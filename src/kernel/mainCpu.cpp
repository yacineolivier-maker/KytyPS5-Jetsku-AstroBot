#include "kernel/mainCpu.h"

#include "common/common.h"
#include "graphics/presentation/fluidityLog.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x00000002
#endif
#endif

namespace Libs::LibKernel::MainCpu {

bool g_sampler_enabled = [] {
	const char* value = std::getenv("KYTY_MAIN_CPU_SAMPLER");
	return value != nullptr && std::strcmp(value, "1") == 0;
}();

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
namespace {

constexpr uint64_t EbootBase       = 0x900000000ull;
constexpr uint64_t EbootEnd        = 0x908000000ull; // the eboot's code ends at +0x74bd57c
constexpr uint64_t GuestEnd        = 0xa00000000ull;
constexpr uint64_t ReportPeriodNs  = 10'000'000'000ull;

uint64_t NowNs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

enum Cat : uint32_t {
	TbbWait,
	LabelPoll,
	Atomics,
	Game,
	GuestModule,
	KernelWait,
	KernelYield,
	WindowsOther,
	Emulator,
	OtherModule,
	Unknown,
	CatCount
};
const char* CatName(uint32_t cat) {
	switch (cat) {
		case TbbWait: return "attente active tbb";
		case LabelPoll: return "attente d'etiquette GPU";
		case Atomics: return "atomiques du jeu";
		case Game: return "code du jeu";
		case GuestModule: return "module invite (libc...)";
		case KernelWait: return "noyau : attente";
		case KernelYield: return "noyau : yield/sommeil";
		case WindowsOther: return "Windows (autre)";
		case Emulator: return "emulateur";
		case OtherModule: return "pilote/autre module";
		default: return "inconnu";
	}
}

struct Thread {
	HANDLE      handle   = nullptr;
	DWORD       tid      = 0;
	std::string name;
	bool        main     = false;
	uint64_t    last_cpu = 0; // 100 ns units
	std::atomic<uint32_t> processor {0xffffffffu};
	bool                  last_running = false; // last sample: not in a wait or a sleep stub // last logical processor the thread reported (NoteCpu)
	uint64_t    cpu_delta = 0;
	bool        active   = false;
	// window statistics
	std::array<uint64_t, CatCount>        ns {};
	std::unordered_map<uint64_t, uint64_t> game_rips;
	std::unordered_map<uint64_t, uint64_t> emu_offsets;
	std::unordered_map<uint64_t, uint64_t> win_rips;
	// Main thread, samples inside a sleep (NtDelayExecution / NtYieldExecution) or a blocking wait:
	// key = first kyty_emulator.exe return address on the stack (offset) << 32 | first eboot one (va).
	std::unordered_map<uint64_t, uint64_t> sleep_callers;
	std::unordered_map<uint64_t, uint64_t> wait_callers;
	uint64_t                              total = 0;
};

std::mutex            g_mutex;
thread_local Thread*  t_self = nullptr; // this thread's record (registered threads)
// Physical core of each logical processor of group 0 (placement report).
std::array<int, 64>   g_core_of {};
// Placement window (sampler thread only).
std::array<uint64_t, 64>                         g_main_on_cpu {};
uint64_t                                         g_main_running   = 0;
uint64_t                                         g_main_shared    = 0;
std::unordered_map<std::string, uint64_t>        g_main_shared_by;
std::vector<Thread*>  g_threads;
std::atomic<uint64_t> g_frames {0};
// Frame pacing (16/11 evening): sequence and end time of the last frame, histogram of the window.
std::atomic<uint64_t> g_frame_seq {0};
std::atomic<uint64_t> g_frame_end_ns {0};
std::atomic<uint64_t> g_frame_prev_ns {0};
constexpr int         HistBuckets = 6; // <34, 34-40, 40-50, 50-67, 67-100, >=100 ms
std::array<std::atomic<uint64_t>, HistBuckets> g_hist {};
// Guest sleeps (16/11 evening): by requested size, and by caller.
constexpr int SleepBuckets = 6; // 0, <=1 us, <=100 us, <=1 ms, <=10 ms, more
std::array<std::atomic<uint64_t>, SleepBuckets> g_sleep_count {};
std::array<std::atomic<uint64_t>, SleepBuckets> g_sleep_req_ns {};
std::array<std::atomic<uint64_t>, SleepBuckets> g_sleep_act_ns {};
std::mutex                                       g_sleep_mutex;
std::unordered_map<uint64_t, std::pair<uint64_t, uint64_t>> g_sleep_callers; // ra -> count, actual ns
double SpikeMs() {
	static const double ms = [] {
		const char* value = std::getenv("KYTY_MAIN_CPU_SPIKE_MS");
		return value != nullptr && *value != 0 ? std::strtod(value, nullptr) : 45.0;
	}();
	return ms;
}
std::atomic<bool>     g_started {false};

struct Ntdll {
	uint64_t wait[5] {};
	uint64_t yield[2] {};
	HMODULE  modules[4] {};
	HMODULE  exe = nullptr;
	uint64_t exe_size = 0;
};
Ntdll g_nt;

void InitNtdll() {
	HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
	const char* waits[] = {"NtWaitForAlertByThreadId", "NtWaitForSingleObject", "NtWaitForMultipleObjects",
	                       "NtRemoveIoCompletion", "NtWaitForKeyedEvent"};
	for (int i = 0; i < 5; i++) {
		g_nt.wait[i] = reinterpret_cast<uint64_t>(GetProcAddress(ntdll, waits[i]));
	}
	g_nt.yield[0]   = reinterpret_cast<uint64_t>(GetProcAddress(ntdll, "NtDelayExecution"));
	g_nt.yield[1]   = reinterpret_cast<uint64_t>(GetProcAddress(ntdll, "NtYieldExecution"));
	g_nt.modules[0] = ntdll;
	g_nt.modules[1] = GetModuleHandleW(L"kernelbase.dll");
	g_nt.modules[2] = GetModuleHandleW(L"kernel32.dll");
	g_nt.modules[3] = GetModuleHandleW(L"win32u.dll");
	g_nt.exe        = GetModuleHandleW(nullptr);
	{
		const auto* base = reinterpret_cast<const uint8_t*>(g_nt.exe);
		const auto* dos  = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
		const auto* nt   = reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + dos->e_lfanew);
		g_nt.exe_size    = nt->OptionalHeader.SizeOfImage;
	}
}

bool InStub(uint64_t rip, uint64_t start) {
	return start != 0 && rip >= start && rip < start + 0x20;
}

uint32_t Classify(uint64_t rip, uint64_t* emu_offset) {
	if (rip >= EbootBase && rip < EbootEnd) {
		const auto va = rip - EbootBase;
		if ((va >= 0xaee4a0 && va < 0xaee930) || (va >= 0xaec980 && va < 0xaeca40)) {
			return TbbWait;
		}
		if (va >= 0x114b2e0 && va < 0x114b330) {
			return LabelPoll;
		}
		if (va >= 0x250 && va < 0x400) {
			return Atomics;
		}
		return Game;
	}
	if (rip >= EbootEnd && rip < GuestEnd) {
		return GuestModule;
	}
	for (auto start: g_nt.wait) {
		if (InStub(rip, start)) {
			return KernelWait;
		}
	}
	for (auto start: g_nt.yield) {
		if (InStub(rip, start)) {
			return KernelYield;
		}
	}
	HMODULE module = nullptr;
	if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
	                           GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
	                       reinterpret_cast<LPCWSTR>(rip), &module) == 0 ||
	    module == nullptr) {
		return Unknown;
	}
	if (module == g_nt.exe) {
		*emu_offset = rip - reinterpret_cast<uint64_t>(module);
		return Emulator;
	}
	for (auto* windows: g_nt.modules) {
		if (module == windows) {
			return WindowsOther;
		}
	}
	return OtherModule;
}

// "module!export+0xN" for an address in a loaded DLL: the nearest export at or below it (internal functions
// show as the export before them).
std::unordered_map<uint64_t, std::string> g_win_name_cache;
std::string WindowsName(uint64_t rip) {
	if (auto it = g_win_name_cache.find(rip); it != g_win_name_cache.end()) {
		return it->second;
	}
	std::string name = "?";
	HMODULE     module = nullptr;
	if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
	                       reinterpret_cast<LPCWSTR>(rip), &module) != 0 &&
	    module != nullptr) {
		char path[MAX_PATH] {};
		GetModuleFileNameA(module, path, MAX_PATH);
		const char* base_name = std::strrchr(path, '\\');
		base_name             = base_name != nullptr ? base_name + 1 : path;
		const auto  base      = reinterpret_cast<const uint8_t*>(module);
		const auto  offset    = static_cast<uint32_t>(rip - reinterpret_cast<uint64_t>(module));
		const auto* dos       = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
		const auto* nt        = reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + dos->e_lfanew);
		const auto& dir       = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT];
		const char* best      = nullptr;
		uint32_t    best_rva  = 0;
		if (dir.VirtualAddress != 0) {
			const auto* exp       = reinterpret_cast<const IMAGE_EXPORT_DIRECTORY*>(base + dir.VirtualAddress);
			const auto* functions = reinterpret_cast<const uint32_t*>(base + exp->AddressOfFunctions);
			const auto* names     = reinterpret_cast<const uint32_t*>(base + exp->AddressOfNames);
			const auto* ordinals  = reinterpret_cast<const uint16_t*>(base + exp->AddressOfNameOrdinals);
			for (uint32_t i = 0; i < exp->NumberOfNames; i++) {
				const uint32_t rva = functions[ordinals[i]];
				if (rva <= offset && rva >= best_rva && (rva < dir.VirtualAddress || rva >= dir.VirtualAddress + dir.Size)) {
					best_rva = rva;
					best     = reinterpret_cast<const char*>(base + names[i]);
				}
			}
		}
		char text[256];
		if (best != nullptr) {
			std::snprintf(text, sizeof(text), "%s!%s+0x%x", base_name, best, offset - best_rva);
		} else {
			std::snprintf(text, sizeof(text), "%s+0x%x", base_name, offset);
		}
		name = text;
	}
	g_win_name_cache.emplace(rip, name);
	return name;
}

// The start of the eboot function holding `rip`: the first byte after the int3 padding before it.
std::unordered_map<uint64_t, uint64_t> g_function_cache;
uint64_t FunctionStart(uint64_t rip) {
	if (auto it = g_function_cache.find(rip); it != g_function_cache.end()) {
		return it->second;
	}
	constexpr size_t      Window = 0x4000;
	std::vector<uint8_t>  bytes(Window + 1);
	const uint64_t        lo     = rip > EbootBase + Window ? rip - Window : EbootBase;
	SIZE_T                read   = 0;
	uint64_t              result = rip;
	if (ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<LPCVOID>(lo), bytes.data(),
	                      static_cast<SIZE_T>(rip - lo + 1), &read) != 0 &&
	    read == rip - lo + 1) {
		for (size_t i = static_cast<size_t>(rip - lo); i > 0; i--) {
			if (bytes[i - 1] == 0xcc && bytes[i] != 0xcc) {
				result = lo + i;
				break;
			}
		}
	}
	g_function_cache.emplace(rip, result);
	return result;
}

uint64_t CpuTime(HANDLE handle) {
	FILETIME creation {}, exit {}, kernel {}, user {};
	if (GetThreadTimes(handle, &creation, &exit, &kernel, &user) == 0) {
		return 0;
	}
	const auto k = (uint64_t {kernel.dwHighDateTime} << 32u) | kernel.dwLowDateTime;
	const auto u = (uint64_t {user.dwHighDateTime} << 32u) | user.dwLowDateTime;
	return k + u;
}

uint32_t ExtraThreads() {
	static const uint32_t count = [] {
		const char* value = std::getenv("KYTY_MAIN_CPU_SAMPLER_THREADS");
		return static_cast<uint32_t>(
		    std::min<unsigned long>(value != nullptr ? std::strtoul(value, nullptr, 10) : 3ul, 16ul));
	}();
	return count;
}

// Every 10 s, registry lock held: the threads to sample next (the main thread and the guest
// threads that used the most CPU), dropping threads that ended.
void ChooseActive() {
	for (auto it = g_threads.begin(); it != g_threads.end();) {
		DWORD code = 0;
		if (GetExitCodeThread((*it)->handle, &code) != 0 && code != STILL_ACTIVE) {
			CloseHandle((*it)->handle);
			delete *it;
			it = g_threads.erase(it);
			continue;
		}
		const auto cpu   = CpuTime((*it)->handle);
		(*it)->cpu_delta = cpu - std::min(cpu, (*it)->last_cpu);
		(*it)->last_cpu  = cpu;
		(*it)->active    = (*it)->main;
		++it;
	}
	std::vector<Thread*> others;
	for (auto* thread: g_threads) {
		if (!thread->main) {
			others.push_back(thread);
		}
	}
	std::sort(others.begin(), others.end(),
	          [](const Thread* a, const Thread* b) { return a->cpu_delta > b->cpu_delta; });
	for (size_t i = 0; i < others.size() && i < ExtraThreads(); i++) {
		others[i]->active = true;
	}
}

void Report(uint64_t window_ns) {
	const auto   frames = g_frames.exchange(0, std::memory_order_relaxed);
	const double fdiv   = frames != 0 ? static_cast<double>(frames) : 1.0;
	std::printf("Kyty echantillons CPU 10 s : %.1f images/s ; temps par image (part de la fenetre)\n",
	            static_cast<double>(frames) * 1e9 / static_cast<double>(window_ns));
	{
		uint64_t h[HistBuckets];
		for (int i = 0; i < HistBuckets; i++) {
			h[i] = g_hist[i].exchange(0, std::memory_order_relaxed);
		}
		std::printf("Kyty cadence 10 s : images <34 ms %" PRIu64 " ; 34-40 %" PRIu64 " ; 40-50 %" PRIu64
		            " ; 50-67 %" PRIu64 " ; 67-100 %" PRIu64 " ; >=100 %" PRIu64 "\n",
		            h[0], h[1], h[2], h[3], h[4], h[5]);
	}
	const auto ms = [fdiv](uint64_t ns) { return static_cast<double>(ns) / 1e6 / fdiv; };
	if (g_main_running != 0) {
		std::string text;
		char        part[160];
		std::snprintf(part, sizeof(part),
		              "Kyty placement 10 s : fil principal actif %" PRIu64 " echantillons, frere SMT occupe par un fil suivi %.0f %% (",
		              g_main_running, 100.0 * static_cast<double>(g_main_shared) / static_cast<double>(g_main_running));
		text = part;
		for (const auto& [name, n]: g_main_shared_by) {
			std::snprintf(part, sizeof(part), " %s %.0f %% ;", name.c_str(),
			              100.0 * static_cast<double>(n) / static_cast<double>(g_main_running));
			text += part;
		}
		text += " ) ; processeurs :";
		for (int i = 0; i < 64; i++) {
			if (g_main_on_cpu[i] != 0) {
				std::snprintf(part, sizeof(part), " %d:%.0f%%", i,
				              100.0 * static_cast<double>(g_main_on_cpu[i]) / static_cast<double>(g_main_running));
				text += part;
			}
		}
		std::printf("%s\n", text.c_str());
		g_main_on_cpu.fill(0);
		g_main_running = 0;
		g_main_shared  = 0;
		g_main_shared_by.clear();
	}
	{
		static const char* names[SleepBuckets] = {"0", "<=1us", "<=100us", "<=1ms", "<=10ms", ">10ms"};
		std::string text = "Kyty sommeils invites 10 s (nombre, demande moyen / obtenu moyen us, ms/image) :";
		for (int i = 0; i < SleepBuckets; i++) {
			const auto n   = g_sleep_count[i].exchange(0, std::memory_order_relaxed);
			const auto req = g_sleep_req_ns[i].exchange(0, std::memory_order_relaxed);
			const auto act = g_sleep_act_ns[i].exchange(0, std::memory_order_relaxed);
			if (n == 0) {
				continue;
			}
			char part[160];
			std::snprintf(part, sizeof(part), " %s %" PRIu64 " (%.1f / %.1f, %.2f) ;", names[i], n,
			              static_cast<double>(req) / 1e3 / static_cast<double>(n),
			              static_cast<double>(act) / 1e3 / static_cast<double>(n), ms(act));
			text += part;
		}
		std::printf("%s\n", text.c_str());
		std::vector<std::pair<uint64_t, std::pair<uint64_t, uint64_t>>> callers;
		{
			std::lock_guard lock(g_sleep_mutex);
			callers.assign(g_sleep_callers.begin(), g_sleep_callers.end());
			g_sleep_callers.clear();
		}
		std::sort(callers.begin(), callers.end(),
		          [](const auto& a, const auto& b) { return a.second.second > b.second.second; });
		std::string top = "  sommeils par appelant (jeu va : nombre, obtenu moyen us, ms/image) :";
		for (size_t i = 0; i < callers.size() && i < 8; i++) {
			const auto ra = callers[i].first;
			char       part[160];
			std::snprintf(part, sizeof(part), " 0x%" PRIx64 " : %" PRIu64 ", %.1f, %.2f ;",
			              ra >= EbootBase && ra < EbootEnd ? ra - EbootBase : ra, callers[i].second.first,
			              static_cast<double>(callers[i].second.second) / 1e3 /
			                  static_cast<double>(callers[i].second.first),
			              ms(callers[i].second.second));
			top += part;
		}
		std::printf("%s\n", top.c_str());
	}
	for (auto* thread: g_threads) {
		if (thread->total == 0) {
			continue;
		}
		std::printf("  %s%s : %.1f ms/image echantillonnes (%.0f %% de la fenetre)\n", thread->name.c_str(),
		            thread->main ? " (fil principal)" : "", ms(thread->total),
		            100.0 * static_cast<double>(thread->total) / static_cast<double>(window_ns));
		std::string line = "   ";
		std::vector<std::pair<uint64_t, uint32_t>> cats;
		for (uint32_t c = 0; c < CatCount; c++) {
			if (thread->ns[c] != 0) {
				cats.emplace_back(thread->ns[c], c);
			}
		}
		std::sort(cats.rbegin(), cats.rend());
		for (const auto& [ns, c]: cats) {
			char part[96];
			std::snprintf(part, sizeof(part), " %s %.2f (%.0f %%) ;", CatName(c), ms(ns),
			              100.0 * static_cast<double>(ns) / static_cast<double>(thread->total));
			line += part;
		}
		std::printf("%s\n", line.c_str());
		// Game functions.
		std::unordered_map<uint64_t, uint64_t> functions;
		for (const auto& [rip, ns]: thread->game_rips) {
			functions[FunctionStart(rip)] += ns;
		}
		std::vector<std::pair<uint64_t, uint64_t>> sorted(functions.begin(), functions.end());
		std::sort(sorted.begin(), sorted.end(),
		          [](const auto& a, const auto& b) { return a.second > b.second; });
		std::string top = "    fonctions du jeu :";
		for (size_t i = 0; i < sorted.size() && i < 10; i++) {
			char part[80];
			std::snprintf(part, sizeof(part), " 0x%" PRIx64 " %.2f ;", sorted[i].first - EbootBase,
			              ms(sorted[i].second));
			top += part;
		}
		std::printf("%s (%zu fonctions)\n", top.c_str(), sorted.size());
		std::vector<std::pair<uint64_t, uint64_t>> emu(thread->emu_offsets.begin(),
		                                               thread->emu_offsets.end());
		std::sort(emu.begin(), emu.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
		if (!emu.empty()) {
			std::string text = "    emulateur (kyty_emulator.exe+) :";
			for (size_t i = 0; i < emu.size() && i < 5; i++) {
				char part[64];
				std::snprintf(part, sizeof(part), " 0x%" PRIx64 " %.2f ;", emu[i].first, ms(emu[i].second));
				text += part;
			}
			std::printf("%s\n", text.c_str());
		}
		// Windows and other modules, by export (sampled RIP; inside a system call it is the stub).
		std::unordered_map<std::string, uint64_t> win;
		for (const auto& [rip, ns]: thread->win_rips) {
			win[WindowsName(rip)] += ns;
		}
		std::vector<std::pair<std::string, uint64_t>> win_sorted(win.begin(), win.end());
		std::sort(win_sorted.begin(), win_sorted.end(),
		          [](const auto& a, const auto& b) { return a.second > b.second; });
		if (!win_sorted.empty()) {
			std::string text = "    Windows/modules :";
			for (size_t i = 0; i < win_sorted.size() && i < 8; i++) {
				char part[320];
				std::snprintf(part, sizeof(part), " %s %.2f ;", win_sorted[i].first.c_str(), ms(win_sorted[i].second));
				text += part;
			}
			std::printf("%s\n", text.c_str());
		}
		thread->ns.fill(0);
		thread->game_rips.clear();
		thread->emu_offsets.clear();
		thread->win_rips.clear();
		for (int kind = 0; kind < 2; kind++) {
			auto& callers = kind == 0 ? thread->sleep_callers : thread->wait_callers;
			if (callers.empty()) {
				continue;
			}
			std::vector<std::pair<uint64_t, uint64_t>> list(callers.begin(), callers.end());
			std::sort(list.begin(), list.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
			std::string text = kind == 0 ? "    sommeils, appelants (emu+off / jeu va) :" : "    attentes, appelants (emu+off / jeu va) :";
			for (size_t i = 0; i < list.size() && i < 8; i++) {
				char part[96];
				std::snprintf(part, sizeof(part), " 0x%" PRIx64 "/0x%" PRIx64 " %.2f ;", list[i].first >> 32,
				              list[i].first & 0xffffffffull, ms(list[i].second));
				text += part;
			}
			std::printf("%s\n", text.c_str());
			callers.clear();
		}
		thread->total = 0;
	}
	std::fflush(stdout);
}

void SamplerRun() {
	InitNtdll();
	{
		g_core_of.fill(-1);
		DWORD length = 0;
		GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &length);
		std::vector<uint8_t> buffer(length);
		auto* info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data());
		if (length != 0 && GetLogicalProcessorInformationEx(RelationProcessorCore, info, &length) != 0) {
			int core = 0;
			for (DWORD offset = 0; offset < length;) {
				auto* entry = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data() + offset);
				if (entry->Relationship == RelationProcessorCore && entry->Processor.GroupCount > 0 &&
				    entry->Processor.GroupMask[0].Group == 0) {
					const KAFFINITY mask = entry->Processor.GroupMask[0].Mask;
					for (int i = 0; i < 64; i++) {
						if ((mask >> i) & 1) {
							g_core_of[i] = core;
						}
					}
					core++;
				}
				offset += entry->Size;
			}
		}
	}
	SetThreadDescription(GetCurrentThread(), L"emu:echantillonneur CPU");
	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
	uint64_t last_report = NowNs();
	uint64_t last_tick   = last_report;
	{
		std::lock_guard lock(g_mutex);
		ChooseActive();
	}
	// Per-frame breakdown of the main thread, printed for frames longer than KYTY_MAIN_CPU_SPIKE_MS.
	std::array<uint64_t, CatCount>         spike_ns {};
	std::unordered_map<uint64_t, uint64_t> spike_rips; // game function starts and Windows rips
	uint64_t                               spike_seq     = g_frame_seq.load(std::memory_order_acquire);
	uint32_t                               spike_printed = 0;
	for (;;) {
		// About 1 ms: the high-resolution timer (Sleep(1) may be 15.6 ms).
		static thread_local HANDLE timer =
		    CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
		if (timer != nullptr) {
			LARGE_INTEGER due {};
			due.QuadPart = -10'000; // 1 ms
			SetWaitableTimerEx(timer, &due, 0, nullptr, nullptr, nullptr, 0);
			WaitForSingleObject(timer, INFINITE);
		} else {
			Sleep(1);
		}
		const auto now    = NowNs();
		const auto weight = now - last_tick;
		last_tick         = now;
		std::lock_guard lock(g_mutex);
		for (auto* thread: g_threads) {
			if (!thread->active) {
				continue;
			}
			if (SuspendThread(thread->handle) == static_cast<DWORD>(-1)) {
				continue;
			}
			CONTEXT context {};
			context.ContextFlags = CONTEXT_CONTROL;
			const bool ok        = GetThreadContext(thread->handle, &context) != 0;
			// Main thread inside a sleep or a wait: who called it (stack read while suspended).
			uint64_t caller_key  = 0;
			int      caller_kind = 0; // 1 sleep, 2 wait
			if (ok && thread->main) {
				for (auto start: g_nt.yield) {
					if (InStub(context.Rip, start)) {
						caller_kind = 1;
					}
				}
				for (auto start: g_nt.wait) {
					if (InStub(context.Rip, start)) {
						caller_kind = 2;
					}
				}
				if (caller_kind != 0 && context.Rsp != 0) {
					const auto  exe_base = reinterpret_cast<uint64_t>(g_nt.exe);
					const auto* stack    = reinterpret_cast<const uint64_t*>(context.Rsp);
					uint64_t    emu = 0, game = 0;
					for (int i = 0; i < 160 && (emu == 0 || game == 0); i++) {
						const uint64_t q = stack[i];
						if (emu == 0 && q > exe_base && q < exe_base + g_nt.exe_size) {
							emu = q - exe_base;
						} else if (game == 0 && q > EbootBase && q < EbootEnd) {
							game = q - EbootBase;
						}
					}
					caller_key = (emu << 32) | (game & 0xffffffffull);
				}
			}
			ResumeThread(thread->handle);
			if (!ok) {
				continue;
			}
			// Classification after the thread runs again (it may hold the loader or heap lock).
			uint64_t   emu_offset = 0;
			const auto cat        = Classify(context.Rip, &emu_offset);
			thread->ns[cat] += weight;
			thread->last_running = cat != KernelWait && cat != KernelYield;
			thread->total += weight;
			if (cat == Game) {
				thread->game_rips[context.Rip] += weight;
			} else if (cat == Emulator) {
				thread->emu_offsets[emu_offset & ~uint64_t {0xf}] += weight;
			} else if (cat == WindowsOther || cat == OtherModule || cat == KernelWait) {
				thread->win_rips[context.Rip] += weight;
			}
			if (caller_kind == 1) {
				thread->sleep_callers[caller_key] += weight;
			} else if (caller_kind == 2) {
				thread->wait_callers[caller_key] += weight;
			}
			if (thread->main) {
				spike_ns[cat] += weight;
				if (cat == Game || cat == WindowsOther || cat == OtherModule || cat == KernelWait ||
				    cat == KernelYield || cat == GuestModule) {
					spike_rips[context.Rip] += weight;
				}
			}
		}
		// Placement: is the main thread's SMT sibling running another sampled thread?
		{
			const Thread* main_thread = nullptr;
			for (auto* thread: g_threads) {
				if (thread->main && thread->active) {
					main_thread = thread;
				}
			}
			if (main_thread != nullptr && main_thread->last_running) {
				const uint32_t cpu = main_thread->processor.load(std::memory_order_relaxed);
				if (cpu < 64 && g_core_of[cpu] >= 0) {
					g_main_running++;
					g_main_on_cpu[cpu]++;
					for (auto* thread: g_threads) {
						if (thread == main_thread || !thread->active || !thread->last_running) {
							continue;
						}
						const uint32_t other = thread->processor.load(std::memory_order_relaxed);
						if (other < 64 && other != cpu && g_core_of[other] == g_core_of[cpu]) {
							g_main_shared++;
							g_main_shared_by[thread->name]++;
							break;
						}
					}
				}
			}
		}
		if (const auto seq = g_frame_seq.load(std::memory_order_acquire); seq != spike_seq) {
			const uint64_t end  = g_frame_end_ns.load(std::memory_order_relaxed);
			const uint64_t prev = g_frame_prev_ns.load(std::memory_order_relaxed);
			const double   ms   = prev != 0 && end > prev ? static_cast<double>(end - prev) / 1e6 : 0.0;
			if (seq == spike_seq + 1 && ms >= SpikeMs() && spike_printed < 40) {
				spike_printed++;
				std::vector<std::pair<uint64_t, uint32_t>> cats;
				for (uint32_t c = 0; c < CatCount; c++) {
					if (spike_ns[c] != 0) {
						cats.emplace_back(spike_ns[c], c);
					}
				}
				std::sort(cats.rbegin(), cats.rend());
				std::string line;
				char        part[320];
				std::snprintf(part, sizeof(part), "Kyty pic : image %.1f ms ; fil principal :", ms);
				line = part;
				for (size_t i = 0; i < cats.size() && i < 5; i++) {
					std::snprintf(part, sizeof(part), " %s %.1f ;", CatName(cats[i].second),
					              static_cast<double>(cats[i].first) / 1e6);
					line += part;
				}
				std::unordered_map<std::string, uint64_t> named;
				for (const auto& [rip, ns]: spike_rips) {
					if (rip >= EbootBase && rip < EbootEnd) {
						std::snprintf(part, sizeof(part), "jeu 0x%" PRIx64, FunctionStart(rip) - EbootBase);
						named[part] += ns;
					} else if (rip >= EbootEnd && rip < GuestEnd) {
						named["module invite"] += ns;
					} else {
						named[WindowsName(rip)] += ns;
					}
				}
				std::vector<std::pair<std::string, uint64_t>> top(named.begin(), named.end());
				std::sort(top.begin(), top.end(), [](const auto& x, const auto& y) { return x.second > y.second; });
				line += " | detail :";
				for (size_t i = 0; i < top.size() && i < 5; i++) {
					std::snprintf(part, sizeof(part), " %s %.1f ;", top[i].first.c_str(),
					              static_cast<double>(top[i].second) / 1e6);
					line += part;
				}
				std::printf("%s\n", line.c_str());
				std::fflush(stdout);
			}
			spike_seq = seq;
			spike_ns.fill(0);
			spike_rips.clear();
		}
		if (now - last_report >= ReportPeriodNs) {
			Report(now - last_report);
			last_report   = now;
			spike_printed = 0;
			ChooseActive();
		}
	}
}

void Register(const char* name, bool main) {
	auto*  thread = new Thread;
	t_self        = thread;
	HANDLE handle = nullptr;
	if (DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &handle,
	                    THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, 0) == 0) {
		delete thread;
		return;
	}
	thread->handle   = handle;
	thread->tid      = GetCurrentThreadId();
	thread->name     = name != nullptr ? name : "?";
	thread->main     = main;
	thread->last_cpu = CpuTime(handle);
	std::lock_guard lock(g_mutex);
	g_threads.push_back(thread);
}

void PinMainThread() {
	const char* value = std::getenv("KYTY_MAIN_THREAD_CORE");
	if (value == nullptr || *value == 0) {
		return;
	}
	const auto logical = std::strtoul(value, nullptr, 10);
	const char* reserve = std::getenv("KYTY_CPU_RESERVE");
	const char* sets    = std::getenv("KYTY_CPU_SETS");
	if ((reserve != nullptr && *reserve != 0 && std::strcmp(reserve, "off") != 0 &&
	     std::strcmp(reserve, "0") != 0) ||
	    (sets != nullptr && *sets != 0)) {
		std::printf("Kyty fil principal : KYTY_MAIN_THREAD_CORE ignore (KYTY_CPU_RESERVE ou KYTY_CPU_SETS "
		            "gerent deja les ensembles de processeurs)\n");
		std::fflush(stdout);
		return;
	}
	ULONG length = 0;
	(void)GetSystemCpuSetInformation(nullptr, 0, &length, GetCurrentProcess(), 0);
	std::vector<uint8_t> buffer(length);
	if (length == 0 || GetSystemCpuSetInformation(reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(buffer.data()),
	                                              length, &length, GetCurrentProcess(), 0) == 0) {
		std::printf("Kyty fil principal : GetSystemCpuSetInformation a echoue (%lu)\n", GetLastError());
		std::fflush(stdout);
		return;
	}
	struct Set {
		ULONG id;
		BYTE  logical;
		BYTE  core;
		WORD  group;
	};
	std::vector<Set> all;
	for (size_t offset = 0; offset < length;) {
		const auto* info = reinterpret_cast<const SYSTEM_CPU_SET_INFORMATION*>(buffer.data() + offset);
		if (info->Type == CpuSetInformation) {
			all.push_back({info->CpuSet.Id, info->CpuSet.LogicalProcessorIndex, info->CpuSet.CoreIndex,
			               info->CpuSet.Group});
		}
		offset += info->Size;
	}
	const auto target = std::find_if(all.begin(), all.end(), [logical](const Set& s) {
		return s.group == 0 && s.logical == logical;
	});
	if (target == all.end()) {
		std::printf("Kyty fil principal : processeur logique %lu introuvable\n", logical);
		std::fflush(stdout);
		return;
	}
	std::vector<ULONG> others;
	std::string        siblings;
	for (const auto& s: all) {
		if (s.group == target->group && s.core == target->core) {
			siblings += " " + std::to_string(s.logical);
		} else {
			others.push_back(s.id);
		}
	}
	const bool defaults = SetProcessDefaultCpuSets(GetCurrentProcess(), others.data(),
	                                               static_cast<ULONG>(others.size())) != 0;
	const ULONG own   = target->id;
	const bool  main  = SetThreadSelectedCpuSets(GetCurrentThread(), &own, 1) != 0;
	std::printf("Kyty fil principal : processeur logique %lu (coeur %u, processeurs du coeur :%s) ; "
	            "autres fils hors de ce coeur : %s ; fil principal : %s (KYTY_MAIN_THREAD_CORE)\n",
	            logical, static_cast<unsigned>(target->core), siblings.c_str(), defaults ? "oui" : "ECHEC",
	            main ? "oui" : "ECHEC");
	std::fflush(stdout);
}

void SetMainPriority() {
	const char* value = std::getenv("KYTY_MAIN_THREAD_PRIORITY");
	if (value == nullptr || *value == 0) {
		return;
	}
	const int  level = std::clamp(static_cast<int>(std::strtol(value, nullptr, 10)), -2, 2);
	const bool ok    = SetThreadPriority(GetCurrentThread(), level) != 0;
	std::printf("Kyty fil principal : priorite Windows %d (%s) (KYTY_MAIN_THREAD_PRIORITY)\n", level,
	            ok ? "appliquee" : "ECHEC");
	std::fflush(stdout);
}

} // namespace
#endif

void OnMainThreadStart() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	if (Libs::Graphics::FluidityEnabled()) {
		HANDLE handle = nullptr;
		if (DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &handle,
		                    THREAD_QUERY_LIMITED_INFORMATION, FALSE, 0) != 0) {
			Libs::Graphics::FluiditySetMainThread(handle);
		}
	}
	PinMainThread();
	SetMainPriority();
	if (g_sampler_enabled) {
		Register("MainThread", true);
		bool expected = false;
		if (g_started.compare_exchange_strong(expected, true)) {
			std::printf("Kyty echantillons CPU : fil principal + %u fils invites, ~1 kHz "
			            "(KYTY_MAIN_CPU_SAMPLER)\n",
			            ExtraThreads());
			std::fflush(stdout);
			std::thread(SamplerRun).detach();
		}
	}
#endif
}

void OnGuestThreadStart(const char* name) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	if (g_sampler_enabled) {
		Register(name, false);
	}
#else
	(void)name;
#endif
}

void NoteGuestSleepSlow(uint64_t ra, uint64_t requested_ns, uint64_t actual_ns) noexcept {
	const int b = requested_ns == 0 ? 0 : requested_ns <= 1000 ? 1 : requested_ns <= 100'000 ? 2
	            : requested_ns <= 1'000'000 ? 3 : requested_ns <= 10'000'000 ? 4 : 5;
	g_sleep_count[b].fetch_add(1, std::memory_order_relaxed);
	g_sleep_req_ns[b].fetch_add(requested_ns, std::memory_order_relaxed);
	g_sleep_act_ns[b].fetch_add(actual_ns, std::memory_order_relaxed);
	std::lock_guard lock(g_sleep_mutex);
	auto& entry = g_sleep_callers[ra];
	entry.first++;
	entry.second += actual_ns;
}

void NoteCpuSlow() noexcept {
	if (auto* self = t_self; self != nullptr) {
		self->processor.store(GetCurrentProcessorNumber(), std::memory_order_relaxed);
	}
}

void NoteFrameSlow() noexcept {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	g_frames.fetch_add(1, std::memory_order_relaxed);
	const uint64_t now  = NowNs();
	const uint64_t prev = g_frame_end_ns.exchange(now, std::memory_order_relaxed);
	g_frame_prev_ns.store(prev, std::memory_order_relaxed);
	g_frame_seq.fetch_add(1, std::memory_order_release);
	if (prev != 0) {
		const double ms = static_cast<double>(now - prev) / 1e6;
		const int    b  = ms < 34.0 ? 0 : ms < 40.0 ? 1 : ms < 50.0 ? 2 : ms < 67.0 ? 3 : ms < 100.0 ? 4 : 5;
		g_hist[b].fetch_add(1, std::memory_order_relaxed);
	}
#endif
}

} // namespace Libs::LibKernel::MainCpu
