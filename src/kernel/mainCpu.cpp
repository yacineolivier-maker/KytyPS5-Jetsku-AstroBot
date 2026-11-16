#include "kernel/mainCpu.h"

#include "common/common.h"

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
	uint64_t    cpu_delta = 0;
	bool        active   = false;
	// window statistics
	std::array<uint64_t, CatCount>        ns {};
	std::unordered_map<uint64_t, uint64_t> game_rips;
	std::unordered_map<uint64_t, uint64_t> emu_offsets;
	uint64_t                              total = 0;
};

std::mutex            g_mutex;
std::vector<Thread*>  g_threads;
std::atomic<uint64_t> g_frames {0};
std::atomic<bool>     g_started {false};

struct Ntdll {
	uint64_t wait[5] {};
	uint64_t yield[2] {};
	HMODULE  modules[4] {};
	HMODULE  exe = nullptr;
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
	const auto ms = [fdiv](uint64_t ns) { return static_cast<double>(ns) / 1e6 / fdiv; };
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
		thread->ns.fill(0);
		thread->game_rips.clear();
		thread->emu_offsets.clear();
		thread->total = 0;
	}
	std::fflush(stdout);
}

void SamplerRun() {
	InitNtdll();
	SetThreadDescription(GetCurrentThread(), L"emu:echantillonneur CPU");
	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
	uint64_t last_report = NowNs();
	uint64_t last_tick   = last_report;
	{
		std::lock_guard lock(g_mutex);
		ChooseActive();
	}
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
			ResumeThread(thread->handle);
			if (!ok) {
				continue;
			}
			// Classification after the thread runs again (it may hold the loader or heap lock).
			uint64_t   emu_offset = 0;
			const auto cat        = Classify(context.Rip, &emu_offset);
			thread->ns[cat] += weight;
			thread->total += weight;
			if (cat == Game) {
				thread->game_rips[context.Rip] += weight;
			} else if (cat == Emulator) {
				thread->emu_offsets[emu_offset & ~uint64_t {0xf}] += weight;
			}
		}
		if (now - last_report >= ReportPeriodNs) {
			Report(now - last_report);
			last_report = now;
			ChooseActive();
		}
	}
}

void Register(const char* name, bool main) {
	auto*  thread = new Thread;
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

void NoteFrameSlow() noexcept {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	g_frames.fetch_add(1, std::memory_order_relaxed);
#endif
}

} // namespace Libs::LibKernel::MainCpu
