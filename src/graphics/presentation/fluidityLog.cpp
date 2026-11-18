#include "graphics/presentation/fluidityLog.h"

#include "common/cpuPlacement.h"
#include "common/debugCounters.h"
#include "graphics/host_gpu/renderer/gpuOpProfiler.h"
#include "graphics/host_gpu/renderer/gpuTiming.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <intrin.h>
#include <windows.h>
// windows.h first
#include <tlhelp32.h>
#endif

// See fluidityLog.h. Nothing here runs unless KYTY_FLUIDITE=1.

namespace Libs::Graphics {

namespace {

using Clock  = std::chrono::steady_clock;
namespace fs = std::filesystem;
namespace DC = Common::DebugCounters;

double EnvNumber(const char* name, double fallback) {
	const char* value = std::getenv(name);
	if (value == nullptr || value[0] == 0) {
		return fallback;
	}
	char*        end    = nullptr;
	const double parsed = std::strtod(value, &end);
	return end != value && parsed > 0.0 ? parsed : fallback;
}

std::string LocalTime(const char* format) {
	const std::time_t now = std::time(nullptr);
	std::tm           local {};
#if defined(_WIN32)
	localtime_s(&local, &now);
#else
	localtime_r(&now, &local);
#endif
	char text[32] {};
	std::strftime(text, sizeof(text), format, &local);
	return text;
}

struct Settings {
	bool        enabled    = false;
	double      spike_ms   = 50.0;
	double      threshold  = 28.0;
	double      gap_s      = 5.0;
	uint32_t    images     = 500;
	uint32_t    reports    = 6;
	uint64_t    quota      = 1024ull << 20u;
	bool        gpu_report = false; // KYTY_GPU_OP_PROFILE=demande
	std::string camera;             // KYTY_FLUIDITE_CAMERA, see ReadCamera
	std::string sampler;            // KYTY_FLUIDITE_ECHANT: echantillonneur.exe (v2, automatic stacks)
	uint32_t    stacks = 6;         // KYTY_FLUIDITE_PILES: stack captures per session
	std::string base;               // _Fluidite
	std::string dir;                // _Fluidite\<date-time>
};

const Settings& Config() {
	static const Settings settings = [] {
		Settings s;
		const char* on = std::getenv("KYTY_FLUIDITE");
		s.enabled      = on != nullptr && on[0] == '1';
		s.spike_ms     = EnvNumber("KYTY_FLUIDITE_PIC_MS", 50.0);
		s.threshold    = EnvNumber("KYTY_FLUIDITE_SEUIL", 28.0);
		s.gap_s        = EnvNumber("KYTY_FLUIDITE_ECART_S", 5.0);
		s.images       = static_cast<uint32_t>(EnvNumber("KYTY_FLUIDITE_IMAGES", 500.0));
		s.reports      = static_cast<uint32_t>(EnvNumber("KYTY_FLUIDITE_RAPPORTS", 6.0));
		s.quota        = static_cast<uint64_t>(EnvNumber("KYTY_FLUIDITE_QUOTA_MO", 1024.0) * 1048576.0);
		if (const char* profile = std::getenv("KYTY_GPU_OP_PROFILE"); profile != nullptr) {
			s.gpu_report = std::strcmp(profile, "demande") == 0;
		}
		if (const char* cam = std::getenv("KYTY_FLUIDITE_CAMERA"); cam != nullptr) {
			s.camera = cam;
		}
		if (const char* sampler = std::getenv("KYTY_FLUIDITE_ECHANT"); sampler != nullptr) {
			s.sampler = sampler;
		}
		s.stacks = static_cast<uint32_t>(EnvNumber("KYTY_FLUIDITE_PILES", 6.0));
		fs::path base;
		if (const char* dir = std::getenv("KYTY_FLUIDITE_DIR"); dir != nullptr && dir[0] != 0) {
			base = dir;
		} else {
#if defined(_WIN32)
			wchar_t exe[MAX_PATH] {};
			if (GetModuleFileNameW(nullptr, exe, MAX_PATH) != 0) {
				base = fs::path(exe).parent_path() / "_Fluidite";
			}
#endif
			if (base.empty()) {
				base = "_Fluidite";
			}
		}
		s.base = base.string();
		s.dir  = (base / LocalTime("%Y%m%d-%H%M%S")).string();
		return s;
	}();
	return settings;
}

// ---- Worker thread: every disk write, and the place budget -------------------------------------

struct Job {
	std::string          path;
	std::vector<uint8_t> bytes;
	bool                 append  = false;
	bool                 capture = false; // check the budget after it
};

std::atomic<bool> g_quota_blocked {false}; // text journals alone exceed the budget

// Captures that rotation may delete: img-* in a session folder, gpuops-<digits>.csv in rapport-gpu.
bool Rotatable(const fs::path& path) {
	const auto name = path.filename().string();
	if (name.rfind("img-", 0) == 0) {
		return true;
	}
	if (path.parent_path().filename() == "rapport-gpu" && name.rfind("gpuops-", 0) == 0 && name.size() > 11 &&
	    name.ends_with(".csv")) {
		const auto middle = name.substr(7, name.size() - 11);
		return !middle.empty() && std::all_of(middle.begin(), middle.end(), [](char c) { return c >= '0' && c <= '9'; });
	}
	return false;
}

// Scans the whole _Fluidite folder; over budget, deletes the oldest rotatable captures.
void EnforceBudget() {
	struct Item {
		fs::file_time_type time;
		fs::path           path;
		uint64_t           size;
	};
	std::vector<Item> items;
	uint64_t          total = 0;
	std::error_code   error;
	for (auto it = fs::recursive_directory_iterator(Config().base, error);
	     !error && it != fs::recursive_directory_iterator(); it.increment(error)) {
		std::error_code item_error;
		if (!it->is_regular_file(item_error)) {
			continue;
		}
		const uint64_t size = it->file_size(item_error);
		if (item_error) {
			continue;
		}
		total += size;
		if (Rotatable(it->path())) {
			items.push_back({it->last_write_time(item_error), it->path(), size});
		}
	}
	if (total <= Config().quota) {
		g_quota_blocked.store(false, std::memory_order_relaxed);
		return;
	}
	std::sort(items.begin(), items.end(), [](const Item& a, const Item& b) { return a.time < b.time; });
	// Down to 90 % of the budget, so that rotation does not run on every capture.
	const uint64_t target = Config().quota / 10u * 9u;
	for (const auto& item: items) {
		if (total <= target) {
			break;
		}
		std::error_code remove_error;
		if (fs::remove(item.path, remove_error)) {
			total -= std::min(total, item.size);
		}
	}
	g_quota_blocked.store(total > Config().quota, std::memory_order_relaxed);
}

class Writer {
public:
	void Push(Job job) {
		{
			std::scoped_lock lock(m_mutex);
			if (m_jobs.size() > 256) { // never let a stuck disk hold memory
				return;
			}
			m_jobs.push_back(std::move(job));
			if (!m_started) {
				m_started = true;
				std::thread([this] { Run(); }).detach();
			}
		}
		m_wake.notify_one();
	}

private:
	void Run() {
		std::error_code error;
		fs::create_directories(Config().dir, error);
		EnforceBudget();
		auto last_check = Clock::now();
		for (;;) {
			Job job;
			{
				std::unique_lock lock(m_mutex);
				m_wake.wait_for(lock, std::chrono::seconds(30), [this] { return !m_jobs.empty(); });
				if (!m_jobs.empty()) {
					job = std::move(m_jobs.front());
					m_jobs.pop_front();
				}
			}
			if (!job.path.empty()) {
				std::error_code dir_error;
				fs::create_directories(fs::path(job.path).parent_path(), dir_error);
				std::ofstream file(job.path, std::ios::binary | (job.append ? std::ios::app : std::ios::trunc));
				file.write(reinterpret_cast<const char*>(job.bytes.data()),
				           static_cast<std::streamsize>(job.bytes.size()));
			}
			// GPU reports are written by gpuOpProfiler's own thread: checked at least every minute.
			const auto now = Clock::now();
			if (job.capture || now - last_check >= std::chrono::minutes(1)) {
				EnforceBudget();
				last_check = now;
			}
		}
	}

	std::mutex              m_mutex;
	std::condition_variable m_wake;
	std::deque<Job>         m_jobs;
	bool                    m_started = false;
};

Writer& GetWriter() {
	static auto* writer = new Writer; // never destroyed: the thread is detached
	return *writer;
}

std::string SessionFile(const std::string& name) {
	return (fs::path(Config().dir) / name).string();
}

void AppendLine(const char* file, const char* header, const std::string& line) {
	static std::mutex               mutex;
	static std::vector<std::string> started;
	std::string                     text;
	{
		std::scoped_lock lock(mutex);
		if (std::find(started.begin(), started.end(), file) == started.end()) {
			started.emplace_back(file);
			text = std::string("\xEF\xBB\xBF") + header + "\r\n"; // UTF-8 BOM for Excel
		}
	}
	text += line + "\r\n";
	GetWriter().Push({SessionFile(file), std::vector<uint8_t>(text.begin(), text.end()), true, false});
}

// ---- Recent guest file opens ---------------------------------------------------------------------

std::mutex                 g_open_mutex;
std::array<std::string, 6> g_opens;
size_t                     g_open_next = 0;

std::string RecentOpens() {
	std::scoped_lock lock(g_open_mutex);
	std::string      text;
	for (size_t i = 0; i < g_opens.size(); i++) {
		const auto& path = g_opens[(g_open_next + g_opens.size() - 1 - i) % g_opens.size()];
		if (path.empty()) {
			continue;
		}
		if (!text.empty()) {
			text += " | ";
		}
		text += path;
	}
	return text;
}

// ---- Camera: KYTY_FLUIDITE_CAMERA=[eboot+]<addr>[,<offset>...][:<floats>] -------------------------
// "eboot+" adds the game executable's load address. Astro Bot 01.019.000, from its own debug line
// "CAMERA POS %5.2f,%5.2f,%5.2f | DIR %5.2f,%5.2f,%5.2f" (eboot 0x2a8565):
//   eboot+0xf372cf8,0x308a80,0x1d330,0x260:8  ->  position x y z (w), direction x y z (w).
// <addr> is read as a pointer, each <offset> but the last is added then read as a pointer, the last
// is added, and <floats> floats (default 16) are read there. Reads go through ReadProcessMemory on
// the own process, so a stale chain gives "?" instead of a fault.

std::atomic<uint64_t> g_eboot_base {0};

bool ReadOwn(uint64_t address, void* out, size_t size) {
#if defined(_WIN32)
	SIZE_T done = 0;
	return ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<const void*>(address), out, size, &done) != 0 &&
	       done == size;
#else
	(void)address;
	(void)out;
	(void)size;
	return false;
#endif
}

// The camera floats: "" on success, else the reason ("?", "?base", "?spec").
const char* ReadCameraFloats(std::array<float, 32>& values, size_t& floats) {
	const auto& spec = Config().camera;
	floats           = 16;
	if (spec.empty()) {
		return "?off";
	}
	std::vector<uint64_t> chain;
	const char*           p      = spec.c_str();
	uint64_t              base   = 0;
	if (std::strncmp(p, "eboot+", 6) == 0) {
		base = g_eboot_base.load(std::memory_order_acquire);
		if (base == 0) {
			return "?base";
		}
		p += 6;
	}
	while (*p != 0) {
		char* end = nullptr;
		chain.push_back(std::strtoull(p, &end, 0));
		if (end == p) {
			return "?spec";
		}
		p = end;
		if (*p == ',') {
			p++;
		} else if (*p == ':') {
			floats = std::clamp<size_t>(std::strtoull(p + 1, nullptr, 0), 1, 32);
			break;
		} else if (*p != 0) {
			return "?spec";
		}
	}
	if (chain.empty()) {
		return "?spec";
	}
	uint64_t address = base + chain[0];
	for (size_t i = 1; i < chain.size(); i++) {
		uint64_t pointer = 0;
		if (!ReadOwn(address, &pointer, sizeof(pointer))) {
			return "?";
		}
		address = pointer + chain[i];
	}
	if (!ReadOwn(address, values.data(), floats * sizeof(float))) {
		return "?";
	}
	return "";
}

std::string ReadCamera() {
	if (Config().camera.empty()) {
		return {};
	}
	std::array<float, 32> values {};
	size_t                floats = 0;
	if (const char* error = ReadCameraFloats(values, floats); *error != 0) {
		return error;
	}
	std::string text;
	char        number[32];
	for (size_t i = 0; i < floats; i++) {
		std::snprintf(number, sizeof(number), "%s%.4g", i == 0 ? "" : " ", values[i]);
		text += number;
	}
	return text;
}

// ---- Host threads and machine, once a second (v2, 18/11) ------------------------------------------
// A monitor thread reads the cycle time of every named thread of the process (QueryThreadCycleTime,
// TSC units) and the system and process CPU times. Published: the load of Thread_Gpu (CP
// translation) and of the guest main thread in % of one core, the top threads, the CPU the other
// processes use (in cores) and the share of the machine in use.

struct Load {
	float       cp_pct     = -1; // Thread_Gpu, -1 unknown
	float       main_pct   = -1; // Thread_Main / MainThread
	float       others     = -1; // cores used by other processes
	float       system_pct = -1; // whole machine, % of all logical processors
	std::string top;             // "Thread_Gpu=96 Thread_Main=71 ..."
};

std::mutex g_load_mutex;
Load       g_load;

Load CurrentLoad() {
	std::scoped_lock lock(g_load_mutex);
	return g_load;
}

#if defined(_WIN32)
uint64_t FileTime(const FILETIME& t) {
	return (static_cast<uint64_t>(t.dwHighDateTime) << 32u) | t.dwLowDateTime;
}

void MonitorLoop() {
	struct Info {
		HANDLE      handle = nullptr;
		std::string name;
		ULONG64     cycles = 0;
		bool        seen   = false;
	};
	std::vector<std::pair<DWORD, Info>> threads;
	const DWORD                         pid = GetCurrentProcessId();
	SYSTEM_INFO                         si {};
	GetSystemInfo(&si);
	const double cpus      = std::max<DWORD>(1, si.dwNumberOfProcessors);
	uint64_t     last_tsc  = __rdtsc();
	auto         last_time = Clock::now();
	FILETIME     idle {}, kernel {}, user {}, c {}, e {}, pk {}, pu {};
	GetSystemTimes(&idle, &kernel, &user);
	GetProcessTimes(GetCurrentProcess(), &c, &e, &pk, &pu);
	uint64_t last_busy = FileTime(kernel) + FileTime(user) - FileTime(idle);
	uint64_t last_proc = FileTime(pk) + FileTime(pu);
	for (;;) {
		std::this_thread::sleep_for(std::chrono::seconds(1));
		const uint64_t tsc   = __rdtsc();
		const auto     now   = Clock::now();
		const double   d_tsc = static_cast<double>(tsc - last_tsc);
		const double   d_s   = std::chrono::duration<double>(now - last_time).count();
		last_tsc             = tsc;
		last_time            = now;
		for (auto& t: threads) {
			t.second.seen = false;
		}
		struct Row {
			std::string name;
			double      pct;
		};
		std::vector<Row> rows;
		if (HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0); snap != INVALID_HANDLE_VALUE) {
			THREADENTRY32 te {};
			te.dwSize = sizeof(te);
			for (BOOL ok = Thread32First(snap, &te); ok; ok = Thread32Next(snap, &te)) {
				if (te.th32OwnerProcessID != pid) {
					continue;
				}
				auto it = std::find_if(threads.begin(), threads.end(), [&](const auto& p) { return p.first == te.th32ThreadID; });
				if (it == threads.end()) {
					Info info;
					info.handle = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE, te.th32ThreadID);
					if (info.handle == nullptr) {
						continue;
					}
					PWSTR text = nullptr;
					if (SUCCEEDED(GetThreadDescription(info.handle, &text)) && text != nullptr) {
						for (const wchar_t* w = text; *w != 0; w++) {
							info.name += *w < 128 ? static_cast<char>(*w) : '?';
						}
						LocalFree(text);
					}
					QueryThreadCycleTime(info.handle, &info.cycles);
					info.seen = true;
					threads.emplace_back(te.th32ThreadID, std::move(info));
					continue;
				}
				auto&   info   = it->second;
				ULONG64 cycles = 0;
				info.seen      = true;
				if (QueryThreadCycleTime(info.handle, &cycles) == 0) {
					continue;
				}
				const double pct = d_tsc > 0 ? 100.0 * static_cast<double>(cycles - info.cycles) / d_tsc : 0.0;
				info.cycles      = cycles;
				if (!info.name.empty() && pct >= 1.0) {
					rows.push_back({info.name, pct});
				}
			}
			CloseHandle(snap);
		}
		std::erase_if(threads, [](auto& p) {
			if (!p.second.seen) {
				CloseHandle(p.second.handle);
				return true;
			}
			return false;
		});
		std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.pct > b.pct; });
		Load load;
		for (const auto& r: rows) {
			if (r.name == "Thread_Gpu") {
				load.cp_pct = std::max(load.cp_pct, static_cast<float>(r.pct));
			} else if (r.name == "Thread_Main" || r.name == "MainThread") {
				load.main_pct = std::max(load.main_pct, static_cast<float>(r.pct));
			}
		}
		for (size_t i = 0; i < rows.size() && i < 6; i++) {
			load.top += (i == 0 ? "" : " ") + rows[i].name + "=" + std::to_string(static_cast<int>(rows[i].pct + 0.5));
		}
		GetSystemTimes(&idle, &kernel, &user);
		GetProcessTimes(GetCurrentProcess(), &c, &e, &pk, &pu);
		const uint64_t s_busy = FileTime(kernel) + FileTime(user) - FileTime(idle);
		const uint64_t p_time = FileTime(pk) + FileTime(pu);
		const double   window = d_s * 1e7; // 100 ns units
		if (window > 0) {
			const double busy  = static_cast<double>(s_busy - last_busy);
			const double proc  = static_cast<double>(p_time - last_proc);
			load.system_pct    = static_cast<float>(100.0 * busy / (window * cpus));
			load.others        = static_cast<float>(std::max(0.0, busy - proc) / window);
		}
		last_busy = s_busy;
		last_proc = p_time;
		std::scoped_lock lock(g_load_mutex);
		g_load = std::move(load);
	}
}
#endif

void StartMonitor() {
#if defined(_WIN32)
	std::thread([] {
		SetThreadDescription(GetCurrentThread(), L"Kyty fluidite fils");
		Common::PlaceCurrentThreadCold();
		MonitorLoop();
	}).detach();
#endif
}

// ---- Automatic stacks (v2, KYTY_FLUIDITE_ECHANT) -------------------------------------------------
// After 3 s of a sustained drop, the external sampler (Developpement\outils\echantillonneur) samples
// the guest main thread and the CP translation thread for 5 s: pile-<episode>-<thread>.txt in the
// session folder. Each launch runs on its own short thread (CreateProcess off the video-out thread).

void LaunchStacks(uint32_t episode) {
#if defined(_WIN32)
	const std::string sampler = Config().sampler;
	const std::string dir     = Config().dir;
	const DWORD       pid     = GetCurrentProcessId();
	std::thread([=] {
		for (const char* thread: {"Thread_Main", "Thread_Gpu"}) {
			const std::string out = (fs::path(dir) / ("pile-" + std::to_string(episode) + "-" + thread + ".txt")).string();
			std::string cmd = "\"" + sampler + "\" " + std::to_string(pid) + " \"" + thread + "\" 5 1 \"" + out + "\"";
			STARTUPINFOA        si {};
			PROCESS_INFORMATION pi {};
			si.cb = sizeof(si);
			if (CreateProcessA(nullptr, cmd.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi) !=
			    0) {
				CloseHandle(pi.hThread);
				CloseHandle(pi.hProcess);
			}
		}
	}).detach();
#else
	(void)episode;
#endif
}

// ---- Black box -----------------------------------------------------------------------------------

std::atomic<void*> g_main_thread {nullptr};

struct Sample {
	float    t_s        = 0; // since the session start
	float    frame_ms   = 0;
	float    cpu_ms     = 0; // guest main thread
	float    gpu_ms     = 0; // GPU busy published since the previous flip (KYTY_GPU_TIMING)
	uint32_t draws      = 0;
	uint32_t dispatches = 0;
	uint32_t shaders    = 0; // shaders + pipelines created
	float    shader_ms  = 0;
	float    wait_ms    = 0; // CPU blocked on the GPU
	uint32_t upload_kb  = 0;
	uint32_t read_kb    = 0; // guest file reads
	float    cp_pct     = -1; // Thread_Gpu load (monitor, last second), -1 unknown
	float    main_pct   = -1; // guest main thread load (monitor)
	float    others     = -1; // cores used by other processes (monitor)
	float    system_pct = -1; // machine in use, % (monitor)
};

struct Raw {
	uint64_t cycles = 0, gpu = 0, draws = 0, dispatches = 0, shaders = 0, shader_ns = 0, wait_ns = 0, upload = 0,
	         read = 0;
};

Raw ReadRaw() {
	Raw r;
#if defined(_WIN32)
	if (auto* handle = g_main_thread.load(std::memory_order_acquire); handle != nullptr) {
		ULONG64 cycles = 0;
		if (QueryThreadCycleTime(static_cast<HANDLE>(handle), &cycles) != 0) {
			r.cycles = cycles;
		}
	}
#endif
	r.gpu        = GpuTiming::TotalBusyNs();
	r.draws      = DC::Get(DC::Counter::Draws) + DC::Get(DC::Counter::IndirectDraws);
	r.dispatches = DC::Get(DC::Counter::Dispatches);
	r.shaders    = DC::Get(DC::Counter::ShadersCompiled) + DC::Get(DC::Counter::PipelinesCreated);
	r.shader_ns  = DC::Get(DC::Counter::ShaderCompileNs) + DC::Get(DC::Counter::PipelineCreateNs);
	r.wait_ns    = DC::Get(DC::Counter::GpuWaitNs);
	r.upload     = DC::Get(DC::Counter::TextureUploadBytes) + DC::Get(DC::Counter::BufferUploadBytes);
	r.read       = DC::Get(DC::Counter::GuestFileReadBytes);
	return r;
}

uint64_t Tsc() {
#if defined(_WIN32)
	return __rdtsc();
#else
	return 0;
#endif
}

std::string Fixed(double value, int digits) {
	char text[32];
	std::snprintf(text, sizeof(text), "%.*f", digits, value);
	std::string out = text;
	std::replace(out.begin(), out.end(), '.', ','); // French Excel
	return out;
}

constexpr size_t RING       = 256;
constexpr size_t BOX_BEFORE = 30;  // frames kept from before the episode (about 1 s)
constexpr size_t BOX_MAX    = 300; // frames per episode in boite-noire.csv

struct Means {
	double   frame = 0, cpu = 0, gpu = 0, wait = 0, shader_ms = 0;
	uint64_t shaders = 0, read_kb = 0, upload_kb = 0, draws = 0;
	size_t   n = 0;
	double   cp = -1, main = -1, others = -1, system = -1; // monitor means, -1 unknown
};

// Running sums over any number of frames (episode, tranche, trace line): the black box keeps only
// the first BOX_MAX frames of an episode, so v1 means covered its first ~10 s only.
struct Accum {
	Means  m;
	double cp = 0, main = 0, others = 0, system = 0, worst = 0;
	size_t n_cp = 0, n_main = 0, n_others = 0, n_system = 0;
	void   Add(const Sample& s) {
        m.frame += s.frame_ms;
        m.cpu += s.cpu_ms;
        m.gpu += s.gpu_ms;
        m.wait += s.wait_ms;
        m.shader_ms += s.shader_ms;
        m.shaders += s.shaders;
        m.read_kb += s.read_kb;
        m.upload_kb += s.upload_kb;
        m.draws += s.draws;
        m.n++;
        worst = std::max(worst, static_cast<double>(s.frame_ms));
        auto add = [](double& sum, size_t& n, float v) {
            if (v >= 0) {
                sum += v;
                n++;
            }
        };
        add(cp, n_cp, s.cp_pct);
        add(main, n_main, s.main_pct);
        add(others, n_others, s.others);
        add(system, n_system, s.system_pct);
	}
	[[nodiscard]] Means Get() const {
		Means r = m;
		if (r.n != 0) {
			const auto n = static_cast<double>(r.n);
			r.frame /= n;
			r.cpu /= n;
			r.gpu /= n;
			r.wait /= n;
			r.draws /= r.n;
		}
		r.cp     = n_cp != 0 ? cp / static_cast<double>(n_cp) : -1.0;
		r.main   = n_main != 0 ? main / static_cast<double>(n_main) : -1.0;
		r.others = n_others != 0 ? others / static_cast<double>(n_others) : -1.0;
		r.system = n_system != 0 ? system / static_cast<double>(n_system) : -1.0;
		return r;
	}
};

std::string Opt(double value, int digits) {
	return value < 0 ? std::string("-") : Fixed(value, digits);
}

// The probable cause, from the episode's own frames.
const char* Cause(const Means& m, double seconds, double worst_ms, bool gpu_timed, bool cpu_timed) {
	if (m.n == 0) {
		return "?";
	}
	if (m.shader_ms >= 0.25 * seconds * 1000.0) {
		return "shaders";
	}
	if (m.read_kb >= 20u * 1024u && worst_ms >= 100.0) {
		return "disque";
	}
	// v2 (18/11): the stage that takes the largest share of the frame. Measured on Pirate Island at
	// 13-16 fps: no stage at 100 % (Thread_Gpu 70-86 % of the time, GPU 28 of 71 ms, game 14 ms): the
	// stages run one after the other. The old "attente-gpu" (GpuWaitNs, host completion waits
	// included) mostly meant the CP translation.
	if (m.others >= 6.0) {
		return "machine"; // other processes use 6+ cores (baseline here: 2.5-4, window and driver included)
	}
	const double cp_share   = m.cp >= 0 ? m.cp / 100.0 : 0.0;
	const double game_share = cpu_timed && m.frame > 0 ? m.cpu / m.frame : 0.0;
	const double gpu_share  = gpu_timed && m.frame > 0 ? m.gpu / m.frame : 0.0;
	const double top        = std::max({cp_share, game_share, gpu_share});
	if (top >= 0.6) {
		return top == cp_share ? "traduction-cp" : (top == game_share ? "cpu-jeu" : "gpu");
	}
	if (m.wait >= 0.3 * m.frame) {
		return "attente";
	}
	return "mixte";
}

// ---- Episodes ------------------------------------------------------------------------------------

struct State {
	std::array<Clock::time_point, RING> flips {};
	std::array<Sample, RING>            ring {};
	size_t                              count = 0;
	size_t                              next  = 0;
	Clock::time_point                   last {};
	Clock::time_point                   session_start {};
	Raw                                 raw {};
	uint64_t                            tsc0      = 0;
	bool                                cpu_timed = false;
	bool                                gpu_timed = false;

	bool                active = false;
	Clock::time_point   start {};
	Clock::time_point   above_since {};
	Clock::time_point   low_since {};
	Clock::time_point   last_capture {};
	Clock::time_point   last_episode_capture {};
	Clock::time_point   last_report {};
	uint32_t            frames       = 0;
	double              worst_ms     = 0.0;
	double              min_fps      = 1000.0;
	uint32_t            episode_id   = 0;
	uint32_t            images_done  = 0;
	uint32_t            reports_done = 0;
	bool                report       = false;
	std::string         clock;
	std::string         images;
	std::vector<Sample> box;
	size_t              box_start = 0; // first episode frame in box

	Clock::time_point minute_start {};
	uint32_t          minute_frames = 0;
	double            minute_worst  = 0.0;
	uint32_t          minute_slow   = 0; // frames longer than 1000 / threshold ms
	double            minute_cpu    = 0.0;
	double            minute_gpu    = 0.0;

	// v2 (18/11): whole-episode sums, a "tranche" line every TRANCHE of an episode, trace.csv.
	Accum             episode_acc;
	Accum             tranche_acc;
	Clock::time_point tranche_start {};
	double            tranche_min_fps = 1000.0;
	Accum             trace_acc;
	Clock::time_point trace_start {};
	std::string       top; // busiest threads (monitor)
	uint32_t          stacks_done = 0;
	bool              stacks      = false; // taken in this episode
	Clock::time_point last_stacks {};
};

constexpr auto TRANCHE = std::chrono::seconds(5);
constexpr auto TRACE   = std::chrono::milliseconds(500);

State                  g_state; // video-out thread only
std::atomic<uint32_t>  g_capture_request {0};
std::atomic<uint32_t>  g_next_image {1};
std::atomic<bool>      g_mark_request {false};
std::mutex             g_pad_mutex;
std::array<uint8_t, 8> g_pad_sticks {}; // bit 0 L3, bit 1 R3, per pad id

// A thumbnail of the next presented frame: its file name, or "" when none is allowed.
std::string RequestCapture(State& s) {
	if (s.images_done >= Config().images || g_quota_blocked.load(std::memory_order_relaxed)) {
		return {};
	}
	uint32_t       expected = 0;
	const uint32_t number   = g_next_image.fetch_add(1, std::memory_order_relaxed);
	if (!g_capture_request.compare_exchange_strong(expected, number, std::memory_order_acq_rel)) {
		return "img-" + std::to_string(expected) + ".jpg"; // one pending: shared
	}
	s.images_done++;
	return "img-" + std::to_string(number) + ".jpg";
}

bool RequestReport(State& s, Clock::time_point now, std::chrono::seconds gap) {
	if (!Config().gpu_report || s.reports_done >= Config().reports ||
	    g_quota_blocked.load(std::memory_order_relaxed) ||
	    (s.last_report != Clock::time_point {} && now - s.last_report < gap)) {
		return false;
	}
	s.last_report = now;
	s.reports_done++;
	GpuOpProfiler::RequestCapture();
	return true;
}

const char* JOURNAL_HEADER = "n;heure;type;duree_s;images;ips_min_1s;pire_image_ms;cause;image_ms_moy;"
                             "cpu_jeu_ms_moy;gpu_ms_moy;attente_gpu_ms_moy;dessins_moy;shaders_crees;shaders_ms;"
                             "lecture_ko;envoi_gpu_ko;vignettes;rapport_gpu;fichiers_recents;camera;"
                             "fil_cp_pct;fil_jeu_pct;autres_coeurs;machine_pct;fils;t_s";
const char* TRACE_HEADER   = "t_s;heure;ips;pire_image_ms;image_ms_moy;cpu_jeu_ms;gpu_ms;attente_ms;dessins_image;"
                             "fil_cp_pct;fil_jeu_pct;autres_coeurs;machine_pct;episode;camera";

// One journal line (v1 columns, then the v2 ones).
std::string JournalLine(const State& s, const std::string& id, const char* kind, double seconds, uint32_t frames,
                        double min_fps, double worst_ms, const Means& m, const char* cause, const std::string& images,
                        bool report, double t_s) {
	return id + ";" + LocalTime("%H:%M:%S") + ";" + kind + ";" + Fixed(seconds, 1) + ";" + std::to_string(frames) + ";" +
	       (min_fps >= 1000.0 ? std::string("-") : Fixed(min_fps, 1)) + ";" + Fixed(worst_ms, 0) + ";" + cause + ";" +
	       Fixed(m.frame, 1) + ";" + (s.cpu_timed ? Fixed(m.cpu, 1) : std::string("-")) + ";" +
	       (s.gpu_timed ? Fixed(m.gpu, 1) : std::string("-")) + ";" + Fixed(m.wait, 1) + ";" + std::to_string(m.draws) +
	       ";" + std::to_string(m.shaders) + ";" + Fixed(m.shader_ms, 0) + ";" + std::to_string(m.read_kb) + ";" +
	       std::to_string(m.upload_kb) + ";" + images + ";" + (report ? "oui" : "") + ";" + RecentOpens() + ";" +
	       ReadCamera() + ";" + Opt(m.cp, 0) + ";" + Opt(m.main, 0) + ";" + Opt(m.others, 1) + ";" + Opt(m.system, 0) +
	       ";" + s.top + ";" + Fixed(t_s, 1);
}

double SessionSeconds(const State& s, Clock::time_point t) {
	return std::chrono::duration<double>(t - s.session_start).count();
}

// Start of an episode: written at once, with the camera where the drop begins.
void WriteEpisodeStart(State& s, const Sample& first) {
	Accum a;
	a.Add(first);
	AppendLine("journal.csv", JOURNAL_HEADER,
	           JournalLine(s, std::to_string(s.episode_id), "debut", 0.0, 1, 1000.0, first.frame_ms, a.Get(), "", s.images,
	                       false, SessionSeconds(s, s.start)));
}

// Every TRANCHE of an episode: what these seconds were, and where (a long drop crosses places; an
// episode still open when the game closes is no longer lost).
void WriteTranche(State& s, Clock::time_point now) {
	const double seconds = std::chrono::duration<double>(now - s.tranche_start).count();
	const Means  m       = s.tranche_acc.Get();
	AppendLine("journal.csv", JOURNAL_HEADER,
	           JournalLine(s, std::to_string(s.episode_id), "tranche", seconds, static_cast<uint32_t>(m.n),
	                       s.tranche_min_fps, s.tranche_acc.worst, m,
	                       Cause(m, seconds, s.tranche_acc.worst, s.gpu_timed, s.cpu_timed), "", false,
	                       SessionSeconds(s, now)));
	s.tranche_acc     = {};
	s.tranche_start   = now;
	s.tranche_min_fps = 1000.0;
}
const char* BOX_HEADER     = "n;t_s;image_ms;cpu_jeu_ms;gpu_ms;attente_gpu_ms;dessins;calculs;shaders_crees;"
                             "shaders_ms;envoi_gpu_ko;lecture_ko;dans_episode";

void WriteEpisode(State& s, Clock::time_point end) {
	const double seconds = std::chrono::duration<double>(end - s.start).count();
	const char*  kind    = s.worst_ms >= 1000.0 ? "arret" : (s.min_fps >= 1000.0 ? "pic" : "baisse");
	const Means  m       = s.episode_acc.Get(); // v2: every frame of the episode, not the first BOX_MAX
	const char*  cause   = Cause(m, seconds, s.worst_ms, s.gpu_timed, s.cpu_timed);
	// The summary keeps the episode's start time in "heure" (v1); its camera is the end's.
	std::string line = JournalLine(s, std::to_string(s.episode_id), kind, seconds, s.frames, s.min_fps, s.worst_ms, m,
	                               cause, s.images, s.report, SessionSeconds(s, end));
	line.replace(line.find(';') + 1, 8, s.clock);
	AppendLine("journal.csv", JOURNAL_HEADER, line);

	std::string box;
	for (size_t i = 0; i < s.box.size(); i++) {
		const auto& f = s.box[i];
		box += std::to_string(s.episode_id) + ";" + Fixed(f.t_s, 2) + ";" + Fixed(f.frame_ms, 1) + ";" +
		       Fixed(f.cpu_ms, 1) + ";" + Fixed(f.gpu_ms, 1) + ";" + Fixed(f.wait_ms, 1) + ";" +
		       std::to_string(f.draws) + ";" + std::to_string(f.dispatches) + ";" + std::to_string(f.shaders) + ";" +
		       Fixed(f.shader_ms, 1) + ";" + std::to_string(f.upload_kb) + ";" + std::to_string(f.read_kb) + ";" +
		       (i >= s.box_start ? "1" : "0");
		if (i + 1 < s.box.size()) {
			box += "\r\n";
		}
	}
	if (!box.empty()) {
		AppendLine("boite-noire.csv", BOX_HEADER, box);
	}
}

void WriteMark(State& s, Clock::time_point now) {
	const std::string image  = RequestCapture(s);
	const bool        report = RequestReport(s, now, std::chrono::seconds(60));
	const std::string line   = "M;" + LocalTime("%H:%M:%S") + ";marque;-;-;-;-;-;-;-;-;-;-;-;-;-;-;" + image + ";" +
	                         (report ? "oui" : "") + ";" + RecentOpens() + ";" + ReadCamera();
	AppendLine("journal.csv", JOURNAL_HEADER, line);
}

} // namespace

bool FluidityEnabled() noexcept {
	return Config().enabled;
}

bool FluidityCamera(float* position, float* direction) noexcept {
	std::array<float, 32> values {};
	size_t                floats = 0;
	if (*ReadCameraFloats(values, floats) != 0 || floats < 7) {
		return false;
	}
	for (int i = 0; i < 3; i++) {
		position[i]  = values[i];
		direction[i] = values[4 + i];
	}
	return !(position[0] == 0.0f && position[1] == 0.0f && position[2] == 0.0f); // not read yet (menus)
}

void FluidityRecordFlip() noexcept {
	if (!Config().enabled) {
		return;
	}
	auto&      s   = g_state;
	const auto now = Clock::now();
	if (s.session_start == Clock::time_point {}) {
		DC::SetCollect(true);
		s.session_start = now;
		s.minute_start  = now;
		s.tsc0          = Tsc();
		s.raw           = ReadRaw();
		s.box.reserve(BOX_BEFORE + BOX_MAX);
		s.trace_start = now;
		StartMonitor();
		{ // reglages.txt: every KYTY_* setting of this session, to compare sessions place by place
			std::string text;
			if (char** env = _environ; env != nullptr) {
				for (; *env != nullptr; ++env) {
					if (std::strncmp(*env, "KYTY_", 5) == 0) {
						text += *env;
						text += "\r\n";
					}
				}
			}
			GetWriter().Push({SessionFile("reglages.txt"), std::vector<uint8_t>(text.begin(), text.end()), false, false});
		}
		AppendLine("journal.csv", JOURNAL_HEADER,
		           "S;" + LocalTime("%H:%M:%S") + ";session;seuil " + Fixed(Config().threshold, 0) + " i/s, pic " +
		               Fixed(Config().spike_ms, 0) + " ms;;;;;;;;;;;;;;;;" +
		               (Config().gpu_report ? "rapports permis" : "") + ";;");
	}
	const double dt_ms =
	    s.last == Clock::time_point {} ? 0.0 : std::chrono::duration<double, std::milli>(now - s.last).count();
	s.last = now;

	// Black-box sample.
	const Raw    raw  = ReadRaw();
	const double ns_s = std::chrono::duration<double, std::nano>(now - s.session_start).count();
	const double tsc_per_ns = ns_s > 1e8 && s.tsc0 != 0 ? static_cast<double>(Tsc() - s.tsc0) / ns_s : 0.0;
	Sample       sample;
	sample.t_s      = static_cast<float>(ns_s * 1e-9);
	sample.frame_ms = static_cast<float>(dt_ms);
	sample.cpu_ms   = tsc_per_ns > 0.0 && raw.cycles >= s.raw.cycles && s.raw.cycles != 0
	                      ? static_cast<float>(static_cast<double>(raw.cycles - s.raw.cycles) / tsc_per_ns * 1e-6)
	                      : 0.0f;
	sample.gpu_ms     = static_cast<float>(static_cast<double>(raw.gpu - s.raw.gpu) * 1e-6);
	sample.draws      = static_cast<uint32_t>(raw.draws - s.raw.draws);
	sample.dispatches = static_cast<uint32_t>(raw.dispatches - s.raw.dispatches);
	sample.shaders    = static_cast<uint32_t>(raw.shaders - s.raw.shaders);
	sample.shader_ms  = static_cast<float>(static_cast<double>(raw.shader_ns - s.raw.shader_ns) * 1e-6);
	sample.wait_ms    = static_cast<float>(static_cast<double>(raw.wait_ns - s.raw.wait_ns) * 1e-6);
	sample.upload_kb  = static_cast<uint32_t>((raw.upload - s.raw.upload) >> 10u);
	sample.read_kb    = static_cast<uint32_t>((raw.read - s.raw.read) >> 10u);
	{
		Load load         = CurrentLoad();
		sample.cp_pct     = load.cp_pct;
		sample.main_pct   = load.main_pct;
		sample.others     = load.others;
		sample.system_pct = load.system_pct;
		s.top             = std::move(load.top);
	}
	s.cpu_timed       = s.cpu_timed || raw.cycles != 0;
	s.gpu_timed       = s.gpu_timed || raw.gpu != 0;
	s.raw             = raw;

	s.flips[s.next] = now;
	s.ring[s.next]  = sample;
	s.next          = (s.next + 1) % RING;
	s.count         = std::min(s.count + 1, RING);

	if (g_mark_request.exchange(false, std::memory_order_acq_rel)) {
		WriteMark(s, now);
	}

	// Flips in the last second (newest first; the ring holds at most RING of them).
	uint32_t in_second = 0;
	for (size_t i = 0; i < s.count; i++) {
		if (now - s.flips[(s.next + RING - 1 - i) % RING] > std::chrono::seconds(1)) {
			break;
		}
		in_second++;
	}
	const bool   warm = now - s.session_start > std::chrono::seconds(2);
	const double fps  = static_cast<double>(in_second);

	// Minute summary.
	s.minute_frames++;
	s.minute_worst = std::max(s.minute_worst, dt_ms);
	s.minute_cpu += sample.cpu_ms;
	s.minute_gpu += sample.gpu_ms;
	if (dt_ms > 1000.0 / Config().threshold) {
		s.minute_slow++;
	}
	if (now - s.minute_start >= std::chrono::minutes(1)) {
		const double seconds = std::chrono::duration<double>(now - s.minute_start).count();
		const double n       = std::max<double>(1.0, s.minute_frames);
		AppendLine("minutes.csv", "heure;ips_moyen;images_lentes;pire_image_ms;cpu_jeu_ms_moy;gpu_ms_moy;fichiers_recents",
		           LocalTime("%H:%M:%S") + ";" + Fixed(s.minute_frames / seconds, 1) + ";" +
		               std::to_string(s.minute_slow) + ";" + Fixed(s.minute_worst, 0) + ";" +
		               (s.cpu_timed ? Fixed(s.minute_cpu / n, 1) : std::string("-")) + ";" +
		               (s.gpu_timed ? Fixed(s.minute_gpu / n, 1) : std::string("-")) + ";" + RecentOpens());
		s.minute_start  = now;
		s.minute_frames = 0;
		s.minute_worst  = 0.0;
		s.minute_slow   = 0;
		s.minute_cpu    = 0.0;
		s.minute_gpu    = 0.0;
	}

	// trace.csv: every TRACE, in or out of an episode: the base of place-by-place comparisons.
	s.trace_acc.Add(sample);
	if (now - s.trace_start >= TRACE) {
		const double seconds = std::chrono::duration<double>(now - s.trace_start).count();
		const Means  m       = s.trace_acc.Get();
		AppendLine("trace.csv", TRACE_HEADER,
		           Fixed(SessionSeconds(s, now), 1) + ";" + LocalTime("%H:%M:%S") + ";" +
		               Fixed(static_cast<double>(m.n) / seconds, 1) + ";" + Fixed(s.trace_acc.worst, 0) + ";" +
		               Fixed(m.frame, 1) + ";" + (s.cpu_timed ? Fixed(m.cpu, 1) : std::string("-")) + ";" +
		               (s.gpu_timed ? Fixed(m.gpu, 1) : std::string("-")) + ";" + Fixed(m.wait, 1) + ";" +
		               std::to_string(m.draws) + ";" + Opt(m.cp, 0) + ";" + Opt(m.main, 0) + ";" + Opt(m.others, 1) + ";" +
		               Opt(m.system, 0) + ";" + (s.active ? std::to_string(s.episode_id) : std::string()) + ";" +
		               ReadCamera());
		s.trace_acc   = {};
		s.trace_start = now;
	}

	const bool spike = dt_ms >= Config().spike_ms;
	const bool low   = warm && fps < Config().threshold;
	if (!s.active) {
		if (!spike && !low) {
			return;
		}
		s.active      = true;
		s.start       = now;
		s.above_since = {};
		s.low_since   = {};
		s.frames      = 0;
		s.worst_ms    = 0.0;
		s.min_fps     = 1000.0;
		s.report      = false;
		s.episode_id++;
		s.clock = LocalTime("%H:%M:%S");
		s.images.clear();
		// The black box: about 1 s before the episode (the current frame is added below).
		s.box.clear();
		const size_t before = std::min(BOX_BEFORE, s.count > 0 ? s.count - 1 : 0);
		for (size_t i = before; i > 0; i--) {
			s.box.push_back(s.ring[(s.next + RING - 1 - i) % RING]);
		}
		s.box_start = s.box.size();
		if (s.last_capture == Clock::time_point {} ||
		    now - s.last_capture >= std::chrono::duration<double>(Config().gap_s)) {
			s.last_capture         = now;
			s.last_episode_capture = now;
			s.images               = RequestCapture(s);
		}
		s.stacks          = false;
		s.episode_acc     = {};
		s.tranche_acc     = {};
		s.tranche_start   = now;
		s.tranche_min_fps = 1000.0;
		WriteEpisodeStart(s, sample);
	}
	s.frames++;
	s.episode_acc.Add(sample);
	s.tranche_acc.Add(sample);
	if (low) {
		s.tranche_min_fps = std::min(s.tranche_min_fps, fps);
	}
	if (now - s.tranche_start >= TRANCHE) {
		WriteTranche(s, now);
	}
	s.worst_ms = std::max(s.worst_ms, dt_ms);
	if (s.box.size() < BOX_BEFORE + BOX_MAX) {
		s.box.push_back(sample);
	}
	if (low) {
		s.min_fps = std::min(s.min_fps, fps);
		if (s.low_since == Clock::time_point {}) {
			s.low_since = now;
		}
		// v2: automatic stacks, once per episode, 60 s apart.
		if (!s.stacks && !Config().sampler.empty() && s.stacks_done < Config().stacks &&
		    now - s.low_since >= std::chrono::seconds(3) && dt_ms < 1000.0 &&
		    (s.last_stacks == Clock::time_point {} || now - s.last_stacks >= std::chrono::seconds(60))) {
			s.stacks      = true;
			s.last_stacks = now;
			s.stacks_done++;
			LaunchStacks(s.episode_id);
		}
		// A sustained drop (not a hitch nor a loading stop): one GPU frame report.
		if (!s.report && now - s.low_since >= std::chrono::seconds(2) && dt_ms < 1000.0) {
			s.report = RequestReport(s, now, std::chrono::seconds(180));
		}
	} else {
		s.low_since = {};
	}
	if (s.last_episode_capture != Clock::time_point {} && now - s.last_episode_capture >= std::chrono::seconds(10)) {
		s.last_capture         = now;
		s.last_episode_capture = now;
		if (auto image = RequestCapture(s); !image.empty()) {
			s.images += (s.images.empty() ? "" : " ") + image;
		}
	}
	if (spike || low) {
		s.above_since = {};
		return;
	}
	if (s.above_since == Clock::time_point {}) {
		s.above_since = now;
		return;
	}
	if (now - s.above_since >= std::chrono::milliseconds(1500)) {
		WriteEpisode(s, s.above_since);
		s.active = false;
	}
}

void FluidityMark() noexcept {
	if (Config().enabled) {
		g_mark_request.store(true, std::memory_order_release);
	}
}

void FluidityNotePadButton(int id, bool left_stick, bool down) noexcept {
	if (!Config().enabled || id < 0) {
		return;
	}
	std::scoped_lock lock(g_pad_mutex);
	auto&            bits   = g_pad_sticks[static_cast<size_t>(id) % g_pad_sticks.size()];
	const uint8_t    before = bits;
	const uint8_t    bit    = left_stick ? 1u : 2u;
	bits                    = down ? static_cast<uint8_t>(bits | bit) : static_cast<uint8_t>(bits & ~bit);
	if (before != 3u && bits == 3u) {
		g_mark_request.store(true, std::memory_order_release);
	}
}

void FluidityNoteOpen(const char* path) noexcept {
	if (!Config().enabled || path == nullptr) {
		return;
	}
	if (std::strncmp(path, "/app0/", 6) != 0) {
		return;
	}
	std::scoped_lock lock(g_open_mutex);
	const auto&      previous = g_opens[(g_open_next + g_opens.size() - 1) % g_opens.size()];
	if (previous == path + 6) {
		return;
	}
	g_opens[g_open_next] = path + 6;
	g_open_next          = (g_open_next + 1) % g_opens.size();
}

void FluiditySetMainThread(void* handle) noexcept {
	void* expected = nullptr;
	if (!g_main_thread.compare_exchange_strong(expected, handle, std::memory_order_acq_rel)) {
#if defined(_WIN32)
		CloseHandle(static_cast<HANDLE>(handle));
#endif
	}
}

void FluiditySetEbootBase(uint64_t base) noexcept {
	g_eboot_base.store(base, std::memory_order_release);
}

bool FluidityTakeCaptureRequest(uint32_t* number) noexcept {
	if (!Config().enabled) {
		return false;
	}
	const uint32_t value = g_capture_request.exchange(0, std::memory_order_acq_rel);
	if (value == 0) {
		return false;
	}
	*number = value;
	return true;
}

std::string FluiditySessionDir() {
	return Config().dir;
}

std::string FluidityImagePath(uint32_t number) {
	return SessionFile("img-" + std::to_string(number) + ".jpg");
}

void FluidityWriteFile(std::string path, std::vector<uint8_t> bytes) {
	GetWriter().Push({std::move(path), std::move(bytes), false, true});
}

} // namespace Libs::Graphics
