#include "graphics/presentation/fluidityLog.h"

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

std::string ReadCamera() {
	const auto& spec = Config().camera;
	if (spec.empty()) {
		return {};
	}
	std::vector<uint64_t> chain;
	size_t                floats = 16;
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
	std::array<float, 32> values {};
	if (!ReadOwn(address, values.data(), floats * sizeof(float))) {
		return "?";
	}
	std::string text;
	char        number[32];
	for (size_t i = 0; i < floats; i++) {
		std::snprintf(number, sizeof(number), "%s%.4g", i == 0 ? "" : " ", values[i]);
		text += number;
	}
	return text;
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
};

Means Average(const std::vector<Sample>& box, size_t from) {
	Means m;
	for (size_t i = from; i < box.size(); i++) {
		const auto& s = box[i];
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
	}
	if (m.n != 0) {
		const auto n = static_cast<double>(m.n);
		m.frame /= n;
		m.cpu /= n;
		m.gpu /= n;
		m.wait /= n;
		m.draws /= m.n;
	}
	return m;
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
	if (gpu_timed && m.gpu >= 0.85 * m.frame) {
		return "gpu";
	}
	if (cpu_timed && m.cpu >= 0.85 * m.frame) {
		return "cpu-jeu";
	}
	if (m.wait >= 0.3 * m.frame) {
		return "attente-gpu";
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
};

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
                             "lecture_ko;envoi_gpu_ko;vignettes;rapport_gpu;fichiers_recents;camera";
const char* BOX_HEADER     = "n;t_s;image_ms;cpu_jeu_ms;gpu_ms;attente_gpu_ms;dessins;calculs;shaders_crees;"
                             "shaders_ms;envoi_gpu_ko;lecture_ko;dans_episode";

void WriteEpisode(State& s, Clock::time_point end) {
	const double seconds = std::chrono::duration<double>(end - s.start).count();
	const char*  kind    = s.worst_ms >= 1000.0 ? "arret" : (s.min_fps >= 1000.0 ? "pic" : "baisse");
	const Means  m       = Average(s.box, s.box_start);
	const char*  cause   = Cause(m, seconds, s.worst_ms, s.gpu_timed, s.cpu_timed);
	std::string  line =
	    std::to_string(s.episode_id) + ";" + s.clock + ";" + kind + ";" + Fixed(seconds, 1) + ";" +
	    std::to_string(s.frames) + ";" + (s.min_fps >= 1000.0 ? std::string("-") : Fixed(s.min_fps, 1)) + ";" +
	    Fixed(s.worst_ms, 0) + ";" + cause + ";" + Fixed(m.frame, 1) + ";" +
	    (s.cpu_timed ? Fixed(m.cpu, 1) : std::string("-")) + ";" + (s.gpu_timed ? Fixed(m.gpu, 1) : std::string("-")) +
	    ";" + Fixed(m.wait, 1) + ";" + std::to_string(m.draws) + ";" + std::to_string(m.shaders) + ";" +
	    Fixed(m.shader_ms, 0) + ";" + std::to_string(m.read_kb) + ";" + std::to_string(m.upload_kb) + ";" + s.images +
	    ";" + (s.report ? "oui" : "") + ";" + RecentOpens() + ";" + ReadCamera();
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
	}
	s.frames++;
	s.worst_ms = std::max(s.worst_ms, dt_ms);
	if (s.box.size() < BOX_BEFORE + BOX_MAX) {
		s.box.push_back(sample);
	}
	if (low) {
		s.min_fps = std::min(s.min_fps, fps);
		if (s.low_since == Clock::time_point {}) {
			s.low_since = now;
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
