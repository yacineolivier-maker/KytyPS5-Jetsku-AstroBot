#include "graphics/presentation/fluidityLog.h"

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
#include <windows.h>
#endif

// See fluidityLog.h. Nothing here runs unless KYTY_FLUIDITE=1.

namespace Libs::Graphics {

namespace {

using Clock = std::chrono::steady_clock;

double EnvNumber(const char* name, double fallback) {
	const char* value = std::getenv(name);
	if (value == nullptr || value[0] == 0) {
		return fallback;
	}
	char*        end    = nullptr;
	const double parsed = std::strtod(value, &end);
	return end != value && parsed > 0.0 ? parsed : fallback;
}

struct Settings {
	bool        enabled   = false;
	double      spike_ms  = 50.0;
	double      threshold = 28.0;
	double      gap_s     = 5.0;
	std::string camera; // KYTY_FLUIDITE_CAMERA, see ReadCamera
	std::string dir;
};

const Settings& Config() {
	static const Settings settings = [] {
		Settings s;
		const char* on = std::getenv("KYTY_FLUIDITE");
		s.enabled      = on != nullptr && on[0] == '1';
		s.spike_ms     = EnvNumber("KYTY_FLUIDITE_PIC_MS", 50.0);
		s.threshold    = EnvNumber("KYTY_FLUIDITE_SEUIL", 28.0);
		s.gap_s        = EnvNumber("KYTY_FLUIDITE_ECART_S", 5.0);
		if (const char* cam = std::getenv("KYTY_FLUIDITE_CAMERA"); cam != nullptr) {
			s.camera = cam;
		}
		std::filesystem::path base;
		if (const char* dir = std::getenv("KYTY_FLUIDITE_DIR"); dir != nullptr && dir[0] != 0) {
			base = dir;
		} else {
#if defined(_WIN32)
			wchar_t exe[MAX_PATH] {};
			if (GetModuleFileNameW(nullptr, exe, MAX_PATH) != 0) {
				base = std::filesystem::path(exe).parent_path() / "_Fluidite";
			}
#endif
			if (base.empty()) {
				base = "_Fluidite";
			}
		}
		const std::time_t now = std::time(nullptr);
		std::tm           local {};
#if defined(_WIN32)
		localtime_s(&local, &now);
#else
		localtime_r(&now, &local);
#endif
		char name[32] {};
		std::strftime(name, sizeof(name), "%Y%m%d-%H%M%S", &local);
		s.dir = (base / name).string();
		return s;
	}();
	return settings;
}

std::string WallClock() {
	const std::time_t now = std::time(nullptr);
	std::tm           local {};
#if defined(_WIN32)
	localtime_s(&local, &now);
#else
	localtime_r(&now, &local);
#endif
	char text[16] {};
	std::strftime(text, sizeof(text), "%H:%M:%S", &local);
	return text;
}

// ---- Worker thread: every disk write -------------------------------------------------------------

struct Job {
	std::string          path;
	std::vector<uint8_t> bytes;
	bool                 append = false;
};

class Writer {
public:
	void Push(Job job) {
		{
			std::scoped_lock lock(m_mutex);
			if (m_jobs.size() > 64) { // never let a stuck disk hold memory
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
		std::filesystem::create_directories(Config().dir, error);
		for (;;) {
			Job job;
			{
				std::unique_lock lock(m_mutex);
				m_wake.wait(lock, [this] { return !m_jobs.empty(); });
				job = std::move(m_jobs.front());
				m_jobs.pop_front();
			}
			std::ofstream file(job.path, std::ios::binary | (job.append ? std::ios::app : std::ios::trunc));
			file.write(reinterpret_cast<const char*>(job.bytes.data()), static_cast<std::streamsize>(job.bytes.size()));
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

std::string SessionFile(const char* name) {
	return (std::filesystem::path(Config().dir) / name).string();
}

void AppendLine(const char* file, const std::string& header, const std::string& line) {
	static std::mutex mutex;
	static bool       journal_started = false;
	static bool       minutes_started = false;
	std::string       text;
	{
		std::scoped_lock lock(mutex);
		bool& started = std::strcmp(file, "journal.csv") == 0 ? journal_started : minutes_started;
		if (!started) {
			started = true;
			text    = "\xEF\xBB\xBF" + header + "\r\n"; // UTF-8 BOM for Excel
		}
	}
	text += line + "\r\n";
	GetWriter().Push({SessionFile(file), std::vector<uint8_t>(text.begin(), text.end()), true});
}

// ---- Recent guest file opens ---------------------------------------------------------------------

std::mutex               g_open_mutex;
std::array<std::string, 6> g_opens;
size_t                   g_open_next = 0;

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

std::atomic<uint64_t> g_eboot_base {0};

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
	if (chain.size() > 1) {
		for (size_t i = 1; i < chain.size(); i++) {
			uint64_t pointer = 0;
			if (!ReadOwn(address, &pointer, sizeof(pointer))) {
				return "?";
			}
			address = pointer + chain[i];
		}
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

// ---- Episodes ------------------------------------------------------------------------------------

constexpr size_t RING = 256;

struct State {
	std::array<Clock::time_point, RING> flips {};
	size_t                              count = 0;
	size_t                              next  = 0;
	Clock::time_point                   last {};
	Clock::time_point                   session_start {};

	bool              active = false;
	Clock::time_point start {};
	Clock::time_point above_since {};
	Clock::time_point last_capture {};
	Clock::time_point last_episode_capture {};
	uint32_t          frames     = 0;
	double            worst_ms   = 0.0;
	double            min_fps    = 1000.0;
	uint32_t          episode_id = 0;
	std::string       clock;
	std::string       images;

	Clock::time_point minute_start {};
	uint32_t          minute_frames = 0;
	double            minute_worst  = 0.0;
	uint32_t          minute_slow   = 0; // frames longer than 1000 / threshold ms
};

State                 g_state; // video-out thread only
std::atomic<uint32_t> g_capture_request {0};
std::atomic<uint32_t> g_next_image {1};
std::atomic<bool>     g_mark_request {false};
std::mutex            g_pad_mutex;
std::array<uint8_t, 8> g_pad_sticks {}; // bit 0 L3, bit 1 R3, per pad id

uint32_t RequestCapture() {
	const uint32_t number = g_next_image.fetch_add(1, std::memory_order_relaxed);
	uint32_t       expected = 0;
	// One request at a time: a pending one is kept.
	if (!g_capture_request.compare_exchange_strong(expected, number, std::memory_order_acq_rel)) {
		return expected;
	}
	return number;
}

std::string Fixed(double value, int digits) {
	char text[32];
	std::snprintf(text, sizeof(text), "%.*f", digits, value);
	std::string out = text;
	std::replace(out.begin(), out.end(), '.', ','); // French Excel
	return out;
}

const char* JOURNAL_HEADER = "n;heure;type;duree_s;images;ips_min_1s;pire_image_ms;vignettes;fichiers_recents;camera";

void WriteEpisode(State& s, Clock::time_point now) {
	const double seconds = std::chrono::duration<double>(now - s.start).count();
	const char*  kind    = s.worst_ms >= 1000.0 ? "arret" : (s.min_fps >= 1000.0 ? "pic" : "baisse");
	std::string  line    = std::to_string(s.episode_id) + ";" + s.clock + ";" + kind + ";" + Fixed(seconds, 1) + ";" +
	                   std::to_string(s.frames) + ";" + (s.min_fps >= 1000.0 ? std::string("-") : Fixed(s.min_fps, 1)) +
	                   ";" + Fixed(s.worst_ms, 0) + ";" + s.images + ";" + RecentOpens() + ";" + ReadCamera();
	AppendLine("journal.csv", JOURNAL_HEADER, line);
}

void WriteMark() {
	const uint32_t image = RequestCapture();
	std::string    line  = "M;" + WallClock() + ";marque;-;-;-;-;img-" + std::to_string(image) + ".png;" +
	                   RecentOpens() + ";" + ReadCamera();
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
		s.session_start = now;
		s.minute_start  = now;
	}
	if (g_mark_request.exchange(false, std::memory_order_acq_rel)) {
		WriteMark();
	}
	const double dt_ms = s.last == Clock::time_point {} ? 0.0 : std::chrono::duration<double, std::milli>(now - s.last).count();
	s.last             = now;
	s.flips[s.next]    = now;
	s.next             = (s.next + 1) % RING;
	s.count            = std::min(s.count + 1, RING);

	// Flips in the last second (newest first; the ring holds at most RING of them).
	uint32_t in_second = 0;
	for (size_t i = 0; i < s.count; i++) {
		const auto t = s.flips[(s.next + RING - 1 - i) % RING];
		if (now - t > std::chrono::seconds(1)) {
			break;
		}
		in_second++;
	}
	const bool   warm = now - s.session_start > std::chrono::seconds(2);
	const double fps  = static_cast<double>(in_second);

	// Minute summary.
	s.minute_frames++;
	s.minute_worst = std::max(s.minute_worst, dt_ms);
	if (dt_ms > 1000.0 / Config().threshold) {
		s.minute_slow++;
	}
	if (now - s.minute_start >= std::chrono::minutes(1)) {
		const double seconds = std::chrono::duration<double>(now - s.minute_start).count();
		AppendLine("minutes.csv", "heure;ips_moyen;images_lentes;pire_image_ms;fichiers_recents",
		           WallClock() + ";" + Fixed(s.minute_frames / seconds, 1) + ";" + std::to_string(s.minute_slow) + ";" +
		               Fixed(s.minute_worst, 0) + ";" + RecentOpens());
		s.minute_start  = now;
		s.minute_frames = 0;
		s.minute_worst  = 0.0;
		s.minute_slow   = 0;
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
		s.frames      = 0;
		s.worst_ms    = 0.0;
		s.min_fps     = 1000.0;
		s.episode_id++;
		s.clock  = WallClock();
		s.images.clear();
		if (s.last_capture == Clock::time_point {} ||
		    now - s.last_capture >= std::chrono::duration<double>(Config().gap_s)) {
			s.last_capture         = now;
			s.last_episode_capture = now;
			s.images               = "img-" + std::to_string(RequestCapture()) + ".png";
		}
	}
	s.frames++;
	s.worst_ms = std::max(s.worst_ms, dt_ms);
	if (low) {
		s.min_fps = std::min(s.min_fps, fps);
	}
	if (s.last_episode_capture != Clock::time_point {} && now - s.last_episode_capture >= std::chrono::seconds(10)) {
		s.last_capture         = now;
		s.last_episode_capture = now;
		s.images += (s.images.empty() ? "" : " ") + std::string("img-") + std::to_string(RequestCapture()) + ".png";
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

std::string FluidityImagePath(uint32_t number) {
	return SessionFile(("img-" + std::to_string(number) + ".png").c_str());
}

void FluiditySetEbootBase(uint64_t base) noexcept {
	g_eboot_base.store(base, std::memory_order_release);
}

void FluidityWriteFile(std::string path, std::vector<uint8_t> bytes) {
	GetWriter().Push({std::move(path), std::move(bytes), false});
}

} // namespace Libs::Graphics
