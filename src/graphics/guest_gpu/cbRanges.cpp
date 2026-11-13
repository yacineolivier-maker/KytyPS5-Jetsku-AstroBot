#include "graphics/guest_gpu/cbRanges.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace Libs::Graphics::CbRanges {

namespace {

constexpr size_t MaxRanges = 64;

std::array<std::atomic<uint64_t>, MaxRanges> g_begin {};
std::array<std::atomic<uint64_t>, MaxRanges> g_end {};
std::atomic<uint32_t>                        g_count {0};
std::mutex                                   g_mutex;

struct Window {
	int64_t  start = 0;
	uint64_t in_n = 0, out_n = 0, in_ns = 0, out_ns = 0, draws = 0;
};
Window g_window; // fil GPU

int64_t NowMs() {
	return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch())
	    .count();
}

} // namespace

bool Enabled() {
	static const bool enabled = [] {
		const char* v = std::getenv("KYTY_CB_CONST_STATS");
		return v != nullptr && std::strcmp(v, "1") == 0;
	}();
	return enabled;
}

void Note(const void* bottom, const void* top) {
	if (!Enabled()) {
		return;
	}
	const auto b = reinterpret_cast<uint64_t>(bottom);
	const auto t = reinterpret_cast<uint64_t>(top);
	const auto n = g_count.load(std::memory_order_acquire);
	for (uint32_t i = 0; i < n; i++) {
		if (g_begin[i].load(std::memory_order_relaxed) == b) {
			return;
		}
	}
	std::lock_guard lock(g_mutex);
	const auto m = g_count.load(std::memory_order_relaxed);
	for (uint32_t i = 0; i < m; i++) {
		if (g_begin[i].load(std::memory_order_relaxed) == b) {
			return;
		}
	}
	if (m >= MaxRanges) {
		return;
	}
	g_begin[m].store(b, std::memory_order_relaxed);
	g_end[m].store(t, std::memory_order_relaxed);
	g_count.store(m + 1, std::memory_order_release);
	std::printf("CbRanges: zone de liste %u : 0x%016" PRIx64 "-0x%016" PRIx64 " (%.1f Mio)\n", m, b, t,
	            static_cast<double>(t - b) / 1048576.0);
	std::fflush(stdout);
}

bool Contains(uint64_t address) {
	const auto n = g_count.load(std::memory_order_acquire);
	for (uint32_t i = 0; i < n; i++) {
		if (address >= g_begin[i].load(std::memory_order_relaxed) && address < g_end[i].load(std::memory_order_relaxed)) {
			return true;
		}
	}
	return false;
}

void NoteBinding(bool in_command_buffer, uint64_t ns) {
	auto& w = g_window;
	if (in_command_buffer) {
		w.in_n++;
		w.in_ns += ns;
	} else {
		w.out_n++;
		w.out_ns += ns;
	}
	const auto now = NowMs();
	if (w.start == 0) {
		w.start = now;
		return;
	}
	if (now - w.start < 10000) {
		return;
	}
	const double total = static_cast<double>(std::max<uint64_t>(1, w.in_n + w.out_n));
	std::printf("CbConst 10s: tampons lies %" PRIu64 " ; dans une liste de commandes du jeu %" PRIu64
	            " (%.0f %%), %.3f us chacun, %.1f ms en tout ; ailleurs %" PRIu64 " (%.0f %%), %.3f us chacun, %.1f ms\n",
	            w.in_n + w.out_n, w.in_n, 100.0 * static_cast<double>(w.in_n) / total,
	            w.in_n ? static_cast<double>(w.in_ns) / 1000.0 / static_cast<double>(w.in_n) : 0.0,
	            static_cast<double>(w.in_ns) / 1e6, w.out_n, 100.0 * static_cast<double>(w.out_n) / total,
	            w.out_n ? static_cast<double>(w.out_ns) / 1000.0 / static_cast<double>(w.out_n) : 0.0,
	            static_cast<double>(w.out_ns) / 1e6);
	std::fflush(stdout);
	w = Window {};
	w.start = now;
}

} // namespace Libs::Graphics::CbRanges
