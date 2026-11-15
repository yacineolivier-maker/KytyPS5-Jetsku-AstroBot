#include "graphics/host_gpu/renderer/drawPrep/laneShare.h"

#include "graphics/guest_gpu/command_processor/cpParallel.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

// Astro Bot PC (KYTY_CP_PAR=2b, P3 prerequisites, laneShare.h).
namespace Libs::Graphics::LaneShare {

namespace {

constexpr uint32_t kLanes = 4;
constexpr uint32_t kStats = static_cast<uint32_t>(Stat::Count);

constexpr std::array<const char*, kStats> kNames {
    "stream rings", "stream maps", "stream wraps", "stream overlaps", "stream foreign",
    "descriptor heaps", "descriptor sets", "descriptor pools", "descriptor double", "descriptor early",
    "touches deferred", "touches flushed", "touches stale",
};

struct alignas(64) LaneCounts {
	std::array<std::atomic<uint64_t>, kStats> values {};
};

std::array<LaneCounts, kLanes>                     g_counts;
std::array<std::array<uint64_t, kStats>, kLanes>   g_previous {}; // g_print_mutex
std::atomic<uint64_t>                              g_last_ns {0};
std::mutex                                         g_print_mutex;

bool Switch(const char* name) {
	const char* value = std::getenv(name);
	return value != nullptr && std::strcmp(value, "1") == 0;
}

uint64_t NowNs() noexcept {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

void Print(uint64_t elapsed_ns) {
	const std::lock_guard lock(g_print_mutex);
	std::printf("CpPar 10s: 2b lane share (%.0fs; stream %s, descriptors %s, prereqs %s, verify %s):",
	            static_cast<double>(elapsed_ns) * 1e-9, g_stream ? "on" : "off",
	            g_descriptors ? "on" : "off", g_prereqs ? "on" : "off", g_verify ? "on" : "off");
	for (uint32_t lane = 0; lane < kLanes; lane++) {
		std::string text;
		for (uint32_t i = 0; i < kStats; i++) {
			const auto now   = g_counts[lane].values[i].load(std::memory_order_relaxed);
			const auto delta = now - g_previous[lane][i];
			g_previous[lane][i] = now;
			if (delta != 0) {
				text += (text.empty() ? " " : ", ") + std::string(kNames[i]) + " " + std::to_string(delta);
			}
		}
		if (!text.empty()) {
			std::printf(" %c:%s;", static_cast<char>('A' + lane), text.c_str());
		}
	}
	std::printf("\n");
	std::fflush(stdout);
}

} // namespace

void Configure(bool lanes_2b) {
	const bool stream      = Switch("KYTY_CP_PAR_LANE_STREAM");
	const bool descriptors = Switch("KYTY_CP_PAR_LANE_DESCRIPTORS");
	const bool prereqs     = Switch("KYTY_CP_PAR_LANE_PREREQS");
	if (!stream && !descriptors && !prereqs) {
		return;
	}
	if (!lanes_2b) {
		std::printf("Kyty CpPar: KYTY_CP_PAR_LANE_STREAM/_DESCRIPTORS/_PREREQS need KYTY_CP_PAR=2b: off\n");
		return;
	}
	g_stream      = stream;
	g_descriptors = descriptors;
	g_prereqs     = prereqs;
	g_verify      = CpPar::VerifyEnabled();
	g_any         = true;
	std::printf("Kyty CpPar: 2b lane share: stream rings per lane %s (KYTY_CP_PAR_LANE_STREAM), descriptor "
	            "heaps per lane %s (KYTY_CP_PAR_LANE_DESCRIPTORS), small prerequisites %s "
	            "(KYTY_CP_PAR_LANE_PREREQS), self-checks %s (KYTY_CP_PAR_VERIFY)\n",
	            stream ? "on" : "off", descriptors ? "on" : "off", prereqs ? "on" : "off",
	            g_verify ? "on" : "off");
}

void Note(uint32_t lane, Stat stat, uint64_t value) noexcept {
	g_counts[lane < kLanes ? lane : kLanes - 1].values[static_cast<uint32_t>(stat)].fetch_add(
	    value, std::memory_order_relaxed);
}

void MaybePrint() noexcept {
	if (!g_any) {
		return;
	}
	auto       last = g_last_ns.load(std::memory_order_acquire);
	const auto now  = NowNs();
	if (last == 0) {
		(void)g_last_ns.compare_exchange_strong(last, now, std::memory_order_acq_rel);
		return;
	}
	if (now - last >= 10'000'000'000ull &&
	    g_last_ns.compare_exchange_strong(last, now, std::memory_order_acq_rel)) {
		Print(now - last);
	}
}

} // namespace Libs::Graphics::LaneShare
