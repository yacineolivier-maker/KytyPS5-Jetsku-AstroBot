#include "graphics/host_gpu/renderer/drawPrep/fastCheck.h"

#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

// Astro Bot PC (KYTY_CP_PAR_FASTCHECK=1, fastCheck.h).
namespace Libs::Graphics::FastCheck {

namespace {

constexpr uint32_t kCauses = static_cast<uint32_t>(Cause::Count);
constexpr uint32_t kLanes  = 4;

static_assert(kCauses <= 64, "the causes are bits of a uint64_t");

constexpr std::array<uint8_t, kCauses> kConditionOf {
    1, 1, 1,       // NotPublished, InheritPending, Snapshot
    2, 2,          // HeadUnclaimed, HeadWaited
    3, 3, 3, 3,    // PrepMissing, PrepRefused, PlanNotQuiet, PipelineNotPlanned
    4, 4, 4, 4,    // TargetOp, Index8, RestartScan, MrtSplit
    5, 5,          // RunLateFallback, TargetsAcquired
    6, 6, 6, 6, 6, 6, // TextureMiss, TextureViewMiss, TextureDcc, Transition, ImageAdopt, ImageOverlay
    7, 7, 7, 7, 7, // BufferWritten, BufferCreate, BufferSync, BufferStream, BufferRejoin
    8, 8, 8,       // SamplerMiss, OcclusionFlush, RecorderDrain
    9, 9, 9, 9,    // PendingOps, Submit, ServicesRun, SliceYield
};

constexpr std::array<const char*, kCauses> kNames {
    "not-published", "inherit-pending", "snapshot",
    "head-unclaimed", "head-waited",
    "prep-missing", "prep-refused", "plan-not-quiet", "pipeline-not-planned",
    "target-op", "index8", "restart-scan", "mrt-split",
    "run-late-fallback", "targets-acquired",
    "texture-miss", "texture-view-miss", "texture-dcc", "transition", "image-adopt", "image-overlay",
    "buffer-written", "buffer-create", "buffer-sync", "buffer-stream", "buffer-rejoin",
    "sampler-miss", "occlusion-flush", "recorder-drain",
    "pending-ops", "submit", "services", "slice-yield",
};

// The order of the causes is the order of the conditions: the lowest set bit is the first failure.
static_assert(kConditionOf[static_cast<uint32_t>(Cause::SliceYield)] == 9);
static_assert(kConditionOf[static_cast<uint32_t>(Cause::BufferRejoin)] == 7);
static_assert(kConditionOf[static_cast<uint32_t>(Cause::TargetsAcquired)] == 5);

struct alignas(64) LaneCounts {
	std::array<std::atomic<uint64_t>, 2>       total {}; // [0] DrawIndex, [1] DrawAuto
	std::array<std::atomic<uint64_t>, 2>       fast {};
	std::atomic<uint64_t>                      empty {0};
	std::array<std::atomic<uint64_t>, kCauses> first {};
	std::array<std::atomic<uint64_t>, kCauses> sole {};
};

struct Snapshot {
	std::array<uint64_t, 2>       total {};
	std::array<uint64_t, 2>       fast {};
	uint64_t                      empty = 0;
	std::array<uint64_t, kCauses> first {};
	std::array<uint64_t, kCauses> sole {};
};

std::array<LaneCounts, kLanes> g_lanes;
std::atomic<uint64_t>          g_last_ns {0};
std::mutex                     g_print_mutex;
std::array<Snapshot, kLanes>   g_previous; // g_print_mutex

uint64_t NowNs() noexcept {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

Snapshot Take(const LaneCounts& lane) noexcept {
	Snapshot s;
	for (uint32_t i = 0; i < 2; i++) {
		s.total[i] = lane.total[i].load(std::memory_order_relaxed);
		s.fast[i]  = lane.fast[i].load(std::memory_order_relaxed);
	}
	s.empty = lane.empty.load(std::memory_order_relaxed);
	for (uint32_t i = 0; i < kCauses; i++) {
		s.first[i] = lane.first[i].load(std::memory_order_relaxed);
		s.sole[i]  = lane.sole[i].load(std::memory_order_relaxed);
	}
	return s;
}

std::string CauseList(const std::array<uint64_t, kCauses>& now, const std::array<uint64_t, kCauses>& before) {
	std::string text;
	uint32_t    condition = 0;
	for (uint32_t i = 0; i < kCauses; i++) {
		const auto delta = now[i] - before[i];
		if (delta == 0) {
			continue;
		}
		if (kConditionOf[i] != condition) {
			condition = kConditionOf[i];
			text += text.empty() ? "" : " |";
			text += " " + std::to_string(condition) + ":";
		}
		text += " " + std::string(kNames[i]) + " " + std::to_string(delta);
	}
	return text.empty() ? std::string(" none") : text;
}

void Print(uint64_t elapsed_ns) {
	const std::lock_guard lock(g_print_mutex);
	std::printf("CpPar 10s: 2b fast-draw candidates (%.0fs; condition (a) 1-9, docs-p3-drawindex.md):\n",
	            static_cast<double>(elapsed_ns) * 1e-9);
	for (uint32_t lane = 0; lane < kLanes; lane++) {
		const auto now    = Take(g_lanes[lane]);
		auto&      before = g_previous[lane];
		const auto index  = now.total[0] - before.total[0];
		const auto autos  = now.total[1] - before.total[1];
		if (index + autos != 0) {
			const auto fast_index = now.fast[0] - before.fast[0];
			const auto fast_autos = now.fast[1] - before.fast[1];
			const auto empty      = now.empty - before.empty;
			const auto all        = index + autos;
			std::printf("  lane %c: DrawIndex %" PRIu64 " (fast %" PRIu64 "), DrawAuto %" PRIu64
			            " (fast %" PRIu64 "); fast %.1f%%, empty %" PRIu64 ", failed %" PRIu64
			            "\n    first failure:%s\n    sole cause:%s\n",
			            static_cast<char>('A' + lane), index, fast_index, autos, fast_autos,
			            100.0 * static_cast<double>(fast_index + fast_autos) / static_cast<double>(all),
			            empty, all - fast_index - fast_autos - empty, CauseList(now.first, before.first).c_str(),
			            CauseList(now.sole, before.sole).c_str());
		}
		before = now;
	}
	std::fflush(stdout);
}

} // namespace

namespace Detail {

bool ReadEnabled() noexcept {
	const char* value = std::getenv("KYTY_CP_PAR_FASTCHECK");
	const bool  on    = value != nullptr && std::strcmp(value, "1") == 0;
	if (on) {
		std::printf("Kyty CpPar: fast-draw candidates census on (KYTY_CP_PAR_FASTCHECK=1, no behaviour change)\n");
	}
	return on;
}

void End(uint32_t lane, bool auto_draw) noexcept {
	t_active          = false;
	const auto causes = t_causes;
	auto&      counts = g_lanes[lane < kLanes ? lane : kLanes - 1];
	counts.total[auto_draw ? 1 : 0].fetch_add(1, std::memory_order_relaxed);
	if (!t_recorded) {
		counts.empty.fetch_add(1, std::memory_order_relaxed);
	} else if (causes == 0) {
		counts.fast[auto_draw ? 1 : 0].fetch_add(1, std::memory_order_relaxed);
	} else {
		counts.first[std::countr_zero(causes)].fetch_add(1, std::memory_order_relaxed);
		if (std::has_single_bit(causes)) {
			counts.sole[std::countr_zero(causes)].fetch_add(1, std::memory_order_relaxed);
		}
	}
	// The last print time is read before the clock (a later time of another lane cannot be newer
	// than this clock reading).
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

} // namespace Detail

} // namespace Libs::Graphics::FastCheck
