#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MASTERSEMAPHORE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MASTERSEMAPHORE_H_

#include "common/common.h"
#include "graphics/host_gpu/queueSubmission.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <memory>

namespace Libs::Graphics {

struct GraphicContext;

class MasterSemaphore {
public:
	// track_dispatch: host submissions of this timeline happen on another thread than the one
	// allocating its ticks even without the submission broker (the CP recorder, commandRecorder.h);
	// a tick then counts as known complete only after its vkQueueSubmit returned.
	explicit MasterSemaphore(GraphicContext& graphics, bool track_dispatch = false);
	~MasterSemaphore();
	KYTY_CLASS_NO_COPY(MasterSemaphore);

	[[nodiscard]] uint64_t CurrentTick() const noexcept {
		if (m_ranges.load(std::memory_order_relaxed)) {
			// Astro Bot PC (KYTY_CP_PAR=2b): the calling thread's lane (see the tick ranges below).
			auto lane = t_tick_lane;
			if (lane >= MaxRangeLanes) {
				lane = m_oldest_lane.load(std::memory_order_relaxed);
			}
			if (lane < MaxRangeLanes) {
				return m_lane_tick[lane].load(std::memory_order_acquire);
			}
		}
		return m_current_tick.load(std::memory_order_acquire);
	}
	[[nodiscard]] uint64_t KnownGpuTick() const noexcept {
		// In queued mode the GPU can finish before the host driver returns. Every
		// resource-retirement caller must also wait for that host call to finish.
		const auto gpu_tick = m_gpu_tick.load(std::memory_order_acquire);
		return m_submission_progress
		           ? std::min(gpu_tick, m_submission_progress->dispatched_tick.load(
		                                    std::memory_order_acquire))
		           : gpu_tick;
	}
	[[nodiscard]] bool     IsFree(uint64_t tick) const noexcept { return KnownGpuTick() >= tick; }
	[[nodiscard]] uint64_t NextTick() noexcept {
		return m_current_tick.fetch_add(1, std::memory_order_release);
	}
	[[nodiscard]] vk::Semaphore Handle() const noexcept { return m_semaphore; }
	[[nodiscard]] const std::shared_ptr<SubmissionProgress>& GetSubmissionProgress() const noexcept {
		return m_submission_progress;
	}

	void Refresh();
	void Wait(uint64_t tick);

	// Astro Bot PC (KYTY_CP_PAR=2b, cpParallel.h): tick ranges. Each lane segment reserves, in list
	// order, a range of RangeTicks consecutive ticks (BeginRange, which returns the segment's order:
	// 0, 1, 2... from EnableRanges on). The lane's submits take the range's ticks in turn and the
	// segment's last submit (the hand-off, `range_end`) takes the range's last tick. A segment that
	// submitted nothing gives its range back when no later range was reserved (TryReleaseRange: the
	// next segment then reuses its ticks, as 2a's skipped hand-off reused the tick); otherwise its
	// hand-off submits (empty), so that every tick below a lane's current one is signalled without
	// that lane's help (a thread may wait for any tick it saw below its current one). Ticks thus stay
	// increasing in queue order (the lanes' buffers are handed to the queue in range order), with
	// gaps: a timeline value may be skipped, and a wait for a skipped value ends with the next
	// higher signal. So the tick a lane records for is known while it records, whatever the older
	// lanes still submit: what is deferred to CurrentTick() completes no earlier than the buffer
	// holding the recording (the first submit at or above it in queue order). CurrentTick(): the
	// calling thread's lane's next tick (SetThreadTickLane); a thread without a lane reads the
	// oldest open segment's lane (SetOldestLane), else the next range's base. Start-up only
	// (EnableRanges: lane 0 gets the first range, order 0); no NextTick() afterwards.
	static constexpr uint32_t MaxRangeLanes = 4;
	static constexpr uint64_t RangeTicks    = uint64_t {1} << 20u;
	void                      EnableRanges();
	[[nodiscard]] bool        Ranges() const noexcept { return m_ranges.load(std::memory_order_relaxed); }
	[[nodiscard]] uint64_t    RangeBase() const noexcept { return m_range_base; }
	uint64_t                  BeginRange(uint32_t lane);
	[[nodiscard]] bool        TryReleaseRange(uint32_t lane);
	uint64_t                  NextLaneTick(uint32_t lane, bool range_end);
	void SetOldestLane(uint32_t lane) noexcept { m_oldest_lane.store(lane, std::memory_order_relaxed); }
	static void SetThreadTickLane(uint32_t lane) noexcept { t_tick_lane = lane; }
	[[nodiscard]] static uint32_t ThreadTickLane() noexcept { return t_tick_lane; }

private:
	// Astro Bot PC (KYTY_CP_PAR=2b): the tick ranges (above). m_lane_tick: each lane's next tick;
	// m_lane_end: its range's last tick (written by the lane's own thread only).
	inline static thread_local uint32_t              t_tick_lane = UINT32_MAX;
	std::atomic<bool>                                m_ranges {false};
	uint64_t                                         m_range_base = 0;
	std::array<std::atomic<uint64_t>, MaxRangeLanes> m_lane_tick {};
	std::array<uint64_t, MaxRangeLanes>              m_lane_end {};
	std::atomic<uint32_t>                            m_oldest_lane {UINT32_MAX};
	std::atomic<uint64_t>                            m_next_order {0};
	GraphicContext&       m_graphics;
	vk::Semaphore         m_semaphore = nullptr;
	std::atomic<uint64_t> m_gpu_tick {0};
	std::atomic<uint64_t> m_current_tick {1};
	std::shared_ptr<SubmissionProgress> m_submission_progress;
	uint32_t                            m_watchdog_timeline = UINT32_MAX;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MASTERSEMAPHORE_H_
