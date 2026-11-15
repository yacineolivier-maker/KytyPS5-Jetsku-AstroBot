#include "graphics/host_gpu/renderer/masterSemaphore.h"

#include "common/assert.h"
#include "common/hangWatchdog.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "common/debugCounters.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/drawPrep/cpGaps.h"

#include <cinttypes>
#include <optional>

namespace Libs::Graphics {

MasterSemaphore::MasterSemaphore(GraphicContext& graphics, bool track_dispatch)
    : m_graphics(graphics) {
	if (graphics.submission_queue.Enabled() || track_dispatch) {
		m_submission_progress = std::make_shared<SubmissionProgress>();
	}
	vk::SemaphoreTypeCreateInfo type_info {};
	type_info.semaphoreType = vk::SemaphoreType::eTimeline;
	type_info.initialValue  = 0;

	vk::SemaphoreCreateInfo create_info {};
	create_info.pNext = &type_info;

	const auto result = m_graphics.device.createSemaphore(&create_info, nullptr, &m_semaphore);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || m_semaphore == nullptr);
	m_watchdog_timeline = HangWatchdog::RegisterTimeline(
	    reinterpret_cast<uint64_t>(static_cast<VkSemaphore>(m_semaphore)));
}

// Astro Bot PC (KYTY_CP_PAR=2b, masterSemaphore.h): tick ranges.
void MasterSemaphore::EnableRanges() {
	if (Ranges()) {
		return;
	}
	m_range_base = m_current_tick.load(std::memory_order_acquire);
	(void)BeginRange(0);
	m_oldest_lane.store(0, std::memory_order_relaxed);
	m_ranges.store(true, std::memory_order_release);
}

uint64_t MasterSemaphore::BeginRange(uint32_t lane) {
	EXIT_IF(lane >= MaxRangeLanes);
	// The order and the range are taken together by the lane holding the turn to begin (in list
	// order: one segment begins at a time, under the scheduler's operation lock).
	const auto order = m_next_order.fetch_add(1, std::memory_order_acq_rel);
	const auto base  = m_current_tick.fetch_add(RangeTicks, std::memory_order_acq_rel);
	m_lane_end[lane] = base + RangeTicks - 1u;
	m_lane_tick[lane].store(base, std::memory_order_release);
	return order;
}

bool MasterSemaphore::TryReleaseRange(uint32_t lane) {
	EXIT_IF(lane >= MaxRangeLanes);
	const auto end  = m_lane_end[lane];
	const auto base = end + 1u - RangeTicks;
	if (m_lane_tick[lane].load(std::memory_order_relaxed) != base) {
		return false; // the segment submitted: its range end must be signalled
	}
	auto expected = end + 1u;
	if (!m_current_tick.compare_exchange_strong(expected, base, std::memory_order_acq_rel)) {
		return false; // a later range was reserved
	}
	m_lane_tick[lane].store(end + 1u, std::memory_order_release); // closed
	return true;
}

uint64_t MasterSemaphore::NextLaneTick(uint32_t lane, bool range_end) {
	EXIT_IF(lane >= MaxRangeLanes);
	const auto tick = m_lane_tick[lane].load(std::memory_order_relaxed);
	const auto end  = m_lane_end[lane];
	if (tick > end) {
		EXIT("Kyty CpPar 2b: lane %u submits outside its tick range (tick %" PRIu64 ", range end %" PRIu64
		     ")\n",
		     lane, tick, end);
	}
	if (range_end) {
		m_lane_tick[lane].store(end + 1u, std::memory_order_release);
		return end;
	}
	if (tick == end) {
		EXIT("Kyty CpPar 2b: lane %u used up its tick range (%" PRIu64 " submits in one segment)\n", lane,
		     RangeTicks - 1u);
	}
	m_lane_tick[lane].store(tick + 1u, std::memory_order_release);
	return tick;
}

MasterSemaphore::~MasterSemaphore() {
	if (m_semaphore != nullptr) {
		m_graphics.device.destroySemaphore(m_semaphore, nullptr);
	}
}

void MasterSemaphore::Refresh() {
	HangWatchdog::Scope query("master-counter-query",
	                          reinterpret_cast<uint64_t>(static_cast<VkSemaphore>(m_semaphore)),
	                          HangWatchdog::Enabled() ? CurrentTick() : 0);
	uint64_t   counter = 0;
	const auto result  = m_graphics.device.getSemaphoreCounterValue(m_semaphore, &counter);
	if (result != vk::Result::eSuccess) {
		HangWatchdog::Scope error("master-counter-error",
		                          reinterpret_cast<uint64_t>(static_cast<VkSemaphore>(m_semaphore)),
		                          CurrentTick(),
		                          static_cast<uint64_t>(static_cast<int64_t>(result)));
		if (result == vk::Result::eErrorDeviceLost) DumpDeviceLossDiagnostics(m_graphics, CurrentTick());
		EXIT("MasterSemaphore: counter query failed: %s (submission tick %" PRIu64 ", gpu tick %" PRIu64 ")\n",
		     vk::to_string(result).c_str(), CurrentTick(), m_gpu_tick.load(std::memory_order_acquire));
	}

	auto known = m_gpu_tick.load(std::memory_order_acquire);
	while (known < counter &&
	       !m_gpu_tick.compare_exchange_weak(known, counter, std::memory_order_release,
	                                         std::memory_order_relaxed)) {
	}
	if (HangWatchdog::Enabled()) {
		HangWatchdog::UpdateTimeline(
		    m_watchdog_timeline, CurrentTick(), counter,
		    m_submission_progress ? m_submission_progress->dispatched_tick.load() : CurrentTick() - 1);
	}
}

void MasterSemaphore::Wait(uint64_t tick) {
	if (IsFree(tick)) {
		return;
	}
	// KYTY_CP_GAP_STATS (GPU thread only): readback drains, occlusion publication, the rest.
	const auto      reason = Profiler::CurrentGpuWaitReason();
	const CpGaps::Scope gap(reason == Profiler::FrameWait::GpuWaitDrain ? CpGaps::Cat::GpuWaitDrain
	                        : reason == Profiler::FrameWait::GpuWaitOcclusion
	                            ? CpGaps::Cat::GpuWaitOcclusion
	                            : CpGaps::Cat::GpuWaitOther);
	// Attribute CP-thread blocking to its caller (Profiler::ScopedGpuWaitReason). Other threads
	// (the completion runner, guest threads) wait here by design and are not counted.
	std::optional<Profiler::ScopedFrameWait> frame_wait;
	if (GuestGpu::IsGpuThread() && Profiler::AggregateEnabled()) {
		frame_wait.emplace(Profiler::CurrentGpuWaitReason());
	}
	if (m_submission_progress) {
		auto submitted = m_submission_progress->dispatched_tick.load(std::memory_order_acquire);
		HangWatchdog::Scope dispatch(
		    "master-dispatch", reinterpret_cast<uint64_t>(static_cast<VkSemaphore>(m_semaphore)),
		    tick, submitted, 0, HangWatchdog::Enabled() ? CurrentTick() : 0);
		while (submitted < tick) {
			m_submission_progress->dispatched_tick.wait(submitted, std::memory_order_acquire);
			submitted = m_submission_progress->dispatched_tick.load(std::memory_order_acquire);
			dispatch.Observed(submitted);
		}
	}
	if (IsFree(tick)) {
		return;
	}
	Refresh();
	if (IsFree(tick)) {
		return;
	}

	vk::SemaphoreWaitInfo wait_info {};
	wait_info.semaphoreCount = 1;
	wait_info.pSemaphores    = &m_semaphore;
	wait_info.pValues        = &tick;

	HangWatchdog::Scope wait("master-gpu",
	                         reinterpret_cast<uint64_t>(static_cast<VkSemaphore>(m_semaphore)),
	                         tick, HangWatchdog::Enabled() ? KnownGpuTick() : 0, 0,
	                         HangWatchdog::Enabled() ? CurrentTick() : 0);
	Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::GpuWaits,
	                                         Common::DebugCounters::Counter::GpuWaitNs);
	const auto result = m_graphics.device.waitSemaphores(&wait_info, UINT64_MAX);
	if (result != vk::Result::eSuccess) {
		HangWatchdog::Scope error("master-wait-error",
		                          reinterpret_cast<uint64_t>(static_cast<VkSemaphore>(m_semaphore)),
		                          tick, static_cast<uint64_t>(static_cast<int64_t>(result)));
		if (result == vk::Result::eErrorDeviceLost) {
			DumpDeviceLossDiagnostics(m_graphics, tick);
		}
		EXIT("MasterSemaphore: wait for tick %" PRIu64 " failed: %s (gpu tick %" PRIu64 ")\n", tick,
		     vk::to_string(result).c_str(), m_gpu_tick.load(std::memory_order_acquire));
	}
	Refresh();
}

} // namespace Libs::Graphics
