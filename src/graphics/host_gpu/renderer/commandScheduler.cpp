#include "graphics/host_gpu/renderer/commandScheduler.h"

#include "graphics/guest_gpu/command_processor/cpParallel.h" // Astro Bot PC (KYTY_CP_PAR=1b3)

#include "common/assert.h"
#include "common/cpuPlacement.h"
#include "common/hangTrace.h"
#include "common/hangWatchdog.h"
#include "common/liveSwitch.h"
#include "common/debugCounters.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/rendererBatch.h"
#include "common/threads.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "graphics/host_gpu/renderer/drawPrep/cpGaps.h"
#include "graphics/host_gpu/renderer/drawPrep/readSet.h"
#include "graphics/host_gpu/renderer/eopTimestamps.h"
#include "graphics/host_gpu/renderer/gpuOpProfiler.h"
#include "graphics/host_gpu/renderer/gpuTiming.h"
#include "graphics/host_gpu/watchdogSubmit.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <thread>

#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif

namespace Libs::Graphics {

static thread_local CommandScheduler* g_deferred_callback_scheduler = nullptr;
// Astro Bot PC (KYTY_CP_PAR=2b, tick ranges): the calling lane thread's Submit is its segment's
// hand-off (EndContextSegment): it takes the range's last tick and never waits for its turn.
static thread_local bool t_lane_end_submit = false;
// Astro Bot PC (KYTY_CP_PAR=1b5, commandScheduler.h): the calling thread's interrupt capture.
static thread_local CommandScheduler::PriorityCapture t_priority_capture = nullptr;

void CommandScheduler::SetThreadPriorityCapture(PriorityCapture capture) noexcept {
	t_priority_capture = capture;
}

namespace {

void ReportVulkanFatal(const char* what, vk::Result result, uint64_t tick, uint32_t debug_op,
                       uint64_t debug_submit, uint32_t arg0, uint32_t arg1, uint32_t arg2,
                       uint32_t arg3, uint64_t arg4) {
	LOGF("%s failed: %s (%d), tick=%" PRIu64 " debug_op=%u debug_submit=%" PRIu64
	     " args=%u,%u,%u,%u,0x%016" PRIx64 "\n",
	     what, vk::to_string(result).c_str(), static_cast<int>(result), tick, debug_op,
	     debug_submit, arg0, arg1, arg2, arg3, arg4);
	std::printf("%s failed: %s (%d), tick=%" PRIu64 " debug_op=%u debug_submit=%" PRIu64
	            " args=%u,%u,%u,%u,0x%016" PRIx64 "\n",
	            what, vk::to_string(result).c_str(), static_cast<int>(result), tick, debug_op,
	            debug_submit, arg0, arg1, arg2, arg3, arg4);
	std::fflush(stdout);
}

} // namespace

CommandScheduler::CommandPool::CommandPool(GraphicContext& graphics, MasterSemaphore& master)
    : m_graphics(graphics), m_master(master) {
	EXIT_IF(graphics.queue_family == static_cast<uint32_t>(-1));
	vk::CommandPoolCreateInfo create {};
	create.queueFamilyIndex = graphics.queue_family;
	create.flags            = vk::CommandPoolCreateFlagBits::eTransient |
	                          vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
	const auto result       = graphics.device.createCommandPool(&create, nullptr, &m_pool);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || m_pool == nullptr);
}

CommandScheduler::CommandPool::~CommandPool() {
	m_graphics.device.destroyCommandPool(m_pool, nullptr);
}

size_t CommandScheduler::CommandPool::Grow() {
	const auto first = m_ticks.size();
	m_ticks.resize(first + GrowStep);
	m_buffers.resize(first + GrowStep);

	vk::CommandBufferAllocateInfo allocate {};
	allocate.commandPool        = m_pool;
	allocate.level              = vk::CommandBufferLevel::ePrimary;
	allocate.commandBufferCount = static_cast<uint32_t>(GrowStep);
	EXIT_IF(m_graphics.device.allocateCommandBuffers(&allocate, m_buffers.data() + first) !=
	        vk::Result::eSuccess);
	return first;
}

vk::CommandBuffer CommandScheduler::CommandPool::Commit() {
	auto       gpu_tick = m_master.KnownGpuTick();
	const auto search   = [this, &gpu_tick](size_t begin, size_t end) -> std::optional<size_t> {
		for (size_t index = begin; index < end; ++index) {
			if (gpu_tick >= m_ticks[index]) {
				m_ticks[index] = m_master.CurrentTick();
				return index;
			}
		}
		return std::nullopt;
	};

	auto found = search(m_hint, m_ticks.size());
	if (!found) {
		m_master.Refresh();
		gpu_tick = m_master.KnownGpuTick();
		found    = search(m_hint, m_ticks.size());
	}
	if (!found) {
		found = search(0, m_hint);
	}
	if (!found) {
		if (m_recorder != nullptr) {
			// vkAllocateCommandBuffers needs the pool the recorder may be recording from.
			m_recorder->Drain(nullptr, false);
		}
		found           = Grow();
		m_ticks[*found] = m_master.CurrentTick();
	}

	m_hint = (*found + 1) % m_ticks.size();
	return m_buffers[*found];
}

// Astro Bot PC (KYTY_CP_PAR=2a, commandScheduler.h).
void CommandScheduler::CommandPool::Return(vk::CommandBuffer buffer) {
	for (size_t index = 0; index < m_buffers.size(); ++index) {
		if (m_buffers[index] == buffer) {
			m_ticks[index] = 0;
			return;
		}
	}
	EXIT("CommandPool::Return: unknown command buffer\n");
}

bool CommandScheduler::InDeferredOperation() noexcept {
	return g_deferred_callback_scheduler != nullptr;
}

CommandScheduler::CommandScheduler(RenderContext& context, GraphicContext& graphics, Role role)
    : m_master(graphics,
               role == Role::Guest && CommandRecorder::ConfiguredMode() != CommandRecorder::Mode::Off),
      m_context(context), m_graphics(graphics),
      m_command_pool(graphics, m_master), m_command(*this), m_command_b(*this),
      m_priority_thread([this](std::stop_token stop) { PriorityOperationsThread(stop); }) {
	if (role == Role::Guest && GpuTiming::Enabled()) {
		m_gpu_timing = std::make_unique<GpuTimestampRing>(graphics, GpuTimestampRing::GuestPairs);
		if (!m_gpu_timing->Valid()) {
			m_gpu_timing.reset();
		}
	}
	if (role == Role::Guest && EopTimestamps::GpuEnabled()) {
		m_eop_timestamps = std::make_unique<EopTimestampRing>(graphics);
		if (!m_eop_timestamps->Valid()) {
			m_eop_timestamps.reset();
		}
	}
	m_gpu_ops = role == Role::Guest && GpuOpProfiler::Enabled();
	if (const auto mode = CommandRecorder::ConfiguredMode();
	    role == Role::Guest && mode != CommandRecorder::Mode::Off) {
		m_recorder = std::make_unique<CommandRecorder>(graphics, m_master, m_gpu_timing.get(),
		                                               m_gpu_ops, mode);
		m_command.m_recorder = m_recorder.get();
		m_command.m_encoder  = &m_recorder->Encoder();
		m_command_pool.SetRecorder(m_recorder.get());
	}
}

CommandScheduler::~CommandScheduler() {
	Shutdown();
}

void CommandScheduler::Shutdown() {
	{
		std::unique_lock lock(m_operation_mutex);
		if (m_operation_state == OperationState::Closed) {
			return;
		}
		if (g_deferred_callback_scheduler == this) {
			EXIT_IF(m_operation_state == OperationState::Open);
			// A priority callback cannot join its own runner, while a normal callback can be
			// executing inside the shutdown owner's final PopPendingOperations. The owning
			// thread will finish shutdown after this callback returns.
			return;
		}
		if (m_operation_state == OperationState::Draining) {
			m_operation_available.wait(
			    lock, [this] { return m_operation_state == OperationState::Closed; });
			return;
		}
		m_operation_state = OperationState::Draining;
	}
	if (!m_command.IsInvalid()) {
		Submit({}, true);
	}
	m_master.Wait(LastSubmittedTick()); // Astro Bot PC (KYTY_CP_PAR=2b: tick ranges have gaps)
	if (m_recorder != nullptr) {
		// Every tick was recorded and handed to the queue: stop the recorder thread before the
		// timing ring (its single producer until now) and the pool are used from here.
		m_recorder->Stop();
	}
	if (m_gpu_timing) {
		// Every submitted tick is complete: collect the remaining pairs before the ring (and its
		// query pool) is destroyed with this scheduler. No slot is left recording here.
		m_gpu_timing->Collect(m_master.KnownGpuTick(), true);
	}
	if (m_gpu_ops) {
		GpuOpProfiler::OnSchedulerShutdown(m_graphics);
	}
	PopPendingOperations();
	DrainPriorityOperations();
	m_priority_thread.request_stop();
	m_operation_available.notify_all();
	{
		// The runner re-checks its stop token only under the lock it sleeps with.
		std::lock_guard lock(m_operation_mutex);
	}
	m_priority_available.notify_all();
	if (m_priority_thread.joinable()) {
		m_priority_thread.join();
	}
	{
		std::lock_guard lock(m_operation_mutex);
		EXIT_IF(!m_pending_operations.empty() || !m_priority_operations.empty() ||
		        m_priority_active);
		m_operation_state = OperationState::Closed;
	}
	m_operation_available.notify_all();
}

void CommandScheduler::Begin(HW::Context& registers, HW::UserConfig& user_config,
                             HW::Shader& shaders) {
	{
		std::lock_guard lock(m_operation_mutex);
		EXIT_IF(m_operation_state != OperationState::Open);
	}
	m_command.Bind(registers, user_config, shaders);
	if (m_dual) {
		m_command_b.Bind(registers, user_config, shaders); // Astro Bot PC (KYTY_CP_PAR=1b2)
		for (uint32_t lane = 2; lane < m_lane_count; lane++) {
			m_command_x[lane]->Bind(registers, user_config, shaders); // Astro Bot PC (2a, N lanes)
		}
	}

	// Astro Bot PC (KYTY_CP_PAR=2a): the calling thread's lane (the other lane names no buffer).
	if ((m_lanes ? Cmd() : m_command).IsInvalid()) {
		BeginNext();
	}
}

void CommandScheduler::BeginRendering(const RenderState& state) {
	Current().BeginRendering(state);
}

void CommandScheduler::EndRendering() {
	if (Active() && !Cmd().IsInvalid()) {
		Current().EndRendering();
	}
}

void CommandScheduler::Flush() {
	SubmitInfo submit;
	Flush(submit);
}

void CommandScheduler::Flush(SubmitInfo& submit) {
	Submit(submit);
	BeginNext();
}

uint64_t CommandScheduler::FlushObservable() {
	CheckActive();
	const auto tick = Submit({}, true);
	BeginNext();
	return tick;
}

void CommandScheduler::FlushAndWait() {
	const auto tick = Submit({}, true);
	m_master.Wait(tick);
	BeginNext();
}

void CommandScheduler::Finish() {
	CheckActive();
	if (!(m_lanes ? Cmd() : m_command).IsInvalid()) { // Astro Bot PC (KYTY_CP_PAR=2a)
		Submit({}, true);
	}
	m_master.Wait(LastSubmittedTick()); // Astro Bot PC (KYTY_CP_PAR=2b: tick ranges have gaps)
	BeginNext();
	PopPendingOperations();
}

void CommandScheduler::Wait(uint64_t tick) {
	EXIT_IF(tick > CurrentTick());
	if (tick == CurrentTick()) {
		CheckActive();
		// A stream-buffer wrap can wait while a draw is being prepared through a reference to
		// Current(). The wrapper stays stable while its pooled Vulkan buffer is retired. Deferred
		// resources are released only at the next GPU operation boundary.
		const auto submitted_tick = Submit({}, true);
		EXIT_IF(submitted_tick != tick);
		m_master.Wait(tick);
		BeginNext();
	} else {
		m_master.Wait(tick);
	}
}

void CommandScheduler::PopPendingOperations() {
	PopOperations(true);
}

// KYTY_PENDING_OPS_NOWAIT (default on; =0 waits as PopPendingOperations does). Live switch
// (common/liveSwitch.h): each call picks one of two valid behaviours.
static Live::Switch g_pending_ops_nowait("KYTY_PENDING_OPS_NOWAIT", Live::ParseDefaultOn);

static bool PendingOpsNoWait() {
	return g_pending_ops_nowait.On();
}

void CommandScheduler::PopReadyOperations() {
	PopOperations(!PendingOpsNoWait());
}

bool CommandScheduler::PriorityDoneLocked(uint64_t tick) const noexcept {
	const bool active_before_or_at = m_priority_active && m_priority_active_tick <= tick;
	const bool queued_before_or_at =
	    !m_priority_operations.empty() && m_priority_operations.front().tick <= tick;
	return !active_before_or_at && !queued_before_or_at;
}

// KYTY_PENDING_REFRESH_US=<n> (default 0: off; BryanKAdams/KytyPS5 e4a7551): the non-waiting pop
// at draw entry (PopReadyOperations) queries the GPU's progress (vkGetSemaphoreCounterValue) at
// most once per n microseconds; between queries it runs what the last known progress allows. The
// operations are deletions, recycling and fault-buffer processing that nothing waits for; every
// blocking pop, explicit wait and allocation path queries as before. Live switch
// (common/liveSwitch.h): a parameter read at every pop.
static Live::Switch g_pending_refresh_us("KYTY_PENDING_REFRESH_US", [](const char* value) -> int64_t {
	return value == nullptr ? 0 : static_cast<int64_t>(std::strtoull(value, nullptr, 10));
});

static uint64_t PendingRefreshIntervalNs() {
	return static_cast<uint64_t>(g_pending_refresh_us.Get()) * 1000u;
}

void CommandScheduler::PopOperations(bool wait_for_priority) {
	if (Common::RendererBatchEnabled()) {
		uint64_t first_tick = 0;
		{
			std::lock_guard lock(m_operation_mutex);
			if (m_pending_operations.empty()) return;
			first_tick = m_pending_operations.front().tick;
		}
		// A callback on the recording tick cannot have completed. Known completed
		// ticks need no driver query; explicit waits and allocation paths still refresh.
		if (first_tick >= CurrentTick()) return;
		if (!m_master.IsFree(first_tick)) {
			if (const auto interval = PendingRefreshIntervalNs(); interval != 0 && !wait_for_priority) {
				// The operations are queued in tick order: with the first one not known complete,
				// none is, until the next query.
				const auto now = GpuTiming::NowNs();
				if (now - m_last_pending_refresh_ns < interval) return;
				m_last_pending_refresh_ns = now;
			}
			m_master.Refresh();
		}
	} else {
		m_master.Refresh();
	}
	for (;;) {
		PendingOperation operation;
		{
			std::lock_guard lock(m_operation_mutex);
			if (m_pending_operations.empty() ||
			    !m_master.IsFree(m_pending_operations.front().tick)) {
				return;
			}
			if (!wait_for_priority && !PriorityDoneLocked(m_pending_operations.front().tick)) {
				// The runner has not finished this tick's priority operations. Leave this operation
				// and the ones after it queued, in order: a later pop runs them, and every blocking
				// pop (Finish, the fault manager) waits as before.
				Profiler::CountFrameEvent(Profiler::FrameEvent::PendingOpsDeferred);
				Profiler::CountFrameEvent(Profiler::FrameEvent::PendingOpsDeferredDepth,
				                          m_pending_operations.size());
				HangTrace::NotePendingOperations(m_pending_operations.size());
				return;
			}
			operation = std::move(m_pending_operations.front());
			m_pending_operations.pop();
		}
		if (wait_for_priority) {
			WaitPriorityOperations(operation.tick);
		}
		// Without waiting, the priority operations of this tick and earlier are done: new ones
		// carry the recording tick, which is later than any completed one.
		RunOperation(std::move(operation.callback));
	}
}

void CommandScheduler::DeferOperation(Common::UniqueFunction<void>&& operation) {
	CheckActive();
	EXIT_IF(!operation);
	std::unique_lock lock(m_operation_mutex);
	if (m_operation_state == OperationState::Open) {
		if (m_graphics.submission_queue.Enabled()) {
			m_preserve_current_completion = true;
		}
		if (Profiler::AggregateEnabled()) {
			m_diagnostic_generic_completion = true;
		}
		if (m_lanes_2b) {
			// Astro Bot PC (KYTY_CP_PAR=2b): a younger segment's operation waits for its turn.
			if (const auto order = HoldOrderLocked(); order != UINT64_MAX) {
				m_held[order].normal.push_back({std::move(operation), CurrentTick()});
				CpPar::NoteLaneStat(MasterSemaphore::ThreadTickLane(), CpPar::LaneStat::HeldOperations, 1);
				return;
			}
		}
		m_pending_operations.push({std::move(operation), CurrentTick()});
		return;
	}
	if (g_deferred_callback_scheduler == this) {
		lock.unlock();
		operation();
		return;
	}
	m_operation_available.wait(lock,
	                           [this] { return m_operation_state == OperationState::Closed; });
	lock.unlock();
	operation();
}

void CommandScheduler::DeferPriorityOperation(Common::UniqueFunction<void>&& operation,
                                             PriorityOperationKind kind) {
	CheckActive();
	EXIT_IF(!operation);
	if (kind == PriorityOperationKind::EopInterrupt && CpPar::Concurrent()) {
		// Astro Bot PC (KYTY_CP_PAR=1b5, cpParallel.h): a run-ahead op's interrupt waits for the
		// merge; one raised by CP resolver #2 outside the capture would reach the guest early.
		if (t_priority_capture != nullptr && t_priority_capture(operation)) {
			return;
		}
		if (CpPar::Deferred() && CpPar::OnResolver2()) {
			CpPar::NoteStrayInterrupt();
		}
	}
	std::unique_lock lock(m_operation_mutex);
	if (m_operation_state == OperationState::Open) {
		if (m_graphics.submission_queue.Enabled()) {
			m_preserve_current_completion = true;
		}
		if (Profiler::AggregateEnabled()) {
			if (kind == PriorityOperationKind::EopInterrupt) {
				m_diagnostic_eop_completion = true;
			} else {
				m_diagnostic_generic_completion = true;
			}
		}
		if (m_lanes_2b) {
			// Astro Bot PC (KYTY_CP_PAR=2b): a younger segment's operation waits for its turn.
			if (const auto order = HoldOrderLocked(); order != UINT64_MAX) {
				m_held[order].priority.push_back({std::move(operation), CurrentTick()});
				CpPar::NoteLaneStat(MasterSemaphore::ThreadTickLane(), CpPar::LaneStat::HeldOperations, 1);
				return;
			}
		}
		const bool was_empty = m_priority_operations.empty();
		m_priority_operations.push({std::move(operation), CurrentTick()});
		lock.unlock();
		if (!PriorityWakeupsBatched()) {
			m_operation_available.notify_one();
			m_priority_available.notify_one();
		} else if (was_empty) {
			// The runner sleeps only on an empty queue; otherwise it pops this one by itself.
			m_priority_available.notify_one();
		}
		return;
	}
	if (g_deferred_callback_scheduler == this) {
		lock.unlock();
		operation();
		return;
	}
	m_operation_available.wait(lock,
	                           [this] { return m_operation_state == OperationState::Closed; });
	lock.unlock();
	operation();
}

bool CommandScheduler::PriorityWakeupsBatched() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_PRIORITY_WAKE_BATCH");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

void CommandScheduler::SetProgressHook(ProgressHook hook, void* context) {
	std::lock_guard lock(m_operation_mutex);
	m_progress_hook         = hook;
	m_progress_hook_context = context;
}

void CommandScheduler::PriorityOperationsThread(std::stop_token stop) {
	// It blocks on the operation queue and on the timeline semaphore; the CP waits for it
	// (WaitPriorityOperations) and so does the guest (end-of-pipe interrupts, flips, readbacks).
	Common::RaiseServiceThreadPriority();
	const bool batched         = PriorityWakeupsBatched();
	bool       has_previous    = false;
	uint64_t   previous_tick   = 0;
	uint32_t   placement_count = 0; // placement samples (common/cpuPlacement.h), every 16th
	while (!stop.stop_requested()) {
		PendingOperation operation;
		ProgressHook     hook         = nullptr;
		void*            hook_context = nullptr;
		{
			std::unique_lock lock(m_operation_mutex);
			if (has_previous) {
				has_previous           = false;
				m_priority_active      = false;
				m_priority_active_tick = 0;
				// After the state change, under the lock: a spinning waiter that sees the new
				// value and then takes the lock finds the operation finished.
				m_priority_progress.fetch_add(1, std::memory_order_release);
				// Waiters need every operation of their tick done. Wake them when the next queued
				// operation belongs to a later tick or the queue is empty, and only if any wait.
				const bool tick_done = m_priority_operations.empty() ||
				                       m_priority_operations.front().tick != previous_tick;
				if (!batched || (m_priority_waiters != 0 && tick_done)) {
					m_operation_available.notify_all();
					if (batched) {
						Profiler::CountFrameEvent(Profiler::FrameEvent::PriorityWaiterWakeups);
					}
				}
			}
			m_priority_available.wait(lock, [this, &stop] {
				return stop.stop_requested() || !m_priority_operations.empty();
			});
			if (stop.stop_requested()) {
				return;
			}
			operation = std::move(m_priority_operations.front());
			m_priority_operations.pop();
			m_priority_active      = true;
			m_priority_active_tick = operation.tick;
			hook                   = m_progress_hook;
			hook_context           = m_progress_hook_context;
		}
		m_master.Wait(operation.tick);
		if (!stop.stop_requested()) {
			RunOperation(std::move(operation.callback));
			Profiler::CountFrameEvent(Profiler::FrameEvent::PriorityOperationsRun);
			if ((++placement_count & 15u) == 0u) {
				Common::SamplePlacement(Common::ThreadRole::Host);
			}
			// Still marked active: an owner clearing the hook and then draining this runner
			// never races with this call.
			if (hook != nullptr) {
				hook(hook_context);
			}
		}
		has_previous  = true;
		previous_tick = operation.tick;
	}
	if (has_previous) {
		{
			std::lock_guard lock(m_operation_mutex);
			m_priority_active      = false;
			m_priority_active_tick = 0;
			m_priority_progress.fetch_add(1, std::memory_order_release);
		}
		m_operation_available.notify_all();
	}
}

void CommandScheduler::DrainPriorityOperations() {
	HangWatchdog::Scope wait("priority-drain", reinterpret_cast<uint64_t>(this));
	EXIT_IF(g_deferred_callback_scheduler == this);
	std::unique_lock lock(m_operation_mutex);
	++m_priority_waiters;
	m_operation_available.wait(
	    lock, [this] { return m_priority_operations.empty() && !m_priority_active; });
	--m_priority_waiters;
}

// KYTY_PRIORITY_WAIT_SPIN_US (default 0): WaitPriorityOperations spins this long before blocking.
// Live switch (common/liveSwitch.h), read at each wait.
static Live::Switch g_priority_wait_spin_us("KYTY_PRIORITY_WAIT_SPIN_US", [](const char* value) -> int64_t {
	const auto us = value != nullptr ? std::strtoull(value, nullptr, 10) : 0ull;
	return static_cast<int64_t>(std::min<unsigned long long>(us, 100'000ull));
});

static uint64_t PriorityWaitSpinNs() {
	return static_cast<uint64_t>(g_priority_wait_spin_us.Get()) * 1000u;
}

static void PriorityWaitRelax() {
#if defined(_M_X64) || defined(__x86_64__)
	_mm_pause();
#else
	std::this_thread::yield();
#endif
}

void CommandScheduler::WaitPriorityOperations(uint64_t tick) {
	EXIT_IF(g_deferred_callback_scheduler == this);
	std::unique_lock lock(m_operation_mutex);
	if (PriorityDoneLocked(tick)) {
		return;
	}
	HangWatchdog::Scope wait("priority-completion", reinterpret_cast<uint64_t>(this), tick,
	                         m_priority_active_tick, 0, m_priority_operations.size());
	if (const auto spin_ns = PriorityWaitSpinNs(); spin_ns != 0) {
		// Spin on the runner's progress counter and take the lock only when it moves: the
		// runner needs the lock to finish each operation.
		Profiler::CountFrameEvent(Profiler::FrameEvent::PriorityWaitSpins);
		auto       seen  = m_priority_progress.load(std::memory_order_acquire);
		const auto start = GpuTiming::NowNs();
		lock.unlock();
		for (uint32_t spins = 1;; spins++) {
			PriorityWaitRelax();
			if (const auto progress = m_priority_progress.load(std::memory_order_acquire);
			    progress != seen) {
				seen = progress;
				lock.lock();
				if (PriorityDoneLocked(tick)) {
					Profiler::CountFrameEvent(Profiler::FrameEvent::PriorityWaitSpinHits);
					return;
				}
				lock.unlock();
			}
			if ((spins & 63u) == 0u && GpuTiming::NowNs() - start > spin_ns) {
				break;
			}
		}
		lock.lock();
	}
	++m_priority_waiters;
	m_operation_available.wait(lock, [this, tick] { return PriorityDoneLocked(tick); });
	--m_priority_waiters;
}

void CommandScheduler::RunOperation(Common::UniqueFunction<void>&& operation) {
	HangWatchdog::Scope callback("priority-or-pending-callback", reinterpret_cast<uint64_t>(this));
	auto* previous                = g_deferred_callback_scheduler;
	g_deferred_callback_scheduler = this;
	operation();
	g_deferred_callback_scheduler = previous;
}

bool CommandScheduler::IsFree(uint64_t tick) {
	if (m_master.IsFree(tick)) {
		return true;
	}
	m_master.Refresh();
	return m_master.IsFree(tick);
}

void CommandScheduler::CheckActive() const {
	EXIT_IF(!Active());
	// Draw-prep workers only prepare; recording belongs to the GPU thread.
	EXIT_IF(DrawPrep::IsWorkerThread());
}

CommandBuffer& CommandScheduler::Current() {
	auto& command = Cmd(); // Astro Bot PC (KYTY_CP_PAR=1b2): the calling thread's context
	EXIT_IF(command.IsInvalid());
	return command;
}

uint32_t CommandScheduler::ThreadContext() noexcept {
	return CommandRecorder::ThreadStream();
}

// Astro Bot PC (KYTY_CP_PAR=1b2, commandScheduler.h).
bool CommandScheduler::EnableContextB() {
	if (m_dual) {
		return true;
	}
	if (m_recorder == nullptr || !m_recorder->EnableSecondStream()) {
		return false;
	}
	m_command_b.m_recorder = m_recorder.get();
	m_command_b.m_encoder  = &m_recorder->EncoderB();
	m_command_b.m_buffer   = m_command.m_buffer;
	if (m_command.m_registers != nullptr) {
		m_command_b.Bind(*m_command.m_registers, *m_command.m_user_config, *m_command.m_shaders);
	}
	// Rendering instance serials and descriptor epochs identify state across draws (draw-run
	// continuations, RenderExecutor::DrawRunCommandUnchanged); B's never equal A's.
	m_command_b.m_rendering_serial = uint64_t {1} << 62u;
	for (auto& epoch: m_command_b.m_descriptor_epochs) {
		epoch = uint64_t {1} << 62u;
	}
	m_dynamic_state_b.Invalidate();
	m_dual = true;
	return true;
}

// Astro Bot PC (KYTY_CP_PAR=2a, commandScheduler.h).
bool CommandScheduler::EnableLanes(uint32_t count) {
	if (m_lanes) {
		return count == m_lane_count;
	}
	if (!m_dual || m_recorder == nullptr || count < 2 || count > MaxLanes ||
	    !m_recorder->EnableStreams(count)) {
		return false;
	}
	m_command_pool_b = std::make_unique<CommandPool>(m_graphics, m_master);
	m_command_pool_b->SetRecorder(m_recorder.get());
	// Context B names no native buffer until its first segment begins one of its own.
	EXIT_IF(m_command_b.m_rendering || m_command_b.m_direct_window || !m_command_b.m_pending.Empty());
	m_command_b.m_buffer = nullptr;
	// Astro Bot PC (KYTY_CP_PAR=2a, N lanes): lanes 2.. as EnableContextB sets context B up, each
	// with its own stream, command pool, dynamic state and serial range (lane << 62).
	for (uint32_t lane = 2; lane < count; lane++) {
		auto& command      = m_command_x[lane];
		command.reset(new CommandBuffer(*this)); // private constructor (friend)
		command->m_recorder = m_recorder.get();
		command->m_encoder  = &m_recorder->StreamEncoder(lane);
		command->m_buffer   = nullptr;
		if (m_command.m_registers != nullptr) {
			command->Bind(*m_command.m_registers, *m_command.m_user_config, *m_command.m_shaders);
		}
		command->m_rendering_serial = uint64_t {lane} << 62u;
		for (auto& epoch: command->m_descriptor_epochs) {
			epoch = uint64_t {lane} << 62u;
		}
		m_dynamic_state_x[lane] = std::make_unique<DynamicState>();
		m_dynamic_state_x[lane]->Invalidate();
		m_command_pool_x[lane] = std::make_unique<CommandPool>(m_graphics, m_master);
		m_command_pool_x[lane]->SetRecorder(m_recorder.get());
	}
	m_lane_count = count;
	m_lanes      = true;
	return true;
}

// Astro Bot PC (KYTY_CP_PAR=2b, commandScheduler.h).
bool CommandScheduler::EnableLanes2b() {
	if (m_lanes_2b) {
		return true;
	}
	if (!m_lanes || m_recorder == nullptr || m_gpu_timing != nullptr || m_gpu_ops ||
	    m_eop_timestamps != nullptr || !m_recorder->EnableLaneReplay()) {
		return false;
	}
	// Tick ranges (commandScheduler.h): lane 0's current recording is the segment of order 0.
	m_master.EnableRanges();
	m_recorder->EnableRangeOrder();
	{
		std::lock_guard lock(m_operation_mutex);
		m_lane_order.fill(UINT64_MAX);
		m_lane_order[0] = 0;
		m_turn          = 0;
		// Steps 4-5 (commandScheduler.h): the registry runs with KYTY_CP_PAR_VERIFY=1.
		m_deps_active = CpPar::VerifyEnabled();
	}
	m_lanes_2b = true;
	return true;
}

// Astro Bot PC (KYTY_CP_PAR=2b, steps 4-5, commandScheduler.h).
void CommandScheduler::NoteLaneBuffer(uint64_t address, uint64_t size, bool written) {
	if (!m_deps_active || size == 0) {
		return;
	}
	const auto lane = MasterSemaphore::ThreadTickLane();
	if (lane >= MaxLanes) {
		return;
	}
	const auto order = m_lane_order[lane]; // written by this thread only
	if (order == UINT64_MAX) {
		return;
	}
	const LaneRange range {address, address + size};
	const auto      overlaps = [&range](const std::vector<LaneRange>& ranges) {
		for (const auto& r: ranges) {
			if (r.begin < range.end && range.begin < r.end) {
				return true;
			}
		}
		return false;
	};
	std::unique_lock lock(m_operation_mutex);
	CpPar::NoteLaneStat(lane, CpPar::LaneStat::DependencyChecks, 1);
	// Step 4: the youngest older open segment that wrote the range ends first (then the next one).
	for (;;) {
		uint64_t writer = UINT64_MAX;
		for (const auto& [other, ranges]: m_segment_ranges) {
			if (other >= order) {
				break;
			}
			if (overlaps(ranges.written)) {
				writer = other;
			}
		}
		if (writer == UINT64_MAX) {
			break;
		}
		const auto          start = CpPar::NowNs();
		HangWatchdog::Scope wait("cp-par-2b-dependency", reinterpret_cast<uint64_t>(this), lane, order,
		                         writer, address);
		m_turn_changed.wait(lock, [this, writer] { return OrderEndedLocked(writer); });
		CpPar::NoteLaneStat(lane, CpPar::LaneStat::DependencyWaits, 1);
		CpPar::NoteLaneStat(lane, CpPar::LaneStat::DependencyWaitNs, CpPar::NowNs() - start);
	}
	// Step 5: a younger open segment bound the range before this write.
	if (written) {
		for (auto it = m_segment_ranges.upper_bound(order); it != m_segment_ranges.end(); ++it) {
			if (overlaps(it->second.bound)) {
				CpPar::NoteLaneStat(lane, CpPar::LaneStat::OrderViolations, 1);
				static std::atomic<uint32_t> logged {0};
				if (logged.fetch_add(1, std::memory_order_relaxed) < 16u) {
					std::printf("Kyty CpPar 2b: order violation: segment %" PRIu64 " (lane %c) writes "
					            "0x%016" PRIx64 "+0x%" PRIx64 " that younger segment %" PRIu64
					            " already bound\n",
					            order, static_cast<char>('A' + lane), address, size, it->first);
				}
				break;
			}
		}
	}
	auto& mine = m_segment_ranges[order];
	mine.bound.push_back(range);
	if (written) {
		mine.written.push_back(range);
	}
}

// Astro Bot PC (KYTY_CP_PAR=2b, commandScheduler.h): the producer-side turn.
uint64_t CommandScheduler::HoldOrderLocked() const noexcept {
	const auto lane = MasterSemaphore::ThreadTickLane();
	if (lane >= MaxLanes) {
		return UINT64_MAX; // not a lane thread: queued (its tick is the oldest segment's)
	}
	const auto order = m_lane_order[lane];
	return order != UINT64_MAX && order != m_turn ? order : UINT64_MAX;
}

void CommandScheduler::AdvanceTurnLocked() {
	m_turn++;
	const auto held = m_held.find(m_turn);
	if (held == m_held.end()) {
		return;
	}
	// The new oldest segment's held operations: their ticks are above every queued one (older
	// segments' ranges), so both queues stay in tick order.
	const bool was_empty = m_priority_operations.empty();
	for (auto& operation: held->second.normal) {
		m_pending_operations.push(std::move(operation));
	}
	for (auto& operation: held->second.priority) {
		m_priority_operations.push(std::move(operation));
	}
	const bool priority = !held->second.priority.empty();
	m_held.erase(held);
	if (priority && (was_empty || !PriorityWakeupsBatched())) {
		m_priority_available.notify_one();
	}
}

void CommandScheduler::EndOrderLocked(uint64_t order) {
	m_segment_ranges.erase(order); // steps 4-5: fully recorded
	if (order == m_turn) {
		AdvanceTurnLocked();
		while (m_ended_early.erase(m_turn) != 0) {
			AdvanceTurnLocked();
		}
	} else {
		EXIT_IF(order < m_turn || !m_ended_early.insert(order).second);
	}
	uint32_t oldest = UINT32_MAX;
	for (uint32_t lane = 0; lane < m_lane_count && lane < MaxLanes; lane++) {
		if (m_lane_order[lane] == m_turn) {
			oldest = lane;
		}
	}
	m_master.SetOldestLane(oldest);
	m_turn_changed.notify_all();
}

void CommandScheduler::WaitLaneTurn(uint32_t lane) {
	std::unique_lock lock(m_operation_mutex);
	if (lane >= MaxLanes || m_lane_order[lane] == UINT64_MAX || m_lane_order[lane] == m_turn) {
		return;
	}
	const auto          start = CpPar::NowNs();
	HangWatchdog::Scope wait("cp-par-2b-submit-turn", reinterpret_cast<uint64_t>(this), lane,
	                         m_lane_order[lane], m_turn);
	m_turn_changed.wait(lock, [this, lane] { return m_lane_order[lane] == m_turn; });
	CpPar::NoteLaneStat(lane, CpPar::LaneStat::SubmitTurnWaits, 1);
	CpPar::NoteLaneStat(lane, CpPar::LaneStat::SubmitTurnNs, CpPar::NowNs() - start);
}

CommandBuffer& CommandScheduler::BeginLaneCommand() {
	EXIT_IF(!m_lanes || m_recorder == nullptr);
	auto& command = Cmd();
	EXIT_IF(!command.IsInvalid());
	auto& pool       = LanePool(ThreadContext());
	command.m_buffer = pool.Commit();
	command.Begin();
	GetDynamicState().Invalidate();
	// As BeginCommand with the recorder: the recorder begins the native buffer (in this lane's
	// ring) and stamps KYTY_GPU_TIMING / GpuOpProfiler with this recording's tick. In turn, no
	// other lane submits before this buffer does, so the tick is the one Submit allocates.
	// Lazy begin: the Begin packet is encoded only before the buffer's first packet (or a drain);
	// a segment that encodes nothing gives the buffer back (EndContextSegment).
	m_recorder->DeferBegin(command.m_buffer, m_master.CurrentTick(),
	                       m_gpu_timing ? GpuTiming::NowNs() : 0);
	if (m_eop_timestamps) {
		m_eop_timestamps->BeginCommand(command.StateSink());
	}
	return command;
}

void CommandScheduler::BeginContextSegment() {
	EXIT_IF(!m_dual);
	const auto context = LaneIndex(); // Astro Bot PC (KYTY_CP_PAR=2a): 0 or 1 outside 2a
	auto&      command = Cmd();
	if (m_lanes) {
		// Astro Bot PC (KYTY_CP_PAR=2a): the lane's previous segment submitted its buffer; this one
		// begins a new native buffer of the lane's own pool (Begin resets the wrapper's shadows).
		EXIT_IF(!command.IsInvalid() || command.m_rendering || command.m_direct_window ||
		        !command.m_pending.Empty());
		m_segment_submits[context] = m_submits[context];
		if (m_lanes_2b) {
			// Astro Bot PC (KYTY_CP_PAR=2b): the segment's tick range, in list order.
			std::lock_guard lock(m_operation_mutex);
			const auto      order = m_master.BeginRange(context);
			EXIT_IF(m_lane_order[context] != UINT64_MAX || order < m_turn);
			m_lane_order[context] = order;
			if (order == m_turn) {
				m_master.SetOldestLane(context);
			}
		}
		BeginLaneCommand();
		m_segment_packets[context] = m_recorder->StreamPackets(context);
		return;
	}
	// The other context may have submitted and begun a native buffer meanwhile.
	command.m_buffer = m_command.m_buffer;
	EXIT_IF(command.m_rendering || command.m_direct_window || !command.m_pending.Empty());
	// What CommandBuffer::Begin resets: nothing recorded by the other context is assumed.
	command.m_bound_pipelines = {};
	for (auto& state: command.m_descriptor_states) {
		state.layout    = nullptr;
		state.bound_set = nullptr;
	}
	for (auto& epoch: command.m_descriptor_epochs) {
		epoch++;
	}
	command.m_push_constants.valid = false;
	command.ResetBarrierState();
	GetDynamicState().Invalidate();
	m_segment_packets[context] = m_recorder->StreamPackets(context);
	m_segment_submits[context] = m_submits[context];
}

CommandScheduler::SegmentCounts CommandScheduler::EndContextSegment() {
	EXIT_IF(!m_dual);
	const auto context = LaneIndex(); // Astro Bot PC (KYTY_CP_PAR=2a): 0 or 1 outside 2a
	auto&      command = Cmd();
	if (m_lanes) {
		// Astro Bot PC (KYTY_CP_PAR=2a): the merge. The lane's own native buffer is submitted now,
		// in list order (Submit ends rendering, flushes the barrier batch, closes the direct window
		// and allocates the tick), then the Switch packet ends the ring segment.
		SegmentCounts counts;
		if (!command.IsInvalid()) {
			// Lazy begin: what Submit would encode before its packet comes first (the owner's
			// end-of-buffer work, the rendering end, the barrier batch); if the buffer's Begin is
			// still pending after that, nothing was encoded into it: it goes back to the pool and
			// nothing is submitted (no tick allocated; work deferred to the current tick waits for
			// the next submit, never less).
			if (m_pre_submit_hook != nullptr && !m_in_pre_submit && Active()) {
				m_in_pre_submit = true;
				m_pre_submit_hook(m_pre_submit_hook_context);
				m_in_pre_submit = false;
				EXIT_IF(command.IsInvalid());
			}
			command.EndRendering();
			command.FlushBarriers();
			// Astro Bot PC (KYTY_CP_PAR=2b): a segment skips its hand-off only if it can give its tick
			// range back (MasterSemaphore::TryReleaseRange); otherwise it submits, empty.
			bool skip = m_recorder->BeginPending();
			if (skip && m_lanes_2b) {
				std::lock_guard lock(m_operation_mutex);
				skip = m_master.TryReleaseRange(context);
			}
			if (skip) {
				EXIT_IF(command.m_rendering || !command.m_pending.Empty());
				command.CloseDirectWindow();
				m_recorder->CancelPendingBegin();
				LanePool(context).Return(command.m_buffer);
				command.m_buffer    = nullptr;
				counts.lane_skipped = true;
			} else {
				// Astro Bot PC (KYTY_CP_PAR=2b): a Begin still pending is encoded (empty hand-off).
				m_recorder->FlushPendingBegin();
				// Empty: only the Begin packet (or a drain marker) was encoded since the segment began.
				const bool empty = m_recorder->StreamPackets(context) - m_segment_packets[context] <= 1;
				// Astro Bot PC (KYTY_CP_PAR=2b): the hand-off submit takes the range's last tick.
				t_lane_end_submit = m_lanes_2b;
				Submit({}, false);
				t_lane_end_submit = false;
				counts.lane_submitted     = true;
				counts.lane_empty         = empty;
				counts.new_command_buffer = true;
			}
		} else {
			command.CloseDirectWindow();
		}
		// The Switch names the lane taking the token (N lanes: the replay cannot guess it).
		if (m_lanes_2b) {
			// Astro Bot PC (KYTY_CP_PAR=2b): the Switch ends the segment's order for the replay; the
			// producer-side turn moves on. (Only this thread writes its lane's entry: read unlocked;
			// the Switch is encoded outside the lock, its ring may be full.)
			const auto order = m_lane_order[context];
			EXIT_IF(order == UINT64_MAX);
			m_recorder->EndStreamSegment(CpPar::HandoffTarget(), order);
			std::lock_guard lock(m_operation_mutex);
			m_lane_order[context] = UINT64_MAX;
			EndOrderLocked(order);
		} else {
			m_recorder->EndStreamSegment(CpPar::HandoffTarget());
		}
		counts.packets          = m_recorder->StreamPackets(context) - m_segment_packets[context];
		counts.submits          = m_submits[context] - m_segment_submits[context];
		m_segment_tick[context] = CurrentTick();
		return counts;
	}
	if (Active() && !command.IsInvalid()) {
		// As Submit does before its packet: the segment's stream ends outside rendering, with no
		// batched barrier, upload copy or kept depth ordering left in this wrapper.
		command.EndRendering();
		command.FlushBarriers();
	}
	command.CloseDirectWindow();
	m_recorder->EndStreamSegment();
	SegmentCounts counts;
	counts.packets = m_recorder->StreamPackets(context) - m_segment_packets[context];
	counts.submits = m_submits[context] - m_segment_submits[context];
	if (counts.packets > 1 && m_segment_tick[context] != CurrentTick()) {
		m_segment_tick[context]   = CurrentTick();
		counts.new_command_buffer = true;
	}
	return counts;
}

CommandBuffer& CommandScheduler::BeginCommand() {
	if (m_lanes) {
		return BeginLaneCommand(); // Astro Bot PC (KYTY_CP_PAR=2a): the calling thread's lane
	}
	EXIT_IF(!m_command.IsInvalid());
	m_command.m_buffer = m_command_pool.Commit();
	auto& command      = Cmd(); // Astro Bot PC (KYTY_CP_PAR=1b2): the calling thread's context
	if (m_dual) {
		// Both wrappers follow the one native buffer; the other context's wrapper is reset at its
		// next segment's begin.
		m_command_b.m_buffer = m_command.m_buffer;
		m_dynamic_state_b.Invalidate();
	}
	command.Begin();
	m_dynamic_state.Invalidate();
	if (m_recorder != nullptr) {
		// The recorder begins the native buffer and stamps KYTY_GPU_TIMING / GpuOpProfiler there,
		// with this recording's tick and start time.
		m_recorder->Begin(command.m_buffer, m_master.CurrentTick(),
		                  m_gpu_timing ? GpuTiming::NowNs() : 0);
		if (m_eop_timestamps) {
			// KYTY_EOP_TIMESTAMPS=gpu: reset the slots whose results were read, as recorder
			// packets after the recorder's Begin (no drain). Without it no slot would ever become
			// free.
			m_eop_timestamps->BeginCommand(command.StateSink());
		}
		return command;
	}
	if (m_gpu_timing) {
		// Reuse the retirement point that just recycled a command buffer: read completed pairs
		// without waiting, then reset and stamp this buffer's pair before any rendering begins.
		m_gpu_timing->Collect(m_master.KnownGpuTick());
		m_gpu_timing->BeginCommand(m_command.m_buffer);
	}
	if (m_eop_timestamps) {
		// Outside rendering: reset the guest timestamp slots whose results were read.
		m_eop_timestamps->BeginCommand(m_command.StateSink());
	}
	if (m_gpu_ops) {
		GpuOpProfiler::OnBeginCommand(m_graphics, m_command.m_buffer, m_master.CurrentTick(),
		                              m_master.KnownGpuTick());
	}
	return m_command;
}

uint64_t CommandScheduler::Submit(SubmitInfo submit, bool force_completion) {
	GpuOpProfiler::NoteSubmitStack(); // Astro Bot PC (KYTY_RENDER_END_STATS)
	const CpGaps::Scope gap(CpGaps::Cat::Flush);
	HangWatchdog::NoteSubmission();
	if (m_dual && CpPar::Concurrent()) {
		// Astro Bot PC (KYTY_CP_PAR=1b3): the run-ahead's unmerged packets are merged first.
		CpPar::BeforeNativeSubmit();
	}
	auto& command = Cmd(); // Astro Bot PC (KYTY_CP_PAR=1b2): the calling thread's context
	EXIT_IF(command.IsInvalid());
	// Astro Bot PC (KYTY_CP_PAR=2a): in turn, only the submitting lane names a native buffer; a
	// younger lane's unsubmitted work would otherwise be overtaken by this tick.
	if (m_lanes) {
		for (uint32_t lane = 0; lane < m_lane_count; lane++) {
			EXIT_IF(&LaneCmd(lane) != &command && !LaneCmd(lane).IsInvalid());
		}
	}
	EXIT_IF(submit.num_wait_semaphores > SubmitInfo::MaxSemaphores ||
	        submit.num_signal_semaphores >= SubmitInfo::MaxSemaphores);
	if (m_pre_submit_hook != nullptr && !m_in_pre_submit && Active()) {
		// Work its owner deferred to the end of this command buffer (SetPreSubmitHook).
		m_in_pre_submit = true;
		m_pre_submit_hook(m_pre_submit_hook_context);
		m_in_pre_submit = false;
		EXIT_IF(command.IsInvalid());
	}
	for (auto* dependency: m_submit_dependencies) {
		if (dependency == nullptr) {
			continue;
		}
		// Staging bytes that recorded commands read may still be copied by a host worker (or, for
		// the upload DMA, by the transfer queue).
		if (const auto value = dependency->PendingValue(); value != 0) {
			if (!SubmitWaitBeforeSignal()) {
				// Waited for by the thread that submits the batch (SubmitDependency).
				submit.AddHostDependency(dependency, value);
			}
			const auto semaphore = dependency->Semaphore();
			if (semaphore == nullptr) {
				continue; // host work: done before vkQueueSubmit
			}
			if (submit.num_wait_semaphores < SubmitInfo::MaxSemaphores) {
				submit.AddWait(semaphore, value, dependency->WaitStages());
			} else {
				dependency->WaitHost(value);
			}
		}
	}
	// KYTY_TEXTURE_SPARSE_RESIDENCY: memory bound behind image levels on the side queue
	// (GraphicContext::BindSparseImageLevels, already submitted) before this batch uses it.
	if (const auto sparse = m_graphics.SparseBindPendingValue(); sparse != 0) {
		if (submit.num_wait_semaphores < SubmitInfo::MaxSemaphores) {
			submit.AddWait(m_graphics.SparseBindSemaphore(), sparse,
			               vk::PipelineStageFlagBits::eAllCommands);
		} else {
			m_graphics.WaitSparseBinds(); // no free wait slot: the binds are done before submission
		}
	}
	const auto count_boundary = [this, &submit, force_completion] {
		if (!Profiler::AggregateEnabled()) {
			return;
		}
		// Explicit waits/signals independently prevent native submit coalescing. Include
		// them as generic obstacles so EopOnly describes the potential optimization scope.
		// The caller holds m_operation_mutex across this and the submit tick transition.
		const bool generic = m_diagnostic_generic_completion || force_completion ||
		                     submit.num_wait_semaphores != 0 || submit.num_signal_semaphores != 0;
		const auto kind = m_diagnostic_eop_completion
		                      ? (generic ? Profiler::FrameEvent::SubmitBoundaryEopMixed
		                                 : Profiler::FrameEvent::SubmitBoundaryEopOnly)
		                      : (generic ? Profiler::FrameEvent::SubmitBoundaryGeneric
		                                 : Profiler::FrameEvent::SubmitBoundaryUnprotected);
		Profiler::CountFrameEvent(kind);
		m_diagnostic_eop_completion     = false;
		m_diagnostic_generic_completion = false;
	};

	const uint64_t submit_ns = m_gpu_timing ? GpuTiming::NowNs() : 0;
	if (m_recorder != nullptr) {
		// KYTY_CP_RECORDER: the final render-pass end and barrier batch are encoded like every
		// other command (or recorded natively in an open direct window); the recorder then stamps
		// the timing end, ends the buffer and hands it to the broker or the queue. The tick is
		// allocated here, exactly as in queued mode.
		command.EndRendering();
		command.FlushBarriers();
		command.CloseDirectWindow();
		CommandStream::SubmitPacket packet;
		packet.submit       = submit;
		packet.submit_ns    = submit_ns;
		packet.debug_op     = command.m_debug_op;
		packet.debug_submit = command.m_debug_submit_id;
		packet.debug_arg0   = command.m_debug_arg0;
		packet.debug_arg1   = command.m_debug_arg1;
		packet.debug_arg2   = command.m_debug_arg2;
		packet.debug_arg3   = command.m_debug_arg3;
		packet.debug_arg4   = command.m_debug_arg4;
		if (m_lanes_2b && !t_lane_end_submit) {
			// Astro Bot PC (KYTY_CP_PAR=2b): a younger segment submits only once it is the oldest.
			WaitLaneTurn(LaneIndex());
		}
		{
			std::lock_guard lock(m_operation_mutex);
			count_boundary();
			// Astro Bot PC (KYTY_CP_PAR=2b): the calling lane's tick range.
			packet.tick = m_lanes_2b ? m_master.NextLaneTick(LaneIndex(), t_lane_end_submit)
			                         : m_master.NextTick();
			if (m_lanes_2b) {
				packet.lane_order = m_lane_order[LaneIndex()]; // the replay's order gate
			}
			m_last_submit_tick.store(packet.tick, std::memory_order_release);
			packet.submit.AddSignal(m_master.Handle(), packet.tick);
			packet.preserve_completion =
			    (force_completion || m_preserve_current_completion) ? 1u : 0u;
			m_preserve_current_completion = false;
		}
		m_recorder->Submit(packet);
		command.m_buffer = nullptr;
		if (m_dual) {
			// Astro Bot PC (KYTY_CP_PAR=1b2): the other wrapper named the same native buffer
			// (2a: each lane names its own; the other one names none).
			if (!m_lanes) {
				m_command.m_buffer   = nullptr;
				m_command_b.m_buffer = nullptr;
			}
			m_submits[LaneIndex()]++;
		}
		return packet.tick;
	}
	{
		KYTY_PROFILER_DETAIL_BLOCK("CommandScheduler::SubmitEnd");
		if (m_gpu_timing) {
			// End's own EndRendering becomes a no-op; the end stamp follows the final store ops.
			command.EndRendering();
			// Pending batched barriers belong before the end stamp, as when recorded directly.
			command.FlushBarriers();
			m_gpu_timing->EndCommand(command.m_buffer);
		}
		command.End();
	}
	const auto buffer   = command.m_buffer;
	auto&      graphics = m_graphics;
	EXIT_IF(graphics.queue == nullptr);

	if (graphics.submission_queue.Enabled()) {
		KYTY_PROFILER_DETAIL_BLOCK("CommandScheduler::QueueDispatch");
		QueuedSubmission queued {.submit = submit,
		                         .progress = m_master.GetSubmissionProgress(),
		                         .master_semaphore = m_master.Handle(),
		                         .command = buffer,
		                         .debug_op = command.m_debug_op,
		                         .debug_submit = command.m_debug_submit_id,
		                         .debug_arg0 = command.m_debug_arg0,
		                         .debug_arg1 = command.m_debug_arg1,
		                         .debug_arg2 = command.m_debug_arg2,
		                         .debug_arg3 = command.m_debug_arg3,
		                         .debug_arg4 = command.m_debug_arg4};
		{
			// The callback runner may already have popped a callback. Record the
			// boundary at registration, never by inspecting the callback queues here.
			std::lock_guard lock(m_operation_mutex);
			count_boundary();
			queued.tick = m_master.NextTick();
			queued.submit.AddSignal(m_master.Handle(), queued.tick);
			queued.preserve_completion = force_completion || m_preserve_current_completion;
			m_preserve_current_completion = false;
		}
		const auto tick = queued.tick;
		if (m_gpu_timing) {
			// The broker stores the native submit time through this pointer before it publishes
			// the tick; the slot is not reused until KnownGpuTick() covers the tick.
			queued.dispatch_ns = m_gpu_timing->Submitted(tick, submit_ns);
		}
		graphics.submission_queue.Enqueue(std::move(queued));
		command.m_buffer = nullptr;
		return tick;
	}

	// Direct submission: this thread waits for the work the batch reads (SubmitDependency).
	submit.WaitHostDependencies();
	vk::Result result;
	uint64_t   tick;
	{
		KYTY_PROFILER_DETAIL_BLOCK("CommandScheduler::QueueDispatch");
		Common::LockGuard lock(graphics.queue_mutex);
		if (Profiler::AggregateEnabled()) {
			std::lock_guard operation_lock(m_operation_mutex);
			count_boundary();
			tick = m_master.NextTick();
		} else {
			tick = m_master.NextTick();
		}
		submit.AddSignal(m_master.Handle(), tick);

		vk::TimelineSemaphoreSubmitInfo timeline_info {};
		timeline_info.waitSemaphoreValueCount   = submit.num_wait_semaphores;
		timeline_info.pWaitSemaphoreValues      = submit.wait_ticks.data();
		timeline_info.signalSemaphoreValueCount = submit.num_signal_semaphores;
		timeline_info.pSignalSemaphoreValues    = submit.signal_ticks.data();

		vk::SubmitInfo submit_info {};
		submit_info.pNext                = &timeline_info;
		submit_info.waitSemaphoreCount   = submit.num_wait_semaphores;
		submit_info.pWaitSemaphores      = submit.wait_semaphores.data();
		submit_info.pWaitDstStageMask    = submit.wait_stages.data();
		submit_info.commandBufferCount   = 1;
		submit_info.pCommandBuffers      = &buffer;
		submit_info.signalSemaphoreCount = submit.num_signal_semaphores;
		submit_info.pSignalSemaphores    = submit.signal_semaphores.data();

		{
			KYTY_PROFILER_DETAIL_BLOCK("CommandScheduler::DriverSubmit");
			HangWatchdog::Scope native(
			    "vkQueueSubmit-direct",
			    reinterpret_cast<uint64_t>(static_cast<VkQueue>(graphics.queue)), tick, 0,
			    submit.num_wait_semaphores, submit.num_signal_semaphores);
			NoteWatchdogSubmit(graphics.queue, submit_info, tick);
			result = graphics.queue.submit(1, &submit_info, nullptr);
		}
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::QueueSubmits);

	if (result == vk::Result::eErrorDeviceLost) {
		DumpDeviceLossDiagnostics(graphics, tick);
	}
	if (result != vk::Result::eSuccess) {
		ReportVulkanFatal("vkQueueSubmit", result, tick, command.m_debug_op,
		                  command.m_debug_submit_id, command.m_debug_arg0,
		                  command.m_debug_arg1, command.m_debug_arg2, command.m_debug_arg3,
		                  command.m_debug_arg4);
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
	if (m_gpu_timing) {
		if (auto* dispatch_ns = m_gpu_timing->Submitted(tick, submit_ns); dispatch_ns != nullptr) {
			*dispatch_ns = GpuTiming::NowNs();
		}
	}

	command.m_buffer = nullptr;
	return tick;
}

void CommandScheduler::WaitRecorded(uint64_t tick, bool from_producer) {
	if (m_recorder != nullptr && m_recorder->GetMode() == CommandRecorder::Mode::Thread) {
		m_recorder->WaitRecorded(tick, from_producer);
	}
}

void CommandScheduler::BeginNext() {
	KYTY_PROFILER_DETAIL_BLOCK("CommandScheduler::BeginNext");
	const CpGaps::Scope gap(CpGaps::Cat::Flush);
	CheckActive();
	BeginCommand();
}

} // namespace Libs::Graphics
