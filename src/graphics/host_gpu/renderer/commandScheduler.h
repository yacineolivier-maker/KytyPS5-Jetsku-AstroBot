#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_

#include "common/common.h"
#include "common/uniqueFunction.h"
#include "graphics/host_gpu/renderer/drawPrep/laneShare.h"
#include "graphics/host_gpu/renderer/dynamicState.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"
#include "graphics/host_gpu/renderer/render.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <thread>
#include <vector>

namespace Libs::Graphics {

class GpuTimestampRing;
class CommandRecorder;
class EopTimestampRing;

// SubmitDependency (queueSubmission.h): host work that commands of the current recording read and
// that finishes after recording (TextureCache staging copies, upload DMA transfers). Submit hands
// it to the batch; the thread that calls vkQueueSubmit waits for it, so the recording thread never
// blocks on it and no GPU wait precedes its signal.

class CommandScheduler {
public:
	// Diagnostic attribution only; both kinds retain the same completion boundary.
	enum class PriorityOperationKind { Generic, EopInterrupt };
	// Only the guest scheduler carries KYTY_GPU_TIMING timestamps. Presenter submissions wait on
	// image acquisition at the transfer stage, so their top-of-pipe spans would include that wait.
	enum class Role { Guest, Presenter };

	CommandScheduler(RenderContext& context, GraphicContext& graphics, Role role);
	~CommandScheduler();
	KYTY_CLASS_NO_COPY(CommandScheduler);

	void           Begin(HW::Context& registers, HW::UserConfig& user_config, HW::Shader& shaders);
	void           BeginRendering(const RenderState& state);
	void           EndRendering();
	void           Flush();
	void           Flush(SubmitInfo& submit);
	void           FlushAndWait();
	// Submits the current recording with an observable completion (as Wait does for the current
	// tick, without the wait) and begins the next one; returns the submitted tick.
	uint64_t       FlushObservable();
	void           Finish();
	CommandBuffer& BeginCommand();
	uint64_t       Submit(SubmitInfo submit = {}, bool force_completion = false);
	// Deferred callbacks can observe an externally owned drain, but cannot initiate shutdown:
	// the priority runner cannot join itself.
	void                      Shutdown();
	void                      Wait(uint64_t tick);
	// Runs every normal operation whose tick has completed, in order. Before each one it waits for
	// the priority operations of that tick and earlier: a normal operation may free what they
	// still use.
	void                      PopPendingOperations();
	// The same, for the draw and dispatch paths' opportunistic housekeeping. With
	// KYTY_PENDING_OPS_NOWAIT (default on) it never waits for the priority runner. At the first
	// completed operation whose tick still has priority operations pending, it stops and leaves
	// that operation and those after it queued, in order, for a later pop. Every blocking caller
	// (Finish, the fault manager, explicit waits) still uses PopPendingOperations.
	void                      PopReadyOperations();
	void                      DrainPriorityOperations();
	// KYTY_PRIORITY_WAIT_SPIN_US (default 0): first spin that long for the priority runner.
	void                      WaitPriorityOperations(uint64_t tick);
	void                      DeferOperation(Common::UniqueFunction<void>&& operation);
	// Astro Bot PC (KYTY_CP_PAR_CONCURRENT=1, cpParallel.h): DeferOperation for the destruction of an
	// object several lanes may use (cached buffers and images, shared scratch). With lanes running
	// at the same time, a younger open segment may already have recorded a use of it at a later
	// tick than the caller's: the operation is then held with the youngest open segment and queued
	// at its tick when it becomes the oldest (nothing waits for these operations). Otherwise, and
	// outside that mode, exactly DeferOperation.
	void                      DeferRetire(Common::UniqueFunction<void>&& operation);
	void                      DeferPriorityOperation(
	    Common::UniqueFunction<void>&& operation,
	    PriorityOperationKind kind = PriorityOperationKind::Generic);
	[[nodiscard]] static bool InDeferredOperation() noexcept;
	// Astro Bot PC (KYTY_CP_PAR=1b5, cpParallel.h): while set on the calling thread, its
	// EopInterrupt priority operations are handed to `capture` first; when it returns true it kept
	// the operation (the run-ahead queues it until its merge) and nothing is queued here. nullptr
	// clears it.
	using PriorityCapture = bool (*)(Common::UniqueFunction<void>& operation);
	static void SetThreadPriorityCapture(PriorityCapture capture) noexcept;
	// Called on the completion runner after every priority operation (e.g. to wake queues
	// suspended on what it published). Clear it, then DrainPriorityOperations, before the
	// context dies: the call happens while the operation is still marked active.
	using ProgressHook = void (*)(void* context);
	void SetProgressHook(ProgressHook hook, void* context);
	// Called at the start of every Submit while the scheduler is active, on the recording thread,
	// before the command buffer is ended: an owner that defers work to the end of the command
	// buffer records it there (KYTY_OCCLUSION_BATCH: OcclusionCounter::FlushBatch). Never
	// re-entered from a Submit it causes. The owner clears it before it is destroyed.
	using PreSubmitHook = void (*)(void* context);
	void SetPreSubmitHook(PreSubmitHook hook, void* context) noexcept {
		m_pre_submit_hook         = hook;
		m_pre_submit_hook_context = context;
	}
	// KYTY_PRIORITY_WAKE_BATCH=0 restores a runner wake per queued operation and a waiter
	// broadcast after every operation.
	[[nodiscard]] static bool PriorityWakeupsBatched();

	// Draw-prep: the register set the current command buffer reads. A committed draw's buffer is
	// pointed at that draw's register snapshot (the live registers may already belong to later
	// packets) and restored afterwards; it survives command-buffer restarts in between because
	// the scheduler reuses one CommandBuffer wrapper.
	struct RegisterBinding {
		HW::Context*    registers   = nullptr;
		HW::UserConfig* user_config = nullptr;
		HW::Shader*     shaders     = nullptr;
	};
	[[nodiscard]] RegisterBinding BindRegisters(HW::Context& registers, HW::UserConfig& user_config,
	                                            HW::Shader& shaders) noexcept {
		auto&                 command = Cmd(); // Astro Bot PC (KYTY_CP_PAR=1b2)
		const RegisterBinding previous {command.m_registers, command.m_user_config,
		                                command.m_shaders};
		command.Bind(registers, user_config, shaders);
		return previous;
	}
	void RestoreRegisters(const RegisterBinding& binding) noexcept {
		auto& command         = Cmd(); // Astro Bot PC (KYTY_CP_PAR=1b2)
		command.m_registers   = binding.registers;
		command.m_user_config = binding.user_config;
		command.m_shaders     = binding.shaders;
	}
	// The bound shader registers, writable: the command processor's own SH register writes during
	// an op (an indirect multi-draw's per-record draw index).
	[[nodiscard]] HW::Shader* BoundShaders() const noexcept { return Cmd().m_shaders; }

	// Astro Bot PC (KYTY_CP_PAR_LANE_PREREQS=1, W13): the calling thread's wrapper (every wrapper is
	// bound together, CommandScheduler::Begin): a lane no longer reads lane A's register pointer,
	// which lane A rewrites at every committed draw (same value: never null once begun).
	[[nodiscard]] bool Active() const noexcept {
		return (LaneShare::g_prereqs ? Cmd() : m_command).m_registers != nullptr;
	}
	void                           CheckActive() const;
	CommandBuffer&                 Current();
	// Astro Bot PC (KYTY_CP_PAR=1b3): the calling thread's wrapper names a native buffer.
	[[nodiscard]] bool CurrentValid() const noexcept { return !Cmd().IsInvalid(); }
	[[nodiscard]] uint64_t         CurrentTick() const noexcept { return m_master.CurrentTick(); }
	// KYTY_EOP_TIMESTAMPS=gpu: the guest timestamp query ring; null in record mode.
	[[nodiscard]] EopTimestampRing* GuestTimestamps() const noexcept { return m_eop_timestamps.get(); }
	[[nodiscard]] bool             IsFree(uint64_t tick);
	[[nodiscard]] MasterSemaphore& GetMasterSemaphore() noexcept { return m_master; }
	[[nodiscard]] DynamicState&    GetDynamicState() noexcept {
		if (!m_dual) {
			return m_dynamic_state;
		}
		// Astro Bot PC (KYTY_CP_PAR=2a, N lanes): lanes 2.. have their own.
		const auto lane = ThreadContext();
		return lane == 0 ? m_dynamic_state : (lane == 1 ? m_dynamic_state_b : *m_dynamic_state_x[lane]);
	}
	[[nodiscard]] RenderContext&   Context() const noexcept { return m_context; }
	[[nodiscard]] GraphicContext&  Graphics() const noexcept { return m_graphics; }
	// Recording-thread only; the owner clears it before it is destroyed. Slot 0: TextureCache
	// staging copies; slot 1: BufferCache upload DMA.
	static constexpr uint32_t SubmitDependencySlots = 2;
	void SetSubmitDependency(SubmitDependency* dependency, uint32_t slot = 0) noexcept {
		m_submit_dependencies[slot] = dependency;
	}
	// KYTY_CP_RECORDER (commandRecorder.h): the guest scheduler's recorder, or null.
	[[nodiscard]] CommandRecorder* Recorder() const noexcept { return m_recorder.get(); }
	// Waits until the recorder has handed tick `tick` (and every older one) to the queue or the
	// submission broker; a no-op without a recorder. Code that assumes "every tick older than the
	// current recording reached the queue or the broker" calls it first. from_producer: called by
	// the recording thread (the recorder is then woken first). Never call it while holding
	// GraphicContext::queue_mutex: the recorder may need it to submit.
	void WaitRecorded(uint64_t tick, bool from_producer = true);

	// Astro Bot PC (KYTY_CP_PAR=1b2, cpParallel.h): a second recording context. Its CommandBuffer
	// wrapper (m_command_b: barrier batcher, pipeline/descriptor/push-constant shadows, rendering
	// instance, register binding) and DynamicState are the calling thread's when its recorder
	// stream (CommandRecorder::ThreadStream) is 1; both wrappers name the one native command
	// buffer being recorded. EnableContextB (once, before the second context records; needs the
	// recorder in thread mode without verify) returns whether it is active. A context's segment
	// lasts while its thread holds the CP token: EndContextSegment (the giver) ends its rendering
	// instance, flushes its barrier batch, closes its direct window and ends its recorder segment;
	// BeginContextSegment (the taker) starts its wrapper with no shadow state, as at a command
	// buffer's begin. Both on the calling thread's context.
	bool EnableContextB();
	[[nodiscard]] bool ContextB() const noexcept { return m_dual; }
	struct SegmentCounts {
		uint64_t packets            = 0; // packets encoded in the context's ring during the segment
		uint64_t submits            = 0; // Submit packets among them
		bool     new_command_buffer = false; // first segment of the context in its native buffer
		// Astro Bot PC (KYTY_CP_PAR=2a): the segment ended with the hand-off submit of the lane's
		// own native buffer; nothing had been recorded into it since the segment began.
		bool     lane_submitted = false;
		bool     lane_empty     = false;
		// Astro Bot PC (KYTY_CP_PAR=2a, lazy begin): nothing was encoded into the lane's buffer (its
		// Begin was still pending): it went back to the pool, nothing was submitted.
		bool     lane_skipped   = false;
	};
	void          BeginContextSegment();
	SegmentCounts EndContextSegment();
	// Astro Bot PC (KYTY_CP_PAR=2a, cpParallel.h): "lanes". Needs EnableContextB; once, before the
	// second context records. Context B's wrapper then names native buffers of its own command
	// pool (never context A's): each segment begins a native buffer of its lane's pool
	// (BeginContextSegment) and ends by submitting it (EndContextSegment, the merge in list order,
	// where its tick is allocated). Execution stays in turn: at most one wrapper names a native
	// buffer at any time (Submit checks it).
	// Astro Bot PC (KYTY_CP_PAR=2a, N lanes): `count` lanes (2..MaxLanes); lanes 2.. get their own
	// wrapper, DynamicState, command pool and recorder stream (CommandRecorder::EnableStreams).
	static constexpr uint32_t MaxLanes = 4;
	bool               EnableLanes(uint32_t count = 2);
	// Astro Bot PC (KYTY_CP_PAR=2b, cpParallel.h): after EnableLanes, start-up only. Each lane's
	// stream gets its own replay thread (CommandRecorder::EnableLaneReplay). Refused (2a goes on)
	// with KYTY_GPU_TIMING or the GPU op profiler (single-producer rings of the replay).
	// KYTY_EOP_TIMESTAMPS=gpu is allowed: its ring is used by the recording side only (lanes in
	// turn), its slot resets and timestamp writes are commands ordered on the GPU by the queue order,
	// and its read queue is in tick order (ticks stay increasing in list order).
	bool               EnableLanes2b();
	[[nodiscard]] bool Lanes2b() const noexcept { return m_lanes_2b; }
	// Astro Bot PC (KYTY_CP_PAR=2b): tick ranges (MasterSemaphore::EnableRanges) are on with 2b.
	// Every lane segment (BeginContextSegment) reserves the next range, in list order: its order.
	// Its hand-off submit takes the range's last tick; its Switch packet carries the order, and the
	// replay hands no buffer of a later order to the queue before that Switch (CommandRecorder::
	// EnableRangeOrder). Producer side, the "turn" is the order of the oldest segment not ended:
	//  - a submit of a younger segment that is not its hand-off waits for its turn (counted);
	//  - an operation deferred by a younger segment's thread (DeferOperation,
	//    DeferPriorityOperation) is held with that segment and queued when it becomes the oldest,
	//    so both queues stay in tick order (counted).
	// Executed in turn (P2 steps 1-2), the lanes are always the oldest: nothing waits, nothing is
	// held.
	// The thread's lane for the tick ranges (MasterSemaphore::SetThreadTickLane).
	static void SetThreadTickLane(uint32_t lane) noexcept { MasterSemaphore::SetThreadTickLane(lane); }
	// Astro Bot PC (KYTY_CP_PAR=2b, P2 steps 4-5): dependencies between lanes. One storage-buffer
	// binding by the calling lane thread's segment (`written`: the shader may write the range;
	// pipeline/descriptors.cpp NativeStorageBuffer, the GPU-thread binding site). A registry keeps,
	// per open segment order, the ranges it bound and the ones it wrote (dropped when the segment
	// ends: it is fully recorded then). Step 4: a binding of a range that an OLDER open segment
	// wrote waits until that segment has ended, so that it sees that segment's last write as serial
	// order would (counted with the time waited). Step 5: a write to a range that a YOUNGER open
	// segment already bound is an order violation (the younger one bound it before this write;
	// serial order is the other way round; only a re-execution, P4, could repair it): counted, the
	// first 16 logged. Active with KYTY_CP_PAR_VERIFY=1 (also when the lanes run in turn, as a
	// self-test of the bookkeeping: no other segment is then ever open, so nothing may wait or be
	// violated); without it the registry costs nothing (lanes in turn need none). Images and render
	// targets are not registered yet.
	void NoteLaneBuffer(uint64_t address, uint64_t size, bool written);
	// Astro Bot PC (KYTY_CP_PAR=2b, morceau (d)): lane `lane` has no open segment or its open
	// segment is the oldest (always in turn); the lane whose open segment follows lane `lane`'s
	// (its order + 1), or UINT32_MAX.
	[[nodiscard]] bool     LaneSegmentOldest(uint32_t lane);
	// Astro Bot PC (KYTY_CP_PAR=2b, suite 1): the calling thread's lane waits until its open
	// segment is the oldest (never in turn; counted as a submit turn wait). Never with the lanes'
	// lock held (the older segment needs it to end).
	void WaitSegmentTurn() {
		if (m_lanes_2b) {
			WaitLaneTurn(LaneIndex());
		}
	}
	[[nodiscard]] uint32_t LaneSuccessor(uint32_t lane);
	// Astro Bot PC (KYTY_CP_PAR=2b, obstacle 2): Thread_Gpu reserves the order (and tick range) of
	// lane `lane`'s next segment when it hands it a list, in list order (CpPar::LaneReserveEnabled);
	// the lane's BeginContextSegment consumes it (counted; a lane other than A beginning without a
	// reservation is counted too). The lane must have no open or reserved segment.
	void               ReserveLaneOrder(uint32_t lane);
	// The lane has an open (or reserved) segment order.
	[[nodiscard]] bool LaneOrderOpen(uint32_t lane);
	// Astro Bot PC (KYTY_CP_PAR_CONCURRENT=1, stall watchdog): the lanes' orders (UINT64_MAX: none),
	// reserved flags, the turn and the segment orders with held operations; false when the
	// operation lock is busy (never waits: the watchdog runs while threads may be stuck).
	bool DebugLaneOrders(std::array<uint64_t, MaxLanes>& orders, std::array<bool, MaxLanes>& reserved,
	                     uint64_t& turn, uint64_t& held_orders, uint64_t& last_submit);
	[[nodiscard]] bool Lanes() const noexcept { return m_lanes; }
	[[nodiscard]] uint32_t LaneCount() const noexcept { return m_lane_count; }
	// The calling thread's context (0 unless it records into stream 1).
	[[nodiscard]] static uint32_t ThreadContext() noexcept;

private:
	[[nodiscard]] CommandBuffer& Cmd() noexcept {
		return m_dual ? LaneCmd(ThreadContext()) : m_command;
	}
	[[nodiscard]] const CommandBuffer& Cmd() const noexcept {
		return m_dual ? const_cast<CommandScheduler*>(this)->LaneCmd(ThreadContext()) : m_command;
	}
	// Astro Bot PC (KYTY_CP_PAR=2a, N lanes): the wrapper of context `lane` (dual contexts on).
	[[nodiscard]] CommandBuffer& LaneCmd(uint32_t lane) noexcept {
		return lane == 0 ? m_command : (lane == 1 ? m_command_b : *m_command_x[lane]);
	}
	// The calling thread's context as an index of the per-context arrays.
	[[nodiscard]] static uint32_t LaneIndex() noexcept {
		const auto lane = ThreadContext();
		return lane < MaxLanes ? lane : MaxLanes - 1;
	}
	class CommandPool {
	public:
		CommandPool(GraphicContext& graphics, MasterSemaphore& master);
		~CommandPool();
		KYTY_CLASS_NO_COPY(CommandPool);

		vk::CommandBuffer Commit();
		// Astro Bot PC (KYTY_CP_PAR=2a, lazy begin): `buffer`, committed and never begun nor
		// submitted, is free again.
		void Return(vk::CommandBuffer buffer);
		// KYTY_CP_RECORDER: the recorder records from this pool; growing it (the only pool call
		// the recording thread makes) drains the recorder first.
		void SetRecorder(CommandRecorder* recorder) noexcept { m_recorder = recorder; }

	private:
		static constexpr size_t GrowStep = 4;

		size_t Grow();

		GraphicContext&                m_graphics;
		MasterSemaphore&               m_master;
		CommandRecorder*               m_recorder = nullptr;
		vk::CommandPool                m_pool = nullptr;
		std::vector<vk::CommandBuffer> m_buffers;
		std::vector<uint64_t>          m_ticks;
		size_t                         m_hint = 0;
	};

	enum class OperationState { Open, Draining, Closed };

	struct PendingOperation {
		Common::UniqueFunction<void> callback;
		uint64_t                     tick = 0;
	};

	void BeginNext();
	// Astro Bot PC (KYTY_CP_PAR=2a): BeginCommand of the calling thread's lane (its own pool).
	CommandBuffer& BeginLaneCommand();
	void PriorityOperationsThread(std::stop_token stop);
	void RunOperation(Common::UniqueFunction<void>&& operation);
	// Pops normal operations (see PopPendingOperations/PopReadyOperations).
	void PopOperations(bool wait_for_priority);
	// m_operation_mutex held: no priority operation of `tick` or earlier is queued or running.
	[[nodiscard]] bool PriorityDoneLocked(uint64_t tick) const noexcept;

	MasterSemaphore              m_master;
	RenderContext&               m_context;
	GraphicContext&              m_graphics;
	CommandPool                  m_command_pool;
	CommandBuffer                m_command;
	// Astro Bot PC (KYTY_CP_PAR=1b2): the second context's wrapper and dynamic state, and, per
	// context, its packets and submits at its segment's begin and the tick it last recorded in.
	CommandBuffer                m_command_b;
	DynamicState                 m_dynamic_state;
	DynamicState                 m_dynamic_state_b;
	bool                         m_dual = false;
	std::array<uint64_t, MaxLanes> m_segment_packets {};
	std::array<uint64_t, MaxLanes> m_segment_submits {};
	std::array<uint64_t, MaxLanes> m_submits {};
	std::array<uint64_t, MaxLanes> m_segment_tick {};
	// Astro Bot PC (KYTY_CP_PAR=2a): lane B's command pool (null unless EnableLanes), lanes on.
	std::unique_ptr<CommandPool> m_command_pool_b;
	bool                         m_lanes = false;
	bool                         m_lanes_2b = false; // Astro Bot PC (KYTY_CP_PAR=2b)
	// Astro Bot PC (KYTY_CP_PAR=2b, tick ranges, above; m_operation_mutex): each lane's open
	// segment order (UINT64_MAX: none), the turn, segments ended before their turn, the operations
	// held per segment order, the turn's waiters; the last tick submitted (the hand-off submit in
	// progress is a thread-local flag of commandScheduler.cpp).
	struct HeldOperations {
		std::vector<PendingOperation> normal;
		std::vector<PendingOperation> priority;
	};
	std::array<uint64_t, MaxLanes>           m_lane_order {UINT64_MAX, UINT64_MAX, UINT64_MAX, UINT64_MAX};
	// Astro Bot PC (KYTY_CP_PAR=2b, obstacle 2): m_lane_order[lane] was reserved by Thread_Gpu
	// (ReserveLaneOrder) and its segment has not begun yet.
	std::array<bool, MaxLanes>               m_lane_reserved {};
	uint64_t                                 m_turn = 0;
	std::set<uint64_t>                       m_ended_early;
	std::map<uint64_t, HeldOperations>       m_held;
	std::condition_variable                  m_turn_changed;

	std::atomic<uint64_t>                    m_last_submit_tick {0};
	// Astro Bot PC (KYTY_CP_PAR_CONCURRENT=1; m_operation_mutex): each lane's last submitted tick and
	// the largest tick submitted by the segments ended so far (all older than any open one): Finish
	// waits for those, not for a younger lane's out-of-turn submit (which the GPU runs only after
	// the caller's segment ends: waiting for it there would never end).
	std::array<uint64_t, MaxLanes>           m_lane_last_submit {};
	uint64_t                                 m_ended_submit_tick = 0;
	// Astro Bot PC (KYTY_CP_PAR=2b, steps 4-5; m_operation_mutex): the ranges each open segment
	// (by order) bound and wrote; whether the registry runs.
	struct LaneRange {
		uint64_t begin = 0;
		uint64_t end   = 0;
	};
	struct SegmentRanges {
		std::vector<LaneRange> bound;
		std::vector<LaneRange> written;
	};
	std::map<uint64_t, SegmentRanges>        m_segment_ranges;
	bool                                     m_deps_active = false;
	[[nodiscard]] bool OrderEndedLocked(uint64_t order) const noexcept {
		return order < m_turn || m_ended_early.count(order) != 0;
	}
	// m_operation_mutex held: segment `order` ended (the turn moves on; held operations of the new
	// oldest segment are queued).
	void EndOrderLocked(uint64_t order);
	void AdvanceTurnLocked();
	// m_operation_mutex held: the calling thread's segment order if its operations must be held
	// (a younger segment), else UINT64_MAX.
	[[nodiscard]] uint64_t HoldOrderLocked() const noexcept;
	// The calling thread's lane: a submit that is not its hand-off waits for its turn.
	void WaitLaneTurn(uint32_t lane);
	// Everything submitted so far (2b: the last submitted tick; otherwise CurrentTick() - 1).
	[[nodiscard]] uint64_t LastSubmittedTick() const noexcept {
		return m_lanes_2b ? m_last_submit_tick.load(std::memory_order_acquire) : CurrentTick() - 1;
	}
	// Astro Bot PC (KYTY_CP_PAR=2a, N lanes): lanes 2.. (index = lane): wrappers, dynamic states
	// and command pools (null below index 2 and unless EnableLanes(3+)); the lanes in use.
	std::array<std::unique_ptr<CommandBuffer>, MaxLanes> m_command_x;
	std::array<std::unique_ptr<DynamicState>, MaxLanes>  m_dynamic_state_x;
	std::array<std::unique_ptr<CommandPool>, MaxLanes>   m_command_pool_x;
	uint32_t                                             m_lane_count = 1;
	[[nodiscard]] CommandPool& LanePool(uint32_t lane) noexcept {
		return lane == 0 ? m_command_pool : (lane == 1 ? *m_command_pool_b : *m_command_pool_x[lane]);
	}
	std::queue<PendingOperation> m_pending_operations;
	std::queue<PendingOperation> m_priority_operations;
	std::mutex                   m_operation_mutex;
	std::condition_variable      m_operation_available;
	// The completion runner sleeps on its own condition, so a push can wake exactly it.
	std::condition_variable      m_priority_available;
	bool                         m_priority_active      = false;
	uint64_t                     m_priority_active_tick = 0;
	// Threads in WaitPriorityOperations/DrainPriorityOperations (m_operation_mutex).
	uint32_t                     m_priority_waiters     = 0;
	// Bumped by the priority runner after each operation; WaitPriorityOperations' spin polls it
	// instead of taking m_operation_mutex.
	std::atomic<uint64_t>        m_priority_progress {0};
	ProgressHook                 m_progress_hook         = nullptr;
	void*                        m_progress_hook_context = nullptr;
	PreSubmitHook                m_pre_submit_hook         = nullptr;
	void*                        m_pre_submit_hook_context = nullptr;
	bool                         m_in_pre_submit           = false;
	OperationState               m_operation_state      = OperationState::Open;
	// Guarded by m_operation_mutex, alongside callback registration and the
	// queued-submit tick transition. Captured into each owned submission record.
	bool                         m_preserve_current_completion = false;
	// Aggregate tracing reasons, guarded by m_operation_mutex. They never affect submission.
	bool                         m_diagnostic_eop_completion     = false;
	bool                         m_diagnostic_generic_completion = false;
	// KYTY_GPU_TIMING ring; null when disabled. Owned by the recording producer, like m_command.
	std::unique_ptr<GpuTimestampRing> m_gpu_timing;
	// KYTY_EOP_TIMESTAMPS=gpu query ring (guest scheduler only); null otherwise. Recording
	// producer, except the deferred-label reads (EopTimestampRing::TakeDeferred) on the priority
	// runner, which is joined before members declared above it are destroyed.
	std::unique_ptr<EopTimestampRing> m_eop_timestamps;
	// Guest scheduler with KYTY_GPU_OP_PROFILE / counters enabled (gpuOpProfiler.h).
	bool m_gpu_ops = false;
	// KYTY_PENDING_REFRESH_US: when the draw-entry pop last queried the GPU's progress (steady
	// clock nanoseconds; the recording producer only).
	uint64_t m_last_pending_refresh_ns = 0;
	std::array<SubmitDependency*, SubmitDependencySlots> m_submit_dependencies {};
	// KYTY_CP_RECORDER: the recorder thread of the guest scheduler (after m_gpu_timing, whose
	// ring it drives; destroyed before it).
	std::unique_ptr<CommandRecorder> m_recorder;
	// Declared last: the runner starts in the constructor and uses the members above.
	std::jthread m_priority_thread;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_
