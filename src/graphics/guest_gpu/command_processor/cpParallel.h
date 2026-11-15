#ifndef EMULATOR_SRC_GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_CPPARALLEL_H_
#define EMULATOR_SRC_GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_CPPARALLEL_H_

// Astro Bot PC (KYTY_CP_PAR), 13/11/2026: parallel translation of the command lists, stages 1a,
// 1b-i, 1b-ii (docs\PLAN-PARALLELE.md), plumbing without concurrency; 1b-iii-a (1b3), op-level
// concurrency under one lock.
//
// KYTY_CP_PAR unset or "0": off, the emulator behaves exactly as before (nothing here runs).
// KYTY_CP_PAR=1a (requires KYTY_CP_SEQ=1): the queue-0 graphics submissions of a frame are
// assigned to two "resolver contexts" by their rank since the last reset_processor submission
// (ranks 1, 3, 5... -> A; 2, 4... -> B). Each context has its own op ring
// (CommandProcessor::m_ops for A, m_ops_b for B): the sequencer emits a submission's ops into its
// context's ring. Op sequence numbers stay global (the rings share one numbering), so every
// sequencer wait threshold, the snapshot ring and the draw-prep window keep working unchanged:
// with fully serialized execution, the global op order is still the execution order.
//
// Context A is resolved by Thread_Gpu, as before. Context B is resolved by a second host thread,
// "CP resolver #2", with the same code (CommandProcessor::ResolveSubmission), but strictly in
// turn: Thread_Gpu hands it the token for one slice of a B submission (RunSlice) and waits until
// it gets the token back. Queue 0 is FIFO, so B starts submission N only after A finished N - 1,
// and A starts N + 1 only after B finished N. Only the executing thread changes.
//
// Thread identity: whoever holds the token is "the GPU thread". At every hand-off the giver runs
// Hooks::give (Thread_Gpu: resigns the renderer lock's ownership; resolver #2: also drops its
// borrowed g_gpu_thread / g_gpu_state) and the taker runs Hooks::take (becomes the RenderMutex
// owner; resolver #2 also borrows g_gpu_thread / g_gpu_state). Hand-offs happen only between ops,
// outside any renderer lock acquisition.
//
// What stays on Thread_Gpu: compute queues, flip preparations, the constant-engine handoff path,
// slice scheduling and flushing; service commands seen between ops and Flip / WaitFlipDone ops
// of a B submission are executed by Thread_Gpu at resolver #2's request (RunOnGpuThread) while
// resolver #2 waits. Service commands run from inside an op (draw-prep commit waits, predication
// waits on deferred labels) run on resolver #2 under the borrowed identity, Thread_Gpu being
// parked meanwhile (still fully serialized).
//
// KYTY_CP_PAR=1b1 (stage 1b-i; same execution in turn as 1a): context B also gets its own register
// snapshot ring (CpSeq::Sequencer::Snapshots(1)) and, with KYTY_DRAW_PREP=parallel, its own
// draw-prep engine (CommandProcessor::m_draw_prep_b: its own window and its own worker set,
// "DrawPrep B#k", KYTY_CP_PAR_B_WORKERS default 3, KYTY_CP_PAR_B_HOT default 1). The sequencer
// publishes a submission's draws (and takes its snapshots) in its context's window (ring), with
// its own pending-op lists; each context's resolver commits and retires only its own window, so a
// context's in-order retire no longer depends on the other context's progress. The worker
// threads are not shared between the two windows: a worker's loop, its parking gate and its wake
// predicates follow one window (workerGate.h RunPreparationWorker), and making them serve two
// would touch the default path.
//
// KYTY_CP_PAR=1b2 (stage 1b-ii; 1b1 plus, still executed in turn): context B records into its own
// target. Design "S" (stream level): the guest CommandScheduler has a second CommandBuffer
// wrapper (barrier batcher, pipeline/descriptor/push-constant shadows, rendering instance, its own
// DynamicState) and the CP recorder a second CommandStream ring and encoder. The thread decides
// the target (CommandRecorder::ThreadStream: resolver #2 -> B, every other thread -> A). Both
// wrappers name the same native command buffer; one recorder thread replays both rings into it.
// The order between the rings is carried by Switch packets: every hand-off of the token ends
// the giver's segment (rendering instance ended, barrier batch flushed, direct window closed,
// then a Switch packet: "continue in the other ring"), and the taker's segment begins with no
// CommandBuffer-level shadow state (as at a command buffer's begin). Ticks: unchanged, one tick
// per native command buffer (per Submit, whichever context submits); a Submit (a flush) of B is
// encoded in ring B, in its place in the global order, and both wrappers then follow the next
// native buffer. Needs KYTY_CP_RECORDER=1 (thread) without KYTY_CP_RECORDER_VERIFY; otherwise
// 1b2 runs as 1b1. KYTY_STREAM_COPY_DEFER is suspended in 1b2 (its ring positions are of one
// ring).
// Image layouts (CpPar::NoteImageState): per context segment, the first state each image was
// found in (its "entry requirement") and its last state; at the segment's end (the merge point,
// FixUpImage) the entry is compared with the state the image was left in by the segments merged
// before (or by an access outside any segment). In-turn execution keeps the global state exact,
// so every such fix-up must be a no-op: "CpPar 10s" counts the ones that would not be.
//
// Fallbacks (the submission stays in context A, on Thread_Gpu) are counted by reason; a
// "CpPar 10s: ..." console line reports them every 10 s with the submissions and resolving time
// of each thread.
//
// KYTY_CP_PAR=1b3 (stage 1b-iii-a; needs what 1b2 needs, otherwise runs as 1b1): real concurrency
// of the two resolver threads, op by op, under one "CP op lock" (OpLockAcquire below) that
// serializes every op touching the renderer (the caches stay serialized, as with the renderer
// lock). Design "run-ahead":
//  - Thread_Gpu resolves every queue-0 submission in queue order, as without KYTY_CP_PAR (both
//    contexts; a context = op ring, draw-prep window, snapshot ring, recorder stream and
//    CommandBuffer wrapper of its submissions). It is always the OLDEST context and holds the op
//    lock all the time, except at its interleave points (below) and while it waits for the
//    run-ahead to stop.
//  - CP resolver #2 is the RUN-AHEAD: while Thread_Gpu resolves submission N, it executes the
//    leading ops of submission N + 1 when N + 1 is admitted and belongs to the other context
//    (ranks alternate), each op under the op lock. It executes only "free" ops: StreamBegin,
//    SkipSlots, direct draws and direct dispatches that carry their registers (snapshot or
//    draw-prep slot) and do not inherit the instance count. At its first other op it stops (a
//    "promotion": that op and the rest of N + 1 are executed by Thread_Gpu, after N has been
//    merged, i.e. serially and in admission order). Thread_Gpu takes N + 1 over (hand-over) when it
//    reaches it in the queue.
//  - Thread_Gpu releases the op lock between two of its ops only while N's remaining ops are known
//    (the sequencer emitted N's StreamEnd) and contain no op that may submit the native command
//    buffer on purpose, wait, flip, write guest memory the run-ahead may read or need a service
//    command (TailQuiet: only direct draws and dispatches, end-of-pipe / RELEASE_MEM / EVENT_WRITE
//    but occlusion dumps). Meanwhile its optional submits (idle flushes, batched end-of-pipe
//    flushes, eager readback flushes), slice yields and service commands are deferred. So A and B
//    interleave op by op in N's "quiet tail".
//  - Recording order = admission order: the run-ahead records into its own recorder ring, after
//    that ring's last Switch; the replay reaches it only when N's segment ends (the merge at N's
//    StreamEnd: Switch into the run-ahead's ring). Image layouts: while it runs ahead, every image
//    the run-ahead touches gets a per-context overlay (Image::CpParTouch: its entry requirement,
//    the global state at the first touch, and its own current state, swapped in for each of its
//    ops); at the merge, a fix-up barrier from the state N left to each entry requirement (and one
//    global memory barrier) is recorded at the end of N's segment, before the Switch, then the
//    overlay's states become the global ones.
//  - Ticks: the run-ahead's packets are replayed into the native buffer current at the merge,
//    which is the one they were encoded for only if no native submit happened in between. A
//    native submit while the run-ahead has unmerged packets (or a recorder drain by the run-ahead,
//    which cannot reach its position otherwise) forces an "insertion": the run-ahead's packets so
//    far are merged at that point (fix-ups, Switch to its ring, Switch back). Lifetimes and
//    layouts stay exact; only the data order of N's remaining work and N + 1's ops so far is
//    inverted. Insertions are counted; KYTY_CP_PAR_VERIFY=1 stops the emulator at the first one.
//  - Not per context (serialized by the op lock instead): caches (buffer, texture, page),
//    descriptor heap, occlusion counter, deferred operations, the RenderExecutor (its
//    command-buffer shadows are forgotten whenever the op lock changes threads; draw-run
//    continuations end then). Per context: the CP fields the resolver loop keeps across ops
//    (swapped by the run-ahead around each of its ops). The sequencer's executed-op counter only
//    follows Thread_Gpu; the run-ahead wakes the sequencer after each op (its window and snapshot
//    slots) and the merge moves the counter past the run-ahead's ops. KYTY_STREAM_COPY_DEFER
//    stays off (dual streams).
// A "CpPar 10s: 1b3 ..." line reports the run-ahead (ops, overlap, promotions by reason, fix-ups,
// insertions, op lock waits).
//
// Astro Bot PC (KYTY_CP_PAR=1b4, 13/11/2026; needs what 1b3 needs, otherwise runs as 1b1): 1b3
// with a large run-ahead window. Measured (Aerial Garden, KYTY_CP_PAR_CENSUS): every WAIT_REG_MEM
// of a submission waits on an address last written by that same submission; the playable config
// writes end-of-pipe labels when they are recorded (KYTY_LABEL_MODE=record). So:
//  - The run-ahead also executes N + 1's own CP writes and its waits on them (RunAheadSubmission,
//    graphicsRun.cpp, "1b4 decision"): RELEASE_MEM / EVENT_WRITE_EOP labels and WRITE_DATA written
//    at record time, barrier-only RELEASE_MEM / EVENT_WRITE (cache flushes: a barrier recorded in
//    its own ring), DMA_DATA without GDS (buffer cache copies and fills, recorded in its ring, so
//    ordered after N on the GPU timeline), and WAIT_REG_MEM whose awaited dwords were last written
//    by N + 1 itself, still hold that value and are GPU-clean (no synchronizing read); it answers
//    the sequencer like Thread_Gpu. A "CP address book" (op lock) keeps the last CP writer and
//    value of every written dword, from both threads.
//  - It still promotes (serially, in order, as 1b3): any end-of-pipe interrupt or event-queue
//    signal (Interrupt: their order against N's interrupts is visible to the guest); clock and
//    timestamp writes (Clock); GDS reads and DMA_DATA with GDS (Gds: synchronizing, one GDS);
//    occlusion dumps (Occlusion); a label TryDeferLabel would defer (LabelDefer: completion/split
//    label modes, KYTY_LABEL_COMPLETION_RANGES, a pending proxy label, or any deferred label pending
//    in record mode; the completion runner is FIFO per tick, so N + 1's deferred write would land
//    before N's later ones: the run-ahead never queues deferred labels); a write to a GPU-dirty
//    range (GpuDirty: the write would synchronize); a write to an address that is not a learned
//    CP-internal label (External, KYTY_CP_PAR_WRITES=local, the default: a record-time write of
//    N + 1 is visible to the guest CPU before N's remaining labels, e.g. a "frame done" label that
//    lets the guest reuse memory N's remaining draws still upload; only dwords that a WAIT_REG_MEM
//    of the submission that wrote them has awaited, and that no other submission awaited, are taken
//    as CP-internal; KYTY_CP_PAR_WRITES=all allows any address, for measurement only); a wait on a
//    dword not last written by N + 1, changed since, unclean, or that would not pass (ForeignWait);
//    flips, LOD statistics, constant RAM dumps, predication, COND_EXEC, branches, lockstep reads,
//    indirect and register-less draws as in 1b3. Deferred end-of-pipe flush requests
//    (BufferFlushForEop) only count up: optional submits stay deferred while the run-ahead is dirty.
//  - Interleave points: Thread_Gpu releases the op lock before each op of N once no op that may
//    submit on purpose, flip, wait on a tick or need a service command remains in N ("hard": flips,
//    flip waits, predication, COND_EXEC, branches, lockstep reads, indirect draws and dispatches,
//    occlusion dumps, GDS end-of-pipe reads, DMA_DATA with GDS, LOD statistics, constant RAM
//    dumps, handoffs; published at N's StreamEnd as in 1b3, which the sequencer always emits
//    before any op of N + 1). N's labels, writes, DMA_DATA without GDS and waits interleave
//    (KYTY_CP_PAR_OLDEST_WRITES=hard makes N's WRITE_DATA / DMA_DATA / clock writes hard again:
//    the run-ahead's draws may read memory at record time that such a write of N changes). A wait
//    of N that does not pass, or a write of N that synchronizes, then submits while the run-ahead
//    is dirty: an insertion, as in 1b3 (counted; KYTY_CP_PAR_VERIFY=1 stops). "Once no hard op
//    remains" rather than "the next op is not hard": a hard op of N reached while the run-ahead has
//    unmerged packets forces an insertion, whatever stops it then; KYTY_CP_PAR_INTERLEAVE=next
//    applies the per-op rule anyway, to measure the difference.
//  - KYTY_CP_PAR_VERIFY=1 also stops at: a run-ahead wait satisfied by a value another submission
//    wrote; Thread_Gpu reading (WAIT_REG_MEM, DMA_DATA source) a dword the run-ahead wrote during
//    the overlap; Thread_Gpu writing a dword the run-ahead wrote or read during the overlap; the
//    run-ahead reading a dword whose last writer is N. Not checked: shader and draw-time reads.
// A "CpPar 10s: 1b4 ..." line (the 1b3 counters) and a "CpPar 10s: 1b4 window ..." line report
// run-ahead ops by kind, the new promotions, local waits and writes, interleave points and cross
// accesses.
//
// Astro Bot PC (KYTY_CP_PAR=1b5, 13/11/2026; needs what 1b4 needs, otherwise runs as 1b1): 1b4,
// and the run-ahead no longer promotes at end-of-pipe interrupts and event-queue signals (EOP
// INT_SEL 1/2, RELEASE_MEM INT_SEL 1/2/4: every path that ends in TriggerEopEventAtEndOfPipe):
//  - While it executes an op, the run-ahead thread's EopInterrupt priority operations are captured
//    (CommandScheduler::SetThreadPriorityCapture) into a per-run-ahead list instead of being
//    queued on the completion runner, with the op's sequence. An interrupt op's label is written
//    at record time when the book knows it as a CP-internal label (as 1b4's labels); otherwise
//    (External in 1b4) its store is captured too, in the same list, before the interrupt: the
//    guest does not see N + 1's signalled label before N's remaining work either.
//  - Delivery (GuestGpu::CpParMerge -> CommandProcessor::CpParDeliverDeferred, Thread_Gpu, op
//    lock held): at N's StreamEnd, right after N's last op and before any later op of N + 1 runs
//    on Thread_Gpu, i.e. where serial execution would have reached them at the earliest. In list
//    order: each captured store is written (memcpy, CP write notes, address book), then each
//    interrupt is queued with DeferPriorityOperation (EopInterrupt) for the current tick, after
//    every priority operation N queued (the runner is FIFO). So the label precedes its signal as
//    in serial, the signal fires only once the tick holding the run-ahead's replayed packets
//    completes, and the guest sees N's interrupts, then N + 1's, in sequence order. After an
//    insertion the list stays queued until the merge (the signal is then later than serial, never
//    earlier). A list still pending at a hand-over (shutdown) is delivered there, counted.
//  - Still promoted: interrupts whose label TryDeferLabel would defer (LabelDefer: the deferred
//    write raises the interrupt itself), GDS ones (Gds), clocks and the other 1b4 reasons.
//  - KYTY_CP_PAR_VERIFY=1 also stops when: a queue-0 interrupt is queued with an op sequence not
//    above the last one queued (deferred or serial: the guest would see them out of order); a
//    run-ahead op raises an interrupt that escapes the capture; a deferred list is not empty at a
//    new offer.
// A "CpPar 10s: 1b5 ..." line adds: deferred interrupts and stores delivered; a histogram stop
// reason -> op kind (the op it stopped at; "none": no op emitted yet); per run-ahead outcome
// (measured when Thread_Gpu completes N + 1): its ops and draws executed ahead, those left to
// Thread_Gpu, and a histogram of the fraction done ahead; and for the stop reason "stop", the
// window (offer to merge), the time the run-ahead waited for the sequencer to emit N + 1, and
// how many stops found no op of N + 1 emitted yet. The first 16 foreign-wait and external
// promotions are logged ("Kyty CpPar 1b5: ...") with the address, the last CP writer (submission,
// its rank among the queue-0 submissions of its frame fence, op kind) and whether memory still
// holds that value (if not, it was written outside the CP since: guest CPU or GPU).
//
// Astro Bot PC (KYTY_CP_PAR=2a, 14/11/2026, docs\CONCEPTION-MULTICOEUR.md step P1 "decoupled lanes,
// in turn"; needs what 1b2 needs, otherwise runs as 1b1): 1b2 (the token, strictly in turn), and
// each context ("lane": A = Thread_Gpu, B = CP resolver #2) also has
//  - its own native command buffers, from its own command pool (CommandScheduler::EnableLanes):
//    the two CommandBuffer wrappers no longer name one shared native buffer;
//  - its own RenderExecutor (RenderContext::GetRenderExecutor picks the calling thread's lane;
//    context B's draw-prep workers read B's texture binding memo);
//  - (as 1b2) its own DynamicState, recorder ring and encoder; one recorder thread still replays
//    both rings, following the Switch packets.
// Submission in list order: at every token hand-off the giver's segment ends with a Submit of its
// own native buffer (the "merge"; its tick is allocated then, CommandScheduler::Submit), then the
// Switch packet; the taker's segment begins a new native buffer of its own pool. So at most one
// native buffer is open at any time (checked at every Submit), ticks stay monotonic in queue order
// and a younger lane never submits while an older one has unsubmitted work. Cost: one extra
// submit per hand-off (empty when the segment recorded nothing: counted). Image layouts: 1b2's
// bookkeeping (fix-ups must be no-ops). A "CpPar 10s: 2a lanes ..." line reports the hand-off
// submits per lane and the empty ones.
// Lazy begin: a lane's Begin packet stays pending in its encoder until its first packet (or a
// recorder drain, which every direct window starts with); a segment that encoded nothing gives its
// native buffer back to the pool and submits nothing ("skipped").
// N lanes (KYTY_CP_PAR_LANES=2|3|4, default 2): contexts A..D, rank r of a frame -> context
// (r - 1) % N, one "CP resolver #k+1" thread per lane k >= 1, each lane with its own op ring,
// snapshot ring, draw-prep engine, recorder stream, wrapper, dynamic state, command pool and
// RenderExecutor. Still in turn: Thread_Gpu hands the token to one lane at a time (RunSlice(lane));
// the giver's Switch packet names the taker (HandoffTarget), so the one recorder thread follows
// the rings in list order.
//
// Astro Bot PC (KYTY_CP_PAR=2b, 15/11/2026, docs\CONCEPTION-MULTICOEUR.md step P2 "simultaneous
// lanes", built in steps; docs-cp-voies.md has the exact state): 2a, and
//  - per-lane replay (step 1): each lane's recorder stream has its own replay thread ("CP recorder
//    B".., CommandRecorder::EnableLaneReplay); the recorder thread replays stream 0 only and the
//    Switch packets no longer chain the rings. The native buffers are still handed to the queue in
//    list order: a replay thread hands a buffer over only after every earlier one (gate in the
//    native executor's Submit). Needs no KYTY_GPU_TIMING or GPU op profiler (KYTY_EOP_TIMESTAMPS=gpu: allowed since 3a)
//    (otherwise 2b runs as 2a). A "CpPar 10s: 2b" line reports each lane's replay (busy time, waits
//    for its turn).
//  - tick ranges (step 2, MasterSemaphore::EnableRanges, CommandScheduler): each lane segment
//    reserves, in list order, a range of ticks (its "order"); its submits take the range's ticks
//    in turn and its hand-off submit the range's last one, so the tick a lane records for is known
//    while it records, even with older lanes still recording (no provisional tick to translate:
//    everything deferred to CurrentTick() completes with the buffer holding the recording or a
//    later one). The replay hands a buffer to the queue only once every lower order has ended (the
//    Switch packet now carries the segment's order). Producer side, a younger segment's submit
//    that is not its hand-off waits for its turn, and the operations it defers are held until its
//    turn (queues stay in tick order). Executed in turn, nothing waits and nothing is held (the
//    2b line counts both).
//  - dependencies between lanes (steps 4-5, CommandScheduler::NoteLaneBuffer, storage buffers):
//    with KYTY_CP_PAR_VERIFY=1, a registry of the ranges each open segment bound and wrote; a
//    binding of a range an older open segment wrote waits for that segment's end, a write to a
//    range a younger open segment already bound is an order violation (counted, first 16 logged).
//    In turn no other segment is open: both stay 0 (self-test).
//  - step 3b-1: shared objects retired by another user than the last one (descriptor pools) retire
//    at the largest tick of their users; UploadDma's submit dependency is per lane. "early retires
//    avoided" counts the cases where the caller's tick would have been too early (0 in turn).
//  - step 3b-2: the lanes' global lock (LaneLockAcquire) around each op of a lane; IsGpuThread is
//    also true for its holder. Still in turn (the token): never contended.
// Not done yet (step 3b-3): simultaneous execution; the CP fields kept across ops, RenderMutex
// ownership and the sequencer's executed-op counter are still those of the token holder.

#include <cstdint>

namespace Libs::Graphics::CpPar {

enum class Mode : uint8_t {
	Off,
	Serial1a,
	Serial1b1,
	Serial1b2,
	Concurrent1b3,
	Concurrent1b4,
	Concurrent1b5, // Astro Bot PC (KYTY_CP_PAR=1b5)
	Lanes2a,       // Astro Bot PC (KYTY_CP_PAR=2a, P1: decoupled lanes, in turn)
	Lanes2b,       // Astro Bot PC (KYTY_CP_PAR=2b, P2: 2a plus per-lane replay and tick ranges)
};

// KYTY_CP_PAR, read once.
[[nodiscard]] Mode GetMode();
[[nodiscard]] inline bool Enabled() {
	return GetMode() != Mode::Off;
}
// Astro Bot PC (KYTY_CP_PAR=2a or 2b): the lane modes (2b is 2a plus its own additions, see above).
[[nodiscard]] inline bool LanesMode() {
	return GetMode() == Mode::Lanes2a || GetMode() == Mode::Lanes2b;
}
// KYTY_CP_PAR=1b1 (and 1b2, 1b3): context B has its own snapshot ring and draw-prep engine (see
// above).
[[nodiscard]] inline bool SeparateWindows() {
	return GetMode() == Mode::Serial1b1 || GetMode() == Mode::Serial1b2 ||
	       GetMode() == Mode::Concurrent1b3 || GetMode() == Mode::Concurrent1b4 ||
	       GetMode() == Mode::Concurrent1b5 || // Astro Bot PC (KYTY_CP_PAR=1b4, 1b5)
	       LanesMode();                        // Astro Bot PC (KYTY_CP_PAR=2a, 2b)
}
// KYTY_CP_PAR=1b2 (and 1b3) requested: context B records into its own target (see above).
// Whether it is active (recorder requirements met) is StreamsActive().
[[nodiscard]] inline bool SeparateStreams() {
	return GetMode() == Mode::Serial1b2 || GetMode() == Mode::Concurrent1b3 ||
	       GetMode() == Mode::Concurrent1b4 ||
	       GetMode() == Mode::Concurrent1b5 || // Astro Bot PC (KYTY_CP_PAR=1b4, 1b5)
	       LanesMode();                        // Astro Bot PC (KYTY_CP_PAR=2a, 2b)
}

// Astro Bot PC (KYTY_CP_PAR=1b2): set once at start-up, before resolver #2 runs, when the two
// recording targets are active; read by Image (state bookkeeping) without a call.
inline bool g_streams_active = false;
[[nodiscard]] inline bool StreamsActive() noexcept {
	return g_streams_active;
}
// Astro Bot PC (KYTY_CP_PAR=1b2 only): the image state hash bookkeeping below runs (set with
// g_streams_active; 1b3 tracks image states with its overlay instead).
inline bool g_hash_book = false;
[[nodiscard]] inline bool HashBookActive() noexcept {
	return g_hash_book;
}
// Astro Bot PC (KYTY_CP_PAR=1b3): set once at start-up (two recording targets and the run-ahead
// thread active), before Thread_Gpu starts.
inline bool g_concurrent = false;
[[nodiscard]] inline bool Concurrent() noexcept {
	return g_concurrent;
}
// Astro Bot PC (KYTY_CP_PAR=1b4): set with g_concurrent when the mode is 1b4 (large window).
inline bool g_wide = false;
[[nodiscard]] inline bool Wide() noexcept {
	return g_wide;
}
// Astro Bot PC (KYTY_CP_PAR=1b5): set with g_wide when the mode is 1b5 (deferred interrupts).
inline bool g_deferred = false;
[[nodiscard]] inline bool Deferred() noexcept {
	return g_deferred;
}
// Astro Bot PC (KYTY_CP_PAR=2a, P1): set once at start-up, with g_streams_active, when each
// lane also has its own native command buffers (pool), RenderExecutor and hand-off submits
// (CommandScheduler::EnableLanes). Read on the recording paths without a call.
inline bool g_lanes = false;
[[nodiscard]] inline bool Lanes() noexcept {
	return g_lanes;
}
// Astro Bot PC (KYTY_CP_PAR=2b, P2): set once at start-up, with g_lanes, when the 2b additions are
// active (see the 2b section above). Read on the recording paths without a call.
inline bool g_lanes_2b = false;
[[nodiscard]] inline bool Lanes2b() noexcept {
	return g_lanes_2b;
}
// Astro Bot PC (KYTY_CP_PAR=2a, N lanes): at most MaxLanes resolver contexts (A = Thread_Gpu,
// B, C, D = CP resolver #2, #3, #4). KYTY_CP_PAR_LANES=2|3|4 (2a only, default 2), read once:
// RequestedLanes() is what the start-up creates rings, snapshot rings, draw-prep engines and
// RenderExecutors for (2 in every other mode). g_lane_count: set with g_lanes (the lanes in use).
// ContextCount(): the rank split's modulus, (rank - 1) % ContextCount() -> context.
inline constexpr uint32_t MaxLanes = 4;
[[nodiscard]] uint32_t    RequestedLanes();
inline uint32_t           g_lane_count = 2;
[[nodiscard]] inline uint32_t ContextCount() noexcept {
	return g_lanes ? g_lane_count : 2u;
}
// The calling thread's resolver context: 0 for Thread_Gpu (and any other thread), k for CP
// resolver #k+1 (the 1b3 run-ahead thread: 1).
[[nodiscard]] uint32_t ThreadLane() noexcept;
// Astro Bot PC (KYTY_CP_PAR=2b, step 3b-2): the lane whose work the calling thread executes: the
// lane of the request Thread_Gpu is serving (RunOnGpuThread, inside RunSlice), else ThreadLane().
[[nodiscard]] uint32_t EffectiveLane() noexcept;
// The lane that takes the token at the hand-off in progress (set before the giver's give hook):
// a lane always gives it back to Thread_Gpu (0); Thread_Gpu gives it to the slice's lane. The
// giver's Switch packet names it (2a).
[[nodiscard]] uint32_t HandoffTarget() noexcept;
// Letter of a lane in the console lines ('A'..'D').
[[nodiscard]] inline char LaneLetter(uint32_t lane) noexcept {
	return static_cast<char>('A' + (lane < MaxLanes ? lane : MaxLanes - 1));
}
// "1b5", "1b4" or "1b3" (the concurrent mode's name in the console lines).
[[nodiscard]] inline const char* ConcurrentName() noexcept {
	return g_deferred ? "1b5" : (g_wide ? "1b4" : "1b3");
}
// KYTY_CP_PAR_WRITES=all (1b4): the run-ahead writes any address (not only learned CP-internal
// labels). KYTY_CP_PAR_OLDEST_WRITES=hard (1b4): N's WRITE_DATA / DMA_DATA / clock writes are hard
// (no interleave point before the last one). Read once.
[[nodiscard]] bool WritesAll();
[[nodiscard]] bool OldestWritesHard();
// KYTY_CP_PAR_INTERLEAVE=next (1b4, measurement): an interleave point before every op of N that is
// not hard itself, even with a hard op of N still ahead (that op then forces an insertion if it
// submits while the run-ahead is dirty). Default: once no hard op remains in N. Read once.
[[nodiscard]] bool InterleaveNext();
// KYTY_CP_PAR_VERIFY=1 (1b3): stop at the first insertion (data-order inversion). Read once.
[[nodiscard]] bool VerifyEnabled();
// The calling thread executes a run-ahead op before its merge: image accesses use its overlay
// (Image::CpParTouch).
inline thread_local bool t_overlay = false;
// Astro Bot PC (KYTY_CP_PAR=2b, morceau (d)): the overlay generalized to N lanes. g_lane_overlay:
// set once at start-up in 2b with the lanes' lock (KYTY_CP_PAR_OVERLAY=0 turns it off), before
// Thread_Gpu starts. t_overlay_lane: the lane whose overlay the calling thread's op uses while
// t_overlay is set (1b3: 0). The oldest open segment works on the global image states (as 1b3's
// oldest context); an op of a younger segment works on its lane's overlay (entry requirement at
// the first touch, its own state swapped in for the op, swapped back at the op's end, lanes' lock
// held). At the end of the oldest segment (its hand-off, lanes' lock held), its successor's
// overlay is merged: fix-up barriers from the global states to the successor's entry
// requirements are recorded at the end of the oldest segment's buffer (which the queue runs right
// before the successor's), then the successor's states become the global ones. In turn there is
// never a younger open segment: no overlay is ever used (the 1b2 book's "not no-ops" stays 0).
inline bool                  g_lane_overlay = false;
inline thread_local uint32_t t_overlay_lane = 0;
[[nodiscard]] bool LaneOverlayConfigured(); // 2b: KYTY_CP_PAR_OVERLAY (default on), read once
// The run-ahead has begun a recorder segment the replay has not reached ("dirty"): set and
// cleared under the op lock; read by the native submit and drain hooks and by Thread_Gpu's
// optional work (deferred while it is set).
[[nodiscard]] bool RunAheadDirty() noexcept;
void               SetRunAheadDirty(bool dirty) noexcept;

// ---- KYTY_CP_PAR=1b3: the CP op lock (FIFO), the run-ahead thread and its statistics ----
// who: 0 Thread_Gpu, 1 CP resolver #2. Returns true when the previous holder was the other one.
bool OpLockAcquire(uint32_t who);
void OpLockRelease(uint32_t who);
// Thread_Gpu: CP resolver #2 runs `task(argument)` (the run-ahead) as soon as it is free.
void OfferRunAhead(void (*task)(void*), void* argument);
// Thread_Gpu, holding the op lock: the run-ahead stops at its next op boundary (or never starts);
// returns once resolver #2 is idle, holding the op lock again.
// Returns true when resolver #2 held the op lock meanwhile (as OpLockAcquire).
bool StopRunAhead();
// Resolver #2, inside the run-ahead task.
[[nodiscard]] bool RunAheadStopRequested() noexcept;
// Sequencer thread: submission `submission` is fully emitted; `last_sync` is the sequence + 1 of
// its last op that is not quiet for the oldest context (0: none). Read by TailQuiet.
void PublishStreamEnd(uint64_t submission, uint64_t last_sync);
// Thread_Gpu: every op of `submission` from sequence `next_op` on is emitted and quiet.
[[nodiscard]] bool TailQuiet(uint64_t submission, uint64_t next_op) noexcept;
// Native submit (CommandScheduler::Submit) and recorder drain (CommandRecorder::Drain) hooks: an
// insertion runs first when the run-ahead is dirty (see above).
using InsertHook = void (*)(void* context, uint32_t trigger);
void SetInsertHook(InsertHook hook, void* context);
void BeforeNativeSubmit();
void BeforeRecorderDrain();
enum class Insertion : uint32_t { OldestSubmit, RunAheadSubmit, RunAheadDrain, Count };
enum class Promotion : uint32_t {
	StreamEnd,   // the submission's end (completion is Thread_Gpu's)
	Lockstep,    // WAIT_REG_MEM, predication, COND_EXEC, branches, lockstep reads, flip waits
	Write,       // labels, WRITE_DATA, DMA_DATA, clocks, LOD stats, constant RAM dumps
	Flip,        // flips
	Indirect,    // indirect draws and dispatches
	Inherit,     // a draw inheriting the instance count (back state of earlier draws)
	NoRegisters, // a draw or dispatch without its own registers (snapshot or prepared slot)
	Ring,        // its recorder ring is half full (the replay cannot drain it before the merge)
	Stop,        // hand-over, or the oldest submission ended first
	// Astro Bot PC (KYTY_CP_PAR=1b4): why a write or wait of the kinds the run-ahead may execute
	// is promoted anyway (see above).
	Interrupt,   // end-of-pipe interrupt / event-queue signal
	LabelDefer,  // TryDeferLabel would defer it (label mode, ranges, proxy, pending deferred labels)
	Clock,       // reference clock or timestamp write
	Gds,         // GDS read or DMA_DATA with GDS
	Occlusion,   // occlusion counter dump
	External,    // a write to an address that is not a learned CP-internal label
	GpuDirty,    // a write or wait on a GPU-dirty range (it would synchronize)
	ForeignWait, // a wait on a dword not last written (with that value) by this submission
	Count,
};
// Astro Bot PC (KYTY_CP_PAR=1b4) statistics: a run-ahead op of kind `kind` (CpSeq::OpKind) executed;
// a run-ahead wait satisfied by its own write; a run-ahead write to an address checked by the book
// (learned label, or any with KYTY_CP_PAR_WRITES=all); an interleave point of Thread_Gpu (offered:
// the op lock released before an op of N; refused: the run-ahead could run but N still has a hard op
// ahead); a cross access during the overlap (the verify checks above, counted when not stopping);
// the address book's sizes (written dwords, learned labels).
enum class Cross : uint32_t {
	OldestReadAhead,   // Thread_Gpu read a dword the run-ahead wrote
	OldestWriteAhead,  // Thread_Gpu wrote a dword the run-ahead wrote or read
	AheadReadOldest,   // the run-ahead read a dword whose last writer is N
	AheadForeignValue, // a run-ahead wait passed on a value it did not write
	Count,
};
void NoteRunAheadKind(uint32_t kind);
void NoteLocalWait();
void NoteLocalWrite();
void NoteInterleave(bool offered);
// Counts it; with KYTY_CP_PAR_VERIFY=1, stops the emulator (address for the message).
void NoteCross(Cross which, uint64_t address);
void NoteBook(uint64_t written, uint64_t labels);
// Astro Bot PC (KYTY_CP_PAR=1b5) statistics and checks (cpParallel.h, above):
// the run-ahead stopped for `reason` at an op of kind `kind` (CpSeq::OpKind; OpKind::Count: no op
// of N + 1 emitted yet);
void NoteStopKind(Promotion reason, uint32_t kind);
// Thread_Gpu completed a run-ahead target: its ops (non-transport) and draws (direct and indirect
// draws and dispatches) executed ahead and by Thread_Gpu after the run-ahead stopped;
void NoteOutcome(bool started, uint64_t ahead_ops, uint64_t ahead_draws, uint64_t rest_ops,
                 uint64_t rest_draws);
// at a merge: the offer-to-merge window; at a run-ahead's end: the time it waited for the
// sequencer to emit N + 1's ops;
void NoteWindow(uint64_t window_ns);
void NoteStarved(uint64_t ns);
// a deferred list delivered (interrupts, stores; `late`: at a hand-over, not at the merge);
void NoteDeferredDelivery(uint64_t interrupts, uint64_t stores, bool late);
// a run-ahead interrupt op whose label store was captured (External in 1b4) / written at record
// time (learned CP-internal label);
void NoteDeferredOp(bool store_captured);
// a queue-0 interrupt queued for op `sequence` (Thread_Gpu: serial or delivered): with
// KYTY_CP_PAR_VERIFY=1, stops when it is not above the last one (`deferred`: for the message);
void NoteInterruptOrder(uint64_t sequence, bool deferred);
// an interrupt raised on resolver #2 outside the capture (KYTY_CP_PAR_VERIFY=1: stops);
void NoteStrayInterrupt();
// a deferred list found non-empty where it must be empty (KYTY_CP_PAR_VERIFY=1: stops).
void NoteDeferredLeft(uint64_t entries);
// Rate limit of the "Kyty CpPar 1b5:" stop logs (true for the first 16 of `reason`).
[[nodiscard]] bool TakeStopLog(Promotion reason);
void NotePromotion(Promotion reason);
void NoteRunAheadStart();
// A run-ahead op executed, holding the op lock for `ns`.
void NoteRunAheadOp(uint64_t ns);
// A merge (at the oldest's end) or an insertion: fix-up barriers recorded, fix-ups impossible
// (entry requirement "undefined" against defined contents), overlay images.
void NoteMerge(uint64_t fixups, uint64_t impossible, uint64_t images);
void NoteInsertion(Insertion trigger, uint64_t fixups, uint64_t impossible, uint64_t images);
// A thread starts (busy) or stops (idle) resolving: overlap accounting.
void NoteBusy(uint32_t who, bool busy);
// Image state bookkeeping (see above), any thread: `image` changed (or was found unchanged by
// an access) from the state hashed as `before` to `after`. Attributed to the calling thread's
// open segment (resolver #2: B; Thread_Gpu: A), else applied as an access outside segments.
void NoteImageState(const void* image, uint64_t before, uint64_t after);
// The image is being destroyed: forget it.
void ForgetImage(const void* image);
// Token hand-off (the hooks, 1b2 only): the calling thread's context begins / ends a segment.
// EndSegment also takes the recording counts of the segment (packets encoded in its ring, Submit
// packets among them, whether it shared a native command buffer not seen before).
void BeginSegment(uint32_t context);
void EndSegment(uint32_t context, uint64_t packets, uint64_t submits, bool new_command_buffer);
// Astro Bot PC (KYTY_CP_PAR=2a): a lane's segment ended with its hand-off submit (its own native
// command buffer handed to the queue, in list order); `empty`: nothing was recorded in it;
// `skipped` (lazy begin): nothing was encoded, the buffer went back to its pool unsubmitted.
void NoteLaneHandoff(uint32_t context, bool empty, bool skipped);
// Astro Bot PC (KYTY_CP_PAR=2b) per-lane statistics (any thread; relaxed), reported every 10 s in
// the "CpPar 10s: 2b" line. Times in nanoseconds.
enum class LaneStat : uint32_t {
	ReplayBusyNs,    // the lane's replay thread replaying its ring
	ReplayTurnWaits, // the lane's replay thread waited for its turn to hand a buffer to the queue
	ReplayTurnNs,
	SubmitTurnWaits, // tick ranges: a younger segment's submit (not its hand-off) waited for its turn
	SubmitTurnNs,
	HeldOperations,  // tick ranges: operations deferred by a younger segment, held until its turn
	DependencyChecks, // KYTY_CP_PAR_VERIFY=1: storage-buffer bindings checked against the registry
	DependencyWaits,  // a binding waited for an older open segment that wrote the range to end
	DependencyWaitNs,
	OrderViolations,  // a write to a range a younger open segment already bound
	EarlyRetires,     // 3b-1: a retire / submit dependency at the caller's tick would have been too early
	LockTakes,        // 3b-2: the lanes' global lock taken around one op
	LockWaitNs,       //       time waiting for it
	LockHoldNs,       //       time holding it
	LockedBounces,    // morceau (c): a service / flip bounced to Thread_Gpu with the lane lock held
	LockDelegated,    //       Thread_Gpu re-took the lock as the delegated holder while serving it
	OcclusionRebatches, // morceau (b): a lane's occlusion batch reduced early (another lane's slots in between)
	OcclusionUnlocked,  //       an occlusion dump outside the lanes' global lock
	SeqPositionBelow,   // morceau (a): the sequencer's per-lane position below the former global counter
	OverlayOps,         // morceau (d): ops of a younger segment run on its lane's image overlay
	OverlayImages,      //       overlay images merged (at the end of the older segment)
	OverlayFixups,      //       fix-up barriers recorded there (not no-ops)
	OverlayImpossible,  //       fix-ups no barrier can express, or with no buffer to record into
	OverlayLeft,        //       a segment ended with its overlay unmerged (its older one still open)
	ForeignRetries,     // obstacle 1: suspended ops pending (any lane) when a lane starts a list
	ReservedOrders,     // obstacle 2: segment orders Thread_Gpu reserved for the lane at the hand-out
	UnreservedBegins,   //       a lane (not A) began a segment without a reserved order (0 expected)
	ListEndClosures,    //       lane A's segment closed (and reopened) at the end of an A list
	KeptSlices,         // obstacle 3: a slice ended (suspended / yielded) with the lane's segment kept open
	KeptBounces,        //       a flip / service bounce with the lane's segment kept open
	ResumedSegments,    //       the lane took the token back into its kept segment (no closing)
	ResumedNotOldest,   //       ... while its segment was not the oldest (0 expected)
	DelegatedRecordings, //      Thread_Gpu took the token and recorded into the lane's kept segment
	OpenAtListEnd,      //       a lane's list completed with its segment still open (0 expected)
	// Astro Bot PC (KYTY_CP_PAR_CONCURRENT=1, obstacles 4-5): reported in the "CpPar 10s: 2b
	// concurrent" line.
	Dispatched,         //       slices handed to the lane without waiting (RunSliceAsync)
	CollectedUnfinished, //      ... that came back with the list unfinished (kept, retried)
	ActiveNs,           //       time the lane translated (in a slice, not waiting for its turn or the lock)
	OutOfTurnSubmits,   //       a submit inside an op of a younger segment, lock held: not waited for
	VisibleNotOldest,   //       an op visible to the guest executed while not the oldest (0 expected)
	VisibleOutOfOrder,  //       ... with an op sequence below the last visible one (0 expected)
	DependencyHazards,  //       KYTY_CP_PAR_VERIFY: a dependency wait refused (lock held), counted only
	YoungerServices,    //       services not run by a younger segment (left to the oldest)
	RetiresHeld,        //       a shared object's retirement held for a younger open segment (DeferRetire)
	Count,
};
void NoteLaneStat(uint32_t lane, LaneStat stat, uint64_t value);
// Astro Bot PC (KYTY_CP_PAR=2b, step 3b-2): the lanes' global lock (FIFO, N lanes), taken by a
// lane thread around each op it executes (ResolveSubmission), so that once lanes run
// simultaneously (3b-3) every op touching the caches, image states, stream buffer, descriptor heap
// and renderer is serialized. In turn it is never contended (statistics only). On in 2b unless
// KYTY_CP_PAR_LOCK=0. Re-entrant on the holding thread. LaneLockHeld: the calling thread holds it
// (GuestGpu::IsGpuThread also answers true for it in 2b).
[[nodiscard]] bool LaneLockEnabled();
// Astro Bot PC (KYTY_CP_PAR=2b, obstacle 2): the order (tick range) of a lane's segment is
// reserved by Thread_Gpu when it hands the lane a list (right after its own segment ended; the
// lane's BeginContextSegment consumes it), and lane A's segment is closed at the end of every A
// list: segment orders follow list order whatever thread begins first. On in 2b unless
// KYTY_CP_PAR_RESERVE=0.
[[nodiscard]] bool LaneReserveEnabled();
// Astro Bot PC (KYTY_CP_PAR=2b, obstacle 3): a lane keeps its segment open until the end of its
// list. When it gives the token back for a request (flip, service: GivingForRequest) or at the end
// of a slice that did not complete its list (suspended wait, starved, yielded: the slice task calls
// SetListUnfinished on the lane thread), its segment does not end; Thread_Gpu, taking the token,
// records into that kept segment (its recording stream, tick lane and image book segment:
// AdoptSegment / DropSegment) until it gives the token back to the lane, which resumes the
// segment without beginning a new one. So no list handed out later can get an order older than
// the rest of an unfinished list. On in 2b unless KYTY_CP_PAR_KEEP=0 (then every slice and bounce
// ends the segment, as before).
[[nodiscard]] bool LaneKeepEnabled();
// Astro Bot PC (KYTY_CP_PAR=2b, KYTY_CP_PAR_CONCURRENT=1, obstacles 4 and 5): the lanes run their
// lists at the same time. Off unless KYTY_CP_PAR_CONCURRENT=1, and only with 2b active and the
// lanes' lock, the reservation (obstacle 2), the kept segments (obstacle 3) and the overlay (d) on;
// decided once at start-up (g_lane_concurrent, before Thread_Gpu starts). Then:
//  - queue-0 graphics lists go to lanes B.. only (rank r -> lane 1 + (r - 1) % (N - 1)); the lists
//    that must stay on A (constant engine) and flip preparations run on Thread_Gpu with no lane out;
//  - Thread_Gpu hands each list to its lane without waiting (RunSliceAsync), in list order, the
//    lane's order reserved at the hand-out, and goes on with the next list while the lane is free;
//    it collects the slices in list order (PollSlice), serves the lanes' requests meanwhile, and
//    hands out nothing past an unfinished (kept) list; its own work (other queues, services, A
//    lists) waits until no lane is out, or, with an unfinished list at the head, is done in that
//    list's kept segment under the lanes' lock;
//  - a younger segment waits for its turn (oldest open segment) before every op the guest can see
//    (labels, end-of-pipe, writes, waits, indirect or register-less draws, flips, services) and
//    before ending its list or keeping an unfinished one; direct draws and dispatches carrying
//    their registers run out of turn, under the lanes' lock, on the lane's image overlay.
// Statistics: "CpPar 10s: 2b concurrent" (overlap, turn waits by reason, controls).
inline bool g_lane_concurrent = false;
[[nodiscard]] inline bool LaneConcurrent() noexcept {
	return g_lane_concurrent;
}
[[nodiscard]] bool LaneConcurrentConfigured(); // KYTY_CP_PAR_CONCURRENT=1 in 2b, read once
// Why a lane waits for its turn (counted per reason in the concurrent line). The calling thread's
// reason is set for the scope of a wait (TurnReasonScope); Submit otherwise.
enum class TurnReason : uint32_t {
	Submit,   // a submit (scheduler), the default
	Label,    // end-of-pipe, RELEASE_MEM, EVENT_WRITE
	Write,    // WRITE_DATA, DMA_DATA, clocks, LOD statistics, constant RAM dumps
	Wait,     // WAIT_REG_MEM, predication, COND_EXEC, branches, lockstep reads
	Indirect, // indirect draws and dispatches (read GPU data)
	Draw,     // draws inheriting the instance count or without their own registers
	Flip,     // flips and flip waits (bounced to Thread_Gpu)
	Service,  // service commands (bounced to Thread_Gpu)
	Slice,    // a slice ending with the list unfinished (its segment is kept)
	ListEnd,  // the list's end (hand-off submit, overlay merge)
	Other,
	Count,
};
void                     SetTurnReason(TurnReason reason) noexcept;
[[nodiscard]] TurnReason CurrentTurnReason() noexcept;
class TurnReasonScope {
public:
	explicit TurnReasonScope(TurnReason reason): m_previous(CurrentTurnReason()) {
		SetTurnReason(reason);
	}
	~TurnReasonScope() { SetTurnReason(m_previous); }
	TurnReasonScope(const TurnReasonScope&)            = delete;
	TurnReasonScope& operator=(const TurnReasonScope&) = delete;

private:
	TurnReason m_previous;
};
// A turn wait of `lane` for `reason` lasted `ns` (CommandScheduler::WaitLaneTurn).
void NoteTurnWait(uint32_t lane, TurnReason reason, uint64_t ns);
// The calling lane thread translates (active) or waits (turn, lanes' lock): overlap accounting
// ("2+ lanes active"). Nested calls keep the outer state (LaneActiveScope restores it).
void               SetLaneActive(bool active);
[[nodiscard]] bool LaneActive() noexcept;
class LaneIdleScope {
public:
	LaneIdleScope(): m_was(LaneActive()) {
		if (m_was) {
			SetLaneActive(false);
		}
	}
	~LaneIdleScope() {
		if (m_was) {
			SetLaneActive(true);
		}
	}
	LaneIdleScope(const LaneIdleScope&)            = delete;
	LaneIdleScope& operator=(const LaneIdleScope&) = delete;

private:
	bool m_was;
};
// An op visible to the guest (a TurnReason op) executes on lane `lane` (lanes' lock held):
// `oldest`: its segment is the oldest open one (VisibleNotOldest otherwise); op sequences of visible
// ops must not go back (VisibleOutOfOrder).
void NoteVisibleOp(uint32_t lane, uint64_t sequence, bool oldest);
// Thread_Gpu statistics of the concurrent mode: flights out after a hand-out, its own work done
// under the lanes' lock (lists out), garbage collections deferred to a moment with no list out.
void NoteFlightsOut(uint32_t count);
void NoteOwnWorkLocked();
void NoteGcDeferred();
// Thread_Gpu: the token's hooks called directly (concurrent mode: no token passes).
// HookGive: the give hook with `target` as the hand-off target (HandoffTarget).
void HookGive(uint32_t target);
void HookTake();
// Thread_Gpu: lane `lane` runs `task(argument)` (a slice) without Thread_Gpu waiting; PollSlice
// serves one pending RunOnGpuThread request of the lane (Thread_Gpu as the lanes' lock's delegated
// holder, as in RunSlice) and returns true once the slice has ended (the lane is idle again).
void               RunSliceAsync(uint32_t lane, void (*task)(void*), void* argument);
[[nodiscard]] bool PollSlice(uint32_t lane);
// The lanes (slice end, request) and Enqueue wake Thread_Gpu waiting in WaitDispatchEvent.
void WakeDispatcher();
void WaitDispatchEvent(uint64_t timeout_ns);
void               SetListUnfinished(bool unfinished) noexcept;
[[nodiscard]] bool ListUnfinished() noexcept;
[[nodiscard]] bool GivingForRequest() noexcept;
void               AdoptSegment(uint32_t context);
void               DropSegment();
void               LaneLockAcquire();
void               LaneLockRelease();
[[nodiscard]] bool LaneLockHeld() noexcept;
// Astro Bot PC (KYTY_CP_PAR=2b, suite 2): the lanes' lock for a scope, in 2b with the lock on
// (re-entrant; delegated on Thread_Gpu serving the holder); nothing otherwise.
class LaneLockScope {
public:
	LaneLockScope(): m_on(Lanes2b() && LaneLockEnabled()) {
		if (m_on) {
			LaneLockAcquire();
		}
	}
	~LaneLockScope() {
		if (m_on) {
			LaneLockRelease();
		}
	}
	LaneLockScope(const LaneLockScope&)            = delete;
	LaneLockScope& operator=(const LaneLockScope&) = delete;

private:
	bool m_on;
};

enum class Fallback : uint32_t {
	NoSequencer, // KYTY_CP_SEQ is not 1 (or the sequencer is off): no op ring to split
	Handoff,     // constant-engine submission: parsed and executed by Thread_Gpu itself
	Verify,      // KYTY_CP_SEQ_VERIFY: the reference front follows one ring
	Count,
};

// The token's hand-off hooks (see above). `context`: the GuestGpu.
struct Hooks {
	void (*take)(void* context) = nullptr;
	void (*give)(void* context) = nullptr;
	void* context               = nullptr;
};

using Task = void (*)(void* argument);

// Starts "CP resolver #2" (GuestGpu, once, when 1a mode is active with the sequencer).
void Start(const Hooks& hooks);
// Astro Bot PC (KYTY_CP_PAR=1b3): starts CP resolver #2 as the run-ahead thread instead.
void StartRunAheadThread();
// Stops and joins it (GuestGpu::Shutdown, after Thread_Gpu returned).
void Stop();
[[nodiscard]] bool Started();

// Thread_Gpu: runs `task` on resolver #2 and waits until it returns, serving its RunOnGpuThread
// requests meanwhile.
void RunSlice(Task task, void* argument);
// Astro Bot PC (KYTY_CP_PAR=2a, N lanes): the same on the resolver of lane `lane` (1..N-1).
void RunSlice(uint32_t lane, Task task, void* argument);
// Resolver #2, inside a RunSlice task: runs `task` on Thread_Gpu (with the token) and waits.
void RunOnGpuThread(Task task, void* argument);
// The calling thread is resolver #2.
[[nodiscard]] bool OnResolver2() noexcept;

// Statistics (any thread; relaxed).
void NoteFallback(Fallback reason);
// A submission resolved to its end by context A (Thread_Gpu) or B (resolver #2).
void NoteCompleted(uint32_t context);
// Resolving time of one slice on Thread_Gpu (context 0) or resolver #2 (context 1).
void NoteResolveTime(uint32_t context, uint64_t ns);
// Resolver #2 asked Thread_Gpu to run something: service commands, or a flip op.
void NoteBounce(bool flip_op);
// KYTY_CP_PAR=1b1, Thread_Gpu after a slice of `context`: draws committed (preparation accepted)
// and fallen back (serial preparation) during it, and the totals of that context's draw-prep
// window so far (positions published, retired).
void NotePrep(uint32_t context, uint64_t committed, uint64_t fell_back, uint64_t published,
              uint64_t retired);
// Thread_Gpu: prints "CpPar 10s: ..." once 10 s have passed since the last line.
void MaybeReport();

[[nodiscard]] uint64_t NowNs();

} // namespace Libs::Graphics::CpPar

#endif // EMULATOR_SRC_GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_CPPARALLEL_H_
