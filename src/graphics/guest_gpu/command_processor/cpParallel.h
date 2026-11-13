#ifndef EMULATOR_SRC_GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_CPPARALLEL_H_
#define EMULATOR_SRC_GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_CPPARALLEL_H_

// Astro Bot PC (KYTY_CP_PAR), 13/11/2026: parallel translation of the command lists, stages 1a,
// 1b-i, 1b-ii (docs\PLAN-PARALLELE.md). Plumbing only, no concurrency.
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

#include <cstdint>

namespace Libs::Graphics::CpPar {

enum class Mode : uint8_t { Off, Serial1a, Serial1b1, Serial1b2 };

// KYTY_CP_PAR, read once.
[[nodiscard]] Mode GetMode();
[[nodiscard]] inline bool Enabled() {
	return GetMode() != Mode::Off;
}
// KYTY_CP_PAR=1b1 (and 1b2): context B has its own snapshot ring and draw-prep engine (see above).
[[nodiscard]] inline bool SeparateWindows() {
	return GetMode() == Mode::Serial1b1 || GetMode() == Mode::Serial1b2;
}
// KYTY_CP_PAR=1b2 requested: context B records into its own target (see above). Whether it is
// active (recorder requirements met) is StreamsActive().
[[nodiscard]] inline bool SeparateStreams() {
	return GetMode() == Mode::Serial1b2;
}

// Astro Bot PC (KYTY_CP_PAR=1b2): set once at start-up, before resolver #2 runs, when the two
// recording targets are active; read by Image (state bookkeeping) without a call.
inline bool g_streams_active = false;
[[nodiscard]] inline bool StreamsActive() noexcept {
	return g_streams_active;
}
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
// Stops and joins it (GuestGpu::Shutdown, after Thread_Gpu returned).
void Stop();
[[nodiscard]] bool Started();

// Thread_Gpu: runs `task` on resolver #2 and waits until it returns, serving its RunOnGpuThread
// requests meanwhile.
void RunSlice(Task task, void* argument);
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
