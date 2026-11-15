#ifndef EMULATOR_SRC_GRAPHICS_GUEST_GPU_FRAMEPATH_H_
#define EMULATOR_SRC_GRAPHICS_GUEST_GPU_FRAMEPATH_H_

#include <chrono>
#include <cstdint>

// Astro Bot PC (KYTY_FRAME_PATH=1, diagnostic, default off): the latency chain of a frame, from
// the guest's graphics submission to the GPU's end, and the time the guest's main thread waits.
// Every 10 s one line "Kyty chemin image 10 s : ..." (stdout) gives the mean and the 95th
// percentile of each stage, for the queue-0 graphics submissions completed in that window:
//   file          GuestGpu::Enqueue -> the CP starts the submission (Thread_Gpu or a lane)
//   traduction    CP start -> CP complete (slices, suspended waits and yields included). With
//                 KYTY_LABEL_MODE=record the guest sees an end-of-pipe label as soon as the CP
//                 executes it, so "jeu->fin CP" (file + traduction) bounds "the guest submitted
//                 -> the guest sees the submission's last label".
//   enregistrement  CP complete -> its last tick handed to the Vulkan submission queue (CP
//                 recorder replay; the slice end submits, BufferFlush)
//   soumission    handed over -> vkQueueSubmit returned (submission broker, coalescing, driver)
//   GPU           vkQueueSubmit returned -> the timeline semaphore reached that tick (observed by
//                 a monitor thread that waits on it, KYTY_FRAME_PATH only)
// plus the end-of-pipe labels written at record time (submission enqueue -> label write), the
// deferred labels (tick completion seen by the completion runner -> label written on Thread_Gpu),
// sceAgcSuspendPoint waits (any thread), and the guest main thread's waits (share of the window:
// yields, with the pause loops between close yields estimated as spinning, kernel semaphores,
// condition variables, event queues, sleeps, access faults).
// Off: every hook is one load of a global flag.
namespace Libs::Graphics {
class MasterSemaphore;
} // namespace Libs::Graphics

namespace Libs::Graphics::FramePath {

extern bool g_enabled; // KYTY_FRAME_PATH=1, read once at static initialization

[[nodiscard]] inline bool Enabled() noexcept {
	return g_enabled;
}

[[nodiscard]] inline uint64_t NowNs() noexcept {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

// Thread_Gpu (or a lane): a queue-0 graphics submission completed on the CP. tail_tick: the last
// tick submitted after it (its slice end flushes). Starts the monitor thread on first use.
void NoteSubmission(MasterSemaphore& master, uint64_t enqueue_ns, uint64_t start_ns,
                    uint64_t complete_ns, uint64_t tail_tick);
// CP recorder: the buffer of `tick` was handed to the submission queue (or submitted directly).
void NoteRecorded(uint64_t semaphore, uint64_t tick);
// Submission broker / recorder: vkQueueSubmit returned for `tick` of timeline `semaphore`.
void NoteDispatched(uint64_t semaphore, uint64_t tick);
// ~MasterSemaphore: stops the monitor thread if it waits on this timeline.
void Detach(const MasterSemaphore* master);

// The CP thread's current slice: the guest enqueue time of its submission (0: none).
void BeginSlice(uint64_t enqueue_ns) noexcept;
// An end-of-pipe label written at record time by the CP thread.
void NoteLabelWrite() noexcept;
// A deferred label: the completion runner ran its operation at completed_ns; written at visible_ns.
void NoteDeferredLabel(uint64_t completed_ns, uint64_t visible_ns) noexcept;
// sceAgcSuspendPoint (any guest thread): one call, wait_ns blocked.
void NoteAgcDone(uint64_t wait_ns) noexcept;

// Guest main thread (RuntimeLinker::Execute).
void MarkMainThread() noexcept;
[[nodiscard]] bool OnMainThread() noexcept;

enum class Wait : uint32_t { Semaphore, Cond, Equeue, Sleep, Fault, Count };
void NoteMainWait(Wait kind, uint64_t ns) noexcept;
// PthreadYield on the main thread: [begin, end] of the yield.
void NoteMainYield(uint64_t begin_ns, uint64_t end_ns) noexcept;

// Times a guest wait of the main thread (nothing when off or on another thread).
class MainWaitScope {
public:
	explicit MainWaitScope(Wait kind) noexcept
	    : m_kind(kind), m_start(Enabled() && OnMainThread() ? NowNs() : 0) {}
	~MainWaitScope() {
		if (m_start != 0) {
			NoteMainWait(m_kind, NowNs() - m_start);
		}
	}
	MainWaitScope(const MainWaitScope&)            = delete;
	MainWaitScope& operator=(const MainWaitScope&) = delete;

private:
	Wait     m_kind;
	uint64_t m_start;
};

} // namespace Libs::Graphics::FramePath

#endif
