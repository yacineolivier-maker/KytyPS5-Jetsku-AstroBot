#ifndef EMULATOR_SRC_KERNEL_WAITGRAPH_H_
#define EMULATOR_SRC_KERNEL_WAITGRAPH_H_

#include <cstdint>

// Astro Bot PC (KYTY_WAIT_GRAPH=1, diagnostic, default off): who wakes whom among the guest
// threads, to find what sets the frame rate.
//
// Every blocking wait of a guest thread on a guest primitive (kernel semaphores, POSIX semaphores,
// pthread condition variables and mutexes, event flags, event queues, sceAgcSuspendPoint, sleeps)
// is timed. The signalling side (signal, post, unlock, set, trigger, the CP's progress) writes
// "last signal" facts for its object into a hash table: who (thread), when, and the guest return
// address. When the wait ends, its object's last signal, if not older than the wait, names the
// waker; otherwise the waker is "inconnu" (timeout, cancel, a signal path without a hook). Signals
// made by emulator threads carry a readable name: "GPU : interruption EOP", "VideoOut : flip/vblank",
// "CP : fin des soumissions (AgcSuspendPoint)", ... Yield loops (scePthreadYield and sleeps of at
// most 1 us called less than 100 us apart) are timed per burst, keyed by the guest address that
// loops (eboot 0xaee726: the game's tbb wait; 0x114b2f0: its GPU-label poll).
//
// Every 10 s on stdout ("Kyty graphe d'attente 10 s : ..."): the guest main thread, then the three
// other guest threads that waited most, each with its wait time per frame (frames =
// sceAgcSuspendPoint calls) split by waker (thread or GPU source), primitive and guest address of
// the wait, and the longest chains "A attend B qui attend C" (each hop: the waker A waited for
// most, if it is a guest thread).
//
// Off: every hook is one load of a global flag.
namespace Libs::LibKernel::WaitGraph {

extern bool g_enabled; // KYTY_WAIT_GRAPH=1, read once at static initialization

[[nodiscard]] inline bool Enabled() noexcept {
	return g_enabled;
}

enum class Prim : uint8_t {
	KernelSema,
	PosixSem,
	Cond,
	Mutex,
	EventFlag,
	Equeue,
	AgcDone,
	Sleep,
	Yield,
	Count
};

// The calling thread signals `object` (wakes, posts, unlocks, sets, triggers). `label`: the waker's
// name when the caller is not a guest thread (a string literal; ignored on guest threads).
// Off: inline, one load of the flag and no call (the out-of-line parts run only when on).
void SignalSlow(const void* object, uint64_t return_address, const char* label) noexcept;
inline void Signal(const void* object, uint64_t return_address, const char* label = nullptr) noexcept {
	if (Enabled()) {
		SignalSlow(object, return_address, label);
	}
}

// A blocking wait of the calling thread on `object`; waits shorter than `min_ns` are not counted
// (uncontended mutexes).
class WaitScope {
public:
	WaitScope(const void* object, Prim prim, uint64_t return_address, uint64_t min_ns = 0) noexcept {
		if (Enabled()) {
			Begin(object, prim, return_address, min_ns);
		}
	}
	~WaitScope() {
		if (m_start != 0) {
			End();
		}
	}
	WaitScope(const WaitScope&)            = delete;
	WaitScope& operator=(const WaitScope&) = delete;

private:
	void Begin(const void* object, Prim prim, uint64_t return_address, uint64_t min_ns) noexcept;
	void End() noexcept;

	const void* m_object  = nullptr;
	uint64_t    m_ra      = 0;
	uint64_t    m_start   = 0;
	uint64_t    m_min_ns  = 0;
	Prim        m_prim    = Prim::Count;
};

// A yield (or a sleep of at most 1 us) of the calling thread, called from `return_address`.
void NoteYieldSlow(uint64_t return_address) noexcept;
inline void NoteYield(uint64_t return_address) noexcept {
	if (Enabled()) {
		NoteYieldSlow(return_address);
	}
}

// The calling thread is the guest main thread (RuntimeLinker::Execute).
void MarkMainThread() noexcept;

// One guest frame (sceAgcSuspendPoint).
void NoteFrameSlow() noexcept;
inline void NoteFrame() noexcept {
	if (Enabled()) {
		NoteFrameSlow();
	}
}

} // namespace Libs::LibKernel::WaitGraph

#endif
