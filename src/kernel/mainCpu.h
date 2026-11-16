#ifndef EMULATOR_SRC_KERNEL_MAINCPU_H_
#define EMULATOR_SRC_KERNEL_MAINCPU_H_

#include <cstdint>

// Astro Bot PC: where the guest main thread's time goes, and its placement (Windows; default off).
//
// KYTY_MAIN_CPU_SAMPLER=1: a helper thread samples, about every millisecond, the instruction
// pointer of the guest main thread and of the KYTY_MAIN_CPU_SAMPLER_THREADS (default 3) other
// guest threads that used the most CPU time in the previous 10 s (the sampled thread is suspended
// only while its context is read). Each sample, weighted by the time since the previous one, goes
// to a category:
//   attente active tbb   eboot 0xaee4a0-0xaee930 and 0xaec980-0xaeca40 (the game's tbb wait:
//                        steal attempts, pause loop, scePthreadYield call)
//   attente d'etiquette  eboot 0x114b2e0-0x114b330 (the GPU-label poll)
//   atomiques            eboot 0x250-0x400 (the game's atomic helpers)
//   code du jeu          the rest of the eboot, also by function (start found in guest memory:
//                        the first byte after int3 padding), top 10 per thread
//   module invite        other guest modules (0x9xxxxxxxx outside the eboot: libc.prx, ...)
//   noyau : attente      inside NtWaitForAlertByThreadId / NtWaitForSingleObject(s) /
//                        NtWaitForMultipleObjects / NtRemoveIoCompletion (blocked)
//   noyau : yield/sommeil inside NtDelayExecution / NtYieldExecution
//   Windows (autre)      other ntdll, kernelbase, kernel32, win32u code
//   emulateur            kyty_emulator.exe (top 5 offsets per thread, for llvm-symbolizer)
//   pilote/autre module  any other module (GPU driver, ...)
// Every 10 s: "Kyty echantillons CPU 10 s : ..." per thread, in ms per frame (frames =
// sceAgcSuspendPoint calls) and percent of the window.
// Also every 10 s: "Kyty cadence 10 s : ..." (frame-time histogram), and for each frame of the
// main thread longer than KYTY_MAIN_CPU_SPIKE_MS (default 45): "Kyty pic : image N ms ; ..." with
// that frame's categories and top functions (game, guest module, Windows export), 40 per 10 s at most.
//
// KYTY_MAIN_THREAD_CORE=<logical processor>: the guest main thread runs on that logical processor
// only, and every processor of the same physical core (its SMT siblings, read with
// GetSystemCpuSetInformation) leaves the process default CPU sets, so every other thread (guest and
// emulator, unless it selects CPU sets itself) runs elsewhere. Ignored, with a message, when
// KYTY_CPU_RESERVE or KYTY_CPU_SETS manage the CPU sets.
// KYTY_MAIN_THREAD_PRIORITY=<-2..2>: SetThreadPriority of the guest main thread (0 normal, 1 above
// normal, 2 highest; the guest's own priority changes of that thread come after it).
namespace Libs::LibKernel::MainCpu {

extern bool g_sampler_enabled; // KYTY_MAIN_CPU_SAMPLER=1

// RuntimeLinker::Execute, on the guest main thread, before guest code.
void OnMainThreadStart();
// A guest thread starts (pthread run begin), with its name.
void OnGuestThreadStart(const char* name);
// One guest frame (sceAgcSuspendPoint).
void NoteFrameSlow() noexcept;
inline void NoteFrame() noexcept {
	if (g_sampler_enabled) {
		NoteFrameSlow();
	}
}
// A guest sleep (sceKernelUsleep / nanosleep): its guest return address, the time asked for and
// the time it took. Every 10 s: "Kyty sommeils invites 10 s : ..." by requested size, and the
// callers that slept the longest.
void NoteGuestSleepSlow(uint64_t ra, uint64_t requested_ns, uint64_t actual_ns) noexcept;
inline void NoteGuestSleep(uint64_t ra, uint64_t requested_ns, uint64_t actual_ns) noexcept {
	if (g_sampler_enabled) {
		NoteGuestSleepSlow(ra, requested_ns, actual_ns);
	}
}

} // namespace Libs::LibKernel::MainCpu

#endif
