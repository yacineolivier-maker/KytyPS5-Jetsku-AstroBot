#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_FLUIDITYLOG_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_FLUIDITYLOG_H_

// KYTY_FLUIDITE=1 (Astro Bot PC): journal of the places where the frame rate drops, kept while the
// player plays (sessions of one to two hours), for later reconstruction of each situation.
//
// Per guest flip (main bus), on the video-out thread: one clock read, one QueryThreadCycleTime of the
// guest main thread, a dozen relaxed counter loads (Common::DebugCounters, collected even with the
// overlay hidden) and the GPU busy total (GpuTiming, when KYTY_GPU_TIMING=1). Kept in a ring: the
// "black box".
//
// Episode: a frame >= KYTY_FLUIDITE_PIC_MS (50) or the last second under KYTY_FLUIDITE_SEUIL fps (28);
// it ends after 1.5 s back above. For each episode:
// - journal.csv: time, duration, frames, worst frame, min fps, probable cause (shaders, disque, gpu,
//   cpu-jeu, attente-gpu, mixte) with the means behind it, thumbnails, GPU report, recent app0 file
//   opens (level), camera (KYTY_FLUIDITE_CAMERA, see fluidityLog.cpp);
// - boite-noire.csv: one line per frame from 1 s before the start to the end (at most 300);
// - a thumbnail (JPEG 640 wide, game image read back by the presenter, never the desktop): at most
//   one every KYTY_FLUIDITE_ECART_S s (5), one every 10 s in a long episode, KYTY_FLUIDITE_IMAGES
//   per session (500);
// - after 2 s under the threshold, a GPU frame report (KYTY_GPU_OP_PROFILE=demande, gpuOpProfiler.h):
//   at most KYTY_FLUIDITE_RAPPORTS per session (6), 3 min apart.
// F9 or L3+R3: manual mark (thumbnail, and a report if allowed). minutes.csv: one line per minute.
//
// Files: KYTY_FLUIDITE_DIR (default _Fluidite next to the emulator)\<date-time>\. Place budget:
// KYTY_FLUIDITE_QUOTA_MO (1024) for the whole _Fluidite folder. Over it, the oldest captures
// (img-*, rapport-gpu\gpuops-<n>.csv) of all sessions are deleted, oldest first (porteur's rule,
// 17/11/2026); text journals are never deleted, and if they alone exceed the budget no capture is
// taken any more. All disk work runs on one worker thread.

#include <cstdint>
#include <string>
#include <vector>

namespace Libs::Graphics {

[[nodiscard]] bool FluidityEnabled() noexcept;

// Camera position (x y z) and direction (x y z), from KYTY_FLUIDITE_CAMERA (works even when the
// journal is off); false when unset or not read yet. Any thread (internal pad driver "aller").
[[nodiscard]] bool FluidityCamera(float* position, float* direction) noexcept;

// Called for each main-bus guest flip (video-out thread).
void FluidityRecordFlip() noexcept;

// Manual mark (window thread, input thread).
void FluidityMark() noexcept;

// Pad stick buttons, for the L3+R3 mark.
void FluidityNotePadButton(int id, bool left_stick, bool down) noexcept;

// v3 (19/11): a pad button change (any pad, any thread): manette.csv, for replaying a session's path.
void FluidityNoteButton(uint32_t button, bool down) noexcept;

// v3: a line the game printed (any thread); "ActivateCheckPoint..." lines go to points.csv.
void FluidityNoteGameText(const char* text) noexcept;

// A guest file open (any thread); only app0 data paths are kept (v3: level changes to niveaux.csv).
void FluidityNoteOpen(const char* path) noexcept;

// Guest main thread handle (THREAD_QUERY_LIMITED_INFORMATION), from its start.
void FluiditySetMainThread(void* handle) noexcept;

// Load address of the game executable, for "eboot+<offset>" in KYTY_FLUIDITE_CAMERA.
void FluiditySetEbootBase(uint64_t base) noexcept;

// Present thread: true once when a thumbnail is wanted; returns its number.
[[nodiscard]] bool FluidityTakeCaptureRequest(uint32_t* number) noexcept;

// The session folder, and the path of thumbnail <number> in it.
[[nodiscard]] std::string FluiditySessionDir();
[[nodiscard]] std::string FluidityImagePath(uint32_t number);

// Hands a finished capture file to the worker thread (written there, counted in the budget).
void FluidityWriteFile(std::string path, std::vector<uint8_t> bytes);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_FLUIDITYLOG_H_
