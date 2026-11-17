#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_FLUIDITYLOG_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_FLUIDITYLOG_H_

// KYTY_FLUIDITE=1 (Astro Bot PC): journal of the places where the frame rate drops, kept while the
// player plays, for later reconstruction of each situation.
//
// - Each main-bus guest flip is timed (one clock read, a few additions under no lock).
// - An episode starts on a long frame (KYTY_FLUIDITE_PIC_MS, default 50) or when the frames of the
//   last second fall under KYTY_FLUIDITE_SEUIL fps (default 28); it ends after 1.5 s back above.
// - At the start of an episode (at most one every KYTY_FLUIDITE_ECART_S seconds, default 5, and one
//   every 10 s during a long episode), the presenter reads back the next presented frame and saves
//   a thumbnail (game image only: never the desktop or other windows).
// - F9 on the keyboard or L3+R3 on the pad mark the current place by hand, with a thumbnail.
// - Recent guest file opens (level data) and, when KYTY_FLUIDITE_CAMERA describes where the game
//   keeps it, the camera, are written with each line.
// - Files go to KYTY_FLUIDITE_DIR (default: _Fluidite next to the emulator), one folder per session:
//   journal.csv (episodes and marks), minutes.csv (one line per minute), img-<n>.png.
//   Disk writes are done on a worker thread, never on the flip or present threads.

#include <cstdint>
#include <string>
#include <vector>

namespace Libs::Graphics {

[[nodiscard]] bool FluidityEnabled() noexcept;

// Called for each main-bus guest flip (video-out thread).
void FluidityRecordFlip() noexcept;

// Manual mark (window thread, input thread).
void FluidityMark() noexcept;

// Pad stick buttons, for the L3+R3 mark.
void FluidityNotePadButton(int id, bool left_stick, bool down) noexcept;

// A guest file open (any thread); only app0 data paths are kept.
void FluidityNoteOpen(const char* path) noexcept;

// Present thread: true once when a thumbnail is wanted; returns its number.
[[nodiscard]] bool FluidityTakeCaptureRequest(uint32_t* number) noexcept;

// Path of the thumbnail number (in the session folder).
[[nodiscard]] std::string FluidityImagePath(uint32_t number);

// Load address of the game executable, for "eboot+<offset>" in KYTY_FLUIDITE_CAMERA.
void FluiditySetEbootBase(uint64_t base) noexcept;

// Hands a finished file to the worker thread (written there).
void FluidityWriteFile(std::string path, std::vector<uint8_t> bytes);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_FLUIDITYLOG_H_
