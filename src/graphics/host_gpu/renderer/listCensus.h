#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_LISTCENSUS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_LISTCENSUS_H_

#include <cstdint>

// Astro Bot PC (KYTY_CP_PAR_CENSUS2=1, diagnostic, default off; step P0 of
// docs\CONCEPTION-MULTICOEUR.md): if each guest command list were translated by its own lane, a
// later list could upload (copy from guest memory) a buffer range that an earlier list of the same
// frame writes on the GPU, and so read stale bytes. This census tags every storage-buffer binding
// with its list (submission, in resolve order) and, one frame in 4, intersects the ranges a list
// writes with the ranges any LATER list of the frame binds. One "ListCensus 10s: ..." line every
// 10 s. Nothing it measures feeds back into emulation.
namespace Libs::Graphics::ListCensus {

[[nodiscard]] bool Enabled();
// GPU thread: the resolver starts a guest submission.
void BeginList(uint64_t submission);
// GPU thread: one storage-buffer binding of the current list.
void NoteBuffer(uint64_t address, uint64_t size, bool written);
// A guest flip: the frame's bindings are analysed (one frame in 4) and cleared.
void EndFrame();

} // namespace Libs::Graphics::ListCensus

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_LISTCENSUS_H_ */
