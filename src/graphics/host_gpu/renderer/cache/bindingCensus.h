#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_BINDINGCENSUS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_BINDINGCENSUS_H_

#include "common/onceValue.h"

#include <atomic>
#include <cstdint>

// Astro Bot PC (KYTY_BINDING_CENSUS=1, diagnostic, default off; step S1 of the "unified memory"
// track, docs\PLAN-PARALLELE.md): where the V# binding time of RenderExecutor::RebindBuffers goes,
// by guest mapping. Each binding is bucketed by the guest virtual range holding its address
// (VirtualRanges: name, direct/flexible/..., base) x what the lookup did (memo hit, memo stream
// hit, lean miss, stream copy, cache buffer with upload = CPU-dirty before, cache buffer clean,
// texel, written, other) x its time (TSC around NativeStorageBuffer). Per 16 KiB page bound: the
// frames it was bound, the frames it was CPU-dirty when bound, whether it was ever GPU-dirty or
// written by a binding, whether an image may overlap it (TextureCache 1 MiB pages). With the
// native capture, the time per engine render thread slot of the draw op being executed.
// Every 10 s, "BindingCensus 10s: ..." lines: totals, share of binding time of the top 1/2/4
// ranges, then the top 16 ranges by binding time. GPU thread only (other threads are counted and
// ignored); nothing it measures feeds back into emulation. Off: one predictable branch per hook.
namespace Libs::Graphics {
class RenderContext;
} // namespace Libs::Graphics

namespace Libs::Graphics::BindingCensus {

namespace Detail {
[[nodiscard]] bool ReadEnabled();
extern thread_local uint8_t t_outcome; // BufferLookupStats::Outcome of the binding being obtained
extern thread_local uint8_t t_slot;    // 0: not a native draw op; else engine thread slot + 1
} // namespace Detail

[[nodiscard]] inline bool Enabled() {
	static std::atomic<int64_t> enabled_once {Common::OnceUnset};
	return Common::Once(enabled_once, [] { return Detail::ReadEnabled(); }) != 0;
}

// BufferLookupStats::SetOutcome (bufferLookupStats.h), whatever KYTY_CP_DIAGNOSTICS is.
inline void SetOutcome(uint8_t outcome) {
	Detail::t_outcome = outcome;
}
// graphicsRun.cpp: the draw op being executed (op flags carry the native slot).
inline void SetNativeSlot(uint8_t slot_plus_one) {
	Detail::t_slot = slot_plus_one;
}

// What Before saw, for After.
struct Probe {
	uint64_t vaddr     = 0;
	uint64_t size      = 0;
	uint64_t t0        = 0;
	bool     active    = false; // GPU thread, non-null range
	bool     cpu_dirty = false; // some bound page was CPU-dirty before the lookup
};

// Before the binding's lookup: the dirty state of its pages (the lookup may upload and clear it).
void Before(RenderContext& context, uint64_t vaddr, uint64_t size, Probe& probe);
// After it: classification and time since Before returned.
void After(RenderContext& context, const Probe& probe, bool written, bool formatted);

} // namespace Libs::Graphics::BindingCensus

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_BINDINGCENSUS_H_ */
