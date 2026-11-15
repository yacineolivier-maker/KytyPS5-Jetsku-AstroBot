#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_FASTCHECK_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_FASTCHECK_H_

#include <cstdint>

// Astro Bot PC (KYTY_CP_PAR_FASTCHECK=1, default 0; docs-p3-drawindex.md §3 (a) and (b)): "would this
// draw have taken the fast path?" A census of the committed draws (DrawIndex, DrawAuto) under the
// lanes' lock as it is, which changes nothing: the existing commit code marks, at the decisions it
// takes anyway, the events that the fast path's condition (a) 1-9 forbids (a prepared slot not
// ready, a refused preparation, targets acquired instead of continued, a texture memo miss, a real
// image transition, a written or synchronised buffer, a submit...). The marks are thread-local bits
// of the op's thread; nothing in the emulator's state is read for the census beyond what the commit
// reads itself, and nothing is written (no LRU, memo or flag). At the op's end, the draw is counted
// per lane: fast (no mark, something recorded), empty (nothing recorded: a draw without work), or
// failed, by its first failing condition (the lowest-numbered mark) and, when it has only one mark,
// by that "sole" cause (the draws one fix would make fast). One console line every 10 s:
// "CpPar 10s: 2b fast-draw candidates". Off: one load of a constant per hook.
namespace Libs::Graphics::FastCheck {

// In the order of condition (a): a cause's condition is kConditionOf[cause].
enum class Cause : uint8_t {
	// 1: a published draw, without a pending inherited count or a register snapshot
	NotPublished,
	InheritPending,
	Snapshot,
	// 2: the slot was prepared (done) when the commit took it
	HeadUnclaimed,
	HeadWaited,
	// 3: the preparation accepted, the plan's hardware checks quiet, the plan's pipeline used
	PrepMissing,
	PrepRefused,
	PlanNotQuiet,
	PipelineNotPlanned,
	// 4: no target operation, no 8-bit indices, no restart scan, no MRT split
	TargetOp,
	Index8,
	RestartScan,
	MrtSplit,
	// 5: targets of a continuation (draw run) or a reused acquisition (a continuation that fell
	// back late, then the targets acquired: both are marked, the late fallback is reported first)
	RunLateFallback,
	TargetsAcquired,
	// 6: texture memo hits, no DCC application, no real image transition
	TextureMiss,
	TextureViewMiss,
	TextureDcc,
	Transition,
	// (P3 step 2, strict) the image state is adopted without a barrier, or a younger lane's op
	// swaps an image's state with its lane overlay (in place, for the rest of the op: no other lane
	// may read that image meanwhile)
	ImageAdopt,
	ImageOverlay,
	// 7: buffers read-only and clean
	BufferWritten,
	BufferCreate,
	BufferSync,
	BufferStream,
	BufferRejoin,
	// 8: no sampler memo miss, no occlusion flush, no recorder drain
	SamplerMiss,
	OcclusionFlush,
	RecorderDrain,
	// 9: nothing global: no pending operation run, no submit, no services, no slice end
	PendingOps,
	Submit,
	ServicesRun,
	SliceYield,
	Count
};

namespace Detail {
bool                                     ReadEnabled() noexcept;
inline const bool                        g_enabled = ReadEnabled();
inline constinit thread_local bool       t_active  = false;
inline constinit thread_local uint64_t   t_causes  = 0;
inline constinit thread_local bool       t_recorded = false;
void                                     End(uint32_t lane, bool auto_draw) noexcept;
} // namespace Detail

[[nodiscard]] inline bool Enabled() noexcept {
	return Detail::g_enabled;
}

// The op's thread, around one DrawIndex / DrawAuto op (the lanes' lock held as it is).
inline void Begin() noexcept {
	if (Detail::g_enabled) [[unlikely]] {
		Detail::t_active   = true;
		Detail::t_causes   = 0;
		Detail::t_recorded = false;
	}
}
inline void End(uint32_t lane, bool auto_draw) noexcept {
	if (Detail::g_enabled && Detail::t_active) [[unlikely]] {
		Detail::End(lane, auto_draw);
	}
}

// A mark of the census (any thread; only the op's thread between Begin and End counts).
inline void Fail(Cause cause) noexcept {
	if (Detail::g_enabled && Detail::t_active) [[unlikely]] {
		Detail::t_causes |= uint64_t {1} << static_cast<uint32_t>(cause);
	}
}

// The draw reached its pipeline selection: something is recorded (not an empty draw).
inline void Recorded() noexcept {
	if (Detail::g_enabled && Detail::t_active) [[unlikely]] {
		Detail::t_recorded = true;
	}
}

} // namespace Libs::Graphics::FastCheck

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_FASTCHECK_H_
