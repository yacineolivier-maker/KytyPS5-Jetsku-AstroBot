#ifndef KYTY_LOADER_GAME_CAPTURE_H_
#define KYTY_LOADER_GAME_CAPTURE_H_

// KYTY_GAME_HOOKS=capture (Astro Bot PC, "native rendering backend" project, stage "capture";
// docs\CARTE-BACKEND-GPU.md, section "Capture"): observation only, rendering is unchanged.
//
// The hook set (gameHooksCapture.cpp) builds a "native draw record" from the engine's backend
// context object at every draw. This header is the small surface the rest of the emulator
// calls, so that the game's view of a draw can be compared with what the emulator's command
// processor sees for the same draw:
//
//   - libAgc HLE draw functions (libs/agc.cpp) call NoteDrawPacket() with the address of the
//     PM4 draw packet they just wrote, on the calling game thread. A sampled hooked draw (1 in
//     1000 per thread) is published under that packet address.
//   - The command processor (graphics/guest_gpu/graphicsRun.cpp) calls CheckCpDraw() for every
//     draw packet it parses (not for reference/prefetch fronts), with its own state; a sample
//     with the same packet address is compared field by field and consumed.
//
// Both calls are a single relaxed load of a global flag when capture is off.

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace Loader::GameCapture {

extern std::atomic<bool> g_active; // set once the capture hooks are installed

[[nodiscard]] inline bool Active() {
	return g_active.load(std::memory_order_relaxed);
}

// PM4 opcode of the packet (IT_DRAW_INDEX_OFFSET_2, IT_DRAW_INDEX_AUTO, ...) and its size in
// dwords (header included; 0 for a packet that must never be replaced).
//
// KYTY_NATIVE (stage N0, docs\PLAN-PARALLELE.md): when the draw flush of a selected engine
// thread chose the draw as "native", a direct draw packet (DRAW_INDEX_2, DRAW_INDEX_OFFSET_2,
// DRAW_INDEX_AUTO) is rewritten here, in place and before libAgc returns it, as a NOP of the same
// size: header KYTY_PM4(size, IT_NOP, R_NATIVE_DRAW), word 1 = kNativeMagic << 16 | version << 8
// | thread slot, word 2 = per-thread serial, other words 0. State packets (NUM_INSTANCES, ...)
// are left as written.
void NoteDrawPacketSlow(uint32_t* packet, uint32_t opcode, uint32_t size_dw);
inline void NoteDrawPacket(uint32_t* packet, uint32_t opcode, uint32_t size_dw = 0) {
	if (Active()) {
		NoteDrawPacketSlow(packet, opcode, size_dw);
	}
}

constexpr uint32_t kNativeMagic   = 0x4e41u; // "NA"
constexpr uint32_t kNativeVersion = 1u;

// Command processor: an R_NATIVE_DRAW marker was parsed (not by a reference / speculative front).
// `body` points after the header, `body_dw` >= 2.
void NoteNativeMarker(const uint32_t* body, uint32_t body_dw);

// KYTY_NATIVE_MODE=splice (stage N1): before writing a marker, the game thread publishes the
// original draw packet (header included) in its slot's native ring (lock-free, one producer: the
// game thread; readers: the command processor fronts), indexed by the marker's serial. Ring full
// (the CP has not consumed far enough): no marker, the normal packet stays (counted). The CP's
// CpOpNativeDraw looks the record up and runs the original packet's handler at the marker's
// position, with the CP state of that position, so the image is the emulator's own.
enum class NativeMode : uint8_t { Off, Drop, Splice, N2, N3, N3b, S0, S1 };
extern std::atomic<uint8_t> g_native_mode; // NativeMode, set once the hooks are installed
[[nodiscard]] inline NativeMode GetNativeMode() {
	return static_cast<NativeMode>(g_native_mode.load(std::memory_order_relaxed));
}
// The CP hands the original packet to its handler at the marker: N1 (splice), N2 and N3, whose
// emission is still the emulator's own draw path (N3: with the segment resolution windows below).
// S0 (KYTY_NATIVE_MODE=s0, docs\CONCEPTION-RENDU-NATIF.md stage S0): N2 plus the learning and
// diagnostics of the standalone native renderer (below); emission is still N1's.
[[nodiscard]] inline bool NativeSplices() {
	const auto mode = GetNativeMode();
	return mode == NativeMode::Splice || mode == NativeMode::N2 || mode == NativeMode::N3 ||
	       mode == NativeMode::N3b || mode == NativeMode::S0 || mode == NativeMode::S1;
}
// N2, N3, S0 and S1: game-thread records, op tags, record verification, resolver timing.
[[nodiscard]] inline bool NativeRecords() {
	const auto mode = GetNativeMode();
	return mode == NativeMode::N2 || mode == NativeMode::N3 || mode == NativeMode::N3b ||
	       mode == NativeMode::S0 || mode == NativeMode::S1;
}
// S0 learning (and serial op tags, segment lengths): s0, and s1 which adds native emission.
[[nodiscard]] inline bool NativeS0() {
	const auto mode = GetNativeMode();
	return mode == NativeMode::S0 || mode == NativeMode::S1;
}
// S1 (KYTY_NATIVE_MODE=s1, stage S1, renderer/nativeLearn.h): S0 plus native emission of Shadow
// continuation draws.
[[nodiscard]] inline bool NativeS1() {
	return GetNativeMode() == NativeMode::S1;
}
// N3b: lean continuations (renderer/renderDraw.cpp, RenderExecutor::NativeLeanPrecheck).
[[nodiscard]] inline bool NativeLean() {
	return GetNativeMode() == NativeMode::N3b;
}

// Marker word 1, low byte: bits 0..3 the engine thread slot; bit 7 (N2 only) the draw starts a
// native segment (NativeDraw::seg_start).
constexpr uint32_t kNativeSlotMask     = 0x0fu;
constexpr uint32_t kNativeSegStartFlag = 0x80u;

constexpr uint32_t kNativeMaxPacketDw = 6; // DRAW_INDEX_2

struct NativeDrawPacket {
	uint32_t words[kNativeMaxPacketDw] {}; // the original packet, header first
	uint32_t size_dw = 0;
};

// Splice lookup for a marker (`marker_size_dw`: the marker's size, header included). `consume`:
// the parse is the real one (not a reference / speculative front): the record is released to the
// producer. False on a miss (counted by reason); KYTY_NATIVE_VERIFY=1 exits on a miss.
[[nodiscard]] bool LookupNativeDraw(const uint32_t* body, uint32_t marker_size_dw, bool consume,
                                    NativeDrawPacket& out);
// Splice: the record's packet was handed to its draw handler (real parse only).
void NoteNativeSpliced(const uint32_t* body);

// ---- KYTY_NATIVE_MODE=n2 (stage N2, docs\PLAN-PARALLELE.md) --------------------------------------
//
// N2 = N1 (the CP splices the original packet into the emulator's draw path at the marker, so
// the image is the emulator's own) plus:
//  - on the game thread, right after the draw flush of a native draw and without touching any
//    emulator cache, the draw is decoded into a compact NativeDraw: program identity, the raw
//    fixed-function registers from the game's own register shadow (depth/stencil, raster, clip,
//    blend, depth bias, viewport/scissor), user data, draw arguments, the render-target set and
//    a memo key. It is published under the marker serial in a second ring of the slot (same
//    seqlock discipline as the packet ring);
//  - segments: consecutive native draws of a thread on one context and one render-target set,
//    with no other hooked draw, dispatch flush, tool-library draw or list begin of that thread in
//    between; the first draw of a segment has seg_start (also in the marker byte, above);
//  - KYTY_NATIVE_VERIFY=1: the CP decodes each record with its own register decoders and compares
//    it with its registers at the marker (real parse), field by field;
//  - the resolver measures, per thread, the Thread_Gpu TSC time of every native draw op, of every
//    segment (first op start to last op end) and how the emulator's draw run treated the draw.
// NativeDraw::known: which register groups the game re-emitted on this context since its list
// began; a register it did not re-emit keeps the queue's previous value: the record cannot vouch
// for it and the comparison skips it (counted as "unknown").
enum NativeKnown : uint32_t {
	kNkPgmEs      = 1u << 0,
	kNkPgmGs      = 1u << 1,
	kNkPgmPs      = 1u << 2,
	kNkRenderCtl  = 1u << 3,
	kNkDepthCtl   = 1u << 4,
	kNkStencilCtl = 1u << 5,
	kNkModeCtl    = 1u << 6,
	kNkClipCtl    = 1u << 7,
	kNkColorCtl   = 1u << 8,
	kNkTargetMask = 1u << 9,
	kNkBlend0     = 1u << 10, // + colour slot (8 bits)
	kNkPolyOffset = 1u << 18, // the six PA_SU_POLY_OFFSET_* registers
	kNkViewport   = 1u << 19, // PA_CL_VPORT_*SCALE/OFFSET of viewport 0
	kNkScissor    = 1u << 20, // PA_SC_VPORT_SCISSOR_0_TL/BR
	kNkStencilRef = 1u << 21, // DB_STENCILREFMASK(_BF)
};

struct NativeDraw {
	uint32_t serial    = 0; // marker serial
	uint8_t  kind      = 0; // engine draw kind: 0 indexed, 1 indexed + offset, 2 auto, 3 auto + offset
	uint8_t  geom_slot = 0; // engine stage slot of the first geometry stage
	uint8_t  active    = 0; // engine stage slots active
	uint8_t  fail      = 0; // why the capture record is not "complete" (gameHooksCapture.cpp Fail)
	uint32_t known     = 0; // NativeKnown
	uint8_t  seg_start = 0; // first draw of a native segment
	uint8_t  pad0[3] {};
	// S0, segment length: in the head record (seg_start) the game thread keeps the number of draws
	// of the segment whose marker was written so far (1 at the head, then +1 per draw); the list
	// is complete before it is submitted, so the CP reads the final count at the head marker.
	// seg_closed: a later draw started another segment, or the segment was broken.
	uint32_t seg_count  = 0;
	uint32_t seg_closed = 0;
	uint32_t prim = 0, count = 0, instances = 0, first = 0, vertex_offset = 0;
	uint32_t ib_count = 0, ib_32 = 0;
	uint32_t user_gs_known = 0, user_ps_known = 0; // bit per user SGPR re-emitted since the list began
	uint64_t ib_addr   = 0;
	uint64_t pipeline  = 0; // engine pipeline object (ctx + 0xe28)
	uint64_t code_geom = 0; // engine shader object code of the geometry slot
	uint64_t code_ps   = 0; // and of the pixel slot (0: inactive)
	uint64_t pgm_es = 0, pgm_gs = 0, pgm_ps = 0; // SPI_SHADER_PGM_LO/HI_* from the shadow (bytes)
	uint64_t rt_hash    = 0; // colour target objects + depth target object
	uint64_t state_hash = 0; // programs + fixed-function registers below
	uint64_t memo_key   = 0; // pipeline object mixed with state_hash
	uint32_t user_gs[32] {}; // SPI_SHADER_USER_DATA_GS_0..31 (shadow)
	uint32_t user_ps[32] {}; // SPI_SHADER_USER_DATA_PS_0..31 (shadow)
	uint32_t db_render_control = 0, db_depth_control = 0, db_stencil_control = 0;
	uint32_t db_stencilrefmask = 0, db_stencilrefmask_bf = 0;
	uint32_t pa_su_sc_mode_cntl = 0, pa_cl_clip_cntl = 0, cb_color_control = 0, cb_target_mask = 0;
	uint32_t cb_blend[8] {};
	uint32_t poly_offset[6] {}; // DB_FMT_CNTL, CLAMP, FRONT_SCALE, FRONT_OFFSET, BACK_SCALE, BACK_OFFSET
	uint32_t vport[6] {};       // XSCALE, XOFFSET, YSCALE, YOFFSET, ZSCALE, ZOFFSET (viewport 0)
	uint32_t vport_scissor[2] {}, screen_scissor[2] {};
	// The engine's own copy of the 32 user SGPRs of the geometry stage slot and of the pixel slot
	// (stage slot + 0x10, after the flush wrote this draw's descriptors into them; 0: no copy).
	uint32_t sgpr_geom[32] {};
	uint32_t sgpr_ps[32] {};
};
constexpr uint32_t kNativeDrawDw = sizeof(NativeDraw) / 4u;
static_assert(sizeof(NativeDraw) % 4u == 0);
constexpr uint32_t kNativeSegCountWord  = offsetof(NativeDraw, seg_count) / 4u;
constexpr uint32_t kNativeSegClosedWord = offsetof(NativeDraw, seg_closed) / 4u;
static_assert(offsetof(NativeDraw, seg_count) % 4u == 0);

// The record of a marker, from any CP front (no consumption: the packet lookup consumes).
// False when the mode is not n2 or the entry was rewritten (counted on a real parse).
[[nodiscard]] bool LookupNativeRecord(const uint32_t* body, uint32_t marker_size_dw, bool real,
                                      NativeDraw& out);

// KYTY_NATIVE_VERIFY=1 with n2: the CP compared a record with its registers at the marker (real
// parse). `mismatch` / `unknown`: bits of NativeVerifyField; `detail`: the differing fields'
// values, logged for the first 8 mismatching draws.
enum NativeVerifyField : uint32_t {
	kNvPgmEs, kNvPgmGs, kNvPgmPs, kNvUserGs, kNvUserPs, kNvRenderCtl, kNvDepthCtl, kNvStencilCtl,
	kNvModeCtl, kNvClipCtl, kNvColorCtl, kNvTargetMask, kNvBlend, kNvPolyOffset, kNvViewport,
	kNvScissor, kNvIndexBase, kNvIndexType, kNvCount
};
[[nodiscard]] bool NativeVerifyEnabled();
void NoteNativeVerify(const uint32_t* body, uint32_t mismatch, uint32_t unknown, const char* detail);

// The op flag bits (graphics/guest_gpu/command_processor/cpOps.h, DrawIndexOp / DrawAutoOp
// flags) of a spliced native draw: bit 8 native, bits 9..12 the slot, bit 13 segment start.
constexpr uint32_t kNativeOpFlag      = 1u << 8u;
constexpr uint32_t kNativeOpSlotShift = 9u;
constexpr uint32_t kNativeOpSegStart  = 1u << 13u;
constexpr uint32_t kNativeOpMask      = 0x3fu << 8u;
// S0 only: bits 16..31 carry the low 16 bits of the marker serial, so that the executor can find
// the draw's record (LookupNativeRecordForOp). Other modes leave them 0.
constexpr uint32_t kNativeOpSerialShift = 16u;
[[nodiscard]] inline uint32_t NativeOpTag(const uint32_t* body) {
	return kNativeOpFlag | ((body[0] & kNativeSlotMask) << kNativeOpSlotShift) |
	       ((body[0] & kNativeSegStartFlag) != 0 ? kNativeOpSegStart : 0u) |
	       (NativeS0() ? (body[1] & 0xffffu) << kNativeOpSerialShift : 0u);
}
// How the emulator's draw run (KYTY_DRAW_RUN) treated a resolved native draw.
enum NativeRun : uint32_t {
	kNrContinued, // continuation of the previous draw's structure
	kNrRecalled,  // continuation of an earlier record (KYTY_DRAW_RUN_RECALL)
	kNrSwapped,   // continuation with a pipeline swap (KYTY_DRAW_RUN_PIPESWAP)
	kNrTargets,   // new structure, targets kept (KYTY_DRAW_RUN_TARGETS)
	kNrStarted,   // new structure, resolved in full
	kNrLean,      // n3b: a continuation committed on the lean path
	kNrNative,    // s1: emitted natively (renderer/nativeLearn.h), not through the emulator's commit
	kNrCount
};
// Resolver (Thread_Gpu): a native draw op ran from `start_tsc` to `end_tsc`.
void NoteNativeResolved(uint32_t op_flags, uint64_t start_tsc, uint64_t end_tsc, uint32_t run);

// ---- KYTY_NATIVE_MODE=n3b (stage N3b, docs\PLAN-PARALLELE.md) ------------------------------------
//
// N3b = N2 plus a lean commit of native continuations (RenderExecutor, renderDraw.cpp "N3b"): a
// native draw that continues the previous native draw of its segment (same engine thread slot, not a
// segment start, the previous executed draw a native draw of that slot that left a run record, then
// KYTY_DRAW_RUN's own certificate: same structure key, activity epoch, command buffer, tick,
// rendering instance, programs, validated preparation; no recall, swap, buffer writes, indirect or
// mesh draw) skips the generic work whose result is the previous draw's: the colour/depth target
// operations (gated by the register test that decides them), the hardware register checks, the
// barrier-safety evaluation and the run-record eligibility. Everything per draw (certificate,
// shader data, buffer bindings, vertex/index, descriptors, push constants, rendering begin,
// pipeline bind, draw) runs as on the normal path, so the packets are the normal path's.
// Why a native draw took the normal path (or kNlLean), reported per slot in "Native 10s".
enum NativeLeanReason : uint32_t {
	kNlLean,     // took the lean path
	kNlSegStart, // first draw of a native segment
	kNlChain,    // the previous executed draw was not a recorded native draw of this slot
	kNlOff,      // not an engine commit, or KYTY_DRAW_RUN not 1
	kNlRegs,     // registers may run a target operation, debug dump, MRT split pass
	kNlNotCont,  // KYTY_DRAW_RUN did not continue (start, targets kept, ...)
	kNlRecall,   // continued an earlier record (recall)
	kNlSwap,     // continued with a pipeline swap
	kNlWrites,   // the run's programs write buffers
	kNlShape,    // indirect, mesh, plain-pixel variant changed, shader writes, no lean record
	kNlLate,     // the continuation fell back late (images or command changed)
	kNlSample,   // KYTY_NATIVE_VERIFY=1: sampled for verification (normal path, predictions checked)
	kNlEarly,    // the draw returned before recording
	kNlCount
};
// Resolver: why a native draw took its path; `verify`: a sampled draw's predictions were checked,
// `mismatch`: and differed (verify mode stops the emulator after logging).
void NoteNativeLean(uint32_t op_flags, uint32_t reason, bool verified, bool mismatch);

// ---- KYTY_NATIVE_MODE=n3 (stage N3a, docs\PLAN-PARALLELE.md) -------------------------------------
//
// N3a = N2 plus "segment resolution windows" in the buffer cache (BufferCache::NativeTierBegin,
// renderer/cache/bufferCache.h): while the resolver executes a native draw op, a read binding
// (storage/uniform V#, vertex range, index range: ObtainBuffer neither written nor texel) whose
// range lies in a window certified earlier in the same native segment is answered from the window
// (buffer + linear offset), without the binding memo, tracker queries or ObtainBufferNow. A window
// is learned from the emulator's own result for a read binding of a cache buffer: the aligned range
// around it (KYTY_NATIVE_N3_WINDOW bytes, default 64 KiB, else its 4 KiB pages) inside that buffer,
// with no CPU-dirty page (hot pages are CPU-dirty) under an unchanged tracker signature. It holds
// while the sync epoch (syncEpoch.h), the buffer structure epoch and the unregister count are the
// ones it was learned under, within one native segment of one engine thread. That is the fact
// TryLeanReadMiss proves per binding ("no CPU-dirty page, registered buffer covering it: that buffer
// at Offset(vaddr)"), taken once per window and reused within its epoch, as syncEpoch.h allows
// (a CPU write between two ordering points races the GPU on the hardware too).
// KYTY_NATIVE_N3_STRICT=1: every window answer also re-reads the window's tracker signature (the
// binding memo's own condition). KYTY_NATIVE_N3_TIER=0: n3 without windows (A/B against n2).
// KYTY_NATIVE_VERIFY=1 (n3): every window answer also runs the normal path, which decides; a
// different buffer or offset is a mismatch, or a race when the window's signature moved.
// Per native draw, what the windows did (resolver; the buffer cache fills it, the CP reports it).
struct NativeN3Draw {
	uint32_t hits          = 0; // read bindings answered by a window
	uint32_t learned       = 0; // normal path, then a window recorded from its result
	uint32_t stream        = 0; // normal path gave a stream copy (CPU-dirty range): no window
	uint32_t dirty         = 0; // normal path gave a cache buffer, window refused (CPU-dirty pages)
	uint32_t other         = 0; // normal path, result not a plain cache-buffer answer
	uint32_t ineligible    = 0; // written / texel / memo off: normal path, never windowed
	uint32_t inv_epoch     = 0; // window found, sync epoch moved
	uint32_t inv_structure = 0; // window found, buffer structure changed
	uint32_t inv_unregister = 0; // window found, a buffer was unregistered
	uint32_t inv_signature = 0; // window found, tracker signature moved (strict mode)
	uint32_t verify_checks = 0, verify_races = 0, verify_mismatches = 0;
};
// Resolver: a native draw op ran under the windows from `start_tsc` to `end_tsc`.
void NoteNativeN3(uint32_t op_flags, uint64_t start_tsc, uint64_t end_tsc, const NativeN3Draw& draw);

// ---- KYTY_NATIVE_MODE=s0 (stage S0, docs\CONCEPTION-RENDU-NATIF.md) ------------------------------
//
// S0 = N2 (emission is still the emulator's N1 path) plus what a standalone native emitter needs to
// learn, and the proof that it could rebuild each draw from the record alone:
//  - segment length: the head record holds the segment's draw count (NativeDraw::seg_count); the CP
//    reads it at the head marker and checks it against the markers it then parses;
//  - learning (renderer/nativeLearn.h): right after the pipeline selection of a native draw on the
//    normal path, a native pipeline table keyed by NativeKey (ES and PS programs, record state hash,
//    target formats, primitive, specialization signature) keeps the pipeline, both compiled shaders,
//    mesh info, push-data start and a "resource recipe": for every V#/T#/S# and flat SRT dword, where
//    it comes from in terms of the draw's inputs (a user-data SGPR, a constant, or one load from a
//    user-data pointer + immediate offset); a deeper descriptor graph is not native-eligible;
//  - KYTY_NATIVE_VERIFY=1: 1 native draw in 64 is rebuilt from the record + recipe (V#/T#/S#, flat
//    SRT, shader-data user dwords, push constants, mesh draw dwords) and compared with what the
//    normal path committed; the first differences are logged;
//  - user-data mapping: the record's GS user registers (shadow) and the engine's slot copies are
//    compared with the user data the emulator's ES/mesh stage actually used.
// Executor (Thread_Gpu): the record of the draw of a native op (op flags carry slot and serial).
[[nodiscard]] bool LookupNativeRecordForOp(uint32_t op_flags, NativeDraw& out);
// Why a native draw is not covered by the native emitter (S0 learning), reported per slot.
enum NativeS0Reason : uint32_t {
	kS0Simple,      // every resource dword has a simple recipe
	kS0Flow,        // simple recipes, but resource sources gated by shader control flow
	kS0NoRecord,    // the op's record was rewritten or missing
	kS0WrongDraw,   // the executed draw is not the record's (an earlier pending draw)
	kS0NoPlan,      // no resource plan for a stage's permutation
	kS0Tess,        // more than one vertex stage
	kS0Indirect,    // an indirect (table-selected) descriptor
	kS0Deep,        // a descriptor dword deeper than one load from a user-data pointer
	kS0ConstBuffer, // a descriptor dword read through a V# (ReadConstBuffer)
	kS0TableFull,   // the native pipeline table is full
	kS0Skipped,     // the op's draw never reached the pipeline selection (no draw, copy, skipped)
	kS0ReasonCount
};
struct NativeS0Draw {
	uint32_t reason      = kS0NoRecord;
	bool     found       = false; // NativeKey already learned
	bool     learned     = false; // inserted by this draw
	bool     conflict    = false; // found, but a different program pair / layout: the key is too weak
	bool     replaced    = false; // found, same programs and layout, another pipeline object
	bool     verified    = false; // KYTY_NATIVE_VERIFY=1 sample rebuilt and compared
	uint32_t compared    = 0;     // dwords compared
	uint32_t mismatches  = 0;     // dwords that differ
	uint32_t nulled      = 0;     // recipe value non-zero, committed value zero (inactive / nulled)
	uint32_t deep_opcode = 0;     // kS0Deep: the IR opcode that stopped the recipe (+1, 0: none)
	// User-data mapping (verified samples): the emulator's ES/mesh user data vs the record.
	uint32_t ud_compared   = 0;  // user dwords compared
	uint32_t ud_shadow_eq  = 0;  // record shadow (GS user registers) == emulator, known dwords
	uint32_t ud_shadow_ne  = 0;  // and different
	uint32_t ud_shadow_unk = 0;  // shadow dword not known
	uint32_t ud_copy_eq    = 0;  // engine slot copy == emulator
	uint32_t ud_copy_ne    = 0;
	// S0b: why a found key's programs differ (NativeS0Conflict; first differing component), and for
	// a static-state difference the first differing word.
	uint32_t conflict_cause = 0;
	uint32_t conflict_word  = 0;
	// S1 verify (KYTY_NATIVE_VERIFY=1): this normal-path draw also had a native prediction
	// (sampled continuation or segment head), compared here.
	bool     s1_checked    = false;
	uint32_t s1_mismatches = 0;
};
// S0b: the first component in which a key conflict's two permutations differ.
enum NativeS0Conflict : uint32_t {
	kCfNone,      // no conflict
	kCfLayout,    // same permutations, another pipeline layout
	kCfVsHash,    // vertex stage: another source (code hash)
	kCfVsUdc,     // vertex stage: user-data count
	kCfVsCode,    // vertex stage: code size
	kCfVsStatic,  // vertex stage: static state (conflict_word: first differing word)
	kCfVsPush,    // vertex stage: push-data start
	kCfVsSpec,    // vertex stage: specialization
	kCfPsHash,
	kCfPsUdc,
	kCfPsCode,
	kCfPsStatic,
	kCfPsPush,
	kCfPsSpec,
	kCfPlain,     // plain pixel variant
	kCfOther,     // same identities (a duplicate module?)
	kCfCount
};
// Executor: one native draw's S0 outcome (after its draw command, or at an early stop).
void NoteNativeS0(uint32_t op_flags, const NativeS0Draw& draw);
// Executor: a verify sample differed (text logged for the first 16, then counted only).
void NoteNativeS0Mismatch(uint32_t op_flags, const char* text);
// Executor: the native pipeline table size (distinct NativeKeys, and distinct keys without the
// specialization signature), for the report.
void NoteNativeS0Table(uint32_t keys, uint32_t base_keys);

// ---- KYTY_NATIVE_MODE=s1 (stage S1, first subset; renderer/nativeLearn.h "S1") ----------------------
// Why a native op's draw was not emitted natively (kS1Emitted: it was), per slot in "Native 10s".
enum NativeS1Reason : uint32_t {
	kS1Emitted,   // emitted natively
	kS1SegStart,  // the head of a native segment (always the normal path)
	kS1Slot,      // not a Shadow-thread draw (first subset)
	kS1NoSeg,     // no segment state: the previous draw of the segment did not complete here
	kS1Activity,  // something else ran on the command processor, or another draw, since
	kS1Command,   // command buffer, tick or rendering instance changed, or no rendering instance
	kS1Regs,      // context / user-config registers differ from the segment's last normal draw
	kS1Kind,      // not an indexed draw (DrawIndex), 8-bit indices, primitive restart
	kS1Record,    // no record, or the executed draw is not the record's
	kS1Pixel,     // a pixel stage is active and KYTY_NATIVE_S1_PIXEL=0
	kS1Miss,      // NativeKey not learned (or conflicted, table full)
	kS1Shape,     // learned entry not covered (NativeS1Shape: why, per stage)
	kS1Read,      // a recipe load refused (guest bytes not clean)
	kS1Spec,      // specialization rejected a V#, T# or S#
	kS1Late,      // the binding work changed command buffer / rendering / image state
	kS1Sample,    // KYTY_NATIVE_VERIFY=1: sampled, normal path, prediction compared
	kS1Barriers,  // barriers pending and the segment's rendering state has a clear
	kS1Feedback,  // S1b: a texture of the draw is one of the segment's attachments
	kS1Plain,     // S1b: the pixel stage records mip statistics (the plain variant does not apply)
	kS1Count
};
// S1b: why a learned entry is not covered (first uncovered property; diagnostic breakdown of
// kS1Shape, per stage: vertex/mesh, pixel).
enum NativeS1Shape : uint32_t {
	kShCovered,
	kShTess,        // more than one vertex stage
	kShFlow,        // recipe with control flow (plan control flow)
	kShDeep,        // recipe deeper than user data / one load (or indirect descriptor, const buffer)
	kShWrites,      // written buffers / images, atomics, address writes
	kShBinding,     // binding kind not covered (GDS, shared memory, fault buffer...)
	kShUniform,     // uniform fill
	kShFetch,       // embedded vertex fetch or vertex buffers
	kShClip,        // user clip planes
	kShPrefix,      // user-data prefix neither zero nor the GS user-data address
	kShIndirect,    // a resource recipe without dwords (indirect / unknown source)
	kShFlat,        // flattened SRT size differs
	kShUdCount,     // bound user SGPR count differs from the learned one
	kShVariant,     // the full key's entry differs from the base key's representative
	kShPipeline,    // no pipeline object (or mesh flag differs)
	kShLodVerify,   // pixel stage with KYTY_LOD_STATS_PLAIN_VARIANT=verify (canary)
	kShImageClass,  // pixel/vertex image not covered (storage, indirect image table)
	kShCount
};
struct NativeS1Draw {
	uint32_t reason      = kS1Emitted;
	uint32_t shape       = kShCovered; // kS1Shape: NativeS1Shape
	uint32_t shape_stage = 0;          // kS1Shape: 0 vertex/mesh, 1 pixel
	bool     pixel       = false;      // the draw has a pixel stage
	// S1c (kS1Regs with KYTY_NATIVE_S1_REGS_DIAG=1): the register groups that differ from the
	// segment's last normal draw, bit = NativeS1RegPart.
	uint32_t regs_parts  = 0;
};
// S1c: register groups of a kS1Regs fallback (HW::Context::VisitKeyDiagParts groups, then the user
// clip planes, the user-config registers and context bytes outside every group).
enum NativeS1RegPart : uint32_t {
	kRpTargets, kRpClear, kRpDepth, kRpBlend, kRpBlendColor, kRpDepthState, kRpStencil, kRpClearValues, kRpViewport,
	kRpRaster, kRpShRegs, kRpUserClip, kRpUserConfig, kRpOther, kRpCount
};
// Executor: one native op's S1 outcome (emitted, or why not).
void NoteNativeS1(uint32_t op_flags, const NativeS1Draw& draw);
// Executor: a verify prediction differed (logged; verify mode then stops the emulator).
void NoteNativeS1Mismatch(uint32_t op_flags, const char* text);
// Whether S1 emits natively for the op's engine thread slot (KYTY_NATIVE_S1: slot keys or "all";
// default "shadow", the first subset).
[[nodiscard]] bool NativeS1SlotEnabled(uint32_t op_flags);

// Guest flips (graphics/presentation/videoOut.cpp), for the per-frame averages.
extern std::atomic<uint64_t> g_flips;
inline void NoteFlip() {
	if (Active()) {
		g_flips.fetch_add(1, std::memory_order_relaxed);
	}
}

// What the command processor knows when it parses a draw packet.
struct CpDrawView {
	uint64_t packet       = 0; // guest address of the packet header
	uint32_t opcode       = 0;
	bool     speculative  = false; // parsed by the speculative (prefetch) front
	uint32_t index_count  = 0; // from the packet (vertex count for DRAW_INDEX_AUTO)
	uint32_t index_offset = 0; // DRAW_INDEX_OFFSET_2 only
	uint32_t instances    = 0; // current instance count (front view)
	bool     instances_known = true;
	uint64_t ib_base      = 0; // INDEX_BASE
	uint32_t ib_type      = 0; // VGT_INDEX_TYPE (0 = 16-bit, 1 = 32-bit)
	uint64_t rt0_base     = 0; // CB_COLOR0_BASE(+EXT) as decoded by the emulator
	uint64_t ps_addr      = 0; // SPI_SHADER_PGM_LO/HI_PS
	uint64_t gs_addr      = 0; // SPI_SHADER_PGM_LO/HI_GS
	// SPI_SHADER_PGM_LO/HI_ES: the program the emulator runs as the vertex/geometry stage
	// (es_regs, DrawHasValidVertexShader). The engine's GS-family code is compared with it: the GS
	// pair held one constant value in every logged "geom s2" mismatch (14/11).
	uint64_t es_addr      = 0;
	uint64_t hs_addr      = 0; // SPI_SHADER_PGM_LO/HI_HS
};

// KYTY_NATIVE_DUMP=<n> (graphics/presentation/window/swapchain.cpp): the presented frame number
// to write as a PNG (0 = off), and the directory (KYTY_NATIVE_DUMP_DIR, default
// E:\JEUX\ASTROBOT_PC\Outils_Claude\work\native).
[[nodiscard]] uint64_t    NativeDumpFrame();
[[nodiscard]] const char* NativeDumpDir();

void CheckCpDrawSlow(const CpDrawView& view);
inline void CheckCpDraw(const CpDrawView& view) {
	if (Active()) {
		CheckCpDrawSlow(view);
	}
}

} // namespace Loader::GameCapture

#endif // KYTY_LOADER_GAME_CAPTURE_H_
