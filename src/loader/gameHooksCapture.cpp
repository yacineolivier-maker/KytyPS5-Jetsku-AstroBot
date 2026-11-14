// KYTY_GAME_HOOKS=capture: "native draw record" capture (Astro Bot PC, native rendering backend,
// stage "capture"; docs\CARTE-BACKEND-GPU.md, sections 4, 6 and "Capture"). Observation only:
// no guest register or memory is changed, rendering is unchanged.
//
// Astro Bot (PPSA21567) engine GPU backend, ELF addresses (SysV; ctx = the ~0x8d08-byte backend
// context object, one per command list):
//   0x4828d0 (ctx)                       draw state flush: After hook + entry observer. Entry:
//                                        dirty mask (+0x6070) before the flush. Exit: the
//                                        record is built from the post-flush context.
//   0x484800 (ctx)                       dispatch state flush: After + entry (scope, stage use).
//   0x483c20 (ctx, prim, count, instances, first, vertex_offset)     indexed draw   } Before:
//   0x483e30 (same + ...)                                             indexed + ofs  } arguments,
//   0x4836f0 (ctx, prim, count, vertex_offset)                        auto draw      } read before
//   0x4838c0 (ctx, prim, count, instances, vertex_offset, inst_ofs)   auto + ofs     } the flush
//   0x4840ac (ctx, prim, buffer id, args offset)                      indexed indirect (reached by
//                                        fall-through from 0x4840a0, which returns on id 0xffff)
//   0x48d450 (cb = ctx+0xe50, acc = ctx+0x1208, ps, instances)  tool-library full-screen draw
//                                        (own shaders, bypasses the context): After + entry.
//   0x47ee00 (ctx)                       command list begin (ResetQueue, defaults, marks every
//                                        group dirty): After + entry.
//   0x6ef22b0 / 0x6ef2400 / 0x6ef2550 (acc, pairs, count)  static Agc register accumulators
//                                        (context / SH / uconfig) — Before: every {offset, value}
//                                        pair written on a context's accumulator (ctx + 0x1208)
//                                        updates a per-context shadow register file, classified
//                                        by the scope it comes from (flush, list begin, tool
//                                        library, inside a hooked draw, or "outside": direct
//                                        calls by the render stages).
//
// Threads: every hook runs on the calling game thread. Per-thread state is thread_local and
// written by its thread only (relaxed atomics where the reporter reads it); a context object is
// used by one thread at a time (one command list), so its shadow needs no lock. Locks: thread
// and context registration (once each), the 10 s report (try-lock).

//
// KYTY_NATIVE (stage N0 of the native backend, docs\PLAN-PARALLELE.md): on the selected engine
// threads, a hooked indexed / auto draw (not indirect, no tessellation stage) is chosen as
// "native" at its flush exit; the direct draw packet libAgc then writes for it is replaced by a
// NOP marker of the same size (gameCapture.h), which the command processor counts and drops
// (pm4Handlers.cpp CpOpNativeDraw). KYTY_NATIVE unset: nothing of this runs.
// Stage N1 (KYTY_NATIVE_MODE=splice): before the rewrite, the original packet is published in the
// slot's native ring under the marker serial (NativeRing, gameCapture.h); the CP looks it up and
// runs the packet's own handler at the marker (image unchanged). Ring full: the normal packet
// stays (fallback "ring-full"). KYTY_NATIVE_VERIFY=1: a record miss on the CP exits.
// Stage N2 (KYTY_NATIVE_MODE=n2, gameCapture.h "KYTY_NATIVE_MODE=n2"): N1, plus the NativeDraw
// record of every native draw (built at its flush exit from the context and the register shadow,
// published under the marker serial in the slot's record ring), native segments (seg_start), the
// CP's field-by-field comparison of the record with its registers (KYTY_NATIVE_VERIFY=1) and the
// resolver's Thread_Gpu time per native draw and per segment ("Native 10s" n2 part).
// Stage N3a (KYTY_NATIVE_MODE=n3, gameCapture.h "KYTY_NATIVE_MODE=n3"): N2, plus the buffer cache's
// segment resolution windows during every native draw op; "Native 10s" n3 part: draws and
// Thread_Gpu time by window path, binding outcomes, window invalidations, verify results.

#include "common/logging/log.h"
#include "graphics/guest_gpu/pm4.h"
#include "kernel/pthread.h"
#include "loader/gameCapture.h"
#include "loader/gameHooks.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <intrin.h>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace Loader::GameCapture {
std::atomic<bool>     g_active {false};
std::atomic<uint64_t> g_flips {0};
std::atomic<uint8_t>  g_native_mode {0};
} // namespace Loader::GameCapture

namespace Loader::GameHooks::Capture {

// ---- KYTY_NATIVE configuration ------------------------------------------------------------------

namespace {

constexpr uint32_t    kEngineThreads                     = 5;
constexpr const char* kEngineThreadNames[kEngineThreads] = {"Draw Geometry", "Draw Extra Geometry", "Draw Shadow",
                                                            "Draw Decal", "DrawThread"};
constexpr const char* kEngineThreadKeys[kEngineThreads]  = {"geometry", "extra", "shadow", "decal", "drawthread"};

std::string Lower(std::string text) {
	for (auto& ch: text) {
		ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
	}
	return text;
}

std::string Trim(const std::string& text) {
	const auto begin = text.find_first_not_of(" \t");
	if (begin == std::string::npos) {
		return {};
	}
	const auto end = text.find_last_not_of(" \t");
	return text.substr(begin, end - begin + 1);
}

int EngineThreadSlot(const char* name) {
	for (uint32_t s = 0; s < kEngineThreads; s++) {
		if (std::strcmp(name, kEngineThreadNames[s]) == 0) {
			return static_cast<int>(s);
		}
	}
	return -1;
}

struct NativeConfig {
	bool        requested = false; // KYTY_NATIVE set to something other than 0 / off / none
	bool        valid     = false; // requested, mode supported and at least one thread selected
	uint32_t    mask      = 0;     // bit = engine thread slot
	std::string unknown;           // tokens that name no engine thread
	std::string mode;
	bool        splice = false;    // KYTY_NATIVE_MODE=splice (stage N1) or n2, else drop (N0)
	bool        n2     = false;    // KYTY_NATIVE_MODE=n2 or n3 (stage N2: N1 + records, verify, timing)
	bool        n3     = false;    // KYTY_NATIVE_MODE=n3 (stage N3a: N2 + segment resolution windows)
	bool        n3b    = false;    // KYTY_NATIVE_MODE=n3b (stage N3b: N2 + lean native continuations)
	bool        s0     = false;    // KYTY_NATIVE_MODE=s0 (stage S0: N2 + native learning, recipes, verify)
	bool        s1     = false;    // KYTY_NATIVE_MODE=s1 (stage S1: S0 + native emission of continuations)
	uint32_t    s1_mask = 1u << 2u; // KYTY_NATIVE_S1: engine thread slots S1 emits for (default shadow)
	bool        verify = false;    // KYTY_NATIVE_VERIFY=1: a splice record miss exits
};

NativeConfig ParseNative() {
	NativeConfig config;
	const char*  value = std::getenv("KYTY_NATIVE");
	if (value == nullptr) {
		return config;
	}
	const std::string all = Lower(Trim(value));
	if (all.empty() || all == "0" || all == "off" || all == "none") {
		return config;
	}
	config.requested = true;
	std::string token;
	const auto  take = [&] {
		const auto t = Lower(Trim(token));
		token.clear();
		if (t.empty()) {
			return;
		}
		if (t == "all") {
			config.mask = (1u << kEngineThreads) - 1u;
			return;
		}
		for (uint32_t s = 0; s < kEngineThreads; s++) {
			if (t == kEngineThreadKeys[s] || t == Lower(kEngineThreadNames[s])) {
				config.mask |= 1u << s;
				return;
			}
		}
		config.unknown += " " + t;
	};
	for (const char* p = value;; p++) {
		if (*p == '\0' || *p == ',' || *p == ';') {
			take();
			if (*p == '\0') {
				break;
			}
		} else {
			token += *p;
		}
	}
	const char* mode = std::getenv("KYTY_NATIVE_MODE");
	config.mode      = Lower(Trim(mode != nullptr ? mode : ""));
	if (config.mode.empty()) {
		config.mode = "drop";
	}
	config.n3     = config.mode == "n3";
	config.n3b    = config.mode == "n3b";
	config.s1     = config.mode == "s1";
	config.s0     = config.mode == "s0" || config.s1;
	if (const char* s1 = std::getenv("KYTY_NATIVE_S1"); s1 != nullptr && s1[0] != '\0') {
		const auto list = Lower(Trim(s1));
		config.s1_mask  = 0;
		for (uint32_t s = 0; s < kEngineThreads; s++) {
			if (list == "all" || list.find(kEngineThreadKeys[s]) != std::string::npos) config.s1_mask |= 1u << s;
		}
	}
	config.n2     = config.mode == "n2" || config.n3 || config.n3b || config.s0;
	config.splice = config.mode == "splice" || config.n2;
	config.valid  = (config.mode == "drop" || config.splice) && config.mask != 0;
	const char* verify = std::getenv("KYTY_NATIVE_VERIFY");
	config.verify      = verify != nullptr && std::strcmp(verify, "0") != 0 && verify[0] != 0;
	return config;
}

const NativeConfig& Native() {
	static const NativeConfig config = ParseNative();
	return config;
}

} // namespace

bool NativeRequested() {
	return Native().requested;
}

} // namespace Loader::GameHooks::Capture

namespace Loader::GameHooks::Capture {

namespace {

// ---- Engine layout (docs\CARTE-BACKEND-GPU.md § 4) ---------------------------------------------

constexpr uint64_t kCtxPipelineObj  = 0xe28;
constexpr uint64_t kCtxPipelineType = 0xe30; // int: 0 none, 1..4 graphics, 5 compute
constexpr uint64_t kCtxCb           = 0xe50; // Agc command buffer
constexpr uint64_t kCtxSlots        = 0xe88; // 8 stage slots
constexpr uint64_t kSlotStride      = 0x70;
constexpr uint64_t kSlotShader      = 0x00; // Agc shader object
constexpr uint64_t kSlotSgpr        = 0x10; // pointer to the 32 user SGPR copy
constexpr uint64_t kSlotVtxTable    = 0x20;
constexpr uint64_t kSlotActive      = 0x58; // byte
constexpr uint64_t kCtxAcc          = 0x1208; // context register accumulator
constexpr uint64_t kCtxSuScMode     = 0x13f8; // PA_SU_SC_MODE_CNTL value (pair at 0x13f4)
constexpr uint64_t kCtxRt           = 0x60c0; // 8 color targets, stride 0x38
constexpr uint64_t kRtStride        = 0x38;
constexpr uint64_t kCtxDepthObj     = 0x6288;
constexpr uint64_t kCtxDirty        = 0x6070;
constexpr uint64_t kCtxDepthControl = 0x6654;
constexpr uint64_t kCtxStencilRef   = 0x6658;
constexpr uint64_t kCtxStencilRefBf = 0x665c;
constexpr uint64_t kCtxStencilCtl   = 0x6664;
constexpr uint64_t kCtxRenderCtl    = 0x666c;
constexpr uint64_t kCtxIbCount      = 0x6678;
constexpr uint64_t kCtxIb32         = 0x6680;
constexpr uint64_t kCtxIbAddr       = 0x6688;
constexpr uint64_t kCtxVbMode       = 0x66a0;
constexpr uint64_t kCtxBlendRt0     = 0x7184; // CB_BLEND0_CONTROL value
constexpr uint64_t kCtxBlendRt1     = 0x70dc; // CB_BLEND1..7_CONTROL values, stride 0x18
constexpr uint64_t kCtxBlendColor   = 0x7188; // 4 dwords, stride 4
constexpr uint64_t kCtxAlphaToMask  = 0x71a0;
constexpr uint64_t kCtxPrim         = 0x8cd0;

// Agc shader object (graphics/shader/shader.h, struct Shader).
constexpr uint64_t kShaderCode = 0x10;
constexpr uint64_t kShaderType = 0x5a; // byte: 0 Cs 1 Ps 2 Gs 3 Hs 4 GsFront 5 HsFront 6 GsBack 7 HsBack

constexpr uint64_t kAllGroups = 0x1fffffffffffffffull; // what 0x47ee00 stores in the mask

// Registers (offsets in the context / SH register spaces, graphics/guest_gpu/pm4.h).
constexpr uint32_t kCxNum             = 0x400;
constexpr uint32_t kShNum             = 0x300;
constexpr uint32_t kRegSelectorMask   = 0x70000000u;
constexpr uint32_t kCbColor0Base      = 0x318; // + 15 * slot
constexpr uint32_t kCbColor0Info      = 0x31c;
constexpr uint32_t kCbColor0BaseExt   = 0x390; // + slot
constexpr uint32_t kCbColor0Attrib2   = 0x3b0; // + slot (mip0 width/height)
constexpr uint32_t kCbTargetMask      = 0x8e;
constexpr uint32_t kDbZInfo           = 0x10;
constexpr uint32_t kDbZReadBase       = 0x12;
constexpr uint32_t kDbZReadBaseHi     = 0x1a;
constexpr uint32_t kDbDepthSizeXy     = 0x7;
constexpr uint32_t kPaClVportXscale   = 0x10f; // 6 registers
constexpr uint32_t kPaScVportScissor0 = 0x94;  // TL, BR
constexpr uint32_t kPaScScreenScissor = 0xc;   // TL, BR
constexpr uint32_t kSpiPgmLoPs        = 0x8;
constexpr uint32_t kSpiPgmLoGs        = 0x88;
constexpr uint32_t kSpiPgmLoHs        = 0x108;

// ---- Hooked functions ---------------------------------------------------------------------------

constexpr uint64_t kFlushDraw     = 0x4828d0;
constexpr uint64_t kFlushDispatch = 0x484800;
constexpr uint64_t kDrawIndexed   = 0x483c20;
constexpr uint64_t kDrawIndexedOf = 0x483e30;
constexpr uint64_t kDrawAuto      = 0x4836f0;
constexpr uint64_t kDrawAutoOf    = 0x4838c0;
constexpr uint64_t kDrawIndirect  = 0x4840ac;
constexpr uint64_t kToolkitDraw   = 0x48d450;
constexpr uint64_t kListBegin     = 0x47ee00;
constexpr uint64_t kAccCx         = 0x6ef22b0;
constexpr uint64_t kAccSh         = 0x6ef2400;
constexpr uint64_t kAccUc         = 0x6ef2550;

// Bytes read from Outils_Claude\work\ghidra\eboot.elf (PT_LOAD 0 maps vaddr 0 at file offset
// 0x4000). All start with push rbp; mov rbp,rsp; push r15 (6 relocatable bytes).
const std::vector<uint8_t> kBytesFlushDraw     = {}; // original game bytes not published (fill in locally to enable this hook)
const std::vector<uint8_t> kBytesFlushDispatch = {}; // original game bytes not published (fill in locally to enable this hook)
const std::vector<uint8_t> kBytesDrawIndexed   = {}; // original game bytes not published (fill in locally to enable this hook)
const std::vector<uint8_t> kBytesDrawIndexedOf = {}; // original game bytes not published (fill in locally to enable this hook)
const std::vector<uint8_t> kBytesDrawAuto      = {}; // original game bytes not published (fill in locally to enable this hook)
const std::vector<uint8_t> kBytesDrawAutoOf    = {}; // original game bytes not published (fill in locally to enable this hook)
const std::vector<uint8_t> kBytesDrawIndirect  = {}; // original game bytes not published (fill in locally to enable this hook)
const std::vector<uint8_t> kBytesToolkitDraw   = {}; // original game bytes not published (fill in locally to enable this hook)
const std::vector<uint8_t> kBytesListBegin     = {}; // original game bytes not published (fill in locally to enable this hook)
const std::vector<uint8_t> kBytesAccCx         = {}; // original game bytes not published (fill in locally to enable this hook)
const std::vector<uint8_t> kBytesAccSh         = {}; // original game bytes not published (fill in locally to enable this hook)
const std::vector<uint8_t> kBytesAccUc         = {}; // original game bytes not published (fill in locally to enable this hook)

// ---- Guest memory --------------------------------------------------------------------------------

template <typename T>
T Rd(uint64_t base, uint64_t offset) {
	T value;
	std::memcpy(&value, reinterpret_cast<const void*>(base + offset), sizeof(T));
	return value;
}

// ---- Records --------------------------------------------------------------------------------------

enum DrawKind : uint8_t { kKindIndexed, kKindIndexedOfs, kKindAuto, kKindAutoOfs, kKindIndirect, kKindUnknown };
constexpr const char* kKindNames[] = {"idx", "idx+ofs", "auto", "auto+ofs", "indirect", "flush-only"};

// Reasons a draw is not "complete" (see IsComplete in the "Capture" section of the map).
enum Fail : uint32_t {
	kFailPipeline = 1u << 0, // no pipeline object, or not a graphics pipeline type (1..4)
	kFailStage    = 1u << 1, // first geometry slot inactive, or an active slot 1..7 without a
	                         // shader or whose shader type differs from the slot index
	kFailTarget   = 1u << 2, // no color target object and no depth target object
	kFailIndex    = 1u << 3, // indexed: no index buffer address, or first + count > IB size
	kFailOutside  = 1u << 4, // a register written on this context outside the context paths
	                         // (stage code or tool library) and not re-emitted since
	kFailNoBegin  = 1u << 5, // no list begin (0x47ee00) seen on this context since start
};

struct RtRec {
	uint64_t obj     = 0;
	uint64_t base    = 0; // shadow CB_COLORn_BASE(+EXT), byte address
	uint32_t info    = 0; // shadow CB_COLORn_INFO
	uint32_t attrib2 = 0; // shadow CB_COLORn_ATTRIB2 (mip0 size)
	uint16_t id      = 0;
	uint8_t  view    = 0;
	uint8_t  known   = 0; // base emitted on this context since its list began
};

struct DrawRecord {
	std::atomic<uint64_t> seq {0}; // 0 while being written
	uint64_t ctx = 0;
	uint32_t kind = 0, prim = 0, count = 0, instances = 0, first = 0, vertex_offset = 0;
	uint32_t indirect_id = 0, indirect_offset = 0;
	uint64_t dirty = 0;
	uint64_t pipeline = 0;
	uint32_t pipeline_type = 0;
	uint8_t  active = 0, geom_slot = 0, fail = 0, stale_outside = 0;
	uint64_t shader[8] {};
	uint64_t code[8] {};
	uint8_t  type[8] {};
	int8_t   sgpr_slot[3] {-1, -1, -1}; // first geometry slot, pixel slot, other geometry family
	uint32_t sgpr[3][32] {};
	RtRec    rt[8];
	uint64_t depth_obj = 0, z_base = 0;
	uint32_t z_info = 0, depth_size_xy = 0, cb_target_mask = 0;
	uint32_t vport[6] {}, vport_scissor[2] {}, screen_scissor[2] {};
	uint64_t ib_addr = 0;
	uint32_t ib_count = 0, ib_32 = 0, vb_mode = 0;
	uint64_t vb_table = 0;
	uint32_t depth_control = 0, stencil_control = 0, stencil_ref = 0, stencil_ref_bf = 0;
	uint32_t render_control = 0, su_sc_mode = 0, alpha_to_mask = 0;
	uint32_t blend[8] {}, blend_color[4] {};
	uint64_t rt_hash = 0; // colour target objects + depth target object (KYTY_NATIVE n2 segments)
};

constexpr size_t kRing = 32;

// ---- Per-context shadow -----------------------------------------------------------------------------

constexpr uint32_t kPendingMax = 64;
constexpr uint8_t  kFlagOutside = 1; // written outside the context paths, not re-emitted since
constexpr uint8_t  kFlagBegin   = 2; // last written by the list begin

struct ContextState {
	uint64_t ctx        = 0;
	uint32_t epoch      = 0; // list begins seen (0: none)
	bool     first_pending = false;
	uint32_t pending_count = 0;
	bool     pending_overflow = false;
	uint16_t pending[kPendingMax] {}; // bank << 12 | offset (bank 0 cx, 1 sh)
	uint32_t cx[kCxNum] {};
	uint32_t cx_epoch[kCxNum] {}; // epoch + 1 of the last write (0: never)
	uint8_t  cx_flags[kCxNum] {};
	uint32_t sh[kShNum] {};
	uint32_t sh_epoch[kShNum] {};
	uint8_t  sh_flags[kShNum] {};

	[[nodiscard]] bool CxKnown(uint32_t reg) const { return epoch != 0 && cx_epoch[reg] == epoch + 1; }
	[[nodiscard]] bool ShKnown(uint32_t reg) const { return epoch != 0 && sh_epoch[reg] == epoch + 1; }
};

constexpr size_t                                 kMaxContexts = 256;
std::array<std::atomic<ContextState*>, kMaxContexts> g_contexts {};
std::mutex                                       g_contexts_mutex;
std::atomic<uint32_t>                            g_contexts_dropped {0};

ContextState* FindOrAddContext(uint64_t ctx) {
	for (auto& slot: g_contexts) {
		auto* state = slot.load(std::memory_order_acquire);
		if (state == nullptr) {
			break;
		}
		if (state->ctx == ctx) {
			return state;
		}
	}
	std::lock_guard lock(g_contexts_mutex);
	for (auto& slot: g_contexts) {
		auto* state = slot.load(std::memory_order_acquire);
		if (state == nullptr) {
			auto* fresh = new ContextState;
			fresh->ctx  = ctx;
			slot.store(fresh, std::memory_order_release);
			return fresh;
		}
		if (state->ctx == ctx) {
			return state;
		}
	}
	g_contexts_dropped.fetch_add(1, std::memory_order_relaxed);
	return nullptr;
}

// Global histogram of registers written "outside" (bank 0 cx, 1 sh).
std::array<std::atomic<uint32_t>, kCxNum> g_outside_cx {};
std::array<std::atomic<uint32_t>, kShNum> g_outside_sh {};
std::array<std::atomic<uint32_t>, kCxNum> g_begin_cx {};

// ---- Per-thread state ----------------------------------------------------------------------------------

enum Ctr : uint32_t {
	kCDraw0, // + DrawKind (6)
	kCToolkit = kCDraw0 + 6,
	kCFlush,
	kCFlushNoDraw,
	kCDispatchFlush,
	kCRecords,     // hooked draws with a record (kinds 0..4)
	kCComplete,
	kCFail0,       // + bit index of Fail (6)
	kCRtResolved = kCFail0 + 6,
	kCShaderSelfOk,
	kCShaderSelfBad,
	kCDirtyBits,
	kCFirstLists,
	kCFirstFull,
	kCFirstDispatch,
	kCNoBeginDraws,
	kCAcc0, // + AccClass (6)
	kCHleHooked = kCAcc0 + 6,
	kCHleToolkit,
	kCHleOther,
	kCSamples,
	kCFam0, // + family (16)
	kCCount = kCFam0 + 16
};

enum AccClass : uint32_t { kAccFlush, kAccBegin, kAccToolkit, kAccInDraw, kAccOutside, kAccForeign };
constexpr const char* kAccNames[] = {"flush", "begin", "toolkit", "in-draw", "outside", "foreign"};

struct Family {
	const char* name;
	uint64_t    mask;
};
constexpr uint64_t Bits(int lo, int hi) {
	return ((hi == 63 ? ~0ull : ((1ull << (hi + 1)) - 1)) >> lo) << lo;
}
constexpr Family kFamilies[16] = {
    {"rt", Bits(0, 7)},       {"depth", Bits(8, 8)},  {"msaa", Bits(9, 10)},  {"vp", Bits(11, 14)},
    {"ib", Bits(15, 17)},     {"ds", Bits(18, 20)},   {"vb", Bits(21, 21)},   {"blend", Bits(22, 31)},
    {"smp", Bits(32, 43)},    {"pipe", Bits(44, 44)}, {"tex", Bits(45, 50)},  {"rw", Bits(51, 51)},
    {"srt", Bits(52, 57)},    {"dbnd", Bits(58, 58)}, {"b59", Bits(59, 59)},  {"cons", Bits(60, 60)},
};

// KYTY_NATIVE counters (per thread).
enum NCtr : uint32_t {
	kNChosen,     // draws chosen native at their flush exit
	kNWritten,    // markers written in place of their draw packet
	kNIncomplete, // chosen draws whose record is not "complete" (information only)
	kNFbIndirect, // fallback: indirect draw
	kNFbTess,     // fallback: tessellation (hull-shader geometry slot)
	kNFbPacket,   // fallback: the next libAgc draw packet was not a direct draw of known size
	kNFbLost,     // fallback: window closed without any libAgc draw packet
	kNFbRing,     // fallback (splice): the slot's native ring was full, normal packet kept
	kNSegments,   // n2: native draws that start a segment (game side)
	kNRecIncomplete, // n2: records whose fixed-function registers are not all known
	kNCount
};
constexpr uint32_t kNFb0 = kNFbIndirect;
constexpr const char* kNFbNames[] = {"indirect", "tess", "packet", "lost", "ring-full"};
constexpr uint32_t    kNFbKinds   = 5;

// KYTY_NATIVE_MODE=splice: per engine-thread slot ring of original draw packets, indexed by the
// marker serial. One producer (the game thread that owns the slot), readers on the command
// processor fronts. An entry is a seqlock: `serial` 0 while it is rewritten, then the serial
// (release). The producer reuses an entry only once the CP consumed a later serial (`consumed`,
// a running maximum written back by the real parse), so a record stays valid while frames in
// flight still hold its marker.
constexpr uint32_t kNativeRingSize = 16384; // power of two; ~3 frames of Extra Geometry draws
struct NativeEntry {
	std::atomic<uint32_t>                                  serial {0};
	std::atomic<uint32_t>                                  size_dw {0};
	std::array<std::atomic<uint32_t>, GameCapture::kNativeMaxPacketDw> words {};
};
struct NativeRing {
	alignas(64) std::atomic<uint32_t> consumed {0};
	alignas(64) std::atomic<uint32_t> owner {0}; // thread id + 1 of the producer (claimed once)
	std::array<NativeEntry, kNativeRingSize> entries {};
};
std::array<NativeRing*, kEngineThreads> g_native_rings {}; // allocated at install (splice)

// KYTY_NATIVE_MODE=n2: per slot ring of NativeDraw records, indexed like the packet ring by the
// marker serial (same producer, same seqlock; the packet ring's `consumed` guards reuse: a record
// is always written together with its packet entry).
struct NativeRecordEntry {
	std::atomic<uint32_t>                                     serial {0};
	std::array<std::atomic<uint32_t>, GameCapture::kNativeDrawDw> words {};
};
struct NativeRecordRing {
	std::array<NativeRecordEntry, kNativeRingSize> entries {};
};
std::array<NativeRecordRing*, kEngineThreads> g_native_records {}; // allocated at install (n2)

// n2 counters. CP side (any front, real parse): record lookups that missed; verify comparisons.
std::array<std::atomic<uint64_t>, kEngineThreads + 1> g_n2_rec_ok {};
std::array<std::atomic<uint64_t>, kEngineThreads + 1> g_n2_rec_miss {};
std::array<std::atomic<uint64_t>, kEngineThreads + 1> g_n2_verified {};
std::array<std::atomic<uint64_t>, kEngineThreads + 1> g_n2_verify_bad {};
std::array<std::atomic<uint64_t>, GameCapture::kNvCount> g_n2_field_bad {};
std::array<std::atomic<uint64_t>, GameCapture::kNvCount> g_n2_field_unknown {};
std::atomic<uint32_t>                                  g_n2_logged {0};
constexpr const char* kNvNames[GameCapture::kNvCount] = {
    "pgm-es", "pgm-gs", "pgm-ps", "user-gs", "user-ps", "render-ctl", "depth-ctl", "stencil-ctl", "mode-ctl", "clip-ctl",
    "color-ctl", "target-mask", "blend", "poly-offset", "viewport", "scissor", "index-base", "index-type"};
// Resolver side (Thread_Gpu): native draw ops by draw-run treatment, their TSC cycles, segments
// (first op start to last op end of a run of ops of one segment) and their draws.
constexpr const char* kNrNames[GameCapture::kNrCount] = {"cont", "recall", "swap", "targets", "start", "lean", "s1-native"};
// n3b (resolver): why each native draw took the lean or the normal path, and the verify samples.
constexpr const char* kNlNames[GameCapture::kNlCount] = {"lean", "seg-start", "chain", "off", "regs", "not-cont", "recall",
                                                         "swap", "writes", "shape", "late", "sample", "early"};
struct NlResolve {
	std::array<std::atomic<uint64_t>, GameCapture::kNlCount> reasons {};
	std::atomic<uint64_t>                                    verified {0};
	std::atomic<uint64_t>                                    mismatches {0};
};
std::array<NlResolve, kEngineThreads> g_nl_resolve {};
struct NlResolveLast {
	std::array<uint64_t, GameCapture::kNlCount> reasons {};
	uint64_t                                    verified = 0, mismatches = 0;
};
std::array<NlResolveLast, kEngineThreads> g_nl_resolve_last {}; // reporter only
struct N2Resolve {
	std::array<std::atomic<uint64_t>, GameCapture::kNrCount> draws {};
	std::array<std::atomic<uint64_t>, GameCapture::kNrCount> ticks {};
	std::atomic<uint64_t>                                    segments {0};
	std::atomic<uint64_t>                                    segment_ticks {0};
	std::atomic<uint64_t>                                    segment_draws {0};
	std::array<std::atomic<uint64_t>, 5>                     seg_hist {}; // 1, 2-4, 5-16, 17-64, 65+
	std::atomic<uint64_t>                                    unmarked_starts {0}; // span opened without seg_start
};
std::array<N2Resolve, kEngineThreads> g_n2_resolve {};
struct N2ResolveLast {
	std::array<uint64_t, GameCapture::kNrCount> draws {}, ticks {};
	uint64_t segments = 0, segment_ticks = 0, segment_draws = 0, unmarked = 0;
	std::array<uint64_t, 5> seg_hist {};
};
std::array<N2ResolveLast, kEngineThreads> g_n2_resolve_last {}; // reporter only
// n3 (resolver): native draws and their cycles by window path (window: every read binding the
// windows could see was answered by one; mixed: some; normal: none), the sums of the per-draw
// NativeN3Draw fields, and resolver segments whose draws after the first all took the window path.
constexpr uint32_t    kN3Paths               = 3;
constexpr const char* kN3PathNames[kN3Paths] = {"window", "mixed", "normal"};
constexpr uint32_t    kN3Fields              = 13;
constexpr const char* kN3FieldNames[kN3Fields] = {"hits", "learned", "stream", "dirty", "other", "ineligible", "epoch", "structure",
                                                  "unregister", "signature", "verify-checks", "verify-races", "verify-mismatches"};
struct N3Resolve {
	std::array<std::atomic<uint64_t>, kN3Paths>  draws {};
	std::array<std::atomic<uint64_t>, kN3Paths>  ticks {};
	std::array<std::atomic<uint64_t>, kN3Fields> fields {};
	std::atomic<uint64_t>                        segments {0};
	std::atomic<uint64_t>                        segments_window {0};
};
std::array<N3Resolve, kEngineThreads> g_n3_resolve {};
struct N3ResolveLast {
	std::array<uint64_t, kN3Paths>  draws {}, ticks {};
	std::array<uint64_t, kN3Fields> fields {};
	uint64_t                        segments = 0, segments_window = 0;
};
std::array<N3ResolveLast, kEngineThreads> g_n3_resolve_last {}; // reporter only
// S0 (executor, Thread_Gpu): learning outcomes per slot; (CP, real parse) segment lengths.
constexpr const char* kS0ReasonNames[GameCapture::kS0ReasonCount] = {"simple",    "flow", "no-record", "wrong-draw", "no-plan",
                                                                     "tess",      "indirect", "deep", "const-buffer", "table-full",
                                                                     "skipped"};
constexpr uint32_t kS0Fields = 18;
enum S0Field : uint32_t {
	kS0fDraws, kS0fFound, kS0fLearned, kS0fConflict, kS0fReplaced, kS0fVerified, kS0fCompared, kS0fMismatch, kS0fBadDraws,
	kS0fNulled, kS0fUd, kS0fUdShadowEq, kS0fUdShadowNe, kS0fUdShadowUnk, kS0fUdCopyEq, kS0fUdCopyNe, kS0fSegHeads, kS0fSegDraws
};
constexpr uint32_t kS0SegFields = 6; // heads, draws in heads, length ok, length bad, open at parse, orphan
struct S0Resolve {
	std::array<std::atomic<uint64_t>, GameCapture::kS0ReasonCount> reasons {};
	std::array<std::atomic<uint64_t>, kS0Fields>                   fields {};
	std::array<std::atomic<uint64_t>, kS0SegFields>                seg {}; // CP side
	std::array<std::atomic<uint64_t>, 5>                           seg_hist {}; // head counts 1, 2-4, 5-16, 17-64, 65+
	// kS0Deep: the IR opcodes that stopped a recipe (+1), a small open table (executor only writes).
	std::array<std::atomic<uint32_t>, 16> deep_op {};
	std::array<std::atomic<uint64_t>, 16> deep_n {};
};
std::array<S0Resolve, kEngineThreads> g_s0 {};
struct S0ResolveLast {
	std::array<uint64_t, GameCapture::kS0ReasonCount> reasons {};
	std::array<uint64_t, kS0Fields>                   fields {};
	std::array<uint64_t, kS0SegFields>                seg {};
	std::array<uint64_t, 5>                           seg_hist {};
};
std::array<S0ResolveLast, kEngineThreads> g_s0_last {}; // reporter only
std::atomic<uint32_t> g_s0_keys {0}, g_s0_base_keys {0};
// S0b: key conflicts by cause (NativeS0Conflict) and, for static-state causes, by word (0..63).
constexpr const char* kS0ConflictNames[GameCapture::kCfCount] = {"none",     "layout",    "vs-hash",  "vs-udc",   "vs-code",  "vs-static",
                                                                 "vs-push",  "vs-spec",   "ps-hash",  "ps-udc",   "ps-code",  "ps-static",
                                                                 "ps-push",  "ps-spec",   "plain",    "other"};
struct S0Conflicts {
	std::array<std::atomic<uint64_t>, GameCapture::kCfCount> cause {};
	std::array<std::atomic<uint64_t>, 64>                    vs_word {};
	std::array<std::atomic<uint64_t>, 64>                    ps_word {};
};
std::array<S0Conflicts, kEngineThreads> g_s0_conflicts {};
std::array<std::array<uint64_t, GameCapture::kCfCount>, kEngineThreads> g_s0_conflicts_last {};
// S1 (executor): native emission outcomes per slot.
constexpr const char* kS1ReasonNames[GameCapture::kS1Count] = {"emitted", "seg-start", "slot",  "no-seg", "activity", "command",
                                                               "regs",    "kind",      "record", "pixel", "miss",     "shape",
                                                               "read",    "spec",      "late",  "sample", "barriers"};
struct S1Resolve {
	std::array<std::atomic<uint64_t>, GameCapture::kS1Count> reasons {};
	std::atomic<uint64_t>                                    checked {0}, mismatched {0};
};
std::array<S1Resolve, kEngineThreads> g_s1 {};
struct S1ResolveLast {
	std::array<uint64_t, GameCapture::kS1Count> reasons {};
	uint64_t                                    checked = 0, mismatched = 0;
};
std::array<S1ResolveLast, kEngineThreads> g_s1_last {};
std::atomic<uint32_t> g_s1_logged {0};
std::atomic<uint32_t> g_s0_logged {0};
std::atomic<uint32_t> g_s0_seg_logged {0};
std::array<uint64_t, kEngineThreads + 1> g_n2_rec_ok_last {}, g_n2_rec_miss_last {}, g_n2_verified_last {},
    g_n2_verify_bad_last {};
std::array<uint64_t, GameCapture::kNvCount> g_n2_field_bad_last {}, g_n2_field_unknown_last {};
// TSC calibration of the report (reporter only): the pair read at install, and the cycles per
// microsecond measured over the whole run at each report.
uint64_t g_n2_tsc_last   = 0;
int64_t  g_n2_ns_last    = 0;
double   g_n2_tsc_per_us = 0.0;

// CP-side splice counters, per slot (+1: malformed marker).
enum NMiss : uint32_t {
	kMissMalformed, // magic / version / slot
	kMissNoRing,    // slot without a ring (mode is not splice)
	kMissSerial,    // entry overwritten, being rewritten, or never written
	kMissPacket,    // size / opcode of the record not those of the marker
	kMissCount
};
constexpr const char* kMissNames[kMissCount] = {"malformed", "no-ring", "serial", "packet"};
std::array<std::atomic<uint64_t>, kEngineThreads + 1>                           g_native_spliced {};
std::array<uint64_t, kEngineThreads + 1>                                        g_native_spliced_last {};
std::array<std::array<std::atomic<uint64_t>, kMissCount>, kEngineThreads + 1> g_native_miss {};
std::array<std::array<uint64_t, kMissCount>, kEngineThreads + 1>              g_native_miss_last {};

// Markers parsed by the command processor, per thread slot (+1: malformed).
std::array<std::atomic<uint64_t>, kEngineThreads + 1> g_native_seen {};
std::array<uint64_t, kEngineThreads + 1>              g_native_seen_last {}; // reporter only
uint64_t                                              g_flips_last = 0;      // reporter only

enum Scope : uint8_t { kScopeNone, kScopeDrawFlush, kScopeDispatchFlush, kScopeBegin, kScopeToolkit };

struct PendingDraw {
	uint64_t ctx = 0;
	uint32_t kind = 0, prim = 0, count = 0, instances = 0, first = 0, vertex_offset = 0;
	uint32_t indirect_id = 0, indirect_offset = 0;
};

constexpr size_t kSetSize = 4096; // open addressing, power of two

struct PtrSet {
	std::array<uint64_t, kSetSize> keys {};
	std::atomic<uint32_t>          size {0};
	void Insert(uint64_t key) {
		if (key == 0) {
			return;
		}
		auto h = (key * 0x9E3779B97F4A7C15ull) >> 52u; // 12 bits
		for (size_t probe = 0; probe < 64; probe++) {
			auto& slot = keys[(h + probe) & (kSetSize - 1)];
			if (slot == key) {
				return;
			}
			if (slot == 0) {
				slot = key;
				size.store(size.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
				return;
			}
		}
	}
};

struct alignas(64) ThreadState {
	char     name[32] {};
	uint32_t thread_id = 0;

	// Scope stack (flush / begin / toolkit), with the context each one works on.
	Scope    scope[8] {};
	uint64_t scope_ctx[8] {};
	uint32_t depth = 0;
	uint64_t flush_mask = 0; // dirty mask at the entry of the current draw flush

	bool        draw_pending = false; // a hooked draw entered, its flush not yet left
	PendingDraw draw;
	bool        emit_window = false; // between a hooked draw's flush exit and its packet

	// KYTY_NATIVE: engine thread slot when this thread is selected (-1 otherwise); the open
	// window means the next direct draw packet of emit_window is replaced by a marker.
	int         native_slot   = -1;
	bool        native_window = false;
	uint32_t    native_serial = 0;
	// n2: the record of the open native window, and the segment the next native draw continues
	// (broken by any other draw, dispatch flush, tool-library draw or list begin of this thread).
	GameCapture::NativeDraw native_rec;
	bool                    seg_broken = true;
	uint64_t                seg_ctx    = 0;
	uint64_t                seg_rt     = 0;
	// S0: serial of the open segment's head record (0: none) and its draws with a marker so far.
	uint32_t                seg_head_serial = 0;
	uint32_t                seg_len         = 0;
	std::array<std::atomic<uint64_t>, kNCount> n {};
	std::array<uint64_t, kNCount>              n_last {}; // reporter only

	// Packet sample of the last record (published by NoteDrawPacket).
	bool     sample_pending = false;
	uint64_t sample_words[7] {};
	uint64_t sample_counter = 0;

	ContextState* cache_state[4] {};
	uint32_t      cache_next = 0;

	std::array<std::atomic<uint64_t>, kCCount> c {};
	std::atomic<uint64_t> first_missing {0}; // OR of the groups not dirty at a list's first flush
	std::array<std::array<std::atomic<uint32_t>, 10>, 8> slot_type {};    // draws: slot -> type
	std::array<std::array<std::atomic<uint32_t>, 10>, 8> cs_slot_type {}; // dispatch flushes
	std::array<std::array<std::atomic<uint32_t>, 256>, 6> ptype_active {}; // pipeline type x active mask

	PtrSet shaders;
	PtrSet rtsets;

	std::array<DrawRecord, kRing> ring;
	uint64_t                      ring_seq = 0;

	std::array<uint64_t, kCCount> last {}; // reporter only
};

std::mutex                g_threads_mutex;
std::vector<ThreadState*> g_threads; // never freed
thread_local ThreadState* t_state = nullptr;

void Inc(std::atomic<uint64_t>& counter, uint64_t by = 1) {
	counter.store(counter.load(std::memory_order_relaxed) + by, std::memory_order_relaxed);
}
void Inc32(std::atomic<uint32_t>& counter) {
	counter.store(counter.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
}

ThreadState* T() {
	auto* state = t_state;
	if (state != nullptr) [[likely]] {
		return state;
	}
	state = new ThreadState;
	if (auto* self = Libs::LibKernel::PthreadSelfOrNull(); self != nullptr) {
		Libs::LibKernel::PthreadGetname(self, state->name);
	}
	if (state->name[0] == '\0') {
		std::strncpy(state->name, "?", sizeof(state->name) - 1);
	}
	state->thread_id = static_cast<uint32_t>(Libs::LibKernel::PthreadGetthreadid());
	if (const auto& native = Native(); native.valid) {
		const int slot = EngineThreadSlot(state->name);
		if (slot >= 0 && ((native.mask >> static_cast<uint32_t>(slot)) & 1u) != 0) {
			state->native_slot = slot;
			// Splice: one producer per ring; a second thread of the same name stays non-native.
			if (auto* ring = g_native_rings[slot]; ring != nullptr) {
				uint32_t expected = 0;
				if (!ring->owner.compare_exchange_strong(expected, state->thread_id + 1u)) {
					state->native_slot = -1;
					Log::WriteToConsoleAndLog(fmt::format("GameNative: thread \"{}\" ({}) not native: slot owned by thread {}\n",
					                                      state->name, state->thread_id, expected - 1u));
				}
			}
		}
	}
	{
		std::lock_guard lock(g_threads_mutex);
		g_threads.push_back(state);
	}
	t_state = state;
	return state;
}

// Closes the emit window (and a native window still open: its draw wrote no packet).
void ResetEmit(ThreadState* t) {
	t->emit_window = false;
	if (t->native_window) [[unlikely]] {
		t->native_window = false;
		t->seg_broken    = true;
		Inc(t->n[kNFbLost]);
	}
}

ContextState* StateOf(ThreadState* t, uint64_t ctx) {
	for (auto* state: t->cache_state) {
		if (state != nullptr && state->ctx == ctx) {
			return state;
		}
	}
	auto* state = FindOrAddContext(ctx);
	if (state != nullptr) {
		t->cache_state[t->cache_next++ & 3u] = state;
	}
	return state;
}

void PushScope(ThreadState* t, Scope scope, uint64_t ctx) {
	if (t->depth < 8) {
		t->scope[t->depth]     = scope;
		t->scope_ctx[t->depth] = ctx;
	}
	t->depth++;
}
void PopScope(ThreadState* t) {
	if (t->depth > 0) {
		t->depth--;
	}
}
Scope TopScope(const ThreadState* t, uint64_t* ctx) {
	if (t->depth == 0 || t->depth > 8) {
		*ctx = 0;
		return kScopeNone;
	}
	*ctx = t->scope_ctx[t->depth - 1];
	return t->scope[t->depth - 1];
}

// ---- Packet samples (game side -> command processor) -------------------------------------------------

constexpr size_t kSamples = 4096;
struct SampleSlot {
	std::atomic<uint64_t>                packet {0};
	std::array<std::atomic<uint64_t>, 7> w {};
};
std::array<SampleSlot, kSamples> g_samples;

size_t SampleIndex(uint64_t packet) {
	return static_cast<size_t>(((packet >> 2u) * 0x9E3779B97F4A7C15ull) >> 52u) & (kSamples - 1);
}

// Sample words: 0 count | first << 32; 1 instances | kind << 32 | ib32 << 40 | rt0 known << 41 |
// has ps << 42 | geom slot << 48; 2 ib address; 3 rt0 base; 4 ps code; 5 geometry code; 6 ms.
enum CmpField : uint32_t { kCmpOpcode, kCmpCount, kCmpFirst, kCmpInstances, kCmpIb, kCmpIbType, kCmpRt0, kCmpPs, kCmpGeom, kCmpFields };
constexpr const char* kCmpNames[] = {"opcode", "count", "first", "instances", "ib", "ib-type", "rt0", "ps", "geom"};

struct CpStats {
	std::atomic<uint64_t> draws {0}, published {0}, matched {0}, torn {0}, speculative {0};
	std::array<std::atomic<uint64_t>, kCmpFields> compared {};
	std::array<std::atomic<uint64_t>, kCmpFields> mismatched {};
	std::atomic<uint32_t> logged {0};
};
CpStats g_cp;
std::array<uint64_t, 5 + 2 * kCmpFields> g_cp_last {}; // reporter only

std::atomic<int64_t> g_next_report {0};
std::mutex           g_report_mutex;
int64_t              g_last_report = 0;

int64_t NowMs() {
	return std::chrono::duration_cast<std::chrono::milliseconds>(
	           std::chrono::steady_clock::now().time_since_epoch())
	    .count();
}

// ---- Accumulators ----------------------------------------------------------------------------------------

template <uint32_t kBank> // 0 cx, 1 sh, 2 uc
void OnAccumulator(GuestCallContext& ctx) {
	auto*          t     = T();
	const uint64_t acc   = ctx.rdi;
	const uint64_t owner = acc - kCtxAcc;
	if (acc == 0 || Rd<uint64_t>(acc, 0) != owner + kCtxCb) {
		Inc(t->c[uint32_t {kCAcc0} + kAccForeign]);
		return;
	}
	uint64_t scope_ctx = 0;
	const auto scope   = TopScope(t, &scope_ctx);
	AccClass   cls     = kAccOutside;
	if (scope != kScopeNone && scope_ctx == owner) {
		cls = scope == kScopeBegin ? kAccBegin : scope == kScopeToolkit ? kAccToolkit : kAccFlush;
	} else if (t->draw_pending && t->draw.ctx == owner) {
		cls = kAccInDraw;
	}
	Inc(t->c[uint32_t {kCAcc0} + cls]);
	if constexpr (kBank == 2) {
		return;
	} else {
		auto* state = StateOf(t, owner);
		if (state == nullptr) {
			return;
		}
		const auto* pairs = reinterpret_cast<const uint32_t*>(ctx.rsi);
		const auto  count = static_cast<uint32_t>(ctx.rdx);
		if (pairs == nullptr) {
			return;
		}
		const bool     outside = (cls == kAccOutside || cls == kAccToolkit);
		const uint32_t limit   = kBank == 0 ? kCxNum : kShNum;
		for (uint32_t i = 0; i < count; i++) {
			const uint32_t raw = pairs[2 * i];
			if (raw == 0xffffffffu) {
				continue;
			}
			const uint32_t reg = raw & ~kRegSelectorMask;
			if (reg >= limit) {
				continue;
			}
			uint32_t* values = kBank == 0 ? state->cx : state->sh;
			uint32_t* epochs = kBank == 0 ? state->cx_epoch : state->sh_epoch;
			uint8_t*  flags  = kBank == 0 ? state->cx_flags : state->sh_flags;
			values[reg]      = pairs[2 * i + 1];
			epochs[reg]      = state->epoch + 1;
			if (outside) {
				if ((flags[reg] & kFlagOutside) == 0) {
					flags[reg] |= kFlagOutside;
					if (state->pending_count < kPendingMax) {
						state->pending[state->pending_count++] = static_cast<uint16_t>((kBank << 12u) | reg);
					} else {
						state->pending_overflow = true;
					}
				}
				if constexpr (kBank == 0) {
					g_outside_cx[reg].fetch_add(1, std::memory_order_relaxed);
				} else {
					g_outside_sh[reg].fetch_add(1, std::memory_order_relaxed);
				}
			} else {
				flags[reg] = static_cast<uint8_t>(cls == kAccBegin ? kFlagBegin : 0);
				if constexpr (kBank == 0) {
					if (cls == kAccBegin) {
						g_begin_cx[reg].fetch_add(1, std::memory_order_relaxed);
					}
				}
			}
		}
	}
}

// Registers written outside and still not re-emitted; rebuilds the pending list.
uint32_t StaleOutside(ContextState* state) {
	if (state->pending_overflow) {
		state->pending_count    = 0;
		state->pending_overflow = false;
		uint32_t stale          = 0;
		for (uint32_t bank = 0; bank < 2; bank++) {
			const uint32_t limit = bank == 0 ? kCxNum : kShNum;
			const uint8_t* flags = bank == 0 ? state->cx_flags : state->sh_flags;
			for (uint32_t reg = 0; reg < limit; reg++) {
				if ((flags[reg] & kFlagOutside) != 0) {
					stale++;
					if (state->pending_count < kPendingMax) {
						state->pending[state->pending_count++] = static_cast<uint16_t>((bank << 12u) | reg);
					} else {
						state->pending_overflow = true;
					}
				}
			}
		}
		return stale;
	}
	uint32_t kept = 0;
	for (uint32_t i = 0; i < state->pending_count; i++) {
		const auto entry = state->pending[i];
		const auto bank  = entry >> 12u;
		const auto reg   = entry & 0xfffu;
		const auto flag  = bank == 0 ? state->cx_flags[reg] : state->sh_flags[reg];
		if ((flag & kFlagOutside) != 0) {
			state->pending[kept++] = entry;
		}
	}
	state->pending_count = kept;
	return kept;
}

// ---- Draw entry points (Before) ------------------------------------------------------------------------------

void BeginDraw(GuestCallContext& ctx, DrawKind kind) {
	auto* t         = T();
	ResetEmit(t);
	t->sample_pending = false;
	t->draw_pending = true;
	auto& d         = t->draw;
	d               = PendingDraw {};
	d.ctx           = ctx.rdi;
	d.kind          = kind;
	d.prim          = static_cast<uint32_t>(ctx.rsi);
	switch (kind) {
		case kKindIndexed:
		case kKindIndexedOfs:
			d.count         = static_cast<uint32_t>(ctx.rdx);
			d.instances     = static_cast<uint32_t>(ctx.rcx);
			d.first         = static_cast<uint32_t>(ctx.r8);
			d.vertex_offset = static_cast<uint32_t>(ctx.r9);
			break;
		case kKindAuto:
			d.count         = static_cast<uint32_t>(ctx.rdx);
			d.instances     = 1;
			d.vertex_offset = static_cast<uint32_t>(ctx.rcx);
			break;
		case kKindAutoOfs:
			d.count         = static_cast<uint32_t>(ctx.rdx);
			d.instances     = static_cast<uint32_t>(ctx.rcx);
			d.vertex_offset = static_cast<uint32_t>(ctx.r8);
			break;
		case kKindIndirect:
			d.indirect_id     = static_cast<uint32_t>(ctx.rdx) & 0xffffu;
			d.indirect_offset = static_cast<uint32_t>(ctx.rcx);
			break;
		default: break;
	}
	Inc(t->c[uint32_t {kCDraw0} + kind]);
}

void OnDrawIndexed(GuestCallContext& ctx) { BeginDraw(ctx, kKindIndexed); }
void OnDrawIndexedOfs(GuestCallContext& ctx) { BeginDraw(ctx, kKindIndexedOfs); }
void OnDrawAuto(GuestCallContext& ctx) { BeginDraw(ctx, kKindAuto); }
void OnDrawAutoOfs(GuestCallContext& ctx) { BeginDraw(ctx, kKindAutoOfs); }
void OnDrawIndirect(GuestCallContext& ctx) { BeginDraw(ctx, kKindIndirect); }

// ---- List begin, tool library ----------------------------------------------------------------------------------

void OnListBeginEnter(GuestCallContext& ctx) {
	auto* t        = T();
	ResetEmit(t);
	t->seg_broken  = true;
	if (auto* state = StateOf(t, ctx.rdi); state != nullptr) {
		state->epoch++;
		state->first_pending = true;
	}
	PushScope(t, kScopeBegin, ctx.rdi);
}
void OnListBeginExit(GuestCallContext& /*ctx*/) {
	PopScope(T());
}

void OnToolkitEnter(GuestCallContext& ctx) {
	auto* t        = T();
	ResetEmit(t);
	t->seg_broken  = true;
	// cb = ctx + 0xe50 and acc = ctx + 0x1208 when called on a backend context.
	const uint64_t owner = (ctx.rsi == ctx.rdi + (kCtxAcc - kCtxCb)) ? ctx.rdi - kCtxCb : 0;
	Inc(t->c[kCToolkit]);
	PushScope(t, kScopeToolkit, owner);
}
void OnToolkitExit(GuestCallContext& /*ctx*/) {
	PopScope(T());
}

// ---- Dispatch flush ----------------------------------------------------------------------------------------------

void OnDispatchFlushEnter(GuestCallContext& ctx) {
	auto*          t   = T();
	const uint64_t cxp = ctx.rdi;
	ResetEmit(t);
	t->seg_broken      = true;
	Inc(t->c[kCDispatchFlush]);
	if (auto* state = StateOf(t, cxp); state != nullptr && state->first_pending) {
		Inc(t->c[kCFirstDispatch]);
	}
	uint8_t active = 0;
	for (uint32_t s = 0; s < 8; s++) {
		const uint64_t slot = cxp + kCtxSlots + s * kSlotStride;
		if (Rd<uint8_t>(slot, kSlotActive) == 0) {
			continue;
		}
		active |= static_cast<uint8_t>(1u << s);
		const auto shader = Rd<uint64_t>(slot, kSlotShader);
		const auto type   = shader != 0 ? std::min<uint32_t>(Rd<uint8_t>(shader, kShaderType), 9u) : 9u;
		Inc32(t->cs_slot_type[s][type]);
	}
	const auto ptype = static_cast<uint32_t>(Rd<int32_t>(cxp, kCtxPipelineType));
	Inc32(t->ptype_active[std::min<uint32_t>(ptype, 5u)][active]);
	PushScope(t, kScopeDispatchFlush, cxp);
}
void OnDispatchFlushExit(GuestCallContext& /*ctx*/) {
	PopScope(T());
}

// ---- Draw flush: the record --------------------------------------------------------------------------------------

void Report(int64_t now);

void OnDrawFlushEnter(GuestCallContext& ctx) {
	auto*          t   = T();
	const uint64_t cxp = ctx.rdi;
	ResetEmit(t);
	t->flush_mask      = Rd<uint64_t>(cxp, kCtxDirty);
	Inc(t->c[kCFlush]);
	if (auto* state = StateOf(t, cxp); state != nullptr && state->first_pending) {
		state->first_pending = false;
		Inc(t->c[kCFirstLists]);
		if ((t->flush_mask & kAllGroups) == kAllGroups) {
			Inc(t->c[kCFirstFull]);
		} else {
			t->first_missing.fetch_or(~t->flush_mask & kAllGroups, std::memory_order_relaxed);
		}
	}
	PushScope(t, kScopeDrawFlush, cxp);
}

uint8_t FirstGeometrySlot(uint64_t cxp) {
	// The draws' own rule (0x483c20): 5, else 7, else 3, else 4, else 6 or 2.
	const auto active = [&](uint32_t s) { return Rd<uint8_t>(cxp + kCtxSlots + s * kSlotStride, kSlotActive) != 0; };
	if (active(5)) return 5;
	if (active(7)) return 7;
	if (active(3)) return 3;
	if (active(4)) return 4;
	return active(6) ? 6 : 2;
}

void CopySgprs(DrawRecord& r, int index, uint64_t cxp, uint32_t s) {
	r.sgpr_slot[index] = static_cast<int8_t>(s);
	const auto sgpr    = Rd<uint64_t>(cxp + kCtxSlots + s * kSlotStride, kSlotSgpr);
	if (sgpr != 0) {
		std::memcpy(r.sgpr[index], reinterpret_cast<const void*>(sgpr), sizeof(r.sgpr[index]));
	} else {
		std::memset(r.sgpr[index], 0, sizeof(r.sgpr[index]));
	}
}

// KYTY_NATIVE_MODE=n2: the compact record of a native draw, from the capture record (context
// fields) and the context's register shadow (what the game emitted since the list began). Pure
// reads of the game's own memory; no emulator state.
uint64_t Fnv(uint64_t hash, const void* data, size_t size) {
	const auto* bytes = static_cast<const uint8_t*>(data);
	for (size_t i = 0; i < size; i++) {
		hash = (hash ^ bytes[i]) * 0x100000001b3ull;
	}
	return hash;
}

void BuildNativeDraw(ThreadState* t, const DrawRecord& r, const ContextState* state) {
	namespace Pm4 = Libs::Graphics::Pm4;
	auto& n       = t->native_rec;
	n             = GameCapture::NativeDraw {};
	n.kind        = static_cast<uint8_t>(r.kind);
	n.geom_slot   = r.geom_slot;
	n.active      = r.active;
	n.fail        = r.fail;
	n.prim        = r.prim;
	n.count       = r.count;
	n.instances   = r.instances;
	n.first       = r.first;
	n.vertex_offset = r.vertex_offset;
	n.ib_count    = r.ib_count;
	n.ib_32       = r.ib_32;
	n.ib_addr     = r.ib_addr;
	n.pipeline    = r.pipeline;
	n.code_geom   = r.code[r.geom_slot];
	n.code_ps     = (r.active & 2u) != 0 ? r.code[1] : 0;
	n.rt_hash     = r.rt_hash;
	// The engine's user SGPR copies of the geometry and pixel slots (CopySgprs, after the flush).
	if (r.sgpr_slot[0] >= 0) std::memcpy(n.sgpr_geom, r.sgpr[0], sizeof(n.sgpr_geom));
	if (r.sgpr_slot[1] >= 0) std::memcpy(n.sgpr_ps, r.sgpr[1], sizeof(n.sgpr_ps));
	uint32_t known = 0;
	if (state != nullptr) {
		const auto cx = [&](uint32_t reg, uint32_t bit, uint32_t& out) {
			out = state->cx[reg];
			if (state->CxKnown(reg)) known |= bit;
		};
		const auto pgm = [&](uint32_t lo, uint32_t bit, uint64_t& out) {
			out = (static_cast<uint64_t>(state->sh[lo]) << 8u) | ((static_cast<uint64_t>(state->sh[lo + 1]) & 0xffu) << 40u);
			if (state->ShKnown(lo) && state->ShKnown(lo + 1)) known |= bit;
		};
		pgm(Pm4::SPI_SHADER_PGM_LO_ES, GameCapture::kNkPgmEs, n.pgm_es);
		pgm(Pm4::SPI_SHADER_PGM_LO_GS, GameCapture::kNkPgmGs, n.pgm_gs);
		pgm(Pm4::SPI_SHADER_PGM_LO_PS, GameCapture::kNkPgmPs, n.pgm_ps);
		for (uint32_t i = 0; i < 32; i++) {
			n.user_gs[i] = state->sh[Pm4::SPI_SHADER_USER_DATA_GS_0 + i];
			n.user_ps[i] = state->sh[Pm4::SPI_SHADER_USER_DATA_PS_0 + i];
			if (state->ShKnown(Pm4::SPI_SHADER_USER_DATA_GS_0 + i)) n.user_gs_known |= 1u << i;
			if (state->ShKnown(Pm4::SPI_SHADER_USER_DATA_PS_0 + i)) n.user_ps_known |= 1u << i;
		}
		cx(Pm4::DB_RENDER_CONTROL, GameCapture::kNkRenderCtl, n.db_render_control);
		cx(Pm4::DB_DEPTH_CONTROL, GameCapture::kNkDepthCtl, n.db_depth_control);
		cx(Pm4::DB_STENCIL_CONTROL, GameCapture::kNkStencilCtl, n.db_stencil_control);
		cx(Pm4::PA_SU_SC_MODE_CNTL, GameCapture::kNkModeCtl, n.pa_su_sc_mode_cntl);
		cx(Pm4::PA_CL_CLIP_CNTL, GameCapture::kNkClipCtl, n.pa_cl_clip_cntl);
		cx(Pm4::CB_COLOR_CONTROL, GameCapture::kNkColorCtl, n.cb_color_control);
		cx(Pm4::CB_TARGET_MASK, GameCapture::kNkTargetMask, n.cb_target_mask);
		for (uint32_t k = 0; k < 8; k++) {
			cx(Pm4::CB_BLEND0_CONTROL + k, GameCapture::kNkBlend0 << k, n.cb_blend[k]);
		}
		uint32_t all = GameCapture::kNkPolyOffset;
		for (uint32_t k = 0; k < 6; k++) {
			uint32_t bit = 0;
			cx(Pm4::PA_SU_POLY_OFFSET_DB_FMT_CNTL + k, 0, n.poly_offset[k]);
			bit = state->CxKnown(Pm4::PA_SU_POLY_OFFSET_DB_FMT_CNTL + k) ? all : 0;
			all &= bit;
		}
		known |= all;
		uint32_t vp = GameCapture::kNkViewport;
		for (uint32_t k = 0; k < 6; k++) {
			n.vport[k] = state->cx[Pm4::PA_CL_VPORT_XSCALE + k];
			if (!state->CxKnown(Pm4::PA_CL_VPORT_XSCALE + k)) vp = 0;
		}
		known |= vp;
		n.vport_scissor[0]  = state->cx[Pm4::PA_SC_VPORT_SCISSOR_0_TL];
		n.vport_scissor[1]  = state->cx[Pm4::PA_SC_VPORT_SCISSOR_0_TL + 1];
		if (state->CxKnown(Pm4::PA_SC_VPORT_SCISSOR_0_TL) && state->CxKnown(Pm4::PA_SC_VPORT_SCISSOR_0_TL + 1)) {
			known |= GameCapture::kNkScissor;
		}
		n.screen_scissor[0] = state->cx[Pm4::PA_SC_SCREEN_SCISSOR_TL];
		n.screen_scissor[1] = state->cx[Pm4::PA_SC_SCREEN_SCISSOR_TL + 1];
		n.db_stencilrefmask    = state->cx[Pm4::DB_STENCILREFMASK];
		n.db_stencilrefmask_bf = state->cx[Pm4::DB_STENCILREFMASK_BF];
		if (state->CxKnown(Pm4::DB_STENCILREFMASK) && state->CxKnown(Pm4::DB_STENCILREFMASK_BF)) {
			known |= GameCapture::kNkStencilRef;
		}
	}
	n.known = known;
	// The fixed-function part every native draw needs (programs of the active stages, depth,
	// raster, clip): a record without it cannot stand for the draw on its own.
	constexpr uint32_t kNeeded = GameCapture::kNkDepthCtl | GameCapture::kNkModeCtl | GameCapture::kNkClipCtl |
	                             GameCapture::kNkPolyOffset | GameCapture::kNkViewport | GameCapture::kNkScissor;
	if ((known & kNeeded) != kNeeded || (n.code_ps != 0 && (known & GameCapture::kNkPgmPs) == 0)) {
		Inc(t->n[kNRecIncomplete]);
	}
	uint64_t h = 0xcbf29ce484222325ull;
	h          = Fnv(h, &n.pgm_es, sizeof(uint64_t) * 3);
	h          = Fnv(h, &n.db_render_control, sizeof(uint32_t) * 9);
	h          = Fnv(h, n.cb_blend, sizeof(n.cb_blend));
	h          = Fnv(h, n.poly_offset, sizeof(n.poly_offset));
	h          = Fnv(h, &n.prim, sizeof(n.prim));
	n.state_hash = h;
	n.memo_key   = (n.pipeline * 0x9E3779B97F4A7C15ull) ^ h;
	// Segment: same context and render-target set as the previous native draw, nothing between.
	const bool start = t->seg_broken || t->seg_ctx != r.ctx || t->seg_rt != r.rt_hash;
	n.seg_start      = start ? 1 : 0;
	t->seg_broken    = false; // a fallback at the packet breaks it again (WriteNativeMarker)
	t->seg_ctx       = r.ctx;
	t->seg_rt        = r.rt_hash;
	if (start) {
		Inc(t->n[kNSegments]);
	}
}

void OnDrawFlushExit(GuestCallContext& /*ctx*/) {
	auto*    t   = T();
	uint64_t cxp = 0;
	if (TopScope(t, &cxp) != kScopeDrawFlush) {
		PopScope(t);
		return;
	}
	PopScope(t);
	const bool hooked = t->draw_pending && t->draw.ctx == cxp;
	t->draw_pending   = false;
	if (!hooked) {
		t->seg_broken = true;
		Inc(t->c[kCFlushNoDraw]);
		return;
	}
	auto* state = StateOf(t, cxp);

	auto& r = t->ring[t->ring_seq % kRing];
	r.seq.store(0, std::memory_order_relaxed);
	std::atomic_thread_fence(std::memory_order_release);

	const auto& d   = t->draw;
	r.ctx           = cxp;
	r.kind          = d.kind;
	r.prim          = d.prim;
	r.count         = d.count;
	r.instances     = d.instances;
	r.first         = d.first;
	r.vertex_offset = d.vertex_offset;
	r.indirect_id   = d.indirect_id;
	r.indirect_offset = d.indirect_offset;
	r.dirty         = t->flush_mask;
	r.pipeline      = Rd<uint64_t>(cxp, kCtxPipelineObj);
	r.pipeline_type = static_cast<uint32_t>(Rd<int32_t>(cxp, kCtxPipelineType));
	r.geom_slot     = FirstGeometrySlot(cxp);

	uint32_t fail = 0;
	uint8_t  active = 0;
	for (uint32_t s = 0; s < 8; s++) {
		const uint64_t slot = cxp + kCtxSlots + s * kSlotStride;
		r.shader[s]         = Rd<uint64_t>(slot, kSlotShader);
		r.code[s]           = 0;
		r.type[s]           = 0xff;
		if (Rd<uint8_t>(slot, kSlotActive) == 0) {
			continue;
		}
		active |= static_cast<uint8_t>(1u << s);
		if (r.shader[s] == 0) {
			if (s != 0) fail |= kFailStage;
			Inc32(t->slot_type[s][9]);
			continue;
		}
		r.code[s] = Rd<uint64_t>(r.shader[s], kShaderCode);
		r.type[s] = Rd<uint8_t>(r.shader[s], kShaderType);
		Inc32(t->slot_type[s][std::min<uint32_t>(r.type[s], 9u)]);
		if (s != 0 && r.type[s] != s) {
			fail |= kFailStage;
		}
		t->shaders.Insert(r.shader[s]);
	}
	r.active = active;
	if ((active & (1u << r.geom_slot)) == 0 || r.shader[r.geom_slot] == 0) {
		fail |= kFailStage;
	}
	if (r.pipeline == 0 || r.pipeline_type < 1 || r.pipeline_type > 4) {
		fail |= kFailPipeline;
	}
	Inc32(t->ptype_active[std::min<uint32_t>(r.pipeline_type, 5u)][active]);

	// User SGPR copies (after the flush wrote this draw's descriptors into them).
	CopySgprs(r, 0, cxp, r.geom_slot);
	r.sgpr_slot[1] = -1;
	r.sgpr_slot[2] = -1;
	if ((active & 2u) != 0) {
		CopySgprs(r, 1, cxp, 1);
	}
	if ((r.geom_slot & 1u) != 0) { // tessellation: also the GS-family stage
		for (uint32_t s: {4u, 6u, 2u}) {
			if ((active & (1u << s)) != 0) {
				CopySgprs(r, 2, cxp, s);
				break;
			}
		}
	}

	// Targets: objects from the context, registers from the shadow.
	bool     any_target  = false;
	bool     rt_resolved = true;
	uint64_t rt_hash     = 0xcbf29ce484222325ull;
	for (uint32_t n = 0; n < 8; n++) {
		const uint64_t slot = cxp + kCtxRt + n * kRtStride;
		auto&          rt   = r.rt[n];
		rt.id               = Rd<uint16_t>(slot, 0);
		rt.view             = Rd<uint8_t>(slot, 4);
		rt.obj              = Rd<uint64_t>(slot, 8);
		rt_hash             = (rt_hash ^ rt.obj) * 0x100000001b3ull;
		rt.known            = 0;
		rt.base = 0;
		rt.info = 0;
		rt.attrib2 = 0;
		if (rt.obj == 0) {
			continue;
		}
		any_target = true;
		if (state != nullptr) {
			const uint32_t base_reg = kCbColor0Base + 15 * n;
			rt.known   = state->CxKnown(base_reg) ? 1 : 0;
			rt.base    = (static_cast<uint64_t>(state->cx[base_reg]) << 8u) |
			             ((static_cast<uint64_t>(state->cx[kCbColor0BaseExt + n]) & 0xffu) << 40u);
			rt.info    = state->cx[kCbColor0Info + 15 * n];
			rt.attrib2 = state->cx[kCbColor0Attrib2 + n];
		}
		if (rt.known == 0) {
			rt_resolved = false;
		}
	}
	r.depth_obj = Rd<uint64_t>(cxp, kCtxDepthObj);
	rt_hash     = (rt_hash ^ r.depth_obj) * 0x100000001b3ull;
	if (r.depth_obj != 0) {
		any_target = true;
		if (state == nullptr || !state->CxKnown(kDbZReadBase)) {
			rt_resolved = false;
		}
	}
	if (!any_target) {
		fail |= kFailTarget;
	}
	t->rtsets.Insert(rt_hash == 0 ? 1 : rt_hash);
	r.rt_hash = rt_hash;
	if (state != nullptr) {
		r.z_base = (static_cast<uint64_t>(state->cx[kDbZReadBase]) << 8u) |
		           ((static_cast<uint64_t>(state->cx[kDbZReadBaseHi]) & 0xffu) << 40u);
		r.z_info         = state->cx[kDbZInfo];
		r.depth_size_xy  = state->cx[kDbDepthSizeXy];
		r.cb_target_mask = state->cx[kCbTargetMask];
		for (uint32_t i = 0; i < 6; i++) r.vport[i] = state->cx[kPaClVportXscale + i];
		r.vport_scissor[0]  = state->cx[kPaScVportScissor0];
		r.vport_scissor[1]  = state->cx[kPaScVportScissor0 + 1];
		r.screen_scissor[0] = state->cx[kPaScScreenScissor];
		r.screen_scissor[1] = state->cx[kPaScScreenScissor + 1];
	}

	// Index / vertex buffers, fixed state (context fields).
	r.ib_count = Rd<uint32_t>(cxp, kCtxIbCount);
	r.ib_32    = Rd<uint32_t>(cxp, kCtxIb32);
	r.ib_addr  = Rd<uint64_t>(cxp, kCtxIbAddr);
	r.vb_mode  = Rd<uint32_t>(cxp, kCtxVbMode);
	r.vb_table = Rd<uint64_t>(cxp + kCtxSlots + r.geom_slot * kSlotStride, kSlotVtxTable);
	if (d.kind == kKindIndexed || d.kind == kKindIndexedOfs) {
		if (r.ib_addr == 0 || static_cast<uint64_t>(d.first) + d.count > r.ib_count) {
			fail |= kFailIndex;
		}
	} else if (d.kind == kKindIndirect && r.ib_addr == 0) {
		fail |= kFailIndex;
	}
	r.depth_control  = Rd<uint32_t>(cxp, kCtxDepthControl);
	r.stencil_ref    = Rd<uint32_t>(cxp, kCtxStencilRef);
	r.stencil_ref_bf = Rd<uint32_t>(cxp, kCtxStencilRefBf);
	r.stencil_control = Rd<uint32_t>(cxp, kCtxStencilCtl);
	r.render_control = Rd<uint32_t>(cxp, kCtxRenderCtl);
	r.su_sc_mode     = Rd<uint32_t>(cxp, kCtxSuScMode);
	r.alpha_to_mask  = Rd<uint32_t>(cxp, kCtxAlphaToMask);
	r.blend[0]       = Rd<uint32_t>(cxp, kCtxBlendRt0);
	for (uint32_t k = 1; k < 8; k++) r.blend[k] = Rd<uint32_t>(cxp, kCtxBlendRt1 + 0x18 * (k - 1));
	for (uint32_t k = 0; k < 4; k++) r.blend_color[k] = Rd<uint32_t>(cxp, kCtxBlendColor + 4 * k);
	r.prim = Rd<uint32_t>(cxp, kCtxPrim);

	// Context consistency: stale outside writes, list begin seen.
	uint32_t stale = 0;
	if (state != nullptr) {
		stale = StaleOutside(state);
		if (state->epoch == 0) {
			fail |= kFailNoBegin;
			Inc(t->c[kCNoBeginDraws]);
		}
	} else {
		fail |= kFailNoBegin;
	}
	if (stale != 0) {
		fail |= kFailOutside;
	}
	r.stale_outside = static_cast<uint8_t>(std::min<uint32_t>(stale, 255u));
	r.fail          = static_cast<uint8_t>(fail);

	// Self check: does the shadow's program address equal the Agc shader object's code?
	if (state != nullptr && (active & 2u) != 0 && r.shader[1] != 0 && state->ShKnown(kSpiPgmLoPs)) {
		Inc(t->c[(static_cast<uint64_t>(state->sh[kSpiPgmLoPs]) << 8u) == (r.code[1] & 0xffffffffffull)
		             ? kCShaderSelfOk
		             : kCShaderSelfBad]);
	}
	if (state != nullptr && r.shader[r.geom_slot] != 0) {
		const uint32_t lo = (r.geom_slot & 1u) != 0 ? kSpiPgmLoHs : kSpiPgmLoGs;
		if (state->ShKnown(lo)) {
			Inc(t->c[(static_cast<uint64_t>(state->sh[lo]) << 8u) == (r.code[r.geom_slot] & 0xffffffffffull)
			             ? kCShaderSelfOk
			             : kCShaderSelfBad]);
		}
	}

	// Counters.
	Inc(t->c[kCRecords]);
	if (fail == 0) {
		Inc(t->c[kCComplete]);
	}
	for (uint32_t b = 0; b < 6; b++) {
		if ((fail & (1u << b)) != 0) Inc(t->c[uint32_t {kCFail0} + b]);
	}
	if (rt_resolved) {
		Inc(t->c[kCRtResolved]);
	}
	Inc(t->c[kCDirtyBits], static_cast<uint64_t>(std::popcount(r.dirty & kAllGroups)));
	for (uint32_t f = 0; f < 16; f++) {
		if ((r.dirty & kFamilies[f].mask) != 0) Inc(t->c[uint32_t {kCFam0} + f]);
	}

	t->ring_seq++;
	r.seq.store(t->ring_seq, std::memory_order_release);

	// KYTY_NATIVE: indexed / auto draws of a selected thread, without tessellation, are native.
	if (t->native_slot >= 0) [[unlikely]] {
		if (d.kind == kKindIndirect || d.kind > kKindAutoOfs) {
			t->seg_broken = true;
			Inc(t->n[kNFbIndirect]);
		} else if ((r.geom_slot & 1u) != 0) {
			t->seg_broken = true;
			Inc(t->n[kNFbTess]);
		} else {
			t->native_window = true;
			Inc(t->n[kNChosen]);
			if (fail != 0) {
				Inc(t->n[kNIncomplete]);
			}
			if (g_native_records[t->native_slot] != nullptr) {
				BuildNativeDraw(t, r, state);
			}
		}
	}

	// One draw in 1000: published under its packet address when libAgc writes the packet.
	t->emit_window = true;
	if (++t->sample_counter % 1000 == 0) {
		t->sample_pending = true;
		const bool rt0_known = r.rt[0].obj != 0 && r.rt[0].known != 0;
		const bool has_ps    = (active & 2u) != 0 && r.code[1] != 0;
		t->sample_words[0]   = d.count | (static_cast<uint64_t>(d.first) << 32u);
		t->sample_words[1]   = d.instances | (static_cast<uint64_t>(d.kind) << 32u) |
		                     (static_cast<uint64_t>(r.ib_32 != 0) << 40u) |
		                     (static_cast<uint64_t>(rt0_known) << 41u) |
		                     (static_cast<uint64_t>(has_ps) << 42u) |
		                     (static_cast<uint64_t>(r.geom_slot) << 48u);
		t->sample_words[2] = r.ib_addr;
		t->sample_words[3] = r.rt[0].base;
		t->sample_words[4] = has_ps ? r.code[1] : 0;
		t->sample_words[5] = r.code[r.geom_slot];
		t->sample_words[6] = static_cast<uint64_t>(NowMs());
	}

	if ((t->ring_seq & 63u) == 0) {
		const auto now  = NowMs();
		auto       next = g_next_report.load(std::memory_order_relaxed);
		if (next == 0) {
			g_next_report.compare_exchange_strong(next, now + 10000);
		} else if (now >= next && g_next_report.compare_exchange_strong(next, now + 10000)) {
			Report(now);
		}
	}
}

// ---- Report -------------------------------------------------------------------------------------------------------

// KYTY_NATIVE_MODE=n2 report parts (reporter only).
double N2TicksPerUs() {
	const uint64_t tsc = __rdtsc();
	const int64_t  ns  = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
	double         per = 0.0;
	if (g_n2_ns_last != 0 && ns > g_n2_ns_last) {
		per = static_cast<double>(tsc - g_n2_tsc_last) * 1000.0 / static_cast<double>(ns - g_n2_ns_last);
	}
	return per;
}

// KYTY_NATIVE_MODE=n3 part of a slot (reporter only).
std::string N3SlotReport(uint32_t s) {
	auto&                          r    = g_n3_resolve[s];
	auto&                          last = g_n3_resolve_last[s];
	const auto                     take = [](std::atomic<uint64_t>& v, uint64_t& l) {
		const auto now = v.load(std::memory_order_relaxed);
		const auto d   = now - l;
		l              = now;
		return d;
	};
	std::array<uint64_t, kN3Paths>  draws {}, ticks {};
	std::array<uint64_t, kN3Fields> fields {};
	uint64_t                        all = 0;
	for (uint32_t k = 0; k < kN3Paths; k++) {
		draws[k] = take(r.draws[k], last.draws[k]);
		ticks[k] = take(r.ticks[k], last.ticks[k]);
		all += draws[k];
	}
	for (uint32_t k = 0; k < kN3Fields; k++) fields[k] = take(r.fields[k], last.fields[k]);
	const auto   segments        = take(r.segments, last.segments);
	const auto   segments_window = take(r.segments_window, last.segments_window);
	const double tpu             = g_n2_tsc_per_us;
	const auto   us              = [&](uint64_t t, uint64_t n) {
        return tpu <= 0.0 || n == 0 ? 0.0 : static_cast<double>(t) / tpu / static_cast<double>(n);
	};
	std::string out = fmt::format(" | emission: emulator path (splice) {} with n3 resolution windows (stage N3a; pre-encoded segments: not yet)", all);
	out += " | n3 draws by path:";
	for (uint32_t k = 0; k < kN3Paths; k++) {
		out += fmt::format(" {} {} ({:.1f}%) {:.2f} us", kN3PathNames[k], draws[k],
		                   all == 0 ? 0.0 : 100.0 * static_cast<double>(draws[k]) / static_cast<double>(all), us(ticks[k], draws[k]));
	}
	const uint64_t bindings = fields[0] + fields[1] + fields[2] + fields[3] + fields[4] + fields[5];
	out += fmt::format(" | n3 segments {} all-window {} | n3 read bindings {} per draw {:.1f}: window {:.1f}%", segments, segments_window, bindings,
	                   all == 0 ? 0.0 : static_cast<double>(bindings) / static_cast<double>(all),
	                   bindings == 0 ? 0.0 : 100.0 * static_cast<double>(fields[0]) / static_cast<double>(bindings));
	out += " (normal path:";
	for (uint32_t k = 1; k <= 5; k++) out += fmt::format(" {} {}", kN3FieldNames[k], fields[k]);
	out += ") window invalidated:";
	for (uint32_t k = 6; k <= 9; k++) out += fmt::format(" {} {}", kN3FieldNames[k], fields[k]);
	if (Native().verify) {
		out += fmt::format(" | n3 verify: checks {} races {} mismatches {}", fields[10], fields[11], fields[12]);
	}
	out += " | worker encode: none (N3a)";
	return out;
}

// KYTY_NATIVE_MODE=n3b part of a slot (reporter only): lean draws and their Thread_Gpu time against
// the continuations that took the normal path, and why the others did not take the lean path.
std::string N3bSlotReport(uint32_t s, const std::array<uint64_t, GameCapture::kNrCount>& draws,
                          const std::array<uint64_t, GameCapture::kNrCount>& ticks) {
	auto&      r    = g_nl_resolve[s];
	auto&      last = g_nl_resolve_last[s];
	const auto take = [](std::atomic<uint64_t>& v, uint64_t& l) {
		const auto now = v.load(std::memory_order_relaxed);
		const auto d   = now - l;
		l              = now;
		return d;
	};
	std::array<uint64_t, GameCapture::kNlCount> reasons {};
	uint64_t                                    all = 0;
	for (uint32_t k = 0; k < GameCapture::kNlCount; k++) {
		reasons[k] = take(r.reasons[k], last.reasons[k]);
		all += reasons[k];
	}
	const auto   verified   = take(r.verified, last.verified);
	const auto   mismatches = take(r.mismatches, last.mismatches);
	const double tpu        = g_n2_tsc_per_us;
	const auto   us         = [&](uint64_t t, uint64_t n) {
        return tpu <= 0.0 || n == 0 ? 0.0 : static_cast<double>(t) / tpu / static_cast<double>(n);
	};
	const auto lean = GameCapture::kNrLean;
	const auto cont = GameCapture::kNrContinued;
	std::string out = fmt::format(" | n3b lean {} ({:.1f}%) {:.2f} us/draw vs normal-path continuation {} {:.2f} us/draw | n3b normal path:", draws[lean],
	                              all == 0 ? 0.0 : 100.0 * static_cast<double>(reasons[GameCapture::kNlLean]) / static_cast<double>(all),
	                              us(ticks[lean], draws[lean]), draws[cont], us(ticks[cont], draws[cont]));
	for (uint32_t k = 1; k < GameCapture::kNlCount; k++) {
		if (reasons[k] != 0) out += fmt::format(" {} {}", kNlNames[k], reasons[k]);
	}
	if (Native().verify) {
		out += fmt::format(" | n3b verify: samples {} mismatches {}", verified, mismatches);
	}
	return out;
}

// S0: what the learning found for this slot's native draws in the window, and the CP's check of
// the segment lengths the game thread wrote.
std::string S0SlotReport(uint32_t s) {
	auto&      r    = g_s0[s];
	auto&      last = g_s0_last[s];
	const auto take = [](std::atomic<uint64_t>& v, uint64_t& l) {
		const auto now = v.load(std::memory_order_relaxed);
		const auto d   = now - l;
		l              = now;
		return d;
	};
	std::array<uint64_t, kS0Fields> f {};
	for (uint32_t k = 0; k < kS0Fields; k++) f[k] = take(r.fields[k], last.fields[k]);
	const uint64_t draws = f[kS0fDraws];
	const auto     pct   = [&](uint64_t v) { return draws == 0 ? 0.0 : 100.0 * static_cast<double>(v) / static_cast<double>(draws); };
	std::array<uint64_t, GameCapture::kS0ReasonCount> reasons {};
	for (uint32_t k = 0; k < GameCapture::kS0ReasonCount; k++) reasons[k] = take(r.reasons[k], last.reasons[k]);
	std::string out = fmt::format(" | s0: native draws {} learned-key found {} ({:.1f}%) new {} conflict {} pipeline-replaced {} | recipe simple {} ({:.1f}%)",
	                              draws, f[kS0fFound], pct(f[kS0fFound]), f[kS0fLearned], f[kS0fConflict], f[kS0fReplaced],
	                              reasons[GameCapture::kS0Simple], pct(reasons[GameCapture::kS0Simple]));
	out += " not covered:";
	for (uint32_t k = 1; k < GameCapture::kS0ReasonCount; k++) {
		if (reasons[k] != 0) out += fmt::format(" {} {}", kS0ReasonNames[k], reasons[k]);
	}
	std::string deep;
	for (uint32_t k = 0; k < r.deep_op.size(); k++) {
		const auto op = r.deep_op[k].load(std::memory_order_relaxed);
		if (op != 0) deep += fmt::format(" op{}x{}", op - 1u, r.deep_n[k].load(std::memory_order_relaxed));
	}
	if (!deep.empty()) out += " (deep, cumulative:" + deep + ")";
	if (Native().verify) {
		out += fmt::format(" | s0 verify: samples {} dwords {} differ {} (draws {}) nulled {} | user data (ES/mesh stage vs record): dwords {} shadow eq {} ne {} unknown {}, slot copy eq {} ne {}",
		                   f[kS0fVerified], f[kS0fCompared], f[kS0fMismatch], f[kS0fBadDraws], f[kS0fNulled], f[kS0fUd], f[kS0fUdShadowEq],
		                   f[kS0fUdShadowNe], f[kS0fUdShadowUnk], f[kS0fUdCopyEq], f[kS0fUdCopyNe]);
	} else {
		out += " | s0 verify off (KYTY_NATIVE_VERIFY=1)";
	}
	std::array<uint64_t, kS0SegFields> seg {};
	for (uint32_t k = 0; k < kS0SegFields; k++) seg[k] = take(r.seg[k], last.seg[k]);
	std::array<uint64_t, 5> hist {};
	for (uint32_t k = 0; k < 5; k++) hist[k] = take(r.seg_hist[k], last.seg_hist[k]);
	out += fmt::format(" | segment length (CP, head records): heads {} draws/segment {:.1f} (len 1:{} 2-4:{} 5-16:{} 17-64:{} 65+:{}) marker check ok {} bad {} open {} orphan {}",
	                   seg[0], seg[0] == 0 ? 0.0 : static_cast<double>(seg[1]) / static_cast<double>(seg[0]), hist[0], hist[1], hist[2], hist[3],
	                   hist[4], seg[2], seg[3], seg[4], seg[5]);
	// S0b: the key conflicts' causes (first differing component of the two permutations).
	{
		auto&       c    = g_s0_conflicts[s];
		auto&       cl   = g_s0_conflicts_last[s];
		std::string text;
		for (uint32_t k = 1; k < GameCapture::kCfCount; k++) {
			const auto now = c.cause[k].load(std::memory_order_relaxed);
			const auto d   = now - cl[k];
			cl[k]          = now;
			if (d != 0) text += fmt::format(" {} {}", kS0ConflictNames[k], d);
		}
		std::string words;
		for (uint32_t w = 0; w < 64; w++) {
			const auto v = c.vs_word[w].load(std::memory_order_relaxed);
			const auto p = c.ps_word[w].load(std::memory_order_relaxed);
			if (v != 0) words += fmt::format(" vs{}x{}", w, v);
			if (p != 0) words += fmt::format(" ps{}x{}", w, p);
		}
		out += fmt::format(" | s0b conflict causes:{}{}", text.empty() ? " none" : text,
		                   words.empty() ? "" : " (static words, cumulative:" + words + ")");
	}
	if (Native().s1) {
		auto&                                     r    = g_s1[s];
		auto&                                     last = g_s1_last[s];
		std::array<uint64_t, GameCapture::kS1Count> n {};
		uint64_t                                  all  = 0;
		for (uint32_t k = 0; k < GameCapture::kS1Count; k++) {
			n[k] = take(r.reasons[k], last.reasons[k]);
			all += n[k];
		}
		const auto checked    = take(r.checked, last.checked);
		const auto mismatched = take(r.mismatched, last.mismatched);
		out += fmt::format(" | s1: {} {} native-emitted {} ({:.1f}%) fallback:", ((Native().s1_mask >> s) & 1u) != 0 ? "on" : "off", all,
		                   n[GameCapture::kS1Emitted],
		                   all == 0 ? 0.0 : 100.0 * static_cast<double>(n[GameCapture::kS1Emitted]) / static_cast<double>(all));
		for (uint32_t k = 1; k < GameCapture::kS1Count; k++) {
			if (n[k] != 0) out += fmt::format(" {} {}", kS1ReasonNames[k], n[k]);
		}
		out += fmt::format(" | s1 verify: predictions checked {} mismatched {}{}", checked, mismatched,
		                   Native().verify ? "" : " (off: KYTY_NATIVE_VERIFY=1)");
		out += " | s1 HOT copies: through the emulator's BufferCache (ObtainBuffer) in this subset, not counted separately";
	}
	return out;
}

std::string N2SlotReport(uint32_t s, uint64_t segments_game, uint64_t rec_incomplete, uint64_t flips) {
	auto& r    = g_n2_resolve[s];
	auto& last = g_n2_resolve_last[s];
	std::array<uint64_t, GameCapture::kNrCount> draws {}, ticks {};
	uint64_t                       all_draws = 0, all_ticks = 0;
	for (uint32_t k = 0; k < GameCapture::kNrCount; k++) {
		const auto d = r.draws[k].load(std::memory_order_relaxed);
		const auto t = r.ticks[k].load(std::memory_order_relaxed);
		draws[k]      = d - last.draws[k];
		ticks[k]      = t - last.ticks[k];
		last.draws[k] = d;
		last.ticks[k] = t;
		all_draws += draws[k];
		all_ticks += ticks[k];
	}
	const auto take = [](std::atomic<uint64_t>& v, uint64_t& l) {
		const auto now = v.load(std::memory_order_relaxed);
		const auto d   = now - l;
		l              = now;
		return d;
	};
	const auto segments = take(r.segments, last.segments);
	const auto seg_tick = take(r.segment_ticks, last.segment_ticks);
	const auto seg_draw = take(r.segment_draws, last.segment_draws);
	const auto unmarked = take(r.unmarked_starts, last.unmarked);
	std::array<uint64_t, 5> hist {};
	for (uint32_t k = 0; k < 5; k++) hist[k] = take(r.seg_hist[k], last.seg_hist[k]);
	const auto ok      = g_n2_rec_ok[s].load(std::memory_order_relaxed);
	const auto miss    = g_n2_rec_miss[s].load(std::memory_order_relaxed);
	const auto rec_ok  = ok - g_n2_rec_ok_last[s];
	const auto rec_bad = miss - g_n2_rec_miss_last[s];
	g_n2_rec_ok_last[s]   = ok;
	g_n2_rec_miss_last[s] = miss;
	const double tpu = g_n2_tsc_per_us;
	const auto   us  = [&](uint64_t t, uint64_t n) { return tpu <= 0.0 || n == 0 ? 0.0 : static_cast<double>(t) / tpu / static_cast<double>(n); };
	std::string out = fmt::format(" | n2: records ok {} miss {} incomplete {} | segments (game) {}", rec_ok, rec_bad, rec_incomplete,
	                              segments_game);
	if (Native().n3) {
		out += N3SlotReport(s);
	} else if (Native().n3b) {
		out += N3bSlotReport(s, draws, ticks);
	} else if (Native().s0) {
		out += fmt::format(" | emission: emulator path (splice) {}", all_draws);
		out += S0SlotReport(s);
	} else {
		out += fmt::format(" | emission: emulator path (splice) {} n2-native 0 (fallback reason: native emission not implemented)", all_draws);
	}
	out += fmt::format(" | Thread_Gpu {:.2f} us/native draw ({:.2f} ms/frame)", us(all_ticks, all_draws),
	                   tpu <= 0.0 || flips == 0 ? 0.0 : static_cast<double>(all_ticks) / tpu / 1000.0 / static_cast<double>(flips));
	out += " [";
	for (uint32_t k = 0; k < GameCapture::kNrCount; k++) {
		out += fmt::format("{}{} {} ({:.1f}%) {:.2f} us", k == 0 ? "" : ", ", kNrNames[k], draws[k],
		                   all_draws == 0 ? 0.0 : 100.0 * static_cast<double>(draws[k]) / static_cast<double>(all_draws), us(ticks[k], draws[k]));
	}
	out += fmt::format("] | resolver segments {} draws/segment {:.1f} span {:.2f} us/segment {:.2f} us/draw (len 1:{} 2-4:{} 5-16:{} 17-64:{} 65+:{}, "
	                   "opened mid-segment {})",
	                   segments, segments == 0 ? 0.0 : static_cast<double>(seg_draw) / static_cast<double>(segments), us(seg_tick, segments),
	                   us(seg_tick, seg_draw), hist[0], hist[1], hist[2], hist[3], hist[4], unmarked);
	return out;
}

std::string N2VerifyReport() {
	if (!Native().verify) {
		return " | n2 verify off (KYTY_NATIVE_VERIFY=1)";
	}
	uint64_t verified = 0, bad = 0;
	for (uint32_t s = 0; s <= kEngineThreads; s++) {
		const auto v = g_n2_verified[s].load(std::memory_order_relaxed);
		const auto b = g_n2_verify_bad[s].load(std::memory_order_relaxed);
		verified += v - g_n2_verified_last[s];
		bad += b - g_n2_verify_bad_last[s];
		g_n2_verified_last[s]   = v;
		g_n2_verify_bad_last[s] = b;
	}
	std::string out = fmt::format(" | n2 verify: {} draws compared, {} with a difference; field differ/unknown:", verified, bad);
	for (uint32_t f = 0; f < GameCapture::kNvCount; f++) {
		const auto b = g_n2_field_bad[f].load(std::memory_order_relaxed);
		const auto u = g_n2_field_unknown[f].load(std::memory_order_relaxed);
		out += fmt::format(" {} {}/{}", kNvNames[f], b - g_n2_field_bad_last[f], u - g_n2_field_unknown_last[f]);
		g_n2_field_bad_last[f]     = b;
		g_n2_field_unknown_last[f] = u;
	}
	return out;
}

std::string TopRegisters(const char* prefix, const std::atomic<uint32_t>* hist, uint32_t n, size_t max) {
	std::vector<std::pair<uint32_t, uint32_t>> top;
	for (uint32_t i = 0; i < n; i++) {
		if (const auto v = hist[i].load(std::memory_order_relaxed); v != 0) top.emplace_back(v, i);
	}
	std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
	std::string out = fmt::format("{} distinct", top.size());
	for (size_t k = 0; k < top.size() && k < max; k++) {
		out += fmt::format(" {}0x{:x}={}", prefix, top[k].second, top[k].first);
	}
	return out;
}

double Pct(uint64_t part, uint64_t whole) {
	return whole == 0 ? 0.0 : 100.0 * static_cast<double>(part) / static_cast<double>(whole);
}

void Report(int64_t now) {
	std::unique_lock report(g_report_mutex, std::try_to_lock);
	if (!report.owns_lock()) {
		return;
	}
	const double seconds = g_last_report != 0 ? static_cast<double>(now - g_last_report) / 1000.0 : 10.0;
	g_last_report        = now;

	std::string line = fmt::format("GameCapture 10s ({:.1f} s):", seconds);
	std::string stages = "GameCapture stages (cumulative, slot:type=count; Cs0 Ps1 Gs2 Hs3 GsF4 HsF5 GsB6 HsB7, 9=null):";
	std::string sample;
	uint64_t    best_records = 0;
	std::array<uint64_t, kCCount> sum {};
	uint64_t    first_missing = 0;
	{
		std::lock_guard lock(g_threads_mutex);
		for (auto* t: g_threads) {
			std::array<uint64_t, kCCount> d {};
			bool                          any = false;
			for (uint32_t i = 0; i < kCCount; i++) {
				const auto v = t->c[i].load(std::memory_order_relaxed);
				d[i]         = v - t->last[i];
				t->last[i]   = v;
				sum[i] += d[i];
				any = any || d[i] != 0;
			}
			first_missing |= t->first_missing.load(std::memory_order_relaxed);
			if (!any) {
				continue;
			}
			const uint64_t draws = d[kCRecords];
			line += fmt::format(" | \"{}\" (tid {}) draws {} [", t->name, t->thread_id, draws);
			for (uint32_t k = 0; k < 5; k++) line += fmt::format("{}{} {}", k == 0 ? "" : ", ", kKindNames[k], d[uint32_t {kCDraw0} + k]);
			line += fmt::format("] toolkit {} flushes {} (no hooked draw {}) dispatch-flushes {} | shaders {} rt-sets {} | complete {} ({:.1f} %) fail pipe {} stage {} target {} index {} outside {} no-begin {} | rt-resolved {:.1f} % | dirty {:.1f} bits/draw:",
			                   d[kCToolkit], d[kCFlush], d[kCFlushNoDraw], d[kCDispatchFlush],
			                   t->shaders.size.load(std::memory_order_relaxed), t->rtsets.size.load(std::memory_order_relaxed),
			                   d[kCComplete], Pct(d[kCComplete], draws), d[uint32_t {kCFail0} + 0], d[uint32_t {kCFail0} + 1], d[uint32_t {kCFail0} + 2],
			                   d[uint32_t {kCFail0} + 3], d[uint32_t {kCFail0} + 4], d[uint32_t {kCFail0} + 5], Pct(d[kCRtResolved], draws),
			                   draws == 0 ? 0.0 : static_cast<double>(d[kCDirtyBits]) / static_cast<double>(draws));
			for (uint32_t f = 0; f < 16; f++) {
				if (d[uint32_t {kCFam0} + f] != 0) line += fmt::format(" {} {:.0f}%", kFamilies[f].name, Pct(d[uint32_t {kCFam0} + f], draws));
			}
			line += fmt::format(" | acc");
			for (uint32_t k = 0; k < 6; k++) line += fmt::format(" {} {}", kAccNames[k], d[uint32_t {kCAcc0} + k]);

			stages += fmt::format(" | \"{}\" draws", t->name);
			for (uint32_t s = 0; s < 8; s++) {
				for (uint32_t ty = 0; ty < 10; ty++) {
					if (const auto v = t->slot_type[s][ty].load(std::memory_order_relaxed); v != 0) stages += fmt::format(" {}:{}={}", s, ty, v);
				}
			}
			stages += " dispatch";
			for (uint32_t s = 0; s < 8; s++) {
				for (uint32_t ty = 0; ty < 10; ty++) {
					if (const auto v = t->cs_slot_type[s][ty].load(std::memory_order_relaxed); v != 0) stages += fmt::format(" {}:{}={}", s, ty, v);
				}
			}
			stages += " ptype/active";
			for (uint32_t p = 0; p < 6; p++) {
				for (uint32_t m = 0; m < 256; m++) {
					if (const auto v = t->ptype_active[p][m].load(std::memory_order_relaxed); v != 0) stages += fmt::format(" {}/0x{:02x}={}", p, m, v);
				}
			}

			if (draws > best_records && t->ring_seq != 0) {
				// Latest record of the busiest thread (copied, then validated by its sequence).
				const auto& rec = t->ring[(t->ring_seq - 1) % kRing];
				const auto  seq = rec.seq.load(std::memory_order_acquire);
				if (seq != 0) {
					std::string s = fmt::format(
					    "GameCapture record \"{}\" #{}: ctx 0x{:x} {} prim {} count {} inst {} first {} vofs {} dirty 0x{:x} pipe 0x{:x} type {} active 0x{:02x} geom {} fail 0x{:x} stale {} |",
					    t->name, seq, rec.ctx, kKindNames[std::min<uint32_t>(rec.kind, 5u)], rec.prim, rec.count, rec.instances, rec.first,
					    rec.vertex_offset, rec.dirty, rec.pipeline, rec.pipeline_type, rec.active, rec.geom_slot, rec.fail, rec.stale_outside);
					for (uint32_t k = 0; k < 8; k++) {
						if ((rec.active & (1u << k)) != 0) s += fmt::format(" s{} 0x{:x} t{} code 0x{:x}", k, rec.shader[k], rec.type[k], rec.code[k]);
					}
					for (uint32_t k = 0; k < 3; k++) {
						if (rec.sgpr_slot[k] < 0) continue;
						s += fmt::format(" | sgpr s{}:", rec.sgpr_slot[k]);
						for (uint32_t w = 0; w < 16; w++) s += fmt::format(" {:08x}", rec.sgpr[k][w]);
					}
					for (uint32_t k = 0; k < 8; k++) {
						if (rec.rt[k].obj != 0)
							s += fmt::format(" | rt{} obj 0x{:x} id {} base 0x{:x} info 0x{:x} size 0x{:x}{}", k, rec.rt[k].obj, rec.rt[k].id,
							                 rec.rt[k].base, rec.rt[k].info, rec.rt[k].attrib2, rec.rt[k].known != 0 ? "" : " (unknown)");
					}
					s += fmt::format(" | depth obj 0x{:x} base 0x{:x} zinfo 0x{:x} size 0x{:x} | cb-mask 0x{:x} vp {:08x} {:08x} {:08x} {:08x} scissor {:08x} {:08x}"
					                 " | ib 0x{:x} n {} 32b {} vb mode {} table 0x{:x} | dsc 0x{:x} stc 0x{:x} rnd 0x{:x} su 0x{:x} blend0 0x{:x}\n",
					                 rec.depth_obj, rec.z_base, rec.z_info, rec.depth_size_xy, rec.cb_target_mask, rec.vport[0], rec.vport[1],
					                 rec.vport[2], rec.vport[3], rec.vport_scissor[0], rec.vport_scissor[1], rec.ib_addr, rec.ib_count, rec.ib_32,
					                 rec.vb_mode, rec.vb_table, rec.depth_control, rec.stencil_control, rec.render_control, rec.su_sc_mode, rec.blend[0]);
					if (rec.seq.load(std::memory_order_acquire) == seq) {
						sample       = std::move(s);
						best_records = draws;
					}
				}
			}
		}
	}
	const uint64_t draws = sum[kCRecords];
	line += fmt::format(" | total draws {} ({:.0f}/s) complete {:.1f} % rt-resolved {:.1f} % | shader self-check ok {} bad {}\n", draws,
	                    static_cast<double>(draws) / seconds, Pct(sum[kCComplete], draws), Pct(sum[kCRtResolved], draws),
	                    sum[kCShaderSelfOk], sum[kCShaderSelfBad]);
	stages += "\n";

	std::string lists = fmt::format(
	    "GameCapture lists: first draw flush after a list begin {} (all groups dirty {}, missing groups 0x{:x}; a dispatch flush came first {}) | draws on a context without begin {} | contexts {}{} | outside writes: cx {} ; sh {} | begin cx {}\n",
	    sum[kCFirstLists], sum[kCFirstFull], first_missing, sum[kCFirstDispatch], sum[kCNoBeginDraws],
	    [] {
		    size_t n = 0;
		    for (auto& s: g_contexts) n += s.load(std::memory_order_relaxed) != nullptr ? 1 : 0;
		    return n;
	    }(),
	    g_contexts_dropped.load(std::memory_order_relaxed) != 0 ? " (table full)" : "",
	    TopRegisters("cx", g_outside_cx.data(), kCxNum, 16), TopRegisters("sh", g_outside_sh.data(), kShNum, 16),
	    TopRegisters("cx", g_begin_cx.data(), kCxNum, 8));

	// Command processor cross-check (window deltas).
	std::array<uint64_t, 5 + 2 * kCmpFields> cur {};
	cur[0] = g_cp.draws.load(std::memory_order_relaxed);
	cur[1] = g_cp.published.load(std::memory_order_relaxed);
	cur[2] = g_cp.matched.load(std::memory_order_relaxed);
	cur[3] = g_cp.torn.load(std::memory_order_relaxed);
	cur[4] = g_cp.speculative.load(std::memory_order_relaxed);
	for (uint32_t i = 0; i < kCmpFields; i++) {
		cur[5 + i]              = g_cp.compared[i].load(std::memory_order_relaxed);
		cur[5 + kCmpFields + i] = g_cp.mismatched[i].load(std::memory_order_relaxed);
	}
	std::array<uint64_t, 5 + 2 * kCmpFields> dc {};
	for (size_t i = 0; i < cur.size(); i++) {
		dc[i]        = cur[i] - g_cp_last[i];
		g_cp_last[i] = cur[i];
	}
	std::string cp = fmt::format(
	    "GameCapture CP: draw packets written by libAgc: hooked draws {}, tool library {}, other code {} | parsed by the CP {} (speculative front {}) | samples published {} matched {} torn {} | compared/mismatched:",
	    sum[kCHleHooked], sum[kCHleToolkit], sum[kCHleOther], dc[0], dc[4], dc[1], dc[2], dc[3]);
	for (uint32_t i = 0; i < kCmpFields; i++) {
		cp += fmt::format(" {} {}/{}", kCmpNames[i], dc[5 + i], dc[5 + kCmpFields + i]);
	}
	cp += "\n";

	// KYTY_NATIVE.
	std::string native;
	if (Native().n2) {
		g_n2_tsc_per_us = N2TicksPerUs();
	}
	if (Native().valid) {
		const uint64_t flips_now = GameCapture::g_flips.load(std::memory_order_relaxed);
		const uint64_t flips     = flips_now - g_flips_last;
		g_flips_last             = flips_now;
		std::array<std::array<uint64_t, kNCount>, kEngineThreads> per {};
		{
			std::lock_guard lock(g_threads_mutex);
			for (auto* t: g_threads) {
				if (t->native_slot < 0) {
					continue;
				}
				for (uint32_t i = 0; i < kNCount; i++) {
					const auto v = t->n[i].load(std::memory_order_relaxed);
					per[t->native_slot][i] += v - t->n_last[i];
					t->n_last[i] = v;
				}
			}
		}
		std::array<uint64_t, kEngineThreads + 1> seen {};
		for (uint32_t s = 0; s <= kEngineThreads; s++) {
			const auto v          = g_native_seen[s].load(std::memory_order_relaxed);
			seen[s]               = v - g_native_seen_last[s];
			g_native_seen_last[s] = v;
		}
		std::array<uint64_t, kEngineThreads + 1>                          spliced {};
		std::array<std::array<uint64_t, kMissCount>, kEngineThreads + 1> miss {};
		for (uint32_t s = 0; s <= kEngineThreads; s++) {
			const auto v             = g_native_spliced[s].load(std::memory_order_relaxed);
			spliced[s]               = v - g_native_spliced_last[s];
			g_native_spliced_last[s] = v;
			for (uint32_t k = 0; k < kMissCount; k++) {
				const auto m             = g_native_miss[s][k].load(std::memory_order_relaxed);
				miss[s][k]               = m - g_native_miss_last[s][k];
				g_native_miss_last[s][k] = m;
			}
		}
		const auto per_frame = [&](uint64_t v) { return flips == 0 ? 0.0 : static_cast<double>(v) / static_cast<double>(flips); };
		native = fmt::format("Native 10s ({:.1f} s, mode {}): flips {}", seconds, Native().mode, flips);
		std::array<uint64_t, kNCount> total {};
		uint64_t                      total_seen    = 0;
		uint64_t                      total_spliced = 0;
		uint64_t                      total_miss    = 0;
		for (uint32_t s = 0; s < kEngineThreads; s++) {
			if (((Native().mask >> s) & 1u) == 0) {
				continue;
			}
			const auto& p = per[s];
			native += fmt::format(" | \"{}\" chosen {} written {} cp-seen {} fallback", kEngineThreadNames[s], p[kNChosen], p[kNWritten], seen[s]);
			for (uint32_t k = 0; k < kNFbKinds; k++) native += fmt::format(" {} {}", kNFbNames[k], p[kNFb0 + k]);
			native += fmt::format(" incomplete {}", p[kNIncomplete]);
			if (Native().splice) {
				uint64_t misses = 0;
				for (uint32_t k = 0; k < kMissCount; k++) misses += miss[s][k];
				native += fmt::format(" spliced {} record-miss {}", spliced[s], misses);
				for (uint32_t k = 0; k < kMissCount; k++) {
					if (miss[s][k] != 0) native += fmt::format(" ({} {})", kMissNames[k], miss[s][k]);
				}
				if (const auto* ring = g_native_rings[s]; ring != nullptr) {
					native += fmt::format(" ring-consumed {}", ring->consumed.load(std::memory_order_relaxed));
				}
			}
			if (Native().n2) {
				native += N2SlotReport(s, p[kNSegments], p[kNRecIncomplete], flips);
			}
			native += fmt::format(" | per frame chosen {:.1f} written {:.1f} cp-seen {:.1f}", per_frame(p[kNChosen]), per_frame(p[kNWritten]),
			                      per_frame(seen[s]));
			if (Native().splice) native += fmt::format(" spliced {:.1f}", per_frame(spliced[s]));
			total_spliced += spliced[s];
			for (uint32_t k = 0; k < kMissCount; k++) total_miss += miss[s][k];
			for (uint32_t i = 0; i < kNCount; i++) total[i] += p[i];
			total_seen += seen[s];
		}
		uint64_t malformed_miss = 0;
		for (uint32_t k = 0; k < kMissCount; k++) malformed_miss += miss[kEngineThreads][k];
		if (Native().splice) {
			native += fmt::format(" | total spliced {} record-miss {} ring-full {}", total_spliced, total_miss + malformed_miss, total[kNFbRing]);
		}
		if (Native().n2) {
			native += N2VerifyReport();
		}
		if (Native().s0) {
			native += fmt::format(" | s0 native pipeline table: keys {} (without specialization {})",
			                      g_s0_keys.load(std::memory_order_relaxed), g_s0_base_keys.load(std::memory_order_relaxed));
		}
		native += fmt::format(" | total chosen {} written {} cp-seen {} (other slots {}, malformed {}) per frame written {:.1f}\n", total[kNChosen],
		                      total[kNWritten], total_seen,
		                      [&] {
			                      uint64_t other = 0;
			                      for (uint32_t s = 0; s < kEngineThreads; s++) {
				                      if (((Native().mask >> s) & 1u) == 0) other += seen[s];
			                      }
			                      return other;
		                      }(),
		                      seen[kEngineThreads], per_frame(total[kNWritten]));
	}

	Log::WriteToConsoleAndLog(line);
	Log::WriteToConsoleAndLog(stages);
	Log::WriteToConsoleAndLog(lists);
	Log::WriteToConsoleAndLog(cp);
	if (!sample.empty()) {
		Log::WriteToConsoleAndLog(sample);
	}
	if (!native.empty()) {
		Log::WriteToConsoleAndLog(native);
	}
}

// ---- Registration -------------------------------------------------------------------------------------------------

int g_id_flush = -1;
std::vector<std::pair<int, const char*>> g_ids;

int Add(uint64_t elf, Kind kind, ObserverFn fn, ObserverFn entry, const std::vector<uint8_t>& bytes, const char* name) {
	HookSpec spec;
	spec.name        = name;
	spec.elf_address = elf;
	spec.kind        = kind;
	spec.expected    = bytes;
	spec.observer_fn = fn;
	spec.entry_fn    = entry;
	const int id     = Register(std::move(spec));
	g_ids.emplace_back(id, name);
	return id;
}

} // namespace

void RegisterHooks() {
	if (const auto& native = Native(); native.valid && native.splice) {
		for (uint32_t s = 0; s < kEngineThreads; s++) {
			if (((native.mask >> s) & 1u) != 0 && g_native_rings[s] == nullptr) {
				g_native_rings[s] = new NativeRing; // never freed (read by the CP until exit)
			}
			if (native.n2 && ((native.mask >> s) & 1u) != 0 && g_native_records[s] == nullptr) {
				g_native_records[s] = new NativeRecordRing; // never freed
			}
		}
		if (native.n2) {
			g_n2_tsc_last = __rdtsc();
			g_n2_ns_last  = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
		}
	}
	g_id_flush = Add(kFlushDraw, Kind::After, &OnDrawFlushExit, &OnDrawFlushEnter, kBytesFlushDraw, "capture_flush_draw");
	Add(kFlushDispatch, Kind::After, &OnDispatchFlushExit, &OnDispatchFlushEnter, kBytesFlushDispatch, "capture_flush_dispatch");
	Add(kDrawIndexed, Kind::Before, &OnDrawIndexed, nullptr, kBytesDrawIndexed, "capture_draw_indexed");
	Add(kDrawIndexedOf, Kind::Before, &OnDrawIndexedOfs, nullptr, kBytesDrawIndexedOf, "capture_draw_indexed_ofs");
	Add(kDrawAuto, Kind::Before, &OnDrawAuto, nullptr, kBytesDrawAuto, "capture_draw_auto");
	Add(kDrawAutoOf, Kind::Before, &OnDrawAutoOfs, nullptr, kBytesDrawAutoOf, "capture_draw_auto_ofs");
	Add(kDrawIndirect, Kind::Before, &OnDrawIndirect, nullptr, kBytesDrawIndirect, "capture_draw_indirect");
	Add(kToolkitDraw, Kind::After, &OnToolkitExit, &OnToolkitEnter, kBytesToolkitDraw, "capture_toolkit_draw");
	Add(kListBegin, Kind::After, &OnListBeginExit, &OnListBeginEnter, kBytesListBegin, "capture_list_begin");
	Add(kAccCx, Kind::Before, &OnAccumulator<0>, nullptr, kBytesAccCx, "capture_acc_cx");
	Add(kAccSh, Kind::Before, &OnAccumulator<1>, nullptr, kBytesAccSh, "capture_acc_sh");
	Add(kAccUc, Kind::Before, &OnAccumulator<2>, nullptr, kBytesAccUc, "capture_acc_uc");
}

void OnInstalled() {
	std::string missing;
	for (const auto& [id, name]: g_ids) {
		if (!IsInstalled(id)) {
			missing += fmt::format(" {}", name);
		}
	}
	const bool ok = IsInstalled(g_id_flush);
	Log::WriteToConsoleAndLog(fmt::format("GameCapture: {} ({} hooks registered{}{})\n", ok ? "active" : "inactive (draw flush hook not installed)",
	                                      g_ids.size(), missing.empty() ? "" : ", not installed:", missing));
	if (const auto& native = Native(); native.requested) {
		std::string threads;
		for (uint32_t s = 0; s < kEngineThreads; s++) {
			if (((native.mask >> s) & 1u) != 0) threads += fmt::format(" \"{}\"", kEngineThreadNames[s]);
		}
		Log::WriteToConsoleAndLog(fmt::format(
		    "GameNative: KYTY_NATIVE {} | mode {}{}{} | threads:{}{}{}\n",
		    !native.valid ? "ignored" : !ok ? "inactive (capture hooks not installed)" : native.n3 ? "active (stage N3a)" : native.n3b ? "active (stage N3b)" : native.s1 ? "active (stage S1)" : native.s0 ? "active (stage S0)" : native.n2 ? "active (stage N2)" : native.splice ? "active (stage N1)" : "active (stage N0)",
		    native.mode, native.mode == "drop" || native.splice ? "" :  " (unsupported: drop, splice, n2, n3, n3b, s0 or s1)",
		    native.splice ? fmt::format(" (ring {} entries{})", kNativeRingSize, native.verify ? ", verify" : "") : std::string {},
		    threads.empty() ? " none" : threads, native.unknown.empty() ? "" : " | unknown:", native.unknown));
		if (native.valid && ok) {
			GameCapture::g_native_mode.store(static_cast<uint8_t>(native.n3       ? GameCapture::NativeMode::N3
			                                                      : native.n3b    ? GameCapture::NativeMode::N3b
			                                                      : native.s1     ? GameCapture::NativeMode::S1
			                                                      : native.s0     ? GameCapture::NativeMode::S0
			                                                      : native.n2     ? GameCapture::NativeMode::N2
			                                                      : native.splice ? GameCapture::NativeMode::Splice
			                                                                      : GameCapture::NativeMode::Drop),
			                                 std::memory_order_release);
		}
	}
	GameCapture::g_active.store(ok, std::memory_order_release);
}

} // namespace Loader::GameHooks::Capture

// ---- Public surface (gameCapture.h) ------------------------------------------------------------------------------------

namespace Loader::GameCapture {

using namespace Loader::GameHooks::Capture; // NOLINT(google-build-using-namespace)

namespace {

// KYTY_NATIVE: the direct draw packet becomes an R_NATIVE_DRAW NOP of the same size.
enum class MarkerResult : uint8_t { NotDirect, Written, RingFull };
MarkerResult WriteNativeMarker(ThreadState* t, uint32_t* packet, uint32_t opcode, uint32_t size_dw) {
	namespace Pm4 = Libs::Graphics::Pm4;
	const uint32_t expected = opcode == Pm4::IT_DRAW_INDEX_2          ? 6u
	                          : opcode == Pm4::IT_DRAW_INDEX_OFFSET_2 ? 5u
	                          : opcode == Pm4::IT_DRAW_INDEX_AUTO     ? 3u
	                                                                   : 0u;
	if (packet == nullptr || expected == 0 || size_dw != expected || KYTY_PM4_LEN(packet[0]) != size_dw ||
	    ((packet[0] >> 8u) & 0xffu) != opcode) {
		return MarkerResult::NotDirect;
	}
	const uint32_t serial = t->native_serial + 1u;
	if (serial == 0) [[unlikely]] {
		return MarkerResult::NotDirect; // 2^32 draws on one thread: serial 0 = entry being rewritten
	}
	if (auto* ring = g_native_rings[t->native_slot]; ring != nullptr) {
		// Splice: publish the original packet before it becomes a marker.
		if (serial - ring->consumed.load(std::memory_order_acquire) >= kNativeRingSize) {
			Inc(t->n[kNFbRing]);
			t->seg_broken = true;
			return MarkerResult::RingFull; // the normal packet stays
		}
		if (auto* records = g_native_records[t->native_slot]; records != nullptr) {
			// n2: the record first, under the same serial (the CP reads it at the marker).
			t->native_rec.serial = serial;
			// S0: segment length. A head closes the previous segment and counts 1; a continuation
			// adds itself to its head's count (both only for draws whose marker is written: every
			// fallback breaks the segment, so the next native draw is a head).
			const bool s0_head = Native().s0 && t->native_rec.seg_start != 0;
			if (s0_head) {
				if (t->seg_head_serial != 0) {
					auto& prev = records->entries[t->seg_head_serial & (kNativeRingSize - 1u)];
					if (prev.serial.load(std::memory_order_relaxed) == t->seg_head_serial) {
						prev.words[GameCapture::kNativeSegClosedWord].store(1, std::memory_order_release);
					}
				}
				t->native_rec.seg_count  = 1;
				t->native_rec.seg_closed = 0;
			}
			uint32_t words[GameCapture::kNativeDrawDw];
			std::memcpy(words, &t->native_rec, sizeof(words));
			auto& r = records->entries[serial & (kNativeRingSize - 1u)];
			r.serial.store(0, std::memory_order_relaxed);
			std::atomic_thread_fence(std::memory_order_release);
			for (uint32_t i = 0; i < GameCapture::kNativeDrawDw; i++) {
				r.words[i].store(words[i], std::memory_order_relaxed);
			}
			r.serial.store(serial, std::memory_order_release);
			if (s0_head) {
				t->seg_head_serial = serial;
				t->seg_len         = 1;
			} else if (Native().s0 && t->seg_head_serial != 0) {
				// Within one segment the head is at most a few thousand serials old: its entry is the
				// producer's own and cannot have been reused (ring of 16384, reuse after consumption).
				t->seg_len++;
				auto& head = records->entries[t->seg_head_serial & (kNativeRingSize - 1u)];
				if (head.serial.load(std::memory_order_relaxed) == t->seg_head_serial) {
					head.words[GameCapture::kNativeSegCountWord].store(t->seg_len, std::memory_order_release);
				}
			}
		}
		auto& e = ring->entries[serial & (kNativeRingSize - 1u)];
		e.serial.store(0, std::memory_order_relaxed);
		std::atomic_thread_fence(std::memory_order_release);
		e.size_dw.store(size_dw, std::memory_order_relaxed);
		for (uint32_t i = 0; i < size_dw; i++) {
			e.words[i].store(packet[i], std::memory_order_relaxed);
		}
		e.serial.store(serial, std::memory_order_release);
	}
	t->native_serial = serial;
	packet[1]        = (kNativeMagic << 16u) | (kNativeVersion << 8u) | static_cast<uint32_t>(t->native_slot) |
	            (g_native_records[t->native_slot] != nullptr && t->native_rec.seg_start != 0 ? kNativeSegStartFlag : 0u);
	packet[2]             = serial;
	for (uint32_t i = 3; i < size_dw; i++) {
		packet[i] = 0;
	}
	packet[0] = KYTY_PM4(size_dw, Pm4::IT_NOP, Pm4::R_NATIVE_DRAW);
	Inc(t->n[kNWritten]);
	return MarkerResult::Written;
}

} // namespace

void NoteDrawPacketSlow(uint32_t* packet, uint32_t opcode, uint32_t size_dw) {
	auto* t = T();
	if (t->emit_window) {
		t->emit_window = false;
		Inc(t->c[kCHleHooked]);
		if (t->native_window) [[unlikely]] {
			t->native_window = false;
			const auto result = WriteNativeMarker(t, packet, opcode, size_dw);
			if (result == MarkerResult::Written) {
				t->sample_pending = false; // the CP will not see this draw packet
				return;
			}
			if (result == MarkerResult::NotDirect) {
				t->seg_broken = true;
				Inc(t->n[kNFbPacket]);
			} // RingFull (splice): the normal packet stays, counted as ring-full
		}
		if (t->sample_pending) {
			t->sample_pending = false;
			const auto address = reinterpret_cast<uint64_t>(packet);
			auto&      slot    = g_samples[SampleIndex(address)];
			slot.packet.store(0, std::memory_order_relaxed);
			std::atomic_thread_fence(std::memory_order_release);
			for (size_t i = 0; i < 7; i++) slot.w[i].store(t->sample_words[i], std::memory_order_relaxed);
			slot.packet.store(address, std::memory_order_release);
			g_cp.published.fetch_add(1, std::memory_order_relaxed);
		}
		return;
	}
	uint64_t   scope_ctx = 0;
	const auto scope     = TopScope(t, &scope_ctx);
	Inc(t->c[scope == kScopeToolkit ? kCHleToolkit : kCHleOther]);
}

void CheckCpDrawSlow(const CpDrawView& view) {
	(view.speculative ? g_cp.speculative : g_cp.draws).fetch_add(1, std::memory_order_relaxed);
	auto&      slot   = g_samples[SampleIndex(view.packet)];
	const auto packet = slot.packet.load(std::memory_order_acquire);
	if (packet == 0 || packet != view.packet) {
		return;
	}
	std::array<uint64_t, 7> w {};
	for (size_t i = 0; i < 7; i++) w[i] = slot.w[i].load(std::memory_order_relaxed);
	std::atomic_thread_fence(std::memory_order_acquire);
	auto expected = packet;
	if (!slot.packet.compare_exchange_strong(expected, 0, std::memory_order_acq_rel)) {
		g_cp.torn.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	g_cp.matched.fetch_add(1, std::memory_order_relaxed);

	const auto count     = static_cast<uint32_t>(w[0]);
	const auto first     = static_cast<uint32_t>(w[0] >> 32u);
	const auto instances = static_cast<uint32_t>(w[1]);
	const auto kind      = static_cast<uint32_t>((w[1] >> 32u) & 0xffu);
	const bool ib32      = ((w[1] >> 40u) & 1u) != 0;
	const bool rt0_known = ((w[1] >> 41u) & 1u) != 0;
	const bool has_ps    = ((w[1] >> 42u) & 1u) != 0;
	const auto geom_slot = static_cast<uint32_t>((w[1] >> 48u) & 0xffu);
	const bool indexed   = kind == kKindIndexed || kind == kKindIndexedOfs;
	const bool indirect  = kind == kKindIndirect;

	uint32_t   bad = 0;
	const auto cmp = [&](CmpField field, bool equal) {
		g_cp.compared[field].fetch_add(1, std::memory_order_relaxed);
		if (!equal) {
			g_cp.mismatched[field].fetch_add(1, std::memory_order_relaxed);
			bad |= 1u << field;
		}
	};
	constexpr uint32_t kOpOffset2 = 0x35, kOpAuto = 0x2d, kOpIndexIndirect = 0x25;
	const uint32_t     want       = indexed ? kOpOffset2 : indirect ? kOpIndexIndirect : kOpAuto;
	cmp(kCmpOpcode, view.opcode == want);
	if (!indirect) {
		cmp(kCmpCount, view.index_count == count);
		if (view.instances_known) cmp(kCmpInstances, view.instances == instances);
	}
	if (indexed) {
		cmp(kCmpFirst, view.index_offset == first);
	}
	if (indexed || indirect) {
		cmp(kCmpIb, view.ib_base == w[2]);
		cmp(kCmpIbType, view.ib_type == (ib32 ? 1u : 0u));
	}
	if (rt0_known) {
		cmp(kCmpRt0, view.rt0_base == w[3]);
	}
	if (has_ps) {
		cmp(kCmpPs, view.ps_addr == (w[4] & 0xffffffffffffull));
	}
	if (w[5] != 0) {
		// GS-family slot: the CP holds the engine's geometry program in the ES pair (es_regs, what the
		// emulator draws with); the GS pair is accepted too.
		const uint64_t code = w[5] & 0xffffffffffffull;
		cmp(kCmpGeom, (geom_slot & 1u) != 0 ? view.hs_addr == code : (view.es_addr == code || view.gs_addr == code));
	}
	if (bad != 0 && g_cp.logged.fetch_add(1, std::memory_order_relaxed) < 8) {
		Log::WriteToConsoleAndLog(fmt::format(
		    "GameCapture CP mismatch 0x{:x} at packet 0x{:x}: game {} count {} first {} inst {} ib 0x{:x}/{} rt0 0x{:x} ps 0x{:x} geom s{} 0x{:x} | "
		    "CP op 0x{:x} count {} first {} inst {}{} ib 0x{:x}/{} rt0 0x{:x} ps 0x{:x} es 0x{:x} gs 0x{:x} hs 0x{:x}\n",
		    bad, view.packet, kKindNames[std::min<uint32_t>(kind, 5u)], count, first, instances, w[2], ib32 ? 32 : 16, w[3], w[4], geom_slot,
		    w[5], view.opcode, view.index_count, view.index_offset, view.instances, view.instances_known ? "" : "?", view.ib_base,
		    view.ib_type, view.rt0_base, view.ps_addr, view.es_addr, view.gs_addr, view.hs_addr));
	}
}

namespace {

uint32_t MarkerSlot(const uint32_t* body) {
	if (body != nullptr && (body[0] >> 16u) == kNativeMagic && ((body[0] >> 8u) & 0xffu) == kNativeVersion &&
	    (body[0] & 0x70u) == 0 && (body[0] & kNativeSlotMask) < kEngineThreads) {
		return body[0] & kNativeSlotMask;
	}
	return kEngineThreads; // malformed
}

// S0 (CP, real parse): per CP thread and slot, the open segment (the head record's count and the
// markers of the slot parsed since). A segment never spans two lists (a list begin breaks it), and
// one list is parsed by one CP thread at a time.
struct S0SegCheck {
	uint32_t expected = 0, seen = 0, head = 0;
	bool     open = false;
};
thread_local std::array<S0SegCheck, kEngineThreads> t_s0_seg {};

void NoteS0SegmentMarker(uint32_t slot, const NativeDraw& n) {
	auto& r = g_s0[slot];
	auto& c = t_s0_seg[slot];
	if (n.seg_start != 0) {
		if (c.open) {
			if (c.seen == c.expected) {
				Inc(r.seg[2]);
			} else {
				Inc(r.seg[3]);
				if (g_s0_seg_logged.fetch_add(1, std::memory_order_relaxed) < 8) {
					Log::WriteToConsoleAndLog(fmt::format("GameNative s0: \"{}\" segment head serial {}: head count {} but {} markers parsed\n",
					                                      kEngineThreadNames[slot], c.head, c.expected, c.seen));
				}
			}
		}
		c.open     = true;
		c.expected = n.seg_count;
		c.seen     = 1;
		c.head     = n.serial;
		Inc(r.seg[0]);
		Inc(r.seg[1], n.seg_count);
		const uint32_t len    = n.seg_count;
		const uint32_t bucket = len <= 1 ? 0 : len <= 4 ? 1 : len <= 16 ? 2 : len <= 64 ? 3 : 4;
		Inc(r.seg_hist[bucket]);
		if (n.seg_closed == 0) Inc(r.seg[4]);
	} else if (c.open) {
		c.seen++;
	} else {
		Inc(r.seg[5]);
	}
}

[[noreturn]] void NativeVerifyExit(const char* what, const uint32_t* body, uint32_t marker_size_dw, const NativeDrawPacket* packet) {
	const auto msg = fmt::format("GameNative: KYTY_NATIVE_VERIFY: splice record miss ({}): marker word1 0x{:08x} serial {} size {} | record size {} "
	                             "header 0x{:08x}\n",
	                             what, body != nullptr ? body[0] : 0u, body != nullptr ? body[1] : 0u, marker_size_dw,
	                             packet != nullptr ? packet->size_dw : 0u, packet != nullptr ? packet->words[0] : 0u);
	Log::WriteToConsoleAndLog(msg);
	std::fflush(nullptr);
	std::_Exit(3);
}

} // namespace

void NoteNativeMarker(const uint32_t* body, uint32_t body_dw) {
	const uint32_t slot = body_dw >= 2 ? MarkerSlot(body) : kEngineThreads;
	g_native_seen[slot].fetch_add(1, std::memory_order_relaxed);
}

bool LookupNativeDraw(const uint32_t* body, uint32_t marker_size_dw, bool consume, NativeDrawPacket& out) {
	namespace Pm4      = Libs::Graphics::Pm4;
	const uint32_t slot = marker_size_dw >= 3 ? MarkerSlot(body) : kEngineThreads;
	const auto     miss = [&](NMiss reason, const NativeDrawPacket* packet) {
		if (consume) {
			g_native_miss[slot][reason].fetch_add(1, std::memory_order_relaxed);
			if (Native().verify) {
				NativeVerifyExit(kMissNames[reason], body, marker_size_dw, packet);
			}
		}
		return false;
	};
	if (slot >= kEngineThreads) {
		return miss(kMissMalformed, nullptr);
	}
	auto* ring = g_native_rings[slot];
	if (ring == nullptr) {
		return miss(kMissNoRing, nullptr);
	}
	const uint32_t serial = body[1];
	auto&          e      = ring->entries[serial & (kNativeRingSize - 1u)];
	const uint32_t s1     = e.serial.load(std::memory_order_acquire);
	out.size_dw           = e.size_dw.load(std::memory_order_relaxed);
	const uint32_t n      = std::min<uint32_t>(out.size_dw, kNativeMaxPacketDw);
	for (uint32_t i = 0; i < n; i++) {
		out.words[i] = e.words[i].load(std::memory_order_relaxed);
	}
	std::atomic_thread_fence(std::memory_order_acquire);
	const uint32_t s2 = e.serial.load(std::memory_order_relaxed);
	if (serial == 0 || s1 != serial || s2 != serial) {
		return miss(kMissSerial, nullptr);
	}
	const uint32_t opcode   = (out.words[0] >> 8u) & 0xffu;
	const uint32_t expected = opcode == Pm4::IT_DRAW_INDEX_2          ? 6u
	                          : opcode == Pm4::IT_DRAW_INDEX_OFFSET_2 ? 5u
	                          : opcode == Pm4::IT_DRAW_INDEX_AUTO     ? 3u
	                                                                   : 0u;
	if (expected == 0 || out.size_dw != expected || out.size_dw != marker_size_dw || KYTY_PM4_LEN(out.words[0]) != out.size_dw) {
		return miss(kMissPacket, &out);
	}
	if (consume) {
		// Running maximum: lists of one thread may be parsed by two CP threads (KYTY_CP_PAR).
		uint32_t current = ring->consumed.load(std::memory_order_relaxed);
		while (static_cast<int32_t>(serial - current) > 0 &&
		       !ring->consumed.compare_exchange_weak(current, serial, std::memory_order_release, std::memory_order_relaxed)) {
		}
	}
	return true;
}

void NoteNativeSpliced(const uint32_t* body) {
	g_native_spliced[MarkerSlot(body)].fetch_add(1, std::memory_order_relaxed);
}

bool LookupNativeRecord(const uint32_t* body, uint32_t marker_size_dw, bool real, NativeDraw& out) {
	const uint32_t slot = marker_size_dw >= 3 ? MarkerSlot(body) : kEngineThreads;
	if (slot >= kEngineThreads || g_native_records[slot] == nullptr) {
		if (real) g_n2_rec_miss[slot].fetch_add(1, std::memory_order_relaxed);
		return false;
	}
	const uint32_t serial = body[1];
	auto&          e      = g_native_records[slot]->entries[serial & (kNativeRingSize - 1u)];
	uint32_t       words[kNativeDrawDw];
	const uint32_t s1 = e.serial.load(std::memory_order_acquire);
	for (uint32_t i = 0; i < kNativeDrawDw; i++) {
		words[i] = e.words[i].load(std::memory_order_relaxed);
	}
	std::atomic_thread_fence(std::memory_order_acquire);
	const uint32_t s2 = e.serial.load(std::memory_order_relaxed);
	std::memcpy(&out, words, sizeof(words));
	if (serial == 0 || s1 != serial || s2 != serial || out.serial != serial) {
		if (real) g_n2_rec_miss[slot].fetch_add(1, std::memory_order_relaxed);
		return false;
	}
	if (real) g_n2_rec_ok[slot].fetch_add(1, std::memory_order_relaxed);
	if (real && Native().s0) {
		NoteS0SegmentMarker(slot, out);
	}
	return true;
}

namespace {
// Executor-side record copy (any thread): the entry of the op's slot whose serial has the op's low
// 16 bits, read under its seqlock.
bool ReadRecordEntry(uint32_t slot, uint32_t serial16, NativeDraw& out) {
	if (slot >= kEngineThreads || g_native_records[slot] == nullptr) {
		return false;
	}
	auto&          e = g_native_records[slot]->entries[serial16 & (kNativeRingSize - 1u)];
	uint32_t       words[kNativeDrawDw];
	const uint32_t s1 = e.serial.load(std::memory_order_acquire);
	for (uint32_t i = 0; i < kNativeDrawDw; i++) {
		words[i] = e.words[i].load(std::memory_order_relaxed);
	}
	std::atomic_thread_fence(std::memory_order_acquire);
	const uint32_t s2 = e.serial.load(std::memory_order_relaxed);
	std::memcpy(&out, words, sizeof(words));
	return s1 != 0 && s1 == s2 && out.serial == s1 && (s1 & 0xffffu) == serial16;
}
} // namespace

bool LookupNativeRecordForOp(uint32_t op_flags, NativeDraw& out) {
	if ((op_flags & kNativeOpFlag) == 0) {
		return false;
	}
	return ReadRecordEntry((op_flags >> kNativeOpSlotShift) & kNativeSlotMask, op_flags >> kNativeOpSerialShift, out);
}

void NoteNativeS0(uint32_t op_flags, const NativeS0Draw& d) {
	const uint32_t slot = (op_flags >> kNativeOpSlotShift) & kNativeSlotMask;
	if (slot >= kEngineThreads) {
		return;
	}
	auto& r = g_s0[slot];
	Inc(r.fields[kS0fDraws]);
	Inc(r.reasons[std::min<uint32_t>(d.reason, kS0ReasonCount - 1u)]);
	if (d.found) Inc(r.fields[kS0fFound]);
	if (d.learned) Inc(r.fields[kS0fLearned]);
	if (d.conflict) {
		Inc(r.fields[kS0fConflict]);
		auto& c = g_s0_conflicts[slot];
		Inc(c.cause[std::min<uint32_t>(d.conflict_cause, kCfCount - 1u)]);
		if (d.conflict_cause == kCfVsStatic) Inc(c.vs_word[std::min<uint32_t>(d.conflict_word, 63u)]);
		if (d.conflict_cause == kCfPsStatic) Inc(c.ps_word[std::min<uint32_t>(d.conflict_word, 63u)]);
	}
	if (d.s1_checked) {
		Inc(g_s1[slot].checked);
		if (d.s1_mismatches != 0) Inc(g_s1[slot].mismatched);
	}
	if (d.replaced) Inc(r.fields[kS0fReplaced]);
	if (d.verified) {
		Inc(r.fields[kS0fVerified]);
		Inc(r.fields[kS0fCompared], d.compared);
		Inc(r.fields[kS0fMismatch], d.mismatches);
		if (d.mismatches != 0) Inc(r.fields[kS0fBadDraws]);
		Inc(r.fields[kS0fNulled], d.nulled);
		Inc(r.fields[kS0fUd], d.ud_compared);
		Inc(r.fields[kS0fUdShadowEq], d.ud_shadow_eq);
		Inc(r.fields[kS0fUdShadowNe], d.ud_shadow_ne);
		Inc(r.fields[kS0fUdShadowUnk], d.ud_shadow_unk);
		Inc(r.fields[kS0fUdCopyEq], d.ud_copy_eq);
		Inc(r.fields[kS0fUdCopyNe], d.ud_copy_ne);
	}
	if (d.reason == kS0Deep && d.deep_opcode != 0) {
		for (uint32_t k = 0; k < r.deep_op.size(); k++) {
			const auto op = r.deep_op[k].load(std::memory_order_relaxed);
			if (op == d.deep_opcode || op == 0) {
				r.deep_op[k].store(d.deep_opcode, std::memory_order_relaxed);
				Inc(r.deep_n[k]);
				break;
			}
		}
	}
}

void NoteNativeS0Mismatch(uint32_t op_flags, const char* text) {
	const uint32_t slot = (op_flags >> kNativeOpSlotShift) & kNativeSlotMask;
	if (g_s0_logged.fetch_add(1, std::memory_order_relaxed) < 16) {
		Log::WriteToConsoleAndLog(fmt::format("GameNative s0 verify: \"{}\" serial(lo16) {}: {}\n",
		                                      slot < kEngineThreads ? kEngineThreadNames[slot] : "?", op_flags >> kNativeOpSerialShift,
		                                      text != nullptr ? text : ""));
	}
}

void NoteNativeS1(uint32_t op_flags, const NativeS1Draw& d) {
	const uint32_t slot = (op_flags >> kNativeOpSlotShift) & kNativeSlotMask;
	if (slot >= kEngineThreads) {
		return;
	}
	Inc(g_s1[slot].reasons[std::min<uint32_t>(d.reason, kS1Count - 1u)]);
}

void NoteNativeS1Mismatch(uint32_t op_flags, const char* text) {
	const uint32_t slot = (op_flags >> kNativeOpSlotShift) & kNativeSlotMask;
	if (g_s1_logged.fetch_add(1, std::memory_order_relaxed) < 32) {
		Log::WriteToConsoleAndLog(fmt::format("GameNative s1 verify: \"{}\" serial(lo16) {}: {}\n",
		                                      slot < kEngineThreads ? kEngineThreadNames[slot] : "?", op_flags >> kNativeOpSerialShift,
		                                      text != nullptr ? text : ""));
	}
}

bool NativeS1SlotEnabled(uint32_t op_flags) {
	const uint32_t slot = (op_flags >> kNativeOpSlotShift) & kNativeSlotMask;
	return slot < kEngineThreads && ((Native().s1_mask >> slot) & 1u) != 0;
}

void NoteNativeS0Table(uint32_t keys, uint32_t base_keys) {
	g_s0_keys.store(keys, std::memory_order_relaxed);
	g_s0_base_keys.store(base_keys, std::memory_order_relaxed);
}

bool NativeVerifyEnabled() {
	return Native().verify;
}

void NoteNativeVerify(const uint32_t* body, uint32_t mismatch, uint32_t unknown, const char* detail) {
	const uint32_t slot = MarkerSlot(body);
	g_n2_verified[slot].fetch_add(1, std::memory_order_relaxed);
	for (uint32_t f = 0; f < kNvCount; f++) {
		if ((mismatch & (1u << f)) != 0) g_n2_field_bad[f].fetch_add(1, std::memory_order_relaxed);
		if ((unknown & (1u << f)) != 0) g_n2_field_unknown[f].fetch_add(1, std::memory_order_relaxed);
	}
	if (mismatch == 0) {
		return;
	}
	g_n2_verify_bad[slot].fetch_add(1, std::memory_order_relaxed);
	if (g_n2_logged.fetch_add(1, std::memory_order_relaxed) < 8) {
		std::string fields;
		for (uint32_t f = 0; f < kNvCount; f++) {
			if ((mismatch & (1u << f)) != 0) fields += fmt::format(" {}", kNvNames[f]);
		}
		Log::WriteToConsoleAndLog(fmt::format("GameNative n2 verify: \"{}\" serial {}: record differs from the CP registers:{} | {}\n",
		                                      slot < kEngineThreads ? kEngineThreadNames[slot] : "?", body[1], fields,
		                                      detail != nullptr ? detail : ""));
	}
}

namespace {
// Resolver (Thread_Gpu, or CP resolver #2 with KYTY_CP_PAR): the open segment span.
struct N2Span {
	int      slot   = -1;
	uint64_t start  = 0;
	uint64_t end    = 0;
	uint64_t draws  = 0;
};
thread_local N2Span t_span;
// A gap longer than this (TSC cycles, ~1 ms) between two ops of a segment closes the span: the
// resolver left the list (another submission, a wait) and the span would count foreign time.
constexpr uint64_t kSpanGapTicks = 3'000'000;

void CloseSpan(N2Span& span) {
	if (span.slot < 0) {
		return;
	}
	auto& r = g_n2_resolve[static_cast<uint32_t>(span.slot)];
	Inc(r.segments);
	Inc(r.segment_ticks, span.end - span.start);
	Inc(r.segment_draws, span.draws);
	const uint32_t bucket = span.draws <= 1 ? 0 : span.draws <= 4 ? 1 : span.draws <= 16 ? 2 : span.draws <= 64 ? 3 : 4;
	Inc(r.seg_hist[bucket]);
	span.slot = -1;
}
} // namespace

void NoteNativeResolved(uint32_t op_flags, uint64_t start_tsc, uint64_t end_tsc, uint32_t run) {
	const uint32_t slot = (op_flags >> kNativeOpSlotShift) & kNativeSlotMask;
	if (slot >= kEngineThreads || run >= kNrCount) {
		return;
	}
	auto& r = g_n2_resolve[slot];
	Inc(r.draws[run]);
	Inc(r.ticks[run], end_tsc - start_tsc);
	auto&      span      = t_span;
	const bool seg_start = (op_flags & kNativeOpSegStart) != 0;
	if (seg_start || span.slot != static_cast<int>(slot) || start_tsc - span.end > kSpanGapTicks) {
		CloseSpan(span);
		if (!seg_start) {
			Inc(r.unmarked_starts); // continued a segment whose start was not resolved here
		}
		span.slot  = static_cast<int>(slot);
		span.start = start_tsc;
		span.draws = 0;
	}
	span.end = end_tsc;
	span.draws++;
}

void NoteNativeLean(uint32_t op_flags, uint32_t reason, bool verified, bool mismatch) {
	const uint32_t slot = (op_flags >> kNativeOpSlotShift) & kNativeSlotMask;
	if (slot >= kEngineThreads || reason >= kNlCount) {
		return;
	}
	auto& r = g_nl_resolve[slot];
	Inc(r.reasons[reason]);
	if (verified) {
		Inc(r.verified);
	}
	if (mismatch) {
		Inc(r.mismatches);
	}
}

namespace {
// Resolver: the open n3 segment (whether its draws after the first all took the window path).
struct N3Span {
	int      slot       = -1;
	uint64_t draws      = 0;
	bool     all_window = true;
};
thread_local N3Span t_n3_span;
void CloseN3Span(N3Span& span) {
	if (span.slot < 0) {
		return;
	}
	auto& r = g_n3_resolve[static_cast<uint32_t>(span.slot)];
	Inc(r.segments);
	if (span.all_window && span.draws > 1) {
		Inc(r.segments_window);
	}
	span.slot = -1;
}
} // namespace

void NoteNativeN3(uint32_t op_flags, uint64_t start_tsc, uint64_t end_tsc, const NativeN3Draw& d) {
	const uint32_t slot = (op_flags >> kNativeOpSlotShift) & kNativeSlotMask;
	if (slot >= kEngineThreads) {
		return;
	}
	auto&          r      = g_n3_resolve[slot];
	const uint32_t normal = d.learned + d.stream + d.dirty + d.other + d.ineligible;
	const uint32_t path   = d.hits != 0 ? (normal == 0 ? 0u : 1u) : 2u;
	Inc(r.draws[path]);
	Inc(r.ticks[path], end_tsc - start_tsc);
	const uint32_t values[kN3Fields] = {d.hits,           d.learned,       d.stream,         d.dirty,         d.other,
	                                    d.ineligible,     d.inv_epoch,     d.inv_structure,  d.inv_unregister, d.inv_signature,
	                                    d.verify_checks,  d.verify_races,  d.verify_mismatches};
	for (uint32_t k = 0; k < kN3Fields; k++) {
		if (values[k] != 0) {
			Inc(r.fields[k], values[k]);
		}
	}
	auto& span = t_n3_span;
	if ((op_flags & kNativeOpSegStart) != 0 || span.slot != static_cast<int>(slot)) {
		CloseN3Span(span);
		span.slot       = static_cast<int>(slot);
		span.draws      = 0;
		span.all_window = true;
	} else if (path != 0) {
		span.all_window = false; // a draw after the first did not take the window path
	}
	span.draws++;
}

uint64_t NativeDumpFrame() {
	static const uint64_t frame = [] {
		const char* value = std::getenv("KYTY_NATIVE_DUMP");
		return value != nullptr ? std::strtoull(value, nullptr, 10) : uint64_t {0};
	}();
	return frame;
}

const char* NativeDumpDir() {
	static const std::string dir = [] {
		const char* value = std::getenv("KYTY_NATIVE_DUMP_DIR");
		return std::string(value != nullptr && value[0] != 0 ? value : "E:\\JEUX\\ASTROBOT_PC\\Outils_Claude\\work\\native");
	}();
	return dir.c_str();
}

} // namespace Loader::GameCapture
