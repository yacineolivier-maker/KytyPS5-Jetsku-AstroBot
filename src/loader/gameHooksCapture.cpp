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
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace Loader::GameCapture {
std::atomic<bool>     g_active {false};
std::atomic<uint64_t> g_flips {0};
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
	config.valid = config.mode == "drop" && config.mask != 0;
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
	kNCount
};
constexpr uint32_t kNFb0 = kNFbIndirect;
constexpr const char* kNFbNames[] = {"indirect", "tess", "packet", "lost"};

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
			Inc(t->n[kNFbIndirect]);
		} else if ((r.geom_slot & 1u) != 0) {
			Inc(t->n[kNFbTess]);
		} else {
			t->native_window = true;
			Inc(t->n[kNChosen]);
			if (fail != 0) {
				Inc(t->n[kNIncomplete]);
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
		const auto per_frame = [&](uint64_t v) { return flips == 0 ? 0.0 : static_cast<double>(v) / static_cast<double>(flips); };
		native = fmt::format("Native 10s ({:.1f} s, mode {}): flips {}", seconds, Native().mode, flips);
		std::array<uint64_t, kNCount> total {};
		uint64_t                      total_seen = 0;
		for (uint32_t s = 0; s < kEngineThreads; s++) {
			if (((Native().mask >> s) & 1u) == 0) {
				continue;
			}
			const auto& p = per[s];
			native += fmt::format(" | \"{}\" chosen {} written {} cp-seen {} fallback", kEngineThreadNames[s], p[kNChosen], p[kNWritten], seen[s]);
			for (uint32_t k = 0; k < 4; k++) native += fmt::format(" {} {}", kNFbNames[k], p[kNFb0 + k]);
			native += fmt::format(" incomplete {} | per frame chosen {:.1f} written {:.1f} cp-seen {:.1f}", p[kNIncomplete], per_frame(p[kNChosen]),
			                      per_frame(p[kNWritten]), per_frame(seen[s]));
			for (uint32_t i = 0; i < kNCount; i++) total[i] += p[i];
			total_seen += seen[s];
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
		    "GameNative: KYTY_NATIVE {} | mode {}{} | threads:{}{}{}\n",
		    !native.valid ? "ignored" : ok ? "active (stage N0)" : "inactive (capture hooks not installed)", native.mode,
		    native.mode == "drop" ? "" : " (unsupported: only drop)", threads.empty() ? " none" : threads,
		    native.unknown.empty() ? "" : " | unknown:", native.unknown));
	}
	GameCapture::g_active.store(ok, std::memory_order_release);
}

} // namespace Loader::GameHooks::Capture

// ---- Public surface (gameCapture.h) ------------------------------------------------------------------------------------

namespace Loader::GameCapture {

using namespace Loader::GameHooks::Capture; // NOLINT(google-build-using-namespace)

namespace {

// KYTY_NATIVE: the direct draw packet becomes an R_NATIVE_DRAW NOP of the same size.
bool WriteNativeMarker(ThreadState* t, uint32_t* packet, uint32_t opcode, uint32_t size_dw) {
	namespace Pm4 = Libs::Graphics::Pm4;
	const uint32_t expected = opcode == Pm4::IT_DRAW_INDEX_2          ? 6u
	                          : opcode == Pm4::IT_DRAW_INDEX_OFFSET_2 ? 5u
	                          : opcode == Pm4::IT_DRAW_INDEX_AUTO     ? 3u
	                                                                   : 0u;
	if (packet == nullptr || expected == 0 || size_dw != expected || KYTY_PM4_LEN(packet[0]) != size_dw ||
	    ((packet[0] >> 8u) & 0xffu) != opcode) {
		return false;
	}
	const uint32_t serial = ++t->native_serial;
	packet[1]             = (kNativeMagic << 16u) | (kNativeVersion << 8u) | static_cast<uint32_t>(t->native_slot);
	packet[2]             = serial;
	for (uint32_t i = 3; i < size_dw; i++) {
		packet[i] = 0;
	}
	packet[0] = KYTY_PM4(size_dw, Pm4::IT_NOP, Pm4::R_NATIVE_DRAW);
	Inc(t->n[kNWritten]);
	return true;
}

} // namespace

void NoteDrawPacketSlow(uint32_t* packet, uint32_t opcode, uint32_t size_dw) {
	auto* t = T();
	if (t->emit_window) {
		t->emit_window = false;
		Inc(t->c[kCHleHooked]);
		if (t->native_window) [[unlikely]] {
			t->native_window = false;
			if (WriteNativeMarker(t, packet, opcode, size_dw)) {
				t->sample_pending = false; // the CP will not see this draw
				return;
			}
			Inc(t->n[kNFbPacket]);
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
		cmp(kCmpGeom, ((geom_slot & 1u) != 0 ? view.hs_addr : view.gs_addr) == (w[5] & 0xffffffffffffull));
	}
	if (bad != 0 && g_cp.logged.fetch_add(1, std::memory_order_relaxed) < 8) {
		Log::WriteToConsoleAndLog(fmt::format(
		    "GameCapture CP mismatch 0x{:x} at packet 0x{:x}: game {} count {} first {} inst {} ib 0x{:x}/{} rt0 0x{:x} ps 0x{:x} geom s{} 0x{:x} | "
		    "CP op 0x{:x} count {} first {} inst {}{} ib 0x{:x}/{} rt0 0x{:x} ps 0x{:x} gs 0x{:x} hs 0x{:x}\n",
		    bad, view.packet, kKindNames[std::min<uint32_t>(kind, 5u)], count, first, instances, w[2], ib32 ? 32 : 16, w[3], w[4], geom_slot,
		    w[5], view.opcode, view.index_count, view.index_offset, view.instances, view.instances_known ? "" : "?", view.ib_base,
		    view.ib_type, view.rt0_base, view.ps_addr, view.gs_addr, view.hs_addr));
	}
}

void NoteNativeMarker(const uint32_t* body, uint32_t body_dw) {
	uint32_t slot = kEngineThreads; // malformed
	if (body != nullptr && body_dw >= 2 && (body[0] >> 16u) == kNativeMagic &&
	    ((body[0] >> 8u) & 0xffu) == kNativeVersion && (body[0] & 0xffu) < kEngineThreads) {
		slot = body[0] & 0xffu;
	}
	g_native_seen[slot].fetch_add(1, std::memory_order_relaxed);
}

} // namespace Loader::GameCapture
