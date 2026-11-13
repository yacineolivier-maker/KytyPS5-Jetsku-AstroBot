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
	uint64_t hs_addr      = 0; // SPI_SHADER_PGM_LO/HI_HS
};

void CheckCpDrawSlow(const CpDrawView& view);
inline void CheckCpDraw(const CpDrawView& view) {
	if (Active()) {
		CheckCpDrawSlow(view);
	}
}

} // namespace Loader::GameCapture

#endif // KYTY_LOADER_GAME_CAPTURE_H_
