#ifndef KYTY_LOADER_GAME_HOOKS_CODEGEN_H_
#define KYTY_LOADER_GAME_HOOKS_CODEGEN_H_

// Machine code for KYTY_GAME_HOOKS (gameHooks.cpp): a length decoder for position-independent
// prologue instructions and the x86-64 trampolines. Standard library only, so that it can be
// tested on its own (the trampolines are exercised by a host-only test outside the emulator).

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

namespace Loader::GameHooks::Codegen {

constexpr size_t kRel32JumpSize = 5;
constexpr size_t kAbsJumpSize   = 14; // jmp [rip+0]; dq target

// Offsets in the trampoline frame (GuestCallContext layout, gameHooks.h).
constexpr uint32_t kFrameSize      = 0x170;
constexpr uint32_t kSavedRspOffset = 0x160;
constexpr uint32_t kSkipOffset     = 0x168;

inline std::string Hex8(uint8_t value) {
	static constexpr char kDigits[] = "0123456789ABCDEF";
	return std::string {kDigits[value >> 4u], kDigits[value & 15u]};
}

// ---- Minimal x86-64 length decoder for position-independent prologue instructions ----------
//
// Accepted: endbr64, nop, push/pop reg, mov reg,imm, and ModRM ALU/mov/lea/test/cmp forms
// (01 03 09 0B 21 23 29 2B 31 33 39 3B 85 89 8B 8D, 83 ib, 81 id, C7 /0 id, 0F 1F nop) whose
// memory operand is not RIP-relative. Everything else (branches, calls, RIP-relative operands,
// operand-size/segment prefixes, VEX...) is refused: such bytes cannot be copied verbatim.
// Returns the instruction length, or 0 with *why set.
inline size_t ModRmLength(const uint8_t* p, size_t avail, std::string* why) {
	if (avail < 1) {
		*why = "truncated ModRM";
		return 0;
	}
	const uint8_t modrm = p[0];
	const uint8_t mod   = modrm >> 6u;
	const uint8_t rm    = modrm & 7u;
	size_t        len   = 1;
	if (mod == 3) {
		return len;
	}
	if (rm == 4) {
		if (avail < 2) {
			*why = "truncated SIB";
			return 0;
		}
		const uint8_t base = p[1] & 7u;
		len++;
		if (mod == 0 && base == 5) {
			len += 4; // [index*scale + disp32]: absolute, position independent
		}
	} else if (mod == 0 && rm == 5) {
		*why = "RIP-relative operand";
		return 0;
	}
	if (mod == 1) {
		len += 1;
	} else if (mod == 2) {
		len += 4;
	}
	return len;
}

inline size_t RelocatableLength(const uint8_t* p, size_t avail, std::string* why) {
	if (avail >= 4 && p[0] == 0xF3 && p[1] == 0x0F && p[2] == 0x1E && p[3] == 0xFA) {
		return 4; // endbr64
	}
	size_t  pos = 0;
	uint8_t rex = 0;
	if (avail > 0 && (p[0] & 0xF0u) == 0x40) {
		rex = p[0];
		pos = 1;
	}
	if (pos >= avail) {
		*why = "truncated instruction";
		return 0;
	}
	const uint8_t op    = p[pos++];
	const bool    rex_w = (rex & 0x08u) != 0;

	if (op == 0x90 && rex == 0) {
		return 1;
	}
	if (op >= 0x50 && op <= 0x5F) {
		return pos; // push / pop r64
	}
	if (op >= 0xB8 && op <= 0xBF) {
		const size_t len = pos + (rex_w ? 8 : 4);
		if (len > avail) {
			*why = "truncated immediate";
			return 0;
		}
		return len;
	}
	size_t imm = 0;
	switch (op) {
		case 0x01:
		case 0x03:
		case 0x09:
		case 0x0B:
		case 0x21:
		case 0x23:
		case 0x29:
		case 0x2B:
		case 0x31:
		case 0x33:
		case 0x39:
		case 0x3B:
		case 0x85:
		case 0x89:
		case 0x8B:
		case 0x8D: imm = 0; break;
		case 0x83: imm = 1; break;
		case 0x81: imm = 4; break;
		case 0xC7:
			if (pos >= avail || ((p[pos] >> 3u) & 7u) != 0) {
				*why = "unsupported C7 form";
				return 0;
			}
			imm = 4;
			break;
		case 0x0F:
			if (pos < avail && p[pos] == 0x1F) {
				pos++;
				imm = 0;
				break;
			}
			*why = "unsupported opcode 0F " + Hex8(pos < avail ? p[pos] : 0);
			return 0;
		default: *why = "unsupported opcode " + Hex8(op); return 0;
	}
	const size_t modrm = ModRmLength(p + pos, avail - pos, why);
	if (modrm == 0) {
		return 0;
	}
	const size_t len = pos + modrm + imm;
	if (len > avail) {
		*why = "truncated instruction";
		return 0;
	}
	return len;
}

// Whole instructions covering at least `needed` bytes of `bytes`; 0 with *why if impossible.
inline size_t RelocatablePrologue(const std::vector<uint8_t>& bytes, size_t needed, std::string* why) {
	size_t total = 0;
	while (total < needed) {
		std::string  reason;
		const size_t len = RelocatableLength(bytes.data() + total, bytes.size() - total, &reason);
		if (len == 0) {
			*why = "+" + std::to_string(total) + ": " + reason;
			return 0;
		}
		total += len;
	}
	return total;
}

// ---- Code emission ----------------------------------------------------------------------------

struct Code {
	std::vector<uint8_t> bytes;

	void B(std::initializer_list<uint8_t> list) { bytes.insert(bytes.end(), list); }
	void U32(uint32_t value) {
		const auto* p = reinterpret_cast<const uint8_t*>(&value);
		bytes.insert(bytes.end(), p, p + sizeof(value));
	}
	void U64(uint64_t value) {
		const auto* p = reinterpret_cast<const uint8_t*>(&value);
		bytes.insert(bytes.end(), p, p + sizeof(value));
	}
	void AbsJump(uint64_t target) {
		B({0xFF, 0x25, 0x00, 0x00, 0x00, 0x00}); // jmp [rip+0]
		U64(target);
	}
};

// Frame on the 16-byte aligned stack: a GuestCallContext (0x170 bytes) at rsp. On entry to the
// block, r11 holds (address of the pushed original r11); ctx.rsp = r11 + rsp_bias.
inline void EmitSaveCallRestore(Code* c, uint8_t rsp_bias, uint64_t hook, uint64_t dispatcher) {
	c->B({0x48, 0x83, 0xE4, 0xF0});                   // and rsp, -16
	c->B({0x48, 0x81, 0xEC, 0x70, 0x01, 0x00, 0x00}); // sub rsp, 0x170
	c->B({0x48, 0x89, 0x3C, 0x24});                   // mov [rsp], rdi
	c->B({0x48, 0x89, 0x74, 0x24, 0x08});             // mov [rsp+0x08], rsi
	c->B({0x48, 0x89, 0x54, 0x24, 0x10});             // mov [rsp+0x10], rdx
	c->B({0x48, 0x89, 0x4C, 0x24, 0x18});             // mov [rsp+0x18], rcx
	c->B({0x4C, 0x89, 0x44, 0x24, 0x20});             // mov [rsp+0x20], r8
	c->B({0x4C, 0x89, 0x4C, 0x24, 0x28});             // mov [rsp+0x28], r9
	c->B({0x48, 0x89, 0x44, 0x24, 0x30});             // mov [rsp+0x30], rax
	c->B({0x4C, 0x89, 0x54, 0x24, 0x38});             // mov [rsp+0x38], r10
	c->B({0x49, 0x8B, 0x03});                         // mov rax, [r11]  (original r11)
	c->B({0x48, 0x89, 0x44, 0x24, 0x40});             // mov [rsp+0x40], rax
	c->B({0x49, 0x8D, 0x43, rsp_bias});               // lea rax, [r11+bias]
	c->B({0x48, 0x89, 0x44, 0x24, 0x48});             // mov [rsp+0x48], rax
	c->B({0x48, 0xB8});                               // mov rax, hook
	c->U64(hook);
	c->B({0x48, 0x89, 0x44, 0x24, 0x50});                         // mov [rsp+0x50], rax
	c->B({0x48, 0xC7, 0x44, 0x24, 0x58, 0x00, 0x00, 0x00, 0x00}); // mov qword [rsp+0x58], 0
	for (uint8_t i = 0; i < 8; i++) {
		c->B({0xC5, 0xFE, 0x7F, static_cast<uint8_t>(0x84u | (i << 3u)), 0x24}); // vmovdqu [rsp+d], ymmi
		c->U32(0x60u + 32u * i);
	}
	c->B({0x4C, 0x89, 0x9C, 0x24, 0x60, 0x01, 0x00, 0x00}); // mov [rsp+0x160], r11
	c->B({0x48, 0xC7, 0x84, 0x24, 0x68, 0x01, 0x00, 0x00}); // mov qword [rsp+0x168], 0
	c->U32(0);
	c->B({0x48, 0x89, 0xE7}); // mov rdi, rsp
	c->B({0x48, 0xB8});       // mov rax, dispatcher
	c->U64(dispatcher);
	c->B({0xFF, 0xD0}); // call rax
	for (uint8_t i = 0; i < 8; i++) {
		c->B({0xC5, 0xFE, 0x6F, static_cast<uint8_t>(0x84u | (i << 3u)), 0x24}); // vmovdqu ymmi, [rsp+d]
		c->U32(0x60u + 32u * i);
	}
	c->B({0x48, 0x8B, 0x3C, 0x24});       // mov rdi, [rsp]
	c->B({0x48, 0x8B, 0x74, 0x24, 0x08}); // mov rsi, [rsp+0x08]
	c->B({0x48, 0x8B, 0x54, 0x24, 0x10}); // mov rdx, [rsp+0x10]
	c->B({0x48, 0x8B, 0x4C, 0x24, 0x18}); // mov rcx, [rsp+0x18]
	c->B({0x4C, 0x8B, 0x44, 0x24, 0x20}); // mov r8, [rsp+0x20]
	c->B({0x4C, 0x8B, 0x4C, 0x24, 0x28}); // mov r9, [rsp+0x28]
	c->B({0x4C, 0x8B, 0x54, 0x24, 0x38}); // mov r10, [rsp+0x38]
	c->B({0x48, 0x8B, 0x44, 0x24, 0x30}); // mov rax, [rsp+0x30]
}

// Hook entry (Before/After), reached by the jump written at the guest function entry.
inline void EmitEntryTrampoline(Code* c, uint64_t hook, uint64_t dispatcher, uint64_t original) {
	c->B({0x41, 0x53});       // push r11
	c->B({0x49, 0x89, 0xE3}); // mov r11, rsp
	EmitSaveCallRestore(c, 8, hook, dispatcher);
	c->B({0x80, 0xBC, 0x24, 0x68, 0x01, 0x00, 0x00, 0x00}); // cmp byte [rsp+0x168], 0
	c->B({0x48, 0x8B, 0xA4, 0x24, 0x60, 0x01, 0x00, 0x00}); // mov rsp, [rsp+0x160]
	c->B({0x41, 0x5B});                                     // pop r11 (flags kept)
	c->B({0x75, static_cast<uint8_t>(kAbsJumpSize)});       // jne skip
	c->AbsJump(original);                                   // run the original function
	c->B({0xC3});                                           // skip: ret (ctx.rax...)
}

// Shared After return stub: the diverted return address of every After hook.
inline void EmitReturnStub(Code* c, uint64_t dispatcher) {
	c->B({0x48, 0x83, 0xEC, 0x08}); // sub rsp, 8   (slot for the real return address)
	c->B({0x41, 0x53});             // push r11
	c->B({0x49, 0x89, 0xE3});       // mov r11, rsp
	EmitSaveCallRestore(c, 16, 0, dispatcher);
	c->B({0x48, 0x8B, 0xA4, 0x24, 0x60, 0x01, 0x00, 0x00}); // mov rsp, [rsp+0x160]
	c->B({0x41, 0x5B});                                     // pop r11
	c->B({0xC3});                                           // ret -> real return address
}

} // namespace Loader::GameHooks::Codegen

#endif // KYTY_LOADER_GAME_HOOKS_CODEGEN_H_
