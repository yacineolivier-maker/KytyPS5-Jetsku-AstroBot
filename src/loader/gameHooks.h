#ifndef KYTY_LOADER_GAME_HOOKS_H_
#define KYTY_LOADER_GAME_HOOKS_H_

#include "common/common.h"

#include <cstddef>
#include <cstdint>
#include <vector>

// KYTY_GAME_HOOKS (off by default; x86-64 hosts): host code in place of, or around, functions
// INSIDE the main guest program (the eboot), not only its library imports.
//
//   KYTY_GAME_HOOKS=<set>[,<set>...]   built-in hook sets to register ("trace": see
//                                      gameHooksTrace.cpp); unset, "", "0" or "off" = nothing
//                                      is registered, allocated or written.
//
// A hook is registered with the function's ELF virtual address (the guest address is
// main_program->base_vaddr + elf_address) and the bytes expected there. At load time (after
// relocation and game patches, before the game's entry point) Install() checks each hook's bytes
// and skips it with a log line if they differ (other game version, or an instruction patcher
// already rewrote the prologue). The entry of the function becomes a 5-byte "jmp rel32" to a
// code cave next to the program (or, when no cave is within +-2 GiB, a 14-byte absolute jump).
//
//   Replace: the cave jumps straight to host_fn, a KYTY_SYSV_ABI function with the guest
//            function's own signature: it gets the guest's System V arguments (rdi, rsi, rdx,
//            rcx, r8, r9, xmm0-7, then the stack) and its return value (rax/rdx/xmm0) goes back to
//            the guest caller. The guest function body never runs. If the prologue can be
//            relocated, Original(id) is a callable copy of the original function.
//   Before:  host_fn (an ObserverFn) runs with a GuestCallContext holding the argument registers,
//            then the original function runs from a relocated copy of its prologue. Changes to
//            the context's argument registers are passed on; ctx.skip_original = 1 returns to the
//            caller at once with ctx.rax/rdx/ymm0 as the return value.
//   After:   the return address is diverted (per-thread shadow stack); host_fn sees the return
//            value in ctx.rax/rdx/ymm0/ymm1 and may change it.
//
// Threads: hooks run on the calling guest thread (game threads such as "Draw Geometry"), on the
// guest stack, possibly on several threads at once and re-entrantly. Host functions must be
// thread-safe, must not throw, should keep a small stack frame (no large locals) and must not
// block for long: the game thread is stalled meanwhile. Guest memory is in the same address
// space and is read with plain pointers. The registry is immutable once Install() returns.
namespace Loader {

struct Program;

namespace GameHooks {

enum class Kind { Replace, Before, After };

struct HookRecord;

// The registers saved by the hook trampolines. Layout is fixed (used by generated code).
struct GuestCallContext {
	uint64_t rdi; // 0x00  argument 1 ("this" for a member function)
	uint64_t rsi; // 0x08  argument 2
	uint64_t rdx; // 0x10  argument 3 (After: second return register)
	uint64_t rcx; // 0x18  argument 4
	uint64_t r8;  // 0x20  argument 5
	uint64_t r9;  // 0x28  argument 6
	uint64_t rax; // 0x30  Before: al = vector register count of a variadic call; After: return value
	uint64_t r10; // 0x38
	uint64_t r11; // 0x40  (read only: not restored)
	// Before: the guest rsp at the function entry ([rsp] = return address, stack arguments from
	// rsp + 8). After: the caller's rsp after the return.
	uint64_t          rsp;            // 0x48
	const HookRecord* hook;           // 0x50
	uint64_t          return_address; // 0x58  After: where the function returns to
	uint8_t           ymm[8][32];     // 0x60  ymm0-7 (vector arguments / return values)
	uint64_t          saved_rsp;      // 0x160 internal
	uint64_t          skip_original;  // 0x168 Before: set to 1 to return to the caller at once

	// Integer/pointer argument `index` (0-based) of a function without vector arguments, valid
	// in a Before hook: the first six in registers, then the stack.
	[[nodiscard]] uint64_t Arg(int index) const;
	[[nodiscard]] double   ArgDouble(int index) const; // xmm<index>, index < 8
	[[nodiscard]] uint32_t GuestThreadId() const;      // calling guest thread (pthread id)
	[[nodiscard]] const char* HookName() const;
	[[nodiscard]] void*       User() const;
};
static_assert(sizeof(GuestCallContext) == 0x170);

using ObserverFn = void (*)(GuestCallContext& ctx);

struct HookSpec {
	const char*          name        = nullptr;
	uint64_t             elf_address = 0;
	Kind                 kind        = Kind::Before;
	std::vector<uint8_t> expected; // original bytes at elf_address (at least the patch size)
	void*                replace_fn  = nullptr; // Replace: KYTY_SYSV_ABI fn, guest signature
	ObserverFn           observer_fn = nullptr; // Before/After
	void*                user        = nullptr; // GuestCallContext::User()
};

// Returns the hook id (>= 0), or -1 if rejected (logged). Call before Install(), from the main
// thread (hook sets register from Install() itself).
int Register(HookSpec spec);
int Register(uint64_t elf_address, Kind kind, void* host_fn, std::vector<uint8_t> expected,
             const char* name, void* user = nullptr);

// True when KYTY_GAME_HOOKS lists `set` (case-insensitive).
bool SetEnabled(const char* set);

// Registers the built-in sets named by KYTY_GAME_HOOKS and installs every registered hook in the
// main program. No-op when KYTY_GAME_HOOKS is off.
void Install(Program* main_program);

// After Install(): whether hook `id` is live, and a callable entry to the original function
// (Before/After always; Replace only if its prologue could be relocated), else nullptr.
// (No per-hook call counter: a shared atomic would bounce between the draw threads' cores.)
bool  IsInstalled(int id);
void* Original(int id);

void Clear();

} // namespace GameHooks

} // namespace Loader

#endif // KYTY_LOADER_GAME_HOOKS_H_
