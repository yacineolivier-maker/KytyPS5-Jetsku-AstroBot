#include "loader/gameHooks.h"

#include "common/abi.h"
#include "common/assert.h"
#include "common/logging/log.h"
#include "common/stringUtils.h"
#include "common/virtualMemory.h"
#include "kernel/memory.h"
#include "kernel/pthread.h"
#include "loader/elf.h"
#include "loader/gameHooksCodegen.h"
#include "loader/runtimeLinker.h"

#include <algorithm>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <initializer_list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace Loader::GameHooks {

namespace Trace {
void RegisterHooks(); // gameHooksTrace.cpp
} // namespace Trace
namespace Capture {
void RegisterHooks();  // gameHooksCapture.cpp
void OnInstalled();    // gameHooksCapture.cpp: after Install() placed the hooks
} // namespace Capture

struct HookRecord {
	int         id = -1;
	HookSpec    spec;
	std::string name;
	uint64_t    guest_address = 0;
	bool        installed     = false;
	void*       original      = nullptr; // relocated prologue + jump back
};

namespace {

using namespace Codegen; // NOLINT(google-build-using-namespace)

#if defined(__x86_64__) || defined(_M_X64)
constexpr bool kSupportedHost = true;
#else
constexpr bool kSupportedHost = false;
#endif

constexpr uint64_t kPageSize       = 0x4000;
constexpr size_t   kBytesPerHook   = 768;
constexpr size_t   kSharedCodeSize = 512;

std::vector<std::unique_ptr<HookRecord>> g_hooks;
uint64_t                                 g_cave_addr = 0;
uint64_t                                 g_cave_size = 0;
uint64_t                                 g_return_stub = 0;
bool                                     g_installed   = false;

const char* KindName(Kind kind) {
	switch (kind) {
		case Kind::Replace: return "Replace";
		case Kind::Before: return "Before";
		case Kind::After: return "After";
	}
	return "?";
}

std::vector<std::string> EnvTokens() {
	std::vector<std::string> tokens;
	const char*              value = std::getenv("KYTY_GAME_HOOKS");
	if (value == nullptr) {
		return tokens;
	}
	std::string current;
	for (const char* p = value;; p++) {
		if (*p == '\0' || *p == ',' || *p == ';' || *p == ' ') {
			if (!current.empty()) {
				tokens.push_back(Common::ToLower(current));
			}
			current.clear();
			if (*p == '\0') {
				break;
			}
		} else {
			current += *p;
		}
	}
	return tokens;
}

bool Enabled() {
	const auto tokens = EnvTokens();
	return std::any_of(tokens.begin(), tokens.end(), [](const std::string& token) {
		return token != "0" && token != "off" && token != "none";
	});
}

// ---- Dispatch (called from generated code, on the guest thread and stack) --------------------

struct ShadowEntry {
	uint64_t          return_address;
	uint64_t          caller_rsp; // rsp after the return
	const HookRecord* hook;
};

thread_local std::vector<ShadowEntry> t_shadow;

KYTY_SYSV_ABI void DispatchEnter(GuestCallContext* ctx) {
	const auto* hook = ctx->hook;
	if (hook->spec.kind == Kind::Before) {
		hook->spec.observer_fn(*ctx);
		return;
	}
	// After
	if (hook->spec.entry_fn != nullptr) {
		hook->spec.entry_fn(*ctx);
	}
	auto* ret_slot = reinterpret_cast<uint64_t*>(ctx->rsp);
	t_shadow.push_back({*ret_slot, ctx->rsp + 8, hook});
	*ret_slot = g_return_stub;
}

KYTY_SYSV_ABI void DispatchExit(GuestCallContext* ctx) {
	// Frames skipped by longjmp/unwinding left deeper (lower rsp) entries behind.
	while (!t_shadow.empty() && t_shadow.back().caller_rsp < ctx->rsp) {
		t_shadow.pop_back();
	}
	if (t_shadow.empty() || t_shadow.back().caller_rsp != ctx->rsp) {
		EXIT("GameHooks: After return stub reached without a matching entry (rsp=0x%016" PRIx64
		     ")\n",
		     ctx->rsp);
	}
	const auto entry = t_shadow.back();
	t_shadow.pop_back();
	ctx->hook           = entry.hook;
	ctx->return_address = entry.return_address;
	entry.hook->spec.observer_fn(*ctx);
	*reinterpret_cast<uint64_t*>(ctx->rsp - 8) = entry.return_address;
}

// ---- Installation ------------------------------------------------------------------------------

bool InExecutableSegment(const Program& program, uint64_t elf_address, size_t size) {
	const auto* ehdr = program.elf->GetEhdr();
	const auto* phdr = program.elf->GetPhdr();
	for (Elf64_Half i = 0; i < ehdr->e_phnum; i++) {
		if (phdr[i].p_type == PT_LOAD && (phdr[i].p_flags & PF_X) != 0 &&
		    elf_address >= phdr[i].p_vaddr && elf_address + size <= phdr[i].p_vaddr + phdr[i].p_filesz) {
			return true;
		}
	}
	return false;
}

void Print(const std::string& text) {
	Log::WriteToConsoleAndLog(text);
}

bool FitsRel32(uint64_t from_next, uint64_t to) {
	const auto delta = static_cast<int64_t>(to - from_next);
	return delta >= INT32_MIN && delta <= INT32_MAX;
}

} // namespace

// ---- GuestCallContext helpers ------------------------------------------------------------------

uint64_t GuestCallContext::Arg(int index) const {
	switch (index) {
		case 0: return rdi;
		case 1: return rsi;
		case 2: return rdx;
		case 3: return rcx;
		case 4: return r8;
		case 5: return r9;
		default: return *reinterpret_cast<const uint64_t*>(rsp + 8 + 8 * static_cast<uint64_t>(index - 6));
	}
}

double GuestCallContext::ArgDouble(int index) const {
	double value = 0.0;
	if (index >= 0 && index < 8) {
		std::memcpy(&value, ymm[index], sizeof(value));
	}
	return value;
}

uint32_t GuestCallContext::GuestThreadId() const {
	return static_cast<uint32_t>(Libs::LibKernel::PthreadGetthreadid());
}

const char* GuestCallContext::HookName() const {
	return hook != nullptr ? hook->name.c_str() : "";
}

void* GuestCallContext::User() const {
	return hook != nullptr ? hook->spec.user : nullptr;
}

// ---- Public API --------------------------------------------------------------------------------

bool SetEnabled(const char* set) {
	const auto tokens = EnvTokens();
	const auto wanted = Common::ToLower(set);
	return std::find(tokens.begin(), tokens.end(), wanted) != tokens.end();
}

int Register(HookSpec spec) {
	const std::string name = spec.name != nullptr ? spec.name : fmt::format("0x{:x}", spec.elf_address);
	const auto        reject = [&](const char* why) {
        Print(fmt::format("GameHooks: rejected {} at ELF 0x{:x}: {}\n", name, spec.elf_address, why));
        return -1;
	};
	if (g_installed) {
		return reject("registered after Install()");
	}
	if (spec.kind == Kind::Replace ? spec.replace_fn == nullptr : spec.observer_fn == nullptr) {
		return reject("no host function for this kind");
	}
	if (spec.expected.size() < kRel32JumpSize) {
		return reject("fewer than 5 expected bytes");
	}
	for (const auto& hook: g_hooks) {
		if (hook->spec.elf_address == spec.elf_address) {
			return reject("address already hooked");
		}
	}
	auto record  = std::make_unique<HookRecord>();
	record->id   = static_cast<int>(g_hooks.size());
	record->name = name;
	record->spec = std::move(spec);
	g_hooks.push_back(std::move(record));
	return g_hooks.back()->id;
}

int Register(uint64_t elf_address, Kind kind, void* host_fn, std::vector<uint8_t> expected,
             const char* name, void* user) {
	HookSpec spec;
	spec.name        = name;
	spec.elf_address = elf_address;
	spec.kind        = kind;
	spec.expected    = std::move(expected);
	spec.user        = user;
	if (kind == Kind::Replace) {
		spec.replace_fn = host_fn;
	} else {
		spec.observer_fn = reinterpret_cast<ObserverFn>(host_fn);
	}
	return Register(std::move(spec));
}

void Install(Program* main_program) {
	if (g_installed || !Enabled()) {
		return;
	}
	if (!kSupportedHost) {
		Print("GameHooks: KYTY_GAME_HOOKS ignored (x86-64 host only)\n");
		return;
	}
	if (SetEnabled("trace")) {
		Trace::RegisterHooks();
	}
	const bool capture = SetEnabled("capture");
	if (capture) {
		Capture::RegisterHooks();
	}
	g_installed = true;
	if (main_program == nullptr || main_program->elf == nullptr || main_program->elf->IsShared() ||
	    main_program->base_vaddr == 0) {
		Print("GameHooks: no main program, nothing installed\n");
		return;
	}
	if (g_hooks.empty()) {
		Print("GameHooks: KYTY_GAME_HOOKS set but no hook registered\n");
		return;
	}

	// Validate every hook against the loaded image.
	std::vector<HookRecord*> valid;
	for (auto& hook: g_hooks) {
		const auto& spec    = hook->spec;
		hook->guest_address = main_program->base_vaddr + spec.elf_address;
		if (!InExecutableSegment(*main_program, spec.elf_address, spec.expected.size())) {
			Print(fmt::format("GameHooks: skipped {}: ELF 0x{:x} is not in an executable segment\n",
			                hook->name, spec.elf_address));
			continue;
		}
		const auto* live = reinterpret_cast<const uint8_t*>(hook->guest_address);
		if (std::memcmp(live, spec.expected.data(), spec.expected.size()) != 0) {
			std::string found;
			for (size_t i = 0; i < spec.expected.size(); i++) {
				found += fmt::format("{:02x}", live[i]);
			}
			Print(fmt::format("GameHooks: skipped {}: bytes at ELF 0x{:x} differ (found {})\n",
			                hook->name, spec.elf_address, found));
			continue;
		}
		valid.push_back(hook.get());
	}
	if (valid.empty()) {
		return;
	}

	// One code cave, searched for just past the program so that "jmp rel32" reaches it.
	const auto cave_size =
	    (kSharedCodeSize + kBytesPerHook * valid.size() + kPageSize - 1) & ~(kPageSize - 1);
	const auto search =
	    (main_program->base_vaddr + main_program->mapped_size + kPageSize - 1) & ~(kPageSize - 1);
	g_cave_addr = Libs::LibKernel::Memory::AllocateRuntimeMemory(
	    search, cave_size, Common::VirtualMemory::Mode::ExecuteReadWrite, "game_hooks_cave");
	if (g_cave_addr == 0) {
		Print("GameHooks: could not allocate the code cave, nothing installed\n");
		return;
	}
	g_cave_size = cave_size;
	uint64_t cursor = g_cave_addr;
	const auto place = [&](const Code& code) {
		const auto at = cursor;
		EXIT_IF(at + code.bytes.size() > g_cave_addr + g_cave_size);
		std::memcpy(reinterpret_cast<void*>(at), code.bytes.data(), code.bytes.size());
		cursor = (at + code.bytes.size() + 15) & ~uint64_t {15};
		return at;
	};

	if (std::any_of(valid.begin(), valid.end(), [](const HookRecord* h) { return h->spec.kind == Kind::After; })) {
		Code stub;
		EmitReturnStub(&stub, reinterpret_cast<uint64_t>(&DispatchExit));
		g_return_stub = place(stub);
	}

	for (auto* hook: valid) {
		const auto& spec       = hook->spec;
		const auto  guest      = hook->guest_address;
		// The entry patch: rel32 if the cave is in reach of the whole cave range, else absolute.
		const bool  near       = FitsRel32(guest + kRel32JumpSize, g_cave_addr) &&
		                         FitsRel32(guest + kRel32JumpSize, g_cave_addr + g_cave_size);
		const auto  patch_size = near ? kRel32JumpSize : kAbsJumpSize;
		if (spec.expected.size() < patch_size) {
			Print(fmt::format("GameHooks: skipped {}: {} expected bytes, the entry jump needs {}\n",
			                hook->name, spec.expected.size(), patch_size));
			continue;
		}
		std::string  why;
		const size_t relocated = RelocatablePrologue(spec.expected, patch_size, &why);
		if (relocated == 0 && spec.kind != Kind::Replace) {
			Print(fmt::format("GameHooks: skipped {} ({}): prologue not relocatable at {}\n", hook->name,
			                KindName(spec.kind), why));
			continue;
		}

		// Original: relocated prologue, then back into the function.
		uint64_t original = 0;
		if (relocated != 0) {
			Code code;
			code.bytes.assign(spec.expected.begin(), spec.expected.begin() + static_cast<ptrdiff_t>(relocated));
			code.AbsJump(guest + relocated);
			original = place(code);
		}

		uint64_t target = 0;
		if (spec.kind == Kind::Replace) {
			Code code;
			code.AbsJump(reinterpret_cast<uint64_t>(spec.replace_fn));
			target = place(code);
		} else {
			Code code;
			EmitEntryTrampoline(&code, reinterpret_cast<uint64_t>(hook), reinterpret_cast<uint64_t>(&DispatchEnter),
			                    original);
			target = place(code);
		}

		// Patch the guest entry. Program pages are host read-write-execute (as for game patches).
		const size_t written = std::max(patch_size, relocated);
		std::vector<uint8_t> patch(written, 0xCC);
		if (near) {
			const auto rel = static_cast<int32_t>(static_cast<int64_t>(target - (guest + kRel32JumpSize)));
			patch[0]       = 0xE9;
			std::memcpy(patch.data() + 1, &rel, sizeof(rel));
		} else {
			Code code;
			code.AbsJump(target);
			std::copy(code.bytes.begin(), code.bytes.end(), patch.begin());
		}
		std::memcpy(reinterpret_cast<void*>(guest), patch.data(), patch.size());
		Common::VirtualMemory::FlushInstructionCache(guest, patch.size());

		hook->installed = true;
		hook->original  = reinterpret_cast<void*>(original);
		Print(fmt::format("GameHooks: installed {} ({}) at ELF 0x{:x} = 0x{:x}: {} jump, {}\n", hook->name,
		                KindName(spec.kind), spec.elf_address, guest, near ? "rel32" : "absolute",
		                relocated != 0 ? fmt::format("{} prologue bytes relocated", relocated)
		                               : fmt::format("no original entry ({})", why)));
	}
	Common::VirtualMemory::FlushInstructionCache(g_cave_addr, g_cave_size);
	if (capture) {
		Capture::OnInstalled();
	}
}

bool IsInstalled(int id) {
	return id >= 0 && id < static_cast<int>(g_hooks.size()) && g_hooks[id]->installed;
}

void* Original(int id) {
	return IsInstalled(id) ? g_hooks[id]->original : nullptr;
}

void Clear() {
	if (g_cave_addr != 0) {
		EXIT_IF(!Libs::LibKernel::Memory::FreeGuestMemory(g_cave_addr, g_cave_size));
	}
	g_cave_addr   = 0;
	g_cave_size   = 0;
	g_return_stub = 0;
	g_hooks.clear();
	g_installed = false;
}

} // namespace Loader::GameHooks
