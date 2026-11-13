// KYTY_GAME_HOOKS=trace: demonstration hook set.
//
// Astro Bot (PPSA21567): a Before hook on the engine's main draw method, ELF 0x483c20 (SysV:
// rdi = the backend context object, esi/edx/ecx/r8d/r9d = integers; it reads the context's fields
// 0x8cd0..0x8d00, 0xe88, 0xf50, 0x1030..0x1238; see docs\ETUDE-RENDU-NATIF.md). It counts the calls
// of each calling guest thread and the distinct context pointers each thread passes, and every
// 10 s prints one "GameHooks 10s:" line. One context per draw thread is expected.
//
// Cost on the hot path: one thread_local lookup, two relaxed atomic stores to a per-thread cache
// line, a scan of up to kContexts pointers and, every 64 calls, a clock read.

#include "common/logging/log.h"
#include "kernel/pthread.h"
#include "loader/gameHooks.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <fmt/format.h>
#include <mutex>
#include <string>
#include <vector>

namespace Loader::GameHooks::Trace {

namespace {

constexpr uint64_t kDrawElf  = 0x483c20;
constexpr size_t   kContexts = 4;

// push rbp; mov rbp,rsp; push r15; push r14; push r13; push r12; push rbx; sub rsp,0x18;
// mov rax,[rip+0x89cd7e0] (read from Outils_Claude\work\ghidra\eboot.elf: PT_LOAD 0 maps vaddr
// 0 at file offset 0x4000, so ELF 0x483c20 is file offset 0x487c20).
const std::vector<uint8_t> kDrawExpected = {}; // original game bytes not published (fill in locally to enable this hook)

struct alignas(64) ThreadStats {
	char                                    name[32] {};
	uint32_t                                thread_id = 0;
	std::atomic<uint64_t>                   calls {0};
	std::array<std::atomic<uint64_t>, kContexts> contexts {};
	std::atomic<uint64_t>                   more_contexts {0}; // calls with a context beyond kContexts
	uint64_t                                reported = 0;      // reporter only
};

std::mutex                g_threads_mutex;
std::vector<ThreadStats*> g_threads; // never freed: a thread may exit while being reported
thread_local ThreadStats* t_stats = nullptr;

std::atomic<int64_t> g_next_report {0};
std::mutex           g_report_mutex;
int64_t              g_last_report = 0;

int64_t NowMs() {
	return std::chrono::duration_cast<std::chrono::milliseconds>(
	           std::chrono::steady_clock::now().time_since_epoch())
	    .count();
}

ThreadStats* RegisterThread(const GuestCallContext& ctx) {
	auto* stats = new ThreadStats;
	if (auto* self = Libs::LibKernel::PthreadSelfOrNull(); self != nullptr) {
		Libs::LibKernel::PthreadGetname(self, stats->name);
	}
	if (stats->name[0] == '\0') {
		std::strncpy(stats->name, "?", sizeof(stats->name) - 1);
	}
	stats->thread_id = ctx.GuestThreadId();
	std::lock_guard lock(g_threads_mutex);
	g_threads.push_back(stats);
	return stats;
}

void Report(int64_t now) {
	std::unique_lock report(g_report_mutex, std::try_to_lock);
	if (!report.owns_lock()) {
		return;
	}
	const double seconds = g_last_report != 0 ? static_cast<double>(now - g_last_report) / 1000.0 : 10.0;
	g_last_report        = now;

	std::string            line = "GameHooks 10s: draw 0x483c20";
	uint64_t               total = 0;
	std::vector<uint64_t>  distinct;
	{
		std::lock_guard lock(g_threads_mutex);
		for (auto* stats: g_threads) {
			const auto calls = stats->calls.load(std::memory_order_relaxed);
			const auto delta = calls - stats->reported;
			stats->reported  = calls;
			if (delta == 0) {
				continue;
			}
			total += delta;
			line += fmt::format(" | \"{}\" (tid {}) {} calls ctx", stats->name, stats->thread_id, delta);
			for (const auto& slot: stats->contexts) {
				const auto value = slot.load(std::memory_order_relaxed);
				if (value == 0) {
					break;
				}
				line += fmt::format(" 0x{:x}", value);
				if (std::find(distinct.begin(), distinct.end(), value) == distinct.end()) {
					distinct.push_back(value);
				}
			}
			if (const auto more = stats->more_contexts.load(std::memory_order_relaxed); more != 0) {
				line += fmt::format(" (+{} calls with other contexts)", more);
			}
		}
	}
	line = fmt::format("{} | total {} calls ({:.0f}/s), {} distinct contexts\n", line, total,
	                   static_cast<double>(total) / seconds, distinct.size());
	Log::WriteToConsoleAndLog(line);
}

void OnDraw(GuestCallContext& ctx) {
	auto* stats = t_stats;
	if (stats == nullptr) {
		stats   = RegisterThread(ctx);
		t_stats = stats;
	}
	const auto calls = stats->calls.load(std::memory_order_relaxed) + 1;
	stats->calls.store(calls, std::memory_order_relaxed); // single writer: this thread

	const uint64_t context = ctx.rdi;
	bool           known   = false;
	for (auto& slot: stats->contexts) {
		const auto value = slot.load(std::memory_order_relaxed);
		if (value == context) {
			known = true;
			break;
		}
		if (value == 0) {
			slot.store(context, std::memory_order_relaxed);
			known = true;
			break;
		}
	}
	if (!known) {
		stats->more_contexts.fetch_add(1, std::memory_order_relaxed);
	}

	if ((calls & 63u) == 0) {
		const auto now  = NowMs();
		auto       next = g_next_report.load(std::memory_order_relaxed);
		if (next == 0) {
			g_next_report.compare_exchange_strong(next, now + 10000);
		} else if (now >= next && g_next_report.compare_exchange_strong(next, now + 10000)) {
			Report(now);
		}
	}
}

} // namespace

void RegisterHooks() {
	Register(kDrawElf, Kind::Before, reinterpret_cast<void*>(&OnDraw), kDrawExpected,
	         "astrobot_draw_trace");
}

} // namespace Loader::GameHooks::Trace
