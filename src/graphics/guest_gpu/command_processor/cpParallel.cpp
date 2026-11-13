// Astro Bot PC (KYTY_CP_PAR): stage 1a of the parallel translation of the command lists
// (cpParallel.h, docs\PLAN-PARALLELE.md).

#include "graphics/guest_gpu/command_processor/cpParallel.h"

#include "common/assert.h"
#include "common/hangWatchdog.h"
#include "common/profiler.h"
#include "common/threads.h"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>

namespace Libs::Graphics::CpPar {

namespace {

struct Stats {
	std::array<std::atomic<uint64_t>, 2> completed {};
	std::array<std::atomic<uint64_t>, 2> resolve_ns {};
	std::array<std::atomic<uint64_t>, static_cast<size_t>(Fallback::Count)> fallbacks {};
	std::atomic<uint64_t> slices_b {0};        // RunSlice hand-offs
	std::atomic<uint64_t> bounces_service {0}; // service commands run on Thread_Gpu for B
	std::atomic<uint64_t> bounces_flip {0};    // Flip / WaitFlipDone ops run on Thread_Gpu for B
	// KYTY_CP_PAR=1b1 (NotePrep): per context, committed / fallen-back draws since the last line,
	// the window totals (published, retired) now, and at the last line (Thread_Gpu only).
	std::array<std::atomic<uint64_t>, 2> prep_committed {};
	std::array<std::atomic<uint64_t>, 2> prep_fell_back {};
	std::array<std::atomic<uint64_t>, 2> window_published {};
	std::array<std::atomic<uint64_t>, 2> window_retired {};
	std::array<uint64_t, 2>              reported_published {};
	std::array<uint64_t, 2>              reported_retired {};
};

Stats& GetStats() {
	static Stats stats;
	return stats;
}

thread_local bool t_resolver2 = false;

// The token: exactly one of Thread_Gpu and resolver #2 runs CP work at any time.
class Resolver2 {
public:
	explicit Resolver2(const Hooks& hooks): m_hooks(hooks) {
		m_thread = std::thread([this] { Run(); });
	}
	~Resolver2() {
		{
			std::lock_guard lock(m_mutex);
			EXIT_IF(m_state != State::Idle);
			m_state = State::Stop;
		}
		m_wake_b.notify_all();
		if (m_thread.joinable()) {
			m_thread.join();
		}
	}
	Resolver2(const Resolver2&)            = delete;
	Resolver2& operator=(const Resolver2&) = delete;

	// Thread_Gpu.
	void RunSlice(Task task, void* argument) {
		EXIT_IF(t_resolver2);
		Give();
		{
			std::unique_lock lock(m_mutex);
			EXIT_IF(m_state != State::Idle);
			m_task     = task;
			m_argument = argument;
			m_state    = State::Slice;
			m_wake_b.notify_one();
			HangWatchdog::Scope wait("cp-par-token", reinterpret_cast<uint64_t>(this));
			for (;;) {
				m_wake_a.wait(lock, [this] {
					return m_state == State::SliceDone || m_state == State::Request;
				});
				if (m_state == State::SliceDone) {
					break;
				}
				// Resolver #2 waits in RunOnGpuThread: run its request with the token.
				const auto request          = m_request;
				auto*      request_argument = m_request_argument;
				lock.unlock();
				Take();
				request(request_argument);
				Give();
				lock.lock();
				m_state = State::Slice;
				m_wake_b.notify_one();
			}
			m_state = State::Idle;
		}
		Take();
	}

	// Resolver #2, inside a slice.
	void RunOnGpuThread(Task task, void* argument) {
		EXIT_IF(!t_resolver2);
		Give();
		{
			std::unique_lock lock(m_mutex);
			EXIT_IF(m_state != State::Slice);
			m_request          = task;
			m_request_argument = argument;
			m_state            = State::Request;
			m_wake_a.notify_one();
			m_wake_b.wait(lock, [this] { return m_state == State::Slice; });
		}
		Take();
	}

private:
	enum class State : uint8_t { Idle, Slice, Request, SliceDone, Stop };

	void Take() { m_hooks.take(m_hooks.context); }
	void Give() { m_hooks.give(m_hooks.context); }

	void Run() {
		t_resolver2 = true;
		KYTY_PROFILER_THREAD("CP resolver #2");
		// Like Thread_Gpu, whose work it does in turn.
		Common::RaiseCurrentThreadPriority();
		std::unique_lock lock(m_mutex);
		for (;;) {
			m_wake_b.wait(lock, [this] { return m_state == State::Slice || m_state == State::Stop; });
			if (m_state == State::Stop) {
				return;
			}
			const auto task     = m_task;
			auto*      argument = m_argument;
			lock.unlock();
			Take();
			task(argument);
			Give();
			lock.lock();
			m_state = State::SliceDone;
			m_wake_a.notify_one();
		}
	}

	Hooks                   m_hooks;
	std::mutex              m_mutex;
	std::condition_variable m_wake_a; // Thread_Gpu waits on it
	std::condition_variable m_wake_b; // resolver #2 waits on it
	State                   m_state            = State::Idle;
	Task                    m_task             = nullptr;
	void*                   m_argument         = nullptr;
	Task                    m_request          = nullptr;
	void*                   m_request_argument = nullptr;
	std::thread             m_thread;
};

std::unique_ptr<Resolver2> g_resolver2;

} // namespace

Mode GetMode() {
	static const Mode mode = [] {
		const auto* value = std::getenv("KYTY_CP_PAR");
		if (value == nullptr || *value == '\0' || std::strcmp(value, "0") == 0) {
			return Mode::Off;
		}
		if (std::strcmp(value, "1a") == 0) {
			std::printf("Kyty CpPar: KYTY_CP_PAR=1a (two resolver contexts, executed in turn)\n");
			return Mode::Serial1a;
		}
		if (std::strcmp(value, "1b1") == 0) {
			std::printf("Kyty CpPar: KYTY_CP_PAR=1b1 (two resolver contexts, executed in turn, each "
			            "with its own snapshot ring and draw-prep window)\n");
			return Mode::Serial1b1;
		}
		std::printf("Kyty CpPar: KYTY_CP_PAR=%s is not supported (only 1a, 1b1); off\n", value);
		return Mode::Off;
	}();
	return mode;
}

uint64_t NowNs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

void Start(const Hooks& hooks) {
	EXIT_IF(g_resolver2 != nullptr || hooks.take == nullptr || hooks.give == nullptr);
	g_resolver2 = std::make_unique<Resolver2>(hooks);
	std::printf("Kyty CpPar: CP resolver #2 started\n");
}

void Stop() {
	g_resolver2.reset();
}

bool Started() {
	return g_resolver2 != nullptr;
}

void RunSlice(Task task, void* argument) {
	EXIT_IF(g_resolver2 == nullptr);
	GetStats().slices_b.fetch_add(1, std::memory_order_relaxed);
	g_resolver2->RunSlice(task, argument);
}

void RunOnGpuThread(Task task, void* argument) {
	EXIT_IF(g_resolver2 == nullptr);
	g_resolver2->RunOnGpuThread(task, argument);
}

bool OnResolver2() noexcept {
	return t_resolver2;
}

void NoteFallback(Fallback reason) {
	GetStats().fallbacks[static_cast<size_t>(reason)].fetch_add(1, std::memory_order_relaxed);
}

void NoteCompleted(uint32_t context) {
	GetStats().completed[context != 0 ? 1 : 0].fetch_add(1, std::memory_order_relaxed);
}

void NoteResolveTime(uint32_t context, uint64_t ns) {
	GetStats().resolve_ns[context != 0 ? 1 : 0].fetch_add(ns, std::memory_order_relaxed);
}

void NoteBounce(bool flip_op) {
	(flip_op ? GetStats().bounces_flip : GetStats().bounces_service)
	    .fetch_add(1, std::memory_order_relaxed);
}

void NotePrep(uint32_t context, uint64_t committed, uint64_t fell_back, uint64_t published,
              uint64_t retired) {
	const size_t index = context != 0 ? 1 : 0;
	auto&        stats = GetStats();
	stats.prep_committed[index].fetch_add(committed, std::memory_order_relaxed);
	stats.prep_fell_back[index].fetch_add(fell_back, std::memory_order_relaxed);
	stats.window_published[index].store(published, std::memory_order_relaxed);
	stats.window_retired[index].store(retired, std::memory_order_relaxed);
}

void MaybeReport() {
	static uint64_t last_ns = 0; // Thread_Gpu only
	const auto      now     = NowNs();
	if (last_ns == 0) {
		last_ns = now;
		return;
	}
	if (now - last_ns < 10'000'000'000ull) {
		return;
	}
	last_ns     = now;
	auto& stats = GetStats();
	auto  take  = [](std::atomic<uint64_t>& value) {
		return value.exchange(0, std::memory_order_relaxed);
	};
	const auto a      = take(stats.completed[0]);
	const auto b      = take(stats.completed[1]);
	const auto a_ns   = take(stats.resolve_ns[0]);
	const auto b_ns   = take(stats.resolve_ns[1]);
	const auto slices = take(stats.slices_b);
	const auto bs     = take(stats.bounces_service);
	const auto bf     = take(stats.bounces_flip);
	const auto f_seq  = take(stats.fallbacks[static_cast<size_t>(Fallback::NoSequencer)]);
	const auto f_ho   = take(stats.fallbacks[static_cast<size_t>(Fallback::Handoff)]);
	const auto f_ver  = take(stats.fallbacks[static_cast<size_t>(Fallback::Verify)]);
	std::printf("CpPar 10s: A (Thread_Gpu) %" PRIu64 " subs, %.1f ms resolving; B (CP resolver #2) %" PRIu64
	            " subs, %.1f ms resolving, %" PRIu64 " slices; bounces to Thread_Gpu: service %" PRIu64
	            ", flip ops %" PRIu64 "; fallbacks to A: no-sequencer %" PRIu64 ", handoff %" PRIu64
	            ", verify %" PRIu64 "\n",
	            a, static_cast<double>(a_ns) / 1e6, b, static_cast<double>(b_ns) / 1e6, slices, bs, bf,
	            f_seq, f_ho, f_ver);
	if (SeparateWindows()) {
		// Astro Bot PC (KYTY_CP_PAR=1b1): each context's draw-prep window over the last 10 s.
		uint64_t values[2][4] = {};
		for (size_t c = 0; c < 2; c++) {
			const auto published = stats.window_published[c].load(std::memory_order_relaxed);
			const auto retired   = stats.window_retired[c].load(std::memory_order_relaxed);
			values[c][0]         = published - stats.reported_published[c];
			values[c][1]         = retired - stats.reported_retired[c];
			values[c][2]         = take(stats.prep_committed[c]);
			values[c][3]         = take(stats.prep_fell_back[c]);
			stats.reported_published[c] = published;
			stats.reported_retired[c]   = retired;
		}
		std::printf("CpPar 10s: DrawPrep windows: A published %" PRIu64 ", retired %" PRIu64
		            ", committed %" PRIu64 ", fell back %" PRIu64 "; B published %" PRIu64
		            ", retired %" PRIu64 ", committed %" PRIu64 ", fell back %" PRIu64 "\n",
		            values[0][0], values[0][1], values[0][2], values[0][3], values[1][0],
		            values[1][1], values[1][2], values[1][3]);
	}
	std::fflush(stdout);
}

} // namespace Libs::Graphics::CpPar
