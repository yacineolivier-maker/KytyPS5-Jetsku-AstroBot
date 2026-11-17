#ifndef EMULATOR_SRC_COMMON_CONDWAITUNTIL_H_
#define EMULATOR_SRC_COMMON_CONDWAITUNTIL_H_

#include "common/threads.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif

namespace Common {

// Astro Bot PC (KYTY_COND_SUBMS=1, default off, read once): timed guest waits (pthread condition
// variables, kernel semaphores) no longer round what is left below 1 ms up to a whole millisecond
// of the host condition variable / WaitOnAddress: the last part (< 1 ms) is waited for by spinning
// (pause, a yield every ~100 us) on a value the waker changes atomically, then the deadline is
// checked as before. A longer wait sleeps until about 1 ms before its deadline first.
inline bool CondSubMs() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_COND_SUBMS");
		const bool  r     = value != nullptr && std::strcmp(value, "1") == 0;
		if (r) {
			std::printf("Kyty attentes chronometrees : reste < 1 ms attendu precisement (KYTY_COND_SUBMS)\n");
			std::fflush(stdout);
		}
		return r;
	}();
	return on;
}

// Astro Bot PC (KYTY_COND_STATS=1, default off, read once): counters of the timed guest waits
// (pthread condition variables and kernel semaphores that block), printed every 10 s:
// "Kyty attentes chronometrees 10 s : ...": how many, the histogram of the time asked for, how
// many ended on their deadline and how late on average, the sub-millisecond spins (KYTY_COND_SUBMS)
// and how many of those were woken while spinning.
inline bool CondStatsOn() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_COND_STATS");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return on;
}

struct CondStatsCounters {
	static constexpr int BUCKETS = 8; // < 100 us, < 500 us, < 1 ms, < 2 ms, < 5 ms, < 10 ms, < 100 ms, more
	std::atomic<uint64_t> waits {0};
	std::atomic<uint64_t> hist[BUCKETS] {};
	std::atomic<uint64_t> timeouts {0};
	std::atomic<uint64_t> late_ns {0};
	std::atomic<uint64_t> spins {0};
	std::atomic<uint64_t> spins_woken {0};
	std::atomic<uint64_t> spin_ns {0};
	std::atomic<uint64_t> report_ns {0};
};

inline CondStatsCounters& CondStats() {
	static CondStatsCounters c;
	return c;
}

inline uint64_t CondNowNs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

inline void CondStatsMaybeReport() {
	auto&      c    = CondStats();
	const auto now  = CondNowNs();
	auto       last = c.report_ns.load(std::memory_order_relaxed);
	if (last == 0) {
		c.report_ns.compare_exchange_strong(last, now, std::memory_order_relaxed);
		return;
	}
	if (now - last < 10'000'000'000ull || !c.report_ns.compare_exchange_strong(last, now, std::memory_order_relaxed)) {
		return;
	}
	uint64_t h[CondStatsCounters::BUCKETS];
	for (int i = 0; i < CondStatsCounters::BUCKETS; i++) {
		h[i] = c.hist[i].exchange(0, std::memory_order_relaxed);
	}
	const auto waits    = c.waits.exchange(0, std::memory_order_relaxed);
	const auto timeouts = c.timeouts.exchange(0, std::memory_order_relaxed);
	const auto late     = c.late_ns.exchange(0, std::memory_order_relaxed);
	const auto spins    = c.spins.exchange(0, std::memory_order_relaxed);
	const auto woken    = c.spins_woken.exchange(0, std::memory_order_relaxed);
	const auto spin_ns  = c.spin_ns.exchange(0, std::memory_order_relaxed);
	const double seconds = static_cast<double>(now - last) / 1e9;
	std::printf("Kyty attentes chronometrees 10 s : %" PRIu64 " attentes (demande <100us %" PRIu64 ", <500us %" PRIu64
	            ", <1ms %" PRIu64 ", <2ms %" PRIu64 ", <5ms %" PRIu64 ", <10ms %" PRIu64 ", <100ms %" PRIu64
	            ", plus %" PRIu64 "), %" PRIu64 " echeances (retard moyen %.0f us), %" PRIu64
	            " attentes actives < 1 ms dont %" PRIu64 " reveillees, %.1f ms/s d'attente active\n",
	            waits, h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], timeouts,
	            timeouts != 0 ? static_cast<double>(late) / 1e3 / static_cast<double>(timeouts) : 0.0, spins,
	            woken, static_cast<double>(spin_ns) / 1e6 / seconds);
	std::fflush(stdout);
}

// A timed wait that is about to block, `requested_us` from now.
inline void CondStatsRecordWait(uint64_t requested_us) {
	if (!CondStatsOn()) {
		return;
	}
	auto&      c = CondStats();
	const int  b = requested_us < 100     ? 0
	               : requested_us < 500   ? 1
	               : requested_us < 1000  ? 2
	               : requested_us < 2000  ? 3
	               : requested_us < 5000  ? 4
	               : requested_us < 10000 ? 5
	               : requested_us < 100000 ? 6
	                                       : 7;
	c.waits.fetch_add(1, std::memory_order_relaxed);
	c.hist[b].fetch_add(1, std::memory_order_relaxed);
	CondStatsMaybeReport();
}

// A timed wait that ended on its deadline, `late_ns` after it.
inline void CondStatsRecordTimeout(uint64_t late_ns) {
	if (!CondStatsOn()) {
		return;
	}
	auto& c = CondStats();
	c.timeouts.fetch_add(1, std::memory_order_relaxed);
	c.late_ns.fetch_add(late_ns, std::memory_order_relaxed);
}

// KYTY_COND_SUBMS: spins until woken() or the deadline; true when woken() was seen. woken() must be
// safe without the waiter's lock (an atomic load). Never returns before the deadline unless woken.
template <class Woken>
bool SpinUntilPrecise(const Woken& woken, std::chrono::steady_clock::time_point deadline) {
	const auto start      = std::chrono::steady_clock::now();
	auto       next_yield = start + std::chrono::microseconds(100);
	bool       seen       = false;
	for (;;) {
		if (woken()) {
			seen = true;
			break;
		}
		for (int i = 0; i < 16; i++) {
#if defined(_M_X64) || defined(__x86_64__)
			_mm_pause();
#else
			std::this_thread::yield();
#endif
		}
		const auto now = std::chrono::steady_clock::now();
		if (now >= deadline) {
			seen = woken();
			break;
		}
		if (now >= next_yield) {
			// Leaves the core to a ready thread of the same core (possibly the waker).
			(void)YieldToReadyThread();
			next_yield = now + std::chrono::microseconds(100);
		}
	}
	if (CondStatsOn()) {
		auto& c = CondStats();
		c.spins.fetch_add(1, std::memory_order_relaxed);
		if (seen) {
			c.spins_woken.fetch_add(1, std::memory_order_relaxed);
		}
		c.spin_ns.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
		                                              std::chrono::steady_clock::now() - start)
		                                              .count()),
		                    std::memory_order_relaxed);
	}
	return seen;
}

// Waits on `cv` (with `lock` held) until ready() or the deadline. Every poll_us at the latest, and
// after each wake-up that is not ready, dispatch() runs with the lock released (pending guest
// signals). precise (KYTY_PRECISE_COND_WAITS, Senaxx d069e1e99): Windows times condition-variable
// waits in whole milliseconds on the system tick, so a ~5 ms wait woke up to ~1 ms late; precise
// waits time the last 2.5 ms with the high-resolution timer in slices of at most 0.5 ms (a wake-up
// in that stretch is seen within a slice) and a short yield loop. Not precise: one
// condition-variable wait per poll, as before.
// subms (KYTY_COND_SUBMS, takes precedence over precise): the condition variable is waited on
// until about 1 ms before the deadline, the rest (< 1 ms) is spun with SpinUntilPrecise on
// ready(), which must then be safe to call without the lock (an atomic load).
template <class Lock, class Ready, class Dispatch>
void CondWaitUntil(Lock& lock, std::condition_variable& cv, const Ready& ready,
                   std::chrono::steady_clock::time_point deadline, uint32_t poll_us, bool precise,
                   Dispatch&& dispatch, bool subms = false) {
	constexpr auto coarse_margin = std::chrono::microseconds(2500);
	constexpr auto spin_margin   = std::chrono::microseconds(300);
	constexpr auto fine_slice    = std::chrono::microseconds(500);
	constexpr auto subms_margin  = std::chrono::microseconds(1000);
	const auto     poll          = std::chrono::steady_clock::duration(std::chrono::microseconds(poll_us));
	while (!ready()) {
		const auto now = std::chrono::steady_clock::now();
		if (now >= deadline) {
			break;
		}
		const auto remaining = deadline - now;
		if (subms) {
			if (remaining < subms_margin) {
				lock.unlock();
				(void)SpinUntilPrecise(ready, deadline);
				lock.lock();
				continue; // ready() and the deadline are checked again under the lock
			}
			cv.wait_for(lock, std::min<std::chrono::steady_clock::duration>(remaining - subms_margin, poll));
		} else if (!precise) {
			cv.wait_for(lock, std::min(remaining, poll));
		} else if (remaining > coarse_margin) {
			cv.wait_for(lock, std::min<std::chrono::steady_clock::duration>(remaining - coarse_margin, poll));
		} else {
			lock.unlock();
			if (remaining > spin_margin) {
				const auto slice =
				    std::min<std::chrono::steady_clock::duration>(remaining - spin_margin, fine_slice);
				Thread::SleepMicro(static_cast<uint32_t>(
				    std::chrono::duration_cast<std::chrono::microseconds>(slice).count()));
			} else {
				std::this_thread::yield();
			}
			lock.lock();
			continue;
		}
		if (!ready()) {
			lock.unlock();
			dispatch();
			lock.lock();
		}
	}
}

} // namespace Common

#endif // EMULATOR_SRC_COMMON_CONDWAITUNTIL_H_
