#include "kernel/semaphore.h"

#include "common/assert.h"
#include "common/condWaitUntil.h" // Astro Bot PC (KYTY_COND_SUBMS, KYTY_COND_STATS)
#include "common/hangWatchdog.h"
#include "common/logging/log.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "common/timer.h"
#include "graphics/guest_gpu/framePath.h" // Astro Bot PC (KYTY_FRAME_PATH)
#include "kernel/mainCpu.h"            // Astro Bot PC (KYTY_MAIN_CPU_SAMPLER)
#include "kernel/waitGraph.h"           // Astro Bot PC (KYTY_WAIT_GRAPH)
#include "libs/errno.h"
#include "libs/libs.h"

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#pragma comment(lib, "Synchronization.lib") // WaitOnAddress (KYTY_GUEST_SEMA_LIGHT)
#endif

#include <algorithm>
#include <atomic>
#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif
#include <thread>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <vector>

namespace Libs::LibKernel {
void KernelDispatchPendingSignalForCurrentThread();
} // namespace Libs::LibKernel

namespace Libs::LibKernel::Semaphore {

LIB_NAME("libkernel", "libkernel");

constexpr uint32_t SIGNAL_APC_POLL_MICROS = 10000;

// Astro Bot PC (KYTY_GUEST_SEMA_LIGHT, default 0, Windows only): the guest kernel semaphores'
// lock and wake-ups on lighter primitives. 1: an SRW lock instead of the critical section (the
// 15/11 profile showed Thread_Main ~5.5 % in RtlLeaveCriticalSection -> NtSetInformationThread
// under KernelSignalSema) and each waiter parked on its own word (WaitOnAddress, woken by
// WakeByAddressSingle under the lock) instead of a condition variable. Same rules otherwise:
// waiter order, need counts, timeouts, cancel, delete, the 10 ms pending-signal poll.
// "verify": the same, plus checks under the lock (0 <= count <= max after each change, no
// waiter both served and still counted) and late wake-ups (a served waiter that resumed more than
// 15 ms after it was served: a lost wake-up, saved by the 10 ms poll); counters every 10 s:
// "Kyty semaphores invites 10 s : ...".
enum class SemaMode { Classic, Light, Verify };

static SemaMode GuestSemaMode() {
	static const SemaMode mode = [] {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		const char* value = std::getenv("KYTY_GUEST_SEMA_LIGHT");
		SemaMode    m     = SemaMode::Classic;
		if (value != nullptr && std::strcmp(value, "1") == 0) {
			m = SemaMode::Light;
		} else if (value != nullptr && std::strcmp(value, "verify") == 0) {
			m = SemaMode::Verify;
		}
		if (m != SemaMode::Classic) {
			std::printf("Kyty semaphores invites : verrou SRW et attente par adresse%s "
			            "(KYTY_GUEST_SEMA_LIGHT)\n",
			            m == SemaMode::Verify ? ", avec verification" : "");
			std::fflush(stdout);
		}
		return m;
#else
		return SemaMode::Classic;
#endif
	}();
	return mode;
}

// Astro Bot PC (KYTY_GUEST_SEMA_SPIN_US=<us>, default 0 = off, 1..1000): a kernel semaphore wait
// that would block first spins (pause) up to <us> for the count, without being registered as a
// waiter. Astro Bot's main thread and its "SystemWakeup" thread exchange ~1,300 signal/wait pairs
// per frame through the game's semaphore wrappers (eboot 0x17a0 wait, 0x17f0 timed wait; the
// answer comes ~4 us later, KYTY_WAIT_GRAPH 16/11): a waiter that gets its count while spinning
// costs no sleep, and the signaller finds no sleeping waiter, so it makes no wake-up call. Same
// rules: a spinner takes the count only as a wait called at that moment would (count available,
// under the lock); registered waiters are served first by Signal (WakeWaiters) as before; the
// spin is deducted from a timed wait's time; a poll (time 0) never spins; a deleted semaphore
// ends the spin. Works with and without KYTY_GUEST_SEMA_LIGHT. Every 10 s:
// "Kyty semaphores invites attente active 10 s : ...".
static uint32_t GuestSemaSpinUs() {
	static const uint32_t us = [] {
		const char* value  = std::getenv("KYTY_GUEST_SEMA_SPIN_US");
		const auto  parsed = value != nullptr ? std::strtoul(value, nullptr, 10) : 0ul;
		const auto  result = static_cast<uint32_t>(std::min<unsigned long>(parsed, 1000ul));
		if (result != 0) {
			std::printf("Kyty semaphores invites : attente active de %u us avant sommeil "
			            "(KYTY_GUEST_SEMA_SPIN_US)\n",
			            result);
			std::fflush(stdout);
		}
		return result;
	}();
	return us;
}

// Astro Bot PC (KYTY_SEMA_HANDOFF, default 0 = off, read once, Windows only; audit 17/11): direct
// hand-off of a kernel semaphore's count to its waiters. The game makes ~2,000 mostly blocking
// signals per frame; before, the signaller marked the waiter ready and woke it under the lock,
// and the woken waiter took the lock twice more (back from its condition wait, then to remove
// itself). With the hand-off the signaller, under the lock, takes the count for the waiter,
// writes its result, removes it from the list and flips its parking word to "served"; it calls
// WakeByAddressSingle after unlocking and only if the waiter was asleep. The served waiter
// returns without taking the lock again. Each waiter first spins briefly on its word when the
// semaphore's recent service delay (moving average, measured by the signaller) is below 20 us,
// then sleeps on it (WaitOnAddress) by slices of at most 10 ms (pending guest signals are
// dispatched between slices, and the word is checked again: safety net). Same rules as before:
// waiter order (FIFO or priority by attribute), need counts served as before (any waiter whose
// need fits, in list order), timeouts, cancel, delete, poll. 1: adaptive spin; 2: no spin.
// KYTY_GUEST_SEMA_SPIN_US is ignored in this mode. KYTY_COND_STATS=1 prints counters every 10 s:
// "Kyty semaphores remise directe 10 s : ...".
static int SemaHandoffMode() {
	static const int mode = [] {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		const char* value = std::getenv("KYTY_SEMA_HANDOFF");
		int         m     = 0;
		if (value != nullptr && std::strcmp(value, "1") == 0) {
			m = 1;
		} else if (value != nullptr && std::strcmp(value, "2") == 0) {
			m = 2;
		}
		if (m != 0) {
			std::printf("Kyty semaphores invites : remise directe du signal%s (KYTY_SEMA_HANDOFF)\n",
			            m == 1 ? ", attente active adaptative" : ", sans attente active");
			std::fflush(stdout);
		}
		return m;
#else
		return 0;
#endif
	}();
	return mode;
}

// KYTY_SEMA_HANDOFF: a waiter's parking word.
constexpr uint32_t PARK_ACTIVE   = 0; // registered, running (spinning, checking, dispatching)
constexpr uint32_t PARK_SLEEPING = 1; // registered, asleep on the word (WaitOnAddress)
constexpr uint32_t PARK_SERVED   = 2; // served by a signaller, cancel or delete (result written)
constexpr uint32_t PARK_LEAVING  = 3; // timed out: no longer served, removes itself under the lock

// KYTY_SEMA_HANDOFF adaptive spin: spin while the recent service delay is below this.
constexpr uint64_t HANDOFF_SPIN_THRESHOLD_NS = 20'000;
constexpr uint64_t HANDOFF_SPIN_MAX_NS       = 30'000;
constexpr uint64_t HANDOFF_SPIN_MIN_NS       = 2'000;

namespace {
std::atomic<uint64_t> g_ho_spin_served {0};  // served while spinning (no sleep, no wake-up call)
std::atomic<uint64_t> g_ho_sleep_served {0}; // served while asleep or between slices
std::atomic<uint64_t> g_ho_wake_calls {0};   // WakeByAddressSingle calls by signallers
std::atomic<uint64_t> g_ho_timeouts {0};
std::atomic<uint64_t> g_ho_leaving_skips {0}; // signaller found the waiter already timing out
std::atomic<uint64_t> g_ho_spin_ns {0};
std::atomic<uint64_t> g_ho_delay_ns {0}; // sum of service delays (registration -> served)
std::atomic<uint64_t> g_ho_delays {0};
std::atomic<uint64_t> g_ho_report_ns {0};

void HandoffReport(uint64_t now) {
	auto last = g_ho_report_ns.load(std::memory_order_relaxed);
	if (last == 0) {
		g_ho_report_ns.compare_exchange_strong(last, now, std::memory_order_relaxed);
		return;
	}
	if (now - last < 10'000'000'000ull ||
	    !g_ho_report_ns.compare_exchange_strong(last, now, std::memory_order_relaxed)) {
		return;
	}
	const double seconds = static_cast<double>(now - last) / 1e9;
	const auto   delays  = g_ho_delays.exchange(0, std::memory_order_relaxed);
	const auto   delay   = g_ho_delay_ns.exchange(0, std::memory_order_relaxed);
	std::printf("Kyty semaphores remise directe 10 s : %.0f/s servies en attente active, %.0f/s servies "
	            "endormies (%.0f/s reveils), %.0f/s echeances (%.0f/s au moment du service), delai de "
	            "service moyen %.1f us, %.1f ms/s d'attente active\n",
	            static_cast<double>(g_ho_spin_served.exchange(0, std::memory_order_relaxed)) / seconds,
	            static_cast<double>(g_ho_sleep_served.exchange(0, std::memory_order_relaxed)) / seconds,
	            static_cast<double>(g_ho_wake_calls.exchange(0, std::memory_order_relaxed)) / seconds,
	            static_cast<double>(g_ho_timeouts.exchange(0, std::memory_order_relaxed)) / seconds,
	            static_cast<double>(g_ho_leaving_skips.exchange(0, std::memory_order_relaxed)) / seconds,
	            delays != 0 ? static_cast<double>(delay) / 1e3 / static_cast<double>(delays) : 0.0,
	            static_cast<double>(g_ho_spin_ns.exchange(0, std::memory_order_relaxed)) / 1e6 / seconds);
	std::fflush(stdout);
}

// KYTY_SEMA_HANDOFF: parking words to wake once the semaphore's lock is released. Only the
// addresses are kept: WakeByAddressSingle never reads the memory, so waking a waiter that has
// already seen its word and returned (its stack word reused) is at worst a spurious wake-up, which
// every WaitOnAddress caller tolerates.
struct HandoffWakeList {
	static constexpr int N = 8;
	void*                addr[N] {};
	int                  n = 0;
	void                 Add(void* a) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		if (n < N) {
			addr[n++] = a;
		} else {
			WakeByAddressSingle(a); // more than N sleepers served at once: wake under the lock
		}
#else
		(void)a;
#endif
	}
	void Flush() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		for (int i = 0; i < n; i++) {
			WakeByAddressSingle(addr[i]);
		}
#endif
		if (n != 0) {
			g_ho_wake_calls.fetch_add(static_cast<uint64_t>(n), std::memory_order_relaxed);
		}
		n = 0;
	}
};

// Audit 17/11: one condition variable per host thread, reused by all its classic semaphore waits
// (WaitingThread used to own one: an allocation and a free per blocking wait). A thread waits on
// one semaphore at a time; a guest signal handler that waits again on this thread while an outer
// wait is registered shares it, which at worst gives one of them a spurious wake-up: both re-check
// their own `ready` flag under their semaphore's lock before sleeping again.
Common::CondVar& ThreadSemaCond() {
	thread_local Common::CondVar cond;
	return cond;
}
} // namespace

namespace {
std::atomic<uint64_t> g_spin_served {0};  // waits served while spinning (no sleep, no wake-up call)
std::atomic<uint64_t> g_spin_slept {0};   // waits that spun, then slept
std::atomic<uint64_t> g_spin_ns {0};      // time spent spinning
std::atomic<uint64_t> g_spin_report_ns {0};

void SpinReport(uint64_t now) {
	auto last = g_spin_report_ns.load(std::memory_order_relaxed);
	if (last == 0) {
		g_spin_report_ns.compare_exchange_strong(last, now, std::memory_order_relaxed);
		return;
	}
	if (now - last < 10'000'000'000ull ||
	    !g_spin_report_ns.compare_exchange_strong(last, now, std::memory_order_relaxed)) {
		return;
	}
	const double seconds = static_cast<double>(now - last) / 1e9;
	const auto   served  = g_spin_served.exchange(0, std::memory_order_relaxed);
	const auto   slept   = g_spin_slept.exchange(0, std::memory_order_relaxed);
	const auto   spun_ns = g_spin_ns.exchange(0, std::memory_order_relaxed);
	std::printf("Kyty semaphores invites attente active 10 s : %.0f/s servies pendant l'attente active "
	            "(sommeil et reveil evites), %.0f/s endormies ensuite, %.1f ms/s d'attente active\n",
	            static_cast<double>(served) / seconds, static_cast<double>(slept) / seconds,
	            static_cast<double>(spun_ns) / 1e6 / seconds);
	std::fflush(stdout);
}

std::atomic<uint64_t> g_sema_signals {0};
std::atomic<uint64_t> g_sema_waits {0};
std::atomic<uint64_t> g_sema_blocking {0};
std::atomic<uint64_t> g_sema_late_wakes {0};
std::atomic<uint64_t> g_sema_max_wake_us {0};
std::atomic<uint64_t> g_sema_violations {0};
std::atomic<uint64_t> g_sema_report_ns {0};

uint64_t SemaNowNs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

void SemaVerifyReport() {
	const auto now  = SemaNowNs();
	auto       last = g_sema_report_ns.load(std::memory_order_relaxed);
	if (last == 0) {
		g_sema_report_ns.compare_exchange_strong(last, now, std::memory_order_relaxed);
		return;
	}
	if (now - last < 10'000'000'000ull ||
	    !g_sema_report_ns.compare_exchange_strong(last, now, std::memory_order_relaxed)) {
		return;
	}
	std::printf("Kyty semaphores invites 10 s : %" PRIu64 " signaux, %" PRIu64 " attentes dont %" PRIu64
	            " bloquantes, reveils tardifs (>15 ms) %" PRIu64 " (pire %" PRIu64
	            " us), controles en defaut %" PRIu64 "\n",
	            g_sema_signals.exchange(0, std::memory_order_relaxed),
	            g_sema_waits.exchange(0, std::memory_order_relaxed),
	            g_sema_blocking.exchange(0, std::memory_order_relaxed),
	            g_sema_late_wakes.exchange(0, std::memory_order_relaxed),
	            g_sema_max_wake_us.exchange(0, std::memory_order_relaxed),
	            g_sema_violations.load(std::memory_order_relaxed));
	std::fflush(stdout);
}
} // namespace

class KernelSemaPrivate {
public:
	enum class Result { Ok, TimedOut, Canceled, Deleted, InvalCount };

	KernelSemaPrivate(const std::string& name, bool fifo, int init_count, int max_count)
	    : m_name(name), m_fifo_order(fifo), m_count(init_count), m_init_count(init_count),
	      m_max_count(max_count), m_mode(GuestSemaMode()) {
		PublishCount();
	};
	virtual ~KernelSemaPrivate();

	KYTY_CLASS_NO_COPY(KernelSemaPrivate);

	Result Cancel(int set_count, int* num_waiting_threads);
	Result Signal(int signal_count);
	Result Wait(int need_count, uint32_t* ptr_micros);

	Result Poll(int need_count) {
		uint32_t micros = 0;
		return Wait(need_count, &micros);
	}

	[[nodiscard]] const std::string& GetName() const { return m_name; }

private:
	enum class Status { Set, Deleted };

	struct WaitingThread {
		int    id         = 0;
		int    priority   = 700;
		int    need_count = 0;
		Result result     = Result::TimedOut;
		bool   ready      = false;
		// Astro Bot PC (13/11) : chaque fil en attente a sa variable de condition ; un signal ne reveille que les
		// fils servis (avant : SignalAll a chaque Signal, tous les fils reveilles se disputaient le verrou).
		// Audit 17/11 : celle du fil, reutilisee (ThreadSemaCond), plus d'allocation par attente.
		Common::CondVar* cond = nullptr;
		// KYTY_GUEST_SEMA_LIGHT: the word this waiter parks on (1 once served), and when it was
		// served (verify). Also set by Wake in the classic mode (KYTY_COND_SUBMS spins on it).
		std::atomic<uint32_t> word {0};
		uint64_t              served_ns = 0;
		// KYTY_SEMA_HANDOFF: the parking word (PARK_*), whether the server removed the waiter from
		// the list (false only for delete: the destructor waits for its waiters to leave), and when
		// it was registered (service delay, steady clock ns).
		std::atomic<uint32_t> park {PARK_ACTIVE};
		bool                  removed = false;
		uint64_t              enq_ns  = 0;
	};

	// The semaphore's lock (KYTY_GUEST_SEMA_LIGHT: SRW; otherwise the critical section).
	void Lock() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		if (m_mode != SemaMode::Classic) {
			AcquireSRWLockExclusive(&m_srw);
			return;
		}
#endif
		m_mutex.Lock();
	}
	void Unlock() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		if (m_mode != SemaMode::Classic) {
			ReleaseSRWLockExclusive(&m_srw);
			return;
		}
#endif
		m_mutex.Unlock();
	}
	class Guard {
	public:
		explicit Guard(KernelSemaPrivate& sema): m_sema(sema) { m_sema.Lock(); }
		~Guard() { m_sema.Unlock(); }
		Guard(const Guard&)            = delete;
		Guard& operator=(const Guard&) = delete;

	private:
		KernelSemaPrivate& m_sema;
	};

	// Lock held: the waiter was marked ready; wakes it.
	void Wake(WaitingThread* waiter) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		if (m_mode != SemaMode::Classic) {
			if (m_mode == SemaMode::Verify) {
				waiter->served_ns = SemaNowNs();
			}
			waiter->word.store(1, std::memory_order_release);
			WakeByAddressSingle(static_cast<void*>(&waiter->word));
			return;
		}
#endif
		waiter->word.store(1, std::memory_order_release); // KYTY_COND_SUBMS spin
		waiter->cond->Signal();
	}
	// Lock held: parks until woken, `micros` at most (ms granularity, as the condition variable);
	// returns with the lock held.
	void Park(WaitingThread* waiter, uint32_t micros) {
		if (micros < 1000 && Common::CondSubMs()) {
			// KYTY_COND_SUBMS: less than 1 ms left (the caller passes the time left of a timed
			// wait; untimed waits pass 10 ms): spin on the word Wake sets, lock released.
			const auto deadline = std::chrono::steady_clock::now() + std::chrono::microseconds(micros);
			Unlock();
			(void)Common::SpinUntilPrecise(
			    [waiter] { return waiter->word.load(std::memory_order_acquire) != 0; }, deadline);
			Lock();
			return;
		}
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		if (m_mode != SemaMode::Classic) {
			Unlock();
			uint32_t   expected = 0;
			const auto ms       = static_cast<DWORD>(micros < 1000 ? 1 : micros / 1000);
			(void)WaitOnAddress(static_cast<volatile void*>(&waiter->word), &expected, sizeof(expected), ms);
			Lock();
			return;
		}
#endif
		waiter->cond->WaitFor(&m_mutex, micros);
	}
	// KYTY_GUEST_SEMA_LIGHT=verify, lock held.
	void Check() {
		if (m_mode != SemaMode::Verify) {
			return;
		}
		bool bad = m_count < 0 || m_count > m_max_count;
		for (const auto* waiter: m_waiting_threads) {
			if (waiter->ready != (waiter->word.load(std::memory_order_relaxed) != 0) &&
			    m_status == Status::Set && waiter->result != Result::Canceled) {
				bad = true;
			}
		}
		if (bad && g_sema_violations.fetch_add(1, std::memory_order_relaxed) < 16) {
			std::printf("Kyty semaphores invites : controle en defaut sur '%s' (compte %d, max %d, "
			            "%zu en attente)\n",
			            m_name.c_str(), m_count, m_max_count, m_waiting_threads.size());
			std::fflush(stdout);
		}
	}

	void AddWaiter(WaitingThread* waiter);
	void RemoveWaiter(WaitingThread* waiter);
	void WakeWaiters();

	// KYTY_SEMA_HANDOFF (see SemaHandoffMode).
	Result WaitHandoff(int need_count, uint32_t* ptr_micros);
	// Lock held: serves `waiter` with `result` unless it is already leaving (timed out) or served.
	// Everything the waiter reads is written before its word turns PARK_SERVED; once it has, the
	// waiter may return at any moment and its WaitingThread (on its stack) must not be touched
	// again: only the word's address is kept, for a wake-up after the unlock if it was asleep.
	bool HandoffClaim(WaitingThread* waiter, Result result, bool remove, HandoffWakeList* wake) {
		auto* const word = &waiter->park;
		uint32_t    prev = word->load(std::memory_order_relaxed);
		// Only a signaller (under this lock) makes a word PARK_SERVED; the waiter itself moves it
		// between ACTIVE, SLEEPING and LEAVING only.
		if (prev == PARK_SERVED) {
			return false;
		}
		waiter->result  = result;
		waiter->removed = remove;
		waiter->ready   = true;
		if (m_mode == SemaMode::Verify) {
			waiter->served_ns = SemaNowNs();
		}
		for (;;) {
			if (prev == PARK_LEAVING) {
				// Timed out first: it is still registered and removes itself under this lock, so
				// it is alive; it reports a timeout and reads none of the fields above.
				waiter->ready = false;
				g_ho_leaving_skips.fetch_add(1, std::memory_order_relaxed);
				return false;
			}
			if (word->compare_exchange_weak(prev, PARK_SERVED, std::memory_order_acq_rel,
			                                std::memory_order_relaxed)) {
				break;
			}
		}
		// `waiter` may be gone from here on.
		if (prev == PARK_SLEEPING) {
			wake->Add(static_cast<void*>(word));
		}
		return true;
	}
	// Lock held: serves the waiters whose need fits, in list order (as WakeWaiters), removing them.
	void HandoffServe(HandoffWakeList* wake) {
		uint64_t now = 0;
		auto     out = m_waiting_threads.begin();
		for (auto it = m_waiting_threads.begin(); it != m_waiting_threads.end(); ++it) {
			auto* waiter = *it;
			if (!waiter->ready && waiter->need_count <= m_count) {
				const int      need   = waiter->need_count; // read before the claim
				const uint64_t enq_ns = waiter->enq_ns;
				if (HandoffClaim(waiter, Result::Ok, true, wake)) {
					m_count -= need;
					if (now == 0) {
						now = SemaNowNs();
					}
					const uint64_t delay = now > enq_ns ? now - enq_ns : 0;
					// Moving average (1/8) of the service delay, read by the next waiters.
					m_ho_avg_ns = m_ho_avg_ns - m_ho_avg_ns / 8 + std::min<uint64_t>(delay, 10'000'000) / 8;
					if (Common::CondStatsOn()) {
						g_ho_delay_ns.fetch_add(delay, std::memory_order_relaxed);
						g_ho_delays.fetch_add(1, std::memory_order_relaxed);
					}
					continue; // removed
				}
			}
			*out++ = waiter;
		}
		m_waiting_threads.erase(out, m_waiting_threads.end());
	}
	// Lock held: how long a new waiter spins before sleeping (adaptive, KYTY_SEMA_HANDOFF=1).
	[[nodiscard]] uint64_t HandoffSpinNs() const {
		if (SemaHandoffMode() != 1 || m_ho_avg_ns >= HANDOFF_SPIN_THRESHOLD_NS) {
			return 0;
		}
		return std::clamp<uint64_t>(2 * m_ho_avg_ns + 1'000, HANDOFF_SPIN_MIN_NS, HANDOFF_SPIN_MAX_NS);
	}
	uint64_t m_ho_avg_ns = 0; // KYTY_SEMA_HANDOFF: service delay moving average (lock held)

	Common::Mutex               m_mutex;
	Common::CondVar             m_cond_var;
	std::string                 m_name;
	Status                      m_status = Status::Set;
	std::vector<WaitingThread*> m_waiting_threads;
	bool                        m_fifo_order;
	int                         m_count;
	// KYTY_GUEST_SEMA_SPIN_US: m_count as last stored under the lock, read by spinning waiters.
	std::atomic<int> m_count_hint {0};
	void             PublishCount() noexcept { m_count_hint.store(m_count, std::memory_order_release); }
	// KYTY_GUEST_SEMA_SPIN_US: waiters spinning unregistered (under the lock) and the deletion,
	// which they read while spinning.
	int               m_spinners = 0;
	std::atomic<bool> m_deleted_hint {false};
	int                         m_init_count;
	int                         m_max_count;
	SemaMode                    m_mode;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	SRWLOCK m_srw = SRWLOCK_INIT;
#endif
};

KernelSemaPrivate::~KernelSemaPrivate() {
	Guard lock(*this);

	while (m_status != Status::Set) {
		Unlock();
		Common::Thread::SleepMicro(10);
		Lock();
	}

	m_status = Status::Deleted;
	m_deleted_hint.store(true, std::memory_order_release); // KYTY_GUEST_SEMA_SPIN_US

	if (SemaHandoffMode() != 0) {
		// Served but left in the list: each waiter removes itself under the lock, and the loop
		// below waits for the list to empty before the semaphore is freed. Leaving (timed-out)
		// waiters are skipped and remove themselves too.
		HandoffWakeList wake;
		for (auto* waiter: m_waiting_threads) {
			if (!waiter->ready) {
				(void)HandoffClaim(waiter, Result::Deleted, false, &wake);
			}
		}
		wake.Flush();
	} else {
		for (auto* waiter: m_waiting_threads) {
			if (!waiter->ready) {
				waiter->result = Result::Deleted;
				waiter->ready  = true;
				Wake(waiter);
			}
		}
	}

	while (!m_waiting_threads.empty() || m_spinners != 0) {
		Unlock();
		Common::Thread::SleepMicro(10);
		Lock();
	}
}

KernelSemaPrivate::Result KernelSemaPrivate::Cancel(int set_count, int* num_waiting_threads) {
	Guard lock(*this);

	if (set_count > m_max_count) {
		return Result::InvalCount;
	}

	if (m_status == Status::Deleted) {
		return Result::Deleted;
	}

	if (num_waiting_threads != nullptr) {
		*num_waiting_threads = static_cast<int>(
		    std::count_if(m_waiting_threads.begin(), m_waiting_threads.end(),
		                  [](const WaitingThread* waiter) { return !waiter->ready; }));
	}

	m_count = (set_count < 0 ? m_init_count : set_count);
	PublishCount();

	if (SemaHandoffMode() != 0) {
		// Served and removed (a canceled waiter never touches the semaphore again); leaving ones
		// stay and time out. Woken under the lock here (cancel is rare; the wake-up reads no
		// memory, so it is safe even if the waiter has already returned).
		HandoffWakeList wake;
		auto            out = m_waiting_threads.begin();
		for (auto it = m_waiting_threads.begin(); it != m_waiting_threads.end(); ++it) {
			auto* waiter = *it;
			if (!waiter->ready && HandoffClaim(waiter, Result::Canceled, true, &wake)) {
				continue;
			}
			*out++ = waiter;
		}
		m_waiting_threads.erase(out, m_waiting_threads.end());
		wake.Flush();
		Check();
		return Result::Ok;
	}

	for (auto* waiter: m_waiting_threads) {
		if (!waiter->ready) {
			waiter->result = Result::Canceled;
			waiter->ready  = true;
			Wake(waiter);
		}
	}
	Check();

	return Result::Ok;
}

KernelSemaPrivate::Result KernelSemaPrivate::Signal(int signal_count) {
	if (m_mode == SemaMode::Verify) {
		g_sema_signals.fetch_add(1, std::memory_order_relaxed);
		SemaVerifyReport();
	}
	if (SemaHandoffMode() != 0) {
		HandoffWakeList wake;
		{
			Guard lock(*this);
			if (m_status == Status::Deleted) {
				return Result::Deleted;
			}
			if (m_count + signal_count > m_max_count) {
				return Result::InvalCount;
			}
			m_count += signal_count;
			if (!m_waiting_threads.empty()) {
				HandoffServe(&wake);
			}
			PublishCount();
			Check();
		}
		wake.Flush(); // after the unlock: the woken waiter does not run into the lock
		if (Common::CondStatsOn()) {
			HandoffReport(SemaNowNs());
		}
		return Result::Ok;
	}
	Guard lock(*this);

	if (m_status == Status::Deleted) {
		return Result::Deleted;
	}

	if (m_count + signal_count > m_max_count) {
		return Result::InvalCount;
	}

	m_count += signal_count;

	WakeWaiters(); // reveille seulement les fils servis
	PublishCount(); // KYTY_GUEST_SEMA_SPIN_US: what is left for a spinning waiter
	Check();

	return Result::Ok;
}

void KernelSemaPrivate::AddWaiter(WaitingThread* waiter) {
	if (m_fifo_order) {
		m_waiting_threads.push_back(waiter);
		return;
	}

	auto it = m_waiting_threads.begin();
	while (it != m_waiting_threads.end() && (*it)->priority <= waiter->priority) {
		++it;
	}
	m_waiting_threads.insert(it, waiter);
}

void KernelSemaPrivate::RemoveWaiter(WaitingThread* waiter) {
	auto it = std::find(m_waiting_threads.begin(), m_waiting_threads.end(), waiter);
	if (it != m_waiting_threads.end()) {
		m_waiting_threads.erase(it);
	}
}

void KernelSemaPrivate::WakeWaiters() {
	for (auto* waiter: m_waiting_threads) {
		if (!waiter->ready && waiter->need_count <= m_count) {
			m_count -= waiter->need_count;
			waiter->result = Result::Ok;
			waiter->ready  = true;
			Wake(waiter);
		}
	}
}

KernelSemaPrivate::Result KernelSemaPrivate::Wait(int need_count, uint32_t* ptr_micros) {
	if (SemaHandoffMode() != 0) {
		return WaitHandoff(need_count, ptr_micros);
	}
	if (m_mode == SemaMode::Verify) {
		g_sema_waits.fetch_add(1, std::memory_order_relaxed);
	}
	Guard lock(*this);

	if (need_count < 1 || need_count > m_max_count) {
		return Result::InvalCount;
	}

	if (m_status == Status::Deleted) {
		return Result::Deleted;
	}

	uint32_t micros     = 0;
	bool     infinitely = true;
	if (ptr_micros != nullptr) {
		micros     = *ptr_micros;
		infinitely = false;
	}

	if (m_count >= need_count) {
		m_count -= need_count;
		PublishCount();
		Check();
		if (ptr_micros != nullptr) {
			*ptr_micros = micros;
		}
		return Result::Ok;
	}

	if (!infinitely && micros == 0) {
		return Result::TimedOut;
	}

	if (const auto spin_us = GuestSemaSpinUs(); spin_us != 0) {
		// KYTY_GUEST_SEMA_SPIN_US: spin unregistered, then take the count as a wait called now
		// would; the lock is held again whatever happens (Guard).
		const auto spin_start = std::chrono::steady_clock::now();
		const auto budget_us  = infinitely ? spin_us : std::min<uint32_t>(spin_us, micros);
		const auto deadline   = spin_start + std::chrono::microseconds(budget_us);
		bool       served     = false;
		m_spinners++; // under the lock: the destructor waits for spinners too
		Unlock();
		for (;;) {
			for (int i = 0; i < 16; i++) {
#if defined(_M_X64) || defined(__x86_64__)
				_mm_pause();
#else
				std::this_thread::yield();
#endif
			}
			const bool deleted = m_deleted_hint.load(std::memory_order_acquire);
			if (deleted || m_count_hint.load(std::memory_order_acquire) >= need_count) {
				Lock();
				if (m_status == Status::Deleted) {
					m_spinners--;
					return Result::Deleted; // Guard unlocks
				}
				if (m_count >= need_count) {
					m_count -= need_count;
					PublishCount();
					Check();
					served = true;
					break; // lock held
				}
				Unlock();
			}
			if (std::chrono::steady_clock::now() >= deadline) {
				Lock();
				break;
			}
		}
		m_spinners--; // lock held again
		const auto spun = std::chrono::steady_clock::now() - spin_start;
		const auto spun_ns =
		    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(spun).count());
		g_spin_ns.fetch_add(spun_ns, std::memory_order_relaxed);
		(served ? g_spin_served : g_spin_slept).fetch_add(1, std::memory_order_relaxed);
		SpinReport(static_cast<uint64_t>(
		    std::chrono::duration_cast<std::chrono::nanoseconds>(
		        std::chrono::steady_clock::now().time_since_epoch())
		        .count()));
		if (served) {
			if (ptr_micros != nullptr) {
				const auto used = static_cast<uint32_t>(spun_ns / 1000u);
				*ptr_micros     = micros > used ? micros - used : 0;
			}
			return Result::Ok;
		}
		if (m_status == Status::Deleted) {
			return Result::Deleted;
		}
		if (m_count >= need_count) { // came in just after the deadline
			m_count -= need_count;
			PublishCount();
			Check();
			if (ptr_micros != nullptr) {
				const auto used = static_cast<uint32_t>(spun_ns / 1000u);
				*ptr_micros     = micros > used ? micros - used : 0;
			}
			return Result::Ok;
		}
		if (!infinitely) {
			const auto used = static_cast<uint32_t>(spun_ns / 1000u);
			if (used >= micros) {
				*ptr_micros = 0;
				return Result::TimedOut;
			}
			micros -= used;
		}
	}

	if (m_mode == SemaMode::Verify) {
		g_sema_blocking.fetch_add(1, std::memory_order_relaxed);
	}

	uint32_t      elapsed = 0;
	Common::Timer t;
	t.Start();

	int id = Common::Thread::GetThreadIdUnique();

	HangWatchdog::Scope wait("guest-semaphore", reinterpret_cast<uint64_t>(this), need_count,
	                         m_count, 0, micros);
	WaitingThread waiter {};
	waiter.id         = id;
	waiter.priority   = (m_fifo_order ? 0 : Libs::LibKernel::PthreadGetCurrentPriorityForKernel());
	waiter.need_count = need_count;
	waiter.cond       = &ThreadSemaCond();
	AddWaiter(&waiter);
	if (!infinitely) {
		Common::CondStatsRecordWait(micros); // KYTY_COND_STATS
	}

	while (!waiter.ready) {
		if ((elapsed >= micros && !infinitely)) {
			RemoveWaiter(&waiter);
			*ptr_micros = 0;
			Common::CondStatsRecordTimeout(static_cast<uint64_t>(elapsed - micros) * 1000u);
			return Result::TimedOut;
		}

		Park(&waiter, infinitely ? SIGNAL_APC_POLL_MICROS : micros - elapsed);

		Unlock();
		LibKernel::KernelDispatchPendingSignalForCurrentThread();
		Lock();

		elapsed = static_cast<uint32_t>(t.GetTimeS() * 1000000.0);
	}

	if (m_mode == SemaMode::Verify && waiter.served_ns != 0) {
		const auto delay_us = (SemaNowNs() - waiter.served_ns) / 1000u;
		if (delay_us > 15'000) {
			g_sema_late_wakes.fetch_add(1, std::memory_order_relaxed);
		}
		auto worst = g_sema_max_wake_us.load(std::memory_order_relaxed);
		while (delay_us > worst &&
		       !g_sema_max_wake_us.compare_exchange_weak(worst, delay_us, std::memory_order_relaxed)) {
		}
	}

	RemoveWaiter(&waiter);

	if (ptr_micros != nullptr) {
		*ptr_micros = (elapsed >= micros ? 0 : micros - elapsed);
	}

	return waiter.result;
}

// KYTY_SEMA_HANDOFF (see SemaHandoffMode). The races, all decided by the waiter's parking word:
// - served while running or asleep: the signaller (under the lock) writes the result, removes the
//   waiter and CASes ACTIVE/SLEEPING -> SERVED; it wakes the address after the unlock only if it
//   saw SLEEPING. The waiter only sleeps after CASing ACTIVE -> SLEEPING, and WaitOnAddress
//   returns at once when the word is no longer SLEEPING: no lost wake-up. A missed wake-up would
//   still be caught at the end of the slice (10 ms at most).
// - timeout at the moment of service: the waiter CASes ACTIVE -> LEAVING before it takes the
//   lock to remove itself. If that fails the word is SERVED: the wait succeeded (its count was
//   taken). If it succeeds the signaller, which only serves ACTIVE/SLEEPING words, skips it and
//   gives the count to the next waiter whose need fits (or keeps it).
// - delete: the waiters are served with Deleted but left in the list; each one takes the lock to
//   remove itself, and the destructor waits for the list to empty. Cancel removes them (they do
//   not touch the semaphore again, so a delete right after a cancel is safe).
// - a served waiter touches neither the semaphore nor its lock: the semaphore may be deleted as
//   soon as the signal returns. Only the signaller updates the service-delay average.
KernelSemaPrivate::Result KernelSemaPrivate::WaitHandoff(int need_count, uint32_t* ptr_micros) {
	if (m_mode == SemaMode::Verify) {
		g_sema_waits.fetch_add(1, std::memory_order_relaxed);
	}
	Lock();

	if (need_count < 1 || need_count > m_max_count) {
		Unlock();
		return Result::InvalCount;
	}
	if (m_status == Status::Deleted) {
		Unlock();
		return Result::Deleted;
	}

	uint32_t   micros     = (ptr_micros != nullptr ? *ptr_micros : 0);
	const bool infinitely = (ptr_micros == nullptr);

	if (m_count >= need_count) {
		m_count -= need_count;
		PublishCount();
		Check();
		Unlock();
		return Result::Ok; // *ptr_micros unchanged
	}
	if (!infinitely && micros == 0) {
		Unlock();
		return Result::TimedOut;
	}

	if (m_mode == SemaMode::Verify) {
		g_sema_blocking.fetch_add(1, std::memory_order_relaxed);
	}

	const auto start    = std::chrono::steady_clock::now();
	const auto deadline = start + std::chrono::microseconds(micros);

	HangWatchdog::Scope wait("guest-semaphore", reinterpret_cast<uint64_t>(this), need_count, m_count, 0,
	                         micros);
	WaitingThread waiter {};
	waiter.id         = Common::Thread::GetThreadIdUnique();
	waiter.priority   = (m_fifo_order ? 0 : Libs::LibKernel::PthreadGetCurrentPriorityForKernel());
	waiter.need_count = need_count;
	waiter.enq_ns     = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(start.time_since_epoch()).count());
	AddWaiter(&waiter);
	const uint64_t spin_ns = HandoffSpinNs();
	Unlock();
	// From here on the semaphore is touched only while `waiter` is still in its list (not served,
	// or served by delete).

	if (!infinitely) {
		Common::CondStatsRecordWait(micros); // KYTY_COND_STATS
	}
	const bool stats  = Common::CondStatsOn();
	bool       served = false;
	auto       is_served = [&waiter] { return waiter.park.load(std::memory_order_acquire) == PARK_SERVED; };

	if (spin_ns != 0) {
		auto spin_end = start + std::chrono::nanoseconds(spin_ns);
		if (!infinitely && spin_end > deadline) {
			spin_end = deadline;
		}
		for (;;) {
			if (is_served()) {
				served = true;
				break;
			}
			for (int i = 0; i < 8; i++) {
#if defined(_M_X64) || defined(__x86_64__)
				_mm_pause();
#else
				std::this_thread::yield();
#endif
			}
			if (std::chrono::steady_clock::now() >= spin_end) {
				served = is_served();
				break;
			}
		}
		if (stats) {
			g_ho_spin_ns.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
			                                                 std::chrono::steady_clock::now() - start)
			                                                 .count()),
			                       std::memory_order_relaxed);
			if (served) {
				g_ho_spin_served.fetch_add(1, std::memory_order_relaxed);
			}
		}
	}

	if (!served) {
		const bool subms = Common::CondSubMs();
		for (;;) {
			if (is_served()) {
				served = true;
				break;
			}
			const auto now = std::chrono::steady_clock::now();
			if (!infinitely && now >= deadline) {
				uint32_t expected = PARK_ACTIVE;
				if (waiter.park.compare_exchange_strong(expected, PARK_LEAVING, std::memory_order_acq_rel)) {
					// Still registered: remove it under the lock (the destructor waits for it).
					Lock();
					RemoveWaiter(&waiter);
					Check();
					Unlock();
					if (stats) {
						g_ho_timeouts.fetch_add(1, std::memory_order_relaxed);
						Common::CondStatsRecordTimeout(static_cast<uint64_t>(
						    std::chrono::duration_cast<std::chrono::nanoseconds>(now - deadline).count()));
					}
					*ptr_micros = 0;
					return Result::TimedOut;
				}
				served = true; // expected == PARK_SERVED: served just before the deadline
				break;
			}
			uint64_t left_us = SIGNAL_APC_POLL_MICROS;
			if (!infinitely) {
				left_us = std::min<uint64_t>(
				    left_us, static_cast<uint64_t>(
				                 std::chrono::duration_cast<std::chrono::microseconds>(deadline - now).count()));
				if (subms && left_us < 1000) {
					// KYTY_COND_SUBMS: the last part of a timed wait, spun precisely.
					(void)Common::SpinUntilPrecise(is_served, deadline);
					continue;
				}
			}
			// Sleep slice: never longer than 10 ms (the pending-signal poll and the safety net).
			// KYTY_COND_SUBMS: wakes about 1 ms before the deadline, the rest is spun.
			auto ms = static_cast<uint32_t>(left_us / 1000);
			if (subms && !infinitely && left_us < SIGNAL_APC_POLL_MICROS) {
				ms = static_cast<uint32_t>((left_us - 1000) / 1000);
			}
			ms = std::max<uint32_t>(ms, 1);
			uint32_t expected = PARK_ACTIVE;
			if (!waiter.park.compare_exchange_strong(expected, PARK_SLEEPING, std::memory_order_acq_rel)) {
				served = true; // only a signaller changes an ACTIVE word: PARK_SERVED
				break;
			}
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
			uint32_t sleeping = PARK_SLEEPING;
			(void)WaitOnAddress(static_cast<volatile void*>(&waiter.park), &sleeping, sizeof(sleeping), ms);
#endif
			expected = PARK_SLEEPING;
			if (!waiter.park.compare_exchange_strong(expected, PARK_ACTIVE, std::memory_order_acq_rel)) {
				served = true; // PARK_SERVED
				break;
			}
			// Not served (slice over or spurious wake-up): pending guest signals, then check again.
			LibKernel::KernelDispatchPendingSignalForCurrentThread();
		}
		if (stats) {
			g_ho_sleep_served.fetch_add(1, std::memory_order_relaxed);
		}
	}

	// Served (the word is PARK_SERVED, read with acquire: the result is visible).
	const Result result = waiter.result;
	if (m_mode == SemaMode::Verify && waiter.served_ns != 0) {
		const auto delay_us = (SemaNowNs() - waiter.served_ns) / 1000u;
		if (delay_us > 15'000) {
			g_sema_late_wakes.fetch_add(1, std::memory_order_relaxed);
		}
		auto worst = g_sema_max_wake_us.load(std::memory_order_relaxed);
		while (delay_us > worst &&
		       !g_sema_max_wake_us.compare_exchange_weak(worst, delay_us, std::memory_order_relaxed)) {
		}
	}
	if (!waiter.removed) {
		// Delete: still in the list, the destructor waits for it.
		Lock();
		RemoveWaiter(&waiter);
		Unlock();
	}
	// The old path dispatched pending guest signals after every wake-up, served or not.
	LibKernel::KernelDispatchPendingSignalForCurrentThread();
	if (ptr_micros != nullptr) {
		const auto elapsed_us = static_cast<uint64_t>(
		    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());
		*ptr_micros = (elapsed_us >= micros ? 0 : micros - static_cast<uint32_t>(elapsed_us));
	}
	return result;
}

int KYTY_SYSV_ABI KernelCreateSema(KernelSema* sem, const char* name, uint32_t attr, int init,
                                   int max, void* opt) {
	PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(sem == nullptr);

	if (name == nullptr || attr > 2 || init < 0 || max <= 0 || init > max) {
		return KERNEL_ERROR_EINVAL;
	}

	bool fifo = false;

	switch (attr) {
		case 0x01: fifo = true; break;
		case 0x02:
		default: fifo = false; break;
	}

	*sem = new KernelSemaPrivate(std::string(name), fifo, init, max);

	return OK;
}

int KYTY_SYSV_ABI KernelDeleteSema(KernelSema sem) {
	PRINT_NAME();

	if (sem == nullptr) {
		return KERNEL_ERROR_ESRCH;
	}

	delete sem;

	return OK;
}

int KYTY_SYSV_ABI KernelWaitSema(KernelSema sem, int need, KernelUseconds* time) {
	MainCpu::NoteCpu(); // Astro Bot PC (KYTY_MAIN_CPU_SAMPLER): placement
	if (sem == nullptr) {
		return KERNEL_ERROR_ESRCH;
	}

	const ::Libs::Graphics::FramePath::MainWaitScope frame_path_wait(
	    ::Libs::Graphics::FramePath::Wait::Semaphore); // Astro Bot PC (KYTY_FRAME_PATH)
	const WaitGraph::WaitScope wait_graph(sem, WaitGraph::Prim::KernelSema,
	                                      reinterpret_cast<uint64_t>(__builtin_return_address(0)));
	auto result = sem->Wait(need, time);

	int ret = OK;

	switch (result) {
		case KernelSemaPrivate::Result::Ok: ret = OK; break;
		case KernelSemaPrivate::Result::InvalCount: ret = KERNEL_ERROR_EINVAL; break;
		case KernelSemaPrivate::Result::TimedOut: ret = KERNEL_ERROR_ETIMEDOUT; break;
		case KernelSemaPrivate::Result::Canceled: ret = KERNEL_ERROR_ECANCELED; break;
		case KernelSemaPrivate::Result::Deleted: ret = KERNEL_ERROR_EACCES; break;
	}

	return ret;
}

int KYTY_SYSV_ABI KernelPollSema(KernelSema sem, int need) {
	PRINT_NAME();

	if (sem == nullptr) {
		return KERNEL_ERROR_ESRCH;
	}

	auto result = sem->Poll(need);

	int ret = OK;

	switch (result) {
		case KernelSemaPrivate::Result::Ok: ret = OK; break;
		case KernelSemaPrivate::Result::InvalCount: ret = KERNEL_ERROR_EINVAL; break;
		case KernelSemaPrivate::Result::TimedOut:
		case KernelSemaPrivate::Result::Canceled:
		case KernelSemaPrivate::Result::Deleted: ret = KERNEL_ERROR_EBUSY; break;
	}

	return ret;
}

int KYTY_SYSV_ABI KernelSignalSema(KernelSema sem, int count) {
	if (sem == nullptr) {
		return KERNEL_ERROR_ESRCH;
	}

	WaitGraph::Signal(sem, reinterpret_cast<uint64_t>(__builtin_return_address(0))); // KYTY_WAIT_GRAPH
	auto result = sem->Signal(count);

	int ret = OK;

	switch (result) {
		case KernelSemaPrivate::Result::Ok: ret = OK; break;
		case KernelSemaPrivate::Result::InvalCount:
		case KernelSemaPrivate::Result::TimedOut:
		case KernelSemaPrivate::Result::Canceled:
		case KernelSemaPrivate::Result::Deleted: ret = KERNEL_ERROR_EINVAL; break;
	}

	return ret;
}

int KYTY_SYSV_ABI KernelCancelSema(KernelSema sem, int count, int* threads) {
	PRINT_NAME();

	if (sem == nullptr) {
		return KERNEL_ERROR_ESRCH;
	}

	WaitGraph::Signal(sem, reinterpret_cast<uint64_t>(__builtin_return_address(0))); // KYTY_WAIT_GRAPH
	auto result = sem->Cancel(count, threads);

	int ret = OK;

	switch (result) {
		case KernelSemaPrivate::Result::Ok: ret = OK; break;
		case KernelSemaPrivate::Result::InvalCount:
		case KernelSemaPrivate::Result::TimedOut:
		case KernelSemaPrivate::Result::Canceled:
		case KernelSemaPrivate::Result::Deleted: ret = KERNEL_ERROR_EINVAL; break;
	}

	return ret;
}

} // namespace Libs::LibKernel::Semaphore

namespace Libs::Posix {

LIB_NAME("Posix", "libkernel");

namespace {

constexpr uint16_t POSIX_SEM_MAGIC        = 0x09fa;
constexpr int      POSIX_SEM_VALUE_MAX    = 0x7fffffff;
constexpr uint32_t SIGNAL_APC_POLL_MICROS = 10000;

struct PosixSemGuest {
	uint16_t          magic;
	uint16_t          nameid;
	volatile uint32_t has_waiters;
	volatile uint32_t count;
	uint32_t          flags;
};

static_assert(sizeof(PosixSemGuest) == 16);

class PosixSemPrivate {
public:
	PosixSemPrivate(PosixSemGuest* guest, unsigned int value, uint32_t flags)
	    : m_guest(guest), m_count(static_cast<int>(value)) {
		SyncGuest(flags);
	}

	int Wait(uint32_t* micros) {
		Common::LockGuard lock(m_mutex);

		uint32_t      elapsed    = 0;
		const bool    timed_wait = (micros != nullptr);
		Common::Timer timer;
		timer.Start();

		while (m_count <= 0) {
			HangWatchdog::Scope wait("guest-posix-semaphore", reinterpret_cast<uint64_t>(m_guest),
			                         1, m_count);
			if (timed_wait && elapsed >= *micros) {
				SyncGuest();
				return POSIX_ETIMEDOUT;
			}

			m_waiters++;
			SyncGuest();

			if (timed_wait) {
				m_cond_var.WaitFor(&m_mutex, *micros - elapsed);
			} else {
				m_cond_var.WaitFor(&m_mutex, SIGNAL_APC_POLL_MICROS);
			}

			m_mutex.Unlock();
			LibKernel::KernelDispatchPendingSignalForCurrentThread();
			m_mutex.Lock();

			m_waiters--;
			elapsed = static_cast<uint32_t>(timer.GetTimeS() * 1000000.0);
		}

		m_count--;
		SyncGuest();

		return OK;
	}

	int TryWait() {
		Common::LockGuard lock(m_mutex);

		if (m_count <= 0) {
			SyncGuest();
			return POSIX_EAGAIN;
		}

		m_count--;
		SyncGuest();

		return OK;
	}

	int Post() {
		Common::LockGuard lock(m_mutex);

		if (m_count == POSIX_SEM_VALUE_MAX) {
			return POSIX_EOVERFLOW;
		}

		m_count++;
		SyncGuest();
		m_cond_var.Signal();

		return OK;
	}

	int GetValue() {
		Common::LockGuard lock(m_mutex);

		return m_count;
	}

private:
	void SyncGuest(uint32_t flags = std::numeric_limits<uint32_t>::max()) {
		if (m_guest != nullptr) {
			m_guest->magic       = POSIX_SEM_MAGIC;
			m_guest->nameid      = 0;
			m_guest->has_waiters = (m_waiters > 0 ? 1u : 0u);
			m_guest->count       = static_cast<uint32_t>(m_count);
			if (flags != std::numeric_limits<uint32_t>::max()) {
				m_guest->flags = flags;
			}
		}
	}

	Common::Mutex   m_mutex;
	Common::CondVar m_cond_var;
	PosixSemGuest*  m_guest   = nullptr;
	int             m_count   = 0;
	uint32_t        m_waiters = 0;
};

Common::Mutex                                        g_posix_sem_mutex;
std::unordered_map<PosixSemGuest*, PosixSemPrivate*> g_posix_sems;

static int PosixToKernel(int posix_errno) {
	return (posix_errno >= 0 && posix_errno <= 102 ? posix_errno - 2147352576
	                                               : LibKernel::KERNEL_ERROR_EINVAL);
}

static PosixSemPrivate* GetSem(void* sem) {
	if (sem == nullptr) {
		return nullptr;
	}

	Common::LockGuard lock(g_posix_sem_mutex);

	const auto it = g_posix_sems.find(static_cast<PosixSemGuest*>(sem));
	return (it != g_posix_sems.end() ? it->second : nullptr);
}

static int SetErrnoReturn(int posix_errno) {
	*GetErrorAddr() = posix_errno;
	return -1;
}

static int SemInitImpl(void* sem, int pshared, unsigned int value, const char* name) {
	if (sem == nullptr || value > POSIX_SEM_VALUE_MAX) {
		return POSIX_EINVAL;
	}

	auto* guest = static_cast<PosixSemGuest*>(sem);
	auto* obj   = new PosixSemPrivate(guest, value, static_cast<uint32_t>(pshared));

	{
		Common::LockGuard lock(g_posix_sem_mutex);

		if (auto it = g_posix_sems.find(guest); it != g_posix_sems.end()) {
			delete it->second;
			it->second = obj;
		} else {
			g_posix_sems.insert({guest, obj});
		}
	}

	static std::atomic<uint32_t> log_count {0};
	const auto                   count = log_count.fetch_add(1);
	if (count < 32) {
		LOGF("\t POSIX semaphore init: %s, value = %u, pshared = %d\n",
		     (name != nullptr ? name : "sem"), value, pshared);
	}

	return OK;
}

static int SemDestroyImpl(void* sem) {
	if (sem == nullptr) {
		return POSIX_EINVAL;
	}

	auto*            guest = static_cast<PosixSemGuest*>(sem);
	PosixSemPrivate* obj   = nullptr;

	{
		Common::LockGuard lock(g_posix_sem_mutex);

		const auto it = g_posix_sems.find(guest);
		if (it == g_posix_sems.end()) {
			return POSIX_EINVAL;
		}

		obj = it->second;
		g_posix_sems.erase(it);
	}

	delete obj;
	guest->magic       = 0;
	guest->has_waiters = 0;
	guest->count       = 0;
	guest->flags       = 0;

	return OK;
}

static int SemTimedwaitImpl(void* sem, uint32_t* micros) {
	auto* obj = GetSem(sem);
	if (obj == nullptr) {
		return POSIX_EINVAL;
	}

	return obj->Wait(micros);
}

static bool AbsTimespecToMicros(const LibKernel::KernelTimespec* abstime, uint32_t* micros) {
	if (abstime == nullptr || micros == nullptr || abstime->tv_sec < 0 || abstime->tv_nsec < 0 ||
	    abstime->tv_nsec >= 1000000000) {
		return false;
	}

	LibKernel::KernelTimespec now {};
	if (LibKernel::KernelClockGettime(0, &now) != OK) {
		return false;
	}

	const int64_t sec_delta  = abstime->tv_sec - now.tv_sec;
	const int64_t nsec_delta = abstime->tv_nsec - now.tv_nsec;
	const int64_t usec_delta = sec_delta * 1000000 + nsec_delta / 1000;

	if (usec_delta <= 0) {
		*micros = 0;
	} else if (usec_delta > std::numeric_limits<uint32_t>::max()) {
		*micros = std::numeric_limits<uint32_t>::max();
	} else {
		*micros = static_cast<uint32_t>(usec_delta);
	}

	return true;
}

} // namespace

int KYTY_SYSV_ABI sem_init(void* sem, int pshared, unsigned int value) {
	PRINT_NAME();

	const int result = SemInitImpl(sem, pshared, value, nullptr);
	return (result == OK ? 0 : SetErrnoReturn(result));
}

int KYTY_SYSV_ABI sem_destroy(void* sem) {
	PRINT_NAME();

	const int result = SemDestroyImpl(sem);
	return (result == OK ? 0 : SetErrnoReturn(result));
}

int KYTY_SYSV_ABI sem_wait(void* sem) {
	const LibKernel::WaitGraph::WaitScope wait_graph(
	    sem, LibKernel::WaitGraph::Prim::PosixSem,
	    reinterpret_cast<uint64_t>(__builtin_return_address(0))); // Astro Bot PC (KYTY_WAIT_GRAPH)
	const int result = SemTimedwaitImpl(sem, nullptr);
	return (result == OK ? 0 : SetErrnoReturn(result));
}

int KYTY_SYSV_ABI sem_trywait(void* sem) {
	auto* obj = GetSem(sem);
	if (obj == nullptr) {
		return SetErrnoReturn(POSIX_EINVAL);
	}

	const int result = obj->TryWait();
	return (result == OK ? 0 : SetErrnoReturn(result));
}

int KYTY_SYSV_ABI sem_reltimedwait_np(void* sem, uint32_t usec) {
	const LibKernel::WaitGraph::WaitScope wait_graph(
	    sem, LibKernel::WaitGraph::Prim::PosixSem,
	    reinterpret_cast<uint64_t>(__builtin_return_address(0))); // Astro Bot PC (KYTY_WAIT_GRAPH)
	const int result = SemTimedwaitImpl(sem, &usec);
	return (result == OK ? 0 : SetErrnoReturn(result));
}

int KYTY_SYSV_ABI sem_timedwait(void* sem, const LibKernel::KernelTimespec* abstime) {
	uint32_t micros = 0;
	if (!AbsTimespecToMicros(abstime, &micros)) {
		return SetErrnoReturn(POSIX_EINVAL);
	}

	const LibKernel::WaitGraph::WaitScope wait_graph(
	    sem, LibKernel::WaitGraph::Prim::PosixSem,
	    reinterpret_cast<uint64_t>(__builtin_return_address(0))); // Astro Bot PC (KYTY_WAIT_GRAPH)
	const int result = SemTimedwaitImpl(sem, &micros);
	return (result == OK ? 0 : SetErrnoReturn(result));
}

int KYTY_SYSV_ABI sem_post(void* sem) {
	auto* obj = GetSem(sem);
	if (obj == nullptr) {
		return SetErrnoReturn(POSIX_EINVAL);
	}

	LibKernel::WaitGraph::Signal(sem, reinterpret_cast<uint64_t>(__builtin_return_address(0))); // KYTY_WAIT_GRAPH
	const int result = obj->Post();
	return (result == OK ? 0 : SetErrnoReturn(result));
}

int KYTY_SYSV_ABI sem_getvalue(void* sem, int* value) {
	auto* obj = GetSem(sem);
	if (obj == nullptr || value == nullptr) {
		return SetErrnoReturn(POSIX_EINVAL);
	}

	*value = obj->GetValue();

	return 0;
}

} // namespace Libs::Posix

namespace Libs::LibKernel::Semaphore {

int KYTY_SYSV_ABI PthreadSemInit(void* sem, int flag, unsigned int value, const char* name) {
	PRINT_NAME();

	if (flag != 0) {
		return KERNEL_ERROR_EINVAL;
	}

	const int result = Posix::SemInitImpl(sem, 0, value, name);
	return (result == OK ? OK : Posix::PosixToKernel(result));
}

int KYTY_SYSV_ABI PthreadSemDestroy(void* sem) {
	PRINT_NAME();

	const int result = Posix::SemDestroyImpl(sem);
	return (result == OK ? OK : Posix::PosixToKernel(result));
}

int KYTY_SYSV_ABI PthreadSemWait(void* sem) {
	const LibKernel::WaitGraph::WaitScope wait_graph(
	    sem, LibKernel::WaitGraph::Prim::PosixSem,
	    reinterpret_cast<uint64_t>(__builtin_return_address(0))); // Astro Bot PC (KYTY_WAIT_GRAPH)
	const int result = Posix::SemTimedwaitImpl(sem, nullptr);
	return (result == OK ? OK : Posix::PosixToKernel(result));
}

int KYTY_SYSV_ABI PthreadSemTrywait(void* sem) {
	auto* obj = Posix::GetSem(sem);
	if (obj == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	const int result = obj->TryWait();
	return (result == OK ? OK : Posix::PosixToKernel(result));
}

int KYTY_SYSV_ABI PthreadSemTimedwait(void* sem, KernelUseconds usec) {
	uint32_t  micros = usec;
	const LibKernel::WaitGraph::WaitScope wait_graph(
	    sem, LibKernel::WaitGraph::Prim::PosixSem,
	    reinterpret_cast<uint64_t>(__builtin_return_address(0))); // Astro Bot PC (KYTY_WAIT_GRAPH)
	const int result = Posix::SemTimedwaitImpl(sem, &micros);
	return (result == OK ? OK : Posix::PosixToKernel(result));
}

int KYTY_SYSV_ABI PthreadSemPost(void* sem) {
	auto* obj = Posix::GetSem(sem);
	if (obj == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	LibKernel::WaitGraph::Signal(sem, reinterpret_cast<uint64_t>(__builtin_return_address(0))); // KYTY_WAIT_GRAPH
	const int result = obj->Post();
	return (result == OK ? OK : Posix::PosixToKernel(result));
}

int KYTY_SYSV_ABI PthreadSemGetvalue(void* sem, int* value) {
	auto* obj = Posix::GetSem(sem);
	if (obj == nullptr || value == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	*value = obj->GetValue();

	return OK;
}

} // namespace Libs::LibKernel::Semaphore
