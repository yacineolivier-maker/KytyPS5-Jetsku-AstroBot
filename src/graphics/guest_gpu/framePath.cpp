#include "graphics/guest_gpu/framePath.h"

#include "graphics/host_gpu/renderer/masterSemaphore.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cinttypes>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>

namespace Libs::Graphics::FramePath {

bool g_enabled = [] {
	const auto* value = std::getenv("KYTY_FRAME_PATH");
	const bool  on    = value != nullptr && std::strcmp(value, "1") == 0;
	if (on) {
		std::printf("Kyty chemin image : mesure active, une ligne toutes les 10 s (KYTY_FRAME_PATH)\n");
		std::fflush(stdout);
	}
	return on;
}();

namespace {

constexpr uint64_t ReportPeriodNs = 10'000'000'000ull;

// Latencies in log-spaced buckets (4 per power of two of microseconds, up to ~2 s).
class Histogram {
public:
	static constexpr uint32_t Buckets = 88;

	void Add(uint64_t ns) noexcept {
		m_buckets[Index(ns / 1000u)].fetch_add(1, std::memory_order_relaxed);
		m_sum_ns.fetch_add(ns, std::memory_order_relaxed);
		m_count.fetch_add(1, std::memory_order_relaxed);
	}

	struct Summary {
		uint64_t count   = 0;
		double   mean_ms = 0.0;
		double   p95_ms  = 0.0;
	};

	// Takes the window's values and clears them.
	Summary Take() noexcept {
		std::array<uint64_t, Buckets> counts {};
		uint64_t                      total = 0;
		for (uint32_t i = 0; i < Buckets; i++) {
			counts[i] = m_buckets[i].exchange(0, std::memory_order_relaxed);
			total += counts[i];
		}
		const auto sum = m_sum_ns.exchange(0, std::memory_order_relaxed);
		(void)m_count.exchange(0, std::memory_order_relaxed);
		Summary summary;
		summary.count = total;
		if (total == 0) {
			return summary;
		}
		summary.mean_ms        = static_cast<double>(sum) / static_cast<double>(total) / 1e6;
		const uint64_t rank    = (total * 95u + 99u) / 100u; // the 95th percentile's rank (1-based)
		uint64_t       running = 0;
		for (uint32_t i = 0; i < Buckets; i++) {
			running += counts[i];
			if (running >= rank) {
				summary.p95_ms = static_cast<double>(UpperUs(i)) / 1000.0;
				break;
			}
		}
		return summary;
	}

private:
	static uint32_t Index(uint64_t us) noexcept {
		if (us == 0) {
			return 0;
		}
		const auto log  = static_cast<uint32_t>(63 - std::countl_zero(us));
		const auto frac = static_cast<uint32_t>(log >= 2 ? (us >> (log - 2u)) & 3u : (us << (2u - log)) & 3u);
		return std::min<uint32_t>(1u + log * 4u + frac, Buckets - 1u);
	}
	// The bucket's upper bound in microseconds.
	static uint64_t UpperUs(uint32_t index) noexcept {
		if (index == 0) {
			return 1;
		}
		const uint32_t log  = (index - 1u) / 4u;
		const uint32_t frac = (index - 1u) % 4u;
		return std::max<uint64_t>(1, ((uint64_t {5u} + frac) << log) >> 2u);
	}

	std::array<std::atomic<uint64_t>, Buckets> m_buckets {};
	std::atomic<uint64_t>                      m_sum_ns {0};
	std::atomic<uint64_t>                      m_count {0};
};

enum Stage : uint32_t {
	Queue,      // enqueue -> CP start
	Translate,  // CP start -> CP complete
	GuestToCp,  // enqueue -> CP complete
	Record,     // CP complete -> handed to the submission queue
	Dispatch,   // handed over -> vkQueueSubmit returned
	Gpu,        // vkQueueSubmit returned -> timeline reached
	GuestToGpu, // enqueue -> timeline reached
	LabelRecord,
	LabelDeferred,
	AgcDone,
	StageCount
};

std::array<Histogram, StageCount> g_hist;

// Per-tick times of the monitored timeline (ring indexed by tick).
struct TickSlot {
	std::atomic<uint64_t> tick {0};
	std::atomic<uint64_t> ns {0};
};
constexpr uint64_t                 RingSize = 8192;
std::array<TickSlot, RingSize>     g_recorded;
std::array<TickSlot, RingSize>     g_dispatched;
std::atomic<uint64_t>              g_semaphore {0}; // the monitored timeline (VkSemaphore), 0: none

void StoreTick(std::array<TickSlot, RingSize>& ring, uint64_t tick, uint64_t ns) noexcept {
	auto& slot = ring[tick % RingSize];
	slot.tick.store(0, std::memory_order_relaxed);
	slot.ns.store(ns, std::memory_order_relaxed);
	slot.tick.store(tick, std::memory_order_release);
}

uint64_t LoadTick(const std::array<TickSlot, RingSize>& ring, uint64_t tick) noexcept {
	const auto& slot = ring[tick % RingSize];
	if (slot.tick.load(std::memory_order_acquire) != tick) {
		return 0;
	}
	const auto ns = slot.ns.load(std::memory_order_relaxed);
	return slot.tick.load(std::memory_order_acquire) == tick ? ns : 0;
}

struct Pending {
	uint64_t enqueue_ns  = 0;
	uint64_t start_ns    = 0;
	uint64_t complete_ns = 0;
	uint64_t tick        = 0;
};

std::mutex              g_mutex;
std::condition_variable g_cv;
std::deque<Pending>     g_pending;
MasterSemaphore*        g_master  = nullptr; // under g_mutex
std::thread*            g_monitor = nullptr; // leaked on purpose (no join at static destruction)
bool                    g_stop    = false;
std::atomic<uint64_t>   g_submissions {0};
std::atomic<uint64_t>   g_tick_misses {0};

// Main thread waits (ns in the window).
thread_local bool                                            t_main_thread = false;
std::array<std::atomic<uint64_t>, static_cast<size_t>(Wait::Count)> g_main_wait {};
std::atomic<uint64_t> g_main_yield_ns {0};
std::atomic<uint64_t> g_main_spin_ns {0};
std::atomic<uint64_t> g_main_yields {0};
uint64_t              g_main_last_yield_end = 0; // main thread only
constexpr uint64_t    SpinGapNs             = 50'000; // a gap this short between yields: a spin

std::atomic<uint64_t> g_agc_done_calls {0};

thread_local uint64_t t_slice_enqueue_ns = 0;

void Line(char* out, size_t size, const char* name, Histogram::Summary s) {
	std::snprintf(out, size, " %s %.2f/%.1f", name, s.mean_ms, s.p95_ms);
}

void Report(uint64_t window_ns) {
	const auto take = [](Stage stage) { return g_hist[stage].Take(); };
	const auto queue     = take(Queue);
	const auto translate = take(Translate);
	const auto guest_cp  = take(GuestToCp);
	const auto record    = take(Record);
	const auto dispatch  = take(Dispatch);
	const auto gpu       = take(Gpu);
	const auto guest_gpu = take(GuestToGpu);
	const auto label_rec = take(LabelRecord);
	const auto label_def = take(LabelDeferred);
	const auto agc       = take(AgcDone);
	const auto frames    = g_agc_done_calls.exchange(0, std::memory_order_relaxed);
	const auto subs      = g_submissions.exchange(0, std::memory_order_relaxed);
	const auto misses    = g_tick_misses.exchange(0, std::memory_order_relaxed);
	const double seconds = static_cast<double>(window_ns) / 1e9;

	std::array<uint64_t, static_cast<size_t>(Wait::Count)> wait {};
	for (size_t i = 0; i < wait.size(); i++) {
		wait[i] = g_main_wait[i].exchange(0, std::memory_order_relaxed);
	}
	const auto yield_ns = g_main_yield_ns.exchange(0, std::memory_order_relaxed);
	const auto spin_ns  = g_main_spin_ns.exchange(0, std::memory_order_relaxed);
	const auto yields   = g_main_yields.exchange(0, std::memory_order_relaxed);
	const auto pct = [window_ns](uint64_t ns) {
		return 100.0 * static_cast<double>(ns) / static_cast<double>(window_ns);
	};
	uint64_t total_wait = yield_ns + spin_ns;
	for (auto ns: wait) {
		total_wait += ns;
	}
	const double per_frame_ms =
	    frames != 0 ? static_cast<double>(total_wait) / 1e6 / static_cast<double>(frames) : 0.0;

	char stages[7][48];
	Line(stages[0], sizeof(stages[0]), "file", queue);
	Line(stages[1], sizeof(stages[1]), "traduction", translate);
	Line(stages[2], sizeof(stages[2]), "jeu->fin CP", guest_cp);
	Line(stages[3], sizeof(stages[3]), "enregistrement", record);
	Line(stages[4], sizeof(stages[4]), "soumission", dispatch);
	Line(stages[5], sizeof(stages[5]), "GPU", gpu);
	Line(stages[6], sizeof(stages[6]), "jeu->fin GPU", guest_gpu);
	std::printf("Kyty chemin image 10 s : %.1f images/s (AgcSuspendPoint), %" PRIu64
	            " soumissions | moy/p95 ms :%s%s%s%s%s%s%s [n=%" PRIu64 ", ticks manques %" PRIu64
	            "] | etiquettes EOP jeu->ecrite %.2f/%.1f [n=%" PRIu64
	            "] differees GPU->visible %.2f/%.1f [n=%" PRIu64 "] | AgcSuspendPoint %.0f ms/10 s "
	            "p95 %.1f | Thread_Main attend %.0f %% (%.1f ms/image) : yield %.0f %% + boucle "
	            "estimee %.0f %% (%" PRIu64 " yields/s), semaphore %.0f %%, cond %.0f %%, equeue %.0f "
	            "%%, sommeil %.0f %%, fautes %.0f %%\n",
	            static_cast<double>(frames) / seconds, subs, stages[0], stages[1], stages[2],
	            stages[3], stages[4], stages[5], stages[6], gpu.count, misses, label_rec.mean_ms,
	            label_rec.p95_ms, label_rec.count, label_def.mean_ms, label_def.p95_ms,
	            label_def.count, agc.mean_ms * static_cast<double>(agc.count) * (10.0 / seconds),
	            agc.p95_ms, pct(total_wait), per_frame_ms, pct(yield_ns), pct(spin_ns),
	            static_cast<uint64_t>(static_cast<double>(yields) / seconds),
	            pct(wait[static_cast<size_t>(Wait::Semaphore)]),
	            pct(wait[static_cast<size_t>(Wait::Cond)]),
	            pct(wait[static_cast<size_t>(Wait::Equeue)]),
	            pct(wait[static_cast<size_t>(Wait::Sleep)]),
	            pct(wait[static_cast<size_t>(Wait::Fault)]));
	std::fflush(stdout);
}

void MonitorRun() {
	uint64_t next_report = NowNs() + ReportPeriodNs;
	uint64_t last_report = NowNs();
	bool     failed      = false; // the timeline wait failed (device lost): stages C-D stop
	const auto maybe_report = [&] {
		const auto now = NowNs();
		if (now >= next_report) {
			Report(now - last_report);
			last_report = now;
			next_report = now + ReportPeriodNs;
		}
	};
	for (;;) {
		Pending          entry;
		MasterSemaphore* master = nullptr;
		{
			std::unique_lock lock(g_mutex);
			const auto       wait_ns = next_report > NowNs() ? next_report - NowNs() : 0;
			g_cv.wait_for(lock, std::chrono::nanoseconds(wait_ns),
			              [] { return g_stop || !g_pending.empty(); });
			if (g_stop) {
				return;
			}
			if (!g_pending.empty()) {
				entry = g_pending.front();
				g_pending.pop_front();
				master = g_master;
			}
		}
		if (master != nullptr && !failed) {
			// Waits for the tick in short steps (Detach stops it within one step).
			for (;;) {
				const int result = master->WaitQuiet(entry.tick, 5'000'000);
				if (result > 0) {
					break;
				}
				if (result < 0) {
					failed = true;
					break;
				}
				{
					std::lock_guard lock(g_mutex);
					if (g_stop) {
						return;
					}
				}
				maybe_report();
			}
			if (!failed) {
				const auto gpu_ns      = NowNs();
				const auto recorded    = LoadTick(g_recorded, entry.tick);
				const auto dispatched  = LoadTick(g_dispatched, entry.tick);
				if (dispatched != 0 && dispatched >= entry.complete_ns && gpu_ns >= dispatched) {
					if (recorded != 0 && recorded >= entry.complete_ns && dispatched >= recorded) {
						g_hist[Record].Add(recorded - entry.complete_ns);
						g_hist[Dispatch].Add(dispatched - recorded);
					}
					g_hist[Gpu].Add(gpu_ns - dispatched);
				} else {
					g_tick_misses.fetch_add(1, std::memory_order_relaxed);
				}
				if (entry.enqueue_ns != 0 && gpu_ns >= entry.enqueue_ns) {
					g_hist[GuestToGpu].Add(gpu_ns - entry.enqueue_ns);
				}
			}
		}
		maybe_report();
	}
}

} // namespace

void NoteSubmission(MasterSemaphore& master, uint64_t enqueue_ns, uint64_t start_ns,
                    uint64_t complete_ns, uint64_t tail_tick) {
	g_submissions.fetch_add(1, std::memory_order_relaxed);
	if (enqueue_ns != 0 && start_ns >= enqueue_ns) {
		g_hist[Queue].Add(start_ns - enqueue_ns);
	}
	if (start_ns != 0 && complete_ns >= start_ns) {
		g_hist[Translate].Add(complete_ns - start_ns);
	}
	if (enqueue_ns != 0 && complete_ns >= enqueue_ns) {
		g_hist[GuestToCp].Add(complete_ns - enqueue_ns);
	}
	std::lock_guard lock(g_mutex);
	if (g_stop) {
		return;
	}
	if (g_master == nullptr) {
		g_master = &master;
		g_semaphore.store(reinterpret_cast<uint64_t>(static_cast<VkSemaphore>(master.Handle())),
		                  std::memory_order_release);
		g_monitor = new std::thread(MonitorRun);
	}
	if (g_master != &master || tail_tick == 0) {
		return;
	}
	if (g_pending.size() >= 4096) {
		g_pending.pop_front(); // the monitor fell behind (should not happen): keep the newest
	}
	g_pending.push_back({enqueue_ns, start_ns, complete_ns, tail_tick});
	g_cv.notify_one();
}

void NoteRecorded(uint64_t semaphore, uint64_t tick) {
	if (semaphore != 0 && semaphore == g_semaphore.load(std::memory_order_acquire)) {
		StoreTick(g_recorded, tick, NowNs());
	}
}

void NoteDispatched(uint64_t semaphore, uint64_t tick) {
	if (semaphore != 0 && semaphore == g_semaphore.load(std::memory_order_acquire)) {
		StoreTick(g_dispatched, tick, NowNs());
	}
}

void Detach(const MasterSemaphore* master) {
	std::thread* monitor = nullptr;
	{
		std::lock_guard lock(g_mutex);
		if (g_master != master) {
			return;
		}
		g_stop   = true;
		g_master = nullptr;
		g_semaphore.store(0, std::memory_order_release);
		monitor = g_monitor;
		g_cv.notify_all();
	}
	if (monitor != nullptr && monitor->joinable() &&
	    monitor->get_id() != std::this_thread::get_id()) {
		monitor->join();
	}
}

void BeginSlice(uint64_t enqueue_ns) noexcept {
	t_slice_enqueue_ns = enqueue_ns;
}

void NoteLabelWrite() noexcept {
	const auto enqueue = t_slice_enqueue_ns;
	if (enqueue == 0) {
		return;
	}
	const auto now = NowNs();
	if (now >= enqueue) {
		g_hist[LabelRecord].Add(now - enqueue);
	}
}

void NoteDeferredLabel(uint64_t completed_ns, uint64_t visible_ns) noexcept {
	if (completed_ns != 0 && visible_ns >= completed_ns) {
		g_hist[LabelDeferred].Add(visible_ns - completed_ns);
	}
}

void NoteAgcDone(uint64_t wait_ns) noexcept {
	g_agc_done_calls.fetch_add(1, std::memory_order_relaxed);
	g_hist[AgcDone].Add(wait_ns);
}

void MarkMainThread() noexcept {
	t_main_thread = true;
}

bool OnMainThread() noexcept {
	return t_main_thread;
}

void NoteMainWait(Wait kind, uint64_t ns) noexcept {
	g_main_wait[static_cast<size_t>(kind)].fetch_add(ns, std::memory_order_relaxed);
}

void NoteMainYield(uint64_t begin_ns, uint64_t end_ns) noexcept {
	g_main_yields.fetch_add(1, std::memory_order_relaxed);
	if (end_ns >= begin_ns) {
		g_main_yield_ns.fetch_add(end_ns - begin_ns, std::memory_order_relaxed);
	}
	const auto last = g_main_last_yield_end;
	if (last != 0 && begin_ns >= last && begin_ns - last < SpinGapNs) {
		g_main_spin_ns.fetch_add(begin_ns - last, std::memory_order_relaxed);
	}
	g_main_last_yield_end = end_ns;
}

} // namespace Libs::Graphics::FramePath
