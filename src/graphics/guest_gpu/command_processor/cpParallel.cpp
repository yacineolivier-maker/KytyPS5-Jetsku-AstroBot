// Astro Bot PC (KYTY_CP_PAR): stage 1a of the parallel translation of the command lists
// (cpParallel.h, docs\PLAN-PARALLELE.md).

#include "graphics/guest_gpu/command_processor/cpParallel.h"

#include "graphics/guest_gpu/command_processor/cpOps.h" // Astro Bot PC (KYTY_CP_PAR=1b4): op names

#include "common/assert.h"
#include "common/hangWatchdog.h"
#include "common/profiler.h"
#include "common/threads.h"

#include <algorithm>
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
#include <string>
#include <thread>
#include <unordered_map>

namespace Libs::Graphics::CpPar {

namespace {

struct Stats {
	std::array<std::atomic<uint64_t>, MaxLanes> completed {};
	std::array<std::atomic<uint64_t>, MaxLanes> resolve_ns {};
	std::array<std::atomic<uint64_t>, static_cast<size_t>(Fallback::Count)> fallbacks {};
	std::atomic<uint64_t> slices_b {0};        // RunSlice hand-offs
	std::atomic<uint64_t> bounces_service {0}; // service commands run on Thread_Gpu for B
	std::atomic<uint64_t> bounces_flip {0};    // Flip / WaitFlipDone ops run on Thread_Gpu for B
	// KYTY_CP_PAR=1b1 (NotePrep): per context, committed / fallen-back draws since the last line,
	// the window totals (published, retired) now, and at the last line (Thread_Gpu only).
	std::array<std::atomic<uint64_t>, MaxLanes> prep_committed {};
	std::array<std::atomic<uint64_t>, MaxLanes> prep_fell_back {};
	std::array<std::atomic<uint64_t>, MaxLanes> window_published {};
	std::array<std::atomic<uint64_t>, MaxLanes> window_retired {};
	std::array<uint64_t, MaxLanes>       reported_published {};
	std::array<uint64_t, MaxLanes>       reported_retired {};
	// KYTY_CP_PAR=1b2 (EndSegment, FixUpImage): per context, segments, packets encoded in its
	// ring, Submit packets among them, native command buffers it recorded into (first segment in
	// each); image fix-ups checked at merge points and those that would not be no-ops; image
	// accesses outside any segment.
	std::array<std::atomic<uint64_t>, MaxLanes> segments {};
	std::array<std::atomic<uint64_t>, MaxLanes> segment_packets {};
	std::array<std::atomic<uint64_t>, MaxLanes> segment_submits {};
	std::array<std::atomic<uint64_t>, MaxLanes> segment_buffers {};
	std::array<std::atomic<uint64_t>, MaxLanes> fixups_checked {};
	std::array<std::atomic<uint64_t>, MaxLanes> fixups_needed {};
	std::atomic<uint64_t>                image_outside {0};
	// Astro Bot PC (KYTY_CP_PAR=2a): per lane, hand-off submits and the empty ones among them.
	std::array<std::atomic<uint64_t>, MaxLanes> lane_submits {};
	std::array<std::atomic<uint64_t>, MaxLanes> lane_empty {};
	std::array<std::atomic<uint64_t>, MaxLanes> lane_skipped {}; // lazy begin: no submit at the hand-off
	// Astro Bot PC (KYTY_CP_PAR=2b): per lane, the LaneStat counters.
	std::array<std::array<std::atomic<uint64_t>, static_cast<size_t>(LaneStat::Count)>, MaxLanes> lane_stats {};
	std::atomic<uint64_t>                fixups_needed_total {0};
	// KYTY_CP_PAR=1b3: run-ahead starts and ops, promotions by reason, merges and insertions with
	// their fix-ups, op lock waits per thread, time both threads were busy (overlap).
	std::atomic<uint64_t> runahead_starts {0};
	std::atomic<uint64_t> runahead_ops {0};
	std::atomic<uint64_t> runahead_ns {0};
	std::array<std::atomic<uint64_t>, static_cast<size_t>(Promotion::Count)> promotions {};
	std::atomic<uint64_t> merges {0};
	std::atomic<uint64_t> merge_fixups {0};
	std::atomic<uint64_t> merge_impossible {0};
	std::atomic<uint64_t> merge_images {0};
	std::array<std::atomic<uint64_t>, static_cast<size_t>(Insertion::Count)> insertions {};
	std::atomic<uint64_t> insertion_fixups {0};
	std::array<std::atomic<uint64_t>, MaxLanes> lock_wait_ns {};
	std::array<std::atomic<uint64_t>, MaxLanes> lock_takes {};
	std::atomic<uint64_t> overlap_ns {0};
	// Astro Bot PC (KYTY_CP_PAR=1b4): run-ahead ops by kind, its waits satisfied by its own writes,
	// its writes to checked addresses, Thread_Gpu's interleave points, cross accesses, book sizes.
	std::array<std::atomic<uint64_t>, static_cast<size_t>(CpSeq::OpKind::Count)> runahead_kinds {};
	std::atomic<uint64_t> local_waits {0};
	std::atomic<uint64_t> local_writes {0};
	std::atomic<uint64_t> interleave_offered {0};
	std::atomic<uint64_t> interleave_refused {0};
	std::array<std::atomic<uint64_t>, static_cast<size_t>(Cross::Count)> cross {};
	std::atomic<uint64_t> book_written {0};
	std::atomic<uint64_t> book_labels {0};
	// Astro Bot PC (KYTY_CP_PAR=1b5): stop reason -> op kind (OpKind::Count: none emitted yet);
	// outcomes of the run-aheads (completed targets) and their fraction done ahead (0, <25 %,
	// <50 %, <75 %, <100 %, 100 %); windows; deferred deliveries; order checks.
	std::array<std::array<std::atomic<uint64_t>, static_cast<size_t>(CpSeq::OpKind::Count) + 1>,
	           static_cast<size_t>(Promotion::Count)>
	    stop_kinds {};
	std::atomic<uint64_t>                outcomes {0};
	std::atomic<uint64_t>                outcomes_started {0};
	std::atomic<uint64_t>                outcome_ahead_ops {0};
	std::atomic<uint64_t>                outcome_ahead_draws {0};
	std::atomic<uint64_t>                outcome_rest_ops {0};
	std::atomic<uint64_t>                outcome_rest_draws {0};
	std::atomic<uint64_t>                outcome_max_ops {0};
	std::array<std::atomic<uint64_t>, 6> outcome_done {};
	std::atomic<uint64_t>                windows {0};
	std::atomic<uint64_t>                window_ns {0};
	std::atomic<uint64_t>                starved_ns {0};
	std::atomic<uint64_t>                deferred_interrupts {0};
	std::atomic<uint64_t>                deferred_stores {0};
	std::atomic<uint64_t>                deferred_late {0};
	std::atomic<uint64_t>                deferred_ops_captured {0};
	std::atomic<uint64_t>                deferred_ops_recorded {0};
	std::atomic<uint64_t>                order_violations {0};
	std::atomic<uint64_t>                stray_interrupts {0};
	std::atomic<uint64_t>                deferred_left {0};
	std::array<std::atomic<uint32_t>, static_cast<size_t>(Promotion::Count)> stop_logs {};
};

Stats& GetStats() {
	static Stats stats;
	return stats;
}

// Astro Bot PC (KYTY_CP_PAR=2a, N lanes): the calling thread's resolver context (0: Thread_Gpu or
// any other thread; k: CP resolver #k+1; the 1b3 run-ahead thread: 1).
thread_local uint32_t t_lane = 0;
// Index of a context in the per-context statistics (contexts 0 and 1 outside 2a).
inline size_t LaneIndex(uint32_t context) noexcept {
	return context < MaxLanes ? context : MaxLanes - 1;
}
// The lane taking the token at the hand-off in progress (CpPar::HandoffTarget).
std::atomic<uint32_t> g_handoff_target {0};
thread_local bool t_busy      = false; // Astro Bot PC (KYTY_CP_PAR=1b3): NoteBusy's last state

// Astro Bot PC (KYTY_CP_PAR=1b2): image state bookkeeping (cpParallel.h). An image's state is a
// hash of its tracked state (Image::CpParStateHash); per context segment, the state each image
// was first found in (entry requirement) and its last state; `merged`: the state each image was
// left in by the segments merged so far (or by an access outside any segment).
struct ImageBook {
	struct Segment {
		bool                                       open = false;
		std::unordered_map<const void*, uint64_t>  entry;
		std::unordered_map<const void*, uint64_t>  last;
	};
	std::mutex                                 mutex;
	std::array<Segment, MaxLanes>                  segments;
	std::unordered_map<const void*, uint64_t>  merged;
};

ImageBook& GetImageBook() {
	static ImageBook book;
	return book;
}

// The calling thread's open segment (-1: none). Set by BeginSegment / EndSegment on that thread.
thread_local int t_segment = -1;

// The merge point of one image of a segment of `context` (EndSegment, book locked): the
// transition from the state the merged segments left (`global`) to the segment's entry
// requirement. Stage 1b-iii records it there (into the merged stream, before the segment); in
// 1b2 execution is in turn, so it must be a no-op: only counted (and the first ones logged).
void FixUpImage(uint32_t context, const void* image, uint64_t global, uint64_t entry) {
	auto& stats = GetStats();
	stats.fixups_checked[context].fetch_add(1, std::memory_order_relaxed);
	if (global == entry) {
		return;
	}
	stats.fixups_needed[context].fetch_add(1, std::memory_order_relaxed);
	if (stats.fixups_needed_total.fetch_add(1, std::memory_order_relaxed) < 16) {
		std::printf("Kyty CpPar: image %p of a context %c segment was found in state 0x%016" PRIx64
		            ", the merged segments left it in 0x%016" PRIx64 " (fix-up not a no-op)\n",
		            image, LaneLetter(context), entry, global);
	}
}

// The token: exactly one of Thread_Gpu and resolver #2 runs CP work at any time.
// Astro Bot PC (KYTY_CP_PAR=2a, N lanes): one Resolver2 per lane 1..N-1 ("CP resolver #k+1"),
// still in turn: Thread_Gpu hands the token to one of them at a time.
class Resolver2;
thread_local Resolver2* t_resolver = nullptr; // the resolver object of the calling resolver thread
class Resolver2 {
public:
	Resolver2(const Hooks& hooks, uint32_t lane): m_hooks(hooks), m_lane(lane) {
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
		EXIT_IF(t_lane != 0);
		GiveToLane();
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
				GiveToLane();
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
		EXIT_IF(t_lane != m_lane);
		GiveToGpuThread();
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
	// Astro Bot PC (KYTY_CP_PAR=2a): the giver names the taker first (HandoffTarget).
	void GiveToLane() {
		g_handoff_target.store(m_lane, std::memory_order_relaxed);
		m_hooks.give(m_hooks.context);
	}
	void GiveToGpuThread() {
		g_handoff_target.store(0, std::memory_order_relaxed);
		m_hooks.give(m_hooks.context);
	}

	void Run() {
		t_lane     = m_lane;
		t_resolver = this;
		switch (m_lane) {
			case 1: KYTY_PROFILER_THREAD("CP resolver #2"); break;
			case 2: KYTY_PROFILER_THREAD("CP resolver #3"); break;
			default: KYTY_PROFILER_THREAD("CP resolver #4"); break;
		}
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
			GiveToGpuThread();
			lock.lock();
			m_state = State::SliceDone;
			m_wake_a.notify_one();
		}
	}

	Hooks                   m_hooks;
	uint32_t                m_lane = 1;
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

// Astro Bot PC (KYTY_CP_PAR=2a, N lanes): index = lane (1..N-1); [0] unused.
std::array<std::unique_ptr<Resolver2>, MaxLanes> g_resolvers;

// Astro Bot PC (KYTY_CP_PAR=1b3, cpParallel.h): the CP op lock. FIFO (tickets), so that two
// contending threads alternate op by op; `last` is the previous holder.
class OpLock {
public:
	bool Acquire(uint32_t who) {
		std::unique_lock lock(m_mutex);
		const auto ticket = m_next++;
		if (m_serving != ticket) {
			const auto start = NowNs();
			m_waiters++;
			HangWatchdog::Scope wait("cp-par-op-lock", reinterpret_cast<uint64_t>(this), who);
			m_wake.wait(lock, [this, ticket] { return m_serving == ticket; });
			m_waiters--;
			GetStats().lock_wait_ns[who != 0 ? 1 : 0].fetch_add(NowNs() - start,
			                                                    std::memory_order_relaxed);
		}
		EXIT_IF(m_holder != UINT32_MAX);
		m_holder           = who;
		const bool changed = m_last != UINT32_MAX && m_last != who;
		m_last             = who;
		GetStats().lock_takes[who != 0 ? 1 : 0].fetch_add(1, std::memory_order_relaxed);
		return changed;
	}
	void Release(uint32_t who) {
		uint32_t waiters = 0;
		{
			std::lock_guard lock(m_mutex);
			EXIT_IF(m_holder != who);
			m_holder = UINT32_MAX;
			m_serving++;
			waiters = m_waiters;
		}
		if (waiters != 0) {
			m_wake.notify_all();
		}
	}

private:
	std::mutex              m_mutex;
	std::condition_variable m_wake;
	uint64_t                m_next    = 0;
	uint64_t                m_serving = 0;
	uint32_t                m_waiters = 0;
	uint32_t                m_holder  = UINT32_MAX;
	uint32_t                m_last    = UINT32_MAX;
};

OpLock& GetOpLock() {
	static OpLock lock;
	return lock;
}

std::atomic<bool> g_runahead_dirty {false};

// Astro Bot PC (KYTY_CP_PAR=1b3): CP resolver #2 as the run-ahead thread. Thread_Gpu offers a
// task; the thread runs it once; Stop cancels an offer not taken yet or waits for the task's end.
class RunAheadThread {
public:
	RunAheadThread() {
		m_thread = std::thread([this] { Run(); });
	}
	~RunAheadThread() {
		{
			std::lock_guard lock(m_mutex);
			EXIT_IF(m_busy || m_offered);
			m_quit = true;
		}
		m_wake.notify_all();
		if (m_thread.joinable()) {
			m_thread.join();
		}
	}
	RunAheadThread(const RunAheadThread&)            = delete;
	RunAheadThread& operator=(const RunAheadThread&) = delete;

	void Offer(void (*task)(void*), void* argument) {
		{
			std::lock_guard lock(m_mutex);
			EXIT_IF(m_busy || m_offered);
			m_task     = task;
			m_argument = argument;
			m_offered  = true;
			m_stop.store(false, std::memory_order_release);
		}
		m_wake.notify_all();
	}
	// Thread_Gpu, holding the op lock.
	bool Stop() {
		{
			std::lock_guard lock(m_mutex);
			if (m_offered) {
				m_offered = false; // not taken yet
				return false;
			}
			if (!m_busy) {
				return false;
			}
			m_stop.store(true, std::memory_order_release);
		}
		// The task needs the op lock to reach its next op boundary.
		OpLockRelease(0);
		{
			std::unique_lock    lock(m_mutex);
			HangWatchdog::Scope wait("cp-par-runahead-stop", reinterpret_cast<uint64_t>(this));
			m_idle.wait(lock, [this] { return !m_busy; });
		}
		return OpLockAcquire(0);
	}
	[[nodiscard]] bool StopRequested() const noexcept {
		return m_stop.load(std::memory_order_acquire);
	}

private:
	void Run() {
		t_lane = 1;
		KYTY_PROFILER_THREAD("CP resolver #2");
		Common::RaiseCurrentThreadPriority();
		std::unique_lock lock(m_mutex);
		for (;;) {
			m_wake.wait(lock, [this] { return m_offered || m_quit; });
			if (m_quit) {
				return;
			}
			m_offered           = false;
			m_busy              = true;
			const auto task     = m_task;
			auto*      argument = m_argument;
			lock.unlock();
			task(argument);
			lock.lock();
			m_busy = false;
			m_stop.store(false, std::memory_order_release);
			m_idle.notify_all();
		}
	}

	std::mutex              m_mutex;
	std::condition_variable m_wake;
	std::condition_variable m_idle;
	bool                    m_offered  = false;
	bool                    m_busy     = false;
	bool                    m_quit     = false;
	std::atomic<bool>       m_stop {false};
	void (*m_task)(void*)              = nullptr;
	void*                   m_argument = nullptr;
	std::thread             m_thread;
};

std::unique_ptr<RunAheadThread> g_runahead;

// Astro Bot PC (KYTY_CP_PAR=1b3): the quiet tails the sequencer published (PublishStreamEnd), a
// few submissions deep (it may parse ahead of Thread_Gpu).
struct TailSlot {
	std::atomic<uint64_t> submission {0};
	std::atomic<uint64_t> last_sync {0};
};
std::array<TailSlot, 16> g_tails;

InsertHook g_insert_hook         = nullptr;
void*      g_insert_hook_context = nullptr;

// Overlap accounting (NoteBusy): which threads are busy, since when both are.
std::mutex g_busy_mutex;
uint32_t   g_busy_mask  = 0;
uint64_t   g_both_since = 0;

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
		if (std::strcmp(value, "1b2") == 0) {
			std::printf("Kyty CpPar: KYTY_CP_PAR=1b2 (as 1b1, and context B records into its own "
			            "command buffer wrapper and stream ring, executed in turn)\n");
			return Mode::Serial1b2;
		}
		if (std::strcmp(value, "1b3") == 0) {
			std::printf("Kyty CpPar: KYTY_CP_PAR=1b3 (as 1b2, and CP resolver #2 runs the next "
			            "submission's leading draws ahead, op by op under the CP op lock)\n");
			return Mode::Concurrent1b3;
		}
		if (std::strcmp(value, "1b4") == 0) {
			// Astro Bot PC (KYTY_CP_PAR=1b4, cpParallel.h).
			std::printf("Kyty CpPar: KYTY_CP_PAR=1b4 (as 1b3, and the run-ahead also executes the next "
			            "submission's own CP writes and waits; interleave points at every op of the "
			            "oldest once no hard op remains in it)\n");
			return Mode::Concurrent1b4;
		}
		if (std::strcmp(value, "1b5") == 0) {
			// Astro Bot PC (KYTY_CP_PAR=1b5, cpParallel.h).
			std::printf("Kyty CpPar: KYTY_CP_PAR=1b5 (as 1b4, and the run-ahead's end-of-pipe "
			            "interrupts are captured and delivered in order at the merge, after the "
			            "oldest submission's last op)\n");
			return Mode::Concurrent1b5;
		}
		if (std::strcmp(value, "2a") == 0) {
			// Astro Bot PC (KYTY_CP_PAR=2a, cpParallel.h).
			std::printf("Kyty CpPar: KYTY_CP_PAR=2a (as 1b2, and each lane records into its own native "
			            "command buffers with its own RenderExecutor; submitted in list order at "
			            "every hand-off, executed in turn)\n");
			return Mode::Lanes2a;
		}
		if (std::strcmp(value, "2b") == 0) {
			// Astro Bot PC (KYTY_CP_PAR=2b, cpParallel.h).
			std::printf("Kyty CpPar: KYTY_CP_PAR=2b (as 2a, and each lane has its own replay thread; the "
			            "lanes' native buffers are handed to the queue in list order)\n");
			return Mode::Lanes2b;
		}
		std::printf("Kyty CpPar: KYTY_CP_PAR=%s is not supported (only 1a, 1b1, 1b2, 1b3, 1b4, 1b5, 2a, "
		            "2b); off\n",
		            value);
		return Mode::Off;
	}();
	return mode;
}

bool VerifyEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_CP_PAR_VERIFY");
		const bool  on    = value != nullptr && std::strcmp(value, "1") == 0;
		if (on && GetMode() == Mode::Concurrent1b3) {
			std::printf("Kyty CpPar: KYTY_CP_PAR_VERIFY=1 (stop at the first insertion)\n");
		}
		if (on && GetMode() == Mode::Concurrent1b4) {
			// Astro Bot PC (KYTY_CP_PAR=1b4).
			std::printf("Kyty CpPar: KYTY_CP_PAR_VERIFY=1 (stop at the first insertion, foreign wait "
			            "value or cross access of the overlap)\n");
		}
		if (on && GetMode() == Mode::Concurrent1b5) {
			// Astro Bot PC (KYTY_CP_PAR=1b5).
			std::printf("Kyty CpPar: KYTY_CP_PAR_VERIFY=1 (stop at the first insertion, foreign wait "
			            "value, cross access of the overlap, out-of-order or uncaptured interrupt)\n");
		}
		return on;
	}();
	return enabled;
}

// Astro Bot PC (KYTY_CP_PAR=1b4, cpParallel.h).
bool WritesAll() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_CP_PAR_WRITES");
		const bool  all   = value != nullptr && std::strcmp(value, "all") == 0;
		if (GetMode() == Mode::Concurrent1b4 || GetMode() == Mode::Concurrent1b5) {
			std::printf("Kyty CpPar: 1b4 run-ahead writes: %s (KYTY_CP_PAR_WRITES)\n",
			            all ? "any address (measurement only)" : "learned CP-internal labels only");
		}
		return all;
	}();
	return enabled;
}

bool OldestWritesHard() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_CP_PAR_OLDEST_WRITES");
		const bool  hard  = value != nullptr && std::strcmp(value, "hard") == 0;
		if (hard && (GetMode() == Mode::Concurrent1b4 || GetMode() == Mode::Concurrent1b5)) {
			std::printf("Kyty CpPar: 1b4: the oldest's WRITE_DATA / DMA_DATA / clock writes are hard "
			            "(KYTY_CP_PAR_OLDEST_WRITES)\n");
		}
		return hard;
	}();
	return enabled;
}

bool InterleaveNext() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_CP_PAR_INTERLEAVE");
		const bool  next  = value != nullptr && std::strcmp(value, "next") == 0;
		if (GetMode() == Mode::Concurrent1b4 || GetMode() == Mode::Concurrent1b5) {
			std::printf("Kyty CpPar: 1b4 interleave points: %s (KYTY_CP_PAR_INTERLEAVE)\n",
			            next ? "before every op of the oldest that is not hard (measurement)"
			                 : "before every op of the oldest once no hard op remains in it");
		}
		return next;
	}();
	return enabled;
}

void NoteRunAheadKind(uint32_t kind) {
	if (kind < static_cast<uint32_t>(CpSeq::OpKind::Count)) {
		GetStats().runahead_kinds[kind].fetch_add(1, std::memory_order_relaxed);
	}
}

void NoteLocalWait() {
	GetStats().local_waits.fetch_add(1, std::memory_order_relaxed);
}

void NoteLocalWrite() {
	GetStats().local_writes.fetch_add(1, std::memory_order_relaxed);
}

void NoteInterleave(bool offered) {
	(offered ? GetStats().interleave_offered : GetStats().interleave_refused)
	    .fetch_add(1, std::memory_order_relaxed);
}

void NoteCross(Cross which, uint64_t address) {
	static constexpr const char* names[] = {
	    "Thread_Gpu read a dword the run-ahead wrote during the overlap",
	    "Thread_Gpu wrote a dword the run-ahead wrote or read during the overlap",
	    "the run-ahead read a dword last written by the oldest submission",
	    "a run-ahead wait passed on a value another submission wrote"};
	const auto index = static_cast<size_t>(which);
	if (VerifyEnabled()) {
		EXIT("Kyty CpPar: KYTY_CP_PAR_VERIFY=1: %s (0x%016" PRIx64 ")\n", names[index], address);
	}
	if (GetStats().cross[index].fetch_add(1, std::memory_order_relaxed) == 0) {
		std::printf("Kyty CpPar: first cross access: %s (0x%016" PRIx64 "; counted in \"CpPar 10s: "
		            "%s window\")\n",
		            names[index], address, ConcurrentName());
	}
}

void NoteBook(uint64_t written, uint64_t labels) {
	GetStats().book_written.store(written, std::memory_order_relaxed);
	GetStats().book_labels.store(labels, std::memory_order_relaxed);
}

// ---- Astro Bot PC (KYTY_CP_PAR=1b5, cpParallel.h) ----

void NoteStopKind(Promotion reason, uint32_t kind) {
	const auto r = static_cast<size_t>(reason);
	const auto k = std::min<size_t>(kind, static_cast<size_t>(CpSeq::OpKind::Count));
	if (r < static_cast<size_t>(Promotion::Count)) {
		GetStats().stop_kinds[r][k].fetch_add(1, std::memory_order_relaxed);
	}
}

void NoteOutcome(bool started, uint64_t ahead_ops, uint64_t ahead_draws, uint64_t rest_ops,
                 uint64_t rest_draws) {
	auto& stats = GetStats();
	stats.outcomes.fetch_add(1, std::memory_order_relaxed);
	if (started) {
		stats.outcomes_started.fetch_add(1, std::memory_order_relaxed);
	}
	stats.outcome_ahead_ops.fetch_add(ahead_ops, std::memory_order_relaxed);
	stats.outcome_ahead_draws.fetch_add(ahead_draws, std::memory_order_relaxed);
	stats.outcome_rest_ops.fetch_add(rest_ops, std::memory_order_relaxed);
	stats.outcome_rest_draws.fetch_add(rest_draws, std::memory_order_relaxed);
	const auto total = ahead_ops + rest_ops;
	auto       max   = stats.outcome_max_ops.load(std::memory_order_relaxed);
	while (total > max &&
	       !stats.outcome_max_ops.compare_exchange_weak(max, total, std::memory_order_relaxed)) {
	}
	size_t bucket = 0;
	if (ahead_ops == 0) {
		bucket = 0;
	} else if (rest_ops == 0) {
		bucket = 5;
	} else {
		bucket = 1 + std::min<size_t>(3, static_cast<size_t>((ahead_ops * 4u) / total));
	}
	stats.outcome_done[bucket].fetch_add(1, std::memory_order_relaxed);
}

void NoteWindow(uint64_t window_ns) {
	auto& stats = GetStats();
	stats.windows.fetch_add(1, std::memory_order_relaxed);
	stats.window_ns.fetch_add(window_ns, std::memory_order_relaxed);
}

void NoteStarved(uint64_t ns) {
	GetStats().starved_ns.fetch_add(ns, std::memory_order_relaxed);
}

void NoteDeferredDelivery(uint64_t interrupts, uint64_t stores, bool late) {
	auto& stats = GetStats();
	stats.deferred_interrupts.fetch_add(interrupts, std::memory_order_relaxed);
	stats.deferred_stores.fetch_add(stores, std::memory_order_relaxed);
	if (late && stats.deferred_late.fetch_add(1, std::memory_order_relaxed) == 0) {
		std::printf("Kyty CpPar: 1b5: a deferred interrupt list was delivered at a hand-over, not at "
		            "the merge (counted in \"CpPar 10s: 1b5\")\n");
	}
}

void NoteDeferredOp(bool store_captured) {
	(store_captured ? GetStats().deferred_ops_captured : GetStats().deferred_ops_recorded)
	    .fetch_add(1, std::memory_order_relaxed);
}

void NoteInterruptOrder(uint64_t sequence, bool deferred) {
	static uint64_t last = 0; // Thread_Gpu only (queue-0 ops and the merge run there)
	if (sequence <= last && last != 0) {
		if (VerifyEnabled()) {
			EXIT("Kyty CpPar: KYTY_CP_PAR_VERIFY=1: %s interrupt of op %" PRIu64
			     " queued after the interrupt of op %" PRIu64 " (out of order)\n",
			     deferred ? "a deferred" : "a serial", sequence, last);
		}
		if (GetStats().order_violations.fetch_add(1, std::memory_order_relaxed) == 0) {
			std::printf("Kyty CpPar: 1b5: first out-of-order interrupt: op %" PRIu64 " after op %" PRIu64
			            " (%s)\n",
			            sequence, last, deferred ? "deferred" : "serial");
		}
	}
	last = std::max(last, sequence);
}

void NoteStrayInterrupt() {
	if (VerifyEnabled()) {
		EXIT("Kyty CpPar: KYTY_CP_PAR_VERIFY=1: CP resolver #2 raised an end-of-pipe interrupt "
		     "outside the run-ahead's capture\n");
	}
	if (GetStats().stray_interrupts.fetch_add(1, std::memory_order_relaxed) == 0) {
		std::printf("Kyty CpPar: 1b5: first interrupt raised by CP resolver #2 outside the capture\n");
	}
}

void NoteDeferredLeft(uint64_t entries) {
	if (VerifyEnabled()) {
		EXIT("Kyty CpPar: KYTY_CP_PAR_VERIFY=1: %" PRIu64 " deferred interrupt entries left at a "
		     "new offer\n",
		     entries);
	}
	GetStats().deferred_left.fetch_add(1, std::memory_order_relaxed);
}

bool TakeStopLog(Promotion reason) {
	const auto r = static_cast<size_t>(reason);
	return r < static_cast<size_t>(Promotion::Count) &&
	       GetStats().stop_logs[r].fetch_add(1, std::memory_order_relaxed) < 16u;
}

// ---- Astro Bot PC (KYTY_CP_PAR=1b3) ----

bool RunAheadDirty() noexcept {
	return g_runahead_dirty.load(std::memory_order_acquire);
}

void SetRunAheadDirty(bool dirty) noexcept {
	g_runahead_dirty.store(dirty, std::memory_order_release);
}

bool OpLockAcquire(uint32_t who) {
	// A busy thread is idle while it waits for the lock (overlap statistics).
	const bool busy = t_busy;
	if (busy) {
		NoteBusy(who, false);
	}
	const bool changed = GetOpLock().Acquire(who);
	if (busy) {
		NoteBusy(who, true);
	}
	return changed;
}

void OpLockRelease(uint32_t who) {
	GetOpLock().Release(who);
}

void StartRunAheadThread() {
	EXIT_IF(g_resolvers[1] != nullptr || g_runahead != nullptr);
	g_runahead = std::make_unique<RunAheadThread>();
	std::printf("Kyty CpPar: CP resolver #2 started (run-ahead)\n");
}

void OfferRunAhead(void (*task)(void*), void* argument) {
	EXIT_IF(g_runahead == nullptr);
	g_runahead->Offer(task, argument);
}

bool StopRunAhead() {
	if (g_runahead != nullptr) {
		return g_runahead->Stop();
	}
	return false;
}

bool RunAheadStopRequested() noexcept {
	return g_runahead != nullptr && g_runahead->StopRequested();
}

void PublishStreamEnd(uint64_t submission, uint64_t last_sync) {
	auto& slot = g_tails[submission % g_tails.size()];
	slot.submission.store(0, std::memory_order_release);
	slot.last_sync.store(last_sync, std::memory_order_release);
	slot.submission.store(submission, std::memory_order_release);
}

bool TailQuiet(uint64_t submission, uint64_t next_op) noexcept {
	auto& slot = g_tails[submission % g_tails.size()];
	if (slot.submission.load(std::memory_order_acquire) != submission) {
		return false;
	}
	const auto last_sync = slot.last_sync.load(std::memory_order_acquire);
	// Re-checked: the slot was not reused for a later submission meanwhile.
	return slot.submission.load(std::memory_order_acquire) == submission && next_op >= last_sync;
}

void SetInsertHook(InsertHook hook, void* context) {
	g_insert_hook         = hook;
	g_insert_hook_context = context;
}

void BeforeNativeSubmit() {
	if (!RunAheadDirty() || g_insert_hook == nullptr) {
		return;
	}
	g_insert_hook(g_insert_hook_context, static_cast<uint32_t>(OnResolver2()
	                                                                ? Insertion::RunAheadSubmit
	                                                                : Insertion::OldestSubmit));
}

void BeforeRecorderDrain() {
	// Thread_Gpu's drains reach its marker (its ring is the one being replayed).
	if (!OnResolver2() || !RunAheadDirty() || g_insert_hook == nullptr) {
		return;
	}
	g_insert_hook(g_insert_hook_context, static_cast<uint32_t>(Insertion::RunAheadDrain));
}

void NotePromotion(Promotion reason) {
	GetStats().promotions[static_cast<size_t>(reason)].fetch_add(1, std::memory_order_relaxed);
}

void NoteRunAheadStart() {
	GetStats().runahead_starts.fetch_add(1, std::memory_order_relaxed);
}

void NoteRunAheadOp(uint64_t ns) {
	GetStats().runahead_ns.fetch_add(ns, std::memory_order_relaxed);
	GetStats().runahead_ops.fetch_add(1, std::memory_order_relaxed);
}

void NoteMerge(uint64_t fixups, uint64_t impossible, uint64_t images) {
	auto& stats = GetStats();
	stats.merges.fetch_add(1, std::memory_order_relaxed);
	stats.merge_fixups.fetch_add(fixups, std::memory_order_relaxed);
	stats.merge_impossible.fetch_add(impossible, std::memory_order_relaxed);
	stats.merge_images.fetch_add(images, std::memory_order_relaxed);
}

void NoteInsertion(Insertion trigger, uint64_t fixups, uint64_t impossible, uint64_t images) {
	auto& stats = GetStats();
	static constexpr const char* names[] = {"a native submit of the oldest context",
	                                        "a native submit of the run-ahead",
	                                        "a recorder drain of the run-ahead"};
	const auto index = static_cast<size_t>(trigger);
	if (VerifyEnabled()) {
		EXIT("Kyty CpPar: KYTY_CP_PAR_VERIFY=1: insertion forced by %s (the run-ahead's ops so far "
		     "are recorded before the oldest submission's remaining work)\n",
		     names[index]);
	}
	if (stats.insertions[index].fetch_add(1, std::memory_order_relaxed) == 0) {
		std::printf("Kyty CpPar: first insertion forced by %s (data order of the overlap "
		            "inverted; counted in \"CpPar 10s: %s\")\n",
		            names[index], ConcurrentName());
	}
	stats.insertion_fixups.fetch_add(fixups, std::memory_order_relaxed);
	stats.merge_impossible.fetch_add(impossible, std::memory_order_relaxed);
	stats.merge_images.fetch_add(images, std::memory_order_relaxed);
}

void NoteBusy(uint32_t who, bool busy) {
	t_busy              = busy;
	const uint32_t  bit = who != 0 ? 2u : 1u;
	std::lock_guard lock(g_busy_mutex);
	const bool      both_before = g_busy_mask == 3u;
	g_busy_mask                 = busy ? (g_busy_mask | bit) : (g_busy_mask & ~bit);
	const bool both_after       = g_busy_mask == 3u;
	if (!both_before && both_after) {
		g_both_since = NowNs();
	} else if (both_before && !both_after) {
		GetStats().overlap_ns.fetch_add(NowNs() - g_both_since, std::memory_order_relaxed);
	}
}

uint64_t NowNs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

uint32_t RequestedLanes() {
	static const uint32_t lanes = [] {
		if (!LanesMode()) { // Astro Bot PC (KYTY_CP_PAR=2a, 2b)
			return 2u;
		}
		const auto* value = std::getenv("KYTY_CP_PAR_LANES");
		if (value == nullptr || *value == '\0') {
			return 2u;
		}
		const auto parsed = std::strtoul(value, nullptr, 10);
		const auto count  = static_cast<uint32_t>(std::clamp<unsigned long>(parsed, 2ul, MaxLanes));
		if (count != parsed) {
			std::printf("Kyty CpPar: KYTY_CP_PAR_LANES must be 2..%u (got '%s'): %u\n", MaxLanes,
			            value, count);
		}
		return count;
	}();
	return lanes;
}

uint32_t ThreadLane() noexcept {
	return t_lane;
}

uint32_t HandoffTarget() noexcept {
	return g_handoff_target.load(std::memory_order_relaxed);
}

void Start(const Hooks& hooks) {
	EXIT_IF(g_resolvers[1] != nullptr || hooks.take == nullptr || hooks.give == nullptr);
	// Astro Bot PC (KYTY_CP_PAR=2a): one resolver per lane after A (one outside 2a).
	const uint32_t count = ContextCount();
	for (uint32_t lane = 1; lane < count; lane++) {
		g_resolvers[lane] = std::make_unique<Resolver2>(hooks, lane);
	}
	std::printf("Kyty CpPar: CP resolver #2%s started\n",
	            count == 2 ? "" : (count == 3 ? " and #3" : " to #4"));
}

void Stop() {
	for (auto& resolver: g_resolvers) {
		resolver.reset();
	}
	g_runahead.reset(); // Astro Bot PC (KYTY_CP_PAR=1b3)
}

bool Started() {
	return g_resolvers[1] != nullptr || g_runahead != nullptr;
}

void RunSlice(Task task, void* argument) {
	RunSlice(1, task, argument);
}

void RunSlice(uint32_t lane, Task task, void* argument) {
	EXIT_IF(lane == 0 || lane >= MaxLanes || g_resolvers[lane] == nullptr);
	GetStats().slices_b.fetch_add(1, std::memory_order_relaxed);
	g_resolvers[lane]->RunSlice(task, argument);
}

void RunOnGpuThread(Task task, void* argument) {
	EXIT_IF(t_resolver == nullptr);
	t_resolver->RunOnGpuThread(task, argument);
}

bool OnResolver2() noexcept {
	return t_lane != 0;
}

void NoteFallback(Fallback reason) {
	GetStats().fallbacks[static_cast<size_t>(reason)].fetch_add(1, std::memory_order_relaxed);
}

void NoteCompleted(uint32_t context) {
	GetStats().completed[LaneIndex(context)].fetch_add(1, std::memory_order_relaxed);
}

void NoteResolveTime(uint32_t context, uint64_t ns) {
	GetStats().resolve_ns[LaneIndex(context)].fetch_add(ns, std::memory_order_relaxed);
}

void NoteBounce(bool flip_op) {
	(flip_op ? GetStats().bounces_flip : GetStats().bounces_service)
	    .fetch_add(1, std::memory_order_relaxed);
}

void NotePrep(uint32_t context, uint64_t committed, uint64_t fell_back, uint64_t published,
              uint64_t retired) {
	const size_t index = LaneIndex(context);
	auto&        stats = GetStats();
	stats.prep_committed[index].fetch_add(committed, std::memory_order_relaxed);
	stats.prep_fell_back[index].fetch_add(fell_back, std::memory_order_relaxed);
	stats.window_published[index].store(published, std::memory_order_relaxed);
	stats.window_retired[index].store(retired, std::memory_order_relaxed);
}

// Astro Bot PC (KYTY_CP_PAR=1b2): image state bookkeeping and segments (cpParallel.h).
void NoteImageState(const void* image, uint64_t before, uint64_t after) {
	auto&            book = GetImageBook();
	const int        open = t_segment;
	std::lock_guard  lock(book.mutex);
	if (open < 0 || !book.segments[open].open) {
		book.merged[image] = after;
		GetStats().image_outside.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	auto& segment = book.segments[open];
	segment.entry.try_emplace(image, before);
	segment.last[image] = after;
}

void ForgetImage(const void* image) {
	auto&           book = GetImageBook();
	std::lock_guard lock(book.mutex);
	book.merged.erase(image);
	for (auto& segment: book.segments) {
		segment.entry.erase(image);
		segment.last.erase(image);
	}
}

void BeginSegment(uint32_t context) {
	const size_t    index = LaneIndex(context);
	auto&           book  = GetImageBook();
	std::lock_guard lock(book.mutex);
	EXIT_IF(book.segments[index].open);
	book.segments[index].open = true;
	t_segment                 = static_cast<int>(index);
}

void EndSegment(uint32_t context, uint64_t packets, uint64_t submits, bool new_command_buffer) {
	const size_t index = LaneIndex(context);
	auto&        stats = GetStats();
	stats.segments[index].fetch_add(1, std::memory_order_relaxed);
	stats.segment_packets[index].fetch_add(packets, std::memory_order_relaxed);
	stats.segment_submits[index].fetch_add(submits, std::memory_order_relaxed);
	if (new_command_buffer) {
		stats.segment_buffers[index].fetch_add(1, std::memory_order_relaxed);
	}
	auto&           book = GetImageBook();
	std::lock_guard lock(book.mutex);
	auto&           segment = book.segments[index];
	if (!segment.open) {
		return; // the first hand-off of Thread_Gpu: no segment was begun
	}
	// The merge point: each image's entry requirement against the merged state, then the
	// segment's last states become the merged ones.
	for (const auto& [image, entry]: segment.entry) {
		if (const auto found = book.merged.find(image); found != book.merged.end()) {
			FixUpImage(static_cast<uint32_t>(index), image, found->second, entry);
		}
	}
	for (const auto& [image, last]: segment.last) {
		book.merged[image] = last;
	}
	segment.entry.clear();
	segment.last.clear();
	segment.open = false;
	t_segment    = -1;
}

void NoteLaneHandoff(uint32_t context, bool empty, bool skipped) {
	const size_t index = LaneIndex(context);
	auto&        stats = GetStats();
	if (skipped) {
		stats.lane_skipped[index].fetch_add(1, std::memory_order_relaxed);
		return;
	}
	stats.lane_submits[index].fetch_add(1, std::memory_order_relaxed);
	if (empty) {
		stats.lane_empty[index].fetch_add(1, std::memory_order_relaxed);
	}
}

// Astro Bot PC (KYTY_CP_PAR=2b, cpParallel.h).
void NoteLaneStat(uint32_t lane, LaneStat stat, uint64_t value) {
	const auto index = static_cast<size_t>(stat);
	if (index < static_cast<size_t>(LaneStat::Count)) {
		GetStats().lane_stats[LaneIndex(lane)][index].fetch_add(value, std::memory_order_relaxed);
	}
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
	// Astro Bot PC (KYTY_CP_PAR=2a, N lanes): each context's draw-prep counts, also for the lanes
	// line below (contexts C, D: only there).
	uint64_t values[MaxLanes][4] = {};
	if (SeparateWindows()) {
		// Astro Bot PC (KYTY_CP_PAR=1b1): each context's draw-prep window over the last 10 s.
		for (size_t c = 0; c < MaxLanes; c++) {
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
	if (Concurrent()) {
		// Astro Bot PC (KYTY_CP_PAR=1b3): the run-ahead over the last 10 s.
		uint64_t p[static_cast<size_t>(Promotion::Count)] = {};
		for (size_t i = 0; i < std::size(p); i++) {
			p[i] = take(stats.promotions[i]);
		}
		uint64_t ins[static_cast<size_t>(Insertion::Count)] = {};
		for (size_t i = 0; i < std::size(ins); i++) {
			ins[i] = take(stats.insertions[i]);
		}
		const auto starts     = take(stats.runahead_starts);
		const auto ops        = take(stats.runahead_ops);
		const auto merges     = take(stats.merges);
		const auto fixups     = take(stats.merge_fixups);
		const auto impossible = take(stats.merge_impossible);
		const auto images     = take(stats.merge_images);
		const auto ins_fixups = take(stats.insertion_fixups);
		const auto wait_a     = take(stats.lock_wait_ns[0]);
		const auto wait_b     = take(stats.lock_wait_ns[1]);
		const auto takes_a    = take(stats.lock_takes[0]);
		const auto takes_b    = take(stats.lock_takes[1]);
		const auto overlap    = take(stats.overlap_ns);
		const auto ops_ns     = take(stats.runahead_ns);
		std::printf("CpPar 10s: %s: run-ahead %" PRIu64 " starts, %" PRIu64 " ops (%.2f ms "
		            "holding the op lock); overlap %.2f ms; "
		            "promotions: stream-end %" PRIu64 ", lockstep %" PRIu64 ", write %" PRIu64
		            ", flip %" PRIu64 ", indirect %" PRIu64 ", inherit %" PRIu64
		            ", no-registers %" PRIu64 ", ring %" PRIu64 ", stop %" PRIu64 "; merges %" PRIu64
		            " (%" PRIu64 " overlay images, fix-ups %" PRIu64 " + %" PRIu64
		            " at insertions, impossible %" PRIu64 "); insertions: oldest submit %" PRIu64
		            ", run-ahead submit %" PRIu64 ", run-ahead drain %" PRIu64
		            "; op lock: Thread_Gpu %" PRIu64 " takes, %.2f ms waiting; resolver #2 %" PRIu64
		            " takes, %.2f ms waiting\n",
		            ConcurrentName(), starts, ops, static_cast<double>(ops_ns) / 1e6,
		            static_cast<double>(overlap) / 1e6,
		            p[0], p[1], p[2], p[3], p[4], p[5],
		            p[6], p[7], p[8], merges, images, fixups, ins_fixups, impossible, ins[0], ins[1],
		            ins[2], takes_a, static_cast<double>(wait_a) / 1e6, takes_b,
		            static_cast<double>(wait_b) / 1e6);
		if (Wide()) {
			// Astro Bot PC (KYTY_CP_PAR=1b4): the window (cpParallel.h).
			std::string kinds;
			for (size_t k = 0; k < stats.runahead_kinds.size(); k++) {
				if (const auto n = take(stats.runahead_kinds[k]); n != 0) {
					char item[64];
					std::snprintf(item, sizeof(item), " %s=%" PRIu64,
					              CpSeq::OpKindName(static_cast<CpSeq::OpKind>(k)), n);
					kinds += item;
				}
			}
			uint64_t c[static_cast<size_t>(Cross::Count)] = {};
			for (size_t i = 0; i < std::size(c); i++) {
				c[i] = take(stats.cross[i]);
			}
			const auto fixups_total = fixups + ins_fixups;
			std::printf("CpPar 10s: %s window: overlap %.2f ms; run-ahead ops by kind:%s; "
			            "promotions: interrupt %" PRIu64 ", label-defer %" PRIu64 ", clock %" PRIu64
			            ", gds %" PRIu64 ", occlusion %" PRIu64 ", external %" PRIu64
			            ", gpu-dirty %" PRIu64 ", foreign-wait %" PRIu64 " (and the 1b3 reasons "
			            "above); waits satisfied locally %" PRIu64 ", writes to checked addresses %" PRIu64
			            "; fix-ups emitted %" PRIu64 "; interleave points offered %" PRIu64
			            ", refused (hard op ahead, or N not fully emitted) %" PRIu64 "; cross accesses: oldest read %" PRIu64
			            ", oldest write %" PRIu64 ", run-ahead read %" PRIu64 ", foreign value %" PRIu64
			            "; book %" PRIu64 " dwords, %" PRIu64 " learned labels\n",
			            ConcurrentName(), static_cast<double>(overlap) / 1e6,
			            kinds.empty() ? " none" : kinds.c_str(),
			            p[static_cast<size_t>(Promotion::Interrupt)],
			            p[static_cast<size_t>(Promotion::LabelDefer)],
			            p[static_cast<size_t>(Promotion::Clock)], p[static_cast<size_t>(Promotion::Gds)],
			            p[static_cast<size_t>(Promotion::Occlusion)],
			            p[static_cast<size_t>(Promotion::External)],
			            p[static_cast<size_t>(Promotion::GpuDirty)],
			            p[static_cast<size_t>(Promotion::ForeignWait)], take(stats.local_waits),
			            take(stats.local_writes), fixups_total, take(stats.interleave_offered),
			            take(stats.interleave_refused), c[0], c[1], c[2], c[3],
			            stats.book_written.load(std::memory_order_relaxed),
			            stats.book_labels.load(std::memory_order_relaxed));
		}
		if (Deferred()) {
			// Astro Bot PC (KYTY_CP_PAR=1b5): deferred interrupts, stops by op kind, outcomes,
			// windows (cpParallel.h).
			static constexpr const char* reasons[] = {
			    "stream-end", "lockstep", "write", "flip", "indirect", "inherit", "no-registers",
			    "ring", "stop", "interrupt", "label-defer", "clock", "gds", "occlusion", "external",
			    "gpu-dirty", "foreign-wait"};
			static_assert(std::size(reasons) == static_cast<size_t>(Promotion::Count));
			std::string histogram;
			uint64_t    stop_none = 0;
			uint64_t    stop_ops  = 0;
			for (size_t r = 0; r < stats.stop_kinds.size(); r++) {
				for (size_t k = 0; k < stats.stop_kinds[r].size(); k++) {
					const auto n = take(stats.stop_kinds[r][k]);
					if (n == 0) {
						continue;
					}
					const bool none = k == static_cast<size_t>(CpSeq::OpKind::Count);
					if (r == static_cast<size_t>(Promotion::Stop)) {
						(none ? stop_none : stop_ops) += n;
					}
					char item[96];
					std::snprintf(item, sizeof(item), " %s/%s=%" PRIu64, reasons[r],
					              none ? "none" : CpSeq::OpKindName(static_cast<CpSeq::OpKind>(k)), n);
					histogram += item;
				}
			}
			const auto outcomes  = take(stats.outcomes);
			const auto started   = take(stats.outcomes_started);
			const auto a_ops     = take(stats.outcome_ahead_ops);
			const auto a_draws   = take(stats.outcome_ahead_draws);
			const auto r_ops     = take(stats.outcome_rest_ops);
			const auto r_draws   = take(stats.outcome_rest_draws);
			const auto max_ops   = take(stats.outcome_max_ops);
			uint64_t   done[6]   = {};
			for (size_t i = 0; i < std::size(done); i++) {
				done[i] = take(stats.outcome_done[i]);
			}
			const auto windows   = take(stats.windows);
			const auto window_ns = take(stats.window_ns);
			const auto starved   = take(stats.starved_ns);
			const auto per = [](uint64_t value, uint64_t count) {
				return count == 0 ? 0.0 : static_cast<double>(value) / static_cast<double>(count);
			};
			std::printf(
			    "CpPar 10s: 1b5: deferred interrupts delivered %" PRIu64 " (stores %" PRIu64
			    ", late %" PRIu64 "; interrupt ops with label store captured %" PRIu64
			    ", without a captured store (learned label, clock or no data) %" PRIu64
			    "); order violations %" PRIu64
			    ", stray interrupts %" PRIu64 ", lists left %" PRIu64 "; stops (reason/op kind):%s; "
			    "targets completed %" PRIu64 " (%" PRIu64 " started): ops ahead %" PRIu64
			    ", ops left to Thread_Gpu %" PRIu64 " (%.1f per target, largest target %" PRIu64
			    " ops), draws ahead %" PRIu64 " of %" PRIu64 " (%.1f per target); done ahead 0%% %" PRIu64
			    ", <25%% %" PRIu64 ", <50%% %" PRIu64 ", <75%% %" PRIu64 ", <100%% %" PRIu64
			    ", 100%% %" PRIu64 "; stop: %" PRIu64 " with no op of N+1 emitted, %" PRIu64
			    " with ops pending; windows %" PRIu64 " (offer to merge %.3f ms each, run-ahead "
			    "waiting for the sequencer %.3f ms each)\n",
			    take(stats.deferred_interrupts), take(stats.deferred_stores),
			    take(stats.deferred_late), take(stats.deferred_ops_captured),
			    take(stats.deferred_ops_recorded), take(stats.order_violations),
			    take(stats.stray_interrupts), take(stats.deferred_left),
			    histogram.empty() ? " none" : histogram.c_str(), outcomes, started, a_ops, r_ops,
			    per(r_ops, outcomes), max_ops, a_draws, a_draws + r_draws,
			    per(a_draws + r_draws, outcomes), done[0], done[1], done[2], done[3], done[4], done[5],
			    stop_none, stop_ops, windows, per(window_ns, windows) / 1e6,
			    per(starved, windows) / 1e6);
		}
	}
	if (StreamsActive() && !Concurrent()) {
		// Astro Bot PC (KYTY_CP_PAR=1b2): each context's recording target over the last 10 s.
		uint64_t s[2][6] = {};
		for (size_t c = 0; c < 2; c++) {
			s[c][0] = take(stats.segments[c]);
			s[c][1] = take(stats.segment_packets[c]);
			s[c][2] = take(stats.segment_submits[c]);
			s[c][3] = take(stats.segment_buffers[c]);
			s[c][4] = take(stats.fixups_checked[c]);
			s[c][5] = take(stats.fixups_needed[c]);
		}
		const auto outside = take(stats.image_outside);
		std::printf("CpPar 10s: streams: B %" PRIu64 " segments, %" PRIu64 " packets, %" PRIu64
		            " submits, %" PRIu64 " command buffers; A %" PRIu64 " segments, %" PRIu64
		            " packets, %" PRIu64 " submits, %" PRIu64 " command buffers; image fix-ups "
		            "checked A %" PRIu64 " B %" PRIu64 ", not no-ops A %" PRIu64 " B %" PRIu64
		            " (expected 0); image accesses outside segments %" PRIu64 "\n",
		            s[1][0], s[1][1], s[1][2], s[1][3], s[0][0], s[0][1], s[0][2], s[0][3], s[0][4],
		            s[1][4], s[0][5], s[1][5], outside);
	}
	if (Lanes()) {
		// Astro Bot PC (KYTY_CP_PAR=2a): the hand-off submits of each lane over the last 10 s.
		// Per lane: lists resolved to their end, translation (resolving) time, hand-off submits
		// (empty: only the Begin packet; skipped: nothing encoded, no submit), draw-prep commits
		// and fallbacks. Lanes A and B reuse the counts taken for the lines above.
		std::string lanes;
		for (uint32_t lane = 0; lane < g_lane_count && lane < MaxLanes; lane++) {
			const uint64_t subs = lane == 0 ? a : (lane == 1 ? b : take(stats.completed[lane]));
			const uint64_t ns   = lane == 0 ? a_ns : (lane == 1 ? b_ns : take(stats.resolve_ns[lane]));
			char item[256];
			std::snprintf(item, sizeof(item),
			              "%s%c: %" PRIu64 " lists, %.1f ms translating, submits %" PRIu64
			              " (empty %" PRIu64 ", skipped %" PRIu64 "), prep committed %" PRIu64
			              " fell back %" PRIu64,
			              lane == 0 ? "" : "; ", LaneLetter(lane), subs, static_cast<double>(ns) / 1e6,
			              take(stats.lane_submits[lane]), take(stats.lane_empty[lane]),
			              take(stats.lane_skipped[lane]), values[lane][2], values[lane][3]);
			lanes += item;
		}
		std::printf("CpPar 10s: 2a lanes (%u): %s\n", g_lane_count, lanes.c_str());
	}
	if (Lanes2b()) {
		// Astro Bot PC (KYTY_CP_PAR=2b): per lane, its replay thread (busy time, waits for its turn
		// to hand a buffer to the queue).
		std::string lanes;
		for (uint32_t lane = 0; lane < g_lane_count && lane < MaxLanes; lane++) {
			uint64_t v[static_cast<size_t>(LaneStat::Count)] = {};
			for (size_t i = 0; i < std::size(v); i++) {
				v[i] = take(stats.lane_stats[lane][i]);
			}
			const auto ms = [&v](LaneStat stat) {
				return static_cast<double>(v[static_cast<size_t>(stat)]) / 1e6;
			};
			const auto n = [&v](LaneStat stat) { return v[static_cast<size_t>(stat)]; };
			char item[512];
			std::snprintf(item, sizeof(item),
			              "%s%c: replay %.1f ms busy, turn waits %" PRIu64 " (%.1f ms)",
			              lane == 0 ? "" : "; ", LaneLetter(lane), ms(LaneStat::ReplayBusyNs),
			              n(LaneStat::ReplayTurnWaits), ms(LaneStat::ReplayTurnNs));
			lanes += item;
		}
		std::printf("CpPar 10s: 2b (%u): %s\n", g_lane_count, lanes.c_str());
	}
	std::fflush(stdout);
}

} // namespace Libs::Graphics::CpPar
