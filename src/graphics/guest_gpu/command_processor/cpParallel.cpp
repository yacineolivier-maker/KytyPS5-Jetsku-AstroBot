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
#include <unordered_map>

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
	// KYTY_CP_PAR=1b2 (EndSegment, FixUpImage): per context, segments, packets encoded in its
	// ring, Submit packets among them, native command buffers it recorded into (first segment in
	// each); image fix-ups checked at merge points and those that would not be no-ops; image
	// accesses outside any segment.
	std::array<std::atomic<uint64_t>, 2> segments {};
	std::array<std::atomic<uint64_t>, 2> segment_packets {};
	std::array<std::atomic<uint64_t>, 2> segment_submits {};
	std::array<std::atomic<uint64_t>, 2> segment_buffers {};
	std::array<std::atomic<uint64_t>, 2> fixups_checked {};
	std::array<std::atomic<uint64_t>, 2> fixups_needed {};
	std::atomic<uint64_t>                image_outside {0};
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
	std::array<std::atomic<uint64_t>, 2> lock_wait_ns {};
	std::array<std::atomic<uint64_t>, 2> lock_takes {};
	std::atomic<uint64_t> overlap_ns {0};
};

Stats& GetStats() {
	static Stats stats;
	return stats;
}

thread_local bool t_resolver2 = false;
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
	std::array<Segment, 2>                     segments;
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
		            image, context != 0 ? 'B' : 'A', entry, global);
	}
}

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
		t_resolver2 = true;
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
		std::printf("Kyty CpPar: KYTY_CP_PAR=%s is not supported (only 1a, 1b1, 1b2, 1b3); off\n",
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
		return on;
	}();
	return enabled;
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
	EXIT_IF(g_resolver2 != nullptr || g_runahead != nullptr);
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
		            "inverted; counted in \"CpPar 10s: 1b3\")\n",
		            names[index]);
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

void Start(const Hooks& hooks) {
	EXIT_IF(g_resolver2 != nullptr || hooks.take == nullptr || hooks.give == nullptr);
	g_resolver2 = std::make_unique<Resolver2>(hooks);
	std::printf("Kyty CpPar: CP resolver #2 started\n");
}

void Stop() {
	g_resolver2.reset();
	g_runahead.reset(); // Astro Bot PC (KYTY_CP_PAR=1b3)
}

bool Started() {
	return g_resolver2 != nullptr || g_runahead != nullptr;
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
	const size_t    index = context != 0 ? 1 : 0;
	auto&           book  = GetImageBook();
	std::lock_guard lock(book.mutex);
	EXIT_IF(book.segments[index].open);
	book.segments[index].open = true;
	t_segment                 = static_cast<int>(index);
}

void EndSegment(uint32_t context, uint64_t packets, uint64_t submits, bool new_command_buffer) {
	const size_t index = context != 0 ? 1 : 0;
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
		std::printf("CpPar 10s: 1b3: run-ahead %" PRIu64 " starts, %" PRIu64 " ops (%.2f ms "
		            "holding the op lock); overlap %.2f ms; "
		            "promotions: stream-end %" PRIu64 ", lockstep %" PRIu64 ", write %" PRIu64
		            ", flip %" PRIu64 ", indirect %" PRIu64 ", inherit %" PRIu64
		            ", no-registers %" PRIu64 ", ring %" PRIu64 ", stop %" PRIu64 "; merges %" PRIu64
		            " (%" PRIu64 " overlay images, fix-ups %" PRIu64 " + %" PRIu64
		            " at insertions, impossible %" PRIu64 "); insertions: oldest submit %" PRIu64
		            ", run-ahead submit %" PRIu64 ", run-ahead drain %" PRIu64
		            "; op lock: Thread_Gpu %" PRIu64 " takes, %.2f ms waiting; resolver #2 %" PRIu64
		            " takes, %.2f ms waiting\n",
		            starts, ops, static_cast<double>(ops_ns) / 1e6, static_cast<double>(overlap) / 1e6,
		            p[0], p[1], p[2], p[3], p[4], p[5],
		            p[6], p[7], p[8], merges, images, fixups, ins_fixups, impossible, ins[0], ins[1],
		            ins[2], takes_a, static_cast<double>(wait_a) / 1e6, takes_b,
		            static_cast<double>(wait_b) / 1e6);
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
	std::fflush(stdout);
}

} // namespace Libs::Graphics::CpPar
