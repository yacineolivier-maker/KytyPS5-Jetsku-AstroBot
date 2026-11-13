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
};

Stats& GetStats() {
	static Stats stats;
	return stats;
}

thread_local bool t_resolver2 = false;

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
		std::printf("Kyty CpPar: KYTY_CP_PAR=%s is not supported (only 1a, 1b1, 1b2); off\n", value);
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
	if (StreamsActive()) {
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
