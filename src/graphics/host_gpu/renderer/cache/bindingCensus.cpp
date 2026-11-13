#include "graphics/host_gpu/renderer/cache/bindingCensus.h"

#include "graphics/guest_gpu/cbRanges.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/cache/bufferLookupStats.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__)
#include <x86intrin.h>
#endif

// Astro Bot PC: KYTY_BINDING_CENSUS (bindingCensus.h). Diagnostic only.
namespace Libs::Graphics::BindingCensus {

namespace Detail {
thread_local uint8_t t_outcome = 0;
thread_local uint8_t t_slot    = 0;

bool ReadEnabled() {
	const auto* value = std::getenv("KYTY_BINDING_CENSUS");
	return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}
} // namespace Detail

namespace {

constexpr uint64_t PageBits        = 14; // 16 KiB, BufferCache::CACHING_PAGEBITS
constexpr uint64_t PageSize        = uint64_t {1} << PageBits;
constexpr uint64_t MaxProbePages   = 64;    // pages probed per binding (1 MiB)
constexpr size_t   MaxPagesPerRange = 1u << 16u;
constexpr uint64_t UnknownKeyFlag  = uint64_t {1} << 63u;
constexpr size_t   TopRanges       = 16;
constexpr size_t   SlotCount       = 17; // 0: not native, 1..16: engine thread slot + 1

// Census classes of a binding.
enum Class : uint8_t {
	kMemo,       // memo hit (same-epoch or cross-epoch cache buffer)
	kMemoStream, // memo stream-copy hit (bytes copied again)
	kLean,       // memo miss, lean path (KYTY_BINDING_LEAN_MISS)
	kStream,     // memo miss, stream copy into the ring
	kUpload,     // memo miss, cache buffer, some page CPU-dirty before (synchronized/uploaded)
	kCache,      // memo miss, cache buffer, no page seen CPU-dirty
	kTexel,      // formatted read
	kWritten,    // writable binding
	kOther,      // unclassified (no memo, n3 window, other thread...)
	kClassCount
};
constexpr std::array<const char*, kClassCount> ClassNames = {
    "memo", "mstr", "lean", "strm", "upld", "cach", "texl", "wrtn", "othr"};

uint64_t ReadTsc() {
#if defined(_M_X64) || defined(__x86_64__)
	return __rdtsc();
#else
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
#endif
}

uint64_t NowNs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

struct PageStats {
	uint32_t last_bound_frame = UINT32_MAX;
	uint32_t last_dirty_frame = UINT32_MAX;
	uint32_t bound_frames     = 0;
	uint32_t dirty_frames     = 0;
	bool     gpu_dirty        = false; // seen GPU-dirty when bound
	bool     written          = false; // bound writable
	bool     image            = false; // an image may overlap (1 MiB pages)
	bool     unknown          = false; // tracker could not answer
};

struct RangeStats {
	uint64_t start   = 0;
	uint64_t size    = 0;
	char     name[Libs::LibKernel::Memory::KERNEL_MAXIMUM_NAME_LENGTH + 1] {};
	char     kind    = '?'; // D direct, F flexible, S stack, P pooled, R reserved/other, ? none
	int32_t  protection = 0;
	std::array<uint64_t, kClassCount> count {};
	std::array<uint64_t, kClassCount> cycles {};
	uint64_t bytes       = 0;
	uint64_t max_size    = 0;
	uint64_t cb          = 0; // bindings inside a game command-buffer zone (KYTY_CB_CONST_STATS)
	uint64_t big         = 0; // bindings larger than the probed pages
	uint64_t page_overflow = 0;
	std::unordered_map<uint64_t, PageStats> pages;

	[[nodiscard]] uint64_t TotalCycles() const {
		uint64_t t = 0;
		for (auto c: cycles) {
			t += c;
		}
		return t;
	}
	[[nodiscard]] uint64_t TotalCount() const {
		uint64_t t = 0;
		for (auto c: count) {
			t += c;
		}
		return t;
	}
};

struct CachedRange {
	uint64_t    start = 0;
	uint64_t    end   = 0;
	RangeStats* stats = nullptr;
};

struct State {
	std::unordered_map<uint64_t, RangeStats> ranges;
	std::vector<CachedRange>                 cache; // sorted by start, never overlapping
	size_t                                   last_hit   = 0;
	uint64_t                                 generation = UINT64_MAX;
	std::array<uint64_t, SlotCount>          slot_cycles {};
	std::array<uint64_t, SlotCount>          slot_count {};
	std::array<uint64_t, kClassCount>        class_count {};
	std::array<uint64_t, kClassCount>        class_cycles {};
	uint64_t                                 probe_cycles = 0; // the census's own Before cost
	uint64_t                                 queries      = 0; // virtual-range queries
	uint64_t                                 no_tracker   = 0; // page probes the tracker skipped
	uint32_t                                 frame_first  = 0;
	uint32_t                                 frame_last   = 0;
	bool                                     frame_set    = false;
	uint64_t                                 interval_ns  = 0;
	uint64_t                                 interval_tsc = 0;
};

std::atomic<uint64_t> g_off_thread {0};

State& GetState() {
	static State state; // GPU thread only
	return state;
}

char KindOf(const Libs::LibKernel::Memory::VirtualQueryInfo& info) {
	if (info.is_direct != 0) {
		return 'D';
	}
	if (info.is_flexible != 0) {
		return 'F';
	}
	if (info.is_stack != 0) {
		return 'S';
	}
	if (info.is_pooled != 0) {
		return 'P';
	}
	return info.is_committed != 0 ? 'C' : 'R';
}

RangeStats* Lookup(State& st, uint64_t vaddr) {
	const auto generation = Libs::LibKernel::Memory::VirtualRangesGeneration();
	if (generation != st.generation) {
		st.cache.clear();
		st.last_hit   = 0;
		st.generation = generation;
	}
	if (st.last_hit < st.cache.size()) {
		const auto& c = st.cache[st.last_hit];
		if (vaddr >= c.start && vaddr < c.end) {
			return c.stats;
		}
	}
	auto it = std::upper_bound(st.cache.begin(), st.cache.end(), vaddr,
	                           [](uint64_t v, const CachedRange& c) { return v < c.start; });
	if (it != st.cache.begin()) {
		const auto prev = std::prev(it);
		if (vaddr >= prev->start && vaddr < prev->end) {
			st.last_hit = static_cast<size_t>(prev - st.cache.begin());
			return prev->stats;
		}
	}
	st.queries++;
	Libs::LibKernel::Memory::VirtualQueryInfo info {};
	CachedRange                               entry {};
	RangeStats*                               stats = nullptr;
	if (Libs::LibKernel::Memory::DiagQueryVirtualRange(vaddr, &info) && info.end > info.start &&
	    vaddr >= info.start && vaddr < info.end) {
		entry.start = info.start;
		entry.end   = info.end;
		auto [r, inserted] = st.ranges.try_emplace(info.start);
		stats              = &r->second;
		if (inserted || stats->size != info.end - info.start) {
			stats->start      = info.start;
			stats->size       = info.end - info.start;
			stats->kind       = KindOf(info);
			stats->protection = info.protection;
			std::memcpy(stats->name, info.name, sizeof(info.name));
			stats->name[sizeof(stats->name) - 1] = '\0';
		}
	} else {
		// No guest range: one bucket per 16 MiB, cached per 16 KiB page.
		entry.start        = vaddr & ~(PageSize - 1);
		entry.end          = entry.start + PageSize;
		const auto key     = UnknownKeyFlag | (vaddr >> 24u);
		auto [r, inserted] = st.ranges.try_emplace(key);
		stats              = &r->second;
		if (inserted) {
			stats->start = (vaddr >> 24u) << 24u;
			stats->size  = uint64_t {1} << 24u;
			stats->kind  = '?';
			std::snprintf(stats->name, sizeof(stats->name), "<no range>");
		}
	}
	entry.stats = stats;
	// A cached entry never overlaps another (ranges never overlap; a negative page is dropped
	// when it would).
	it                = std::upper_bound(st.cache.begin(), st.cache.end(), entry.start,
	                                     [](uint64_t v, const CachedRange& c) { return v < c.start; });
	const bool overlap = (it != st.cache.end() && it->start < entry.end) ||
	                     (it != st.cache.begin() && std::prev(it)->end > entry.start);
	if (!overlap && st.cache.size() < 65536) {
		it          = st.cache.insert(it, entry);
		st.last_hit = static_cast<size_t>(it - st.cache.begin());
	}
	return stats;
}

void Print(State& st) {
	const auto now_ns  = NowNs();
	const auto now_tsc = ReadTsc();
	if (st.interval_ns == 0) {
		st.interval_ns  = now_ns;
		st.interval_tsc = now_tsc;
		return;
	}
	if (now_ns - st.interval_ns < 10'000'000'000ull) {
		return;
	}
	const double ns_per_tick = static_cast<double>(now_ns - st.interval_ns) /
	                           static_cast<double>(now_tsc - st.interval_tsc + 1);
	const double seconds     = static_cast<double>(now_ns - st.interval_ns) * 1e-9;
	const auto   frames      = st.frame_set ? std::max<uint32_t>(st.frame_last - st.frame_first, 1u) : 1u;
	auto         ms          = [&](uint64_t cycles) { return static_cast<double>(cycles) * ns_per_tick * 1e-6; };

	uint64_t total_cycles = 0;
	uint64_t total_count  = 0;
	for (size_t k = 0; k < kClassCount; k++) {
		total_cycles += st.class_cycles[k];
		total_count += st.class_count[k];
	}
	std::vector<const RangeStats*> order;
	order.reserve(st.ranges.size());
	for (const auto& [key, r]: st.ranges) {
		if (r.TotalCount() != 0) {
			order.push_back(&r);
		}
	}
	std::sort(order.begin(), order.end(), [](const RangeStats* a, const RangeStats* b) {
		return a->TotalCycles() > b->TotalCycles();
	});
	auto top_share = [&](size_t n) {
		uint64_t c = 0;
		for (size_t i = 0; i < std::min(n, order.size()); i++) {
			c += order[i]->TotalCycles();
		}
		return total_cycles != 0 ? 100.0 * static_cast<double>(c) / static_cast<double>(total_cycles) : 0.0;
	};

	std::string line;
	char        text[512];
	std::snprintf(text, sizeof(text),
	              "BindingCensus 10s: %.1f s, %u frames, %" PRIu64 " bindings (%.0f/frame), %.2f ms/frame "
	              "binding time (census probe %.2f ms/frame), %zu ranges, top1 %.1f%% top2 %.1f%% top4 %.1f%%"
	              " top16 %.1f%% |",
	              seconds, frames, total_count, static_cast<double>(total_count) / frames,
	              ms(total_cycles) / frames, ms(st.probe_cycles) / frames, order.size(), top_share(1),
	              top_share(2), top_share(4), top_share(16));
	line += text;
	for (size_t k = 0; k < kClassCount; k++) {
		if (st.class_count[k] == 0) {
			continue;
		}
		std::snprintf(text, sizeof(text), " %s %.0f/f %.0fns", ClassNames[k],
		              static_cast<double>(st.class_count[k]) / frames,
		              static_cast<double>(st.class_cycles[k]) * ns_per_tick /
		                  static_cast<double>(st.class_count[k]));
		line += text;
	}
	std::snprintf(text, sizeof(text), " | vq %" PRIu64 ", no-tracker %" PRIu64 ", off-thread %" PRIu64,
	              st.queries, st.no_tracker, g_off_thread.exchange(0, std::memory_order_relaxed));
	line += text;
	bool slots = false;
	for (size_t s = 1; s < SlotCount; s++) {
		slots = slots || st.slot_count[s] != 0;
	}
	if (slots) {
		line += " | slots:";
		for (size_t s = 0; s < SlotCount; s++) {
			if (st.slot_count[s] == 0) {
				continue;
			}
			std::snprintf(text, sizeof(text), " %s%zu %.1f%% (%.0f/f)", s == 0 ? "non-native" : "t",
			              s == 0 ? size_t {0} : s - 1,
			              total_cycles != 0 ? 100.0 * static_cast<double>(st.slot_cycles[s]) /
			                                      static_cast<double>(total_cycles)
			                                : 0.0,
			              static_cast<double>(st.slot_count[s]) / frames);
			line += text;
		}
	}
	std::printf("%s\n", line.c_str());

	for (size_t i = 0; i < std::min(TopRanges, order.size()); i++) {
		const auto& r      = *order[i];
		const auto  cycles = r.TotalCycles();
		const auto  count  = r.TotalCount();
		line.clear();
		std::snprintf(text, sizeof(text),
		              "BindingCensus 10s: #%-2zu %c 0x%010" PRIx64 "+0x%" PRIx64 " '%s' prot=0x%x: %.1f%% "
		              "time, %.2f ms/f, %.0f bind/f, %.0fns/bind, avg %.0f B max 0x%" PRIx64 " |",
		              i + 1, r.kind, r.start, r.size, r.name, r.protection,
		              total_cycles != 0 ? 100.0 * static_cast<double>(cycles) / static_cast<double>(total_cycles) : 0.0,
		              ms(cycles) / frames, static_cast<double>(count) / frames,
		              static_cast<double>(cycles) * ns_per_tick / static_cast<double>(count),
		              static_cast<double>(r.bytes) / static_cast<double>(count), r.max_size);
		line += text;
		for (size_t k = 0; k < kClassCount; k++) {
			if (r.count[k] == 0) {
				continue;
			}
			std::snprintf(text, sizeof(text), " %s %.0f%%", ClassNames[k],
			              100.0 * static_cast<double>(r.count[k]) / static_cast<double>(count));
			line += text;
		}
		// Pages: distinct bound, CPU-dirty ever, mean dirty frames / bound frames of those, GPU,
		// written, image.
		uint64_t pages = 0, dirty_pages = 0, dirty_frames = 0, bound_frames = 0, gpu = 0, written = 0,
		         image = 0, every_frame_dirty = 0, unknown = 0;
		for (const auto& [page, p]: r.pages) {
			pages++;
			bound_frames += p.bound_frames;
			if (p.dirty_frames != 0) {
				dirty_pages++;
				dirty_frames += p.dirty_frames;
				every_frame_dirty += p.dirty_frames * 10u >= p.bound_frames * 9u ? 1u : 0u;
			}
			gpu += p.gpu_dirty ? 1u : 0u;
			written += p.written ? 1u : 0u;
			image += p.image ? 1u : 0u;
			unknown += p.unknown ? 1u : 0u;
		}
		std::snprintf(text, sizeof(text),
		              " | pages %" PRIu64 "%s (bound %.1f f/page), cpu-dirty %" PRIu64
		              " (%.0f%% of their bound frames, %" PRIu64 " dirty >=90%%), gpu %" PRIu64
		              ", wrtn %" PRIu64 ", img? %" PRIu64 "%s",
		              pages, r.page_overflow != 0 ? "+" : "",
		              pages != 0 ? static_cast<double>(bound_frames) / static_cast<double>(pages) : 0.0,
		              dirty_pages,
		              bound_frames != 0 && dirty_pages != 0
		                  ? 100.0 * static_cast<double>(dirty_frames) /
		                        static_cast<double>(std::max<uint64_t>(bound_frames, 1))
		                  : 0.0,
		              every_frame_dirty, gpu, written, image, unknown != 0 ? " (tracker n/a on some)" : "");
		line += text;
		if (r.big != 0 || r.cb != 0) {
			std::snprintf(text, sizeof(text), " | >1MiB %" PRIu64 ", in game CB %.0f%%", r.big,
			              100.0 * static_cast<double>(r.cb) / static_cast<double>(count));
			line += text;
		}
		std::printf("%s\n", line.c_str());
	}
	std::fflush(stdout);

	st.ranges.clear();
	st.cache.clear();
	st.last_hit = 0;
	st.slot_cycles.fill(0);
	st.slot_count.fill(0);
	st.class_count.fill(0);
	st.class_cycles.fill(0);
	st.probe_cycles = 0;
	st.queries      = 0;
	st.no_tracker   = 0;
	st.frame_set    = false;
	st.interval_ns  = now_ns;
	st.interval_tsc = now_tsc;
}

} // namespace

void Before(RenderContext& context, uint64_t vaddr, uint64_t size, Probe& probe) {
	const auto start = ReadTsc();
	probe.vaddr      = vaddr;
	probe.size       = size;
	probe.active     = false;
	probe.cpu_dirty  = false;
	Detail::t_outcome = 0;
	if (vaddr == 0 || size == 0 || UINT64_MAX - vaddr < size) {
		probe.t0 = ReadTsc();
		return;
	}
	if (!GuestGpu::IsGpuThread()) {
		g_off_thread.fetch_add(1, std::memory_order_relaxed);
		probe.t0 = ReadTsc();
		return;
	}
	auto&       st     = GetState();
	auto&       cache  = context.GetBufferCache();
	auto&       images = context.GetTextureCache();
	const auto  frame  = cache.CensusFrame();
	if (!st.frame_set) {
		st.frame_first = frame;
		st.frame_set   = true;
	}
	st.frame_last      = frame;
	RangeStats* range  = Lookup(st, vaddr);
	const auto  first  = vaddr >> PageBits;
	const auto  last   = (vaddr + size - 1) >> PageBits;
	const auto  probed = std::min<uint64_t>(last - first + 1, MaxProbePages);
	for (uint64_t i = 0; i < probed; i++) {
		const auto page = first + i;
		PageStats* p    = nullptr;
		if (auto it = range->pages.find(page); it != range->pages.end()) {
			p = &it->second;
		} else if (range->pages.size() < MaxPagesPerRange) {
			p = &range->pages[page];
		} else {
			range->page_overflow++;
			continue;
		}
		const bool new_frame = p->last_bound_frame != frame;
		if (new_frame) {
			p->last_bound_frame = frame;
			p->bound_frames++;
			p->image = p->image || images.CensusMayOverlapImages(page << PageBits, PageSize);
		}
		bool cpu = false;
		bool gpu = false;
		const auto lo = std::max(vaddr, page << PageBits);
		const auto hi = std::min(vaddr + size, (page + 1) << PageBits);
		if (!cache.CensusDirtyState(lo, hi - lo, cpu, gpu)) {
			p->unknown = true;
			st.no_tracker++;
			continue;
		}
		if (cpu) {
			probe.cpu_dirty = true;
			if (p->last_dirty_frame != frame) {
				p->last_dirty_frame = frame;
				p->dirty_frames++;
			}
		}
		p->gpu_dirty = p->gpu_dirty || gpu;
	}
	if (last - first + 1 > MaxProbePages) {
		range->big++;
	}
	probe.active = true;
	const auto t0 = ReadTsc();
	st.probe_cycles += t0 - start;
	probe.t0 = t0;
}

void After(RenderContext& context, const Probe& probe, bool written, bool formatted) {
	const auto cycles = ReadTsc() - probe.t0;
	if (!probe.active) {
		return;
	}
	auto&       st    = GetState();
	const auto  start = ReadTsc();
	RangeStats* range = Lookup(st, probe.vaddr);
	using O           = BufferLookupStats::Outcome;
	const auto outcome = static_cast<O>(Detail::t_outcome);
	Class      cls     = kOther;
	if (written) {
		cls = kWritten;
	} else if (formatted) {
		cls = kTexel;
	} else {
		switch (outcome) {
			case O::MemoHit:
			case O::MemoCross: cls = kMemo; break;
			case O::MemoStream: cls = kMemoStream; break;
			case O::MissLean: cls = kLean; break;
			case O::MissStream: cls = kStream; break;
			case O::MissCached: cls = probe.cpu_dirty ? kUpload : kCache; break;
			default: cls = kOther; break;
		}
	}
	range->count[cls]++;
	range->cycles[cls] += cycles;
	range->bytes += probe.size;
	range->max_size = std::max(range->max_size, probe.size);
	if (CbRanges::Enabled() && CbRanges::Contains(probe.vaddr)) {
		range->cb++;
	}
	if (written) {
		const auto first  = probe.vaddr >> PageBits;
		const auto last   = (probe.vaddr + probe.size - 1) >> PageBits;
		const auto probed = std::min<uint64_t>(last - first + 1, MaxProbePages);
		for (uint64_t i = 0; i < probed; i++) {
			if (auto it = range->pages.find(first + i); it != range->pages.end()) {
				it->second.written = true;
			}
		}
	}
	st.class_count[cls]++;
	st.class_cycles[cls] += cycles;
	const auto slot = std::min<size_t>(Detail::t_slot, SlotCount - 1);
	st.slot_count[slot]++;
	st.slot_cycles[slot] += cycles;
	st.probe_cycles += ReadTsc() - start;
	(void)context;
	Print(st);
}

} // namespace Libs::Graphics::BindingCensus
