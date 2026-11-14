#include "graphics/host_gpu/renderer/listCensus.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

// Astro Bot PC: KYTY_CP_PAR_CENSUS2 (listCensus.h).
namespace Libs::Graphics::ListCensus {

namespace {

struct Range {
	uint64_t start   = 0;
	uint64_t end     = 0; // exclusive
	uint32_t list    = 0; // ordinal of the list in the frame
	bool     written = false;
};

struct State {
	std::mutex             mutex;
	std::vector<Range>     ranges;
	uint32_t               list_ordinal = 0;
	uint64_t               last_submission = UINT64_MAX;
	uint64_t               frame = 0;
	// 10 s accumulators.
	uint64_t frames_analysed = 0, lists = 0, bindings = 0, written = 0;
	uint64_t cross_written   = 0; // written ranges of a list bound again by a later list
	uint64_t cross_read      = 0; // ... of which the later binding only reads (an upload candidate)
	uint64_t cross_bytes     = 0;
	std::unordered_map<uint32_t, uint64_t> cross_by_pair; // (earlier list << 8 | later list) -> count
	std::chrono::steady_clock::time_point last_print = std::chrono::steady_clock::now();
};

State& S() {
	static State state;
	return state;
}

void Analyse(State& s) {
	auto& r = s.ranges;
	std::sort(r.begin(), r.end(), [](const Range& a, const Range& b) { return a.start < b.start; });
	s.frames_analysed++;
	s.lists += s.list_ordinal;
	s.bindings += r.size();
	// For each written range, look for overlapping ranges of a later list (sweep over the sorted
	// ranges: candidates start before the written range ends).
	for (size_t i = 0; i < r.size(); i++) {
		if (!r[i].written) continue;
		s.written++;
		bool counted = false;
		// Ranges starting before r[i] may still overlap it: scan backwards a little, then forwards.
		const auto overlaps = [&](const Range& o) { return o.start < r[i].end && r[i].start < o.end; };
		const auto consider = [&](const Range& o) {
			if (counted || o.list <= r[i].list || !overlaps(o)) return;
			counted = true;
			s.cross_written++;
			if (!o.written) s.cross_read++;
			s.cross_bytes += std::min(o.end, r[i].end) - std::max(o.start, r[i].start);
			s.cross_by_pair[(std::min<uint32_t>(r[i].list, 255u) << 8u) | std::min<uint32_t>(o.list, 255u)]++;
		};
		for (size_t j = i; j-- > 0 && i - j < 64;) consider(r[j]);
		for (size_t j = i + 1; j < r.size() && r[j].start < r[i].end; j++) consider(r[j]);
	}
}

void MaybePrint(State& s) {
	const auto now = std::chrono::steady_clock::now();
	if (now - s.last_print < std::chrono::seconds(10)) return;
	s.last_print = now;
	const double f = s.frames_analysed != 0 ? static_cast<double>(s.frames_analysed) : 1.0;
	std::printf("ListCensus 10s: %" PRIu64 " frames analysed, %.1f lists/frame, %.0f buffer bindings/frame, %.0f written "
	            "ranges/frame; written by a list and bound again by a LATER list: %.1f/frame (%.1f read-only, %.1f KiB)",
	            s.frames_analysed, static_cast<double>(s.lists) / f, static_cast<double>(s.bindings) / f,
	            static_cast<double>(s.written) / f, static_cast<double>(s.cross_written) / f,
	            static_cast<double>(s.cross_read) / f, static_cast<double>(s.cross_bytes) / f / 1024.0);
	std::vector<std::pair<uint64_t, uint32_t>> pairs;
	for (const auto& [key, count]: s.cross_by_pair) pairs.emplace_back(count, key);
	std::sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
	std::printf(" | by list pair (earlier>later):");
	for (size_t i = 0; i < pairs.size() && i < 8; i++) {
		std::printf(" %u>%u=%.1f", pairs[i].second >> 8u, pairs[i].second & 0xffu, static_cast<double>(pairs[i].first) / f);
	}
	std::printf("\n");
	std::fflush(stdout);
	s.frames_analysed = s.lists = s.bindings = s.written = 0;
	s.cross_written = s.cross_read = s.cross_bytes = 0;
	s.cross_by_pair.clear();
}

} // namespace

bool Enabled() {
	static const bool enabled = [] {
		const char* v = std::getenv("KYTY_CP_PAR_CENSUS2");
		return v != nullptr && v[0] != 0 && std::strcmp(v, "0") != 0;
	}();
	return enabled;
}

void BeginList(uint64_t submission) {
	if (!Enabled()) return;
	auto&           s = S();
	std::lock_guard lock(s.mutex);
	if (submission != s.last_submission) {
		s.last_submission = submission;
		s.list_ordinal++;
	}
}

void NoteBuffer(uint64_t address, uint64_t size, bool written) {
	if (!Enabled() || size == 0) return;
	auto&           s = S();
	std::lock_guard lock(s.mutex);
	if ((s.frame & 3u) != 0) return; // one frame in 4
	s.ranges.push_back({address, address + size, s.list_ordinal, written});
}

void EndFrame() {
	if (!Enabled()) return;
	auto&           s = S();
	std::lock_guard lock(s.mutex);
	if ((s.frame & 3u) == 0 && !s.ranges.empty()) Analyse(s);
	s.ranges.clear();
	s.list_ordinal    = 0;
	s.last_submission = UINT64_MAX;
	s.frame++;
	MaybePrint(s);
}

} // namespace Libs::Graphics::ListCensus
