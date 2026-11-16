#include "kernel/waitGraph.h"

#include "common/common.h"
#include "kernel/pthread.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace Libs::LibKernel::WaitGraph {

bool g_enabled = [] {
	const auto* value = std::getenv("KYTY_WAIT_GRAPH");
	const bool  on    = value != nullptr && std::strcmp(value, "1") == 0;
	if (on) {
		std::printf("Kyty graphe d'attente : actif, une ligne toutes les 10 s (KYTY_WAIT_GRAPH)\n");
		std::fflush(stdout);
	}
	return on;
}();

namespace {

constexpr uint64_t ReportPeriodNs = 10'000'000'000ull;
constexpr uint64_t YieldBurstGapNs = 100'000;
constexpr size_t   SignalSlots     = 1u << 16u;

uint64_t NowNs() noexcept {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

const char* PrimName(Prim prim) {
	switch (prim) {
		case Prim::KernelSema: return "semaphore noyau";
		case Prim::PosixSem: return "sem Posix";
		case Prim::Cond: return "cond";
		case Prim::Mutex: return "mutex";
		case Prim::EventFlag: return "event flag";
		case Prim::Equeue: return "equeue";
		case Prim::AgcDone: return "AgcSuspendPoint";
		case Prim::Sleep: return "sommeil";
		case Prim::Yield: return "boucle de yield";
		case Prim::Count: break;
	}
	return "?";
}

// Aggregated waits of one thread, by (waker, primitive, guest address of the wait).
struct Key {
	uint32_t waker = 0;
	Prim     prim  = Prim::Count;
	uint64_t ra    = 0;
	bool     operator==(const Key& other) const noexcept {
		return waker == other.waker && prim == other.prim && ra == other.ra;
	}
};
struct KeyHash {
	size_t operator()(const Key& key) const noexcept {
		return std::hash<uint64_t>()(key.ra * 31u + (uint64_t {key.waker} << 8u) +
		                             static_cast<uint64_t>(key.prim));
	}
};
struct Agg {
	uint64_t ns    = 0;
	uint64_t count = 0;
};

// A thread, or a pseudo-thread (an emulator-side waker with a readable name).
struct ThreadRec {
	uint32_t    id     = 0;
	bool        pseudo = false;
	bool        guest  = false;
	bool        main   = false;
	std::string name;
	std::mutex  mutex; // waits (the owner adds, the reporter takes)
	std::unordered_map<Key, Agg, KeyHash> waits;
	// Yield burst (owner only).
	uint64_t burst_start = 0;
	uint64_t burst_last  = 0;
	uint64_t burst_ra    = 0;
	uint32_t burst_count = 0;
};

std::mutex               g_registry_mutex;
std::vector<ThreadRec*>  g_threads; // index = id; leaked (threads may outlive the reporter)
std::unordered_map<std::string, uint32_t> g_pseudo;
std::atomic<uint64_t>    g_frames {0};
std::atomic<bool>        g_reporter_started {false};

thread_local ThreadRec* t_self = nullptr;
thread_local bool       t_main = false; // the guest main thread (MarkMainThread)

uint32_t Register(ThreadRec* rec) {
	std::lock_guard lock(g_registry_mutex);
	rec->id = static_cast<uint32_t>(g_threads.size());
	g_threads.push_back(rec);
	return rec->id;
}

void StartReporter();

ThreadRec* Self() {
	if (t_self != nullptr) {
		return t_self;
	}
	auto* rec = new ThreadRec;
	char  name[64] {};
	auto  self = PthreadSelfOrNull();
	if (self != nullptr && PthreadGetname(self, name) == 0 && name[0] != 0) {
		rec->name  = name;
		rec->guest = true;
	} else if (self != nullptr) {
		rec->name  = "invite";
		rec->guest = true;
	} else {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		PWSTR description = nullptr;
		if (SUCCEEDED(GetThreadDescription(GetCurrentThread(), &description)) && description != nullptr) {
			for (const wchar_t* c = description; *c != 0 && rec->name.size() < 40; c++) {
				rec->name.push_back(*c < 128 ? static_cast<char>(*c) : '?');
			}
			LocalFree(description);
		}
#endif
		if (rec->name.empty()) {
			rec->name = "hote";
		}
		rec->name = "emu:" + rec->name;
	}
	rec->main = t_main;
	Register(rec);
	if (rec->guest) {
		rec->name += "#" + std::to_string(rec->id);
	}
	t_self = rec;
	StartReporter();
	return rec;
}

uint32_t PseudoId(const char* label) {
	std::lock_guard lock(g_registry_mutex);
	auto it = g_pseudo.find(label);
	if (it != g_pseudo.end()) {
		return it->second;
	}
	auto* rec   = new ThreadRec;
	rec->pseudo = true;
	rec->name   = label;
	rec->id     = static_cast<uint32_t>(g_threads.size());
	g_threads.push_back(rec);
	g_pseudo.emplace(label, rec->id);
	return rec->id;
}

// Last signal of an object (lock-free; a torn read only misnames one wait).
struct Slot {
	std::atomic<uint64_t> key {0};
	std::atomic<uint64_t> ns {0};
	std::atomic<uint64_t> ra {0};
	std::atomic<uint32_t> waker {0};
};
std::array<Slot, SignalSlots> g_slots;

size_t SlotIndex(const void* object) noexcept {
	auto value = reinterpret_cast<uint64_t>(object);
	value ^= value >> 29u;
	value *= 0xbf58476d1ce4e5b9ull;
	value ^= value >> 32u;
	return static_cast<size_t>(value & (SignalSlots - 1u));
}

uint32_t UnknownId() {
	static const uint32_t id = PseudoId("inconnu (delai, annulation ou signal non suivi)");
	return id;
}

void AddWait(ThreadRec* rec, uint32_t waker, Prim prim, uint64_t ra, uint64_t ns) {
	std::lock_guard lock(rec->mutex);
	auto& agg = rec->waits[Key {waker, prim, ra}];
	agg.ns += ns;
	agg.count++;
}

void FlushBurst(ThreadRec* rec) {
	if (rec->burst_count >= 2 && rec->burst_last > rec->burst_start) {
		static const uint32_t loop_id = PseudoId("boucle de yield");
		AddWait(rec, loop_id, Prim::Yield, rec->burst_ra, rec->burst_last - rec->burst_start);
	}
	rec->burst_start = 0;
	rec->burst_last  = 0;
	rec->burst_count = 0;
}

std::string Address(uint64_t ra) {
	char text[48];
	if (ra == 0) {
		return "";
	}
	if (ra == 0x90114b305ull || ra == 0x90114b315ull) {
		return " @eboot 0x114b2f0 (attente d'etiquette GPU)";
	}
	if ((ra >> 32u) == 0x9u && ra < 0x910000000ull) {
		std::snprintf(text, sizeof(text), " @eboot 0x%" PRIx64, ra - 0x900000000ull);
	} else if ((ra >> 32u) == 0x9u) {
		std::snprintf(text, sizeof(text), " @0x%" PRIx64, ra);
	} else {
		std::snprintf(text, sizeof(text), " @emu");
	}
	return text;
}

struct ThreadSummary {
	ThreadRec*                       rec   = nullptr;
	uint64_t                         total = 0;
	std::vector<std::pair<Key, Agg>> waits; // sorted, largest first
	uint32_t top_thread_waker = UINT32_MAX;  // the guest thread it waited for most
	uint64_t top_thread_ns    = 0;
	Prim     top_thread_prim  = Prim::Count;
};

void Report(uint64_t window_ns) {
	const auto frames  = g_frames.exchange(0, std::memory_order_relaxed);
	const double fdiv  = frames != 0 ? static_cast<double>(frames) : 1.0;
	std::vector<ThreadRec*> threads;
	{
		std::lock_guard lock(g_registry_mutex);
		threads = g_threads;
	}
	std::vector<ThreadSummary> summaries(threads.size());
	for (size_t i = 0; i < threads.size(); i++) {
		auto* rec      = threads[i];
		auto& summary  = summaries[i];
		summary.rec    = rec;
		if (rec->pseudo) {
			continue;
		}
		std::unordered_map<Key, Agg, KeyHash> waits;
		{
			std::lock_guard lock(rec->mutex);
			waits.swap(rec->waits);
		}
		std::unordered_map<uint32_t, std::pair<uint64_t, Prim>> by_waker;
		for (const auto& [key, agg]: waits) {
			summary.total += agg.ns;
			summary.waits.emplace_back(key, agg);
			if (key.waker < threads.size() && threads[key.waker]->guest && key.waker != rec->id) {
				auto& entry = by_waker[key.waker];
				entry.first += agg.ns;
				entry.second = key.prim;
			}
		}
		std::sort(summary.waits.begin(), summary.waits.end(),
		          [](const auto& a, const auto& b) { return a.second.ns > b.second.ns; });
		for (const auto& [waker, entry]: by_waker) {
			if (entry.first > summary.top_thread_ns) {
				summary.top_thread_ns    = entry.first;
				summary.top_thread_waker = waker;
				summary.top_thread_prim  = entry.second;
			}
		}
	}
	const double seconds = static_cast<double>(window_ns) / 1e9;
	std::printf("Kyty graphe d'attente 10 s : %.1f images/s (AgcSuspendPoint) ; attente par image, "
	            "par reveilleur [primitive] @adresse invitee de l'attente (nombre par image)\n",
	            static_cast<double>(frames) / seconds);
	std::vector<size_t> order;
	for (size_t i = 0; i < summaries.size(); i++) {
		if (!summaries[i].rec->pseudo && summaries[i].rec->guest && summaries[i].total != 0) {
			order.push_back(i);
		}
	}
	std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
		if (summaries[a].rec->main != summaries[b].rec->main) {
			return summaries[a].rec->main;
		}
		return summaries[a].total > summaries[b].total;
	});
	const size_t shown = std::min<size_t>(order.size(), 4);
	for (size_t n = 0; n < shown; n++) {
		const auto& s = summaries[order[n]];
		std::printf("  %s%s : attend %.2f ms/image (%.0f %% du temps)\n", s.rec->name.c_str(),
		            s.rec->main ? " (fil principal)" : "", static_cast<double>(s.total) / 1e6 / fdiv,
		            100.0 * static_cast<double>(s.total) / static_cast<double>(window_ns));
		size_t lines = 0;
		for (const auto& [key, agg]: s.waits) {
			if (lines++ == 6) {
				break;
			}
			const char* waker = key.waker < threads.size() ? threads[key.waker]->name.c_str() : "?";
			std::printf("    %6.2f ms/image  <- %s [%s]%s (%.1f/image, moy %.0f us)\n",
			            static_cast<double>(agg.ns) / 1e6 / fdiv, waker, PrimName(key.prim),
			            Address(key.ra).c_str(), static_cast<double>(agg.count) / fdiv,
			            static_cast<double>(agg.ns) / 1e3 / static_cast<double>(std::max<uint64_t>(agg.count, 1)));
		}
	}
	// Chains: from each shown thread, follow the guest thread it waited for most.
	for (size_t n = 0; n < shown; n++) {
		std::string chain   = summaries[order[n]].rec->name;
		auto        current = order[n];
		std::vector<size_t> seen {current};
		for (int hop = 0; hop < 5; hop++) {
			const auto& s = summaries[current];
			if (s.top_thread_waker == UINT32_MAX || s.top_thread_waker >= summaries.size()) {
				break;
			}
			char step[160];
			std::snprintf(step, sizeof(step), " -(%s %.2f ms/image)-> %s", PrimName(s.top_thread_prim),
			              static_cast<double>(s.top_thread_ns) / 1e6 / fdiv,
			              threads[s.top_thread_waker]->name.c_str());
			chain += step;
			current = s.top_thread_waker;
			if (std::find(seen.begin(), seen.end(), current) != seen.end()) {
				chain += " (boucle)";
				break;
			}
			seen.push_back(current);
		}
		std::printf("  chaine : %s\n", chain.c_str());
	}
	std::fflush(stdout);
}

void ReporterRun() {
	uint64_t last = NowNs();
	for (;;) {
		std::this_thread::sleep_for(std::chrono::nanoseconds(ReportPeriodNs));
		const auto now = NowNs();
		Report(now - last);
		last = now;
	}
}

void StartReporter() {
	bool expected = false;
	if (g_reporter_started.compare_exchange_strong(expected, true)) {
		std::thread(ReporterRun).detach();
	}
}

} // namespace

void SignalSlow(const void* object, uint64_t return_address, const char* label) noexcept {
	if (!Enabled() || object == nullptr) {
		return;
	}
	auto*    rec   = Self();
	uint32_t waker = rec->id;
	if (!rec->guest && label != nullptr) {
		waker = PseudoId(label);
	}
	auto& slot = g_slots[SlotIndex(object)];
	slot.key.store(0, std::memory_order_relaxed);
	slot.ns.store(NowNs(), std::memory_order_relaxed);
	slot.ra.store(return_address, std::memory_order_relaxed);
	slot.waker.store(waker, std::memory_order_relaxed);
	slot.key.store(reinterpret_cast<uint64_t>(object), std::memory_order_release);
}

void WaitScope::Begin(const void* object, Prim prim, uint64_t return_address,
                      uint64_t min_ns) noexcept {
	if (!Enabled()) {
		return;
	}
	auto* rec = Self();
	if (!rec->guest) {
		return;
	}
	if (rec->burst_count != 0) {
		FlushBurst(rec); // a blocking wait ends a yield burst
	}
	m_object = object;
	m_prim   = prim;
	m_ra     = return_address;
	m_min_ns = min_ns;
	m_start  = NowNs();
}

void WaitScope::End() noexcept {
	const auto end = NowNs();
	const auto ns  = end - m_start;
	if (ns < m_min_ns) {
		return;
	}
	uint32_t waker = UnknownId();
	if (m_prim == Prim::Sleep) {
		static const uint32_t sleep_id = PseudoId("sommeil (minuterie)");
		waker                          = sleep_id;
	} else if (m_object != nullptr) {
		const auto& slot = g_slots[SlotIndex(m_object)];
		if (slot.key.load(std::memory_order_acquire) == reinterpret_cast<uint64_t>(m_object)) {
			const auto when = slot.ns.load(std::memory_order_relaxed);
			const auto who  = slot.waker.load(std::memory_order_relaxed);
			if (when >= m_start && when <= end &&
			    slot.key.load(std::memory_order_acquire) == reinterpret_cast<uint64_t>(m_object)) {
				waker = who;
			}
		}
	}
	AddWait(t_self, waker, m_prim, m_ra, ns);
}

void NoteYieldSlow(uint64_t return_address) noexcept {
	if (!Enabled()) {
		return;
	}
	auto* rec = Self();
	if (!rec->guest) {
		return;
	}
	const auto now = NowNs();
	if (rec->burst_count != 0 && (now - rec->burst_last > YieldBurstGapNs || now < rec->burst_last)) {
		FlushBurst(rec);
	}
	if (rec->burst_count == 0) {
		rec->burst_start = now;
		rec->burst_ra    = return_address;
	}
	rec->burst_count++;
	rec->burst_last = now;
}

void MarkMainThread() noexcept {
	t_main = true;
}

void NoteFrameSlow() noexcept {
	if (Enabled()) {
		g_frames.fetch_add(1, std::memory_order_relaxed);
	}
}

} // namespace Libs::LibKernel::WaitGraph
