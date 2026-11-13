#include "graphics/guest_gpu/command_processor/parCensus.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>

namespace Libs::Graphics::ParCensus {

namespace {

constexpr size_t Slots   = 128;
constexpr size_t MaxRank = 8;

struct Record {
	uint64_t sequence = UINT64_MAX;
	uint32_t rank     = 0;
	int64_t  admit_ns = 0;
};

struct Current {
	bool     active = false;
	uint32_t rank   = 0;
	int64_t  admit_ns = 0, begin_ns = 0;
	uint64_t draws = 0, dispatches = 0, serial = 0, draws_before_serial = 0;
	bool     serial_seen = false;
	std::array<uint64_t, static_cast<size_t>(CpSeq::OpKind::Count)> kinds {};
};

struct RankTotals {
	uint64_t subs = 0, draws = 0, dispatches = 0, serial = 0, draws_before_serial = 0;
	int64_t  wait_ns = 0, resolve_ns = 0;
	int64_t  admit_after_frame_ns = 0; // admission - admission de la soumission de rang 1 de l'image
	std::array<uint64_t, static_cast<size_t>(CpSeq::OpKind::Count)> kinds {}; // operations serielles par type
};

std::mutex                 g_mutex;
std::array<Record, Slots>  g_records {};
uint32_t                   g_rank = 0;          // sous g_mutex
int64_t                    g_frame_admit_ns = 0; // sous g_mutex
std::array<int64_t, Slots> g_frame_admit {};      // admission du rang 1 de l'image de chaque soumission

Current                              g_cur; // fil GPU
uint64_t                             g_frame = 0; // images vues par le fil GPU (rang 1)
struct Writer {
	uint64_t frame = 0;
	uint32_t rank  = 0;
	uint64_t seq   = 0;
};
std::unordered_map<uint64_t, Writer> g_writers; // adresse ecrite -> derniere liste qui l'a ecrite (fil GPU)
// Attentes WaitRegMem par rang et par origine : 0 meme liste, 1 liste precedente de l'image, 2 image precedente,
// 3 inconnue (file de calcul, processeur, autre).
std::array<std::array<uint64_t, 4>, MaxRank + 1> g_waits {};
uint64_t                                         g_cur_seq = 0;
std::array<RankTotals, MaxRank + 1>  g_totals {};
int64_t                              g_window_start = 0;

int64_t NowNs() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
	    .count();
}

bool Serial(CpSeq::OpKind kind) {
	if (CpSeq::IsLockstep(kind)) {
		return true;
	}
	switch (kind) {
		case CpSeq::OpKind::EndOfPipe:
		case CpSeq::OpKind::ReleaseMem:
		case CpSeq::OpKind::EventWrite:
		case CpSeq::OpKind::WriteData:
		case CpSeq::OpKind::ReferenceClock:
		case CpSeq::OpKind::DmaData:
		case CpSeq::OpKind::LodStats:
		case CpSeq::OpKind::Flip: return true;
		default: return false;
	}
}

void Report(int64_t now) {
	const double secs = static_cast<double>(now - g_window_start) / 1e9;
	for (size_t r = 1; r <= MaxRank; r++) {
		const auto& t = g_totals[r];
		if (t.subs == 0) {
			continue;
		}
		const double n = static_cast<double>(t.subs);
		std::printf("ParCensus 10s rang %zu: %" PRIu64 " soumissions (%.1f/s) | admission %.2f ms apres le rang 1, attente "
		            "avant traduction %.2f ms, traduction %.2f ms | %.0f dessins, %.0f dispatchs, %.1f operations serielles "
		            "par soumission ; dessins avant la 1re operation serielle %.0f %%\n",
		            r, t.subs, n / secs, static_cast<double>(t.admit_after_frame_ns) / 1e6 / n,
		            static_cast<double>(t.wait_ns) / 1e6 / n, static_cast<double>(t.resolve_ns) / 1e6 / n,
		            static_cast<double>(t.draws) / n, static_cast<double>(t.dispatches) / n,
		            static_cast<double>(t.serial) / n,
		            100.0 * static_cast<double>(t.draws_before_serial) / static_cast<double>(std::max<uint64_t>(1, t.draws)));
		const auto& w = g_waits[r];
		std::printf("ParCensus 10s rang %zu, attentes WaitRegMem par soumission : meme liste %.1f, liste precedente %.1f, "
		            "image precedente %.1f, inconnue %.1f\n",
		            r, static_cast<double>(w[0]) / n, static_cast<double>(w[1]) / n, static_cast<double>(w[2]) / n,
		            static_cast<double>(w[3]) / n);
		std::printf("ParCensus 10s rang %zu, operations serielles par soumission :", r);
		for (size_t k = 0; k < t.kinds.size(); k++) {
			if (t.kinds[k] != 0) {
				std::printf(" %s=%.1f", CpSeq::OpKindName(static_cast<CpSeq::OpKind>(k)), static_cast<double>(t.kinds[k]) / n);
			}
		}
		std::printf("\n");
	}
	std::fflush(stdout);
	g_totals.fill({});
	for (auto& w: g_waits) {
		w.fill(0);
	}
	if (g_writers.size() > 200000) {
		g_writers.clear();
	}
	g_window_start = now;
}

} // namespace

bool Enabled() {
	static const bool enabled = [] {
		const char* v = std::getenv("KYTY_CP_PAR_CENSUS");
		const bool  on = v != nullptr && std::strcmp(v, "1") == 0;
		if (on) {
			std::printf("ParCensus: on (KYTY_CP_PAR_CENSUS)\n");
			std::fflush(stdout);
		}
		return on;
	}();
	return enabled;
}

void NoteAdmit(uint64_t sequence, bool frame_start) {
	if (!Enabled()) {
		return;
	}
	const auto      now = NowNs();
	std::lock_guard lock(g_mutex);
	if (frame_start) {
		g_rank           = 0;
		g_frame_admit_ns = now;
	}
	g_rank++;
	auto& rec    = g_records[sequence % Slots];
	rec.sequence = sequence;
	rec.rank     = std::min<uint32_t>(g_rank, MaxRank);
	rec.admit_ns = now;
	g_frame_admit[sequence % Slots] = g_frame_admit_ns;
}

void NoteBegin(uint64_t sequence) {
	if (!Enabled()) {
		return;
	}
	Record  rec;
	int64_t frame_admit = 0;
	{
		std::lock_guard lock(g_mutex);
		rec         = g_records[sequence % Slots];
		frame_admit = g_frame_admit[sequence % Slots];
	}
	g_cur = {};
	if (rec.sequence != sequence) {
		return; // pas une soumission graphique notee (ou anneau depasse)
	}
	g_cur.active   = true;
	g_cur.rank     = rec.rank;
	g_cur_seq      = sequence;
	if (rec.rank == 1) {
		g_frame++;
	}
	g_cur.admit_ns = rec.admit_ns;
	g_cur.begin_ns = NowNs();
	g_totals[rec.rank].admit_after_frame_ns += rec.admit_ns - frame_admit;
}

void NoteOp(CpSeq::OpKind kind, const void* payload) {
	if (!Enabled() || !g_cur.active) {
		return;
	}
	auto& c = g_cur;
	const auto note_write = [&](uint64_t address) {
		if (address != 0) {
			g_writers[address] = Writer {g_frame, c.rank, g_cur_seq};
		}
	};
	if (kind == CpSeq::OpKind::ReleaseMem) {
		const auto* body = static_cast<const CpSeq::ReleaseMemOp*>(payload)->body;
		if (((body[1] >> 29u) & 0x7u) != 0) {
			note_write(body[2] | (static_cast<uint64_t>(body[3]) << 32u));
		}
	} else if (kind == CpSeq::OpKind::WriteData) {
		note_write(static_cast<const CpSeq::WriteDataOp*>(payload)->dst);
	} else if (kind == CpSeq::OpKind::EndOfPipe) {
		note_write(static_cast<const CpSeq::EndOfPipeOp*>(payload)->dst);
	} else if (kind == CpSeq::OpKind::WaitRegMem) {
		const auto addr = static_cast<const CpSeq::WaitRegMemOp*>(payload)->addr;
		const auto it   = g_writers.find(addr);
		uint32_t   origin = 3;
		if (it != g_writers.end()) {
			if (it->second.seq == g_cur_seq) {
				origin = 0;
			} else if (it->second.frame == g_frame) {
				origin = 1;
			} else {
				origin = 2;
			}
		}
		g_waits[c.rank][origin]++;
	}
	switch (kind) {
		case CpSeq::OpKind::DrawIndex:
		case CpSeq::OpKind::DrawAuto:
		case CpSeq::OpKind::DrawIndirect:
		case CpSeq::OpKind::DrawIndirectMulti:
			c.draws++;
			if (!c.serial_seen) {
				c.draws_before_serial++;
			}
			break;
		case CpSeq::OpKind::DispatchDirect:
		case CpSeq::OpKind::DispatchIndirect: c.dispatches++; break;
		default:
			if (Serial(kind)) {
				c.kinds[static_cast<size_t>(kind)]++;
				c.serial++;
				c.serial_seen = true;
			}
			break;
	}
}

void NoteEnd(uint64_t /*sequence*/) {
	if (!Enabled() || !g_cur.active) {
		return;
	}
	const auto now = NowNs();
	auto&      c   = g_cur;
	auto&      t   = g_totals[c.rank];
	t.subs++;
	t.draws += c.draws;
	t.dispatches += c.dispatches;
	t.serial += c.serial;
	t.draws_before_serial += c.draws_before_serial;
	for (size_t k = 0; k < t.kinds.size(); k++) {
		t.kinds[k] += c.kinds[k];
	}
	t.wait_ns += c.begin_ns - c.admit_ns;
	t.resolve_ns += now - c.begin_ns;
	c.active = false;
	if (g_window_start == 0) {
		g_window_start = now;
	} else if (now - g_window_start >= 10'000'000'000LL) {
		Report(now);
	}
}

} // namespace Libs::Graphics::ParCensus
