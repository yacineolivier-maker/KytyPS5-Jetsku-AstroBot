#include "graphics/guest_gpu/command_processor/stateTrace.h"

#include "graphics/guest_gpu/pm4.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace Libs::Graphics::StateTrace {

namespace {

struct Reg {
	uint64_t sub     = 0; // soumission qui l'a ecrit en dernier (0 : jamais depuis la remise a zero)
	uint32_t value   = 0;
	bool     varying = false; // a deja recu une valeur differente
};

struct State {
	std::array<Reg, Pm4::CX_NUM> cx {};
	std::array<Reg, Pm4::SH_NUM> sh {};
	std::array<uint64_t, Pm4::CX_NUM> cx_hits {}; // herite variable au premier dessin (fenetre)
	std::array<uint64_t, Pm4::SH_NUM> sh_hits {};
	uint64_t cur          = 0;
	bool     first_draw   = true;
	bool     on_thread    = false;
	uint64_t subs_in_frame = 0;
	// Fenetre de 10 s.
	int64_t  window_start = 0;
	uint64_t frames = 0, subs = 0, subs_with_draws = 0, draws = 0;
	uint64_t first_inherited = 0, first_inherited_var = 0, first_max_var = 0;
	uint64_t draws_with_var = 0;
	uint64_t self_contained = 0; // soumissions dont le premier dessin n'herite d'aucun registre variable
	std::array<uint64_t, 9> draws_by_index {}; // dessins par rang de la soumission dans l'image (8 = 8e et suivantes)
};

State& S() {
	static State s; // fil du sequenceur
	return s;
}

thread_local bool t_sequencer = false;

int64_t NowMs() {
	return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch())
	    .count();
}

template <size_t N>
void Write(std::array<Reg, N>& regs, uint32_t offset, uint32_t count, const uint32_t* values, uint64_t cur) {
	for (uint32_t i = 0; i < count; i++) {
		const uint32_t o = offset + i;
		if (o >= N) {
			break;
		}
		auto& r = regs[o];
		if (r.sub != 0 && r.value != values[i]) {
			r.varying = true;
		}
		r.sub   = cur;
		r.value = values[i];
	}
}

void Report(State& s, int64_t now) {
	const double ns = static_cast<double>(std::max<uint64_t>(1, s.subs_with_draws));
	std::printf("StateTrace 10s: images %" PRIu64 ", soumissions %" PRIu64 " (dont avec dessins %" PRIu64 "), dessins %" PRIu64
	            " | au 1er dessin d'une soumission : registres herites %.1f en moyenne, dont variables %.1f (max %" PRIu64
	            ") ; soumissions sans heritage variable %" PRIu64 " (%.0f %%) | dessins avec heritage variable %" PRIu64 "\n",
	            s.frames, s.subs, s.subs_with_draws, s.draws, static_cast<double>(s.first_inherited) / ns,
	            static_cast<double>(s.first_inherited_var) / ns, s.first_max_var, s.self_contained,
	            100.0 * static_cast<double>(s.self_contained) / ns, s.draws_with_var);
	std::vector<std::pair<uint64_t, uint32_t>> top; // (compte, registre | 0x80000000 pour un registre de shader)
	for (uint32_t i = 0; i < Pm4::CX_NUM; i++) {
		if (s.cx_hits[i] != 0) top.emplace_back(s.cx_hits[i], i);
	}
	for (uint32_t i = 0; i < Pm4::SH_NUM; i++) {
		if (s.sh_hits[i] != 0) top.emplace_back(s.sh_hits[i], i | 0x80000000u);
	}
	std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
	std::printf("StateTrace herites variables (%zu registres) :", top.size());
	for (size_t k = 0; k < top.size() && k < 160; k++) {
		std::printf(" %s0x%x=%" PRIu64, (top[k].second & 0x80000000u) != 0 ? "sh" : "cx", top[k].second & 0x7fffffffu,
		            top[k].first);
	}
	std::printf("\n");
	std::printf("StateTrace dessins par rang de soumission dans l'image :");
	for (size_t k = 1; k < s.draws_by_index.size(); k++) {
		std::printf(" %zu=%" PRIu64, k, s.draws_by_index[k]);
	}
	std::printf("\n");
	std::fflush(stdout);
	const auto cx = s.cx;
	const auto sh = s.sh;
	const auto cur = s.cur;
	const bool first = s.first_draw;
	const auto in_frame = s.subs_in_frame;
	s = State {};
	s.cx = cx;
	s.sh = sh;
	s.cur = cur;
	s.first_draw = first;
	s.subs_in_frame = in_frame;
	s.on_thread = true;
	s.window_start = now;
}

} // namespace

bool Enabled() {
	static const bool enabled = [] {
		const char* v = std::getenv("KYTY_STATE_TRACE");
		const bool  on = v != nullptr && std::strcmp(v, "1") == 0;
		if (on) {
			std::printf("StateTrace: on (KYTY_STATE_TRACE)\n");
			std::fflush(stdout);
		}
		return on;
	}();
	return enabled;
}

void BeginSubmission(uint64_t sequence, bool frame_start) {
	if (!Enabled()) {
		return;
	}
	t_sequencer = true;
	auto& s = S();
	if (frame_start) {
		// Remise a zero du processeur : plus aucun registre n'est herite.
		for (auto& r: s.cx) r.sub = 0;
		for (auto& r: s.sh) r.sub = 0;
		s.frames++;
		s.subs_in_frame = 0;
	}
	s.cur        = sequence + 1; // jamais 0
	s.first_draw = true;
	s.subs++;
	s.subs_in_frame++;
}

void NoteContext(uint32_t offset, uint32_t count, const uint32_t* values) {
	if (!Enabled() || !t_sequencer) {
		return;
	}
	auto& s = S();
	Write(s.cx, offset, count, values, s.cur);
}

void NoteShader(uint32_t offset, uint32_t count, const uint32_t* values) {
	if (!Enabled() || !t_sequencer) {
		return;
	}
	auto& s = S();
	Write(s.sh, offset, count, values, s.cur);
}

void NoteDraw() {
	if (!Enabled() || !t_sequencer) {
		return;
	}
	auto& s = S();
	s.draws++;
	s.draws_by_index[std::min<uint64_t>(8, s.subs_in_frame)]++;
	uint64_t inherited = 0, inherited_var = 0;
	const bool first = s.first_draw;
	for (uint32_t i = 0; i < Pm4::CX_NUM; i++) {
		const auto& r = s.cx[i];
		if (r.sub != 0 && r.sub != s.cur) {
			inherited++;
			if (r.varying) {
				inherited_var++;
				if (first) s.cx_hits[i]++;
			}
		}
	}
	for (uint32_t i = 0; i < Pm4::SH_NUM; i++) {
		const auto& r = s.sh[i];
		if (r.sub != 0 && r.sub != s.cur) {
			inherited++;
			if (r.varying) {
				inherited_var++;
				if (first) s.sh_hits[i]++;
			}
		}
	}
	if (inherited_var != 0) {
		s.draws_with_var++;
	}
	if (first) {
		s.first_draw = false;
		s.subs_with_draws++;
		s.first_inherited += inherited;
		s.first_inherited_var += inherited_var;
		s.first_max_var = std::max(s.first_max_var, inherited_var);
		if (inherited_var == 0) {
			s.self_contained++;
		}
	}
	const auto now = NowMs();
	if (s.window_start == 0) {
		s.window_start = now;
	} else if (now - s.window_start >= 10000) {
		Report(s, now);
	}
}

} // namespace Libs::Graphics::StateTrace
