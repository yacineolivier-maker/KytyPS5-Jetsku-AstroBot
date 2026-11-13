#include "graphics/host_gpu/renderer/passStats.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics::PassStats {

namespace {

struct Key {
	uint64_t address       = 0;
	uint64_t depth_address = 0;
	uint32_t width         = 0;
	uint32_t height        = 0;
	uint32_t format        = 0;
	uint32_t colors        = 0;
	uint64_t vs            = 0; // KYTY_PASS_STATS_SHADERS : empreintes des shaders
	uint64_t ps            = 0;
	uint8_t  kind          = 0; // 0 dessin avec cible, 1 dessin sans cible (abandonne), 2 dispatch
	uint8_t  depth         = 0;

	bool operator==(const Key& o) const noexcept {
		return address == o.address && depth_address == o.depth_address && width == o.width &&
		       height == o.height && format == o.format && colors == o.colors && kind == o.kind &&
		       depth == o.depth && vs == o.vs && ps == o.ps;
	}
};

struct KeyHash {
	size_t operator()(const Key& k) const noexcept {
		uint64_t h = k.address * 0x9E3779B97F4A7C15ull ^ k.depth_address * 0xC2B2AE3D27D4EB4Full;
		h ^= (static_cast<uint64_t>(k.width) << 32u | k.height) * 0x165667B19E3779F9ull;
		h ^= (static_cast<uint64_t>(k.format) << 16u | k.colors << 8u | k.kind << 4u | k.depth);
		h ^= k.vs * 0x27D4EB2F165667C5ull ^ k.ps * 0x94D049BB133111EBull;
		return static_cast<size_t>(h ^ (h >> 29u));
	}
};

struct Totals {
	uint64_t count = 0;
	int64_t  ticks = 0;
};

struct State {
	bool                                       target_set = false;
	Key                                        target;
	int                                        depth = 0;
	std::unordered_map<Key, Totals, KeyHash>   window;
	int64_t                                    window_start = 0;
	uint64_t                                   first_frame  = 0;
	uint64_t                                   last_frame   = 0;
	int64_t                                    all_ticks    = 0;
};

State& Get() {
	static State state; // fil GPU seulement
	return state;
}

int64_t Now() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

double TicksToUs(int64_t ticks) {
	return static_cast<double>(ticks) / 1000.0;
}

bool ShadersEnabled() {
	static const bool enabled = [] {
		const char* v = std::getenv("KYTY_PASS_STATS_SHADERS");
		return v != nullptr && std::strcmp(v, "1") == 0;
	}();
	return enabled;
}

void Report(State& s, int64_t now) {
	const auto frames = s.last_frame > s.first_frame ? s.last_frame - s.first_frame : 1u;
	std::vector<std::pair<Key, Totals>> v(s.window.begin(), s.window.end());
	std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.second.ticks > b.second.ticks; });
	uint64_t all_count = 0;
	for (const auto& e: v) {
		all_count += e.second.count;
	}
	const double wall_us = TicksToUs(now - s.window_start);
	std::printf("PassStats: %" PRIu64 " images en %.1f s, fil GPU dans les dessins et dispatchs %.2f ms/image "
	            "(%.0f %% du temps), %.0f operations/image, %zu cibles\n",
	            frames, wall_us / 1e6, TicksToUs(s.all_ticks) / 1000.0 / static_cast<double>(frames),
	            100.0 * TicksToUs(s.all_ticks) / std::max(1.0, wall_us),
	            static_cast<double>(all_count) / static_cast<double>(frames), v.size());
	const size_t shown = std::min<size_t>(v.size(), ShadersEnabled() ? 60 : 40);
	for (size_t i = 0; i < shown; i++) {
		const auto& k = v[i].first;
		const auto& t = v[i].second;
		const char* kind = k.kind == 2 ? "dispatch" : k.kind == 1 ? "sans-cible" : (k.colors == 0 ? "profondeur" : "couleur");
		std::printf("PassStats   %5.2f ms/image %6.1f ops/image %5.1f us/op  %-10s %4ux%-4u fmt %3u  "
		            "cibles %u%s  rt 0x%010" PRIx64 " z 0x%010" PRIx64 " vs %016" PRIx64 " ps %016" PRIx64 "\n",
		            TicksToUs(t.ticks) / 1000.0 / static_cast<double>(frames),
		            static_cast<double>(t.count) / static_cast<double>(frames),
		            TicksToUs(t.ticks) / static_cast<double>(std::max<uint64_t>(1, t.count)), kind, k.width,
		            k.height, k.format, k.colors, k.depth ? "+z" : "", k.address, k.depth_address, k.vs, k.ps);
	}
	std::fflush(stdout);
	s.window.clear();
	s.all_ticks    = 0;
	s.window_start = now;
	s.first_frame  = s.last_frame;
}

} // namespace

bool Enabled() {
	static const bool enabled = [] {
		const char* v = std::getenv("KYTY_PASS_STATS");
		return v != nullptr && std::strcmp(v, "1") == 0;
	}();
	return enabled;
}

void SetTarget(uint64_t address, uint32_t width, uint32_t height, uint32_t format, uint32_t colors, bool depth,
               uint64_t depth_address, uint64_t vs, uint64_t ps) {
	if (!Enabled()) {
		return;
	}
	auto& s              = Get();
	s.target_set         = true;
	s.target.address     = address;
	s.target.width       = width;
	s.target.height      = height;
	s.target.format      = format;
	s.target.colors      = colors;
	s.target.depth       = depth ? 1u : 0u;
	s.target.depth_address = depth_address;
	s.target.kind        = 0;
	s.target.vs          = ShadersEnabled() ? vs : 0;
	s.target.ps          = ShadersEnabled() ? ps : 0;
}

Scope::Scope(Kind kind, uint64_t frame) {
	if (!Enabled()) {
		return;
	}
	auto& s = Get();
	if (s.depth++ != 0) {
		return; // imbrique : compte dans l'operation exterieure
	}
	m_active     = true;
	m_kind       = kind;
	m_frame      = frame;
	s.target_set = false;
	m_start      = Now();
}

Scope::~Scope() {
	if (!Enabled()) {
		return;
	}
	auto& s = Get();
	s.depth--;
	if (!m_active) {
		return;
	}
	const auto now = Now();
	Key        key;
	if (m_kind == Kind::Dispatch) {
		key.kind = 2;
	} else if (s.target_set) {
		key = s.target;
	} else {
		key.kind = 1;
	}
	auto& t = s.window[key];
	t.count++;
	t.ticks += now - m_start;
	s.all_ticks += now - m_start;
	if (s.window_start == 0) {
		s.window_start = now;
		s.first_frame  = m_frame;
	}
	s.last_frame = std::max(s.last_frame, m_frame);
	// KYTY_PASS_ORDER_FRAME=<n> : ordre des passes de l'image n (operations consecutives de meme cible regroupees),
	// ecrit une fois au debut de l'image suivante.
	// KYTY_PASS_ORDER_AT_S=<s> : l'image qui suit <s> secondes apres la premiere operation mesuree.
	static uint64_t order_frame = [] {
		const char* v = std::getenv("KYTY_PASS_ORDER_FRAME");
		return v != nullptr ? std::strtoull(v, nullptr, 10) : 0ull;
	}();
	static const double order_at_us = [] {
		const char* v = std::getenv("KYTY_PASS_ORDER_AT_S");
		return v != nullptr ? std::strtod(v, nullptr) * 1e6 : 0.0;
	}();
	static int64_t first_op = 0;
	if (first_op == 0) {
		first_op = now;
	}
	if (order_frame == 0 && order_at_us > 0.0 && TicksToUs(now - first_op) >= order_at_us) {
		order_frame = m_frame + 1;
	}
	if (order_frame != 0) {
		static std::vector<std::pair<Key, uint64_t>> order;
		static bool                                  printed = false;
		if (m_frame == order_frame) {
			if (!order.empty() && order.back().first == key) {
				order.back().second++;
			} else {
				order.emplace_back(key, 1);
			}
		} else if (m_frame > order_frame && !printed) {
			printed = true;
			std::printf("PassOrder: image %" PRIu64 ", %zu groupes\n", order_frame, order.size());
			for (size_t i = 0; i < order.size(); i++) {
				const auto& k = order[i].first;
				std::printf("PassOrder %4zu  %5" PRIu64 " ops  %-10s %4ux%-4u fmt %3u cibles %u%s rt 0x%010" PRIx64
				            " z 0x%010" PRIx64 "\n",
				            i, order[i].second,
				            k.kind == 2 ? "dispatch" : k.kind == 1 ? "sans-cible" : (k.colors == 0 ? "profondeur" : "couleur"),
				            k.width, k.height, k.format, k.colors, k.depth ? "+z" : "", k.address, k.depth_address);
			}
			std::fflush(stdout);
		}
	}
	if (TicksToUs(now - s.window_start) >= 10e6) {
		Report(s, now);
	}
}

void NoteColors(const uint64_t* address, const uint32_t* format, const uint32_t* width, const uint32_t* height,
                uint32_t count) {
	if (!Enabled()) {
		return;
	}
	static std::vector<uint64_t> seen;
	uint64_t                     h = count;
	for (uint32_t i = 0; i < count; i++) {
		h = (h ^ address[i]) * 0x100000001B3ull ^ format[i];
	}
	if (std::find(seen.begin(), seen.end(), h) != seen.end() || seen.size() > 512) {
		return;
	}
	seen.push_back(h);
	std::printf("PassColors: %u cibles :", count);
	for (uint32_t i = 0; i < count; i++) {
		std::printf(" [%u] 0x%010" PRIx64 " fmt %u %ux%u", i, address[i], format[i], width[i], height[i]);
	}
	std::printf("\n");
	std::fflush(stdout);
}

} // namespace Libs::Graphics::PassStats
