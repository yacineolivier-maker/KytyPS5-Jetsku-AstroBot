#include "graphics/presentation/quitMenu.h"

#include "common/logging/log.h"
#include "graphics/presentation/window/hostInput.h"
#include "imgui.h"
#include "libs/controller.h"

#include <SDL3/SDL.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <initializer_list>
#include <limits>
#include <mutex>
#include <utility>
#include <vector>

// Astro Bot PC (22/11/2026) : menu de confirmation de sortie, dessine dans la surcouche ImGui
// (systemOverlay.cpp) par-dessus l'image du jeu. Dessin original (aucun element du jeu ni de
// Sony) : carte claire aux coins tres arrondis, bleu vif et blanc, etincelles simples.
//
// Entrees : WindowContext::ProcessEvent appelle QuitMenuProcessEvent en premier. Menu ouvert,
// tout appui (clavier, souris, boutons et sticks de manette, ecran tactile) est consomme ; seuls
// passent les relachements de touches/boutons appuyes AVANT l'ouverture (pour que l'etat interne
// des entrees hote reste juste) et les capteurs de mouvement (les figer serait pire). L'emulation
// n'est pas gelee : le jeu continue de tourner, sans recevoir d'appui.

namespace Libs::Graphics {

namespace {

using Clock = std::chrono::steady_clock;

constexpr int BUTTON_CONTINUE = 0;
constexpr int BUTTON_QUIT     = 1;

constexpr int16_t STICK_PRESS   = 16000;
constexpr int16_t STICK_NEUTRAL = 8000;
constexpr int16_t TRIGGER_IDLE  = 4000;

struct PadButton {
	SDL_JoystickID which;
	uint8_t        button;
	bool           operator==(const PadButton&) const = default;
};

struct State {
	bool                   open     = false;
	bool                   quitting = false;
	int                    selected = BUTTON_CONTINUE;
	int                    pressed  = -1; // bouton sous la souris au moment du clic gauche
	int                    stick    = 0;  // sens du stick gauche deja pris en compte (-1, 0, 1)
	Clock::time_point      opened_at {};
	std::vector<SDL_Keycode> keys;    // touches dont l'appui a ete consomme
	std::vector<PadButton>   buttons; // boutons de manette dont l'appui a ete consomme
	uint32_t                 mouse = 0;
};

std::mutex               g_mutex;
State                    g_state;
std::atomic_bool         g_active {false};
std::atomic<uint64_t>    g_revision {0};
std::atomic<int64_t>     g_opened_ms {0};
std::atomic<uint32_t>    g_extent_width {0};
std::atomic<uint32_t>    g_extent_height {0};
ImFont*                  g_font_regular = nullptr;
ImFont*                  g_font_bold    = nullptr;

int64_t NowMs() {
	return std::chrono::duration_cast<std::chrono::milliseconds>(
	           Clock::now().time_since_epoch())
	    .count();
}

struct Rect {
	float x0, y0, x1, y1;
	[[nodiscard]] bool Contains(float x, float y) const {
		return x >= x0 && x <= x1 && y >= y0 && y <= y1;
	}
	[[nodiscard]] float Cx() const { return (x0 + x1) * 0.5f; }
	[[nodiscard]] float Cy() const { return (y0 + y1) * 0.5f; }
};

// Mise en page en pixels de l'image presentee ; reference 1920x1080, mise a l'echelle ensuite.
struct Layout {
	float s;
	float width;
	float height;
	Rect  card;
	Rect  buttons[2];
	float title_y;
	float subtitle_y;
	float dots_y;
	float hint_y;
	float band_y;
};

constexpr float CARD_W = 900.0f;
constexpr float CARD_H = 470.0f;

Layout ComputeLayout(float width, float height) {
	Layout l {};
	l.width  = width;
	l.height = height;
	float s  = std::clamp(std::min(width / 1920.0f, height / 1080.0f), 0.3f, 4.0f);
	s        = std::min(s, (width - 32.0f) / CARD_W);
	s        = std::min(s, (height - 32.0f) / (CARD_H + 70.0f));
	s        = std::max(s, 0.2f);
	l.s      = s;

	const float cx = width * 0.5f;
	const float cy = height * 0.5f + 20.0f * s;
	l.card         = {cx - CARD_W * s * 0.5f, cy - CARD_H * s * 0.5f, cx + CARD_W * s * 0.5f,
	                  cy + CARD_H * s * 0.5f};
	l.title_y      = l.card.y0 + 78.0f * s;
	l.dots_y       = l.card.y0 + 160.0f * s;
	l.subtitle_y   = l.card.y0 + 180.0f * s;
	l.band_y       = l.card.y0 + 250.0f * s;
	const float bw = 350.0f * s;
	const float bh = 96.0f * s;
	const float gap = 40.0f * s;
	const float by  = l.card.y0 + 280.0f * s;
	l.buttons[BUTTON_CONTINUE] = {cx - gap * 0.5f - bw, by, cx - gap * 0.5f, by + bh};
	l.buttons[BUTTON_QUIT]     = {cx + gap * 0.5f, by, cx + gap * 0.5f + bw, by + bh};
	l.hint_y                   = l.card.y0 + 405.0f * s;
	return l;
}

Layout CurrentLayout(SDL_Window* window) {
	auto w = static_cast<float>(g_extent_width.load(std::memory_order_acquire));
	auto h = static_cast<float>(g_extent_height.load(std::memory_order_acquire));
	if ((w <= 0.0f || h <= 0.0f) && window != nullptr) {
		int pw = 0;
		int ph = 0;
		if (SDL_GetWindowSizeInPixels(window, &pw, &ph)) {
			w = static_cast<float>(pw);
			h = static_cast<float>(ph);
		}
	}
	return ComputeLayout(std::max(w, 64.0f), std::max(h, 64.0f));
}

// Coordonnees souris (fenetre) -> pixels de l'image presentee.
int HitTest(SDL_WindowID window_id, float x, float y) {
	SDL_Window* window = SDL_GetWindowFromID(window_id);
	const auto  layout = CurrentLayout(window);
	if (window != nullptr) {
		int ww = 0;
		int wh = 0;
		if (SDL_GetWindowSize(window, &ww, &wh) && ww > 0 && wh > 0) {
			x = x * layout.width / static_cast<float>(ww);
			y = y * layout.height / static_cast<float>(wh);
		}
	}
	for (int i = 0; i < 2; i++) {
		if (layout.buttons[i].Contains(x, y)) {
			return i;
		}
	}
	return -1;
}

void Touch() {
	g_revision.fetch_add(1, std::memory_order_acq_rel);
}

void Select(int index) {
	if (g_state.selected != index) {
		g_state.selected = index;
		Touch();
	}
}

void OpenLocked() {
	g_state.open      = true;
	g_state.quitting  = false;
	g_state.selected  = BUTTON_CONTINUE;
	g_state.pressed   = -1;
	g_state.stick     = 0;
	g_state.opened_at = Clock::now();
	g_opened_ms.store(NowMs(), std::memory_order_release);
	g_active.store(true, std::memory_order_release);
	Touch();
	HostInputSetSuspended(true);
	Controller::ResetInputState();
	LOGF("[QuitMenu] ouvert\n");
}

void CloseLocked() {
	g_state.open    = false;
	g_state.pressed = -1;
	g_active.store(false, std::memory_order_release);
	Touch();
	HostInputSetSuspended(false);
	LOGF("[QuitMenu] ferme (retour au jeu)\n");
}

void ActivateLocked(int index) {
	if (g_state.quitting) {
		return;
	}
	if (index == BUTTON_CONTINUE) {
		CloseLocked();
		return;
	}
	// Meme chemin que la croix de la fenetre : SDL_EVENT_QUIT -> GameEventQuit -> fin de
	// WindowContext::Run -> sauvegarde du cache de pipelines et arret normal.
	g_state.quitting = true;
	Touch();
	LOGF("[QuitMenu] quitter demande\n");
	SDL_Event quit {};
	quit.type = SDL_EVENT_QUIT;
	if (!SDL_PushEvent(&quit)) {
		LOGF("[QuitMenu] SDL_PushEvent(QUIT) a echoue : %s\n", SDL_GetError());
		g_state.quitting = false;
	}
}

template<class T>
bool EraseOne(std::vector<T>& list, const T& value) {
	const auto it = std::find(list.begin(), list.end(), value);
	if (it == list.end()) {
		return false;
	}
	list.erase(it);
	return true;
}

template<class T>
void AddOnce(std::vector<T>& list, const T& value) {
	if (std::find(list.begin(), list.end(), value) == list.end()) {
		list.push_back(value);
	}
}

bool AxisNeutral(uint8_t axis, int16_t value) {
	if (axis == SDL_GAMEPAD_AXIS_LEFT_TRIGGER || axis == SDL_GAMEPAD_AXIS_RIGHT_TRIGGER) {
		return value < TRIGGER_IDLE;
	}
	return value > -STICK_NEUTRAL && value < STICK_NEUTRAL;
}

// ---- dessin ----

ImU32 Rgba(int r, int g, int b, float a) {
	return IM_COL32(r, g, b, static_cast<int>(std::clamp(a, 0.0f, 1.0f) * 255.0f + 0.5f));
}

void Sparkle(ImDrawList* draw, ImVec2 c, float r, ImU32 color) {
	const float inner = r * 0.26f;
	const ImVec2 tips[4]  = {{c.x, c.y - r}, {c.x + r, c.y}, {c.x, c.y + r}, {c.x - r, c.y}};
	const float  d        = inner * 0.7071f;
	const ImVec2 waist[4] = {{c.x + d, c.y - d}, {c.x + d, c.y + d}, {c.x - d, c.y + d},
	                         {c.x - d, c.y - d}};
	for (int i = 0; i < 4; i++) {
		draw->AddTriangleFilled(c, tips[i], waist[i], color);
		draw->AddTriangleFilled(c, waist[(i + 3) % 4], tips[i], color);
	}
}

ImFont* FontOr(ImFont* font) {
	return font != nullptr ? font : ImGui::GetFont();
}

// Texte centre sur cx, reduit si besoin pour tenir dans max_width.
void CenteredText(ImDrawList* draw, ImFont* font, float size, float cx, float top, float max_width,
                  ImU32 color, const char* text) {
	font            = FontOr(font);
	const auto meas = font->CalcTextSizeA(size, std::numeric_limits<float>::max(), 0.0f, text);
	if (meas.x > max_width && meas.x > 0.0f) {
		size *= max_width / meas.x;
	}
	const auto fitted = font->CalcTextSizeA(size, std::numeric_limits<float>::max(), 0.0f, text);
	draw->AddText(font, size, {std::floor(cx - fitted.x * 0.5f), std::floor(top)}, color, text);
}

void DrawButton(ImDrawList* draw, const Rect& base, float s, bool selected, float pulse, float alpha,
                const char* label) {
	Rect r = base;
	if (selected) {
		const float grow = 1.0f + 0.025f * pulse;
		const float hw   = (base.x1 - base.x0) * 0.5f * grow;
		const float hh   = (base.y1 - base.y0) * 0.5f * grow;
		r                = {base.Cx() - hw, base.Cy() - hh, base.Cx() + hw, base.Cy() + hh};
	}
	const float rounding = (r.y1 - r.y0) * 0.5f;
	if (selected) {
		// Halo bleu qui respire.
		for (int k = 4; k >= 1; k--) {
			const float e = (static_cast<float>(k) * 5.0f + pulse * 6.0f) * s;
			draw->AddRectFilled({r.x0 - e, r.y0 - e}, {r.x1 + e, r.y1 + e},
			                    Rgba(30, 107, 255, alpha * (0.10f - 0.018f * static_cast<float>(k)) *
			                                           (0.7f + 0.6f * pulse)),
			                    rounding + e);
		}
		draw->AddRectFilled({r.x0, r.y0}, {r.x1, r.y1},
		                    Rgba(30 + static_cast<int>(18 * pulse), 107 + static_cast<int>(14 * pulse),
		                         255, alpha),
		                    rounding);
		// Reflet clair en haut du bouton.
		draw->AddRectFilled({r.x0 + 10.0f * s, r.y0 + 6.0f * s},
		                    {r.x1 - 10.0f * s, r.Cy() - 2.0f * s}, Rgba(255, 255, 255, alpha * 0.16f),
		                    rounding, ImDrawFlags_RoundCornersTop);
		// Contour lumineux.
		const float o = 5.0f * s;
		draw->AddRect({r.x0 - o, r.y0 - o}, {r.x1 + o, r.y1 + o},
		              Rgba(150, 200, 255, alpha * (0.55f + 0.45f * pulse)), rounding + o, 0,
		              3.0f * s);
		draw->AddRect({r.x0 + 2.0f * s, r.y0 + 2.0f * s}, {r.x1 - 2.0f * s, r.y1 - 2.0f * s},
		              Rgba(255, 255, 255, alpha * 0.55f), rounding - 2.0f * s, 0, 2.0f * s);
	} else {
		draw->AddRectFilled({r.x0, r.y0}, {r.x1, r.y1}, Rgba(255, 255, 255, alpha), rounding);
		draw->AddRect({r.x0, r.y0}, {r.x1, r.y1}, Rgba(30, 107, 255, alpha * 0.85f), rounding, 0,
		              3.0f * s);
	}
	const float size = 33.0f * s;
	const ImU32 text = selected ? Rgba(255, 255, 255, alpha) : Rgba(30, 107, 255, alpha);
	CenteredText(draw, g_font_bold, size, r.Cx(), r.Cy() - size * 0.68f, (r.x1 - r.x0) - 36.0f * s,
	             text, label);
}

} // namespace

bool QuitMenuEnabled() noexcept {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_QUIT_MENU");
		const bool  on    = value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
		if (on) {
			LOGF("[QuitMenu] actif (KYTY_QUIT_MENU=%s) : Echap ouvre le menu de sortie\n", value);
		}
		return on;
	}();
	return enabled;
}

bool QuitMenuActive() noexcept {
	return g_active.load(std::memory_order_acquire);
}

uint64_t QuitMenuRevision() noexcept {
	uint64_t revision = g_revision.load(std::memory_order_acquire);
	if (QuitMenuActive()) {
		// Pas d'animation d'environ 33 ms : force le rafraichissement de la surcouche meme si le
		// jeu ne presente plus d'image.
		revision += static_cast<uint64_t>((NowMs() - g_opened_ms.load(std::memory_order_acquire)) / 33);
	}
	return revision;
}

bool QuitMenuProcessEvent(const SDL_Event& event) {
	if (!QuitMenuEnabled()) {
		return false;
	}
	std::scoped_lock lock(g_mutex);
	auto&            st = g_state;
	switch (event.type) {
		case SDL_EVENT_KEY_DOWN: {
			const SDL_Keycode key = event.key.key;
			if (!st.open) {
				if (key == SDLK_ESCAPE && !event.key.repeat && (event.key.mod & SDL_KMOD_ALT) == 0) {
					AddOnce(st.keys, key);
					OpenLocked();
					return true;
				}
				return false;
			}
			AddOnce(st.keys, key);
			if (st.quitting) {
				return true;
			}
			switch (key) {
				case SDLK_LEFT: Select(BUTTON_CONTINUE); break;
				case SDLK_RIGHT: Select(BUTTON_QUIT); break;
				case SDLK_TAB:
				case SDLK_UP:
				case SDLK_DOWN:
					if (!event.key.repeat) {
						Select(1 - st.selected);
					}
					break;
				case SDLK_RETURN:
				case SDLK_RETURN2:
				case SDLK_KP_ENTER:
				case SDLK_SPACE:
					if (!event.key.repeat) {
						ActivateLocked(st.selected);
					}
					break;
				case SDLK_ESCAPE:
					if (!event.key.repeat) {
						CloseLocked();
					}
					break;
				default: break;
			}
			return true;
		}
		case SDL_EVENT_KEY_UP:
			// Relachement d'une touche appuyee avant l'ouverture : on le laisse passer.
			return EraseOne(st.keys, event.key.key) || (st.open && st.quitting);

		case SDL_EVENT_TEXT_INPUT:
		case SDL_EVENT_TEXT_EDITING:
		case SDL_EVENT_MOUSE_WHEEL:
		case SDL_EVENT_FINGER_DOWN:
		case SDL_EVENT_FINGER_MOTION:
		case SDL_EVENT_FINGER_UP: return st.open;

		case SDL_EVENT_MOUSE_MOTION:
			if (!st.open) {
				return false;
			}
			if (!st.quitting) {
				if (const int hit = HitTest(event.motion.windowID, event.motion.x, event.motion.y);
				    hit >= 0) {
					Select(hit);
				}
			}
			return true;

		case SDL_EVENT_MOUSE_BUTTON_DOWN: {
			if (!st.open) {
				return false;
			}
			st.mouse |= 1u << (event.button.button & 31u);
			if (event.button.button == SDL_BUTTON_LEFT && !st.quitting) {
				st.pressed = HitTest(event.button.windowID, event.button.x, event.button.y);
				if (st.pressed >= 0) {
					Select(st.pressed);
				}
			}
			return true;
		}
		case SDL_EVENT_MOUSE_BUTTON_UP: {
			const uint32_t bit = 1u << (event.button.button & 31u);
			if ((st.mouse & bit) == 0) {
				return st.open && st.quitting;
			}
			st.mouse &= ~bit;
			if (st.open && event.button.button == SDL_BUTTON_LEFT && st.pressed >= 0) {
				const int target = std::exchange(st.pressed, -1);
				if (HitTest(event.button.windowID, event.button.x, event.button.y) == target) {
					ActivateLocked(target);
				}
			}
			return true;
		}

		case SDL_EVENT_GAMEPAD_BUTTON_DOWN: {
			if (!st.open) {
				return false;
			}
			AddOnce(st.buttons, PadButton {event.gbutton.which, event.gbutton.button});
			if (st.quitting) {
				return true;
			}
			switch (event.gbutton.button) {
				case SDL_GAMEPAD_BUTTON_DPAD_LEFT: Select(BUTTON_CONTINUE); break;
				case SDL_GAMEPAD_BUTTON_DPAD_RIGHT: Select(BUTTON_QUIT); break;
				case SDL_GAMEPAD_BUTTON_SOUTH: ActivateLocked(st.selected); break; // A / croix
				case SDL_GAMEPAD_BUTTON_EAST:                                      // B / rond
				case SDL_GAMEPAD_BUTTON_START: CloseLocked(); break;
				default: break;
			}
			return true;
		}
		case SDL_EVENT_GAMEPAD_BUTTON_UP:
			return EraseOne(st.buttons, PadButton {event.gbutton.which, event.gbutton.button}) ||
			       (st.open && st.quitting);

		case SDL_EVENT_GAMEPAD_AXIS_MOTION: {
			if (!st.open) {
				return false;
			}
			const auto axis  = event.gaxis.axis;
			const auto value = event.gaxis.value;
			if (axis == SDL_GAMEPAD_AXIS_LEFTX && !st.quitting) {
				if (value <= -STICK_PRESS && st.stick != -1) {
					st.stick = -1;
					Select(BUTTON_CONTINUE);
				} else if (value >= STICK_PRESS && st.stick != 1) {
					st.stick = 1;
					Select(BUTTON_QUIT);
				} else if (value > -STICK_NEUTRAL && value < STICK_NEUTRAL) {
					st.stick = 0;
				}
			}
			// Un retour au neutre ne fait que relacher : il peut aller au jeu.
			return st.quitting || !AxisNeutral(axis, value);
		}

		case SDL_EVENT_GAMEPAD_TOUCHPAD_DOWN:
		case SDL_EVENT_GAMEPAD_TOUCHPAD_MOTION: return st.open;

		default: return false;
	}
}

void QuitMenuLoadFonts(ImGuiIO& io) {
	g_font_regular = nullptr;
	g_font_bold    = nullptr;
	if (!QuitMenuEnabled()) {
		return;
	}
	const auto load = [&io](std::initializer_list<const char*> paths) -> ImFont* {
		for (const char* path: paths) {
			std::error_code error;
			if (std::filesystem::is_regular_file(path, error)) {
				ImFontConfig config;
				config.Flags = ImFontFlags_NoLoadError;
				if (auto* font = io.Fonts->AddFontFromFileTTF(path, 32.0f, &config); font != nullptr) {
					return font;
				}
			}
		}
		return nullptr;
	};
	g_font_regular = load({"C:/Windows/Fonts/segoeui.ttf"});
	g_font_bold    = load({"C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/seguisb.ttf",
	                       "C:/Windows/Fonts/segoeui.ttf"});
	if (g_font_regular == nullptr || g_font_bold == nullptr) {
		LOGF("[QuitMenu] police Segoe UI introuvable : police par defaut d'ImGui\n");
	}
}

void DrawQuitMenu(vk::Extent2D extent) {
	bool              quitting = false;
	int               selected = BUTTON_CONTINUE;
	Clock::time_point opened_at;
	{
		std::scoped_lock lock(g_mutex);
		if (!g_state.open) {
			return;
		}
		quitting  = g_state.quitting;
		selected  = g_state.selected;
		opened_at = g_state.opened_at;
	}
	if (extent.width == 0 || extent.height == 0) {
		return;
	}
	g_extent_width.store(extent.width, std::memory_order_release);
	g_extent_height.store(extent.height, std::memory_order_release);

	const auto  l      = ComputeLayout(static_cast<float>(extent.width), static_cast<float>(extent.height));
	const float s      = l.s;
	const float t      = std::chrono::duration<float>(Clock::now() - opened_at).count();
	const float appear = std::clamp(t / 0.2f, 0.0f, 1.0f);
	const float a      = 1.0f - (1.0f - appear) * (1.0f - appear); // ease-out
	const float pulse  = 0.5f + 0.5f * std::sin(t * 2.0f * 3.14159265f * 0.85f);
	auto*       draw   = ImGui::GetForegroundDrawList();

	// Fond : l'image du jeu assombrie, teinte bleu nuit.
	draw->AddRectFilledMultiColor({0.0f, 0.0f}, {l.width, l.height}, Rgba(8, 22, 64, a * 0.62f),
	                              Rgba(8, 22, 64, a * 0.62f), Rgba(2, 8, 30, a * 0.80f),
	                              Rgba(2, 8, 30, a * 0.80f));

	// Petite entree en douceur de la carte (glisse de quelques pixels).
	const float lift = (1.0f - a) * 24.0f * s;
	Rect        card = l.card;
	card.y0 += lift;
	card.y1 += lift;
	const float R = 56.0f * s;

	// Etincelles decoratives autour de la carte (scintillement).
	struct Star {
		float x, y, r, phase;
	};
	constexpr Star stars[] = {{-0.60f, -0.50f, 20.0f, 0.0f}, {0.62f, -0.40f, 14.0f, 1.7f},
	                          {-0.66f, 0.30f, 12.0f, 3.1f},  {0.58f, 0.52f, 22.0f, 4.2f},
	                          {-0.28f, -0.70f, 10.0f, 2.4f}, {0.30f, -0.68f, 16.0f, 5.3f},
	                          {0.70f, 0.05f, 9.0f, 0.9f},    {-0.52f, 0.62f, 15.0f, 3.8f}};
	const float    hw      = (card.x1 - card.x0) * 0.5f;
	const float    hh      = (card.y1 - card.y0) * 0.5f;
	for (const auto& star: stars) {
		const float twinkle = 0.55f + 0.45f * std::sin(t * 2.2f + star.phase);
		const ImVec2 c {card.Cx() + star.x * hw * 1.45f, card.Cy() + star.y * hh * 1.55f};
		if (c.x < 0.0f || c.y < 0.0f || c.x > l.width || c.y > l.height) {
			continue;
		}
		Sparkle(draw, c, star.r * s * (0.85f + 0.15f * twinkle), Rgba(200, 225, 255, a * twinkle));
	}

	// Halo bleu derriere la carte, puis ombre portee.
	for (int k = 6; k >= 1; k--) {
		const float e = static_cast<float>(k) * 12.0f * s;
		draw->AddRectFilled({card.x0 - e, card.y0 - e}, {card.x1 + e, card.y1 + e},
		                    Rgba(30, 107, 255, a * 0.035f), R + e);
	}
	for (int k = 4; k >= 1; k--) {
		const float e = static_cast<float>(k) * 6.0f * s;
		draw->AddRectFilled({card.x0 - e * 0.5f, card.y0 + e}, {card.x1 + e * 0.5f, card.y1 + e * 1.6f},
		                    Rgba(0, 8, 30, a * 0.10f), R + e);
	}

	// Carte : blanche en haut, bleu tres clair sous les boutons.
	draw->AddRectFilled({card.x0, card.y0}, {card.x1, card.y1}, Rgba(255, 255, 255, a), R);
	draw->AddRectFilled({card.x0, l.band_y + lift}, {card.x1, card.y1}, Rgba(229, 240, 255, a), R,
	                    ImDrawFlags_RoundCornersBottom);
	draw->AddRect({card.x0, card.y0}, {card.x1, card.y1}, Rgba(176, 206, 255, a), R, 0, 3.0f * s);

	// Pastille ronde en haut de la carte, etincelle blanche au centre.
	const ImVec2 badge {card.Cx(), card.y0};
	const float  br = 46.0f * s;
	draw->AddCircleFilled(badge, br + (8.0f + 6.0f * pulse) * s, Rgba(30, 107, 255, a * 0.22f), 48);
	draw->AddCircleFilled(badge, br, Rgba(30, 107, 255, a), 48);
	draw->AddCircle(badge, br, Rgba(255, 255, 255, a), 48, 6.0f * s);
	Sparkle(draw, badge, 24.0f * s * (0.92f + 0.08f * pulse), Rgba(255, 255, 255, a));

	const float text_w = (card.x1 - card.x0) - 80.0f * s;
	CenteredText(draw, g_font_bold, 60.0f * s, card.Cx(), l.title_y + lift, text_w,
	             Rgba(14, 34, 92, a), "Quitter Astro Bot\xC2\xA0?");

	// Trois points d'accent sous le titre.
	for (int i = -1; i <= 1; i++) {
		const float bounce = std::sin(t * 4.0f - static_cast<float>(i + 1) * 0.6f);
		draw->AddCircleFilled({card.Cx() + static_cast<float>(i) * 22.0f * s,
		                       l.dots_y + lift - std::max(bounce, 0.0f) * 3.0f * s},
		                      5.5f * s, i == 0 ? Rgba(120, 175, 255, a) : Rgba(30, 107, 255, a), 16);
	}

	const char* subtitle = quitting ? "Fermeture en cours\xE2\x80\xA6"
	                                : "La progression d\xC3\xA9j\xC3\xA0 sauvegard\xC3\xA9"
	                                  "e par le jeu est conserv\xC3\xA9"
	                                  "e.";
	CenteredText(draw, g_font_regular, 29.0f * s, card.Cx(), l.subtitle_y + lift, text_w,
	             Rgba(70, 88, 128, a), subtitle);

	for (int i = 0; i < 2; i++) {
		Rect r = l.buttons[i];
		r.y0 += lift;
		r.y1 += lift;
		DrawButton(draw, r, s, i == selected, quitting ? 0.0f : pulse, a * (quitting ? 0.6f : 1.0f),
		           i == BUTTON_CONTINUE ? "Continuer \xC3\xA0 jouer" : "Quitter");
	}

	CenteredText(draw, g_font_regular, 22.0f * s, card.Cx(), l.hint_y + lift, text_w,
	             Rgba(100, 118, 156, a),
	             "Entr\xC3\xA9"
	             "e / A : valider   \xC2\xB7   \xC3\x89"
	             "chap / B : annuler   \xC2\xB7   Fl\xC3\xA8"
	             "ches : choisir");
}

} // namespace Libs::Graphics
