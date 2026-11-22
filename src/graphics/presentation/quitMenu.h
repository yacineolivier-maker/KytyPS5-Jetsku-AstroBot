#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_QUITMENU_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_QUITMENU_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>

union SDL_Event;
struct ImGuiIO;

namespace Libs::Graphics {

// Astro Bot PC (22/11/2026) : menu « Quitter Astro Bot ? » ouvert par Echap.
// Interrupteur KYTY_QUIT_MENU=1 ; absent ou 0 : aucun effet (Echap garde son role d'origine).

[[nodiscard]] bool QuitMenuEnabled() noexcept;
// Vrai tant que le menu est affiche (les entrees ne vont plus au jeu).
[[nodiscard]] bool QuitMenuActive() noexcept;
// Change a chaque ouverture/fermeture/selection et, menu ouvert, a chaque pas d'animation.
[[nodiscard]] uint64_t QuitMenuRevision() noexcept;

// Fil de la fenetre, en tete de WindowContext::ProcessEvent. Vrai si l'evenement est consomme
// par le menu (il ne doit alors aller ni au jeu ni aux autres surcouches).
[[nodiscard]] bool QuitMenuProcessEvent(const SDL_Event& event);

// Charge les polices Segoe UI du menu dans l'atlas ImGui de la surcouche systeme (si actif).
void QuitMenuLoadFonts(ImGuiIO& io);

// Fil de presentation, entre ImGui::NewFrame() et ImGui::Render().
void DrawQuitMenu(vk::Extent2D extent);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_QUITMENU_H_
