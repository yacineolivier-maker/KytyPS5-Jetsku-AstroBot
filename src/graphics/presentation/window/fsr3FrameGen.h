#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_WINDOW_FSR3FRAMEGEN_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_WINDOW_FSR3FRAMEGEN_H_

// Astro Bot PC (13/11/2026) : generation d'images AMD FSR 3 (FidelityFX SDK 1.1.4, licence MIT) a la presentation.
//
// KYTY_FSR3_FG=1 : la chaine de presentation de l'emulateur est remplacee par celle de FSR 3
// (amd_fidelityfx_vk.dll, a cote de l'executable), qui intercale une image interpolee entre deux images du jeu et
// regle elle-meme la cadence d'affichage. Prototype A : profondeur neutre et vecteurs de mouvement nuls (FSR 3
// s'appuie alors sur son flux optique). Desactive par defaut ; en cas d'echec, la presentation normale reprend.
// KYTY_FSR3_FG_DEBUG=1 : affiche les reperes de cadence et de reinitialisation de FSR 3.

#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>

namespace Libs::Graphics {
struct GraphicContext;
} // namespace Libs::Graphics

namespace Libs::Graphics::Fsr3 {

[[nodiscard]] bool Requested();
// La chaine FSR 3 est en place (CreateSwapchain a reussi).
[[nodiscard]] bool Active();

// Remplace vkCreateSwapchainKHR. Faux : rien n'est cree, l'appelant cree sa chaine normale.
bool CreateSwapchain(GraphicContext& graphics, const VkSwapchainCreateInfoKHR& info, VkSwapchainKHR* swapchain);
void DestroySwapchain(GraphicContext& graphics);

VkResult GetSwapchainImages(VkDevice device, VkSwapchainKHR swapchain, uint32_t* count, VkImage* images);
VkResult AcquireNextImage(VkDevice device, VkSwapchainKHR swapchain, uint64_t timeout, VkSemaphore semaphore,
                          VkFence fence, uint32_t* index);
VkResult QueuePresent(VkQueue queue, const VkPresentInfoKHR* info);
// Images affichees par seconde (reelles + generees), mesurees chaque seconde ; 0 si inactif.
[[nodiscard]] double DisplayedFps();
// Vrai pendant l'une des 3 premieres presentations de FSR 3 (au plus 3 s chacune) : le fil de la fenetre ne relit
// pas ses messages. 13/11 : gel 1 lancement sur 6 a la premiere presentation, fil de la fenetre bloque sur un verrou
// du pilote NVIDIA (crochet de fenetre appele depuis PeekMessage) pendant que la presentation ne revient pas.
[[nodiscard]] bool HoldWindowPump();

// Image du jeu prete a etre presentee : configure la generation et enregistre sa preparation dans `command`
// (le tampon de commandes de la presentation, avant la presentation).
void PrepareFrame(VkCommandBuffer command);

} // namespace Libs::Graphics::Fsr3

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_WINDOW_FSR3FRAMEGEN_H_
