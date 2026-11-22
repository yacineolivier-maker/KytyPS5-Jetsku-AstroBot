#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_WINDOW_PRESENTANTIALIASING_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_WINDOW_PRESENTANTIALIASING_H_

#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace Libs::Graphics {

// Astro Bot PC (KYTY_PRESENT_AA) : antialiasing post-traitement de l'image du jeu, juste avant sa
// copie (blit ou PresentFilter) vers la chaine de presentation (Swapchain::RecordPresentCommands).
//   absent, vide ou "0" : rien ne change (aucune ressource creee, aucune commande enregistree) ;
//   "smaa"             : SMAA 1x, preset High (seuil 0,1, 16 pas, diagonales 8, coins 25) ;
//   "smaa_ultra"       : SMAA 1x, preset Ultra (seuil 0,05, 32 pas, diagonales 16).
// KYTY_PRESENT_AA_TIMING=1 : cout GPU moyen des passes (requetes timestamp) ecrit dans la console
// toutes les 10 s.
// SMAA : Jorge Jimenez, Jose I. Echevarria, Tiago Sousa, Diego Gutierrez (2012), licence MIT,
// sources officielles github.com/iryoku/smaa 71c806a8 dans 3rdparty/smaa (LICENSES/SMAA-MIT.txt).
enum class PresentAaMode : uint8_t { Off, SmaaHigh, SmaaUltra };

[[nodiscard]] PresentAaMode PresentAaRequested() noexcept;

class PresentAntiAliasing final {
public:
	PresentAntiAliasing() = default;
	~PresentAntiAliasing();
	KYTY_CLASS_NO_COPY(PresentAntiAliasing);

	// Vrai si l'image `format` / `extent` peut etre lissee (format SDR a virgule fixe, cible de
	// rendu et source de blit sur cette carte).
	[[nodiscard]] bool Supports(GraphicContext& graphics, vk::Format format) const;

	// Enregistre les trois passes sur `source_view` (image du jeu deja en
	// eShaderReadOnlyOptimal, lue par les nuanceurs). Le resultat est dans l'image de sortie
	// (meme taille, meme format), laissee en eColorAttachmentOptimal apres ecriture ; a passer
	// ensuite par TransitOutput. Faux (rien enregistre) si l'image ne peut pas etre lissee.
	[[nodiscard]] bool Record(GraphicContext& graphics, vk::CommandBuffer command,
	                          vk::ImageView source_view, vk::Format format, vk::Extent2D extent);

	// Image lissee de la derniere passe (valide apres un Record reussi).
	[[nodiscard]] vk::Image     OutputImage() const noexcept { return m_targets->output.image; }
	[[nodiscard]] vk::ImageView OutputView() const noexcept { return m_targets->output_view; }
	// Barriere de l'image lissee vers `layout` (eTransferSrcOptimal pour le blit,
	// eShaderReadOnlyOptimal pour PresentFilter).
	void TransitOutput(vk::CommandBuffer command, vk::ImageLayout layout, vk::AccessFlags2 access,
	                   vk::PipelineStageFlags2 stage);

	// Libere tout ; le GPU ne doit plus utiliser ces ressources (apres queue.waitIdle).
	void Release(GraphicContext& graphics);

private:
	struct Targets {
		VulkanImage   edges;
		VulkanImage   weights;
		VulkanImage   output;
		vk::ImageView edges_view   = nullptr;
		vk::ImageView weights_view = nullptr;
		vk::ImageView output_view  = nullptr;
		vk::Pipeline  blend        = nullptr; // passe 2, au format de l'image du jeu
		vk::Format    format       = vk::Format::eUndefined;
		vk::Extent2D  extent {};
		uint32_t      frames_left  = 0; // Record restants avant destruction (une fois retirees)
	};

	void CreateShared(GraphicContext& graphics, vk::CommandBuffer command);
	[[nodiscard]] std::unique_ptr<Targets> CreateTargets(GraphicContext& graphics, vk::Format format,
	                                                     vk::Extent2D extent) const;
	static void DestroyTargets(GraphicContext& graphics, Targets& targets);
	void        CollectRetired(GraphicContext& graphics, bool all);
	vk::Pipeline CreatePipeline(vk::Device device, uint32_t pass, vk::Format format) const;
	void         ReadTimings(GraphicContext& graphics, uint32_t slot);

	PresentAaMode m_mode = PresentAaMode::Off;

	// Ressources independantes de la taille : tables AreaTex / SearchTex, echantillonneur,
	// disposition, pipelines des passes 0 et 1 (formats fixes R8G8 / R8G8B8A8).
	VulkanImage                 m_area;
	VulkanImage                 m_search;
	vk::ImageView               m_area_view          = nullptr;
	vk::ImageView               m_search_view        = nullptr;
	VkBuffer                    m_staging            = VK_NULL_HANDLE;
	VmaAllocation               m_staging_allocation = nullptr;
	vk::Sampler                 m_sampler            = nullptr;
	vk::DescriptorSetLayout     m_descriptors        = nullptr;
	vk::PipelineLayout          m_layout             = nullptr;
	std::array<vk::Pipeline, 2> m_pipelines {};

	// Cibles a la taille et au format de l'image du jeu.
	std::unique_ptr<Targets> m_targets;
	// Cibles remplacees (changement de taille) : detruites apres assez d'images pour que le GPU
	// ait fini (AcquireNextImage attend l'image de meme indice, au plus quelques-unes en vol).
	std::vector<std::unique_ptr<Targets>> m_retired;

	// Chronometrage (KYTY_PRESENT_AA_TIMING=1).
	static constexpr uint32_t kTimingSlots = 8;
	vk::QueryPool             m_queries    = nullptr;
	std::array<bool, kTimingSlots> m_slot_pending {};
	uint32_t                  m_slot          = 0;
	double                    m_sum_ms        = 0.0;
	double                    m_max_ms        = 0.0;
	uint64_t                  m_samples       = 0;
	uint64_t                  m_report_due_ms = 0;
	bool                      m_reported_unsupported = false;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_WINDOW_PRESENTANTIALIASING_H_
