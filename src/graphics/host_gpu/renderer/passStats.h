#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PASSSTATS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PASSSTATS_H_

// Astro Bot PC (13/11/2026) : KYTY_PASS_STATS=1 mesure le temps du fil GPU par passe de rendu.
// Chaque dessin et chaque dispatch est chronometre (premier niveau seulement) et range sous sa cible :
// la premiere cible couleur (adresse invitee, taille, format), sinon la cible de profondeur. Toutes les
// 10 s, une ligne "PassStats" par cible parmi les plus couteuses : dessins par image, temps par image.
// Desactive (par defaut), rien n'est mesure : une lecture de booleen par operation.

#include <cstdint>

namespace Libs::Graphics::PassStats {

[[nodiscard]] bool Enabled();

enum class Kind : uint8_t { Draw, Dispatch };

// Cible du dessin en cours (fil GPU). colors : nombre de cibles couleur ; depth : cible de profondeur.
void SetTarget(uint64_t address, uint32_t width, uint32_t height, uint32_t format, uint32_t colors,
               bool depth, uint64_t depth_address, uint64_t vs = 0, uint64_t ps = 0);

// Toutes les cibles couleur d'un dessin : une ligne "PassColors" par combinaison nouvelle.
void NoteColors(const uint64_t* address, const uint32_t* format, const uint32_t* width, const uint32_t* height,
                uint32_t count);

class Scope {
public:
	explicit Scope(Kind kind, uint64_t frame);
	~Scope();
	Scope(const Scope&)            = delete;
	Scope& operator=(const Scope&) = delete;

private:
	bool     m_active = false;
	Kind     m_kind   = Kind::Draw;
	uint64_t m_frame  = 0;
	int64_t  m_start  = 0;
};

} // namespace Libs::Graphics::PassStats

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PASSSTATS_H_
