#ifndef EMULATOR_SRC_GRAPHICS_GUEST_GPU_CBRANGES_H_
#define EMULATOR_SRC_GRAPHICS_GUEST_GPU_CBRANGES_H_

// Astro Bot PC (13/11/2026) : KYTY_CB_CONST_STATS=1, diagnostic de l'etude « rendu natif ».
// Les zones memoire des listes de commandes du jeu (objets CommandBuffer de libAgc : [bottom, top)), notees par la
// bibliotheque a leur premiere allocation. Le jeu range dans le haut de ces zones (cursor_down) les constantes de ses
// dessins ; RenderExecutor::RebindBuffers mesure la part et le cout des tampons lies qui s'y trouvent.

#include <atomic>
#include <cstdint>

namespace Libs::Graphics::CbRanges {

[[nodiscard]] bool Enabled();
// Bibliotheque (fils du jeu) : une zone de liste de commandes.
void Note(const void* bottom, const void* top);
// N'importe quel fil : l'adresse est-elle dans une zone notee ?
[[nodiscard]] bool Contains(uint64_t address);
// Fil GPU : un tampon lie (RebindBuffers), sa zone et le temps de sa liaison (ns) ; ligne toutes les 10 s.
void NoteBinding(bool in_command_buffer, uint64_t ns);

} // namespace Libs::Graphics::CbRanges

#endif // EMULATOR_SRC_GRAPHICS_GUEST_GPU_CBRANGES_H_
