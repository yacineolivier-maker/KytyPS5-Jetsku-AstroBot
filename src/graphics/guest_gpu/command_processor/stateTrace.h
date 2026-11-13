#ifndef EMULATOR_SRC_GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_STATETRACE_H_
#define EMULATOR_SRC_GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_STATETRACE_H_

// Astro Bot PC (13/11/2026) : KYTY_STATE_TRACE=1, diagnostic de l'etude « rendu natif » (docs\ETUDE-RENDU-NATIF.md).
//
// Question : chaque liste de commandes (une soumission de la file graphique, une par fil de rendu du jeu) est-elle
// complete a elle seule, ou ses dessins dependent-ils de registres ecrits par une liste precedente de la meme image ?
// Le processeur de commandes est remis a zero a la premiere soumission de chaque image (reset_processor) : une liste ne
// peut heriter que des listes precedentes de la meme image.
//
// Pour chaque registre de contexte et de shader : la soumission qui l'a ecrit en dernier, sa valeur, et s'il a deja
// change de valeur. Au premier dessin de chaque soumission, et a chaque dessin : le nombre de registres ecrits par une
// soumission precedente de l'image (« herites »), dont ceux qui changent de valeur d'une ecriture a l'autre (« herites
// variables » : une vraie dependance, contrairement a une constante posee une fois). Une ligne toutes les 10 s,
// « StateTrace 10s: ... », et les registres herites variables les plus frequents.
//
// Fil du sequenceur seulement (KYTY_CP_SEQ=1) : les ecritures des autres fils sont ignorees.

#include <cstdint>

namespace Libs::Graphics::StateTrace {

[[nodiscard]] bool Enabled();
void BeginSubmission(uint64_t sequence, bool frame_start);
void NoteContext(uint32_t offset, uint32_t count, const uint32_t* values);
void NoteShader(uint32_t offset, uint32_t count, const uint32_t* values);
void NoteDraw();

} // namespace Libs::Graphics::StateTrace

#endif // EMULATOR_SRC_GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_STATETRACE_H_
