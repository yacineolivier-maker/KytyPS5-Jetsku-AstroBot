#ifndef EMULATOR_SRC_GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_PARCENSUS_H_
#define EMULATOR_SRC_GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_PARCENSUS_H_

// Astro Bot PC (13/11/2026) : KYTY_CP_PAR_CENSUS=1, etape 0 de la traduction parallele des listes de commandes
// (docs\ETUDE-RENDU-NATIF.md). Ne change rien au comportement.
//
// Pour chaque soumission graphique de la file 0, par rang dans l'image (1 = celle qui remet le processeur a zero) :
//  - attente entre l'admission (GuestGpu::Enqueue) et le debut de sa traduction (StreamBegin sur le fil GPU) :
//    le temps pendant lequel elle pourrait etre traduite en parallele de la precedente ;
//  - duree de sa traduction (StreamBegin -> StreamEnd) ;
//  - dessins, dispatchs, et operations qui forceraient un retour en serie dans une traduction parallele
//    (operations « lockstep » et ecritures du processeur de commandes : EndOfPipe, ReleaseMem, EventWrite, WriteData,
//    DmaData, ReferenceClock, LodStats, Flip) ; part des dessins traduits avant la premiere de ces operations.
// Une ligne par rang toutes les 10 s : « ParCensus 10s rang N: ... ».

#include "graphics/guest_gpu/command_processor/cpOps.h"

#include <cstdint>

namespace Libs::Graphics::ParCensus {

[[nodiscard]] bool Enabled();
// GuestGpu::Enqueue (sous m_queue_mutex), soumission graphique de la file 0.
void NoteAdmit(uint64_t sequence, bool frame_start);
// Fil GPU (ResolveSubmission).
void NoteBegin(uint64_t sequence);
void NoteOp(CpSeq::OpKind kind, const void* payload);
void NoteEnd(uint64_t sequence);

} // namespace Libs::Graphics::ParCensus

#endif // EMULATOR_SRC_GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_PARCENSUS_H_
