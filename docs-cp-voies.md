# Étape P1 « voies découplées, à tour de rôle » (`KYTY_CP_PAR=2a`)

Branche `cp-voies` (partie de `mt-liaisons`). Conception : `docs\CONCEPTION-MULTICOEUR.md` (étape P1 du
découpage (a)). Rien n'a été essayé en jeu : les essais sont faits par la session principale.

## Ce que fait 2a

`KYTY_CP_PAR=2a` = 1b2 (jeton, exécution strictement à tour de rôle, deux contextes A = Thread_Gpu et
B = « CP resolver #2 », listes de rang impair → A, pair → B) **plus**, pour chaque voie :

| Élément | Avant (1b2) | 2a |
|---|---|---|
| Enveloppe `CommandBuffer` (ombres, lot de barrières, instance de rendu) | une par contexte | une par voie (inchangé) |
| `DynamicState` | une par contexte | une par voie (inchangé) |
| Anneau `CommandStream` + encodeur | un par contexte | un par voie (inchangé) |
| Tampon natif `VkCommandBuffer` | **un seul, partagé** | **un par voie, pris dans le pool propre de la voie** (`CommandScheduler::EnableLanes`, `m_command_pool_b`) |
| `RenderExecutor` | un seul | **un par voie** (`RenderContext::m_render_executor_b`, choisi par `GetRenderExecutor()` selon le flux du fil appelant) |
| Mémo de liaisons de textures lu par les préparateurs DrawPrep B | celui de l'unique exécuteur | celui de l'exécuteur B |
| Rejeu | un fil recorder, paquets `Switch` | inchangé (un fil recorder suit les `Switch`) |

Soumission dans l'ordre des listes (« fusion ») : à chaque passage du jeton, la voie qui le rend termine son
segment par un `Submit` de **son** tampon natif (le tick est attribué à ce moment, comme pour toute soumission),
puis le paquet `Switch` ; la voie qui le prend commence un nouveau tampon natif de **son** pool. Il n'y a donc
jamais plus d'un tampon natif ouvert : `CommandScheduler::Submit` s'arrête (`EXIT_IF`) si l'autre voie nomme encore
un tampon, ce qui garantit qu'aucune voie plus jeune ne soumet avant qu'une voie plus ancienne ait soumis.

Coût attendu : un `Submit` de plus par passage du jeton (début et fin de chaque tranche B, et chaque renvoi
`RunOnGpuThread` vers Thread_Gpu : services, flips). Un segment vide produit une soumission vide : compté.

Sans `KYTY_CP_PAR=2a` rien ne change : toutes les branches nouvelles sont gardées par `m_lanes` (planificateur),
`m_render_executor_b != nullptr` (contexte de rendu, nul hors 2a) ou `CpPar::Lanes()`.

## Fichiers touchés

- `src/graphics/guest_gpu/command_processor/cpParallel.h/.cpp` : mode `Lanes2a`, `g_lanes`/`Lanes()`,
  `NoteLaneHandoff`, ligne console « CpPar 10s: 2a lanes: hand-off submits A n (empty n), B n (empty n) ».
- `src/graphics/host_gpu/renderer/commandScheduler.h/.cpp` : `EnableLanes`, `BeginLaneCommand`, branches
  « voies » de `BeginContextSegment` / `EndContextSegment` / `BeginCommand` / `Begin` / `Finish` / `Submit`.
- `src/graphics/host_gpu/renderer/renderContext.h/.cpp` : second `RenderExecutor` (créé seulement si
  `KYTY_CP_PAR=2a` et `KYTY_CP_RECORDER=1` sans vérification), `GetRenderExecutor()` par voie,
  `GetRenderExecutor(lane)`, `HasLaneExecutor()`.
- `src/graphics/host_gpu/renderer/drawPrep/drawPrep.cpp` : `MakeBindingPlanContext(renderer, executor)`.
- `src/graphics/guest_gpu/graphicsRun.cpp` : activation au démarrage (après `EnableContextB`), compte des
  soumissions de passage dans le crochet `give`.

## Variables pour l'essai en jeu

Configuration de 1b2 plus le mode : `KYTY_CP_SEQ=1`, `KYTY_CP_RECORDER=1` (sans `KYTY_CP_RECORDER_VERIFY`),
`KYTY_DRAW_PREP=parallel` (fenêtres DrawPrep par contexte), `KYTY_CP_PAR=2a`.
À vérifier dans la console : « Kyty CpPar: 2a lanes: each lane has its own command pool… » au démarrage, puis
toutes les 10 s « CpPar 10s: streams … image fix-ups … not no-ops A 0 B 0 » (doit rester 0) et
« CpPar 10s: 2a lanes: hand-off submits … » (nombre de soumissions ajoutées, part de vides).
Comparer image, i/s (3 tours) et stabilité (5 min) avec `KYTY_CP_PAR=1b2` et sans `KYTY_CP_PAR`.

## Risques connus

1. **Fil hors jeton qui enregistre dans le planificateur invité** (fil du jeu sur faute de page, capture…) :
   son flux est 0 → enveloppe A, qui n'a pas de tampon pendant un segment B → arrêt `EXIT_IF` dans `Current()`.
   En 1b2 un tel enregistrement serait déjà allé dans l'anneau A au mauvais endroit ; 1b2 a tourné sans
   plantage, ce cas paraît absent, mais rien ne le prouve.
2. **Soumissions supplémentaires** (dont vides) : coût CPU (fil recorder, pilote) et petites bulles GPU.
3. **Exécuteurs séparés** : mémos (textures, échantillonneurs, descriptions, DrawRun, réemploi de sets, ensembles
   « xframe ») froids et dédoublés ; corrects (validés par versions de page / tick / tampon) mais moins de
   réussites au début. Les sessions d'apprentissage natives (`KYTY_NATIVE_MODE` s0/s1) sont aussi par exécuteur.
4. Si Thread_Gpu valide des dessins de la fenêtre B (requête `RunOnGpuThread`), il les exécute avec l'exécuteur A
   et des étiquettes de mémo prises dans le mémo B : repli sur le chemin lent (correct, plus lent).

## Suite pour finir P1 puis préparer P2

- **Début paresseux** du tampon natif d'une voie (au premier paquet encodé, pas au début du segment) pour
  supprimer les soumissions vides : l'encodeur émettrait le paquet `Begin` en attente avant tout autre paquet ;
  en fin de segment sans paquet, rendre l'emplacement du pool et ne rien soumettre.
- **3–4 voies** : aujourd'hui deux contextes fixes (rang pair/impair, un seul resolver #2). Généraliser
  `m_command_b`/`m_dynamic_state_b`/`m_command_pool_b`/`m_render_executor_b`/anneau B en tableaux indexés par
  voie, rang modulo N, N−1 fils resolver, `Switch` vers une voie désignée (le paquet doit porter la voie cible).
- **P2 (`=2b`, simultanéité sous verrou du cache)** : le tick ne peut plus être `CurrentTick()` pendant
  l'enregistrement d'une voie plus jeune. Il faudra : tampon d'une voie plus jeune enregistré sans tick (ou avec
  un tick provisoire), tick attribué à la fusion ; ressources différées (`DeferOperation`, durées de vie des
  tampons de flux, `CommandPool::m_ticks`, mémos tick) rattachées au tick de fusion ; interdiction des
  soumissions intermédiaires d'une voie plus jeune (flush idle, attentes) ou promotion ; rejeu par voie (un fil
  recorder par voie, soumission ordonnée par un courtier) au lieu du `Switch` sérialisé.
