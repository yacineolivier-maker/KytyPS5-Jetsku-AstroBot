# Étape P1 « voies découplées, à tour de rôle » (`KYTY_CP_PAR=2a`)

Branche `cp-voies` (partie de `mt-liaisons`). Conception : `docs\CONCEPTION-MULTICOEUR.md` (étape P1 du
découpage (a)). Les essais en jeu sont faits par la session principale.

## État au 15/11/2026 (P1 terminée, à essayer en jeu)

| Commit | Contenu | Build |
|---|---|---|
| `7e1f5472` | 2a à deux voies (soumission à chaque passage du jeton) | essayé en jeu le 15/11 |
| `2232c7ed` | **début paresseux** du tampon natif d'une voie, plus de soumission vide | BUILD_OK |
| `ad390590` | **N voies** (`KYTY_CP_PAR_LANES=2|3|4`), toujours à tour de rôle | BUILD_OK |
| `bd021e65` | compteurs par voie dans la ligne « CpPar 10s: 2a lanes » | BUILD_OK |

Mesure du 15/11 avec `7e1f5472` (Aerial Garden) : aucune erreur, image correcte sans correctif, **22,0 i/s avec
2a contre 18,1 sans** ; « hand-off submits A 1641 (empty 133), B 1641 (empty 6) » par 10 s ; Thread_Gpu 5,5 s et
CP resolver #2 4,1 s de traduction par 10 s, à tour de rôle. Les commits suivants n'ont **pas encore été essayés**.

## Ce que fait 2a

`KYTY_CP_PAR=2a` = 1b2 (jeton, exécution strictement à tour de rôle) **plus**, pour chaque voie :

| Élément | Avant (1b2) | 2a |
|---|---|---|
| Enveloppe `CommandBuffer` (ombres, lot de barrières, instance de rendu) | une par contexte | une par voie |
| `DynamicState` | une par contexte | une par voie |
| Anneau `CommandStream` + encodeur du recorder | un par contexte | un par voie |
| Tampon natif `VkCommandBuffer` | **un seul, partagé** | **un par voie, pris dans le pool propre de la voie** |
| `RenderExecutor` | un seul | **un par voie** (`RenderContext::GetRenderExecutor()` selon le flux du fil appelant) |
| Anneau d'opérations, anneau d'instantanés, moteur DrawPrep | un par contexte (A, B) | un par voie (A..D) |
| Rejeu | un fil recorder, paquets `Switch` « vers l'autre anneau » | un fil recorder, `Switch` **nommant l'anneau cible** |

Répartition : la liste de rang r d'une image (r = 1 pour la soumission `reset_processor`) va au contexte
`(r − 1) mod N`. N = 2 : rangs impairs → A (Thread_Gpu), pairs → B (CP resolver #2), comme avant. N = 3 :
1, 4, 7… → A ; 2, 5… → B ; 3, 6… → C. Une voie k ≥ 1 est servie par le fil « CP resolver #k+1 ».

Soumission dans l'ordre des listes (« fusion ») : à chaque passage du jeton, la voie qui le rend termine son
segment par un `Submit` de **son** tampon natif (le tick est attribué à ce moment), puis le paquet `Switch` qui
nomme la voie preneuse (`CpPar::HandoffTarget()` : Thread_Gpu donne à la voie de la tranche, une voie rend
toujours à Thread_Gpu). Il n'y a jamais plus d'un tampon natif ouvert : `CommandScheduler::Submit` s'arrête
(`EXIT_IF`) si une autre voie nomme encore un tampon.

**Début paresseux** (`2232c7ed`) : en début de segment la voie prend un tampon de son pool, mais le paquet `Begin`
reste en attente dans son encodeur (`CommandStream::Encoder::DeferBegin`) ; il est encodé juste avant le premier
autre paquet, ou par toute vidange du recorder (`CommandRecorder::Drain`, par laquelle commence toute fenêtre
directe `Handle()`/`StateHandle()` ; la vidange « inactive » ne peut donc plus ouvrir une fenêtre sur un tampon non
commencé). En fin de segment, après ce que `Submit` encoderait d'abord (crochet de fin de tampon des requêtes
d'occlusion, fin de rendu, lot de barrières), si le `Begin` est encore en attente : il est annulé, le tampon rendu
au pool (`CommandPool::Return`), rien n'est soumis, aucun tick n'est attribué (le travail différé au tick courant
attend la soumission suivante : jamais plus tôt). Compté « skipped ». Un `Switch` avec un `Begin` en attente est
une erreur (`EXIT_IF`).

Sans `KYTY_CP_PAR=2a` rien ne change : `DeferBegin` n'est appelé que par `BeginLaneCommand` (voies), les tableaux
par voie ne servent qu'aux indices 0/1 hors 2a, le `Switch` sans cible garde la bascule de 1b2/1b3, la répartition
reste `(r − 1) mod 2`.

## Fichiers touchés

- `command_processor/cpParallel.h/.cpp` : `MaxLanes`, `RequestedLanes()` (`KYTY_CP_PAR_LANES`), `ContextCount()`,
  `ThreadLane()`, `HandoffTarget()`, un `Resolver2` par voie (`RunSlice(lane, …)`), statistiques par voie,
  ligne « CpPar 10s: 2a lanes ».
- `command_processor/commandProcessor.h`, `cpSequencer.h`, `guest_gpu/graphicsRun.cpp` : anneaux d'opérations
  (`OpsOf`), anneaux d'instantanés (`AddSnapshotRings`), moteurs DrawPrep (`LanePrep`), listes en attente par
  contexte, répartition par rang, crochets du jeton (`ThreadLane`), tranche sur la voie de la soumission.
- `renderer/commandStream.h/.cpp` : `DeferBegin` / `FlushPendingBegin` / `CancelPendingBegin`.
- `renderer/commandRecorder.h/.cpp` : flux 1..3 (`EnableStreams`, `StreamEncoder`), `EndStreamSegment(target)`
  (cible dans l'octet haut du numéro du `Switch`), vidange qui encode le `Begin` en attente.
- `renderer/commandScheduler.h/.cpp` : `EnableLanes(count)`, `LaneCmd`, `LanePool`, `CommandPool::Return`,
  fin de segment paresseuse.
- `renderer/renderContext.h/.cpp` : un `RenderExecutor` par voie après A ; `drawPrep.h/.cpp` : `Engine(…, lane)`.

## Variables pour l'essai en jeu

Base (comme le 15/11) : `KYTY_CP_SEQ=1`, `KYTY_CP_RECORDER=1` (sans `KYTY_CP_RECORDER_VERIFY`),
`KYTY_DRAW_PREP=parallel`, `KYTY_CP_PAR=2a`.

1. **N = 2** (défaut, ou `KYTY_CP_PAR_LANES=2`) : doit refaire la mesure du 15/11, avec moins de soumissions.
2. **N = 3** : `KYTY_CP_PAR_LANES=3`. Optionnel pour limiter les fils DrawPrep des voies B..D (3 chacun par
   défaut, 1 « chaud ») : `KYTY_CP_PAR_B_WORKERS=2`.
3. (N = 4 possible, `KYTY_CP_PAR_LANES=4`.)

À lire dans la console : au démarrage « Kyty CpPar: 2a lanes (N): … (lazy begin) » et « CP resolver #2 [and #3]
started » ; toutes les 10 s « image fix-ups … not no-ops A 0 B 0 » (doit rester 0) et
« CpPar 10s: 2a lanes (N): A: n lists, x ms translating, submits n (empty n, skipped n), prep committed n fell
back n; B: … ; C: … ». Attendus : « skipped » ≈ l'ancien « empty » (≈ 140/10 s à N = 2), « empty » ≈ 0 ; à N = 3
le temps de traduction se répartit sur trois fils mais **le total ne baisse pas** (toujours à tour de rôle : P1 ne
fait que préparer P2). Comparer image, i/s (3 tours), stabilité (5 min) entre N = 2, N = 3 et sans `KYTY_CP_PAR`.

## Risques connus

1. **Fil hors jeton qui enregistre dans le planificateur invité** : son flux est 0 → enveloppe A, sans tampon
   pendant le segment d'une autre voie → arrêt `EXIT_IF`. Non observé en 1b2 ni en 2a (15/11), non prouvé absent.
2. **Début paresseux** : le `Begin` différé garde le tick et l'heure du moment où la voie a pris le tampon ; un
   chemin qui enregistrerait nativement sans passer par `Drain` (aucun trouvé : toute fenêtre directe passe par
   `OpenDirectWindow` → `Drain`) écrirait dans un tampon non commencé. Avec `KYTY_EOP_TIMESTAMPS=gpu` le début
   peut ne plus être paresseux (`EopTimestampRing::BeginCommand` encode en tête de tampon les remises à zéro de
   requêtes) : correct, sans gain. Non vérifié en jeu.
3. **N voies** : mémos des exécuteurs froids et dédoublés N fois (corrects, moins de réussites) ; N − 1 moteurs
   DrawPrep en plus (fils supplémentaires, dont un chaud par voie) sur un processeur déjà chargé ; anneaux du
   recorder de 16 Mo et anneaux d'opérations par voie (mémoire). La dépendance « liste 1 → 5/6 » reste exacte
   puisque l'exécution est toujours à tour de rôle.
4. Si Thread_Gpu valide des dessins d'une fenêtre B..D (requête `RunOnGpuThread`), il les exécute avec l'exécuteur
   A et des étiquettes du mémo de la voie : repli sur le chemin lent (correct).
5. 1b3/1b4/1b5 restent à deux contextes (`KYTY_CP_PAR_LANES` n'est lu qu'en 2a).

## Suite : P2 (`=2b`, voies simultanées)

Ce qui manque pour faire tourner les voies en même temps (sous un verrou global des caches au début) :

- **Tick provisoire** : aujourd'hui le tick d'un tampon est `CurrentTick()` à son début, juste parce qu'aucune
  autre voie ne soumet avant lui. En simultané, une voie plus jeune enregistre sans connaître son tick : tick
  provisoire (numéro de voie + numéro de segment) résolu à la fusion, quand la voie plus ancienne a soumis.
- **Ressources différées rattachées au tick de fusion** : `DeferOperation`/`DeferPriorityOperation`, durées de vie
  des tampons de flux (`StreamBuffer`, copies de mise en place), `CommandPool::m_ticks`, mémos validés par tick
  (liaisons, `DrawRun`, réemploi de sets), étiquettes différées : tout ce qui lit `CurrentTick()` pendant
  l'enregistrement d'une voie plus jeune doit lire son tick provisoire, traduit à la fusion.
- **Soumissions intermédiaires d'une voie plus jeune** (vidanges « idle GPU », attentes d'étiquettes, lectures
  CPU) : interdites tant qu'une voie plus ancienne n'a pas soumis, ou « promotion » (la voie attend son tour).
- **Rejeu** : un fil recorder par voie et un courtier qui soumet les tampons dans l'ordre des listes, au lieu du
  `Switch` sérialisé ; le `Switch` nommant sa cible est déjà le format de cet ordre.
- **Dépendances entre listes** : le recensement P0 (`KYTY_CP_PAR_CENSUS2`) a mesuré à Aerial Garden 83–106
  plages par image écrites par le GPU dans une liste puis liées par une liste suivante, surtout listes 1→5 et
  1→6. Avec N = 3, 1 et 4 sont sur A, 2 et 5 sur B, 3 et 6 sur C : 1→5 et 1→6 franchissent des voies. Il faut,
  avant de lier une plage dans une voie plus jeune, savoir si une voie plus ancienne non fusionnée l'écrit
  (registre des plages écrites par segment ouvert) : attendre la fusion de cette voie (simple, coûte du
  parallélisme) ou reprendre la voie plus jeune (P4). Les états d'image (surcouche par contexte de 1b3) et les
  correctifs de disposition à la fusion sont à reprendre de 1b3.
- **Verrou** : d'abord un verrou unique autour des caches (BufferCache, TextureCache, pages, tas de descripteurs,
  occlusion) comme 1b3 (`OpLockAcquire`), les 59 contrôles « fil GPU » requalifiés ; mesurer l'attente avant P3
  (lectures sans verrou).
