# Voies de traduction : P1 (`KYTY_CP_PAR=2a`) et P2 (`KYTY_CP_PAR=2b`)

Branche `cp-voies` (partie de `mt-liaisons`). Conception : `docs\CONCEPTION-MULTICOEUR.md` (découpage (a),
étapes P1 et P2). Les essais en jeu sont faits par la session principale ; rien de ce qui suit n'a encore été
essayé en jeu.

## P2 « voies simultanées » (`KYTY_CP_PAR=2b`) — état au 15/11/2026 soir

**Ce qui est fait : les étapes 1, 2, 4 et 5. Ce qui n'est pas fait : l'étape 3 (les voies traduisent encore à
tour de rôle, par le jeton de 2a ; il n'y a pas encore de verrou global des caches).** 2b ne peut donc pas encore
apporter de gain de traduction ; il met en place et fait tourner, à tour de rôle, tout ce dont la simultanéité aura
besoin pour ne pas perdre le GPU. `KYTY_CP_PAR=2a` et le mode par défaut sont inchangés (tout le code 2b est
derrière `CpPar::Lanes2b()` / `m_lanes_2b`, faux hors 2b).

| Commit | Étape | Contenu | Build |
|---|---|---|---|
| `87d2964e` | 1 | rejeu par voie, remise à la file dans l'ordre des listes | BUILD_OK |
| `d4e6056d` | 2 | plages de ticks par segment de voie (au lieu d'un tick provisoire), ordre de remise par segment, soumissions et opérations différées d'un segment plus jeune retenues jusqu'à son tour | BUILD_OK |
| `48d04f8c` | 3a | 2b accepté avec `KYTY_EOP_TIMESTAMPS=gpu` (préréglage U59) ; tas de descripteurs : un pool se retire au plus grand tick de ses utilisateurs | BUILD_OK |
| `ae518a8e` | 3b-1 | retraits « au tick du fil appelant » relevés ; dépendance de soumission UploadDma par voie ; compteur « early retires avoided » | BUILD_OK |
| `36a6a7ee` | 3b-2 (partielle) | verrou global FIFO des voies autour de chaque opération (`KYTY_CP_PAR_LOCK=0` le coupe), `IsGpuThread` vrai pour son détenteur ; compteurs prises/attente/temps tenu | BUILD_OK |
| `8aae2d32` | 3b-2 (suite) | état du `CommandProcessor` gardé d'une opération à l'autre, par voie (`LaneCpState`, indice = voie effective) ; `m_resolve_lane` remplacé en 2b | BUILD_OK |
| `ba1b4058` | 4–5 | registre des plages liées/écrites par segment ouvert, attente de dépendance, violations d'ordre (`KYTY_CP_PAR_VERIFY=1`), ligne « CpPar 10s: 2b » complète | BUILD_OK |

### Ce que fait 2b (en plus de 2a)

**Étape 1 — rejeu par voie** (`CommandRecorder::EnableLaneReplay`). Chaque flux de voie (B, C, D) a son propre fil
de rejeu (« CP recorder B/C/D ») avec son exécuteur natif et son état de rejeu ; le fil « CP recorder » ne rejoue
plus que le flux A. Les paquets `Switch` ne chaînent plus les anneaux. Chaque voie enregistre déjà dans les tampons
de son propre pool (2a) : deux fils de rejeu n'écrivent jamais dans le même pool. Une vidange (`Drain`, début de
toute fenêtre directe) n'attend que l'anneau du fil appelant. 2b est refusé (2a continue, message au démarrage)
avec `KYTY_GPU_TIMING`, le profileur d'opérations GPU ou `KYTY_EOP_TIMESTAMPS=gpu` (anneaux à producteur unique).

**Étape 2 — plages de ticks** (`MasterSemaphore::EnableRanges/BeginRange/NextLaneTick/TryReleaseRange`). La
demande était un « tick provisoire résolu à la soumission ». Plutôt que de traduire après coup les ~100 endroits
qui lisent `CurrentTick()` (opérations différées, durées de vie du flux, ticks du pool, mémos indexés par tick,
étiquettes…), chaque segment de voie **réserve dans l'ordre des listes une plage de 2^20 ticks** (son « ordre »
0, 1, 2…) au début du segment. Ses soumissions intermédiaires prennent les ticks de la plage dans l'ordre ; sa
soumission de passation prend **le dernier tick de la plage**. Le sémaphore chronologique saute donc des valeurs
(autorisé par Vulkan : attendre une valeur sautée se termine au signal supérieur suivant). Conséquence : le tick
d'un enregistrement est **connu et définitif dès l'enregistrement**, même si une voie plus ancienne soumet
encore ; tout ce qui est rattaché à `CurrentTick()` se termine avec le tampon qui contient l'enregistrement (ou un
suivant), donc rien n'est à « rattacher » à la soumission. Détails :
- `CurrentTick()` (dans `MasterSemaphore`, donc aussi pour les appels directs) rend le prochain tick de la voie
  du fil appelant (`SetThreadTickLane`, posé par Thread_Gpu au démarrage et par chaque fil au début de son
  segment) ; un fil sans voie (ouvriers DrawPrep, exécuteur des opérations prioritaires, fil principal) lit celui
  de la voie du segment le plus ancien ouvert.
- Un segment qui n'a rien encodé rend sa plage (`TryReleaseRange`) si aucune plage plus récente n'a été réservée
  (toujours vrai à tour de rôle : le segment suivant réutilise ses ticks, comme 2a réutilisait le tick) ; sinon il
  soumet un tampon vide avec le dernier tick de sa plage. Ainsi **toute valeur vue sous le tick courant d'une voie
  finit signalée sans l'aide de cette voie** (sinon une attente de ce tick par la voie elle-même bloquerait).
- Rejeu : le `Switch` porte l'ordre du segment, le `Submit` porte `lane_order` ; un fil de rejeu ne remet un
  tampon à la file (ou au courtier) qu'une fois tous les ordres plus anciens terminés (leur `Switch` rejoué).
  Remplace la porte « tick précédent » de l'étape 1 (les ticks ont des trous).
- Producteur (`CommandScheduler`, sous `m_operation_mutex`) : le « tour » est l'ordre du plus ancien segment non
  terminé. Une soumission intermédiaire d'un segment plus jeune **attend son tour** (`WaitLaneTurn`) ; une
  opération différée (`DeferOperation`, `DeferPriorityOperation`) par le fil d'un segment plus jeune est
  **retenue** avec ce segment et mise en file quand il devient le plus ancien : les deux files restent en ordre de
  ticks (sinon `PriorityDoneLocked` pourrait conclure trop tôt, et les interruptions sortiraient dans le désordre).
- `Finish` et `Shutdown` attendent le dernier tick soumis au lieu de `CurrentTick() - 1`.

**Étapes 4–5 — dépendances entre voies** (`CommandScheduler::NoteLaneBuffer`, appelé au point de liaison des
tampons de stockage `NativeStorageBuffer`, sur le fil de la voie). Avec `KYTY_CP_PAR_VERIFY=1`, un registre garde
par segment ouvert les plages qu'il a liées et celles qu'il a écrites (effacées à la fin du segment). Une liaison
d'une plage écrite par un segment **plus ancien encore ouvert** attend la fin de ce segment (étape 4, compté avec
le temps). Une écriture d'une plage qu'un segment **plus jeune ouvert** a déjà liée est une **violation d'ordre**
(étape 5 : comptée, les 16 premières journalisées « Kyty CpPar 2b: order violation … ») — seule une reprise (P4)
la réparerait. À tour de rôle aucun autre segment n'est jamais ouvert : les deux compteurs doivent rester à 0
(auto-contrôle de la tenue des ordres). Sans `KYTY_CP_PAR_VERIFY` le registre ne coûte rien. Les images et
cibles de rendu ne sont pas encore enregistrées.

**Ligne console** toutes les 10 s, en plus de « CpPar 10s: 2a lanes (N): … » (listes et temps de traduction par
voie) : « CpPar 10s: 2b (N): A: replay x ms busy, turn waits n (y ms); submit turn waits n (y ms), held operations
n; dependencies checked n, waits n (y ms), order violations n; B: … ». Pas encore de « temps sous verrou » : il n'y
a pas de verrou (étape 3).

**Fichiers touchés (P2)** : `command_processor/cpParallel.{h,cpp}` (mode, `LaneStat`, ligne 2b, message de
vérification), `guest_gpu/graphicsRun.cpp` (mise en route, voie du fil pour les ticks), `renderer/commandRecorder.{h,cpp}`
(fils de rejeu par voie, portes d'ordre, `Switch` portant l'ordre), `renderer/commandStream.h` (`lane_order` du
paquet `Submit`), `renderer/masterSemaphore.{h,cpp}` (plages de ticks), `renderer/commandScheduler.{h,cpp}` (ordres,
tour, opérations retenues, registre des dépendances), `renderer/pipeline/descriptors.cpp` (point d'appel du
registre), `renderer/renderContext.cpp` (exécuteurs par voie aussi en 2b).

### Étapes 3b-1 et 3b-2 (toujours à tour de rôle)

**3b-1.** Relevé des objets retirés au tick du fil appelant. Pages du flux (`StreamBuffer::Commit`), plages de
l'anneau `UploadDma`, emplacements d'occlusion et de statistiques LOD : chacun est daté au tick de la voie qui
l'utilise (avec les plages de ticks : le tick du tampon qui le lit) et récupéré dans l'ordre de son anneau (un
tick plus grand en tête ne fait que retarder) : correct aussi en simultané. Retiré par un autre que son dernier
utilisateur : les pools de descripteurs (corrigé en 3a : plus grand tick des utilisateurs). Corrigé en 3b-1 :
la dépendance de soumission d'`UploadDma` (un seul `m_stage_tick`) devient propre à chaque voie. « early
retires avoided » (ligne 2b) compte les cas où l'ancien calcul aurait été trop tôt : 0 attendu à tour de rôle.

Mesuré le 15/11 16 h 14 (Aerial Garden, préréglage U59, EOP `gpu`) : 2b N=2 22,1 i/s, N=3 24,2, vérification
16,7, verrou coupé 20,8 ; aucune perte du GPU, 0 violation, 0 retrait trop tôt ; verrou ~0,7 M prises/10 s, ~40 ms
d'attente, ~4 s tenu.

**3b-2, suite (`8aae2d32`).** `submit_id`, `deferred_eop_flushes`, `packets_since_eop_request`,
`idle_flush_tick/draws`, `slice_draws` : un jeu par voie en 2b (`CommandProcessor::LaneCpState`, indice
`CpPar::EffectiveLane()` = voie du fil, ou voie dont Thread_Gpu sert la requête). `m_resolve_lane` n'est plus lu
en 2b. `g_current_execution` / `g_current_processor` étaient déjà propres à chaque fil. Effet à tour de rôle : les
compteurs de vidange différée et de vidange « GPU au repos » ne se cumulent plus entre voies (chaque voie vide
selon ses propres seuils). **Restent** : `RenderMutex` (propriétaire unique donné avec le jeton : en 3b-3, tout
accès « propriétaire » doit devenir « tient le verrou des voies ») et le compteur d'opérations exécutées du
séquenceur (ne suit que Thread_Gpu ; par voie, comme les fenêtres DrawPrep).

**3b-2 (partielle).** Verrou global FIFO des voies (`CpPar::LaneLockAcquire`), pris autour de chaque opération
d'une voie dans `ResolveSubmission` (pas autour des flips renvoyés à Thread_Gpu) ; `IsGpuThread()` répond aussi
vrai pour son détenteur (requalifie tous les contrôles « fil GPU » d'un coup, sans changement à tour de rôle).
Ligne 2b : « lane lock n takes, x ms waiting, y ms held » (attente ≈ 0 attendue). **Pas faits** : état du
`CommandProcessor` gardé d'une opération à l'autre (`m_submit_id`, `m_deferred_eop_flushes`,
`m_packets_since_eop_request`, `m_idle_flush_*`, `m_resolve_lane`, `m_slice_draws`, `g_current_execution`…) par
voie, `RenderMutex` (propriétaire unique donné avec le jeton), compteur d'opérations exécutées du séquenceur. Les
services renvoyés à Thread_Gpu (`RunOnGpuThread`) s'exécutent pendant que la voie tient le verrou : sans risque
à tour de rôle, à revoir en 3b-3 (Thread_Gpu ne doit pas prendre le verrou pour eux, ou la voie doit le lâcher).

### Passation (15/11 soir) : morceaux suivants, à tour de rôle, non commencés

État du code au commit `8aae2d32`. Pistes relevées pour l'agent suivant (vérifiées dans le code, non écrites) :

- **(a) `RenderMutex`** (`renderer/renderContext.h`) : verrou asymétrique à **un seul propriétaire** (`t_owner`,
  `BecomeOwner`/`ResignOwner` dans les crochets du jeton, `graphicsRun.cpp` ~l. 828/858). À tour de rôle, rien à
  changer (le détenteur du jeton est le propriétaire). Pour la simultanéité, le plus sûr : en mode simultané,
  **personne ne devient propriétaire** (tous les `Lock()` passent par `RequesterLock`, mutex symétrique) ;
  l'asymétrie n'a de sens qu'avec un seul fil de traduction. Ne pas faire `BecomeOwner` par opération.
- **(a) compteur du séquenceur** (`cpSequencer.h`, `NoteExecuted(next_op)`, `m_executed`) : sens « toutes les
  opérations avant `next_op` sont exécutées », numérotation globale des opérations sur tous les anneaux. À tour de
  rôle il reste exact. En simultané : position par voie (prochaine opération non exécutée de chaque anneau), et
  `Executed()` = minimum sur les voies qui ont des opérations émises non exécutées (les seuils d'attente du
  séquenceur, fenêtres et écritures en attente, restent globaux).
- **(b) occlusion** (`renderer/occlusion.cpp`) : lot en cours (`m_batch_count`, requêtes en attente, `m_prepared`,
  emplacements `m_slot_ticks`) partagé ; vidé par le crochet de pré-soumission (`SetPreSubmitHook`, appelé aussi à
  la passation d'une voie). Piste : un objet d'état de lot par voie (indice `EffectiveLane`), le crochet vidant
  celui de la voie qui soumet ; `m_issued`/emplacements restent communs sous le verrou des voies.
- **(c) services renvoyés à Thread_Gpu** (`Resolver2::RunSlice`, boucle des requêtes, `cpParallel.cpp`) :
  `t_serving_lane` y est déjà posé ; ajouter un « détenteur délégué » : pendant la requête, Thread_Gpu compte comme
  détenteur du verrou des voies (`LaneLockHeld()` vrai via `t_serving_lane` quand la voie servie tient le verrou),
  sans prendre le verrou (la voie le tient et attend dans `RunOnGpuThread`).
- **(d) surcouche d'état d'image** : celle de 1b3 (`Image::CpParTouch`, `t_overlay`, fusion dans
  `GuestGpu::CpParMerge`/`CpParInsert`) est à deux contextes (A et la course en avant) ; à généraliser à N voies
  (une surcouche par segment ouvert, correctifs enregistrés à la fin du segment plus ancien). À tour de rôle le
  livre d'états de 1b2 (`CpPar::NoteImageState`, « not no-ops ») doit rester à 0 : c'est le contrôle.

### Morceaux de la passation faits le 15/11 soir (toujours à tour de rôle)

**(c) Services renvoyés à Thread_Gpu sous le verrou des voies.** En réalité (vérifié dans le code) les services
(`ProcessCommands`) et les flips renvoyés par une voie à Thread_Gpu (`RunOnGpuThreadScoped`) n'étaient **pas** sous
le verrou (contrairement à ce que disait la note 3b-2). En 2b (verrou actif), la voie prend désormais le verrou
autour de ces renvois, comme autour d'une opération ; Thread_Gpu sert la requête en **détenteur délégué** :
`LaneLockHeld()` est vrai sur lui quand la voie servie (`t_serving_lane`) est celle qui tient le verrou
(`g_lane_lock_holder`, posé à la prise), et un `LaneLockAcquire` de sa part pendant la requête ne passe pas par la
file FIFO (sinon interblocage dès que les voies se chevaucheront). Les services exécutés par Thread_Gpu pour sa
propre voie A, à la frontière d'opérations, sont aussi pris sous le verrou. Ligne 2b : « bounces under it n
(delegated takes n) ». À tour de rôle rien ne change (le verrou n'est jamais disputé). Reste hors verrou : les
services que Thread_Gpu exécute dans sa boucle principale, hors `ResolveSubmission` (à prendre sous le verrou en
3b-3).

**(b) Occlusion : état de lot par voie.** `OcclusionCounter::LaneBatch` : pool de requêtes, requêtes terminées en
attente de réduction (`OcclusionQueryState`), fenêtre de remise à zéro, lot de vidages (`batch_first_slot`,
`batch_count`), `prepared`, `verified`, diagnostics de portée ; un par flux d'enregistrement en 2b, le seul utilisé
(indice 0) hors 2b. **Indice : `CommandScheduler::ThreadContext()` (le flux dont le tampon natif reçoit les
requêtes), pas `EffectiveLane()`** : quand Thread_Gpu sert un flip de la voie k, il enregistre dans le flux 0 ;
l'état de lot doit suivre le tampon où sont les requêtes, et le crochet de pré-soumission (appelé par le fil qui
soumet, y compris à la passation) vide le lot de ce flux. Le pool de la voie k (k > 0) est créé à son premier
usage. Restent communs (touchés par `Dump`, dans une opération, donc sous le verrou des voies en 2b) : compteur
d'émission `m_issued`, emplacements de publication et leurs ticks, table des préfixes, paires de vidages, porte,
compteur cumulatif sur le GPU (accumulé dans l'ordre de soumission, donc dans l'ordre des listes), tampon de
résultats (temporaire GPU, ordonné par les barrières). Le lot d'une voie doit occuper des emplacements
consécutifs (le nuanceur de lot le suppose) : si une autre voie a pris les emplacements suivants, le lot est
réduit d'abord (« occlusion early batches », 0 attendu à tour de rôle). « dumps outside the lock » compte les
vidages faits hors verrou en 2b (à surveiller : une course une fois les voies simultanées). À tour de rôle :
mêmes requêtes, mêmes lots qu'avant (la passation vide déjà le lot de la voie qui finit), mais chaque voie
utilise son propre pool.

**(a) `RenderMutex` sans propriétaire, position du séquenceur par voie.** En 2b, `BecomeOwner`/`ResignOwner` ne
font plus rien (2b est décidé dans le constructeur de `GuestGpu`, avant le démarrage de Thread_Gpu) : tout
`Lock()` passe par `RequesterLock`, mutex ordinaire ré-entrant. Sans `KYTY_RENDER_LOCK_ASYMMETRIC=1` (absent du
préréglage U59), le propriétaire prenait déjà ce mutex à chaque prise : même coût. Séquenceur
(`cpSequencer.{h,cpp}`) : `NoteExecuted(next_op, lane)` (voie = contexte de l'anneau, passée seulement en 2b)
garde la position de chaque voie ; `NoteLaneRun(lane, start)`, appelé par le fil du séquenceur à chaque
changement d'anneau, note les séries d'opérations consécutives émises dans un même anneau. `Executed()` devient
la première opération (numérotation globale) que sa voie n'a pas encore exécutée : première série non terminée
par sa voie, `max(position de la voie, début de la série)` (la passation demandait « minimum sur les voies en
attente » ; le minimum brut des positions serait faux : une voie inactive garde la position de la fin de sa
série précédente, plus petite que tout ce qui est en cours, ce qui pouvait bloquer le séquenceur). À tour de rôle
c'est exactement l'ancien compteur. Le réveil d'un séquenceur garé calcule ce minimum (sous un petit mutex,
seulement quand il est garé). Auto-contrôle : « seq position below » (ligne 2b) compte les fois où la position
par voie est sous l'ancien compteur (une liste plus récente aurait exécuté avant la fin d'une plus ancienne) :
**0 attendu à tour de rôle**. Les seuils d'attente du séquenceur (écritures en attente, fenêtres, instantanés)
restent globaux.



### Variables pour l'essai en jeu

Base (comme P1) : `KYTY_CP_SEQ=1`, `KYTY_CP_RECORDER=1` (sans `KYTY_CP_RECORDER_VERIFY`),
`KYTY_DRAW_PREP=parallel`, **`KYTY_CP_PAR=2b`**. Ne pas activer `KYTY_GPU_TIMING` ni `KYTY_GPU_OP_PROFILE`
(2b retomberait en 2a). Depuis `48d04f8c`, `KYTY_EOP_TIMESTAMPS=gpu` est accepté (avant : repli en 2a ; avec
`gpu` le début des tampons de voie n'est plus paresseux, donc une soumission par passage de jeton).

Mesuré le 15/11 15 h 30 (session principale, Aerial Garden, `KYTY_EOP_TIMESTAMPS=record`, commits jusqu'à
`ba1b4058`) : 2b N=2 22,3 i/s, N=3 22,2, 2a 23,1 ; aucune perte du GPU ; avec `KYTY_CP_PAR_VERIFY=1` des millions
de dépendances vérifiées, 0 attente, 0 violation, 0 attente de tour de soumission ; rejeu A ~1,1–1,4 s, B ~0,9–1,4 s
par 10 s.

1. **N = 2** (défaut, ou `KYTY_CP_PAR_LANES=2`).
2. **N = 3** : `KYTY_CP_PAR_LANES=3` (option `KYTY_CP_PAR_B_WORKERS=2` comme en P1).
3. Une fois chacun : ajouter `KYTY_CP_PAR_VERIFY=1` (auto-contrôle, un peu plus lent).

Au démarrage : « Kyty CpPar: KYTY_CP_PAR=2b … », « 2a lanes (N): … », « 2b: each lane has its own replay
thread … ». Attendus toutes les 10 s : « not no-ops A 0 B 0 » ; dans la ligne 2b : **submit turn waits 0, held
operations 0, dependency waits 0, order violations 0** (à tour de rôle) ; « turn waits » du rejeu > 0 est normal
(le rejeu d'une voie attend le `Switch` de la précédente). i/s attendus : comme 2a (aucun gain prévu tant que
l'étape 3 manque) ; un léger gain est possible si le rejeu de A était chargé, une légère perte si les fils de
rejeu supplémentaires (attente active de 30 µs, `KYTY_CP_RECORDER_SPIN_US`) gênent un processeur déjà chargé.
Comparer image, i/s (3 tours), stabilité (5 min) à 2a.

### Risques (où une perte du GPU serait possible)

1. **Remise hors ordre** (étapes 1–2) : si la porte d'ordre du rejeu laissait passer le tampon d'un segment plus
   récent avant un plus ancien, des ressources libérées au tick plus ancien seraient encore lues → perte du GPU
   possible. Contrôles : `EXIT_IF` si un `Switch` termine un ordre hors tour (`CommandRecorder::EndOrder`) ou si
   le tour producteur recule (`EndOrderLocked`, `BeginContextSegment`). Interblocage possible au lieu d'une perte
   si un segment ne finissait jamais (le chien de garde nomme « recorder-lane-order »).
2. **Ticks à trous** (étape 2) : code qui supposerait des ticks consécutifs. Relevé : la fusion de soumissions du
   courtier (`CanCoalesceAfter`, tick + 1) fusionne moins (sans danger) ; l'heuristique de vidange « GPU au repos »
   (`KnownGpuTick() + 1 < current`) déclenche moins ; `Finish`/`Shutdown` corrigés ; `FramePool`
   (`CurrentTick() - 1`) est sur l'ordonnanceur du présentateur (sans plages). Si un appel `Wait(t)` sur une valeur
   jamais signalée subsistait, on aurait un **blocage** (pas une perte) — empêché par la règle de restitution des
   plages (ci-dessus). Plages épuisées (> 2^20 soumissions dans un segment) : arrêt `EXIT` explicite.
3. **Rejeu concurrent** (étape 1) : deux fils enregistrent en même temps dans deux tampons de pools distincts
   (permis par Vulkan). Objets partagés touchés au rejeu : `vkUpdateDescriptorSets` (chaque set est écrit par une
   seule voie), pools de requêtes d'occlusion (réinitialisations/requêtes dans des tampons différents, ordonnées
   par l'ordre de soumission), copies hôte (`StreamCopyDefer` déjà suspendu en double flux). Une course ici
   donnerait au pire une perte du GPU au rejeu ; non observée (non essayé).
4. **Fils sans voie** : leur `CurrentTick()` est celui du segment le plus ancien (ou la base de plage suivante
   entre deux segments) : un mémo daté par eux est au pire manqué (repli lent, correct).
5. Hérités de P1 : fil hors jeton qui enregistrerait (`EXIT_IF`), début paresseux (voir plus bas).

### Suite

**Étape 3b-3 (non faite) — exécution simultanée, opération entière sous le verrou.** Seule la préparation 3a est faite
(`48d04f8c`). La simultanéité réelle n'a pas été écrite : elle change la boucle des files de Thread_Gpu, le
partage du `CommandProcessor` (un seul objet, champs gardés d'une opération à l'autre : `SetSubmitId`,
`BufferInit/BufferFlush`, compteur d'opérations exécutées du séquenceur qui ne suit que Thread_Gpu), la propriété
du `RenderMutex` (propriétaire unique donné avec le jeton) et l'identité « fil GPU » empruntée (`g_gpu_thread`,
59 contrôles), et ne pouvait pas être faite sans risque d'interblocage ou de perte du GPU impossible à vérifier
sans essai. Ce qu'il faut, dans l'ordre :
1. Remplacer le jeton par une exécution simultanée : aujourd'hui Thread_Gpu attend dans `RunSlice` la fin de la
   tranche d'une voie. Il faut que la boucle des files (`ProcessSequenced`) confie une soumission de file 0 à sa
   voie **sans attendre** et passe à la suivante (file 0 regardée au-delà de sa tête), la fin étant constatée plus
   tard ; les tranches cédées (`Yielded`) et les soumissions de calcul restent à gérer.
2. Un verrou global (comme `OpLockAcquire` de 1b3, mais pour N voies) autour de tout ce qui touche BufferCache,
   TextureCache, état d'image (reprendre la surcouche `Image::CpParTouch` de 1b3 : état d'entrée et correctifs de
   disposition à la fin du segment, enregistrés en fin de segment de la voie plus ancienne), StreamBuffer, tas de
   descripteurs, occlusion, `RenderMutex` (aujourd'hui « propriétaire » unique, donné avec le jeton) ; les 59
   contrôles « fil GPU » à requalifier ; commencer par une opération entière sous le verrou (sûr, sans gain), puis
   sortir du verrou la préparation, l'émission, les descripteurs et les constantes poussées. (Le retrait des
   pools de descripteurs au tick du fil appelant est corrigé en 3a : plus grand tick des utilisateurs.)
   Autres retraits « au tick du fil appelant » à revoir de la même façon avant la simultanéité (perte du GPU
   possible sinon) : pages du flux (`StreamBuffer`, `streamBuffer.cpp:410`), copies de mise en place et DMA
   (`uploadDma.cpp`), emplacements d'occlusion et de statistiques LOD (`occlusion.cpp`, `lodStats.cpp`), tout
   objet partagé entre voies qui serait libéré par une voie plus ancienne alors qu'une plus jeune l'a utilisé.
3. Ce que le segment plus jeune ne peut pas faire avant son tour (classement 1b4/1b5 à reprendre) : écritures du
   CP visibles par le jeu (étiquettes, `WRITE_DATA` hors étiquettes internes apprises), attentes `WAIT_REG_MEM`
   sur une valeur d'une autre liste, interruptions de fin de canal (à capturer et livrer dans l'ordre comme 1b5),
   flips, lectures du GPU par le CPU, commandes de service (aujourd'hui renvoyées à Thread_Gpu) : la voie attend
   alors son tour (le mécanisme `WaitLaneTurn` existe déjà pour les soumissions).
4. Puis activer le registre des étapes 4–5 en permanence (pas seulement en vérification) et y ajouter images et
   cibles de rendu ; mesurer attentes de dépendance et violations ; les violations demanderont la reprise (P4) ou
   une déclaration préalable des plages écrites (le recensement P0 a mesuré 83–106 plages/image, surtout listes
   1 → 5 et 1 → 6).
5. **P3 (`2c`), lectures sans verrou** : mémos par voie, libération différée, sous-anneaux de flux par voie,
   pour sortir les lectures de cache du verrou (gain visé dans la conception : ~28–32 ms/image à 3 voies).

## P1 « voies découplées, à tour de rôle » (`KYTY_CP_PAR=2a`)

Branche `cp-voies` (partie de `mt-liaisons`). Conception : `docs\CONCEPTION-MULTICOEUR.md` (étape P1 du
découpage (a)). Les essais en jeu sont faits par la session principale.

### État au 15/11/2026 (P1 terminée, à essayer en jeu)

| Commit | Contenu | Build |
|---|---|---|
| `7e1f5472` | 2a à deux voies (soumission à chaque passage du jeton) | essayé en jeu le 15/11 |
| `2232c7ed` | **début paresseux** du tampon natif d'une voie, plus de soumission vide | BUILD_OK |
| `ad390590` | **N voies** (`KYTY_CP_PAR_LANES=2|3|4`), toujours à tour de rôle | BUILD_OK |
| `bd021e65` | compteurs par voie dans la ligne « CpPar 10s: 2a lanes » | BUILD_OK |

Mesure du 15/11 avec `7e1f5472` (Aerial Garden) : aucune erreur, image correcte sans correctif, **22,0 i/s avec
2a contre 18,1 sans** ; « hand-off submits A 1641 (empty 133), B 1641 (empty 6) » par 10 s ; Thread_Gpu 5,5 s et
CP resolver #2 4,1 s de traduction par 10 s, à tour de rôle. Les commits suivants n'ont **pas encore été essayés**.

### Ce que fait 2a

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

### Fichiers touchés

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

### Variables pour l'essai en jeu

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

### Risques connus

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

### Suite prévue en P1 pour P2 (historique ; état réel : section P2 ci-dessus)

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
