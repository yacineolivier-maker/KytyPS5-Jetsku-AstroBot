# Voies de traduction : P1 (`KYTY_CP_PAR=2a`) et P2 (`KYTY_CP_PAR=2b`)

Branche `cp-voies` (partie de `mt-liaisons`). Conception : `docs\CONCEPTION-MULTICOEUR.md` (découpage (a),
étapes P1 et P2). Les essais en jeu sont faits par la session principale ; rien de ce qui suit n'a encore été
essayé en jeu.

## P2 « voies simultanées » (`KYTY_CP_PAR=2b`) — état au 15/11/2026 soir (morceaux (a)–(d) et obstacles 1–3 faits, à tour de rôle ; obstacles 4–5 faits derrière `KYTY_CP_PAR_CONCURRENT=1`, non essayés en jeu)

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
| `a2c83dea` | (c) | services et flips renvoyés à Thread_Gpu sous le verrou des voies, Thread_Gpu détenteur délégué | BUILD_OK |
| `f2e94af8` | (b) | occlusion : état de lot par flux (pool, requêtes, fenêtre, lot) ; émission et emplacements communs | BUILD_OK |
| `492ed57b` | (a) | `RenderMutex` sans propriétaire en 2b ; position des opérations exécutées du séquenceur par voie | BUILD_OK |
| `c00a785c` | (d) | surcouche d'état d'image par voie (segments plus jeunes), fusion à la fin du plus ancien | BUILD_OK |
| `3be1ee4d` | obstacle 1 | champs du `CommandProcessor` modifiés entre les opérations, par voie (`LaneCpState`) ; contrôle « retries pending at list start » | BUILD_OK |
| `9e7f0ce0` | obstacle 2 | ordre réservé par Thread_Gpu à la remise de la liste ; segment de A fermé à la fin de chaque liste A (`KYTY_CP_PAR_RESERVE=0` coupe) | BUILD_OK |
| `25a85844` | obstacle 3 | une voie garde son segment jusqu'à la fin de sa liste ; Thread_Gpu enregistre entre-temps dans ce segment (`KYTY_CP_PAR_KEEP=0` coupe) | BUILD_OK |
| `ba1b4058` | 4–5 | registre des plages liées/écrites par segment ouvert, attente de dépendance, violations d'ordre (`KYTY_CP_PAR_VERIFY=1`), ligne « CpPar 10s: 2b » complète | BUILD_OK |
| `69c35cc1` | obstacle 4 | `KYTY_CP_PAR_CONCURRENT=1` : listes confiées sans attendre, recueillies dans l'ordre | BUILD_OK |
| `c89637e1` | obstacle 5 | attente du tour avant toute opération visible ; retraits partagés tenus ; `Finish` par voie | BUILD_OK |
| `68d86418` | compteurs | ligne « CpPar 10s: 2b concurrent », contrôles d'ordre | BUILD_OK |
| `9e05e27b` | blocage 15/11 | délai de sécurité de la barrière de trame rétabli ; tranche inachevée soumise | BUILD_OK |
| `1ff2bee9` | chien de garde | état de chaque voie après 20 s sans image | BUILD_OK |
| `63f97b56` | blocage 19 h 23 | liste reprise placée avant les listes plus jeunes dehors | BUILD_OK |
| `390956fe` | arrêt 19 h 30 | une voie commence son segment à côté d'un segment gardé ; messages d'arrêt détaillés | BUILD_OK |

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

### Passation (15/11 soir) : morceaux (a)–(d) — faits (voir la section suivante)

Pistes laissées au commit `8aae2d32` par l'agent précédent ; toutes codées le 15/11 soir, dans l'ordre (c), (b), (a),
(d), avec deux écarts justifiés plus bas (occlusion indexée par flux d'enregistrement et non par voie effective ;
position du séquenceur par séries d'opérations et non minimum brut). Texte d'origine :

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

**(d) Surcouche d'état d'image à N voies.** La surcouche de 1b3 (`Image::CpParTouch`, entrée à la première
touche, état propre échangé pour la durée d'une opération, rendu à la fin de l'opération) existe maintenant en
un exemplaire par voie (`GetCpParOverlay(lane)`, `CpPar::t_overlay_lane` ; 1b3 n'utilise que l'exemplaire 0,
inchangé). Active en 2b avec le verrou des voies (`CpPar::g_lane_overlay`, `KYTY_CP_PAR_OVERLAY=0` la coupe).
Règle, généralisation exacte de 1b3 : **le segment ouvert le plus ancien travaille sur l'état global** (comme le
contexte le plus ancien de 1b3) ; une opération d'un segment plus jeune (`!LaneSegmentOldest`) travaille sur la
surcouche de sa voie, sous le verrou. À la fin du segment le plus ancien (crochet de passation, avant la
soumission de passation, verrou pris), la surcouche de son **successeur** (ordre + 1, `LaneSuccessor`) est
fusionnée : correctifs (état global → exigence d'entrée du successeur) enregistrés à la fin du tampon du plus
ancien, que la file exécute juste avant celui du successeur ; puis les états du successeur deviennent globaux.
Les correctifs du segment s ne peuvent donc aller que dans le tampon de s−1 : **un segment plus jeune ne doit
pas finir avant d'être le plus ancien** (en simultané, sa fin devra attendre son tour) ; compté « left » (avec
fusion sans correctifs, comptée « impossible »). À tour de rôle aucun segment plus jeune n'est jamais ouvert :
la surcouche n'est jamais utilisée (overlay ops 0, images 0, fix-ups 0, left 0 attendus) et le livre de 1b2
(« not no-ops ») reste à 0 par construction ; ce morceau prépare la simultanéité mais n'est **pas exercé** à
tour de rôle. Correction annexe : Thread_Gpu reste « au service » de la voie (`t_serving_lane`) jusqu'après le
crochet de passation qui suit une requête (sinon ce crochet, qui prend le verrou, attendrait la voie qui le
tient : interblocage). Hors opérations, les services et flips d'un segment plus jeune toucheraient l'état global
(à faire attendre leur tour en 3b-3).

**Essai en jeu (session principale, 15/11 17 h 17, (a)–(d)).** Aerial Garden, préréglage U59 : N=2 20,8 i/s,
N=3 22,3, vérification 22,4 ; aucune perte du GPU ni blocage ; order violations, occlusion early batches, seq
position below et surcouche à 0 partout. « delegated takes » 0 : **normal si « bounces under it » est aussi à 0**
(une prise déléguée n'a lieu que dans le crochet de passation qui suit un renvoi fait sous le verrou ; une prise
non déléguée à cet endroit aurait bloqué Thread_Gpu sur le verrou tenu par la voie : l'absence de blocage montre
que, s'il y a eu des renvois, la délégation a joué). Les voies B/C n'ont vraisemblablement ni flip ni service.

**Suite 1 et 2 (même commit).** (1) `CommandScheduler::WaitSegmentTurn()` (public, attend que le segment de la
voie du fil soit le plus ancien ; compté comme « submit turn waits ») : avant la passation (crochet `give`, hors
verrou ; sous verrou, c'est-à-dire pendant un renvoi, l'attente a été faite avant de le prendre) et avant tout
flip ou service renvoyé par une voie à Thread_Gpu (avant la prise du verrou : attendre en le tenant bloquerait le
segment plus ancien). (2) `GuestGpu::ProcessCommands` prend le verrou des voies lui-même (`CpPar::LaneLockScope`,
ré-entrant, délégué) : tous les appelants (boucle principale de Thread_Gpu, attentes DrawPrep, frontières
d'opérations) ; le vidage d'occlusion de l'exécuteur (`counter.Dump`, aussi atteint hors opération en mode direct)
prend le verrou avant le `RenderMutex` (même ordre qu'une opération). À tour de rôle : aucune attente (le segment
qui tourne est toujours le plus ancien), « dumps outside the lock » doit tomber à 0.





### Arrêt au démarrage (essai du 15/11, 19 h 30) : segment gardé pris pour le sien

**Version** : la chaîne « 33f6d59-dirty » de cet essai était bien le contenu de `63f97b56` (compilé à
19 h 29 min 07 s avant le commit de 19 h 29 min 43 s, d'où « dirty » sur le commit précédent). Désormais
le commit précède la compilation : la build actuelle affiche `390956fe`.

**Arrêt** : « condition (stream != 0 || t_cp_par_adopted != UINT32_MAX) » dans le crochet `take`.
**Cause** : Thread_Gpu confie une liste à la voie X pendant que la voie Y, plus ancienne, est encore
dehors ; X prend son crochet `take` plus tard, sur son propre fil (après le verrou des voies). Si entre-
temps Y a fini sa tranche inachevée et gardé son segment, X voyait `g_cp_par_kept_lane = Y` et entrait
dans la branche « adoption », réservée à Thread_Gpu. À tour de rôle c'était impossible (une seule voie à
la fois). Le réordonnancement de `63f97b56` n'y est pour rien ; c'est la remise sans attente elle-même.
**Correctif** (`390956fe`) : en mode simultané, seule la voie gardée reprend son segment et seul
Thread_Gpu adopte ; toute autre voie commence son propre segment (ordre réservé à la remise, plus jeune
que le segment gardé), exactement comme les listes déjà dehors pendant une tête gardée. Compteur
« began beside a kept segment » (> 0 possible). Les arrêts sur les invariants des segments gardés
(adoption, garde double, segment adopté rendu à une autre voie) écrivent maintenant « Kyty CpPar 2b:
… calling thread lane, lane, kept lane, adopted by Thread_Gpu, concurrent, lists out (last
published) … ».

**Cas voisins vérifiés** : garde pour un renvoi (flip, services) — la voie tient le verrou des voies
pendant tout le renvoi et reprend son segment avant de le lâcher, donc une autre voie (dont le crochet
prend ce verrou) ne voit jamais ce segment gardé ; Thread_Gpu sert le renvoi en détenteur délégué et
adopte ce segment (voulu). Garde double : seule la plus ancienne garde (après son tour), donc jamais deux.
Thread_Gpu « chez lui » avec un segment adopté ne reçoit pas de renvoi (seule la plus ancienne en
envoie, et c'est la tête gardée, qui n'est pas dehors). Une voie commencée à côté d'un segment gardé
reste plus jeune : ses opérations visibles, sa fin, ses renvois attendent la fin de la tête gardée.

### Deuxième blocage (essai du 15/11, 19 h 23, après 4 images) : tour jamais rendu

**Rapport du chien de garde** (`console-20261115-192354.txt`, « STALL ») : Thread_Gpu « waiting for a
lane », listes dehors `[list 32 lane C] [list 30 lane B ended]` ; B inactive ; C en attente du tour
(opération d'étiquette, son ordre 0x1a = 26, tour 0x19 = 25) ; segment gardé B ; ordres B 25 (le plus
ancien), C 26 ; verrou libre.

**Cause (prouvée par l'ordre affiché des listes dehors).** La liste 30 (B) était revenue inachevée
(attente de flip) : recueillie, remise en tête de file 0, segment gardé. Pendant ce temps la 32 (C) était
déjà dehors. `CpParTryDispatch` a reconfié la 30 et l'a **ajoutée à la fin** des listes dehors
(`push_back`), donc derrière la 32. Les fins ne sont recueillies que par l'avant (`CpParConcurrentStep`) :
la 30, revenue de nouveau inachevée, n'a jamais été recueillie, n'a jamais été remise en file ni
reconfiée, son segment (ordre 25) n'a jamais été repris ni terminé, et C (ordre 26) attendait son tour
pour toujours. Thread_Gpu restait en « wait » (`CpParCanGoHome` faux : la voie du segment gardé est
encore parmi les listes dehors). Corrigé (`63f97b56`) : les listes dehors sont **insérées dans l'ordre
des listes** ; seule la liste reprise (la plus ancienne) peut passer devant des listes dehors (sinon
`EXIT`). Compteurs « resumed before younger » (> 0 attendu dans le mode : c'est le cas qui bloquait) et
contrôle « out-of-order ends » (une liste finie inachevée derrière une liste encore dehors ; **0
attendu**, journal « ended unfinished behind list … » la première fois).

**« Lane D » avec 3 voies** : le chien de garde affichait les 5 emplacements ; celui de D n'est jamais
écrit avec `KYTY_CP_PAR_LANES=3` (« running (no wait noted) » = emplacement vide). Il n'y a pas de 4e
voie ; les emplacements des voies absentes ne sont plus affichés.

**Autres cas « fin de liste / tour non rendu » cherchés par la même logique** :
- une liste qui finit complète termine son segment dans son crochet `give`, seulement à son tour : toutes
  les listes plus anciennes ont donc fini, et (dans l'ordre maintenant garanti) sont recueillies avant
  elle ;
- une liste qui finit inachevée garde son segment seulement à son tour (la plus ancienne) : elle est
  alors en tête des listes dehors, recueillie, remise en tête de file 0 (devant toute liste non confiée)
  et reprise avant toute autre (`CpParTryDispatch` refuse toute autre voie tant qu'un segment est gardé) ;
- le segment A de Thread_Gpu ne s'ouvre qu'en l'absence de liste dehors et se ferme à la remise suivante ;
  le segment gardé adopté par Thread_Gpu lui est rendu à la reprise (crochet `give`, « drop ») ;
- retraits tenus (`DeferRetire`) et opérations retenues : libérés quand le tour atteint leur segment,
  donc dès que les segments plus anciens se terminent (cas ci-dessus) ;
- reste possible : « open at list end » (une liste finie dont le segment resterait ouvert) — déjà compté,
  0 attendu ; le chien de garde le montrerait (ordre d'une voie sans liste dehors).

**À vérifier en jeu** : même essai. Attendu : plus de blocage ; « resumed before younger » > 0 ;
« out-of-order ends » 0 ; contrôles d'ordre à 0. En cas de blocage : les lignes « STALL ».

### Blocage à l'écran-titre (essai du 15/11, 19 h 01) et chien de garde

**Constat** (journal `Jeu\KytyPS5_U59\journaux\console-20261115-190156.txt`, `KYTY_CP_PAR_LANES=3`) :
après ~10 s normales (B 8319 tranches dont 7826 inachevées — les attentes de flip —, C 329, contrôles à
0), plus rien : 0 tranche confiée, 0 liste, « most lists out 0 », alors que Thread_Gpu tourne encore
(la ligne « CpPar 10s » est écrite par lui toutes les 10 s) ; processeur à 2 %. Donc Thread_Gpu ne
confiait plus aucune liste et n'avait pas de travail propre possible.

**Cause la plus probable, trouvée dans le code (non prouvée faute d'état au moment du blocage).** En mode
simultané la sélection de la boucle principale ne regarde plus la file 0 (`CpParQueue0Held`, testé
*avant* le test de barrière de trame) et `CpParTryDispatch` rendait faux quand la tête ne passait pas
`FrameFencePassed`, sans noter `fence_hold_ns`. Or le délai de sécurité de la barrière (2 s,
`FrameFenceTimeoutNs`) se compte depuis `fence_hold_ns`, que seule la sélection posait. Une tête retenue
par une barrière que rien d'autre ne lève (à tour de rôle : passée au bout de 2 s, « CP frame fence: …
running it anyway ») restait donc retenue pour toujours : exactement l'état observé (rien dehors, file 0
non vide, aucune activité). Corrigé (`9e05e27b`) : `fence_hold_ns` noté à chaque refus ; compteurs
« fence holds (timeouts) » ; message « frame fence: list … handed out anyway » au premier délai dépassé.

**Écart corrigé au passage** (cycle possible, non observé) : à tour de rôle, après une tranche inachevée,
Thread_Gpu soumet (`BufferFlush`) ce que la voie a enregistré dans son segment gardé. Le mode simultané
ne le faisait pas : étiquettes, interruptions ou demandes de flip enregistrées avant l'attente restaient
dans le tampon ouvert, alors que l'attente (ou une file de calcul, ou le jeu) peut dépendre de ce
travail. Désormais la voie soumet elle-même à la fin d'une tranche inachevée qui a avancé (à son tour,
sous le verrou) ; compteur « unfinished-slice flushes ».

**Autres cycles examinés (pas de cycle trouvé)** : attente du tour d'une opération visible contre tête
suspendue (la tête gardée est la plus ancienne ; Thread_Gpu la reconfie après l'attente des files
suspendues, ≤ 1 s) ; Thread_Gpu qui ne confie rien au-delà d'une liste inachevée (seule la tête attend,
les plus jeunes déjà dehors finissent ou attendent leur tour sans rien tenir) ; WaitRegMem d'une voie
plus jeune sur une étiquette d'une autre liste (opération du tour : exécutée seulement quand toutes les
listes plus anciennes sont finies, comme en série ; une attente sur une liste plus jeune bloquerait aussi
en série) ; flips et services renvoyés (seulement par la plus ancienne ; Thread_Gpu les sert dans chaque
pas, en détenteur délégué du verrou). Le séquenceur s'arrête à toute attente pas à pas non répondue de la
tête : les voies plus jeunes attendent alors leurs opérations en tournant (`yield`), sans rien tenir.
Non fait : « rendre son segment » pour une voie plus jeune (il faudrait plusieurs segments gardés à la
fois ; l'ordre de segment est fixé à la remise).

**Chien de garde** (`1ff2bee9`, actif avec `KYTY_CP_PAR_CONCURRENT=1`, message au démarrage « stall
watchdog on »). Si aucun flip ne sort pendant 20 s, une fois (réarmé quand les images reprennent), des
lignes « Kyty CpPar 2b concurrent: STALL: no frame for … » puis :
- pour Thread_Gpu, chaque voie et le séquenceur : ce qu'il attend et depuis combien de temps, avec deux
  valeurs (attente du tour par type : propre ordre, tour ; verrou des voies : titulaire, ticket ; requête
  à Thread_Gpu ; inactif ; famine : liste ; tick GPU : tick attendu, tick connu ; opérations
  prioritaires ; séquenceur garé : opération de réveil ; Thread_Gpu : « waiting for a lane », « every
  queue suspended or held », « no work », « own work ») — « running (no wait noted) » sinon ;
- verrou des voies : titulaire et tickets pris/servis ;
- Thread_Gpu publié (toutes les 50 ms) : dernier pas (run / progress / wait), chez lui ou non, segment
  adopté, listes dehors (liste, voie, finie) ;
- file 0 : nombre, tête (liste, voie, bloquée, commencée, barrière passée ou **FENCE HELD** et depuis
  quand), autres files, soumissions comptées, services en attente, `m_in_flight` (taille, premier) ;
- segment gardé, tour, ordre de chaque voie (réservé, plus ancien), segments à opérations retenues,
  dernier tick soumis (lecture de l'ordonnanceur sans attendre ; « scheduler busy » sinon).
Au démarrage, si les shaders prennent plus de 20 s avant la première image, un rapport « STALL » peut
sortir une fois sans blocage réel.

**À vérifier en jeu** : même essai (`KYTY_CP_PAR_LANES=3`, `KYTY_CP_PAR_CONCURRENT=1`). Si le jeu passe
l'écran-titre : relever « fence holds (timeouts) » (des délais dépassés confirmeraient la cause ; ils
existent aussi à tour de rôle, à comparer) et « unfinished-slice flushes », puis les attendus de la
section précédente. S'il bloque encore : envoyer les lignes « STALL » (elles nomment le cycle).

### `KYTY_CP_PAR_CONCURRENT=1` : obstacles 4 et 5 — faits le 15/11 au soir

Trois commits compilés (BUILD_OK chacun, aucun nouvel avertissement hors « profile data may be out of
date » des fonctions modifiées, inhérent au PGO). **Défaut 0 : rien ne change** (tout est derrière
`CpPar::LaneConcurrent()`, faux sans la variable). Le mode ne s'active qu'en 2b avec verrou des voies,
surcouche, réservation et segments gardés actifs (sinon message « need … ; lanes stay in turn ») ;
les commits `69c35cc1` et `c89637e1` ne doivent pas être essayés séparément (le premier seul exécuterait
des opérations visibles hors tour).

**Ce que fait le mode (obstacle 4, `69c35cc1`).**
- Répartition : les listes graphiques de la file 0 vont **aux voies B.. seulement** (rang r → voie
  1 + (r − 1) mod (N − 1)). Avec N = 2 tout va à B : aucun recouvrement possible (le mode tourne mais ne
  sert qu'à vérifier la mécanique). **Il faut N = 3 (B et C alternent) ou N = 4** pour que deux listes
  se recouvrent. Restent sur Thread_Gpu : listes à moteur de constantes (repli A), préparations de flip.
- Thread_Gpu ne prend plus le jeton : `CpParConcurrentStep` (début de chaque tour de sa boucle)
  1) interroge chaque voie dehors (`PollSlice` : sert sa requête flip/services en détenteur délégué du
  verrou, comme `RunSlice`), 2) recueille les tranches finies **dans l'ordre des listes** (retrait de la
  file, `m_in_flight`, comptes ; ramasse-miettes différé à un moment sans liste dehors), 3) confie la tête
  de la file 0 à sa voie si elle est libre (`RunSliceAsync`), l'ordre de segment réservé à la remise
  (`ReserveLaneOrder`, ou crochet `give` qui ferme d'abord le segment de A). **Rien n'est confié
  au-delà d'une liste inachevée** (segment gardé) : seule cette liste repart. Variante non faite :
  segments A intercalés à ordre réservé.
- Travail propre de Thread_Gpu (autres files, services de sa boucle, listes A, préparations de flip) :
  seulement quand aucune liste n'est dehors (il ouvre alors son segment A, plus jeune que tout), ou
  quand la tête est une liste inachevée revenue (il adopte son segment gardé, le plus ancien) — et
  alors **sous le verrou des voies** tant que des listes plus jeunes sont dehors. Sinon il attend
  (`WaitDispatchEvent`, réveil par les voies et `Enqueue`, 1 ms au plus).
- Une voie ne termine sa liste, et ne garde un segment inachevé, qu'à son tour (attente hors verrou),
  puis toute la fin (fusion de surcouche, soumission de passation, avance du tour) sous le verrou :
  aucune opération plus jeune ne s'intercale entre la fusion et la fin du segment. Crochets `take`
  sous le verrou aussi. Les voies n'empruntent plus `g_gpu_thread` (`IsGpuThread` = détenteur du
  verrou). Le contrôle de `Submit` « une seule voie a un tampon ouvert » est levé dans ce mode (les
  plages de ticks et la porte d'ordre du rejeu gardent l'ordre de la file).

**Obstacle 5 (`c89637e1`).** Dans `ResolveSubmission`, chaque opération est classée
(`CpParTurnReason`, repris de 1b3–1b5, prudent) : **attendent le tour** (hors verrou) les étiquettes
et fins de canal (EndOfPipe, ReleaseMem, EventWrite), écritures (WriteData, DmaData, horloges,
LodStats, DumpConstRam), attentes (WaitRegMem, prédication, COND_EXEC, branches, lectures pas à pas,
ReadCheck), tirages indirects, tirages qui héritent du nombre d'instances ou sans registres, flips ;
puis **toute l'opération** (liaison des registres, vidange de fin de canal par nombre de paquets
comprises) se fait sous le verrou. **Hors tour** : tirages et dispatchs directs portant leurs registres
(instantané ou emplacement préparé) — le travail Vulkan pur ; pour un segment plus jeune ils tournent
sur la surcouche d'image de sa voie. En plus, une voie plus jeune : ne fait aucune soumission
facultative (vidange « GPU au repos », lot de fins de canal, relecture anticipée), ne cède pas sa
tranche aux files de calcul, ne lance aucun service (ni en tête de boucle, ni dans une attente
DrawPrep : `ProcessCommands` les laisse à la plus ancienne), attend ses opérations du séquenceur sur
place au lieu de rendre sa tranche. Sûreté sous simultanéité :
- soumission forcée dans une opération hors tour (verrou tenu) : elle **part hors tour** au lieu
  d'attendre (attendre bloquerait les segments plus anciens) ; son tick est dans la plage du segment et
  la porte d'ordre du rejeu ne remet le tampon qu'après tous les ordres plus anciens ;
- `Finish` (attente GPU des prédications d'occlusion, vidages proxy, services de démappage) n'attend
  que le dernier tick de la voie appelante et des segments terminés (pas une soumission hors tour plus
  jeune, que le GPU ne verra qu'après la fin du segment appelant) ;
- **retraits partagés** (`DeferRetire`) : destruction d'un tampon ou d'une image du cache, et du LDS de
  calcul commun, tenue avec le segment ouvert le plus jeune et mise en file à son tick quand il devient
  le plus ancien (sinon la voie plus ancienne libérerait au plus tôt un objet déjà utilisé par une plus
  jeune : perte du GPU) ;
- registre de dépendances (`KYTY_CP_PAR_VERIFY=1`) : une liaison se fait dans une opération, verrou
  tenu ; elle n'attend plus le segment plus ancien écrivain (interblocage), l'aléa est compté.
- Partagés restants : `m_synthetic_occlusion_counter` (EventWrite) et `m_num_instances` /
  `m_pending_num_instances` (écrits par les tirages indirects, lus par les tirages qui héritent) ne sont
  touchés que par des opérations du tour, donc par le segment le plus ancien, dans l'ordre des listes,
  sous le verrou : comme à tour de rôle. Le contrôle « retries pending at list start » ne lit plus que
  le champ de la voie elle-même dans ce mode.

**Ligne « CpPar 10s: 2b concurrent (N) »** (`68d86418`) : « overlap 2+ lanes x ms, 3+ lanes y ms »
(temps où au moins 2 / 3 voies traduisent : dans une tranche, hors attente du tour, du verrou et du
séquenceur — c'est le recouvrement réel) ; « most lists out » ; « Thread_Gpu own work under the lanes'
lock » ; « garbage collections deferred » ; par voie : « active » (temps de traduction), « slices handed
out (back unfinished) », « turn waits: » par type (submit, label, write, wait, indirect, draw, flip,
service, slice, list-end, other : nombre et ms), « out-of-turn submits », « visible ops not oldest »,
« out of order », « dependency hazards », « services left to the oldest », « retires held for younger ».
Au démarrage : « 2b lanes run their lists at the same time: requested (KYTY_CP_PAR_CONCURRENT) » puis
« 2b concurrent lanes: on (lists on lanes B.. only, k lane(s) translating …) ».

**À vérifier en jeu** (session principale ; mêmes variables que plus bas, plus
`KYTY_CP_PAR_CONCURRENT=1`, **`KYTY_CP_PAR_LANES=3`** puis 4) :
1. Pas de perte du GPU, pas de blocage (chien de garde : noms « cp-par-2b-submit-turn »,
   « cp-par-2b-lane-lock », « cp-all-queues-suspended »), image correcte (contrôle visuel obligatoire).
2. **0 attendu** : visible ops not oldest, out of order (journal « Kyty CpPar 2b concurrent: … » si non
   nul), order violations, early retires avoided, seq position below, impossible, left, open at list
   end, resumed not oldest, unreserved begins ; turn waits « slice » ≈ 0 (une voie plus jeune ne revient
   inachevée que si elle est devenue la plus ancienne).
3. **Non nuls attendus** : overlap 2+ lanes > 0 (sinon aucun gain possible ; avec N = 2 il reste à 0),
   turn waits label/write/wait/list-end sur B et C, overlay ops/images/fix-ups > 0 (les segments plus
   jeunes sont maintenant exercés : la surcouche et ses correctifs servent pour de bon ; « not no-ops »
   de la ligne 1b2 peut devenir non nul, ce sont ces correctifs), held operations > 0, submit turn waits
   > 0, retires held > 0 possible.
4. À relever : out-of-turn submits (combien de soumissions forcées dans des tirages), dependency hazards
   (avec `KYTY_CP_PAR_VERIFY=1` ; chacun est une lecture possible d'une plage avant l'écriture d'un
   segment plus ancien : à traiter par l'obstacle 6), i/s comparés à 2b à tour de rôle (N = 3).
   Aucun gain n'est garanti : les opérations restent sérialisées par le verrou des voies ; seul le
   travail entre opérations (séquenceur, DrawPrep, rejeu) se recouvre. Le gain de traduction viendra de
   P3 (sortir du verrou).
5. En cas de problème : revenir à `KYTY_CP_PAR_CONCURRENT=0` (comportement d'avant, à tour de rôle) ;
   il n'y a pas d'interrupteur par morceau à l'intérieur du mode (4 sans 5 n'est pas sûr) ; les
   interrupteurs `KYTY_CP_PAR_LOCK/OVERLAY/RESERVE/KEEP=0` coupent le mode entier (prérequis).

**Risques restants (concurrent).**
- **Blocage** (pas une perte) si, dans un tirage hors tour, une voie plus jeune attend le GPU sur son
  propre tick (relecture CPU forcée après une soumission hors tour) ou attend de la place (anneau du
  rejeu plein, 16 Mo, derrière une soumission hors tour en attente de la porte d'ordre ; pool de
  descripteurs ou flux épuisé attendant un tick plus ancien non soumis) : le verrou tenu empêche la
  plus ancienne de finir. Non observé (non essayé) ; « out-of-turn submits » en donne la fréquence.
- **Données d'ordre** (image fausse, pas de perte attendue) : un tirage plus jeune lit une plage
  (tampon, image, cible) qu'une liste plus ancienne écrit ensuite au CPU ou dont l'état de barrière du
  cache est mis à jour dans un ordre différent — c'est l'obstacle 6 (registre permanent avec images et
  cibles). Les états d'image passent par la surcouche ; les tampons non.
- **Perte du GPU encore possible** si un objet partagé est libéré « au tick de l'appelant » par une
  voie plus ancienne ailleurs que dans les trois retraits convertis (`DeferRetire`) : vues et autres
  objets détruits hors `DeferOperation`, recyclage interne d'un cache (`TrimRetiredImages` réutilise
  les images natives retirées : couvert puisque l'image n'est retirée qu'à l'effacement de son
  emplacement, tenu). Les temporaires par opération (tampons de mise en place, tuilage) restent au
  tick de l'appelant (utilisés par lui seul).
- Un service exécuté par la voie la plus ancienne (démappage) pendant qu'une voie plus jeune a déjà
  enregistré un accès à la plage démappée : `Finish` ne couvre pas cet enregistrement non soumis
  (le jeu ne démappe normalement que ce qu'aucune liste en attente n'utilise).
- Statistiques globales (CommitStats, PrepPhases, DrawPrep) mélangées entre voies ; le décompte
  DrawPrep par voie de la ligne 2a n'est plus fait pour les listes confiées.

### Obstacles 1, 2 et 3 de la passation ci-dessous — faits le 15/11 (toujours à tour de rôle, non essayés en jeu)

Trois commits compilés (BUILD_OK chacun), sans 2b rien ne change (tout est derrière `Lanes2b()`).

**Obstacle 1 (`3be1ee4d`) — champs du `CommandProcessor` par voie.** Ajoutés à `LaneCpState` (indice
`EffectiveLane()`, donc aussi la voie servie quand Thread_Gpu exécute un flip renvoyé) : `resolver_packets`,
`retry_op`, `executing`, `placement_packets` (écrits entre les opérations, hors verrou) et deux états d'opération
gardés d'une opération à l'autre au sein d'une liste, relevés en plus : `defer_next_label` (posé par un
`EventWrite` de vidage de proxy de visibilité, lu par l'étiquette de fin de canal suivante) et
`flip_wait_suspended` (`WAIT_FLIP_DONE` suspendu puis repris). Hors 2b l'indice reste 0 : 1b3–1b5 (qui
sauvaient/restauraient `m_resolver_packets`/`m_retry_op` autour des opérations de la course en avant) sont
inchangés. `m_resolve_lane` n'est plus écrit en 2b (il n'y était déjà plus lu). `m_verifier` reste commun : 2b
n'existe jamais avec `KYTY_CP_SEQ_VERIFY` (`EXIT_IF` au `StreamBegin` si une liste demandait la vérification).
**Relevés mais laissés communs** (à revoir avant la simultanéité) : `m_synthetic_occlusion_counter` (compteur
d'`EventWrite`, sous le verrou des voies : seul l'ordre des valeurs entre voies changerait) ;
`m_pending_num_instances`/`m_num_instances` (nombre d'instances hérité d'un tirage indirect, touché à la fois par
le front du séquenceur et par les opérations : partage déjà présent en 2a, à analyser) ; `m_predicate_skip`,
`m_ce_complete`, `m_flip` (front du séquenceur, pas des voies). Contrôle : « retries pending at list start »
(nombre de voies ayant une opération suspendue à reprendre quand une voie commence une liste) : **0 attendu**
(file 0 FIFO, et une voie garde sa liste jusqu'au bout).

**Obstacle 2 (`9e7f0ce0`) — ordre réservé à la remise, A fermé à chaque fin de liste.**
`CommandScheduler::ReserveLaneOrder(lane)` : dans le crochet `give` de Thread_Gpu qui confie une liste à la voie
k (cible `HandoffTarget()`), juste **après** la fin du segment de A (pour que A puisse encore rendre une plage
inutilisée, `TryReleaseRange`), Thread_Gpu réserve l'ordre et la plage de ticks de k (`BeginRange`, sous
`m_operation_mutex`, `m_lane_order[k]` posé, `m_lane_reserved[k]`) ; `BeginContextSegment` de la voie les
consomme. Le segment réservé compte comme ouvert pour le tour (les fils sans voie lisent déjà ses ticks). À la
fin de chaque liste A (`ProcessSequenced`, contexte 0, liste complète, hors listes de passation au mode direct et
de préparation de flip), `GuestGpu::CpParCloseLaneA` termine le segment de A (la soumission de passation emporte
le travail de la tranche, à la place du `BufferFlush` : `BufferFlush(false)` fait le reste) et en rouvre un.
À tour de rôle, mêmes ordres qu'avant (la réservation se fait au même moment que l'ancien début de segment),
mêmes soumissions (la fermeture remplace la soumission du `BufferFlush` ; le segment de A rouvert, vide, rend sa
plage au passage suivant). `KYTY_CP_PAR_RESERVE=0` revient au comportement précédent. Compteurs : « orders
reserved at hand-out » (voies B, C… : ≈ nombre de listes confiées, plus les renvois si `KYTY_CP_PAR_KEEP=0`),
« unreserved begins » (**0 attendu** pour B, C…), « list-end closures » (voie A : ≈ listes A).

**Obstacle 3 (`25a85844`) — une voie garde son segment jusqu'à la fin de sa liste.** Quand une voie rend le jeton
pour un renvoi (flip, services : `CpPar::GivingForRequest`, posé dans `Resolver2::RunOnGpuThread`) ou à la fin
d'une tranche qui n'a pas fini sa liste (suspendue, affamée, cédée : la tâche de tranche appelle
`CpPar::SetListUnfinished(result != Complete)`), son crochet `give` **ne termine pas** le segment
(`g_cp_par_kept_lane`). Thread_Gpu, en reprenant le jeton, **enregistre dans ce segment** (flux d'enregistrement,
voie de ticks, exécuteur, pool, état dynamique, lot d'occlusion de k, segment du livre d'images :
`CpPar::AdoptSegment`) au lieu d'ouvrir un segment A : le flip renvoyé, le `BufferFlush` de fin de tranche
(soumission intermédiaire dans la plage de k), les autres files et les services qu'il traite en attendant la
reprise. Quand il rend le jeton à k (`give`, `t_cp_par_adopted`), rien ne se termine ni ne se réserve
(`DropSegment`) ; k reprend son segment sans en ouvrir un (`take`). Le segment se termine à la fin de la tranche
qui achève la liste (ou la passe au mode direct). Ordre GPU identique à l'ancien (k jusque-là, travail de
Thread_Gpu, suite de k), mais sans segment intermédiaire : **aucune liste confiée plus tard ne peut recevoir un
ordre plus ancien que la suite d'une liste inachevée**. Pourquoi Thread_Gpu enregistre dans le segment de k au
lieu d'ouvrir le sien : un segment A plus jeune ouvert pendant que k attend devrait attendre son tour pour
soumettre ; si k attend un résultat de calcul que Thread_Gpu enregistre (file de calcul, `WAIT_REG_MEM`), ce serait
un interblocage. `KYTY_CP_PAR_KEEP=0` revient au comportement précédent. Compteurs (voie k) : « segments kept over
slices », « over bounces », « resumed without closing » (= somme des deux), « not oldest » (**0 attendu** : une
reprise est toujours le segment le plus ancien), « delegated recordings » (prises du jeton par Thread_Gpu dans le
segment de k), « open at list end » (**0 attendu**). Contrôles `EXIT_IF` : un seul segment gardé à la fois ;
Thread_Gpu ne rend le segment gardé qu'à sa voie ; la fermeture de A n'a jamais lieu pendant une adoption.

**Ce que la session principale doit vérifier en jeu** (mêmes variables que plus bas, `KYTY_CP_PAR=2b`) : au
démarrage, les lignes « segment orders reserved at the hand-out … on » et « lanes keep their segment … on »
(affichées au premier usage) ; dans la ligne « CpPar 10s: 2b », les nouveaux champs ci-dessus avec les valeurs
attendues, plus tous les anciens zéros (submit turn waits, held operations, order violations, seq position below,
overlay…). « submit turn waits » doit rester à 0 : une attente montrerait qu'un segment plus jeune soumet pendant
qu'un plus ancien est ouvert. i/s : comme avant (aucun gain attendu ; un peu moins de soumissions vides
possible, puisqu'un renvoi ne coupe plus le segment). En cas de blocage ou d'image fausse, essayer
`KYTY_CP_PAR_KEEP=0`, puis `KYTY_CP_PAR_RESERVE=0`, pour isoler l'obstacle.

**Risques (3)** : Thread_Gpu utilise maintenant l'exécuteur, le pool et l'état dynamique de la voie k pendant les
flips renvoyés et entre deux tranches (auparavant les siens) ; un état propre au fil qui supposerait « Thread_Gpu =
flux 0 » (état gardé dans l'exécuteur de A, `thread_local` d'un module de rendu) serait faux → image fausse ou
erreur de validation, à surveiller. Le `BufferFlush` qui suit une tranche est compté dans `LaneCp()` de A (comme
avant) alors qu'il soumet le tampon de k : sans danger (k vide un peu plus tôt ensuite). (2) : une soumission
vide de plus est possible quand une liste A se termine sans rien avoir enregistré après la dernière soumission.

### Passation (15/11 soir, après `b758f3d3`) : `KYTY_CP_PAR_CONCURRENT=1` non commencé

**Mise à jour (après `25a85844`) : les obstacles 1, 2 et 3 sont levés (section précédente). Restent 4 à 7.**
**Mise à jour (après `68d86418`) : 4 et 5 sont faits derrière `KYTY_CP_PAR_CONCURRENT=1` (section
« obstacles 4 et 5 » plus haut). Restent 6 (registre de dépendances permanent avec images et cibles de
rendu, au lieu des aléas seulement comptés) et 7 pour la partie temps sous verrou déjà présente ; puis
P3 (lectures sans verrou).**
Pour 4 (boucle des files sans attente de la tranche), noter ce que 2 et 3 imposent : en simultané, la voie A ne
pourra pas « adopter » le segment d'une autre voie en attente sans s'arrêter (règle de 3, faite pour le tour de
rôle) ; il faudra soit que Thread_Gpu ne confie pas de liste au-delà d'une tête suspendue (et n'enregistre rien en
attendant hors du segment gardé), soit des segments A intercalés à ordre réservé d'avance. La réservation
(`ReserveLaneOrder`) se fait déjà au moment de confier : avec `RunSliceAsync`, l'appeler au moment de la remise
(elle exige que la voie n'ait ni segment ouvert ni réservé). Les champs de `LaneCpState` sont lus par
`EffectiveLane()` : un fil de voie ne touche que les siens.

Demandé : lever le jeton (désactivé par défaut, sur 2b). **Pas écrit**, faute de place, et parce que l'examen du
code a montré des obstacles qu'il faut traiter avant, sous peine d'un mode qui bloque ou perd le GPU dès le premier
essai. Relevé (vérifié dans le code, `graphicsRun.cpp`) :
1. **Champs du `CommandProcessor` encore communs**, modifiés entre les opérations, hors verrou : `m_retry_op`,
   `m_resolver_packets`, `m_executing`, `m_verifier`, `m_resolve_lane` (1b1), instantanés et fenêtre de
   `ResolveSubmission` (1b3 les sauvait/restaurait autour de chaque opération de la course en avant : l. ~7590,
   `saved_packets`, `saved_retry`). À mettre dans `LaneCpState` (indice = contexte de l'anneau) avant tout.
2. **Ordre des segments = ordre des listes.** Les plages de ticks sont réservées à `BeginContextSegment`, dans
   l'ordre où les segments commencent. En simultané, Thread_Gpu doit fermer son segment de voie A à la fin de
   *chaque* liste A (aujourd'hui il reste ouvert jusqu'au passage du jeton), et une voie ne doit commencer le
   segment de la liste n+1 qu'après la réservation de celui de la liste n. Le plus simple : Thread_Gpu réserve
   l'ordre au moment où il confie la liste (nouvelle fonction `MasterSemaphore::BeginRangeFor(lane)` appelée
   par Thread_Gpu, consommée par la voie à son `BeginContextSegment`).
3. **Tranches qui cèdent ou se suspendent** (`Pm4ProcessResult::Blocked`, attente `WAIT_REG_MEM` non passée,
   `m_yield`) : aujourd'hui la tranche se termine (passation = fin de segment) et la liste reprend plus tard dans
   un nouveau segment. En simultané, un segment plus jeune aurait alors déjà réservé un ordre plus petit que la
   reprise : remise hors ordre, **perte du GPU**. Règle à écrire : en mode simultané, une voie garde son segment
   ouvert jusqu'à la fin de sa liste (elle attend sur place, sans passation), ou bien Thread_Gpu ne regarde
   au-delà de la tête que si la tête n'est pas suspendue.
4. **Boucle des files** (`ProcessGraphicsSequenced` → `CpPar::RunSlice`, l. ~2703) : remplacer l'appel
   bloquant par `RunSliceAsync(lane)` + constat de fin (état `SliceDone` déjà présent dans `Resolver2`) ; le
   retrait de la tête de file 0, `NoteCompleted`, `RunGarbageCollector`, `NoteStarted` du séquenceur et la
   barrière de trame doivent rester dans l'ordre des listes (seulement quand la tête est finie). Les hooks
   `take`/`give` du jeton (début/fin de segment, `g_gpu_thread`) deviennent début/fin de tranche sur le fil de
   la voie, sans jeton.
5. **Opérations visibles par le jeu** (à faire attendre le tour, `WaitSegmentTurn`, hors verrou) : `EndOfPipe`,
   `ReleaseMem`, `WriteData` hors étiquettes internes, `WaitRegMem` sur une valeur d'une autre liste, toute
   opération qui lève une interruption, lectures GPU par le CPU (renvois de service). Classement à reprendre de
   1b4/1b5 (`CpParOpRaisesInterrupt`, `CpWriteRange`). Flips et services : faits (suite 1).
6. **Registre des dépendances** permanent : `NoteLaneBuffer` n'enregistre que les tampons de stockage ; images et
   cibles absentes ; une écriture d'un segment plus ancien sur une plage déjà liée par un plus jeune reste une
   violation non réparable (P4).
7. Compteurs demandés (temps de traduction superposé, temps sous verrou, attentes) : le temps sous verrou et
   les attentes (verrou, tour, dépendances) existent déjà dans la ligne 2b ; le temps superposé reste à ajouter
   (somme des intervalles où ≥ 2 voies sont dans `ResolveSubmission`).

Points où, une fois écrit, une **perte du GPU** serait possible : 2, 3 (ordres), 1 (champs communs : paquets ou
reprises mélangés entre voies), barrières d'image si une opération plus jeune échappe à la surcouche (services,
chemins hors `ExecuteOp`), retraits au tick du fil appelant non relevés. **Blocages** possibles : attente du tour
en tenant le verrou des voies, crochet `give` de Thread_Gpu hors service pendant qu'une voie tient le verrou,
séquenceur attendant une opération d'une liste suspendue (`Executed()` par voie le rend plus prudent).

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
thread … », « 2b global lane lock … on », « 2b image state overlay of younger segments: on ». Attendus toutes
les 10 s : « not no-ops A 0 B 0 » ; dans la ligne 2b : **submit turn waits 0, held operations 0, dependency waits 0,
order violations 0** (à tour de rôle) ; depuis les morceaux (a)–(d) aussi **occlusion early batches 0, seq position
below 0, overlay ops 0, images 0, fix-ups 0, impossible 0, left 0** ; « bounces under it » > 0 (flips et services
renvoyés) avec « delegated takes » > 0 (le crochet de passation pris par Thread_Gpu pour la voie servie) ; « dumps
outside the lock » : à relever (vidages d'occlusion hors opération, sans danger à tour de rôle) ; « turn waits » du rejeu > 0 est normal
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
6. **Morceaux (a)–(d)** (15/11 soir), à tour de rôle : (c) le verrou des voies est maintenant tenu pendant les
   flips et services renvoyés à Thread_Gpu et pris par les crochets de passation (morceau (d)) ; un oubli de
   délégation donnerait un **interblocage** (chien de garde « cp-par-2b-lane-lock »), pas une perte du GPU.
   (b) chaque voie a son pool de requêtes d'occlusion : une erreur de remise à zéro donnerait des valeurs
   d'occlusion fausses (objets qui clignotent) ou une erreur de validation ; « Occlusion counter: batch slot check
   failed » le signalerait. (a) un `Executed()` trop petit bloquerait le séquenceur (chien de garde
   « sequencer-progress ») ; trop grand, il lirait des octets de commande avant une écriture du CP ; « seq position
   below » 0 attendu. `RenderMutex` : mutex ordinaire, interblocage possible seulement si une voie le tenait au
   moment d'un renvoi à Thread_Gpu (avant, `ResignOwner` l'aurait arrêté). (d) jamais exercé à tour de rôle.

### Suite

**Après les morceaux (a)–(d) (15/11 soir), ce qui reste avant de lever le jeton** (ordre conseillé) :
1. Essai en jeu, à tour de rôle, des commits `a2c83dea`…`c00a785c` : mêmes attendus que plus haut, plus les nouveaux
   compteurs à 0. Si « delegated takes » reste à 0 alors que « bounces under it » > 0, ou si le chien de garde
   nomme « cp-par-2b-lane-lock », la délégation (c) est à revoir.
2. Fin de segment à son tour : un segment plus jeune doit attendre d'être le plus ancien avant sa passation
   (règle de (d), compteur « left ») et avant tout flip ou service renvoyé à Thread_Gpu (ils touchent l'état
   global hors surcouche, et la passation du renvoi termine le segment) : réutiliser `WaitLaneTurn`.
3. Services de Thread_Gpu hors `ResolveSubmission` (sa boucle principale) sous le verrou des voies ; vidages
   d'occlusion hors opération (« dumps outside the lock ») idem.
4. Puis l'étape 1 de 3b-3 ci-dessous (boucle des files sans attente de la tranche).

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
