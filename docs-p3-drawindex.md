# P3 étape 2 : ce que le commit d'un DrawIndex préparé touche d'état partagé entre voies

Relevé du 15/11/2026 (lecture seule du code au commit `1b3a93e7`, trois lectures parallèles : préparation,
descripteurs/flux, caches). Rien n'a été essayé en jeu. Chemins relatifs à `src\graphics\`.

## Verdict

Aujourd'hui, aucun DrawIndex n'est en lecture seule, même tout préparé et même quand il continue le précédent
(draw run). Le cas courant écrit encore, sans protection, dans quatre structures communes aux voies : l'anneau
de flux, le tas de descripteurs, les drapeaux de liaison des images et l'état de layout des images. Trois autres
obstacles :

- **Deuxième verrou global.** `RenderExecutor::DrawIndex` prend aussi `RenderLockGuard`
  (`renderDraw.cpp:6202`). En 2b, `RenderMutex` est un mutex ordinaire pour tous les fils
  (`renderContext.h:52-53`) : il sérialise lui aussi tout le dessin.
- **`IsGpuThread()` n'est vrai que sous le verrou des voies** (`graphicsRun.cpp:9008`). Hors verrou : les
  `EXIT_IF(!IsGpuThread())` de `Engine::CommitPublished` (`drawPrep.cpp:1478`) arrêtent le programme ;
  `DrawPrep::Validate` refuse le certificat (`IsGpuCleanForRead`, `kernel\memory.cpp:1669`) ; le cache de
  tampons abandonne ses chemins sans verrou (`bufferCache.cpp:3584`, `3613`, `3694`).
- **Le cache de tampons n'a aucun mutex général.** Tout repose sur l'hypothèse d'un seul fil GPU
  (`memoryTracker.h:44-49`).

Il faut donc d'abord privatiser par voie quelques structures (§2, C1 à C4) ; ensuite seulement un chemin rapide
est possible pour le sous-ensemble défini en §3 (a).

## 1. Zone tenue sous le verrou

**Bornes** : de `CpPar::LaneLockAcquire()` (`guest_gpu\graphicsRun.cpp:8857`) à `concurrent_op.Release()`
(`:8957`) ; prise ré-entrante en `:8934` avec la surcouche d'image.

**Chaîne** : `ExecuteOp` (`:5891`) → `ExecDrawIndex` (`:3787`) → `Engine::CommitPublished`
(`drawPrep.cpp:1477`) → `CommitHead` → `ReadyHead` (`:1536`) → `Engine::Commit` (`:1850`) →
`RenderExecutor::DrawIndex` (`renderDraw.cpp:6156`) → `PrepareDrawRenderState` → `ExecutePreparedDraw` →
`ResetBindings` ; puis `m_after_commit` (`MaybeFlushIdleGpu`), `Retire`, `MaybeYieldSlice`.

**Un seul exemplaire** dans `RenderContext` (`renderContext.h:213-231`) : `CommandScheduler`, `DescriptorHeap`,
`PipelineCache`, `SamplerCache`, `PageManager`, `BufferCache`, `TextureCache`, `OcclusionCounter`, `LodStats`,
`RenderMutex`.

**Déjà par voie** (indice `CommandRecorder::ThreadStream()`) : l'exécuteur et ses mémos
(`renderContext.h:143`) ; l'enveloppe `CommandBuffer` via `Cmd()`/`LaneCmd` (`commandScheduler.h:273-282`) ;
`DynamicState` (`:146`), encodeur, anneau du rejoueur, pool de tampons de commande ; moteur DrawPrep, fenêtre,
ouvriers (`LanePrep`, `commandProcessor.h:590`) ; `LaneCp()`, anneau d'instantanés ; lots d'occlusion en 2b
(`occlusion.cpp:140-145`) ; compteurs Profiler (bloc par fil, `common\profiler.h:1443`).

## 2. Relevé

Légende — **L/É** lecture/écriture ; **C** cas courant, **R** création/téléversement/synchronisation/repli ;
**Besoin** : **Voie** copie par voie, **Lect.** verrou partagé, **Atom.** atomique, **Global** retomber au
verrou global.

### 2.1 Bloquants du cas courant

| # | Élément (fichier:ligne) | L/É | Cas | Protection | Besoin |
|---|---|---|---|---|---|
| C1 | Anneau de flux `BufferCache::m_stream_buffer` (`bufferCache.h:956`) via `UploadShaderData`→`NativeUpload` (`descriptors.cpp:1372-1380`), petites lectures sales CPU (`bufferCache.cpp:4422-4511`), index 8 bits (`renderDraw.cpp:6334`, `:3445`) ; `m_offset`, `m_mapped_size`, veilles au tick (`streamBuffer.cpp:358-455`) | É | C | rien | **Voie**. Un mutex ne suffit pas : au débordement, `Wait` sur le tick d'une autre voie → `EXIT_IF(tick > CurrentTick())` (`commandScheduler.cpp:306`) ou blocage |
| C2 | `DescriptorHeap::Commit` (`descriptorHeap.cpp:57-94`, appel `descriptors.cpp:3153`) : `m_sets` (lot de 32), `m_current_pool`, `m_current_pool_tick`, `m_pending_pools` ; allocation Vulkan tous les 32 jeux, pool créé/remis à zéro quand plein | É | C / R | rien (seul le retrait au tick max, `:61-64`, connaît les voies) | **Voie** (pool par voie) |
| C3 | `Image::binding` (`image.h:58-66`) : posé par `BindImage`/`BindRenderTarget` (`descriptors.cpp:1501-1522`), `DrawRunMarkBindings` (`renderDraw.cpp:2721-2750`) ; effacé par `ResetBindings` (`descriptors.cpp:1524-1531`, appel `renderDraw.cpp:6361`) ; lu par `CommitBindings` (`:2848-2874`) | L/É | C | rien | **Voie** (table ImageId → drapeaux dans l'exécuteur) |
| C4 | `Image::backing.state` / `subresource_states` : `Transit`/`AdoptState` dans `CommitBindings` (`descriptors.cpp:2835-2884`) et `AcquireRenderTargets` (`renderDraw.cpp:929`, `1078`, `1084`) ; la surcouche `CpParTouch` échange en place (`image\image.cpp:1203-1207`, `1226-1229`) | L/É | C (sauf continuation à images gardées ; une transition sans effet passe aussi par `CpParTouch`) | rien | **Global** dès qu'une transition réelle ; le chemin rapide n'accepte que `TransitIsNoOp` (`image.h:107-113`) sous **Lect.**, ou un état par voie réconcilié à la fusion |
| C5 | `RenderMutex` (`renderDraw.cpp:6202`) | É | C | lui-même | **Lect.** pour les dessins, exclusif pour la présentation (`swapchain.cpp:1319`, `1347`, `1461`, `1603`) et `graphicsRun.cpp:4422`, `5283`, `5390` |
| C6 | Identité fil GPU (`graphicsRun.cpp:9008`), ~35 sites dans `host_gpu` dont des changements de chemin : `bufferCache.cpp:3584`, `3613`, `3694`, `3741`, `2816`, `3101` ; `textureCache.cpp:4330`, `5661` ; `kernel\memory.cpp:1247`, `1632`, `1669` | L | C | — | Identité `t_fast_lane` vraie seulement aux sites de lecture sûrs |
| C7 | `PopReadyOperations` en tête de `DrawIndex` (`renderDraw.cpp:6188` ; `commandScheduler.cpp:358-410`) : rappels différés (destructions, recyclage, `ProcessFaultBuffer`) ; `m_last_pending_refresh_ns` simple (`:375`) | É | C | file seule (`m_operation_mutex`) | **Global** : sauté dans le chemin rapide |

### 2.2 Écritures du cas courant à rendre par voie ou atomiques

| # | Élément | L/É | Cas | Protection | Besoin |
|---|---|---|---|---|---|
| W1 | `TouchImage` via `GetImage` : `frame_accessed_last`, `lru_tick`, `m_lru_cache.Touch` (`textureCache.cpp:1221-1254`) | É | C | rien | **Atom.** + LRU différé par voie |
| W2 | `TouchBuffer` (`bufferCache.cpp:1228-1239`) | É | C | rien | idem |
| W3 | Mémo de liaison `m_binding_memo`/`m_binding_hot_memo` (`bufferCache.cpp:3967-4127`) | L/É | C | rien ; réservé à `IsGpuThread` (`:3694`) | **Voie** |
| W4 | Totaux `m_binding_memo_totals`, `m_relaxed_totals`, `m_range_memo_totals`, `m_written_sync_totals` (`bufferCache.h:822-871`) | É | C | non atomiques | **Voie**/**Atom.** |
| W5 | `m_upload_batch_depth` + `FlushBarriers` sur `Current()` (`bufferCache.cpp:1628-1651`), à chaque dessin (`descriptors.cpp:2198`, `2530`) | É | C | rien | **Voie** |
| W6 | `image.usage.texture`, `NoteExactRangeStaleUse` (`descriptors.cpp:2303-2305`, `2356-2359`, `2421-2423`) | É | C | rien | **Atom.** |
| W7 | `feedback_*` de profondeur (`renderDraw.cpp:1119-1135`) | É | C (profondeur) | rien | **Voie**/**Global** |
| W8 | `g_dynamic_state_emitted` (`renderDraw.cpp:420`, `441`, `5694`, `5713`) | É | C | `static` simple | `thread_local` |
| W9 | Résumés 10 s appelés par `Engine::Commit` (`drawPrep.cpp:1909-1912`) : `PrintDrawPrepSummary`, `DrawRun::PrintSummary` (`drawRun.cpp:467`), `TexelImageLookup::PrintSummary` (`textureCache.cpp:4462`), `XFrameReuse::PrintSummary` (`xframeReuse.cpp:64`) ; statiques `last_ns`, `last[]`, `calls` | L/É | C | rien | une seule voie, ou `last_ns` atomique |
| W10 | `OwnedCounters::Add` (`ownedCounters.h:33-49`) | É | C | atomique en mode 0 | imposer le mode 0 |
| W11 | `g_activity_misses` (`drawRun.cpp:128`, `425-431`), `static logged` (`renderDraw.cpp:2653`, `:667`), `last_ns` (`pipelineCache.cpp:7462`) | É | diagnostic | rien | **Atom.** |
| W12 | `BufferCache::m_eager_flush` et budget (`bufferCache.cpp:2964-2980`), lus par `MaybeFlushIdleGpu` (`graphicsRun.cpp:1812-1864`, crochet `:6177`) | L/É | C | rien | **Atom.**/**Voie** ; `BufferFlush` sous **Global** |
| W13 | `CommandScheduler::Active()` lit `m_command.m_registers` de la voie 0 (`commandScheduler.h:136`) que la voie 0 réécrit à chaque dessin (`:118-131`) ; appelé par `ExecuteOp` (`graphicsRun.cpp:5931`), `Image::Transit`, `CheckActive` | L | C | rien (course) | lire `Cmd().m_registers` |
| W14 | `LaneSegmentOldest` prend `m_operation_mutex` à chaque opération (`commandScheduler.cpp:912-918`, appel `graphicsRun.cpp:8938`) ; `HasRunnableComputeWork` prend `m_queue_mutex` tous les 128 dessins (`:1237`) | L | C | mutex | atomique |

### 2.3 Lectures du cas courant

| # | Élément | Protection | Besoin |
|---|---|---|---|
| R1 | `m_slot_images.try_get` (`textureCache.h:561`, `slotVector.h:51-123`) ; `insert`/`erase` hors `m_lock` (`textureCache.cpp:1175`) | rien | **Lect.** + exclusif pour insert/erase |
| R2 | `image.views` lu sans verrou (`renderDraw.cpp:984`, `descriptors.cpp:2856`) ; `FindView` peut réallouer | rien | **Lect.** |
| R3 | `needs_rebind`/`registered` (`descriptors.cpp:2321-2325`, `2512-2514` ; `renderDraw.cpp:913-916`, `945-947`) | rien | **Lect.** |
| R4 | Sous `m_lock` (spinlock exclusif, `textureCache.h:548`) : `TextureBindingMemo::Try*`, `TryRepeatLookup` (`textureCache.cpp:4365-4417`), `DrawRunImagesChange` (`renderDraw.cpp:2610`), `IsRegionGpuModified` (`textureCache.cpp:6316-6354`) ; écrivent aussi (`tick_accessed_last`, LRU, `info.metadata`, `m_surface_metas`, `m_clean_image_pages` `:6351`) | `m_lock` | scinder lecture / écriture différée |
| R5 | `HasGpuDirtyBytes` : `m_gpu_modified_ranges`, `m_early_released` (`bufferCache.cpp:4945-4951`) | rien | **Lect.** ou seqlock |
| R6 | `FindBuffer` : `m_page_table`, `m_slot_buffers` (`bufferCache.cpp:2982-2997`) | rien | **Lect.** |
| R7 | `RangeSignature`, `SyncEpoch::Current()`, `m_bda_structure_epoch` (`bufferCache.cpp:3996-4048`), `QueryDirtyRelaxed` | atomiques | rien si seul le chemin verrouillé change les bits sale |
| R8 | `m_rebind_epoch = UnregisterEpoch()` (`descriptors.cpp:2532`) | non atomique | **Atom.** |
| R9 | Certificat DrawPrep : `ShaderMapGeneration`, `Coherence::g_log`, `HasPendingBackingPublication` | atomiques/seqlock | rien (voir risque « validation périmée ») |
| R10 | Pipeline prévu : `PlannedPipeline` (`bindingPlan.cpp:650-669`), `NotePlannedPipeline` (`pipelineCache.cpp:7262`) | atomiques/TLS | rien |
| R11 | `DrawRun::g_activity` (`drawRun.cpp:132-163`), `ImageStateEpoch` (`image.h:54`) | atomiques globaux | époque par voie (sinon les continuations s'effondrent) |
| R12 | Occlusion `GateOpen()` : `m_pairs`, `m_gate_broken` (`occlusion.h:138-140`) | rien | **Lect.**/atomique |
| R13 | `LodStats::CanaryReady()` (`lodStats.h:68`), `CounterBuffer()` paresseux (`lodStats.cpp:85-97`) | rien | **Atom.**, création à l'init |
| R14 | `MasterSemaphore` : `IsFree`, `Refresh`, `CurrentTick` (`masterSemaphore.cpp:115-119`) | atomiques | rien ; vérifier que `t_tick_lane` et `ThreadStream` désignent la même voie |
| R15 | `SyncEpoch::Advance` (`graphicsRun.cpp:8874`), `HangWatchdog::NotePacket` | atomiques | rien |

### 2.4 Création, téléversement, synchronisation, repli : verrou global

- Tampons : `CreateBuffer`/`ResolveOverlaps`/`JoinOverlap`/`DeleteBuffer` (`bufferCache.cpp:3059-3085`,
  `1241-1257`) ; `SynchronizeBuffer` après la sortie précoce (`:3087`), staging (`:3172`) ; toute liaison
  **écrite** (`:4524-4538`, `:4566-4578`) ; `PrepareBda` (`renderContext.cpp:391-399`).
- Textures : `FindImage` (`textureCache.cpp:4118-4282`) ; `FindRenderTarget`/`FindDepthTarget`
  (`:4631-4726`, **tout dessin dont les cibles ne viennent pas d'une continuation ou d'une réutilisation**) ;
  `FindTexture`/`FindView` sur mémo raté.
- Opérations de cible : `DepthStencilCopy` (`depthRenderTarget.cpp:457-497`), `ResolveColorTargets` mode 3
  (`renderDraw.cpp:6826-6848`), `ConsumeMetadataColorOperation`.
- `GetGraphicsPrograms` (repli DrawPrep), `GetGraphicsPipelineImpl` (`pipelineCache.cpp:7361-7531`).
- Occlusion : `Initialize`, `FlushPending`/`FlushBatch` (`occlusion.cpp:164-218`) ; `Drain` du rejoueur
  (`commandRecorder.cpp:924-973`, `m_drain_serial` partagé).
- `NoteDepthFeedback`, second passage MRT, `NativeS1`/`n3b`, `BufferFlush`, redémarrage de primitive Scan.
- DrawPrep : `ReadyHead` qui prépare lui-même (`drawPrep.cpp:1545-1548`) ou vole (`AwaitHead`) ; garde-fou
  2 ms → `ProcessCommands` (`:1588-1590`) qui prend le verrou des voies (interblocage si un verrou lecteur est
  tenu).

Listes de retrait : `DeferRetire` (`commandScheduler.cpp:1016-1044`, tick de la voie la plus jeune ouverte en
concurrent) ; `DescriptorHeap::m_pending_pools` (sans verrou) ; veilles de l'anneau (sans verrou) ;
`m_pending_operations`/`m_held` (`m_operation_mutex`) ; pipelines remplacés détruits après 60 s d'horloge
(`pipelineCache.cpp:4114`, `4417-4423`).

## 3. Conclusions

### (a) Condition du chemin rapide sûr

Prérequis : C1, C2, C3, C5, C6, W5 faits ; reste de §2.2 atomique ou par voie ; verrous lecteurs de §2.3.
Ensuite, un DrawIndex est rapide si et seulement si, sous les verrous lecteurs tenus jusqu'à la fin de
l'enregistrement :

1. Dessin publié : `DrawFlagPublished`, sans `DrawFlagInheritInstances` avec `m_pending_num_instances` non
   vide, sans instantané de registres (`graphicsRun.cpp:3796-3807`).
2. Emplacement prêt : `window.HeadDone()` avant la prise (ni `TryClaimHead`, ni vol, ni attente).
3. Préparation acceptée : `prepared.ok`, `DrawPrep::Validate` vrai (`renderDraw.cpp:2029-2030`) ; plan valide
   et `hw_checks_quiet` (`:806-807`) ; `PlannedPipeline != nullptr`.
4. Aucune opération de cible (`MayRunTargetOperation(hw) == false`, `renderDraw.cpp:1287-1297`) ; pas
   d'index 8 bits ; pas de redémarrage Scan ; `m_mrt_split_slots == 0`.
5. Cibles d'une continuation (`m_run_active`) ou d'une réutilisation (`DrawRunAcquireCandidate == On`,
   `:3180-3229`).
6. Textures : succès de mémo (`HitImage`, `textureBindingMemo.cpp:244-253`), `!entry.dcc`, `RefreshIsNoOp`
   (`:165-179`) ; transitions `TransitIsNoOp`.
7. Tampons en lecture seule et propres (mémo `bufferCache.cpp:4012-4066` ou `RelaxedNothingToUpload`) ; aucun
   hors flux ; aucune liaison écrite ; `UnregisterEpoch()` inchangée.
8. Aucun mémo d'échantillonneur raté, pas de réduction d'occlusion due, pas de `Handle()`/`Drain`.
9. Rien de mondial : `PopReadyOperations` sauté ; pas de `BufferFlush` dû ; pas de fin de tranche.

Échec avant tout enregistrement → lâcher les verrous lecteurs, prendre le verrou global, chemin normal. Les
décisions 5 à 7 doivent précéder toute commande enregistrée.

### (b) Part des 8 µs qui resterait sous verrou

Non mesurable dans la build MC (`KYTY_CP_COMMIT_STATS` exige `KYTY_CP_DIAGNOSTICS=ON`). Estimation : pour un
dessin qui passe (a), rien sous le verrou exclusif des voies ; 0,5 à 1 µs restent sérialisés si `m_lock` reste
un spinlock exclusif (2 à 4 prises par dessin), ~0 avec un verrou lecteur. Pour l'ensemble, tout dépend de la
part qui passe (a) : à mesurer avec le compteur « aurait été rapide » (`KYTY_CP_PAR_FASTCHECK=1`). L'objectif
« temps tenu divisé par 3 » demande qu'environ 70 % des DrawIndex passent.

### (c) Risques de perte du GPU et détection par `KYTY_CP_PAR_VERIFY`

| Risque | Mécanisme | Détection |
|---|---|---|
| Plage de flux réécrite en vol (C1) | deux voies dans `Map`/`Commit` ; plage datée d'un tick d'une autre voie | registre des plages (voie, tick, début, fin) : chevauchements vivants, recyclages avant `IsFree` |
| Jeu de descripteurs donné deux fois (C2) | `--batch.size` concurrent ; pool remis à zéro trop tôt | table jeu → (voie, tick) ; au retrait d'un pool, tick ≥ max des ticks de ses jeux |
| Mauvais `oldLayout` / liaison effacée (C3, C4) | `ResetBindings` d'une autre voie ; ordre d'enregistrement ≠ ordre de soumission | journal par voie (image, layout d'entrée, de sortie) comparé à la fusion ; étendre `NoteLaneBuffer` (`commandScheduler.cpp:778`) aux images et cibles |
| Objet libéré encore référencé | `DeleteBuffer`/`FreeImage`/`DeferOperation` ; `erase` de `m_slot_images` hors verrou | tick de dernière liaison par objet ; compter les retraits plus tôt, ou pendant un verrou lecteur |
| Écriture GPU non vue entre voies | barrières dans le seul tampon de la voie qui écrit | registre étendu aux images ; « dependency hazards » / « order violations » à 0 |
| Validation périmée | certificat et mémos lus, puis transition d'une autre voie | revalidation brève sous verrou après un dessin rapide (générations, époques, signatures) : compteur « fast draw stale » |
| Seul écrivain des bits sale rompu (`memoryTracker.h:44-49`) | deux voies effacent/posent des bits | comme `KYTY_CP_OWNED_COUNTERS=verify` : compter les écritures depuis un fil en chemin rapide |

Non vérifié : durée de vie des programmes de `state.programs` face à une éviction ; atteignabilité en 2b des
chemins qui ne posent que `SetThreadStream` ; contenu exact des rappels différés ; mode dense de `SlotVector`.

## 4. Recensement « aurait été rapide » (`KYTY_CP_PAR_FASTCHECK=1`, défaut 0)

Mesure de (b) sans rien changer au comportement (`drawPrep\fastCheck.{h,cpp}`). Chaque opération DrawIndex ou
DrawAuto exécutée sous le verrou des voies (`ResolveSubmission`) est encadrée par `FastCheck::Begin`/`End`. Le
code du commit marque, **aux décisions qu'il prend de toute façon**, les événements que la condition (a)
interdit : ce sont des bits `thread_local` du fil de l'opération. Aucune recherche supplémentaire, aucune
écriture de LRU, de mémo ni de drapeau : le recensement ne lit rien de plus que le commit et n'écrit que ses
propres compteurs (atomiques, par voie). Désactivé : une lecture d'une constante par point de marquage.

Points de marquage (cause → condition) :

| Condition | Causes | Où |
|---|---|---|
| 1 | `not-published`, `inherit-pending`, `snapshot` | `ExecDrawIndex`/`ExecDrawAuto`, `OpSnapshot` (`graphicsRun.cpp`) |
| 2 | `head-unclaimed` (le commit prépare lui-même), `head-waited` (attente sous le verrou) | `Engine::ReadyHead` |
| 3 | `prep-missing`, `prep-refused` (repli `GetGraphicsPrograms`), `plan-not-quiet` (contrôles matériels exécutés), `pipeline-not-planned` | `RefreshShaders`, `RunDrawChecks`, `ExecutePreparedDraw` |
| 4 | `target-op` (métadonnées, copie profondeur, résolution, copie profondeur→couleur), `index8`, `restart-scan`, `mrt-split` | `DrawIndex`/`DrawAuto`, `ResolvePrimitiveRestart`, `ExecutePreparedDraw` |
| 5 | `run-late-fallback` (continuation perdue en route), `targets-acquired` (`AcquireRenderTargets` appelé : ni continuation ni réutilisation) | `ExecutePreparedDraw` |
| 6 | `texture-miss` (`ResolveTextureFull`), `texture-view-miss` (`FindTexture`), `texture-dcc` (`ApplyDcc`), `transition` (barrière d'image réellement enregistrée) | `descriptors.cpp`, `textureBindingMemo.cpp`, `image.cpp` |
| 7 | `buffer-written` (liaison écrite), `buffer-create`, `buffer-sync` (synchronisation au-delà des sorties précoces), `buffer-stream` (petite lecture sale CPU copiée dans le flux), `buffer-rejoin` | `bufferCache.cpp`, `ExecutePreparedDraw` |
| 8 | `sampler-miss`, `occlusion-flush`, `recorder-drain` | `NativeSampler`, `occlusion.cpp`, `CommandRecorder::Drain` |
| 9 | `pending-ops` (une opération différée exécutée), `submit` (`CommandScheduler::Submit`), `services` (garde-fou 2 ms), `slice-yield` | `commandScheduler.cpp`, `drawPrep.cpp`, `MaybeYieldSlice` |

Classement à la fin de l'opération : **fast** (aucune marque et le dessin a atteint la sélection de pipeline,
ou une opération de cible), **empty** (rien d'enregistré : compte nul, programme absent, rien à dessiner,
pipeline sauté), sinon **failed** : compté à sa **première** cause (la plus petite dans l'ordre ci-dessus, donc
la première condition qui échoue) et, s'il n'a qu'une marque, à sa **cause seule** (les dessins qu'une seule
correction rendrait rapides).

Ligne toutes les 10 s (une voie la publie) :

```
CpPar 10s: 2b fast-draw candidates (10s; condition (a) 1-9, docs-p3-drawindex.md):
  lane B: DrawIndex n (fast n), DrawAuto n (fast n); fast x%, empty n, failed n
    first failure: 2: head-waited n | 5: targets-acquired n | 7: buffer-written n ...
    sole cause: 5: targets-acquired n | 6: transition n ...
```

Limites : les conditions qui portent sur l'état partagé lui-même (verrous lecteurs, §2.1 C1-C6) ne sont pas
des marques : ce sont des prérequis structurels, vrais pour tous les dessins une fois faits. Une marque posée
pendant une opération par un travail qui n'est pas le dessin (un `DrainPreparedDraws`, un service) compte pour
ce dessin. Aucune condition n'a dû être classée « non évaluable » : toutes sont lues à une décision existante.

## 5. Prérequis de P3 (étapes 1 à 3), chacun derrière son interrupteur (défaut 0)

Mesure de départ (session principale, `console-20261115-204841.txt`, Aerial Garden, 20,5 i/s,
`KYTY_CP_PAR_FASTCHECK=1`) : B ~327 000 DrawIndex / 10 s, fast 35–40 % ; C ~485 000, fast 30–32 % ;
causes seules : `buffer-stream` (B ~86 000, C ~140 000), `buffer-written` (C ~111 000), puis
`targets-acquired`, `head-waited`, `texture-dcc`, `submit`, `slice-yield` (quelques milliers) ;
DrawAuto : 0 fast. L'anneau par voie ferait passer B à ~66 %, C à ~61 % ; les tampons écrits ensuite
C à ~85 %.

Module commun `drawPrep\laneShare.{h,cpp}` : lecture des interrupteurs une fois au démarrage (après
l'activation de 2b ; refusés hors 2b), compteurs par voie, ligne toutes les 10 s
« CpPar 10s: 2b lane share (…): A: …; B: …; ». Avec `KYTY_CP_PAR_VERIFY=1` chaque étape se contrôle.

| Interrupteur | Étape | Ce qui change | Auto-contrôle (`KYTY_CP_PAR_VERIFY=1`, 0 attendu) |
|---|---|---|---|
| `KYTY_CP_PAR_LANE_STREAM=1` | C1 | `StreamBuffer::Lane()` : un anneau de flux de 64 Mio par voie B.. (même mémoire que l'anneau commun, créé au premier usage). Passent par lui : `UploadShaderData`/`NativeUpload` (via `GetUtilityBuffer(Stream)`), petites lectures sales CPU (`ObtainBufferNow`), index 8 bits, paramètres du tiler. Le mémo de liaison « flux » n'est repris que si sa copie est dans l'anneau de la voie. FASTCHECK ne compte plus `buffer-stream` | `stream overlaps` : une plage remise chevauche l'arc des plages encore en vol (tick non libre) ; `stream foreign` : un anneau servi à un fil d'une autre voie |
| `KYTY_CP_PAR_LANE_DESCRIPTORS=1` | C2 | `DescriptorHeap::Commit` va au tas de la voie du fil (pools, lots, retrait au plus grand tick de ses jeux, donc au tick de la voie) | `descriptor double` : un jeu remis alors que le tick de sa remise précédente n'est pas libre (table commune à tous les tas) ; `descriptor early` : un pool retiré sous le plus grand tick de ses jeux |
| `KYTY_CP_PAR_LANE_PREREQS=1` | C3, W1/W2, W5, W12, W13 | C3 : drapeaux de liaison d'image par voie (`Image::Binding()`), les décisions du cache (image liée ou cible quelque part, `needs_rebind`) regardent toutes les voies ; W5 : profondeur d'`UploadBatch` par voie ; W13 : `CommandScheduler::Active()` lit l'enveloppe du fil ; W1/W2 : contacts LRU d'images et de tampons d'un fil de voie sous le verrou mis de côté et appliqués à la fin de l'opération (encore sous le verrou) et avant toute collecte (entrée périmée : abandonnée, comptée) ; W12 : demande de vidage anticipé par voie | `touches stale` est informatif (entrées libérées entre-temps) |
| aucun | W8, R8 | `g_dynamic_state_emitted` par fil (la différence est toujours prise dans un même dessin, sur son fil) ; `m_unregister_epoch` atomique (relaxé) : mêmes valeurs | — |

Rien de cela ne sort une opération du verrou des voies : ce sont des prérequis du chemin rapide (§3 (a)).
Restent à faire avant lui : C4 (états d'image), C5 (`RenderMutex` partagé), C6 (identité de voie rapide),
le reste de §2.2–2.3, puis le chemin rapide lui-même.

Limites connues : la copie de flux différée (`KYTY_STREAM_COPY_DEFER`, préréglage U59) est déjà coupée
dès que le recorder a plusieurs flux (2a/2b) ; le mémo de plages (`RecordRangeFact`) et le mémo de
liaison restent communs (W3) ; `KYTY_CP_OWNED_COUNTERS=1` (préréglage U59) garde ses compteurs non
atomiques (W10) : sans effet tant que tout est sous le verrou.

## 6. Chemin rapide (`KYTY_CP_PAR_FASTDRAW=2`) : pourquoi il n'est pas codé, et ce qui le débloque

Mesure (session principale, Aerial Garden, C1/C2/prérequis actifs, contrôles à 0) : B 64,2 % et C
60,7 % de dessins « fast » au sens de §3 (a). Causes seules restantes : B `targets-acquired` ~12 900,
`texture-dcc` ~3 700, `head-waited` ~2 600, `submit` ~1 600, `slice-yield` ~1 500 ; C `buffer-written`
~131 000, `targets-acquired` ~11 900, `submit` ~4 200, `texture-dcc` ~3 200. Repère : U59 sans 2b
28,3 i/s en déplacement, MC 2b à tour de rôle 23,4.

### Ce que la relecture du code a établi (bloquants du chemin rapide tel que spécifié en §3 (a))

1. **La surcouche d'état d'image échange l'état en place pour toute la durée de l'opération.**
   `Image::CpParTouchSlow` (`image\image.cpp`) met l'état de la voie plus jeune dans `backing.state`
   de l'image commune, et `Image::CpParEndOp` ne le rend qu'à la fin de l'opération. Sous le verrou
   global c'est sûr (une seule opération à la fois). Hors verrou, une autre voie qui lit l'image pendant
   ce temps (même un `Transit` sans effet, un `DrawRunImagesUnchanged`) lit l'état de l'autre voie :
   mauvais `oldLayout`, perte du GPU possible. `CpParTouch` est appelé par **toute** lecture ou
   écriture d'état d'image d'une opération de voie plus jeune, y compris les transitions sans effet.
   C4 « seulement si TransitIsNoOp » ne suffit donc pas : il faut qu'un dessin rapide ne lise ni
   n'écrive aucun état d'image, ou que la surcouche cesse d'être en place (accès à l'état par un
   accesseur qui redirige vers la surcouche de la voie du fil).
2. **Des replis existent au milieu du dessin, après la préparation des liaisons.** Exemple :
   `ExecutePreparedDraw` (`renderDraw.cpp`, bloc « late fallback ») : une continuation dont les images
   ont changé reprend `DrawRunTargets` (acquisition des cibles, transitions, `FindImage` possible)
   après `PrepareBindings`. De même `RebindAfterJoin`, et les recherches de tampons qui créent ou
   synchronisent pendant `PrepareBindings`. Décisions et mutations sont entremêlées sur ~3 000 lignes
   (`PrepareDrawRenderState`, `PrepareBindings`, `PrepareGraphicsBindings`, `AcquireVertexBuffers`) :
   il n'existe pas de point « tout est décidé, rien n'est encore modifié ». Évaluer (a) « avant tout
   enregistrement » demande donc un prédicat séparé, en lectures pures, qui reproduise ces décisions à
   partir du plan de liaisons des ouvriers (`BindingPlan::vertex/pixel.ranges`, `texture_hashes`,
   `samplers`, `vertex_ranges`, `pipeline`).
3. **Un repli à mi-dessin hors verrou n'a pas de sortie sûre.** Relâcher le verrou partagé pour prendre
   le verrou global au milieu d'un dessin laisse d'autres opérations changer des images et tampons déjà
   résolus par ce dessin ; une montée de verrou partagé vers exclusif interbloque dès que deux voies
   rapides la demandent.
4. **Les mémos du cache de tampons sont écrits même sur un succès** (`ObtainReadBindingMemo` : copie
   « chaude », époque, `touched` ; `RecordBinding`/`RecordRangeFact` sur un échec) : écritures
   concurrentes entre voies rapides sur un même emplacement, donc entrée déchirée, donc mauvais tampon
   lié.

Conclusion : coder `KYTY_CP_PAR_FASTDRAW=2` maintenant donnerait soit un chemin rapide qui peut perdre le
GPU (surcouche en place, replis à mi-dessin), soit un chemin si restreint qu'il ne gagne rien. Il n'est
pas codé. Les prérequis C5/C6/C7, W6/W7/W9–W11, W14, R1–R6/R12/R13 n'ont d'effet qu'avec lui (sous le
verrou global ils ne changent rien et ne peuvent pas être vérifiés) : ils seront faits avec lui.

### Conception proposée (F2), à valider avant de coder

- **Verrou de structure S** (`std::shared_mutex`) : toute opération sous le verrou des voies le prend en
  exclusif ; un dessin rapide le prend en partagé du début à la fin. Les seules concurrences restantes
  sont alors rapide contre rapide.
- **Prédicat P avant tout enregistrement**, lectures pures sous S partagé, à partir du plan de liaisons :
  emplacement prêt, préparation acceptée, plan `hw_checks_quiet`, pipeline prévu, aucune opération de
  cible, cibles d'une continuation (images inchangées), chaque texture trouvée dans le mémo de
  l'exécuteur avec sa vue, **aucun accès à l'état d'image** (continuation avec `keep_images`, ou voie la
  plus ancienne, ou surcouche non en place), chaque tampon lié existant et propre (page présente dans
  `m_page_table`, `IsInBounds`, aucune page sale CPU), aucune liaison écrite.
- **Pendant le dessin rapide** : mémos de tampons en lecture seule (copie chaude, époque et
  `RecordBinding` sautés), contacts LRU différés (fait), `PopReadyOperations` sauté, `RenderMutex` en
  partagé, identité « fil GPU » de voie rapide aux seuls sites de lecture (`RelaxedDirtySnapshot`,
  `RelaxedNothingToUpload`, `IsGpuCleanForRead`, `ObtainReadBindingMemo`), compteurs non atomiques par
  voie. Une page devenue sale côté CPU après P (écriture invitée concurrente) n'est pas téléversée par ce
  dessin : équivalent à un dessin exécuté avant l'écriture ; le dessin lent suivant téléverse.
- **Filet** : chaque point de marquage FASTCHECK (création, synchronisation, liaison écrite, transition,
  surcouche, soumission…) devient en mode rapide un arrêt explicite (« fast path invariant violated at
  … ») plutôt qu'une mutation sans verrou ; avec `KYTY_CP_PAR_VERIFY=1`, revalidation de P sous le verrou
  global après le dessin (« fast draw stale ») et registre des dépendances étendu aux images et cibles.

Pour mesurer ce que P laisserait passer **sans rien changer**, FASTCHECK compte désormais aussi
`image-adopt` (état adopté sans barrière) et `image-overlay` (une opération de voie plus jeune touche
l'état d'une image : échange en place). Si `image-overlay` est fréquent sur B et C, la surcouche non en
place (point 1) est le vrai prochain chantier, avant le chemin rapide.

### buffer-written (cause seule n° 1 de C, ~131 000 / 10 s) : proposition, non codée

Une liaison écrite fait aujourd'hui (`ObtainBufferNow`/`ObtainWrittenBuffer`, `bufferCache.cpp`) :
synchronisation écrite (souvent sautée si la plage appartient déjà au GPU, `WrittenSyncSkip`),
`PreserveImagesForGpuWrite`, `MarkContentWritten`, ajout à `m_gpu_modified_ranges`,
`CleanVerdict::Invalidate`, `NoteBufferContentWrite` (`m_write_ticks`, vidage anticipé),
`ForgetKnownFills`. Classe « écriture stable » proposée, sûre si chaque test est exact : plage déjà
entièrement au GPU (`IsRangeGpuOwned`), déjà contenue dans `m_gpu_modified_ranges`, aucun remplissage
connu, aucune image sur la plage (test pur de ce que `PreserveImagesForGpuWrite` ferait). Il ne reste
alors que `MarkContentWritten` (compteur de révision, en `fetch_add` atomique) et le tick d'écriture
(journal par voie ajouté sous le verrou global à la prochaine opération lente, vidé avant toute lecture
de `m_write_ticks`, comme les contacts LRU différés). Le vrai risque n'est pas la perte du GPU mais
l'**ordre entre voies** : une voie plus jeune qui écrit une plage qu'une plus ancienne lit encore
(« dependency hazards » / « order violations » de `NoteLaneBuffer`, déjà comptés). À garder sous
`KYTY_CP_PAR_VERIFY=1` et à n'activer que si ces compteurs restent à 0 en simultané.

### Surcoût de 2b (23,4 contre 28,3 i/s)

À traiter avant ou avec le chemin rapide. Piste peu coûteuse : `LaneSegmentOldest` prend
`m_operation_mutex` à chaque opération (`commandScheduler.cpp`, appel dans `graphicsRun.cpp`, W14),
mutex aussi pris par les soumissions et les opérations différées de toutes les voies ; le rendre sans
verrou demande de vérifier que `m_turn` n'avance que sous le verrou des voies (non vérifié). Un profil
échantillonné de la build MC 2b à tour de rôle contre U59 dirait où vont les 5 i/s.
