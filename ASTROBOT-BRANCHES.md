# Astro Bot branches of this fork

> Prepared with AI assistance (Claude); built and tested by the repository owner as stated per branch; not submitted upstream.

This fork of [Jetsku/KytyPS5](https://github.com/Jetsku/KytyPS5) keeps the changes made while running ASTRO BOT
(PS5) on a Windows PC with this emulator. Base of every branch: `744705ee` ("Release notes: int21"), the `main` of
Jetsku/KytyPS5 when the fork was made. The default branch (`main`) is left unchanged.

Test machine: Intel Core i5-12400F (Intel host, so the AMD instruction patches are active), NVIDIA RTX 3060,
Windows 11, 1080p, 30 fps mode. No game data, keys or firmware are included in any branch; guest (eboot) addresses
quoted in comments are only locations in the game's code.

All the switches below are environment variables read once at start-up. Unless written otherwise they are **off by
default**, and with the switch off the behaviour is the one of the base.

## Summary

| Branch | Kind | Switch | Status |
|---|---|---|---|
| `astrobot-pc` | the playable configuration (5 fixes below in one commit) | see below | compiled; tested in game on the Jetsku U59 base |
| `astrobot-experiments` | archive of all experiments (125 commits) | many, all off by default | used in game for measurements; state of 17 Nov, replaced by `astrobot-experiments-22-11` |
| `astrobot-experiments-22-11` | `astrobot-experiments` + 26 commits (17-22 Nov 2026) | many, all off by default unless stated | the build played on the test machine since 22 Nov; see the last section |
| `fix/esc-exits-fullscreen` | fix | none (always on) | compiled; tested in game on the Jetsku U59 base (as part of `astrobot-pc`) |
| `feature/app0-overlay-subst` | feature | `KYTY_APP0_OVERLAY`, `KYTY_APP0_SUBST` | compiled; tested in game on the Jetsku U59 base (as part of `astrobot-pc`) |
| `fix/windows-thread-names` | fix | none (Windows only) | compiled; tested in game on the Jetsku U59 base (as part of `astrobot-pc`) |
| `diag/render-pass-stats` | diagnostic | `KYTY_CP_COMMIT_STATS` (needs `KYTY_CP_DIAGNOSTICS=ON` at build time) | compiled; in the `astrobot-pc` source, not run with the switch in this isolated form |
| `fix/amd-rsqrt-on-intel` | fix / speed | `KYTY_AMD_RSQRT=exact\|fast\|native` (default `exact`) | compiled; tested in game on the Jetsku U59 base (`fast`, as part of `astrobot-pc`) |
| `fix/kernel-sema-targeted-wake` | fix / speed | none (always on) | compiled; the same change was used in game on the Jetsku U59 base (experiments builds); no isolated measurement |
| `diag/wait-graph` | diagnostic | `KYTY_WAIT_GRAPH=1` | compiled; the same instrumentation was used in game on the Jetsku U59 base (experiments build, 16 Nov 2026) |
| `diag/frame-path` | diagnostic | `KYTY_FRAME_PATH=1` | compiled; the same instrumentation was used in game on the Jetsku U59 base (experiments build, 15-16 Nov 2026) |
| `perf/host-mutex-srw` | speed | `KYTY_HOST_MUTEX_SRW=1` | measured in game in the working branch (+1.7 fps, 5% low 22 vs 16-20); cherry-picked cleanly here, not compiled on its own |
| `fix/astro-cb-scale-01019` | fix (crash) | `KYTY_ASTRO_CB_SCALE` (x2 by default) | crash of the EU 1.019 build fixed in game in the working branch; cherry-picked cleanly here, not compiled on its own |
| `fix/upload-last-reused` | fix | `KYTY_UPLOAD_LAST_REUSED=1` | real defect found by code reading (17 Nov 2026); compiled in the working branch; not yet measured in game |

The single-fix branches carry the same code as in `astrobot-pc` / `astrobot-experiments`, with English comments and
console messages (the console messages of `KYTY_APP0_SUBST` were translated; the reports of the two `diag/` switches
stay in French). They were each compiled on their own (Release, clang-cl, Windows); they were not run one by one in
the game.

## `astrobot-pc`

- **Content**: `744705ee` + `d87d3df3`, one commit grouping the five fixes `fix/esc-exits-fullscreen`,
  `feature/app0-overlay-subst`, `fix/windows-thread-names`, `diag/render-pass-stats`, `fix/amd-rsqrt-on-intel`
  (original French comments).
- **Use**: the configuration the game is played with on the test machine, with `KYTY_AMD_RSQRT=fast` and a
  `KYTY_APP0_SUBST` rule file that lowers shadow distances, built with `KYTY_CP_DIAGNOSTICS=OFF`.
- **Status**: compiled; tested in game on the Jetsku U59 base (full playthrough possible with a controller,
  20-30 fps depending on the level).

## `astrobot-experiments`

- **Content**: the whole working branch (`cp-voies`), `744705ee` + 125 commits, kept as is (French commit messages).
  Every switch is off by default. It includes, among others: per-pass GPU-thread timing (`KYTY_PASS_STATS`), a frame
  generation prototype with FSR 3 (`KYTY_FSR3_FG`), parallel command-list translation lanes (`KYTY_CP_PAR`,
  `KYTY_CP_PAR_CONCURRENT`, ...), game function hooks (`KYTY_GAME_HOOKS`), native rendering of selected passes
  (`KYTY_NATIVE*`), host import of guest memory (`KYTY_HOST_IMPORT`), the two diagnostics `KYTY_FRAME_PATH` and
  `KYTY_WAIT_GRAPH`, and guest-thread wait tweaks (`KYTY_GUEST_YIELD_BACKOFF`, `KYTY_GUEST_SEMA_LIGHT`,
  `KYTY_GUEST_SEMA_SPIN_US`, `KYTY_APPLYSPAN_FAST`, `KYTY_HOT_PAGE_*`, `KYTY_PAGE_HOST_LOCK_SLICES`).
- **Honest result**: most of these were measured **without a frame-rate gain** on the test machine (for example
  `KYTY_GUEST_SEMA_SPIN_US=30`: 23.9 -> 24.2 fps, within noise; native passes slower than the emulator path; FSR 3 frame
  generation unreliable). The branch is an archive and a source of instrumentation, not a recommended build.
- **Rewritten for publication**: the original bytes of the game functions hooked by `KYTY_GAME_HOOKS`
  (`src/loader/gameHooksCapture.cpp`, `src/loader/gameHooksTrace.cpp`, used to check a hook target before patching it)
  are game code and were replaced by empty lists in every commit, so those hooks refuse to install until the bytes are
  filled in locally. Because of this rewrite the commit ids from "KYTY_GAME_HOOKS : accroche de fonctions..." on
  differ from the original working branch. `KYTY_NATIVE_DUMP` still defaults to an output folder of the test machine:
  set `KYTY_NATIVE_DUMP_DIR` before using it.
- **Status**: compiled and used in game on the Jetsku U59 base (measurement builds, before the rewrite); for this fork
  only the two rewritten files were recompiled, not the whole branch.

## `fix/esc-exits-fullscreen`

- **Goal**: F11 and Alt+Enter put the window in desktop fullscreen; Escape now also brings it back to windowed mode
  (only when fullscreen, not on auto-repeat). The key is still forwarded to the game.
- **Astro Bot problem**: with a controller-only setup there was no obvious key to leave fullscreen.
- **Evidence**: symptom; used daily in the playable configuration.
- **Switch**: none.
- **Status**: compiled; tested in game on the Jetsku U59 base (as part of `astrobot-pc`).

## `feature/app0-overlay-subst`

- **Goal**: change small files of the game without modifying the dump.
  - `KYTY_APP0_OVERLAY=<folder>`: a file under `/app0` is served from this folder when a copy exists there.
  - `KYTY_APP0_SUBST=<rules file>`: same-length byte replacements, or a clamp of the number after a tag, applied in
    memory to data read from `/app0` files (`KernelRead`, `KernelPread`, AMPR host reads).
- **Astro Bot problem**: choosing the start level through the game's argument file, replacing button-prompt
  textures (Xbox A/B/X/Y instead of PlayStation symbols), and testing lighter visual settings stored in text files.
- **Evidence**: used during the port (button prompts, level selection) and in the playable configuration
  (shadow-distance clamp). Measured effect of such visual reductions on the frame rate: at most about 1 fps.
- **Switch**: `KYTY_APP0_OVERLAY`, `KYTY_APP0_SUBST` (both off when unset).
- **Status**: compiled; tested in game on the Jetsku U59 base (as part of `astrobot-pc`).

## `fix/windows-thread-names`

- **Goal**: `Profiler::SetThreadName` also sets the Windows thread description, and each guest thread gets
  `jeu:<name given by the game>` ("jeu" = game) when it starts.
- **Astro Bot problem**: per-thread CPU measurements (which thread saturates a core) were impossible with anonymous
  thread ids.
- **Evidence**: used by the per-thread CPU sampling done during the port.
- **Switch**: none (Windows only).
- **Status**: compiled; tested in game on the Jetsku U59 base (as part of `astrobot-pc`).

## `diag/render-pass-stats`

- **Goal**: adds a "Passes" line to the `KYTY_CP_COMMIT_STATS` report: draws, pass instances, commit and gap time per
  frame, grouped by render-pass shape (size, layers, color attachments, depth), top 20.
- **Astro Bot problem**: find which passes (shadow maps, main pass, post-processing) cost the most command-processor
  time; the command processor thread (about 4,300 draws per frame) was one of the main costs found on the test machine.
- **Evidence**: diagnostic only.
- **Switch**: `KYTY_CP_COMMIT_STATS`, compiled only with `KYTY_CP_DIAGNOSTICS=ON` (the CMake default).
- **Status**: compiled; part of the `astrobot-pc` source, not run with the switch in this isolated form.

## `fix/amd-rsqrt-on-intel`

- **Goal**: on Intel hosts the AMD instruction patch replaces each 128-bit `VRSQRTPS` with an exact 1/sqrt in double
  precision (MXCSR saved and restored), which is costly in hot loops. `KYTY_AMD_RSQRT=fast` uses single-precision
  `VSQRTPS` + `VDIVPS` (a few ulp from exact, better than the ~12 bits AMD's `VRSQRTPS` guarantees);
  `KYTY_AMD_RSQRT=native` keeps Intel's own `VRSQRTPS`.
- **Astro Bot problem**: the game executes this instruction in hot code; the exact replacement cost frame rate on
  Intel.
- **Evidence**: 12 Nov 2026, 1080p, 30 fps mode, 24 main levels measured one by one: with `fast`, 11 levels at 30 fps
  (and 5 just below) instead of 7 at 30 fps before; for example Aerial Garden 21.9 -> 23.4 fps, Sky Aztec
  25.4 -> 26.9, Pirate Island 26.1 -> 29.4, Mana Tree 26.5 -> 28.7.
- **Switch**: `KYTY_AMD_RSQRT=exact|fast|native` (default `exact` = unchanged).
- **Status**: compiled; tested in game on the Jetsku U59 base (`fast`, as part of `astrobot-pc`).

## `fix/kernel-sema-targeted-wake`

- **Goal**: `KernelSemaPrivate::Signal` called `SignalAll` after `WakeWaiters`, so every waiting thread woke on each
  signal and contended for the mutex. Each waiter now has its own condition variable, signalled only when it is
  served (or cancelled / deleted). Order, counts and results are unchanged.
- **Astro Bot problem**: the game signals kernel semaphores about 55,000 times per second (about 2,000 per frame).
- **Evidence**: part of the experiments builds from 13 Nov 2026 on; no isolated frame-rate measurement was recorded,
  and later measurements showed that semaphore traffic is not what sets the frame rate on the test machine.
- **Switch**: none.
- **Status**: compiled; the same change was used in game on the Jetsku U59 base; this branch was not run on its own.
  (The official KytyPS5 already has an equivalent change, commit `a0268f1d`.)

## `diag/wait-graph`

- **Goal**: `KYTY_WAIT_GRAPH=1` times the blocking waits of guest threads (kernel and POSIX semaphores, pthread
  condition variables and contended mutexes, event flags, event queues, `sceAgcSuspendPoint`, sleeps), names the
  waker of each wait from the last signal of the same object, counts yield loops by guest address, and prints every
  10 s the main thread and the three other guest threads that wait most, with the "A waits for B waits for C" chains.
- **Astro Bot problem**: find what the game's main thread waits for.
- **Evidence**: 16 Nov 2026 in Aerial Garden: the main thread waits only ~8 ms per frame (18 %) and computes ~82 % of
  the time; the largest wait is ~1,300 semaphore round trips per frame with one helper thread (4-5 ms per frame).
- **Switch**: `KYTY_WAIT_GRAPH=1` (off: each hook is one inline flag test).
- **Status**: compiled; the same instrumentation was used in game on the Jetsku U59 base (experiments build); this
  isolated branch owns the main-thread marker itself (no dependency on `diag/frame-path`) and was not run on its own.

## `diag/frame-path`

- **Goal**: `KYTY_FRAME_PATH=1` prints every 10 s the mean and 95th percentile of each stage of a frame's graphics
  submissions (queue -> command-processor translation -> recording -> `vkQueueSubmit` -> GPU completion), label
  latencies, frame-fence holds, `sceAgcSuspendPoint` waits and the share of the main thread's time spent in yields,
  semaphores, condition variables, event queues, sleeps and access faults.
- **Astro Bot problem**: tell whether the GPU, the command processor or the game's own threads set the frame rate.
- **Evidence**: 15-16 Nov 2026 in Aerial Garden: queue before translation 18-20 ms, translation 7-8 ms, GPU 3-5 ms,
  `sceAgcSuspendPoint` never blocks; the main thread runs game code ~75 % of the time. Conclusion on that machine:
  the limit is the CPU time of the game's threads under the emulation load. The final version costs no measurable
  frame rate when on.
- **Switch**: `KYTY_FRAME_PATH=1`.
- **Status**: compiled; the same instrumentation was used in game on the Jetsku U59 base (experiments build); this
  isolated branch was not run on its own.

## Added on 17 Nov 2026

Measured on the test machine (i5-12400F, RTX 3060): the game is limited by its own main thread (about 16 ms of game
code per frame plus task-scheduler waits), not by the GPU (60-70 % busy); lowering the resolution gives nothing. A
"bimodal" behaviour was seen: with identical settings a run gives ~20 fps or ~24.8 fps.

- `perf/host-mutex-srw`: host mutexes and condition variables on SRW locks. **Measured gain** (3 alternating pairs).
- `fix/astro-cb-scale-01019`: the command buffer size patch extended to Astro Bot 01.019.000 (crash at guest pc
  +0x4a0c3e). **Measured**: no cost; the crash was not seen again in the following test runs.
- `fix/upload-last-reused`: see the commit message. Lost saving only, never a wrong image.
- `astrobot-experiments` gained 17 commits, **not measured in game yet unless stated** (all off by default):
  - `KYTY_HOST_MUTEX_SRW`, `KYTY_ASTRO_CB_SCALE` 01.019 (measured, as above);
  - `KYTY_MAIN_CPU_SAMPLER`: sampling profiler of the guest main thread (frame cadence histogram, long frames,
    callers of its sleeps and waits, Windows detail, SMT placement, top 10 of the guest libc module);
  - `KYTY_FLUIDITE=1`: a frame-drop journal for long play sessions (episodes with probable cause, per-frame black
    box, JPEG thumbnails of the game image, on-demand GPU frame reports, camera read through a pointer chain,
    F9 / L3+R3 marks, a capped folder with rotation of the oldest captures);
  - audit of 17 Nov: `KYTY_SEMA_HANDOFF` (direct hand-off of guest semaphore signals), `KYTY_COND_SUBMS` /
    `KYTY_COND_STATS`, `KYTY_NO_THROTTLE`, `KYTY_THREAD_LAYOUT_LOG` / `KYTY_IDEAL_LAYOUT` (deterministic ideal
    processors for the heavy threads, aimed at the bimodal behaviour), `KYTY_CP_BLOCKED_WAKE_GEN`,
    `KYTY_DRAW_PREP_SPIN_ADAPT`, `KYTY_UPLOAD_LAST_REUSED`, `KYTY_PUSH_DELTA` (push descriptors re-encode only the
    changed elements), `insert-after:` / `insert-before:` / `add:` rules for `KYTY_APP0_SUBST`; a few always-on
    equivalent changes (per-thread condition variable for semaphore waits, pthread signal outside the lock, cached
    QPC frequency, ...).
  - Also measured without a gain and kept only as archive: `KYTY_MAIN_YIELD_SPIN`, `KYTY_MAIN_THREAD_CORE`,
    `KYTY_MAIN_THREAD_PRIORITY`.
- Not published here because they depend on earlier commits of the working branch: the sampler, the frame-drop
  journal and the `KYTY_APP0_SUBST` insertion rules exist only in `astrobot-experiments`.

## Added on 22 Nov 2026: `astrobot-experiments-22-11`

- **Content**: `astrobot-experiments` (`ce4d0b4b`) + 26 commits of 17 to 22 Nov 2026 from the working branch, kept as
  they are (French commit messages and comments). It is a fast-forward of `astrobot-experiments` and **replaces it**:
  `astrobot-experiments` stays at `ce4d0b4b` as the state of 17 Nov.
- **Use**: the emulator the game is played with on the test machine since 22 Nov 2026 (own profile-guided build),
  with the switches marked "in use" below. About 29.6 fps on average in a real session, 1080p, 30 fps mode (with a
  1080p render patch of the game, not part of this repository; see `docs/RENDER-1080P.md` in AstroBot-PC).
- **Third-party code**: SMAA (Jimenez, Echevarria, Sousa, Gutierrez), MIT licence, official sources
  (github.com/iryoku/smaa `71c806a8`) in `3rdparty/smaa`, licence copy in `LICENSES/SMAA-MIT.txt`.
- **Status**: compiled (Release, clang-cl, Windows, PGO) and used in game on the Jetsku U59 base. Same as before:
  every switch is off by default unless stated; with a switch off the behaviour is the one of `astrobot-experiments`,
  except the always-on changes listed at the end.

### In use on the test machine

| Switch | What it does | Evidence |
|---|---|---|
| `KYTY_PAD_ACTIVE_FIRST=1` | The pad on which a button is pressed becomes player 1. Pads are given to players in the order Windows reports them; with a two-slot Xbox wireless receiver (second pad off), the pad in use could become player 2, which Astro Bot's title screen ignores. | Real problem after a reboot (22 Nov), fixed in game |
| `KYTY_PRESENT_AA=smaa\|smaa_ultra` | SMAA 1x (High or Ultra preset) on the game image just before its copy to the swapchain. Unset or 0: no resource, no command. HDR, float or black images left as they are. `KYTY_PRESENT_AA_TIMING=1` prints the GPU cost of the three passes every 10 s. | 0.73-0.85 ms per frame on a 4K image; not measured on the 1080p image. `smaa_ultra` in use |
| `KYTY_QUIT_MENU=1` | Esc opens a "Quit Astro Bot?" box in the ImGui overlay (mouse or controller). While it is open, input does not reach the game. "Quit" sends the same event as the window's close button. | In use; checked by the owner |
| `KYTY_AVPLAYER_STEADY=1` | Video frames advance by a regular whole number of frames per game call (the gesture thumbnails, 60 fps videos read by a 30 fps game, advanced 1 to 4 frames per call). | Steps over 40 ms: 12 instead of 24-45 in the first video of Sky Aztec |
| `KYTY_AVPLAYER_SKIP_LATE` (**on by default**, 0 = old behaviour), `KYTY_AVPLAYER_VIDEO_THREADS` (default 4) | Each game request gets the frame of the present time (late frames skipped); multi-threaded H.264 decoding. Statistics printed when each video stops. | At 30 fps the boot logo played at half speed, 8.5 s behind the sound; mean lateness 4,304 ms → 11 ms |
| `KYTY_ASTRO_CB_MIB=a,b,c,d,e` | Size in MiB (1-32) of each of Astro Bot's five draw-thread command buffers, overriding `KYTY_ASTRO_CB_SCALE`. x3 for all five runs the game out of its fixed graphics heap (Mana Tree, guest pc +0x4a7e0a). | `14,14,12,12,12` in use; the +0x4a0c3e crash became rare, not impossible |
| `KYTY_DRAW_PREP_TAKEOVER=stats\|<µs>` | `Thread_Gpu` prepares itself a draw-preparation window head that a worker has held longer than `<µs>`; `stats` only measures head waits. | Head waits 26-68 ms/s in Sky Aztec, but no fps gain beyond noise; left at 50 by the owner's choice |
| `KYTY_FLUIDITE` journal v2 to v5.2 | Frame-drop journal: v2 continuous episodes, `trace.csv` every 0.5 s, automatic stacks (`KYTY_FLUIDITE_ECHANT`, `KYTY_FLUIDITE_PILES`); v3 GPU idle time and corrected cause, `manette.csv`, `points.csv`, `niveaux.csv`; v4 `fichiers.csv` (first access to each data file) and `commandes.csv` (pad state read by the game, frame by frame); v5 `secondes.csv` (one row per second, including NVML GPU data) and `chargements.csv` (one row per loading); v5.1 loading duration from the read activity; v5.2 no rows lost at start-up. | In use; diagnostic (reads once per second, nothing per draw) |

### Test and diagnostic tools

| Switch | What it does |
|---|---|
| `KYTY_PAD_FILE=<file>` | Internal pad pilot for unattended tests: a thread reads `maintenir` / `touche` / `attendre` / `boucle` commands and drives the host pad directly (works without focus and with the session locked). Later commands guided by the camera position (`aller`, `tenir`, `passer`, `bouton`, `attendre-pres`, `attendre-calme`, `rejoindre`) and frame-by-frame replay of `commandes.csv` (`rejouer`). `KYTY_PAD_SIDE=1` flips the stick side of `aller`. Needs `KYTY_FLUIDITE_CAMERA` (a pointer chain specific to the game version) for the guided commands. |
| `KYTY_RT_SIZE_LOG=1` | Prints each new color render-target size once (at which resolution each step renders). |
| Console lines `VideoOut:` (always on) | Size of the display buffers declared by the game, the output resolution code reported to it, and for `sceVideoOutSetBufferAttribute2` / `RegisterBuffers2` the return address and game-code addresses found in the stack (used to find where the game picks 3840x2160). |
| `KYTY_VIDEO_OUT_RESOLUTION=<n>` | Forces the output resolution code reported by `sceVideoOutGetOutputStatus`. **No effect** on Astro Bot: it declares 3840x2160 display buffers with every code from 0 to 4. |
| `KYTY_TEX_REPEAT_REASONS=1` | Counts why repeated texture sets were refused by the texture binding memo. |

### Measured without a gain (kept off)

| Switch | What it does | Result |
|---|---|---|
| `KYTY_CP_STARVED_NO_YIELD=1` | `Thread_Gpu` no longer yields its core while waiting for the next operation | Sky Aztec 27.0 / 26.8 against 26.8 / 27.8; worse with `KYTY_DRAW_PREP_SPIN_ADAPT=0` |
| `KYTY_APR_PARALLEL=<n>` | Reads of a command buffer's file requests in parallel | No gain (cached Crashsite 9 s classic, 13 s parallel); the real cost of loading was antivirus scanning at file opening |
| `KYTY_PRELOAD_DIR` (+ `KYTY_PRELOAD_THREADS`) | Read-ahead of a level's files into the Windows cache when the level opens (v2: file starts only) | No measurable gain on 14 levels |
| `KYTY_TRANSIT_NARROW=1` | Image-transition stages narrowed to the actual access | Within noise |
| `KYTY_TEXTURE_MEMO_OVERLAP=1\|verify` | Memorizes texture lookups answered by overlap (depth images read as R32F) while none of their pages change | Misses 1.3 M → 60 k, 0 mismatches in verify mode, no fps gain beyond noise |
| `KYTY_DRAW_RUN_MESH=1`, `KYTY_MRT_SPLIT_REUSE=1\|verify` | Mesh draws may form draw runs; the second pass of a split MRT draw keeps the program preparation of the first | 0 mismatches in verify mode, but almost never used at the measured spot (Pirate Island); not measured further |
| `KYTY_LOD_STATS_COUNT_ONCE=1` | The GET_LOD_STATS counter of shaders stops at 1 (the game only tests "non zero") | 22.9 / 23.6 against 23.8 / 24.2 fps; the heavy draw kept its cost |

### Always-on changes

- Draw counters of the frame-drop journal re-wired in the draw paths (one counter add per draw).
- The `VideoOut:` console lines above, the pad-state notes for the journal (`manette.csv`, `commandes.csv`; written
  only when the journal is on).
- The video player changes above (`KYTY_AVPLAYER_SKIP_LATE` and 4 decoding threads by default).
- Not rewritten for publication: the comments of the read-ahead code (`src/kernel/fileSystem.cpp`) contain a few
  wrongly encoded accented letters; cosmetic only.

### Checked before publication

The diff `astrobot-experiments..astrobot-experiments-22-11` was searched for game bytes (byte arrays, hex strings),
personal paths and e-mail addresses: none, apart from the SMAA licence notices (authors' addresses, kept as the
licence requires) and Windows system paths (fonts, NVML). Guest (eboot) addresses in comments and console messages
are locations in the game's code only.
