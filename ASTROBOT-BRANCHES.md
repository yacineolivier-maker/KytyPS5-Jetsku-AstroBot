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
| `astrobot-experiments` | archive of all experiments (108 commits) | many, all off by default | used in game for measurements; see below |
| `fix/esc-exits-fullscreen` | fix | none (always on) | compiled; tested in game on the Jetsku U59 base (as part of `astrobot-pc`) |
| `feature/app0-overlay-subst` | feature | `KYTY_APP0_OVERLAY`, `KYTY_APP0_SUBST` | compiled; tested in game on the Jetsku U59 base (as part of `astrobot-pc`) |
| `fix/windows-thread-names` | fix | none (Windows only) | compiled; tested in game on the Jetsku U59 base (as part of `astrobot-pc`) |
| `diag/render-pass-stats` | diagnostic | `KYTY_CP_COMMIT_STATS` (needs `KYTY_CP_DIAGNOSTICS=ON` at build time) | compiled; in the `astrobot-pc` source, not run with the switch in this isolated form |
| `fix/amd-rsqrt-on-intel` | fix / speed | `KYTY_AMD_RSQRT=exact\|fast\|native` (default `exact`) | compiled; tested in game on the Jetsku U59 base (`fast`, as part of `astrobot-pc`) |
| `fix/kernel-sema-targeted-wake` | fix / speed | none (always on) | compiled; the same change was used in game on the Jetsku U59 base (experiments builds); no isolated measurement |
| `diag/wait-graph` | diagnostic | `KYTY_WAIT_GRAPH=1` | compiled; the same instrumentation was used in game on the Jetsku U59 base (experiments build, 16 Nov 2026) |
| `diag/frame-path` | diagnostic | `KYTY_FRAME_PATH=1` | compiled; the same instrumentation was used in game on the Jetsku U59 base (experiments build, 15-16 Nov 2026) |

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

- **Content**: the whole working branch (`cp-voies`), `744705ee` + 108 commits, kept as is (French commit messages).
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
