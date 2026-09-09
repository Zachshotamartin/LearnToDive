# Integration handoff

LearnToDive is a standalone Vite project and an importable source package. Root owns Git, GitHub, publication and Portfolio integration; this agent has made no commits or repository initialization and has not edited any other project.

## API and assets

```js
import { mountExperiment, metadata } from '@zachshotamartin/learn-to-dive';
import '@zachshotamartin/learn-to-dive/style.css';
const experiment = mountExperiment(element, { embedded: true });
experiment.dispose();
```

- `metadata.id = 'learn-to-dive'`; title, description, instructions, technique, limitations included.
- The synchronous mount owns one `.learn-to-dive` subtree. Async simulation is internal. `dispose()` is idempotent and stops pending work/worker/RAF, listeners, observers, controls, geometry/material resources, renderer context, export URLs and the subtree.
- Ready contract: `.learn-to-dive[data-state='ready']`; input changes synchronously set `data-state='pending'`, failures set `error`. The first state is `loading`. A completed trajectory can be replayed immediately; these states describe simulation availability, not playback completion.
- `embedded: true` omits the standalone heading. Both WebGL clear and preview panel background are transparent, allowing the host's exact page texture/color to show through. No opaque ground/background plane remains; pool and board are actual geometry.
- No copied public assets or `assetBase` setup needed. Model/evaluation imports total 6,955 bytes uncompressed. Worker is referenced through a Vite-compatible relative `new URL`.
- Dependency: Three.js ^0.180.0. Development: Vite ^8.2.1 (lockfile8.2.2), Playwright ^1.62.1. Node≥22. Port5184. No GitHub Actions, secrets, server or API costs.
- Browser entry is ~540KB/~137KB gzip including Three.js in standalone. Portfolio should lazy-import this experiment and share/dedupe Three.js if already installed. Simulation worker ~8.1KB.

## Actual learning and evaluation

Frozen policy trained with genuine reward-based cross-entropy motor-policy search, not imitation or fabricated neural-network training. 20 coefficients; context→takeoff somersault/twist impulses+tuck/open motor program. 829,440 training dives, CPU40.81s. Validation384 contexts; held-out evaluation2000, seed990517.

- Pretrained: 100% valid,90.4438/100 mean score,7.0142° mean entry,1.5023 somersaults,.7917 twists.
- Initial zero weights:10.55% valid,8.5859 mean score.
- Analytic half-flip:100% valid,76.9747 score,1.3003° mean entry. It has better alignment; UI/README openly state the tradeoff.
- 162 distinct boundary-grid cases:162 valid,worst angle25.1028°.
- Wilson95% valid-rate interval for2000/2000 is99.808–100%. Measured success is not a guarantee.
- `training/evaluate.py` reproduces frozen metrics without changing assets; `training/train.py` reproduces training/model/evaluation/fixtures with seeded NumPy. No Torch requirement.

The exact symmetric-top rotation conserves world angular momentum and fixed-shape energy. It replaces an earlier explicit step that failed an energy audit; the shipped model was retrained on corrected physics using a new test seed. Water-contact/validity gates all score components. Reduced equivalent-body model, continuous toy rotation counts, initial twist impulse, and splash proxy limitations are explicit. This is an open-loop learned motor program and does not react to new midair disturbances.

## Verification

- `npm test`:13/13 passed. Full per-step Python/JS parity in3 contextual trajectories, quaternion norm, conserved world momentum, conserved fixed-shape rotational energy, ballistic motion, tuck/inertia response, terminal behavior, actual-water-contact gating, invalid spin-farm gating, fresh seeded contexts, deterministic replay and bounds.
- `npm run test:browser`:6/6 Chromium tests passed. Policies, settings, matching JSON export, replay, scrub, quarter speed, camera, new takeoff/reset,390px no overflow, keyboard controls, embedded/no-heading, batched offscreen→onscreen callback, disposal, actual transparent scene pixels, and unclipped2m/12m camera framing.
- `npm run build`:passes; ordinary standalone Three.js>500KB chunk advisory only.
- Captures inspected visually and produced by live renderer; numeric alpha tests confirm transparent background.

## Showcase media

Primary scene-only transparent PNGs from the same interactive tool:

1. `examples/pretrained-flight.png` — trained5m diver in rotation, time0.40s, orbit view.
2. `examples/head-first-entry.png` — trained8m dive shortly before water contact, close entry view.
3. `examples/splash-proxy.png` — optional initial-weights8m impact illustrating splash proxy, orbit view.

`examples/captures.json` contains reproducible exact settings and times. `npm run capture` starts its own Vite server if needed, waits for real rendered state, uses live `canvas.toDataURL` to preserve alpha, and closes its browser/server. Full interface QA screenshot is in ignored `test-results/interface-check.png`, not showcase media. No UI overlays/borders or generated illustrations in primary PNGs.

## Status

Implementation complete and ready for root review/integration. Files frozen except responding to concrete review findings.

## Exact files to stage

```text
.gitignore
LICENSE
README.md
NOTES.md
index.html
package.json
package-lock.json
playwright.config.js
src/index.js
src/main.js
src/scene.js
src/sim.worker.js
src/style.css
src/core/physics.js
src/data/model.js
src/data/evaluation.js
scripts/capture.mjs
examples/captures.json
examples/pretrained-flight.png
examples/head-first-entry.png
examples/splash-proxy.png
tests/physics.test.js
tests/parity-fixtures.json
tests/browser/experiment.spec.js
training/sim.py
training/train.py
training/evaluate.py
training/requirements.txt
training/checkpoint.json
training/evaluation.json
training/evaluation-check.json
```

`node_modules`, `dist`, `test-results`, Python caches and any local virtualenv are ignored. No Actions/workflow files exist. Final camera review replaced world-box fitting with fitting projected actual geometry/path bounds. Overview is centered at roughly79% image height; Inspect entry provides a close, paused physical entry frame. The second showcase uses that action and hides the path line via the normal Flight path control.

## Reset/replay camera follow-up

A close entry inspection no longer survives a fresh start. Reset, Replay, and Play after the completed timeline restore the overview; Play from an inspected near-entry frame continues that inspection. No physics, coefficients, evaluation or capture changes. Existing6 browser checks passed with the fix; the added focused regression passed after correcting a test-helper argument and compares exact rendered overview PNGs after all3 restart paths. Build passed. Stage `src/index.js`, `tests/browser/experiment.spec.js`, and this note for the follow-up PR.
