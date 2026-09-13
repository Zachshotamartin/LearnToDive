# Review snapshot: Diving

This branch preserves the current source and development browser assets. It is not a model-quality or production-readiness claim. No GitHub Actions were added.

## Checks run before push

- `npm run build`: passed.
- `npm run test:runtime`: passed for trained and initial models in an actual browser, without page errors.
- `npm test`: 49 passed / 0 failed. Stale native geometry references were regenerated independently and fingerprinted; tolerances unchanged.
- The preceding native training verification passed 9 diving tests and 17 focused hide-and-seek tests, plus a bounded-torque/momentum audit and pure browser-policy inference parity. Native training verification remains separate from browser/model quality.

## Training boundary

Current runs execute from frozen source copies on the owner's SSD. This Git push does not restart them or replace the site's model. Native optimizer/checkpoint archives and transient output directories are not committed. Public model files are the existing development exports. Read the version-specific trainer README for the new training path; the root README describes the existing browser runtime.

New system: `training_v3` (self-declared v10), with `training_v2` and `training` preserved for provenance. At the status check, the first 96x96 pilot completed 10,240,000 steps; the second had 9,472,000. The first pilot's final fixed-development evaluation had 0 mean execution/points and 0 clean entries; it is not qualified. The new v10 inference module exists separately and is not wired into the published browser.

## Review repairs (v11, 2026-09-11)

An independent review established why the v10 pilots scored zero execution on every dive: the reset pose stood on the toe corner with the centre of mass already past the platform edge (every dive was a fall; ascent 0.000 in 144 of 144 evaluated dives), the leg servos braked any joint faster than their slewing target so no jump was possible, the summed deductions exceeded ten points on every dive so the execution gradient was identically zero, the sideways-entry measure read angular velocity, preparation bounces counted contact flicker while falling, and the water model decelerated the body at about nine g. `training_v3` now carries the v11 repairs (see its README) with regression tests for each, including physical feasibility probes for balance and a forward takeoff. The three completed 96x96 pilots are invalid as architecture evidence; the suite was stopped and relaunched under v11. Nothing here is a model-quality claim.

## Source restyle (2026-09-11)

`training_v3` was rewritten from the dense one-statement-per-line convention into ordinary readable Python (four-space indentation, one statement per line, named constants, small functions, modules under 600 lines) without changing behaviour. `engine.py` was split into `geometry.py`, `stance.py` and `engine.py`; `train.py` into `train.py`, `evaluation.py` and `checkpointing.py`; `test_system.py` into per-subject test modules. Seventeen dead per-environment arrays left over from earlier reward shaping were removed from the engine. Behaviour was verified bit-exactly against references captured before the rewrite: a 512-step deterministic training run (model, optimizer, training state, environment arrays, RNG states and every surviving physics array identical), a six-case held-out evaluation and the physics audit (JSON identical). Because every learning file changed, checkpoints written by the earlier dense source (including the live SSD run, which executes from its own frozen copy) will not exact-resume under this source; that is the contract working as intended, not a regression. One behavioural fix rode along in the suite controller: a trial that crashed after already resuming once now recomputes its remaining budget from its latest checkpoint instead of reusing the stale count.

## v12 reward repair and relaunch (2026-09-12)

The v11 suite finished: 256x256x128 won all pilots and its continuation plateaued at 102,400,000 steps with 17% valid dives, 1.7 points and no clean entry. Diagnosis on the final checkpoints: no dive jumped at 10M steps and the final policy leaves the edge at a 55 to 63 degree lean with 0.4 m of rise at best, because the training signal weighted entry angle about eleven times more than the takeoff and the position kernels had no gradient away from the shape. v12 (see `training_v3/README.md`) fixes both and the suite was relaunched on the 256x256x128 architecture only, three seeds, with a longer plateau horizon. The checkpoint exported to the site on 2026-09-11 (45M steps) is a v11 development export and is not qualified.

## Handoff verification (2026-09-12, 17:42 UTC)

Read-only inspection verified that the v12 continuation is alive at 86,876,160 steps. The suite's selected pilot is seed **109311**, not 109312 as stated in the incoming handoff. Its latest completed fixed-development evaluation is `diving/continued/evaluations/86016000.json` under `2026-09-12-v12-v5.3`: 144 cases, 48.6% jumped, 0.143 m mean rise, 95.98 degrees mean entry angle, 0.105 position quality, zero valid dives and zero points. The separate current 50-case training sample reports 44% jumped; these are different cohorts.

Failure counts overlap: 130 cases miss the declared somersault count and 114 report invalid takeoff or platform contact. Median rotation error is 0.935 turns. The frozen judge matches the repository byte for byte. In v12 the extra per-degree angle term is gone, but `entryAlignment = min(6, angle / 10)` still saturates at 60 degrees; the training cost contributes the same 1.5 for all angles above that threshold. This is a potential learning-signal plateau, not proof of the sole cause. Review completion/entry credit if the fixed-development curve remains near 90 degrees around 150M steps, as requested in the handoff. No reward, checkpoint, frozen source, process or deployment was changed by this inspection.

## Controlled training improvements (2026-09-12)

29 native regression tests passed; the three-seed motor-control benchmark passed. The 28-case takeoff search found witnesses for 25 starts and leaves armstand unresolved. Alternative learning settings are separate comparison arms; defaults preserve the preceding objective. Existing frozen training runs and published assets were left untouched. These checks establish implementation behavior, not model-quality improvement. Read the versioned trainer README for comparison commands and limitations.

Controlled comparisons launched in `/Volumes/Zach's SSD/PortfolioTraining/2026-09-12-controlled-comparisons`; immutable source/input hashes and controller commands are recorded there. Three paired seeds, no automatic promotion or monitoring. The first active arms are continuous-entry feedback (diving) and outcome-based opponent sampling (hide-and-seek). The original v12/v5.3 controllers remain untouched. Current test evidence: 29 diving native tests, 35 hide-and-seek native tests; the hide-and-seek evaluator smoke completed 48 episodes across four play lengths. These are implementation checks, not evidence that the new models are better.

## Additional research changes (2026-09-12)

Motor-credit lambda comparisons, a diverse learning-progress recovery archive, and conditional correlated exploration are implemented. Defaults retain the previous behavior; each new setting is explicit, checkpointed, and included in the comparison contract. See the versioned README for mechanisms, evidence and limitations. No new model is qualified and no browser assets were replaced. Existing frozen runs remain untouched.

The research suites are running from `/Volumes/Zach's SSD/PortfolioTraining/2026-09-12-research-training-options` with three matched seeds, six arms per project and source/input hash manifests. Verified implementation evidence is in `review-evidence/2026-09-12/research-options/`; 74 native tests passed across both projects. No result has been selected for publication.

Additional assessment changes (2026-09-12): separate score/execution/clean checkpoint champions; conservative development replacement gates; phase-specific native diagnostics; paired deterministic/stochastic evaluation; fresh-versus-inherited initialization suite; and paired training-seed uncertainty reports. These are working-tree changes only. Previously launched frozen suites were not modified or restarted. Fresh-initialization long comparisons are available but have not been launched alongside the existing suites.

Final assessment verification: 38 tests passed. Evidence: `review-evidence/2026-09-12/assessment-and-selection/`. Saved-model native assessments completed; these verify mechanics and report behavior, not improved trained performance.

## v14 motor-learning restart (2026-09-12)

The v13 diver was gracefully paused at 36,341,760 steps. Its best-points actor at 20,480,000 steps initializes a new objective with a fresh independent critic and optimizer. The new native format separates selector, motor and critic trunks, uses continuous independent motor errors plus potential feedback, and mixes autonomous full dives with explicitly marked reusable motor practice. No prescribed dive trajectory is supplied. The published browser model is unchanged; v14 requires a separate 233-input browser integration before export.

Validation: 53 native tests passed, including exact adaptive-training resume, physical reward regressions, actor transfer parity, critic/actor gradient isolation and reference-policy qualification guards. Eight short runs (102,400 steps each, two seeds across successive comparisons) completed with finite updates. Results were mixed; all full-dive clean rates remained zero. They establish execution of the new learning path, not superior trained diving. The critic was separated after code review identified that value-loss gradients could modify inherited actor features. See `training_v3/MOTOR_TRAINING.md` and `review-evidence/2026-09-12/motor-v14/` for evidence and limitations.

The replacement long run uses frozen `source-final` under `/Volumes/Zach's SSD/PortfolioTraining/2026-09-12-v14-motor-phases`, seed 109316, a 512,000,000-step budget, and no plateau stop before 256,000,000 steps. Scheduled archives, raw metric champions, exact-resume state and initial/previous-production/inherited-best references are preserved. Qualification requires full-dive competence and regression guards before a disjoint final test; there is no automatic publication. Hide-and-seek processes and models were not modified by this restart.
