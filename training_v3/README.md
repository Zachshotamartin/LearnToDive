# Autonomous diving training v11

One shared hybrid PPO policy chooses a legal declaration, then operates nine bounded joint controls in MuJoCo. The trainer supplies a category, apparatus, height and routine history, not a specific dive. Categorical and motor outputs share the trunk; they are not separate specialist policies. Nothing in the trainer scripts a takeoff, a flight shape or an entry: the open-loop jumps in the tests are feasibility probes of the physics only.

Official difficulty values are recorded in `difficulty.json` with a source URL and PDF hash. Reproduce them with `python training_v3/extract_difficulty.py downloaded-regulations.pdf`. The declaration mask enforces category, available apparatus/height and no repeated dive number across positions. Flying-action declarations remain excluded until a separate flying-phase judge exists. Official scored heights are springboard 1/3 m and platform 5/7.5/10 m; arbitrary heights must not claim official table values.

Execution is an explicitly approximate geometric judge, separate from official difficulty; `judge.py` documents which World Aquatics articles each check approximates. It observes the full water crossing, pointed feet, leg/hand placement, takeoff, rotation and twist completion. Failed declarations get no difficulty points. Water and splash remain physical approximations. Score is three times difficulty times execution. The training signal is a dense version of the same judged components plus a bounded completion bonus, so a dive whose official execution is clipped to zero still receives gradient toward each fault; it is never positive for a failed dive and extra spins never pay.

## What changed in v12 (the fall-off-the-edge optimum)

The v11 continuation plateaued at 102M steps with a policy that never jumped: it tipped forward off the edge, pushed a little and entered near vertical. Two reasons were measurable. The dense training signal charged up to 4.2 for the entry angle but at most 0.375 for a missing takeoff, so a jump that briefly worsened the entry was never worth it; and position quality used Gaussian kernels that were numerically zero (with zero gradient) once the hips were a radian off, so the position component of the judge could not be felt at all. v12 scores rise, upward departure speed and lean at departure as takeoff outcomes (rule 10.4.3, control and height), removes the extra per-degree angle cost, and replaces the position and entry-form kernels with heavy-tailed ones (`positions.py`). Nothing prescribes a motion; the judge now pays for what a takeoff produces. Every evaluation summary reports whether the athlete jumped, its rise, departure speed, lean and position quality, and the continuation may only stop on a plateau after 256M steps and twenty-five evaluation windows.

## What changed in v11 (review repairs)

The v10 pilots never scored: the start pose stood on the toe corner with the centre of mass past the platform edge, the leg servos braked any joint moving faster than their slewing target, and eight judged deductions summed past ten points on every dive, so execution and its gradient were identically zero. v11 starts flat-footed with the centre of mass over the feet (randomised lean, toe overhang and knee bend; the controller must balance, press and jump), uses athletic bounded torque caps with low servo damping and a 14 rad/s leg target slew, and calibrates water drag so axial flow through slender parts is streamlined (a 10 m entry now reaches about 3.5 m instead of 1.2 m). A bounded random search over open-loop jumps reaches 0.34 m of rise forward and 0.57 m back-facing; the policy has to find its own. The judge's sideways-entry measure read angular velocity (it now reads sideways linear speed), preparation bounces now require an upward hop rather than contact flicker, and position faults follow the rule's 0.5 to 2 point range. The armstand start is balanced over shoulder-width hands. PPO weights every decision once with a separate declaration entropy coefficient, value targets from recovery-practice declarations are excluded, the exact-resume contract hashes only the learning files, a run resumed after its final update still produces its final evaluation, and the suite refuses to rank architectures that all scored zero.

Controls have native joint torque limits and no root actuator. Platform support is rigid; springboard starts at static preload. Recovery practice comes only from this learner's physically reached states with the same declaration, height and apparatus; full routines are retained. Adaptive category sampling never supplies a named maneuver.

## Run and resume

`python training_v3/run_suite.py --output /mounted/ssd/diving-run`

The controller compares 96x96, 256x256 and 256x256x128 across three seeds, then continues the median seed of the best development architecture. Default pilots: 10,240,000 interactions each. Continuation budget: 512,000,000 total interactions, with a plateau decision only after 102,400,000 and ten evaluation windows. An optional explicitly labeled motor warm-start arm is supported, but is not enabled by default.

SIGTERM the PID in `SUITE.json` to pause at a saved update boundary. Repeat the identical command to resume. Checkpoints retain optimizer, RNG, physical worlds, routine history and recovery bank. `latest.pt`, immutable million-step checkpoints and separately evaluated `best.pt` are retained. Changing a learning file invalidates exact resume; the suite controller and audit scripts are outside that contract. Budgets are limits, not claims that a model is good.

## Layout

- `geometry.py`: sensor layout, quaternion helpers and body-geometry measurements (pure functions of a sensor row).
- `stance.py`: the flat-footed standing start and the balanced armstand start.
- `engine.py`: the batched MuJoCo arena with takeoff, flight and entry measurements.
- `water.py`: the stateless water force model.
- `rules.py`, `judge.py`: legal declarations from `difficulty.json` and the approximate geometric judge.
- `environment.py`: routines, declarations, recovery snapshots and the 223-value observation.
- `policy.py`, `losses.py`: the hybrid declaration/motor actor with PopArt value scaling and the PPO surrogate.
- `train.py`, `evaluation.py`, `checkpointing.py`: the resumable trainer, held-out evaluation and atomic writes.
- `run_suite.py`, `audit_physics.py`, `check_browser_parity.py`, `extract_difficulty.py`: tooling outside the exact-resume contract.

## Verification and release boundary

Run `python -m unittest discover -s training_v3 -p 'test_*.py'`, `python training_v3/audit_physics.py`, and `python training_v3/check_browser_parity.py`. The tests are split by subject: `test_judge.py`, `test_environment.py`, `test_measurements.py`, `test_stance.py` and `test_training.py`; `probes.py` holds the shared single-athlete helpers.

The standalone browser inference module is `src/core/autonomousPolicy.js` (format `self-declared-diver-v11`); it is not plugged into the existing published v9 runtime, and `training_v3/diver.xml` is this trainer's model, not the browser's `public/physics/diver.xml`. No training script replaces website assets. After training, frozen unseen routines, perturbed entries, category coverage, physics/browser parity, and actual browser playback must qualify a checkpoint before switching the app to v11. All development exports are marked unqualified.

## Controlled reward comparison (2026-09-12)

`--reward-mode v12` preserves the old training objective. `--reward-mode continuous-entry` replaces only its saturated alignment training cost: the original slope through 30 degrees, then a positive slope through 180 degrees, with the same 1.5 maximum cost. The competition judge, legal declarations, difficulty table, physical forces, takeoff and position incentives remain unchanged. Terminal records expose the alignment, takeoff, rotation, twist and position costs separately. Improved training reward alone never qualifies a model.

`--initialize-from CHECKPOINT` starts a new objective comparison using a same-schema actor with fresh optimizer, value head and worlds. It records the input hash and inherited step count separately from new-objective steps. It is not exact resume. `--resume` still requires matching source hashes and reward mode.

Run `compare_rewards.py --output NEW_FOLDER --initialize-from IMMUTABLE_CHECKPOINT` for paired seeds 91201/91202/91203, baseline versus continuous entry, 2,048,000 additional steps per arm. Each arm uses 32 environments and one physics thread. Fixed 144-dive evaluations run at milestones and completion. SIGTERM checkpoints the active child at its update boundary; repeat the same command to resume. No automatic continuation or publication follows the comparison.

`benchmark_control.py --output report.json` uses the actual motor-policy, GAE and PopArt/PPO optimization path on an inertial point-mass task. Three seeds reduced fixed-evaluation position MSE from about 1.5 to below 0.002 in 160 updates. This checks the motor learner, not self-play or whole-dive feasibility.

`audit_takeoffs.py --output report.json` searches bounded diagnostic servo programs at all 28 legal category/apparatus/height starts. It found upward takeoff witnesses for 25 combinations, but none for the three armstand platform heights. This is a finite search: armstand feasibility remains unresolved, not proven impossible. The commands and diagnostic reports are never training data. No successful complete dive is inferred from a takeoff witness. Saved evidence is under `review-evidence/2026-09-12/`.

## Research training options (2026-09-12)

These options implement controlled comparisons, not qualified policies. They do not change the physical diver, legal dive table, competition scoring, or published website.

| Option | Behavior | Default |
| --- | --- | --- |
| `--gae-lambda .98` or `.99` | Longer motor credit trace; the declaration policy still uses its separate return estimator. | `.95` |
| `--recovery-mode progress` | Up to 512 diverse real flight states per category; outcome-progress, coverage and staleness sampling with 25% uniform exposure. | `recent` |
| `--exploration state-covariance` | State-dependent full joint covariance with bounded low-rank factors and exact squashed-Gaussian PPO likelihood. | `diagonal` |

Recovery states are captured once per early/middle/late descent stage. Later-stage control increases earlier-stage sampling, while every stage retains a nonzero chance. They match the learner's declared dive, apparatus and height. Outcomes affect sampling only, not the score or reward. All physical states came from this learner; no reference motion or prescribed dive is inserted. Archive IDs, visit statistics, readiness, selection and RNG are checkpointed. Full evaluation always starts on the board with recovery practice disabled.

State-covariance exploration is a new conditional Gaussian at every decision, combined with the existing autoregressive mean. It is **not** a literal gSDE implementation with a persistent sampled noise matrix. The new exploration head initializes without perturbing common actor/value weights or the rollout RNG. Its browser export carries a distinct format so an older runtime cannot silently interpret it as the old policy.

`compare_rewards.py --suite research --output NEW_FOLDER --initialize-from FROZEN_PARENT` runs combined, baseline, lambda .98, lambda .99, recovery-only and exploration-only arms on three matched seeds. All start from the same actor and fresh optimizer/value/worlds; each receives 2,048,000 new steps and fixed 144-dive evaluations. The combined arm uses continuous-entry feedback, lambda .98, progress recovery and state-covariance exploration. This is a screening budget, not proof that convergence has been reached. Separate effects and the combined result must be compared before longer continuation or publication.

Verification: 33 native tests pass, including exact combined-option resume, credit-trace boundary handling, correlated-action likelihood/gradients, and archive restoration. A native 25,600-step smoke run collected 240 recovery states across all three stages and reused them 66 times; its evaluation completed 24 board-start dives with zero practice episodes. These numbers verify execution, not better dive quality.

Research foundations: [GAE](https://arxiv.org/abs/1506.02438), [reverse curriculum](https://arxiv.org/abs/1707.05300), [smooth exploration](https://arxiv.org/abs/2005.05719). Methods here are original adaptations with the differences stated above.

The optional learner check `benchmark_control.py --exploration state-covariance --gae-lambda .98 --output report.json` passed on seeds 811/812/813: fixed-evaluation position MSE fell from approximately 1.5 to 0.00104, 0.000218 and 0.0000189. This validates learning on a simple motor-control task, not the quality of complete dives.

### Checkpoint selection and behavioral assessment (2026-09-12)

New runs retain full resumable `best-points.pt`, `best-execution.pt`, and `best-clean.pt` checkpoints independently. `best.pt` is the conservative development incumbent: a replacement requires at least a 0.5-point mean gain with a positive paired confidence bound and noninferiority on execution (0.1 point), clean entries (2 percentage points), valid dives (2 points), jumps (5 points), and alignment (2 degrees). Whole routines/worlds are the resampling unit, not frames or individual dives within a routine. Simultaneous intervals cover the primary and guard metrics; observed material category regressions also block replacement. `selection.json` records the decision, and that evidence is part of exact-resume state. All exports remain `qualified: false`. Repeated development selection is not a final holdout test.

`python training_v3/assess_behavior.py --checkpoint IMMUTABLE.pt --output NEW_REPORT.json` compares mean and sampled actions on paired worlds. Each episode records takeoff, completed declared rotation, entry alignment, angular speed, and rotation after water contact. The last measure is a 20 ms sampled angular-speed integral, not a signed flip count or a new judging rule. Evaluation always starts on the board, disables recovery practice, and preserves the caller's RNG. Sampling changes neither the physical force limits nor the competition score.

`python training_v3/compare_rewards.py --suite initialization --initialize-from PARENT.pt --output NEW_FOLDER` compares fresh and inherited actors with identical new-experience budgets and hyperparameters, fresh value heads and optimizers, and matched seeds. Inherited lifetime experience is reported separately; this does not claim equal lifetime compute. The runner's default is now five seeds. Existing frozen three-seed suites are unchanged.

Completed comparison reports include paired training-seed bootstrap intervals, multiplicity adjustment, practical gain thresholds, and noninferiority guards. Fewer than five independent training seeds are explicitly preliminary; five is a minimum screening rule, not a guarantee of adequate statistical power. `comparison_summary.py --results RESULTS.json --baseline BASELINE_ARM --output NEW_REPORT.json` can assess completed older comparisons without changing their training or reports. No assessment automatically publishes weights, and final unused holdout validation is still required.


## v14 motor-learning restart

See [MOTOR_TRAINING.md](MOTOR_TRAINING.md) for continuous phase feedback, independent critic/selector/motor trunks, reusable motor practice, immutable reference policies, qualification gates and the new 233-input native format. Current browser exports are unchanged.
