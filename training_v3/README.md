# Autonomous diving training v11

One shared hybrid PPO policy chooses a legal declaration, then operates nine bounded joint controls in MuJoCo. The trainer supplies a category, apparatus, height and routine history, not a specific dive. Categorical and motor outputs share the trunk; they are not separate specialist policies. Nothing in the trainer scripts a takeoff, a flight shape or an entry: the open-loop jumps in the tests are feasibility probes of the physics only.

Official difficulty values are recorded in `difficulty.json` with a source URL and PDF hash. Reproduce them with `python training_v3/extract_difficulty.py downloaded-regulations.pdf`. The declaration mask enforces category, available apparatus/height and no repeated dive number across positions. Flying-action declarations remain excluded until a separate flying-phase judge exists. Official scored heights are springboard 1/3 m and platform 5/7.5/10 m; arbitrary heights must not claim official table values.

Execution is an explicitly approximate geometric judge, separate from official difficulty; `judge.py` documents which World Aquatics articles each check approximates. It observes the full water crossing, pointed feet, leg/hand placement, takeoff, rotation and twist completion. Failed declarations get no difficulty points. Water and splash remain physical approximations. Score is three times difficulty times execution. The training signal is a dense version of the same judged components plus a bounded completion bonus, so a dive whose official execution is clipped to zero still receives gradient toward each fault; it is never positive for a failed dive and extra spins never pay.

## What changed in v13 (the final-run configuration)

Diagnosis of the v12 and completion-first runs (`FINAL_RUN_PLAN.md`, with the open-loop ceiling probes under `review-evidence/2026-09-15/ceiling-probes/`): a random search over servo schedules finds 9/10 forward dives on every apparatus in minutes, so the physics is not the limit; the learner never entered vertically because entry alignment was worth 0.36 of a scale where rotation was worth 16, the entry-only sub-task had never succeeded once, and the entry angle is a razor-sharp function of the takeoff (the best scripted dive drops from 6° to 89° under the full stance randomisation).

v13 changes, none of which prescribes a motion:

- `motor_objective.conjunctive_components`: the credit is `16 x rotation progress x entry quality x takeoff quality`, so the maximum needs all three phases; every judge deduction keeps its competition weight, alignment keeps its slope past the judge's cap, and rise, departure speed and lean are charged as takeoff outcomes. Physical failures cost the whole credit. Codex's potential-based rotation shaping is unchanged.
- `mastery.py`: six mastery stages (platform 10 m forward and back with straight and tuck first; then inward and pike; reverse and one and a half; all platform heights and twists; armstand; springboard). A stage widens after three consecutive held-out evaluations with a clean rate of at least 30% on its own scope. Each stage also sets the stance randomisation width (a quarter of the full width to start) and the single-dive practice share.
- Exploration: the motor noise is an AR(1) process with correlation 0.9 (about 200 ms) observed through nine previous-noise columns, so the likelihood stays exact while the perturbation is coherent over a dive phase; the control autoregression keeps its own faster constant.
- Observation: ballistic time to the surface and the height above water at full resolution over the last three metres; running input normalisation folded into the first layers at export (`self-declared-diver-v13`), so the browser needs no normalisation code.
- The entry sub-task starts near the pose an entry needs and widens with mastery; a task that is being lost narrows again instead of staying unwinnable. Direction practice stays inside the current scope.
- Clean form threshold 0.7 (the best scripted dives score 0.67 to 0.79 on this rigid rig).
- Milestones: at 10.24M steps the entry sub-task should succeed more than half the time with at least one clean dive; at 30.72M the clean rate on the current scope should exceed 20%. Each is measured once, recorded in `STATUS.json` under `gates` and printed in the log as `milestoneMissed`. A missed milestone is evidence for review; training continues to its budget either way. Evaluations report per-declaration clean rates and entry-angle percentiles; `best.pt` follows the clean rate once any clean dive exists; the suite ranks pilots by clean rate.
- Suite defaults: 128 environments, batch 2048, three epochs, three seeds of 256x256x128, then the median seed continues with a 256M-step minimum before any plateau decision.

## What changed in v13.3 (departure lean in the takeoff practice cost, 2026-09-16)

Decomposing the v13.1 checkpoint's takeoff practice attempts at the first rung showed the binding term was not the rise or the speed but the invalid-takeoff flag: every attempt (48 of 48, sampled or deterministic) left the platform past horizontal, so the engine charged the flat 2.0 and the ladder had nothing to climb. The flag is binary, so leaving a little more upright was worth nothing. `takeoff_cost` now also charges the departure lean (0.5 × lean / 90°, capped at twice that), measured live as the body's pitch until it leaves the board and as the recorded departure pitch afterwards, so the practice reward has a slope toward leaving upright before the takeoff becomes valid. The judge's validity rule and the full-dive credit are unchanged; a valid hop leaning 20° still passes the first rung.

## What changed in v13.4 (an exploration ceiling, 2026-09-17)

The v13.3 continuation reached 53% valid dives, a 63% jump rate, 46° entries and 9.4 points by 19M steps, then collapsed between 29M and 34M: the takeoff practice task stopped succeeding entirely (frozen at 14,893 successes for 14M steps), the entry task fell to zero readiness, the jump rate fell to 8% and points to zero. The archived checkpoints show why. The motor exploration scale rose monotonically the whole run: 0.154 at 12.3M, 0.168 at 18.4M, 0.180 at 24.6M, 0.189 at 30.7M, 0.201 at 36.9M, 0.223 at 47.8M. Measured directly on this rig, holding the entry pose passes the entry sub-task 25% of the time at 0.14 and 3% at 0.22, so the run crossed its own precision threshold around 30M and the skills that depend on precision became unsamplable.

`--motor-logstd-max` puts a ceiling on that scale (the suite uses -1.9, a marginal scale of 0.15). The policy may still explore less, so precision remains learnable; it can no longer drift into the regime where a clean entry cannot be sampled. The ceiling is applied wherever the distribution is built and to the exported weights, so a browser runtime reproduces the scale training used. The entropy bonus that pushes the scale upward simply saturates against it.

## What changed in v13.2 (after the second gate failure, 2026-09-16)

The v13.1 pilot (run `2026-09-16-v13.1-final`) stopped on the same gate at 10.24M steps, but much further along: 50% valid dives (back dives 202A at 100%, 31° entries, 7.5 to 11.5 points) against 5% and zero points in v13. Entry sub-task success was 0.29 (needs 0.5) and there was no clean dive, because every forward dive (101A) still failed with an invalid takeoff (50° departure lean, no jump). The takeoff practice task never succeeded once in 23,769 attempts: its cost fell from 4.3 to 2.8 but success needed 0.3 m of rise and 2.8 m/s at every level, so the difficulty never moved and the skill had no ladder, unlike the entry and shape tasks.

v13.2 change (`motor_curriculum.takeoff_cost`, `TAKEOFF_RISE`, `TAKEOFF_SPEED`): the takeoff task asks for a 0.1 m hop at 1.2 m/s at level 0 and widens linearly to the competition takeoff (0.3 m, 2.8 m/s) at level 1, with the same mastery rule as the other tasks. The full-dive credit still asks for the competition takeoff; only the practice rung moves. `test_takeoff_ladder.py` covers the targets, the cost and the live cost at both ends. The gates are unchanged.

## What changed in v13.1 (after the first gate failure, 2026-09-15)

The v13 pilot (seed 109310, run `2026-09-15-v13-final`) stopped on the first gate at 10.24M steps: entry sub-task success 0.0 (22 successes in 19,460 practice attempts), no clean dive, takeoff sub-task 3.6%, every practice level still at zero. Two probes on the frozen source explain it, and neither is about reward weights:

- The entry sub-task is winnable as designed. A constant action that simply holds the straight, arms-overhead pose passes it 69% of the time at level 0 (cost median 0.59 against the 0.8 threshold, entry angle median 22°); a centred action (every joint half bent, the `--random-motor-init` default of the v13 suite) never passes (cost median 7.8). The trained policy's entries had hip-bend losses of 13 and knee-bend losses of 17, that is, it never left the bent default.
- The exploration scale made the skill unobservable. With the coherent AR(1) noise the servo no longer averages the perturbation away, so the marginal scale σ = exp(−0.5) = 0.61 is what the joints see. The same pose-holding policy sampled with the training noise passes 0% (cost median 6.7); it passes 3% at σ 0.22, 25% at σ 0.14 and 56% at σ 0.08. Over the pilot the learned log standard deviations drifted up (−0.5 to between −0.4 and −0.1), so no clean entry could ever be sampled and no gradient toward precision existed.

v13.1 changes (`train.py`, `policy.py`, `run_suite.py`): `--motor-init entry-pose` biases the untrained motor head to hold the straight entry pose (a posture prior the agent moves away from freely, not a motion; the old `--random-motor-init` and the board-stance default remain available), `--motor-logstd -2` starts the coherent exploration at σ 0.135, and the motor entropy coefficient is 0.002 so the scale is learned by the objective rather than pushed up by the bonus. `test_motor_init.py` checks that the untrained entry-pose policy passes the entry sub-task deterministically and at σ 0.135 under sampling, and that the historical scale never does. The gates, the objective and the curriculum are unchanged; the failed run's evidence is in its `evaluations/` and `STATUS.json`.

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
# Completion-first specific-dive training

`--reward-mode completion-first` prioritizes actually completing the declared
dive. It leaves the competition judge, execution report, physical model,
observation schema and action space unchanged.

- A valid dive earns **16 + 2 × degree of difficulty** in training units.
  Merely selecting a difficult declaration earns no credit.
- Failed dives receive separate continuous somersault and twist deductions,
  each `8 × log(1 + count error)`. Signed somersault error penalizes the wrong
  direction; overshooting or adding unrequested twists also increases error.
- Once the dive is valid, each count deduction uses a weight of 0.5, leaving
  execution as the main refinement objective within that completed target.
- Execution faults remain individually measured and charged. Each has its own
  smooth bounded deduction; their budgets sum to **4**. Worsening one fault
  never changes another fault's deduction. Height and departure speed remain
  separate. Entry failures and board collisions have separate safety costs.
- Existing potential shaping and motor fundamentals practice are retained.
  Target-curriculum mastery uses 85% valid completion and 15% execution, so
  imperfect but completed targets can unlock more complex practice.

Use `--continue-from` with a saved **phase-dense, goal-practice** checkpoint and
the same training configuration, changing only the reward mode. This explicit
phase migration preserves every network parameter, Adam state, cumulative
step/update count, and motor curriculum readiness. Four critic-only updates
recalibrate values before actor updates resume. New episodes prevent mixing
partial old-objective returns with the new rewards. Old archives remain in the
parent run; only target mastery averages reset because their definition changed.
The new phase must write into a separate output directory. Exact resume within
the new phase remains supported and is tested against uninterrupted training.

**The existing running trainer cannot hot-reload this change.** A graceful
checkpoint at an update boundary followed by continuation is required. This
relaunches the process; it does not reset or replace the trained model. Do not
claim a running process has adopted a reward change just because files changed.

Verification: `python -m unittest test_completion_first test_motor_curriculum
test_direction_practice test_reward_comparison test_training test_judge -q`.

### Fresh initialization and entry/position scoring (v17)

`--random-motor-init` starts random networks without the neutral-pose motor bias.
Use a new output directory and omit `--continue-from`, `--warm-start`, and
`--initialize-from`. No actor, critic, optimizer, rollout, recovery bank,
selection history or curriculum mastery is transferred. The resulting run still
supports exact `--resume` from its own checkpoints.

Entry feedback and assessment share arm references: straight arms beside the
body for feet-first entry, extended overhead for head-first entry. Shoulder
pitch, roll, and elbow deductions remain independent. Actual hand position
relative to the head determines the severe wrong-arm cap. Entry measurements
persist across surface contact and stop charging each submerged segment.

Declared A/B/C/D position determines table DD. Flight shape quality now checks
both legs. Sustained recognition of a clearly different A/B/C position is
separate from malformed execution (such as a pike with bent knees), and only
that recognition triggers the wrong-position two-point cap. Recognition uses
conservative geometric thresholds and at least five flight samples; it is an
automated approximation, not an official scoring algorithm. It never silently
reclassifies a malformed pike as a lower-DD tuck.

Source: World Aquatics Competition Regulations, 18 February 2026, Part Four,
10.1.4–10.1.6, 10.5.5, and 10.6.3–10.6.5:
https://www.worldaquatics.com/rules/competition-regulations

Regression coverage: `test_declared_form.py`, including actual MuJoCo head-first
and feet-first poses, physical feet-first drops, individual arm faults,
asymmetric knees, pike-vs-tuck recognition and unchanged declared DD.

Completion-first v2 also requires the declared body position to be shown before
awarding the training completion and DD bonuses. This is distinct from numeric
validity: a wrong-position dive can remain numerically valid with a competition
cap. A recognizable pike with execution faults still earns completion credit;
a tuck substituted for a declared pike does not. The curriculum uses the same
completed-declaration result when updating target mastery.
