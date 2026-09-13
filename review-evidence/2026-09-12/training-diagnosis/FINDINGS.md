# Diving training diagnosis — September 12, 2026

This is a diagnostic assessment, not a claim that a better policy has been trained. No running trainer, checkpoint, production asset, or reward implementation was changed during this assessment.

## What was compared

Three exact exported actors: the initial weights exposed by the app; production's 45,056,000-step actor; and the local v13 actor selected at 20,480,000 additional steps. All ran in the same current native MuJoCo simulator and v13 judge, with the same 24 evaluation worlds and six dives per world. Production's live browser judge is older, so these execution numbers are comparisons under the current judge, not quotations of the production UI.

Each actor was evaluated with deterministic and sampled actions (864 dives). An additional comparison forced the same legal declarations, in the same per-world sequence, on every actor (432 dives). Those forced choices are diagnostic only and were not used for training. The choice sequence came from the local v13 actor. This isolates motor performance from the quality of the declaration selector.

Raw reports include individual measurements, deductions, rewards, categories, and asset hashes. The scripts in this directory reproduce the checks with the paths available on this machine. Statistical scope: one fixed evaluation seed, not independent training seeds or a generalization claim.

| Identical 144 declarations | Initial weights | Production actor | Local v13 actor |
|---|---:|---:|---:|
| Valid by current automated judge | 33.33% | 20.83% | 23.61% |
| Mean declared-position quality (0–1) | 0.3005 | 0.0831 | 0.1012 |
| Mean execution (0–10) | 0.3629 | 0.4030 | 0.4086 |
| Clean dives | 0 | 0 | 0 |
| Mean training return | -3.8956 | -3.7343 | -3.3467 |
| Upward departure speed > 0.5 m/s | 0% | 54.86% | 54.86% |

The initial actor is worse at takeoff and earns lower reward. However, training improves reward while losing validity and declared-position quality. The user's observation that the initial actor can look better has a measurable basis. It would be incorrect to claim the initial actor is a better diver on every criterion.

With their own choices, the production actor passes the entry arm-position check in 1/144 dives, and v13 in 0/144. All of those declarations require head-first entry. The initial actor passes this check on 72/144 dives, but all 72 are feet-first declarations; it also fails every head-first arm-position check. Reporting that initial 50% as an overhead-arm success would be misleading.

Sampled actions do not rescue performance: mean execution drops from 0.403 to 0.309 for production and from 0.409 to 0.358 for v13. Zero clean dives occur in all six free-choice evaluations. Switching browser playback from the mean to random actions is not a demonstrated solution.

## Answers to the diagnostic questions

### Can the physical controls produce the missing movements?

Partly established. Fresh isolated, torque-limited probes reached a tuck (hip 1.400, knee 2.000 radians), pike (hip 1.499, knee 0.0002), and an overhead entry pose (shoulders 3.052 radians, elbows approximately 0.0019, toes about 4.3 degrees from the leg line). These were free-flight joint tests, not successful full dives.

A bounded 16-candidate takeoff search found valid release witnesses from forward and backward platform starts, with observed maximum rises of 0.396 m and 0.243 m respectively. The maxima describe the probe batch; a completed valid dive was not demonstrated. The small armstand search did not find a qualifying upward release; that is inconclusive, not proof that release is impossible. Earlier larger takeoff probes in the parent evidence directory also found forward/backward jumps.

The nine motor channels already include hip, knee, ankle, separate shoulder pitch/roll, elbow and adduction commands. There is no need to add a direct root spin or a scripted pose to explain missing tucks and overhead arms. A safe, full twisting dive under the existing limits still needs a dedicated feasibility demonstration.

### Do the rewards favor the right behavior?

Not reliably enough. In `training_v3/judge.py`, `trainingValue` uses difficulty multiplied by the **capped, validity-gated** execution score. Failing any validity condition zeros that credit. Independent negative deductions survive, but they are much smaller than the full successful-dive credit and are predominantly terminal.

The `positionCap` holds execution at no more than 2 while measured position quality is below 0.5. In every tested policy, the maximum awarded execution is exactly 2. The policy can therefore improve substantial aspects of form without receiving the corresponding improvement in competition-derived credit until it crosses this threshold. The local actor even bends its hips and knees substantially while declaring straight-position dives. Actual motor control is present; its coordination and incentives are wrong.

The direct shoulder-pitch training penalty is at most 0.25, while takeoff height alone can cost 1.375. All named shoulder/elbow/hand penalties together have a maximum of 0.6875. This is not evidence that one arbitrary coefficient change will solve the problem. It demonstrates that separate reporting does not make the incentives equally effective.

`engine.py::_assess_flight` samples position only between 15% and 80% of the **declared rotation**. If the diver rotates in the wrong direction or never reaches that interval, the measured position quality stays zero regardless of how much its joints improve. Reverse and inward categories score zero position quality in the tested actors. This conditional measurement blocks useful feedback precisely where a beginner needs it.

Armstand is also charged the same rise/upward-speed targets as standing takeoffs. Its press/release needs a separately validated physical and judging definition.

### Is the learner getting useful success examples and timely credit?

The tested policies produced no clean dives. The current run's recovery mode samples its own recent descending states; it does not preferentially preserve demonstrated clean entries or independently acquired motor competence. Replaying a failing distribution does not establish a path to successful twisting or entry.

Except for small foot/leg/velocity regularizers, the important components are delivered when the dive finishes. At 50 decisions/s, gamma 0.995 and GAE lambda 0.95, the direct contribution of a terminal TD residual to an action three seconds earlier is approximately `(0.995 * 0.95)^150 = 0.000215`. The critic can propagate information over learning; this is not the total eventual gradient. Nevertheless, it is a poor starting situation for discovering a coordinated countermovement jump from delayed feedback.

The PPO implementation previously passed a separate inertial-control benchmark with the same motor update path. That rules out a completely nonfunctional optimizer, not task-specific learning defects. Enlarging the network has no demonstrated benefit until these failures are addressed.

### Is behavior collapsing?

Yes, the measured behavior is narrow. Free-choice deterministic declarations shrink from 35 unique initial choices to 10 trained choices. Diversity alone is not success, but it accompanies almost no meaningful twisting and zero clean entries. For v13, average absolute twist over all categories is 0.073 turns. Its improvement in reward comes with worse entry angle than the production actor (38.45 vs. 24.28 degrees under the same native geometry).

## Recommended changes, in order

1. **Separate competition scoring from the learning signal.** Keep actual validity failures and difficulty for the displayed score. Supply continuous, independently logged physical-error feedback on failed attempts too. Remove the 2-point position plateau from the motor learning objective. Explicitly test that a controlled improvement to arms, knees, toes or body position improves its corresponding learning term while other errors remain.
2. **Move feedback to the relevant phase.** Assess takeoff at release/ascent, useful body control during flight, and entry form before and through contact. Use a documented potential or event-based scheme with terminal reconciliation so it does not double-count the same error or reward delaying entry. Body-shape feedback must not disappear when rotation is wrong. Preserve named independent deductions.
3. **Teach reusable physical control before joint optimization of whole routines.** Learn balance/press, crouch-extension, fold/open, and aerial orientation/entry recovery across randomized states and heights, using the same force-limited body. Then train the self-declaring policy to combine them. The curriculum should not prescribe a particular dive number, flip count or timed animation. Entry practice must gradually extend back to actual takeoff, with full-from-board evaluations throughout.
4. **Validate twisting mechanics and usable flight time.** Optimize disposable control probes in our own simulator, including takeoff angular momentum and asymmetrical arm actions, to establish reachable somersault/twist ranges without adding external mid-air torque. Tune physical limits only from that evidence. Armstand release should be validated separately. A jump-height-only success is insufficient.
5. **Use learned motor structure and coherent exploration.** Compare a reusable motor policy with a slower maneuver selector against the current single actor; explore coordinated joint changes over meaningful durations. The current state-covariance option is available but is not evidence of a solution. Run controlled ablations after the reward/measurement changes, rather than combining every option and attributing gains speculatively.
6. **Gate promotion on visible competence across categories.** Retain initial, production and previous-champion controls. Require improvements in full-dive validity, entry arms, position, actual completed rotations and clean entry, with no severe category regressions. Use disjoint evaluation seeds for selection and final testing. Higher reward alone must not justify replacing a more competent motion policy.

## Relevant primary research

- [Crawford, *Learning Control of Complex Skills*, Berkeley, 1998](https://digicoll.lib.berkeley.edu/record/139678/files/ERL-98-53.pdf): directly studies learning a simulated diver. It trains joint-level neural controllers from simulation and coordinates movements at a higher level, initialized from observations and refined by Q-learning. Its specified dive sequences are not identical to our autonomous maneuver-discovery goal; the transferable idea is separating reliable motor control from coordination.
- [Koschorreck and Mombaur, *Modeling and optimal control of human platform diving with somersaults and twists*, 2012](https://link.springer.com/article/10.1007/s11081-011-9169-8): whole-body models with bounded joint actuation, contact and flight phases optimized together. This is optimal control, not PPO. It supports using a feasibility optimizer to audit our mechanics before spending more RL steps.
- [Yeadon and Hiley, *The limits of aerial techniques for producing twist in forward 1½ somersault dives*, 2018](https://pubmed.ncbi.nlm.nih.gov/29408164/): studies arm/hip shape changes with movement-time constraints and simulated-annealing optimization. Useful for checking whether our arm actions, angular momentum and available flight time actually support twisting. It is not a pretrained policy we can import.
- [Peng et al., *DeepMimic*, 2018, sections 6 and 10.4](https://arxiv.org/html/1804.02717v3): successful-state exposure matters for acrobatics. Without reference-state initialization, an ablation learned a small backward hop instead of the intended full flip despite similar return. The paper uses motion imitation; adapting its lesson to self-discovery would mean a justified curriculum and useful simulated states, not claiming it validates unguided learning or copying its motions.

These papers support the proposed direction, not a guarantee of a resulting performance level. A new long training run should follow demonstrated mechanics, reward tests and short controlled learning comparisons.
