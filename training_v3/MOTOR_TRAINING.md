# Motor learning revision (v14)

This changes the native learner. It does not claim that the browser's diver has improved or replace a browser model automatically.

## Why this revision exists

The September 12 diagnosis compared initial, deployed and v13 actors under one physical model and one current judge, including identical declarations. All had zero clean entries. The trained actors used worse declared body positions than the initial actor, while the v13 training return improved. Rotation-gated posture measurements and delayed, capped feedback were poor teaching signals. The report and raw evaluations are in `review-evidence/2026-09-12/training-diagnosis/`.

## Objective and credit

`--reward-mode phase-dense` leaves the competition judge intact. It gives the learner a separate continuous ledger for each measured entry fault, takeoff height, takeoff speed, lean, body position, rotation/twist count, contacts and entry completion. Execution caps and clipped competition points do not enter that ledger. Difficulty earns credit only when the agent actually completes a legal self-declared dive.

Flight posture is measured even when rotation is incorrect. Its weight fades as actual height and velocity indicate an approaching water surface; this allows an athlete to open out before entry. Pose and takeoff potentials provide earlier value-learning feedback. Each increment is `gamma * Phi(next) - Phi(previous)`, with a zero terminal potential on both success and failure. It telescopes in discounted returns, so holding or cycling a pose cannot farm its potential. Small physical regularizers remain separate.

The proposed long run uses GAE lambda 0.99 rather than 0.95. With gamma 0.995, the direct weight on a terminal TD residual 150 ticks earlier is about 0.104 rather than 0.000215. This is a credit-trace calculation, not a guarantee about an entire policy gradient. The declaration selector and its value target use the full return to the episode boundary.

## Reusable skills, not assigned dive numbers

`--motor-curriculum adaptive` mixes real, autonomous board-start dives with short motor tasks:

- Takeoff: real board/foot or board/hand contact; no injected launch velocity.
- Body shape: varied initial joint positions; learn straight, pike and tuck shapes.
- Aerial control: varied orientation and angular velocity; control body orientation through the joints.
- Entry: downward airborne starts; coordinate alignment, arms, hands, knees and toes through water entry.

Airborne starts are explicitly practice. They do not count as successful full dives, enter the full-dive recovery bank, contribute to a full routine's used-dive history, train declaration selection, or qualify a checkpoint for publication. All full evaluations start from the actual board. At least 25% of task assignments are full dives initially; their share increases as motor success rises. Practice difficulty increases from measured success. Goals are physical skills, not predefined multi-frame motion trajectories or assigned dive codes. The full-dive policy still chooses its declaration within category, apparatus, height and nonrepeat rules.

The observation gains ten task/goal columns immediately before its nine previous-action columns: 233 values rather than 223. New columns start with zero weights during actor transfer, preserving the inherited policy's initial outputs. `--architecture split` gives declaration selection a separate 256/256/128 trunk; a single 256/256/128 motor trunk remains shared across all dives and skills. This prevents selector gradients from directly changing the motor trunk. A third independent 256/256/128 trunk supplies the critic: value-loss gradients cannot update either actor trunk. The critic and optimizer restart because the objective changed; the parent actor and its ancestry remain recorded.

## Mechanics and evidence

The actuator search uses only bounded joint servo targets, with the existing force and slew limits. It starts on the board and does not apply a root wrench or change the root pose after release. The search found twisting takeoffs; an independent angular-velocity replay verifies real axial rotation. The witness has substantial lateral tilt and is **not** a demonstration of a clean or valid competitive twisting dive. Its searched control sequence is never imported into the learner or used as a browser animation.

The new native tests cover independent fault feedback even on invalid dives, potential cancellation, wrong-rotation posture feedback, curriculum isolation, exact pause/resume of adaptive training, actor-transfer parity, selector/motor gradient separation and full-dive qualification guards. Bounded paired training runs test execution of the learning path, not long-run generalization.

## Checkpoints and selection

Every update writes `latest.pt` and status atomically. Archives retain scheduled step counts. Exact resume includes optimizer, RNG states, simulator states, practice goals, levels, statistics, recovery bank and routine history. A changed objective uses explicit actor initialization, never an alleged exact resume.

`best-points.pt`, `best-execution.pt` and `best-clean.pt` preserve separate development champions. The incumbent's replacement guards include arms, body shape, takeoff and completed rotation, including per-category regressions. Qualification additionally requires competent full dives in every category and superiority to immutable initial, previous-production and inherited-best references on paired development worlds. Only then can the run use its one disjoint final cohort. Passing saves `qualified-for-review.pt`; nothing automatically publishes it.

This v14 native format needs a matching 233-input browser adapter and parity checks before export. Feeding it to the current 223-input browser runtime would be an incompatible-schema error. Existing browser assets remain unchanged.

## Research basis and limits

- [Crawford, Learning Control of Complex Skills (1998)](https://digicoll.lib.berkeley.edu/record/139678/files/ERL-98-53.pdf): coordinating lower-level physical control skills. Its use of predefined dives is not adopted here.
- [Koschorreck and Mombaur (2012)](https://link.springer.com/article/10.1007/s11081-011-9169-8): articulated diving control across contact and flight; optimal control, not evidence that PPO alone will converge.
- [Yeadon and Hiley (2018)](https://pubmed.ncbi.nlm.nih.gov/29408164/): timing and body-shape changes in twisting dives. No motion sequence is copied.
- [DeepMimic (2018)](https://arxiv.org/html/1804.02717v3): varied state initialization can make difficult motor learning reachable. DeepMimic uses imitation; this implementation uses explicitly marked motor practice and no imitation loss.

The implementations are original. These papers motivate testable changes, not a claim to reproduce their results.
