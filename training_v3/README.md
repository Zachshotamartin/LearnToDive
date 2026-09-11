# Autonomous diving training v11

One shared hybrid PPO policy chooses a legal declaration, then operates nine bounded joint controls in MuJoCo. The trainer supplies a category, apparatus, height and routine history, not a specific dive. Categorical and motor outputs share the trunk; they are not separate specialist policies. Nothing in the trainer scripts a takeoff, a flight shape or an entry: the open-loop jumps in the tests are feasibility probes of the physics only.

Official difficulty values are recorded in `difficulty.json` with a source URL and PDF hash. Reproduce them with `python training_v3/extract_difficulty.py downloaded-regulations.pdf`. The declaration mask enforces category, available apparatus/height and no repeated dive number across positions. Flying-action declarations remain excluded until a separate flying-phase judge exists. Official scored heights are springboard 1/3 m and platform 5/7.5/10 m; arbitrary heights must not claim official table values.

Execution is an explicitly approximate geometric judge, separate from official difficulty; `judge.py` documents which World Aquatics articles each check approximates. It observes the full water crossing, pointed feet, leg/hand placement, takeoff, rotation and twist completion. Failed declarations get no difficulty points. Water and splash remain physical approximations. Score is three times difficulty times execution. The training signal is a dense version of the same judged components plus a bounded completion bonus, so a dive whose official execution is clipped to zero still receives gradient toward each fault; it is never positive for a failed dive and extra spins never pay.

## What changed in v11 (review repairs)

The v10 pilots never scored: the start pose stood on the toe corner with the centre of mass past the platform edge, the leg servos braked any joint moving faster than their slewing target, and eight judged deductions summed past ten points on every dive, so execution and its gradient were identically zero. v11 starts flat-footed with the centre of mass over the feet (randomised lean, toe overhang and knee bend; the controller must balance, press and jump), uses athletic bounded torque caps with low servo damping and a 14 rad/s leg target slew, and calibrates water drag so axial flow through slender parts is streamlined (a 10 m entry now reaches about 3.5 m instead of 1.2 m). A bounded random search over open-loop jumps reaches 0.34 m of rise forward and 0.57 m back-facing; the policy has to find its own. The judge's sideways-entry measure read angular velocity (it now reads sideways linear speed), preparation bounces now require an upward hop rather than contact flicker, and position faults follow the rule's 0.5 to 2 point range. The armstand start is balanced over shoulder-width hands. PPO weights every decision once with a separate declaration entropy coefficient, value targets from recovery-practice declarations are excluded, the exact-resume contract hashes only the learning files, a run resumed after its final update still produces its final evaluation, and the suite refuses to rank architectures that all scored zero.

Controls have native joint torque limits and no root actuator. Platform support is rigid; springboard starts at static preload. Recovery practice comes only from this learner's physically reached states with the same declaration, height and apparatus; full routines are retained. Adaptive category sampling never supplies a named maneuver.

## Run and resume

`python training_v3/run_suite.py --output /mounted/ssd/diving-run`

The controller compares 96x96, 256x256 and 256x256x128 across three seeds, then continues the median seed of the best development architecture. Default pilots: 10,240,000 interactions each. Continuation budget: 512,000,000 total interactions, with a plateau decision only after 102,400,000 and ten evaluation windows. An optional explicitly labeled motor warm-start arm is supported, but is not enabled by default.

SIGTERM the PID in `SUITE.json` to pause at a saved update boundary. Repeat the identical command to resume. Checkpoints retain optimizer, RNG, physical worlds, routine history and recovery bank. `latest.pt`, immutable million-step checkpoints and separately evaluated `best.pt` are retained. Changing a learning file invalidates exact resume; the suite controller and audit scripts are outside that contract. Budgets are limits, not claims that a model is good.

## Verification and release boundary

Run `python -m unittest discover -s training_v3 -p 'test_*.py'`, `python training_v3/audit_physics.py`, and `python training_v3/check_browser_parity.py`.

The standalone browser inference module is `src/core/autonomousPolicy.js` (format `self-declared-diver-v11`); it is not plugged into the existing published v9 runtime, and `training_v3/diver.xml` is this trainer's model, not the browser's `public/physics/diver.xml`. No training script replaces website assets. After training, frozen unseen routines, perturbed entries, category coverage, physics/browser parity, and actual browser playback must qualify a checkpoint before switching the app to v11. All development exports are marked unqualified.
