# Autonomous diving training v10

One shared hybrid PPO policy chooses a legal declaration, then operates nine bounded joint controls in MuJoCo. The trainer supplies a category, apparatus, height and routine history, not a specific dive. Categorical and motor outputs share the trunk; they are not separate specialist policies.

Official difficulty values are recorded in `difficulty.json` with a source URL and PDF hash. Reproduce them with `python training_v3/extract_difficulty.py downloaded-regulations.pdf`. The declaration mask enforces category, available apparatus/height and no repeated dive number across positions. Flying-action declarations remain excluded until a separate flying-phase judge exists. Official scored heights are springboard 1/3 m and platform 5/7.5/10 m; arbitrary heights must not claim official table values.

Execution is an explicitly approximate geometric judge, separate from official difficulty. It observes the full water crossing, pointed feet, leg/hand placement, takeoff, rotation and twist completion. Failed declarations get no difficulty points. Water and splash remain physical approximations. Score is three times difficulty times execution; continuous failure feedback helps learning but is not shown as competition points.

Controls have native joint torque limits and no root actuator. Platform support is rigid; springboard starts at static preload. Armstand starts with supported hands. Recovery practice comes only from this learner's physically reached states with the same declaration, height and apparatus; full routines are retained. Adaptive category sampling never supplies a named maneuver.

## Run and resume

`python training_v3/run_suite.py --output /mounted/ssd/diving-run`

The controller compares 96x96, 256x256 and 256x256x128 across three seeds, then continues the median seed of the best development architecture. Default pilots: 10,240,000 interactions each. Continuation budget: 512,000,000 total interactions, with a plateau decision only after 102,400,000 and ten evaluation windows. An optional explicitly labeled motor warm-start arm is supported, but is not enabled by default.

SIGTERM the PID in `SUITE.json` to pause at a saved update boundary. Repeat the identical command to resume. Checkpoints retain optimizer, RNG, physical worlds, routine history and recovery bank. `latest.pt`, immutable million-step checkpoints and separately evaluated `best.pt` are retained. Source changes invalidate exact resume. Budgets are limits, not claims that a model is good.

## Verification and release boundary

Run `python -m unittest discover -s training_v3 -p 'test_*.py'`, `python training_v3/audit_physics.py`, and `python training_v3/check_browser_parity.py`.

The standalone browser inference module is `src/core/autonomousPolicy.js`; it is not plugged into the existing published v9 runtime. No training script replaces website assets. After training, frozen unseen routines, perturbed entries, category coverage, physics/browser parity, and actual browser playback must qualify a checkpoint before switching the app to v10. All development exports are marked unqualified.
