# Remaining rotation continuation — 2026-09-14

Enable `--rotation-progress` for the two extra measured inputs and potential shaping. The default preserves the old 233-feature adaptive motor observation; the enabled version has 235 features. Competition judging, physical joints, torque limits, declarations, and completion-first terminal scoring are unchanged.

## Signals

The base observation inserts two columns at 214, before the action history. The motor curriculum then inserts its existing task context before the final nine action-history columns, as before.

- Flip remainder: signed declared turns minus unwrapped measured turns, divided by five. Sign is the same world-axis convention used by the judge; reversing direction does not count toward a forward declaration.
- Twist remainder: declared twists minus the magnitude of unwrapped measured twists, divided by five. Either twist hand is legal; a negative remainder means too many twists. Back-and-forth twisting uses net unwrapped rotation, not accumulated absolute travel.

Both features are zero while selecting a declaration. Current rotation, speeds, position and original declared goals remain available.

## Earlier feedback, unchanged terminal objective

For each count, define `c(error) = abs(error) / (1 + abs(error))` and `Phi = 2 * (c(initial error) - c(current error))`. Phi is largest at the requested count; undershooting, overshooting, and wrong-way movement reduce it. At zero initial rotation Phi is zero for every declaration. No trajectory, specific joint pose, desired speed, or predicted future is supplied.

Use the existing shaping mechanism: `F = 0.995 * Phi(next) - Phi(current)`, with zero potential on every terminal transition, including failures and timeouts. Discounted shaping telescopes, so waiting or repeatedly reversing rotation cannot create additional total reward. This redistributes learning feedback; it is not a per-frame bonus for spinning and does not change the competition score. Fundamental motor tasks keep their existing separate rewards; full dives and assigned full-dive practice receive the added count shaping.

## Continue without resetting learned weights

`--continue-from` with this flag explicitly migrates a compatible completion-first, split-trunk 233-feature motor checkpoint into a separate output phase. It preserves every old model parameter, PopArt value normalization, Adam step/moment, cumulative training count, and curriculum mastery. Two zero columns are inserted into each first-layer weight and optimizer moment tensor. Initial policy behavior is therefore preserved. Four critic-only updates calibrate values before actor updates resume.

Partial episodes and recovery snapshots remain in the parent checkpoint. New episodes start at the handoff so old returns cannot mix with new shaping. Source/physics/contract guards reject incompatible migrations. New-phase checkpoints subsequently use ordinary exact resume with the flag retained.

The operational continuation is `/Volumes/Zach's SSD/PortfolioTraining/2026-09-14-rotation-progress`. `continue-diver.py` uses its frozen source, latest full-state checkpoint and original cumulative target (595,824,640 steps). This is a process checkpoint/relaunch, not in-process source hot reload or fresh-weight training. Hide-and-seek is independent and remains running.

Inherited best checkpoints retain their original source contracts and observation sizes. A best model is replaced only when the existing selection criteria improve; the new schema alone does not make a model better. Browser publication must use the correct observation contract, never substitute a 235-feature checkpoint into an old 233-feature client.

## Verification

`python -m unittest test_rotation_progress test_completion_first test_declared_form test_training test_motor_curriculum -q`

The tests cover under/overshoot, reverse/inward signs, either twist hand, zero-twist declarations, discount-correct cancellation, observations matching the judge, exact old weights/Adam moments, initial forward behavior, PPO updates, exact resume, assigned-dive evaluation and motor-skill evaluation. Real-checkpoint verification is recorded in the operational folder as `MIGRATION_VERIFIED.json`. Evaluation logs now report mean flip/twist error and separate flip shortfall/overshoot.
