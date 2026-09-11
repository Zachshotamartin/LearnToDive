# Surface-entry training v9

Isolated from `training/` and the accepted browser runtime. Old runs remain resumable using their original sources. This is a weights migration with fresh optimizer state, not additional experience on the old training contract.

- Entry is measured as each body part crosses the water. Hand alignment/separation, shoulder/elbow extension, hip/knee extension, foot lines and leg separation are sampled at the relevant crossings. Completed crossings are latched; later underwater movement cannot alter them. Body alignment continues through the legs/feet to catch tail-flop entries.
- `validEntry` retains broad safety/form thresholds. `cleanEntry` adds <=15 degrees and >=0.8 form; `cleanCompletionRate` additionally requires the requested rotation, position and legal launch/execution. These are simulation criteria, not official judging cutoffs.
- A bounded smooth alignment/form reward prioritizes entry. Difficulty credit is gated by valid entry. Splash is explicitly a geometric/velocity proxy; the water solver does not simulate a free surface or resolve a rip entry.
- Entry practice replays reachable descending states collected from actual full dives. No prescribed action labels or trajectory overrides. Practice fraction falls from 75% to 10% as entry success improves. Full dives always remain in training. Practice metrics are separate from full-dive metrics; evaluations always begin at takeoff.
- Checkpoints include optimizer, RNG, environment, curriculum bank and counters. Separate source hashes prevent accidental exact resume across edits.
- Controller evaluates fixed full-dive cases every 2,048,000 steps with both v9 and the unchanged legacy judge. Keep best snapshots; plateau is not a success declaration. Final held-out evaluation remains separate.

References: https://www.usadiving.org/about-us/diving-101/judging-and-scoring and https://www.usadiving.org/about-us/diving-101/glossary . Engineering thresholds are checked against the existing physically reachable straight/pointed-toe pose and malformed pose tests, not represented as official scores.

Active run: `output/surface-entry-v9/status.json`. Command and PID are recorded in `output/surface-entry-v9/launch.json`. To pause, send SIGTERM to the controller PID; it saves at an update boundary. Re-run the recorded command to resume. To continue after plateau, start a new controller output from the selected complete checkpoint.
