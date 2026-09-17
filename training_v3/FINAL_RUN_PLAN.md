# Diving: everything to change before the last restart (2026-09-15)

Status: implemented as v13 on 2026-09-15 (see `README.md`, section "What changed in v13"). Items not implemented: the learning-rate anneal (5.22) and the stochastic evaluation (7.36), both deferred as low-risk follow-ups; the springboard hop deduction (12d) applies automatically because bounces are a judge deduction in the training ledger.

Evidence base: the live run `2026-09-14-bounded-fresh` at 100.8M steps (evaluation `100352000.json`), its frozen source, the three earlier reward versions (v11, v12, completion-first), and the open-loop ceiling probes run today (`scratchpad/dive_ceiling_search*.py`, logs `ceiling-10m.log`, `ceiling-5m.log`, `ceiling-3m-board.log`).

## 0. The one fact that changes the diagnosis

A random search over open-loop servo schedules, with no learning at all, found forward dives (101A) that the live judge scores **9.05 / 10 with a 5.7° entry from the 10 m platform, 8.66 from 5 m and 8.83 from the 3 m springboard**, each within a few hundred candidates (section 8). The environment, the judge and the actuators can produce a near-clean dive on every apparatus; a dumb search finds one in minutes. After 100M PPO steps the learner's mean execution is 0.27. Whatever is wrong is in the learning problem, not the physics.

The second fact is the flip side. The same best 10 m schedule replayed under the training stance randomisation (lean, knee, toe-over) gives a median entry of 63° at half the randomisation width and 89° at the full width, with only 33 of 64 dives still valid. The entry angle is a razor-sharp function of the takeoff; the learned policy's 87° median is what an approximately open-loop policy produces. A diver has to sense the stance and correct in flight, and the reward has to pay for the correction.

## 1. Objective (root cause; must change)

1. **Entry alignment is worth 0.36 reward out of a scale where rotation progress is worth 16.** Going from today's 87° entry to a clean 15° is worth 0.145. Nothing else on this list matters until this is fixed.
2. **Stop re-weighting an additive sum.** v11 (entry 4.2 / takeoff 0.375) produced a faller; v12 (takeoff added) produced a flat-landing jumper; completion-first (rotation 16 / entry 0.36) produced a flat-landing spinner. A dive is conjunctive. Make the credit conjunctive:
   - `credit = 16 · progress · entryQuality`, with `entryQuality = 1 / (1 + angle / 15°)` (1.0 at vertical, 0.5 at 15°, 0.15 at 87°). Rotation still pays, but only through a vertical entry.
   - Keep every judge deduction as a dense cost **at the judge's own weights** (alignment up to 6, position 2, form 2, takeoff 1.5 + lean 1, feet/legs/hands 3, lateral 1, distance 2). Do not renormalise them into a shared budget of 4; that is what erased the entry.
   - Keep the takeoff outcome costs from v12 (rise short of 0.3 m, upward speed short of 2 m/s, lean beyond 15°) at their v12 weights.
   - Keep Codex's potential-based rotation shaping exactly as is (it telescopes to zero and cannot be farmed).
   - Safety failures (no water, board contact, incomplete entry) stay above the maximum credit but on the same scale (16, not 32).
3. **Count error tolerance in the credit, not the judge.** The judge's ±0.25-turn validity is a rule; the credit should use the continuous error so a 10° miss is worth more than an 80° miss.
4. **Difficulty pays only for clean dives.** `points = 3 · DD · execution` already does this in the judge; the training credit must not add a DD bonus for merely declaring a hard dive (completion-first got this right; keep it).
5. **Position credit through the position deduction only** (2 points, judge weight), using the 1/(1+d) kernels so the slope never vanishes. Drop the separate `motorPositionQuality` path if it disagrees with the judge's.
6. **One reward function.** Delete the `reward_mode` switch and the four versions; the run must not carry dead alternatives that can be re-enabled by a flag.

## 2. Task scope and curriculum (second cause)

7. **Narrow first, widen on mastery.** Today every routine forces all six groups (forward, back, reverse, inward, twisting, armstand) on both apparatus at five heights, and the evaluation shows 13 distinct declared dives. Start the final run on the 10 m platform, groups 1 and 2, at most 1 somersault, positions A and C. Widen the declaration mask automatically when the clean rate on the current set exceeds 30% over three evaluations: add group 4 (inward), then 3 (reverse), then 1.5 somersaults, then the twisting group, then armstand, then 5 m and 7.5 m, then the springboard. This is the same self-declaration mechanism with a smaller legal set; nothing prescribes a motion.
8. **Springboard as a second stage.** Board recoil is a different takeoff skill; do not split the early budget across both apparatus.
9. **Make the entry sub-task the gate and make it winnable.** The curriculum's entry-only task starts the athlete in the air and has succeeded 0 of 24 times with mean error 4.4 against a threshold of 0.8, so its difficulty level has never risen from 0. Start it from near-vertical, slowly rotating states and widen the start distribution only as the success rate passes 30%; today's random pitch ±0.65 rad and random angular velocity make the first success improbable. Do not expand full-dive credit until this task succeeds more than half the time.
10. **Auxiliary single-dive episodes early.** Raise the auxiliary practice fraction from 25% to 60% for the first stage so most episodes are one dive from the mastery set rather than a six-dive routine.
11. **Recovery bank stays** (own states only, same declaration), rate 0.15 to 0.35.
12. **Reverse curriculum from the learner's own late-flight states** (banked states within 0.5 s of the water with the right rotation) as a separate task, so entry timing gets many short, dense episodes.

## 2b. Start-state randomisation (new, from the sensitivity probe)

12b. **Randomisation curriculum.** Train the first stage with the stance randomisation at a quarter of its width (lean ±0.015 rad, knee ±0.04, toe-over ±0.015) and widen it with the clean rate, the same way the mastery gate widens the declaration mask. The full width is the right end point for robustness, not the right starting point for a policy that has not yet learned to enter vertically once.
12c. **Keep the evaluation stance fixed** (it is) so progress is measured on one target, and add a second evaluation at full randomisation so the gate to widening is honest.
12d. **Springboard hop detector**: keep it, but in the training reward charge a bounce as a deduction (2 points) rather than a failure (32) during the first springboard stage; the judge still fails it.

## 3. Observation

13. **Add the ballistic time-to-water** (`(vz + sqrt(vz² + 2 g h)) / g`, clipped). v10 had it; v11 removed it as "a future estimate". It is what a diver sees. Timing the extension is the skill that is missing.
14. **Rescale height above water.** `above_water / 10` compresses the last metre, where timing is decided, into 0.1 of a unit. Use `min(above_water, 3) / 3` and keep the /10 copy.
15. **Keep the remaining-rotation inputs** Codex added (they match the judge's signed unwrapped measurement).
16. **Running observation normalisation** (mean/variance tracked, frozen at export) instead of fixed constants. Standard in every mature PPO implementation; the fixed scalings here were guessed.
17. Keep: pelvis quaternion and angular velocity, joint positions and velocities, servo targets, previous action, water fraction, contact flags, intent, group, round, used dives.

## 4. Exploration (why PPO fails where random search succeeds)

18. **Temporally coherent exploration.** The motor noise is Gaussian per 20 ms step with AR(1) ρ = 0.6, a correlation time of about 40 ms. A dive is made of 200 to 400 ms phases; the open-loop search explores in that space and finds 9/10 dives in 200 tries. Raise ρ to 0.9 to 0.95 (correlation 200 to 400 ms) or use pink noise (Eberhard et al., ICLR 2023). This is the single largest exploration change available and costs nothing.
19. **Do not let the standard deviation collapse before the first clean dives.** Floor stays at exp(−2.8); hold the entropy coefficient at 0.006 until the clean rate is nonzero, then anneal.
20. **Explore the takeoff timing directly.** Keep the stance randomisation (lean, preload, knee, toe-over); add ±0.2 m/s random initial joint velocities for the first stage.

## 5. Optimisation

21. **More samples per update.** 64 envs × 160 steps = 10,240 samples per update is small for a 9-dimensional continuous task with terminal rewards; use 256 envs (65,536 per update) with 3 epochs. Throughput is CPU-bound; pause the hide-and-seek suite while diving trains (12 cores) or accept half speed.
22. **Learning-rate anneal** from 3e-4 to 1e-4 over the run; entropy anneal with it once clean dives exist.
23. **GAE λ 0.99, γ 0.995 stay** (terminal reward, 2 to 4 s episodes).
24. **Split critic stays** (Codex); add value clipping (0.2) and keep PopArt.
25. **KL early stop 0.02 stays**; log how often it triggers (it did at 0.016 in the last update).
26. **Network 256×256×128 tanh stays**; the architecture sweep already chose it. No recurrent net: the state is fully observed.
27. **Seeds.** Run three seeds for the first 10M steps, continue the best by clean rate. With hide-and-seek paused this fits the machine.

## 6. Physics and engine checks (all verified, listed so they are not re-done)

28. Actuators: hips 240 N m, knees 260, ankles 180, shoulders 120/70, elbows 70, kv 6 to 12; the servo-braking bug is gone; slew 14 rad/s on legs. Enough for a 9/10 forward dive (section 0).
29. Stance: flat-footed, centre of mass over the feet, balanced for more than 1 s without control.
30. Water: streamlined axial drag; 10 m entry reaches about 3.5 m depth; entry scoring stops at full submersion or 1.2 s.
31. Failure triggers that are hard zeros: departure past horizontal, feet leaving more than 40 ms apart, non-foot board contact, rotated recontact, sideways tumbling. 25 of 144 evaluated dives still fail on takeoff contact. Keep them as failures but report which trigger fired in every evaluation so a trigger that blocks learning is visible.
32. Judge tolerance: "clean" requires entry ≤ 15°, form ≥ 0.8, valid geometry, valid limbs, valid arms. Check against the best scripted dive from section 8 that "clean" is reachable with this rigid rig; if the scripted 9/10 dive is not clean, the form thresholds are the reason and must be loosened to what the rig can do (documented, not hidden).

## 7. Evaluation, metrics and gates

33. **Gates instead of a 512M-step wait.** Gate 1 at 10M steps: entry sub-task success above 50% and at least one clean dive on the fixed worlds. Gate 2 at 30M: clean rate above 20% on the mastery set. A failed gate stops the run for a change; it does not wait for a plateau.
34. **Primary metric is clean rate on the current mastery set**, then execution, then points. Best-checkpoint selection by that order (Codex added `best-clean.pt`; make it the default `best.pt`).
35. **Report per declared dive**: clean rate, entry-angle percentiles, rotation error, jump rate, rise, lean, which failure trigger fired.
36. **Evaluate stochastically as well as deterministically.** A deterministic mean action can hide a policy whose sampled behaviour is what actually trains.
37. **Do not read "rotation completion" as progress.** It is within ±90°.

## 8. Physics ceiling (open-loop probes, today)

| apparatus / height | best execution | entry angle | rotation | rise | valid in last batch of 64 | form | clean |
|---|---|---|---|---|---|---|---|
| platform 10 m | 9.05 | 5.7° | 0.48 turn | 0.40 m | 15 | 0.79 | no (form 0.79 < 0.80) |
| platform 5 m | 8.66 | 8.1° | 0.48 turn | 0.38 m | 37 | 0.73 | no (form) |
| springboard 3 m | 8.83 | 7.1° | 0.48 turn | 0.44 m | 29 | 0.67 | no (form) |

Robustness of the best 10 m schedule replayed over 64 stances: at zero randomisation all 64 score 9.03; at half the training width 50 valid, median entry 63°; at the full width 33 valid, median entry 89°, failures split between short rotation, wrong end first and takeoff contact. On the springboard 28 of 64 random schedules fail as a "double bounce", so the hop detector is a live trap for a learner pressing the board.

All three best dives fail "clean" only on the form threshold (0.80), by 0.01 at 10 m, with every other clean condition met. The form measure is `1 / (1 + loss)` over thirteen weighted joint and hand terms; 0.80 means a summed loss of 0.25, which a scripted dive with straight legs and hands together barely misses. Set `CLEAN_FORM` to 0.7 (loss 0.43) and report the graded form value; otherwise the headline metric can stay at zero for a diver that is visibly diving. Robustness of the best 10 m schedule to the stance randomisation: [pending].

Nothing from these probes is used for training. They only bound what the reward can ask for.

## 9. Run management

38. Freeze the source per run (done), record amendments (done), gates in `STATUS.json`.
39. Pause hide-and-seek for the diving run, or give diving 8 of 12 cores.
40. Keep the evaluation tick budget derived from the physics (fixed 2026-09-12).
41. One reward, one observation layout, one curriculum; delete the alternatives from the source copy so a resume cannot drift.

## 10. Stop doing

- Re-weighting the additive reward after each symptom.
- Training six groups, two apparatus and five heights from step zero.
- Treating ±0.25 turn as "completed".
- Continuing past a failed gate because the budget is not used up.
- Publishing development exports as if they were models.

## 11. Order of implementation

1. Reward (section 1) with unit tests: a flat-landing spinner must score below a vertical half-somersault; a jump must beat a fall at equal entry; the credit must be monotone in entry angle and in rotation error.
2. Exploration ρ, time-to-water input, height rescale, running normalisation.
3. Mastery-gated declaration mask, randomisation curriculum, the winnable entry task, clean-form threshold 0.7.
4. Gates and per-dive reporting.
5. Three seeds × 10M steps on 10 m platform, groups 1 and 2; continue the best.

## 12. What success looks like, so the gate is not moved later

- 10M steps: the entry sub-task succeeds more than half the time; at least one clean 101A or 201A on the fixed worlds.
- 30M: clean rate above 20% on the mastery set at quarter randomisation; execution above 5 on clean dives.
- 100M: forward, back and inward dives at 1 somersault, clean rate above 20% at full randomisation, springboard stage started.
- If 10M passes and 30M fails, the problem is exploration or credit assignment, not the reward. If 10M fails, the reward or the entry task is still wrong; stop and fix rather than continue.

Scripts and logs behind every number: `scratchpad/dive_ceiling_search.py`, `dive_ceiling_search_board.py`, `dive_ceiling_robustness.py`, logs `ceiling-10m.log`, `ceiling-5m.log`, `ceiling-3m-board.log`, task output `ba4m2etib`.


## 13. Addendum after the first gate failure (2026-09-15, run `2026-09-15-v13-final`)

The first pilot stopped at 10.24M steps with entry sub-task success 0.0 and no clean dive. Probes on the frozen source (see README, "What changed in v13.1"): holding the straight entry pose passes the entry task 69% of the time, the centred default pose never does, and under the coherent exploration noise at σ 0.61 even the held pose passes 0% (25% at σ 0.14, 56% at σ 0.08). The learned noise scale drifted upward during the pilot. Section 4 of this plan chose coherent noise so PPO could discover what random search found; it kept the historical marginal scale, and a coherent perturbation of that size is not averaged away by the servo. Fix: start the motor head at the entry pose, start the exploration at σ 0.135 (`--motor-logstd -2`) and lower the motor entropy bonus to 0.002. Section 12's success criteria and the gates are unchanged.


## 14. Addendum after the second gate failure (2026-09-16, run `2026-09-16-v13.1-final`)

Entry success 0.29, no clean dive, but 50% valid dives and points up to 11.5. Every forward dive failed at the takeoff and the takeoff practice task never succeeded (0 of 23,769). The task demanded the competition takeoff (0.3 m, 2.8 m/s) at every level, so its difficulty ladder never engaged. Fix: the takeoff task starts at a 0.1 m hop and widens to the competition takeoff with mastery (README, "What changed in v13.2"). Gates unchanged.
