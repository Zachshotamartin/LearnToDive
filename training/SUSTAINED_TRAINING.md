# Continue until measured progress flattens

`train_until_plateau.py` resumes a complete checkpoint with the existing PPO,
reward, physical model, joint limits, and judge. It does not increase difficulty
bonuses or change the scoring rules to make a curve look better.

```sh
python training/train_until_plateau.py \
  --checkpoint training/ppo-normalized-entry-v3/evaluated/step-000193536000-update-00000100/checkpoint.pt \
  --output output/sustained-entry-v3
```

Defaults: evaluate 384 fixed development cases after each 2,048,000 new
interactions. Train at least 20,480,000 additional interactions. After that,
stop only when five consecutive evaluation stages bring no meaningful new
best in any individual dive or the aggregate: one percentage point of clean
completion, 0.1 execution points, 0.5 degrees less entry error, or two percentage
points of geometry passes. The five-stage patience window spans 10,240,000
interactions. These thresholds are engineering choices, not proof of convergence.
Small improvements accumulate across the window. Recovery from a regression to
an already attained score does not count as new progress.

There is no fixed maximum interaction count. A plateau ends at
`plateau-awaiting-review`, not “qualified.” Poor performance at a plateau requires
investigating optimization, representation, reward, or capacity. Fixed development
maps are reused for selection; untouched evaluation and browser/native agreement
are separate release requirements. No model is automatically published.

`output/sustained-entry-v3/status.json` records the current phase, child PID,
completed evaluations and pending training stage. `best.json` retains the best
evaluated complete checkpoint. Stage directories preserve optimizer, RNG,
environment state, source hashes, elapsed time and intermediate snapshots.

SIGTERM or SIGINT pauses and requests a checkpoint at the next safe update
boundary. Run the identical command to continue. A new output and prefix can
branch from any complete saved checkpoint, including the evaluated best.
