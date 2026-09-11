# Saved training points and continuation

New runs save an immutable directory under `training/<name>/checkpoints/` at the initial update boundary, every 204,800 additional interactions or 300 seconds (whichever comes first), at the completed target, and after a requested SIGINT/SIGTERM. Signals finish the current PPO update before saving. A hard kill or power failure can lose work since the last saved boundary; arbitrary unsaved instants are not recoverable.

Each directory contains `checkpoint.pt`, its matching `policy.json`, and a hash manifest with UTC time, lineage steps, PPO versus critic-calibration counts, completed updates, segment elapsed time and cumulative known elapsed time. `latest.json` atomically points to the latest complete directory. The compatibility `latest.pt`/`policy.json` copies are replaceable; immutable directories are the source of truth. Existing run directories are never reused.

A v2 checkpoint contains actor, critic, conditional Gaussian variance, complete Adam state, Python/NumPy/Torch RNGs, the environment generator state, all live per-world physical and judging arrays, source hashes, configuration and counters. Native Rollout receives explicit physical state every call; worker scratch buffers are not continuation state. Tests compare actual resumed MuJoCo/PPO updates to uninterrupted updates, including model, optimizer, RNG and complete environment arrays.

Continue **any saved v2 checkpoint** in a new directory:

```sh
python training/train.py --name continued-entry \
  --resume training/previous/checkpoints/step-000193536000-update-00000100/checkpoint.pt \
  --resume-mode exact --additional-steps 2048000
```

Exact mode inherits the saved training configuration, restores physical/RNG/optimizer state and requires identical frozen training sources and physics. Segment-only operations such as optimizer reset or critic calibration are not repeated. Exact continuation is guaranteed only on the same native runtime/platform; native and browser contact dynamics are not promised bitwise identical.

`--additional-steps` (also accepted as `--steps`) means **more** interactions, including after a previous target completed. Alternatively, `--target-steps 200000000` requests a cumulative lineage counter. Requests round upward to a complete PPO batch, and the checkpoint records both the request and actual target. A target already reached is rejected. Batch-boundary timing means a wall-clock checkpoint can occur one update after its configured interval.

Older checkpoints saved RNG values but never restored them and omitted complete environment state. They cannot resume exactly. Use a named continuation explicitly:

```sh
python training/train.py --name changed-objective \
  --resume training/old-run/latest.pt --resume-mode restart \
  --reset-optimizer --additional-steps 2048000
```

Restart mode records freshly reset worlds and RNGs. Omit `--reset-optimizer` only when intentionally retaining compatible Adam state. Changed physical contracts require the additional explicit migration flag and a fresh optimizer. Parent snapshots, hashes and history remain unchanged. Elapsed time before a legacy checkpoint is marked unknown rather than invented.

`python training/select_best.py RUN EVALUATION_DIRECTORY` registers an immutable fixed-case evaluation with its resumable checkpoint. The evaluation must match checkpoint coefficients, policy bytes, physics, judge, evaluator source and ordered case hashes. Candidates are compared by minimum per-skill clean rate, overall clean rate, all-case mean execution, worst-entry angle, then minimum per-skill execution. Every failure remains in the averages. `best.json` retains the earlier model if a newer one ranks worse. It points to an evaluated **development best**, independently of `latest`; it does not claim qualification or publish anything. Per-skill qualification gates are included in the record. The separately seeded native/WASM release qualification still must pass.

The current corrected objective is `normalized-whole-entry-once-v3`: full-window angle, minimum form and bounded geometry are scored once. It removes duplicated water geometry charges that previously rewarded better hand form and a shorter entry even while the body tilted more. All physical limits and the independent judge remain unchanged.
