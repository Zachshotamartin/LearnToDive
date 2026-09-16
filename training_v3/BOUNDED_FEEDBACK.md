# Bounded fresh diver — 2026-09-14

The previous rotation-progress inputs match the signed, unwrapped measurements used by the judge. GAE lambda was already 0.99. No direction-sign bug was found.

Two objective conflicts were found in code: the completion reward jumped by 16 + 2*DD (plus a count-weight reduction) at a binary completion/pose threshold, and reusable aerial practice rewarded a stationary orientation with an angular-speed penalty. These are plausible contributors, not proof of a single cause of the observed regression. The old fixed-target benchmark declined from 16/40 to 9/40 rotation completions and reported no clean entries; that evidence remains in the old run.

Changes:
- Keep every declaration index and official difficulty row, but mask training/preview choices to <=1.5 flips (<=2 for armstands), <=1 twist. This retains 108 position-specific declarations rather than training the unrestricted table.
- Keep signed remaining-flip and unsigned remaining-twist inputs, and valid potential-based shaping with terminal cancellation. Do not remove cancellation: that would allow reward farming.
- Replace threshold completion/DD motor credits with continuous progress from zero rotation toward each requested count. Each active axis contributes equally to a maximum combined credit of 16; undershoot, overshoot, and wrong directions lower it. Count-error deductions remain continuous at a fixed weight. No difficulty bonus for merely declaring a difficult dive.
- Keep each execution fault an independent bounded deduction (combined budget 4). Malformed pike/tuck still loses position quality and official execution; its classification no longer gates motor progress credit.
- Keep distinct safety deductions; incomplete entry, board contact and missed water each cost 32, exceeding the maximum progress credit. Wrong-end/plane still have their own deductions.
- Exclude stationary aerial stabilization from specific-dive curriculum sampling, while retaining real-board takeoff, shape, entry and full-dive practice. The legacy aerial slot remains available for standalone evaluation and checkpoint compatibility.
- Official judge, DD, category, entry-end, nonrepeat rules and physics are unchanged.

The user requested a fresh run: random model weights, empty Adam state, zero counters and curriculum mastery. Preserve the old run and its checkpoints. Save initial.pt before collecting experience; regular checkpoints and best models remain resumable. Do not silently publish an untrained replacement as best.

Validation includes monotonic progress, continuity across judging thresholds, reverse/inward direction, twist handedness, independent form deductions, all apparatus/height masks, nonrepeat masks, fresh initialization and exact checkpoint resume. Long-run improvement must be measured; passing tests does not establish learned competence.

References: Ng et al., Policy Invariance Under Reward Transformations (https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf); Schulman et al., Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438).
