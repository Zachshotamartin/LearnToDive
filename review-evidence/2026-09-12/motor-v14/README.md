# v14 implementation and short-run evidence

- `tests.log`: 53 passing native tests. The deliberate nonfinite-state regression emits a MuJoCo warning; it is testing detection of an unstable state.
- `comparisons.json`: two seeds, 102,400 steps each, terminal feedback vs phase feedback vs phase feedback plus motor practice. Matching inherited actor and 256/256/128 motor/selector widths. The fresh critic still shared movement features in these preliminary runs.
- `critic-comparisons.json`: two more 102,400-step runs after isolating the critic's feature network. Same parent actor, separate fresh critic, phase feedback and motor practice. Constructor randomness differs with the added network; these are bounded diagnostics, not a statistically established architecture comparison.
- Full-dive evaluations use eight worlds / 48 dives each; motor evaluations use eight independent practice contexts per task. All final clean full-dive rates were zero. Some motor errors improved, some regressed. No short-run actor was promoted or used to initialize the long run.
- `twist-feasibility.json`: joint-limited control search and independent angular-velocity replay. This establishes axial twisting capability from actual board contact, but the witness has excessive lateral tilt and does not demonstrate a valid competitive dive. Its control sequence is not used by the trainer.
- `launch.json`: immutable source and parent fingerprints, command, budget and checkpoint cadence for the replacement run. The inherited actor is the original v13 best-points checkpoint, not a selected short-run result.

Raw before/after episode reports, optimizer checkpoints, frozen comparison sources and controller scripts remain on the SSD under the launch root. Competition scores remain separate from practice success. No browser asset was updated.
