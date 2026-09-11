# Review snapshot: Diving

This branch preserves the current source and development browser assets. It is not a model-quality or production-readiness claim. No GitHub Actions were added.

## Checks run before push

- `npm run build`: passed.
- `npm run test:runtime`: passed for trained and initial models in an actual browser, without page errors.
- `npm test`: 49 passed / 0 failed. Stale native geometry references were regenerated independently and fingerprinted; tolerances unchanged.
- The preceding native training verification passed 9 diving tests and 17 focused hide-and-seek tests, plus a bounded-torque/momentum audit and pure browser-policy inference parity. Native training verification remains separate from browser/model quality.

## Training boundary

Current runs execute from frozen source copies on the owner's SSD. This Git push does not restart them or replace the site's model. Native optimizer/checkpoint archives and transient output directories are not committed. Public model files are the existing development exports. Read the version-specific trainer README for the new training path; the root README describes the existing browser runtime.

New system: `training_v3` (self-declared v10), with `training_v2` and `training` preserved for provenance. At the status check, the first 96x96 pilot completed 10,240,000 steps; the second had 9,472,000. The first pilot's final fixed-development evaluation had 0 mean execution/points and 0 clean entries; it is not qualified. The new v10 inference module exists separately and is not wired into the published browser.
