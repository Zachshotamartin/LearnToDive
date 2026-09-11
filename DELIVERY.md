# Delivery selection — development, not frozen

The parent task owns Git operations, package pins and publication. Do not stage the repository wholesale. The ignore rules preserve all local training/visual diagnostics without selecting them for delivery.

After qualification, the exact delivery list will include:

- Runtime: `src/`, the model/XML files named by `public/asset-manifest.json`, that manifest and authoritative `public/physics/diver.xml`.
- Package/build: `package.json`, `package-lock.json`, `index.html`, `vite.config.js`, `playwright.config.js`, `.gitignore`.
- Verification: `tests/`, `scripts/build-mujoco-csp.mjs`, `scripts/vendor/`, `scripts/audit-memory.mjs`, `scripts/audit-policy.mjs`, `scripts/capture.mjs`, and the retained visual-audit scripts that work with the selected final fixture.
- Reproduction: `training/train.py`, `training/sim.py`, `training/contract.py`, `training/evaluate.py`, `training/export_assets.py`, `training/freeze_policy.py`, `training/check_portable_resume.py`, `training/generate_fixtures.py`, `training/qualify.py`, `training/inspect_policy.py`, `training/requirements.txt`, `training/tests/`, `training/evaluation-protocol.json`, `training/candidate-skills.json`, and the genuine current-contract initial policy.
- Selected release evidence: one native optimizer checkpoint, exact source/config/counter provenance, frozen policy and initial weights, complete final evaluation cases and reports, runtime audit, and a concise account of superseded failed judges. These will live in `training/release/`; current phase folders are not the release bundle.
- Documentation/licensing: `README.md`, `NOTES.md`, this file, `LICENSE`, `THIRD_PARTY_NOTICES.md`, and current artwork provenance.
- Examples: two newly captured accepted-controller scene PNGs plus reproducible capture provenance. The old example PNGs must be replaced before publication.

Local `training/ppo-*`, `training/diagnostics`, exploratory evaluation/validation folders, historical initial weights, `output/` and earlier artwork fixtures are intentionally excluded. They have not been deleted. Final checkpoint continuation will be tested from a temporary copy containing only the selected source, physical model and checkpoint; it must not depend on an ignored local phase directory.
