# Learn to Dive

Current training implementation: [training_v3/README.md](training_v3/README.md). Browser runtime and training versions are intentionally separate while new models are evaluated. See [review status](REVIEW_STATUS.md) for known test failures and [the Claude review prompt](CLAUDE_REVIEW_PROMPT.md).

A local 3D experiment with an original pretrained neural joint controller. The athlete has independent physical body segments, 14 constrained joints and a free torso. A frozen goal-conditioned PPO actor reads the current state every 20 ms and commands bounded joint servos. MuJoCo 3.13 runs both offline training and browser physics from the same MJCF model.

**Local V8 development preview, not qualified. The selected frozen model has 193,536,000 lineage interactions, including 2,048,000 PPO interactions and 184,320 frozen-actor critic-calibration interactions on the current physical model. Its 384 fixed development cases yielded 0 clean dives, mean execution 1.83/10, and mean worst full-entry angle 19.59°. It is available for inspection while whole-entry hand and arm control remains incomplete.**

## Run

Node 22 or newer:

```sh
npm install
npm run dev
npm test
npm run test:browser
npm run build
```

The local demo runs at `http://127.0.0.1:5184`. All physics, models and artwork load locally. There are no uploads, inference APIs, visitor training or GitHub Actions.

```js
import { mountExperiment, metadata } from "@zachshotamartin/learn-to-dive";
import "@zachshotamartin/learn-to-dive/style.css";

const experiment = mountExperiment(element, {
  embedded: true,
  assetBase: "/assets/learn-to-dive/",
});
// Synchronous disposal releases UI, worker and graphics resources.
experiment.dispose();
```

`embedded` omits the standalone heading. `assetBase` is optional: without it, Vite emits and resolves the model/XML assets. Hosts supplying it must copy the files in `public/asset-manifest.json` to that base, preserving relative paths and verifying byte counts and SHA-256 hashes. `src/data/assets.js` names immutable content-hashed files. Three.js artwork and the exact MuJoCo WASM binary are emitted by Vite separately.

## Controls and practice rules

Choose an announced dive or a varied practice round, change board height, and compare the trained actor with its initial weights or a named hand-authored feedback controller. Pause, replay, use slow motion, scrub the physics timeline, inspect entry, or follow the diver. The result export includes every actual controller command and the scoring breakdown. Scene PNGs have a transparent background.

A practice round cannot repeat the same numeric dive code: 103B and 103C count as the same dive. The scheduler chooses goals; it does not generate movement. Eligibility will use measured per-skill height ranges after final evaluation. This variable-height springboard practice rig is not a complete competition format. Its reference difficulties come from the 2026 World Aquatics 3 m springboard table, even when the height changes:

| Dive  | Reference difficulty |
| ----- | -------------------: |
| 101C  |                  1.4 |
| 103C  |                  1.5 |
| 103B  |                  1.6 |
| 201C  |                  1.7 |
| 203C  |                  1.9 |
| 5132D |                  2.1 |

Execution is an independent computed estimate from 0–10 in half points. Displayed points equal `3 × execution × reference difficulty`. Wrong declared rotation or an invalid collision-assisted launch scores 0; a missed announced body position is capped at 2. Form, entry angle, splash proxy, a twist begun on the board and preparatory hops have explicit deductions. These are computed estimates, not human judges. No difficulty points are awarded for an incomplete extra rotation.

## What the controller learns

The actor is a 76→96→96→9 multilayer perceptron with tanh hidden layers. Inputs include physical joint positions/velocities, torso quaternion/angular velocity, center-of-mass state, stable foot-contact flags, observed rotation progress, water distance, current targets and the announced skill. Inputs are proprioceptive state, not rendered images. The C posture metric requires both bent hips/knees and each hand within 24 cm of its own lower-leg segment. Central posture coverage has a coarse 44 cm leg-separation limit; entry has the tighter physical gap criteria below. These tolerances account for the simplified rigid hand/leg geometry; fingers and grasp attachment are not simulated.

The first eight outputs command common hip, knee and ankle targets; independent left/right shoulder pitch and roll; and common elbow targets. The ninth commands paired physical hip adduction. These drive 14 independently simulated joints. Targets have a slew-rate limit; each actuator has a force/torque limit. The policy cannot set root position, velocity, orientation or angular momentum. It is evaluated throughout board contact and flight. Changing a midflight force changes its later body commands.

Each new dive is generated by a fresh local simulation. Replay and scrubbing reuse the physical frames recorded for that run. The model contains neural weights, with no saved motion trajectories or skill animation clips. The shared capsule-and-sphere stick figure follows the physical body orientations and endpoints; both learning games use the same original figure package. Rendering interpolates physical frames; it does not author flight actions. The round scheduler and comparison heuristic are separate from the learned actor.

## Physics

`public/physics/diver.xml` is the shared physical model. Units are kilograms, meters, seconds and radians. Native MuJoCo coordinates are X forward, Y lateral, Z up. Rendering maps them to `[X, Z + boardHeight, −Y]`.

- The athlete has 70 kg distributed among pelvis, torso, head, upper/lower arms, hands, upper/lower legs and feet. Joint limits, segment inertia, self-collision and physical board contact are handled by MuJoCo.
- The rigid board has one bounded vertical spring mode. Its 18,000 N/m spring, 140 N·s/m damper, 24 kg mass and deflection limits are explicit in MJCF. Preload is stored initial spring energy. Dives start from a crouched ready stance; no learned walking approach is claimed.
- Hip flexion/knee/ankle torques are capped at 160/180/100 N·m; shoulder pitch/roll and elbow torques at 65/45/35 N·m. Those servo targets slew at 8–10 rad/s. The additional hip-adduction motors have 80 N·m caps and 4 rad/s target slew. These are a simplified actuator envelope, not a complete human muscle model or a hard cap on externally induced joint velocity.
- MuJoCo integrates at 2 ms; the actor acts every 20 ms. Internal motor actions redistribute momentum. Only gravity, contacts and explicitly applied disturbances/fluids supply external forces. Finite-step integration has measurable numerical momentum error; tests bound it rather than claiming exact conservation.
- Water entry is detected from all 15 oriented athlete collider extents at the first 2 ms contact state. Scoring never uses a later pose at the end of a controller tick. The plane-framed quaternion measurement classifies somersaults using the principal upright-referenced takeoff pitch plus unwrapped airborne swing. It never includes accumulated preparation turns. Airborne rotation is also reported separately. Axial twist counts only after the final departure; lateral singularities cannot earn a forward/back code. This avoids rewarding a poor entry merely to rotate another exact 180° after a leaning takeoff.
- Board/athlete and stand/athlete contact forces are measured at every 2 ms physics step. Nonfoot board contact, any stand contact, pressing a foot against the side/underside, or a rotated recontact after a confirmed hop invalidates the launch and scores zero. Contact above 15 N or accumulated forbidden impulse above 0.1 N·s counts; thresholds ignore numerical grazing. Airborne status requires 40 ms without a board contact. Upright preparatory hops receive a one-point deduction each, capped at two. Final foot departures must be within 40 ms; takeoff lean must stay within ±60°. These are explicit computational approximations of balanced two-foot takeoff, not official timing or angle thresholds.
- Fixed stand colliders cover the exposed top slab, spring housing, four columns and upper rails. Column colliders continue below the deck to preserve the exposed surface across variable heights. Decorative diagonal braces and ladder details are not collision meshes.
- Static pool walls and floor match the visible basin. After entry, each submerged body receives bounded approximate drag and buoyancy inside the basin. The last motor target is held during underwater continuation. Splash graphics use the actual contact time and location; their size is a posture/speed proxy, not CFD.

### V6 physical entry form

Each foot hinge now combines ankle and midfoot pitch in one physical degree of freedom. Its range is −0.6…1.4 rad (about 80° plantarflexion), with learned targets bounded at 1.35 rad and the same 100 N·m motor cap. This is a deliberately simplified combined-foot envelope, not an isolated ankle joint or a detailed human foot. Radiographic dancer measurements report about 74° average plantarflexion and show that joints beyond the talocrural joint contribute to the movement ([Cho et al.](https://pubmed.ncbi.nlm.nih.gov/29768088/)). These measurements motivate the aggregate range; they do not validate the whole diver as a biomechanical model.

First-contact form measures each knee→ankle direction against its actual ankle→toe direction. Both errors must be under 20°. Each hand's physical long axis must be within 30° of downward entry, hand tips must be less than 12 cm apart, and their vertical mismatch must be under 4 cm. These are project thresholds for this rigid rig, not World Aquatics tolerances. A contact-free entry pose is reachable with the existing arm geometry; it does not require intersecting the head or artificially snapping the hands together. Native and WASM regressions also drive the actual foot motors from 1.10 to 1.35 rad over 100 ms, respecting target slew and torque limits, without any pose writes during the interval.

Continuous form deductions use squared physical foot-line errors, hand-axis error beyond 15°, tip separation beyond 7 cm, and vertical tip mismatch, in addition to bilateral hip/knee/arm form. Neutral feet produce about 82° line error and cannot earn full form credit. Flight reward also penalizes physical foot-line error. Central declared-position coverage still uses the hip/knee/grip metric described above; it does not implement every official body-position rule, including a strict pointed-foot coverage requirement throughout the whole flight. The shared stick figure follows these physical endpoints without visual toe correction.

### V7 leg closure and takeoff

The previous sagittal-only hip hinges could not bring knee or ankle centerlines closer than 21 cm. V7 adds an adduction hinge to each physical thigh, driven by one symmetric learned action. The new joint torque cap is 80 N·m per hip; the commanded range is −0.15…0.18 rad with 4 rad/s target slew, inside physical limits of −0.2…0.25 rad. Hip attachment points, body masses, original eight action meanings and all original torque caps remain unchanged. The figure follows actual physical endpoints; no visual leg correction is applied.

A motor-driven test reaches knee/ankle/toe gaps of approximately 17.7/14.4/13.0 cm from neutral adduction in 200 ms without any contacts. Natural foot contact limits further closure; the model does not require moving hip origins or making the legs perfectly parallel. Entry gates are knee gap <19 cm, ankle gap <16 cm and toe gap <15 cm. Signed ordering in the pelvis frame rejects crossed legs, even if their absolute gap is small. These centerline limits are approximations for the actual collider sizes, not official judging tolerances. Continuous physical separation loss also encourages close legs during flight.

Takeoff quality measures whole-diver center-of-mass ascent after final departure. Its reference and apex reset on every board recontact; pose changes or an earlier preparation hop cannot create rise credit. The height deduction is `1.5 × (1 − clamp(ascent / 0.30 m, 0, 1))`, so it contributes no penalty after 30 cm of actual rise. This project-specific scale implements the preference for a high takeoff without inventing an official minimum height. Training adds bounded ascent credit only when the declared dive and entry succeed. It uses the same limited-force motors and springboard; no root impulse or trajectory is imposed.

The first 67 observation positions are preserved. Five measurements are appended: both adduction angles, both angular velocities, and the common adduction target. The explicit v6→v7 migration copies all old network computations and starts new actor/critic input columns at exactly zero. The independent ninth output head is initialized near neutral adduction, with a fresh optimizer. The new physical freedom still changes the dynamics, so earlier results do not qualify v7.

## Training and evaluation

Python 3.12 with the exact versions in `training/requirements.txt`:

```sh
python3.12 -m venv .venv
.venv/bin/pip install -r training/requirements.txt
.venv/bin/python -m unittest discover -s training/tests -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python training/train.py \
  --skills 0 1 --min-height 5 --max-height 10 --steps 20480000 --name forward-curriculum
```

`training/train.py` implements PPO using reward from physical rollouts, with no expert trajectories or action labels. It supports goal/height curricula and checkpoint continuation. The current phase uses a tanh-squashed Gaussian, with the change-of-variables
log probability and the entropy of bounded actions. Its latent mean is
`2 × tanh(networkLogit / 2)`; its deterministic browser action is the tanh of
that latent mean. PPO stores latent samples so likelihoods never require an
unstable inverse tanh. Goal-specific exploration scales are training-only.
Earlier archived phases used Gaussian samples clipped by the environment;
those runs are not retrospectively described as squashed policies. Checkpoints and logs record the actual number of policy decisions. Development has included reward, contact-validation and curriculum revisions. The preserved audit in `training/diagnostics/invalid-launch/` documents failed earlier policies that used board assistance; those results are not qualification evidence. Each new phase snapshots its exact simulator, MJCF and trainer source hashes; a bit-identical reconstruction of exploratory runs is not claimed.

The final release will include per-skill held-out height/perturbation results, initial-network and heuristic comparisons, out-of-training-height stress tests, and a paired feedback ablation that replays the undisturbed actor commands under a new physical push. Frozen evaluations emulate browser arithmetic (float64 dot products with float32 layer outputs) and preserve original start parameters separately from terminal positions. Tiny native/WASM rounding differences can be amplified by unstable policies, so final evaluation includes actual runtime rollout comparisons as well as prerecorded-action physics parity. Failed dives remain in all reported means. The current development snapshot is not release-qualified for all listed skills.

`node scripts/audit-policy.mjs path/to/frozen/evaluation --all` evaluates the same frozen inputs and weights with JavaScript inference and MuJoCo WASM. It records per-skill execution distributions separately from the height grid and reports every native/runtime validity difference. The default invocation uses a smaller deterministic diagnostic subset. A passing prerecorded-action parity test alone does not establish identical closed-loop decisions.

The preserved v6 phase warm-started the original 154,603,520-decision v5 weights with a fresh optimizer. Expanding the ankle action target range changes what the same network output commands; the migration is recorded and all v6 decisions are counted separately as `stepsOnContract`. Previous v5 successes are not credited as v6 training or qualification. The genuine zero-experience comparison is freshly initialized with seed 1701 and zero optimizer steps. The v6 phase stopped after 9,646,080 new decisions (164,249,600 lineage decisions) for the requested adduction extension; its exact checkpoint and complete development evaluation remain in `training/diagnostics/v6-frozen-for-v7/`. The first v7 phase completed 20,480,000 decisions; the next entry/ascent refinement is preserved after 9,031,680 more. The corrected first-departure velocity phase is preserved after 12,718,080 additional decisions. The preserved `ppo-contact-exploration-v7` phase branches from the strongest earlier fixed-case entry checkpoint (191,303,680 lineage decisions) with a fresh optimizer. Its Gaussian exploration has separate loaded-foot-contact and flight variances. Only the hip, knee and ankle variance during observed foot load is increased; all deterministic network weights and narrow flight variances are preserved at migration. The conditional distribution uses the same exact tanh Jacobian likelihood. This is a recorded training change; physical limits and judging gates are unchanged. Decisions on the abandoned later branch are not added to the new branch’s inherited counters. The superseded v5 final-seed evidence remains in `training/validation-v5-final-seed/`; the current version uses the still-untouched separately preregistered final seed in `training/evaluation-protocol.json`.

Every current checkpoint records a physical/control contract: XML SHA-256, judge/control/observation versions, exact action bounds, dimensions, source hashes and decisions on that contract. Training refuses cross-contract continuation without an explicit migration and optimizer reset. Export, native evaluation and the browser reject missing or mismatched contracts. Explicit cross-physics diagnostic evaluations are labeled as such and cannot qualify as release evidence. A manifest proves both matching runtime semantics and exact asset bytes; it cannot retroactively make an old checkpoint trained on a new model.

`training/export_assets.py` writes model/XML hashes, byte counts and immutable browser paths. `training/generate_fixtures.py` generates native-engine golden trajectories. Numerical tests cover native/WASM state and observation parity, quaternion rotation cases, first-contact scoring, unsupported motions, force bounds, momentum error, water-force direction and practice scheduling. Browser tests exercise actual moving body poses, inference, controller comparison, disturbances, exports, cameras, visibility, mobile layout and disposal.

## Limits and references

This is an articulated rigid-body control experiment, not a full biomechanical athlete. It omits muscles, tendons, tissue deformation, finger control and a flexible bending board. The water model approximates forces rather than solving fluid flow. The competition-inspired scores and height rules describe this practice simulator.

- [MuJoCo computation and articulated dynamics](https://mujoco.readthedocs.io/en/stable/computation/index.html).
- [MuJoCo contact force sensors](https://mujoco.readthedocs.io/en/stable/XMLreference.html#sensor-contact).
- [MuJoCo API: body velocities, state and stepping](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html).
- [Schulman et al., Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347). Algorithm reference; this project’s training code and weights are original.
- [World Aquatics Competition Regulations, February 2026](https://resources.fina.org/fina/document/2026/02/18/e6815ecc-06d9-4f0b-98e9-4c441cf5e6a3/2026-02-18_World-Aquatics_CR-Final.pdf), Part Four and Appendix 9. Reference difficulty and practice no-repeat rule, not a claim of full event compliance.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for runtime licenses and the reproducible CSP-compatible MuJoCo glue. No JavaScript `unsafe-eval` is required.

Release assets require `training/export_assets.py --status release --qualification <frozen-evidence-directory>`, in addition to the explicit policy and initial paths. The exporter verifies the same preregistered native/WASM quality and provenance checks before changing the public manifest. Development snapshots keep their visible development status.

The V8 whole-entry controller is still under development. The first matched-head 2.048M continuation failed all 384 clean-entry tests and worsened mean full-window angle from 35.73° to 41.82°. Its evidence remains preserved. The completed `normalized-whole-entry-once-v3` phase reduced the fixed-case worst-angle mean to 19.59°, while all 384 cases still failed the complete clean-entry standard. See [training/CHECKPOINTS.md](training/CHECKPOINTS.md) for immutable checkpoints, exact versus restarted continuation, evaluated-best selection and extending a completed target.

## Runtime regression checks

`npm test` checks the committed native reference trajectories and current policy schema. `npm run test:runtime` runs focused browser loading/playback/export checks for trained and initial models. These are correctness checks, not proof of model quality; older browser qualification scenarios are separate.

Regenerate native references only after investigating a contract change, using a Python environment with the pinned training dependencies: `python training/generate_fixtures.py`. References carry source/model hashes so drift is reported explicitly. Existing numerical parity tolerances remain unchanged.
