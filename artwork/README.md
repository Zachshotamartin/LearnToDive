# Shared stick-figure renderer

The visible figure is defined once in `@zachshotamartin/stick-figure`, the same
original package consumed by Hide and Seek. It draws eighteen thin straight
sausage segments and twenty round joint markers, including a small sphere head.
There is no sculpted mesh, GLB, face, skinning, clothing or animation clip. The
same dimensions and code serve both apps; the caller supplies scale, color and
joint landmarks.

MuJoCo supplies the diver's actual body landmarks. The renderer places each
segment between those endpoints, so physical knees, elbows, wrist and entry
positions remain authoritative. Water/spray effects illustrate physical contact;
they are not a fluid solver.

Run `node scripts/visuals/audit-figure.mjs` for actual browser pose, landmark,
mobile and shader checks. Its frozen `fixtures/development-rollout.json` is an
intermediate physics rollout for art review, not performance evidence. The current
fixture is a fixed 201C at 9 m from the 108,523,520-decision development actor:
launch clearance passed, declared dive completed, computed execution 6/10 and
entry angle 34.07° under the preceding judge. The newer tuck-hand proximity criterion was introduced afterward, so that execution score is historical and not a claim under the current judge. Its exact policy SHA is stored in the fixture. This one
rollout does not establish repertoire quality or held-out reliability. Final
learned demonstrations use qualified policy rollouts.

`node scripts/visuals/capture-comparison.mjs` creates a neutral same-pose,
different-color comparison of the shared model. This image is a geometry
illustration, not a simulation or a claimed learned behavior.

All character code is original project work, MIT © Zachary Martin, 2026.
