/** Exercise the real WASM allocation lifetime, including replay and engine disposal. */
import { readFile } from 'node:fs/promises';
import assert from 'node:assert/strict';
import load from '@mujoco/mujoco';
import { PhysicsEngine, simulate } from '../src/core/physics.js';
import { validatePolicy } from '../src/core/control.js';
const manifest = JSON.parse(await readFile('public/asset-manifest.json'));
const policyFile = manifest.files.find((file) => file.key === 'policy');
const policy = validatePolicy(JSON.parse(await readFile('public/' + policyFile.path)));
const xml = await readFile('public/physics/diver.xml', 'utf8');
const mj = await load();
const engine = new PhysicsEngine(mj, xml);
const parameters = { skill: 1, height: 7.5, tilt: .16, preload: -.18, x: -.13, disturbance: 0 };
let reference;
for (let i = 0; i < 8; i++) reference = simulate(engine, parameters, 'pretrained', policy).result;
const warmBytes = engine.data.qpos.buffer.byteLength;
let maximumBytes = warmBytes;
for (let i = 0; i < 64; i++) {
  const result = simulate(engine, parameters, 'pretrained', policy).result;
  assert.equal(result.execution, reference.execution);
  assert.equal(result.angle, reference.angle);
  maximumBytes = Math.max(maximumBytes, engine.data.qpos.buffer.byteLength);
}
assert.equal(maximumBytes, warmBytes, 'WASM memory grew during repeated replay');
engine.dispose();
let recreationWarmBytes;
for (let i = 0; i < 128; i++) {
  const disposable = new PhysicsEngine(mj, xml);
  simulate(disposable, parameters, 'pretrained', policy);
  // Emscripten retains its allocator high-water mark. A first XML recompilation
  // may enlarge the heap once; subsequent model/data lifetimes must plateau.
  if (i === 0) recreationWarmBytes = disposable.data.qpos.buffer.byteLength;
  assert.equal(disposable.data.qpos.buffer.byteLength, recreationWarmBytes, 'WASM memory keeps growing across engine recreation');
  maximumBytes = Math.max(maximumBytes, disposable.data.qpos.buffer.byteLength);
  disposable.dispose();
}
console.log(JSON.stringify({ policySHA256: policyFile.sha256, replays: 64, engineRecreations: 128, warmBytes, recreationWarmBytes, maximumBytes, execution: reference.execution, ownership: 'One MjModel, MjData and reusable DoubleBuffer per engine; no contact/body/geom Embind handle accessors.' }, null, 2));
