import loadMuJoCo from "./vendor/mujoco-csp.js";
import wasm from "@mujoco/mujoco/mujoco.wasm?url";
import {
  AutonomousEngine,
  simulateAutonomous,
} from "./core/autonomousPhysics.js";
import { createAutonomousPolicy } from "./core/autonomousPolicy.js";
import { DIVES } from './data/declarations.js';
let init, engine, policies;
async function read(url, json = false) {
  const r = await fetch(url);
  if (!r.ok) throw Error(`Simulator asset unavailable (${r.status})`);
  return json ? r.json() : r.text();
}
self.onmessage = async ({ data }) => {
  try {
    if (!init)
      init = (async () => {
        const [mj, xml, p, i] = await Promise.all([
          loadMuJoCo({ locateFile: (p) => (p.endsWith(".wasm") ? wasm : p) }),
          read(data.assets.xml),
          read(data.assets.policy, true),
          read(data.assets.initial, true),
        ]);
        const hash = Array.from(
          new Uint8Array(
            await crypto.subtle.digest(
              "SHA-256",
              new TextEncoder().encode(xml),
            ),
          ),
          (x) => x.toString(16).padStart(2, "0"),
        ).join("");
        for (const model of [p, i])
          if (model.contract.sourceHashes["diver.xml"] !== hash)
            throw Error("Physics and model versions do not match");
        engine = new AutonomousEngine(mj, xml);
        policies = {
          pretrained: { actor: createAutonomousPolicy(p), steps: p.steps },
          untrained: { actor: createAutonomousPolicy(i), steps: 0 },
        };
        // Reuse the worker's canonical rule table. Importing it into the page
        // would duplicate the full table in the application JavaScript bundle.
        self.postMessage({ catalog: DIVES.map(({ id, code, name, position, group, turns, twists, difficulty }) =>
          ({ id, code, name, position, group, turns, twists, conditions: Object.keys(difficulty) })) });
      })();
    await init;
    const selected = policies[data.policy];
    if (!selected) throw Error("Unknown model");
    const dive = simulateAutonomous(engine, data.parameters, selected.actor);
    dive.policy = data.policy;
    dive.physics.policySteps = selected.steps;
    self.postMessage({ id: data.id, dive });
  } catch (e) {
    self.postMessage({ id: data.id, error: e.message });
  }
};
