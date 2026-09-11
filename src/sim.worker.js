import loadMuJoCo from "./vendor/mujoco-csp.js";
import wasmURL from "@mujoco/mujoco/mujoco.wasm?url";
import { PhysicsEngine, simulate } from "./core/physics.js";
import { validatePolicy } from "./core/control.js";
let initialization,
  engine,
  policies,
  latest = 0;
async function read(url, maxBytes) {
  const response = await fetch(url);
  if (!response.ok)
    throw new Error(`Local simulator asset unavailable (${response.status}).`);
  const body = await response.text();
  if (body.length > maxBytes)
    throw new Error("Local simulator asset exceeds its size limit.");
  return body;
}
async function initialize(assets) {
  if (!initialization)
    initialization = (async () => {
      const [mj, xml, trained, initial] = await Promise.all([
        loadMuJoCo({
          locateFile: (path) => (path.endsWith(".wasm") ? wasmURL : path),
        }),
        read(assets.xml, 100000),
        read(assets.policy, 1000000),
        read(assets.initial, 1000000),
      ]);
      const xmlHash = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(xml))), b => b.toString(16).padStart(2, "0")).join("");
      policies = {
        pretrained: validatePolicy(JSON.parse(trained), xmlHash),
        untrained: validatePolicy(JSON.parse(initial), xmlHash),
      };
      engine = new PhysicsEngine(mj, xml);
    })();
  return initialization;
}
self.onmessage = async ({ data }) => {
  latest = data.id;
  try {
    await initialize(data.assets);
    if (data.id !== latest) return;
    const dive = simulate(
      engine,
      data.parameters,
      data.policy,
      policies[data.policy] || policies.pretrained,
    );
    dive.physics.actionDistribution = policies[data.policy]?.actionDistribution ?? (data.policy === "baseline" ? "hand-authored" : "legacy-clipped-gaussian");
    dive.physics.contract = policies.pretrained.contract;
    dive.physics.policyContract = policies[data.policy]?.contract ?? null;
    dive.physics.policySteps = policies[data.policy]?.steps ?? null;
    dive.physics.policyAsset =
      data.policy === "baseline"
        ? null
        : data.assets[data.policy === "untrained" ? "initial" : "policy"];
    if (data.id === latest) self.postMessage({ id: data.id, dive });
  } catch (error) {
    if (data.id === latest)
      self.postMessage({ id: data.id, error: error.message });
  }
};
