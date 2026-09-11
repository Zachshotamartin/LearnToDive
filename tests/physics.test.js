import test, { before } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import {createHash} from "node:crypto";
import load from "@mujoco/mujoco";
import { PhysicsEngine, scoreEntry, simulate } from "../src/core/physics.js";
import {
  SKILLS,
  framedAngles,
  wrap,
  targetsFromAction,
  geomExtent,
  firstWaterGeometry,
  parameters,
  validatePolicy,
  policyAction,
  postureProgress,
  entryPosture,
  entryGeometry,
  legGeometry,
  LOW, HIGH, RATE, MAP, CONTROL_SEMANTICS, JUDGE_VERSION,
  baselineAction,
  contactForces,
  tuckGeometry,
} from "../src/core/control.js";
import { DiveRound } from "../src/core/round.js";
const xml = fs.readFileSync(
  new URL("../public/physics/diver.xml", import.meta.url),
  "utf8",
);
const currentContract={schemaVersion:1,waterVersion:"directional-submerged-body-v1",physics:"MuJoCo 3.13.0",xmlSHA256:createHash("sha256").update(xml).digest("hex"),controlSemantics:CONTROL_SEMANTICS,judgeVersion:JUDGE_VERSION,observationSemantics:"upright-adduction-full-entry-v4",observationSize:76,actionSize:9,actionLow:LOW,actionHigh:HIGH};
let mj;
before(async () => {
  mj = await load();
});
const near = (a, b, tol = 1e-8) =>
  assert.ok(
    Math.abs(a - b) <= tol,
    `${a} differs from ${b} (tolerance ${tol})`,
  );
const product = (a, b) => {
  const [w, x, y, z] = a,
    [v, i, j, k] = b;
  return [
    w * v - x * i - y * j - z * k,
    w * i + x * v + y * k - z * j,
    w * j - x * k + y * v + z * i,
    w * k + x * j - y * i + z * v,
  ];
};
const yrot = (a) => [Math.cos(a / 2), 0, Math.sin(a / 2), 0],
  zrot = (a) => [Math.cos(a / 2), 0, 0, Math.sin(a / 2)];
function withEngine(fn) {
  const e = new PhysicsEngine(mj, xml).reset();
  try {
    return fn(e);
  } finally {
    e.dispose();
  }
}

test("articulated model has70kg of independently constrained bodies and bounded actuators", () =>
  withEngine((e) => {
    const m = e.model;
    assert.equal(m.nq, 22);
    assert.equal(m.nv, 21);
    assert.equal(m.nu, 14);
    near(
      Array.from(m.body_mass)
        .slice(2)
        .reduce((a, b) => a + b, 0),
      70,
    );
    assert.deepEqual(
      Array.from(m.actuator_forcerange).filter((_, i) => i % 2 === 1),
      [160, 180, 100, 160, 180, 100, 65, 45, 35, 65, 45, 35, 80, 80],
    );
    assert.equal(m.opt.timestep, 0.002);
    assert.equal(Object.keys(e.frame().pose).length, 20);
  }));

test("native3.13 and WASM3.13 match full trajectories, physical pushes and observations", () => {
  const fixtures = JSON.parse(
    fs.readFileSync(new URL("./fixtures/mujoco-parity.json", import.meta.url)),
  );
  assert.equal(fixtures.contract.xmlSHA256, currentContract.xmlSHA256, 'Native fixture geometry is stale; regenerate with the pinned native generator');
  assert.equal(fixtures.contract.observationSize, currentContract.observationSize);
  assert.equal(fixtures.contract.controlSemantics, CONTROL_SEMANTICS);
  assert.equal(fixtures.contract.judgeVersion, JUDGE_VERSION);
  for (const [name, hash] of Object.entries(fixtures.sources))
    assert.equal(createHash('sha256').update(fs.readFileSync(new URL('../training/' + name, import.meta.url))).digest('hex'), hash, `Native fixture source changed: ${name}`);
  for (const c of fixtures.cases)
    withEngine((e) => {
      e.reset(c.parameters);
      for (const step of c.steps) {
        e.step(step.action);
        e.state().forEach((v, i) => near(v, step.state[i], 1e-7));
        Array.from(e.observation()).forEach((v, i) =>
          near(v, step.observation[i], 2e-6),
        );
        near(e.theta, step.theta, 1e-7);
        near(e.twist, step.twist, 1e-7);
        near(e.shapeAngle, step.shapeAngle, 1e-7);
        baselineAction(e).forEach((v, i) =>
          near(v, step.baselineAction[i], 1e-7),
        );
        if (step.score) {
          const r = scoreEntry(e);
          assert.equal(r.valid, step.score.valid);
          near(r.execution, step.score.execution);
          near(r.splash, step.score.splash, 1e-7);
        }
      }
    });
});

test("framed quaternion counting preserves forward/back somersaults and axial twists", () => {
  for (const back of [0, 1])
    for (const turns of [0.5, 1.5, 2.5])
      for (const twists of [0, 0.5, 1]) {
        let sumA = 0,
          sumB = 0,
          prev = framedAngles(product(yrot(0), zrot(back * Math.PI)));
        for (let i = 1; i <= 1000; i++) {
          const q = product(
              yrot((turns * 2 * Math.PI * i) / 1000),
              zrot(back * Math.PI + (twists * 2 * Math.PI * i) / 1000),
            ),
            a = framedAngles(q);
          sumA += wrap(a[0] - prev[0]);
          sumB += wrap(a[1] - prev[1]);
          near(a[2], 0, 1e-12);
          prev = a;
        }
        near(sumA, turns * 2 * Math.PI, 1e-10);
        near(sumB, twists * 2 * Math.PI, 1e-10);
      }
});

test("lateral cartwheel singularities are identified and cannot earn a declared forward code", () => {
  let max = 0;
  for (let i = 0; i <= 100; i++) {
    const a = (i / 100) * 2 * Math.PI;
    max = Math.max(
      max,
      framedAngles([Math.cos(a / 2), Math.sin(a / 2), 0, 0])[2],
    );
  }
  near(max, 1);
  const e = idealEntry(1);
  e.maxLateral = max;
  const result = scoreEntry(e);
  assert.equal(result.numericCorrect, false);
  assert.equal(result.execution, 0);
});

test("water-plane extents use exact oriented sphere, capsule, box and ellipsoid support", () => {
  near(geomExtent(2, [0.2, 0, 0], [1, 0, 0, 0]), 0.2);
  near(geomExtent(3, [0.1, 0.4, 0], [1, 0, 0, 0]), 0.5);
  near(geomExtent(3, [0.1, 0.4, 0], yrot(Math.PI / 2)), 0.1);
  near(geomExtent(6, [0.3, 0.2, 0.1], yrot(Math.PI / 2)), 0.3);
  near(geomExtent(4, [0.3, 0.2, 0.1], yrot(Math.PI / 2)), 0.3);
  const s = new Array(59).fill(0);
  s.splice(45, 7, 0, 0, -4.91, 1, 0, 0, 0);
  s.splice(52, 7, 0, 0, -4.5, 1, 0, 0, 0);
  const hit = firstWaterGeometry(
    s,
    [
      { type: 2, size: [0.1, 0, 0] },
      { type: 2, size: [0.2, 0, 0] },
    ],
    5,
  );
  assert.equal(hit.contact, true);
  assert.equal(hit.index, 0);
});

test("first contact is recorded at2ms and policy control continues until full submersion", () => {
 const fixtures=JSON.parse(fs.readFileSync(new URL('./fixtures/mujoco-parity.json',import.meta.url)));
 let betweenTicks=false,controlledWater=false;
 for(const c of fixtures.cases)withEngine(e=>{
  e.reset(c.parameters);
  for(const step of c.steps){const wasWater=e.entryTime!==null,previous=e.time;e.step(step.action);
   if(step.entryTime!==null){near(e.entryTime,step.entryTime,1e-8);betweenTicks||=Math.abs(e.entryTime/.02-Math.round(e.entryTime/.02))>.01;}
   if(wasWater&&e.time>previous)controlledWater=true;
  }
  assert.ok(e.actions.some(a=>a.time>=e.entryTime));
 });
 assert.ok(betweenTicks);assert.ok(controlledWater);
});

test("bounded target slew never changes the free root or requests unlimited joint movement", () => {
  const previous = new Array(9).fill(0),
    next = targetsFromAction(new Array(9).fill(100), previous);
  next.forEach((v, i) =>
    assert.ok(Math.abs(v) <= RATE[i] * 0.02 + 1e-12),
  );
  const p = parameters({ height: Infinity, disturbance: 999, tilt: -999 });
  assert.equal(p.height, 7.5);
  assert.equal(p.disturbance, 120);
  assert.equal(p.tilt, 0.04);
});

test("internal motors preserve airborne momentum within measured finite-step error at finite step size", () =>
  withEngine((e) => {
    const d = e.data,
      m = e.model;
    d.qpos[1] = 8;
    d.qpos[2] = 0;
    d.qpos[3] = 25;
    d.qvel.fill(0);
    d.qvel[5] = 4;
    mj.mj_forward(m, d);
    e.read();
    const L0 = Array.from(d.subtree_angmom).slice(6, 9),
      v0 = Array.from(d.subtree_linvel).slice(6, 9);
    let drift = 0;
    for (let i = 0; i < 50; i++) {
      e.step(Array.from({ length: 9 }, (_, j) => 0.5 * Math.sin(i * 0.17 + j)));
      const L = Array.from(d.subtree_angmom).slice(6, 9);
      drift = Math.max(drift, Math.hypot(...L.map((v, j) => v - L0[j])));
    }
    const v = Array.from(d.subtree_linvel).slice(6, 9);
    near(v[0], v0[0], 0.035);
    near(v[1], v0[1], 0.035);
    near(v[2], v0[2] - 9.81, 0.035);
    assert.ok(
      drift < 0.03 * Math.hypot(...L0) + 0.05,
      `finite-step angular drift ${drift}`,
    );
    assert.ok(e.maxTorque <= 180 + 1e-8);
  }));

test("initial board contact and rendering endpoints refer to the same engine bodies", () =>
  withEngine((e) => {
    const f = e.frame();
    near(f.board.position[0], -1.275);
    near(f.board.position[1], e.parameters.height + e.parameters.preload);
    f.board.quaternion.forEach((v, i) => near(v, [0, 0, 0, 1][i]));
    assert.deepEqual(f.pose.pelvis, f.bodies.pelvis.position);
    for (const point of Object.values(f.pose))
      assert.ok(point.every(Number.isFinite));
    assert.ok(
      f.pose.toeL[1] > e.parameters.height + e.parameters.preload - 0.05,
    );
  }));

test("water drag opposes actual body velocity and stays bounded", () =>
  withEngine((e) => {
    const d = e.data,
      m = e.model;
    d.qpos[1] = 3;
    d.qpos[3] = -e.parameters.height - 0.8;
    d.qvel[1] = 2;
    d.qvel[2] = 1;
    d.qvel[3] = -5;
    mj.mj_forward(m, d);
    e.applyWater();
    assert.ok(Array.from(d.xfrc_applied).every(Number.isFinite));
    for (let b = 2; b < m.nbody; b++) {
      assert.ok(Math.abs(d.xfrc_applied[b * 6]) <= 3500);
      assert.ok(d.xfrc_applied[b * 6] <= 0);
      assert.ok(d.xfrc_applied[b * 6 + 1] <= 0);
      assert.ok(d.xfrc_applied[b * 6 + 2] >= 0);
    }
  }));

function idealEntry(skill = 0) {
  const g = SKILLS[skill],
    q = new Array(22).fill(0);
  q[4] = 0;
  q[6] = 1;
  const qadr = [14, 16, 17, 18, 20, 21, 8, 9, 10, 11, 12, 13, 15, 19];
  q[8] = q[11] = 3.05;
  q[9] = -0.3;
  q[12] = 0.3;
  const s = new Array(150).fill(0);
  s[0] = 2;
  s[5] = -10;
  // Ideal geometric fixture, separate from the real-engine reachability test.
  for (const [knee, shin, toe, x] of [[37,115,16,.04],[40,136,19,-.04]]) {
    s.splice(knee, 3, 0, x, 0); s.splice(shin, 3, 0, x, -.21); s.splice(toe, 3, 0, x, -.59);
  }
  s[83] = s[104] = 1; s[9]=1;
  return {
    parameters: { skill },
    q,
    qadr,
    s,
    theta: g.turns * 2 * Math.PI,
    airTheta: g.turns * 2 * Math.PI,
    phaseTheta: g.turns * 2 * Math.PI,
    departurePitch: 0,
    departureCOM: 0.8, apexCOM:1.2, takeoffVerticalSpeed:3,
    airTwist: g.twists * 2 * Math.PI,
    boardInvalid: false,
    twist: g.twists * 2 * Math.PI,
    maxLateral: 0,
    shapeDuration: 0.3,
    shapeAngle: 0.55 * g.turns * 2 * Math.PI,
    shapePeak: 1,
    boardTwist: 0,
    releaseTwist: 0,
    entryTime: 1.5,
    fullEntryTime:1.7,entryMaxAngle:0,entryMinForm:1,entryLimbsValid:true,entryGeometryValid:true,entrySamples:100,
    firstGeometry: 6,
    maxJointSpeed: 8,
    maxTorque: 160,
    maxPower: 800,
  };
}

test("execution is independent of difficulty and a failed hard dive loses to a clean simple dive", () => {
  const clean = scoreEntry(idealEntry(0)),
    hard = idealEntry(5);
  hard.phaseTheta = 0.5 * Math.PI;
  const failed = scoreEntry(hard);
  assert.equal(clean.execution, 10);
  assert.equal(clean.total, 42);
  assert.equal(failed.execution, 0);
  assert.equal(failed.total, 0);
  assert.equal(failed.valid, false);
});

test("missed declared body position caps execution at2 and extra spins cannot farm points", () => {
  const e = idealEntry(1);
  e.shapeDuration = 0;
  e.shapeAngle = 0;
  const wrongShape = scoreEntry(e);
  assert.equal(wrongShape.execution, 2);
  assert.equal(wrongShape.valid, false);
  e.shapeDuration = 0.3;
  e.shapeAngle = 0.55 * SKILLS[1].turns * 2 * Math.PI;
  e.phaseTheta += 2 * Math.PI;
  assert.equal(scoreEntry(e).total, 0);
});

test("rounds reject repeat numeric codes, including a changed position letter", () => {
  const round = new DiveRound(10, 17);
  assert.equal(round.choose(1), 1);
  round.record({ execution: 9, total: 40.5, valid: true });
  assert.notEqual(round.choose(2), 2);
  while (round.remaining().length) {
    round.choose();
    round.record({ execution: 8, total: 30, valid: true });
  }
  assert.equal(
    new Set(round.attempts.map((r) => r.code)).size,
    round.attempts.length,
  );
  assert.equal(round.choose(), null);
  round.nextRound();
  assert.equal(round.attempts.length, 0);
  assert.equal(round.number, 2);
});

test("height eligibility removes difficult skills on low boards", () => {
  const round = new DiveRound(3, 42);
  assert.deepEqual(
    round.eligible().map((x) => x.code),
    [101, 201],
  );
  for (let i = 0; i < 2; i++) {
    round.choose();
    round.record({ execution: 8, total: 30, valid: true });
  }
  assert.equal(round.remaining().length, 0);
});

test("local model validates finite weights and produces state-dependent joint commands", () => {
  const model = validatePolicy(
    JSON.parse(
      fs.readFileSync(
        new URL(
          "../public/" +
            JSON.parse(
              fs.readFileSync(
                new URL("../public/asset-manifest.json", import.meta.url),
              ),
            ).files.find((x) => x.key === "policy").path,
          import.meta.url,
        ),
      ),
    ),
  );
  withEngine((e) => {
    const a = policyAction(model, e.observation());
    const obs = e.observation();
    obs[0] = -0.2;
    obs[2] = 0.9;
    obs[6] = 1;
    obs[53] += 0.2;
    const b = policyAction(model, obs);
    assert.ok(a.every(Number.isFinite));
    assert.ok(a.some((v, i) => Math.abs(v - b[i]) > 0.01));
  });
  assert.throws(() => validatePolicy({ ...model, layers: [] }));
});

test("central posture coverage rejects brief poses and cannot be farmed by oscillation", () => {
  const g = SKILLS[1],
    target = g.turns * 2 * Math.PI,
    j = [1.4, 2, 0.1, 1.4, 2, 0.1, 0.4, 0, 0.2, 0.4, 0, 0.2];
  let maximum = 0,
    angle = 0;
  for (let i = 1; i <= 100; i++) {
    const result = postureProgress(maximum, (target * i) / 100, g, j, true, {
      distances: [0.1, 0.1],
      legSeparation: 0.21,
    });
    maximum = result.maximum;
    angle += result.angle;
  }
  near(angle, 0.55 * target);
  for (let i = 0; i < 100; i++) {
    const result = postureProgress(
      maximum,
      target * (0.2 + 0.5 * (i % 2)),
      g,
      j,
      true,
      { distances: [0.1, 0.1], legSeparation: 0.21 },
    );
    maximum = result.maximum;
    angle += result.angle;
  }
  near(angle, 0.55 * target);
  const e = idealEntry(1);
  e.shapeDuration = 2;
  e.shapeAngle = 0.1 * target;
  assert.equal(scoreEntry(e).shapeCorrect, false);
});
test("entry form penalizes each bent leg and each splayed or bent arm independently", () => {
  for (const [index, value] of [
    [0, 0.7],
    [3, 0.7],
    [1, 0.8],
    [4, 0.8],
    [6, 2],
    [9, 2],
    [7, 0.9],
    [10, -0.9],
    [8, 0.8],
    [11, 0.8],
  ]) {
    const e = idealEntry(1);
    e.q[e.qadr[index]] = value;
    const r = scoreEntry(e);
    assert.equal(r.validEntry, false, `joint ${index}`);
    assert.ok(r.execution <= 2);
  }
  const j = [0, 0, 0, 0, 0, 0, 3.05, -0.3, 0, 3.05, 0.3, 0];
  near(entryPosture(j, entryGeometry(idealEntry().s)).form, 1);
  assert.equal(entryPosture(j).form, 0);
});
test("a stale rollout cannot be recorded as a newly selected numeric dive", () => {
  const round = new DiveRound(10, 1);
  round.choose(0);
  round.choose(3);
  round.record({ execution: 9, total: 37.8, valid: true }, 0);
  assert.equal(round.attempts.length, 0);
  round.record({ execution: 8, total: 40.8, valid: true }, 3);
  assert.equal(round.attempts[0].code, 201);
});

test("clean entry requires actual nearby hands to contact water before the head", () => {
  const e = idealEntry(1);
  e.s[22] = -0.7;
  e.s[25] = 0.7;
  assert.equal(scoreEntry(e).validEntry, false);
  e.s[22] = 0;
  e.s[25] = 0;
  e.firstGeometry = 3;
  assert.equal(scoreEntry(e).validEntry, false);
});

test("a twist performed on the board cannot satisfy an airborne twisting dive", () => {
  const e = idealEntry(5);
  e.releaseTwist = 2 * Math.PI;
  e.airTwist = 0;
  e.boardTwist = 2 * Math.PI;
  assert.equal(scoreEntry(e).numericCorrect, false);
  assert.equal(scoreEntry(e).execution, 0);
});

test("open-loop ablation holds the final recorded command and never resumes feedback", () =>
  withEngine((e) => {
    const action = new Array(9).fill(0);
    const dive = simulate(e, { height: 3, skill: 0 }, "pretrained", null, {
      waterSeconds: 0,
      openLoop: [{ values: action }],
    });
    assert.ok(dive.actions.length > 1);
    assert.ok(dive.actions.every((a) => a.values.every((x) => x === 0)));
  }));

test("real board top, side, underside and stand contacts are measured by MuJoCo force sensors", () => {
  for (const [name, position, support] of [
    ["top", [-1, 0, 0.06], false],
    ["front side", [0.1, 0, -0.1], false],
    ["underside", [-1, 0, -0.21], false],
    ["spring housing", [-2.14, 0, -0.6], true],
    ["support column", [-2.46, 0.64, -1.3], true],
  ])
    withEngine((e) => {
      e.data.qpos[0] = 0;
      e.data.qpos.set(position, 1);
      e.data.qpos.set([1, 0, 0, 0], 4);
      e.data.qvel.fill(0);
      mj.mj_forward(e.model, e.data);
      e.read();
      const forces = contactForces(e.s, support ? 210 : 150);
      assert.ok(
        Math.max(...forces) > 15,
        name + " must produce actual constraint force",
      );
      e.trackLaunch(0.002, 0);
      assert.equal(e.boardInvalid, true, name + " must invalidate launch");
      if (support) assert.ok(e.standPeak > 15);
      else assert.ok(e.boardNonfootPeak > 15);
    });
});

test("ordinary foot support is valid and self-touch cannot masquerade as board contact", () =>
  withEngine((e) => {
    assert.ok(contactForces(e.s)[11] > 15);
    e.trackLaunch(0.002, 0);
    assert.equal(e.boardInvalid, false);
    e.s.fill(0);
    e.s[43] = e.s[44] = 900;
    e.time = 0.1;
    e.theta = 0.2;
    e.trackLaunch(0.04, 0);
    assert.equal(e.released, true);
    near(e.airTheta, 0);
  }));

test("upright preparation bounces reset flight credit at the final departure; rotated recontact fails", () =>
  withEngine((e) => {
    e.s.fill(0);
    e.time = 0.1;
    e.theta = 0.1;
    e.trackLaunch(0.04, 0);
    e.time = 0.2;
    e.theta = 0.4;
    e.trackLaunch(0.002, 0);
    near(e.airTheta, 0.3);
    e.s[206] = 1;
    e.s[194] = 1;
    e.s[197] = 700;
    e.s[124] = e.q[0] + 0.1;
    e.time = 0.3;
    e.trackLaunch(0.002, 0);
    assert.equal(e.released, false);
    near(e.airTheta, 0);
    assert.equal(e.preparationBounces, 1);
    assert.equal(e.boardInvalid, false);
    e.s.fill(0);
    e.time = 0.4;
    e.theta = 0.5;
    e.trackLaunch(0.04, 0);
    near(e.releaseTime, 0.362);
    e.theta = 0.8;
    e.time = 0.5;
    e.trackLaunch(0.002, 0);
    near(e.airTheta, 0.3);
    e.q.splice(4, 4, ...yrot(Math.PI / 2));
    e.s[206] = 1;
    e.s[194] = 1;
    e.s[197] = 700;
    e.s[124] = e.q[0] + 0.1;
    e.trackLaunch(0.002, 0);
    assert.equal(e.rotatedRecontact, true);
    assert.equal(e.boardInvalid, true);
  }));

test("an otherwise perfect entry with a collision-assisted launch receives zero points", () => {
  const e = idealEntry(0);
  e.boardInvalid = true;
  const r = scoreEntry(e);
  assert.equal(r.execution, 0);
  assert.equal(r.valid, false);
  assert.equal(r.launchValid, false);
  assert.equal(r.total, 0);
});

test("classification includes bounded takeoff lean while twist remains strictly airborne", () => {
  const e = idealEntry(0);
  e.departurePitch = (40 * Math.PI) / 180;
  e.airTheta = Math.PI - e.departurePitch;
  e.phaseTheta = e.airTheta + e.departurePitch;
  assert.equal(scoreEntry(e).valid, true);
  near(scoreEntry(e).execution, 10);
  e.phaseTheta += 2 * Math.PI;
  assert.equal(scoreEntry(e).valid, false);
});
test("a takeoff beyond the explicit tilt bound or from staggered feet is invalid", () => {
  for (const stagger of [false, true])
    withEngine((e) => {
      e.s.fill(0);
      e.time = 0.4;
      e.q.splice(4, 4, ...yrot(stagger ? 0.2 : 1.2));
      e.lastFootContact = stagger ? [0.3, 0.2] : [0.3, 0.3];
      e.trackLaunch(0.04, 0);
      assert.equal(e.boardInvalid, true);
      if (stagger) near(e.footDepartureGap, 0.1);
      else assert.equal(e.takeoffTiltInvalid, true);
    });
});

test("accumulated preparation turns cannot enter the final somersault phase", () =>
  withEngine((e) => {
    e.s.fill(0);
    e.q.splice(4, 4, ...yrot(0.4));
    e.time = 0.5;
    e.theta = 2 * Math.PI + 0.4;
    e.trackLaunch(0.04, 0);
    near(e.phaseTheta, 0.4);
    e.theta += Math.PI - 0.4;
    e.time = 1.5;
    e.q.splice(4, 4, ...yrot(Math.PI));
    e.trackLaunch(0.002, 0);
    near(e.phaseTheta, Math.PI);
    near(e.airTheta, Math.PI - 0.4);
  }));
test("a standing preparation hop has a visible execution deduction", () => {
  const e = idealEntry(0);
  e.preparationBounces = 1;
  const r = scoreEntry(e);
  assert.equal(r.execution, 9);
  assert.equal(r.breakdown.preparation, 1);
});

test("tuck coverage requires both actual hands near the lower legs and rejects missing geometry", () => {
  const g = SKILLS[1],
    j = [1.4, 2, 0.1, 1.4, 2, 0.1, 0.4, 0, 0.2, 0.4, 0, 0.2];
  const target = g.turns * 2 * Math.PI;
  for (const distances of [
    [0.7, 0.1],
    [0.1, 0.7],
    [0.7, 0.7],
  ]) {
    const p = postureProgress(0.2 * target, 0.3 * target, g, j, true, {
      distances,
      legSeparation: 0.21,
    });
    near(p.angle, 0);
  }
  near(postureProgress(0.2 * target, 0.3 * target, g, j, true).angle, 0);
  assert.ok(
    postureProgress(0.2 * target, 0.3 * target, g, j, true, {
      distances: [0.12, 0.12],
      legSeparation: 0.21,
    }).angle > 0,
  );
});

test("a compact hand-near-shin tuck is reachable by the actual articulated joint limits", () =>
  withEngine((e) => {
    const joints = [1.3, 2, 0.2, 1.3, 2, 0.2, 0.6, -0.17, 0, 0.6, 0.17, 0];
    e.data.qpos[1] = 5;
    e.data.qpos[3] = 3;
    joints.forEach((v, i) => {
      e.data.qpos[e.qadr[i]] = v;
    });
    mj.mj_forward(e.model, e.data);
    e.read();
    const geometry = tuckGeometry(e.s);
    assert.ok(Math.max(...geometry.distances) < 0.03);
    near(geometry.legSeparation, 0.21, 1e-8);
    const pose = e.frame().pose;
    for (const side of ["L", "R"]) {
      const hand = pose["hand" + side],
        a = pose["knee" + side],
        b = pose["ankle" + side];
      const axis = b.map((v, i) => v - a[i]),
        offset = hand.map((v, i) => v - a[i]);
      const dot = (x, y) => x.reduce((v, c, i) => v + c * y[i], 0);
      const t = Math.max(0, Math.min(1, dot(offset, axis) / dot(axis, axis)));
      const distance = Math.hypot(...offset.map((v, i) => v - t * axis[i]));
      near(distance, geometry.distances[side === "L" ? 0 : 1], 1e-8);
    }
  }));

test("squashed actor export applies the exact bounded latent-mean transformation", () => {
  const layer = (o, n) => ({
    weight: Array.from({ length: o }, () => new Array(n).fill(0)),
    bias: new Array(o).fill(0),
  });
  const model = {
    contract: currentContract,
    format: "mujoco-joint-ppo-v1",
    observationSize: 76,
    actionSize: 9,
    outputActivation: "squashed-mean-v1",
    latentMeanLimit: 2,
    layers: [layer(96, 76), layer(96, 96), layer(9, 96)],
  };
  for (const [key,value] of [["schemaVersion",9],["observationSemantics","old"],["observationSize",66],["actionSize",7],["controlSemantics","v5"],["judgeVersion","v5"]])
    assert.throws(()=>validatePolicy({...model,contract:{...model.contract,[key]:value}}));
  assert.throws(()=>validatePolicy(model,"0".repeat(64)));
  assert.throws(()=>validatePolicy({...model,contract:undefined}));
  model.layers[2].bias = Array.from({ length: 9 }, (_, i) => i - 3.5);
  const a = policyAction(validatePolicy(model), new Float32Array(76));
  a.forEach((v, i) => {
    near(v, Math.tanh(2 * Math.tanh((i - 3.5) / 2)), 1e-7);
    assert.ok(Math.abs(v) < 1);
  });
  assert.throws(() => validatePolicy({ ...model, latentMeanLimit: 20 }));
  assert.throws(() =>
    validatePolicy({ ...model, outputActivation: "unknown" }),
  );
});


test("v6 rejects each physical foot and hand misalignment independently", () => {
  for (const change of [e => e.s[16] += .5, e => e.s[19] += .5,
    e => e.s[24] += .05, e => e.s[22] += .13,
    e => e.s.splice(83,4,...yrot(Math.PI/4)), e => e.s.splice(104,4,...yrot(Math.PI/4))]) {
    const e=idealEntry(1); change(e); const r=scoreEntry(e);
    assert.equal(r.geometryLimits,false); assert.equal(r.validEntry,false); assert.ok(r.execution<=2);
  }
});

test("v6 bounded motors physically reach pointed feet and close hands over 100 ms", () => withEngine(e => {
  const d=e.data,m=e.model;
  d.qpos[1]=2;d.qpos[2]=0;d.qpos[3]=4;
  d.qpos.set([0,0,1,0],4);
  const joints=[0,0,1.1,0,0,1.1,2.8,-.24,0,2.8,.24,0,.04,.04];
  for(let i=0;i<14;i++) d.qpos[e.qadr[i]]=joints[i];
  const targets=[0,0,1.1,2.8,2.8,-.24,.24,0,.04], desired=[0,0,1.35,2.8,2.8,-.24,.24,0,.04];
  const map=MAP,rate=RATE;
  mj.mj_forward(m,d);
  for(let tick=0;tick<5;tick++) {
    targets.forEach((v,i)=>targets[i]=v+Math.max(-rate[i]*.02,Math.min(rate[i]*.02,desired[i]-v)));
    d.ctrl.set(map.map(i=>targets[i]));
    for(let k=0;k<10;k++){mj.mj_step(m,d);assert.equal(d.ncon,0);assert.ok(Array.from(d.actuator_force).every(v=>Math.abs(v)<=180.00001));}
  }
  mj.mj_forward(m,d);
  const geometry=entryGeometry(Array.from(d.sensordata)), posture=entryPosture(e.qadr.map(i=>d.qpos[i]),geometry);
  assert.equal(posture.geometryLimits,true);assert.equal(posture.limbLimits,true);assert.ok(posture.form>.8);
  assert.ok(Math.max(...geometry.footLineAngles)<20*Math.PI/180);
  d.qpos[e.qadr[2]]=d.qpos[e.qadr[5]]=0;mj.mj_forward(m,d);
  const neutral=entryPosture(e.qadr.map(i=>d.qpos[i]),entryGeometry(Array.from(d.sensordata)));
  assert.equal(neutral.geometryLimits,false);assert.ok(neutral.form<.2);
}));


test("v7 physical adduction closes leg gaps with bounded motors and rejects crossing",()=>withEngine(e=>{
 const d=e.data,m=e.model;d.qpos.set([2,0,4],1);d.qpos.set([0,0,1,0],4);
 const joints=[0,0,1.35,0,0,1.35,2.8,-.24,0,2.8,.24,0,0,0];
 joints.forEach((v,i)=>{d.qpos[e.qadr[i]]=v;d.ctrl[i]=v});mj.mj_forward(m,d);
 near(legGeometry(Array.from(d.sensordata)).ankleGap,.21);
 d.ctrl[12]=d.ctrl[13]=.08;
 for(let k=0;k<100;k++){mj.mj_step(m,d);assert.equal(d.ncon,0);assert.ok(Math.abs(d.actuator_force[12])<=80);assert.ok(Math.abs(d.actuator_force[13])<=80)}
 mj.mj_forward(m,d);const s=Array.from(d.sensordata),g=legGeometry(s);
 assert.ok(g.kneeGap<.19 && g.ankleGap<.16 && g.toeGap<.15);assert.equal(g.crossedLegs,false);
 const left=s.slice(16,19);s.splice(16,3,...s.slice(19,22));s.splice(19,3,...left);
 assert.equal(legGeometry(s).crossedLegs,true);
 assert.equal(entryPosture(e.qadr.map(i=>d.qpos[i]),entryGeometry(s)).geometryLimits,false);
}));

test("takeoff height credit uses final-departure COM and cannot farm prep hops",()=>withEngine(e=>{
 e.s.fill(0);e.s[9]=1;e.q[4]=1;e.q[5]=e.q[6]=e.q[7]=0;e.time=.2;
 e.s[194]=1;e.s[2]=8;e.trackLaunch(.002,0);near(e.apexCOM,8);
 e.s[194]=0;e.s[2]=.8;e.trackLaunch(.002,0);near(e.departureCOM,.8);near(e.apexCOM,.8);
 e.s[2]=1.1;e.trackLaunch(.002,0);near(e.apexCOM,1.1);
 e.s[194]=1;e.s[2]=.9;e.trackLaunch(.002,0);near(e.apexCOM,.9);
 e.s[194]=0;e.s[2]=.7;e.trackLaunch(.002,0);near(e.departureCOM,.7);near(e.apexCOM,.7);
 const high=idealEntry(),low=idealEntry();low.apexCOM=low.departureCOM;
 assert.equal(scoreEntry(high).execution-scoreEntry(low).execution,1.5);
 low.boardInvalid=true;assert.equal(scoreEntry(low).execution,0);
}));

test('a good hand-contact snapshot cannot hide a flopped whole entry or timeout',()=>{
 const clean=idealEntry(1);assert.equal(scoreEntry(clean).valid,true);
 const flop=idealEntry(1);flop.entryMaxAngle=48;const bad=scoreEntry(flop);
 assert.equal(bad.firstContactAngle,0);assert.equal(bad.angle,48);assert.equal(bad.validEntry,false);assert.ok(bad.execution<=2);
 const timeout=idealEntry(1);timeout.fullEntryTime=null;timeout.time=2.3;assert.equal(scoreEntry(timeout).fullEntryComplete,false);assert.ok(scoreEntry(timeout).execution<=2);
 const form=idealEntry(1);form.entryGeometryValid=false;assert.equal(scoreEntry(form).validEntry,false);
});

test('slightly underrotated hand contact is compatible with a good complete entry',()=>{
 const e=idealEntry(1);e.q.splice(4,4,...yrot(Math.PI-.1));e.phaseTheta-=.1;e.entryMaxAngle=8;
 const r=scoreEntry(e);assert.equal(r.numericCorrect,true);assert.equal(r.validEntry,true);assert.ok(r.execution>=8);
});
