import { WATER_VERSION } from './water.js';
/** Original policy interface, state measurements and bounded joint commands. */
export const CONTROL_SEMANTICS = "joint-servo-full-entry-v8";
export const JUDGE_VERSION = "full-submersion-physical-entry-v8";
export const DT = 0.02,
  SUBSTEPS = 10,
  PHYSICS_DT = 0.002;
export const SKILLS = [
  {
    id: "101C",
    code: 101,
    name: "Forward dive · tuck",
    turns: 0.5,
    twists: 0,
    back: 0,
    shape: 0,
    minHeight: 3,
    dd: 1.4,
  },
  {
    id: "103C",
    code: 103,
    name: "Forward 1½ · tuck",
    turns: 1.5,
    twists: 0,
    back: 0,
    shape: 0,
    minHeight: 5,
    dd: 1.5,
  },
  {
    id: "103B",
    code: 103,
    name: "Forward 1½ · pike",
    turns: 1.5,
    twists: 0,
    back: 0,
    shape: 1,
    minHeight: 7.5,
    dd: 1.6,
  },
  {
    id: "201C",
    code: 201,
    name: "Back dive · tuck",
    turns: 0.5,
    twists: 0,
    back: 1,
    shape: 0,
    minHeight: 3,
    dd: 1.7,
  },
  {
    id: "203C",
    code: 203,
    name: "Back 1½ · tuck",
    turns: 1.5,
    twists: 0,
    back: 1,
    shape: 0,
    minHeight: 7.5,
    dd: 1.9,
  },
  {
    id: "5132D",
    code: 5132,
    name: "Forward 1½ · one twist",
    turns: 1.5,
    twists: 1,
    back: 0,
    shape: 2,
    minHeight: 9,
    dd: 2.1,
  },
];
export const LOW = [-0.45, 0, -0.55, -0.5, -0.5, -1.1, -1.1, 0, -0.15],
  HIGH = [2.2, 2.6, 1.35, 3.14, 3.14, 1.1, 1.1, 2.3, 0.18],
  RATE = [8, 10, 8, 10, 10, 8, 8, 10, 4],
  MAP = [0, 1, 2, 0, 1, 2, 3, 5, 7, 4, 6, 7, 8, 8];
export const JOINT_NAMES = [
  "hip_L",
  "knee_L",
  "ankle_L",
  "hip_R",
  "knee_R",
  "ankle_R",
  "shoulder_pitch_L",
  "shoulder_roll_L",
  "elbow_L",
  "shoulder_pitch_R",
  "shoulder_roll_R",
  "elbow_R",
  "hip_adduction_L",
  "hip_adduction_R",
];
export const SITE_NAMES = [
  "pelvis",
  "chest",
  "neck",
  "head",
  "shoulderL",
  "shoulderR",
  "elbowL",
  "elbowR",
  "wristL",
  "wristR",
  "handL",
  "handR",
  "hipL",
  "hipR",
  "kneeL",
  "kneeR",
  "ankleL",
  "ankleR",
  "toeL",
  "toeR",
];
export const BODY_NAMES = [
  "board",
  "pelvis",
  "torso",
  "head",
  "upper_arm_L",
  "upper_arm_R",
  "forearm_L",
  "forearm_R",
  "hand_L",
  "hand_R",
  "thigh_L",
  "thigh_R",
  "shin_L",
  "shin_R",
  "foot_L",
  "foot_R",
];
export const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
export const wrap = (x) =>
  ((((x + Math.PI) % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI)) - Math.PI;
export function up(q) {
  const [w, x, y, z] = q;
  return [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)];
}
export function forward(q) {
  const [w, x, y, z] = q;
  return [1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)];
}
const dot = (a, b) => a.reduce((v, x, i) => v + x * b[i], 0),
  cross = (a, b) => [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
export function framedAngles(q) {
  const u = up(q),
    f = forward(q),
    side = [-u[0] * u[1], 1 - u[1] * u[1], -u[2] * u[1]],
    n = Math.max(1e-8, Math.hypot(...side));
  for (let i = 0; i < 3; i++) side[i] /= n;
  const ref = cross(side, u);
  return [
    Math.atan2(u[0], u[2]),
    Math.atan2(dot(f, side), dot(f, ref)),
    Math.abs(u[1]),
  ];
}
export function parameters(raw = {}) {
  const num = (key, initial, a, b) =>
    Number.isFinite(Number(raw[key])) ? clamp(Number(raw[key]), a, b) : initial;
  return {
    height: num("height", 7.5, 3, 10),
    skill: clamp(
      Math.floor(num("skill", 1, 0, SKILLS.length - 1)),
      0,
      SKILLS.length - 1,
    ),
    tilt: num("tilt", 0.16, 0.04, 0.3),
    preload: num("preload", -0.18, -0.22, -0.14),
    x: num("x", -0.13, -0.2, -0.06),
    disturbance: num("disturbance", 0, -120, 120),
    disturbanceTime: num("disturbanceTime", 0.9, 0.4, 1.5),
    seed: Math.floor(num("seed", 1701, 0, 4294967295)),
  };
}
export function targetsFromAction(action, previous) {
  return previous.map((v, i) => {
    const desired =
      LOW[i] + (clamp(action[i], -1, 1) + 1) * 0.5 * (HIGH[i] - LOW[i]);
    return v + clamp(desired - v, -RATE[i] * DT, RATE[i] * DT);
  });
}
export function contactForces(s, offset = 150) {
  return Array.from({ length: 15 }, (_, i) =>
    Math.hypot(...s.slice(offset + i * 4 + 1, offset + i * 4 + 4)),
  );
}
export function observe(state) {
  const {
      q,
      v,
      s,
      targets,
      phaseTheta: theta,
      airTwist: twist,
      time,
      released,
      parameters: p,
      qadr,
      vadr,
    } = state,
    g = SKILLS[p.skill];
  const distance = Math.max(s[2] + p.height - 0.7, 0),
    vz = s[5],
    tgo = clamp((vz + Math.sqrt(vz * vz + 19.62 * distance)) / 9.81, 0.05, 3);
  return Float32Array.from(
    [
      ...q.slice(4, 8),
      ...v.slice(1, 7).map((x) => x / 10),
      ...qadr.slice(0, 12).map((i) => q[i] / 3),
      ...vadr.slice(0, 12).map((i) => v[i] / 15),
      ...targets.slice(0, 8).map((x) => x / 3),
      s[0] / 4,
      s[1] / 2,
      s[2] / 10,
      ...s.slice(3, 6).map((x) => x / 10),
      ...s.slice(6, 9).map((x) => x / 70),
      theta / (2 * Math.PI),
      twist / (2 * Math.PI),
      (g.turns * 2 * Math.PI - theta) / (2 * Math.PI),
      (g.twists * 2 * Math.PI - twist) / (2 * Math.PI),
      tgo / 2,
      p.height / 10,
      time / 3,
      Number(released),
      q[0] * 4,
      v[0] / 3,
      Number(contactForces(s)[11] > 15),
      Number(contactForces(s)[14] > 15),
      g.turns / 2.5,
      g.twists / 2,
      g.back,
      g.shape / 2,
      ...qadr.slice(12).map(i => q[i] / 0.25),
      ...vadr.slice(12).map(i => v[i] / 4),
      targets[8] / 0.25,
      Number(state.entryTime != null),
      state.entryTime == null ? 0 : (time-state.entryTime)/.8,
      (state.aboveWater??0)/2,
      state.waterFraction??0,
    ],
    (x) => clamp(x, -8, 8),
  );
}
export function validatePolicy(model, expectedXMLHash) {
  const c = model?.contract;
  if (!c || c.waterVersion !== WATER_VERSION || c.schemaVersion !== 1 || c.observationSemantics !== "upright-adduction-full-entry-v4" || c.observationSize !== 76 || c.actionSize !== 9 || c.controlSemantics !== CONTROL_SEMANTICS || c.judgeVersion !== JUDGE_VERSION ||
      c.physics !== "MuJoCo 3.13.0" || !/^[a-f0-9]{64}$/.test(c.xmlSHA256) ||
      (expectedXMLHash && c.xmlSHA256 !== expectedXMLHash) ||
      JSON.stringify(c.actionLow) !== JSON.stringify(LOW) || JSON.stringify(c.actionHigh) !== JSON.stringify(HIGH) ||
      model.crossPhysicsBaseline) throw new Error("Local policy and physical/control contract do not match.");
  if (
    model?.format !== "mujoco-joint-ppo-v1" ||
    model.observationSize !== 76 ||
    model.actionSize !== 9 ||
    model.layers?.length !== 3 ||
    (model.outputActivation !== undefined &&
      model.outputActivation !== "squashed-mean-v1") ||
    (model.outputActivation === "squashed-mean-v1" &&
      model.latentMeanLimit !== 2)
  )
    throw new Error("Unsupported local policy format.");
  let inputs = 76;
  for (const [l, layer] of model.layers.entries()) {
    const outputs = l === 2 ? 9 : 96;
    if (
      layer.weight?.length !== outputs ||
      layer.bias?.length !== outputs ||
      layer.weight.some(
        (row) =>
          row.length !== inputs ||
          row.some((v) => !Number.isFinite(v) || Math.abs(v) > 100),
      ) ||
      layer.bias.some((v) => !Number.isFinite(v) || Math.abs(v) > 100)
    )
      throw new Error("Invalid local policy weights.");
    inputs = outputs;
  }
  return model;
}
export function policyAction(model, observation) {
  let x = observation;
  for (let i = 0; i < model.layers.length; i++) {
    const layer = model.layers[i];
    x = Float32Array.from(layer.bias, (bias, j) => {
      let sum = bias;
      for (let k = 0; k < x.length; k++) sum += layer.weight[j][k] * x[k];
      if (i !== model.layers.length - 1) return Math.tanh(sum);
      return model.outputActivation === "squashed-mean-v1"
        ? Math.tanh(2 * Math.tanh(sum / 2))
        : sum;
    });
  }
  return Array.from(x);
}
/** Explicit comparison controller: a hand-authored feedback heuristic, never AI. */
export function baselineAction(state) {
  const g = SKILLS[state.parameters.skill],
    s = state.s,
    remaining = g.turns * 2 * Math.PI - state.phaseTheta,
    h = Math.max(s[2] + state.parameters.height - 0.7, 0),
    tgo = clamp((s[5] + Math.sqrt(s[5] * s[5] + 19.62 * h)) / 9.81, 0.08, 3);
  let target;
  if (!state.released)
    target = [
      g.back ? (g.turns < 1 ? 1.5 : 0.5) : g.shape === 1 ? 1.5 : 0.5,
      g.turns < 1 || g.back ? 2 : 0,
      -0.5,
      3.05,
      3.05,
      0,
      0,
      0,
    ];
  else {
    // Straight-body transverse inertia is approximately18kg·m² in this rig.
    // Reserve the observed extra angular travel during the opening transition.
    const openSpeed = Math.max(0, s[7]) / 18;
    const needsSpeed =
      remaining >
      openSpeed * tgo + Math.max(0, s[14] - openSpeed) * 0.14 + 0.05;
    target =
      needsSpeed && tgo > 0.3
        ? [1.7, g.shape === 1 ? 0.05 : 2.3, 0.3, 0.4, 0.4, 0, 0, 0.8]
        : [0, 0, 1.35, 2.8, 2.8, -0.24, 0.24, 0];
  }
  const axis = up(state.q.slice(4, 8)),
    axial = s.slice(13, 16).reduce((sum, v, i) => sum + v * axis[i], 0);
  if (
    state.released &&
    g.twists > 0 &&
    tgo > 0.4 &&
    g.twists * 2 * Math.PI - state.airTwist > Math.max(axial, 0) * tgo + 0.2
  ) {
    target[3] = 2.8;
    target[4] = 0.3;
    target[5] = -0.8;
    target[6] = 0.8;
  }
  target.push(state.released ? 0.06 : 0);
  return target.map((v, i) => ((v - LOW[i]) * 2) / (HIGH[i] - LOW[i]) - 1);
}
export function geomExtent(type, size, q) {
  const [w, x, y, z] = q,
    row = [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)];
  if (type === 2) return size[0];
  if (type === 3) return size[0] + size[1] * Math.abs(row[2]);
  if (type === 6) return row.reduce((s, v, i) => s + Math.abs(v) * size[i], 0);
  return Math.sqrt(row.reduce((s, v, i) => s + (v * size[i]) ** 2, 0));
}
export function firstWaterGeometry(s, geometry, height) {
  let lowest = Infinity,
    index = -1,
    point;
  for (let i = 0; i < geometry.length; i++) {
    const base = 45 + i * 7,
      g = geometry[i],
      z = s[base + 2] - geomExtent(g.type, g.size, s.slice(base + 3, base + 7));
    if (
      s[base] >= -0.44 &&
      s[base] <= 11.34 &&
      Math.abs(s[base + 1]) <= 3.56 &&
      z < lowest
    ) {
      lowest = z;
      index = i;
      point = [s[base], s[base + 1], z];
    }
  }
  return { contact: lowest <= -height, index, point, lowest };
}

/** Actual bilateral hand-to-own-shin distance, measured in world space. */
export function tuckGeometry(s) {
  const hands = [s.slice(22, 25), s.slice(25, 28)],
    knees = [s.slice(37, 40), s.slice(40, 43)],
    shins = [s.slice(115, 118), s.slice(136, 139)];
  const ankles = shins.map((p, i) => p.map((v, j) => 2 * v - knees[i][j]));
  const distances = hands.map((p, i) => {
    const axis = ankles[i].map((v, j) => v - knees[i][j]);
    const offset = p.map((v, j) => v - knees[i][j]);
    const fraction = clamp(
      dot(offset, axis) / Math.max(dot(axis, axis), 1e-9),
      0,
      1,
    );
    return Math.hypot(...offset.map((v, j) => v - fraction * axis[j]));
  });
  const gap = (p) => Math.hypot(...p[0].map((v, i) => v - p[1][i]));
  return { distances, legSeparation: Math.max(gap(knees), gap(ankles)) };
}

/** Central-angle posture coverage; repeated rotations over the same interval earn nothing. */
export function postureProgress(
  previousMax,
  theta,
  skill,
  joints,
  released,
  geometry = { distances: [Infinity, Infinity], legSeparation: Infinity },
) {
  const goal = skill.turns * 2 * Math.PI,
    maximum = Math.max(previousMax, theta);
  const covered = (x) => clamp(x - 0.15 * goal, 0, 0.55 * goal);
  const correct =
    skill.shape === 1
      ? Math.min(joints[0], joints[3]) > 0.8 &&
        Math.max(joints[1], joints[4]) < 0.4 &&
        geometry.legSeparation < 0.44
      : Math.min(joints[0], joints[3]) > 0.8 &&
        Math.min(joints[1], joints[4]) > 1.2 &&
        Math.max(...geometry.distances) < 0.24 &&
        geometry.legSeparation < 0.44;
  return {
    maximum,
    angle: released && correct ? covered(maximum) - covered(previousMax) : 0,
  };
}

/** Pair gaps and signed left/right ordering in the physical pelvis frame. */
export function legGeometry(s) {
  const knees=[s.slice(37,40),s.slice(40,43)], shins=[s.slice(115,118),s.slice(136,139)];
  const ankles=shins.map((p,i)=>p.map((v,j)=>2*v-knees[i][j]));
  const toes=[s.slice(16,19),s.slice(19,22)], [w,x,y,z]=s.slice(9,13);
  const side=[2*(x*y-w*z),1-2*(x*x+z*z),2*(y*z+w*x)];
  const pairs=[knees,ankles,toes], gaps=pairs.map(p=>Math.hypot(...p[0].map((v,j)=>v-p[1][j])));
  const signed=pairs.map(p=>dot(p[0].map((v,j)=>v-p[1][j]),side));
  return { kneeGap:gaps[0],ankleGap:gaps[1],toeGap:gaps[2],crossedLegs:signed.some(v=>v<=0.02),signedLegGaps:signed };
}

/** Actual first-contact geometry. Angles are radians, lengths are meters.
 * Foot pitch combines ankle and midfoot; no rendered toe correction is used.
 * Missing/degenerate landmarks fail rather than silently earning ideal form.
 */
export function entryGeometry(s) {
  const hands = [s.slice(22, 25), s.slice(25, 28)],
    knees = [s.slice(37, 40), s.slice(40, 43)],
    shins = [s.slice(115, 118), s.slice(136, 139)],
    toes = [s.slice(16, 19), s.slice(19, 22)];
  const ankles = shins.map((p, i) => p.map((v, j) => 2 * v - knees[i][j]));
  const footLineAngles = ankles.map((a, i) => {
    const shin = a.map((v, j) => v - knees[i][j]);
    const toe = toes[i].map((v, j) => v - a[j]);
    const length = Math.hypot(...shin) * Math.hypot(...toe);
    return length > 1e-9 ? Math.acos(clamp(dot(shin, toe) / length, -1, 1)) : Math.PI;
  });
  const handAxisAngles = [83, 104].map((i) => {
    const q = s.slice(i, i + 4);
    return Math.abs(Math.hypot(...q) - 1) < 1e-3 ? Math.acos(clamp(up(q)[2], -1, 1)) : Math.PI;
  });
  return { ...legGeometry(s), footLineAngles, handAxisAngles,
    handSeparation: Math.hypot(...hands[0].map((v, i) => v - hands[1][i])),
    handHeightGap: Math.abs(hands[0][2] - hands[1][2]) };
}

/** Bilateral posture plus physical foot lines and hand-tip alignment. */
export function entryPosture(j, geometry) {
  if (!geometry) return { form: 0, limbLimits: false, geometryLimits: false };
  const {footLineAngles, handAxisAngles, handSeparation, handHeightGap, kneeGap, ankleGap, toeGap, crossedLegs} = geometry;
  const hips = [j[0], j[3]], knees = [j[1], j[4]],
    pitch = [j[6] - 3.05, j[9] - 3.05],
    roll = [j[7] + 0.3, j[10] - 0.3], elbows = [j[8], j[11]];
  const squares = (a) => a.reduce((s, x) => s + x * x, 0),
    within = (a, m) => a.every((x) => Math.abs(x) < m);
  const geometryLimits = within(footLineAngles, 20 * Math.PI / 180) &&
    within(handAxisAngles, Math.PI / 6) && handSeparation < 0.12 && handHeightGap < 0.04 &&
    kneeGap < 0.19 && ankleGap < 0.16 && toeGap < 0.15 && !crossedLegs;
  return {
    form: Math.exp(-1.25 * (squares(hips) + squares(knees)) -
      0.325 * squares(pitch) - 0.65 * squares(roll) - 0.5 * squares(elbows) -
      0.55 * squares(footLineAngles) -
      1.5 * squares(handAxisAngles.map(a => Math.max(a - Math.PI / 12, 0))) -
      4 * Math.max(handSeparation - 0.07, 0) ** 2 - 20 * handHeightGap ** 2 -
      12 * (Math.max(kneeGap - 0.17, 0)**2 + Math.max(ankleGap - 0.13, 0)**2 + Math.max(toeGap - 0.11, 0)**2) - (crossedLegs ? 4 : 0)),
    geometryLimits,
    limbLimits: within(hips, 0.45) && within(knees, 0.5) && within(pitch, 0.6) &&
      within(roll, 0.5) && within(elbows, 0.5),
  };
}
