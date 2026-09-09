export const DT = 0.02,
  GRAVITY = 9.81;
export const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
export const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
export const cross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
export const normalized = (v) => {
  const n = Math.hypot(...v) || 1;
  return v.map((x) => x / n);
};
export function multiply(a, b) {
  const [x, y, z, w] = a,
    [X, Y, Z, W] = b;
  return [
    w * X + x * W + y * Z - z * Y,
    w * Y - x * Z + y * W + z * X,
    w * Z + x * Y - y * X + z * W,
    w * W - x * X - y * Y - z * Z,
  ];
}
export function rotate(q, v) {
  const u = q.slice(0, 3),
    uv = cross(u, v),
    uuv = cross(u, uv);
  return v.map((x, i) => x + 2 * (q[3] * uv[i] + uuv[i]));
}
export const conjugate = (q) => [-q[0], -q[1], -q[2], q[3]];
export function parameters(input = {}) {
  return {
    height: clamp(Number(input.height) || 5, 2, 12),
    spring: clamp(Number(input.spring) || 2.4, 1.5, 3.5),
    wind: clamp(Number(input.wind) || 0, -0.6, 0.6),
    tilt: clamp(Number(input.tilt) || 0, -0.09, 0.09),
  };
}
export function flightTime(p) {
  return (
    (p.spring + Math.sqrt(p.spring * p.spring + 2 * GRAVITY * p.height)) /
    GRAVITY
  );
}
export function features(p) {
  const v = (1 / flightTime(p) - 0.65) * 3;
  return [1, v, v * v, p.tilt / 0.09, p.spring - 2.5];
}
export function decodeActions(raw) {
  return {
    spin: clamp(45 + raw[0] * 18, 10, 95),
    twist: clamp(1.2 + raw[1] * 0.9, 0, 4),
    tuck: clamp(0.5 + raw[2] * 0.22, 0, 0.95),
    openAt: clamp(0.58 + raw[3] * 0.12, 0.18, 0.86),
  };
}
export function policyActions(weights, p) {
  const f = features(p);
  return decodeActions(weights.map((row) => dot(row, f)));
}
export function baselineActions(p) {
  return {
    spin: (12 * (Math.PI - p.tilt)) / flightTime(p),
    twist: 0,
    tuck: 0,
    openAt: 0.2,
  };
}
export function initialState(input, actions) {
  const p = parameters(input),
    q = [0, 0, Math.sin(p.tilt / 2), Math.cos(p.tilt / 2)],
    L = rotate(q, [0, actions.twist, actions.spin]);
  return {
    p,
    actions,
    time: 0,
    q,
    L,
    x: 0,
    y: p.height + 0.9,
    z: 0,
    vx: 1.6,
    vy: p.spring,
    vz: 0,
    tuck: 0,
    flips: 0,
    twists: 0,
    twistPhase: 0,
    done: false,
  };
}
export function bodyExtent(tuck, axis) {
  return (
    (0.9 - 0.42 * tuck) * Math.abs(axis[1]) +
    (0.13 + 0.13 * tuck) * Math.sqrt(Math.max(0, 1 - axis[1] * axis[1]))
  );
}
export function step(state) {
  if (state.done) return state;
  const s = { ...state },
    target = s.time / flightTime(s.p) < s.actions.openAt ? s.actions.tuck : 0;
  s.tuck += clamp(target - s.tuck, -DT * 3.5, DT * 3.5);
  const I = [12 - 8.8 * s.tuck, 0.9 + 0.8 * s.tuck, 12 - 8.8 * s.tuck],
    localL = rotate(conjugate(s.q), s.L),
    localW = localL.map((x, i) => x / I[i]),
    omega = rotate(s.q, localW);
  // Exact symmetric-top flow for this step's fixed principal inertia: world precession, then body-axis spin.
  const Lhat = normalized(s.L),
    precession = (Math.hypot(...s.L) * DT) / I[0],
    spin = (1 / I[1] - 1 / I[0]) * localL[1] * DT;
  const world = [
      ...Lhat.map((x) => x * Math.sin(precession / 2)),
      Math.cos(precession / 2),
    ],
    body = [0, Math.sin(spin / 2), 0, Math.cos(spin / 2)];
  const previousAxis = rotate(s.q, [0, 1, 0]);
  s.q = normalized(multiply(multiply(world, s.q), body));
  const axis = rotate(s.q, [0, 1, 0]);
  const project = (v) =>
      normalized(v.map((x, i) => x - dot(v, Lhat) * Lhat[i])),
    a = project(previousAxis),
    b = project(axis);
  s.flips += Math.atan2(dot(Lhat, cross(a, b)), dot(a, b)) / (2 * Math.PI);
  const right = normalized(cross(axis, Lhat)),
    forward = cross(right, axis),
    bodyRight = rotate(s.q, [1, 0, 0]);
  const phase = Math.atan2(dot(bodyRight, forward), dot(bodyRight, right)),
    change = Math.atan2(
      Math.sin(phase - s.twistPhase),
      Math.cos(phase - s.twistPhase),
    );
  s.twists += change / (2 * Math.PI);
  s.twistPhase = phase;
  s.time += DT;
  s.x = 1.6 * s.time;
  s.z = 0.5 * s.p.wind * s.time * s.time;
  s.y =
    s.p.height + 0.9 + s.p.spring * s.time - 0.5 * GRAVITY * s.time * s.time;
  s.vy = s.p.spring - GRAVITY * s.time;
  s.vz = s.p.wind * s.time;
  s.done = s.y <= bodyExtent(s.tuck, axis) || s.time >= 3.5;
  s.omega = omega;
  s.axis = axis;
  return s;
}
export function score(state) {
  const axis = rotate(state.q, [0, 1, 0]),
    angle = (Math.acos(clamp(-axis[1], -1, 1)) * 180) / Math.PI,
    extension = 1 - state.tuck;
  const inPool = state.x > 0.3 && state.x < 8 && Math.abs(state.z) < 2.8;
  const contact = state.done && state.y <= bodyExtent(state.tuck, axis);
  const valid = contact && inPool && angle <= 30 && extension >= 0.7;
  const flips = Math.abs(state.flips),
    twists = Math.abs(state.twists),
    alignment = Math.max(0, Math.cos((angle * Math.PI) / 180)) ** 2;
  const horizontal = Math.hypot(state.vx, state.vz),
    area =
      0.12 + 0.88 * Math.sin((angle * Math.PI) / 180) ** 2 + 0.45 * state.tuck;
  const splash = clamp(
    area * (0.45 + 0.035 * Math.abs(state.vy)) +
      0.035 * horizontal * horizontal,
    0,
    1,
  );
  const difficulty = clamp(Math.max(0, flips - 0.5) / 2.5 + twists / 2.5, 0, 1);
  const breakdown = {
    difficulty: valid ? 20 * difficulty : 0,
    execution: valid ? 20 * extension : 0,
    alignment: valid ? 45 * alignment : 0,
    splash: valid ? 15 * (1 - splash) : 0,
  };
  const total = Object.values(breakdown).reduce((a, b) => a + b, 0);
  return {
    total,
    valid,
    angle,
    extension,
    flips,
    twists,
    splash,
    entrySpeed: Math.hypot(state.vx, state.vy, state.vz),
    horizontalSpeed: horizontal,
    breakdown,
    outcome: !contact
      ? "Still in flight"
      : !inPool
        ? "Outside the pool"
        : angle > 90
          ? "Feet or back first"
          : angle > 30
            ? "Off-axis entry"
            : extension < 0.7
              ? "Still tucked at entry"
              : "Valid head-first entry",
  };
}
export function simulate(input, actions) {
  let state = initialState(input, actions);
  const frames = [state];
  while (!state.done) {
    state = step(state);
    frames.push(state);
  }
  return { frames, result: score(state), actions, parameters: state.p };
}
