import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { MODEL } from "../src/data/model.js";
import {
  DT,
  GRAVITY,
  parameters,
  initialState,
  step,
  simulate,
  policyActions,
  baselineActions,
  score,
  rotate,
  conjugate,
} from "../src/core/physics.js";
const close = (a, b, tol = 1e-9) =>
  assert.ok(Math.abs(a - b) < tol, `${a} != ${b}`);
const fixtures = JSON.parse(
  readFileSync(new URL("./parity-fixtures.json", import.meta.url)),
);
for (const [i, fixture] of fixtures.entries())
  test(`Python/JavaScript complete trajectory parity ${i + 1}`, () => {
    const p = parameters(fixture.parameters),
      actions = policyActions(MODEL.weights, p),
      dive = simulate(p, actions);
    for (const name of Object.keys(actions))
      close(actions[name], fixture.actions[name]);
    assert.equal(dive.frames.length, fixture.trace.length + 1);
    fixture.trace.forEach((expected, i) => {
      const actual = dive.frames[i + 1];
      for (const [key, value] of Object.entries(expected))
        if (Array.isArray(value))
          value.forEach((x, j) => close(actual[key][j], x));
        else close(actual[key], value);
    });
    close(dive.result.total, fixture.score);
  });
test("world angular momentum and quaternion normalization survive all steps", () => {
  for (const actions of [
    { spin: 72, twist: 3, tuck: 0.8, openAt: 0.65 },
    { spin: 20, twist: 0, tuck: 0, openAt: 0.2 },
  ]) {
    const d = simulate({ height: 12, tilt: 0.08, wind: 0.6 }, actions);
    for (const f of d.frames) {
      assert.deepEqual(f.L, d.frames[0].L);
      close(Math.hypot(...f.q), 1);
      for (const v of [...f.q, f.time, f.x, f.y, f.z, f.flips, f.twists])
        assert.ok(Number.isFinite(v));
    }
    assert.ok(d.frames.length < 177);
  }
});
test("COM trajectory follows ballistic gravity and a lateral force through its center", () => {
  const d = simulate(
    { height: 8, spring: 3, wind: 0.4 },
    baselineActions(parameters({ height: 8, spring: 3, wind: 0.4 })),
  );
  for (const f of d.frames) {
    close(f.x, 1.6 * f.time);
    close(f.y, 8.9 + 3 * f.time - 0.5 * GRAVITY * f.time ** 2);
    close(f.z, 0.2 * f.time ** 2);
  }
});
test("tuck increases rotation speed without adding angular momentum", () => {
  let open = initialState(
      { height: 12 },
      { spin: 30, twist: 0, tuck: 0, openAt: 0.8 },
    ),
    tuck = initialState(
      { height: 12 },
      { spin: 30, twist: 0, tuck: 0.9, openAt: 0.8 },
    );
  for (let i = 0; i < 15; i++) {
    open = step(open);
    tuck = step(tuck);
  }
  assert.ok(Math.hypot(...tuck.omega) > Math.hypot(...open.omega) * 2);
  assert.deepEqual(tuck.L, open.L);
});
test("terminal state is immutable under extra steps", () => {
  const p = parameters(),
    d = simulate(p, baselineActions(p)),
    last = d.frames.at(-1);
  assert.equal(step(last), last);
  assert.ok(last.done);
  assert.ok(last.time > DT);
});
test("invalid entry earns no difficulty or other points even with huge spin counts", () => {
  const s = initialState({}, { spin: 30, twist: 1, tuck: 0, openAt: 0.2 });
  s.x = 3;
  s.flips = 100;
  s.twists = 100;
  s.done = true;
  assert.equal(score(s).valid, false);
  assert.equal(score(s).total, 0);
  s.q = [0, 0, 1, 0];
  s.x = 12;
  assert.equal(score(s).total, 0);
  s.x = 3;
  s.tuck = 0.8;
  assert.equal(score(s).total, 0);
});
test("fresh seeded contexts produce valid learned dives across all height bins", () => {
  let seed = 884019;
  const random = () =>
    (seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 4294967296;
  let learned = 0,
    base = 0,
    valid = 0;
  for (let i = 0; i < 250; i++) {
    const p = parameters({
      height: 2 + 10 * random(),
      spring: 1.5 + 2 * random(),
      wind: 1.2 * random() - 0.6,
      tilt: 0.18 * random() - 0.09,
    });
    const d = simulate(p, policyActions(MODEL.weights, p));
    learned += d.result.total;
    base += simulate(p, baselineActions(p)).result.total;
    valid += Number(d.result.valid);
    assert.ok(d.result.total <= 100);
  }
  assert.ok(valid >= 245);
  assert.ok(learned > base);
});
test("policy is contextual and replay is deterministic", () => {
  const p = parameters({ height: 5 });
  const a = policyActions(MODEL.weights, p);
  assert.notDeepEqual(
    a,
    policyActions(MODEL.weights, parameters({ height: 10 })),
  );
  assert.deepEqual(simulate(p, a), simulate(p, a));
  assert.notDeepEqual(MODEL.weights, MODEL.initialWeights);
});
test("environment settings are bounded and nonfinite inputs cannot poison physics", () => {
  const p = parameters({
    height: Infinity,
    spring: -Infinity,
    wind: NaN,
    tilt: Infinity,
  });
  assert.deepEqual(p, { height: 12, spring: 1.5, wind: 0, tilt: 0.09 });
  assert.ok(
    simulate(p, policyActions(MODEL.weights, p)).frames.every((f) =>
      Number.isFinite(f.y),
    ),
  );
});

test("constant-shape free flight conserves rotational energy", () => {
  for (const twist of [0, 1.5, 4]) {
    const d = simulate(
      { height: 12 },
      { spin: 40, twist, tuck: 0, openAt: 0.2 },
    );
    const energy = (f) => {
      const l = rotate(conjugate(f.q), f.L);
      return 0.5 * (l[0] ** 2 / 12 + l[1] ** 2 / 0.9 + l[2] ** 2 / 12);
    };
    for (const f of d.frames) close(energy(f), energy(d.frames[0]), 1e-8);
  }
});

test("a correctly oriented airborne body cannot earn entry points", () => {
  const s = initialState(
    { height: 5 },
    { spin: 30, twist: 0, tuck: 0, openAt: 0.2 },
  );
  s.q = [0, 0, 1, 0];
  s.x = 3;
  s.flips = 1.5;
  assert.equal(score(s).valid, false);
  assert.equal(score(s).total, 0);
  assert.equal(score(s).outcome, "Still in flight");
});
