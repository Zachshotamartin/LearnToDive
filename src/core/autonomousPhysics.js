/** Browser counterpart of training_v3: declarations, physical motors and whole-entry judging.
 * Keeps the legacy renderer's frame format; observations and scores are v11 only.
 */
import { PhysicsEngine } from "./physics.js";
import {
  MAP,
  LOW,
  HIGH,
  clamp,
  wrap,
  up,
  framedAngles,
  entryGeometry,
  tuckGeometry,
  contactForces,
  firstWaterGeometry,
} from "./control.js";
import { immersion, bodyWater } from "./autonomousWater.js";
import { DIVES, CODES } from "../data/declarations.js";
import { legalDeclarations, requestedDeclaration } from "../data/diveChoices.js";

/** Ballistic seconds until the surface from the height above it and the vertical speed; a kinematic reading, not a plan. */
export function timeToSurface(height, verticalSpeed, limit = 3) {
  const h = Math.max(height, 0);
  const arrival = (verticalSpeed + Math.sqrt(verticalSpeed * verticalSpeed + 2 * GRAVITY_CONSTANT * h)) / GRAVITY_CONSTANT;
  return Math.min(Math.max(arrival, 0), limit);
}
const GRAVITY_CONSTANT = 9.81;
export { legalDeclarations } from "../data/diveChoices.js";
const RATE = [14, 14, 14, 10, 10, 8, 8, 10, 4],
  FEET = [11, 14],
  NEAR_WATER = 3,
  // v13 layout: task-context columns of a full dive (task 0 indicator, zero goals)
  // and the previous exploration noise, which is zero in deterministic playback.
  FULL_DIVE_CONTEXT = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  NO_NOISE = Array(9).fill(0),
  HANDS = [5, 8];
const rad = Math.PI / 180,
  sum = (a) => a.reduce((s, x) => s + x, 0),
  sq = (x) => x * x;
const orientation = (t, y) => [
  Math.cos(t / 2) * Math.cos(y / 2),
  Math.sin(t / 2) * Math.sin(y / 2),
  Math.sin(t / 2) * Math.cos(y / 2),
  Math.cos(t / 2) * Math.sin(y / 2),
];
function bisect(fn, lo, hi) {
  let a = fn(lo);
  for (let i = 0; i < 40; i++) {
    const m = (lo + hi) / 2,
      b = fn(m);
    if (Math.sign(a) === Math.sign(b)) {
      lo = m;
      a = b;
    } else hi = m;
  }
  return (lo + hi) / 2;
}
export class AutonomousEngine extends PhysicsEngine {
  resetContext({
    height = 7.5,
    apparatus = "platform",
    category = 2,
    used = [],
    round = 0,
  } = {}) {
    if (!legalDeclarations(category, apparatus, height, used).some(Boolean))
      throw Error("No legal declaration for these conditions");
    super.reset({ height: Math.max(3, height), skill: 0 });
    this.parameters = {
      height,
      apparatus,
      category,
      used: [...used],
      round,
      preload: apparatus === "platform" ? 0 : -0.05123,
    };
    this.platform = apparatus === "platform";
    this.armstand = false;
    this.headfirst = true;
    this.choosing = true;
    this.firstState = null;
    this.firstSensors = null;
    this.previousActions = Array(9).fill(0);
    this.positionSums = Array(4).fill(0);
    this.positionTicks = 0;
    this.surfaceFinished = Array(15).fill(false);
    this.surfaceLoss = Array(6).fill(0);
    this.surfaceLateral = 0;
    this.entryOmega = 0;
    this.place(false);
    this.read();
    this.waterState();
    this.frames = [];
    this.actions = [];
    return this;
  }
  place(back, armstand = false) {
    const { mj, model: m, data: d } = this,
      p = this.parameters,
      feet = FEET.map((i) => i + 1);
    const place = (ankle) => {
      mj.mj_resetData(m, d);
      d.mocap_pos.set([0, 0, -p.height]);
      d.mocap_quat.set([1, 0, 0, 0]);
      mj.mj_setState(m, d, [Number(this.platform)], 512);
      d.qpos[0] = p.preload;
      d.qpos.set(orientation(0.04, Math.PI * Number(back)), 4);
      this.targets = [0.12, 0.2, ankle, 0.3, 0.3, 0, 0, 0, 0];
      this.qadr.forEach((q, i) => {
        d.qpos[q] = this.targets[MAP[i]];
        d.ctrl[i] = this.targets[MAP[i]];
      });
      mj.mj_forward(m, d);
      return d.geom_xmat[feet[0] * 9 + 6];
    };
    place(bisect(place, -0.6, 1.4));
    const extent = (g, row) =>
      sum(
        [0, 1, 2].map(
          (i) =>
            Math.abs(d.geom_xmat[g * 9 + row * 3 + i]) * m.geom_size[g * 3 + i],
        ),
      );
    d.qpos[3] +=
      p.preload +
      0.055 -
      Math.min(...feet.map((g) => d.geom_xpos[g * 3 + 2] - extent(g, 2))) -
      0.0001;
    d.qpos[1] +=
      0.03 - Math.max(...feet.map((g) => d.geom_xpos[g * 3] + extent(g, 0)));
    mj.mj_forward(m, d);
    this.read();
    this.waterState();
    if (armstand) {
      const pose = (t) => {
        this.targets = [0, 0, 1.2, 3.14, 3.14, 0, 0, 0, 0.06];
        d.qpos.set(orientation(t, Math.PI * Number(back)), 4);
        this.qadr.forEach((q, i) => {
          d.qpos[q] = this.targets[MAP[i]];
          d.ctrl[i] = this.targets[MAP[i]];
        });
        d.qpos[0] = d.qpos[1] = d.qpos[3] = 0;
        mj.mj_forward(m, d);
        const hands = [6, 9];
        const bottom = Math.min(
          ...hands.map(
            (g) =>
              d.geom_xpos[g * 3 + 2] -
              Math.hypot(
                ...[0, 1, 2].map(
                  (i) => d.geom_xmat[g * 9 + 6 + i] * m.geom_size[g * 3 + i],
                ),
              ),
          ),
        );
        d.qpos[3] += 0.055 - bottom - 0.0001;
        d.qpos[1] += -0.2 - sum(hands.map((g) => d.geom_xpos[g * 3])) / 2;
        mj.mj_forward(m, d);
        return d.subtree_com[6] - sum(hands.map((g) => d.geom_xpos[g * 3])) / 2;
      };
      pose(bisect(pose, Math.PI - 0.35, Math.PI + 0.35));
    }
    this.read();
    [this.prevPitch, this.prevTwist] = framedAngles(this.q.slice(4, 8));
  }
  declare(index) {
    const p = this.parameters;
    if (!legalDeclarations(p.category, p.apparatus, p.height, p.used)[index])
      throw Error("Illegal declaration");
    this.declaration = DIVES[index];
    this.declarationIndex = index;
    this.armstand = this.declaration.armstand;
    this.headfirst = this.declaration.headfirst;
    this.place(this.declaration.back, this.armstand);
    this.choosing = false;
    this.frames = [this.frame()];
  }
  observation() {
    const p = this.parameters,
      q = this.q,
      v = this.v,
      s = this.s,
      d = this.declaration,
      intent = Array(9).fill(0);
    if (!this.choosing) {
      intent.splice(
        0,
        5,
        (d.sign * d.turns) / 5,
        d.twists / 5,
        d.back,
        Number(d.armstand),
        d.direction / 4,
      );
      intent[5 + "ABCD".indexOf(d.position)] = 1;
    }
    const forces = contactForces(s);
    return [
      ...q.slice(4, 8),
      ...v.slice(1, 7).map((x) => x / 10),
      ...this.qadr.map((i) => q[i] / 3),
      ...this.vadr.map((i) => v[i] / 15),
      ...this.targets.map((x) => x / 3),
      ...s.slice(0, 3).map((x, i) => x / [4, 2, 10][i]),
      ...s.slice(3, 6).map((x) => x / 10),
      this.phaseTheta / (2 * Math.PI),
      this.airTwist / (2 * Math.PI),
      p.height / 10,
      this.time / 4,
      Number(this.released),
      Number(this.platform),
      Number(this.armstand),
      this.waterFraction,
      this.aboveWater / 10,
      timeToSurface(this.aboveWater, s[5]) / 2,
      Math.min(this.aboveWater, NEAR_WATER) / NEAR_WATER,
      Number(this.choosing),
      p.round / 6,
      ...FEET.map((i) => Number(forces[i] > 15)),
      q[0] * 4,
      v[0] / 3,
      ...Array.from({ length: 6 }, (_, i) => Number(i === p.category - 1)),
      ...intent,
      ...CODES.map((c) => Number(p.used.includes(c))),
      ...this.remainingRotation(),
      ...FULL_DIVE_CONTEXT,
      ...NO_NOISE,
      ...this.previousActions,
    ].map((x) => Math.fround(clamp(x, -10, 10)));
  }
  remainingRotation() {
    // Declared counts minus the measured unwrapped turns, exactly as the judge measures them.
    if (this.choosing) return [0, 0];
    const d = this.declaration;
    return [
      (d.sign * d.turns - this.phaseTheta / (2 * Math.PI)) / 5,
      (d.twists - Math.abs(this.airTwist / (2 * Math.PI))) / 5,
    ];
  }
  waterState(apply = false) {
    if (!this.parameters) return;
    const { data: d, model: m } = this,
      h = this.parameters.height;
    if (apply) d.xfrc_applied.fill(0);
    let mass = 0,
      above = -Infinity;
    this.fractions = [];
    for (const g of this.geometry) {
      const i = g.id - 1,
        pos = this.s.slice(45 + i * 7, 48 + i * 7),
        quat = this.s.slice(48 + i * 7, 52 + i * 7),
        wet = immersion(g.type, g.size, pos, quat, -h);
      mass += wet.fraction * m.body_mass[g.body];
      above = Math.max(above, pos[2] + wet.extent + h);
      this.fractions.push(wet.fraction);
      if (apply) {
        const velocity = this.s.slice(270 + i * 6, 276 + i * 6),
          f = bodyWater({
            type: g.type,
            size: g.size,
            position: pos,
            quaternion: quat,
            com: pos,
            linear: velocity.slice(0, 3),
            angular: velocity.slice(3),
            mass: m.body_mass[g.body],
            waterZ: -h,
          });
        d.xfrc_applied.set([...f.force, ...f.torque], g.body * 6);
      }
    }
    this.waterFraction = mass / 70;
    this.aboveWater = above;
    this.allSubmerged = this.fractions.every((x) => x === 1);
  }
  trackLaunchV11(duration) {
    if (duration <= 0) return;
    const s = this.s,
      q = this.q,
      forces = contactForces(s),
      support = contactForces(s, 210),
      allowed = this.armstand ? HANDS : FEET;
    const assisting = forces.map((x, i) => (allowed.includes(i) ? 0 : x)),
      pitch = framedAngles(q.slice(4, 8))[0],
      u = up(q.slice(4, 8));
    this.boardNonfootImpulse += sum(assisting) * duration;
    this.boardNonfootPeak = Math.max(this.boardNonfootPeak, ...assisting);
    this.standImpulse += sum(support) * duration;
    this.standPeak = Math.max(this.standPeak, ...support);
    this.footSideContact ||=
      !this.armstand &&
      FEET.some((i) => s[47 + i * 7] < q[0] + 0.04 && forces[i] > 15);
    const grounded = Array.from({ length: 15 }, (_, i) => s[150 + i * 4]).some(
      (x) => x > 0,
    );
    allowed.forEach((g, i) => {
      if (s[150 + g * 4] > 0) this.lastFootContact[i] = this.time;
    });
    if (this.released && grounded) {
      if (this.takeoffVerticalSpeed > 0.25) this.preparationBounces++;
      this.rotatedRecontact ||= (this.armstand ? -u[2] : u[2]) < 0.87;
    }
    this.boardInvalid ||=
      this.boardNonfootImpulse > 0.1 ||
      this.boardNonfootPeak > 15 ||
      this.standImpulse > 0.1 ||
      this.standPeak > 15 ||
      this.rotatedRecontact ||
      this.footSideContact;
    if (grounded) {
      this.airClearTime = 0;
      this.departureCOM = this.apexCOM = s[2];
      this.released = false;
      this.maxLateral = 0;
    } else {
      if (this.airClearTime === 0) {
        this.departureCOM = this.apexCOM = s[2];
        this.takeoffVerticalSpeed = s[5];
        this.departurePitch = pitch;
        this.departureTheta = this.theta;
        this.departureTwist = this.twist;
      }
      this.apexCOM = Math.max(this.apexCOM, s[2]);
      this.airClearTime += duration;
      if (
        !this.released &&
        this.airClearTime >= 0.04 - 1e-8 &&
        this.time > 0.08
      ) {
        this.released = true;
        this.releaseTime = this.time - this.airClearTime + 0.002;
        this.releaseTheta = this.departureTheta;
        this.releaseTwist = this.departureTwist;
        this.footDepartureGap = Math.abs(
          this.lastFootContact[0] - this.lastFootContact[1],
        );
        this.takeoffTiltInvalid ||=
          (this.armstand
            ? Math.abs(Math.abs(this.departurePitch) - Math.PI)
            : Math.abs(this.departurePitch)) >
          Math.PI / 2;
        this.boardInvalid ||=
          this.takeoffTiltInvalid || this.footDepartureGap > 0.04 + 1e-8;
      }
    }
    this.airTheta = this.released ? this.theta - this.releaseTheta : 0;
    this.phaseTheta = this.released
      ? this.airTheta +
        (this.armstand
          ? wrap(this.departurePitch - Math.PI)
          : this.departurePitch)
      : 0;
    this.airTwist = this.released ? this.twist - this.releaseTwist : 0;
    this.maxLateral = Math.max(
      this.maxLateral,
      Math.abs(u[1]) * Number(this.released),
    );
  }
  trackEntry() {
    const active = this.fractions.map(
        (x, i) => x > 0 && !this.surfaceFinished[i],
      ),
      g = entryGeometry(this.s),
      head = this.headfirst,
      any = (a) => a.some((i) => active[i]),
      hands = any(HANDS),
      legs = any([9, 10, 11, 12, 13, 14]),
      trunk = any([0, 1, 2]);
    const feet = FEET.map((i) => active[i]),
      hand = HANDS.map((i) => active[i]),
      thigh = [9, 12].map((i) => active[i]),
      shin = [10, 13].map((i) => active[i]),
      shoulder = [3, 6].map((i) => active[i]),
      forearm = [4, 7].map((i) => active[i]);
    const kneeCross = thigh.some(Boolean) || shin.some(Boolean),
      ankleCross = shin.some(Boolean) || feet.some(Boolean),
      toeCross = feet.some(Boolean);
    for (const [key, mask] of [
      ["footLineAngles", feet],
      ["handAxisAngles", hand],
    ])
      this.entryWorstGeometry[key] = this.entryWorstGeometry[key].map((x, i) =>
        Math.max(x, mask[i] ? g[key][i] : 0),
      );
    for (const [key, on] of [
      ["handSeparation", hands],
      ["handHeightGap", hands],
      ["kneeGap", kneeCross],
      ["ankleGap", ankleCross],
      ["toeGap", toeCross],
    ])
      if (on)
        this.entryWorstGeometry[key] = Math.max(
          this.entryWorstGeometry[key],
          g[key],
        );
    this.entryWorstGeometry.crossedLegs ||= legs && g.crossedLegs;
    if (trunk || legs)
      this.entryMaxAngle = Math.max(
        this.entryMaxAngle,
        Math.acos(clamp((head ? -1 : 1) * up(this.q.slice(4, 8))[2], -1, 1)) /
          rad,
      );
    const j = this.qadr.map((i) => this.q[i]),
      hip = [j[0], j[3]],
      knee = [j[1], j[4]],
      pitch = [j[6], j[9]].map((x) => x - (head ? 3.05 : 0)),
      roll = [j[7], j[10]].map((x, i) => x - (head ? [-0.3, 0.3][i] : 0)),
      elbow = [j[8], j[11]],
      pair = (fn) => sum([0, 1].map(fn));
    const losses = [
      pair((i) => 1.25 * sq(hip[i]) * thigh[i]),
      pair((i) => 1.25 * sq(knee[i]) * shin[i]) +
        12 * sq(Math.max(g.kneeGap - 0.17, 0)) * kneeCross,
      pair((i) => 0.55 * sq(g.footLineAngles[i]) * feet[i]) +
        12 *
          (sq(Math.max(g.ankleGap - 0.13, 0)) * ankleCross +
            sq(Math.max(g.toeGap - 0.11, 0)) * toeCross) +
        4 * Number(g.crossedLegs) * toeCross,
      pair((i) => (0.325 * sq(pitch[i]) + 0.65 * sq(roll[i])) * shoulder[i]),
      pair((i) => 0.5 * sq(elbow[i]) * forearm[i]),
      Number(head) *
        (pair(
          (i) =>
            1.5 * sq(Math.max(g.handAxisAngles[i] - Math.PI / 12, 0)) * hand[i],
        ) +
          (4 * sq(Math.max(g.handSeparation - 0.07, 0)) +
            20 * sq(g.handHeightGap)) *
            hands),
    ];
    this.surfaceLoss = this.surfaceLoss.map((x, i) => Math.max(x, losses[i]));
    this.entryMinForm = Math.exp(-sum(this.surfaceLoss));
    this.entryLimbsValid &&= [0, 1].every(
      (i) =>
        (!thigh[i] || Math.abs(hip[i]) < 0.45) &&
        (!shin[i] || Math.abs(knee[i]) < 0.5) &&
        (!shoulder[i] ||
          (Math.abs(pitch[i]) < 0.6 && Math.abs(roll[i]) < 0.5)) &&
        (!forearm[i] || Math.abs(elbow[i]) < 0.5),
    );
    const handOK =
      hand.every((on, i) => !on || g.handAxisAngles[i] < Math.PI / 6) &&
      (!hands || (g.handSeparation < 0.12 && g.handHeightGap < 0.04));
    const legOK =
      feet.every((on, i) => !on || g.footLineAngles[i] < 20 * rad) &&
      (!kneeCross || g.kneeGap < 0.19) &&
      (!ankleCross || g.ankleGap < 0.16) &&
      (!toeCross || (g.toeGap < 0.15 && !g.crossedLegs));
    this.entryGeometryValid &&= (!head || handOK) && legOK;
    this.surfaceLateral = Math.max(
      this.surfaceLateral,
      ...active.map((on, i) => (on ? Math.abs(this.s[271 + i * 6]) : 0)),
    );
    this.surfaceFinished = this.surfaceFinished.map(
      (x, i) => x || this.fractions[i] >= 1,
    );
    if (this.allSubmerged && this.fullEntryTime === null)
      this.fullEntryTime = this.time;
  }
  step(action, { record = true } = {}) {
    const { data: d, model: m, mj } = this,
      startedWet = this.entryTime !== null,
      old = this.state();
    this.targets = this.targets.map(
      (x, i) =>
        x +
        clamp(
          LOW[i] + (clamp(action[i], -1, 1) + 1) * 0.5 * (HIGH[i] - LOW[i]) - x,
          -RATE[i] * 0.02,
          RATE[i] * 0.02,
        ),
    );
    this.qadr.forEach((_, i) => {
      d.ctrl[i] = this.targets[MAP[i]];
    });
    mj.mj_setState(m, d, [Number(this.platform)], 512);
    d.xfrc_applied.fill(0);
    d.qacc_warmstart.fill(0);
    this.actions.push({
      time: this.time,
      values: [...action],
      targets: [...this.targets],
    });
    for (let k = 0; k < 10; k++) {
      if (startedWet) {
        this.applyWater();
        d.qacc_warmstart.fill(0);
      }
      const before = this.state();
      mj.mj_step(m, d);
      this.read();
      const contact = firstWaterGeometry(
        this.s,
        this.geometry,
        this.parameters.height,
      );
      if (!startedWet && contact.contact) {
        this.restore(before);
        this.entryTime = this.time;
        this.firstState = this.q.slice();
        this.firstSensors = this.s.slice();
        this.firstGeometry = contact.index + 1;
        this.contactPoint = [contact.point[0], 0, -contact.point[1]];
      }
      if (!startedWet) {
        const [a, b, lateral] = framedAngles(this.q.slice(4, 8));
        this.maxLateral = Math.max(this.maxLateral, lateral);
        this.theta += wrap(a - this.prevPitch);
        this.twist += wrap(b - this.prevTwist);
        this.prevPitch = a;
        this.prevTwist = b;
        this.trackLaunchV11(this.time - before[0]);
      }
      if (startedWet) {
        mj.mj_forward(m, d);
        this.read();
        this.waterState();
        this.trackEntry();
        if (
          this.fullEntryTime !== null ||
          this.time - this.entryTime >= 1.2 - 1e-9
        )
          break;
      } else if (this.entryTime !== null) {
        this.waterState();
        this.trackEntry();
        break;
      }
    }
    this.previousActions = [...action];
    this.waterState();
    const hip = this.q[this.qadr[0]],
      knee = this.q[this.qadr[1]],
      tuck =
        Math.exp(-sq((hip - 1.4) / 0.8) - sq((knee - 2) / 0.8)) *
        Math.exp(-sq(Math.max(...tuckGeometry(this.s).distances) / 0.45)),
      pike = Math.exp(-sq((hip - 1.5) / 0.8) - sq(knee / 0.35)),
      straight = Math.exp(-sq(hip / 0.35) - sq(knee / 0.35)),
      turns = this.declaration.sign * this.declaration.turns,
      progress =
        this.phaseTheta / (Math.abs(turns) > 1e-6 ? turns * 2 * Math.PI : 1);
    if (
      this.released &&
      this.entryTime === null &&
      ((progress > 0.15 && progress < 0.8) || turns === 0)
    ) {
      const a = [straight, pike, tuck, Math.max(straight, pike, tuck)];
      this.positionSums = this.positionSums.map((x, i) => x + a[i]);
      this.positionTicks++;
    }
    if (this.entryTime !== null)
      this.entryOmega = Math.max(
        this.entryOmega,
        Math.hypot(...this.s.slice(13, 16)),
      );
    if (record) this.frames.push(this.frame());
    return (
      this.fullEntryTime !== null ||
      (this.entryTime !== null
        ? this.time - this.entryTime >= 1.2 - 1e-9
        : this.time >= 3.6) ||
      Math.hypot(...this.q.slice(1, 4).map((x, i) => x - old[2 + i])) > 1
    );
  }
  measurements() {
    const q = this.firstState ?? this.q,
      s = this.firstSensors ?? this.s,
      g = this.entryWorstGeometry;
    return {
      rotation: this.phaseTheta / (2 * Math.PI),
      twist: this.airTwist / (2 * Math.PI),
      boardInvalid: this.boardInvalid,
      water: this.entryTime !== null,
      fullEntryComplete: this.fullEntryTime !== null,
      firstGeometry: this.firstGeometry,
      x: s[0],
      firstContactAngle:
        Math.acos(
          clamp((this.headfirst ? -1 : 1) * up(q.slice(4, 8))[2], -1, 1),
        ) / rad,
      entryAngle: this.entryMaxAngle,
      form: this.entryMinForm,
      entryGeometryValid: this.entryGeometryValid,
      entryLimbsValid: this.entryLimbsValid,
      entryGeometryWorst: {
        ...g,
        footLineAngles: g.footLineAngles.map((x) => x / rad),
        handAxisAngles: g.handAxisAngles.map((x) => x / rad),
      },
      entryArmPositionValid:
        Math.max(
          ...[6, 9].map((i) =>
            Math.abs(q[this.qadr[i]] - (this.headfirst ? 3.05 : 0)),
          ),
        ) < 0.8,
      positionQualities: this.positionSums.map(
        (x) => x / Math.max(1, this.positionTicks),
      ),
      ascent: Math.max(0, this.apexCOM - this.departureCOM),
      preparationBounces: this.preparationBounces,
      surfaceLateralSpeed: this.surfaceLateral,
      maxLateral: this.maxLateral,
      time: this.time,
    };
  }
}
export function judgeDeclaration(d, apparatus, height, m) {
  const rotationError = Math.abs(m.rotation * d.sign - d.turns),
    twistError = Math.abs(Math.abs(m.twist) - d.twists),
    angle = Math.max(m.firstContactAngle, m.entryAngle),
    g = m.entryGeometryWorst,
    position = m.positionQualities["ABCD".indexOf(d.position)],
    failures = [];
  for (const [bad, reason] of [
    [m.boardInvalid, "Invalid takeoff or platform contact"],
    [!m.water, "No water entry"],
    [rotationError > 0.25, "Declared somersault count not completed"],
    [twistError >= 0.25, "Declared twist count not completed"],
    [!m.fullEntryComplete, "Entry did not complete"],
    [
      d.headfirst && [12, 15].includes(m.firstGeometry),
      "Feet entered before head or hands",
    ],
    [
      !d.headfirst && ![12, 15].includes(m.firstGeometry),
      "Feet-first dive did not enter feet first",
    ],
    [apparatus === "springboard" && m.preparationBounces > 0, "Double bounce"],
    [m.maxLateral > 0.85 && d.twists === 0, "Wrong rotation plane"],
  ])
    if (bad) failures.push(reason);
  const deductions = {
    takeoff:
      Math.min(1.5, 1.5 * Math.max(0, 1 - m.ascent / 0.3)) +
      Math.min(2, m.preparationBounces),
    position: 2 * (1 - clamp(position, 0, 1)),
    entryAlignment: Math.min(6, angle / 10),
    entryForm: Math.min(2, 2 * (1 - clamp(m.form, 0, 1))),
    feet: Math.min(1, Math.max(...g.footLineAngles) / 45),
    legs: Math.min(
      1,
      Math.max(0, g.ankleGap - 0.13) * 5 + Number(g.crossedLegs),
    ),
    hands: d.headfirst
      ? Math.min(
          1,
          Math.max(0, g.handSeparation - 0.08) * 5 +
            Math.max(0, g.handHeightGap - 0.02) * 10,
        )
      : 0,
    lateralEntry: Math.min(1, m.surfaceLateralSpeed / 3),
    distance: Math.min(2, Math.max(0, 0.6 - m.x) * 4),
  };
  let execution = clamp(10 - sum(Object.values(deductions)), 0, 10);
  if (position < 0.5 || m.x < 0.2) execution = Math.min(execution, 2);
  if (!m.entryArmPositionValid) execution = Math.min(execution, 4.5);
  if (failures.length) execution = 0;
  const difficulty = d.difficulty[`${apparatus}:${height}`];
  return {
    declaration: d.id,
    category: d.group,
    difficulty,
    execution,
    points: 3 * difficulty * execution,
    valid: !failures.length,
    clean:
      !failures.length &&
      angle <= 15 &&
      m.form >= 0.8 &&
      m.entryGeometryValid &&
      m.entryLimbsValid,
    deductions,
    failures,
    rotationError,
    twistError,
    entryAngle: angle,
    judgeVersion: "self-declared-whole-entry-v11",
    measurements: m,
  };
}
export function simulateAutonomous(engine, parameters, policy) {
  engine.resetContext(parameters);
  const mask = legalDeclarations(
    parameters.category,
    parameters.apparatus,
    parameters.height,
    parameters.used,
  );
  const requested = requestedDeclaration(parameters.dive, mask);
  engine.declare(requested ?? policy.predict(engine.observation(), mask).choice);
  for (let i = 0; i < 250; i++) {
    const r = policy.predict(engine.observation(), mask);
    if (engine.step(r.action)) break;
    if (i === 249) throw Error("Physical dive exceeded its integration budget");
  }
  const result = judgeDeclaration(
    engine.declaration,
    parameters.apparatus,
    parameters.height,
    engine.measurements(),
  );
  return {
    parameters,
    skill: engine.declaration,
    result,
    entryTime: engine.entryTime,
    contactPoint: engine.contactPoint,
    frames: engine.frames,
    actions: engine.actions,
    physics: {
      engine: "MuJoCo 3.13.0",
      dt: 0.002,
      controlDt: 0.02,
      contract: "self-declared-diver-v11",
    },
  };
}
