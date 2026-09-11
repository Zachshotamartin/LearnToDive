import { bodyWater, immersion, WATER_VERSION } from './water.js';
import {
  DT,
  SUBSTEPS,
  PHYSICS_DT,
  SKILLS,
  MAP,
  JOINT_NAMES,
  SITE_NAMES,
  BODY_NAMES,
  clamp,
  wrap,
  up,
  framedAngles,
  parameters,
  targetsFromAction,
  observe,
  policyAction,
  baselineAction,
  firstWaterGeometry,
  postureProgress,
  entryPosture,
  entryGeometry,
  contactForces,
  tuckGeometry,
} from "./control.js";
const values = (array, start, length) =>
  Array.from(array.subarray(start, start + length));
const renderPoint = (p, height) => [p[0], p[2] + height, -p[1]];
const renderVector = (p) => [p[0], p[2], -p[1]];
const renderQuat = (q) => [q[1], q[3], -q[2], q[0]];

/** The same MuJoCo MJCF and integration steps used by native PPO training. */
export class PhysicsEngine {
  constructor(mj, xml) {
    this.mj = mj;
    this.model = mj.MjModel.from_xml_string(xml);
    this.data = new mj.MjData(this.model);
    this.velocityBuffer = new mj.DoubleBuffer(6);
    const m = this.model;
    this.qadr = JOINT_NAMES.map(
      (name) => m.jnt_qposadr[mj.mj_name2id(m, 3, name)],
    );
    this.vadr = JOINT_NAMES.map(
      (name) => m.jnt_dofadr[mj.mj_name2id(m, 3, name)],
    );
    this.sites = Object.fromEntries(
      SITE_NAMES.map((name) => [name, mj.mj_name2id(m, 6, name)]),
    );
    this.bodies = Object.fromEntries(
      BODY_NAMES.map((name) => [name, mj.mj_name2id(m, 1, name)]),
    );
    this.geometry = Array.from({ length: 15 }, (_, i) => ({
      id: i + 1,
      type: m.geom_type[i + 1],
      size: values(m.geom_size, (i + 1) * 3, 3),
      body: m.geom_bodyid[i + 1],
    }));
  }
  reset(raw = {}) {
    this.parameters = parameters(raw);
    const p = this.parameters,
      m = this.model,
      d = this.data,
      mj = this.mj;
    mj.mj_resetData(m, d);
    d.mocap_pos.set([0, 0, -p.height]);
    d.mocap_quat.set([1, 0, 0, 0]);
    const yaw = Math.PI * SKILLS[p.skill].back,
      t = p.tilt;
    d.qpos[0] = p.preload;
    d.qpos.set(
      [
        Math.cos(t / 2) * Math.cos(yaw / 2),
        Math.sin(t / 2) * Math.sin(yaw / 2),
        Math.sin(t / 2) * Math.cos(yaw / 2),
        Math.cos(t / 2) * Math.sin(yaw / 2),
      ],
      4,
    );
    this.targets = [0.5, 1, 0.5, 0.3, 0.3, 0, 0, 0, 0];
    for (let i = 0; i < this.qadr.length; i++) {
      d.qpos[this.qadr[i]] = this.targets[MAP[i]];
      d.ctrl[i] = this.targets[MAP[i]];
    }
    mj.mj_forward(m, d);
    const feet = [this.bodies.foot_L, this.bodies.foot_R].map(
      (b) => m.body_geomadr[b],
    );
    let bottom = Infinity,
      cx = 0;
    for (const g of feet) {
      let extent = 0;
      for (let i = 0; i < 3; i++)
        extent += Math.abs(d.geom_xmat[g * 9 + 6 + i]) * m.geom_size[g * 3 + i];
      bottom = Math.min(bottom, d.geom_xpos[g * 3 + 2] - extent);
      cx += d.geom_xpos[g * 3];
    }
    d.qpos[3] += p.preload + 0.055 - bottom - 0.0001;
    d.qpos[1] += p.x - cx / 2;
    mj.mj_forward(m, d);
    this.phaseTheta = 0;
    this.departurePitch = 0;
    this.departureCOM = 0;
    this.apexCOM = 0;
    this.takeoffVerticalSpeed = 0;
    this.lastFootContact = [0, 0];
    this.footDepartureGap = 0;
    this.takeoffTiltInvalid = false;
    this.airClearTime = 0;
    this.departureTheta = 0;
    this.departureTwist = 0;
    this.airTheta = 0;
    this.airTwist = 0;
    this.releaseTheta = 0;
    this.boardInvalid = false;
    this.boardNonfootImpulse = 0;
    this.boardNonfootPeak = 0;
    this.standImpulse = 0;
    this.standPeak = 0;
    this.footSideContact = false;
    this.rotatedRecontact = false;
    this.preparationBounces = 0;
    this.theta = 0;
    this.twist = 0;
    [this.prevPitch, this.prevTwist] = framedAngles(values(d.qpos, 4, 4));
    this.maxLateral = 0;
    this.shapeDuration = 0;
    this.shapeAngle = 0;
    this.maxProgress = 0;
    this.shapePeak = 0;
    this.shapeQuality = 0;
    this.shapeTicks = 0;
    this.boardTwist = 0;
    this.released = false;
    this.releaseTime = null;
    this.releaseTwist = 0;
    this.entryTime = null;
    this.fullEntryTime = null;
    this.firstContactScore = null;
    this.entryMaxAngle = 0;
    this.entryMinForm = 1;
    this.entryLimbsValid = true;
    this.entryGeometryValid = true;
    this.waterFraction = 0;
    this.aboveWater = 0;
    this.entrySamples = 0;
    this.entryWorstGeometry={footLineAngles:[0,0],handAxisAngles:[0,0],handSeparation:0,handHeightGap:0,kneeGap:0,ankleGap:0,toeGap:0,crossedLegs:false};
    this.contactPoint = null;
    this.firstGeometry = null;
    this.maxJointSpeed = 0;
    this.maxTorque = 0;
    this.maxPower = 0;
    this.appliedImpulse = 0;
    this.pushDuration = 0;
    this.steps = 0;
    this.actions = [];
    this.frames = [];
    this.result = null;
    this.read();
    this.waterState();
    this.frames.push(this.frame());
    return this;
  }
  read() {
    const d = this.data;
    this.q = Array.from(d.qpos);
    this.v = Array.from(d.qvel);
    this.s = Array.from(d.sensordata);
    this.time = d.time;
    return this;
  }
  observation() {
    return observe(this);
  }
  state() {
    return [this.data.time, ...this.data.qpos, ...this.data.qvel];
  }
  restore(state) {
    this.mj.mj_setState(this.model, this.data, state, 15);
    this.mj.mj_forward(this.model, this.data);
    this.read();
  }
  waterState(apply = false) {
    const m=this.model,d=this.data,h=this.parameters.height;
    if(apply)d.xfrc_applied.fill(0);
    let wetMass=0,above=-Infinity,allSubmerged=true;
    for(const g of this.geometry){
      const position=values(d.geom_xpos,g.id*3,3),quaternion=values(d.sensordata,45+(g.id-1)*7+3,4);
      const wet=immersion(g.type,g.size,position,quaternion,-h);
      allSubmerged&&=wet.fraction===1;wetMass+=wet.fraction*m.body_mass[g.body];above=Math.max(above,position[2]+wet.extent+h);
      if(apply&&wet.fraction>0){
        this.mj.mj_objectVelocity(m,d,1,g.body,this.velocityBuffer,0);
        const v=this.velocityBuffer.GetView();
        const force=bodyWater({type:g.type,size:g.size,position,quaternion,com:values(d.xipos,g.body*3,3),linear:values(v,3,3),angular:values(v,0,3),mass:m.body_mass[g.body],waterZ:-h});
        d.xfrc_applied.set([...force.force,...force.torque],g.body*6);
      }
    }
    this.waterFraction=wetMass/70;this.aboveWater=above;this.allSubmerged=allSubmerged;
    return above;
  }
  applyWater(){ this.waterState(true); }
  trackEntry(){
    const angle=Math.acos(clamp(-up(this.q.slice(4,8))[2],-1,1))*180/Math.PI;
    const geometry=entryGeometry(this.s),form=entryPosture(this.qadr.map(i=>this.q[i]),geometry);
    for(const key of ['footLineAngles','handAxisAngles'])this.entryWorstGeometry[key]=this.entryWorstGeometry[key].map((v,i)=>Math.max(v,geometry[key][i]));
    for(const key of ['handSeparation','handHeightGap','kneeGap','ankleGap','toeGap'])this.entryWorstGeometry[key]=Math.max(this.entryWorstGeometry[key],geometry[key]);
    this.entryWorstGeometry.crossedLegs||=geometry.crossedLegs;
    this.entryMaxAngle=Math.max(this.entryMaxAngle,angle);
    this.entryMinForm=Math.min(this.entryMinForm,form.form);
    this.entryLimbsValid&&=form.limbLimits;this.entryGeometryValid&&=form.geometryLimits;this.entrySamples++;
    if(this.allSubmerged && this.fullEntryTime===null)this.fullEntryTime=this.time;
  }
  step(action, { record = true, water = false } = {}) {
    const mj = this.mj,
      m = this.model,
      d = this.data,
      p = this.parameters;
    if ((this.fullEntryTime !== null || (this.entryTime !== null && this.time-this.entryTime>=.8-1e-9)) && !water) return true;
    const startedInWater = this.entryTime !== null;
    if (!water) {
      this.targets = targetsFromAction(action, this.targets);
      for (let i = 0; i < this.qadr.length; i++) d.ctrl[i] = this.targets[MAP[i]];
      this.actions.push({
        time: this.time,
        values: Array.from(action),
        targets: [...this.targets],
      });
    }
    d.xfrc_applied.fill(0);
    if (
      !startedInWater &&
      this.released &&
      this.time >= p.disturbanceTime - 1e-8 &&
      this.time < p.disturbanceTime + 0.12 - 1e-8
    )
      d.xfrc_applied[this.bodies.torso * 6] = p.disturbance;
    const pushForce = d.xfrc_applied[this.bodies.torso * 6];
    for (let k = 0; k < SUBSTEPS; k++) {
      if (startedInWater) {
        mj.mj_forward(m, d);
        this.applyWater();
      }
      const before = this.state();
      mj.mj_step(m, d);
      this.read();
      const contact = firstWaterGeometry(this.s, this.geometry, p.height);
      if (!startedInWater && contact.contact) {
        this.restore(before);
        this.entryTime = this.time;
        this.contactPoint = renderPoint(contact.point, p.height);
        this.contactPoint[1] = 0;
        this.firstGeometry = contact.index + 1;
      }
      if (!startedInWater) {
        const [a, b, lateral] = framedAngles(this.q.slice(4, 8));
        this.theta += wrap(a - this.prevPitch);
        this.twist += wrap(b - this.prevTwist);
        this.prevPitch = a;
        this.prevTwist = b;
        const duration = this.time - before[0];
        this.trackLaunch(duration, lateral);
        const progress = postureProgress(
          this.maxProgress,
          this.airTheta,
          SKILLS[p.skill],
          this.qadr.map((i) => this.q[i]),
          this.released,
          tuckGeometry(this.s),
        );
        this.maxProgress = progress.maximum;
        this.shapeAngle += progress.angle;
      }
      for (let i = 0; i < this.qadr.length; i++) {
        this.maxJointSpeed = Math.max(
          this.maxJointSpeed,
          Math.abs(this.v[this.vadr[i]]),
        );
        this.maxTorque = Math.max(
          this.maxTorque,
          Math.abs(d.actuator_force[i]),
        );
        this.maxPower = Math.max(
          this.maxPower,
          Math.abs(d.actuator_force[i] * this.v[this.vadr[i]]),
        );
      }
      if (!startedInWater && pushForce) {
        this.appliedImpulse += pushForce * (this.time - before[0]);
        this.pushDuration += this.time - before[0];
      }
      if (this.entryTime !== null && !startedInWater) {
        this.firstContactScore = scoreContact(this);
        this.waterState();this.trackEntry();break;
      }
      if(startedInWater&&!water){
        // Score the current physical pose, not the sensor cache from before mj_step.
        mj.mj_forward(m,d);this.read();this.waterState();this.trackEntry();
        if(this.fullEntryTime!==null||this.time-this.entryTime>=.8-1e-9)break;
      }
    }
    if (!startedInWater) {
      const g = SKILLS[p.skill],
        hip = this.q[this.qadr[0]],
        knee = this.q[this.qadr[1]],
        h = Math.max(this.s[2] + p.height - 0.7, 0),
        tgo = clamp(
          (this.s[5] + Math.sqrt(this.s[5] ** 2 + 19.62 * h)) / 9.81,
          0.08,
          3,
        ),
        shape =
          (g.shape === 1
            ? Math.exp(-(((hip - 1.5) / 0.6) ** 2) - (knee / 0.5) ** 2)
            : Math.exp(
                -(((hip - 1.4) / 0.65) ** 2) - ((knee - 2) / 0.7) ** 2,
              )) *
          (g.shape === 0
            ? Math.exp(
                -((Math.max(...tuckGeometry(this.s).distances) / 0.45) ** 2),
              )
            : 1);
      if (this.released && tgo > 0.45) {
        this.shapeQuality += shape;
        this.shapeTicks++;
        this.shapePeak = Math.max(this.shapePeak, shape);
        if (shape > 0.6) this.shapeDuration += DT;
      }
    }
    if(!startedInWater)this.waterState();
    this.steps++;
    if (record) this.frames.push(this.frame());
    return (
      this.fullEntryTime !== null ||
      (this.entryTime !== null && this.time-this.entryTime>=.8-1e-9) ||
      this.time >= 4.4 ||
      this.q.some((x) => !Number.isFinite(x))
    );
  }
  trackLaunch(duration, lateral) {
    if (duration <= 0) return;
    const forces = contactForces(this.s),
      support = contactForces(this.s, 210);
    const nonfoot = forces.filter((_, i) => i !== 11 && i !== 14);
    this.boardNonfootImpulse += nonfoot.reduce((a, b) => a + b, 0) * duration;
    this.boardNonfootPeak = Math.max(this.boardNonfootPeak, ...nonfoot);
    this.standImpulse += support.reduce((a, b) => a + b, 0) * duration;
    this.standPeak = Math.max(this.standPeak, ...support);
    this.footSideContact ||= [11, 14].some(
      (i) => forces[i] > 15 && this.s[45 + i * 7 + 2] < this.q[0] + 0.04,
    );
    const grounded = forces.some((_, i) => this.s[150 + i * 4] > 0);
    for (let i = 0; i < 2; i++)
      if (this.s[[194, 206][i]] > 0) this.lastFootContact[i] = this.time;
    const pitch = framedAngles(this.q.slice(4, 8))[0];
    this.takeoffTiltInvalid ||= grounded && Math.abs(pitch) > Math.PI / 3;
    const recontact = this.released && grounded;
    if (recontact) this.preparationBounces++;
    this.rotatedRecontact ||= recontact && up(this.q.slice(4, 8))[2] < 0.87;
    this.boardInvalid ||=
      this.boardNonfootImpulse > 0.1 ||
      this.boardNonfootPeak > 15 ||
      this.standImpulse > 0.1 ||
      this.standPeak > 15 ||
      this.footSideContact ||
      this.rotatedRecontact;
    if (grounded) {
      this.airClearTime = 0;
      this.departureCOM = this.s[2];
      this.apexCOM = this.s[2];
      this.boardTwist = Math.max(this.boardTwist, Math.abs(this.twist));
      this.released = false;
      this.shapeAngle = 0;
      this.maxProgress = 0;
      this.maxLateral = 0;
      this.shapeDuration = 0;
      this.shapeQuality = 0;
      this.shapeTicks = 0;
      this.shapePeak = 0;
    } else {
      if (this.airClearTime === 0) {
        this.departurePitch = pitch;
        this.departureCOM = this.s[2];
        this.apexCOM = this.s[2];
        this.takeoffVerticalSpeed = this.s[5];
        this.departureTheta = this.theta;
        this.departureTwist = this.twist;
      }
      this.apexCOM = Math.max(this.apexCOM, this.s[2]);
      this.airClearTime += duration;
      if (
        !this.released &&
        this.time > 0.08 &&
        this.airClearTime >= 0.04 - 1e-8
      ) {
        this.released = true;
        this.releaseTime = this.time - this.airClearTime + PHYSICS_DT;
        this.releaseTheta = this.departureTheta;
        this.releaseTwist = this.departureTwist;
        this.footDepartureGap = Math.abs(
          this.lastFootContact[0] - this.lastFootContact[1],
        );
        this.takeoffTiltInvalid ||= Math.abs(this.departurePitch) > Math.PI / 3;
        this.boardInvalid ||= this.footDepartureGap > 0.04 + 1e-8;
      }
    }
    this.boardInvalid ||= this.takeoffTiltInvalid;
    this.airTheta = this.released ? this.theta - this.releaseTheta : 0;
    this.phaseTheta = this.released ? this.airTheta + this.departurePitch : 0;
    this.airTwist = this.released ? this.twist - this.releaseTwist : 0;
    this.maxLateral = Math.max(this.maxLateral, this.released ? lateral : 0);
  }
  frame() {
    const mj = this.mj,
      m = this.model,
      d = this.data,
      h = this.parameters.height;
    const cached = this.s;
    mj.mj_forward(m, d);
    const pose = Object.fromEntries(
      Object.entries(this.sites).map(([name, id]) => [
        name,
        renderPoint(values(d.site_xpos, id * 3, 3), h),
      ]),
    );
    const bodies = Object.fromEntries(
      Object.entries(this.bodies).map(([name, id]) => [
        name,
        {
          position: renderPoint(values(d.xpos, id * 3, 3), h),
          quaternion: renderQuat(values(d.xquat, id * 4, 4)),
        },
      ]),
    );
    const com = renderPoint(
        values(d.subtree_com, this.bodies.pelvis * 3, 3),
        h,
      ),
      velocity = renderVector(
        values(d.subtree_linvel, this.bodies.pelvis * 3, 3),
      ),
      L = renderVector(values(d.subtree_angmom, this.bodies.pelvis * 3, 3));
    this.s = cached;
    return {
      time: this.time,
      x: com[0],
      y: com[1],
      z: com[2],
      vx: velocity[0],
      vy: velocity[1],
      vz: velocity[2],
      q: bodies.pelvis.quaternion,
      pose,
      bodies,
      L,
      board: bodies.board,
      boardDeflection: this.q[0],
      phase:
        this.entryTime !== null ? "water" : this.released ? "flight" : "board",
      released: this.released,
      releaseTime: this.releaseTime,
      entryTime: this.entryTime,
      fullEntryTime: this.fullEntryTime,
      waterFraction: this.waterFraction,
      aboveWater: this.aboveWater,
      contactPoint: this.contactPoint,
      joints: this.qadr.map((i) => this.q[i]),
      targets: [...this.targets],
      handToShin: tuckGeometry(cached).distances,
      tuck: clamp(this.q[this.qadr[1]] / 2.6, 0, 1),
      pike: clamp(this.q[this.qadr[0]] / 2.2, 0, 1),
      somersaults: this.phaseTheta / (2 * Math.PI),
      airborneSomersaults: this.airTheta / (2 * Math.PI),
      takeoffLean: this.departurePitch,
      twists: this.airTwist / (2 * Math.PI),
    };
  }
  dispose() {
    this.velocityBuffer.delete();
    this.data.delete();
    this.model.delete();
  }
}

function scoreContact(engine) {
  const g = SKILLS[engine.parameters.skill],
    q = engine.q,
    s = engine.s,
    j = engine.qadr.map((i) => q[i]),
    u = up(q.slice(4, 8)),
    angle = (Math.acos(clamp(-u[2], -1, 1)) * 180) / Math.PI,
    flips = engine.phaseTheta / (2 * Math.PI),
    twists = engine.airTwist / (2 * Math.PI),
    rotationError = Math.abs(engine.phaseTheta - g.turns * 2 * Math.PI),
    twistError = Math.abs(engine.airTwist - g.twists * 2 * Math.PI),
    airTwistError = Math.abs(engine.airTwist - g.twists * 2 * Math.PI),
    arms = (j[6] + j[9]) * 0.5;
  const geometry = entryGeometry(s), { handSeparation } = geometry;
  const { form, limbLimits, geometryLimits } = entryPosture(j, geometry);
  const clearance = clamp((s[0] - 0.2) / 0.7, 0, 1),
    shapeFraction = clamp(
      engine.shapeAngle / (0.55 * g.turns * 2 * Math.PI),
      0,
      1,
    ),
    shapeCorrect = g.shape === 2 || shapeFraction >= 0.65,
    numericCorrect =
      rotationError < 0.45 &&
      twistError < 0.55 &&
      airTwistError < 0.55 &&
      engine.maxLateral < 0.85,
    validEntry =
      engine.entryTime !== null &&
      angle < 35 &&
      form > 0.55 &&
      limbLimits &&
      geometryLimits &&
      [6, 9].includes(engine.firstGeometry) &&
      clearance > 0.5;
  const splash = clamp(
      0.55 * Math.sin((angle * Math.PI) / 180) ** 2 +
        0.25 * (1 - form) +
        (0.2 * Math.hypot(s[3], s[4])) / 3,
      0,
      1,
    ),
    deductions = {
      height: 1.5 * (1 - clamp((engine.apexCOM - engine.departureCOM) / 0.3, 0, 1)),
      form: clamp((1 - form) * 3, 0, 3),
      entry: clamp(angle / 10, 0, 4),
      splash: splash * 1.5,
      preparation: Math.min(2, engine.preparationBounces || 0),
      takeoff: engine.boardTwist > 0.26 ? clamp(engine.boardTwist, 0.5, 2) : 0,
    };
  let execution = clamp(
    10 - Object.values(deductions).reduce((a, b) => a + b, 0),
    0,
    10,
  );
  if (!shapeCorrect) execution = Math.min(execution, 2);
  if (!validEntry) execution = Math.min(execution, 2);
  if (!numericCorrect || engine.entryTime === null || engine.boardInvalid)
    execution = 0;
  execution = Math.round(execution * 2) / 2;
  const valid =
    !engine.boardInvalid && numericCorrect && validEntry && shapeCorrect;
  return {
    valid,
    validEntry,
    numericCorrect,
    shapeCorrect,
    outcome: engine.boardInvalid
      ? "Board or stand clearance failed"
      : valid
        ? "Declared dive completed"
        : !numericCorrect
          ? "Declared rotation incomplete"
          : !shapeCorrect
            ? "Declared body position missed"
            : "Entry form failed",
    flips,
    airborneFlips: engine.airTheta / (2 * Math.PI),
    takeoffLean: engine.departurePitch,
    footDepartureGap: engine.footDepartureGap,
    takeoffTiltInvalid: engine.takeoffTiltInvalid,
    twists,
    angle,
    splash,
    execution,
    difficulty: g.dd,
    total: 3 * execution * g.dd,
    breakdown: deductions,
    rotationError,
    twistError,
    airTwistError,
    shapeDuration: engine.shapeDuration,
    shapeFraction,
    shapePeak: engine.shapePeak,
    form,
    limbLimits,
    handSeparation,
    handHeightGap: geometry.handHeightGap,
    kneeGap: geometry.kneeGap,
    ankleGap: geometry.ankleGap,
    toeGap: geometry.toeGap,
    crossedLegs: geometry.crossedLegs,
    signedLegGaps: geometry.signedLegGaps,
    ascent: Math.max(0, engine.apexCOM - engine.departureCOM),
    takeoffVerticalSpeed: engine.takeoffVerticalSpeed,
    footLineAngles: geometry.footLineAngles.map(a => a * 180 / Math.PI),
    handAxisAngles: geometry.handAxisAngles.map(a => a * 180 / Math.PI),
    geometryLimits,
    entryJoints: j,
    maxLateral: engine.maxLateral,
    boardTwist: engine.boardTwist,
    launchValid: !engine.boardInvalid,
    boardInvalid: engine.boardInvalid,
    boardNonfootImpulse: engine.boardNonfootImpulse,
    boardNonfootPeak: engine.boardNonfootPeak,
    standImpulse: engine.standImpulse,
    standPeak: engine.standPeak,
    footSideContact: engine.footSideContact,
    rotatedRecontact: engine.rotatedRecontact,
    preparationBounces: engine.preparationBounces,
    releaseTime: engine.releaseTime,
    firstGeometry: engine.firstGeometry,
    maxJointSpeed: engine.maxJointSpeed,
    maxTorque: engine.maxTorque,
    maxJointPower: engine.maxPower,
    appliedImpulse: engine.appliedImpulse,
    pushDuration: engine.pushDuration,
    contactTime: engine.entryTime,
  };
}

export function scoreEntry(engine){
 const first=engine.firstContactScore??scoreContact(engine);
 const complete=Number.isFinite(engine.fullEntryTime);
 const angle=Math.max(first.angle,engine.entryMaxAngle??0),form=Math.min(first.form,engine.entryMinForm??1);
 const validEntry=first.validEntry&&complete&&angle<35&&form>.55&&engine.entryLimbsValid&&engine.entryGeometryValid;
 const splash=clamp(.55*Math.sin(angle*Math.PI/180)**2+.25*(1-form)+.2*Math.hypot(engine.s[3],engine.s[4])/3,0,1);
 const breakdown={...first.breakdown,form:clamp((1-form)*3,0,3),entry:clamp(angle/10,0,4),splash:splash*1.5};
 let execution=clamp(10-Object.values(breakdown).reduce((a,b)=>a+b,0),0,10);
 if(!first.shapeCorrect||!validEntry)execution=Math.min(execution,2);
 if(!first.numericCorrect||engine.entryTime===null||engine.boardInvalid)execution=0;
 execution=Math.round(execution*2)/2;
 const valid=first.numericCorrect&&first.shapeCorrect&&validEntry&&!engine.boardInvalid;
 return {...first,valid,validEntry,execution,maxTorque:engine.maxTorque,maxJointSpeed:engine.maxJointSpeed,maxJointPower:engine.maxPower,total:3*execution*first.difficulty,breakdown,angle,form,splash,
   firstContactAngle:first.angle,firstContactForm:first.form,entryGeometryWorst:engine.entryWorstGeometry?{...engine.entryWorstGeometry,footLineAngles:engine.entryWorstGeometry.footLineAngles.map(a=>a*180/Math.PI),handAxisAngles:engine.entryWorstGeometry.handAxisAngles.map(a=>a*180/Math.PI)}:null,fullEntryTime:engine.fullEntryTime,
   entryDuration:engine.entryTime===null?null:(engine.fullEntryTime??engine.time)-engine.entryTime,
   fullEntryComplete:complete,entrySamples:engine.entrySamples,entryGeometryValid:engine.entryGeometryValid,
   entryLimbsValid:engine.entryLimbsValid,outcome:valid?'Declared dive completed':first.valid?'Whole entry form failed':first.outcome};
}

export function simulate(
  engine,
  raw,
  policy,
  model,
  { waterSeconds = 1.15, openLoop = null } = {},
) {
  engine.reset(raw);
  let i = 0;
  while (engine.time < 4.4 && engine.fullEntryTime === null && (engine.entryTime === null ? engine.time<3.6 : engine.time-engine.entryTime<.8-1e-9)) {
    const action =
      (openLoop?.length
        ? openLoop[Math.min(i, openLoop.length - 1)].values
        : null) ??
      (policy === "baseline"
        ? baselineAction(engine)
        : policyAction(model, engine.observation()));
    engine.step(action);
    i++;
    if (i > 230)
      throw new Error("Simulation exceeded its bounded control budget.");
  }
  // Reclassify earlier upright hops as preparation once the final departure is
  // known. Flight telemetry never credits a preceding bounce's rotation.
  for (const frame of engine.frames) {
    if (frame.time < engine.releaseTime) {
      frame.phase = "board";
      frame.somersaults = 0;
      frame.twists = 0;
    }
  }
  const result = scoreEntry(engine),
    entryTime = engine.entryTime,
    contactPoint = engine.contactPoint;
  const end = engine.time + waterSeconds;
  if (entryTime !== null)
    while (engine.time < end) engine.step([], { water: true });
  return {
    parameters: engine.parameters,
    skill: SKILLS[engine.parameters.skill],
    policy,
    result,
    entryTime,
    contactPoint,
    frames: engine.frames,
    actions: engine.actions,
    physics: { engine: "MuJoCo 3.13.0", dt: PHYSICS_DT, controlDt: DT, waterLaw: WATER_VERSION },
  };
}
