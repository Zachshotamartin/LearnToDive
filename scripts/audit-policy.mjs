/** Independent browser-runtime inference + MuJoCo parity against frozen native cases. */
import { readFile, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import load from "@mujoco/mujoco";
import { PhysicsEngine, scoreEntry } from "../src/core/physics.js";
import { policyAction, validatePolicy, SKILLS } from "../src/core/control.js";
const directory = process.argv[2] || "training/release";
const allCases = process.argv.includes("--all");
const report = JSON.parse(await readFile(`${directory}/evaluation.json`));
const cases = JSON.parse(await readFile(`${directory}/cases.json`));
const bytes = await readFile(`${directory}/policy.json`);
if (createHash("sha256").update(bytes).digest("hex") !== report.policySHA256)
  throw new Error("Frozen policy hash differs from the native evaluation");
const xml = await readFile("public/physics/diver.xml", "utf8");
const xmlHash = createHash("sha256").update(xml).digest("hex");
const model = validatePolicy(JSON.parse(bytes), xmlHash);
const mj = await load(),
  engine = new PhysicsEngine(
    mj,
    xml,
  );
const chosen = [];
for (let skill = 0; skill < 6; skill++) {
  chosen.push(
    ...cases.pretrained.filter((c) => c.skill === skill).slice(0, allCases ? Infinity : 16).map((c) => ({ ...c, evaluationSet: "heldOut" })),
  );
  for (const height of allCases ? [3, 4, 5, 6, 7.5, 9, 10] : [3, 5, 7.5, 9, 10])
    chosen.push(
      ...(cases.heightGrid || [])
        .filter((c) => c.skill === skill && c.height === height)
        .slice(0, allCases ? Infinity : 3).map((c) => ({ ...c, evaluationSet: "heightGrid" })),
    );
}
const summary = {
  contract: model.contract,
  runtimeCatalog: SKILLS.map(({id,minHeight,dd})=>({id,minHeight,difficulty:dd})),
  cases: chosen.length,
  policySHA256: report.policySHA256,
  judge: report.judge,
  physicsSHA256: createHash("sha256").update(await readFile("public/physics/diver.xml")).digest("hex"),
  sourceSHA256: Object.fromEntries(await Promise.all(["src/core/physics.js", "src/core/control.js", "src/core/water.js"].map(async (path) => [path, createHash("sha256").update(await readFile(path)).digest("hex")]))),
  scope: allCases ? "All held-out cases and all height bins" : "Deterministic diagnostic subset",
  maxEntryAngleDifference: 0,
  maxShapeFractionDifference: 0,
  maxContactTimeDifference: 0,
  maxFullEntryTimeDifference:0,
  wholeEntryControlFailures:0,
  fullEntryCompletionDifferences:0,
  maxTorque: 0,
  maxJointSpeed: 0,
  maxJointPower: 0,
  maxBoardImpulseDifference: 0,
  maxStandImpulseDifference: 0,
  invalidLaunchCountedClean: 0,
  validityDifferences: 0,
  executionDifferences: 0,
  mismatches: [],
};
if (summary.physicsSHA256 !== report.physicsSHA256)
  throw new Error("Physics XML differs from the native evaluation");
const runtimeRows = [];
function aggregate(originalRows) {
  const rows=originalRows.map(row=>({...row,result:{...row.result,...(row.result.entryGeometryWorst??{}),geometryLimits:row.result.entryGeometryValid??row.result.geometryLimits}}));
  const quantiles = (values) => {
    const sorted=[...values].sort((a,b)=>a-b);
    return Object.fromEntries([0,.5,.9,.99,1].map(q=>{
      const at=q*(sorted.length-1),lo=Math.floor(at),hi=Math.ceil(at);
      return [q, sorted.length ? sorted[lo]+(sorted[hi]-sorted[lo])*(at-lo) : null];
    }));
  };
  return {
    geometrySummaryScope:'Worst physical geometry across the full scored entry interval; original runtime-case scalar fields remain first-contact telemetry',
    failureCounts: {
      countsMayOverlap: true,
      launch: rows.filter(({result:r})=>r.boardInvalid).length,
      numericRotation: rows.filter(({result:r})=>!r.numericCorrect).length,
      declaredPosition: rows.filter(({result:r})=>!r.shapeCorrect).length,
      entry: rows.filter(({result:r})=>!r.validEntry).length,
      footLine: rows.filter(({result:r})=>Math.max(...r.footLineAngles)>=20).length,
      handAxis: rows.filter(({result:r})=>Math.max(...r.handAxisAngles)>=30).length,
      handSeparation: rows.filter(({result:r})=>r.handSeparation>=.12).length,
      handHeight: rows.filter(({result:r})=>r.handHeightGap>=.04).length,
      legGap: rows.filter(({result:r})=>r.kneeGap>=.19 || r.ankleGap>=.16 || r.toeGap>=.15).length,
      crossedLegs: rows.filter(({result:r})=>r.crossedLegs).length,
      executionBelowSix: rows.filter(({result:r})=>r.execution<6).length,
    },
    geometryPassRate: rows.filter(({result})=>result.geometryLimits).length / rows.length,
    entryGeometry: Object.fromEntries(["footLineAngles","handAxisAngles","handSeparation","handHeightGap","kneeGap","ankleGap","toeGap","ascent","takeoffVerticalSpeed"].map(key=> {
      const values=rows.map(({result})=>Array.isArray(result[key]) ? Math.max(...result[key]) : result[key]);
      return [key,{mean:values.reduce((a,b)=>a+b,0)/values.length,quantiles:quantiles(values)}];
    })),
    episodes: rows.length,
    cleanCompletionRate: rows.filter(({ result }) => result.valid && result.execution >= 6).length / rows.length,
    meanExecution: rows.reduce((sum, { result }) => sum + result.execution, 0) / rows.length,
    meanEntryAngle: rows.reduce((sum, { result }) => sum + result.angle, 0) / rows.length,
    launchClearanceRate: rows.filter(({ result }) => !result.boardInvalid).length / rows.length,
    validityDifferences: rows.filter(({ result, native }) => result.valid !== native.valid).length,
    maxEntryAngleDifference: Math.max(0, ...rows.map(({ result, native }) => Math.abs(result.angle - native.angle))),
    executionHistogram: Object.fromEntries(Array.from({ length: 21 }, (_, i) => [i / 2, rows.filter(({ result }) => result.execution === i / 2).length])),
  };
}
try {
  for (const c of chosen) {
    if (!c.parameters) throw new Error("Evaluation case is missing original input parameters");
    engine.reset(c.parameters);
    for (let i = 0; i < 231; i++) {
      if (
        engine.step(policyAction(model, engine.observation()), {
          record: false,
        })
      )
        break;
    }
    const r = scoreEntry(engine);
    runtimeRows.push({ evaluationSet: c.evaluationSet, parameters: c.parameters, result: r, actionSteps:engine.actions.length,waterActionSteps:engine.actions.filter(a=>a.time>engine.entryTime).length,lastActionTime:engine.actions.at(-1)?.time??null,
      native: { valid: c.valid, execution: c.execution, angle: c.entryAngle, shapeFraction: c.shapeFraction },
    });
    summary.maxEntryAngleDifference = Math.max(
      summary.maxEntryAngleDifference,
      Math.abs(r.angle - c.entryAngle),
    );
    summary.maxShapeFractionDifference = Math.max(
      summary.maxShapeFractionDifference,
      Math.abs(r.shapeFraction - c.shapeFraction),
    );
    summary.maxContactTimeDifference = Math.max(
      summary.maxContactTimeDifference,
      Math.abs((r.contactTime ?? engine.time) - (c.contactTime ?? c.time)),
    );
    summary.fullEntryCompletionDifferences+=Number(r.fullEntryComplete!==c.fullEntryComplete);
    if(r.fullEntryComplete){
      summary.maxFullEntryTimeDifference=Math.max(summary.maxFullEntryTimeDifference,c.fullEntryComplete?Math.abs(r.fullEntryTime-c.fullEntryTime):0);
      if(!(engine.actions.some(a=>a.time>engine.entryTime+.02)&&r.fullEntryTime-engine.actions.at(-1).time<=.020001))summary.wholeEntryControlFailures++;
    }
    summary.maxTorque = Math.max(summary.maxTorque, r.maxTorque);
    summary.maxJointSpeed = Math.max(summary.maxJointSpeed, r.maxJointSpeed);
    summary.maxJointPower = Math.max(summary.maxJointPower, r.maxJointPower);
    summary.maxBoardImpulseDifference = Math.max(
      summary.maxBoardImpulseDifference,
      Math.abs(r.boardNonfootImpulse - c.boardNonfootImpulse),
    );
    summary.maxStandImpulseDifference = Math.max(
      summary.maxStandImpulseDifference,
      Math.abs(r.standImpulse - c.standImpulse),
    );
    summary.invalidLaunchCountedClean += Number(r.valid && r.boardInvalid);
    summary.validityDifferences += Number(r.valid !== c.valid);
    summary.executionDifferences += Number(r.execution !== c.execution);
    if (
      r.boardInvalid !== c.boardInvalid ||
      r.rotatedRecontact !== c.rotatedRecontact ||
      r.footSideContact !== c.footSideContact ||
      r.valid !== c.valid ||
      Math.abs(r.execution - c.execution) > 0.5 ||
      Math.abs(r.angle - c.entryAngle) > 0.5 ||
      Math.abs((r.contactTime ?? engine.time) - (c.contactTime ?? c.time)) > 0.004
    )
      summary.mismatches.push({
        skill: c.skill,
        height: c.height,
        parameters: c.parameters,
        native: {
          valid: c.valid,
          execution: c.execution,
          angle: c.entryAngle,
          contactTime:c.contactTime,fullEntryTime:c.fullEntryTime,fullEntryComplete:c.fullEntryComplete,
          shapeFraction: c.shapeFraction,
          twistError: c.twistError,
        },
        wasm: {
          valid: r.valid,
          execution: r.execution,
          angle: r.angle,
          contactTime:r.contactTime,fullEntryTime:r.fullEntryTime,fullEntryComplete:r.fullEntryComplete,
          shapeFraction: r.shapeFraction,
          twistError: r.twistError,
        },
      });
  }
  if (summary.maxTorque > 180 + 1e-7)
    throw new Error("An actuator exceeded its physical force limit");
  summary.perSkillHeldOut = Object.fromEntries(SKILLS.map((skill, index) => [skill.id,
    aggregate(runtimeRows.filter((row) => row.evaluationSet === "heldOut" && row.parameters.skill === index)),
  ]));
  summary.heightGrid = Object.fromEntries([...new Set(runtimeRows.filter((row) => row.evaluationSet === "heightGrid").map((row) => row.parameters.height))].map((height) => [height,
    Object.fromEntries(SKILLS.map((skill, index) => [skill.id, aggregate(runtimeRows.filter((row) => row.evaluationSet === "heightGrid" && row.parameters.skill === index && row.parameters.height === height))])),
  ]));
  await writeFile(`${directory}/runtime-cases.json`, JSON.stringify(runtimeRows) + "\n");
  await writeFile(
    `${directory}/runtime-audit.json`,
    JSON.stringify(summary, null, 2) + "\n",
  );
  console.log(JSON.stringify(summary, null, 2));
  if (summary.mismatches.length) process.exitCode = 1;
} finally {
  engine.dispose();
}
