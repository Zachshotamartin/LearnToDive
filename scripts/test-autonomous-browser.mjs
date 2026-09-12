import assert from "node:assert/strict";
import fs from "node:fs/promises";
import { spawn } from "node:child_process";
import { chromium } from "@playwright/test";
const port = 5198,
  url = `http://127.0.0.1:${port}`,
  server = spawn(
    process.execPath,
    [
      "node_modules/vite/bin/vite.js",
      "--host",
      "127.0.0.1",
      "--port",
      String(port),
      "--strictPort",
    ],
    { stdio: "ignore" },
  );
let browser;
try {
  for (let i = 0; i < 150; i++) {
    if (
      await fetch(url)
        .then((r) => r.ok)
        .catch(() => false)
    )
      break;
    await new Promise((r) => setTimeout(r, 100));
  }
  browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  await page.goto(url);
  const result = await page.evaluate(async () => {
    const [
      { default: load },
      { AutonomousEngine, judgeDeclaration, legalDeclarations },
      { createAutonomousPolicy },
      { ASSETS },
    ] = await Promise.all([
      import("/src/vendor/mujoco-csp.js"),
      import("/src/core/autonomousPhysics.js"),
      import("/src/core/autonomousPolicy.js"),
      import("/src/data/autonomousAssets.js"),
    ]);
    const wasm = (await import("/node_modules/@mujoco/mujoco/mujoco.wasm?url"))
      .default;
    const mj = await load({
        locateFile: (p) => (p.endsWith(".wasm") ? wasm : p),
      }),
      xml = await fetch(ASSETS.xml).then((r) => r.text()),
      model = await fetch(ASSETS.policy).then((r) => r.json()),
      fixture = await fetch("/tests/fixtures/autonomous-native.json").then(
        (r) => r.json(),
      );
    const engine = new AutonomousEngine(mj, xml),
      policy = createAutonomousPolicy(model),
      reports = [];
    const error = (a, b) => Math.max(...a.map((x, i) => Math.abs(x - b[i])));
    for (const c of fixture.cases) {
      engine.resetContext(c.parameters);
      let obsError = error(engine.observation(), c.observation),
        stateError = 0,
        actionError = 0,
        worst = null;
      const mask = legalDeclarations(
          c.parameters.category,
          c.parameters.apparatus,
          c.parameters.height,
          [],
        ),
        choice = policy.predict(engine.observation(), mask).choice;
      if (choice !== c.choice)
        throw Error(`Declaration mismatch ${choice} vs ${c.choice}`);
      engine.declare(choice);
      for (let i = 0; i < c.rows.length; i++) {
        const row = c.rows[i],
          o = engine.observation();
        obsError = Math.max(obsError, error(o, row.observation));
        actionError = Math.max(
          actionError,
          error(policy.predict(row.observation, mask).action, row.action),
        );
        engine.step(row.action, { record: false });
        const e = error(engine.state(), row.state);
        if (e > stateError) {
          stateError = e;
          worst = { i, time: engine.time, e };
        }
      }
      const score = judgeDeclaration(
        engine.declaration,
        c.parameters.apparatus,
        c.parameters.height,
        engine.measurements(),
      );
      reports.push({
        category: c.parameters.category,
        choice,
        obsError,
        stateError,
        actionError,
        worst,
        score: {
          execution: score.execution,
          valid: score.valid,
          angle: score.entryAngle,
        },
        native: {
          execution: c.result.execution,
          valid: c.result.valid,
          angle: c.result.entryAngle,
        },
      });
    }
    engine.dispose();
    return reports;
  });
  console.log(JSON.stringify(result, null, 2));
  await fs.mkdir("output/browser-v11", { recursive: true });
  await fs.writeFile(
    "output/browser-v11/parity.json",
    JSON.stringify(result, null, 2),
  );
  for (const row of result) {
    assert(row.actionError < 1e-5, "policy parity");
    assert(row.stateError < 0.002, "physics parity");
    assert(row.obsError < 0.002, "observation parity");
    assert.equal(row.score.valid, row.native.valid);
    assert(
      Math.abs(row.score.execution - row.native.execution) < 0.01,
      "judge parity",
    );
  }
} finally {
  await browser?.close();
  server.kill();
}
