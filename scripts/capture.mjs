/** Reproducible focused captures from the actual local physics worker and renderer. */
import { chromium } from "@playwright/test";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
const origin = "http://127.0.0.1:5184";
let server;
try {
  await fetch(origin);
} catch {
  server = spawn(
    process.execPath,
    ["node_modules/vite/bin/vite.js", "--host", "127.0.0.1", "--port", "5184"],
    { stdio: "ignore" },
  );
  for (let i = 0; i < 80; i++) {
    await new Promise((r) => setTimeout(r, 100));
    try {
      await fetch(origin);
      break;
    } catch {}
  }
}
const browser = await chromium.launch({
  args: [
    "--use-gl=angle",
    "--use-angle=swiftshader",
    "--enable-unsafe-swiftshader",
  ],
});
try {
  const page = await browser.newPage({
    viewport: { width: 1600, height: 1100 },
    deviceScaleFactor: 1,
    reducedMotion: "reduce",
  });
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.route("**/capture-harness", (route) =>
    route.fulfill({
      contentType: "text/html",
      body: '<!doctype html><html><head><style>html,body{margin:0;width:100%;height:100%;background:transparent}#scene{width:1600px;height:1100px}canvas{display:block}</style></head><body><div id="scene"></div></body></html>',
    }),
  );
  await page.goto(`${origin}/capture-harness`);
  await page.evaluate(async () => {
    const [{ createScene }, { default: WorkerClass }, { assetURLs }] =
      await Promise.all([
        import("/src/scene.js"),
        import("/src/sim.worker.js?worker"),
        import("/src/data/assets.js"),
      ]);
    window.captureScene = createScene(document.getElementById("scene"));
    await window.captureScene.ready;
    window.physicsWorker = new WorkerClass();
    window.captureRequest = 0;
    window.simulateCapture = (parameters) =>
      new Promise((resolve, reject) => {
        const id = ++window.captureRequest;
        window.physicsWorker.onmessage = ({ data }) => {
          if (data.id === id)
            data.error ? reject(new Error(data.error)) : resolve(data.dive);
        };
        window.physicsWorker.onerror = (e) => reject(new Error(e.message));
        window.physicsWorker.postMessage({
          id,
          parameters,
          policy: "pretrained",
          assets: assetURLs(),
        });
      });
  });
  await mkdir("examples", { recursive: true });
  const images = [];
  for (const spec of [
    {
      file: "pretrained-flight.png",
      skill: 2,
      height: 7.5,
      frame: "pike",
      camera: "athlete",
    },
    {
      file: "head-first-entry.png",
      skill: 1,
      height: 7.5,
      frame: "entry",
      camera: "entry",
    },
    {
      file: "twisting-flight.png",
      skill: 5,
      height: 10,
      frame: "twist",
      camera: "athlete",
    },
  ]) {
    const shot = await page.evaluate(async (spec) => {
      const parameters = {
        skill: spec.skill,
        height: spec.height,
        tilt: 0.16,
        preload: -0.18,
        x: -0.13,
        disturbance: 0,
        disturbanceTime: 0.85,
      };
      const dive = await window.simulateCapture(parameters);
      if (!dive.result.valid || dive.result.execution < 6)
        throw new Error(
          `Capture preset ${dive.skill.id} did not meet the published clean-dive criterion`,
        );
      let frame;
      const airborne = dive.frames.filter(
        (f) =>
          f.released &&
          f.time > dive.frames[0].time + 0.2 &&
          f.time < dive.entryTime - 0.35,
      );
      if (spec.frame === "pike")
        frame = airborne.reduce((a, b) =>
          b.joints[0] - 2 * Math.abs(b.joints[1]) >
          a.joints[0] - 2 * Math.abs(a.joints[1])
            ? b
            : a,
        );
      else if (spec.frame === "twist")
        frame = airborne.reduce((a, b) =>
          Math.abs(b.twists - 0.5) < Math.abs(a.twists - 0.5) ? b : a,
        );
      else
        frame = dive.frames.reduce((a, b) =>
          Math.abs(b.time - (dive.entryTime - 0.06)) <
          Math.abs(a.time - (dive.entryTime - 0.06))
            ? b
            : a,
        );
      const scene = window.captureScene;
      scene.setDive(dive);
      scene.setOptions({ trail: false, momentum: false });
      scene.render(frame.time);
      scene.frameCamera(spec.camera);
      scene.render(frame.time);
      return {
        png: scene.capture().split(",")[1],
        parameters,
        skill: dive.skill,
        result: dive.result,
        time: frame.time,
        camera: spec.camera,
        frameSelection: spec.frame,
        physics: dive.physics,
        diagnostics: {
          triangles: scene.diagnostics().triangles,
          calls: scene.diagnostics().calls,
        },
      };
    }, spec);
    await writeFile(`examples/${spec.file}`, Buffer.from(shot.png, "base64"));
    delete shot.png;
    images.push({ file: spec.file, ...shot });
  }
  const manifest = await readFile("public/asset-manifest.json");
  await writeFile(
    "examples/captures.json",
    JSON.stringify(
      {
        method:
          "Direct transparent PNG from the same MuJoCo worker and Three.js scene as the interactive experiment. Fixed declared presets; no external artwork or edited pixels.",
        viewport: [1600, 1100],
        deviceScaleFactor: 1,
        figure: {
          package: "@zachshotamartin/stick-figure",
          sourceSHA256: createHash("sha256")
            .update(
              await readFile(
                new URL(import.meta.resolve("@zachshotamartin/stick-figure")),
              ),
            )
            .digest("hex"),
        },
        assetManifestSHA256: createHash("sha256")
          .update(manifest)
          .digest("hex"),
        assets: JSON.parse(manifest),
        images,
      },
      null,
      2,
    ) + "\n",
  );
  if (errors.length) throw new Error(errors.join("\n"));
  await page.evaluate(() => {
    window.physicsWorker.terminate();
    window.captureScene.dispose();
  });
  console.log(`Captured ${images.length} actual learned dives.`);
} finally {
  await browser.close();
  server?.kill();
}
