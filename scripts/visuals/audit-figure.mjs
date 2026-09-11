import { chromium } from "@playwright/test";
import { spawn } from "node:child_process";
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const repo = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../..",
);
const fixturePath =
  process.argv[2] ||
  path.join(repo, "artwork/fixtures/development-rollout.json");
const output = process.argv[3] || path.join(repo, "output/visual-audit");
await mkdir(output, { recursive: true });
const fixture = JSON.parse(await readFile(fixturePath, "utf8"));
const server = spawn(
  process.execPath,
  ["node_modules/vite/bin/vite.js", "--host", "127.0.0.1", "--port", "5194"],
  { cwd: repo, stdio: "ignore" },
);
await new Promise((r) => setTimeout(r, 900));
const browser = await chromium.launch({ channel: "chromium" });
const errors = [];
try {
  const page = await browser.newPage({
    viewport: { width: 1400, height: 1000 },
    deviceScaleFactor: 1.5,
  });
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (e) => {
    if (e.type() === "error") errors.push(e.text());
  });
  await page.route("**/visual-audit", (r) =>
    r.fulfill({
      contentType: "text/html",
      body: '<style>body{margin:0;background:#142020}#view{width:100vw;height:100vh}canvas{width:100%;height:100%;display:block}</style><div id="view"></div>',
    }),
  );
  await page.goto("http://127.0.0.1:5194/visual-audit");

  // An optional live portfolio URL supplies its actual computed CSS background.
  // Capture the canvas element after browser compositing, without raster edits.
  let captureBackground = '#142020';
  if (process.env.CAPTURE_PAGE_URL) {
    const reference = await browser.newPage();
    await reference.goto(process.env.CAPTURE_PAGE_URL);
    captureBackground = await reference.evaluate(() => getComputedStyle(document.body).background);
    await reference.close();
  }
  for(const [i,match] of [...captureBackground.matchAll(/url\(["']?([^"')]+)["']?\)/g)].entries()) {
    const url = match[1], response = await fetch(url);
    if(!response.ok)throw Error('Background texture fetch failed: '+response.status);
    const body=Buffer.from(await response.arrayBuffer());
    const localURL=new URL('/capture-background-'+i,new URL(page.url()).origin).href;
    await page.route(localURL,r=>r.fulfill({contentType:response.headers.get('content-type')||'image/webp',body}));
    captureBackground=captureBackground.replace(url,localURL);
  }
  await page.evaluate(async background => {
    document.body.style.background = background;
    const urls = [...background.matchAll(/url\(["']?([^"')]+)["']?\)/g)].map(m => m[1]);
    await Promise.all(urls.map(url => new Promise(resolve => {
      const image = new Image(); image.onload = image.onerror = resolve; image.src = url;
    })));
  }, captureBackground);
  const regressions = await page.evaluate(async (fixture) => {
    const { createDiver } = await import('/src/visuals/diver.js');
    const figure=createDiver();await figure.ready;figure.apply(fixture.frames[0].pose);
    const measured=figure.getRenderedJoints();let maxJointError=0;
    for(const [name,p]of Object.entries(fixture.frames[0].pose))maxJointError=Math.max(maxJointError,Math.hypot(...p.map((x,i)=>x-measured[name][i])));
    if(maxJointError>1e-5)throw Error('Stickfigure joint error '+maxJointError);
    const diagnostics=figure.diagnostics();figure.dispose();
    const {createScene}=await import('/src/scene.js');window.view=createScene(document.querySelector('#view'));view.setDive(fixture);await view.ready;
    return {maxJointError,diagnostics};
  },fixture);
  await page.waitForTimeout(600);
  const releaseTime=fixture.result?.releaseTime ?? fixture.frames.find(f=>f.released)?.time ?? .25;
  const tuckTime=fixture.frames.filter(f=>f.phase==='flight').reduce((best,f)=>(f.tuck??0)>(best.tuck??0)?f:best,fixture.frames[0]).time;
  const times = [
    ["diver-stick-board", 0, "athlete"],
    ["diver-stick-face", 0, "portrait"],
    ["diver-stick-pushoff", releaseTime, "athlete"],
    ["diver-stick-hip-flexion", releaseTime+.15, "athlete"],
    ["diver-stick-tuck", tuckTime, "athlete"],
    ["diver-stick-entry", fixture.entryTime, "athlete"],
    ["diver-stick-underwater", fixture.entryTime + 0.25, "entry"],
    ["diver-stick-pool", 0.25, "orbit"],
  ];
  let maxFrameJointError = 0;
  const captures = [];
  for (const [name, time, camera] of times) {
    const data = await page.evaluate(
      ({ time, camera }) => {
        view.render(time);
        view.frameCamera(camera);
        return { diagnostics: view.diagnostics() };
      },
      { time, camera },
    );
    for (const [name, p] of Object.entries(data.diagnostics.pose)) {
      const a = data.diagnostics.renderedJoints[name];
      maxFrameJointError = Math.max(
        maxFrameJointError,
        Math.hypot(...p.map((x, i) => x - a[i])),
      );
    }
    await page.locator('canvas').screenshot({path:path.join(output,name+'.png')});
    captures.push({
      name,
      time,
      camera,
      triangles: data.diagnostics.triangles,
      calls: data.diagnostics.calls,
    });
  }
  const performanceSample = await page.evaluate(() => {
    const gl = view.canvas.getContext("webgl2"),
      samples = [];
    for (let i = 0; i < 32; i++) {
      const start = performance.now();
      view.render(0.3 + i * 0.01);
      gl.finish();
      if (i > 1) samples.push(performance.now() - start);
    }
    samples.sort((a, b) => a - b);
    const extension = gl.getExtension("WEBGL_debug_renderer_info");
    return {
      width: view.canvas.width,
      height: view.canvas.height,
      frames: samples.length,
      meanMs: samples.reduce((a, b) => a + b, 0) / samples.length,
      p95Ms: samples[Math.floor(samples.length * 0.95)],
      renderer: extension
        ? gl.getParameter(extension.UNMASKED_RENDERER_WEBGL)
        : "unavailable",
      method:
        "Renderer CPU + completed GPU work via gl.finish; physics not included.",
    };
  });
  await page.setViewportSize({ width: 390, height: 580 });
  await page.evaluate(() => {
    view.frameCamera("orbit");
    view.render(0.25);
  });
  await page.waitForTimeout(100);
  await page.locator('canvas').screenshot({path:path.join(output,'diver-stick-mobile.png')});
  await page.evaluate(() => view.dispose());
  if (errors.length) throw new Error(errors.join("\n"));
  if (maxFrameJointError > 1e-5)
    throw new Error("Rendered frame joint mismatch " + maxFrameJointError);
  const report = {
    fixture: fixturePath,
    developmentResult: {policySHA256:fixture.policySHA256??null, modelSteps:fixture.modelSteps??null, launchValid:fixture.result?.launchValid??null, declaredValid:fixture.result?.valid??null,execution:fixture.result?.execution??null},
    captureBackground,
    regression: regressions,
    maxFrameJointError,
    captures,
    browserErrors: errors,
    performanceSample,
    notes:
      "Actual MuJoCo development rollout, used only to check the visual figure. The recorded developmentResult describes this one run; it is not held-out or repertoire qualification. Optical water/splash rendering is illustrative.",
  };
  await writeFile(
    path.join(output, "stick-figure-audit.json"),
    JSON.stringify(report, null, 2),
  );
  console.log(JSON.stringify(report, null, 2));
} finally {
  await browser.close();
  server.kill();
}
