import { chromium } from "@playwright/test";
import { mkdir, writeFile } from "node:fs/promises";
import { spawn } from "node:child_process";
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
  for (let i = 0; i < 50; i++) {
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
    viewport: { width: 1400, height: 1200 },
    deviceScaleFactor: 1.5,
    reducedMotion: "reduce",
  });
  await page.goto(origin);
  await page.waitForSelector('.learn-to-dive[data-state="ready"]');
  await mkdir("examples", { recursive: true });
  const set = async (name, value) =>
    page.locator(`[name="${name}"]`).evaluate((el, value) => {
      el.value = value;
      el.dispatchEvent(new Event("input", { bubbles: true }));
    }, String(value));
  const capture = async (name) => {
    const expected =
      Number(await page.locator("[name=timeline]").inputValue()).toFixed(2) +
      " s";
    await page.waitForFunction(
      (expected) => document.querySelector(".ld-time").textContent === expected,
      expected,
    );
    await page.waitForTimeout(80);
    const png = await page
      .locator("canvas")
      .evaluate((c) => c.toDataURL("image/png").split(",")[1]);
    await writeFile(`examples/${name}.png`, Buffer.from(png, "base64"));
  };
  await set("timeline", 0.4);
  await capture("pretrained-flight");
  await set("height", 8);
  await page.waitForSelector('.learn-to-dive[data-state="ready"]');
  await page
    .getByRole("button", { name: "Inspect entry", exact: true })
    .click();
  const entry =
    Number(await page.locator('[name="timeline"]').getAttribute("max")) - 1.35;
  await set("timeline", entry - 0.06);
  await capture("head-first-entry");
  await page.locator('[name="trail"]').check();
  await page.locator('[name="policy"]').selectOption("untrained");
  await page.waitForSelector('.learn-to-dive[data-state="ready"]');
  await page.getByRole("button", { name: "Orbit view", exact: true }).click();
  const impact =
    Number(await page.locator('[name="timeline"]').getAttribute("max")) - 1.35;
  await set("timeline", impact + 0.22);
  await capture("splash-proxy");
  await mkdir("test-results", { recursive: true });
  await page.screenshot({
    path: "test-results/interface-check.png",
    fullPage: true,
  });
  await writeFile(
    "examples/captures.json",
    JSON.stringify(
      {
        method:
          "Direct PNG from the live Three.js canvas with alpha 0 background. No UI or generated illustration.",
        viewport: [1400, 1200],
        deviceScaleFactor: 1.5,
        images: [
          {
            file: "pretrained-flight.png",
            policy: "pretrained",
            height: 5,
            spring: 2.4,
            wind: 0,
            tilt: 0,
            time: 0.4,
            camera: "orbit",
          },
          {
            file: "head-first-entry.png",
            policy: "pretrained",
            height: 8,
            spring: 2.4,
            wind: 0,
            tilt: 0,
            time: entry - 0.06,
            camera: "entry closeup",
            trail: false,
          },
          {
            file: "splash-proxy.png",
            policy: "untrained",
            height: 8,
            spring: 2.4,
            wind: 0,
            tilt: 0,
            time: impact + 0.22,
            camera: "orbit",
          },
        ],
      },
      null,
      2,
    ),
  );
} finally {
  await browser.close();
  server?.kill();
}
