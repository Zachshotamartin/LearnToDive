import { chromium } from "@playwright/test";
import assert from "node:assert/strict";
import fs from "node:fs/promises";
import { spawn } from "node:child_process";
const server = spawn(
  process.execPath,
  [
    "node_modules/vite/bin/vite.js",
    "--host",
    "127.0.0.1",
    "--port",
    "5199",
    "--strictPort",
  ],
  { stdio: "ignore" },
);
let browser;
try {
  for (let i = 0; i < 150; i++) {
    if (
      await fetch("http://127.0.0.1:5199")
        .then((r) => r.ok)
        .catch(() => false)
    )
      break;
    await new Promise((r) => setTimeout(r, 100));
  }
  browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({
      viewport: { width: 1440, height: 1100 },
    }),
    errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("http://127.0.0.1:5199");
  await page.locator("[name=autoplay]").uncheck();
  await page.locator("[data-state=ready]").waitFor({ timeout: 60000 });
  assert.equal(
    await page.locator(".learn-to-dive").getAttribute("data-steps"),
    "45056000",
  );
  await page.locator("[data-action=play]").click();
  await page.waitForFunction(
    () => Number(document.querySelector("[name=timeline]").value) > 0.3,
  );
  await page.locator("[data-action=play]").click();
  for (const [name, value] of [
    ["category", "6"],
    ["apparatus", "springboard"],
    ["height", "1"],
    ["policy", "untrained"],
  ]) {
    await page.locator(`[name=${name}]`).selectOption(value);
    await page.locator("[data-state=ready]").waitFor({ timeout: 60000 });
  }
  const pending = page.waitForEvent("download");
  await page.locator("[data-action=export]").click();
  const data = JSON.parse(
    await fs.readFile(await (await pending).path(), "utf8"),
  );
  assert.equal(data.physics.policySteps, 0);
  assert.equal(data.physics.contract, "self-declared-diver-v11");
  assert(
    data.actions.every(
      (r) => r.values.length === 9 && r.values.every(Number.isFinite),
    ),
  );
  await page.locator("[name=apparatus]").selectOption("platform");
  await page.locator("[data-state=ready]").waitFor({ timeout: 60000 });
  await page.locator("[name=policy]").selectOption("pretrained");
  await page.locator("[data-state=ready]").waitFor({ timeout: 60000 });
  await page.locator("[name=category]").selectOption("2");
  await page.locator("[data-state=ready]").waitFor({ timeout: 60000 });
  await fs.mkdir("output/browser-v11", { recursive: true });
  await page.screenshot({
    path: "output/browser-v11/interface.png",
    fullPage: true,
  });
  assert.deepEqual(errors, []);
  console.log(
    "PASS: trained/initial model, physical play/pause, six-category selection, apparatus and height changes, finite result export.",
  );
} finally {
  await browser?.close();
  server.kill();
}
