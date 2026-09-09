import { test, expect } from "@playwright/test";
import { readFile } from "node:fs/promises";
async function ready(page) {
  await expect(page.locator(".learn-to-dive")).toHaveAttribute(
    "data-state",
    "ready",
  );
}
async function setRange(page, name, value) {
  await page.locator(`[name="${name}"]`).evaluate((el, value) => {
    el.value = value;
    el.dispatchEvent(new Event("input", { bubbles: true }));
  }, String(value));
}
test("pretrained trajectory, all policies, settings and matching export", async ({
  page,
}) => {
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  await ready(page);
  await expect(page.locator(".ld-outcome")).toHaveText(
    "Valid head-first entry",
  );
  await expect(page.locator('[data-metric="twists"]')).not.toHaveText("0.00");
  await setRange(page, "height", 10);
  await ready(page);
  await expect(page.locator('[data-value="height"]')).toHaveText("10.0 m");
  await page.locator('[name="policy"]').selectOption("untrained");
  await ready(page);
  await expect(page.locator('[data-score="difficulty"]')).toHaveText("0.0");
  await page.locator('[name="policy"]').selectOption("baseline");
  await ready(page);
  await expect(page.locator('[data-metric="twists"]')).toHaveText("0.00");
  const downloadPromise = page.waitForEvent("download");
  await page
    .getByRole("button", { name: "Export result", exact: true })
    .click();
  const download = await downloadPromise;
  const result = JSON.parse(await readFile(await download.path(), "utf8"));
  expect(result.policy).toBe("baseline");
  expect(result.parameters.height).toBe(10);
  expect(result.result.valid).toBe(true);
  await page.getByRole("button", { name: "Reset", exact: true }).click();
  await ready(page);
  await expect(page.locator('[name="policy"]')).toHaveValue("pretrained");
  await expect(page.locator('[data-value="height"]')).toHaveText("5.0 m");
  expect(errors).toEqual([]);
});
test("replay, scrub, slow motion, camera and new takeoff", async ({ page }) => {
  await page.goto("/");
  await ready(page);
  await setRange(page, "timeline", 0.6);
  await expect(page.locator(".ld-time")).toHaveText("0.60 s");
  const before = await page.locator("canvas").screenshot();
  await page.getByRole("button", { name: "Side view", exact: true }).click();
  await page.waitForTimeout(150);
  expect(await page.locator("canvas").screenshot()).not.toEqual(before);
  await page.locator('[name="speed"]').selectOption(".25");
  await page.getByRole("button", { name: "Replay", exact: true }).click();
  await page.waitForTimeout(300);
  await page.getByRole("button", { name: "Pause", exact: true }).click();
  expect(
    Number(await page.locator('[name="timeline"]').inputValue()),
  ).toBeLessThan(0.3);
  await page.getByRole("button", { name: "New takeoff", exact: true }).click();
  await ready(page);
  await expect(page.locator('[data-value="tilt"]')).not.toHaveText("0.0°");
  const entry =
    Number(await page.locator('[name="timeline"]').getAttribute("max")) - 1.35;
  await page
    .getByRole("button", { name: "Inspect entry", exact: true })
    .click();
  await expect(page.locator(".ld-time")).toHaveText(
    `${(entry - 0.06).toFixed(2)} s`,
  );
  await expect(
    page.getByRole("button", { name: "Play dive", exact: true }),
  ).toBeVisible();
});
test("390px viewport has no horizontal overflow and usable controls", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await ready(page);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await page.locator('[name="height"]').focus();
  await page.keyboard.press("ArrowRight");
  await ready(page);
  await expect(page.locator('[data-value="height"]')).toHaveText("5.1 m");
  expect(
    (
      await page
        .getByRole("button", { name: "Replay", exact: true })
        .boundingBox()
    ).height,
  ).toBeGreaterThanOrEqual(48);
});
test("embedded mount has no heading, handles batched visibility and disposes", async ({
  page,
}) => {
  await page.addInitScript(() => {
    window.__observers = [];
    window.IntersectionObserver = class {
      constructor(cb) {
        this.cb = cb;
        window.__observers.push(this);
      }
      observe() {}
      disconnect() {}
    };
  });
  await page.goto("/");
  await ready(page);
  await page.evaluate(async () => {
    const { mountExperiment } = await import("/src/index.js");
    document.querySelector("main").innerHTML = "";
    window.__experiment = mountExperiment(document.querySelector("main"), {
      embedded: true,
    });
  });
  await ready(page);
  await expect(page.locator(".ld-heading")).toHaveCount(0);
  await page.evaluate(() =>
    window.__observers
      .at(-1)
      .cb([{ isIntersecting: false }, { isIntersecting: true }]),
  );
  await page.getByRole("button", { name: "Replay", exact: true }).click();
  await page.waitForTimeout(200);
  expect(
    Number(await page.locator('[name="timeline"]').inputValue()),
  ).toBeGreaterThan(0);
  await page.evaluate(() => window.__experiment.dispose());
  await expect(page.locator("canvas")).toHaveCount(0);
});
test("canvas exports transparent real scene pixels", async ({ page }) => {
  await page.goto("/");
  await ready(page);
  await setRange(page, "timeline", 0.6);
  await page.waitForTimeout(80);
  const counts = await page.locator("canvas").evaluate((canvas) => {
    const c = document.createElement("canvas");
    c.width = canvas.width;
    c.height = canvas.height;
    const ctx = c.getContext("2d");
    ctx.drawImage(canvas, 0, 0);
    const d = ctx.getImageData(0, 0, c.width, c.height).data;
    let clear = 0,
      opaque = 0;
    for (let i = 3; i < d.length; i += 4) {
      if (d[i] === 0) clear++;
      if (d[i] > 200) opaque++;
    }
    return { clear, opaque, n: c.width * c.height };
  });
  expect(counts.clear / counts.n).toBeGreaterThan(0.25);
  expect(counts.opaque / counts.n).toBeGreaterThan(0.05);
});

test("keyboard camera controls and extreme heights retain clear canvas edges", async ({
  page,
}) => {
  await page.goto("/");
  await ready(page);
  const canvas = page.locator("canvas");
  const before = await canvas.screenshot();
  await canvas.focus();
  await page.keyboard.press("ArrowRight");
  await page.keyboard.press("+");
  await page.waitForTimeout(100);
  expect(await canvas.screenshot()).not.toEqual(before);
  for (const height of [2, 12]) {
    await setRange(page, "height", height);
    await ready(page);
    await page.getByRole("button", { name: "Side view", exact: true }).click();
    await page.waitForTimeout(100);
    const nonclear = await canvas.evaluate((canvas) => {
      const c = document.createElement("canvas");
      c.width = canvas.width;
      c.height = canvas.height;
      const ctx = c.getContext("2d");
      ctx.drawImage(canvas, 0, 0);
      const d = ctx.getImageData(0, 0, c.width, c.height).data;
      let count = 0;
      for (let x = 0; x < c.width; x++)
        for (const y of [0, c.height - 1])
          if (d[(y * c.width + x) * 4 + 3] > 0) count++;
      for (let y = 0; y < c.height; y++)
        for (const x of [0, c.width - 1])
          if (d[(y * c.width + x) * 4 + 3] > 0) count++;
      return count;
    });
    expect(nonclear).toBe(0);
  }
});
