import { test, expect } from "@playwright/test";
import { readFile } from "node:fs/promises";
async function ready(page) {
  await expect(page.locator(".learn-to-dive")).toHaveAttribute(
    "data-state",
    "ready",
    { timeout: 45000 },
  );
}
async function setRange(page, name, value) {
  await page.locator(`[name="${name}"]`).evaluate((el, value) => {
    el.value = value;
    el.dispatchEvent(new Event("input", { bubbles: true }));
  }, String(value));
}
async function open(page) {
  await page.goto("/");
  await ready(page);
  await page.locator('[name="autoplay"]').uncheck();
}
async function result(page) {
  const pending = page.waitForEvent("download");
  await page
    .getByRole("button", { name: "Export result", exact: true })
    .click();
  return JSON.parse(await readFile(await (await pending).path(), "utf8"));
}

test("real pretrained joint feedback, initial comparison and matching physics export", async ({
  page,
}) => {
  const errors = [],
    remote = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("request", (r) => {
    if (
      !r.url().startsWith("http://127.0.0.1:5184") &&
      !r.url().startsWith("data:") &&
      !r.url().startsWith("blob:")
    )
      remote.push(r.url());
  });
  await open(page);
  const trained = await result(page);
  expect(trained.physics.engine).toBe("MuJoCo 3.13.0");
  expect(trained.physics.controlDt).toBe(0.02);
  expect(trained.actions.length).toBeGreaterThan(50);
  expect(
    new Set(trained.actions.map((a) => a.targets[0].toFixed(2))).size,
  ).toBeGreaterThan(15);
  expect(trained.result.valid).toBe(true);
  expect(trained.physics.contract.controlSemantics).toBe("joint-servo-adduction-v7");
  expect(trained.physics.contract.judgeVersion).toBe("adduction-com-rise-physical-entry-v7");
  expect(trained.physics.contract.observationSize).toBe(72);
  expect(trained.physics.contract.actionSize).toBe(9);
  expect(Math.max(...trained.result.footLineAngles)).toBeLessThan(20);
  expect(Math.max(...trained.result.handAxisAngles)).toBeLessThan(30);
  expect(trained.result.handSeparation).toBeLessThan(.12);
  expect(trained.result.handHeightGap).toBeLessThan(.04);
  expect(trained.result.kneeGap).toBeLessThan(.19);
  expect(trained.result.ankleGap).toBeLessThan(.16);
  expect(trained.result.toeGap).toBeLessThan(.15);
  expect(trained.result.crossedLegs).toBe(false);
  expect(trained.result.ascent).toBeGreaterThanOrEqual(0);
  expect(Number.isFinite(trained.result.takeoffVerticalSpeed)).toBe(true);
  await page.locator('[name="policy"]').selectOption("untrained");
  await ready(page);
  const initial = await result(page);
  expect(initial.result.execution).toBeLessThan(trained.result.execution);
  expect(initial.actions).not.toEqual(trained.actions);
  await page.locator('[name="policy"]').selectOption("baseline");
  await ready(page);
  expect((await result(page)).policy).toBe("baseline");
  expect(errors).toEqual([]);
  expect(remote).toEqual([]);
});

test("scrubbing changes real pose pixels and joint angles, including synchronous hidden-state inspection", async ({
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
  await open(page);
  await setRange(page, "timeline", 0.2);
  await expect(page.locator(".ld-time")).toHaveText("0.20 s");
  const before = await page.locator("canvas").screenshot(),
    joints = await page.locator(".ld-joints").textContent();
  await setRange(page, "timeline", 0.9);
  await expect(page.locator(".ld-time")).toHaveText("0.90 s");
  expect(await page.locator("canvas").screenshot()).not.toEqual(before);
  expect(await page.locator(".ld-joints").textContent()).not.toEqual(joints);
  await page.evaluate(() =>
    window.__observers.at(-1).cb([{ isIntersecting: false }]),
  );
  await setRange(page, "timeline", 0.7);
  await expect(page.locator(".ld-time")).toHaveText("0.70 s");
  await page.evaluate(() =>
    window.__observers
      .at(-1)
      .cb([{ isIntersecting: false }, { isIntersecting: true }]),
  );
  await page.getByRole("button", { name: "Replay", exact: true }).click();
  await page.waitForTimeout(250);
  expect(
    Number(await page.locator('[name="timeline"]').inputValue()),
  ).toBeGreaterThan(0);
});

test("practice rounds prohibit numeric repeats and enforce height eligibility", async ({
  page,
}) => {
  await open(page);
  expect(await page.locator(".learn-to-dive").getAttribute("data-skill")).toBe(
    "103C",
  );
  await page.getByRole("button", { name: "Next dive", exact: true }).click();
  await ready(page);
  expect(
    await page.locator(".learn-to-dive").getAttribute("data-skill"),
  ).not.toMatch(/^103/);
  await expect(
    page.locator('[name="skill"] option[value="2"]'),
  ).toHaveAttribute("disabled", "");
  await setRange(page, "height", 3);
  await ready(page);
  await expect(
    page.locator('[name="skill"] option[value="1"]'),
  ).toHaveAttribute("disabled", "");
  await expect(
    page.locator('[name="skill"] option[value="5"]'),
  ).toHaveAttribute("disabled", "");
  await expect(page.locator('[data-value="height"]')).toHaveText("3.0 m");
  await page.getByRole("button", { name: "Reset", exact: true }).click();
  await ready(page);
  await expect(page.locator('[data-value="height"]')).toHaveText("7.5 m");
  await expect(page.locator(".ld-round")).toContainText("0 unique codes");
});

test("physical midflight disturbance changes the learned control response", async ({
  page,
}) => {
  await open(page);
  const original = await result(page);
  await page.getByText("Change the conditions", { exact: true }).click();
  await setRange(page, "disturbance", 120);
  await ready(page);
  const disturbed = await result(page);
  expect(disturbed.parameters.disturbance).toBe(120);
  const first = original.actions.filter((a) => a.time < 0.85),
    second = disturbed.actions.filter((a) => a.time < 0.85);
  expect(second).toEqual(first);
  expect(disturbed.actions.filter((a) => a.time > 1)).not.toEqual(
    original.actions.filter((a) => a.time > 1),
  );
  expect(disturbed.result.angle).not.toEqual(original.result.angle);
});

test("Replay and end-to-Play preserve Follow diver; Reset restores overview", async ({
  page,
}) => {
  await open(page);
  await setRange(page, "timeline", 0.4);
  const overview = await page.locator("canvas").screenshot();
  await page.getByRole("button", { name: "Follow diver", exact: true }).click();
  const follow = await page.locator("canvas").screenshot();
  expect(follow).not.toEqual(overview);
  await page.locator('[name="speed"]').selectOption(".25");
  await page.getByRole("button", { name: "Replay", exact: true }).click();
  await page.waitForTimeout(250);
  await page.getByRole("button", { name: "Pause", exact: true }).click();
  expect(
    Number(await page.locator('[name="timeline"]').inputValue()),
  ).toBeLessThan(0.25);
  await setRange(page, "timeline", 0.4);
  expect(await page.locator("canvas").screenshot()).toEqual(follow);
  const end = await page.locator('[name="timeline"]').getAttribute("max");
  // The HTML slider quantizes to its .01 s step; let the actual player reach
  // the final physical frame instead of assuming a scrub sets its exact time.
  await setRange(page, "timeline", Number(end) - .1);
  await page.locator('[name="speed"]').selectOption("1");
  await page.getByRole("button", { name: "Play dive", exact: true }).click();
  await expect(page.locator('[data-action="play"]')).toHaveText("Play dive");
  await page.locator('[name="speed"]').selectOption(".25");
  await page.getByRole("button", { name: "Play dive", exact: true }).click();
  await page.waitForTimeout(200);
  await page.getByRole("button", { name: "Pause", exact: true }).click();
  expect(Number(await page.locator('[name="timeline"]').inputValue())).toBeLessThan(.25);
  await setRange(page, "timeline", .4);
  expect(await page.locator("canvas").screenshot()).toEqual(follow);
  await page.getByRole("button", { name: "Reset", exact: true }).click();
  await ready(page);
  await setRange(page, "timeline", 0.4);
  expect(await page.locator("canvas").screenshot()).toEqual(overview);
});

test("scene PNG has transparent background, actual geometry and no top/bottom borders", async ({
  page,
}) => {
  await open(page);
  await setRange(page, "timeline", 0.8);
  const pixels = await page.locator("canvas").evaluate((canvas) => {
    const copy = document.createElement("canvas");
    copy.width = canvas.width;
    copy.height = canvas.height;
    const ctx = copy.getContext("2d");
    ctx.drawImage(canvas, 0, 0);
    const d = ctx.getImageData(0, 0, copy.width, copy.height).data;
    let clear = 0,
      opaque = 0,
      edge = 0;
    for (let i = 3; i < d.length; i += 4) {
      clear += d[i] === 0;
      opaque += d[i] > 200;
    }
    for (let x = 0; x < copy.width; x++)
      for (const y of [0, copy.height - 1])
        edge += d[(y * copy.width + x) * 4 + 3] > 0;
    return { clear, opaque, edge, n: copy.width * copy.height };
  });
  expect(pixels.clear / pixels.n).toBeGreaterThan(0.25);
  expect(pixels.opaque / pixels.n).toBeGreaterThan(0.05);
  expect(pixels.edge).toBe(0);
  const pending = page.waitForEvent("download");
  await page
    .getByRole("button", { name: "Export scene PNG", exact: true })
    .click();
  const file = await readFile(await (await pending).path());
  expect(file.subarray(0, 8).toString("hex")).toBe("89504e470d0a1a0a");
});

test("390px layout fits, keyboard controls work and the canvas stays visible", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await open(page);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await page.locator('[name="height"]').focus();
  await page.keyboard.press("ArrowRight");
  await ready(page);
  await expect(page.locator('[data-value="height"]')).toHaveText("7.6 m");
  expect(
    (
      await page
        .getByRole("button", { name: "Replay", exact: true })
        .boundingBox()
    ).height,
  ).toBeGreaterThanOrEqual(48);
  await page.locator("canvas").focus();
  const before = await page.locator("canvas").screenshot();
  await page.keyboard.press("ArrowRight");
  expect(await page.locator("canvas").screenshot()).not.toEqual(before);
});

test("embedded mount is heading-free and disposal cancels a pending worker job", async ({
  page,
}) => {
  await page.goto("/");
  await ready(page);
  await page.evaluate(async () => {
    const { mountExperiment } = await import("/src/index.js");
    const host = document.createElement("div");
    host.id = "test-embedded";
    document.body.replaceChildren(host);
    window.__experiment = mountExperiment(host, { embedded: true });
  });
  await ready(page);
  await expect(page.locator(".ld-heading")).toHaveCount(0);
  await setRange(page, "height", 9);
  await page.evaluate(() => window.__experiment.dispose());
  await expect(page.locator("canvas")).toHaveCount(0);
  await page.waitForTimeout(250);
  await expect(page.locator(".learn-to-dive")).toHaveCount(0);
});

test("pending settings and rapid next clicks cannot assign an old score to a new round", async ({
  page,
}) => {
  await open(page);
  // All actions occur within the scheduling debounce, before a replacement rollout.
  await page.evaluate(() => {
    const root = document.querySelector(".learn-to-dive");
    root.querySelector('[data-action="reset"]').click();
    root.querySelector('[data-action="next"]').click();
    root.querySelector('[data-action="next"]').click();
  });
  await ready(page);
  await expect(page.locator(".ld-round")).toContainText("0 unique codes");
  const current = await result(page);
  expect(await page.locator(".learn-to-dive").getAttribute("data-skill")).toBe(
    current.skill.id,
  );
  await page.evaluate(() => {
    const h = document.querySelector('[name="height"]');
    h.value = "7.6";
    h.dispatchEvent(new Event("input", { bubbles: true }));
    document.querySelector('[data-action="next"]').click();
  });
  await ready(page);
  await expect(page.locator(".ld-round")).toContainText("0 unique codes");
});

test("selecting a position variant runs that announced skill without duplicate numeric records", async ({
  page,
}) => {
  await open(page);
  await page.locator('[name="skill"]').selectOption("2");
  await ready(page);
  expect((await result(page)).skill.id).toBe("103B");
  await expect(page.locator(".ld-round")).toContainText("0 unique codes");
});

test("local physical rollout exposes clearance and bounded-command evidence without external requests", async ({
  page,
}) => {
  const remote = [],
    errors = [];
  page.on("request", (r) => {
    if (!/^(http:\/\/127\.0\.0\.1:5184|data:|blob:)/.test(r.url()))
      remote.push(r.url());
  });
  page.on("pageerror", (e) => errors.push(e.message));
  await open(page);
  const d = await result(page);
  expect(typeof d.result.launchValid).toBe("boolean");
  expect(d.result.boardNonfootImpulse).toBeGreaterThanOrEqual(0);
  expect(d.result.standImpulse).toBeGreaterThanOrEqual(0);
  expect(d.result.footDepartureGap).toBeGreaterThanOrEqual(0);
  if (!d.result.launchValid) expect(d.result.execution).toBe(0);
  expect(d.result.maxTorque).toBeLessThanOrEqual(180.000001);
  expect(d.physics.actionDistribution).toBe("tanh-squashed-gaussian-v1");
  expect(d.actions.length).toBeGreaterThan(20);
  expect(d.actions.every(a=>a.values.every(v=>Number.isFinite(v)&&Math.abs(v)<=1))).toBe(true);
  await expect(page.locator(".ld-clearance")).toContainText("Launch clearance");
  expect(errors).toEqual([]);
  expect(remote).toEqual([]);
});
