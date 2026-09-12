import { createScene } from "./scene.js";
import { ASSETS } from "./data/autonomousAssets.js";
import { EVALUATION } from "./data/autonomousEvaluation.js";
export const metadata = {
  id: "learn-to-dive",
  title: "Learn to Dive",
  description:
    "A neural policy chooses a legal dive and controls an articulated athlete throughout the simulation.",
  technique: "Self-declared PPO with torque-limited MuJoCo physics.",
  instructions: [
    "Choose an apparatus and category. The model chooses its own legal declaration.",
    "Play, pause and scrub the physical dive. Compare against initial network weights.",
  ],
  limitations: [
    "Development checkpoint: clean entries are not yet reliable.",
    "Execution uses an automated geometric approximation; water forces are simplified.",
  ],
};
const categories = [
  "Forward",
  "Backward",
  "Reverse",
  "Inward",
  "Twisting",
  "Armstand",
];
export function mountExperiment(element, options = {}) {
  const root = document.createElement("section");
  root.className = "learn-to-dive";
  root.dataset.state = "loading";
  root.setAttribute("aria-label", "Learn to Dive");
  element.append(root);
  root.innerHTML = `<div class="ld-layout"><aside class="ld-controls"><label>Apparatus<select name="apparatus"><option value="platform">Platform</option><option value="springboard">Springboard</option></select></label><label>Height<select name="height"><option value="5">5 m</option><option value="7.5" selected>7.5 m</option><option value="10">10 m</option></select></label><label>Dive category<select name="category">${categories.map((c, i) => `<option value="${i + 1}" ${i === 1 ? "selected" : ""}>${c}</option>`).join("")}</select></label><p>The model chooses the dive number and body position, then controls its joints throughout takeoff, flight and entry.</p><label>Controller<select name="policy"><option value="pretrained">Best evaluated · 45.1M steps</option><option value="untrained">Initial network weights</option></select></label><div class="ld-buttons"><button data-action="next">Next dive</button><button data-action="reset">Reset round</button></div><label class="ld-check"><input name="autoplay" type="checkbox" checked>Continue varied rounds</label><p class="ld-round"></p><fieldset><legend>Camera</legend><div class="ld-buttons"><button data-action="overview">Overview</button><button data-action="side">Side view</button><button data-action="athlete">Follow diver</button></div><label class="ld-check"><input name="trail" type="checkbox">Flight path</label></fieldset></aside><div class="ld-main"><div class="ld-stage"><div class="ld-canvas"></div><div class="ld-stage-label"></div></div><div class="ld-playback"><button data-action="play" disabled>Play dive</button><button data-action="replay" disabled>Replay</button><label>Speed<select name="speed"><option value="1">1×</option><option value=".5" selected>½×</option><option value=".25">¼×</option></select></label><span class="ld-time"></span></div><label class="ld-scrub">Physics timeline<input name="timeline" type="range" min="0" max="4" step=".01" value="0" disabled></label><div class="ld-result"><h3>Execution report</h3><div class="ld-metrics"><div><span>Execution / 10</span><strong data-metric="execution">—</strong></div><div><span>Difficulty</span><strong data-metric="difficulty">—</strong></div><div><span>Dive points</span><strong data-metric="points">—</strong></div><div><span>Entry angle</span><strong data-metric="angle">—</strong></div></div><p class="ld-outcome"></p><details><summary>Execution deductions</summary><dl class="ld-breakdown"></dl></details><div class="ld-buttons"><button data-action="export" disabled>Export result</button><button data-action="png" disabled>Export scene PNG</button></div></div></div></div><p class="ld-status" role="status">Loading physics and local model…</p><details class="ld-evaluation"><summary>Training & scoring</summary><p>Development checkpoint at ${EVALUATION.steps.toLocaleString()} training steps. On 144 fixed evaluation dives: ${(EVALUATION.metrics.valid * 100).toFixed(1)}% valid, ${(EVALUATION.metrics.clean * 100).toFixed(1)}% clean, average execution ${EVALUATION.metrics.execution.toFixed(2)}/10. Backward dives perform best; the other categories still need work.</p><p>Difficulty comes from the World Aquatics 2026 table for the selected apparatus and height. Points = 3 × execution × difficulty. A round cannot repeat a dive number, even with a different body position. The geometric execution judge measures takeoff, declared rotations, body position and the full entry until submersion; it is not an official judging panel. Water is an approximate force model.</p></details>`;
  const $ = (s) => root.querySelector(s),
    input = (n) => $(`[name="${n}"]`),
    button = (n) => $(`[data-action="${n}"]`),
    abort = new AbortController(),
    listen = (el, e, fn) =>
      el.addEventListener(e, fn, { signal: abort.signal });
  const scene = createScene($(".ld-canvas")),
    worker = new Worker(new URL("./autonomous.worker.js", import.meta.url), {
      type: "module",
    });
  let id = 0,
    dive = null,
    playing = false,
    time = 0,
    last = 0,
    raf = 0,
    autoTimer = 0,
    disposed = false,
    used = [],
    round = 0;
  const assets = options.assetBase
    ? Object.fromEntries(
        Object.entries(ASSETS).map(([k, v]) => [
          k,
          new URL(
            v.split("/public/")[1] ?? v.split("/").at(-1),
            new URL(options.assetBase, location.href),
          ).href,
        ]),
      )
    : ASSETS;
  function play(value) {
    playing = value;
    button("play").textContent = value ? "Pause" : "Play dive";
    button("play").setAttribute("aria-pressed", String(value));
    clearTimeout(autoTimer);
    last = 0;
  }
  function request() {
    play(false);
    clearTimeout(autoTimer);
    dive = null;
    root.dataset.state = "loading";
    for (const n of ["play", "replay", "export", "png"])
      button(n).disabled = true;
    input("timeline").disabled = true;
    $(".ld-status").textContent = "Simulating the model’s joint actions…";
    worker.postMessage({
      id: ++id,
      assets,
      policy: input("policy").value,
      parameters: {
        height: Number(input("height").value),
        apparatus: input("apparatus").value,
        category: Number(input("category").value),
        used,
        round,
      },
    });
  }
  function reset() {
    used = [];
    round = 0;
    request();
  }
  function next() {
    if (dive && !used.includes(dive.skill.code)) used.push(dive.skill.code);
    round++;
    if (round >= 6) {
      round = 0;
      used = [];
    }
    const max = input("apparatus").value === "platform" ? 6 : 5;
    input("category").value = String(
      (Number(input("category").value) % max) + 1,
    );
    request();
  }
  worker.onmessage = ({ data }) => {
    if (disposed || data.id !== id) return;
    if (data.error) {
      root.dataset.state = "error";
      $(".ld-status").textContent = `Unable to start: ${data.error}`;
      return;
    }
    dive = data.dive;
    scene.setDive(dive);
    time = 0;
    input("timeline").max = dive.frames.at(-1).time;
    input("timeline").value = 0;
    input("timeline").disabled = false;
    for (const n of ["play", "replay", "export", "png"])
      button(n).disabled = false;
    root.dataset.state = "ready";
    root.dataset.steps = String(dive.physics.policySteps);
    const r = dive.result;
    for (const [key, value] of Object.entries({
      execution: r.execution,
      difficulty: r.difficulty,
      points: r.points,
      angle: r.entryAngle,
    }))
      $(`[data-metric="${key}"]`).textContent = value.toFixed(2);
    $(".ld-stage-label").textContent =
      `${dive.skill.id} · ${categories[r.category - 1]} · ${dive.parameters.height} m ${dive.parameters.apparatus}`;
    $(".ld-outcome").textContent = r.failures.length
      ? r.failures.join(" · ")
      : r.clean
        ? "Clean dive"
        : "Declared dive completed; execution faults remain";
    $(".ld-breakdown").replaceChildren(
      ...Object.entries(r.deductions).flatMap(([k, v]) => {
        const dt = document.createElement("dt"),
          dd = document.createElement("dd");
        dt.textContent = k.replace(/[A-Z]/g, (m) => " " + m.toLowerCase());
        dd.textContent = v.toFixed(2);
        return [dt, dd];
      }),
    );
    $(".ld-round").textContent =
      `Round attempt ${round + 1} of 6${used.length ? " · used " + used.join(", ") : ""}`;
    $(".ld-status").textContent =
      "Simulation ready. Playback shows actual physical joint motion.";
    scene.render(0);
    if (input("autoplay").checked) play(true);
  };
  worker.onerror = (e) => {
    root.dataset.state = "error";
    $(".ld-status").textContent = e.message || "Simulator worker failed";
  };
  for (const n of ["height", "category", "policy"])
    listen(input(n), "change", reset);
  listen(input("apparatus"), "change", () => {
    const platform = input("apparatus").value === "platform";
    input("height").replaceChildren(
      ...(platform ? [5, 7.5, 10] : [1, 3]).map(
        (h) => new Option(`${h} m`, h, false, h === (platform ? 7.5 : 3)),
      ),
    );
    input("category").options[5].disabled = !platform;
    if (!platform && input("category").value === "6")
      input("category").value = "1";
    reset();
  });
  listen(button("next"), "click", next);
  listen(button("reset"), "click", reset);
  listen(button("play"), "click", () => play(!playing));
  listen(button("replay"), "click", () => {
    time = 0;
    play(true);
  });
  listen(input("timeline"), "input", () => {
    play(false);
    time = Number(input("timeline").value);
    scene.render(time);
  });
  listen(input("autoplay"), "change", () => {
    if (!input("autoplay").checked) clearTimeout(autoTimer);
  });
  for (const n of ["overview", "side", "athlete"])
    listen(button(n), "click", () =>
      scene.frameCamera(n === "overview" ? "overview" : n),
    );
  listen(input("trail"), "change", () =>
    scene.setOptions({ trail: input("trail").checked }),
  );
  const download = (url, name) => {
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    a.click();
  };
  listen(button("png"), "click", () =>
    download(scene.capture(), "learn-to-dive.png"),
  );
  listen(button("export"), "click", () => {
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(dive)], { type: "application/json" }),
    );
    download(url, "dive-result.json");
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  function tick(now) {
    if (disposed) return;
    if (playing && dive) {
      if (last)
        time +=
          Math.min(0.05, (now - last) / 1000) * Number(input("speed").value);
      const end = dive.frames.at(-1).time;
      if (time >= end) {
        time = end;
        play(false);
        if (input("autoplay").checked) autoTimer = setTimeout(next, 1600);
      }
      scene.render(time);
      input("timeline").value = time;
    }
    $(".ld-time").textContent = `${time.toFixed(2)} s`;
    last = now;
    raf = requestAnimationFrame(tick);
  }
  raf = requestAnimationFrame(tick);
  request();
  return {
    dispose() {
      disposed = true;
      abort.abort();
      worker.terminate();
      cancelAnimationFrame(raf);
      clearTimeout(autoTimer);
      scene.dispose();
      root.remove();
    },
  };
}
