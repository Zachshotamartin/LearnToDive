import { createScene } from "./scene.js";
import { MODEL } from "./data/model.js";
import { EVALUATION } from "./data/evaluation.js";
export const metadata = {
  id: "learn-to-dive",
  title: "Learn to Dive",
  description:
    "A reward-trained diving policy adapts takeoff and body shape to the height of the board.",
  technique:
    "Offline cross-entropy policy search; quaternion rigid-body rotation with conserved angular momentum and variable inertia.",
  instructions: [
    "Change board height, launch speed or wind, then watch the pretrained policy dive.",
    "Compare initial weights and a simple half-flip baseline. Scrub or slow the replay to inspect body shape and entry.",
    "Drag to orbit the 3D arena, or use the camera buttons.",
  ],
  limitations: [
    "Reduced equivalent-body physics and a learned motor program, not full human biomechanics or an official judging system.",
    "World angular momentum is conserved in flight. Shape changes alter inertia; angular impulse occurs only at board contact.",
    "Splash is a posture/speed proxy and visualization, not a fluid simulation. Pretrained domain: 2–12 m, launch 1.5–3.5 m/s, wind ±0.6 m/s², lean ±0.09 rad.",
  ],
};
export function mountExperiment(element, options = {}) {
  const root = document.createElement("section");
  root.className = "learn-to-dive";
  root.dataset.state = "loading";
  root.setAttribute("aria-label", "Learn to Dive");
  element.append(root);
  root.innerHTML = `<header class="ld-heading"><p class="ld-eyebrow">A learned landing</p><h2>Learn to Dive</h2><p>Raise the board. Change the conditions. Watch a pretrained policy find its entry.</p></header><div class="ld-layout"><aside class="ld-controls"><label>Policy<select name="policy"><option value="pretrained">Pretrained policy</option><option value="untrained">Initial weights</option><option value="baseline">Simple half-flip baseline</option></select></label><p class="ld-policy-note">Reward-trained offline. No training or uploads in your browser.</p><label>Board height <output data-value="height">5.0 m</output><input name="height" aria-label="Board height" type="range" min="2" max="12" step=".1" value="5"></label><label>Launch speed <output data-value="spring">2.4 m/s</output><input name="spring" aria-label="Launch speed" type="range" min="1.5" max="3.5" step=".1" value="2.4"></label><label>Crosswind <output data-value="wind">0.0 m/s²</output><input name="wind" aria-label="Crosswind" type="range" min="-.6" max=".6" step=".05" value="0"></label><label>Takeoff lean <output data-value="tilt">0.0°</output><input name="tilt" aria-label="Takeoff lean" type="range" min="-5" max="5" step=".25" value="0"></label><div class="ld-buttons"><button data-action="trial" type="button">New takeoff</button><button data-action="reset" type="button">Reset</button></div><fieldset><legend>Inspect the dive</legend><label class="ld-check"><input name="trail" type="checkbox" checked>Flight path</label><label class="ld-check"><input name="momentum" type="checkbox">Angular momentum</label><div class="ld-buttons"><button data-action="camera" type="button">Orbit view</button><button data-action="side" type="button">Side view</button></div><button class="ld-entry-button" data-action="entry" type="button" disabled>Inspect entry</button></fieldset><details><summary>What the policy learned</summary><p>It selects takeoff spin, twist and a tuck/open motor program from height, launch speed and measured lean. Its 20 coefficients were optimized using complete simulated dive rewards, without expert labels.</p><p>The body then follows the same physics used in training. No midair torque is added. Tucking changes inertia; opening slows rotation.</p></details></aside><div class="ld-main"><div class="ld-stage"><div class="ld-canvas"></div><div class="ld-stage-label">2–12 m training range</div><div class="ld-stage-score"><span data-stage="phase">Preparing dive</span><strong data-stage="score">—</strong></div></div><div class="ld-playback"><button data-action="play" type="button" disabled>Play dive</button><button data-action="replay" type="button" disabled>Replay</button><label>Speed<select name="speed" aria-label="Playback speed"><option value="1">1×</option><option value=".5">½×</option><option value=".25">¼×</option></select></label><span class="ld-time">0.00 s</span></div><label class="ld-scrub">Dive timeline<input name="timeline" aria-label="Dive timeline" type="range" min="0" max="3" step=".01" value="0" disabled></label><div class="ld-result"><div class="ld-result-title"><h3>Entry report</h3><span class="ld-outcome">Waiting for a simulated dive</span></div><div class="ld-metrics"><div><span>Somersaults</span><strong data-metric="flips">—</strong></div><div><span>Twists</span><strong data-metric="twists">—</strong></div><div><span>Entry angle</span><strong data-metric="angle">—</strong></div><div><span>Splash proxy</span><strong data-metric="splash">—</strong></div></div><dl class="ld-breakdown"><dt>Difficulty / 20</dt><dd data-score="difficulty">—</dd><dt>Execution / 20</dt><dd data-score="execution">—</dd><dt>Alignment / 45</dt><dd data-score="alignment">—</dd><dt>Clean entry / 15</dt><dd data-score="splash">—</dd></dl><p class="ld-score-note">Scores are a toy judging model. Difficulty counts only with a valid, extended head-first entry.</p><button data-action="export" type="button" disabled>Export result</button></div></div></div><p class="ld-status" role="status" aria-live="polite">Loading pretrained policy…</p><details class="ld-evaluation"><summary>Measured performance & model limits</summary><p class="ld-evaluation-intro"></p><div class="ld-table-wrap"><table><thead><tr><th>Policy</th><th>Valid entries</th><th>Mean score</th><th>Entry angle</th></tr></thead><tbody></tbody></table></div><p>The pretrained policy trades some entry alignment for twist difficulty. The half-flip baseline has a cleaner average angle. These results describe this reduced simulator within the trained ranges; they are not a claim about real athletes.</p><p>Splash estimates combine projected body area, entry posture and horizontal/vertical speed. The displayed droplets illustrate that estimate. Water dynamics, muscles, limb collision, board flex and underwater motion are not simulated.</p></details>`;
  if (options.embedded) root.querySelector(".ld-heading").remove();
  const $ = (s) => root.querySelector(s),
    input = (n) => $(`[name="${n}"]`),
    button = (n) => $(`[data-action="${n}"]`),
    abort = new AbortController(),
    listen = (target, event, callback) =>
      target.addEventListener(event, callback, { signal: abort.signal });
  let scene,
    worker = null,
    disposed = false,
    request = 0,
    timer = 0,
    raf = 0,
    dive = null,
    playing = false,
    elapsed = 0,
    last = 0,
    visible = true,
    trial = 1;
  const config = { height: 5, spring: 2.4, wind: 0, tilt: 0 },
    urls = new Map();
  const status = (text) => {
    if ($(".ld-status").textContent !== text)
      $(".ld-status").textContent = text;
  };
  try {
    scene = createScene($(".ld-canvas"));
  } catch (error) {
    status(`3D renderer unavailable: ${error.message}`);
    root.dataset.state = "error";
    return {
      dispose() {
        abort.abort();
        root.remove();
      },
    };
  }
  const labels = {
    pretrained: "Pretrained policy",
    untrained: "Initial weights",
    baseline: "Simple half-flip baseline",
  };
  for (const name of ["pretrained", "untrained", "baseline"]) {
    const r = EVALUATION.evaluations[name],
      row = document.createElement("tr");
    for (const value of [
      labels[name],
      `${(r.validRate * 100).toFixed(1)}%`,
      `${r.meanScore.toFixed(1)} / 100`,
      `${r.meanAngle.toFixed(1)}°`,
    ]) {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    }
    $(".ld-evaluation tbody").append(row);
  }
  $(".ld-evaluation-intro").textContent =
    `${MODEL.trainingEpisodes.toLocaleString()} reward-evaluated training dives. Final evaluation: ${EVALUATION.episodes.toLocaleString()} separate seeded cases with held-out heights, launch speeds, wind and lean. Frozen coefficients run locally.`;
  function updateLabels() {
    for (const name of ["height", "spring", "wind", "tilt"]) {
      $(`[data-value="${name}"]`).textContent =
        name === "tilt"
          ? `${((config.tilt * 180) / Math.PI).toFixed(1)}°`
          : `${config[name].toFixed(1)} ${name === "height" ? "m" : name === "spring" ? "m/s" : "m/s²"}`;
    }
    input("tilt").value = (config.tilt * 180) / Math.PI;
  }
  function stopWorker() {
    worker?.terminate();
    worker = null;
  }
  function setPlaying(value) {
    playing = value;
    button("play").textContent = value ? "Pause" : "Play dive";
    button("play").setAttribute("aria-pressed", String(value));
  }
  function report() {
    if (!dive) return;
    const r = dive.result;
    $(".ld-outcome").textContent = r.outcome;
    $(".ld-outcome").dataset.valid = String(r.valid);
    for (const [key, value] of Object.entries({
      flips: r.flips.toFixed(2),
      twists: r.twists.toFixed(2),
      angle: `${r.angle.toFixed(1)}°`,
      splash: `${(r.splash * 100).toFixed(0)} / 100`,
    }))
      $(`[data-metric="${key}"]`).textContent = value;
    for (const [key, value] of Object.entries(r.breakdown))
      $(`[data-score="${key}"]`).textContent = value.toFixed(1);
  }
  function schedule() {
    clearTimeout(timer);
    request++;
    stopWorker();
    setPlaying(false);
    root.dataset.state = "pending";
    status("Simulating the selected policy…");
    timer = setTimeout(() => {
      if (disposed) return;
      const id = request;
      worker = new Worker(new URL("./sim.worker.js", import.meta.url), {
        type: "module",
      });
      worker.onmessage = ({ data }) => {
        if (disposed || data.id !== request) return;
        if (data.error) {
          status(data.error);
          root.dataset.state = "error";
          return;
        }
        stopWorker();
        dive = { ...data.dive, policy: input("policy").value };
        scene.setDive(dive);
        elapsed = 0;
        input("timeline").max = dive.frames.at(-1).time + 1.35;
        input("timeline").disabled = false;
        for (const name of ["play", "replay", "export", "entry"])
          button(name).disabled = false;
        report();
        root.dataset.state = "ready";
        status(
          `${labels[dive.policy]} · ${dive.result.total.toFixed(1)} / 100 · ${dive.result.outcome}.`,
        );
        setPlaying(!matchMedia("(prefers-reduced-motion: reduce)").matches);
      };
      worker.onerror = () => {
        if (!disposed) {
          status("Simulation worker failed. Reset to try again.");
          root.dataset.state = "error";
        }
      };
      worker.postMessage({
        id,
        parameters: config,
        policy: input("policy").value,
      });
    }, 80);
  }
  function animate(now) {
    if (disposed) return;
    raf = requestAnimationFrame(animate);
    const dt = Math.min(0.05, (now - (last || now)) / 1000);
    last = now;
    if (!visible || document.hidden) return;
    if (playing && dive) {
      elapsed += dt * Number(input("speed").value);
      if (elapsed >= Number(input("timeline").max)) {
        elapsed = Number(input("timeline").max);
        setPlaying(false);
      }
    }
    scene.render(elapsed);
    input("timeline").value = elapsed;
    $(".ld-time").textContent = `${elapsed.toFixed(2)} s`;
    if (dive) {
      const entered = elapsed >= dive.frames.at(-1).time;
      $('[data-stage="phase"]').textContent = entered
        ? dive.result.outcome
        : elapsed === 0
          ? "On the board"
          : "In flight";
      $('[data-stage="score"]').textContent = entered
        ? `${dive.result.total.toFixed(1)} / 100`
        : `${(dive.frames[Math.min(dive.frames.length - 1, Math.floor(elapsed / 0.02))].tuck * 100).toFixed(0)}% tuck`;
    }
  }
  for (const name of ["height", "spring", "wind", "tilt"])
    listen(input(name), "input", () => {
      config[name] =
        Number(input(name).value) * (name === "tilt" ? Math.PI / 180 : 1);
      updateLabels();
      schedule();
    });
  listen(input("policy"), "change", schedule);
  listen(button("play"), "click", () => {
    if (dive && elapsed >= Number(input("timeline").max)) elapsed = 0;
    setPlaying(!playing);
  });
  listen(button("replay"), "click", () => {
    elapsed = 0;
    setPlaying(true);
  });
  listen(input("timeline"), "input", () => {
    elapsed = Number(input("timeline").value);
    setPlaying(false);
  });
  listen(button("reset"), "click", () => {
    Object.assign(config, { height: 5, spring: 2.4, wind: 0, tilt: 0 });
    for (const n of ["height", "spring", "wind"]) input(n).value = config[n];
    input("policy").value = "pretrained";
    input("speed").value = "1";
    updateLabels();
    schedule();
  });
  listen(button("trial"), "click", () => {
    trial = (Math.imul(trial, 1664525) + 1013904223) >>> 0;
    config.tilt = ((trial / 4294967296) * 2 - 1) * 0.085;
    updateLabels();
    schedule();
  });
  for (const name of ["trail", "momentum"])
    listen(input(name), "change", () =>
      scene.setOptions({
        trail: input("trail").checked,
        momentum: input("momentum").checked,
      }),
    );
  listen(button("camera"), "click", () => scene.frameCamera("orbit"));
  listen(button("side"), "click", () => scene.frameCamera("side"));
  listen(button("entry"), "click", () => {
    if (!dive) return;
    elapsed = Math.max(0, dive.frames.at(-1).time - 0.06);
    setPlaying(false);
    scene.frameCamera("entry");
  });
  listen(button("export"), "click", () => {
    if (!dive) return;
    const blob = new Blob(
        [
          JSON.stringify(
            {
              policy: dive.policy,
              parameters: dive.parameters,
              actions: dive.actions,
              result: dive.result,
              trainingEpisodes: MODEL.trainingEpisodes,
              model: "CEM motor policy; equivalent-body simulation",
            },
            null,
            2,
          ),
        ],
        { type: "application/json" },
      ),
      url = URL.createObjectURL(blob),
      a = document.createElement("a");
    a.href = url;
    a.download = `learn-to-dive-${dive.policy}-${dive.parameters.height.toFixed(1)}m.json`;
    a.click();
    urls.set(
      url,
      setTimeout(() => {
        URL.revokeObjectURL(url);
        urls.delete(url);
      }, 1000),
    );
  });
  const observer = new IntersectionObserver(
    (entries) => {
      const entry = entries.at(-1);
      if (entry) visible = entry.isIntersecting;
    },
    { rootMargin: "80px" },
  );
  observer.observe(root);
  listen(document, "visibilitychange", () => {
    last = 0;
  });
  updateLabels();
  schedule();
  raf = requestAnimationFrame(animate);
  return {
    dispose() {
      if (disposed) return;
      disposed = true;
      request++;
      clearTimeout(timer);
      cancelAnimationFrame(raf);
      stopWorker();
      abort.abort();
      observer.disconnect();
      for (const [url, t] of urls) {
        clearTimeout(t);
        URL.revokeObjectURL(url);
      }
      urls.clear();
      scene.dispose();
      root.remove();
    },
  };
}
