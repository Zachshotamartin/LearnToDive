import { createScene } from "./scene.js";
import { SKILLS, parameters } from "./core/control.js";
import { DiveRound } from "./core/round.js";
import { assetURLs } from "./data/assets.js";
import { EVALUATION } from "./data/evaluation.js";
export const metadata = {
  id: "learn-to-dive",
  title: "Learn to Dive",
  description:
    "A pretrained joint controller learns to tuck, rotate and open inside an articulated physics simulation.",
  technique:
    "Original goal-conditioned PPO policy; native and browser MuJoCo 3.13 articulated dynamics with torque-limited joints and physical springboard contact.",
  instructions: [
    "Choose a dive or a varied practice round, then adjust board height.",
    "Pause, slow down or scrub the actual physics replay to inspect the learned body actions.",
    "Compare initial weights or a named heuristic, and add a physical midflight push. Drag the scene to orbit.",
  ],
  limitations: [
    "Variable-height springboard practice, not a regulated competition. Difficulty uses 3 m reference values; execution is a computed estimate.",
    "Simplified 70 kg articulated rigid-body athlete and one-axis compliant board. Board preload is stored initial spring energy. No muscle/tendon or tissue deformation simulation.",
    "A frozen neural policy commands joints every 20 ms. Water buoyancy/drag and splash are approximations, not fluid CFD.",
  ],
};
export function mountExperiment(element, options = {}) {
  const root = document.createElement("section");
  root.className = "learn-to-dive";
  root.dataset.state = "loading";
  root.setAttribute("aria-label", "Learn to Dive");
  element.append(root);
  root.innerHTML = `<header class="ld-heading"><p class="ld-eyebrow">Learning body control</p><h2>Learn to Dive</h2><p>A trained controller moves the athlete’s joints throughout each physical dive.</p></header><div class="ld-layout"><aside class="ld-controls"><label>Dive<select name="skill"><option value="mix">Varied practice round</option>${SKILLS.map((s, i) => `<option value="${i}">${s.id} · ${s.name}</option>`).join("")}</select></label><p class="ld-round" aria-live="polite"></p><label>Board height <output data-value="height"></output><input name="height" aria-label="Board height" type="range" min="3" max="10" step=".1" value="7.5"></label><label>Controller<select name="policy"><option value="pretrained">${EVALUATION.ready ? "Pretrained joint controller" : "Development joint controller"}</option><option value="untrained">Initial network weights</option><option value="baseline">Hand-authored feedback baseline</option></select></label><div class="ld-buttons"><button data-action="next" type="button">Next dive</button><button data-action="reset" type="button">Reset</button></div><label class="ld-check"><input name="autoplay" type="checkbox" checked>Continue varied rounds</label><details><summary>Change the conditions</summary><label>Board preload <output data-value="preload"></output><input name="preload" aria-label="Board preload" type="range" min=".14" max=".22" step=".005" value=".18"></label><label>Starting lean <output data-value="tilt"></output><input name="tilt" aria-label="Starting lean" type="range" min="2.5" max="17" step=".25" value="9.17"></label><label>Midflight push <output data-value="disturbance"></output><input name="disturbance" aria-label="Midflight push" type="range" min="-120" max="120" step="10" value="0"></label><p>A force is scheduled for a 0.12 s window and acts only while airborne. Late takeoff or early entry can shorten it; the result export records the actual duration and impulse. Training used 16–20 cm preload, 5.7–12.6° lean and pushes up to ±80 N; wider settings test generalization.</p><button data-action="trial" type="button">New conditions</button></details><fieldset><legend>Camera & overlays</legend><div class="ld-buttons"><button data-action="camera" type="button">Overview</button><button data-action="athlete" type="button">Follow diver</button><button data-action="side" type="button">Side view</button><button data-action="entry" type="button" disabled>Inspect entry</button></div><label class="ld-check"><input name="trail" type="checkbox">Flight path</label><label class="ld-check"><input name="momentum" type="checkbox">Angular momentum</label></fieldset><details><summary>What is learned?</summary><p>The frozen PPO neural network reads 76 current state and goal measurements 50 times per second. Its 9 commands drive 14 torque-limited joints. It learns when and how to tuck, pike, move the arms, bring the legs together and open.</p><p>Each new dive is a fresh local MuJoCo simulation; Replay and scrubbing reuse that run’s recorded physical frames. The model contains neural weights, with no saved dive trajectories. The mesh follows physical body transforms.</p><p>Board preload is an initial energy condition. The motors and board contacts generate takeoff forces; the controller cannot directly set position, velocity or world rotation.</p></details></aside><div class="ld-main"><div class="ld-stage"><div class="ld-canvas"></div><div class="ld-stage-label"></div><div class="ld-stage-score"><span data-stage="phase">Loading physics</span><strong data-stage="score">—</strong></div></div><div class="ld-playback"><button data-action="play" type="button" disabled>Play dive</button><button data-action="replay" type="button" disabled>Replay</button><label>Speed<select name="speed" aria-label="Playback speed"><option value="1">1×</option><option value=".5" selected>½×</option><option value=".25">¼×</option></select></label><span class="ld-time">0.00 s</span></div><label class="ld-scrub">Physics timeline<input name="timeline" aria-label="Dive timeline" type="range" min="0" max="3" step=".01" value="0" disabled></label><p class="ld-joints" aria-label="Physical joint angles"></p><div class="ld-result"><div class="ld-result-title"><h3>Execution report</h3><span class="ld-outcome">Waiting for the first dive</span></div><div class="ld-metrics"><div><span>Execution</span><strong data-metric="execution">—</strong></div><div><span>Reference difficulty</span><strong data-metric="difficulty">—</strong></div><div><span>Dive points</span><strong data-metric="points">—</strong></div><div><span>Worst entry angle</span><strong data-metric="angle">—</strong></div></div><p class="ld-rotation"></p><p class="ld-clearance"></p><dl class="ld-breakdown"><dt>Takeoff height deduction</dt><dd data-score="height">—</dd><dt>Form deduction</dt><dd data-score="form">—</dd><dt>Entry deduction</dt><dd data-score="entry">—</dd><dt>Splash deduction</dt><dd data-score="splash">—</dd><dt>Preparation deduction</dt><dd data-score="preparation">—</dd><dt>Takeoff deduction</dt><dd data-score="takeoff">—</dd></dl><p class="ld-score-note">Computed execution estimate, 0–10 in half points. Points = 3 × execution × reference difficulty. Wrong rotation or collision-assisted takeoff scores 0; wrong declared body position is capped at 2. Splash is an entry proxy, not a fluid judgment. Entry form follows physical toe directions, leg separation and hand-tip alignment from first water contact until the last body collider submerges. Later underwater motion is not judged. Takeoff height can deduct up to 1.5 points, reaching zero at 30 cm of whole-body rise after final departure; this is a simulator scale. The foot hinge combines ankle and midfoot pitch; its geometric thresholds approximate this simplified rig.</p><div class="ld-buttons"><button data-action="export" type="button" disabled>Export result</button><button data-action="png" type="button" disabled>Export scene PNG</button></div></div></div></div><p class="ld-status" role="status" aria-live="polite">Loading the local physics engine and policy…</p><details class="ld-evaluation"><summary>Training, evaluation & practice rules</summary><p class="ld-evaluation-intro"></p><div class="ld-table-wrap"><table><thead><tr><th>Controller</th><th>Clean dives</th><th>Execution / 10</th><th>Entry angle</th></tr></thead><tbody></tbody></table></div><p>A practice round never repeats a numeric dive code; changing its position letter does not create a new code. ${EVALUATION.ready ? "Height eligibility follows the measured controller range." : "Height limits are provisional while this development controller is being validated."} This variable-height springboard demo is not a complete competition format. Difficulty references the World Aquatics 2026 3 m table at every height.</p><p>Physics uses simplified rigid segments, joint constraints and a single vertical springboard mode. Underwater bodies experience approximate buoyancy and drag. It does not model muscles, tissue, board bending or fluid flow. Scores are computed estimates, not official judging.</p></details>`;
  if (options.embedded) root.querySelector(".ld-heading").remove();
  const $ = (s) => root.querySelector(s),
    input = (n) => $(`[name="${n}"]`),
    button = (n) => $(`[data-action="${n}"]`),
    abort = new AbortController(),
    listen = (el, event, fn) =>
      el.addEventListener(event, fn, { signal: abort.signal });
  let scene,
    worker,
    disposed = false,
    request = 0,
    timer = 0,
    raf = 0,
    elapsed = 0,
    last = 0,
    playing = false,
    visible = true,
    dive = null,
    diveRequest = -1,
    autoTimer = 0,
    config = parameters(),
    round = new DiveRound(config.height, config.seed),
    activeSkill = 1,
    urls = new Set();
  const status = (text) => {
    if ($(".ld-status").textContent !== text)
      $(".ld-status").textContent = text;
  };
  try {
    scene = createScene($(".ld-canvas"));
    worker = new Worker(new URL("./sim.worker.js", import.meta.url), {
      type: "module",
    });
  } catch (error) {
    root.dataset.state = "error";
    status(`Simulator unavailable: ${error.message}`);
    return {
      dispose() {
        abort.abort();
        scene?.dispose();
        root.remove();
      },
    };
  }
  const labels = {
      pretrained: EVALUATION.ready ? "Pretrained joint controller" : "Development joint controller · not yet qualified",
      untrained: "Initial network weights",
      baseline: "Hand-authored feedback baseline",
    },
    assets = assetURLs(options.assetBase);
  function setPlaying(value) {
    playing = value;
    button("play").textContent = value ? "Pause" : "Play dive";
    button("play").setAttribute("aria-pressed", String(value));
    if (!value) clearTimeout(autoTimer);
  }
  function updateLabels() {
    for (const name of ["height", "preload", "tilt", "disturbance"])
      $(`[data-value="${name}"]`).textContent =
        name === "height"
          ? `${config.height.toFixed(1)} m`
          : name === "preload"
            ? `${(-config.preload * 100).toFixed(1)} cm`
            : name === "tilt"
              ? `${((config.tilt * 180) / Math.PI).toFixed(1)}°`
              : `${config.disturbance} N`;
    input("height").value = config.height;
    input("preload").value = -config.preload;
    input("tilt").value = (config.tilt * 180) / Math.PI;
    input("disturbance").value = config.disturbance;
    for (const option of input("skill").options) {
      if (option.value === "mix") continue;
      const i = Number(option.value),
        used =
          round.attempts.some((a) => a.code === SKILLS[i].code) &&
          i !== activeSkill;
      option.disabled = SKILLS[i].minHeight > config.height || used;
    }
    $(".ld-round").textContent =
      `Practice round ${round.number} · ${round.attempts.length} unique codes attempted${round.attempts.length ? " · " + round.attempts.map((a) => a.id).join(", ") : ""}`;
  }
  function report() {
    const r = dive.result;
    $(".ld-outcome").textContent = r.outcome;
    $(".ld-outcome").dataset.valid = String(r.valid);
    for (const [k, v] of Object.entries({
      execution: `${r.execution.toFixed(1)} / 10`,
      difficulty: r.difficulty.toFixed(1),
      points: r.total.toFixed(1),
      angle: `${r.angle.toFixed(1)}°`,
    }))
      $(`[data-metric="${k}"]`).textContent = v;
    $(".ld-rotation").textContent =
      `Measured ${r.flips.toFixed(2)} somersaults (${r.airborneFlips.toFixed(2)} airborne + ${((r.takeoffLean * 180) / Math.PI).toFixed(0)}° takeoff lean) · ${r.twists.toFixed(2)} twists · ${dive.skill.shape === 2 ? "free position" : `${Math.round(r.shapeFraction * 100)}% declared posture coverage`} · COM rise ${(r.ascent * 100).toFixed(0)} cm · knee / ankle gaps ${(r.kneeGap * 100).toFixed(0)} / ${(r.ankleGap * 100).toFixed(0)} cm · feet ${Math.max(...r.footLineAngles).toFixed(0)}° from shin line · hands ${(r.handSeparation * 100).toFixed(0)} cm apart / ${(r.handHeightGap * 100).toFixed(0)} cm uneven · splash proxy ${Math.round(r.splash * 100)} / 100`;
    const launchIssue = r.footSideContact ? "foot contact with the board side or underside"
      : r.rotatedRecontact ? "rotated board recontact"
        : r.takeoffTiltInvalid ? "takeoff lean exceeded the simulator limit"
          : r.footDepartureGap > 0.04000001 ? "the feet left the board too far apart in time"
            : r.standImpulse > 0.1 || r.standPeak > 15 ? "contact with the stand"
              : "nonfoot contact with the board";
    $(".ld-clearance").textContent = r.launchValid
      ? "Launch clearance passed · balanced foot-only takeoff, no stand contact."
      : `Launch clearance failed · ${launchIssue}. No dive points awarded.`;
    for (const [k, v] of Object.entries(r.breakdown))
      $(`[data-score="${k}"]`).textContent = `−${v.toFixed(1)}`;
    $(".ld-stage-label").textContent = `${dive.skill.id} · ${dive.skill.name}`;
  }
  function schedule() {
    clearTimeout(timer);
    clearTimeout(autoTimer);
    request++;
    setPlaying(false);
    root.dataset.state = "pending";
    status("Simulating the selected joint controller…");
    timer = setTimeout(() => {
      if (disposed) return;
      root.dataset.state = "simulating";
      worker.postMessage({
        id: request,
        parameters: { ...config, skill: activeSkill },
        policy: input("policy").value,
        assets,
      });
    }, 90);
  }
  worker.onmessage = async ({ data }) => {
    if (disposed || data.id !== request) return;
    if (data.error) {
      root.dataset.state = "error";
      status(data.error);
      return;
    }
    try {
      await scene.ready;
    } catch (error) {
      if (!disposed) {
        root.dataset.state = "error";
        status(`Diver model unavailable: ${error.message}`);
      }
      return;
    }
    if (disposed || data.id !== request) return;
    dive = data.dive;
    diveRequest = data.id;
    scene.setDive(dive);
    elapsed = 0;
    input("timeline").max = dive.frames.at(-1).time;
    input("timeline").disabled = false;
    for (const n of ["play", "replay", "export", "png", "entry"])
      button(n).disabled = false;
    report();
    display();
    root.dataset.state = "ready";
    root.dataset.completedFrame = String(
      Number(root.dataset.completedFrame || 0) + 1,
    );
    root.dataset.skill = dive.skill.id;
    status(
      `${labels[dive.policy]} · ${dive.result.outcome} · execution ${dive.result.execution.toFixed(1)} / 10.`,
    );
    setPlaying(!matchMedia("(prefers-reduced-motion: reduce)").matches);
  };
  worker.onerror = () => {
    if (!disposed) {
      root.dataset.state = "error";
      status("Physics worker failed. Reset to try again.");
    }
  };
  function recordCurrent() {
    if (dive && diveRequest === request)
      round.record(dive.result, dive.parameters.skill);
  }
  function next() {
    if (dive) recordCurrent();
    let chosen = round.choose();
    if (chosen === null) {
      round.nextRound();
      chosen = round.choose();
    }
    activeSkill = chosen;
    input("skill").value = "mix";
    updateLabels();
    schedule();
  }
  function animate(now) {
    if (disposed) return;
    raf = requestAnimationFrame(animate);
    const dt = Math.min(0.05, (now - (last || now)) / 1000);
    last = now;
    if (!visible || document.hidden || !playing || !dive) return;
    if (playing && dive) {
      elapsed += dt * Number(input("speed").value);
      if (elapsed >= Number(input("timeline").max)) {
        elapsed = Number(input("timeline").max);
        setPlaying(false);
        recordCurrent();
        updateLabels();
        if (input("autoplay").checked && input("skill").value === "mix")
          autoTimer = setTimeout(() => {
            if (!disposed && visible && !document.hidden) next();
          }, 800);
      }
    }
    display();
  }
  function display() {
    scene.render(elapsed);
    input("timeline").value = elapsed;
    $(".ld-time").textContent = `${elapsed.toFixed(2)} s`;
    if (dive) {
      const f =
          dive.frames.findLast((f) => f.time <= elapsed) || dive.frames[0],
        entered = dive.entryTime !== null && elapsed >= dive.entryTime;
      $('[data-stage="phase"]').textContent =
        f.phase === "board"
          ? "Board preparation"
          : entered
            ? "Water contact · drag & buoyancy"
            : "Learned joint control";
      $('[data-stage="score"]').textContent = entered
        ? `${dive.result.execution.toFixed(1)} / 10 execution`
        : `${f.somersaults.toFixed(2)} turns`;
      $(".ld-joints").textContent =
        `Hip ${Math.round((f.joints[0] * 180) / Math.PI)}° · knee ${Math.round((f.joints[1] * 180) / Math.PI)}° · left arm ${Math.round((f.joints[6] * 180) / Math.PI)}° · right arm ${Math.round((f.joints[9] * 180) / Math.PI)}°`;
    }
  }
  listen(input("skill"), "change", () => {
    if (input("skill").value === "mix") {
      round.reset(config.height, config.seed);
      activeSkill = round.choose(1) ?? round.choose();
    } else {
      const preferred = Number(input("skill").value);
      if (dive && dive.skill.code !== SKILLS[preferred].code) recordCurrent();
      if (round.attempts.some((a) => a.code === SKILLS[preferred].code))
        round.nextRound();
      activeSkill = round.choose(preferred);
      if (activeSkill === null) {
        round.nextRound();
        activeSkill = round.choose(preferred);
      }
    }
    updateLabels();
    schedule();
  });
  listen(input("height"), "input", () => {
    config.height = Number(input("height").value);
    round.reset(config.height, config.seed);
    activeSkill = round.choose(activeSkill) ?? 0;
    input("skill").value = "mix";
    updateLabels();
    schedule();
  });
  for (const name of ["preload", "tilt", "disturbance"])
    listen(input(name), "input", () => {
      config[name] =
        Number(input(name).value) *
        (name === "preload" ? -1 : name === "tilt" ? Math.PI / 180 : 1);
      updateLabels();
      schedule();
    });
  listen(input("policy"), "change", schedule);
  listen(button("next"), "click", next);
  listen(button("play"), "click", () => {
    if (dive && elapsed >= Number(input("timeline").max)) {
      elapsed = 0;
    }
    setPlaying(!playing);
  });
  listen(button("replay"), "click", () => {
    elapsed = 0;
    setPlaying(true);
  });
  listen(input("timeline"), "input", () => {
    elapsed = Number(input("timeline").value);
    setPlaying(false);
    display();
  });
  listen(input("autoplay"), "change", () => {
    if (!input("autoplay").checked) clearTimeout(autoTimer);
  });
  listen(button("reset"), "click", () => {
    config = parameters();
    round = new DiveRound(config.height, config.seed);
    activeSkill = round.choose(1);
    input("skill").value = "mix";
    input("policy").value = "pretrained";
    input("speed").value = ".5";
    updateLabels();
    scene.frameCamera("orbit");
    schedule();
  });
  listen(button("trial"), "click", () => {
    config.seed = (Math.imul(config.seed, 1664525) + 1013904223) >>> 0;
    config.tilt = 0.1 + (config.seed / 4294967296) * 0.12;
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
  for (const name of ["camera", "side", "athlete"])
    listen(button(name), "click", () =>
      scene.frameCamera(name === "camera" ? "orbit" : name),
    );
  listen(button("entry"), "click", () => {
    if (!dive) return;
    elapsed = Math.max(0, (dive.entryTime ?? dive.frames.at(-1).time) - 0.06);
    setPlaying(false);
    display();
    scene.frameCamera("entry");
  });
  function download(blob, name) {
    const url = URL.createObjectURL(blob);
    urls.add(url);
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    a.click();
    setTimeout(() => {
      URL.revokeObjectURL(url);
      urls.delete(url);
    }, 1000);
  }
  listen(button("export"), "click", () => {
    if (dive)
      download(
        new Blob(
          [
            JSON.stringify(
              {
                policy: dive.policy,
                parameters: dive.parameters,
                skill: dive.skill,
                result: dive.result,
                actions: dive.actions,
                physics: dive.physics,
              },
              null,
              2,
            ),
          ],
          { type: "application/json" },
        ),
        `learn-to-dive-${dive.skill.id}.json`,
      );
  });
  listen(button("png"), "click", () =>
    scene.canvas.toBlob((blob) => {
      if (blob && !disposed)
        download(blob, `learn-to-dive-${dive?.skill.id || "scene"}.png`);
    }, "image/png"),
  );
  $(".ld-evaluation-intro").textContent =
    EVALUATION.description ||
    `${EVALUATION.trainingSteps?.toLocaleString()} training decisions. Held-out evaluation across measured eligible skills and new physical perturbations.`;
  for (const [name, r] of Object.entries(EVALUATION.policies || {})) {
    const row = document.createElement("tr");
    for (const v of [
      labels[name] || name,
      `${(r.completionRate * 100).toFixed(1)}%`,
      r.meanExecution.toFixed(2),
      `${r.meanAngle.toFixed(1)}°`,
    ]) {
      const cell = document.createElement("td");
      cell.textContent = v;
      row.append(cell);
    }
    $(".ld-evaluation tbody").append(row);
  }
  if (!EVALUATION.ready) $(".ld-table-wrap").hidden = true;
  if (EVALUATION.ready && EVALUATION.perSkill) {
    const wrap = document.createElement("div");
    wrap.className = "ld-table-wrap";
    const table = document.createElement("table");
    const caption = document.createElement("caption");
    caption.textContent = "Browser physics · clean means valid execution of at least 6/10";
    table.append(caption);
    const head = table.createTHead().insertRow();
    for (const label of ["Dive", "Height range", "Clean dives", "Execution / 10"]) {
      const cell = document.createElement("th"); cell.textContent = label; head.append(cell);
    }
    const body = table.createTBody();
    for (const [id, measurement] of Object.entries(EVALUATION.perSkill)) {
      const row = body.insertRow();
      for (const value of [id, `${measurement.minHeight}–10 m`, `${(measurement.completionRate * 100).toFixed(1)}%`, measurement.meanExecution.toFixed(2)])
        row.insertCell().textContent = value;
    }
    wrap.append(table);
    $(".ld-evaluation").append(wrap);
  }
  const bounds = root.getBoundingClientRect();
  visible =
    bounds.bottom > 0 &&
    bounds.top < innerHeight &&
    bounds.right > 0 &&
    bounds.left < innerWidth;
  const observer = new IntersectionObserver((entries) => {
    const entry = entries.at(-1);
    if (!entry) return;
    visible = entry.isIntersecting;
    last = 0;
    if (!visible) clearTimeout(autoTimer);
  });
  observer.observe(root);
  listen(document, "visibilitychange", () => {
    last = 0;
    if (document.hidden) clearTimeout(autoTimer);
  });
  activeSkill = round.choose(1);
  updateLabels();
  schedule();
  raf = requestAnimationFrame(animate);
  return {
    dispose() {
      if (disposed) return;
      disposed = true;
      request++;
      clearTimeout(timer);
      clearTimeout(autoTimer);
      cancelAnimationFrame(raf);
      observer.disconnect();
      abort.abort();
      worker.terminate();
      scene.dispose();
      for (const url of urls) URL.revokeObjectURL(url);
      root.remove();
    },
  };
}
