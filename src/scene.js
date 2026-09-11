import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { createDiver } from "./visuals/diver.js";
import { createPool } from "./visuals/pool.js";

/** Original art layer. Physical body transforms and joint endpoints own the pose. */
export function createScene(container) {
  const scene = new THREE.Scene();
  scene.background = null;
  const renderer = new THREE.WebGLRenderer({
    antialias: true,
    alpha: true,
    preserveDrawingBuffer: true,
  });
  renderer.setClearColor(0, 0);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 0.94;
  const canvas = renderer.domElement;
  container.append(canvas);
  canvas.setAttribute(
    "aria-label",
    "Three-dimensional diving pool and articulated athlete. Drag or use arrow keys to orbit. Plus and minus zoom.",
  );
  canvas.setAttribute("role", "img");
  canvas.tabIndex = 0;
  const environment = new RoomEnvironment(),
    pmrem = new THREE.PMREMGenerator(renderer),
    environmentMap = pmrem.fromScene(environment, 0.04);
  scene.environment = environmentMap.texture;
  scene.environmentIntensity = 0.42;
  environment.dispose();
  pmrem.dispose();
  scene.add(new THREE.HemisphereLight("#e8f4ff", "#577588", 1.2));
  const sun = new THREE.DirectionalLight("#fff1db", 2.5);
  sun.position.set(-5, 17, 8);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.camera.left = -13;
  sun.shadow.camera.right = 13;
  sun.shadow.camera.top = 17;
  sun.shadow.camera.bottom = -10;
  sun.shadow.camera.far = 60;
  sun.shadow.normalBias = 0.016;
  sun.shadow.bias = -0.00015;
  scene.add(sun);
  const fill = new THREE.DirectionalLight("#cde9ff", 0.65);
  fill.position.set(10, 10, -8);
  scene.add(fill);
  const camera = new THREE.PerspectiveCamera(40, 1, 0.06, 150),
    controls = new OrbitControls(camera, canvas);
  controls.enableDamping = false;
  controls.enablePan = false;
  controls.minDistance = 0.7;
  controls.maxDistance = 65;
  controls.minPolarAngle = 0.12;
  controls.maxPolarAngle = Math.PI * 0.49;
  const pool = createPool(scene),
    athlete = createDiver(() => render());
  scene.add(athlete.root);
  athlete.root.visible = false;
  const trailGeometry = new THREE.BufferGeometry(),
    trailMaterial = new THREE.LineBasicMaterial({
      color: "#4b9bbd",
      transparent: true,
      opacity: 0.5,
    }),
    trail = new THREE.Line(trailGeometry, trailMaterial);
  scene.add(trail);
  const momentum = new THREE.ArrowHelper(
    new THREE.Vector3(0, 0, 1),
    new THREE.Vector3(),
    1.5,
    0xeaa247,
    0.22,
    0.08,
  );
  momentum.visible = false;
  scene.add(momentum);
  const particleCount = 180,
    positions = new Float32Array(particleCount * 3),
    splashGeometry = new THREE.BufferGeometry().setAttribute(
      "position",
      new THREE.BufferAttribute(positions, 3),
    ),
    splashMaterial = new THREE.PointsMaterial({
      color: "#e8f9ff",
      size: 0.055,
      transparent: true,
      opacity: 0.9,
      depthWrite: false,
    }),
    splash = new THREE.Points(splashGeometry, splashMaterial);
  scene.add(splash);
  splash.visible = false;
  const bubblePositions = new Float32Array(65 * 3),
    bubbleGeometry = new THREE.BufferGeometry().setAttribute(
      "position",
      new THREE.BufferAttribute(bubblePositions, 3),
    ),
    bubbleMaterial = new THREE.PointsMaterial({
      color: "#d5f6ff",
      size: 0.055,
      transparent: true,
      opacity: 0.55,
      depthWrite: false,
    }),
    bubbles = new THREE.Points(bubbleGeometry, bubbleMaterial);
  scene.add(bubbles);
  bubbles.visible = false;
  for (const material of [splashMaterial, bubbleMaterial])
    material.onBeforeCompile = (shader) => {
      shader.fragmentShader = shader.fragmentShader.replace(
        "#include <color_fragment>",
        `#include <color_fragment>
 vec2 droplet=gl_PointCoord*2.0-1.0;
 float dropletRadius=dot(droplet,droplet);
 if(dropletRadius>1.0)discard;
 diffuseColor.a*=1.0-smoothstep(.12,1.0,dropletRadius);`,
      );
    };
  const rings = [];
  for (let i = 0; i < 3; i++) {
    const material = new THREE.MeshBasicMaterial({
        color: "#b5ecf7",
        transparent: true,
        opacity: 0.32,
        side: THREE.DoubleSide,
        depthWrite: false,
      }),
      ring = new THREE.Mesh(new THREE.RingGeometry(0.97, 1, 72), material);
    ring.rotation.x = -Math.PI / 2;
    ring.visible = false;
    ring.renderOrder = 4;
    scene.add(ring);
    rings.push(ring);
  }
  const crownGeo = new THREE.BufferGeometry(),
    crownMaterial = new THREE.MeshBasicMaterial({
      color: "#c8eff7",
      transparent: true,
      opacity: 0.48,
      side: THREE.DoubleSide,
      depthWrite: false,
    }),
    crown = new THREE.Mesh(crownGeo, crownMaterial);
  scene.add(crown);
  crown.visible = false;
  let trajectory = null,
    cameraView = "orbit",
    lastTime = 0,
    disposed = false,
    showTrail = false,
    showMomentum = false,
    lastFrame = null,
    entry = null,
    frameIndex = 0;
  const vector = (a) => new THREE.Vector3(...a),
    quaternion = (a) => new THREE.Quaternion(...a),
    up = new THREE.Vector3(0, 1, 0);
  function fit(points, direction, padding = 0.88) {
    const right = new THREE.Vector3().crossVectors(up, direction).normalize(),
      vertical = new THREE.Vector3().crossVectors(direction, right),
      ky = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * padding,
      kx = ky * camera.aspect;
    let maxX = -Infinity,
      minX = Infinity,
      maxY = -Infinity,
      minY = Infinity,
      minZ = Infinity,
      maxZ = -Infinity;
    for (const p of points) {
      const x = p.dot(right),
        y = p.dot(vertical),
        z = p.dot(direction);
      maxX = Math.max(maxX, x + kx * z);
      minX = Math.min(minX, x - kx * z);
      maxY = Math.max(maxY, y + ky * z);
      minY = Math.min(minY, y - ky * z);
      minZ = Math.min(minZ, z);
      maxZ = Math.max(maxZ, z);
    }
    const distance = Math.max(
      (maxX - minX) / (2 * kx),
      (maxY - minY) / (2 * ky),
    );
    const center = right
      .clone()
      .multiplyScalar((maxX + minX) / 2)
      .addScaledVector(vertical, (maxY + minY) / 2);
    camera.position.copy(center).addScaledVector(direction, distance);
    controls.target.copy(center).addScaledVector(direction, (minZ + maxZ) / 2);
    camera.lookAt(controls.target);
    controls.update();
  }
  function frameCamera(view = cameraView) {
    cameraView = view;
    if (view === "portrait" && lastFrame?.pose) {
      const center = vector(lastFrame.pose.head),
        direction = new THREE.Vector3(0.35, 0.1, 1)
          .applyQuaternion(athlete.root.quaternion)
          .normalize();
      controls.target.copy(center);
      camera.position.copy(center).addScaledVector(direction, 1.15);
      controls.update();
      draw();
      return;
    }
    if ((view === "athlete" || view === "entry") && trajectory) {
      let center;
      if (view === "entry")
        center = vector(
          entry?.point || [
            trajectory.frames.at(-1).x,
            0,
            trajectory.frames.at(-1).z,
          ],
        ).add(new THREE.Vector3(0, 0.8, 0));
      else
        center = athlete.root.visible
          ? athlete.root.position.clone()
          : new THREE.Vector3(0, pool.getHeight() + 0.8, 0);
      const direction = new THREE.Vector3(
        view === "entry" ? -0.2 : 0.65,
        0.22,
        1,
      ).normalize();
      controls.target.copy(center);
      camera.position
        .copy(center)
        .addScaledVector(direction, view === "entry" ? 6.5 : 4.4);
      controls.update();
      draw();
      return;
    }
    const points = [];
    const bounds = pool.bounds;
    for (const x of [bounds.min.x, bounds.max.x])
      for (const z of [bounds.min.z, bounds.max.z])
        points.push(new THREE.Vector3(x, bounds.min.y, z));
    for (const x of [-4.2, -2.3])
      for (const z of [-1, 1])
        points.push(new THREE.Vector3(x, pool.getHeight() + 1.05, z));
    for (const f of trajectory?.frames || []) {
      if (f.pose) for (const p of Object.values(f.pose)) points.push(vector(p));
      else if (Number.isFinite(f.y))
        points.push(
          new THREE.Vector3(f.x, f.y + 1.1, f.z),
          new THREE.Vector3(f.x, f.y - 1.1, f.z),
        );
    }
    const direction = new THREE.Vector3(
      ...(view === "side" ? [0, 0.1, 1] : [0.33, 0.36, 1]),
    ).normalize();
    fit(points, direction);
    draw();
  }
  function copyFrame(a, b, t) {
    const result = { ...a };
    for (const key of ["x", "y", "z", "vx", "vy", "vz", "boardDeflection"])
      if (Number.isFinite(a[key]) && Number.isFinite(b[key]))
        result[key] = THREE.MathUtils.lerp(a[key], b[key], t);
    if (a.q && b.q)
      result.q = quaternion(a.q).slerp(quaternion(b.q), t).toArray();
    if (a.pose && b.pose) {
      result.pose = {};
      for (const key of Object.keys(a.pose))
        result.pose[key] = b.pose[key]
          ? a.pose[key].map((v, i) =>
              THREE.MathUtils.lerp(v, b.pose[key][i], t),
            )
          : a.pose[key];
    }
    if (a.bodies && b.bodies) {
      result.bodies = {};
      for (const name of Object.keys(a.bodies)) {
        const x = a.bodies[name],
          y = b.bodies[name] || x;
        result.bodies[name] = {
          position: x.position.map((v, i) =>
            THREE.MathUtils.lerp(v, y.position[i], t),
          ),
          quaternion: quaternion(x.quaternion)
            .slerp(quaternion(y.quaternion), t)
            .toArray(),
        };
      }
    }
    if (a.board && b.board) {
      result.board = {
        position: a.board.position.map((v, i) =>
          THREE.MathUtils.lerp(v, b.board.position[i], t),
        ),
        quaternion: quaternion(a.board.quaternion)
          .slerp(quaternion(b.board.quaternion), t)
          .toArray(),
      };
    }
    return result;
  }
  function atTime(time) {
    const frames = trajectory.frames;
    if (time <= frames[0].time) {
      frameIndex = 0;
      return frames[0];
    }
    let lo = 0,
      hi = frames.length - 1;
    while (lo < hi) {
      const mid = Math.ceil((lo + hi) / 2);
      if (frames[mid].time <= time) lo = mid;
      else hi = mid - 1;
    }
    frameIndex = lo;
    const a = frames[lo],
      b = frames[Math.min(lo + 1, frames.length - 1)],
      alpha =
        b.time === a.time
          ? 0
          : THREE.MathUtils.clamp((time - a.time) / (b.time - a.time), 0, 1);
    return copyFrame(a, b, alpha);
  }
  function applyFrame(f) {
    if (!f.pose) {
      athlete.root.visible = false;
      return;
    }
    const center = vector(f.pose.pelvis),
      right = vector(f.pose.shoulderL)
        .sub(vector(f.pose.shoulderR))
        .normalize(),
      vertical = vector(f.pose.neck).sub(center);
    vertical.addScaledVector(right, -vertical.dot(right)).normalize();
    const forward = new THREE.Vector3()
      .crossVectors(right, vertical)
      .normalize();
    vertical.crossVectors(forward, right).normalize();
    const orientation = new THREE.Quaternion().setFromRotationMatrix(
        new THREE.Matrix4().makeBasis(right, vertical, forward),
      ),
      q = orientation.toArray(),
      inverse = orientation.clone().invert();
    const local = Object.fromEntries(
      Object.entries(f.pose).map(([name, p]) => [
        name,
        vector(p).sub(center).applyQuaternion(inverse).toArray(),
      ]),
    );
    const localBodies = f.bodies
      ? Object.fromEntries(
          Object.entries(f.bodies).map(([name, b]) => [
            name,
            {
              quaternion: inverse
                .clone()
                .multiply(quaternion(b.quaternion))
                .toArray(),
            },
          ]),
        )
      : null;
    athlete.root.position.copy(center);
    athlete.root.quaternion.fromArray(q);
    athlete.apply(local, localBodies);
    athlete.root.visible = true;
    pool.setBoard(f.boardDeflection || 0, f.board || f.bodies?.board || null);
  }
  function entryEffects(time) {
    splash.visible = bubbles.visible = crown.visible = false;
    rings.forEach((r) => (r.visible = false));
    if (!entry) return;
    const age = time - entry.time;
    if (age < 0 || age > 2.4) return;
    const point = entry.point,
      spread = THREE.MathUtils.clamp(
        trajectory.result?.splash ?? trajectory.result?.splashProxy ?? 0.3,
        0,
        1,
      ),
      energy = 0.45 + spread * 1.7;
    splash.visible = age < 1.2;
    for (let i = 0; i < particleCount; i++) {
      const angle = i * 2.399963,
        r = (0.3 + (i % 19) / 19) * energy,
        speed = (0.7 + (i % 13) / 13) * energy,
        delay = (i % 7) * 0.006,
        a = Math.max(0, age - delay);
      positions[i * 3] = point[0] + Math.cos(angle) * r * a;
      positions[i * 3 + 1] = Math.max(
        -0.06,
        point[1] + speed * a - 4.9 * a * a,
      );
      positions[i * 3 + 2] = point[2] + Math.sin(angle) * r * a;
    }
    splashGeometry.attributes.position.needsUpdate = true;
    splashMaterial.opacity = Math.max(0, 0.95 - age * 0.8);
    bubbles.visible = age < 2;
    for (let i = 0; i < 65; i++) {
      const a = i * 2.4,
        r = 0.03 + (i % 9) * 0.012,
        b = Math.min(1.25, age * 1.8) * (i / 65);
      bubblePositions[i * 3] = point[0] + Math.cos(a) * r * (1 + age);
      bubblePositions[i * 3 + 1] = -0.1 - b + age * 0.1;
      bubblePositions[i * 3 + 2] = point[2] + Math.sin(a) * r * (1 + age);
    }
    bubbleGeometry.attributes.position.needsUpdate = true;
    bubbleMaterial.opacity = Math.max(0, 0.5 - age * 0.24);
    rings.forEach((ring, i) => {
      const a = age - i * 0.14;
      ring.visible = a > 0 && a < 1.75;
      if (ring.visible) {
        ring.position.set(point[0], 0.03 + i * 0.002, point[2]);
        ring.scale.setScalar(0.1 + a * (0.6 + energy * 0.55));
        ring.material.opacity = Math.max(0, 0.36 - a * 0.2);
      }
    });
    if (age < 0.55) {
      crown.visible = true;
      const vertex = [],
        indices = [],
        radius = 0.09 + age * energy * 0.65,
        h = Math.sin((age / 0.55) * Math.PI) * energy * 0.22;
      for (let i = 0; i <= 48; i++) {
        const a = (i / 48) * Math.PI * 2,
          peak = h * (0.72 + 0.28 * Math.sin(i * 2.8));
        vertex.push(
          point[0] + Math.cos(a) * radius,
          0.015,
          point[2] + Math.sin(a) * radius,
          point[0] + Math.cos(a) * radius * 1.17,
          peak,
          point[2] + Math.sin(a) * radius * 1.17,
        );
        if (i < 48) {
          const j = i * 2;
          indices.push(j, j + 1, j + 3, j, j + 3, j + 2);
        }
      }
      crownGeo.setAttribute(
        "position",
        new THREE.Float32BufferAttribute(vertex, 3),
      );
      crownGeo.setIndex(indices);
      crownMaterial.opacity = 0.42 * (1 - age / 0.55);
    }
  }
  function setDive(dive) {
    trajectory = dive;
    const frames = dive.frames,
      contact = frames.find(
        (f) => f.phase === "contact" || f.phase === "water",
      ),
      entryTime = Number.isFinite(dive.entryTime)
        ? dive.entryTime
        : contact?.time;
    entry = Number.isFinite(entryTime)
      ? {
          time: entryTime,
          point: dive.contactPoint ||
            frames.find((f) => f.contactPoint)?.contactPoint || [
              (contact || frames.at(-1)).x,
              0,
              (contact || frames.at(-1)).z,
            ],
        }
      : null;
    pool.setHeight(dive.parameters.height);
    trailGeometry.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(
        frames.map((f) => [f.x, f.y, f.z]).flat(),
        3,
      ),
    );
    trailGeometry.computeBoundingSphere();
    lastTime = frames[0].time;
    applyFrame(atTime(lastTime));
    frameCamera(cameraView);
    render(lastTime);
  }
  function draw() {
    if (!disposed) renderer.render(scene, camera);
  }
  function render(time = lastTime) {
    if (disposed) return;
    lastTime = Number.isFinite(time) ? time : lastTime;
    if (trajectory) {
      lastFrame = atTime(lastTime);
      applyFrame(lastFrame);
      const last = trajectory.frames.at(-1);
      if (lastTime > last.time + 0.025) athlete.root.visible = false;
      trail.visible = showTrail;
      trailGeometry.setDrawRange(0, frameIndex + 1);
      momentum.visible = showMomentum && athlete.root.visible && !!lastFrame.L;
      if (momentum.visible) {
        momentum.position.copy(athlete.root.position);
        const L = vector(lastFrame.L),
          length = L.length();
        momentum.setDirection(
          length ? L.normalize() : new THREE.Vector3(0, 0, 1),
        );
        momentum.setLength(
          THREE.MathUtils.clamp(length / 35, 0.2, 2),
          0.2,
          0.07,
        );
      }
      entryEffects(lastTime);
      if (cameraView === "athlete" && athlete.root.visible) {
        const center = athlete.root.position.clone(),
          delta = center.clone().sub(controls.target);
        camera.position.add(delta);
        controls.target.copy(center);
        controls.update();
      }
    }
    pool.renderWater(lastTime);
    draw();
  }
  controls.addEventListener("change", draw);
  const observer = new ResizeObserver(() => {
    const { width, height } = container.getBoundingClientRect();
    if (width && height) {
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      frameCamera(cameraView);
    }
  });
  observer.observe(container);
  const key = (e) => {
    if (
      ![
        "ArrowLeft",
        "ArrowRight",
        "ArrowUp",
        "ArrowDown",
        "+",
        "=",
        "-",
      ].includes(e.key)
    )
      return;
    e.preventDefault();
    const offset = camera.position.clone().sub(controls.target),
      s = new THREE.Spherical().setFromVector3(offset);
    if (e.key === "ArrowLeft") s.theta -= 0.15;
    if (e.key === "ArrowRight") s.theta += 0.15;
    if (e.key === "ArrowUp") s.phi -= 0.12;
    if (e.key === "ArrowDown") s.phi += 0.12;
    if (e.key === "+" || e.key === "=") s.radius *= 0.9;
    if (e.key === "-") s.radius *= 1.1;
    s.phi = THREE.MathUtils.clamp(s.phi, 0.12, Math.PI * 0.49);
    s.radius = THREE.MathUtils.clamp(s.radius, 3, 65);
    camera.position.copy(controls.target).add(offset.setFromSpherical(s));
    controls.update();
    draw();
  };
  canvas.addEventListener("keydown", key);
  frameCamera();
  return {
    canvas,
    ready: athlete.ready,
    setDive,
    render,
    frameCamera,
    setOptions(options) {
      showTrail = !!options.trail;
      showMomentum = !!options.momentum;
      render();
    },
    capture() {
      draw();
      return canvas.toDataURL("image/png");
    },
    diagnostics() {
      return {
        calls: renderer.info.render.calls,
        triangles: renderer.info.render.triangles,
        time: lastTime,
        pose: lastFrame?.pose,
        renderedJoints: athlete.getRenderedJoints(),
        worldRoot: athlete.root.position.toArray(),
        quaternion: athlete.root.quaternion.toArray(),
        entry,
        visible: athlete.root.visible,
        skin: athlete.diagnostics(),
      };
    },
    dispose() {
      disposed = true;
      observer.disconnect();
      canvas.removeEventListener("keydown", key);
      controls.removeEventListener("change", draw);
      controls.dispose();
      athlete.dispose();
      pool.dispose();
      trailGeometry.dispose();
      trailMaterial.dispose();
      splashGeometry.dispose();
      splashMaterial.dispose();
      bubbleGeometry.dispose();
      bubbleMaterial.dispose();
      crownGeo.dispose();
      crownMaterial.dispose();
      rings.forEach((r) => {
        r.geometry.dispose();
        r.material.dispose();
      });
      momentum.line.geometry.dispose();
      momentum.line.material.dispose();
      momentum.cone.geometry.dispose();
      momentum.cone.material.dispose();
      environmentMap.dispose();
      renderer.dispose();
      renderer.forceContextLoss();
      canvas.remove();
    },
  };
}
