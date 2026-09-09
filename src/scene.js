import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
export function createScene(container) {
  const scene = new THREE.Scene();
  scene.background = null;
  const renderer = new THREE.WebGLRenderer({
    antialias: true,
    alpha: true,
    preserveDrawingBuffer: true,
  });
  renderer.setClearColor(0x000000, 0);
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  container.append(renderer.domElement);
  renderer.domElement.setAttribute(
    "aria-label",
    "3D diving arena. Drag or use arrow keys to orbit; scroll or use plus and minus to zoom. Camera buttons reset the view.",
  );
  renderer.domElement.setAttribute("role", "img");
  renderer.domElement.tabIndex = 0;
  const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100),
    controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.12;
  controls.minDistance = 4;
  controls.maxDistance = 55;
  controls.maxPolarAngle = Math.PI * 0.48;
  controls.target.set(1.5, 2.4, 0);
  const keyAbort = new AbortController();
  renderer.domElement.addEventListener(
    "keydown",
    (event) => {
      if (
        ![
          "ArrowLeft",
          "ArrowRight",
          "ArrowUp",
          "ArrowDown",
          "+",
          "=",
          "-",
        ].includes(event.key)
      )
        return;
      event.preventDefault();
      const offset = camera.position.clone().sub(controls.target);
      const spherical = new THREE.Spherical().setFromVector3(offset);
      if (event.key === "ArrowLeft") spherical.theta -= 0.15;
      if (event.key === "ArrowRight") spherical.theta += 0.15;
      if (event.key === "ArrowUp") spherical.phi -= 0.12;
      if (event.key === "ArrowDown") spherical.phi += 0.12;
      if (["+", "="].includes(event.key)) spherical.radius *= 0.9;
      if (event.key === "-") spherical.radius *= 1.1;
      spherical.phi = THREE.MathUtils.clamp(
        spherical.phi,
        0.15,
        controls.maxPolarAngle,
      );
      spherical.radius = THREE.MathUtils.clamp(
        spherical.radius,
        controls.minDistance,
        controls.maxDistance,
      );
      camera.position
        .copy(controls.target)
        .add(offset.setFromSpherical(spherical));
      controls.update();
    },
    { signal: keyAbort.signal },
  );
  const ambient = new THREE.HemisphereLight("#e7eee2", "#35433b", 2.2);
  scene.add(ambient);
  const sun = new THREE.DirectionalLight("#fff4dc", 3.2);
  sun.position.set(-2, 14, 8);
  sun.castShadow = true;
  sun.shadow.mapSize.set(1024, 1024);
  sun.shadow.camera.left = -12;
  sun.shadow.camera.right = 12;
  sun.shadow.camera.top = 16;
  sun.shadow.camera.bottom = -8;
  sun.shadow.camera.far = 40;
  sun.shadow.bias = -0.0004;
  scene.add(sun);
  const mats = {
    deck: new THREE.MeshStandardMaterial({ color: "#708676", roughness: 0.78 }),
    edge: new THREE.MeshStandardMaterial({ color: "#bdc9b6", roughness: 0.65 }),
    steel: new THREE.MeshStandardMaterial({
      color: "#9cae9a",
      metalness: 0.4,
      roughness: 0.35,
    }),
    water: new THREE.MeshPhysicalMaterial({
      color: "#356f63",
      roughness: 0.18,
      metalness: 0.1,
      transparent: true,
      opacity: 0.82,
    }),
    skin: new THREE.MeshStandardMaterial({ color: "#d6ac8b", roughness: 0.65 }),
    suit: new THREE.MeshStandardMaterial({ color: "#bdd19d", roughness: 0.7 }),
    dark: new THREE.MeshStandardMaterial({ color: "#577b66", roughness: 0.7 }),
    cap: new THREE.MeshStandardMaterial({ color: "#d99976", roughness: 0.5 }),
  };
  function mesh(geometry, material, parent = scene) {
    const m = new THREE.Mesh(geometry, material);
    m.castShadow = true;
    m.receiveShadow = true;
    parent.add(m);
    return m;
  }
  function box(w, h, d, x, y, z, mat = mats.deck) {
    const m = mesh(new THREE.BoxGeometry(w, h, d), mat);
    m.position.set(x, y, z);
    return m;
  }
  box(8.5, 0.35, 6.5, 4, -0.65, 0, mats.dark);
  box(8.6, 0.2, 0.3, 4, -0.05, 3.15, mats.edge);
  box(8.6, 0.2, 0.3, 4, -0.05, -3.15, mats.edge);
  box(0.3, 0.2, 6.6, -0.15, -0.05, 0, mats.edge);
  box(0.3, 0.2, 6.6, 8.15, -0.05, 0, mats.edge);
  const water = mesh(new THREE.PlaneGeometry(8, 6), mats.water);
  water.rotation.x = -Math.PI / 2;
  water.position.set(4, 0, 0);
  water.castShadow = false;
  const tileMaterial = new THREE.LineBasicMaterial({
    color: "#88aa93",
    transparent: true,
    opacity: 0.23,
  });
  const lines = [];
  for (let x = 0.5; x < 8; x += 0.5) lines.push(x, -0.25, -3, x, -0.25, 3);
  for (let z = -2.5; z < 3; z += 0.5) lines.push(0, -0.25, z, 8, -0.25, z);
  const tiles = new THREE.LineSegments(
    new THREE.BufferGeometry().setAttribute(
      "position",
      new THREE.Float32BufferAttribute(lines, 3),
    ),
    tileMaterial,
  );
  scene.add(tiles);
  const board = box(2.8, 0.13, 0.8, -1.4, 5, 0, mats.edge),
    support = box(0.38, 5, 0.6, -2.6, 2.5, 0, mats.steel),
    platform = box(1.5, 0.2, 2, -3, 5, 0, mats.deck);
  const ladder = new THREE.Group();
  scene.add(ladder);
  const railA = mesh(
      new THREE.CylinderGeometry(0.035, 0.035, 1, 8),
      mats.steel,
      ladder,
    ),
    railB = mesh(
      new THREE.CylinderGeometry(0.035, 0.035, 1, 8),
      mats.steel,
      ladder,
    );
  const rungs = [];
  for (let i = 0; i < 34; i++) {
    const rung = mesh(
      new THREE.CylinderGeometry(0.027, 0.027, 0.75, 8),
      mats.steel,
      ladder,
    );
    rung.rotation.z = Math.PI / 2;
    rungs.push(rung);
  }
  const diver = new THREE.Group();
  scene.add(diver);
  const torso = mesh(
      new THREE.CylinderGeometry(0.195, 0.145, 0.57, 16),
      mats.suit,
      diver,
    ),
    hips = mesh(new THREE.SphereGeometry(0.18, 16, 12), mats.dark, diver),
    head = mesh(new THREE.SphereGeometry(0.15, 20, 16), mats.skin, diver),
    cap = mesh(
      new THREE.SphereGeometry(
        0.153,
        20,
        12,
        0,
        Math.PI * 2,
        0,
        Math.PI * 0.48,
      ),
      mats.cap,
      diver,
    );
  const bones = [];
  for (let i = 0; i < 8; i++)
    bones.push(
      mesh(
        new THREE.CylinderGeometry(
          i < 4 ? 0.055 : 0.073,
          i < 4 ? 0.06 : 0.075,
          1,
          12,
        ),
        i < 4 ? mats.skin : mats.skin,
        diver,
      ),
    );
  const joints = [];
  for (let i = 0; i < 8; i++)
    joints.push(
      mesh(
        new THREE.SphereGeometry(i < 4 ? 0.066 : 0.083, 12, 8),
        mats.skin,
        diver,
      ),
    );
  const hands = [
    mesh(new THREE.SphereGeometry(0.068, 12, 8), mats.skin, diver),
    mesh(new THREE.SphereGeometry(0.068, 12, 8), mats.skin, diver),
  ];
  const feet = [
    mesh(new THREE.BoxGeometry(0.12, 0.09, 0.25), mats.skin, diver),
    mesh(new THREE.BoxGeometry(0.12, 0.09, 0.25), mats.skin, diver),
  ];
  const momentum = new THREE.ArrowHelper(
    new THREE.Vector3(0, 0, 1),
    new THREE.Vector3(),
    1.5,
    0xd9c394,
    0.25,
    0.12,
  );
  momentum.visible = false;
  scene.add(momentum);
  const trailGeometry = new THREE.BufferGeometry(),
    trail = new THREE.Line(
      trailGeometry,
      new THREE.LineBasicMaterial({
        color: "#b8cd99",
        transparent: true,
        opacity: 0.45,
      }),
    );
  scene.add(trail);
  const splashPositions = new Float32Array(150 * 3),
    splashGeometry = new THREE.BufferGeometry().setAttribute(
      "position",
      new THREE.BufferAttribute(splashPositions, 3),
    ),
    splash = new THREE.Points(
      splashGeometry,
      new THREE.PointsMaterial({
        color: "#cadfcd",
        size: 0.07,
        transparent: true,
        opacity: 0.8,
      }),
    );
  scene.add(splash);
  splash.visible = false;
  const ring = mesh(
    new THREE.RingGeometry(0.95, 1, 64),
    new THREE.MeshBasicMaterial({
      color: "#c6d9c4",
      transparent: true,
      opacity: 0.3,
      side: THREE.DoubleSide,
    }),
  );
  ring.rotation.x = -Math.PI / 2;
  ring.visible = false;
  ring.castShadow = false;
  let height = 5,
    trajectory = null,
    showTrail = true,
    showMomentum = false;
  const up = new THREE.Vector3(0, 1, 0),
    v1 = new THREE.Vector3(),
    v2 = new THREE.Vector3();
  function bone(index, a, b) {
    v1.fromArray(a);
    v2.fromArray(b);
    const m = bones[index];
    m.position.copy(v1).add(v2).multiplyScalar(0.5);
    m.scale.y = v1.distanceTo(v2);
    m.quaternion.setFromUnitVectors(up, v2.sub(v1).normalize());
  }
  function pose(t) {
    torso.scale.y = 1 - 0.25 * t;
    torso.position.y = 0.08 - 0.06 * t;
    hips.position.set(0, -0.23 + 0.08 * t, 0.02);
    head.position.set(0, 0.51 - 0.14 * t, 0);
    cap.position.copy(head.position);
    for (let s = 0; s < 2; s++) {
      const side = s ? 1 : -1,
        shoulder = [side * 0.19, 0.32 - 0.1 * t, 0],
        elbow = [side * (0.2 + 0.14 * t), 0.64 - 0.65 * t, 0.02 + 0.35 * t],
        hand = [side * (0.11 + 0.03 * t), 0.98 - 0.8 * t, 0.0 + 0.44 * t],
        hip = [side * 0.11, -0.22 + 0.05 * t, 0],
        knee = [side * 0.12, -0.59 + 0.41 * t, 0.38 * t],
        foot = [side * 0.11, -0.94 + 0.46 * t, 0.12 * t];
      bone(s * 2, shoulder, elbow);
      bone(s * 2 + 1, elbow, hand);
      bone(4 + s * 2, hip, knee);
      bone(5 + s * 2, knee, foot);
      joints[s * 2].position.fromArray(shoulder);
      joints[s * 2 + 1].position.fromArray(elbow);
      joints[4 + s * 2].position.fromArray(hip);
      joints[5 + s * 2].position.fromArray(knee);
      hands[s].position.fromArray(hand);
      feet[s].position.fromArray(foot);
    }
  }
  let cameraView = "orbit";
  function frameCamera(view = cameraView) {
    cameraView = view;
    if (view === "entry" && trajectory) {
      const entry = trajectory.frames.at(-1);
      controls.target.set(entry.x, 1.45, entry.z);
      camera.position
        .copy(controls.target)
        .add(new THREE.Vector3(-0.4, 0.24, 1).normalize().multiplyScalar(7.3));
      camera.lookAt(controls.target);
      controls.update();
      return;
    }
    // Fit projected bounds of actual static geometry and the diver's full path.
    // An enclosing world-space box contains empty corners that visibly bias framing.
    const direction = new THREE.Vector3(
      ...(view === "side" ? [0, 0.06, 1] : [0.45, 0.48, 1.4]),
    ).normalize();
    const right = new THREE.Vector3()
      .crossVectors(new THREE.Vector3(0, 1, 0), direction)
      .normalize();
    const vertical = new THREE.Vector3().crossVectors(direction, right);
    const points = [];
    scene.updateMatrixWorld(true);
    scene.traverseVisible((object) => {
      if (
        !object.isMesh ||
        object.parent === diver ||
        object === ring ||
        object.parent === momentum
      )
        return;
      object.geometry.computeBoundingBox();
      const box = object.geometry.boundingBox;
      for (const x of [box.min.x, box.max.x])
        for (const y of [box.min.y, box.max.y])
          for (const z of [box.min.z, box.max.z])
            points.push(
              new THREE.Vector3(x, y, z).applyMatrix4(object.matrixWorld),
            );
    });
    for (const frame of trajectory?.frames ?? [
      { x: 0, y: height + 0.9, z: 0 },
    ]) {
      const center = new THREE.Vector3(frame.x, frame.y, frame.z);
      for (const x of [-1.05, 1.05])
        for (const y of [-1.05, 1.05])
          for (const z of [-1.05, 1.05])
            points.push(
              center
                .clone()
                .addScaledVector(right, x)
                .addScaledVector(vertical, y)
                .addScaledVector(direction, z),
            );
    }
    const ky = Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2) * 0.84,
      kx = ky * camera.aspect;
    let maxX = -Infinity,
      minX = Infinity,
      maxY = -Infinity,
      minY = Infinity,
      minZ = Infinity,
      maxZ = -Infinity;
    for (const point of points) {
      const x = point.dot(right),
        y = point.dot(vertical),
        z = point.dot(direction);
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
  function setHeight(value) {
    height = value;
    board.position.y = value;
    platform.position.y = value - 0.035;
    support.position.y = value / 2;
    support.scale.y = value / 5;
    for (const [i, rail] of [railA, railB].entries()) {
      rail.scale.y = value + 1;
      rail.position.set(-3.4 + (i ? 0.375 : -0.375), (value + 1) / 2, -0.65);
    }
    for (let i = 0; i < rungs.length; i++) {
      rungs[i].visible = i * 0.38 < value;
      rungs[i].position.set(-3.4, i * 0.38 + 0.2, -0.65);
    }
    frameCamera();
  }
  function setDive(dive) {
    trajectory = dive;
    const positions = dive.frames.flatMap((s) => [s.x, s.y, s.z]);
    trailGeometry.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(positions, 3),
    );
    trailGeometry.computeBoundingSphere();
    setHeight(dive.parameters.height);
  }
  function render(time) {
    if (trajectory) {
      const last = trajectory.frames.at(-1),
        t = Math.max(0, time),
        i = Math.min(trajectory.frames.length - 1, Math.floor(t / 0.02)),
        f = trajectory.frames[i];
      diver.visible = t < last.time + 0.45;
      diver.position.set(
        f.x,
        f.y - Math.max(0, t - last.time) * Math.abs(last.vy) * 0.6,
        f.z,
      );
      diver.quaternion.fromArray(f.q);
      pose(f.tuck);
      trail.visible = showTrail;
      trailGeometry.setDrawRange(0, i + 1);
      momentum.visible = showMomentum && diver.visible;
      momentum.position.copy(diver.position);
      momentum.setDirection(v1.fromArray(f.L).normalize());
      momentum.setLength(Math.min(2.3, Math.hypot(...f.L) / 35), 0.2, 0.1);
      const age = t - last.time;
      const visible = age >= 0 && age < 1.5;
      splash.visible = visible;
      ring.visible = visible;
      if (visible) {
        const force = 0.35 + trajectory.result.splash * 1.9;
        for (let j = 0; j < 150; j++) {
          const a = j * 2.399963,
            speed = (0.4 + (j % 17) / 17) * force;
          splashPositions[j * 3] = last.x + Math.cos(a) * speed * age;
          splashPositions[j * 3 + 1] = Math.max(
            0.015,
            (0.5 + (j % 11) / 11) * force * age - 2.5 * age * age,
          );
          splashPositions[j * 3 + 2] = last.z + Math.sin(a) * speed * age;
        }
        splashGeometry.attributes.position.needsUpdate = true;
        splash.material.opacity = Math.max(0, 0.9 - age * 0.6);
        ring.position.set(last.x, 0.02, last.z);
        ring.scale.setScalar(0.1 + age * (0.5 + force));
        ring.material.opacity = Math.max(0, 0.5 - age * 0.33);
      }
    }
    controls.update();
    renderer.render(scene, camera);
  }
  const observer = new ResizeObserver(() => {
    const { width, height } = container.getBoundingClientRect();
    if (width && height) {
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      frameCamera();
    }
  });
  observer.observe(container);
  setHeight(5);
  return {
    canvas: renderer.domElement,
    setDive,
    render,
    frameCamera,
    setOptions(options) {
      showTrail = options.trail;
      showMomentum = options.momentum;
    },
    dispose() {
      keyAbort.abort();
      observer.disconnect();
      controls.dispose();
      scene.traverse((object) => {
        object.geometry?.dispose();
        if (object.material) {
          for (const material of Array.isArray(object.material)
            ? object.material
            : [object.material])
            material.dispose();
        }
      });
      renderer.dispose();
      renderer.forceContextLoss();
      renderer.domElement.remove();
    },
  };
}
