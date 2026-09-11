import * as THREE from "three";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";
import { POOL_DEPTH } from "../core/water.js";

/** Original pool/structure geometry. Water shading is illustrative, not a fluid solver. */
export function createPool(scene) {
  const root = new THREE.Group();
  scene.add(root);
  const staticRoot = new THREE.Group(),
    tower = new THREE.Group();
  root.add(staticRoot, tower);
  const mat = (color, roughness = 0.7, metalness = 0) =>
    new THREE.MeshStandardMaterial({ color, roughness, metalness });
  const materials = {
    deck: mat("#dedbd2", 0.83),
    concrete: mat("#b8bfc0", 0.77),
    rim: mat("#f3ede1", 0.53),
    tile: mat("#639daf", 0.5),
    blue: mat("#277797", 0.55),
    navy: mat("#17405b", 0.62),
    steel: mat("#bbc5ca", 0.24, 0.78),
    dark: mat("#677981", 0.55),
    board: mat("#99c4c7", 0.55),
    white: mat("#f7f4ea", 0.63),
  };
  const owned = new Set(),
    uniform = { value: 0 };
  // Animated optical caustics are shading only; the physics uses body drag and
  // buoyancy. Patterns move over the real basin, below the separate surface.
  materials.tile.onBeforeCompile = (shader) => {
    shader.uniforms.uWaterTime = uniform;
    shader.vertexShader =
      "varying vec3 vBasinPosition;\n" + shader.vertexShader;
    shader.vertexShader = shader.vertexShader.replace(
      "#include <begin_vertex>",
      "#include <begin_vertex>\nvBasinPosition=(modelMatrix*vec4(transformed,1.0)).xyz;",
    );
    shader.fragmentShader =
      "uniform float uWaterTime;\nvarying vec3 vBasinPosition;\n" +
      shader.fragmentShader;
    shader.fragmentShader = shader.fragmentShader.replace(
      "#include <color_fragment>",
      `#include <color_fragment>
   vec2 basinP=vBasinPosition.xz*2.5+vBasinPosition.y*.32;
   float causticA=sin(basinP.x+sin(basinP.y*1.37+uWaterTime*.23))+sin(basinP.y-sin(basinP.x*1.13-uWaterTime*.17));
   float causticB=sin(basinP.x*1.73-basinP.y*.42+uWaterTime*.29)+sin(basinP.y*1.59+cos(basinP.x*.83));
   float caustic=pow(max(0.0,1.0-abs(causticA)*.7),14.0)*.6+pow(max(0.0,1.0-abs(causticB)*.7),18.0)*.4;
   diffuseColor.rgb*=.88+caustic*.38;`,
    );
  };
  const mesh = (g, m, parent = root, at = [0, 0, 0]) => {
    owned.add(g);
    const o = new THREE.Mesh(g, m);
    o.position.fromArray(at);
    o.castShadow = true;
    o.receiveShadow = true;
    parent.add(o);
    return o;
  };
  const box = (size, at, m, parent = staticRoot, r = 0.04) =>
    mesh(
      new RoundedBoxGeometry(
        ...size,
        2,
        Math.min(r, ...size.map((v) => v * 0.25)),
      ),
      m,
      parent,
      at,
    );
  function tube(
    points,
    radius = 0.025,
    parent = staticRoot,
    material = materials.steel,
  ) {
    return mesh(
      new THREE.TubeGeometry(
        new THREE.CatmullRomCurve3(points.map((p) => new THREE.Vector3(...p))),
        points.length * 6,
        radius,
        10,
        false,
      ),
      material,
      parent,
    );
  }
  const floorY = -POOL_DEPTH,
    wallHeight = POOL_DEPTH,
    wallY = -POOL_DEPTH / 2;
  const floor = box([11.8, 0.16, 7.15], [5.45, floorY - 0.08, 0], materials.tile);
  floor.name = "pool-floor";
  const left = box([0.18, wallHeight, 7.3], [-0.53, wallY, 0], materials.tile),
    right = box([0.18, wallHeight, 7.3], [11.43, wallY, 0], materials.tile);
  box([12.1, wallHeight, 0.16], [5.45, wallY, -3.64], materials.tile);
  box([12.1, wallHeight, 0.16], [5.45, wallY, 3.64], materials.tile);
  // A concrete shell sits outside the ceramic lining; light patterns belong
  // inside the water, not on the exterior structure.
  for (const z of [-3.735, 3.735])
    box([12.16, wallHeight, 0.035], [5.45, wallY, z], materials.concrete);
  for (const x of [-0.637, 11.537])
    box([0.035, wallHeight, 7.49], [x, wallY, 0], materials.concrete);
  // A substantial deck, inset dark gutter and rounded coping retain real depth.
  for (const z of [-4.05, 4.05]) {
    box([13.1, 0.32, 0.75], [5.4, -0.04, z], materials.deck);
    box(
      [12.1, 0.16, 0.26],
      [5.45, 0.065, Math.sign(z) * 3.65],
      materials.rim,
      staticRoot,
      0.065,
    );
    box(
      [12.05, 0.025, 0.075],
      [5.45, 0.056, Math.sign(z) * 3.86],
      materials.dark,
      staticRoot,
      0.006,
    );
  }
  for (const x of [-0.9, 11.82]) {
    box([0.7, 0.32, 7.55], [x, -0.04, 0], materials.deck);
    box(
      [0.26, 0.16, 7.45],
      [x < 0 ? -0.54 : 11.45, 0.065, 0],
      materials.rim,
      staticRoot,
      0.055,
    );
  }
  box([3.1, 0.38, 3.3], [-2.72, -0.08, 0], materials.deck);
  const tilePositions = [];
  for (let x = -0.45; x <= 11.4; x += 0.3)
    tilePositions.push(x, floorY + 0.012, -3.56, x, floorY + 0.012, 3.56);
  for (let z = -3.5; z <= 3.5; z += 0.3)
    tilePositions.push(-0.44, floorY + 0.012, z, 11.34, floorY + 0.012, z);
  for (let y = floorY + 0.14; y < 0; y += 0.3) {
    tilePositions.push(-0.43, y, -3.548, 11.34, y, -3.548);
    tilePositions.push(-0.425, y, -3.55, -0.425, y, 3.55);
    tilePositions.push(11.332, y, -3.55, 11.332, y, 3.55);
  }
  const grout = new THREE.LineSegments(
    new THREE.BufferGeometry().setAttribute(
      "position",
      new THREE.Float32BufferAttribute(tilePositions, 3),
    ),
    new THREE.LineBasicMaterial({
      color: "#dae9e9",
      transparent: true,
      opacity: 0.55,
    }),
  );
  root.add(grout);
  owned.add(grout.geometry);
  for (const z of [-2.25, 0, 2.25]) {
    box(
      [10.4, 0.013, 0.085],
      [5.45, floorY + 0.024, z],
      materials.navy,
      staticRoot,
      0.002,
    );
    for (const x of [0.3, 10.6])
      box(
        [0.085, 0.014, 0.85],
        [x, floorY + 0.026, z],
        materials.navy,
        staticRoot,
        0.002,
      );
    box(
      [0.008, 1.45, 0.09],
      [11.33, -1.1, z],
      materials.navy,
      staticRoot,
      0.002,
    );
  }
  const waterMaterial = new THREE.MeshPhysicalMaterial({
    color: "#b5e4e9",
    metalness: 0,
    roughness: 0.075,
    transmission: 0.88,
    thickness: POOL_DEPTH,
    ior: 1.333,
    attenuationColor: "#62b5c9",
    attenuationDistance: 5,
    transparent: true,
    opacity: 0.95,
    depthWrite: false,
    clearcoat: 1,
    clearcoatRoughness: 0.09,
    side: THREE.DoubleSide,
  });
  waterMaterial.onBeforeCompile = (shader) => {
    shader.uniforms.uWaterTime = uniform;
    shader.vertexShader = "uniform float uWaterTime;\n" + shader.vertexShader;
    shader.vertexShader = shader.vertexShader.replace(
      "#include <begin_vertex>",
      `#include <begin_vertex>\nfloat phaseA=position.x*2.7+position.z*3.1+uWaterTime*.85;\nfloat phaseB=position.x*-4.1+position.z*1.8-uWaterTime*.63;\ntransformed.y+=sin(phaseA)*.009+sin(phaseB)*.005;`,
    );
    shader.vertexShader = shader.vertexShader.replace(
      "#include <beginnormal_vertex>",
      `#include <beginnormal_vertex>\nfloat pa=position.x*2.7+position.z*3.1+uWaterTime*.85;\nfloat pb=position.x*-4.1+position.z*1.8-uWaterTime*.63;\nobjectNormal=normalize(vec3(-cos(pa)*.0243+cos(pb)*.0205,1.0,-cos(pa)*.0279-cos(pb)*.009));`,
    );
  };
  const waveCompile = waterMaterial.onBeforeCompile;
  waterMaterial.onBeforeCompile = (shader) => {
    waveCompile(shader);
    shader.fragmentShader = shader.fragmentShader.replace(
      "#include <opaque_fragment>",
      `float waterFresnel=pow(1.0-clamp(dot(normal,normalize(vViewPosition)),0.0,1.0),3.0);
 outgoingLight=mix(outgoingLight,vec3(.31,.57,.66),waterFresnel*.24);
 diffuseColor.a=mix(.87,.98,waterFresnel);
 #include <opaque_fragment>`,
    );
  };
  const waterGeo = new THREE.PlaneGeometry(11.8, 7.12, 58, 36);
  waterGeo.rotateX(-Math.PI / 2);
  const water = mesh(waterGeo, waterMaterial, root, [5.45, 0, 0]);
  water.castShadow = false;
  water.renderOrder = 3;
  // Pool access ladder with curved rails and submerged rungs.
  for (const x of [8.4, 9.05])
    tube(
      [
        [x, 0.08, 4.15],
        [x, 0.5, 4.07],
        [x, 0.64, 3.8],
        [x, 0.59, 3.38],
        [x, 0.16, 3.16],
        [x, -1.4, 3.16],
      ],
      0.029,
    );
  for (let i = 0; i < 5; i++)
    tube(
      [
        [8.4, -0.25 - i * 0.27, 3.16],
        [9.05, -0.25 - i * 0.27, 3.16],
      ],
      0.025,
    );
  // Original springboard and support assembly, reconfigured to physical height.
  const upper = new THREE.Group();
  tower.add(upper);
  box([1.7, 0.2, 1.8], [-3.1, -0.17, 0], materials.concrete, upper, 0.05);
  box([1.64, 0.035, 1.73], [-3.1, -0.05, 0], materials.rim, upper, 0.025);
  const boardCarrier = new THREE.Group();
  upper.add(boardCarrier);
  boardCarrier.position.set(-1.275, 0, 0);
  const boardGeometry = new RoundedBoxGeometry(2.55, 0.11, 0.96, 2, 0.018),
    board = mesh(boardGeometry, materials.board, boardCarrier);
  const grip = box(
    [0.54, 0.006, 0.84],
    [0.975, 0.058, 0],
    materials.white,
    boardCarrier,
    0.003,
  );
  const gripLines = [];
  for (let i = 0; i < 10; i++)
    gripLines.push(
      0.725 + i * 0.052,
      0.062,
      -0.39,
      0.725 + i * 0.052,
      0.062,
      0.39,
    );
  const traction = new THREE.LineSegments(
    new THREE.BufferGeometry().setAttribute(
      "position",
      new THREE.Float32BufferAttribute(gripLines, 3),
    ),
    new THREE.LineBasicMaterial({
      color: "#849e9f",
      transparent: true,
      opacity: 0.75,
    }),
  );
  boardCarrier.add(traction);
  owned.add(traction.geometry);
  const fulcrum = mesh(
    new THREE.CylinderGeometry(0.13, 0.13, 0.8, 20),
    materials.steel,
    upper,
    [-1.9, -0.185, 0],
  );
  fulcrum.rotation.x = Math.PI / 2;
  box([0.88, 0.12, 1.0], [-2.14, -0.74, 0], materials.dark, upper, 0.025);
  const springs = [];
  for (const z of [-0.33, 0.33]) {
    const points = [];
    for (let i = 0; i <= 100; i++) {
      const a = (i / 100) * Math.PI * 10;
      points.push([
        -1.9 + Math.cos(a) * 0.066,
        i / 100,
        z + Math.sin(a) * 0.066,
      ]);
    }
    const spring = tube(points, 0.011, upper, materials.steel);
    spring.position.y = -0.67;
    spring.scale.y = 0.355;
    springs.push(spring);
  }
  const towerLegs = [];
  for (const x of [-3.66, -2.56])
    for (const z of [-0.64, 0.64]) {
      const column = box(
        [0.22, 1, 0.22],
        [x, 0.5, z],
        materials.concrete,
        tower,
        0.03,
      );
      towerLegs.push(column);
      box([0.4, 0.08, 0.4], [x, 0.11, z], materials.dark, tower, 0.03);
    }
  const braces = [];
  for (const z of [-0.64, 0.64])
    for (const reverse of [false, true]) {
      const g = mesh(
        new THREE.CylinderGeometry(0.036, 0.036, 1, 10),
        materials.steel,
        tower,
      );
      braces.push({ mesh: g, z, reverse });
    }
  for (const z of [-0.83, 0.83]) {
    tube(
      [
        [-3.84, 0.02, z],
        [-3.84, 0.82, z],
        [-3.7, 0.95, z],
        [-2.7, 0.95, z],
        [-2.54, 0.79, z],
        [-2.54, 0.01, z],
      ],
      0.026,
      upper,
    );
    tube(
      [
        [-3.81, 0.45, z],
        [-2.57, 0.45, z],
      ],
      0.022,
      upper,
    );
  }
  tube(
    [
      [-3.85, 0.78, -0.82],
      [-3.94, 0.87, -0.7],
      [-3.94, 0.87, 0.7],
      [-3.85, 0.78, 0.82],
    ],
    0.026,
    upper,
  );
  const accessRails = [
      mesh(
        new THREE.CylinderGeometry(0.031, 0.031, 1, 10),
        materials.steel,
        tower,
      ),
      mesh(
        new THREE.CylinderGeometry(0.031, 0.031, 1, 10),
        materials.steel,
        tower,
      ),
    ],
    rungs = [];
  for (let i = 0; i < 40; i++) {
    const rung = mesh(
      new THREE.CylinderGeometry(0.026, 0.026, 0.72, 10),
      materials.steel,
      tower,
    );
    rung.rotation.z = Math.PI / 2;
    rungs.push(rung);
  }
  let height = 5,
    deflection = 0;
  const up = new THREE.Vector3(0, 1, 0);
  const segment = (m, a, b) => {
    const p = new THREE.Vector3(...a),
      q = new THREE.Vector3(...b),
      d = q.clone().sub(p);
    m.position.copy(p).add(q).multiplyScalar(0.5);
    m.scale.y = d.length();
    m.quaternion.setFromUnitVectors(up, d.normalize());
  };
  function setHeight(value) {
    height = value;
    upper.position.y = value;
    towerLegs.forEach((c) => {
      c.position.y = (value - 0.35) / 2;
      c.scale.y = value - 0.35;
    });
    braces.forEach(({ mesh, z, reverse }) =>
      segment(
        mesh,
        [-3.66, reverse ? 0.4 : value - 0.45, z],
        [-2.56, reverse ? value - 0.45 : 0.4, z],
      ),
    );
    accessRails.forEach((r, i) =>
      segment(
        r,
        [-3.3 + (i ? -0.36 : 0.36), 0.15, 1.4],
        [-3.3 + (i ? -0.36 : 0.36), height + 0.85, 0.85],
      ),
    );
    rungs.forEach((r, i) => {
      const y = 0.3 + i * 0.3;
      r.visible = y < height - 0.05;
      r.position.set(-3.3, y, 1.4 - ((y - 0.15) / (height + 0.7)) * 0.55);
    });
  }
  function setBoard(value = 0, transform = null) {
    deflection = transform?.position ? transform.position[1] - height : value;
    fulcrum.position.y = -0.185 + deflection;
    springs.forEach((s) => (s.scale.y = Math.max(0.05, 0.355 + deflection)));
    // The physical model uses a rigid board on a vertical spring. Follow that
    // exact translation/rotation; do not invent an extra bending animation.
    if (transform?.position && transform?.quaternion) {
      boardCarrier.position.fromArray(transform.position);
      boardCarrier.position.y -= height;
      boardCarrier.quaternion.fromArray(transform.quaternion);
    } else {
      boardCarrier.position.set(-1.275, value, 0);
      boardCarrier.quaternion.identity();
    }
  }

  setHeight(5);
  return {
    root,
    water,
    setHeight,
    setBoard,
    renderWater(time) {
      uniform.value = time;
    },
    bounds: new THREE.Box3(
      new THREE.Vector3(-4.15, floorY - 0.16, -4.48),
      new THREE.Vector3(12.2, 0, 4.48),
    ),
    getHeight: () => height,
    dispose() {
      owned.forEach((g) => g.dispose());
      Object.values(materials).forEach((m) => m.dispose());
      waterMaterial.dispose();
      grout.material.dispose();
      traction.material.dispose();
      root.removeFromParent();
    },
  };
}
