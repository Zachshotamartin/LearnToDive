# Third-party notices

The experiment code, policy training code, learned weights, diver mesh and pool artwork are original work for this project. No external demonstration dataset, behavior code or pretrained policy checkpoint is used.

- **MuJoCo 3.13.0** — Google DeepMind and contributors; Apache License 2.0. Official native engine and unchanged official WASM binary. [Source and license](https://github.com/google-deepmind/mujoco). The CSP-compatible JavaScript binding adapter is reproducibly generated from the pinned official glue using Emscripten’s static wrapper templates; see [the vendor README](src/vendor/README.md), included licenses and `scripts/build-mujoco-csp.mjs`.
- **Three.js** — Three.js authors; MIT License. [Source and license](https://github.com/mrdoob/three.js). Includes OrbitControls, RoomEnvironment, GLTFLoader and geometry helpers from the same distribution.
- **PPO** is an algorithm reference, not borrowed implementation code: [Schulman et al., Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347).

The original Blender asset generator is included as source. Blender is an authoring tool and is not distributed with the runtime package.
