# Prism

A WebGPU path tracer written from scratch. The front end is a zero-interaction gallery; v1 aims to add a command interface that both people (the workbench page) and AI (MCP) can call.
It does not use the three.js render pipeline or a third-party BVH; the render page image is computed in the browser, and gallery covers are static preview images.

- Live demo: https://bijingdabaihua.github.io/prism/
- Stack: TypeScript, WGSL, WebGPU, Web Workers, Vite, glTF 2.0, EXR
- Assets: Poly Haven (CC0); new assets keep the same license

## Current phase and priorities (about 6 weeks)

This project is first and foremost a portfolio piece. When deciding whether a change is worth making, check whether it improves at least one of **visual quality, speed (with numbers), or usability**.

Scenes continue to be assembled by the app layer through the existing API. Add operations as needed when wiring up the workbench and AI. Only consider a JSON format once there is a need to save, share or batch-load scenes.

1. **Weeks 1–2, close the rendering gaps (done)**: smooth glass transmission is done and validated by the "Glass Sphere" scene. Emissive materials and direct sampling of emissive geometry are in, validated by the "Emissive Panel" scene.
2. **Weeks 3–4, workbench and AI (current)**: the artwork page offers a text command panel in the top-right corner, collapsed by default. First validate changing a material's color per object, then connect emission brightness. Voice and image export are deferred.
3. **Weeks 5–6, packaging**: put performance numbers and the AI screen recording in the first screen of the README, and write a one-page technical note.

**Not in v1**: scene editor, animation/skinning/Draco/meshopt, volumetric rendering, spectral rendering, accounts/cloud, npm package.
**If time runs short**: cut the number of materials and presets first; glass and AI stay.

## Architecture

```
src/
  app/          app layer: routing, gallery, render page, fallback page, scene catalog and assembly
  core/api/     Prism facade: main-thread RPC, message protocol, handles
  core/worker/  Worker entry, camera, environment lighting, models, image decoding and path tracing
  core/math/    vectors, matrices, colors; shared by app and core
```

### AI workbench design

- `src/app/` manages the panel and the current scene, and exposes stable identifiers, available operations and current values of editable objects to the Agent.
- `src/agent/` is planned. The Agent sends the text and scene capabilities to the model API, validates the target, operation and parameters in structured commands, then executes them through `src/core/api/`.
- `src/core/api/` provides render commands shared by humans and the Agent. `src/core/worker/` updates render state and resets accumulation.
- The current public demo connects the browser directly to DeepSeek using a dedicated public demo credential. The first version lets the user change the color of one opaque sphere in the current scene with text, and shows the result.

### Hard rules (do not break)

- **Dependencies point one way, downward**: app → core → math. core does not import app, and math does not import anything above it.
- **The main thread never touches the GPU**: all WebGPU calls happen in the Worker. The main thread only communicates through the `prism.ts` facade and receives `ImageBitmap` frames via `camera.listen()`.
- **`src/core/worker/worker.ts` assembles the render modules**; the scene catalog stays in the app layer, so do not let core depend on app.
- **All new capabilities go through commands**: add typed commands to the `prism.ts` facade and the message protocol. UI and AI use the same commands; neither gets a back door.
- **No** three.js, and no third-party BVH or render library.

## Rendering status (read before changing render code)

- Intersection: two-level BVH (TLAS/BLAS), built on the CPU, flattened into a 1D buffer and traversed iteratively on the GPU. The build uses 16-bucket SAH binning, falls back to median split beyond depth 40, and leaves hold at most 2 primitives. Triangle intersection uses Möller–Trumbore in local space.
- Lighting: NEE with brightness-CDF importance sampling of the environment map; MIS with the power heuristic, combining BSDF sampling and light sampling; Russian roulette termination after the 4th bounce.
- Accumulation: 1 spp progressive accumulation per frame, reset only when the camera or parameters change. The number of pixels sampled per frame adapts to measured GPU time, and the sampling order ensures every pixel is covered once per round.
- Other: 2K EXR environment maps; thin-lens depth of field (aperture and focus distance in meters); displacement maps are baked into real geometry through tessellation and vertex displacement before being put into the BVH; material textures use `texture_2d_array` with full mip chains (diffuse, normal, roughness, AO).
- Known limitations: glTF only supports triangles and the 4 texture types above; no transparency or alpha cutout; smooth glass transmission is supported, but rough transmission and volumetric absorption are not yet; double-sided emissive geometry is supported, but there are no separate point lights, spotlights or similar light types yet.

## Common tasks

**Add a scene**: add the camera placement, environment map, geometry and card info to `SCENE_CARDS` in `src/app/catalog.ts`; extend `src/app/assemble.ts` as needed, then put a cover image at `assets/gallery/<key>.png`.

**Commands**: `npm install` installs dependencies, `npm run dev` runs the dev server, `npm run build` builds, `npm run preview` previews the build output. There is no test script yet.

## How we work

- Top-down: confirm the problem and the desired result first, then define the interface, then implement. Before touching the architecture, write out the plan and wait for confirmation; do not make large changes directly.
- Run the reference-image regression test after changing render code. Until the test exists, take screenshots in the gallery scenes and compare before and after.
- Performance-related changes must report before and after numbers: GPU frame time, samples per second, and time to converge to something presentable.
- After changes, keep every gallery scene opening correctly, and browsers without WebGPU support still see the fallback static image.

## Open questions

- [ ] AI integration: MCP + WebSocket connected to the page, or a headless browser running the renderer
- [ ] How many seconds the Wasteland scene needs to converge to presentable (used to set the acceptance number)

## Code conventions

Prism project TypeScript / WGSL code conventions. All AI-generated code must follow them.

### TypeScript

#### File structure: two layers

Every module file is fixed at two layers, separated by a divider line:

```typescript
// ═══════════════ Contract layer (exports) ══════════════

/** Construction entry: declarations only (types + docs), no implementation */
export let createXxx: (deps: XxxDeps) => XxxHandle;

export interface XxxHandle { ... }      // public contract
export interface XxxDeps { ... }        // dependency surface
export type XxxRequest = ...;           // shapes referenced by the framework (e.g. messages)

// ═════════ Implementation layer (not exported) ══════════

class XxxImpl implements XxxHandle { ... }   // private classes, private types

createXxx = (deps) => new XxxImpl(deps);     // assignment of contract-layer variables
```

- **Contract layer = all exports.** It is the module's entire promise: reading this layer alone is enough to know the whole module, with no need to read the implementation layer.
  - Values: only declarations of "types + docs", **with no implementation at all** (not even a one-line delegation);
  - Types: public contracts, dependency surfaces, shapes referenced by the framework (e.g. messages).
- **Implementation layer = all non-exports.** It is the entire detail of the module: private classes, private types, and assignments to contract-layer variables. As long as the contract stays the same, this layer may be refactored freely.
- A symbol's layer is decided by one question only: **is it exported?** Exported means contract layer; not exported means implementation layer. No exceptions.
- Reference direction: the implementation layer may reference the contract layer; the contract layer must not reference any symbol from the implementation layer.
- The standard form of an exported value is `export let` + an assignment in the implementation layer. Read-only access from outside is guaranteed by ESM (import bindings cannot be reassigned); within the module, contract-layer variables are never reassigned, by convention.
  The reason for not using `const`: `const` must be initialized at its declaration, and the initializer is implementation, which is mutually exclusive with "zero implementation in the contract layer".
- Reference example: `src/core/api/background.ts`.

**Modules centered on an exported class** (such as IR data classes): the class is not exported directly and is likewise split into two layers—

- Contract layer: `export interface Xxx` (all fields + method signatures; docs live only here) + `export let Xxx` (the class binding; only the constructor signature and static method signatures, no implementation) + other exported types;
- Implementation layer: `class XxxImpl implements Xxx` (fields carry no docs; consistency with the interface is enforced at compile time by `implements`) + an `Xxx = XxxImpl` assignment at the bottom;
- The interface and the class binding coexist under the same name (type space + value space); callers' `new Xxx()`, `Xxx.static()` and `: Xxx` usages all stay unchanged;
- Reference example: `src/core/worker/model/material/material.ts`.

#### Imports

- All in-project imports use relative paths
- Type-only imports go first

#### Class design

- All members explicitly marked `public` / `private`; omission is not allowed
- No getters/setters; use ordinary methods everywhere (`width()` rather than `get width()`)

#### Class member order

The more open a member, the earlier it goes; follow this order strictly:

```
public static
public properties
public readonly
private properties
private readonly
constructor
public methods
private methods
```

**Property region rules:**
- No blank lines between properties
- No blank lines between visibility groups either (public / public readonly / private / private readonly)
- The whole property region is continuous from top to bottom, with no blank lines inserted

**Method region rules:**
- Method bodies must be expanded across lines; even a one-line return is not allowed in single-line form
  ```typescript
  // ❌ Wrong
  public width(): number { return this.w; }

  // ✅ Correct
  public width(): number {
    return this.w;
  }
  ```
- Methods must be separated by a blank line

#### Comments

- Write short comments only at key logic
- Layer dividers use the format defined in "File structure"; decorative dividers and ASCII structure boxes are otherwise still forbidden
- Do not write comments that repeat what the code already says
- Self-check after changes

### WGSL

- Variable/function names use `snake_case`
- Struct names use `PascalCase`
