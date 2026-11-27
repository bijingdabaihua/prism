import type { CameraView, InstancePlacement } from '../core/api/prism';

/**
 * Scene metadata: gallery cards plus the camera placements, environment maps
 * and geometry lists needed to assemble a scene elsewhere.
 *
 * Pure data: imports no scene builders or importers. Assembly logic lives in
 * `loadScene` in `assemble.ts`; the catalog itself can be safely read by both
 * the gallery and the render page.
 *
 * The scene assembly order is decided by the caller: `camera.set` first, then
 * `background.setImage` / `background.setRotation`, then geometry one by one,
 * and finally `camera.listen`.
 */

/** A procedural ground: one quad with a displacement material tiled into a grid by `tile`. */
export interface TiledGroundGeometry {
    type: 'tiled-ground';
    /** Material archive URL, relative to the site root: a zip with an embedded
     *  `.blend` and a texture directory. */
    material: string;
    /** Real-world size covered by one tile (meters), which is also the tile edge length. */
    tile: number;
    /** Tile-center range (meters). */
    x0: number;
    x1: number;
    z0: number;
    z1: number;
}

/** A glTF asset package: loaded whole; with `placement`, parsed by the caller and
 *  appended at that placement. */
export interface GltfGeometry {
    type: 'gltf';
    /** Asset package URL (`.gltf` / `.glb` / `.zip`), relative to the site root. */
    url: string;
    /** Omitted = keep the transforms in the file. */
    placement?: InstancePlacement;
}

export type SceneGeometry = TiledGroundGeometry | GltfGeometry | { type: 'glass-study' } | { type: 'emission-study' };

/** Scene parameters the renderer needs beyond geometry. */
export interface SceneSetup {
    /** Composition. A different camera placement is a different picture, so it is
     *  part of the scene. */
    camera: CameraView;
    /** Environment map URL, relative to the site root; the render page fetches it
     *  as a Blob for the Worker to decode. */
    env: string | null;
    /** Rotation of the environment map around +y (radians); omitted = source orientation. */
    envRotation?: number;
}

export interface SceneCard {
    title: string;
    /** One-line description on the card */
    subtitle: string;
    /** Technical tags on the card, so what it uses is visible at a glance */
    tags: string[];
    /** Gallery cover, relative to the site root; the card shows a placeholder
     *  pattern when omitted. Files live at assets/gallery/<key>.png */
    cover?: string;
    /** Show the emission brightness control in the workbench. */
    adjustEmission?: boolean;
    setup: SceneSetup;
    /** Geometry list; handed to `assemble.ts` for assembly in order. */
    geometry: SceneGeometry[];
}

/**
 * Sun direction.
 *
 * The three candidate environments have almost the same sun azimuth (all
 * measured at φ ≈ 36.1°); they differ only in elevation: venice_sunset 3.4°,
 * aarfontein 0.3°, sundowner 27.2°. In the first two the sun grazes the
 * horizon, so direct light on a horizontal surface is only cos(90°) ≈ 0 — the
 * ground relies on skylight alone and the image looks flat.
 *
 * So use sundowner, then rotate the environment by 34° to put the sun at
 * φ ≈ −80°: toward the back on the right of the frame. Azimuth only decides
 * which way the shadow sweeps — with the sun on the right, the shadow runs
 * along +z across the left of the frame. Its length is computed:
 * 0.44 / tan(27.2°) ≈ 0.86 m, enough to cross the left half of the frame.
 *
 * Rotate further forward and the shadow shrinks back under the jar, losing that
 * long shadow; rotate further back and the sun moves behind the jar, leaving
 * only skylight on the front and smearing the vase's blue-and-white pattern.
 */
const ENV_ROTATION = (Math.PI / 180) * (36.1 - 100) + Math.PI - 0.6;
const SUN_ROTATION = ENV_ROTATION + 0.6;

/**
 * Wasteland camera placement: 50 mm equivalent, 0.94 m off the ground,
 * 25.8° pitch, f/1.2.
 *
 * `fovY` is derived from the 20.25 mm film height of a 16:9 frame:
 * 2·atan(20.25/2/50) = 0.399. Once the field of view is fixed, aperture is the
 * only remaining degree of freedom in the lens's physical size: 50 mm at
 * f/1.2 gives an entrance pupil 41.7 mm across, radius 20.8 mm. Depth of field
 * is about 3 cm — the front of the jar is sharp, the back rim of its mouth
 * blurs by 3 px, dirt 2.5 m away by 10 px, dirt 40 m away by 22 px. Wider and
 * the jar's mouth goes soft; narrower and the background steals the scene again.
 *
 * The camera rises to 25.8°: the topmost ray of the frustum lands inside the
 * outer edge of the tiled ground (3.165 m), so the frame holds nothing but
 * dirt and leaves no gap for the environment map. Higher looks down too far
 * and the jar's mouth shrinks to an ellipse; lower and the top corners
 * immediately cross the tile edge.
 *
 * phi is the polar angle measured from +Y: at 64.2° the camera pulls back
 * 1.48 m horizontally and rises 0.72 m.
 */
const VASE_CAMERA: CameraView = {
    target: { x: 0, y: 0.22, z: -0.1 },
    distance: 1.646,
    theta: 0.0,
    phi: 1.12,
    fovY: 0.399,
    // Depth of field is off for now (originally 0.0208 / 1.522): each sample
    // jitters the aperture to another point on the disk, and out-of-focus areas
    // have far more variance, so the same spp looks noisier. Muted while tuning
    // materials; restore the two values below to bring it back — the shader
    // falls back to a pinhole when aperture_radius <= 0, so nothing else changes.
    apertureRadius: 0,
    focusDistance: 0,
};

/**
 * Wasteland geometry: a 3×3 grid of 2.11 m dirt tiles, a blue-and-white
 * porcelain jar, and a dead branch.
 *
 * Data only; the glTF files are zipped into asset packages and assembly is
 * `assemble.ts`'s job.
 */
const VASE_ON_DIRT_GEOMETRY: SceneGeometry[] = [
    {
        type: 'tiled-ground',
        material: 'textures/forest_ground_06.zip',
        // One texture tile spans 2.11 m; tile centers land at ±2.11, yielding
        // exactly 3×3 tiles, 6.33 m square.
        tile: 2.11,
        x0: -2.11,
        x1: 2.11,
        z0: -2.11,
        z1: 2.11,
    },
    {
        type: 'gltf',
        url: 'model/antique_ceramic_vase_01_1k.zip',
        // The jar faces the origin: its base is at y = 0, sitting directly on the dirt.
        placement: { position: { x: 0, y: 0, z: 0 }, rotationY: 0.4 },
    },
    {
        type: 'gltf',
        url: 'model/dead_quiver_branch_01_1k.zip',
        // The branch lies on the ground: tipped 90° around X, then rotated 22°
        // around Y to lie diagonally into the lower-right of the frame. The 9 cm
        // lift compensates for the lowest point after tipping (the side that was
        // toward −z) at y = −0.097; 7 mm less leaves it slightly sunk into the
        // dirt rather than perched on glass.
        placement: {
            position: { x: 0.04, y: 0.09, z: -0.27 },
            rotationX: -Math.PI / 2,
            rotationY: 0.38,
        },
    },
];

export const SCENE_CARDS: Readonly<Record<string, SceneCard>> = {
    'emission-study': {
        title: 'Emissive Panel',
        subtitle: 'An emissive panel lights the ground as a block casts a soft shadow',
        tags: ['Emissive material', 'Area light sampling', 'Soft shadows'],
        cover: 'gallery/emission-study.png',
        adjustEmission: true,
        setup: {
            camera: {
                target: { x: -0.42, y: 0.80, z: 0 },
                distance: 4.5,
                theta: 0.88,
                phi: 1.05,
                fovY: 0.60,
                apertureRadius: 0,
                focusDistance: 0,
            },
            env: null,
        },
        geometry: [{ type: 'emission-study' }],
    },
    'glass-study': {
        title: 'Glass Sphere',
        subtitle: 'See the checkerboard refract through the glass sphere and the environment reflect around its rim',
        tags: ['Glass transmission', 'Fresnel reflection', 'Procedural geometry'],
        cover: 'gallery/glass-study.png',
        setup: {
            camera: {
                target: { x: 0, y: 0.46, z: 0 },
                distance: 3.5,
                theta: 0.3,
                phi: 0.88,
                fovY: 0.52,
                apertureRadius: 0,
                focusDistance: 0,
            },
            env: 'env/sundowner_overlook_2k.exr',
            envRotation: SUN_ROTATION,
        },
        geometry: [{ type: 'glass-study' }],
    },
    'vase-on-dirt': {
        title: 'Wasteland',
        subtitle: 'A blue-and-white porcelain jar rests on sunlit dirt that stretches past the frame, leaving only earth and light',
        tags: ['Instanced ground', 'Displacement geometry', 'glTF assets'],
        cover: 'gallery/vase-on-dirt.png',
        setup: {
            camera: VASE_CAMERA,
            env: 'env/sundowner_overlook_2k.exr',
            envRotation: SUN_ROTATION,
        },
        geometry: VASE_ON_DIRT_GEOMETRY,
    },
};
