import type { Vec3 } from '../core/math/types';
import { Instance } from '../core/worker/model/instance';
import { PBRMaterial } from '../core/worker/model/material/material';
import { MeshBuilder } from '../core/worker/model/mesh';
import type { TiledGroundGeometry } from './catalog';

/**
 * Dirt ground — the surface of the wasteland scene.
 *
 * A single plane cannot cover this scene: at a low camera angle the top of the
 * frame nearly touches the horizon, and the edge of a 1 m square slab becomes
 * visible just 2 m away. So this is "one ground tile + instancing"; see
 * `buildTiledGround`.
 *
 * The material all lives in the archive (`.blend` + texture directory) — the
 * four textures, color space, normal strength and displacement Midlevel / Scale
 * are all authored values. This file supplies only the number a file cannot
 * provide: how large one texture tile is (Poly Haven exposes it as `dimensions`
 * in its web API, not in the .blend), taken from the recipe.
 *
 * Displacement is a property of the material, not the geometry: only a flat quad
 * is laid out here; tessellation and lifting happen at compile time in
 * `Mesh.evolve`.
 */

/** Horizontal quad; winding keeps the normal facing +y (matching addQuad's convention). */
function groundQuad(x0: number, x1: number, z0: number, z1: number): [Vec3, Vec3, Vec3, Vec3] {
    return [
        { x: x0, y: 0, z: z1 },
        { x: x1, y: 0, z: z1 },
        { x: x1, y: 0, z: z0 },
        { x: x0, y: 0, z: z0 },
    ];
}

/**
 * A large ground paved from one tile: one mesh, many instances, sharing one
 * BLAS.
 *
 * Why split it up: a ground with 4 mm tiles cannot be cut in one go. A 42 m
 * square is 200 million triangles and 3 GB of vertices alone. Cut into small
 * tiles, the geometry is computed once (about 520k triangles for one wasteland
 * tile), and every additional tile only adds an instance entry — `blasCache`
 * caches per mesh, so instances share the same tree.
 *
 * Why the tile edge must equal the texture period: displacement is sampled from
 * the texture, which repeats every `tile` meters. When the tile edge equals that
 * period, the UVs at neighboring tile boundaries wrap from `tile` back to 0,
 * both sides sample the same height, and the displacement and pattern join up on
 * their own — the paved result matches a single large ground vertex for vertex.
 * The wasteland uses 2.11 m; with an integer 2 m the seams shift by 0.11 m and
 * draw a visible line.
 *
 * Tile centers sit at (i·tile, 0, j·tile), so the origin is a tile center and the
 * jar stands exactly on the middle tile. The range is selected by **tile
 * center**: only tiles whose center falls in [x0, x1] are kept. Two ground
 * sections adjusted with adjacent x0 / x1 meet exactly on a tile seam, with no
 * overlap and no gap.
 *
 * The material archive is fetched by the caller and passed in: this file only
 * builds geometry and does no networking.
 */
export async function buildTiledGround(
    materialZip: Blob,
    recipe: TiledGroundGeometry,
): Promise<Instance[]> {
    // tile sets both UV scale and tile edge length; the .blend cannot supply it,
    // so correct it after construction
    const material = (await PBRMaterial.buildByBlend(materialZip)).retile(recipe.tile);
    const h = recipe.tile * 0.5;
    const builder = new MeshBuilder();
    builder.addQuad(groundQuad(-h, h, -h, h), 0);
    const mesh = builder.build([material], 'ground-tile');

    const instances: Instance[] = [];
    for (let i = Math.ceil(recipe.x0 / recipe.tile); i <= Math.floor(recipe.x1 / recipe.tile); i++) {
        for (let j = Math.ceil(recipe.z0 / recipe.tile); j <= Math.floor(recipe.z1 / recipe.tile); j++) {
            const m = new Float32Array(16);
            m[0] = m[5] = m[10] = m[15] = 1;
            m[12] = i * recipe.tile;
            m[14] = j * recipe.tile;
            instances.push(new Instance(mesh, m));
        }
    }
    return instances;
}
