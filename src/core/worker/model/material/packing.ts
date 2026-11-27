import type { PBRMaterial } from './material';

/**
 * IR → GPU fixed-size layout packing.
 *
 * This is the only place that flattens PBRMaterial into fixed-length floats and puts atlas
 * coordinates into slots; shader-side indices correspond one-to-one, so changing one side
 * requires changing the other.
 *
 * Layout (33 floats / 132 B, slot indices noted in the comments):
 *   0–2   baseColor            3  metallic
 *   4     roughness            5  normalScale
 *   6     occlusionStrength    7  padding
 *   8–11  baseColor TexRef    12–15 roughness TexRef
 *   16–19 normal TexRef       20–23 occlusion TexRef
 *   24–27 metallic-roughness TexRef
 *   28    transmission          29 ior
 *   30–32 emission RGB
 *
 * The file has two layers: contract (all exports) and implementation (no exports);
 * see AGENTS.md for the convention.
 */

// ══════════════════ Contract layer (exports) ═════════════════

/**
 * Position of a texture in a texture_2d_array.
 *
 * Textures are split into four arrays by kind (albedo / normal / roughness / occlusion), so a
 * reference only needs a layer number; at sampling time UVs are multiplied by scale — geometry
 * UVs are in meters, and scale is "repeats per meter", derived from the texture's own meters
 * per tile (`Image.meters`), so texture density on large surfaces like walls does not vary with
 * geometry size.
 */
export interface TexRef {
    layer: number;
    scale: number;
    has: number;
}

export interface MaterialTexSlots {
    baseColor?: TexRef;
    roughness?: TexRef;
    normal?: TexRef;
    occlusion?: TexRef;
    /** Packed metallic-roughness (G = rough, B = metal), stored in the roughness array. */
    metallicRoughness?: TexRef;
}

/** Fixed record size of a material in the SSBO (float count); layout is in the file header. */
export const PBR_MAT_FLOATS = 33;

/** PBRMaterial + texture slots → fixed GPU record (PBR_MAT_FLOATS floats). */
export let packPBRMaterial: (m: PBRMaterial, tex?: MaterialTexSlots) => Float32Array;

// ═════════════ Implementation layer (not exported) ════════════

/** TexRef → 4 floats. Without a texture, layer=0, scale=1, has=0. */
function packTexRef(t: TexRef | undefined): number[] {
    return t ? [t.layer, t.scale, t.has, 0] : [0, 1, 0, 0];
}

packPBRMaterial = (m, tex) => {
    const c = m.baseColor;
    const bc = packTexRef(tex?.baseColor);
    const ro = packTexRef(tex?.roughness);
    const n = packTexRef(tex?.normal);
    const ao = packTexRef(tex?.occlusion);
    const mr = packTexRef(tex?.metallicRoughness);
    return new Float32Array([
        c.r,
        c.g,
        c.b,
        m.metallic,
        m.roughness,
        m.normalScale,
        m.occlusionStrength,
        0.0,
        bc[0],
        bc[1],
        bc[2],
        bc[3],
        ro[0],
        ro[1],
        ro[2],
        ro[3],
        n[0],
        n[1],
        n[2],
        n[3],
        ao[0],
        ao[1],
        ao[2],
        ao[3],
        mr[0],
        mr[1],
        mr[2],
        mr[3],
        m.transmission,
        m.ior,
        m.emission.r,
        m.emission.g,
        m.emission.b,
    ]);
};
