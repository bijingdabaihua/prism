import type { AABB, Vec3 } from '../../math/types';
import type { PBRMaterial } from './material/material';
import type { Image } from './image';
import { sub, dot, cross, normalize } from '../../math/vec3';

/**
 * Mesh IR: geometry plus material references, with no GPU or acceleration structures.
 *
 * BLAS is derived data (buildMeshBvh) and belongs to the compile layer; it is not part of the IR.
 * The one exception is `Mesh.evolve`, which bakes displacement maps from materials into real
 * geometry; see the interface comment.
 *
 * The file has two layers: contract (all exports) and implementation (no exports);
 * see AGENTS.md for the convention.
 */

// ═════════════════ Contract layer (exports) ══════════════════

/**
 * Class binding for Mesh: builds a mesh from pre-packed vertex and triangle data.
 *
 * vertexData is interleaved: pos(3) + normal(3) + uv(2) + tangent(4) = 12 floats/vertex.
 * triangleData holds 4 uint32 per triangle: v0, v1, v2, matIdx. Bounds are computed from
 * the vertices.
 * For procedural construction, use `MeshBuilder`.
 *
 * This is only a declaration; the assignment lives in the implementation layer.
 */
export let Mesh: {
    new (vertexData: Float32Array, triangleData: Uint32Array, materials: readonly PBRMaterial[], name?: string): Mesh;
};

/**
 * Mesh IR: geometry plus material references, with no GPU or acceleration structures.
 *
 * BLAS is derived data (buildMeshBvh) and belongs to the compile layer; it is not part of the IR.
 *
 * The one exception is `evolve()`, which bakes displacement maps from materials into real
 * geometry. A displacement map is a runtime property of the material (supplied by the importer
 * or the scene author), while what a piece of geometry looks like must be settled here: the
 * BVH, bounding box and instance placement all need to read the evolved vertices.
 */
export interface Mesh {
    /** Display name (file name or in-scene label); not used in any computation. */
    readonly name?: string;
    /** Interleaved: pos(3) + normal(3) + uv(2) + tangent(4) = 12 floats/vertex. */
    readonly vertexData: Float32Array;
    /** 4 uint32 per triangle: v0, v1, v2, matIdx. */
    readonly triangleData: Uint32Array;
    /** PBR materials used by this mesh (matIdx in triangleData indexes into this). */
    readonly materials: readonly PBRMaterial[];
    /** Local-space AABB. */
    readonly bounds: AABB;

    /**
     * Evolves the mesh by consuming displacement maps from its materials, returning a mesh
     * ready to render.
     *
     * Triangles with a displacement map are tessellated into a grid at one texel per cell;
     * their vertices are raised according to the material's mid/scale, and normals are
     * recomputed from the height field. Triangles without one keep every vertex untouched.
     * On the returned mesh those materials are copies with `displacement` removed, so no
     * displacement map reaches the GPU.
     *
     * The result is cached on the mesh: multiple instances of the same mesh share one evolved
     * result and therefore one BLAS. A mesh without displacement is returned as-is (=== itself).
     */
    evolve(): Mesh;
}

/**
 * Class binding for MeshBuilder: creates an empty procedural geometry builder.
 *
 * This is only a declaration; the assignment lives in the implementation layer.
 */
export let MeshBuilder: {
    new (): MeshBuilder;
};

/**
 * Procedural scenes: build Instances directly in code without going through any file.
 *
 * Use it as a reference scene for the render pipeline: when the image is wrong, OBJ parsing
 * can be ruled out immediately.
 */
export interface MeshBuilder {
    /**
     * Quad. UVs are in **meters** (along a→b and perpendicular within the face), so the
     * texture repeat count depends only on each map's meters-per-cell (`Image.meters`), not on
     * the quad's size: an 8 m wall and a 1 m plane using the same material keep a continuous
     * texture density.
     *
     * If `uv` is given, coordinates use that projection instead. The default "along the edges"
     * projection switches axes on folded surfaces: where the ground folds into a slope, the
     * texture direction turns 90° with it and the pattern breaks at the seam.
     */
    addQuad(corners: [Vec3, Vec3, Vec3, Vec3], material: number, uv?: UvProjection): void;

    /** Packs into a Mesh. Vertices are not shared across quads. */
    build(materials: readonly PBRMaterial[], name?: string): Mesh;
}

/** Texture coordinates (meters): a vertex's u/v is its offset from the origin projected
 * onto these two axes. */
export interface UvProjection {
    origin: Vec3;
    /** Unit vector: the map's u direction. */
    uAxis: Vec3;
    /** Unit vector: the map's v direction. */
    vAxis: Vec3;
}

// ════════════ Implementation layer (not exported) ═════════════

/**
 * A material's displacement splits into two parts: the map plus two scalars (mid, meters per
 * cell). The cell size need not be supplied; the map determines it.
 *
 * One texel covers `meters / width` meters; use that as the tessellation cell size. Cutting
 * finer samples the same texel several times (wasted triangles); cutting coarser squeezes
 * several texels into one cell (aliasing into spikes).
 *
 * The map must be downsampled to a resolution the grid can sample (`prepareHeightImage`)
 * before it gets here. This happens here rather than at material load time because how far
 * to downsample depends on how many triangles need cutting, which the material does not know.
 */
interface Displacement {
    image: Image;
    /** Height midpoint; see `PBRMaterial.displacementMid`. */
    mid: number;
    /** Height scale (meters); see `PBRMaterial.displacementScale`. */
    scale: number;
    /** Tessellation cell size (meters) = edge length of one texel in the map. */
    cell: number;
}

function displacementOf(material: PBRMaterial): Displacement | undefined {
    const image = material.displacement;
    if (!image) return undefined;
    const { displacementMid: mid, displacementScale: scale } = material;
    if (mid === undefined || scale === undefined) {
        throw new Error('A material with a displacement map must provide displacementMid and displacementScale');
    }
    const prepared = prepareHeightImage(image);
    return { image: prepared, mid, scale, cell: (prepared.meters ?? 1) / prepared.width };
}

// Displacement height field

/**
 * Displacement map → height field for the mesh: downsample once, then hand it to geometry.
 *
 * The grid cell size is the edge length of one texel **after** downsampling (see
 * `displacementOf`), so this step decides how dense the mesh is. Using the original map has
 * two problems: scan noise and JPEG blocks are finer than the grid and get pulled into spikes
 * as if they were real relief, while doubling the grid density simply wastes triangles.
 * After halving, one texel matches one cell exactly.
 *
 * How far to downsample depends on how many triangles need cutting, which the map itself
 * cannot know, so only the evolve step should call this. When a material is shared by
 * multiple meshes the result is cached by source image, so one downsample is enough.
 */
function prepareHeightImage(image: Image): Image {
    const cached = preparedImages.get(image);
    if (cached) return cached;
    const prepared = filterHeightImage(image, Math.max(1, Math.floor(image.width / 2)));
    preparedImages.set(image, prepared);
    return prepared;
}

/** Source image → downsampled result. Displacement maps are shared by multiple instances
 * at load time, so one downsample is enough. */
const preparedImages = new WeakMap<Image, Image>();

/**
 * Downsamples with a 2×2 box filter to within targetSize.
 *
 * Sampling theorem: vertex spacing must be less than half the Nyquist wavelength. The size
 * ratio between displacement map and grid directly determines whether aliasing occurs; one
 * texel per cell is the safe upper limit.
 */
function filterHeightImage(image: Image, targetSize: number): Image {
    let width = image.width;
    let height = image.height;
    let data = image.data;
    while (width > targetSize || height > targetSize) {
        const nw = Math.max(1, width >> 1);
        const nh = Math.max(1, height >> 1);
        const out = new Float32Array(nw * nh * 3);
        for (let y = 0; y < nh; y++) {
            const y0 = Math.min(y * 2, height - 1);
            const y1 = Math.min(y * 2 + 1, height - 1);
            for (let x = 0; x < nw; x++) {
                const x0 = Math.min(x * 2, width - 1);
                const x1 = Math.min(x * 2 + 1, width - 1);
                const a = (y0 * width + x0) * 3;
                const b = (y0 * width + x1) * 3;
                const c = (y1 * width + x0) * 3;
                const d = (y1 * width + x1) * 3;
                const o = (y * nw + x) * 3;
                out[o] = (data[a] + data[b] + data[c] + data[d]) * 0.25;
                out[o + 1] = (data[a + 1] + data[b + 1] + data[c + 1] + data[d + 1]) * 0.25;
                out[o + 2] = (data[a + 2] + data[b + 2] + data[c + 2] + data[d + 2]) * 0.25;
            }
        }
        width = nw;
        height = nh;
        data = out;
    }
    return { ...image, width, height, data };
}

/** Bilinearly samples the height map (grayscale stored as RGB; any channel works); UVs wrap
 * with repeat. */
function sampleHeight(image: Image, u: number, v: number): number {
    const w = image.width;
    const h = image.height;
    const x = (((u % 1) + 1) % 1) * w - 0.5;
    const y = (((v % 1) + 1) % 1) * h - 0.5;
    const x0 = Math.floor(x);
    const y0 = Math.floor(y);
    const fx = x - x0;
    const fy = y - y0;
    const px = (i: number) => ((i % w) + w) % w;
    const py = (i: number) => ((i % h) + h) % h;
    const a = image.data[(py(y0) * w + px(x0)) * 3];
    const b = image.data[(py(y0) * w + px(x0 + 1)) * 3];
    const c = image.data[(py(y0 + 1) * w + px(x0)) * 3];
    const d = image.data[(py(y0 + 1) * w + px(x0 + 1)) * 3];
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy;
}

/**
 * Displacement in meters: `(height - mid) × scale`, the same definition as the Midlevel and
 * Scale of Blender's Displacement node. Both values come from the material; nothing is
 * measured or rescaled here.
 *
 * Earlier versions auto-calibrated from the 5–95th percentiles and used tanh to clamp the
 * tails, on the grounds that scanned maps do not store absolute height. But assets do supply
 * an absolute scale (it is in the material author's .blend), so measuring it again only
 * replaces the author's choice with a guess.
 *
 * u/v are metric UVs, folded into texture repeats using the map's meters per cell
 * (`image.meters`). Only UVs matter: on one triangle, places with the same UV rise equally
 * regardless of position.
 */
function sampleDisplacement(image: Image, u: number, v: number, mid: number, scale: number): number {
    const meters = image.meters ?? 1;
    return (sampleHeight(image, u / meters, v / meters) - mid) * scale;
}

class MeshImpl implements Mesh {
    public readonly name?: string;
    public readonly vertexData: Float32Array;
    public readonly triangleData: Uint32Array;
    public readonly materials: readonly PBRMaterial[];
    public readonly bounds: AABB;
    private evolved?: Mesh;

    constructor(vertexData: Float32Array, triangleData: Uint32Array, materials: readonly PBRMaterial[], name?: string) {
        this.vertexData = vertexData;
        this.triangleData = triangleData;
        this.materials = materials;
        this.name = name;
        this.bounds = computeBounds(vertexData);
    }

    public evolve(): Mesh {
        if (!this.materials.some((m) => m.displacement)) return this;
        if (!this.evolved) this.evolved = evolveMesh(this);
        return this.evolved;
    }
}

/** Bakes displacement maps into geometry; the implementation of `evolve()`. */
function evolveMesh(mesh: Mesh): Mesh {
    const vd = mesh.vertexData;
    const td = mesh.triangleData;
    // Material copies with the displacement map and related scalars consumed, all other
    // fields intact
    const materials = mesh.materials.map((m) => m.withoutDisplacement());

    const verts: number[] = [];
    const tris: number[] = [];
    for (let t = 0; t < td.length; t += 4) {
        const ia = td[t];
        const ib = td[t + 1];
        const ic = td[t + 2];
        const mat = td[t + 3];
        const displacement = displacementOf(mesh.materials[mat]);

        if (!displacement) {
            const base = verts.length / 12;
            for (const v of [ia, ib, ic]) {
                for (let k = 0; k < 12; k++) verts.push(vd[v * 12 + k]);
            }
            tris.push(base, base + 1, base + 2, mat);
            continue;
        }

        tessellate(vd, ia, ib, ic, mat, displacement, verts, tris);
    }

    return new MeshImpl(new Float32Array(verts), new Uint32Array(tris), materials, mesh.name);
}

/**
 * Lays out a triangle as a barycentric grid: divide each edge by the longest edge / cell
 * size, raise grid points by `displacement`, and recompute normals from the height field.
 *
 * Both adjacent triangles use the same subdivision count, so shared edges line up and
 * neighboring triangles do not crack apart. Vertices are not shared across triangles (each
 * cell carries its own), at the cost of duplicate vertices near edges; in return there is no
 * welding to do and normals do not break at seams.
 */
function tessellate(
    vd: Float32Array,
    ia: number,
    ib: number,
    ic: number,
    mat: number,
    displacement: Displacement,
    verts: number[],
    tris: number[],
): void {
    const a = vd.subarray(ia * 12, ia * 12 + 12);
    const b = vd.subarray(ib * 12, ib * 12 + 12);
    const c = vd.subarray(ic * 12, ic * 12 + 12);
    const longest = Math.max(edgeLength(a, b), edgeLength(a, c), edgeLength(b, c));
    const n = Math.max(1, Math.ceil(longest / displacement.cell));

    // Triangle grid laid out by rows: row j has n+1-j vertices (barycentric j/n)
    const rowStart = new Int32Array(n + 1);
    let cursor = verts.length / 12;
    for (let j = 0; j <= n; j++) {
        rowStart[j] = cursor;
        cursor += n + 1 - j;
    }

    const v = new Float32Array(12);
    for (let j = 0; j <= n; j++) {
        for (let i = 0; i <= n - j; i++) {
            const wb = i / n;
            const wc = j / n;
            const wa = 1 - wb - wc;
            for (let k = 0; k < 12; k++) v[k] = a[k] * wa + b[k] * wb + c[k] * wc;
            displace(v, displacement);
            for (let k = 0; k < 12; k++) verts.push(v[k]);
        }
    }

    const at = (j: number, i: number) => rowStart[j] + i;
    for (let j = 0; j < n; j++) {
        for (let i = 0; i < n - j; i++) {
            const p0 = at(j, i);
            const p1 = at(j, i + 1);
            const p2 = at(j + 1, i);
            tris.push(p0, p1, p2, mat);
            if (i < n - 1 - j) {
                const p3 = at(j + 1, i + 1);
                tris.push(p1, p3, p2, mat);
            }
        }
    }
}

/**
 * Raises one vertex and recomputes its normal (writes `v` in place).
 *
 * Normals are not averaged from neighboring triangles; that information only exists after
 * welding. Instead this uses an analytic gradient: displacement is a continuous function of
 * UV, so derivatives along u and v are central-differenced directly, and the surface normal
 * is the cross product of the two tangents. Normals stay smooth even on a coarse grid.
 */
function displace(v: Float32Array, d: Displacement): void {
    const u = v[6];
    const w = v[7];
    const x = v[0];
    const y = v[1];
    const z = v[2];

    // Displace along the vertex normal: height difference 1.0 equals d.scale meters; mid
    // sets the zero
    const nx = v[3];
    const ny = v[4];
    const nz = v[5];
    const offset = sampleDisplacement(d.image, u, w, d.mid, d.scale);
    v[0] = x + nx * offset;
    v[1] = y + ny * offset;
    v[2] = z + nz * offset;

    // Tangent basis: normal plus the tangent along u; bitangent = w × (n × t)
    const sign = v[11] < 0 ? -1 : 1;
    let tx = v[8];
    let ty = v[9];
    let tz = v[10];
    const along = nx * tx + ny * ty + nz * tz;
    tx -= nx * along;
    ty -= ny * along;
    tz -= nz * along;
    const len = Math.hypot(tx, ty, tz);
    if (len < 1e-6) return; // Degenerate tangent: keep the normal as-is rather than divide by zero
    tx /= len;
    ty /= len;
    tz /= len;
    const bx = (ny * tz - nz * ty) * sign;
    const by = (nz * tx - nx * tz) * sign;
    const bz = (nx * ty - ny * tx) * sign;

    // Central difference steps one cell each way (two cells total). The step cannot be
    // smaller: a height-map cell is only a few millimeters, and a narrower difference would
    // treat scan noise as slope.
    const e = d.cell;
    // Samples are already in meters, so the difference is the slope (m/m); no scale needed
    const ku =
        (sampleDisplacement(d.image, u + e, w, d.mid, d.scale) -
            sampleDisplacement(d.image, u - e, w, d.mid, d.scale)) /
        (2 * e);
    const kv =
        (sampleDisplacement(d.image, u, w + e, d.mid, d.scale) -
            sampleDisplacement(d.image, u, w - e, d.mid, d.scale)) /
        (2 * e);

    // Surface tangents along u/v = tangent/bitangent plus the displacement derivative
    // along the normal
    const ux = tx + nx * ku;
    const uy = ty + ny * ku;
    const uz = tz + nz * ku;
    const vx = bx + nx * kv;
    const vy = by + ny * kv;
    const vz = bz + nz * kv;
    let mx = uy * vz - uz * vy;
    let my = uz * vx - ux * vz;
    let mz = ux * vy - uy * vx;
    const m = Math.hypot(mx, my, mz);
    if (m < 1e-12) return;
    mx /= m;
    my /= m;
    mz /= m;
    v[3] = mx;
    v[4] = my;
    v[5] = mz;
}

function edgeLength(a: Float32Array, b: Float32Array): number {
    return Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
}

function computeBounds(vertexData: Float32Array): AABB {
    const min = { x: Infinity, y: Infinity, z: Infinity };
    const max = { x: -Infinity, y: -Infinity, z: -Infinity };
    for (let i = 0; i < vertexData.length; i += 12) {
        const x = vertexData[i];
        const y = vertexData[i + 1];
        const z = vertexData[i + 2];
        if (x < min.x) min.x = x;
        if (y < min.y) min.y = y;
        if (z < min.z) min.z = z;
        if (x > max.x) max.x = x;
        if (y > max.y) max.y = y;
        if (z > max.z) max.z = z;
    }
    return { min, max };
}

class MeshBuilderImpl implements MeshBuilder {
    private positions: number[] = [];
    private normals: number[] = [];
    private uvs: number[] = [];
    private tangents: number[] = [];
    private indices: number[] = [];
    private triMaterials: number[] = [];
    private vertexCount = 0;

    public addQuad(corners: [Vec3, Vec3, Vec3, Vec3], material: number, uv?: UvProjection): void {
        const [a, b, c, d] = corners;
        const e1 = sub(b, a);
        const e2 = sub(c, a);
        const n = normalize(cross(e1, e2));
        const t = normalize(e1);
        const bitangent = cross(n, t);
        const from = uv?.origin ?? a;
        const uAxis = uv?.uAxis ?? t;
        const vAxis = uv?.vAxis ?? bitangent;
        const base = this.vertexCount;
        for (const p of [a, b, c, d]) {
            const offset = sub(p, from);
            this.pushVertex(p, n, dot(offset, uAxis), dot(offset, vAxis), t);
        }
        this.pushTriangle(base, base + 1, base + 2, material);
        this.pushTriangle(base, base + 2, base + 3, material);
    }

    public build(materials: readonly PBRMaterial[], name?: string): Mesh {
        const vertexData = new Float32Array(this.vertexCount * 12);
        for (let i = 0; i < this.vertexCount; i++) {
            const o = i * 12;
            vertexData[o] = this.positions[i * 3];
            vertexData[o + 1] = this.positions[i * 3 + 1];
            vertexData[o + 2] = this.positions[i * 3 + 2];
            vertexData[o + 3] = this.normals[i * 3];
            vertexData[o + 4] = this.normals[i * 3 + 1];
            vertexData[o + 5] = this.normals[i * 3 + 2];
            vertexData[o + 6] = this.uvs[i * 2];
            vertexData[o + 7] = this.uvs[i * 2 + 1];
            vertexData[o + 8] = this.tangents[i * 4];
            vertexData[o + 9] = this.tangents[i * 4 + 1];
            vertexData[o + 10] = this.tangents[i * 4 + 2];
            vertexData[o + 11] = this.tangents[i * 4 + 3];
        }

        const triCount = this.indices.length / 3;
        const triangleData = new Uint32Array(triCount * 4);
        for (let t = 0; t < triCount; t++) {
            const o = t * 4;
            triangleData[o] = this.indices[t * 3];
            triangleData[o + 1] = this.indices[t * 3 + 1];
            triangleData[o + 2] = this.indices[t * 3 + 2];
            triangleData[o + 3] = this.triMaterials[t];
        }

        return new MeshImpl(vertexData, triangleData, materials, name);
    }

    private pushVertex(p: Vec3, n: Vec3, u: number, v: number, t: Vec3): void {
        this.positions.push(p.x, p.y, p.z);
        this.normals.push(n.x, n.y, n.z);
        this.uvs.push(u, v);
        this.tangents.push(t.x, t.y, t.z, 1);
        this.vertexCount++;
    }

    private pushTriangle(a: number, b: number, c: number, material: number): void {
        this.indices.push(a, b, c);
        this.triMaterials.push(material);
    }
}

/** Assignments for contract-layer class bindings. */
Mesh = MeshImpl;
MeshBuilder = MeshBuilderImpl;
