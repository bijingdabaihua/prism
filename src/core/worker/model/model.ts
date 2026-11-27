import type { Instance } from './instance';
import type { Mesh } from './mesh';
import type { TextureSet } from './material/texture-set';
import { INST_FLOATS } from './instance';
import { Mat4 } from '../../math/mat4';
import { buildBvh, meshBlas } from './bvh';
import { MaterialStore } from './material/material-store';

/**
 * Model — the GPU form of a scene: a set of buffers.
 *
 * It is the other side of the same thing as `Instance` in `instance.ts`: Instance is the
 * CPU-side description, and this is the GPU data compiled from it. The render layer only
 * knows this side and never touches any CPU-side geometry object.
 *
 * "Model" here means "the scene's model on the GPU side", not the glTF asset under
 * `assets/model/` — that one belongs to Mesh / Instance.
 *
 * Every `add()` runs the whole compilation chain again over "all current instances";
 * the four stages all live in the implementation layer:
 *
 *   layout   dedupe meshes, compute each mesh's BLAS / vertex / triangle start index
 *   nodes    TLAS + each mesh's BLAS combined into one flat Float32Array, indices rebased
 *   pack     vertices / triangles / instances each laid out into one big array, material
 *            indices rewritten to absolute slots
 *   upload   create buffers and write into them — pure transport, no computation
 *
 * When instances are loaded one at a time, each call appends to the previous result;
 * old buffers already submitted to the GPU cannot be destroyed immediately, and must
 * wait for `queue.onSubmittedWorkDone()`. That is the only thing this file does beyond
 * a one-shot compile.
 *
 * Everything lives in one file because there is only one call chain: the intermediate
 * data structures (SceneOffsets / SceneNodesData) have no other consumer, and splitting
 * them into three modules would just make the reader jump around.
 *
 * The material table is maintained internally from the result of each pack, and the
 * material sampling atlas is owned by the material table (see MaterialStore).
 *
 * The file has two layers: contract (all exports) and implementation (no exports);
 * see AGENTS.md for the convention.
 */

// ══════════════════ Contract layer (exports) ═════════════════

/**
 * Class binding for Model: construction creates an empty material table; the four scene
 * buffers do not exist until the first add().
 *
 * Only declarations here; assignment is in the implementation layer.
 */
export let Model: {
    new (device: GPUDevice): Model;
};

/** The GPU form of a scene — a set of buffers, appendable; see destroy() for release. */
export interface Model {
    /**
     * Combined TLAS + BLAS node buffer: 8 floats per node.
     * Layout [TLAS ...][BLAS ...]; TLAS leaves encode -(instanceID + 1),
     * BLAS leaves encode -(absolute triangle index + 1).
     */
    nodes: GPUBuffer;
    /** Combined vertex data, 12 floats/vertex */
    geometry: GPUBuffer;
    /** Combined triangles, 4 × u32 (v0, v1, v2, absolute material slot) */
    triangles: GPUBuffer;
    emitters: GPUBuffer;
    emissionScale(): number;
    setEmissionScale(scale: number): void;
    /** Instance table: 36 floats/instance (blas_node_offset + transform + inv_transform) */
    instanceBuffer: GPUBuffer;

    /**
     * Whether geometry / materials have been compiled to the GPU.
     * PathTracePass uses it to decide whether bind groups can be created.
     */
    ready(): boolean;
    /**
     * Scene resource version: incremented on every successful append.
     *
     * Appending replaces nodes / geometry / triangles / instanceBuffer / the material
     * SSBO; existing bind groups become stale with it, and accumulation must restart
     * from zero.
     */
    version(): number;
    /**
     * Appends a group of instances to the scene — may be called repeatedly from outside.
     *
     * New instances are evolved first (displacement maps baked into real geometry), then
     * merged with the already-appended instances and run through the whole compilation
     * chain again. Before `add()` none of the four buffers exists — PathTracePass waits
     * for `ready()` before creating bind groups.
     *
     * New buffers are built first, the material table is committed next, and fields are
     * swapped last: on failure the old scene stays usable; retired old buffers are
     * destroyed after the submitted GPU work completes.
     */
    add(insts: Instance[]): void;
    /** Material SSBO, for direct binding in a bind group. */
    materialBuffer(): GPUBuffer;
    /** Material atlas, for direct binding in a bind group. */
    textures(): TextureSet;
    destroy(): void;
}

// ═════════════ Implementation layer (not exported) ════════════

// Layout

/**
 * Layout of scene resources: geometry / BLAS / triangles are divided by **mesh**, not by
 * instance.
 *
 * Multiple instances of the same mesh share the same geometry region and the same BLAS;
 * the instance buffer stores one transform each plus a base address pointing at the same
 * region. This is what makes instancing actually save VRAM.
 */
interface SceneOffsets {
    /** Deduplicated meshes, in order of first appearance */
    meshes: Mesh[];
    /** BLAS start index for each mesh (float) */
    blas: number[];
    /** Vertex start index for each mesh (vertex) */
    vert: number[];
    /** Triangle start index for each mesh (triangle) */
    tri: number[];
    /** Index into meshes[] of the mesh each instance references */
    instMesh: number[];
}

function computeOffsets(insts: readonly Instance[]): SceneOffsets {
    const meshes: Mesh[] = [];
    const meshIndex = new Map<Mesh, number>();
    const instMesh: number[] = [];
    for (const inst of insts) {
        let idx = meshIndex.get(inst.mesh);
        if (idx === undefined) {
            idx = meshes.length;
            meshIndex.set(inst.mesh, idx);
            meshes.push(inst.mesh);
        }
        instMesh.push(idx);
    }

    const blas: number[] = [];
    const vert: number[] = [];
    const tri: number[] = [];
    let b = 0;
    let v = 0;
    let t = 0;
    for (const mesh of meshes) {
        blas.push(b);
        vert.push(v);
        tri.push(t);
        b += meshBlas(mesh).length;
        v += mesh.vertexData.length / 12;
        t += mesh.triangleData.length / 4;
    }
    return { meshes, blas, vert, tri, instMesh };
}

// Node table

/**
 * TLAS + the BLAS of every mesh combined into one flat Float32Array.
 *
 * A pure CPU compilation product: this only decides "in what order nodes are arranged and
 * how indices are rebased"; writing it into a buffer is the job of the upload* functions
 * later in the file.
 *
 * The immediate reason to merge is binding slots: WebGPU allows at most 8 storage buffers
 * per shader stage by default, and geometry / triangles / materials / instances /
 * environment CDF are already near the limit; merging is what frees a slot for the pixel
 * sample-order table.
 *
 * Layout: [TLAS nodes ...][BLAS nodes ...], 8 floats per node.
 */

/**
 * Instances → TLAS.
 *
 * Leaf nodes encode -(instanceIndex + 1); a single instance degenerates to a single node,
 * matching the convention in `bvh.wgsl`.
 */
function buildTlas(insts: readonly Instance[]): Float32Array {
    if (insts.length === 0) return new Float32Array(0);
    if (insts.length === 1) {
        const b = insts[0].getBounds();
        const n = new Float32Array(8);
        n[0] = b.min.x;
        n[1] = b.min.y;
        n[2] = b.min.z;
        n[3] = b.max.x;
        n[4] = b.max.y;
        n[5] = b.max.z;
        n[6] = -1;
        return n;
    }
    return buildBvh(insts, (i) => {
        const b = i.getBounds();
        return { min: { x: b.min.x, y: b.min.y, z: b.min.z }, max: { x: b.max.x, y: b.max.y, z: b.max.z } };
    });
}

/**
 * Merges each mesh into one BLAS; triangle indices in leaves are rebased to that mesh's
 * global triangle region.
 *
 * @param nodeBase Start index (float) of this BLAS in the combined nodes buffer.
 *   Child offsets of internal nodes have it added; leaves (negative) are triangle
 *   indices and are unaffected.
 */
function mergeBLAS(offsets: SceneOffsets, nodeBase: number): Float32Array {
    const all = offsets.meshes.map((m) => meshBlas(m));
    let total = 0;
    for (const b of all) total += b.length;
    const merged = new Float32Array(total);
    for (let i = 0; i < all.length; i++) {
        const b = all[i];
        const off = offsets.blas[i];
        const triOff = offsets.tri[i];
        for (let n = 0; n < b.length; n += 8) {
            for (let j = 0; j < 6; j++) merged[off + n + j] = b[n + j];
            for (let j = 6; j < 8; j++) {
                const s = b[n + j];
                if (s < 0) {
                    merged[off + n + j] = s - triOff;
                } else {
                    merged[off + n + j] = s > 0 ? s + off + nodeBase : 0;
                }
            }
        }
    }
    return merged;
}

interface SceneNodesData {
    /** [TLAS ...][BLAS ...], 8 floats per node */
    data: Float32Array;
    /** BLAS region start index (float) in data; baked into the instance table's blas_node_offset */
    blasBase: number;
}

/** Builds the TLAS, appends every mesh's BLAS after it, and rebases internal node indices. */
function buildSceneNodes(insts: readonly Instance[], offsets: SceneOffsets): SceneNodesData {
    const tlas = buildTlas(insts);
    const blas = mergeBLAS(offsets, tlas.length);
    const data = new Float32Array(tlas.length + blas.length);
    data.set(tlas, 0);
    data.set(blas, tlas.length);
    return { data, blasBase: tlas.length };
}

// Upload to the GPU

/**
 * Compilation product → GPU buffer.
 *
 * Transport only: create buffers from the arrays computed above and write into them;
 * no further layout computation.
 */

function createBuffer(device: GPUDevice, data: Float32Array | Uint32Array, label: string, emptySize = 4): GPUBuffer {
    const buf = device.createBuffer({
        size: data.byteLength || emptySize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        label,
    });
    if (data.byteLength) device.queue.writeBuffer(buf, 0, data as any);
    return buf;
}

function uploadGeometry(device: GPUDevice, offsets: SceneOffsets): GPUBuffer {
    let total = 0;
    for (const mesh of offsets.meshes) total += mesh.vertexData.length / 12;
    const data = new Float32Array(total * 12);
    for (let i = 0; i < offsets.meshes.length; i++) {
        data.set(offsets.meshes[i].vertexData, offsets.vert[i] * 12);
    }
    return createBuffer(device, data, 'scene-geometry');
}

/**
 * @param matSlots Material table for each mesh: maps the relative material index in a
 *   triangle to an absolute slot
 */
function uploadTriangles(device: GPUDevice, offsets: SceneOffsets, matSlots: number[][]): GPUBuffer {
    let total = 0;
    for (const mesh of offsets.meshes) total += mesh.triangleData.length / 4;
    const data = new Uint32Array(total * 4);
    for (let i = 0; i < offsets.meshes.length; i++) {
        const tri = offsets.meshes[i].triangleData;
        const vOff = offsets.vert[i];
        const mats = matSlots[i];
        const tBase = offsets.tri[i] * 4;
        for (let s = 0; s < tri.length; s += 4) {
            data[tBase + s] = tri[s] + vOff;
            data[tBase + s + 1] = tri[s + 1] + vOff;
            data[tBase + s + 2] = tri[s + 2] + vOff;
            data[tBase + s + 3] = mats[tri[s + 3]];
        }
    }
    return createBuffer(device, data, 'scene-triangles');
}

/**
 * Inverse of the world transform — used in shading to go from world space back to local
 * space. Derived data, so computed at pack time.
 */
function invertTransform(transform: Float32Array): Float32Array {
    return Mat4.from(transform).invert().raw();
}

function uploadInstances(device: GPUDevice, insts: Instance[], offsets: SceneOffsets, blasBase: number): GPUBuffer {
    const data = new Float32Array(insts.length * INST_FLOATS);
    for (let i = 0; i < insts.length; i++) {
        const b = i * INST_FLOATS;
        const inst = insts[i];
        // All instances of the same mesh point at the same BLAS; the BLAS region starts at
        // blasBase in the combined nodes buffer, so bake the base address into offset.
        data[b] = offsets.blas[offsets.instMesh[i]] + blasBase;
        const pm = (m: Float32Array, o: number) => {
            data[b + o] = m[0];
            data[b + o + 1] = m[1];
            data[b + o + 2] = m[2];
            data[b + o + 3] = 0;
            data[b + o + 4] = m[4];
            data[b + o + 5] = m[5];
            data[b + o + 6] = m[6];
            data[b + o + 7] = 0;
            data[b + o + 8] = m[8];
            data[b + o + 9] = m[9];
            data[b + o + 10] = m[10];
            data[b + o + 11] = 0;
            data[b + o + 12] = m[12];
            data[b + o + 13] = m[13];
            data[b + o + 14] = m[14];
            data[b + o + 15] = 0;
        };
        pm(inst.transform, 4);
        pm(invertTransform(inst.transform), 20);
    }
    // WebGPU enforces a minimum binding size for `array<Instance>` (INST_FLOATS*4 B/instance);
    // an empty scene must satisfy it too
    return createBuffer(device, data, 'scene-instances', INST_FLOATS * 4);
}

/**
 * Node table: [TLAS ...][BLAS ...], 8 floats per node
 * (construction is in buildSceneNodes above).
 */
function uploadSceneNodes(device: GPUDevice, data: Float32Array): GPUBuffer {
    return createBuffer(device, data, 'scene-nodes');
}

/**
 * Stores world-space vertices, cumulative area, and emitted radiance for each emissive
 * triangle.
 */
function uploadEmitters(device: GPUDevice, instances: readonly Instance[]): GPUBuffer {
    const records = [0, 0];
    for (const { mesh, transform: m } of instances) {
        if (!mesh.materials.some(({ emission: e }) => e.r > 0 || e.g > 0 || e.b > 0)) continue;
        const vertex = (index: number): [number, number, number] => {
            const i = index * 12;
            const [x, y, z] = mesh.vertexData.subarray(i, i + 3);
            return [
                m[0] * x + m[4] * y + m[8] * z + m[12],
                m[1] * x + m[5] * y + m[9] * z + m[13],
                m[2] * x + m[6] * y + m[10] * z + m[14],
            ];
        };
        const tris = mesh.triangleData;
        for (let i = 0; i < tris.length; i += 4) {
            const e = mesh.materials[tris[i + 3]].emission;
            if (e.r <= 0 && e.g <= 0 && e.b <= 0) continue;
            const a = vertex(tris[i]);
            const b = vertex(tris[i + 1]);
            const c = vertex(tris[i + 2]);
            const [ux, uy, uz] = b.map((v, axis) => v - a[axis]);
            const [vx, vy, vz] = c.map((v, axis) => v - a[axis]);
            const area = 0.5 * Math.hypot(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx);
            if (!(area > 0)) continue;
            records[0]++;
            records[1] += area;
            records.push(...a, ...b, ...c, records[1], e.r, e.g, e.b);
        }
    }
    return createBuffer(device, new Float32Array(records), 'scene-emitters');
}

class ModelImpl implements Model {
    public nodes!: GPUBuffer;
    public geometry!: GPUBuffer;
    public triangles!: GPUBuffer;
    public emitters!: GPUBuffer;
    public instanceBuffer!: GPUBuffer;
    private compiled = false;
    private emissionMultiplier = 1;
    /**
     * Incremented on every successful append; PathTracePass uses it to notice scene
     * buffers were replaced.
     */
    private versionCounter = 0;
    /** Appended (evolved) instances; the next add merges with them and recompiles everything. */
    private instances: Instance[] = [];
    /** Old buffers, destroyed once the GPU is done with them. */
    private readonly retired = new Set<GPUBuffer>();
    private destroyed = false;
    private readonly device: GPUDevice;
    private readonly materialStore: MaterialStore;

    constructor(device: GPUDevice) {
        this.device = device;
        this.materialStore = new MaterialStore(device);
    }

    public emissionScale(): number {
        return this.emissionMultiplier;
    }

    public setEmissionScale(scale: number): void {
        if (!Number.isFinite(Math.fround(scale)) || scale < 0) throw new Error('emission scale must be a non-negative finite number');
        this.emissionMultiplier = Math.fround(scale);
    }

    public ready(): boolean {
        return this.compiled;
    }

    public version(): number {
        return this.versionCounter;
    }

    public add(insts: Instance[]): void {
        if (this.destroyed) throw new Error('Model: already destroyed');
        if (insts.length === 0) return;

        // Evolution: instances whose material has a displacement map are replaced here by
        // tessellated real geometry; from then on the GPU side has no displacement map and
        // no concept of "displacement" at all
        const instances = this.instances.concat(insts.map((inst) => inst.evolve()));
        const offsets = computeOffsets(instances);
        // Materials are registered by object: a material shared by several meshes takes one slot
        const matSlots = offsets.meshes.map((mesh) => mesh.materials.map((m) => this.materialStore.register(m)));

        // Build geometry first; if material-table compilation fails, drop the new geometry
        // and leave the old scene untouched
        const compiled = this.build(offsets, instances, matSlots);
        let previousMaterials: GPUBuffer | null;
        try {
            previousMaterials = this.materialStore.compile();
        } catch (err) {
            destroyCompiled(compiled);
            throw err;
        }

        const retired: GPUBuffer[] = [];
        if (this.compiled) retired.push(this.nodes, this.geometry, this.triangles, this.instanceBuffer, this.emitters);
        if (previousMaterials) retired.push(previousMaterials);

        this.instances = instances;
        this.nodes = compiled.nodes;
        this.geometry = compiled.geometry;
        this.triangles = compiled.triangles;
        this.emitters = compiled.emitters;
        this.instanceBuffer = compiled.instanceBuffer;
        this.compiled = true;
        this.versionCounter++;
        this.retire(retired);
    }

    public materialBuffer(): GPUBuffer {
        return this.materialStore.buffer();
    }

    public textures(): TextureSet {
        return this.materialStore.textures;
    }

    public destroy(): void {
        if (this.destroyed) return;
        this.destroyed = true;
        this.nodes?.destroy();
        this.geometry?.destroy();
        this.triangles?.destroy();
        this.emitters?.destroy();
        this.instanceBuffer?.destroy();
        for (const buf of this.retired) buf.destroy();
        this.retired.clear();
        this.materialStore.destroy();
    }

    /**
     * Delayed release of old buffers.
     *
     * Submitted command buffers may still reference them; `onSubmittedWorkDone()` covers
     * all work submitted before the call. The callback checks the set again: if `destroy()`
     * has already run, it does not destroy twice.
     */
    private retire(buffers: GPUBuffer[]): void {
        if (buffers.length === 0) return;
        for (const buf of buffers) this.retired.add(buf);
        void this.device.queue.onSubmittedWorkDone().then(() => {
            for (const buf of buffers) {
                if (this.retired.delete(buf)) buf.destroy();
            }
        });
    }

    /**
     * Builds a complete set of new buffers; on failure destroys the parts already created
     * so nothing leaks.
     */
    private build(offsets: SceneOffsets, insts: Instance[], matSlots: number[][]): CompiledBuffers {
        const d = this.device;
        const created: GPUBuffer[] = [];
        const track = (buf: GPUBuffer): GPUBuffer => {
            created.push(buf);
            return buf;
        };
        try {
            // The node table is assembled on the CPU; the instance table needs its BLAS
            // base address
            const nodes = buildSceneNodes(insts, offsets);
            return {
                nodes: track(uploadSceneNodes(d, nodes.data)),
                geometry: track(uploadGeometry(d, offsets)),
                triangles: track(uploadTriangles(d, offsets, matSlots)),
                emitters: track(uploadEmitters(d, insts)),

                instanceBuffer: track(uploadInstances(d, insts, offsets, nodes.blasBase)),
            };
        } catch (err) {
            for (const buf of created) buf.destroy();
            throw err;
        }
    }
}

/** Assignment of the contract-layer class binding. */
Model = ModelImpl;

interface CompiledBuffers {
    nodes: GPUBuffer;
    geometry: GPUBuffer;
    triangles: GPUBuffer;
    emitters: GPUBuffer;
    instanceBuffer: GPUBuffer;
}

/** Discards a set of compiled buffers that was never swapped in. */
function destroyCompiled(buffers: CompiledBuffers): void {
    buffers.nodes.destroy();
    buffers.geometry.destroy();
    buffers.triangles.destroy();
    buffers.emitters.destroy();
    buffers.instanceBuffer.destroy();
}
