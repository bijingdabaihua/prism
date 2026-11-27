import type { Vec3 } from '../../math/types';
import type { Image } from './image';
import type { PBRMaterialOptions, TextureKind } from './material/material';
import { Mat4 } from '../../math/mat4';
import { PBRMaterial } from './material/material';
import { Mesh } from './mesh';
import { loadTexture } from './image';

/**
 * Instance IR: a single placement of a mesh, plus its wire codec and glTF
 * importer.
 *
 * The codec crosses threads: IR class instances cannot cross a thread boundary
 * — structured clone only carries enumerable fields and drops all methods, yet
 * in the Worker `engine.addInstances` → `Model.compile` depends on methods such
 * as `evolve()`. So the raw `WireInstance` data crosses the thread, and the
 * Worker rebuilds the class instances from it.
 *
 * The glTF 2.0 parser lives in the implementation layer and is driven by
 * `Instance.buildByGltf`: the node tree in a file is expanded into several
 * Instances there.
 *
 * The file has two layers: contract (all exports) and implementation (no
 * exports); see AGENTS.md for the convention.
 */

// ═════════════════ Contract layer (exports) ══════════════════

/**
 * Class binding for Instance: constructs an instance (mesh + transform); the
 * static method `buildByGltf` imports from glTF, and `encode` / `decode` form
 * the cross-thread codec.
 *
 * Declarations only; the assignments live in the implementation layer.
 */
export let Instance: {
    /**
     * Constructor: copies the transform (identity matrix by default). The mesh
     * is not copied — multiple instances share the same geometry, and the BLAS
     * is built per mesh, not per instance.
     */
    new (mesh: Mesh, transform?: Float32Array): Instance;

    /**
     * Reads a glTF file (.gltf with external resources, or a single-file .glb)
     * and returns Instances ready to add to the scene. The parser lives in the
     * implementation layer.
     *
     * A file may hold several nodes, each a separate placement — hence the
     * array return, which the caller should place as a whole rather than one by
     * one.
     *
     * A URL also supplies the base for external resources; a Blob has no base,
     * so it suits self-contained `.glb` files, `.gltf` files with all resources
     * embedded, or a `.zip` bundling `.gltf` / `.bin` / textures. ZIPs are
     * unpacked in the Worker and external paths are rewritten to in-memory blob
     * URLs, after which the same external-resource flow still applies.
     *
     * Materials are already on the Mesh (textures decoded into Image): the
     * importer hands back exactly what the IR can give the engine, with no
     * second result for the caller to assemble.
     *
     * See the importer's notes in the implementation layer for supported
     * features.
     */
    buildByGltf(source: string | Blob): Promise<Instance[]>;

    /**
     * Encodes an Instance into its wire form (`WireInstance`).
     *
     * postMessage (structured clone) does the actual serialization; encoding
     * defines the wire shape — explicitly picking the fields that cross the
     * boundary and deliberately dropping class identity and the `Mesh` evolution
     * cache (`evolved` is a Worker-side runtime cache and should not come over
     * from the main thread). Typed arrays are handed to the clone by reference
     * and are not copied here.
     */
    encode(instance: Instance): WireInstance;

    /**
     * Decodes the wire form into an Instance.
     *
     * The input is a cloned plain object; validate its shape first, then revive
     * it with the IR constructors: the `Mesh` constructor recomputes bounding
     * boxes, `PBRMaterial` fills in defaults, and the `Instance` constructor
     * copies the transform. A validation failure should name the offending field
     * instead of surfacing as `undefined is not a function`.
     *
     * Revive by content to deduplicate: the same geometry can appear many times
     * in one batch (a procedural floor has one Instance per tile, and 9 tiles
     * share one mesh); constructing each would yield 9 Mesh objects, while the
     * BLAS cache / material table / texture layers all deduplicate by object
     * identity. Identical bytes revive only once, and restoring identity is what
     * makes those three levels of deduplication work.
     */
    decode(raw: unknown): Instance;
};

/**
 * Instance IR: a single placement of a mesh.
 *
 * One mesh can be referenced by several instances — the BLAS should be shared
 * per mesh, not duplicated per instance.
 *
 * The inverse matrix is derived data, computed on the fly when packing
 * (engine/model/model.ts) and not stored in the IR.
 */
export interface Instance {
    /** Column-major 4×4 world transform. Publicly writable: placement modifies it directly. */
    transform: Float32Array;
    readonly mesh: Mesh;

    /**
     * Evolves the instance: lets the mesh consume displacement maps from its
     * materials and returns an instance ready for rendering.
     *
     * A mesh without displacement is returned as-is (`===` itself) with the
     * transform copied; the evolved result is cached on the mesh, so placing the
     * same mesh many times computes it once and shares one BLAS.
     */
    evolve(): Instance;

    /** World-space AABB after the transform (all 8 bounding-box corners transformed). */
    getBounds(): { min: Vec3; max: Vec3 };
}

/**
 * Wire form of an Instance: main-thread `Instance.encode` → postMessage →
 * Worker `Instance.decode`.
 *
 * - `Image` is already a pure-data interface, so it is reused directly on the
 *   wire;
 * - the material wire form is `PBRMaterialOptions` (the constructor arguments
 *   as a plain object);
 * - typed arrays (vertexData / triangleData / transform) keep their type after
 *   cloning and are type-checked on decode.
 */
export interface WireMesh {
    name?: string;
    vertexData: Float32Array;
    triangleData: Uint32Array;
    materials: PBRMaterialOptions[];
}

export interface WireInstance {
    mesh: WireMesh;
    transform?: Float32Array;
}

/**
 * Record size of an instance in the scene-instances SSBO (in floats): BLAS base
 * 1 + transform 16 + inverse transform 16 = 33, padded to 36 for alignment. See
 * engine/model/model.ts for packing.
 */
export const INST_FLOATS = 36;

export interface Placement {
    position: Vec3;
    /**
     * Rotation about X, in radians. Applied before `rotationY`.
     *
     * To lay a standing glTF model down you must rotate about a horizontal axis
     * — spinning about Y alone never lays it flat. Tipping it over with X and
     * then aiming it with Y matches how it feels to set down an object in your
     * hand.
     */
    rotationX?: number;
    /** Rotation about Y, in radians. 0 leaves the model's orientation unchanged. */
    rotationY?: number;
    scale?: number;
}

/**
 * Places a group of instances at an author-specified position / orientation /
 * scale, preserving their relative arrangement.
 *
 * Textures and models are exported in meters, so scale is usually 1 — adjust it
 * only when a model's size is clearly wrong, not to fudge a composition.
 */
export let placeInstances: (instances: readonly Instance[], p: Placement) => void;

// ════════════ Implementation layer (not exported) ═════════════

const IDENTITY_MAT4: Readonly<Float32Array> = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);

class InstanceImpl implements Instance {
    public static async buildByGltf(source: string | Blob): Promise<Instance[]> {
        let buf: ArrayBuffer;
        let baseUrl = '';
        let label = 'glTF Blob';
        const isUrl = typeof source === 'string';
        if (isUrl) {
            const resp = await fetch(source);
            if (!resp.ok) throw new Error(`Failed to load glTF: ${source} (${resp.status})`);
            buf = await resp.arrayBuffer();
            baseUrl = source.substring(0, source.lastIndexOf('/') + 1);
            label = source;
        } else {
            buf = await source.arrayBuffer();
        }
        // Content fingerprint: structured cloning changes a Blob's object
        // identity, so hashing the content lets the same asset reuse its
        // decompressed result and texture layers across loads.
        const cacheId = contentCacheId(buf);
        if (!isUrl) {
            // A non-zip Blob has no directory base; give embedded images a key
            // that will not collide with other Blobs
            baseUrl = `blob:${cacheId}/`;
        }

        // ZIP is only unpacked at this layer: the model file is extracted and
        // continues through glTF parsing, while external resources are
        // rewritten to stable, reusable blob URLs — downstream loadBuffers /
        // loadTexture never need to know they came from an archive.
        const zip = isZip(buf) ? parseZip(buf, label, cacheId) : null;
        let modelPath = '';
        if (zip) {
            modelPath = pickZipModel(zip);
            const model = await readZipEntry(zip, modelPath);
            buf = toArrayBuffer(model);
            const dir = modelPath.substring(0, modelPath.lastIndexOf('/') + 1);
            baseUrl = `zip:${cacheId}/${dir}`;
            label = `${label} (${modelPath})`;
        }

        const { doc, bin } = parseContainer(buf, label);
        if (zip) await rewriteZipUris(doc, zip, modelPath);
        const buffers = await loadBuffers(doc, bin, baseUrl, label);

        const ctx: LoadContext = {
            doc,
            buffers,
            baseUrl,
            textureUrls: new Map(),
            imageUrls: new Map(),
            occlusionUrls: new Map(),
        };

        const materials = await Promise.all((doc.materials ?? []).map((m) => convertMaterial(m, ctx)));
        const meshes = doc.meshes.map((m, i) => buildMesh(ctx, m, materials, `gltf-mesh-${i}`));
        const instances = buildInstances(doc, meshes);

        return instances;
    }

    public static encode(instance: Instance): WireInstance {
        const mesh = instance.mesh;
        return {
            transform: instance.transform,
            mesh: {
                name: mesh.name,
                vertexData: mesh.vertexData,
                triangleData: mesh.triangleData,
                // Spread into a plain object: keeps every field (all constructor
                // args are here) and drops the prototype
                materials: mesh.materials.map((m) => ({ ...m })),
            },
        };
    }

    public static decode(raw: unknown): Instance {
        if (!raw || typeof raw !== 'object') {
            throw new Error('Instance data is not an object');
        }
        const wire = raw as WireInstance;
        if (!wire.mesh || typeof wire.mesh !== 'object') {
            throw new Error('Instance data is missing mesh');
        }
        if (wire.transform !== undefined && !(wire.transform instanceof Float32Array)) {
            throw new Error('Instance data transform is not a Float32Array');
        }
        const { vertexData, triangleData, materials } = wire.mesh;
        if (!(vertexData instanceof Float32Array) || !(triangleData instanceof Uint32Array)) {
            throw new Error('Instance mesh vertex / triangle data is missing or has the wrong type');
        }
        if (!Array.isArray(materials)) {
            throw new Error('Instance mesh is missing its material table');
        }
        const key = meshFingerprint(wire.mesh);
        let mesh = decodedMeshes.get(key);
        if (!mesh) {
            mesh = new Mesh(
                vertexData,
                triangleData,
                materials.map((m) => new PBRMaterial(m)),
                wire.mesh.name,
            );
            decodedMeshes.set(key, mesh);
        }
        return new InstanceImpl(mesh, wire.transform);
    }

    public transform: Float32Array;
    public readonly mesh: Mesh;

    constructor(mesh: Mesh, transform?: Float32Array) {
        this.mesh = mesh;
        this.transform = transform ? new Float32Array(transform) : new Float32Array(IDENTITY_MAT4);
    }

    public evolve(): Instance {
        const mesh = this.mesh.evolve();
        return mesh === this.mesh ? this : new InstanceImpl(mesh, this.transform);
    }

    public getBounds(): { min: Vec3; max: Vec3 } {
        const b = this.mesh.bounds;
        const t = this.transform;
        const corners = [
            { x: b.min.x, y: b.min.y, z: b.min.z },
            { x: b.max.x, y: b.min.y, z: b.min.z },
            { x: b.min.x, y: b.max.y, z: b.min.z },
            { x: b.max.x, y: b.max.y, z: b.min.z },
            { x: b.min.x, y: b.min.y, z: b.max.z },
            { x: b.max.x, y: b.min.y, z: b.max.z },
            { x: b.min.x, y: b.max.y, z: b.max.z },
            { x: b.max.x, y: b.max.y, z: b.max.z },
        ];
        let minX = Infinity,
            minY = Infinity,
            minZ = Infinity;
        let maxX = -Infinity,
            maxY = -Infinity,
            maxZ = -Infinity;
        for (const c of corners) {
            const x = t[0] * c.x + t[4] * c.y + t[8] * c.z + t[12];
            const y = t[1] * c.x + t[5] * c.y + t[9] * c.z + t[13];
            const z = t[2] * c.x + t[6] * c.y + t[10] * c.z + t[14];
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
            if (z < minZ) minZ = z;
            if (z > maxZ) maxZ = z;
        }
        return { min: { x: minX, y: minY, z: minZ }, max: { x: maxX, y: maxY, z: maxZ } };
    }
}

/**
 * Revival dedup table: content fingerprint → Mesh. Only grows for the lifetime
 * of the Worker; like this file's `derivedBlobs`, it is a cross-message reuse
 * cache.
 */
const decodedMeshes = new Map<string, Mesh>();

/**
 * Mesh content fingerprint (non-cryptographic).
 *
 * Its purpose is sharing one copy among byte-identical clones, not tamper
 * detection: identical bytes always produce the same key, while distinct
 * content is told apart by length + head/tail hashes + evenly spaced sampling +
 * material scalars / texture metadata, making collisions practically
 * impossible. Hashing a 20 MB vertex table byte by byte takes on the order of a
 * second, which is why sampling is used.
 */
function meshFingerprint(mesh: WireMesh): string {
    const parts: string[] = [mesh.name ?? ''];
    const sample = (view: ArrayBufferView): string => {
        const u8 = new Uint8Array(view.buffer, view.byteOffset, view.byteLength);
        const head = fnv1a(u8.subarray(0, Math.min(1024, u8.length)));
        const tail = u8.length > 1024 ? fnv1a(u8.subarray(u8.length - 1024)) : 0;
        let stride = 0;
        for (let i = 0; i < u8.length; i += 4096) {
            stride = (stride * 31 + u8[i]) | 0;
        }
        return `${u8.length}:${head}:${tail}:${stride}`;
    };
    parts.push(sample(mesh.vertexData), sample(mesh.triangleData));
    for (const m of mesh.materials) {
        parts.push(
            JSON.stringify([
                m.baseColor,
                m.metallic,
                m.roughness,
                m.transmission,
                m.ior,
                m.emission,
                m.normalScale,
                m.occlusionStrength,
                m.displacementMid,
                m.displacementScale,
            ]),
        );
        const tex = (img: unknown): string => {
            if (!img || typeof img !== 'object') return '-';
            const t = img as { data: Float32Array; width: number; height: number; meters?: number };
            return `${sample(t.data)}:${t.width}x${t.height}:${t.meters ?? 1}`;
        };
        parts.push(
            tex(m.baseColorTexture),
            tex(m.metallicRoughnessTexture),
            tex(m.roughnessTexture),
            tex(m.normalTexture),
            tex(m.occlusionTexture),
            tex(m.displacement),
        );
    }
    return parts.join('|');
}

function fnv1a(bytes: Uint8Array): number {
    let hash = 0x811c9dc5;
    for (let i = 0; i < bytes.length; i++) {
        hash ^= bytes[i];
        hash = Math.imul(hash, 0x01000193);
    }
    return hash >>> 0;
}

/**
 * glTF 2.0 importer: file → Instance[].
 *
 * Supported features, and how each gap fails:
 * - triangles only (mode 4); other primitive modes raise an error instead of
 *   being silently dropped;
 * - reads four maps — baseColor / normal / metallicRoughness / occlusion — into
 *   four separate texture arrays;
 * - node transforms go into Instances while vertices stay in local coordinates,
 *   so a mesh referenced by several nodes shares its geometry (the BLAS is
 *   built per mesh, not per node);
 * - Draco / meshopt compression, morph targets, skinning and animation are not
 *   supported — Draco raises an explicit error, the rest are ignored.
 *
 * Coordinates and UVs need no conversion: glTF is Y-up, in meters, with the UV
 * origin at the top-left of the texture, matching decodeBitmap's decode
 * direction and WebGPU's sampling origin.
 */

// Containers and constants

const GLB_MAGIC = 0x46546c67; // 'glTF' little-endian
const CHUNK_JSON = 0x4e4f534a; // 'JSON'
const CHUNK_BIN = 0x004e4942; // 'BIN\0'

const MODE_TRIANGLES = 4;

const COMPONENT = {
    5120: { view: 'getInt8', size: 1, norm: 127 },
    5121: { view: 'getUint8', size: 1, norm: 255 },
    5122: { view: 'getInt16', size: 2, norm: 32767 },
    5123: { view: 'getUint16', size: 2, norm: 65535 },
    5125: { view: 'getUint32', size: 4, norm: 4294967295 },
    5126: { view: 'getFloat32', size: 4, norm: 1 },
} as const;

const TYPE_COMPONENTS: Record<string, number> = {
    SCALAR: 1,
    VEC2: 2,
    VEC3: 3,
    VEC4: 4,
    MAT2: 4,
    MAT3: 9,
    MAT4: 16,
};

// Parsed document — declares only the fields used

interface GltfDocument {
    accessors: GltfAccessor[];
    bufferViews: { buffer: number; byteOffset?: number; byteLength: number; byteStride?: number }[];
    /** Material-only documents have no buffers; only vertex data needs them. */
    buffers?: { byteLength: number; uri?: string }[];
    meshes: GltfMesh[];
    nodes: GltfNode[];
    materials?: GltfMaterial[];
    textures?: { source?: number; sampler?: number }[];
    images?: { uri?: string; bufferView?: number; mimeType?: string }[];
    scenes?: { nodes?: number[] }[];
    scene?: number;
}

interface GltfAccessor {
    bufferView?: number;
    byteOffset?: number;
    componentType: number;
    normalized?: boolean;
    count: number;
    type: string;
    sparse?: unknown;
}

interface GltfMesh {
    name?: string;
    primitives: GltfPrimitive[];
}

interface GltfPrimitive {
    attributes: Record<string, number>;
    indices?: number;
    material?: number;
    mode?: number;
    extensions?: Record<string, unknown>;
}

interface GltfNode {
    name?: string;
    mesh?: number;
    children?: number[];
    matrix?: number[];
    translation?: number[];
    rotation?: number[];
    scale?: number[];
}

interface GltfTextureRef {
    index: number;
    /** Normal scale for normalTexture. */
    scale?: number;
    /** Occlusion strength for occlusionTexture. */
    strength?: number;
}

interface GltfMaterial {
    emissiveFactor?: number[];
    extensions?: { KHR_materials_emissive_strength?: { emissiveStrength?: number } };
    name?: string;
    normalTexture?: GltfTextureRef;
    occlusionTexture?: GltfTextureRef;
    pbrMetallicRoughness?: {
        baseColorFactor?: number[];
        baseColorTexture?: GltfTextureRef;
        metallicFactor?: number;
        roughnessFactor?: number;
        metallicRoughnessTexture?: GltfTextureRef;
    };
}

// Load context

/** State shared during one load: decoded buffers, texture URL caches, blob URLs to release. */
interface LoadContext {
    doc: GltfDocument;
    buffers: ArrayBuffer[];
    baseUrl: string;
    /**
     * Texture index → resolved texture URL. Occlusion extracts a channel on the
     * fly, so it is cached separately.
     */
    textureUrls: Map<number, Promise<string | null>>;
    /**
     * Image index → resolved URL (displacement maps reference images directly,
     * without going through textures).
     */
    imageUrls: Map<number, Promise<string | null>>;
    occlusionUrls: Map<number, Promise<string | null>>;
}

/**
 * Cache of derived blob URLs, keyed by texture source rather than load batch.
 *
 * Keys must be stable: TextureSet uses them as layer cache keys, and layers are
 * never reclaimed — generating a new blob URL on every load would exhaust the
 * array's layers after a few scene switches. The cost is that the blobs
 * themselves are never released (a few hundred KB), in exchange for one texture
 * always mapping to one layer.
 */
const derivedBlobs = new Map<string, Promise<string | null>>();

// Containers

/** .glb is a 12-byte header plus chunks; .gltf is pure JSON. Decided by magic, not extension. */
function parseContainer(buf: ArrayBuffer, url: string): { doc: GltfDocument; bin: ArrayBuffer | null } {
    const view = new DataView(buf);
    if (buf.byteLength >= 12 && view.getUint32(0, true) === GLB_MAGIC) {
        const version = view.getUint32(4, true);
        if (version !== 2) throw new Error(`glTF ${url}: only glTF 2.0 is supported, this file is version ${version}`);
        let offset = 12;
        let doc: GltfDocument | null = null;
        let bin: ArrayBuffer | null = null;
        while (offset + 8 <= buf.byteLength) {
            const length = view.getUint32(offset, true);
            const type = view.getUint32(offset + 4, true);
            const start = offset + 8;
            if (type === CHUNK_JSON) {
                doc = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, start, length))) as GltfDocument;
            } else if (type === CHUNK_BIN) {
                bin = buf.slice(start, start + length);
            }
            offset = start + length + ((4 - (length % 4)) % 4);
        }
        if (!doc) throw new Error(`glTF ${url}: no JSON chunk in GLB`);
        return { doc, bin };
    }
    const text = new TextDecoder().decode(buf);
    if (text.trimStart().startsWith('<')) {
        throw new Error(`glTF ${url}: looks like HTML/XML, not JSON — likely a 404 page`);
    }
    return { doc: JSON.parse(text) as GltfDocument, bin: null };
}

// ZIP containers
//
// Downloaded glTF assets are often .zip files: one .gltf, one .bin and a
// textures/ directory. A Blob only says "there are some bytes"; without
// unpacking, parseContainer would read the ZIP as JSON. This layer does just
// two things: extract the needed entries and rewrite relative URIs in the glTF
// to in-memory blob URLs; the normal external-resource flow then continues.

interface ZipEntry {
    method: number;
    compressedSize: number;
    uncompressedSize: number;
    /** Byte offset of this entry's data within the ZIP file. */
    dataOffset: number;
}

interface ZipArchive {
    buffer: ArrayBuffer;
    entries: Map<string, ZipEntry>;
    /**
     * Stable identifier for one source across loads, used to reuse decompressed
     * data and blob URLs.
     */
    cacheId: string;
    label: string;
}

const ZIP_LOCAL = 0x04034b50;
const ZIP_CENTRAL = 0x02014b50;
const ZIP_END = 0x06054b50;
const ZIP_METHOD_STORED = 0;
const ZIP_METHOD_DEFLATE = 8;

/**
 * Content fingerprint of source bytes (FNV-1a).
 *
 * A Blob becomes a new object after postMessage reaches the Worker, so caching
 * by object identity fails when the same Blob is passed in one at a time from
 * outside; hashing content is what lets ZIP decompression results and texture
 * layers be reused across messages.
 */
function contentCacheId(buf: ArrayBuffer): string {
    const bytes = new Uint8Array(buf);
    let hash = 0x811c9dc5;
    for (let i = 0; i < bytes.length; i++) {
        hash ^= bytes[i];
        hash = Math.imul(hash, 0x01000193);
    }
    return `content-${(hash >>> 0).toString(16)}-${bytes.length.toString(16)}`;
}

/** ZIP entry → decompressed blob URL; keyed by source + path and reused across loads. */
const zipObjectUrls = new Map<string, Promise<string>>();

function isZip(buf: ArrayBuffer): boolean {
    if (buf.byteLength < 4) return false;
    const sig = new DataView(buf).getUint32(0, true);
    return sig === ZIP_LOCAL || sig === ZIP_END;
}

/**
 * Reads the EOCD → central directory.
 *
 * Only plain ZIPs (single-volume, unencrypted, not ZIP64) are supported, which
 * is what asset packs usually are; unsupported cases raise an explicit error so
 * garbage is not later fed to JSON.parse.
 */
function parseZip(buf: ArrayBuffer, label: string, cacheId: string): ZipArchive {
    const view = new DataView(buf);
    let eocd = -1;
    const earliest = Math.max(0, buf.byteLength - 22 - 0xffff);
    for (let i = buf.byteLength - 22; i >= earliest; i--) {
        if (view.getUint32(i, true) === ZIP_END) {
            eocd = i;
            break;
        }
    }
    if (eocd < 0) throw new Error(`${label}: no central directory found in ZIP (corrupt file or not a ZIP)`);

    const count = view.getUint16(eocd + 10, true);
    let offset = view.getUint32(eocd + 16, true);
    const entries = new Map<string, ZipEntry>();
    const decoder = new TextDecoder();

    for (let i = 0; i < count; i++) {
        if (offset + 46 > buf.byteLength || view.getUint32(offset, true) !== ZIP_CENTRAL) {
            throw new Error(`${label}: corrupt ZIP central directory`);
        }
        const flags = view.getUint16(offset + 8, true);
        const method = view.getUint16(offset + 10, true);
        const compressedSize = view.getUint32(offset + 20, true);
        const uncompressedSize = view.getUint32(offset + 24, true);
        const nameLength = view.getUint16(offset + 28, true);
        const extraLength = view.getUint16(offset + 30, true);
        const commentLength = view.getUint16(offset + 32, true);
        const localOffset = view.getUint32(offset + 42, true);
        if (offset + 46 + nameLength > buf.byteLength) {
            throw new Error(`${label}: ZIP file name out of bounds`);
        }
        const name = decoder
            .decode(new Uint8Array(buf, offset + 46, nameLength))
            .replace(/\\/g, '/')
            .replace(/^\/+/, '');

        if (flags & 1) throw new Error(`${label}: encrypted ZIP is not supported (${name})`);
        if (compressedSize === 0xffffffff || uncompressedSize === 0xffffffff || localOffset === 0xffffffff) {
            throw new Error(`${label}: ZIP64 is not supported (${name})`);
        }

        if (name && !name.endsWith('/')) {
            if (localOffset + 30 > buf.byteLength || view.getUint32(localOffset, true) !== ZIP_LOCAL) {
                throw new Error(`${label}: corrupt ZIP entry header (${name})`);
            }
            const localName = view.getUint16(localOffset + 26, true);
            const localExtra = view.getUint16(localOffset + 28, true);
            const dataOffset = localOffset + 30 + localName + localExtra;
            if (dataOffset + compressedSize > buf.byteLength) {
                throw new Error(`${label}: ZIP entry out of bounds (${name})`);
            }
            entries.set(name, { method, compressedSize, uncompressedSize, dataOffset });
        }

        offset += 46 + nameLength + extraLength + commentLength;
    }

    return { buffer: buf, entries, cacheId, label };
}

/**
 * Picks the model file in an archive: skips macOS metadata and hidden
 * directories, preferring a top-level .gltf.
 */
function pickZipModel(zip: ZipArchive): string {
    const models = [...zip.entries.keys()].filter((name) => {
        if (name.startsWith('__MACOSX/')) return false;
        if (name.split('/').some((part) => part.startsWith('.'))) return false;
        return /\.(gltf|glb)$/i.test(name);
    });
    if (models.length === 0) throw new Error(`${zip.label}: no .gltf / .glb in ZIP`);
    const gltfs = models.filter((name) => /\.gltf$/i.test(name));
    const candidates = gltfs.length > 0 ? gltfs : models;
    const depth = (name: string) => name.split('/').length;
    candidates.sort((a, b) => depth(a) - depth(b) || a.length - b.length || a.localeCompare(b));
    return candidates[0];
}

async function readZipEntry(zip: ZipArchive, path: string): Promise<Uint8Array<ArrayBuffer>> {
    const entry = zip.entries.get(path);
    if (!entry) throw new Error(`${zip.label}: no "${path}" in ZIP`);

    const raw = new Uint8Array(zip.buffer, entry.dataOffset, entry.compressedSize);
    if (entry.method === ZIP_METHOD_STORED) return raw.slice();
    if (entry.method !== ZIP_METHOD_DEFLATE) {
        throw new Error(`${zip.label}: unsupported ZIP compression method ${entry.method} (${path})`);
    }

    if (typeof DecompressionStream === 'undefined') {
        throw new Error(`${zip.label}: this browser has no DecompressionStream, cannot decompress ZIP`);
    }
    const stream = new Blob([raw]).stream().pipeThrough(new DecompressionStream('deflate-raw'));
    const out = new Uint8Array(await new Response(stream).arrayBuffer());
    if (out.byteLength !== entry.uncompressedSize) {
        throw new Error(`${zip.label}: "${path}" has the wrong length after decompression`);
    }
    return out;
}

/** ZIP entry → fetchable blob URL; each path is decompressed and turned into a URL once. */
function zipEntryUrl(zip: ZipArchive, path: string): Promise<string> {
    const key = `${zip.cacheId}\0${path}`;
    const cached = zipObjectUrls.get(key);
    if (cached) return cached;
    const job = (async (): Promise<string> => {
        const bytes = await readZipEntry(zip, path);
        return URL.createObjectURL(new Blob([bytes]));
    })();
    zipObjectUrls.set(key, job);
    return job;
}

/** A glTF URI may be relative; it is relative to the directory containing the model file. */
function resolveZipPath(dir: string, uri: string): string {
    const raw = uri.split('#')[0].split('?')[0];
    let decoded = raw;
    try {
        decoded = decodeURIComponent(raw);
    } catch {
        // Use it as-is; the original path still shows up in errors if lookup fails
    }
    const parts: string[] = [];
    for (const part of `${dir}${decoded}`.split('/')) {
        if (!part || part === '.') continue;
        if (part === '..') parts.pop();
        else parts.push(part);
    }
    return parts.join('/');
}

function isAbsoluteUri(uri: string): boolean {
    return /^[a-z][a-z0-9+.-]*:/i.test(uri);
}

/**
 * Rewrites relative URIs in the glTF's buffers / images to blob URLs for the
 * corresponding ZIP entries.
 *
 * Only these two places change: they are the entry points for all external
 * resources in glTF. Afterwards `loadBuffers` / `imageUrl` / `channelUrl` still
 * work by fetching a URL and need no ZIP-specific branch.
 */
async function rewriteZipUris(doc: GltfDocument, zip: ZipArchive, modelPath: string): Promise<void> {
    const dir = modelPath.substring(0, modelPath.lastIndexOf('/') + 1);
    const refs: { uri?: string }[] = [...(doc.buffers ?? []), ...(doc.images ?? [])];
    await Promise.all(
        refs.map(async (ref) => {
            if (!ref.uri || isAbsoluteUri(ref.uri)) return;
            const path = resolveZipPath(dir, ref.uri);
            if (!zip.entries.has(path)) {
                throw new Error(`${zip.label}: model references "${path}", but the ZIP has no such file`);
            }
            ref.uri = await zipEntryUrl(zip, path);
        }),
    );
}

function toArrayBuffer(bytes: Uint8Array): ArrayBuffer {
    return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
}

/**
 * glTF allows buffers to be spread across external files, or to be a single
 * embedded BIN. Both are read into one array of ArrayBuffers, with accessors
 * indexing buffers by position.
 */
async function loadBuffers(
    doc: GltfDocument,
    bin: ArrayBuffer | null,
    baseUrl: string,
    url: string,
): Promise<ArrayBuffer[]> {
    return Promise.all(
        (doc.buffers ?? []).map(async (b, i) => {
            if (!b.uri) {
                if (!bin) throw new Error(`glTF ${url}: buffer[${i}] has no uri and no embedded BIN chunk`);
                return bin;
            }
            const bufUrl = b.uri.startsWith('data:') ? b.uri : resolveUri(b.uri, baseUrl);
            const resp = await fetch(bufUrl);
            if (!resp.ok) throw new Error(`glTF ${url}: failed to read buffer[${i}] (${bufUrl})`);
            return resp.arrayBuffer();
        }),
    );
}

// accessor

/**
 * Reads into a tightly packed Float32Array.
 *
 * A de-interleave step sits in between: a bufferView may have a byteStride
 * (several attributes interleaved), while the downstream Mesh IR requires tight
 * packing, so elements must be read one by one instead of treating the
 * underlying ArrayBuffer as an array — that would get both alignment and stride
 * wrong.
 */
function readAccessorFloats(ctx: LoadContext, index: number): Float32Array {
    const acc = ctx.doc.accessors[index];
    if (acc.sparse) throw new Error('glTF: sparse accessors are not supported yet');
    if (acc.bufferView === undefined) {
        // A legal "all zeros" accessor: no bufferView means the data defaults to 0.
        return new Float32Array(acc.count * TYPE_COMPONENTS[acc.type]);
    }

    const components = TYPE_COMPONENTS[acc.type];
    const comp = COMPONENT[acc.componentType as keyof typeof COMPONENT];
    if (!comp) throw new Error(`glTF: unknown componentType ${acc.componentType}`);

    const bv = ctx.doc.bufferViews[acc.bufferView];
    const data = new DataView(ctx.buffers[bv.buffer]);
    const base = (bv.byteOffset ?? 0) + (acc.byteOffset ?? 0);
    const stride = bv.byteStride ?? components * comp.size;
    const read = comp.view as 'getFloat32';

    const out = new Float32Array(acc.count * components);
    for (let i = 0; i < acc.count; i++) {
        const elementBase = base + i * stride;
        for (let c = 0; c < components; c++) {
            const raw = data[read](elementBase + c * comp.size, true) as number;
            out[i * components + c] = acc.normalized ? Math.max(raw / comp.norm, -1) : raw;
        }
    }
    return out;
}

/** Indices are widened to Uint32: the Mesh IR's triangleData is already uint32. */
function readIndices(ctx: LoadContext, index: number): Uint32Array {
    const acc = ctx.doc.accessors[index];
    if (acc.type !== 'SCALAR') throw new Error(`glTF: index accessor has type ${acc.type}`);
    if (acc.sparse) throw new Error('glTF: sparse accessors are not supported yet');
    const bv = ctx.doc.bufferViews[acc.bufferView!];
    const data = new DataView(ctx.buffers[bv.buffer]);
    const base = (bv.byteOffset ?? 0) + (acc.byteOffset ?? 0);
    const size = COMPONENT[acc.componentType as keyof typeof COMPONENT].size;
    const stride = bv.byteStride ?? size;
    const read = COMPONENT[acc.componentType as keyof typeof COMPONENT].view as 'getUint32';

    const out = new Uint32Array(acc.count);
    for (let i = 0; i < acc.count; i++) out[i] = data[read](base + i * stride, true) as number;
    return out;
}

// Geometry

/**
 * Merges all primitives of one glTF mesh into a single Mesh IR.
 *
 * Merging works because the Mesh IR is just one vertex table + one triangle
 * table + a material array; multiple primitives inside a mesh share the same
 * node transform, so the merged result still needs only one Instance.
 * Primitives are told apart by the material index in the last of every four
 * numbers in triangleData.
 */
function buildMesh(ctx: LoadContext, gm: GltfMesh, materials: PBRMaterial[], name: string): Mesh {
    // Scan first to get the totals, so several primitives can be laid into one
    // vertex/triangle table at once
    let vertexCount = 0;
    let idxCount = 0;
    for (const prim of gm.primitives) {
        checkPrimitive(prim);
        const count = ctx.doc.accessors[prim.attributes.POSITION].count;
        vertexCount += count;
        idxCount += prim.indices !== undefined ? ctx.doc.accessors[prim.indices].count : count;
    }
    const triCount = idxCount / 3;

    const positions = new Float32Array(vertexCount * 3);
    const normals = new Float32Array(vertexCount * 3);
    const uvs = new Float32Array(vertexCount * 2);
    const gTan = new Float32Array(vertexCount * 4);
    let hasNormals = true;
    let hasTangents = true;
    let hasUvs = true;

    const triangles = new Uint32Array(triCount * 4);
    const meshMaterials: PBRMaterial[] = [];

    let vo = 0; // vertex write offset
    let to = 0; // triangle write offset

    for (const prim of gm.primitives) {
        const pos = readAccessorFloats(ctx, prim.attributes.POSITION);
        const count = ctx.doc.accessors[prim.attributes.POSITION].count;
        positions.set(pos, vo * 3);

        const nrm = prim.attributes.NORMAL !== undefined ? readAccessorFloats(ctx, prim.attributes.NORMAL) : null;
        if (nrm) normals.set(nrm, vo * 3);
        else hasNormals = false;

        const uv =
            prim.attributes.TEXCOORD_0 !== undefined ? readAccessorFloats(ctx, prim.attributes.TEXCOORD_0) : null;
        if (uv) uvs.set(uv, vo * 2);
        else hasUvs = false;

        const tan = prim.attributes.TANGENT !== undefined ? readAccessorFloats(ctx, prim.attributes.TANGENT) : null;
        if (tan) gTan.set(tan, vo * 4);
        else hasTangents = false;

        // Material slot: one PBRMaterial takes a single index within a mesh
        const source = prim.material !== undefined ? materials[prim.material] : undefined;
        const mat = source ?? new PBRMaterial();
        let matIdx = meshMaterials.indexOf(mat);
        if (matIdx < 0) {
            matIdx = meshMaterials.length;
            meshMaterials.push(mat);
        }

        const idx =
            prim.indices !== undefined
                ? readIndices(ctx, prim.indices)
                : Uint32Array.from({ length: count }, (_, i) => i);
        if (idx.length % 3 !== 0) throw new Error(`glTF: primitive index count ${idx.length} is not a multiple of 3`);
        for (let i = 0; i < idx.length; i += 3) {
            triangles[to] = idx[i] + vo;
            triangles[to + 1] = idx[i + 1] + vo;
            triangles[to + 2] = idx[i + 2] + vo;
            triangles[to + 3] = matIdx;
            to += 4;
        }

        vo += count;
    }

    const triTotal = to / 4;

    // Missing normals: the glTF spec calls for face normals (flat shading).
    // Here we do vertex smoothing; the difference only shows at hard edges, and
    // models without normals were never meant to have hard edges anyway.
    if (!hasNormals) {
        const synthetic = [];
        for (let t = 0; t < triTotal; t++) {
            synthetic.push({
                vertices: [
                    { pi: triangles[t * 4], ti: -1, ni: -1 },
                    { pi: triangles[t * 4 + 1], ti: -1, ni: -1 },
                    { pi: triangles[t * 4 + 2], ti: -1, ni: -1 },
                ],
            });
        }
        const smooth = computeSmoothNormals(positions, synthetic);
        for (let v = 0; v < vertexCount; v++) {
            const n = smooth[v];
            if (!n) continue;
            normals[v * 3] = n.x;
            normals[v * 3 + 1] = n.y;
            normals[v * 3 + 2] = n.z;
        }
    }

    // Tangents: use the file's when present (exporters compute them more
    // precisely); otherwise derive them from the UVs.
    const tangents = hasTangents ? gTan : computeTangents(positions, normals, hasUvs ? uvs : null, triangles, triTotal);

    // Interleaved vertex layout: pos(3) + normal(3) + uv(2) + tangent(4), matching the Mesh IR
    const vertexData = new Float32Array(vertexCount * 12);
    for (let v = 0; v < vertexCount; v++) {
        const d = v * 12;
        vertexData[d] = positions[v * 3];
        vertexData[d + 1] = positions[v * 3 + 1];
        vertexData[d + 2] = positions[v * 3 + 2];
        vertexData[d + 3] = normals[v * 3];
        vertexData[d + 4] = normals[v * 3 + 1];
        vertexData[d + 5] = normals[v * 3 + 2];
        vertexData[d + 6] = uvs[v * 2];
        vertexData[d + 7] = uvs[v * 2 + 1];
        vertexData[d + 8] = tangents[v * 4];
        vertexData[d + 9] = tangents[v * 4 + 1];
        vertexData[d + 10] = tangents[v * 4 + 2];
        vertexData[d + 11] = tangents[v * 4 + 3];
    }

    return new Mesh(
        vertexData,
        triangles,
        meshMaterials.length > 0 ? meshMaterials : [new PBRMaterial()],
        gm.name ?? name,
    );
}

/**
 * Unsupported primitives are rejected here up front, so later parsing does not
 * fail with confusing errors.
 */
function checkPrimitive(prim: GltfPrimitive): void {
    if (prim.mode !== undefined && prim.mode !== MODE_TRIANGLES) {
        throw new Error(`glTF: only triangles (mode 4) are supported, this primitive is mode ${prim.mode}`);
    }
    if (prim.attributes.POSITION === undefined) {
        throw new Error('glTF: primitive has no POSITION attribute');
    }
    if (prim.extensions && 'KHR_draco_mesh_compression' in prim.extensions) {
        throw new Error('glTF: this model uses Draco compression; decompress it first or export without compression');
    }
}

// Geometry helpers
//
// Only the glTF importer needs these: a glTF vertex table may lack normals or
// tangents, and its triangle table groups three indices (a fourth — the
// material index — is added here). Procedural geometry goes through MeshBuilder
// with complete vertex data of its own and does not need these.

interface FaceVertex {
    pi: number;
    ti: number;
    ni: number;
}

function computeTangents(
    positions: Float32Array,
    normals: Float32Array,
    uvs: Float32Array | null,
    triIndices: Uint32Array,
    triCount: number,
): Float32Array {
    const n = positions.length / 3;
    const tangents = new Float32Array(n * 4);
    const tan1 = new Float32Array(n * 3);
    const tan2 = new Float32Array(n * 3);
    const triStride = triIndices.length / triCount;

    for (let t = 0; t < triCount; t++) {
        const off = t * triStride;
        const i0 = triIndices[off],
            i1 = triIndices[off + 1],
            i2 = triIndices[off + 2];
        const p0x = positions[i0 * 3],
            p0y = positions[i0 * 3 + 1],
            p0z = positions[i0 * 3 + 2];
        const p1x = positions[i1 * 3],
            p1y = positions[i1 * 3 + 1],
            p1z = positions[i1 * 3 + 2];
        const p2x = positions[i2 * 3],
            p2y = positions[i2 * 3 + 1],
            p2z = positions[i2 * 3 + 2];

        if (uvs) {
            const uv0x = uvs[i0 * 2],
                uv0y = uvs[i0 * 2 + 1];
            const uv1x = uvs[i1 * 2],
                uv1y = uvs[i1 * 2 + 1];
            const uv2x = uvs[i2 * 2],
                uv2y = uvs[i2 * 2 + 1];
            const e1x = p1x - p0x,
                e1y = p1y - p0y,
                e1z = p1z - p0z;
            const e2x = p2x - p0x,
                e2y = p2y - p0y,
                e2z = p2z - p0z;
            const du1 = uv1x - uv0x,
                dv1 = uv1y - uv0y;
            const du2 = uv2x - uv0x,
                dv2 = uv2y - uv0y;
            const denom = du1 * dv2 - du2 * dv1;
            if (Math.abs(denom) > 1e-12) {
                const r = 1.0 / denom;
                const tx = (e1x * dv2 - e2x * dv1) * r;
                const ty = (e1y * dv2 - e2y * dv1) * r;
                const tz = (e1z * dv2 - e2z * dv1) * r;
                const bx = (e2x * du1 - e1x * du2) * r;
                const by = (e2y * du1 - e1y * du2) * r;
                const bz = (e2z * du1 - e1z * du2) * r;
                for (const vi of [i0, i1, i2]) {
                    tan1[vi * 3] += tx;
                    tan1[vi * 3 + 1] += ty;
                    tan1[vi * 3 + 2] += tz;
                    tan2[vi * 3] += bx;
                    tan2[vi * 3 + 1] += by;
                    tan2[vi * 3 + 2] += bz;
                }
            }
        }
    }

    for (let i = 0; i < n; i++) {
        const nx = normals[i * 3],
            ny = normals[i * 3 + 1],
            nz = normals[i * 3 + 2];
        const tx = tan1[i * 3],
            ty = tan1[i * 3 + 1],
            tz = tan1[i * 3 + 2];
        const dotNT = nx * tx + ny * ty + nz * tz;
        let tanX = tx - dotNT * nx,
            tanY = ty - dotNT * ny,
            tanZ = tz - dotNT * nz;
        let tanLen = Math.sqrt(tanX * tanX + tanY * tanY + tanZ * tanZ);
        if (tanLen > 1e-12) {
            tanX /= tanLen;
            tanY /= tanLen;
            tanZ /= tanLen;
        } else {
            const absNx = Math.abs(nx),
                absNy = Math.abs(ny);
            let fx = 0,
                fy = 1,
                fz = 0;
            if (absNx < absNy) {
                fx = 1;
                fy = 0;
            }
            const fd = fx * nx + fy * ny + fz * nz;
            tanX = fx - fd * nx;
            tanY = fy - fd * ny;
            tanZ = fz - fd * nz;
            tanLen = Math.sqrt(tanX * tanX + tanY * tanY + tanZ * tanZ);
            if (tanLen > 1e-12) {
                tanX /= tanLen;
                tanY /= tanLen;
                tanZ /= tanLen;
            }
        }
        const bx = tan2[i * 3],
            by = tan2[i * 3 + 1],
            bz = tan2[i * 3 + 2];
        const handedness =
            (ny * tanZ - nz * tanY) * bx + (nz * tanX - nx * tanZ) * by + (nx * tanY - ny * tanX) * bz < 0 ? -1 : 1;
        tangents[i * 4] = tanX;
        tangents[i * 4 + 1] = tanY;
        tangents[i * 4 + 2] = tanZ;
        tangents[i * 4 + 3] = handedness;
    }
    return tangents;
}

function computeSmoothNormals(
    positions: Float32Array,
    faces: { vertices: FaceVertex[] }[],
): Record<number, { x: number; y: number; z: number }> {
    const acc: Record<number, { x: number; y: number; z: number }> = {};
    const addN = (vi: number, nx: number, ny: number, nz: number) => {
        if (acc[vi]) {
            acc[vi].x += nx;
            acc[vi].y += ny;
            acc[vi].z += nz;
        } else {
            acc[vi] = { x: nx, y: ny, z: nz };
        }
    };
    for (const face of faces) {
        const fv = face.vertices;
        for (let i = 1; i < fv.length - 1; i++) {
            const i0 = fv[0].pi,
                i1 = fv[i].pi,
                i2 = fv[i + 1].pi;
            const ax = positions[i0 * 3],
                ay = positions[i0 * 3 + 1],
                az = positions[i0 * 3 + 2];
            const bx = positions[i1 * 3],
                by = positions[i1 * 3 + 1],
                bz = positions[i1 * 3 + 2];
            const cx = positions[i2 * 3],
                cy = positions[i2 * 3 + 1],
                cz = positions[i2 * 3 + 2];
            let nx = (by - ay) * (cz - az) - (bz - az) * (cy - ay),
                ny = (bz - az) * (cx - ax) - (bx - ax) * (cz - az),
                nz = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
            const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
            if (len > 1e-12) {
                nx /= len;
                ny /= len;
                nz /= len;
            }
            addN(i0, nx, ny, nz);
            addN(i1, nx, ny, nz);
            addN(i2, nx, ny, nz);
        }
    }
    for (const vi of Object.keys(acc)) {
        const nrm = acc[+vi];
        const len = Math.sqrt(nrm.x * nrm.x + nrm.y * nrm.y + nrm.z * nrm.z);
        if (len > 1e-12) {
            nrm.x /= len;
            nrm.y /= len;
            nrm.z /= len;
        }
    }
    return acc;
}

// Nodes

/**
 * Accumulates transforms down the node tree, producing one Instance per node
 * with a mesh.
 *
 * Transforms stay on the Instance instead of being baked into vertices: when
 * several nodes reuse one mesh, the geometry should exist only once (the BLAS
 * is built per mesh).
 */
function buildInstances(doc: GltfDocument, meshes: Mesh[]): Instance[] {
    const out: Instance[] = [];
    const sceneIndex = doc.scene ?? 0;
    const roots = doc.scenes?.[sceneIndex]?.nodes ?? doc.nodes.map((_, i) => i);

    const walk = (nodeIndex: number, parent: Float32Array): void => {
        const node = doc.nodes[nodeIndex];
        if (!node) return;
        const world = mulMat4(parent, localMatrix(node));

        if (node.mesh !== undefined) {
            const mesh = meshes[node.mesh];
            if (mesh) out.push(new InstanceImpl(mesh, world));
        }
        for (const child of node.children ?? []) walk(child, world);
    };

    const identity = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
    for (const root of roots) walk(root, identity);
    return out;
}

function localMatrix(node: GltfNode): Float32Array {
    if (node.matrix) return new Float32Array(node.matrix);

    const [tx, ty, tz] = node.translation ?? [0, 0, 0];
    const [x, y, z, w] = node.rotation ?? [0, 0, 0, 1];
    const [sx, sy, sz] = node.scale ?? [1, 1, 1];

    // Quaternion → rotation matrix (column-major), then right-multiply by
    // scale and fill in translation
    const x2 = x + x,
        y2 = y + y,
        z2 = z + z;
    const xx = x * x2,
        xy = x * y2,
        xz = x * z2;
    const yy = y * y2,
        yz = y * z2,
        zz = z * z2;
    const wx = w * x2,
        wy = w * y2,
        wz = w * z2;
    return new Float32Array([
        (1 - (yy + zz)) * sx,
        (xy + wz) * sx,
        (xz - wy) * sx,
        0,
        (xy - wz) * sy,
        (1 - (xx + zz)) * sy,
        (yz + wx) * sy,
        0,
        (xz + wy) * sz,
        (yz - wx) * sz,
        (1 - (xx + yy)) * sz,
        0,
        tx,
        ty,
        tz,
        1,
    ]);
}

/** Column-major 4×4 multiply, out = a * b (b first, then a). */
function mulMat4(a: Float32Array, b: Float32Array): Float32Array {
    const out = new Float32Array(16);
    for (let c = 0; c < 4; c++) {
        for (let r = 0; r < 4; r++) {
            let sum = 0;
            for (let k = 0; k < 4; k++) sum += a[k * 4 + r] * b[c * 4 + k];
            out[c * 4 + r] = sum;
        }
    }
    return out;
}

// Textures

function resolveUri(uri: string, baseUrl: string): string {
    if (/^https?:\/\//.test(uri) || uri.startsWith('data:') || uri.startsWith('blob:')) return uri;
    return baseUrl + uri.replace(/^\.\//, '');
}

/**
 * Stable identifier of a texture source: an absolute URL for external files,
 * or file + image index for embedded ones.
 */
function imageKey(ctx: LoadContext, imageIndex: number): string | null {
    const image = ctx.doc.images?.[imageIndex];
    if (!image) return null;
    return image.uri ? resolveUri(image.uri, ctx.baseUrl) : `${ctx.baseUrl}#image=${imageIndex}`;
}

/** Image index → directly fetchable URL. Embedded images become blob URLs (once). */
function imageUrl(ctx: LoadContext, imageIndex: number): Promise<string | null> {
    const cached = ctx.imageUrls.get(imageIndex);
    if (cached) return cached;
    const job = (async (): Promise<string | null> => {
        const image = ctx.doc.images?.[imageIndex];
        if (!image) return null;
        if (image.uri) return resolveUri(image.uri, ctx.baseUrl);
        if (image.bufferView === undefined) return null;

        const key = imageKey(ctx, imageIndex);
        if (!key) return null;
        let blob = derivedBlobs.get(key);
        if (!blob) {
            const bv = ctx.doc.bufferViews[image.bufferView];
            const bytes = new Uint8Array(ctx.buffers[bv.buffer], bv.byteOffset ?? 0, bv.byteLength);
            blob = Promise.resolve(URL.createObjectURL(new Blob([bytes], { type: image.mimeType ?? 'image/png' })));
            derivedBlobs.set(key, blob);
        }
        return blob;
    })();
    ctx.imageUrls.set(imageIndex, job);
    return job;
}

/** Texture index → directly fetchable URL. */
function textureUrl(ctx: LoadContext, index: number): Promise<string | null> {
    const cached = ctx.textureUrls.get(index);
    if (cached) return cached;
    const job = (async (): Promise<string | null> => {
        const source = ctx.doc.textures?.[index]?.source;
        return source === undefined ? null : await imageUrl(ctx, source);
    })();
    ctx.textureUrls.set(index, job);
    return job;
}

/**
 * Packed map → single-channel map (copies the chosen channel into RGB).
 *
 * glTF packs channels: roughness in G, metallic in B, occlusion in R. The whole
 * metallicRoughness map goes straight into the roughness array, taking G/B as
 * needed; only AO passes through here — the occlusion array samples read .r,
 * and the R channel is exactly AO. Copying it into RGB only matches the other
 * textures' habit of "single-channel maps repeated across RGB"; it is not
 * required, but costs just one transcode.
 *
 * Only reordering, no color conversion (`colorSpaceConversion: 'none'`), so
 * values are not distorted. Derived results are cached by source texture +
 * channel, so one ORM map referenced by several materials is transcoded once.
 */
function channelUrl(
    ctx: LoadContext,
    index: number,
    channel: number,
    tag: string,
    cache: Map<number, Promise<string | null>>,
): Promise<string | null> {
    const cached = cache.get(index);
    if (cached) return cached;
    const job = (async (): Promise<string | null> => {
        // Key by source texture rather than glTF material: one ARM map referenced
        // by several materials is transcoded once, and reloading after a scene
        // switch still hits the same cache entry instead of consuming another
        // layer.
        const tex = ctx.doc.textures?.[index];
        const key = tex?.source !== undefined ? imageKey(ctx, tex.source) : null;
        if (!key) return null;
        const id = `${tag}:${key}`;
        const hit = derivedBlobs.get(id);
        if (hit) return hit;

        const job = (async (): Promise<string | null> => {
            const url = await textureUrl(ctx, index);
            if (!url) return null;
            const resp = await fetch(url);
            if (!resp.ok) return null;
            const bitmap = await createImageBitmap(await resp.blob(), { colorSpaceConversion: 'none' });
            const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
            const c2d = canvas.getContext('2d');
            if (!c2d) throw new Error('glTF: 2D context unavailable, cannot repack metallicRoughness');
            c2d.drawImage(bitmap, 0, 0);
            const px = c2d.getImageData(0, 0, bitmap.width, bitmap.height);
            const d = px.data;
            for (let i = 0; i < d.length; i += 4) {
                const v = d[i + channel];
                d[i] = v;
                d[i + 1] = v;
                d[i + 2] = v;
            }
            c2d.putImageData(px, 0, 0);
            bitmap.close();
            return URL.createObjectURL(await canvas.convertToBlob({ type: 'image/png' }));
        })();
        derivedBlobs.set(id, job);
        return job;
    })();
    cache.set(index, job);
    return job;
}

/**
 * Texture index → decoded Image.
 *
 * The URL layer exists only to hit the decode cache (embedded images must first
 * become blob URLs): loadTexture deduplicates by kind + URL, so one image
 * referenced by several materials yields the same Image object.
 */
async function textureImage(ctx: LoadContext, index: number, kind: TextureKind): Promise<Image | undefined> {
    const url = await textureUrl(ctx, index);
    return url ? await loadTexture(url, kind) : undefined;
}

/** R channel of occlusion → AO map. */
function occlusionUrl(ctx: LoadContext, index: number): Promise<string | null> {
    return channelUrl(ctx, index, 0, 'ao', ctx.occlusionUrls);
}

// Materials

async function convertMaterial(gm: GltfMaterial, ctx: LoadContext): Promise<PBRMaterial> {
    const pbr = gm.pbrMetallicRoughness ?? {};
    const [r, g, b, a] = pbr.baseColorFactor ?? [1, 1, 1, 1];
    const [er, eg, eb] = gm.emissiveFactor ?? [0, 0, 0];
    const strength = gm.extensions?.KHR_materials_emissive_strength?.emissiveStrength ?? 1;
    const options: PBRMaterialOptions = {
        emission: { r: er * strength, g: eg * strength, b: eb * strength, a: 1 },
        baseColor: { r, g, b, a },
        metallic: pbr.metallicFactor ?? 1,
        roughness: pbr.roughnessFactor ?? 1,
        normalScale: gm.normalTexture?.scale ?? 1,
        occlusionStrength: gm.occlusionTexture?.strength ?? 1,
    };

    if (pbr.baseColorTexture) {
        options.baseColorTexture = await textureImage(ctx, pbr.baseColorTexture.index, 'albedo');
    }
    if (gm.normalTexture) {
        options.normalTexture = await textureImage(ctx, gm.normalTexture.index, 'normal');
    }
    if (pbr.metallicRoughnessTexture) {
        // Keep the whole RGBA map intact: G is roughness and B is metallic, and
        // the shader takes what it needs. Earlier this extracted only G and
        // dropped B — metallic then stayed stuck at its scalar value.
        options.metallicRoughnessTexture = await textureImage(ctx, pbr.metallicRoughnessTexture.index, 'roughness');
    }
    if (gm.occlusionTexture) {
        const url = await occlusionUrl(ctx, gm.occlusionTexture.index);
        if (url) options.occlusionTexture = await loadTexture(url, 'occlusion');
    }

    return new PBRMaterial(options);
}

// Placement

placeInstances = (instances, p) => {
    const s = p.scale ?? 1;
    const cy = Math.cos(p.rotationY ?? 0);
    const sy = Math.sin(p.rotationY ?? 0);
    const cx = Math.cos(p.rotationX ?? 0);
    const sx = Math.sin(p.rotationX ?? 0);
    // Ry · Rx, column-major
    const m = Mat4.from(
        new Float32Array([
            cy * s,
            0,
            -sy * s,
            0,
            sy * sx * s,
            cx * s,
            cy * sx * s,
            0,
            sy * cx * s,
            -sx * s,
            cy * cx * s,
            0,
            p.position.x,
            p.position.y,
            p.position.z,
            1,
        ]),
    );
    for (const inst of instances) {
        inst.transform = m.multiply(Mat4.from(inst.transform)).raw();
    }
};

/** Assignment of the contract-layer binding. */
Instance = InstanceImpl;
