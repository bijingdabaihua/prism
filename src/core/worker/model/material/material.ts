import type { Color } from '../../../math/types';
import type { Image } from '../image';
import type { ZipArchive } from '../zip';
import { loadImage, loadTexture } from '../image';
import { contentCacheId, isAbsoluteUri, parseZip, readZipEntry, resolveZipPath, zipEntryUrl } from '../zip';

/**
 * The renderer's unified material representation.
 *
 * glTF materials are mapped to constructor options by the importer in ir/instance.ts; .blend
 * parsing lives in this file's implementation layer and is driven by `buildByBlend`. The
 * mapping is lossy, and each importer documents its own losses.
 * GPU packing is in engine/model/material/packing.ts.
 *
 * The file has two layers: contract (all exports) and implementation (no exports);
 * see AGENTS.md for the convention.
 */

// ══════════════════ Contract layer (exports) ═════════════════

/**
 * Class binding for PBRMaterial: constructs a material; the static factory `buildByBlend`
 * constructs one from a material archive (a Blob, zip — Poly Haven style: one `.blend`
 * plus a texture directory inside).
 *
 * Choosing `.blend` over glTF: glTF has no displacement, so the author's displacement
 * settings (Blender's Midlevel / Scale) would have to be written into extras by hand,
 * duplicating the single source of truth; MaterialX has a displacement node but no Midlevel,
 * and Poly Haven's own .mtlx and .blend Scale values disagree. `.blend` is the file the
 * author actually uses, and both numbers live in it.
 *
 * Texture tiling (meters per tile — Poly Haven keeps it in the web API's `dimensions`,
 * and the file itself cannot be read for it) is not given at construction: correct it
 * with `retile` once the material is in hand.
 *
 * Only declarations here; assignment is in the implementation layer.
 */
export let PBRMaterial: {
    new (options?: PBRMaterialOptions): PBRMaterial;
    buildByBlend(blob: Blob): Promise<PBRMaterial>;
};

/**
 * The renderer's unified material representation.
 *
 * Texture fields are decoded Images, not URLs — the IR keeps no file paths; the importer
 * decodes images at scene load time, so the compiler layer only ever sees pixels in memory.
 */
export interface PBRMaterial {
    readonly baseColor: Color;
    readonly metallic: number;
    readonly roughness: number;
    /** Weight of smooth dielectric transmission, range 0–1. */
    readonly transmission: number;
    /** Index of refraction of the dielectric relative to air. */
    readonly ior: number;
    /** Linear RGB emitted radiance; values above 1 are allowed. */
    readonly emission: Color;
    readonly baseColorTexture?: Image;
    /**
     * glTF metallic-roughness packed map: G = roughness, B = metallic, R = ambient occlusion.
     * Poly Haven's arm maps (AO / Rough / Metal) use the same channel arrangement and can be
     * used directly.
     *
     * On this path the metallic / roughness scalars are **multipliers** (glTF's
     * metallicFactor / roughnessFactor), not overrides — metallic can finally vary with the map.
     */
    readonly metallicRoughnessTexture?: Image;
    /**
     * Single-channel roughness map (Poly Haven's Rough map).
     *
     * When set together with metallicRoughnessTexture, the latter wins for roughness and
     * this path only overrides roughness (metallic still comes from the packed map); the
     * same texture need not be registered twice under two kinds.
     */
    readonly roughnessTexture?: Image;
    readonly normalTexture?: Image;
    readonly normalScale: number;
    readonly occlusionTexture?: Image;
    readonly occlusionStrength: number;
    /**
     * Displacement map: a grayscale height map, tiled by its own `meters` like albedo.
     *
     * For materials that have one, triangles are tessellated and lifted by the map before
     * rendering (see `Mesh.evolve` in `ir/mesh.ts`). Only the triangle's own UVs are used,
     * not position. After evolution the map is **consumed** — the material that reaches the
     * GPU no longer has it, and the shader never knows it existed.
     *
     * Having a map requires `displacementMid` and `displacementScale`. The grid spacing is
     * not given — it is derived from the map's meters per tile and pixel width (one texel
     * per grid cell).
     */
    readonly displacement?: Image;
    /**
     * Height zero point: the grayscale value treated as "neither raised nor sunk",
     * corresponding to Blender's Displacement node Midlevel. A scan does not store absolute
     * heights, so only the material author can define the zero point.
     */
    readonly displacementMid?: number;
    /**
     * Height scale (meters): a grayscale difference of 1.0 displaces this much,
     * corresponding to Blender's Displacement node Scale.
     * Displacement = `(height - displacementMid) × displacementScale`.
     */
    readonly displacementScale?: number;

    /**
     * Resets the tiling scale of every texture: one tile covers `meters` meters; fields
     * without a texture are left untouched.
     * Returns a new material — the original material and its texture objects are unchanged
     * (it copies and then changes `Image.meters`).
     */
    retile(meters: number): PBRMaterial;

    /**
     * Removes the displacement map and its scale settings, returning a material ready for
     * the GPU.
     *
     * Called by `Mesh.evolve` when baking displacement into geometry: displacement must
     * disappear before reaching the GPU, the shader does not know it exists, and the
     * material SSBO has no slot for it. Without displacement it returns the same object
     * (`===` itself).
     */
    withoutDisplacement(): PBRMaterial;
}

/**
 * Constructor options. Omitted fields take defaults: 0.85 grey, non-metal, roughness 0.5,
 * normal strength 1, occlusion strength 1.
 */
export interface PBRMaterialOptions {
    baseColor?: Color;
    metallic?: number;
    roughness?: number;
    transmission?: number;
    ior?: number;
    emission?: Color;
    baseColorTexture?: Image;
    metallicRoughnessTexture?: Image;
    roughnessTexture?: Image;
    normalTexture?: Image;
    normalScale?: number;
    occlusionTexture?: Image;
    occlusionStrength?: number;
    displacement?: Image;
    displacementMid?: number;
    displacementScale?: number;
}

/**
 * The four channels of a material texture: determines which texture_2d_array it goes into
 * and whether sRGB decoding is needed.
 */
export type TextureKind = 'albedo' | 'normal' | 'roughness' | 'occlusion';

// ═════════════ Implementation layer (not exported) ════════════

/*
 * .blend material parsing (the implementation of buildByBlend; former ir/blend.ts merged in).
 *
 * The archive (ZIP) is unpacked by ir/zip.ts; this file only handles: finding the `.blend`
 * inside, reading blocks as structs via SDNA, extracting material parameters along a fixed
 * node chain, and resolving in-archive texture paths back to ZIP entries for their bytes.
 *
 * It does **not** evaluate arbitrary node graphs. Only the fixed Poly Haven chain is
 * recognized:
 *
 *   TexImage ─┬─→ Principled.Base Color
 *             ├─→ Principled.Roughness
 *             ├─→ NormalMap → Principled.Normal
 *             └─→ Displacement.Height → Output.Displacement
 *
 * Unknown nodes (Mix / ColorRamp / node groups ...) throw immediately. A wrong guess is
 * worse than a failed load: it makes an image look "about right" and then nobody checks again.
 */

// Archive

/**
 * Picks the .blend inside the archive: ignores macOS metadata and hidden directories,
 * preferring the shallowest one.
 */
function pickZipBlend(zip: ZipArchive): string {
    const blends = [...zip.entries.keys()].filter((name) => {
        if (name.startsWith('__MACOSX/')) return false;
        if (name.split('/').some((part) => part.startsWith('.'))) return false;
        return /\.blend$/i.test(name);
    });
    if (blends.length === 0) throw new Error(`${zip.label}: no .blend in ZIP`);
    const depth = (name: string) => name.split('/').length;
    blends.sort((a, b) => depth(a) - depth(b) || a.length - b.length || a.localeCompare(b));
    return blends[0];
}

/** Caches parse results by "source + path": parsing each archive at most once. */
const parsedBlends = new Map<string, Promise<BlendFile>>();

async function openBlend(zip: ZipArchive, blendPath: string): Promise<BlendFile> {
    const key = `${zip.cacheId}\0${blendPath}`;
    let job = parsedBlends.get(key);
    if (!job) {
        job = (async () => {
            const bytes = await readZipEntry(zip, blendPath);
            return new BlendFile(await decompress(bytes.buffer));
        })();
        parsedBlends.set(key, job);
    }
    return job;
}

// File format

interface Block {
    code: string;
    sdna: number;
    /**
     * Memory address at write time, used to resolve pointer references.
     *
     * Must be BigInt: this is a real 64-bit address from the dump (in this example
     * 0x5b46ff59012d4d80); past 2^53, number rounds to multiples of 1024 and different
     * pointers collide into one value — linked lists would then walk into other blocks.
     */
    old: bigint;
    nr: number;
    /** Offset of the block data in the file */
    offset: number;
    len: number;
    /**
     * Owning ID block (index into blocks).
     *
     * In a .blend an `old` address is only unique **within the same ID**: materials,
     * compositor and light groups each have their own node tree, and addresses are reused
     * after unload. So finding a block by DATA pointer must first be scoped to the same ID,
     * or it may fetch a same-named address in another tree.
     */
    owner: number;
}

interface FieldDef {
    name: string;
    type: string;
    offset: number;
    size: number;
    count: number;
    ptr: boolean;
}

interface StructDef {
    name: string;
    fields: Map<string, FieldDef>;
}

/** zstd-compressed .blend (Blender's default since 3.0). */
const ZSTD_MAGIC = [0x28, 0xb5, 0x2f, 0xfd];

async function decompress(buf: ArrayBuffer): Promise<ArrayBuffer> {
    const head = new Uint8Array(buf, 0, 4);
    if (!ZSTD_MAGIC.every((b, i) => head[i] === b)) return buf;
    if (typeof DecompressionStream === 'undefined') {
        throw new Error('.blend is zstd-compressed, but this browser has no DecompressionStream("zstd")');
    }
    const stream = new Blob([buf]).stream().pipeThrough(new DecompressionStream('zstd' as CompressionFormat));
    return new Response(stream).arrayBuffer();
}

/**
 * A parsed .blend: block table + SDNA.
 *
 * Read-only, little-endian only.
 */
class BlendFile {
    private readonly bytes: Uint8Array;
    private readonly view: DataView;
    private readonly structs: StructDef[] = [];
    private readonly blocks: Block[] = [];
    /** ID pointer → block (ID addresses are unique across the file). */
    private readonly byOld = new Map<bigint, Block>();
    /** ID index → (address → block); DATA pointers are only trustworthy within their owning ID. */
    private readonly byOwner = new Map<number, Map<bigint, Block>>();

    constructor(buffer: ArrayBuffer) {
        this.bytes = new Uint8Array(buffer);
        this.view = new DataView(buffer);
        this.readAll();
    }

    // Structs

    /** All blocks of a given struct, looked up by struct name. */
    public blocksOf(structName: string): Block[] {
        return this.blocks.filter((b) => this.structOf(b)?.name === structName);
    }

    public structOf(block: Block): StructDef | undefined {
        return this.structs[block.sdna];
    }

    /**
     * Finds a block by address. Given a context block, searches within the same ID,
     * otherwise falls back to the global map.
     */
    public blockAt(ptr: bigint, near?: Block): Block | undefined {
        if (!ptr) return undefined;
        if (near) {
            const scoped = this.byOwner.get(near.owner)?.get(ptr);
            if (scoped) return scoped;
        }
        return this.byOld.get(ptr);
    }

    /**
     * Reads a field. Pointers return an address (bigint); everything else returns a number
     * (char arrays return a string).
     */
    public read(block: Block, field: string, index = 0): number | string | bigint {
        const def = this.structOf(block);
        const f = def?.fields.get(field);
        if (!f) throw new Error(`.blend: ${def?.name ?? '?'} has no field ${field}`);
        const at = block.offset + f.offset + index * f.size;
        if (f.ptr) return this.view.getBigUint64(at, true);
        if (f.type === 'char') {
            let end = at;
            while (end < at + f.count * f.size && this.bytes[end] !== 0) end++;
            return new TextDecoder().decode(this.bytes.subarray(at, end));
        }
        switch (f.type) {
            case 'float':
                return this.view.getFloat32(at, true);
            case 'int':
            case 'int32_t':
                return this.view.getInt32(at, true);
            case 'short':
            case 'int16_t':
                return this.view.getInt16(at, true);
            case 'uint8_t':
                return this.bytes[at];
            case 'int8_t':
                return this.view.getInt8(at);
            default:
                throw new Error(`.blend: cannot read field type ${f.type}`);
        }
    }

    /** Follows a pointer field to the block it points at. A null pointer returns undefined. */
    public deref(block: Block, field: string): Block | undefined {
        return this.blockAt(this.read(block, field) as bigint, block);
    }

    /**
     * Reads a string field of a substruct embedded in a struct
     * (`Image.colorspace_settings.name`).
     */
    public readNestedString(block: Block, structField: string, field: string): string {
        const def = this.structOf(block);
        const outer = def?.fields.get(structField);
        const inner = this.structs.find((s) => s.name === outer?.type)?.fields.get(field);
        if (!outer || !inner) throw new Error(`.blend: ${def?.name ?? '?'}.${structField}.${field} does not exist`);
        const at = block.offset + outer.offset + inner.offset;
        let end = at;
        while (this.bytes[end] !== 0) end++;
        return new TextDecoder().decode(this.bytes.subarray(at, end));
    }

    /** ListBase { first, last }: first element plus a linked list chained through next. */
    public list(block: Block, field: string): Block[] {
        const def = this.structOf(block);
        const f = def?.fields.get(field);
        if (!f) throw new Error(`.blend: ${def?.name ?? '?'} has no field ${field}`);
        let at = this.view.getBigUint64(block.offset + f.offset, true);
        const out: Block[] = [];
        while (at) {
            const node = this.blockAt(at, block);
            if (!node || !this.structOf(node)?.fields.has('next')) break;
            out.push(node);
            at = this.read(node, 'next') as bigint;
        }
        return out;
    }

    // Parsing

    private readAll(): void {
        const h = this.bytes;
        const magic = String.fromCharCode(...h.subarray(0, 7));
        if (magic !== 'BLENDER') throw new Error('not a .blend file');

        // v0: 12-byte header ('BLENDER' + pointer width + endianness + 3-digit version)
        // v1: 17-byte header (since Blender 5.0, block header is always LargeBHead8)
        const v1 = h[7] === 0x31 && h[8] === 0x37;
        if (!v1 && h[8] !== 0x76) throw new Error('.blend: only little-endian files are supported');
        const pointerSize = v1 ? 8 : h[7] === 0x2d ? 8 : 4;
        if (pointerSize !== 8) throw new Error('.blend: only files with 64-bit pointers are supported');

        let pos = v1 ? 17 : 12;
        let currentId = -1;
        while (pos + 32 <= h.length) {
            let code: string;
            let sdna: number;
            let old: bigint;
            let len: number;
            let nr: number;
            if (v1) {
                // LargeBHead8: code, sdna, old, len, nr
                code = this.codeAt(pos);
                sdna = this.view.getInt32(pos + 4, true);
                old = this.view.getBigUint64(pos + 8, true);
                len = Number(this.view.getBigInt64(pos + 16, true));
                nr = Number(this.view.getBigInt64(pos + 24, true));
                pos += 32;
            } else {
                // SmallBHead8: code, len, old, sdna, nr
                code = this.codeAt(pos);
                len = this.view.getInt32(pos + 4, true);
                old = this.view.getBigUint64(pos + 8, true);
                sdna = this.view.getInt32(pos + 16, true);
                nr = this.view.getInt32(pos + 20, true);
                pos += 24;
            }
            if (code === 'ENDB') break;
            if (code === 'DNA1') this.parseSDNA(pos);
            // Two-character block codes are ID blocks (MA / NT / IM / OB ...); the DATA
            // blocks after one belong to it
            if (code.length === 2) currentId = this.blocks.length;
            const block: Block = { code, sdna, old, nr, offset: pos, len, owner: currentId };
            this.blocks.push(block);
            if (old) {
                this.byOld.set(old, block);
                let scoped = this.byOwner.get(block.owner);
                if (!scoped) this.byOwner.set(block.owner, (scoped = new Map()));
                scoped.set(old, block);
            }
            pos += len;
            // Blocks are 8-byte aligned
            if (pos % 8 && this.codeAtOrNull(pos) === null) pos += 8 - (pos % 8);
        }
    }

    private codeAt(pos: number): string {
        const c = this.codeAtOrNull(pos);
        if (c === null) throw new Error(`.blend: no block header at offset ${pos}`);
        return c;
    }

    /** Block code: 4 chars ('DATA') or 2 chars plus zero padding ('OB'). */
    private codeAtOrNull(pos: number): string | null {
        if (pos + 4 > this.bytes.length) return null;
        let end = pos;
        while (end < pos + 4 && this.bytes[end] !== 0) end++;
        if (end === pos) return null;
        for (let i = pos; i < end; i++) {
            const c = this.bytes[i];
            if (c < 32 || c > 126) return null;
        }
        return String.fromCharCode(...this.bytes.subarray(pos, end));
    }

    /**
     * DNA1: type names / field names / type sizes / struct definitions, all carried by the
     * file itself.
     */
    private parseSDNA(at: number): void {
        const dv = this.view;
        let p = at;
        const tag = (t: string): void => {
            const s = String.fromCharCode(...this.bytes.subarray(p, p + 4));
            if (s !== t) throw new Error(`.blend: expected ${t}, got ${s}`);
            p += 4;
        };
        const align = (): void => {
            p = (p + 3) & ~3;
        };
        const strList = (): string[] => {
            const n = dv.getInt32(p, true);
            p += 4;
            const out: string[] = [];
            for (let i = 0; i < n; i++) {
                let e = p;
                while (this.bytes[e] !== 0) e++;
                out.push(String.fromCharCode(...this.bytes.subarray(p, e)));
                p = e + 1;
            }
            align();
            return out;
        };

        tag('SDNA');
        tag('NAME');
        const names = strList();
        tag('TYPE');
        const types = strList();
        tag('TLEN');
        // TLEN has no count: the number of sizes equals the number of types
        const sizes: number[] = [];
        for (let i = 0; i < types.length; i++) {
            sizes.push(dv.getInt16(p, true));
            p += 2;
        }
        align();
        tag('STRC');
        const count = dv.getInt32(p, true);
        p += 4;

        for (let i = 0; i < count; i++) {
            const typeIdx = dv.getInt16(p, true);
            const nfields = dv.getInt16(p + 2, true);
            p += 4;
            const fields = new Map<string, FieldDef>();
            let offset = 0;
            for (let f = 0; f < nfields; f++) {
                const ft = dv.getInt16(p, true);
                const fn = dv.getInt16(p + 2, true);
                p += 4;
                const rawName = names[fn];
                const type = types[ft];
                // SDNA encodes pointers and arrays in the field name: '*next', 'name[64]'
                const ptr = rawName.startsWith('*');
                let name = rawName.replace(/^\*/, '');
                let count = 1;
                const bracket = name.indexOf('[');
                if (bracket >= 0) {
                    count = parseInt(name.slice(bracket + 1), 10);
                    name = name.slice(0, bracket);
                }
                const base = ptr ? 8 : sizes[ft];
                const align = Math.min(Math.max(base, 1), 8);
                offset = Math.ceil(offset / align) * align;
                fields.set(name, { name, type, offset, size: base, count, ptr });
                offset += base * count;
            }
            this.structs.push({ name: types[typeIdx], fields });
        }
    }
}

// Node graph → material

/**
 * Parses a `.blend` material node graph → constructor options.
 *
 * Returns options rather than a `PBRMaterial`: the material is assembled in the
 * implementation layer; this only produces constructor options.
 */
async function parseBlendMaterial(zip: ZipArchive, blend: BlendFile, blendPath: string): Promise<PBRMaterialOptions> {
    const baseDir = blendPath.substring(0, blendPath.lastIndexOf('/') + 1);

    // Node blocks all share the SDNA struct bNode; the concrete node kind is in the idname field
    const nodes = blend.blocksOf('bNode');
    const output = nodes.find((n) => blend.read(n, 'idname') === 'ShaderNodeOutputMaterial');
    if (!output) throw new Error(`${zip.label}: no Material Output node`);

    const surface = upstream(blend, output, 'Surface');
    if (!surface) throw new Error(`${zip.label}: Material Output's Surface is not connected`);
    const bsdfType = blend.read(surface, 'idname') as string;
    if (bsdfType !== 'ShaderNodeBsdfPrincipled') {
        throw new Error(`${zip.label}: only Principled BSDF is supported; this material is connected to ${bsdfType}`);
    }

    const options: PBRMaterialOptions = {};
    const metallic = scalar(blend, surface, 'Metallic');
    const roughness = scalar(blend, surface, 'Roughness');
    if (metallic !== undefined) options.metallic = metallic;
    if (roughness !== undefined) options.roughness = roughness;

    const baseColorTex = tex(blend, surface, 'Base Color');
    if (baseColorTex) {
        options.baseColorTexture = await loadTexture(await resolveTexUrl(zip, baseDir, baseColorTex.path), 'albedo');
    }

    // Roughness is a float socket in Blender; when a color map is connected it is read as
    // grayscale. Our roughness channel reads .r, and a grayscale map has equal channels, so
    // it is equivalent.
    const roughnessTex = tex(blend, surface, 'Roughness');
    if (roughnessTex) {
        options.roughnessTexture = await loadTexture(await resolveTexUrl(zip, baseDir, roughnessTex.path), 'roughness');
        options.roughness = 1;
    }

    const normalTex = tex(blend, surface, 'Normal');
    if (normalTex) {
        options.normalTexture = await loadTexture(await resolveTexUrl(zip, baseDir, normalTex.path), 'normal');
        options.normalScale = normalTex.strength ?? 1;
    }

    const height = await displacement(zip, blend, output, baseDir);
    if (height) {
        options.displacement = height.image;
        options.displacementMid = height.mid;
        options.displacementScale = height.scale;
    }

    return options;
}

/**
 * In-.blend texture path → fetchable URL: in-archive relative paths go through ZIP entries,
 * absolute URIs pass through.
 */
async function resolveTexUrl(zip: ZipArchive, baseDir: string, path: string): Promise<string> {
    if (isAbsoluteUri(path)) return path;
    const rel = path.startsWith('//') ? path.slice(2) : path;
    const entry = resolveZipPath(baseDir, rel);
    if (!zip.entries.has(entry)) {
        throw new Error(`${zip.label}: .blend references "${entry}", but the ZIP has no such file`);
    }
    return zipEntryUrl(zip, entry);
}

/** A material texture: file + decoding rules, not yet actually loaded. */
interface TexImage {
    /** Path relative to the .blend (`//textures/x.png`) */
    path: string;
    srgb: boolean;
    /** Strength on the NormalMap node */
    strength?: number;
}

const SRGB_SPACES = new Set(['sRGB', 'Filmic sRGB']);

/** Upstream node connected to an input socket. Returns undefined when nothing is linked. */
function upstream(blend: BlendFile, node: Block, socketName: string): Block | undefined {
    const socket = socketOf(blend, node, 'inputs', socketName);
    if (!socket) return undefined;
    const links = blend.blocksOf('bNodeLink');
    for (const link of links) {
        if ((blend.read(link, 'tosock') as bigint) === socket.old) {
            const from = blend.read(link, 'fromnode') as bigint;
            return blend.blockAt(from, node);
        }
    }
    return undefined;
}

/** Default scalar value on an unconnected input socket. */
function scalar(blend: BlendFile, node: Block, socketName: string): number | undefined {
    const socket = socketOf(blend, node, 'inputs', socketName);
    if (!socket) return undefined;
    const value = blend.deref(socket, 'default_value');
    if (!value) return undefined;
    return Number((blend.read(value, 'value') as number).toFixed(6));
}

function socketOf(blend: BlendFile, node: Block, list: 'inputs' | 'outputs', name: string): Block | undefined {
    for (const s of blend.list(node, list)) {
        if ((blend.read(s, 'name') as string) === name) return s;
    }
    return undefined;
}

/** Walks down through a NormalMap to find the texture and picks up its strength on the way. */
function tex(blend: BlendFile, node: Block, socketName: string): TexImage | undefined {
    let source = upstream(blend, node, socketName);
    let strength: number | undefined;
    if (source && (blend.read(source, 'idname') as string) === 'ShaderNodeNormalMap') {
        strength = scalar(blend, source, 'Strength');
        source = upstream(blend, source, 'Color');
    }
    if (!source) return undefined;
    const image = imageOf(blend, source);
    return image ? { ...image, strength } : undefined;
}

/**
 * Texture node → file path + color space. Any Mapping / Reroute in between must be an
 * identity transform.
 */
function imageOf(blend: BlendFile, node: Block): TexImage | undefined {
    const idname = blend.read(node, 'idname') as string;
    if (idname === 'NodeReroute') return rewrite(blend, node);
    if (idname !== 'ShaderNodeTexImage') {
        throw new Error(`.blend: only direct image textures are supported; this path is connected to ${idname}`);
    }

    const image = blend.deref(node, 'id');
    if (!image) return undefined;
    const path = blend.read(image, 'name') as string;
    const space = blend.readNestedString(image, 'colorspace_settings', 'name');

    // In ready-made .blend files the UV goes through a Mapping first (Location/Rotation/Scale).
    // A non-identity Mapping changes how the texture repeats, which we cannot express as
    // "meters per tile", so fail rather than guess.
    const vector = upstream(blend, node, 'Vector');
    if (vector) checkMapping(blend, vector);

    return { path, srgb: SRGB_SPACES.has(space) };
}

function rewrite(blend: BlendFile, node: Block): TexImage | undefined {
    const next = upstream(blend, node, 'Input');
    return next ? imageOf(blend, next) : undefined;
}

function checkMapping(blend: BlendFile, node: Block): void {
    const idname = blend.read(node, 'idname') as string;
    if (idname === 'NodeReroute') {
        const next = upstream(blend, node, 'Input');
        if (next) checkMapping(blend, next);
        return;
    }
    if (idname === 'ShaderNodeTexCoord') return;
    if (idname !== 'ShaderNodeMapping') {
        throw new Error(`.blend: texture coordinates support only UV / unit Mapping; connected to ${idname}`);
    }
    const scale = vectorOf(blend, node, 'Scale');
    if (scale.some((v) => Math.abs(v - 1) > 1e-6)) {
        throw new Error(`.blend: Mapping scale is not 1 (${scale.join(', ')}), which is not supported`);
    }
}

function vectorOf(blend: BlendFile, node: Block, socketName: string): number[] {
    const socket = socketOf(blend, node, 'inputs', socketName);
    const value = socket ? blend.deref(socket, 'default_value') : undefined;
    if (!value) return [];
    return [0, 1, 2].map((i) => blend.read(value, 'value', i) as number);
}

/** Displacement node → height map + zero point + scale. */
async function displacement(
    zip: ZipArchive,
    blend: BlendFile,
    output: Block,
    baseDir: string,
): Promise<{ image: Image; mid: number; scale: number } | undefined> {
    const node = upstream(blend, output, 'Displacement');
    if (!node) return undefined;
    const idname = blend.read(node, 'idname') as string;
    if (idname !== 'ShaderNodeDisplacement') {
        throw new Error(`.blend: displacement input is connected to ${idname}; only a Displacement node is supported`);
    }
    const source = upstream(blend, node, 'Height');
    if (!source) throw new Error('.blend: Displacement node\'s Height is not connected');
    const tex = imageOf(blend, source);
    if (!tex) throw new Error('.blend: Displacement\'s Height is not connected to an image');

    const mid = scalar(blend, node, 'Midlevel');
    const scale = scalar(blend, node, 'Scale');
    if (mid === undefined || scale === undefined) {
        throw new Error('.blend: Displacement node is missing Midlevel / Scale');
    }

    const image = await loadImage(await resolveTexUrl(zip, baseDir, tex.path), { srgb: tex.srgb, layout: 'texture' });
    return { image, mid, scale };
}

// The material itself

class PBRMaterialImpl implements PBRMaterial {
    public static async buildByBlend(blob: Blob): Promise<PBRMaterial> {
        const buf = await blob.arrayBuffer();
        const zip = parseZip(buf, 'material archive', contentCacheId(buf));
        const blendPath = pickZipBlend(zip);
        const blend = await openBlend(zip, blendPath);
        return new PBRMaterialImpl(await parseBlendMaterial(zip, blend, blendPath));
    }

    public readonly baseColor: Color;
    public readonly metallic: number;
    public readonly roughness: number;
    public readonly transmission: number;
    public readonly ior: number;
    public readonly emission: Color;
    public readonly baseColorTexture?: Image;
    public readonly metallicRoughnessTexture?: Image;
    public readonly roughnessTexture?: Image;
    public readonly normalTexture?: Image;
    public readonly normalScale: number;
    public readonly occlusionTexture?: Image;
    public readonly occlusionStrength: number;
    public readonly displacement?: Image;
    public readonly displacementMid?: number;
    public readonly displacementScale?: number;

    constructor(options: PBRMaterialOptions = {}) {
        this.emission = options.emission ?? { r: 0, g: 0, b: 0, a: 1 };
        this.baseColor = options.baseColor ?? { r: 0.85, g: 0.85, b: 0.85, a: 1 };
        this.metallic = options.metallic ?? 0;
        this.roughness = options.roughness ?? 0.5;
        this.transmission = Math.max(0, Math.min(1, options.transmission ?? 0));
        this.ior = Math.max(1, options.ior ?? 1.5);
        this.baseColorTexture = options.baseColorTexture;
        this.metallicRoughnessTexture = options.metallicRoughnessTexture;
        this.roughnessTexture = options.roughnessTexture;
        this.normalTexture = options.normalTexture;
        this.normalScale = options.normalScale ?? 1;
        this.occlusionTexture = options.occlusionTexture;
        this.occlusionStrength = options.occlusionStrength ?? 1;
        this.displacement = options.displacement;
        this.displacementMid = options.displacementMid;
        this.displacementScale = options.displacementScale;
    }

    public retile(meters: number): PBRMaterial {
        const atScale = (img: Image | undefined): Image | undefined => (img ? { ...img, meters } : undefined);
        return new PBRMaterialImpl({
            baseColor: this.baseColor,
            metallic: this.metallic,
            roughness: this.roughness,
            transmission: this.transmission,
            ior: this.ior,
            emission: this.emission,
            baseColorTexture: atScale(this.baseColorTexture),
            metallicRoughnessTexture: atScale(this.metallicRoughnessTexture),
            roughnessTexture: atScale(this.roughnessTexture),
            normalTexture: atScale(this.normalTexture),
            normalScale: this.normalScale,
            occlusionTexture: atScale(this.occlusionTexture),
            occlusionStrength: this.occlusionStrength,
            displacement: atScale(this.displacement),
            displacementMid: this.displacementMid,
            displacementScale: this.displacementScale,
        });
    }

    public withoutDisplacement(): PBRMaterial {
        if (!this.displacement) return this;
        const { displacement: _image, displacementMid: _mid, displacementScale: _scale, ...rest } = this;
        return new PBRMaterialImpl(rest);
    }
}

/** Assignment of the contract-layer class binding. */
PBRMaterial = PBRMaterialImpl;
