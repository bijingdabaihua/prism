import type { Image } from '../image';
import type { TextureKind } from './material';
import { linearToSRGB } from '../../../math/color';

/**
 * Material textures — four texture_2d_arrays (color / normal / roughness / AO), each with a
 * full mip chain.
 *
 * Why not an atlas: in an atlas a 1024 texture occupies only a quarter page, a few material
 * sets fill up the 2048 page, and the whole page shares one mip chain, so small and large
 * textures can only use the same LOD. With separate arrays each texture gets its own layer,
 * the layer number goes into the material table, and the shader picks the LOD explicitly with
 * textureSampleLevel.
 *
 * Why arrays instead of multiple textures: the number of sampled textures per stage is
 * limited (16 by default), and one material set has four maps, so a few materials overflow it.
 * Four arrays pin the binding count at 4.
 *
 * Lifetime follows MaterialStore (cached by key) and is not cleared when the scene changes:
 * textures are assets, not scene data.
 * The four arrays are allocated on demand — with no texture registered they consume no VRAM,
 * only a 1×1 placeholder. Layer 0 is always reserved for placeholder data, and real textures
 * start at layer 1.
 *
 * The file has two layers: contract (all exports) and implementation (no exports);
 * see AGENTS.md for the convention.
 */

// ══════════════════ Contract layer (exports) ═════════════════

/**
 * Class binding for TextureSet: construction creates four 1×1 placeholder arrays and a
 * shared sampler; the full arrays are allocated on first registration.
 *
 * Only declarations here; assignment is in the implementation layer.
 */
export let TextureSet: {
    new (device: GPUDevice, size?: number, layerCount?: number): TextureSet;
};

/**
 * Material texture set: four texture_2d_arrays, deduplicating layer numbers by Image
 * object identity.
 */
export interface TextureSet {
    /**
     * Writes a texture into a layer of the corresponding array and returns the layer number.
     * The same Image (by object identity) occupies only one layer.
     *
     * Texture size must equal the array size: layers in an array share a mip chain, so
     * mismatched sizes cannot be stored.
     */
    register(kind: TextureKind, image: Image): number;
    albedoView(): GPUTextureView;
    normalView(): GPUTextureView;
    roughnessView(): GPUTextureView;
    occlusionView(): GPUTextureView;
    sampler(): GPUSampler;
    destroy(): void;
}

// ═════════════ Implementation layer (not exported) ════════════

/**
 * Color maps are stored as sRGB (8-bit precision preserves shadows), data maps store raw
 * values.
 */
const FORMAT: Record<TextureKind, GPUTextureFormat> = {
    albedo: 'rgba8unorm-srgb',
    normal: 'rgba8unorm',
    roughness: 'rgba8unorm',
    occlusion: 'rgba8unorm',
};

const ENCODE_SRGB: Record<TextureKind, boolean> = {
    albedo: true,
    normal: false,
    roughness: false,
    occlusion: false,
};

/** Placeholder color for layer 0: white, flat normal, fully rough, no occlusion. */
const PLACEHOLDER: Record<TextureKind, [number, number, number]> = {
    albedo: [1, 1, 1],
    normal: [0.5, 0.5, 1],
    roughness: [1, 1, 1],
    occlusion: [1, 1, 1],
};

interface Channel {
    texture: GPUTexture;
    view: GPUTextureView;
    /** Next free layer number; layer 0 is the placeholder. */
    next: number;
    allocated: boolean;
    /** Registered texture → layer number; deduplicated by Image object identity. */
    keys: Map<Image, number>;
}

class TextureSetImpl implements TextureSet {
    private readonly device: GPUDevice;
    private readonly size: number;
    private readonly layerCount: number;
    private readonly channels: Record<TextureKind, Channel>;
    private readonly texSampler: GPUSampler;

    constructor(device: GPUDevice, size = 1024, layerCount = 16) {
        this.device = device;
        this.size = size;
        this.layerCount = layerCount;
        this.channels = {
            albedo: this.createPlaceholder('albedo'),
            normal: this.createPlaceholder('normal'),
            roughness: this.createPlaceholder('roughness'),
            occlusion: this.createPlaceholder('occlusion'),
        };
        this.texSampler = device.createSampler({
            magFilter: 'linear',
            minFilter: 'linear',
            mipmapFilter: 'linear',
            addressModeU: 'repeat',
            addressModeV: 'repeat',
        });
    }

    public register(kind: TextureKind, image: Image): number {
        // Validate size first: ensureChannel allocates the full array, which should not claim
        // VRAM before the size is known to be right
        if (image.width !== this.size || image.height !== this.size) {
            throw new Error(
                `TextureSet: ${kind} texture is ${image.width}×${image.height}, ` +
                    `expected ${this.size}×${this.size}`,
            );
        }
        const channel = this.ensureChannel(kind);
        const cached = channel.keys.get(image);
        if (cached !== undefined) return cached;
        if (channel.next >= this.layerCount) {
            throw new Error(
                `TextureSet: ${kind} array is full (${this.layerCount} layers); raise layerCount or share textures`,
            );
        }

        const layer = channel.next++;
        channel.keys.set(image, layer);
        this.uploadMips(channel, layer, image, ENCODE_SRGB[kind]);
        return layer;
    }

    public albedoView(): GPUTextureView {
        return this.channels.albedo.view;
    }

    public normalView(): GPUTextureView {
        return this.channels.normal.view;
    }

    public roughnessView(): GPUTextureView {
        return this.channels.roughness.view;
    }

    public occlusionView(): GPUTextureView {
        return this.channels.occlusion.view;
    }

    public sampler(): GPUSampler {
        return this.texSampler;
    }

    public destroy(): void {
        for (const kind of Object.keys(this.channels) as TextureKind[]) {
            this.channels[kind].texture.destroy();
        }
    }

    /**
     * Allocates the full array on first real registration (1024² × 16 layers × mips is about
     * 84 MB; four channels total ~335 MB).
     *
     * 16 layers because in a combined scene like "one ground material + a few glTF models",
     * each model brings its own diffuse/normal/roughness maps, and 8 layers only fit two models.
     */
    private ensureChannel(kind: TextureKind): Channel {
        const channel = this.channels[kind];
        if (channel.allocated) return channel;

        const mipLevelCount = Math.floor(Math.log2(this.size)) + 1;
        const texture = this.device.createTexture({
            size: { width: this.size, height: this.size, depthOrArrayLayers: this.layerCount },
            format: FORMAT[kind],
            mipLevelCount,
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
            label: `texture-set-${kind}`,
        });
        channel.texture.destroy();
        channel.texture = texture;
        channel.view = texture.createView({ dimension: '2d-array' });
        channel.allocated = true;
        return channel;
    }

    private createPlaceholder(kind: TextureKind): Channel {
        const texture = this.device.createTexture({
            size: { width: 1, height: 1, depthOrArrayLayers: 1 },
            format: FORMAT[kind],
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
            label: `texture-set-${kind}-placeholder`,
        });
        const view = texture.createView({ dimension: '2d-array' });
        const rgb = PLACEHOLDER[kind];
        const texel = new Uint8Array(256);
        texel[0] = Math.round(rgb[0] * 255);
        texel[1] = Math.round(rgb[1] * 255);
        texel[2] = Math.round(rgb[2] * 255);
        texel[3] = 255;
        this.device.queue.writeTexture({ texture }, texel, { bytesPerRow: 256 }, { width: 1, height: 1 });
        return { texture, view, next: 1, allocated: false, keys: new Map() };
    }

    /**
     * Downsamples level by level and uploads: each level is computed from the previous one,
     * keeping mips continuous.
     */
    private uploadMips(channel: Channel, layer: number, image: Image, srgb: boolean): void {
        let width = image.width;
        let height = image.height;
        let data = image.data;
        for (let level = 0; ; level++) {
            this.writeLevel(channel, layer, level, width, height, data, srgb);
            if (width === 1 && height === 1) break;
            const nextWidth = Math.max(1, width >> 1);
            const nextHeight = Math.max(1, height >> 1);
            data = downsample(data, width, height, nextWidth, nextHeight);
            width = nextWidth;
            height = nextHeight;
        }
    }

    private writeLevel(
        channel: Channel,
        layer: number,
        level: number,
        width: number,
        height: number,
        linear: Float32Array,
        srgb: boolean,
    ): void {
        const rowBytes = width * 4;
        // writeTexture requires bytesPerRow to be a multiple of 256; small mips need
        // row-end padding.
        const bytesPerRow = Math.ceil(rowBytes / 256) * 256;
        const source = encodeRGBA8(linear, width * height, srgb);
        let bytes = source;
        if (bytesPerRow !== rowBytes) {
            bytes = new Uint8Array(bytesPerRow * height);
            for (let y = 0; y < height; y++) {
                bytes.set(source.subarray(y * rowBytes, (y + 1) * rowBytes), y * bytesPerRow);
            }
        }
        this.device.queue.writeTexture(
            { texture: channel.texture, mipLevel: level, origin: { x: 0, y: 0, z: layer } },
            bytes as any,
            { bytesPerRow, rowsPerImage: height },
            { width, height, depthOrArrayLayers: 1 },
        );
    }
}

/** Assignment of the contract-layer class binding. */
TextureSet = TextureSetImpl;

/** Linear RGB float → 8-bit RGBA. Color maps are encoded back to sRGB for hardware decoding. */
function encodeRGBA8(linear: Float32Array, pixels: number, srgb: boolean): Uint8Array {
    const out = new Uint8Array(pixels * 4);
    for (let i = 0; i < pixels; i++) {
        for (let c = 0; c < 3; c++) {
            const value = clamp01(linear[i * 3 + c]);
            const encoded = srgb ? linearToSRGB(value) : value;
            out[i * 4 + c] = Math.round(encoded * 255);
        }
        out[i * 4 + 3] = 255;
    }
    return out;
}

/** 2×2 box downsample with edge clamping. Input/output are linear RGB float. */
function downsample(src: Float32Array, sw: number, sh: number, dw: number, dh: number): Float32Array {
    const dst = new Float32Array(dw * dh * 3);
    for (let y = 0; y < dh; y++) {
        const y0 = Math.min(y * 2, sh - 1);
        const y1 = Math.min(y * 2 + 1, sh - 1);
        for (let x = 0; x < dw; x++) {
            const x0 = Math.min(x * 2, sw - 1);
            const x1 = Math.min(x * 2 + 1, sw - 1);
            const a = (y0 * sw + x0) * 3;
            const b = (y0 * sw + x1) * 3;
            const c = (y1 * sw + x0) * 3;
            const d = (y1 * sw + x1) * 3;
            const o = (y * dw + x) * 3;
            dst[o] = (src[a] + src[b] + src[c] + src[d]) * 0.25;
            dst[o + 1] = (src[a + 1] + src[b + 1] + src[c + 1] + src[d + 1]) * 0.25;
            dst[o + 2] = (src[a + 2] + src[b + 2] + src[c + 2] + src[d + 2]) * 0.25;
        }
    }
    return dst;
}

function clamp01(v: number): number {
    return v < 0 ? 0 : v > 1 ? 1 : v;
}
