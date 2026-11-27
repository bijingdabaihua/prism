import type { Image } from '../model/image';

/**
 * GPU side of environment lighting.
 *
 * Understands only the Image IR: it knows neither a URL nor whether the source
 * file is .exr or .hdr. Its outputs are an envMap texture with a mip chain
 * (sampled directly when a ray misses) and a luminance CDF (environment NEE
 * samples directions by brightness, avoiding fireflies).
 *
 * Why mips: path tracing has no screen-space derivatives, so `textureSample`
 * is unavailable and LOD must be supplied explicitly. Without mips, every ray
 * escaping into the environment fetches a single texel: a smooth surface's
 * reflection lobe lands on this 2k point cloud, adjacent samples jump between
 * bright and dark, and the result is a layer of colored grain over the glaze.
 * The mip chain lets the BSDF side fetch a value already averaged over the lobe
 * width according to roughness (see env_radiance_rough in shade.wgsl).
 *
 * The format is rgba16float rather than rgba32float: in WebGPU f16 is filterable
 * by default, while f32 filtering is a device feature that must be requested
 * separately. Half-float's range (65504) is more than enough for HDRI, where the
 * sun is only a few thousand.
 *
 * The file has two layers: contract (all exports) and implementation (no
 * exports); see AGENTS.md for the convention.
 */

// ═════════════════ Contract layer (exports) ══════════════════

/**
 * Class binding for Background: construction only holds the device; there is no
 * placeholder "no environment map" state, it simply does not exist before setImage.
 *
 * Declaration only; the assignment lives in the implementation layer.
 */
export let Background: {
    new (device: GPUDevice): Background;
};

/**
 * GPU side of environment lighting: an envMap texture with a mip chain plus a
 * luminance CDF.
 *
 * setImage() only swaps the source: each call swaps in one environment map
 * (that is how the facade's background.setImage is used) and resets to the
 * original orientation. The texture and CDF are built right there in the same
 * call; there is no "pending upload" state.
 *
 * Rotation of the environment map around +y is handled separately by
 * setRotation(): it rotates the retained source image, rebuilds the texture and
 * CDF, and sampling and pdf stay consistent. A rotated image leaves no source
 * information before it enters the GPU.
 *
 * There is no "no environment map" state: construction does not reserve a
 * placeholder either, the map simply does not exist before setImage.
 * `hasEnvironment()` is one of PathTracePass's conditions for producing a
 * frame; the other is that Model has finished compiling.
 *
 * After a swap, `version` changes: PathTracePass compares it every frame and,
 * when it differs, rebuilds the bind group and clears accumulation. The old
 * texture has already been destroyed and must not be referenced by any
 * in-flight bind group.
 */
export interface Background {
    /**
     * Sets the environment map: swaps the source and resets to the original orientation.
     *
     * The source image stays in memory for setRotation. There is only one
     * environment map, and this decoded float data buys runtime orientation
     * changes without re-decoding the file.
     */
    setImage(image: Image): void;
    /**
     * Rotates the current environment map: the source stays put, and the texture
     * and luminance CDF are rebuilt at the new angle.
     *
     * Requires setImage first. After the change version changes and accumulation
     * clears. The angle is in radians around +y, sharing the same definition as
     * `envRotation` in scene data.
     */
    setRotation(rotation: number): void;
    /**
     * Whether an environment map is available yet.
     *
     * Scenes load asynchronously after the engine is created, and frames before
     * that have no lighting: the bind group cannot be built, so PathTracePass
     * uses this to decide whether to skip the frame.
     */
    hasEnvironment(): boolean;
    /**
     * Version number of the current environment map: a change means the
     * texture / CDF were swapped.
     */
    version(): number;
    envMapView(): GPUTextureView;
    envDistributionBuffer(): GPUBuffer;
    destroy(): void;
}

// ════════════ Implementation layer (not exported) ═════════════

class BackgroundImpl implements Background {
    private envMapTex: GPUTexture | null = null;
    private envMapTexView: GPUTextureView | null = null;
    /** Luminance CDF (see buildEnvDistribution for the packed layout), used by environment NEE. */
    private envDistBuf: GPUBuffer | null = null;
    /** Unrotated source image, kept for setRotation to rebuild from; null before setImage. */
    private source: Image | null = null;
    /** Current rotation of the source image around +y (radians). */
    private rotation = 0;
    /**
     * Incremented on every image swap / rotation. The render pipeline uses it to
     * rebuild the bind group and clear accumulation.
     */
    private versionCounter = 0;
    private readonly device: GPUDevice;

    constructor(device: GPUDevice) {
        this.device = device;
    }

    public setImage(image: Image): void {
        if (image.layout !== 'equirect') {
            throw new Error(`Unsupported environment layout: ${image.layout}`);
        }
        this.upload(image, 0);
    }

    public setRotation(rotation: number): void {
        if (!this.source) {
            throw new Error('Background: no environment map yet, call setImage first');
        }
        if (rotation === this.rotation) return;
        this.upload(this.source, rotation);
    }

    public hasEnvironment(): boolean {
        return this.envMapTexView !== null;
    }

    public version(): number {
        return this.versionCounter;
    }

    public envMapView(): GPUTextureView {
        if (!this.envMapTexView) {
            throw new Error('Background: environment map is not available yet');
        }
        return this.envMapTexView;
    }

    public envDistributionBuffer(): GPUBuffer {
        if (!this.envDistBuf) {
            throw new Error('Background: environment distribution is not available (destroyed?)');
        }
        return this.envDistBuf;
    }

    public destroy(): void {
        this.release();
        this.source = null;
        this.rotation = 0;
    }

    /** Releases the current texture and CDF, returning to "no environment map yet". */
    private release(): void {
        this.envMapTex?.destroy();
        this.envMapTexView = null;
        this.envDistBuf?.destroy();
        this.envMapTex = null;
        this.envDistBuf = null;
    }

    /**
     * Source + angle → rotate → build GPU resources. On failure the old
     * environment lighting is left intact.
     */
    private upload(image: Image, rotation: number): void {
        const env = rotation === 0 ? image : rotateEquirect(image, rotation);
        // Build the new resources first, then swap out the old ones: on failure
        // the old environment lighting still works.
        const { tex, view, dist } = this.build(env);
        this.release();
        this.envMapTex = tex;
        this.envMapTexView = view;
        this.envDistBuf = dist;
        this.source = image;
        this.rotation = rotation;
        this.versionCounter++;
    }

    private build(image: Image): EnvResources {
        // Store linear HDR: Image.data is already linear RGB, so upload it as-is
        // with no tone mapping. Display encoding happens only in DisplayPass's final step.
        const mips = buildEnvMips(image);
        const tex = this.createTex(image.width, image.height, mips.length);
        try {
            const view = tex.createView();
            this.writeTex(tex, mips);
            const dist = this.createDistribution(image);
            return { tex, view, dist };
        } catch (err) {
            // Clean up allocated resources if the build fails partway, avoiding a leak.
            tex.destroy();
            throw err;
        }
    }

    /** Luminance CDF → storage buffer (used directly as array<f32> in the shader). */
    private createDistribution(image: Image): GPUBuffer {
        const { data } = buildEnvDistribution(image);
        const buffer = this.device.createBuffer({
            size: data.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            label: 'env-distribution',
        });
        try {
            this.device.queue.writeBuffer(buffer, 0, data as GPUAllowSharedBufferSource);
        } catch (err) {
            buffer.destroy();
            throw err;
        }
        return buffer;
    }

    private createTex(width: number, height: number, mipLevels: number): GPUTexture {
        return this.device.createTexture({
            size: { width, height },
            format: 'rgba16float',
            mipLevelCount: mipLevels,
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
            label: 'env-map',
        });
    }

    private writeTex(tex: GPUTexture, mips: EnvMip[]): void {
        for (let level = 0; level < mips.length; level++) {
            const { width, height, data } = mips[level]!;
            const tightRow = width * 8; // rgba16float: 8 bytes per texel
            // bytesPerRow must be 256-byte aligned; trailing mips like 1×1 and
            // 2×2 need padding when the row is too short.
            const bytesPerRow = Math.ceil(tightRow / 256) * 256;
            const tight = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
            let source = tight;
            if (bytesPerRow !== tightRow) {
                source = new Uint8Array(bytesPerRow * height);
                for (let y = 0; y < height; y++) {
                    source.set(tight.subarray(y * tightRow, (y + 1) * tightRow), y * bytesPerRow);
                }
            }
            this.device.queue.writeTexture(
                { texture: tex, mipLevel: level },
                source as GPUAllowSharedBufferSource,
                { bytesPerRow, rowsPerImage: height },
                { width, height },
            );
        }
    }
}

/** Assignment of the contract-layer class binding. */
Background = BackgroundImpl;

/**
 * Rotates the environment map by angle radians around the +y axis: an equirect
 * map shifts as a whole along longitude.
 *
 * The renderer's azimuth convention is φ = atan2(z, x), with φ = 0 pointing to
 * +x and increasing toward +z. A positive angle moves features in the map (the
 * sun, bright patches on the horizon) toward decreasing φ.
 *
 * This lets scene authors align the sun: a room may face due south/north, but
 * the HDR sun may not. The luminance CDF is rebuilt from the rotated image, so
 * sampling and pdf stay consistent with no added bias.
 */
function rotateEquirect(image: Image, angle: number): Image {
    if (image.layout !== 'equirect') {
        throw new Error(`rotateEquirect: expected an equirect image, got "${image.layout}"`);
    }
    const width = image.width;
    const height = image.height;
    const shift = Math.round((angle / (2 * Math.PI)) * width);
    const k = ((shift % width) + width) % width;
    if (k === 0) return image;

    const src = image.data;
    const out = new Float32Array(src.length);
    const rowFloats = width * 3;
    const split = k * 3;
    for (let y = 0; y < height; y++) {
        const row = y * rowFloats;
        // Output column x takes input column x + k
        out.set(src.subarray(row + split, row + rowFloats), row);
        out.set(src.subarray(row, row + split), row + rowFloats - split);
    }
    return { ...image, data: out };
}

interface EnvResources {
    tex: GPUTexture;
    view: GPUTextureView;
    dist: GPUBuffer;
}

/** One mip level: rgba16float with tightly packed rows (padding is done in writeTex). */
interface EnvMip {
    width: number;
    height: number;
    data: Uint16Array;
}

/**
 * Environment-map mip chain: successive 2×2 box downsampling until 1×1.
 *
 * In an equirect projection the solid angle covered by row y is ∝ sinθ, so each
 * of the four taps is weighted by its own row's sinθ. A plain average would
 * overcount the polar rows and make the zenith too bright in reflections.
 * Longitude is uniform, so the two horizontal taps wrap directly (going out of
 * bounds in x wraps to the other side, not clamped).
 *
 * Each mip's blur radius corresponds to the BSDF lobe width; see
 * env_radiance_rough in shade.wgsl for the lookup formula. This function only
 * lays out the image with the correct area weights.
 */
function buildEnvMips(image: Image): EnvMip[] {
    let width = image.width;
    let height = image.height;
    let current = new Float32Array(width * height * 3);
    for (let i = 0; i < width * height; i++) {
        current[i * 3] = image.data[i * 3]!;
        current[i * 3 + 1] = image.data[i * 3 + 1]!;
        current[i * 3 + 2] = image.data[i * 3 + 2]!;
    }

    const mips: EnvMip[] = [encodeLevel(current, width, height)];
    while (width > 1 || height > 1) {
        const nextWidth = Math.max(1, width >> 1);
        const nextHeight = Math.max(1, height >> 1);
        const next = new Float32Array(nextWidth * nextHeight * 3);
        for (let y = 0; y < nextHeight; y++) {
            const y0 = Math.min(y * 2, height - 1);
            const y1 = Math.min(y * 2 + 1, height - 1);
            const w0 = Math.max(Math.sin((Math.PI * (y0 + 0.5)) / height), 1e-6);
            const w1 = Math.max(Math.sin((Math.PI * (y1 + 0.5)) / height), 1e-6);
            const inv = 1 / (2 * (w0 + w1));
            for (let x = 0; x < nextWidth; x++) {
                const x0 = (x * 2) % width;
                const x1 = (x * 2 + 1) % width;
                const i00 = (y0 * width + x0) * 3;
                const i01 = (y0 * width + x1) * 3;
                const i10 = (y1 * width + x0) * 3;
                const i11 = (y1 * width + x1) * 3;
                const o = (y * nextWidth + x) * 3;
                for (let c = 0; c < 3; c++) {
                    next[o + c] =
                        (current[i00 + c]! * w0 +
                            current[i01 + c]! * w0 +
                            current[i10 + c]! * w1 +
                            current[i11 + c]! * w1) *
                        inv;
                }
            }
        }
        mips.push(encodeLevel(next, nextWidth, nextHeight));
        current = next;
        width = nextWidth;
        height = nextHeight;
    }
    return mips;
}

/** Linear RGB → rgba16float (alpha padded to 1). */
function encodeLevel(rgb: Float32Array, width: number, height: number): EnvMip {
    const data = new Uint16Array(width * height * 4);
    for (let i = 0, n = width * height; i < n; i++) {
        data[i * 4] = toHalf(rgb[i * 3]!);
        data[i * 4 + 1] = toHalf(rgb[i * 3 + 1]!);
        data[i * 4 + 2] = toHalf(rgb[i * 3 + 2]!);
        data[i * 4 + 3] = 0x3c00; // 1.0
    }
    return { width, height, data };
}

/**
 * float32 → float16 (round to nearest).
 *
 * Clamp to the half-float finite range first: the sun in an environment map
 * occasionally exceeds 65504, and turning it into Inf would poison the whole
 * path's radiance into NaN.
 */
const halfBits = new Uint32Array(1);
const halfValue = new Float32Array(halfBits.buffer);
function toHalf(value: number): number {
    halfValue[0] = Math.min(Math.max(value, -65504), 65504);
    const x = halfBits[0]!;
    const sign = (x >>> 16) & 0x8000;
    const exponent = (x >>> 23) & 0xff;
    let mantissa = x & 0x7fffff;
    let e = exponent - 127 + 15;
    if (e >= 0x1f) return sign | 0x7c00;
    if (e <= 0) {
        if (e < -10) return sign;
        mantissa |= 0x800000;
        const shift = 14 - e;
        return sign | ((mantissa >>> shift) + ((mantissa >>> (shift - 1)) & 1));
    }
    let m = (mantissa >>> 13) + ((mantissa >>> 12) & 1);
    if (m === 0x400) {
        m = 0;
        e += 1;
        if (e >= 0x1f) return sign | 0x7c00;
    }
    return sign | (e << 10) | m;
}

// Luminance CDF

/**
 * Importance-sampling distribution for the environment map: pure CPU, no GPU.
 *
 * Sampling the environment directly when a ray misses the scene is correct but
 * has enormous variance: the sun occupies only a few pixels of the environment
 * map, cosine-sampled photons hit it only occasionally, and the contribution is
 * blown up by the pdf into fireflies. This pre-bakes a 2D CDF over environment
 * directions so NEE can sample the sun directly by brightness.
 *
 * Sampling distribution: pixel weight = luminance × sinθ (in an equirect
 * projection the solid angle covered by each row is proportional to sinθ).
 * After per-row normalization, the marginal CDF is renormalized by row weights.
 *
 * Packed layout (a single Float32Array, matching array<f32> in the shader):
 *
 *   [0, height*(width+1))         conditional CDF, width+1 per row, normalized to 1 within the row
 *   [height*(width+1), +height+1) marginal CDF, normalized to 1
 */
interface EnvDistribution {
    readonly width: number;
    readonly height: number;
    /** Start index of the marginal CDF in data = height * (width + 1) */
    readonly marginalOffset: number;
    readonly data: Float32Array;
}

function buildEnvDistribution(image: Image): EnvDistribution {
    const w = image.width;
    const h = image.height;
    const stride = w + 1;
    const marginalOffset = h * stride;
    const data = new Float32Array(marginalOffset + h + 1);
    const src = image.data;

    const rowWeights = new Float64Array(h);
    let total = 0;

    for (let y = 0; y < h; y++) {
        // Row 0 of the image is the top, uv.y = (y + 0.5) / h; in the shader
        // uv.y = 1 - theta/π, so this row's pixel centers correspond to
        // theta = π * (1 - (y + 0.5) / h).
        const theta = Math.PI * (1 - (y + 0.5) / h);
        const sinTheta = Math.max(Math.sin(theta), 0);
        const base = y * stride;
        let acc = 0;
        for (let x = 0; x < w; x++) {
            const i = (y * w + x) * 3;
            const lum = 0.2126 * src[i] + 0.7152 * src[i + 1] + 0.0722 * src[i + 2];
            acc += Math.max(lum, 0) * sinTheta;
            data[base + x + 1] = acc;
        }
        rowWeights[y] = acc;
        total += acc;
    }

    if (total <= 0) {
        // All-black environment: no direction to sample, so fall back to a
        // uniform distribution (the pdf stays finite and NEE does not divide by zero).
        for (let y = 0; y < h; y++) {
            const base = y * stride;
            for (let x = 0; x < w; x++) data[base + x + 1] = x + 1;
            rowWeights[y] = 1;
        }
        total = h;
    }

    // Normalize each row to 1. All-zero rows (pure black) get weight 0, so the
    // marginal CDF never selects them, but normalization must still avoid
    // 0 * Infinity = NaN.
    for (let y = 0; y < h; y++) {
        const base = y * stride;
        const rw = rowWeights[y];
        const inv = rw > 0 ? 1 / rw : 0;
        for (let x = 1; x <= w; x++) data[base + x] *= inv;
    }

    let acc = 0;
    for (let y = 0; y < h; y++) {
        acc += rowWeights[y];
        data[marginalOffset + y + 1] = acc / total;
    }

    return { width: w, height: h, marginalOffset, data };
}
