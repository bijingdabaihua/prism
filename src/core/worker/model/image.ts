import { srgbToLinear } from '../../math/color';
import { decodeExr } from './exr';
import type { TextureKind } from './material/material';

/**
 * Images in the scene IR, plus the image and texture importers.
 *
 * Source-format-agnostic, decoded linear floating-point data. Every format
 * (.hdr / .exr / .png / .jpg …) converges here, and the render side knows only this type, so
 * it cannot tell what format the source file was.
 *
 * Decoders live in the implementation layer: `loadImage` is the entry point for any source
 * and `loadTexture` for material textures. This is the **only** place allowed to know that
 * ".exr / .hdr / .png" exist — adding a format means adding a decodeXxx plus a magic check
 * in sniff, with no change to the render layer.
 *
 * The file has two layers: contract (all exports) and implementation (no exports);
 * see AGENTS.md for the convention.
 */

// ═════════════════ Contract layer (exports) ══════════════════

/**
 * An image in the scene IR.
 *
 * Conventions:
 * - Data holds **linear** light values, not sRGB-encoded values. Display encoding happens
 *   later in the GPU compile layer.
 * - Values are not normalized and may exceed 1 (HDR environment-map radiance is naturally > 1).
 * - layout distinguishes two uses: equirect is an equidistant cylindrical environment map
 *   (2:1 aspect ratio), texture is a tileable 2D map. To support cubemaps later, add a variant
 *   here rather than changing the render layer.
 */
export interface Image {
    readonly width: number;
    readonly height: number;
    /** 3-channel RGB, row-major, length = width * height * 3 */
    readonly data: Float32Array;
    readonly layout: 'equirect' | 'texture';
    /**
     * Real-world size covered by one texture cell (meters). Absent means 1 m per cell.
     *
     * The only conversion between image space and physical space, and only the asset source
     * can provide it — texture files do not carry this information (Poly Haven's `dimensions`
     * field is in millimeters). Shading folds it into texel coverage (1/meters repeats per
     * meter), displacement folds metric UVs into texture repeats, and both read the same value.
     */
    readonly meters?: number;
}

export interface ImageOptions {
    /**
     * Whether to decode a bitmap from sRGB to linear. True (the default) for color maps;
     * **data maps** such as normal and roughness must be false — their values are not colors,
     * and an sRGB decode would darken roughness overall, making materials look too dark or
     * too bright.
     */
    srgb?: boolean;
    /** Output use. Environment maps are equirect; material textures are texture. */
    layout?: 'equirect' | 'texture';
    /** Real-world size covered by one texture cell (meters); see `Image.meters`. */
    meters?: number;
}

/**
 * Image entry point for any source: URLs are fetched internally, Blobs are read directly.
 *
 * Format detection and decoding always happen in the implementation layer — callers only
 * supply the source and need not know whether it is EXR, HDR or a bitmap.
 */
export let loadImage: (source: string | Blob, options?: ImageOptions) => Promise<Image>;

/**
 * Decodes material textures and deduplicates them.
 *
 * PBRMaterial texture fields are decoded Images, so decoding happens in the importer — here —
 * and user code awaits it while building materials. The same image (same kind + same URL) is
 * decoded once and returns the same Image object: TextureSet deduplicates layers by object
 * identity, so decoding twice would waste a layer.
 *
 * Color maps are decoded as sRGB; normal, roughness and AO keep their raw values.
 */
export let loadTexture: (url: string, kind: TextureKind, meters?: number) => Promise<Image>;

// ════════════ Implementation layer (not exported) ═════════════

loadImage = async (source, options) => {
    let buf: ArrayBuffer;
    if (typeof source === 'string') {
        const resp = await fetch(source);
        if (!resp.ok) throw new Error(`Failed to load ${source} (${resp.status})`);
        buf = await resp.arrayBuffer();
    } else {
        buf = await source.arrayBuffer();
    }
    const image = await decodeImage(buf, options);
    return {
        ...image,
        layout: options?.layout ?? image.layout,
        meters: options?.meters ?? image.meters,
    };
};

const IS_COLOR: Record<TextureKind, boolean> = {
    albedo: true,
    normal: false,
    roughness: false,
    occlusion: false,
};

const textureCache = new Map<string, Promise<Image>>();

loadTexture = (url, kind, meters) => {
    // Meters per cell is part of the cache key: one image at different scales is a different Image.
    const key = `${kind}:${url}:${meters ?? 1}`;
    let job = textureCache.get(key);
    if (!job) {
        job = loadImage(url, { srgb: IS_COLOR[kind], layout: 'texture', meters });
        textureCache.set(key, job);
    }
    return job;
};

// Decoding

/** Detects the format from magic bytes, not the extension — URL naming does not affect decoding. */
function sniff(buf: ArrayBuffer): 'exr' | 'hdr' | 'bitmap' {
    const b = new Uint8Array(buf, 0, Math.min(8, buf.byteLength));
    // OpenEXR: 0x76 0x2f 0x31 0x01
    if (b.length >= 4 && b[0] === 0x76 && b[1] === 0x2f && b[2] === 0x31 && b[3] === 0x01) {
        return 'exr';
    }
    // Radiance HDR: "#?"
    if (b.length >= 2 && b[0] === 0x23 && b[1] === 0x3f) {
        return 'hdr';
    }
    return 'bitmap';
}

async function decodeImage(buf: ArrayBuffer, options?: ImageOptions): Promise<Image> {
    switch (sniff(buf)) {
        case 'exr':
            return { ...(await decodeExr(buf)), layout: 'equirect' };
        case 'hdr':
            return decodeHdr(buf);
        default:
            return decodeBitmap(buf, options?.srgb ?? true);
    }
}

// Radiance HDR (.hdr) — RGBE / modern scanline RLE

function decodeHdr(buf: ArrayBuffer): Image {
    const bytes = new Uint8Array(buf);
    let pos = 0;

    const readLine = (): string => {
        let s = '';
        while (pos < bytes.length && bytes[pos] !== 0x0a) s += String.fromCharCode(bytes[pos++]);
        pos++;
        return s;
    };

    const magic = readLine();
    if (magic !== '#?RADIANCE' && magic !== '#?RGBE') throw new Error('Not a Radiance HDR file');

    let format = '';
    for (;;) {
        const line = readLine();
        if (line === '') break;
        if (line.startsWith('FORMAT=')) format = line.slice(7).trim();
        if (pos >= bytes.length) throw new Error('Truncated HDR header');
    }
    if (format !== '' && format !== '32-bit_rle_rgbe') {
        throw new Error(`Unsupported HDR format: ${format}`);
    }

    // Resolution line, of the form "-Y 1024 +X 2048"
    const res = readLine().trim().split(/\s+/);
    if (res.length !== 4) throw new Error(`Bad HDR resolution line: "${res.join(' ')}"`);
    const [axisY, hStr, axisX, wStr] = res;
    const height = parseInt(hStr, 10);
    const width = parseInt(wStr, 10);
    if (!(width > 0 && height > 0)) throw new Error(`Bad HDR size: ${width}x${height}`);

    const rgbe = new Uint8Array(width * height * 4);
    const scanline = new Uint8Array(width * 4);
    const yStart = axisY === '-Y' ? 0 : height - 1;
    const yStep = axisY === '-Y' ? 1 : -1;
    const yIndex = (row: number) => yStart + row * yStep;

    for (let row = 0; row < height; row++) {
        const isNewRLE =
            pos + 4 <= bytes.length &&
            bytes[pos] === 2 &&
            bytes[pos + 1] === 2 &&
            ((bytes[pos + 2] << 8) | bytes[pos + 3]) === width &&
            width >= 8 &&
            width < 32768;

        if (isNewRLE) {
            pos += 4;
            // RLE runs separately for each of the 4 channels
            for (let c = 0; c < 4; c++) {
                let x = 0;
                while (x < width) {
                    let count = bytes[pos++];
                    if (count > 128) {
                        count -= 128;
                        const v = bytes[pos++];
                        if (count === 0 || x + count > width) throw new Error('Corrupt HDR RLE');
                        for (let i = 0; i < count; i++) scanline[(x + i) * 4 + c] = v;
                    } else {
                        if (count === 0 || x + count > width) throw new Error('Corrupt HDR RLE');
                        for (let i = 0; i < count; i++) scanline[(x + i) * 4 + c] = bytes[pos++];
                    }
                    x += count;
                }
            }
        } else {
            // Old-style flat data. Legacy (1,1,1,n) run-length encoding is not supported;
            // modern exporters do not use it.
            if (pos + width * 4 > bytes.length) throw new Error('Truncated HDR scanline');
            scanline.set(bytes.subarray(pos, pos + width * 4));
            pos += width * 4;
        }

        const base = yIndex(row) * width * 4;
        if (axisX === '-X') {
            for (let x = 0; x < width; x++) {
                const src = (width - 1 - x) * 4;
                rgbe.set(scanline.subarray(src, src + 4), base + x * 4);
            }
        } else {
            rgbe.set(scanline, base);
        }
    }

    // RGBE → linear floats
    const data = new Float32Array(width * height * 3);
    for (let i = 0; i < width * height; i++) {
        const e = rgbe[i * 4 + 3];
        if (e === 0) continue;
        const f = Math.pow(2, e - 136); // 2^(e-128) / 256
        data[i * 3] = (rgbe[i * 4] + 0.5) * f;
        data[i * 3 + 1] = (rgbe[i * 4 + 1] + 0.5) * f;
        data[i * 3 + 2] = (rgbe[i * 4 + 2] + 0.5) * f;
    }
    return { width, height, data, layout: 'equirect' };
}

// Bitmaps (PNG / JPEG / WebP …) — 8-bit, decoded to linear when srgb=true

async function decodeBitmap(buf: ArrayBuffer, srgb: boolean): Promise<Image> {
    const bitmap = await createImageBitmap(new Blob([buf]), { colorSpaceConversion: 'none' });
    const { width, height } = bitmap;

    const canvas = new OffscreenCanvas(width, height);
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('2D context unavailable');
    ctx.drawImage(bitmap, 0, 0);
    const rgba = ctx.getImageData(0, 0, width, height).data;
    bitmap.close();

    const data = new Float32Array(width * height * 3);
    for (let i = 0; i < width * height; i++) {
        const r = rgba[i * 4] / 255;
        const g = rgba[i * 4 + 1] / 255;
        const b = rgba[i * 4 + 2] / 255;
        data[i * 3] = srgb ? srgbToLinear(r) : r;
        data[i * 3 + 1] = srgb ? srgbToLinear(g) : g;
        data[i * 3 + 2] = srgb ? srgbToLinear(b) : b;
    }
    return { width, height, data, layout: 'equirect' };
}
