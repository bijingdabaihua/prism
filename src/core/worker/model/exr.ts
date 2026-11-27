/**
 * Scanline EXR decoder for RGB or grayscale half/float data; supports uncompressed, ZIPS and ZIP.
 */

export let decodeExr: (buffer: ArrayBuffer) => Promise<{ width: number; height: number; data: Float32Array }>;

// ═════════ Implementation layer (not exported) ══════════

interface Channel {
    name: string;
    type: number;
    bytes: number;
}

decodeExr = async (buffer) => {
    const bytes = new Uint8Array(buffer);
    const view = new DataView(buffer);
    const textDecoder = new TextDecoder();
    let pos = 0;

    function need(count: number): void {
        if (count < 0 || pos + count > bytes.length) throw new Error('EXR file is truncated');
    }

    function u32(): number {
        need(4);
        const value = view.getUint32(pos, true);
        pos += 4;
        return value;
    }

    function cstr(): string {
        const start = pos;
        while (pos < bytes.length && bytes[pos] !== 0) pos++;
        need(1);
        return textDecoder.decode(bytes.subarray(start, pos++));
    }

    if (u32() !== 0x01312f76) throw new Error('Not an OpenEXR file');
    const version = u32();
    if ((version & 0xff) !== 2 || (version & ~0xff) !== 0) {
        throw new Error('Only single-part scanline EXR v2 is supported');
    }

    const channels: Channel[] = [];
    let compression = -1;
    let imageType = '';
    let lineOrder = -1;
    let yMin = 0;
    let width = 0;
    let height = 0;
    while (true) {
        const name = cstr();
        if (!name) break;
        const type = cstr();
        const size = u32();
        need(size);
        const start = pos;
        if (name === 'compression' && type === 'compression' && size === 1) {
            compression = bytes[pos];
        } else if (name === 'dataWindow' && type === 'box2i' && size === 16) {
            const xMin = view.getInt32(pos, true);
            yMin = view.getInt32(pos + 4, true);
            width = view.getInt32(pos + 8, true) - xMin + 1;
            height = view.getInt32(pos + 12, true) - yMin + 1;
        } else if (name === 'channels' && type === 'chlist') {
            const end = pos + size;
            while (pos < end && bytes[pos] !== 0) {
                const channelName = cstr();
                need(16);
                const pixelType = u32();
                pos += 4;
                const xSampling = u32();
                const ySampling = u32();
                if (xSampling !== 1 || ySampling !== 1) throw new Error('EXR channel subsampling is not supported');
                if (pixelType !== 1 && pixelType !== 2) throw new Error('Only EXR half/float channels are supported');
                channels.push({ name: channelName, type: pixelType, bytes: pixelType === 1 ? 2 : 4 });
            }
            if (pos !== end - 1) throw new Error('EXR channel list is corrupt');
        } else if (name === 'type' && type === 'string') {
            imageType = textDecoder.decode(bytes.subarray(pos, pos + size));
        } else if (name === 'lineOrder' && type === 'lineOrder' && size === 1) {
            lineOrder = bytes[pos];
        }
        pos = start + size;
    }

    if (!(width > 0 && height > 0) || width * height > 100_000_000) throw new Error('Invalid EXR image size');
    if (imageType && imageType !== 'scanlineimage') throw new Error(`Unsupported EXR image type ${imageType}`);
    if (lineOrder !== 0 && lineOrder !== 1 && lineOrder !== 2) throw new Error('Invalid EXR line order');
    if (![0, 2, 3].includes(compression)) throw new Error(`Unsupported EXR compression type ${compression}`);
    const grayscale = channels.length === 1 && channels[0].name === 'Y';
    if (!grayscale && !['R', 'G', 'B'].every((name) => channels.some((channel) => channel.name === name))) {
        throw new Error('EXR is missing RGB or grayscale Y channels');
    }

    const linesPerBlock = compression === 3 ? 16 : 1;
    const blocks = Math.ceil(height / linesPerBlock);
    const offsets: number[] = [];
    for (let i = 0; i < blocks; i++) {
        need(8);
        const offset = Number(view.getBigUint64(pos, true));
        pos += 8;
        if (!Number.isSafeInteger(offset) || offset + 8 > bytes.length) throw new Error('Invalid EXR block offset');
        offsets.push(offset);
    }

    const stride = width * channels.reduce((sum, channel) => sum + channel.bytes, 0);
    const data = new Float32Array(width * height * 3);
    const seen = new Uint8Array(blocks);
    for (const offset of offsets) {
        const row = view.getInt32(offset, true) - yMin;
        const size = view.getUint32(offset + 4, true);
        const lines = Math.min(linesPerBlock, height - row);
        if (row < 0 || lines <= 0 || offset + 8 + size > bytes.length) throw new Error('Invalid EXR scanline block');
        if (row % linesPerBlock !== 0 || seen[Math.floor(row / linesPerBlock)]++) {
            throw new Error('EXR scanline block is duplicated or misaligned');
        }
        const expected = stride * lines;
        let raw = bytes.subarray(offset + 8, offset + 8 + size);
        if (compression !== 0 && size < expected) {
            const stream = new Blob([new Uint8Array(raw)]).stream().pipeThrough(new DecompressionStream('deflate'));
            raw = new Uint8Array(await new Response(stream).arrayBuffer());
            if (raw.length !== expected) throw new Error('EXR decompressed length mismatch');
            for (let i = 1; i < raw.length; i++) raw[i] = raw[i - 1] + raw[i] - 128;
            const interleaved = new Uint8Array(raw.length);
            const half = Math.ceil(raw.length / 2);
            for (let i = 0; i < raw.length; i++) interleaved[i] = raw[(i & 1) ? half + (i >> 1) : i >> 1];
            raw = interleaved;
        }
        if (raw.length !== expected) throw new Error('EXR block length mismatch');
        const block = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
        for (let line = 0; line < lines; line++) {
            let channelOffset = line * stride;
            for (const channel of channels) {
                const component = grayscale ? 0 : ['R', 'G', 'B'].indexOf(channel.name);
                if (component >= 0) {
                    for (let x = 0; x < width; x++) {
                        const sampleOffset = channelOffset + x * channel.bytes;
                        const value = channel.type === 2
                            ? block.getFloat32(sampleOffset, true)
                            : halfToFloat(block.getUint16(sampleOffset, true));
                        const pixel = ((height - 1 - row - line) * width + x) * 3;
                        data[pixel + component] = value;
                        if (grayscale) {
                            data[pixel + 1] = value;
                            data[pixel + 2] = value;
                        }
                    }
                }
                channelOffset += width * channel.bytes;
            }
        }
    }
    if (seen.some((count) => count !== 1)) throw new Error('EXR is missing scanline blocks');
    return { width, height, data };
};

function halfToFloat(bits: number): number {
    const sign = bits & 0x8000 ? -1 : 1;
    const exponent = (bits >> 10) & 31;
    const fraction = bits & 1023;
    if (exponent === 0) return sign * fraction * 2 ** -24;
    if (exponent === 31) return fraction ? NaN : sign * Infinity;
    return sign * (1 + fraction / 1024) * 2 ** (exponent - 15);
}
