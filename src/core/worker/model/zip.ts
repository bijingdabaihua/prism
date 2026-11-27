/**
 * ZIP container reading: central-directory parsing, entry decompression, and stable reusable
 * blob URLs for entries.
 *
 * Downloaded asset packs (glTF .zip files, Poly Haven material packs) have only the needed
 * entries extracted, with internal relative paths replaced by blob URLs; the normal external
 * link flow then continues, with no separate decoder path for ZIP.
 * Only ordinary ZIP files are supported (single-volume, unencrypted, not ZIP64);
 * unsupported cases raise an explicit error.
 *
 * The file has two layers: contract (all exports) and implementation (no exports);
 * see AGENTS.md for the convention.
 */

// ═════════════════ Contract layer (exports) ══════════════════

export interface ZipEntry {
    method: number;
    compressedSize: number;
    uncompressedSize: number;
    /** Starting byte of this entry's data within the ZIP file. */
    dataOffset: number;
}

export interface ZipArchive {
    buffer: ArrayBuffer;
    entries: Map<string, ZipEntry>;
    /** Stable identifier for the same source across loads, used to reuse decompressed
     * results and blob URLs. */
    cacheId: string;
    label: string;
}

/** Whether the source bytes are a ZIP (local file header, or EOCD directly for an empty ZIP). */
export let isZip: (buf: ArrayBuffer) => boolean;

/**
 * Content fingerprint of the source bytes (FNV-1a).
 *
 * A Blob becomes a new object after postMessage into the Worker, so caching by object
 * identity fails when the same Blob is passed in separately each time; hashing the content
 * allows ZIP decompression results and texture layers to be reused across messages.
 */
export let contentCacheId: (buf: ArrayBuffer) => string;

/**
 * Reads EOCD → central directory → entry table.
 *
 * Only ordinary ZIP files are supported (single-volume, unencrypted, not ZIP64). Asset packs
 * usually are; unsupported cases raise an explicit error so garbage is not fed to the parser.
 */
export let parseZip: (buf: ArrayBuffer, label: string, cacheId: string) => ZipArchive;

/** Extracts an entry's raw bytes; STORED is sliced directly, DEFLATE goes through
 * DecompressionStream. */
export let readZipEntry: (zip: ZipArchive, path: string) => Promise<Uint8Array<ArrayBuffer>>;

/** ZIP entry → fetchable blob URL; each path in a source is decompressed and URL-created once. */
export let zipEntryUrl: (zip: ZipArchive, path: string) => Promise<string>;

/** Resolves a relative URI under dir (a directory inside the ZIP ending in '/') to a
 * normalized ZIP path. */
export let resolveZipPath: (dir: string, uri: string) => string;

/** Absolute URIs such as `https:` / `data:` are used as-is and not looked up in the ZIP. */
export let isAbsoluteUri: (uri: string) => boolean;

// ════════════ Implementation layer (not exported) ═════════════

const ZIP_LOCAL = 0x04034b50;
const ZIP_CENTRAL = 0x02014b50;
const ZIP_END = 0x06054b50;
const ZIP_METHOD_STORED = 0;
const ZIP_METHOD_DEFLATE = 8;

/** ZIP entry → decompressed blob URL; keyed by source + path and reused across loads. */
const zipObjectUrls = new Map<string, Promise<string>>();

isZip = (buf) => {
    if (buf.byteLength < 4) return false;
    const sig = new DataView(buf).getUint32(0, true);
    return sig === ZIP_LOCAL || sig === ZIP_END;
};

contentCacheId = (buf) => {
    const bytes = new Uint8Array(buf);
    let hash = 0x811c9dc5;
    for (let i = 0; i < bytes.length; i++) {
        hash ^= bytes[i];
        hash = Math.imul(hash, 0x01000193);
    }
    return `content-${(hash >>> 0).toString(16)}-${bytes.length.toString(16)}`;
};

parseZip = (buf, label, cacheId) => {
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
            throw new Error(`${label}: ZIP file name is out of bounds`);
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
                throw new Error(`${label}: ZIP entry is out of bounds (${name})`);
            }
            entries.set(name, { method, compressedSize, uncompressedSize, dataOffset });
        }

        offset += 46 + nameLength + extraLength + commentLength;
    }

    return { buffer: buf, entries, cacheId, label };
};

readZipEntry = async (zip, path) => {
    const entry = zip.entries.get(path);
    if (!entry) throw new Error(`${zip.label}: ZIP has no entry "${path}"`);

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
        throw new Error(`${zip.label}: decompressed length mismatch for "${path}"`);
    }
    return out;
};

zipEntryUrl = (zip, path) => {
    const key = `${zip.cacheId}\0${path}`;
    const cached = zipObjectUrls.get(key);
    if (cached) return cached;
    const job = (async (): Promise<string> => {
        const bytes = await readZipEntry(zip, path);
        return URL.createObjectURL(new Blob([bytes]));
    })();
    zipObjectUrls.set(key, job);
    return job;
};

resolveZipPath = (dir, uri) => {
    const raw = uri.split('#')[0].split('?')[0];
    let decoded = raw;
    try {
        decoded = decodeURIComponent(raw);
    } catch {
        // Use as-is; if lookup fails the error still shows the original path
    }
    const parts: string[] = [];
    for (const part of `${dir}${decoded}`.split('/')) {
        if (!part || part === '.') continue;
        if (part === '..') parts.pop();
        else parts.push(part);
    }
    return parts.join('/');
};

isAbsoluteUri = (uri) => /^[a-z][a-z0-9+.-]*:/i.test(uri);
