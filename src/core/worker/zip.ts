/**
 * ZIP 容器读取：中央目录解析、条目解压、条目 → 稳定复用的 blob URL。
 *
 * 下载来的资产包（glTF 的 .zip、Poly Haven 材质包）只解需要的条目，把内部相对
 * 路径换成 blob URL；之后仍走原来的外链流程，不为 ZIP 分叉解码器。
 * 只支持普通 ZIP（单卷、非加密、非 ZIP64）；不支持的情况明确报错。
 *
 * 文件分两层：契约层（全部导出）+ 实现层（全部不导出），规范见 CODE_STYLE.md。
 */

// ══════════════════════════ 契约层（导出） ══════════════════════════

export interface ZipEntry {
    method: number;
    compressedSize: number;
    uncompressedSize: number;
    /** 该条目数据在 ZIP 文件里的起始字节。 */
    dataOffset: number;
}

export interface ZipArchive {
    buffer: ArrayBuffer;
    entries: Map<string, ZipEntry>;
    /** 同一个来源跨多次加载的稳定标识，用来复用解压结果与 blob URL。 */
    cacheId: string;
    label: string;
}

/** 源字节是不是 ZIP（本地文件头，或空 ZIP 直接就是 EOCD）。 */
export let isZip: (buf: ArrayBuffer) => boolean;

/**
 * 源字节的内容指纹（FNV-1a）。
 *
 * Blob 经 postMessage 到 Worker 后会变成新对象，按对象身份缓存对「外部一个
 * 一个传同一个 Blob」无效；按内容算才能跨消息复用 ZIP 解压结果和贴图层。
 */
export let contentCacheId: (buf: ArrayBuffer) => string;

/**
 * 读 EOCD → 中央目录 → 条目表。
 *
 * 只支持普通 ZIP（单卷、非加密、非 ZIP64）。素材包通常都是这种；不支持
 * 的情况明确报错，免得后面拿一堆乱码去喂解析器。
 */
export let parseZip: (buf: ArrayBuffer, label: string, cacheId: string) => ZipArchive;

/** 解出一个条目的原始字节；STORED 直接切片，DEFLATE 走 DecompressionStream。 */
export let readZipEntry: (zip: ZipArchive, path: string) => Promise<Uint8Array<ArrayBuffer>>;

/** ZIP 条目 → 可 fetch 的 blob URL；同一个来源的同一个路径只解压 / 建 URL 一次。 */
export let zipEntryUrl: (zip: ZipArchive, path: string) => Promise<string>;

/** dir（ZIP 内目录，以 '/' 结尾）下的相对 URI 解析成规范化 ZIP 路径。 */
export let resolveZipPath: (dir: string, uri: string) => string;

/** `https:` / `data:` 这类绝对 URI 原样使用，不进 ZIP 查表。 */
export let isAbsoluteUri: (uri: string) => boolean;

// ══════════════════════════ 实现层（不导出） ══════════════════════════

const ZIP_LOCAL = 0x04034b50;
const ZIP_CENTRAL = 0x02014b50;
const ZIP_END = 0x06054b50;
const ZIP_METHOD_STORED = 0;
const ZIP_METHOD_DEFLATE = 8;

/** ZIP 条目 → 已解压的 blob URL；键按「来源 + 路径」，跨加载复用。 */
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
    if (eocd < 0) throw new Error(`${label}: ZIP 里找不到中央目录（文件损坏或不是 ZIP）`);

    const count = view.getUint16(eocd + 10, true);
    let offset = view.getUint32(eocd + 16, true);
    const entries = new Map<string, ZipEntry>();
    const decoder = new TextDecoder();

    for (let i = 0; i < count; i++) {
        if (offset + 46 > buf.byteLength || view.getUint32(offset, true) !== ZIP_CENTRAL) {
            throw new Error(`${label}: ZIP 中央目录损坏`);
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
            throw new Error(`${label}: ZIP 文件名越界`);
        }
        const name = decoder
            .decode(new Uint8Array(buf, offset + 46, nameLength))
            .replace(/\\/g, '/')
            .replace(/^\/+/, '');

        if (flags & 1) throw new Error(`${label}: 不支持加密 ZIP（${name}）`);
        if (compressedSize === 0xffffffff || uncompressedSize === 0xffffffff || localOffset === 0xffffffff) {
            throw new Error(`${label}: 不支持 ZIP64（${name}）`);
        }

        if (name && !name.endsWith('/')) {
            if (localOffset + 30 > buf.byteLength || view.getUint32(localOffset, true) !== ZIP_LOCAL) {
                throw new Error(`${label}: ZIP 条目头损坏（${name}）`);
            }
            const localName = view.getUint16(localOffset + 26, true);
            const localExtra = view.getUint16(localOffset + 28, true);
            const dataOffset = localOffset + 30 + localName + localExtra;
            if (dataOffset + compressedSize > buf.byteLength) {
                throw new Error(`${label}: ZIP 条目越界（${name}）`);
            }
            entries.set(name, { method, compressedSize, uncompressedSize, dataOffset });
        }

        offset += 46 + nameLength + extraLength + commentLength;
    }

    return { buffer: buf, entries, cacheId, label };
};

readZipEntry = async (zip, path) => {
    const entry = zip.entries.get(path);
    if (!entry) throw new Error(`${zip.label}: ZIP 里没有 "${path}"`);

    const raw = new Uint8Array(zip.buffer, entry.dataOffset, entry.compressedSize);
    if (entry.method === ZIP_METHOD_STORED) return raw.slice();
    if (entry.method !== ZIP_METHOD_DEFLATE) {
        throw new Error(`${zip.label}: 不支持的 ZIP 压缩方式 ${entry.method}（${path}）`);
    }

    if (typeof DecompressionStream === 'undefined') {
        throw new Error(`${zip.label}: 这个浏览器没有 DecompressionStream，无法解压 ZIP`);
    }
    const stream = new Blob([raw]).stream().pipeThrough(new DecompressionStream('deflate-raw'));
    const out = new Uint8Array(await new Response(stream).arrayBuffer());
    if (out.byteLength !== entry.uncompressedSize) {
        throw new Error(`${zip.label}: "${path}" 解压后长度不对`);
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
        // 原样使用；查不到时错误信息里还能看到原始路径
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
