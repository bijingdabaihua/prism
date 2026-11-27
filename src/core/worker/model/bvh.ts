import type { AABB, Vec3 } from '../../math/types';
import type { Mesh } from './mesh';

/**
 * BVH construction: generic binned SAH (leaves hold at most two primitives, nodes are a
 * fixed 8 floats), plus a BLAS cached per mesh.
 *
 * The file has two layers: contract (all exports) and implementation (no exports);
 * see AGENTS.md for the convention.
 */

// ═════════════════ Contract layer (exports) ══════════════════

/**
 * A set of primitives with bounds → a flat BVH (8 floats per node: min.xyz / max.xyz /
 * left and right slots).
 *
 * Binned SAH: bin along each of the three axes and pick the cheapest split. The cost is
 * traversal SAH — `area(left) × count(left) + area(right) × count(right)`, i.e. how many
 * bounding-box tests a random ray is expected to perform through this tree. Construction is
 * a few times slower than a median split, but the tree is built once at load time and every
 * ray traverses it afterward.
 */
export let buildBvh: <T>(items: readonly T[], getBounds: (item: T) => AABB) => Float32Array;

/**
 * BLAS cached per mesh: multiple instances of the same mesh share one tree.
 * Rebuilding the scene keeps mesh objects unchanged, so the cache hits and nothing is recomputed.
 */
export let meshBlas: (mesh: Mesh) => Float32Array;

// ════════════ Implementation layer (not exported) ═════════════

class BvhNode {
    public bounds: AABB;
    public left = 0;
    public right = 0;
    public leftChild?: BvhNode;
    public rightChild?: BvhNode;
    constructor(bounds: AABB) {
        this.bounds = bounds;
    }
}

interface ItemData<T> {
    item: T;
    index: number;
    bounds: AABB;
    centroid: Vec3;
}

buildBvh = <T>(items: readonly T[], getBounds: (item: T) => AABB): Float32Array => {
    if (items.length === 0) return new Float32Array(0);
    const itemData: ItemData<T>[] = items.map((item, i) => {
        const b = getBounds(item);
        return {
            item,
            index: i,
            bounds: b,
            centroid: { x: (b.min.x + b.max.x) * 0.5, y: (b.min.y + b.max.y) * 0.5, z: (b.min.z + b.max.z) * 0.5 },
        };
    });
    const root = buildNode(itemData, 0);
    const flat = flatten(root);
    return flat;
};

/** Number of SAH bins. 16 is a common choice: finer bins only add noise to the gain while
 * construction slows down linearly. */
const SAH_BINS = 16;

/**
 * Depth at which to fall back to a median split.
 *
 * SAH accepts imbalance between left and right for a lower "area × count" cost, so the tree
 * can grow lopsided; the shader's traversal stack holds 64 levels (`array<u32, 64>` in
 * `bvh.wgsl`). At this depth a forced half split applies: each level then takes at least half,
 * so the tree grows at most log2(n) more levels.
 */
const SAH_MAX_DEPTH = 40;

const AXES = ['x', 'y', 'z'] as const;

/**
 * A leaf holds at most two primitives (fixed by the node format: two slots, one primitive
 * each), so this only decides how to split; whether to split is not a choice.
 */
function buildNode<T>(items: ItemData<T>[], depth: number): BvhNode {
    const count = items.length;
    if (count <= 2) return createLeaf(items);

    let split: SahSplit | null = null;
    if (depth < SAH_MAX_DEPTH) split = findSahSplit(items);
    const [left, right] = split ? partitionByBin(items, split) : medianPartition(items);

    const lChild = buildNode(left, depth + 1);
    const rChild = buildNode(right, depth + 1);
    const node = new BvhNode(mergeBounds(lChild.bounds, rChild.bounds));
    node.leftChild = lChild;
    node.rightChild = rChild;
    return node;
}

interface SahSplit {
    /** Which bin each primitive falls into (along the chosen axis). */
    bins: Uint8Array;
    /** Left half = all primitives with bin number ≤ this. */
    splitBin: number;
    cost: number;
}

function findSahSplit<T>(items: ItemData<T>[]): SahSplit | null {
    const n = items.length;
    const cmin = { x: Infinity, y: Infinity, z: Infinity };
    const cmax = { x: -Infinity, y: -Infinity, z: -Infinity };
    for (const it of items) {
        if (it.centroid.x < cmin.x) cmin.x = it.centroid.x;
        if (it.centroid.y < cmin.y) cmin.y = it.centroid.y;
        if (it.centroid.z < cmin.z) cmin.z = it.centroid.z;
        if (it.centroid.x > cmax.x) cmax.x = it.centroid.x;
        if (it.centroid.y > cmax.y) cmax.y = it.centroid.y;
        if (it.centroid.z > cmax.z) cmax.z = it.centroid.z;
    }

    let best: SahSplit | null = null;
    for (const axis of AXES) {
        const lo = cmin[axis];
        const extent = cmax[axis] - lo;
        // Centroids coincide completely on this axis; cannot split
        if (!(extent > 0)) continue;

        const scale = SAH_BINS / extent;
        const bins = new Uint8Array(n);
        const counts = new Uint32Array(SAH_BINS);
        const binBounds: AABB[] = Array.from({ length: SAH_BINS }, emptyBounds);
        for (let i = 0; i < n; i++) {
            // The largest centroid lands exactly on SAH_BINS; clamp it to the last bin
            const b = Math.min(SAH_BINS - 1, Math.floor((items[i]!.centroid[axis] - lo) * scale));
            bins[i] = b;
            counts[b]!++;
            growBounds(binBounds[b]!, items[i]!.bounds);
        }

        // Sweep both sides to get the area and count of each side for a split after bin i
        const leftArea = new Float64Array(SAH_BINS);
        const rightArea = new Float64Array(SAH_BINS);
        const leftCount = new Uint32Array(SAH_BINS);
        const rightCount = new Uint32Array(SAH_BINS);

        const lb = emptyBounds();
        let lc = 0;
        for (let i = 0; i < SAH_BINS - 1; i++) {
            growBounds(lb, binBounds[i]!);
            lc += counts[i]!;
            leftArea[i] = lc > 0 ? area(lb) : Infinity;
            leftCount[i] = lc;
        }
        const rb = emptyBounds();
        let rc = 0;
        for (let i = SAH_BINS - 1; i > 0; i--) {
            growBounds(rb, binBounds[i]!);
            rc += counts[i]!;
            rightArea[i] = rc > 0 ? area(rb) : Infinity;
            rightCount[i] = rc;
        }

        for (let i = 0; i < SAH_BINS - 1; i++) {
            if (leftCount[i] === 0 || rightCount[i + 1] === 0) continue;
            const cost = leftArea[i]! * leftCount[i]! + rightArea[i + 1]! * rightCount[i + 1]!;
            if (!best || cost < best.cost) best = { bins, splitBin: i, cost };
        }
    }
    return best;
}

/** Counting-sorts by bin and splits in half — O(n), faster than re-sorting, and keeps
 * same-bin primitives on one side. */
function partitionByBin<T>(items: ItemData<T>[], split: SahSplit): [ItemData<T>[], ItemData<T>[]] {
    const start = new Uint32Array(SAH_BINS + 1);
    for (const b of split.bins) start[b + 1]!++;
    for (let i = 0; i < SAH_BINS; i++) start[i + 1]! += start[i]!;

    const sorted = new Array<ItemData<T>>(items.length);
    const cursor = start.slice(0, SAH_BINS);
    for (let i = 0; i < items.length; i++) sorted[cursor[split.bins[i]!]!++] = items[i]!;

    const k = start[split.splitBin + 1]!;
    return [sorted.slice(0, k), sorted.slice(k)];
}

/** Fallback: split at the centroid median of the longest axis. Used when centroids all
 * coincide or the depth limit is reached. */
function medianPartition<T>(items: ItemData<T>[]): [ItemData<T>[], ItemData<T>[]] {
    const b = mergeAllBounds(items);
    const ex = b.max.x - b.min.x;
    const ey = b.max.y - b.min.y;
    const ez = b.max.z - b.min.z;
    const axis: (typeof AXES)[number] = ex >= ey && ex >= ez ? 'x' : ey >= ez ? 'y' : 'z';
    items.sort((a, b) => a.centroid[axis] - b.centroid[axis]);
    const mid = items.length >> 1;
    return [items.slice(0, mid), items.slice(mid)];
}

function emptyBounds(): AABB {
    return {
        min: { x: Infinity, y: Infinity, z: Infinity },
        max: { x: -Infinity, y: -Infinity, z: -Infinity },
    };
}

function growBounds(a: AABB, b: AABB): void {
    if (b.min.x < a.min.x) a.min.x = b.min.x;
    if (b.min.y < a.min.y) a.min.y = b.min.y;
    if (b.min.z < a.min.z) a.min.z = b.min.z;
    if (b.max.x > a.max.x) a.max.x = b.max.x;
    if (b.max.y > a.max.y) a.max.y = b.max.y;
    if (b.max.z > a.max.z) a.max.z = b.max.z;
}

/** Surface area (half value; a constant factor does not affect comparisons). */
function area(b: AABB): number {
    const dx = b.max.x - b.min.x;
    const dy = b.max.y - b.min.y;
    const dz = b.max.z - b.min.z;
    return dx * dy + dy * dz + dz * dx;
}

function createLeaf<T>(items: ItemData<T>[]): BvhNode {
    const node = new BvhNode(mergeAllBounds(items));
    for (let i = 0; i < items.length; i++) {
        const slot = i === 0 ? 'left' : ('right' as const);
        node[slot] = -(items[i].index + 1);
    }
    return node;
}

const NODE_STRIDE = 8;
function flatten(root: BvhNode): Float32Array {
    const nodes: number[] = [];
    function walk(node: BvhNode): number {
        const base = nodes.length;
        for (let i = 0; i < NODE_STRIDE; i++) nodes.push(0);
        if (node.leftChild) nodes[base + 6] = walk(node.leftChild) * NODE_STRIDE + 1;
        else if (node.left < 0) nodes[base + 6] = node.left;
        if (node.rightChild) nodes[base + 7] = walk(node.rightChild) * NODE_STRIDE + 1;
        else if (node.right < 0) nodes[base + 7] = node.right;
        nodes[base + 0] = node.bounds.min.x;
        nodes[base + 1] = node.bounds.min.y;
        nodes[base + 2] = node.bounds.min.z;
        nodes[base + 3] = node.bounds.max.x;
        nodes[base + 4] = node.bounds.max.y;
        nodes[base + 5] = node.bounds.max.z;
        return base / NODE_STRIDE;
    }
    walk(root);
    return new Float32Array(nodes);
}

function mergeAllBounds<T>(items: ItemData<T>[]): AABB {
    let mnX = Infinity,
        mnY = Infinity,
        mnZ = Infinity;
    let mxX = -Infinity,
        mxY = -Infinity,
        mxZ = -Infinity;
    for (const it of items) {
        mnX = Math.min(mnX, it.bounds.min.x);
        mnY = Math.min(mnY, it.bounds.min.y);
        mnZ = Math.min(mnZ, it.bounds.min.z);
        mxX = Math.max(mxX, it.bounds.max.x);
        mxY = Math.max(mxY, it.bounds.max.y);
        mxZ = Math.max(mxZ, it.bounds.max.z);
    }
    return { min: { x: mnX, y: mnY, z: mnZ }, max: { x: mxX, y: mxY, z: mxZ } };
}

function mergeBounds(a: AABB, b: AABB): AABB {
    return {
        min: { x: Math.min(a.min.x, b.min.x), y: Math.min(a.min.y, b.min.y), z: Math.min(a.min.z, b.min.z) },
        max: { x: Math.max(a.max.x, b.max.x), y: Math.max(a.max.y, b.max.y), z: Math.max(a.max.z, b.max.z) },
    };
}

/** Mesh IR → BLAS (flat BVH array). Each Mesh should be built only once. */
const blasCache = new WeakMap<Mesh, Float32Array>();

meshBlas = (mesh) => {
    let b = blasCache.get(mesh);
    if (!b) {
        b = buildMeshBvh(mesh);
        blasCache.set(mesh, b);
    }
    return b;
};

function buildMeshBvh(mesh: Mesh): Float32Array {
    const vd = mesh.vertexData;
    const tri = mesh.triangleData;
    const triCount = tri.length / 4;
    const items = Array.from({ length: triCount }, (_, t) => t);
    return buildBvh(items, (t: number) => {
        const off = t * 4;
        let mnX = Infinity,
            mnY = Infinity,
            mnZ = Infinity;
        let mxX = -Infinity,
            mxY = -Infinity,
            mxZ = -Infinity;
        for (const vi of [tri[off], tri[off + 1], tri[off + 2]]) {
            // vertexData is interleaved at 12 floats/vertex; position is the first 3
            const b = vi * 12;
            mnX = Math.min(mnX, vd[b]);
            mxX = Math.max(mxX, vd[b]);
            mnY = Math.min(mnY, vd[b + 1]);
            mxY = Math.max(mxY, vd[b + 1]);
            mnZ = Math.min(mnZ, vd[b + 2]);
            mxZ = Math.max(mxZ, vd[b + 2]);
        }
        return { min: { x: mnX, y: mnY, z: mnZ }, max: { x: mxX, y: mxY, z: mxZ } };
    });
}
