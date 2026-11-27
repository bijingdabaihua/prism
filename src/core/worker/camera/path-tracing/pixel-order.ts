/**
 * Pixel sampling order: shuffles [0, count) into a random permutation.
 *
 * Sampling follows this table in order — each frame takes a contiguous run, and one
 * round covers every pixel exactly once. This gives sampling points a spatially
 * random distribution while ensuring no pixel is missed or sampled twice (until the
 * next round), so the denominator of the running average can be derived from the
 * round count + position in the table.
 *
 * The random source is Fisher–Yates driven by xorshift32: no cryptographic quality
 * needed, only a uniform distribution and no spatial clustering of neighboring
 * samples.
 *
 * The file has two layers: contract (all exports) and implementation (no exports);
 * see AGENTS.md for the convention.
 */

// ═════════════════ Contract layer (exports) ══════════════════

/**
 * Generates the pixel sampling order: a deterministic random permutation of
 * [0, count) (xorshift32 + Fisher–Yates).
 */
export let buildPixelOrder: (count: number) => Uint32Array;

// ════════════ Implementation layer (not exported) ═════════════

buildPixelOrder = (count) => {
    const order = new Uint32Array(count);
    for (let i = 0; i < count; i++) order[i] = i;

    let s = 0x9e3779b9 >>> 0;
    for (let i = count - 1; i > 0; i--) {
        s ^= s << 13;
        s >>>= 0;
        s ^= s >>> 17;
        s ^= s << 5;
        s >>>= 0;
        const j = s % (i + 1);
        const t = order[i];
        order[i] = order[j];
        order[j] = t;
    }
    return order;
};
