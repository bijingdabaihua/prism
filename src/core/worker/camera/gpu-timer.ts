/**
 * Measures per-frame GPU time.
 *
 * Prefers timestamp-query: writes 3 timestamps into the command stream (frame start /
 * sampling end / display end) and reads them back asynchronously after resolve.
 * Timestamps come from the GPU's own clock and are not quantized by vsync — wrapping
 * submit() with performance.now() only measures CPU submission, while rAF frame
 * intervals snap to 16.7/33.3 ms steps, and neither can drive the sampling budget.
 *
 * When the device does not support timestamp-query, falls back to
 * queue.onSubmittedWorkDone(): wall-clock time from submission to GPU completion,
 * including a little driver overhead, but enough for the controller to converge.
 *
 * The file has two layers: contract (all exports) and implementation (no exports);
 * see AGENTS.md for the convention.
 */

// ═════════════════ Contract layer (exports) ══════════════════

/**
 * GpuTimer class binding: probes device capabilities at construction and uses the
 * timestamp-query path when timestamps can be written, otherwise falls back to
 * onSubmittedWorkDone.
 *
 * This is only a declaration; the assignment is in the implementation layer.
 */
export let GpuTimer: {
    new (device: GPUDevice): GpuTimer;
};

/** Per-frame GPU time measurer (timestamp-query preferred, onSubmittedWorkDone fallback). */
export interface GpuTimer {
    /** Called when each measurement completes (async, usually 1–2 frames after submission). */
    onMeasure?: (samplingMs: number, totalMs: number, dispatch: number) => void;

    /**
     * Whether another frame can be submitted.
     *
     * Submitting faster than the GPU runs deepens the queue: once all slots are
     * full, measurements are dropped, the controller gets no new data, and the
     * sample count freezes until the queue drains.
     *
     * The limit is 2 rather than 1: with only one frame in flight, submission has
     * to wait for the previous readback and then for the next rAF, leaving the GPU
     * idle for up to half a refresh interval each frame — the sample count drops by
     * 30%. Two frames in flight keep the GPU busy continuously, and with 3 slots
     * there is always one free slot, so measurements never starve.
     */
    canSubmit(): boolean;
    /** Frame start (before sampling). */
    begin(encoder: GPUCommandEncoder): void;
    /** End of path tracing (copy + compute). */
    afterSampling(encoder: GPUCommandEncoder): void;
    /** End of display (blit); also resolves timestamps and queues the readback. */
    end(encoder: GPUCommandEncoder, dispatch: number): void;
    /**
     * Called after submit: starts the readback (timestamps) or waits for GPU
     * completion (fallback).
     */
    afterSubmit(): void;
    destroy(): void;
}

// ════════════ Implementation layer (not exported) ═════════════

/**
 * GPUCommandEncoder.writeTimestamp() is already a standard API, but @webgpu/types
 * 0.1.71 does not declare it yet, so add a local interface here.
 */
interface TimestampCommandEncoder extends GPUCommandEncoder {
    writeTimestamp(querySet: GPUQuerySet, queryIndex: number): void;
}

function writeTimestamp(encoder: GPUCommandEncoder, querySet: GPUQuerySet, index: number): void {
    (encoder as TimestampCommandEncoder).writeTimestamp(querySet, index);
}

/**
 * Whether the device can actually write timestamps.
 *
 * Checking features alone is not enough: some implementations (experimental Firefox
 * WebGPU, etc.) declare the timestamp-query feature but do not implement
 * encoder.writeTimestamp, and the call throws TypeError. Trust whether the method
 * itself exists; if missing, the whole path falls back to onSubmittedWorkDone.
 */
function supportsTimestamps(device: GPUDevice): boolean {
    if (!device.features.has('timestamp-query')) return false;
    const proto = (globalThis as any).GPUCommandEncoder?.prototype;
    return typeof proto?.writeTimestamp === 'function';
}

/** Frames allowed in flight; one readback slot stays free so measurements never starve. */
const MAX_IN_FLIGHT = 2;
/** Readback slot count. One more than MAX_IN_FLIGHT, which the rule above relies on. */
const SLOT_COUNT = 3;

class GpuTimerImpl implements GpuTimer {
    public onMeasure?: (samplingMs: number, totalMs: number, dispatch: number) => void;
    private querySet: GPUQuerySet | null = null;
    private resolveBuf: GPUBuffer | null = null;
    private slots: Array<{ buf: GPUBuffer; busy: boolean }> = [];
    /** Frames submitted but not yet read back. */
    private inFlight = 0;
    private pending: { slot: { buf: GPUBuffer; busy: boolean }; dispatch: number } | null = null;
    /** CPU start time and current-frame sample count for the fallback path. */
    private cpuStart = 0;
    private fallbackDispatch = 0;
    private fallbackBusy = false;
    private readonly device: GPUDevice;

    constructor(device: GPUDevice) {
        this.device = device;

        if (supportsTimestamps(device)) {
            this.querySet = device.createQuerySet({ type: 'timestamp', count: 3 });
            this.resolveBuf = device.createBuffer({
                size: 24,
                usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
            });
            for (let i = 0; i < SLOT_COUNT; i++) {
                this.slots.push({
                    buf: device.createBuffer({
                        size: 24,
                        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
                    }),
                    busy: false,
                });
            }
        }
    }

    public canSubmit(): boolean {
        if (this.slots.length === 0) return !this.fallbackBusy;
        return this.inFlight < MAX_IN_FLIGHT;
    }

    public begin(encoder: GPUCommandEncoder): void {
        if (this.querySet) {
            writeTimestamp(encoder, this.querySet, 0);
        } else {
            this.cpuStart = performance.now();
        }
    }

    public afterSampling(encoder: GPUCommandEncoder): void {
        if (this.querySet) writeTimestamp(encoder, this.querySet, 1);
    }

    public end(encoder: GPUCommandEncoder, dispatch: number): void {
        if (!this.querySet || !this.resolveBuf) {
            this.fallbackDispatch = dispatch;
            return;
        }
        writeTimestamp(encoder, this.querySet, 2);
        encoder.resolveQuerySet(this.querySet, 0, 3, this.resolveBuf, 0);

        const slot = this.slots.find((s) => !s.busy);
        if (!slot) {
            this.pending = null;
            return;
        }
        slot.busy = true;
        this.inFlight++;
        encoder.copyBufferToBuffer(this.resolveBuf, 0, slot.buf, 0, 24);
        this.pending = { slot, dispatch };
    }

    public afterSubmit(): void {
        if (!this.querySet) {
            this.fallbackAfterSubmit();
            return;
        }

        const pending = this.pending;
        this.pending = null;
        if (!pending) return;

        pending.slot.buf
            .mapAsync(GPUMapMode.READ)
            .then(() => {
                const values = new BigUint64Array(pending.slot.buf.getMappedRange());
                const t0 = Number(values[0]);
                const t1 = Number(values[1]);
                const t2 = Number(values[2]);
                pending.slot.buf.unmap();
                pending.slot.busy = false;
                this.inFlight--;
                if (t2 <= t0 || t1 < t0) return;
                const samplingMs = (t1 - t0) / 1e6;
                this.onMeasure?.(samplingMs, (t2 - t0) / 1e6, pending.dispatch);
            })
            .catch(() => {
                pending.slot.busy = false;
                this.inFlight--;
            });
    }

    public destroy(): void {
        this.resolveBuf?.destroy();
        for (const s of this.slots) s.buf.destroy();
        this.slots = [];
        this.querySet?.destroy();
        this.querySet = null;
    }

    private fallbackAfterSubmit(): void {
        if (this.fallbackBusy) return;
        this.fallbackBusy = true;
        const start = this.cpuStart;
        const dispatch = this.fallbackDispatch;
        this.device.queue
            .onSubmittedWorkDone()
            .then(() => {
                this.fallbackBusy = false;
                const ms = performance.now() - start;
                if (dispatch > 0 && ms > 0) {
                    this.onMeasure?.(ms, ms, dispatch);
                }
            })
            .catch(() => {
                this.fallbackBusy = false;
            });
    }
}

/** Assignment of the contract-layer class binding. */
GpuTimer = GpuTimerImpl;
