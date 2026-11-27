/**
 * Frame loop.
 *
 * Only decides "when to render the next frame" and "the GPU budget per frame"; it
 * knows nothing about scenes, cameras or materials.
 *
 * The file has two layers: contract (all exports) and implementation (no exports);
 * see AGENTS.md for the convention.
 */

// ═════════════════ Contract layer (exports) ══════════════════

/**
 * RenderLoop class binding: construction starts it — idle a few frames first to
 * measure the refresh rate, then enter the rAF loop.
 *
 * This is only a declaration; the assignment is in the implementation layer.
 */
export let RenderLoop: {
    new (frame: FrameSource): RenderLoop;
};

/**
 * Frame loop.
 *
 * Only decides "when to render the next frame" and "the GPU budget per frame"; it
 * knows nothing about scenes, cameras or materials.
 */
export interface RenderLoop {
    stop(): void;
}

/**
 * A "frame source" as the frame loop sees it.
 *
 * The camera keeps its renderer private and exposes only this scheduling interface —
 * the loop is unaware of Frame or the camera.
 */
export interface FrameSource {
    readyForNextFrame(): boolean;
    restarted(): boolean;
    renderFrame(): boolean;
    setTargetFrameMs(ms: number): void;
}

// ════════════ Implementation layer (not exported) ═════════════

/**
 * Target budget as a fraction of the measured refresh interval — two tiers.
 *
 * Use 0.75 while the image is being changed (dragging the camera, resetting
 * accumulation): leave headroom for the compositor and favor frame rate, or the lag
 * after releasing will be visible.
 *
 * Use 1.9 while accumulating at rest: each frame spans two refresh intervals. The
 * image is only adding samples to the same picture, so a half-frame delay still
 * shows the same picture, and doubling the budget doubles the sample count in GPU
 * time that would otherwise idle — after sitting still for tens of seconds the image
 * differs by multiples, not a few percent.
 */
const RESPONSIVE_RATIO = 0.75;
const ACCUMULATING_RATIO = 1.9;
/** Number of frames sampled for refresh rate (median, to avoid startup jitter). */
const REFRESH_SAMPLES = 12;
/**
 * Maximum wait for the refresh measurement: a background tab may get no rAF at
 * all, and startup must not hang.
 */
const REFRESH_TIMEOUT_MS = 600;

/**
 * Background keep-alive.
 *
 * When the tab goes to the background (or the window is fully covered), rAF stops
 * and the image freezes on the last frame. Fall back to a timer feeding frames:
 * timers in hidden pages are first clamped to once per second, then to once per
 * minute after 5 minutes, so each frame's budget must be granted in full — one
 * small budget per second is as good as nothing.
 *
 * 900 ms rather than 1000: leave a tenth for the browser and compositor.
 */
const BACKGROUND_BUDGET_MS = 900;
/** How long without a submitted frame before rAF is considered stopped. */
const STALL_MS = 700;
/** Poll interval while rAF is alive: just a look, no work. */
const KEEPALIVE_POLL_MS = 250;
/** Retry interval after taking over from rAF: keep the GPU busy continuously. */
const KEEPALIVE_RETRY_MS = 16;

class RenderLoopImpl implements RenderLoop {
    private running = false;
    private animFrameId: number | null = null;
    /** boot/stop generation number: prevents an old rAF chain from running after stop. */
    private generation = 0;
    /** Measured refresh interval (milliseconds); the per-frame budget is this times a ratio. */
    private refreshMs = 1000 / 60;
    /** Background keep-alive switch: when on, a timer takes over even if rAF stops. */
    private keepRunning = false;
    private keepAliveId: number | null = null;
    /** Time of the last submitted frame, used to tell whether rAF has stopped. */
    private lastSubmit = 0;
    private readonly frame: FrameSource;

    constructor(frame: FrameSource) {
        this.frame = frame;
        this.boot();
    }

    public stop(): void {
        this.running = false;
        this.generation++;
        if (this.animFrameId !== null) {
            cancelAnimationFrame(this.animFrameId);
            this.animFrameId = null;
        }
        this.setKeepRunning(false);
    }

    private boot(): void {
        if (this.running) return;
        this.running = true;
        const generation = ++this.generation;

        // Idle a few frames first to measure the refresh rate. No GPU work is
        // submitted here, so the rAF interval is the display's presentation
        // cadence; this is more reliable than the "render frame interval" (which
        // GPU load can stretch to 2×/3×).
        this.measureRefresh(generation, (refreshMs) => {
            if (!this.running || generation !== this.generation) return;
            this.refreshMs = refreshMs;
            this.frame.setTargetFrameMs(refreshMs * RESPONSIVE_RATIO);
            this.tick(generation);
        });
    }

    /**
     * Background keep-alive switch: when the tab is hidden and rAF stops, keep
     * feeding the GPU with a timer.
     *
     * Only the driver changes, not the rendered content — sampling budget and
     * accumulation cursor are managed by PathTracePass; in the background the
     * per-frame budget is simply raised to nearly a second.
     */
    private setKeepRunning(on: boolean): void {
        if (on === this.keepRunning) return;
        this.keepRunning = on;
        if (this.keepAliveId !== null) {
            clearTimeout(this.keepAliveId);
            this.keepAliveId = null;
        }
        if (!on) return;
        this.lastSubmit = performance.now();
        this.keepAlive();
    }

    private measureRefresh(generation: number, done: (ms: number) => void): void {
        const deltas: number[] = [];
        let last = 0;
        let finished = false;
        const finish = (ms: number) => {
            if (finished) return;
            finished = true;
            clearTimeout(timeout);
            done(ms);
        };
        // A background tab gets no rAF at all, so waiting for 12 frames means
        // waiting forever. Start at the default 60 Hz if it times out; the real
        // interval takes over naturally once the tab returns to the foreground.
        const timeout = setTimeout(() => finish(this.refreshMs), REFRESH_TIMEOUT_MS);
        const step = (now: number) => {
            if (!this.running || generation !== this.generation) return;
            if (last !== 0) deltas.push(now - last);
            last = now;
            if (deltas.length < REFRESH_SAMPLES) {
                requestAnimationFrame(step);
                return;
            }
            deltas.sort((a, b) => a - b);
            const median = deltas[deltas.length >> 1]!;
            // Clamp to [30Hz, 240Hz]: background throttling or a bad sample must
            // not skew the budget
            finish(Math.min(1000 / 30, Math.max(1000 / 240, median)));
        };
        requestAnimationFrame(step);
    }

    /**
     * The background keep-alive timer chain.
     *
     * While rAF is alive (window visible), only poll every 250 ms and do nothing;
     * once no frame has been submitted for longer than STALL_MS, assume rAF was
     * stopped, submit frames itself and retry every 16 ms. After returning to the
     * foreground, rAF resumes submitting and this chain automatically returns to
     * polling only.
     */
    private keepAlive(): void {
        this.keepAliveId = null;
        if (!this.keepRunning || !this.running) return;

        const stalled = performance.now() - this.lastSubmit > STALL_MS;
        if (!stalled) {
            this.keepAliveId = setTimeout(() => this.keepAlive(), KEEPALIVE_POLL_MS);
            return;
        }

        // Timers in hidden pages fire at most once a second, so the budget must
        // be granted in full.
        this.frame.setTargetFrameMs(BACKGROUND_BUDGET_MS);
        if (this.frame.readyForNextFrame() && this.frame.renderFrame()) {
            this.lastSubmit = performance.now();
        }
        this.keepAliveId = setTimeout(() => this.keepAlive(), KEEPALIVE_RETRY_MS);
    }

    private tick(generation: number): void {
        if (!this.running || generation !== this.generation) return;

        // Skip this frame if the previous frame's GPU work has not finished: the
        // steady-state budget lets one frame span two refresh intervals, so
        // submitting unconditionally would outrun the GPU, deepen the queue, and
        // freeze the sample count once readback slots fill up (seen as being stuck
        // at MIN_DISPATCH). Skipping an rAF is no loss — this frame could not have
        // been drawn in time anyway.
        if (this.frame.readyForNextFrame() && this.frame.renderFrame()) {
            // Budget for the next frame: use the responsive tier if this frame
            // restarted accumulation (the image is being changed), otherwise use
            // the throughput tier since the image is only accumulating. Setting
            // the budget one frame late is exactly right — this frame's sample
            // count was fixed before submission, and this changes the next frame.
            // It also pulls the budget back after background keep-alive raised it
            // to 0.9 s (see keepAlive).
            this.frame.setTargetFrameMs(
                this.refreshMs * (this.frame.restarted() ? RESPONSIVE_RATIO : ACCUMULATING_RATIO),
            );

            this.lastSubmit = performance.now();
        }

        this.animFrameId = requestAnimationFrame(() => this.tick(generation));
    }
}

/** Assignment of the contract-layer class binding. */
RenderLoop = RenderLoopImpl;
