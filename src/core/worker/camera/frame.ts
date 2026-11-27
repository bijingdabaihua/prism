import type { Camera } from './camera';
import type { Model } from '../model/model';
import type { Background } from '../background/background';
import type { RenderContext } from './context';
import { GpuTimer } from './gpu-timer';
import { PathTracePass } from './path-tracing/path-trace-pass';
import { DisplayPass } from './display/display-pass';

/**
 * The submission unit for one frame.
 *
 * The order is fixed: path tracing first (writes the accumulation texture), then
 * display (blit to the canvas). Sampling budget and cursor advancement stay inside
 * PathTracePass; this file only chains the stages together and records GPU
 * timestamps at both ends.
 *
 * The file has two layers: contract (all exports) and implementation (no exports);
 * see AGENTS.md for the convention.
 */

// ═════════════════ Contract layer (exports) ══════════════════

/**
 * Frame class binding: construction assembles the path-tracing and display passes
 * and the GPU timer.
 *
 * This is only a declaration; the assignment is in the implementation layer.
 */
export let Frame: {
    new (context: RenderContext, model: Model, camera: Camera, background: Background): Frame;
};

/**
 * Submission unit for one frame: chains path tracing and display with GPU
 * timestamps at both ends.
 */
export interface Frame {
    /** Number of frames submitted. */
    frameCount(): number;
    /**
     * Copies the most recently presented image on the canvas into a bitmap.
     *
     * Only supported by OffscreenCanvas (in the Worker); returns null when no frame
     * has been submitted yet, and the caller waits for the next one.
     */
    capture(): ImageBitmap | null;
    /**
     * GPU budget per frame (milliseconds), set by RenderLoop from the measured
     * display refresh rate.
     */
    setTargetFrameMs(ms: number): void;
    /**
     * Whether the previous frame restarted accumulation — the render loop switches
     * between the "responsive" and "throughput" budget tiers accordingly.
     */
    restarted(): boolean;
    /**
     * Whether another frame can be submitted. Skip while in-flight frames are at the
     * limit: a deeper queue starves the measurements and freezes the sample count
     * (see GpuTimer.canSubmit()).
     */
    readyForNextFrame(): boolean;
    /**
     * Submits one frame. Applies pending canvas size changes first (deferred resize,
     * before getCurrentTexture); skips the whole frame when the size is 0 (canvas not
     * laid out yet). Returns whether rendering was actually submitted.
     */
    renderFrame(): boolean;
    destroy(): void;
}

// ════════════ Implementation layer (not exported) ═════════════

class FrameImpl implements Frame {
    /**
     * Submitted frame count — the camera uses it to tell whether the canvas has a
     * new image to push.
     */
    private submittedFrames = 0;
    private readonly context: RenderContext;
    private readonly camera: Camera;
    private readonly pathTracePass: PathTracePass;
    private readonly displayPass: DisplayPass;
    private readonly timer: GpuTimer;

    constructor(context: RenderContext, model: Model, camera: Camera, background: Background) {
        this.context = context;
        this.camera = camera;
        this.pathTracePass = new PathTracePass(context, model, camera.buffer, background);
        this.displayPass = new DisplayPass(context);
        this.timer = new GpuTimer(context.device);
        this.timer.onMeasure = (samplingMs, _totalMs, dispatch) =>
            this.pathTracePass.reportFrameTime(samplingMs, dispatch);
    }

    public frameCount(): number {
        return this.submittedFrames;
    }

    public capture(): ImageBitmap | null {
        if (this.submittedFrames === 0) return null;
        try {
            return this.context.canvas().transferToImageBitmap();
        } catch {
            // The canvas has not presented any frame yet
            return null;
        }
    }

    public setTargetFrameMs(ms: number): void {
        this.pathTracePass.setTargetFrameMs(ms);
    }

    public restarted(): boolean {
        return this.pathTracePass.restarted();
    }

    public readyForNextFrame(): boolean {
        return this.timer.canSubmit();
    }

    public renderFrame(): boolean {
        if (this.context.width() === 0 || this.context.height() === 0) return false;

        const cameraChanged = this.camera.upload();

        const encoder = this.context.device.createCommandEncoder();
        this.timer.begin(encoder);
        this.pathTracePass.record(encoder, cameraChanged);
        this.timer.afterSampling(encoder);

        this.displayPass.record(encoder, this.pathTracePass.radianceView(), this.pathTracePass.radianceVersion());
        this.timer.end(encoder, this.pathTracePass.lastDispatch());

        this.context.device.queue.submit([encoder.finish()]);
        this.timer.afterSubmit();
        this.submittedFrames++;

        return true;
    }

    public destroy(): void {
        this.pathTracePass.destroy();
        this.displayPass.destroy();
        this.timer.destroy();
    }
}

/** Assignment of the contract-layer class binding. */
Frame = FrameImpl;
