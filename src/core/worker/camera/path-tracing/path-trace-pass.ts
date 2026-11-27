import type { RenderContext } from '../context';
import type { Model } from '../../model/model';
import type { Background } from '../../background/background';
import type { TextureSet } from '../../model/material/texture-set';
import { buildPixelOrder } from './pixel-order';
import structs from './structs.wgsl?raw';
import aabb from './aabb.wgsl?raw';
import intersect from './intersect.wgsl?raw';
import bvh from './bvh.wgsl?raw';
import shade from './shade.wgsl?raw';
import main from './main.wgsl?raw';

/**
 * Path-tracing compute pass.
 *
 * Each frame samples only a subset of screen pixels: the order table is a random
 * permutation of 0..W*H-1, and the cursor takes a contiguous run from its head
 * each frame, so one sweep covers every pixel exactly once. Dispatch is 1D — one
 * thread maps to one pixel in the table, and the sample count adapts to measured
 * GPU time (see reportFrameTime).
 *
 * The running-average denominator doesn't need per-pixel storage: every time the
 * order table advances past W*H samples, all pixels have been sampled exactly once
 * more, so "how many times this pixel was sampled before" = completed sweeps
 * (+ whether this frame crosses a sweep boundary).
 *
 * The file has two layers: contract (all exports) and implementation (no exports);
 * see AGENTS.md for the convention.
 */

// ═════════════════ Contract layer (exports) ══════════════════

/**
 * Class binding for PathTracePass. Construction builds the environment-map sampler
 * and device-level resources (pipeline / bgl / uniform); size-dependent resources
 * are created lazily on the first resize.
 *
 * Declaration only; the assignment lives in the implementation layer.
 */
export let PathTracePass: {
    new (context: RenderContext, model: Model, cameraUniform: GPUBuffer, background: Background): PathTracePass;
};

/**
 * Path-tracing compute pass: samples a subset of screen pixels each frame and
 * accumulates into ping-pong textures.
 */
export interface PathTracePass {
    /** Accumulation view to display; returns null if there isn't one yet. */
    radianceView(): GPUTextureView | null;
    /**
 * Increments whenever the accumulation textures are rebuilt; DisplayPass uses it
 * to tell whether the blit bind group is stale.
 */
    radianceVersion(): number;
    /** Whether the last frame restarted accumulation (camera / scene / size change). */
    restarted(): boolean;
    /** Number of pixels actually dispatched this frame (read by the budget controller). */
    lastDispatch(): number;
    /** Per-frame GPU budget (ms). RenderLoop sets it from the measured refresh rate. */
    setTargetFrameMs(ms: number): void;
    /**
     * Callback for measured GPU time (asynchronous, usually 1–2 frames behind submission).
     *
     * @param samplingMs GPU time for this frame's copy + compute
     * @param measuredDispatch pixels actually dispatched in the measured frame
     *
     * Uses a **ratio** rather than a cost model: take how many milliseconds this
     * frame spent and scale the sample count by target / measured.
     *
     * The old code computed a target directly as budget / per-pixel cost, assuming
     * time is proportional to sample count. But each frame also has fixed overhead
     * independent of sample count — the full-screen copy of the accumulation
     * textures (under partial sampling, pixels not selected must be carried over
     * from the read texture) and the display blit. At low sample counts this is
     * spread across every pixel, inflating the cost estimate, which in turn lowers
     * the sample count and spreads it further — positive feedback that slides all
     * the way down to MIN_DISPATCH and sticks there. Ratio control accounts for
     * this term naturally: it converges to measured ≈ budget, and the fixed
     * overhead is yielded automatically without needing to know its size.
     *
     * Moves at most 2x per frame; one bad measurement causes only a 2x deviation,
     * and the next frame's measurement corrects it.
     */
    reportFrameTime(samplingMs: number, measuredDispatch: number): void;
    /**
     * Records this frame into the encoder: rebuilds accumulation textures / bind
     * groups as needed, resets or copies the accumulator, writes uniforms, and
     * dispatches the sampling threads. Clears accumulation and restarts when
     * cameraChanged is set.
     */
    record(encoder: GPUCommandEncoder, cameraChanged: boolean): void;
    destroy(): void;
}

// ════════════ Implementation layer (not exported) ═════════════

/**
 * Minimum pixels sampled per frame; never go below this so the image isn't
 * fully black while dragging.
 */
const MIN_DISPATCH = 4096;
/** Workgroup size for the 1D dispatch. */
const WORKGROUP_SIZE = 64;
/** The single-dimension limit of dispatchWorkgroups is 65535; converted to pixels here. */
const MAX_DISPATCH = 65535 * WORKGROUP_SIZE;
/** Frame budget used before measured GPU time arrives (about 75% of 60 Hz). */
const DEFAULT_TARGET_MS = 12.5;
/**
 * Fixed reserve for the display blit and submission gap (ms), subtracted from the
 * per-frame budget up front.
 */
const FIXED_RESERVE_MS = 0.5;

const SHADER = [structs, aabb, intersect, bvh, shade, main].join('\n');

class PathTracePassImpl implements PathTracePass {
    /**
 * Increments whenever the accumulation textures are rebuilt; DisplayPass uses it
 * to tell whether the blit bind group is stale.
 */
    private version = 0;
    /**
     * Whether accumulation restarted this frame (camera/scene/size change).
     *
     * The render loop uses this to switch between "responsive" and "throughput"
     * budgets: latency matters while the image is changing, throughput matters
     * while accumulation is static. It is expressed as "was accumulation reset"
     * rather than "did the camera move" because the loop should not know about the
     * camera — anything that restarts the image should take the same path.
     */
    private restartedThisFrame = false;
    private width = 0;
    private height = 0;
    private pixelCount = 0;
    /** onResize only sets a dirty flag; the actual rebuild is deferred to the next record(). */
    private resizeDirty = false;
    private pipeline?: GPUComputePipeline;
    private bgl?: GPUBindGroupLayout;
    private uniformBuf?: GPUBuffer;
    private accumTextures: [GPUTexture?, GPUTexture?] = [undefined, undefined];
    private accumViews: [GPUTextureView?, GPUTextureView?] = [undefined, undefined];
    private accumIndex = 0;
    private currentWriteView: GPUTextureView | null = null;
    /** Pixel sampling order table (a random permutation of 0..pixelCount-1), rebuilt on resize. */
    private orderBuf?: GPUBuffer;
    /** Ping-pong bind groups, selected by the current write index. */
    private bindGroups: [GPUBindGroup?, GPUBindGroup?] = [undefined, undefined];
    /**
 * Environment-map version when this batch of bind groups was built; rebuild if it
 * differs from Background.version().
 */
    private bindGroupEnvVersion = -1;
    /**
 * Scene-resource version when this batch of bind groups was built; changes when
 * Model appends instances.
 */
    private bindGroupModelVersion = -1;
    /**
 * Write view each cached bind group was created against — a different view means
 * the textures were rebuilt.
 */
    private bindGroupWriteViews: [GPUTextureView?, GPUTextureView?] = [undefined, undefined];
    /** Pixels sampled so far in the current sweep, in [0, pixelCount). */
    private cursor = 0;
    /** Completed full sweeps — a lower bound on each pixel's sample count. */
    private sweep = 0;
    /** Frame index: feeds only the LDS random sequence, not the running average. */
    private frameIndex = 1;
    /** Pixels to dispatch this frame, adapted from measured GPU time. */
    private dispatchTarget = MIN_DISPATCH;
    private lastDispatched = 0;
    /** Per-frame GPU budget (ms), set by RenderLoop from the display refresh rate. */
    private targetFrameMs = DEFAULT_TARGET_MS;
    /** Clears the accumulation buffer next frame and restarts sampling from the beginning. */
    private resetPending = false;
    private lastEmissionScale = 1;
    private readonly context: RenderContext;
    private readonly model: Model;
    private readonly cameraUniform: GPUBuffer;
    private readonly background: Background;
    private readonly textures: TextureSet;
    /**
 * Environment-map sampler: u wraps, v clamps, kept separate from the material
 * textures' repeat/repeat sampler.
 */
    private readonly envSampler: GPUSampler;
    /**
     * Maximum path bounces, written to the shader uniform (u.max_bounces).
     *
     * This is only a safety cap: the shader starts Russian roulette after the 4th
     * bounce, and a normal path decides for itself whether to continue based on
     * throughput. 12 guards against infinite loops in extreme cases.
     */
    private readonly maxBounces = 12;

    constructor(context: RenderContext, model: Model, cameraUniform: GPUBuffer, background: Background) {
        this.context = context;
        this.model = model;
        this.cameraUniform = cameraUniform;
        this.background = background;
        this.textures = model.textures();
        // Environment-map sampling: u wraps, v clamps. v is the polar angle, so
        // repeat would wrap the north pole onto the south. Only the roughness-based
        // mip lookup uses this sampler; NEE still uses texel-exact textureLoad.
        this.envSampler = context.device.createSampler({
            magFilter: 'linear',
            minFilter: 'linear',
            mipmapFilter: 'linear',
            addressModeU: 'repeat',
            addressModeV: 'clamp-to-edge',
        });
        // onResize only sets a dirty flag; the rebuild is deferred to record()
        // (no GPU work in the callback)
        context.onResize(() => {
            this.resizeDirty = true;
        });
        // Device-level resources (pipeline/bgl/uniform) are created once at
        // construction and don't change for the device's lifetime
        this.rebuildDeviceResources();
    }

    public radianceView(): GPUTextureView | null {
        return this.currentWriteView ?? this.outView() ?? null;
    }

    public radianceVersion(): number {
        return this.version;
    }

    public restarted(): boolean {
        return this.restartedThisFrame;
    }

    public lastDispatch(): number {
        return this.lastDispatched;
    }

    public setTargetFrameMs(ms: number): void {
        if (ms > 0) this.targetFrameMs = ms;
    }

    public reportFrameTime(samplingMs: number, measuredDispatch: number): void {
        if (!(samplingMs > 0.02) || measuredDispatch <= 0 || this.pixelCount <= 0) return;

        const budgetMs = this.targetFrameMs - FIXED_RESERVE_MS;
        if (budgetMs <= 1) return;

        const ratio = Math.min(Math.max(budgetMs / samplingMs, 0.5), 2);
        const next = Math.min(Math.max(measuredDispatch * ratio, MIN_DISPATCH), this.pixelCount, MAX_DISPATCH);
        this.dispatchTarget = Math.max(1, Math.round(next));
    }

    public record(encoder: GPUCommandEncoder, cameraChanged: boolean): void {
        const d = this.context.device;
        const bg = this.background;

        this.restartedThisFrame = false;

        // Resize dirty flag → rebuild accumulation textures (size-dependent
        // textures are created lazily; the first frame triggers it via onResize
        // setting the flag)
        if (this.resizeDirty) {
            this.resizeDirty = false;
            this.width = this.context.width();
            this.height = this.context.height();
            this.rebuildAccumTextures();
            this.restartedThisFrame = true;
        }
        if (!this.width || !this.height) return;

        // The scene is assembled externally in the order camera placement →
        // environment map → geometry; if any is missing we can't render: no light
        // source, or the Model, materials and textures haven't compiled, so bind
        // groups can't be built.
        if (!bg.hasEnvironment() || !this.model.ready()) return;

        // Camera moved → clear accumulation, reset the cursor (the sampling budget
        // is kept since cost hasn't changed)
        const emissionScale = this.model.emissionScale();
        if (cameraChanged || emissionScale !== this.lastEmissionScale) this.requestReset();
        this.lastEmissionScale = emissionScale;

        // Bind groups are rebuilt in four cases: none exist yet; resize replaced the
        // accumulation textures with new views (the two ping-pong groups capture
        // the views at build time, so a mismatch on either means the whole batch is
        // stale); the environment map changed (the old texture was destroyed and
        // bindings must follow); or the scene appended instances (geometry /
        // material / instance buffers were all swapped for a new set).
        const envChanged = this.bindGroupEnvVersion !== bg.version();
        const modelChanged = this.bindGroupModelVersion !== this.model.version();
        if (!this.bindGroups[0] || this.bindGroupWriteViews[0] !== this.accumViews[0] || envChanged || modelChanged) {
            // Lighting or scene changed, so the old accumulation is meaningless; on
            // the first bind-group build there are no samples yet
            if ((envChanged || modelChanged) && this.bindGroups[0]) this.requestReset();
            this.rebuildBindGroups();
        }

        const maxPixels = Math.max(1, Math.min(this.pixelCount, MAX_DISPATCH));
        const n = Math.max(1, Math.min(this.dispatchTarget, maxPixels));
        const writeIndex = this.accumIndex;
        const readIndex = 1 - writeIndex;

        if (this.resetPending) {
            // Camera/content change: clear both to black, reset the cursor, and
            // restart from the head of the order table.
            this.resetPending = false;
            this.restartedThisFrame = true;
            this.cursor = 0;
            this.sweep = 0;
            const pass = encoder.beginRenderPass({
                colorAttachments: [
                    {
                        view: this.accumViews[0]!,
                        loadOp: 'clear',
                        storeOp: 'store',
                        clearValue: { r: 0, g: 0, b: 0, a: 0 },
                    },
                    {
                        view: this.accumViews[1]!,
                        loadOp: 'clear',
                        storeOp: 'store',
                        clearValue: { r: 0, g: 0, b: 0, a: 0 },
                    },
                ],
            });
            pass.end();
        } else if (n < this.pixelCount) {
            // Only n pixels are written this frame; the rest must be copied over from
            // the read texture, otherwise they'd stay at their value from two frames
            // ago in the destination (holes/flicker in the image). Once the budget
            // grows to a full screen every pixel is rewritten, so this whole copy is
            // skipped.
            encoder.copyTextureToTexture(
                { texture: this.accumTextures[readIndex]!, origin: { x: 0, y: 0 } },
                { texture: this.accumTextures[writeIndex]!, origin: { x: 0, y: 0 } },
                { width: this.width, height: this.height },
            );
        }

        // Rewrite uniforms every frame (cursor, sweep and budget change; caching would cost more)
        const buf = new ArrayBuffer(48);
        const uni = new Uint32Array(buf);
        uni[0] = this.width;
        uni[1] = this.height;
        uni[2] = this.frameIndex;
        uni[3] = this.maxBounces;
        uni[4] = n;
        uni[5] = this.cursor;
        uni[6] = this.sweep;
        uni[7] = this.pixelCount;
        new Float32Array(buf)[8] = emissionScale;
        d.queue.writeBuffer(this.uniformBuf!, 0, buf);

        const pass = encoder.beginComputePass();
        pass.setPipeline(this.pipeline!);
        pass.setBindGroup(0, this.bindGroups[writeIndex]!);
        pass.dispatchWorkgroups(Math.ceil(n / WORKGROUP_SIZE));
        pass.end();

        this.currentWriteView = this.accumViews[writeIndex]!;
        this.accumIndex = readIndex;
        this.lastDispatched = n;

        // Advance the cursor. n ≤ pixel_count, so it crosses a sweep boundary at most once.
        this.cursor += n;
        if (this.cursor >= this.pixelCount) {
            this.cursor -= this.pixelCount;
            this.sweep++;
        }
        this.frameIndex++;
    }

    public destroy(): void {
        this.uniformBuf?.destroy();
        this.orderBuf?.destroy();
        for (const t of this.accumTextures) t?.destroy();
        this.accumTextures = [undefined, undefined];
        this.accumViews = [undefined, undefined];
        this.bindGroups = [undefined, undefined];
        this.bindGroupWriteViews = [undefined, undefined];
    }

    private outView(): GPUTextureView | undefined {
        return this.accumViews[this.accumIndex];
    }

    /**
     * Clears the accumulation buffer and restarts from the beginning (cursor reset).
     * Must be called when the camera/scene/background/render parameters change,
     * otherwise old and new images blend together.
     */
    private requestReset(): void {
        this.resetPending = true;
        // frameIndex deliberately does not reset: it is the LDS sample index, feeding
        // only random numbers and not the running average. During dragging every frame
        // resets; if the index went back to 1, all pixels would keep using the same
        // random set — a single-sample image would show a fixed screen-space pattern
        // instead of noise.
    }

    /** Device-level resources: only need rebuilding when the GPUDevice changes. */
    private rebuildDeviceResources(): void {
        const d = this.context.device;

        this.uniformBuf?.destroy();
        this.uniformBuf = d.createBuffer({
            size: 48, // Uniforms: 8 × u32, 1 × f32 and alignment padding; see structs.wgsl
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Storage buffers: 2=nodes, 3=pixel order, 4..7=geometry/triangles/
        // materials/instances, 16=environment CDF.
        this.bgl = d.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 20, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                {
                    binding: 8,
                    visibility: GPUShaderStage.COMPUTE,
                    storageTexture: { format: 'rgba32float', access: 'write-only' },
                },
                {
                    binding: 9,
                    visibility: GPUShaderStage.COMPUTE,
                    texture: { sampleType: 'float', viewDimension: '2d' },
                },
                {
                    binding: 10,
                    visibility: GPUShaderStage.COMPUTE,
                    texture: { sampleType: 'float', viewDimension: '2d-array' },
                },
                {
                    binding: 11,
                    visibility: GPUShaderStage.COMPUTE,
                    texture: { sampleType: 'float', viewDimension: '2d-array' },
                },
                {
                    binding: 12,
                    visibility: GPUShaderStage.COMPUTE,
                    texture: { sampleType: 'float', viewDimension: '2d-array' },
                },
                { binding: 17, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
                {
                    binding: 13,
                    visibility: GPUShaderStage.COMPUTE,
                    texture: { sampleType: 'unfilterable-float', viewDimension: '2d' },
                },
                { binding: 15, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 16, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                {
                    binding: 19,
                    visibility: GPUShaderStage.COMPUTE,
                    texture: { sampleType: 'float', viewDimension: '2d-array' },
                },
                { binding: 21, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
            ],
        });

        const module = d.createShaderModule({ code: SHADER });
        const layout = d.createPipelineLayout({ bindGroupLayouts: [this.bgl] });
        this.pipeline = d.createComputePipeline({ layout, compute: { module, entryPoint: 'main' } });
    }

    /** Size-dependent resources: only need rebuilding when the render resolution changes. */
    private rebuildAccumTextures(): void {
        const d = this.context.device;
        if (!this.width || !this.height) return;

        for (const t of this.accumTextures) t?.destroy();
        for (let i = 0; i < 2; i++) {
            // The accumulator must be fp32. fp16 has an ulp of about 2.4e-4 near 0.3,
            // while the Nth sample's contribution to the mean is (radiance - mean)/N —
            // past a few thousand samples it falls within one ulp and the update is
            // swallowed by rounding. This is especially fatal for heavy-tailed
            // distributions (fireflies): the mean sticks near the early samples and
            // stops converging, and the image keeps darkening as samples accumulate.
            //
            // COPY_*: copy the read texture to the write texture each frame (partial
            // sampling only writes the selected pixels).
            // RENDER_ATTACHMENT: clear both to black on reset.
            this.accumTextures[i] = d.createTexture({
                size: { width: this.width, height: this.height },
                format: 'rgba32float',
                usage:
                    GPUTextureUsage.STORAGE_BINDING |
                    GPUTextureUsage.TEXTURE_BINDING |
                    GPUTextureUsage.COPY_SRC |
                    GPUTextureUsage.COPY_DST |
                    GPUTextureUsage.RENDER_ATTACHMENT,
                label: `pt-accum-${i}`,
            });
            this.accumViews[i] = this.accumTextures[i]!.createView();
        }

        // Pixel sampling order table: each pixel appears exactly once per sweep.
        this.pixelCount = this.width * this.height;
        this.orderBuf?.destroy();
        const order = buildPixelOrder(this.pixelCount);
        this.orderBuf = d.createBuffer({
            size: order.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            label: 'pt-pixel-order',
        });
        d.queue.writeBuffer(this.orderBuf, 0, order as any);

        this.accumIndex = 0;
        this.cursor = 0;
        this.sweep = 0;
        this.resetPending = false;
        this.dispatchTarget = Math.min(Math.max(this.dispatchTarget, MIN_DISPATCH), this.pixelCount, MAX_DISPATCH);
        this.version++;
    }

/**
 * Builds both ping-pong bind groups together so slots 0/1 reference the same batch
 * of scene resources.
 */
    private rebuildBindGroups(): void {
        this.bindGroups[0] = this.createBindGroup(this.accumViews[0]!, this.accumViews[1]!);
        this.bindGroups[1] = this.createBindGroup(this.accumViews[1]!, this.accumViews[0]!);
        this.bindGroupWriteViews[0] = this.accumViews[0];
        this.bindGroupWriteViews[1] = this.accumViews[1];
        this.bindGroupEnvVersion = this.background.version();
        this.bindGroupModelVersion = this.model.version();
    }

/**
 * On each rebuild only the two ping-pong views change; all scene resources come
 * from held references.
 */
    private createBindGroup(writeView: GPUTextureView, readView: GPUTextureView): GPUBindGroup {
        const model = this.model;
        const bg = this.background;
        return this.context.device.createBindGroup({
            layout: this.bgl!,
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuf! } },
                { binding: 2, resource: { buffer: model.nodes } },
                { binding: 3, resource: { buffer: this.orderBuf! } },
                { binding: 4, resource: { buffer: model.geometry } },
                { binding: 5, resource: { buffer: model.triangles } },
                { binding: 6, resource: { buffer: model.materialBuffer() } },
                { binding: 20, resource: { buffer: model.emitters } },
                { binding: 7, resource: { buffer: model.instanceBuffer } },
                { binding: 8, resource: writeView },
                { binding: 9, resource: bg.envMapView() },
                { binding: 10, resource: this.textures.albedoView() },
                { binding: 11, resource: this.textures.normalView() },
                { binding: 12, resource: this.textures.roughnessView() },
                { binding: 17, resource: this.textures.sampler() },
                { binding: 13, resource: readView },
                { binding: 15, resource: { buffer: this.cameraUniform } },
                { binding: 16, resource: { buffer: bg.envDistributionBuffer() } },
                { binding: 19, resource: this.textures.occlusionView() },
                { binding: 21, resource: this.envSampler },
            ],
        });
    }
}

/** Assignment of the contract-layer class binding. */
PathTracePass = PathTracePassImpl;
