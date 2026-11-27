import type { RenderContext } from '../context';
import VS from './fullscreen-vertex.wgsl?raw';
import FS from './fullscreen-fragment.wgsl?raw';

/**
 * Display pass: blits the accumulation texture to the canvas — ACES, vignette, grain.
 *
 * The input is an fp32 accumulator and is unfilterable — use textureLoad to fetch
 * each pixel instead of sampling; the display size matches the render size one to
 * one, so interpolation is not needed anyway.
 *
 * The file has two layers: contract (all exports) and implementation (no exports);
 * see AGENTS.md for the convention.
 */

// ═════════════════ Contract layer (exports) ══════════════════

/**
 * DisplayPass class binding: construction compiles the blit pipeline.
 *
 * This is only a declaration; the assignment is in the implementation layer.
 */
export let DisplayPass: {
    new (context: RenderContext): DisplayPass;
};

/** Display pass: blits the accumulation texture to the canvas — ACES, vignette, grain. */
export interface DisplayPass {
    /**
     * @param view    Texture to display (the accumulation result)
     * @param version Incremented when the upstream texture is rebuilt; a change
     *                drops all cached bind groups
     */
    record(encoder: GPUCommandEncoder, view: GPUTextureView | null, version: number): void;
    destroy(): void;
}

// ════════════ Implementation layer (not exported) ═════════════

class DisplayPassImpl implements DisplayPass {
    private readonly context: RenderContext;
    private pipeline: GPURenderPipeline | null = null;
    private blitVersion = -1;
    /**
     * Blit bind groups keyed by radiance view — ping-pong switches between the two
     * views every frame, so each version has exactly 2 cache entries.
     */
    private blitBindGroups = new Map<GPUTextureView, GPUBindGroup>();
    private readonly bgl: GPUBindGroupLayout;

    constructor(context: RenderContext) {
        this.context = context;
        const d = context.device;
        this.bgl = d.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.FRAGMENT,
                    texture: { sampleType: 'unfilterable-float', viewDimension: '2d' },
                },
            ],
        });
        this.pipeline = d.createRenderPipeline({
            layout: d.createPipelineLayout({ bindGroupLayouts: [this.bgl] }),
            vertex: { module: d.createShaderModule({ code: VS }), entryPoint: 'main' },
            fragment: {
                module: d.createShaderModule({ code: FS }),
                entryPoint: 'main',
                targets: [{ format: context.format }],
            },
            primitive: { topology: 'triangle-list' },
        });
    }

    public record(encoder: GPUCommandEncoder, view: GPUTextureView | null, version: number): void {
        const ctx = this.context.gpuContext();
        if (!ctx) return;
        const bindGroup = this.bindGroupFor(view, version);
        if (!bindGroup) return;
        this.draw(encoder, bindGroup, ctx.getCurrentTexture().createView());
    }

    public destroy(): void {
        this.pipeline = null;
        this.blitBindGroups.clear();
    }

    private bindGroupFor(view: GPUTextureView | null, version: number): GPUBindGroup | null {
        if (version !== this.blitVersion) {
            this.blitBindGroups.clear();
            this.blitVersion = version;
        }

        // Cache keyed by view: hits directly while upstream alternates between two
        // views each frame
        if (!view || !this.pipeline) return null;

        const cached = this.blitBindGroups.get(view);
        if (cached) return cached;

        const bindGroup = this.createBlitBindGroup(view);
        this.blitBindGroups.set(view, bindGroup);
        return bindGroup;
    }

    private draw(encoder: GPUCommandEncoder, bindGroup: GPUBindGroup, target: GPUTextureView): void {
        if (!this.pipeline) return;
        const rp = encoder.beginRenderPass({
            colorAttachments: [{ view: target, loadOp: 'clear', storeOp: 'store' }],
        });
        rp.setPipeline(this.pipeline);
        rp.setBindGroup(0, bindGroup);
        rp.draw(3);
        rp.end();
    }

    private createBlitBindGroup(radianceView: GPUTextureView): GPUBindGroup {
        return this.context.device.createBindGroup({
            layout: this.bgl,
            entries: [{ binding: 0, resource: radianceView }],
        });
    }
}

/** Assignment of the contract-layer class binding. */
DisplayPass = DisplayPassImpl;
