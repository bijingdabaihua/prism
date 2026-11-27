/**
 * GPU infrastructure: requests the device (createDevice) and binds GPUDevice and
 * OffscreenCanvas into one unit. RenderContext is the single owner of the render
 * target size.
 *
 * OffscreenCanvas is the only canvas host — the whole render pipeline lives in the
 * Worker. An offscreen canvas cannot read CSS layout (no clientWidth /
 * devicePixelRatio), so the size can only be sent in by the main thread through
 * camera.resize.
 *
 * Canvas width and height are shared infrastructure, not owned by any single pass;
 * size changes are delivered to components (Camera / PathTracePass, etc.) through
 * the onResize subscription, and each responds on its own. The only source of size
 * is setSize() — the resize command received by the Worker — applied and announced
 * immediately, with no per-frame polling.
 *
 * The file has two layers: contract (all exports) and implementation (no exports);
 * see AGENTS.md for the convention.
 */

// ═════════════════ Contract layer (exports) ══════════════════

/**
 * RenderContext class binding: binds device and canvas into one render target.
 *
 * This is only a declaration; the assignment is in the implementation layer.
 */
export let RenderContext: {
    new (device: GPUDevice, canvas: OffscreenCanvas): RenderContext;
};

/**
 * Render target: GPUDevice + canvas as a unit, and the single owner of the render
 * target size.
 *
 * Canvas width and height are shared infrastructure, not owned by any single pass;
 * size changes are delivered to components (Camera / PathTracePass, etc.) through
 * the onResize subscription, and each responds on its own.
 */
export interface RenderContext {
    readonly device: GPUDevice;
    /** Platform-preferred canvas format; device-independent and known at construction. */
    readonly format: GPUTextureFormat;

    gpuContext(): GPUCanvasContext | null;
    canvas(): OffscreenCanvas;
    /** Size comes from outside (the resize command): apply it and notify subscribers. */
    setSize(width: number, height: number): void;
    width(): number;
    height(): number;
    /** Registers a size-change callback. Callbacks fire synchronously inside setSize(). */
    onResize(fn: (width: number, height: number) => void): void;
}

/**
 * WebGPUUnavailableError class binding: thrown when no WebGPU device / adapter is
 * available.
 *
 * Callers use this to route to the fallback page instead of treating it as an
 * ordinary error.
 */
export let WebGPUUnavailableError: {
    new (message: string): WebGPUUnavailableError;
};

/**
 * No WebGPU device — callers use this to route to the fallback page instead of
 * treating it as an ordinary error.
 */
export interface WebGPUUnavailableError extends Error {}

/**
 * Requests an adapter + device. The main thread and the Worker share one path —
 * the device requirements are written once.
 *
 * A browser without WebGPU, or one that cannot get an adapter (driver blocklisted,
 * virtual machine, hardware acceleration disabled), throws WebGPUUnavailableError;
 * failures from requestDevice itself propagate unchanged.
 */
export let createDevice: () => Promise<GPUDevice>;

// ════════════ Implementation layer (not exported) ═════════════

class RenderContextImpl implements RenderContext {
    public readonly device: GPUDevice;
    public readonly format: GPUTextureFormat = navigator.gpu.getPreferredCanvasFormat();
    private ctx: GPUCanvasContext | null = null;
    private _width = 0;
    private _height = 0;
    private resizeListeners: Array<(width: number, height: number) => void> = [];
    private readonly canvasEl: OffscreenCanvas;

    constructor(device: GPUDevice, canvas: OffscreenCanvas) {
        this.device = device;
        this.canvasEl = canvas;
        const ctx = canvas.getContext('webgpu');
        if (!ctx) throw new Error('Failed to get WebGPU context');
        this.ctx = ctx;
        this.ctx.configure({ device: this.device, format: this.format, alphaMode: 'premultiplied' });
    }

    public gpuContext(): GPUCanvasContext | null {
        return this.ctx;
    }

    public canvas(): OffscreenCanvas {
        return this.canvasEl;
    }

    public setSize(width: number, height: number): void {
        if (width <= 0 || height <= 0) return;
        if (width === this._width && height === this._height) return;
        this._width = width;
        this._height = height;
        this.canvasEl.width = width;
        this.canvasEl.height = height;
        for (const fn of this.resizeListeners) fn(width, height);
    }

    public width(): number {
        return this._width;
    }

    public height(): number {
        return this._height;
    }

    public onResize(fn: (width: number, height: number) => void): void {
        this.resizeListeners.push(fn);
    }
}

class WebGPUUnavailableErrorImpl extends Error implements WebGPUUnavailableError {
    constructor(message: string) {
        super(message);
        this.name = 'WebGPUUnavailableError';
    }
}

/** requestAdapter never resolves on some machines; do not let callers wait forever. */
const ADAPTER_TIMEOUT_MS = 8000;

function raceTimeout<T>(p: Promise<T>, ms: number): Promise<T | null> {
    return new Promise((resolve) => {
        const timer = setTimeout(() => resolve(null), ms);
        p.then(
            (v) => {
                clearTimeout(timer);
                resolve(v);
            },
            () => {
                clearTimeout(timer);
                resolve(null);
            },
        );
    });
}

createDevice = async () => {
    if (!navigator.gpu) {
        throw new WebGPUUnavailableErrorImpl('WebGPU not supported. Try Chrome 113+.');
    }

    // On dual-GPU laptops the 'high-performance' tier may fail to get an adapter
    // while the dGPU is asleep; on timeout or null, retry once with the default tier.
    const adapter =
        (await raceTimeout(
            navigator.gpu.requestAdapter({ powerPreference: 'high-performance' }),
            ADAPTER_TIMEOUT_MS,
        )) ?? (await raceTimeout(navigator.gpu.requestAdapter(), ADAPTER_TIMEOUT_MS));
    if (!adapter) {
        throw new WebGPUUnavailableErrorImpl(
            'No WebGPU adapter. Open chrome://gpu to confirm WebGPU is Enabled, ' +
                'and check that "Use graphics acceleration" is on in system settings.',
        );
    }

    // Timestamps measure per-frame GPU sampling time (the sampling-budget
    // controller's input). If unavailable, fall back to onSubmittedWorkDone;
    // rendering is unaffected.
    const requiredFeatures: GPUFeatureName[] = [];
    if (adapter.features.has('timestamp-query')) requiredFeatures.push('timestamp-query');

    return adapter.requestDevice({ requiredFeatures });
};

/** Assigns the contract-layer class bindings and function. */
RenderContext = RenderContextImpl;
WebGPUUnavailableError = WebGPUUnavailableErrorImpl;
