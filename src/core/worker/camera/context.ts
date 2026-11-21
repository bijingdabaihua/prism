/**
 * GPU 基础设施：申请设备（createDevice）并把 GPUDevice 和 OffscreenCanvas 绑定为一个整体。
 * RenderContext 是渲染目标尺寸的单一持有者。
 *
 * 画布宿主只有 OffscreenCanvas 一种 —— 整条渲染管线只活在 Worker 中。离屏画布读不到
 * CSS 布局（没有 clientWidth / devicePixelRatio），尺寸只能由主线程通过 camera.resize
 * 送进来。
 *
 * Canvas 的宽高是共享基础设施，不属于任何单个 pass；尺寸变化通过 onResize 订阅通知
 * 各组件（Camera / PathTracePass 等），各自响应。尺寸的唯一来源是 setSize() ——
 * Worker 收到的 resize 命令 —— 应用即通知，没有逐帧探测。
 *
 * 文件分两层：契约层（全部导出）+ 实现层（全部不导出），规范见 CODE_STYLE.md。
 */

// ══════════════════════════ 契约层（导出） ══════════════════════════

/**
 * RenderContext 的类绑定：把 device 和 canvas 绑成一个渲染目标。
 *
 * 此处只有声明，赋值在实现层。
 */
export let RenderContext: {
    new (device: GPUDevice, canvas: OffscreenCanvas): RenderContext;
};

/**
 * 渲染目标：GPUDevice + canvas 的整体，渲染目标尺寸的单一持有者。
 *
 * Canvas 的宽高是共享基础设施，不属于任何单个 pass；尺寸变化通过 onResize 订阅通知
 * 各组件（Camera / PathTracePass 等），各自响应。
 */
export interface RenderContext {
    readonly device: GPUDevice;
    /** 平台首选 canvas 格式，device 无关，构造时即可确定。 */
    readonly format: GPUTextureFormat;

    gpuContext(): GPUCanvasContext | null;
    canvas(): OffscreenCanvas;
    /** 尺寸来源是外部（resize 命令）：直接应用并通知订阅者。 */
    setSize(width: number, height: number): void;
    width(): number;
    height(): number;
    /** 注册尺寸变化回调。回调在 setSize() 内同步触发。 */
    onResize(fn: (width: number, height: number) => void): void;
}

/**
 * WebGPUUnavailableError 的类绑定：拿不到 WebGPU 设备 / adapter 时抛出。
 *
 * 调用方据此走降级页，而不是当成普通错误。
 */
export let WebGPUUnavailableError: {
    new (message: string): WebGPUUnavailableError;
};

/** 拿不到 WebGPU 设备 —— 调用方据此走降级页，而不是当成普通错误。 */
export interface WebGPUUnavailableError extends Error {}

/**
 * 申请 adapter + device。主线程和 Worker 走同一条 —— 设备的申请条件只写一遍。
 *
 * 浏览器没有 WebGPU、或者拿不到 adapter（驱动进黑名单、虚拟机、关了硬件加速），
 * 都抛 WebGPUUnavailableError；requestDevice 自身的失败原样上抛。
 */
export let createDevice: () => Promise<GPUDevice>;

// ══════════════════════════ 实现层（不导出） ══════════════════════════

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

/** requestAdapter 在某些机器上会一直不 resolve，别让调用方无限等待。 */
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

    // 双显卡笔记本上 dGPU 休眠时 'high-performance' 档可能拿不到 adapter，
    // 超时或 null 都退回默认档再试一次。
    const adapter =
        (await raceTimeout(
            navigator.gpu.requestAdapter({ powerPreference: 'high-performance' }),
            ADAPTER_TIMEOUT_MS,
        )) ?? (await raceTimeout(navigator.gpu.requestAdapter(), ADAPTER_TIMEOUT_MS));
    if (!adapter) {
        throw new WebGPUUnavailableErrorImpl(
            'No WebGPU adapter. 打开 chrome://gpu 确认 WebGPU 为 Enabled，' +
                '并检查系统设置里「使用图形加速」是否开启。',
        );
    }

    // 时间戳用于测量每帧 GPU 采样耗时（采样预算控制器的输入）。
    // 拿不到就退回 onSubmittedWorkDone，不影响渲染。
    const requiredFeatures: GPUFeatureName[] = [];
    if (adapter.features.has('timestamp-query')) requiredFeatures.push('timestamp-query');

    return adapter.requestDevice({ requiredFeatures });
};

/** 契约层类绑定与函数的赋值。 */
RenderContext = RenderContextImpl;
WebGPUUnavailableError = WebGPUUnavailableErrorImpl;
