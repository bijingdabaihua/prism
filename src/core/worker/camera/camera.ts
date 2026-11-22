import type { Vec3 } from '../../math/types';
import type { Model } from '../model/model';
import type { Background } from '../background/background';
import { RenderContext } from './context';
import { Frame } from './frame';
import { RenderLoop } from './loop';
import { vec3, normalize, cross, sub, add, scale } from '../../math/vec3';
import { Mat4 } from '../../math/mat4';

/**
 * 轨道相机：观察状态 + 它的 uniform buffer（112 B，布局见 structs.wgsl）+ 出图。
 *
 * 内外一致：外面（CameraHandle）只有 `camera.listen` 能拿到画面，里面对应的
 * 渲染目标（RenderContext）和帧循环（RenderLoop）也都归相机 —— Engine 只是
 * 把模型、环境图、相机拼起来的组装根。
 *
 * 和 Background 一样是「一份状态 + 它的 GPU 镜像」：相机自己知道参数变化了
 * 没有，脏了就写 buffer，调用方只看返回值决定要不要重置累积。
 *
 * 文件分两层：契约层（全部导出）+ 实现层（全部不导出），规范见 CODE_STYLE.md。
 */

// ══════════════════════════ 契约层（导出） ══════════════════════════

/**
 * 场景机位：场景作者指定的取景描述。
 *
 * 构图是场景的一部分 —— 同一个场景换一套参数就是另一张图，所以它由场景
 * 作者随场景一起给出。Camera 把这份描述变成轨道参数和 112 B uniform；
 * 之后鼠标怎么转，是观察者的事。
 *
 * 这也是 api（`CameraHandle.set` 的报文）和引擎共用的唯一一份定义。api 侧
 * 只以 `import type` 引用它，编译时擦除，不会把引擎运行时代码带进主线程包。
 */
export interface CameraView {
    target: { x: number; y: number; z: number };
    /** 相机到 target 的距离（米，轨道半径） */
    distance: number;
    /** 水平角，π/2 表示从 +z 方向看过来 */
    theta: number;
    /** 俯仰角，π/2 是水平视线，越小相机越高 */
    phi: number;
    /**
     * 垂直视野角，弧度。水平视野 = 2·atan(tan(fovY/2) · 宽高比)，
     * 所以它只管上下开口，左右跟着窗口走。
     */
    fovY: number;
    /**
     * 入射光瞳半径（米）。省略或 0 = 针孔相机，没有景深。
     *
     * 这是镜头的物理尺寸，不是「虚化强度」：薄透镜只有两个自由度，光圈多大、
     * 对焦在多远。f 值要自己按焦距换算（50 mm 镜头开 f/2.8，光瞳直径
     * 17.9 mm，半径 8.9 mm）。
     */
    apertureRadius?: number;
    /**
     * 对焦距离（米，沿视线从相机量起）。需要和 `apertureRadius` 一起给才生效。
     *
     * 对焦是镜头的状态，不跟着鼠标轨道转 —— 观察者绕着场景转时，合焦面停在
     * 场景作者定的那层深度上。
     */
    focusDistance?: number;
}

/**
 * Camera 的类绑定：构造即开工 —— 建渲染目标、uniform、Frame 和帧循环。
 *
 * 初始值只是占位：外部装场景时会用 `applyView` 覆盖掉。
 * 分辨率是渲染目标的事，但相机跟着它变，所以构造时订阅 canvas 尺寸变化。
 *
 * 此处只有声明，赋值在实现层。
 */
export let Camera: {
    new (device: GPUDevice, canvas: OffscreenCanvas, model: Model, background: Background): Camera;
};

/**
 * 轨道相机：观察状态 + 它的 uniform buffer（112 B）+ 出图。
 *
 * 渲染目标（RenderContext）和帧循环（RenderLoop）都归相机 —— Engine 只是
 * 组装根。和 Background 一样是「一份状态 + 它的 GPU 镜像」：相机自己知道
 * 参数变化了没有，脏了就写 buffer，调用方只看 upload() 的返回值决定要不要
 * 重置累积。
 */
export interface Camera {
    /** 112 B 机位 uniform（布局见 structs.wgsl）。 */
    readonly buffer: GPUBuffer;

    /** 参数变化（脏）时写入 uniform buffer；返回是否有变化。 */
    upload(): boolean;
    /**
     * 按间隔把当前画面推出去 —— 画面是相机的输出，所以推送也挂在相机上。
     *
     * 和主线程的 `CameraHandle.listen` 同一个语义：渲染比间隔慢时不重复推
     * 同一帧。同一时刻只保留一个监听者，重复调用会替换旧的。
     */
    startListening(intervalMs: number, onFrame: (frame: ImageBitmap) => void): void;
    stopListening(): void;
    /** 提交一帧（渲染循环驱动）。 */
    renderFrame(): boolean;
    /** 这一帧还能不能提交 —— 渲染循环的节流判断。 */
    readyForNextFrame(): boolean;
    /** 上一帧是否重开了累积 —— 渲染循环据此切换采样预算档位。 */
    restarted(): boolean;
    /** 每帧 GPU 预算（毫秒）—— 渲染循环通过 FrameSource 结构类型调到。 */
    setTargetFrameMs(ms: number): void;
    vpMatrix(): Mat4;
    invViewProj(): Mat4;
    /**
     * 整套机位一次装进来：场景机位、运镜关键帧都走这一条。
     *
     * 近远平面跟着距离走，外部只报距离。没有「按包围盒自动取景」这条退路：
     * 自动取景在画廊里永远是错的构图 —— 相机会退到刚好看全整个场景的地方，
     * 而画廊要的是场景作者选的那一格。
     */
    applyView(view: CameraView): void;
    /** 视口尺寸（物理像素）—— 主线程量好送进来，画布、相机和各 pass 一起变。 */
    setSize(width: number, height: number): void;
    orbitTheta(delta: number): void;
    orbitPhi(delta: number): void;
    zoom(factor: number): void;
    pan(deltaX: number, deltaY: number): void;
    destroy(): void;
}

// ══════════════════════════ 实现层（不导出） ══════════════════════════

class CameraImpl implements Camera {
    public readonly buffer: GPUBuffer;
    /** 正在往哪里推画面；null = 没人监听。 */
    private listener: Listener | null = null;
    private listenTimer: ReturnType<typeof setTimeout> | null = null;
    private dirty = true;
    private target: Vec3 = { x: 0, y: 0, z: 0 };
    private orbitDist = 5;
    private theta = 0;
    private phi = Math.PI / 3;
    private fovYRads = (60 * Math.PI) / 180;
    /** 入射光瞳半径（米）。0 = 针孔。 */
    private apertureRadius = 0;
    /** 对焦距离（米）。0 = 无穷远。 */
    private focusDistance = 0;
    private aspect = 16 / 9;
    private near = 0.01;
    private far = 100;
    private renderWidth = 800;
    private renderHeight = 600;
    private readonly device: GPUDevice;
    /** 渲染目标 —— 相机出的图落在这块画布上。 */
    private readonly context: RenderContext;
    /** 相机的渲染器；画面只从这里出。 */
    private readonly frame: Frame;
    /** 帧循环：什么时候再渲染一帧、每帧多少预算，也由相机管。 */
    private readonly loop: RenderLoop;

    constructor(device: GPUDevice, canvas: OffscreenCanvas, model: Model, background: Background) {
        this.device = device;
        this.context = new RenderContext(device, canvas);
        this.buffer = this.device.createBuffer({
            size: 112,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            label: 'camera-uniform',
        });
        this.frame = new Frame(this.context, model, this, background);
        this.context.onResize((w, h) => this.setResolution(w, h));
        this.loop = new RenderLoop(this);
    }

    public upload(): boolean {
        if (!this.dirty) return false;
        this.dirty = false;

        const buf = new ArrayBuffer(112);
        const f32 = new Float32Array(buf);

        f32.set(this.invViewProj().raw(), 0);

        const eye = this.computeEye();
        f32[16] = eye.x;
        f32[17] = eye.y;
        f32[18] = eye.z;
        f32[19] = 1.0;

        // vec4 resolution：只用得到 x、y，zw 留空
        f32[20] = this.renderWidth;
        f32[21] = this.renderHeight;

        f32[24] = this.fovYRads;
        f32[25] = this.apertureRadius;
        f32[26] = this.focusDistance;

        this.device.queue.writeBuffer(this.buffer, 0, buf);
        return true;
    }

    public startListening(intervalMs: number, onFrame: (frame: ImageBitmap) => void): void {
        this.stopListening();
        const listener: Listener = { intervalMs: Math.max(1, intervalMs), onFrame, lastSent: -1 };
        this.listener = listener;
        this.deliver(listener);
    }

    public stopListening(): void {
        this.listener = null;
        if (this.listenTimer !== null) {
            clearTimeout(this.listenTimer);
            this.listenTimer = null;
        }
    }

    public renderFrame(): boolean {
        return this.frame.renderFrame();
    }

    public readyForNextFrame(): boolean {
        return this.frame.readyForNextFrame();
    }

    public restarted(): boolean {
        return this.frame.restarted();
    }

    public setTargetFrameMs(ms: number): void {
        this.frame.setTargetFrameMs(ms);
    }

    public vpMatrix(): Mat4 {
        const view = Mat4.lookAt(this.computeEye(), this.target, { x: 0, y: 1, z: 0 });
        return Mat4.perspective(this.fovYRads, this.aspect, this.near, this.far).multiply(view);
    }

    public invViewProj(): Mat4 {
        return this.vpMatrix().invert();
    }

    public applyView(view: CameraView): void {
        this.setFovY(view.fovY);
        this.setView(view.target, view.distance, view.theta, view.phi);
        this.setLens(view.apertureRadius ?? 0, view.focusDistance ?? 0);
    }

    public setSize(width: number, height: number): void {
        this.context.setSize(width, height);
    }

    public orbitTheta(delta: number): void {
        this.theta += delta;
        this.dirty = true;
    }

    public orbitPhi(delta: number): void {
        this.phi = Math.max(0.01, Math.min(Math.PI - 0.01, this.phi + delta));
        this.dirty = true;
    }

    public zoom(factor: number): void {
        this.orbitDist = Math.max(0.1, this.orbitDist * factor);
        this.dirty = true;
    }

    public pan(deltaX: number, deltaY: number): void {
        const eye = this.computeEye();
        const viewDir = normalize(sub(this.target, eye));
        const right = normalize(cross(viewDir, { x: 0, y: 1, z: 0 }));
        const up = cross(right, viewDir);
        const speed = this.orbitDist * 0.008;
        this.target = add(this.target, add(scale(right, -deltaX * speed), scale(up, deltaY * speed)));
        this.dirty = true;
    }

    public destroy(): void {
        this.loop.stop();
        this.stopListening();
        this.frame.destroy();
        this.buffer?.destroy();
    }

    private setResolution(w: number, h: number): void {
        this.renderWidth = w;
        this.renderHeight = h;
        this.aspect = w / h;
        this.dirty = true;
    }

    private deliver(listener: Listener): void {
        if (this.listener !== listener) return;
        try {
            const count = this.frame.frameCount();
            if (count !== listener.lastSent) {
                const bitmap = this.frame.capture();
                if (bitmap) {
                    listener.lastSent = count;
                    listener.onFrame(bitmap);
                }
            }
        } finally {
            // 回调里可能已经取消了监听：只有还挂着才排下一次
            if (this.listener === listener) {
                this.listenTimer = setTimeout(() => this.deliver(listener), listener.intervalMs);
            }
        }
    }

    private computeEye(): Vec3 {
        const ct = Math.cos(this.theta),
            st = Math.sin(this.theta);
        const cp = Math.cos(this.phi),
            sp = Math.sin(this.phi);
        return vec3(
            this.target.x + this.orbitDist * sp * ct,
            this.target.y + this.orbitDist * cp,
            this.target.z + this.orbitDist * sp * st,
        );
    }

    /** 机位四元组：目标点 + 距离 + 两个角。近远平面跟着距离走。 */
    private setView(target: Vec3, distance: number, theta: number, phi: number): void {
        this.target = target;
        this.orbitDist = Math.max(distance, 0.01);
        this.theta = theta;
        this.phi = Math.max(0.01, Math.min(Math.PI - 0.01, phi));
        this.near = this.orbitDist * 0.01;
        this.far = Math.max(this.orbitDist * 10, 100);
        this.dirty = true;
    }

    private setFovY(radians: number): void {
        this.fovYRads = radians;
        this.dirty = true;
    }

    /**
     * 镜头：光圈半径 + 对焦距离，两个都是米。任一个为 0 就退回针孔。
     *
     * 单位是米而不是 f 值，因为引擎不知道传感器多大 —— 焦距只活在 fovY 里。
     * 场景作者要把「50 mm 开 f/2.8」换成年半径，那是取景的问题，不是渲染的问题。
     */
    private setLens(apertureRadius: number, focusDistance: number): void {
        this.apertureRadius = Math.max(0, apertureRadius);
        this.focusDistance = Math.max(0, focusDistance);
        this.dirty = true;
    }
}

interface Listener {
    readonly intervalMs: number;
    readonly onFrame: (frame: ImageBitmap) => void;
    /** 上一次推出去的是第几帧。 */
    lastSent: number;
}

/** 契约层类绑定的赋值。 */
Camera = CameraImpl;
