import type { Vec3 } from '../../math/types';
import type { Model } from '../model/model';
import type { Background } from '../background/background';
import { RenderContext } from './context';
import { Frame } from './frame';
import { RenderLoop } from './loop';
import { vec3, normalize, cross, sub, add, scale } from '../../math/vec3';
import { Mat4 } from '../../math/mat4';

/**
 * Orbit camera: view state + its uniform buffer (112 B, layout in structs.wgsl) +
 * image output.
 *
 * Inside and out, the camera owns everything: the outside (CameraHandle) can only
 * get frames through `camera.listen`, and the matching render target
 * (RenderContext) and frame loop (RenderLoop) belong to the camera too — Engine is
 * just the assembly root that wires model, environment map and camera together.
 *
 * Like Background, it is "one piece of state + its GPU mirror": the camera knows
 * whether its parameters changed, writes the buffer when dirty, and the caller only
 * checks the return value to decide whether to reset accumulation.
 *
 * The file has two layers: contract (all exports) and implementation (no exports);
 * see AGENTS.md for the convention.
 */

// ═════════════════ Contract layer (exports) ══════════════════

/**
 * Scene camera placement: the framing description specified by the scene author.
 *
 * Composition is part of the scene — the same scene with different parameters is a
 * different image — so the scene author provides it along with the scene. Camera
 * turns this description into orbit parameters and a 112 B uniform; how the mouse
 * moves it afterwards is the viewer's business.
 *
 * This is also the single definition shared by the API (the `CameraHandle.set`
 * message) and the engine. The API side references it only through `import type`,
 * so it is erased at compile time and does not pull engine runtime code into the
 * main-thread bundle.
 */
export interface CameraView {
    target: { x: number; y: number; z: number };
    /** Distance from the camera to target (meters, orbit radius) */
    distance: number;
    /** Horizontal angle; π/2 means looking from the +z direction */
    theta: number;
    /** Pitch angle; π/2 is a horizontal view, and smaller values raise the camera */
    phi: number;
    /**
     * Vertical field of view in radians. Horizontal FOV = 2·atan(tan(fovY/2) · aspect),
     * so it controls only the vertical opening; the horizontal extent follows the window.
     */
    fovY: number;
    /**
     * Entrance pupil radius (meters). Omitted or 0 = pinhole camera, no depth of field.
     *
     * This is the physical size of the lens, not a "blur strength": a thin lens has
     * only two degrees of freedom, how wide the aperture is and how far it focuses.
     * Convert the f-number from the focal length yourself (a 50 mm lens at f/2.8 has
     * a 17.9 mm pupil diameter, 8.9 mm radius).
     */
    apertureRadius?: number;
    /**
     * Focus distance (meters, measured from the camera along the view direction).
     * Only takes effect together with `apertureRadius`.
     *
     * Focus is a lens state, not something the mouse orbit turns — when the viewer
     * orbits the scene, the focal plane stays at the depth the scene author chose.
     */
    focusDistance?: number;
}

/**
 * Camera class binding: construction starts work — it creates the render target,
 * uniform, Frame and frame loop.
 *
 * Initial values are just placeholders: the caller overwrites them with `applyView`
 * when loading a scene. Resolution belongs to the render target, but the camera
 * follows it, so the constructor subscribes to canvas size changes.
 *
 * This is only a declaration; the assignment is in the implementation layer.
 */
export let Camera: {
    new (device: GPUDevice, canvas: OffscreenCanvas, model: Model, background: Background): Camera;
};

/**
 * Orbit camera: view state + its uniform buffer (112 B) + image output.
 *
 * The render target (RenderContext) and frame loop (RenderLoop) belong to the
 * camera — Engine is just the assembly root. Like Background, it is "one piece of
 * state + its GPU mirror": the camera knows whether its parameters changed, writes
 * the buffer when dirty, and the caller only checks upload()'s return value to
 * decide whether to reset accumulation.
 */
export interface Camera {
    /** 112 B camera uniform (layout in structs.wgsl). */
    readonly buffer: GPUBuffer;

    /**
     * Writes the uniform buffer when parameters changed (dirty); returns whether
     * anything changed.
     */
    upload(): boolean;
    /**
     * Pushes the current image out at an interval — the image is the camera's
     * output, so delivery hangs off the camera too.
     *
     * Same semantics as the main thread's `CameraHandle.listen`: when rendering is
     * slower than the interval, the same frame is not pushed twice. Only one
     * listener is kept at a time; calling again replaces the old one.
     */
    startListening(intervalMs: number, onFrame: (frame: ImageBitmap) => void): void;
    stopListening(): void;
    /** Submits one frame (driven by the render loop). */
    renderFrame(): boolean;
    /** Whether another frame can be submitted — the render loop's throttle check. */
    readyForNextFrame(): boolean;
    /**
     * Whether the previous frame restarted accumulation — the render loop switches
     * sampling-budget tiers accordingly.
     */
    restarted(): boolean;
    /**
     * GPU budget per frame (milliseconds) — called by the render loop through the
     * FrameSource structural type.
     */
    setTargetFrameMs(ms: number): void;
    vpMatrix(): Mat4;
    invViewProj(): Mat4;
    /**
     * Loads a whole camera placement at once: scene placements and camera-move
     * keyframes both go through this.
     *
     * Near and far planes follow the distance; the caller only reports distance.
     * There is no "frame the bounding box automatically" fallback: auto-framing is
     * always the wrong composition for the gallery — the camera backs off until the
     * whole scene just fits, while the gallery wants the framing the scene author
     * picked.
     */
    applyView(view: CameraView): void;
    /**
     * Viewport size (physical pixels) — measured by the main thread and sent in;
     * canvas, camera and all passes change together.
     */
    setSize(width: number, height: number): void;
    orbitTheta(delta: number): void;
    orbitPhi(delta: number): void;
    zoom(factor: number): void;
    pan(deltaX: number, deltaY: number): void;
    destroy(): void;
}

// ════════════ Implementation layer (not exported) ═════════════

class CameraImpl implements Camera {
    public readonly buffer: GPUBuffer;
    /** Where images are being pushed to; null = no listener. */
    private listener: Listener | null = null;
    private listenTimer: ReturnType<typeof setTimeout> | null = null;
    private dirty = true;
    private target: Vec3 = { x: 0, y: 0, z: 0 };
    private orbitDist = 5;
    private theta = 0;
    private phi = Math.PI / 3;
    private fovYRads = (60 * Math.PI) / 180;
    /** Entrance pupil radius (meters). 0 = pinhole. */
    private apertureRadius = 0;
    /** Focus distance (meters). 0 = infinity. */
    private focusDistance = 0;
    private aspect = 16 / 9;
    private near = 0.01;
    private far = 100;
    private renderWidth = 800;
    private renderHeight = 600;
    private readonly device: GPUDevice;
    /** Render target — images from the camera land on this canvas. */
    private readonly context: RenderContext;
    /** The camera's renderer; images come only from here. */
    private readonly frame: Frame;
    /**
     * Frame loop: when to render the next frame and how much budget each frame
     * gets are also owned by the camera.
     */
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

        // vec4 resolution: only x and y are used; zw are left empty
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
            // The callback may have cancelled listening: schedule the next one
            // only if still attached
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

    /**
     * Camera placement tuple: target + distance + two angles. Near and far planes
     * follow the distance.
     */
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
     * Lens: aperture radius + focus distance, both in meters. Either one at 0 falls
     * back to a pinhole.
     *
     * The unit is meters rather than an f-number because the engine does not know
     * the sensor size — focal length lives only in fovY. Converting "50 mm at
     * f/2.8" to a radius in meters is a framing concern, not a rendering one.
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
    /** Which frame was pushed last. */
    lastSent: number;
}

/** Assignment of the contract-layer class binding. */
Camera = CameraImpl;
