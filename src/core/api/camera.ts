import type { CameraView } from '../worker/camera/camera';

/**
 * Main-thread camera handle: contract + implementation.
 *
 * The real camera lives in `../worker/camera.ts`; this side only turns methods into messages.
 * The camera message shape re-exports the shared `CameraView` (defined in `../worker/camera.ts`),
 * so there is a single definition. The factory is `createCamera(rpc)`; the implementation class
 * is not exported.
 */

/**
 * Public contract
 *
 * The types and interfaces below are the external API. Check caller impact before changing.
 */

export type { CameraView } from '../worker/camera/camera';

/** Render viewport size (physical pixels = CSS size × devicePixelRatio). */
export interface ViewportSize {
    width: number;
    height: number;
}

/**
 * Camera handle — all methods are synchronous: they return immediately and do not wait for a frame.
 *
 * Aspect ratio is part of projection, so size belongs to the camera (`resize`), not the canvas:
 * the camera does not know which canvas it draws to, only how wide the image is.
 *
 * Angles are always in radians, translations are always in screen pixels — the former is camera
 * semantics, the latter maps directly to pointer events. The feel coefficient in between (e.g.
 * radians per pixel) stays in the app layer.
 */
export interface CameraHandle {
    /**
     * Sets the camera placement directly (a full frame, not a delta).
     *
     * The usual order for assembling a scene is `set` → `background.setImage` /
     * `background.setRotation` → `scene.load`; loading geometry does not change the camera.
     */
    set(spec: CameraView): void;
    /** Viewport size (physical pixels). Aspect ratio follows it. */
    resize(size: ViewportSize): void;
    /**
     * Orbits around the target in radians. phi is clamped to (0, π), so it cannot flip over
     * the top.
     */
    orbit(deltaTheta: number, deltaPhi: number): void;
    /** Dolly: >1 moves out, <1 moves in. Multiplicative, so any drag distance is proportional. */
    zoom(factor: number): void;
    /** Pans the target in screen pixels — fed directly from pointer movement. */
    pan(deltaX: number, deltaY: number): void;
    /**
     * Pushes the current frame back at an interval — the frame is the camera's output,
     * so this belongs on the camera.
     *
     * Pushes at most one frame per `intervalMs`; if rendering is slower, it waits for the next
     * frame and does not push the same image twice. The frame is an `ImageBitmap`: the main
     * thread pastes it directly with bitmaprenderer's `transferFromImageBitmap()`, with no
     * readback or encoding.
     *
     * Returns a cancel function. Only one listener is allowed at a time.
     */
    listen(intervalMs: number, onFrame: (frame: ImageBitmap) => void): () => void;
}

/**
 * Implementation
 *
 * The following is not part of the public contract: message shapes, RPC dependencies and the
 * handle implementation.
 */

/** Camera commands: all fire-and-forget write operations with no return value. */
export type CameraCommand =
    | { type: 'set'; spec: CameraView }
    | { type: 'resize'; size: ViewportSize }
    | { type: 'orbit'; deltaTheta: number; deltaPhi: number }
    | { type: 'zoom'; factor: number }
    | { type: 'pan'; deltaX: number; deltaY: number };

/** Worker messages related to the camera. */
export type CameraRequest =
    | { type: 'camera'; command: CameraCommand }
    /** Starts pushing frames back to the main thread at an interval. */
    | { type: 'listen/start'; intervalMs: number }
    | { type: 'listen/stop' };

/**
 * The minimal RPC surface the camera handle needs.
 *
 * Does not depend on a concrete RpcClient class; it only requires sending messages and
 * registering a frame callback.
 */
export interface CameraRpc {
    send(request: CameraRequest): void;
    setFrameHandler(handler: ((bitmap: ImageBitmap) => void) | null): void;
}

/**
 * Main-thread camera handle: forwards each command to the Worker and does not wait
 * for an acknowledgement.
 */
class RpcCamera implements CameraHandle {
    private listener: ((frame: ImageBitmap) => void) | null = null;

    constructor(private readonly rpc: CameraRpc) {}

    public set(spec: CameraView): void {
        this.rpc.send({ type: 'camera', command: { type: 'set', spec } });
    }

    public resize(size: ViewportSize): void {
        this.rpc.send({ type: 'camera', command: { type: 'resize', size } });
    }

    public orbit(deltaTheta: number, deltaPhi: number): void {
        this.rpc.send({ type: 'camera', command: { type: 'orbit', deltaTheta, deltaPhi } });
    }

    public zoom(factor: number): void {
        this.rpc.send({ type: 'camera', command: { type: 'zoom', factor } });
    }

    public pan(deltaX: number, deltaY: number): void {
        this.rpc.send({ type: 'camera', command: { type: 'pan', deltaX, deltaY } });
    }

    public listen(intervalMs: number, onFrame: (frame: ImageBitmap) => void): () => void {
        if (this.listener) throw new Error('camera.listen: a listener is already registered');
        this.listener = onFrame;
        this.rpc.setFrameHandler(onFrame);
        this.rpc.send({ type: 'listen/start', intervalMs });
        return () => {
            if (this.listener !== onFrame) return;
            this.listener = null;
            this.rpc.setFrameHandler(null);
            try {
                this.rpc.send({ type: 'listen/stop' });
            } catch {
                // Already disposed: the Worker is gone, so there is nothing to stop
            }
        };
    }
}

/** Factory for the main-thread camera handle; the implementation class is not exported. */
export function createCamera(rpc: CameraRpc): CameraHandle {
    return new RpcCamera(rpc);
}
