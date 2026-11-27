/**
 * The public surface of core: facade + main-thread RPC + message protocol + errors.
 *
 * The contracts and main-thread implementations of the three handles live in `camera.ts` /
 * `background.ts` / `model.ts`; this file only assembles them. The main thread imports only
 * from `core/prism`; the Worker performs pure type imports from the same file, leaving no
 * runtime dependency after compilation.
 */

import type { BackgroundHandle, BackgroundRequest } from './background';
import type { CameraHandle, CameraRequest } from './camera';
import type { ModelHandle, ModelRequest } from './model';
import { createBackground } from './background';
import { createCamera } from './camera';
import { createModel } from './model';

export type { BackgroundHandle } from './background';
export type { CameraHandle, CameraView, ViewportSize } from './camera';
export type { InstancePlacement, ModelHandle } from './model';

/**
 * Public contract
 *
 * The types, interfaces and facade signature below are the external API. Check caller impact
 * before changing; the concrete implementation starts at the "Implementation" section of this file.
 */

/**
 * Facade contract (instance side).
 *
 * The factory is `Prism.create()`; an instance exposes only three read-only handles and
 * `dispose()`. The handle contracts live in `camera.ts` / `background.ts` / `model.ts`.
 */
export interface Prism {
    /** Observer: camera control. */
    readonly camera: CameraHandle;
    /** Environment lighting. */
    readonly background: BackgroundHandle;
    /** Scene geometry: glTF bytes and externally built Instances, two entry points. */
    readonly model: ModelHandle;
    /** Releases the Worker and the GPU device; repeated calls have no effect. */
    dispose(): void;
}

/** Facade factory: the signature of `Prism.create()`. */
export interface PrismFactory {
    create(): Promise<Prism>;
}

/**
 * This machine cannot run WebGPU — the caller uses this to go to the fallback page,
 * not an error state.
 */
export class PrismUnsupportedError extends Error {
    constructor(message = 'WebGPU is not available in this browser') {
        super(message);
        this.name = 'PrismUnsupportedError';
    }
}

/**
 * Implementation
 *
 * The following is not part of the public contract: internal message protocol, the concrete
 * facade implementation and main-thread RPC.
 */

/** Camera command types on the Worker side; defined by the implementation section of camera.ts. */
export type { CameraCommand } from './camera';

/**
 * Main thread → Worker.
 *
 * Two kinds of messages: write operations (camera commands) have no id; calls with a return
 * value carry an auto-incrementing id, and the Worker replies with result / failure.
 * "Cannot run WebGPU" is a separate reply kind, which the main thread turns into
 * PrismUnsupportedError.
 *
 * The message shapes of the four handles are defined in their own files; this is only the union.
 */
export type WorkerRequest =
    | { type: 'init'; id: number }
    | CameraRequest
    | BackgroundRequest
    | ModelRequest;

/** Worker → main thread. */
export type WorkerResponse =
    /** Success: value is this call's return value (absent for void calls). */
    | { type: 'result'; id: number; value?: unknown }
    /** The call failed. */
    | { type: 'failure'; id: number; message: string }
    /** This machine cannot run WebGPU. */
    | { type: 'unsupported'; id: number; message: string }
    /** One frame (pushed by listen, no id). */
    | { type: 'frame'; bitmap: ImageBitmap };

/**
 * Facade: starts the Worker, assembles the three handles and manages the lifecycle.
 *
 * The contracts and main-thread implementations of the three handles live in camera.ts /
 * background.ts / model.ts; this file only creates and holds them. Actual rendering runs in
 * the Worker — one instance per Worker and per separate GPUDevice; the main thread never
 * touches the GPU.
 *
 * The app layer knows only this facade. The frame is the camera's output:
 * `camera.listen(interval, cb)` pushes one ImageBitmap back to the main thread every interval
 * milliseconds; the worker owns the canvas internally.
 */
class PrismImpl implements Prism {
    /**
     * Async factory: one call = one Worker = one separate GPUDevice.
     *
     * The main thread first checks navigator.gpu — if absent, it does not even start the Worker
     * and rejects immediately. This check only means "the browser has this API"; it does not
     * guarantee it works: requestAdapter() in the Worker may still return null (driver
     * blocklisted, VM, hardware acceleration off), and that case is also rejected with
     * PrismUnsupportedError — the message comes back from the Worker.
     *
     * No canvas is passed in: the Worker creates its own OffscreenCanvas and pushes frames back
     * as ImageBitmap through `camera.listen()` — the main thread never touches the GPU.
     */
    public static async create(): Promise<Prism> {
        if (!navigator.gpu) {
            throw new PrismUnsupportedError('WebGPU is not available. Try Chrome 113+.');
        }

        const worker = new Worker(new URL('../worker/worker.ts', import.meta.url), { type: 'module' });
        const rpc = new RpcClient(worker);
        try {
            await withTimeout(
                rpc.init(),
                READY_TIMEOUT_MS,
                `Prism.create: Worker did not become ready within ${READY_TIMEOUT_MS} ms`,
            );
        } catch (err) {
            // A Worker that failed midway cannot be kept around: it may already hold the device
            rpc.dispose();
            worker.terminate();
            throw err;
        }
        return new PrismImpl(worker, rpc);
    }

    /** Observer: camera control. */
    public readonly camera: CameraHandle;
    /** Environment lighting. */
    public readonly background: BackgroundHandle;
    /** Scene geometry: glTF bytes and externally built Instances, two entry points. */
    public readonly model: ModelHandle;
    private disposed = false;
    private readonly rpc: RpcClient;

    private constructor(
        private readonly worker: Worker,
        rpc: RpcClient,
    ) {
        this.rpc = rpc;
        this.camera = createCamera(rpc);
        this.background = createBackground(rpc);
        this.model = createModel(rpc);
    }

    /**
     * Releases this instance's Worker and GPU device.
     *
     * The instance owns one Worker and one GPUDevice; after dispose all handles are invalid,
     * and repeated calls have no effect. Not calling dispose leaves a Worker and a chunk of
     * VRAM occupied indefinitely.
     */
    public dispose(): void {
        if (this.disposed) return;
        this.disposed = true;
        this.rpc.dispose();
        this.worker.terminate();
    }
}

/**
 * Exported facade factory: the value is PrismImpl; externally, instances can only be created
 * through `Prism.create()`. The constructor value is kept so `instanceof` still works; the
 * implementation class itself is not exported separately.
 */
export const Prism: PrismFactory & typeof PrismImpl = PrismImpl;

/**
 * RPC on the main-thread side: send messages, receive replies.
 *
 * Calls with a return value carry an auto-incrementing id; write operations (camera commands)
 * are fire-and-forget. Messages on the same MessagePort are FIFO — so after `camera.set()`
 * immediately followed by `camera.listen()`, the Worker always sees the camera placement first
 * and then starts pushing frames.
 *
 * The three handles use it only through their own minimal RPC interfaces: message senders send
 * messages, request senders send requests, and none of them knows the Worker-side implementation.
 * The concrete class itself is not exported.
 */
class RpcClient {
    private nextId = 1;
    private closed = false;
    /** Frames pushed back by listen. Only one listener is served at a time. */
    private frameHandler: ((bitmap: ImageBitmap) => void) | null = null;
    private readonly pending = new Map<number, Pending>();

    constructor(private readonly worker: Worker) {
        worker.addEventListener('message', (event: MessageEvent<WorkerResponse>) => {
            const message = event.data;
            if (message.type === 'result') {
                this.settle(message.id, (pending) => pending.resolve(message.value));
            } else if (message.type === 'failure') {
                this.settle(message.id, (pending) => pending.reject(new Error(message.message)));
            } else if (message.type === 'unsupported') {
                this.settle(message.id, (pending) =>
                    pending.reject(new PrismUnsupportedError(message.message)),
                );
            } else if (message.type === 'frame') {
                this.frameHandler?.(message.bitmap);
            }
        });
    }

    /** Who is listening for frames. null = no listener (received frames are dropped). */
    public setFrameHandler(handler: ((bitmap: ImageBitmap) => void) | null): void {
        this.frameHandler = handler;
    }

    /** Initialization: creates the device inside the Worker. */
    public init(): Promise<void> {
        return this.request<void>((id) => ({ type: 'init', id }));
    }

    /** One call with a return value; build receives the id and assembles the message. */
    public request<T>(build: (id: number) => WorkerRequest): Promise<T> {
        const id = this.nextId++;
        return new Promise<T>((resolve, reject) => {
            this.pending.set(id, { resolve: resolve as (value: unknown) => void, reject });
            try {
                this.send(build(id));
            } catch (err) {
                this.pending.delete(id);
                reject(err);
            }
        });
    }

    /**
     * Fire-and-forget: camera commands and listen toggles go through here without waiting
     * for a reply.
     */
    public send(request: WorkerRequest): void {
        if (this.closed) throw new Error('Prism has been disposed');
        this.worker.postMessage(request);
    }

    /** The Worker is gone: rejects all pending requests; later calls throw immediately. */
    public dispose(): void {
        this.closed = true;
        this.frameHandler = null;
        for (const pending of this.pending.values()) {
            pending.reject(new Error('Prism has been disposed'));
        }
        this.pending.clear();
    }

    private settle(id: number, act: (pending: Pending) => void): void {
        const pending = this.pending.get(id);
        if (!pending) return;
        this.pending.delete(id);
        act(pending);
    }
}

/**
 * Upper bound for device creation in the Worker. requestAdapter has its own 8 s timeout;
 * this adds another layer.
 */
const READY_TIMEOUT_MS = 15000;

/** Adds a timeout to a Promise; replies that arrive late are ignored by RpcClient. */
function withTimeout<T>(promise: Promise<T>, ms: number, message: string): Promise<T> {
    return new Promise<T>((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error(message)), ms);
        promise.then(
            (value) => {
                clearTimeout(timer);
                resolve(value);
            },
            (err) => {
                clearTimeout(timer);
                reject(err);
            },
        );
    });
}

interface Pending {
    resolve: (value: unknown) => void;
    reject: (err: Error) => void;
}
