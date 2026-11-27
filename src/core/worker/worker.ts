/**
 * Ray-tracing Worker. One per Prism instance, exclusively owning a GPUDevice.
 *
 * The entire render pipeline (Camera / Model / Background) runs on this thread.
 * The canvas is an OffscreenCanvas created by the Worker itself, and the main
 * thread never touches the GPU. Frames are pushed back to the main thread as
 * ImageBitmaps through `camera.listen()`, at most one per interval.
 *
 * This file is both the thread entry point and the assembly root: all runtime
 * state hangs off a single `WorkerHost` instance, which builds the device at
 * init and assembles the model / environment map / camera modules. Message
 * semantics sink straight down into the modules: camera-placement commands and
 * frame listening are delegated to camera, the environment map to background,
 * and geometry to model.add; the top level keeps only the serial queue and
 * replies.
 *
 * The render target and frame loop belong to the camera. On the main thread
 * only the camera handle manages size and frame output, mirroring the same
 * division of labor. This file only wires the three together and owns their
 * lifetimes.
 *
 * The Worker knows nothing about specific scenes: geometry either comes from
 * glTF bytes (decoded inside the worker) or is raw data built by the main
 * thread as IR and pushed over (`Instance.decode` revives it).
 *
 * Destruction is not a message: Prism.dispose simply terminates the Worker,
 * the thread disappears and the device is released with it, so there is no
 * symmetric teardown path.
 */
// Type-only import: the message protocol and facade live in the same file,
// so this edge is fully erased at compile time.
import type { CameraCommand, InstancePlacement, WorkerRequest, WorkerResponse } from '../api/prism';
import { createDevice, WebGPUUnavailableError } from './camera/context';
import { Camera } from './camera/camera';
import { Background } from './background/background';
import { Model } from './model/model';
import { loadImage } from './model/image';
import { Instance, placeInstances } from './model/instance';
import type { WireInstance } from './model/instance';

/**
 * The Worker's runtime state: three modules plus one serial queue.
 *
 * Any message the modules handle themselves (camera placement, listening,
 * environment map, geometry) is delegated down; the top level only queues
 * async operations and sends results back.
 */
class WorkerHost {
    /**
     * Render camera: created at init, owns the render target and frame loop.
     * Exactly one per Prism lifetime.
     */
    private camera: Camera | null = null;
    /** Environment lighting: assembled alongside the camera at init. */
    private background: Background | null = null;
    /** Scene geometry: BVH / material table, the destination for all added geometry. */
    private model: Model | null = null;
    /** Serial chain for async tasks. */
    private taskChain: Promise<void> = Promise.resolve();

    /**
     * Message entry point: commands without return values run directly, those
     * with return values go into the queue.
     */
    public async handle(request: WorkerRequest): Promise<void> {
        switch (request.type) {
            case 'init':
                try {
                    await this.initEngine();
                    this.reply(request.id, undefined);
                } catch (err) {
                    const message = err instanceof Error ? err.message : String(err);
                    if (err instanceof WebGPUUnavailableError) this.unsupported(request.id, message);
                    else this.failure(request.id, message);
                }
                return;
            case 'camera':
                this.applyCamera(request.command);
                return;
            case 'listen/start':
                this.camera?.startListening(Math.max(1, request.intervalMs), postFrame);
                return;
            case 'listen/stop':
                this.camera?.stopListening();
                return;
            case 'background/setImage':
                this.enqueue(() => this.setImage(request.id, request.blob));
                return;
            case 'background/setRotation':
                this.enqueue(() => this.setRotation(request.id, request.rotation));
                return;
            case 'model/loadGltf':
                this.enqueue(() => this.loadGltf(request.id, request.source, request.placement));
                return;
            case 'model/setEmissionScale':
                try {
                    if (!this.model) throw new Error('Prism is not initialized yet');
                    this.model.setEmissionScale(request.scale);
                    this.reply(request.id, undefined);
                } catch (err) {
                    this.failure(request.id, err instanceof Error ? err.message : String(err));
                }
                return;
            case 'model/loadInstances':
                this.enqueue(() => this.loadInstances(request.id, request.instances));
                return;
        }
    }

    /**
     * Builds the device and assembles the three modules; returns early if already built.
     *
     * A Prism instance has exactly one engine, assembled only by init. The
     * scene's camera placement, environment map and geometry are sent in
     * afterwards, separately. The camera is built last: its constructor takes
     * the model and background references.
     */
    private async initEngine(): Promise<void> {
        if (this.camera) return;
        const device = await createDevice();
        const background = new Background(device);
        const model = new Model(device);
        this.background = background;
        this.model = model;
        this.camera = new Camera(device, new OffscreenCanvas(1, 1), model, background);
    }

    /**
     * Serializes async tasks.
     *
     * `loadImage` / `Instance.buildByGltf` are async: call them twice in a row
     * and the earlier call may finish later, clobbering the later one. With the
     * queue, the last call issued is always the last to land.
     */
    private enqueue(task: () => Promise<void>): void {
        this.taskChain = this.taskChain.then(task);
    }

    /**
     * Swaps the environment map: Blob → decode → upload.
     *
     * The main thread only provides bytes; format detection and decoding happen
     * in the Worker. This only swaps the image and leaves the orientation
     * alone; orientation is handled separately by setRotation.
     *
     * Swapping also works while the engine is running: a change in
     * Background.version() makes PathTracePass rebuild the bind group and clear
     * accumulation on the next frame. Decoding stays at the top level
     * (loadImage is the IR entry point); only the decoded Image goes to
     * background.
     */
    private async setImage(id: number, blob: Blob): Promise<void> {
        try {
            if (!this.background) throw new Error('Prism is not initialized yet');
            this.background.setImage(await loadImage(blob));
            this.reply(id, undefined);
        } catch (err) {
            this.failure(id, err instanceof Error ? err.message : String(err));
        }
    }

    /**
     * Rotates the environment map: without re-decoding, rebuilds the texture and
     * CDF at the new angle from the source image kept by Background.
     *
     * Requires setImage first. The angle is in radians around +y, sharing the
     * same definition as `envRotation` in scene data. Accumulation clears
     * automatically: Background.version() changes, so the bind group is rebuilt
     * on the next frame.
     */
    private async setRotation(id: number, rotation: number): Promise<void> {
        try {
            if (!this.background) throw new Error('Prism is not initialized yet');
            this.background.setRotation(rotation);
            this.reply(id, undefined);
        } catch (err) {
            this.failure(id, err instanceof Error ? err.message : String(err));
        }
    }

    /**
     * Loads every Instance from one glTF file and appends them to the current scene.
     *
     * Parsing, texture decoding and optional overall placement all happen here:
     * the main thread only hands over bytes and placement parameters. Like the
     * procedural-geometry entry point, everything ultimately lands in
     * `model.add`; both entries append, so callers can invoke them several times
     * to stitch multiple sources into one scene.
     */
    private async loadGltf(id: number, source: Blob, placement?: InstancePlacement): Promise<void> {
        try {
            if (!this.model) throw new Error('Prism is not initialized yet');
            const instances = await Instance.buildByGltf(source);
            if (placement) placeInstances(instances, placement);
            this.model.add(instances);
            this.reply(id, undefined);
        } catch (err) {
            this.failure(id, err instanceof Error ? err.message : String(err));
        }
    }

    /**
     * Appends a batch of externally built Instances.
     *
     * The main thread uses `Instance.encode` to serialize IR into raw data
     * (class instances cannot cross the thread boundary); here `Instance.decode`
     * validates and revives it into IR class instances, then it takes the same
     * `model.add` path. A typical source is procedural geometry: things like
     * ground planes are built on the app side and do not go through glTF. A
     * batch compiles only once: a floor tile maps to one Instance, but not to
     * one request.
     */
    private async loadInstances(id: number, wires: readonly WireInstance[]): Promise<void> {
        try {
            if (!this.model) throw new Error('Prism is not initialized yet');
            this.model.add(wires.map((wire) => Instance.decode(wire)));
            this.reply(id, undefined);
        } catch (err) {
            this.failure(id, err instanceof Error ? err.message : String(err));
        }
    }

    /**
     * Camera-placement commands: forwarded straight to Camera methods.
     *
     * resize changes the render target size, not camera-placement parameters.
     * The camera owns the render target and frame loop, so this just hands the
     * size over. Commands are dropped when the engine is not built yet: the
     * handle side is synchronous and never throws, and `Prism.create()` only
     * hands out the handle after the init reply, so resize cannot arrive first.
     */
    private applyCamera(command: CameraCommand): void {
        if (command.type === 'resize') {
            this.camera?.setSize(command.size.width, command.size.height);
            return;
        }
        const camera = this.camera;
        if (!camera) return;

        switch (command.type) {
            case 'set':
                camera.applyView(command.spec);
                return;
            case 'orbit':
                camera.orbitTheta(command.deltaTheta);
                camera.orbitPhi(command.deltaPhi);
                return;
            case 'zoom':
                camera.zoom(command.factor);
                return;
            case 'pan':
                camera.pan(command.deltaX, command.deltaY);
                return;
        }
    }

    private reply(id: number, value: unknown): void {
        scope.postMessage({ type: 'result', id, value });
    }

    private failure(id: number, message: string): void {
        scope.postMessage({ type: 'failure', id, message });
    }

    private unsupported(id: number, message: string): void {
        scope.postMessage({ type: 'unsupported', id, message });
    }
}

/**
 * Minimal declaration of the Worker global scope.
 *
 * tsconfig's lib only includes DOM, where `self` is Window, whose postMessage
 * requires a targetOrigin. Workers have no such thing, so this declares one
 * matching actual usage.
 */
const scope = self as unknown as {
    postMessage(message: WorkerResponse, transfer?: Transferable[]): void;
    onmessage: ((event: MessageEvent<WorkerRequest>) => void) | null;
};

/** Frame exit: the camera's listen callback, pushing frames back to the main thread. */
function postFrame(bitmap: ImageBitmap): void {
    scope.postMessage({ type: 'frame', bitmap }, [bitmap]);
}

/** The Worker's single runtime state; all messages are forwarded to it. */
const host = new WorkerHost();

scope.onmessage = (event: MessageEvent<WorkerRequest>) => {
    void host.handle(event.data);
};
