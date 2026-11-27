import { Prism, PrismUnsupportedError } from '../core/api/prism';
import { SCENE_CARDS } from './catalog';
import { loadScene } from './assemble';
import { OrbitControls } from './orbit-controls';
import { createWorkbench } from './workbench';

/** After the scene compiles, give the Worker this long to fill the frame before
 *  hiding the loading overlay. */
const LOADING_HOLD_MS = 1500;
/** Requested frame push interval: about 60 Hz, in practice bounded by the render frame rate. */
const FRAME_INTERVAL_MS = 1000 / 60;

/**
 * Render page shell: canvas, loading state and lifecycle.
 *
 * Scene assembly is left to `loadScene` and pointer interaction to
 * `OrbitControls`; this file only pieces the page together and guards
 * `disposed` — a worker built during initialization must be returned if the
 * user navigates away.
 *
 * It only knows `Prism` — frames arrive as ImageBitmaps pushed by the Worker
 * and the main thread paints them through bitmaprenderer without touching any
 * GPU resource.
 */
export class RenderView {
    private prism: Prism | null = null;
    private canvas: HTMLCanvasElement | null = null;
    private context: ImageBitmapRenderingContext | null = null;
    private stage: HTMLElement | null = null;
    private loading: HTMLElement | null = null;
    private resizeObserver: ResizeObserver | null = null;
    private stopListening: (() => void) | null = null;
    private disposed = false;
    private readonly workbenchAbort = new AbortController();
    private readonly controls = new OrbitControls();
    private readonly key: string;
    /** Invoked when the WebGPU API exists but no device is available; the caller
     *  swaps in the fallback page. */
    private readonly onUnsupported: () => void;

    constructor(key: string, onUnsupported: () => void) {
        this.key = key;
        this.onUnsupported = onUnsupported;
    }

    public async mount(container: HTMLElement): Promise<void> {
        const stage = document.createElement('div');
        stage.className = 'stage';
        this.stage = stage;
        container.appendChild(stage);

        const loading = this.buildLoading();
        this.loading = loading;
        stage.appendChild(loading);

        try {
            const canvas = document.createElement('canvas');
            this.canvas = canvas;
            const context = canvas.getContext('bitmaprenderer');
            if (!context) throw new Error('This browser does not support bitmaprenderer canvas');
            this.context = context;
            stage.insertBefore(canvas, loading);

            this.updateLoading('Initializing GPU...');
            const prism = await Prism.create();
            this.prism = prism;
            // Navigated away during initialization (double-clicked a card) — return the
            // Worker just created
            if (this.disposed) {
                prism.dispose();
                return;
            }

            // The Worker cannot measure the canvas, so the main thread measures and sends the size
            this.watchSize();

            // Entry back to the gallery. Browser Back works too, but visitors will not use it.
            const back = document.createElement('a');
            back.className = 'stage-back';
            back.href = '#/';
            back.textContent = '← All scenes';
            stage.appendChild(back);

            const card = SCENE_CARDS[this.key];
            if (!card) throw new Error(`No scene named "${this.key}"`);

            this.updateLoading('Loading scene...');
            await loadScene(prism, card);
            if (this.disposed) {
                prism.dispose();
                return;
            }

            // Frames come from this one stream: the Worker pushes an ImageBitmap per
            // frame, painted as-is
            const workbench = card.adjustEmission ? createWorkbench(prism, this.workbenchAbort.signal) : null;
            this.stopListening = prism.camera.listen(FRAME_INTERVAL_MS, (bitmap) => {
                if (!this.context) {
                    bitmap.close();
                    return;
                }
                this.context.transferFromImageBitmap(bitmap);
                if (workbench && !workbench.isConnected && !this.disposed) stage.appendChild(workbench);
            });
            this.controls.attach(canvas, prism.camera);

            // Compiled does not mean the frame is filled: the sample budget
            // starts at 4096 pixels/frame, so wait a moment before hiding the
            // loading overlay to avoid revealing a half-lit image.
            await new Promise<void>((resolve) => setTimeout(resolve, LOADING_HOLD_MS));
            loading.classList.add('hidden');
        } catch (err) {
            if (this.disposed) return;
            if (err instanceof PrismUnsupportedError) {
                this.onUnsupported();
                return;
            }
            this.updateLoading(err instanceof Error ? err.message : String(err), true);
            loading.classList.add('error');
        }
    }

    public unmount(): void {
        this.disposed = true;
        this.workbenchAbort.abort();
        this.stopListening?.();
        this.stopListening = null;
        this.resizeObserver?.disconnect();
        this.resizeObserver = null;
        this.controls.detach();
        this.prism?.dispose();
        this.prism = null;

        this.loading?.remove();
        this.loading = null;
        this.context = null;
        this.canvas = null;
        this.stage?.remove();
        this.stage = null;
    }

    /** Feed the canvas's physical pixel size to the camera (the Worker cannot read
     *  clientWidth / dpr). */
    private watchSize(): void {
        const canvas = this.canvas;
        if (!canvas) return;
        const apply = () => {
            const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
            const width = Math.max(1, Math.round(canvas.clientWidth * dpr));
            const height = Math.max(1, Math.round(canvas.clientHeight * dpr));
            // bitmaprenderer does not change the width/height attributes when a bitmap
            // is painted, so keep them in sync with the Worker
            canvas.width = width;
            canvas.height = height;
            this.prism?.camera.resize({ width, height });
        };
        this.resizeObserver = new ResizeObserver(apply);
        this.resizeObserver.observe(canvas);
        apply();
    }

    private buildLoading(): HTMLElement {
        const el = document.createElement('div');
        el.id = 'loading-overlay';
        el.innerHTML = `
      <div class="spinner"></div>
      <div class="msg" id="loading-msg">Initializing...</div>
    `;
        return el;
    }

    private updateLoading(msg: string, isError = false): void {
        const el = this.loading?.querySelector('#loading-msg');
        if (el) el.textContent = msg;
        if (isError) {
            this.loading?.querySelector('.spinner')?.classList.add('hidden');
        }
    }
}
