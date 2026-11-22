import { Prism, PrismUnsupportedError } from '../core/api/prism';
import { SCENE_CARDS } from './catalog';
import { loadScene } from './assemble';
import { OrbitControls } from './orbit-controls';

/** 场景编译完之后，再给 Worker 这么久把画面铺满，然后撤 loading。 */
const LOADING_HOLD_MS = 1500;
/** 请求的画面推送间隔：约 60 Hz，实际受渲染帧率限制。 */
const FRAME_INTERVAL_MS = 1000 / 60;

/**
 * 渲染页外壳：画布、加载状态和生命周期。
 *
 * 场景怎么装交给 `loadScene`，指针怎么转交给 `OrbitControls`；这里只把
 * 页面元素拼起来，再守住 disposed —— 初始化期间被切走时要还回 Worker。
 *
 * 这里只认识 `Prism` —— 画面是 Worker 推回来的 ImageBitmap，主线程用
 * bitmaprenderer 直接贴上，不碰任何 GPU 资源。
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
    private readonly controls = new OrbitControls();
    private readonly key: string;
    /** 有 WebGPU API、但拿不到设备时交给外层换成降级页。 */
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
            if (!context) throw new Error('浏览器不支持 bitmaprenderer canvas');
            this.context = context;
            stage.insertBefore(canvas, loading);

            this.updateLoading('Initializing GPU...');
            const prism = await Prism.create();
            this.prism = prism;
            // 初始化期间被切走（连点两下卡片）—— 把刚建好的 Worker 还回去
            if (this.disposed) {
                prism.dispose();
                return;
            }

            // Worker 里的画布尺寸只能由主线程量好送进去
            this.watchSize();

            // 返回画廊的入口。浏览器后退也行，但访客不会去按。
            const back = document.createElement('a');
            back.className = 'stage-back';
            back.href = '#/';
            back.textContent = '← 全部场景';
            stage.appendChild(back);

            const card = SCENE_CARDS[this.key];
            if (!card) throw new Error(`没有场景 "${this.key}"`);

            this.updateLoading('Loading scene...');
            await loadScene(prism, card);
            if (this.disposed) {
                prism.dispose();
                return;
            }

            // 画面从这一条流里来：Worker 每帧推一张 ImageBitmap，贴上即显示
            this.stopListening = prism.camera.listen(FRAME_INTERVAL_MS, (bitmap) => {
                if (!this.context) {
                    bitmap.close();
                    return;
                }
                this.context.transferFromImageBitmap(bitmap);
            });
            this.controls.attach(canvas, prism.camera);

            // 编译完成不等于画面已经铺满：采样预算从 4096 像素/帧起步，
            // 等一小会儿再撤 loading，避免露出半张没光的画面。
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

    /** 把画布的物理像素尺寸喂给相机（Worker 里读不到 clientWidth / dpr）。 */
    private watchSize(): void {
        const canvas = this.canvas;
        if (!canvas) return;
        const apply = () => {
            const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
            const width = Math.max(1, Math.round(canvas.clientWidth * dpr));
            const height = Math.max(1, Math.round(canvas.clientHeight * dpr));
            // bitmaprenderer 不会因为贴位图而改宽高属性，这里跟 Worker 保持一致
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
