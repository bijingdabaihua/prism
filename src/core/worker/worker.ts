/**
 * 光追 Worker —— 一个 Prism 实例一个，独占一个 GPUDevice。
 *
 * 整个渲染管线（Camera / Model / Background）都跑在这个线程里，画布是
 * Worker 自己建的 OffscreenCanvas，主线程从头到尾不碰 GPU。画面通过
 * `camera.listen()` 以 ImageBitmap 推回主线程 —— 每间隔最多一帧。
 *
 * 这个文件同时是线程入口和组装根：运行态全部挂在 `WorkerHost` 一个实例上，
 * init 时建设备、把模型 / 环境图 / 相机三个模块装配起来。消息语义直接沉到
 * 模块 —— 机位命令和画面监听委托 camera，环境图委托 background，几何委托
 * model.add；顶层只留串行队列和回执。
 *
 * 渲染目标、帧循环都归相机 —— 主线程那边也只有 camera 一个句柄管尺寸和出图，
 * 内外同一套分工。这里只负责把三者拼起来并持有生命周期。
 *
 * Worker 不认识任何具体场景：几何要么来自 glTF 字节（worker 里解码），要么
 * 是主线程构造好 IR 推过来的裸数据（`Instance.decode` 复活）。
 *
 * 销毁不走消息：Prism.dispose 直接 terminate Worker，线程没了设备自然释放，
 * 所以没有对称的 teardown 路径。
 */
// 纯类型导入：消息协议和门面同文件，编译后这条边会完全擦除。
import type { CameraCommand, WorkerRequest, WorkerResponse } from '../api/prism';
import { createDevice, WebGPUUnavailableError } from './camera/context';
import { Camera } from './camera/camera';
import { Background } from './background/background';
import { Model } from './model/model';
import { loadImage } from './model/image';
import { Instance } from './model/instance';

/**
 * Worker 的运行态：三个模块 + 一条串行队列。
 *
 * 消息里凡是「模块自己会」的（机位、监听、环境图、几何）都委托下去；
 * 顶层只负责把异步操作排好队、把结果回执出去。
 */
class WorkerHost {
    /** 渲染相机：init 时建，持有渲染目标和帧循环。Prism 生命周期内唯一一个。 */
    private camera: Camera | null = null;
    /** 环境光照：init 时和相机一起装配。 */
    private background: Background | null = null;
    /** 场景几何：BVH / 材质表，所有加几何的落点。 */
    private model: Model | null = null;
    /** 异步任务的串行链。 */
    private taskChain: Promise<void> = Promise.resolve();

    /** 消息入口：写命令直接做，有返回值的进队列。 */
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
                this.enqueue(() => this.loadGltf(request.id, request.source));
                return;
            case 'model/loadInstance':
                this.enqueue(() => this.loadInstance(request.id, request.instance));
                return;
        }
    }

    /**
     * 建设备 + 装配三个模块；已经建过就直接返回。
     *
     * 一个 Prism 实例只有一套引擎，只能由 init 装配。场景的机位、环境图、
     * 几何随后分别送进来。相机最后建：构造时要拿 model 和 background 的引用。
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
     * 异步任务的串行化。
     *
     * `loadImage` / `Instance.buildByGltf` 都是异步的：连着调两次，先发的那次
     * 可能后回来，把后发的那张顶掉。排队之后「最后发起的那次」一定最后落地。
     */
    private enqueue(task: () => Promise<void>): void {
        this.taskChain = this.taskChain.then(task);
    }

    /**
     * 换环境图：Blob → 解码 → 上传。
     *
     * 主线程只负责提供字节；格式识别和解码都在 Worker 里。只换图，不动
     * 朝向；朝向由 setRotation 单独负责。
     *
     * 引擎已经在跑时也可以换：Background.version() 变化会让 PathTracePass 在
     * 下一帧重建 bind group 并清零累积。解码留在顶层（loadImage 是 ir 的
     * 入口），解码后的 Image 才交给 background。
     */
    private async setImage(id: number, blob: Blob): Promise<void> {
        try {
            if (!this.background) throw new Error('Prism 还没初始化');
            this.background.setImage(await loadImage(blob));
            this.reply(id, undefined);
        } catch (err) {
            this.failure(id, err instanceof Error ? err.message : String(err));
        }
    }

    /**
     * 旋转环境图：不重新解码，拿 Background 留着的原图按新角度重建纹理和 CDF。
     *
     * 需要先 setImage。角度是绕 +y 的弧度，和场景数据里的 `envRotation` 同一套
     * 定义。累积会自动清零 —— Background.version() 变了，下一帧重建 bind group。
     */
    private async setRotation(id: number, rotation: number): Promise<void> {
        try {
            if (!this.background) throw new Error('Prism 还没初始化');
            this.background.setRotation(rotation);
            this.reply(id, undefined);
        } catch (err) {
            this.failure(id, err instanceof Error ? err.message : String(err));
        }
    }

    /**
     * 加载一个 glTF 文件里的全部 Instance，追加到当前场景。
     *
     * 与场景配方入口一样，最终都落到 `model.add`。模型入口都是追加语义，
     * 外部可以一个一个调用，把多个来源拼进同一个场景。
     */
    private async loadGltf(id: number, source: Blob): Promise<void> {
        try {
            if (!this.model) throw new Error('Prism 还没初始化');
            this.model.add(await Instance.buildByGltf(source));
            this.reply(id, undefined);
        } catch (err) {
            this.failure(id, err instanceof Error ? err.message : String(err));
        }
    }

    /**
     * 追加一个外部构造好的 Instance。
     *
     * 主线程用 `Instance.encode` 把 IR 编成裸数据发过来（类实例过不了线程
     * 边界），这里 `Instance.decode` 校验并复活成 IR 类实例，再走同一条
     * `model.add`。典型来源是程序化几何：地面这类东西在 app 侧造好，
     * 不从 glTF 走。
     */
    private async loadInstance(id: number, wire: unknown): Promise<void> {
        try {
            if (!this.model) throw new Error('Prism 还没初始化');
            this.model.add([Instance.decode(wire)]);
            this.reply(id, undefined);
        } catch (err) {
            this.failure(id, err instanceof Error ? err.message : String(err));
        }
    }

    /**
     * 机位命令：直接落到 Camera 的方法上。
     *
     * resize 改的是渲染目标的尺寸，不是机位参数 —— 相机持有渲染目标和帧循环，
     * 这里只把尺寸交给它。引擎还没建好时命令直接丢弃：句柄侧是同步不抛的，
     * 而 `Prism.create()` 等到 init 回执才把句柄交出去，resize 不会先到。
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
 * Worker 全局作用域的最小声明。
 *
 * tsconfig 的 lib 只有 DOM，`self` 是 Window —— 那上面的 postMessage 要求
 * 传 targetOrigin。Worker 里没有这一套，所以按实际用法声明一份。
 */
const scope = self as unknown as {
    postMessage(message: WorkerResponse, transfer?: Transferable[]): void;
    onmessage: ((event: MessageEvent<WorkerRequest>) => void) | null;
};

/** 画面出口：相机的 listen 回调，把帧推回主线程。 */
function postFrame(bitmap: ImageBitmap): void {
    scope.postMessage({ type: 'frame', bitmap }, [bitmap]);
}

/** Worker 唯一的运行态；消息全部转给它。 */
const host = new WorkerHost();

scope.onmessage = (event: MessageEvent<WorkerRequest>) => {
    void host.handle(event.data);
};
