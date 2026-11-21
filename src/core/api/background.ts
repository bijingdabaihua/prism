/**
 * 主线程环境光句柄：主线程只提供字节，格式识别 / 解码 / 上传都在 Worker。
 * 文件分两层：契约层（全部导出）+ 实现层（全部不导出），规范见 CODE_STYLE.md。
 */

// ══════════════════════════ 契约层（导出） ══════════════════════════

/**
 * 构造主线程环境光句柄；返回的是契约接口，实现类不导出。
 * Blob 交给 Worker 去解码、上传 —— 主线程只负责提供字节。
 * 此处只有声明，赋值在实现层。
 */
export let createBackground: (rpc: BackgroundRpc) => BackgroundHandle;

/**
 * 环境光句柄 —— 环境图的远程入口。
 */
export interface BackgroundHandle {
    /**
     * 设置环境图（Blob）。Worker 负责格式识别、解码和上传；每次换图都从
     * 原图方向开始，需要朝向就调用 `setRotation`。Promise 解决时纹理和
     * CDF 已经就位。
     */
    setImage(blob: Blob): Promise<void>;
    /**
     * 旋转当前环境图（绕 +y 的弧度）。需要先 setImage；转完纹理和 CDF 一起
     * 重建，Promise 解决时已经生效。
     */
    setRotation(rotation: number): Promise<void>;
}

/**
 * 环境光句柄依赖的最小 RPC 面：一次带返回值的请求。
 * `build` 产出要发给 Worker 的请求消息（形状见本层 BackgroundRequest），
 * 对契约层不透明。
 */
export interface BackgroundRpc {
    request<T>(build: (id: number) => BackgroundRequest): Promise<T>;
}

/**
 * 环境光相关的 Worker 消息报文。
 * 报文形状不是对外契约，但 prism 的消息分发联合需要枚举它，因此导出、归本层。
 */
export type BackgroundRequest =
    | { type: 'background/setImage'; id: number; blob: Blob }
    | { type: 'background/setRotation'; id: number; rotation: number };

// ══════════════════════════ 实现层（不导出） ══════════════════════════

/** 环境光句柄的主线程实现。 */
class RpcBackground implements BackgroundHandle {
    constructor(private readonly rpc: BackgroundRpc) {}

    public setImage(blob: Blob): Promise<void> {
        return this.rpc.request<void>((id) => ({ type: 'background/setImage', id, blob }));
    }

    public setRotation(rotation: number): Promise<void> {
        return this.rpc.request<void>((id) => ({ type: 'background/setRotation', id, rotation }));
    }
}

/** 契约层构造入口的赋值。 */
createBackground = (rpc) => new RpcBackground(rpc);
