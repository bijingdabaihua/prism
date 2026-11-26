/**
 * 主线程模型句柄：契约 + 实现。
 *
 * 文件解码、位移、BVH 全在 Worker 里；这边只递资产字节。创建入口是
 * `createModel(rpc)`，实现类不导出。
 */

import { Instance } from '../worker/model/instance';
import type { WireInstance } from '../worker/model/instance';

/**
 * 公开契约
 *
 * 以下类型和接口是对外 API。修改前先确认调用方影响。
 */

/** 一组实例的摆放，语义与场景构建器里的 `placeInstances` 一致。 */
export interface InstancePlacement {
    position: { x: number; y: number; z: number };
    /** 绕 X 轴，弧度。先于 rotationY 应用。 */
    rotationX?: number;
    /** 绕 Y 轴，弧度。 */
    rotationY?: number;
    scale?: number;
}

/**
 * 模型句柄 —— 场景几何的远程入口。
 *
 * 两个入口，语义都是追加到当前场景，不是替换：
 * - `loadGltf` 递资产包字节，解析、摆位、解码全在 Worker；
 * - `loadInstances` 递 app 在主线程构造好的一组 IR Instance（如程序化地面），
 *   句柄侧编码成线上形态，Worker 解码复活后再编译。
 */
export interface ModelHandle {
    /**
     * 追加一个 glTF 资产包里的全部 Instance。
     *
     * `source` 是资产包字节：自包含 `.glb`、内嵌资源的 `.gltf`，或带
     * `.bin` / 贴图的 `.zip`。URL 也由调用方自己 fetch 成 Blob 再传进来 ——
     * 主线程只递字节，解析、解包和贴图解码都在 Worker。
     *
     * `placement` 给整组实例再加一层摆放（语义同 `placeInstances`）。放在
     * Worker 里做，主线程就不必为了摆一个模型先把它整个解析一遍。
     */
    loadGltf(source: Blob, placement?: InstancePlacement): Promise<void>;
    /**
     * 追加一组外部构造好的 Instance。
     *
     * 典型用途是程序化几何（`MeshBuilder` 铺的地面等）：几何和材质在主线程
     * 造好，跨线程时由 `Instance.encode` 编成裸数据，Worker 侧 `Instance.decode`
     * 复活成 IR 类实例。贴图字段须是解码后的 `Image`（IR 里不留文件路径）。
     *
     * 一批一次请求：地面铺 9 块砖就是 9 个 Instance，但只编译一次。
     */
    loadInstances(instances: readonly Instance[]): Promise<void>;
    /** 设置全场景发光几何的辐亮度倍率，默认 1。 */
    setEmissionScale(scale: number): Promise<void>;
}

/**
 * 实现
 *
 * 以下内容不属于公开契约：消息形状、RPC 依赖和句柄实现。
 */

/** 模型相关的 Worker 消息。 */
export type ModelRequest =
    | { type: 'model/loadGltf'; id: number; source: Blob; placement?: InstancePlacement }
    | { type: 'model/loadInstances'; id: number; instances: WireInstance[] }
    | { type: 'model/setEmissionScale'; id: number; scale: number };

/** 模型句柄需要的最小 RPC 面：一次带返回值的请求。 */
export interface ModelRpc {
    request<T>(build: (id: number) => ModelRequest): Promise<T>;
}

/** 模型句柄的主线程实现：把两个加载入口翻成带 id 的请求。 */
class RpcModel implements ModelHandle {
    constructor(private readonly rpc: ModelRpc) {}

    public loadGltf(source: Blob, placement?: InstancePlacement): Promise<void> {
        return this.rpc.request<void>((id) => ({ type: 'model/loadGltf', id, source, placement }));
    }

    public setEmissionScale(scale: number): Promise<void> {
        return this.rpc.request<void>((id) => ({ type: 'model/setEmissionScale', id, scale }));
    }

    public loadInstances(instances: readonly Instance[]): Promise<void> {
        return this.rpc.request<void>((id) => ({
            type: 'model/loadInstances',
            id,
            instances: instances.map((instance) => Instance.encode(instance)),
        }));
    }
}

/** 主线程模型句柄的工厂：实现类不导出。 */
export function createModel(rpc: ModelRpc): ModelHandle {
    return new RpcModel(rpc);
}
