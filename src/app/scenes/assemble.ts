import type { ModelHandle } from '../../core/api/prism';
import { Instance, placeInstances } from '../../core/worker/model/instance';
import { buildTiledGround, loadGroundMaterial } from './ground';
import type { SceneGeometry } from './catalog';

/**
 * 场景装配：按清单顺序，把几何一个一个装进渲染器。
 *
 * 两种入口对应清单里的两种条目：
 * - `tiled-ground`：程序化地面，主线程用 `MeshBuilder` / `.blend` 材质造好
 *   Instance，逐块 `loadInstance` 推给 Worker；
 * - `gltf`：资产包字节。不带摆放的整包走 `loadGltf`（解析在 Worker，主线程
 *   不背重活）；带摆放的外部解析、摆位后同样走 `loadInstance`。
 *
 * 清单是纯数据（`catalog.ts`），这里才开始碰导入器 —— 画廊读卡片不会把
 * glTF / blend / 解码器拉进它的加载路径。
 */
export async function loadGeometry(model: ModelHandle, geometry: readonly SceneGeometry[]): Promise<void> {
    for (const object of geometry) {
        if (object.type === 'tiled-ground') {
            const material = await loadGroundMaterial(object.material, object.tile);
            const tiles = buildTiledGround(material, object.tile, object.x0, object.x1, object.z0, object.z1);
            for (const tile of tiles) {
                await model.loadInstance(tile);
            }
        } else {
            const blob = await fetchBlob(object.url);
            if (!object.placement) {
                await model.loadGltf(blob);
                continue;
            }
            const instances = await Instance.buildByGltf(blob);
            placeInstances(instances, object.placement);
            for (const instance of instances) {
                await model.loadInstance(instance);
            }
        }
    }
}

/** URL → Blob。Worker 里的解析以 Blob 为入口，取字节是调用方的事。 */
export async function fetchBlob(url: string): Promise<Blob> {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`Failed to load ${url} (${resp.status})`);
    return resp.blob();
}
