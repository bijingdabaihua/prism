import type { ModelHandle, Prism } from '../core/api/prism';
import type { SceneCard, SceneGeometry } from './catalog';
import { buildGlassStudy } from './glass-study';
import { buildTiledGround } from './ground';

/**
 * 装一个场景：机位 → 环境图 + 朝向 → 几何。
 *
 * 顺序是场景语义的一部分：机位先定构图，环境图决定光从哪里来，几何最后
 * 落地。Prism 侧是三个独立句柄，没有「整场景」接口；这个函数只是把业务层
 * 唯一那套顺序收在一处，渲染页调用它。
 *
 * 清单是纯数据（`catalog.ts`），这里才开始碰导入器和网络 —— 画廊读卡片
 * 不会把 glTF / blend / 解码器拉进它的加载路径。
 */
export async function loadScene(prism: Prism, card: SceneCard): Promise<void> {
    prism.camera.set(card.setup.camera);

    // 环境图以 Blob 交给 Worker：主线程只取字节，解码仍留在 Worker。
    // setImage 从原图方向建好；场景需要别的朝向时再由 setRotation 转。
    await prism.background.setImage(await fetchBlob(card.setup.env));
    await prism.background.setRotation(card.setup.envRotation ?? 0);

    await loadGeometry(prism.model, card.geometry);
}

/**
 * 按清单顺序装几何。程序化几何走实例入口：
 * - `tiled-ground`：主线程造好一批 Instance，一次请求推给 Worker；
 * - `gltf`：只递资产包字节，解析、贴图解码和摆位都在 Worker 里做。
 */
async function loadGeometry(model: ModelHandle, geometry: readonly SceneGeometry[]): Promise<void> {
    for (const object of geometry) {
        if (object.type === 'tiled-ground') {
            const material = await fetchBlob(object.material);
            await model.loadInstances(await buildTiledGround(material, object));
        } else if (object.type === 'glass-study') {
            await model.loadInstances(buildGlassStudy());
        } else {
            await model.loadGltf(await fetchBlob(object.url), object.placement);
        }
    }
}

/** URL → Blob。Worker 里的解析以 Blob 为入口，取字节是调用方的事。 */
async function fetchBlob(url: string): Promise<Blob> {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`Failed to load ${url} (${resp.status})`);
    return resp.blob();
}
