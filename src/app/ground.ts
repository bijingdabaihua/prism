import type { Vec3 } from '../core/math/types';
import { Instance } from '../core/worker/model/instance';
import { PBRMaterial } from '../core/worker/model/material/material';
import { MeshBuilder } from '../core/worker/model/mesh';
import type { TiledGroundGeometry } from './catalog';

/**
 * 砂土地面 —— 荒地场景的地表。
 *
 * 一张平面铺不出这个场景：低机位下画面顶端几乎贴着地平线，一片 1 m 见方的
 * 板 2 m 外就被看见边了。所以这里是「一块地砖 + 实例化」，见 `buildTiledGround`。
 *
 * 材质全在压缩包里（`.blend` + 贴图目录）—— 四张贴图、色彩空间、法线强度、
 * 位移的 Midlevel / Scale 都是作者填的。这里只给一个文件读不出来的数：一格
 * 贴图多大（Poly Haven 把它放在网页 API 的 `dimensions` 里，不在 .blend 中），
 * 它来自配方。
 *
 * 位移是材质的属性，不是几何的属性：这里只铺一张平的四边形，细分和抬起
 * 由 `Mesh.evolve` 在编译时做。
 */

/** 水平四边形，绕序保证法线朝 +y（和 addQuad 的约定一致）。 */
function groundQuad(x0: number, x1: number, z0: number, z1: number): [Vec3, Vec3, Vec3, Vec3] {
    return [
        { x: x0, y: 0, z: z1 },
        { x: x1, y: 0, z: z1 },
        { x: x1, y: 0, z: z0 },
        { x: x0, y: 0, z: z0 },
    ];
}

/**
 * 用一块地砖铺出的大地面：一块 mesh、很多实例、共用一棵 BLAS。
 *
 * 为什么切开：4 mm 一格的大地面没法一次切出来。42 m 见方是 2 亿个三角形，
 * 光顶点就 3 GB。切成小砖之后，几何只算一份（荒地的一块约 52 万三角形），
 * 铺多少块都只多一张实例表 —— `blasCache` 按 mesh 缓存，实例之间共享同一棵树。
 *
 * 为什么砖边长必须等于贴图周期：位移是从贴图上采样的，贴图每 `tile` 米
 * 重复一次。砖边长等于这个周期时，相邻砖边界上的 UV 正好从 `tile` 接回 0，
 * 两边采到的高度相同，位移和图案都自己接上 —— 铺出来的结果和「一整块大地」
 * 逐顶点一样。荒地用 2.11 m；换成整数 2 m，缝上就错开 0.11 m，又是一条线。
 *
 * 砖中心落在 (i·tile, 0, j·tile) 上，所以原点是砖心，罐子正好踩在中间那块上。
 * 范围按**砖心**取舍：中心落在 [x0, x1] 里的砖才有。两段地面用相邻的
 * x0 / x1 调，就正好在一条砖缝上接上，既不重叠也不留缝。
 *
 * 材质压缩包由调用方取好再传进来：这个文件只管造几何，不做网络。
 */
export async function buildTiledGround(
    materialZip: Blob,
    recipe: TiledGroundGeometry,
): Promise<Instance[]> {
    // tile 同时决定 UV 缩放和砖边长；.blend 里读不出这个数，构造后矫正
    const material = (await PBRMaterial.buildByBlend(materialZip)).retile(recipe.tile);
    const h = recipe.tile * 0.5;
    const builder = new MeshBuilder();
    builder.addQuad(groundQuad(-h, h, -h, h), 0);
    const mesh = builder.build([material], 'ground-tile');

    const instances: Instance[] = [];
    for (let i = Math.ceil(recipe.x0 / recipe.tile); i <= Math.floor(recipe.x1 / recipe.tile); i++) {
        for (let j = Math.ceil(recipe.z0 / recipe.tile); j <= Math.floor(recipe.z1 / recipe.tile); j++) {
            const m = new Float32Array(16);
            m[0] = m[5] = m[10] = m[15] = 1;
            m[12] = i * recipe.tile;
            m[14] = j * recipe.tile;
            instances.push(new Instance(mesh, m));
        }
    }
    return instances;
}
