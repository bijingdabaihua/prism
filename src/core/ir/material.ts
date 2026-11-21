import type { Color } from '../math/types';
import type { Image } from './image';
import { parseBlendMaterial } from './blend';

/**
 * 渲染器内部统一的材质表达。
 *
 * glTF 的材质在 ir/instance.ts 的导入器里映射成构造参数；.blend 的导入器在
 * ir/blend.ts，由 `buildByBlend` 驱动。映射是有损的，损失多少由各导入器自己说明。
 * GPU 打包在 engine/model/material/packing.ts。
 *
 * 文件分两层：契约层（全部导出）+ 实现层（全部不导出），规范见 CODE_STYLE.md。
 */

// ══════════════════════════ 契约层（导出） ══════════════════════════

/**
 * PBRMaterial 的类绑定：构造一个材质；静态工厂 `buildByBlend` 从 `.blend` 的
 * 材质节点图构造（解析器在 ir/blend.ts）。
 * 此处只有声明，赋值在实现层。
 */
export let PBRMaterial: {
    new (options?: PBRMaterialOptions): PBRMaterial;
    buildByBlend(url: string, meters?: number): Promise<PBRMaterial>;
};

/**
 * 渲染器内部统一的材质表达。
 *
 * 贴图字段是解码后的 Image，不是 URL —— IR 里不留文件路径，导入器在场景加载时
 * 就把图解开，编译层拿到的全是内存里的像素。
 */
export interface PBRMaterial {
    readonly baseColor: Color;
    readonly metallic: number;
    readonly roughness: number;
    readonly baseColorTexture?: Image;
    /**
     * glTF 的 metallic-roughness 打包图：G = 粗糙度，B = 金属度，R = 环境光遮蔽。
     * Poly Haven 的 arm 图（AO / Rough / Metal）是同一套通道排列，可以直接用。
     *
     * metallic / roughness 两个标量在这条路径上是**乘数**（glTF 的
     * metallicFactor / roughnessFactor），不是覆盖值 —— 金属度终于能随贴图变化了。
     */
    readonly metallicRoughnessTexture?: Image;
    /**
     * 单通道粗糙度图（Poly Haven 的 Rough 图）。
     *
     * 和 metallicRoughnessTexture 同时设置时，粗糙度以后者为准、这条路径只覆盖
     * 粗糙度（金属度仍从打包图取）；同一张贴图不必按两种 kind 注册两次。
     */
    readonly roughnessTexture?: Image;
    readonly normalTexture?: Image;
    readonly normalScale: number;
    readonly occlusionTexture?: Image;
    readonly occlusionStrength: number;
    /**
     * 位移贴图：灰度高度图，和 albedo 一样按它自己的 `meters` 重复。
     *
     * 有它的材质，三角形在进渲染前会被细分、按图抬起（见 `ir/mesh.ts` 的
     * `Mesh.evolve`）。只看三角形自己的 UV，不看位置。演化之后这张图就被
     * **消耗**掉了 —— 上 GPU 的材质里不再有它，着色器也从来不知道它存在。
     *
     * 有图就必须给 `displacementMid` 和 `displacementScale`。格距不用给 ——
     * 由图的一格几米和像素宽度算出来（一个 texel 一格）。
     */
    readonly displacement?: Image;
    /**
     * 高度零点：图上这个灰度值当作「不抬不陷」，对应 Blender Displacement
     * 节点的 Midlevel。扫描图存的不是绝对高度，零点只能由材质作者定。
     */
    readonly displacementMid?: number;
    /**
     * 高度刻度（米）：灰度每差 1.0 位移这么多，对应 Blender Displacement
     * 节点的 Scale。位移量 = `(高度 - displacementMid) × displacementScale`。
     */
    readonly displacementScale?: number;

    /**
     * 去掉位移贴图和相关刻度，返回可以直接上 GPU 的材质。
     *
     * `Mesh.evolve` 把位移烘成几何时调用：位移进 GPU 之前必须消失，着色器
     * 不知道它存在，材质 SSBO 里也没有这一格。没有位移时原样返回（`===` 自己）。
     */
    withoutDisplacement(): PBRMaterial;
}

/**
 * 构造参数。省略的字段取默认值：0.85 灰、非金属、粗糙度 0.5、
 * 法线强度 1、遮蔽强度 1。
 */
export interface PBRMaterialOptions {
    baseColor?: Color;
    metallic?: number;
    roughness?: number;
    baseColorTexture?: Image;
    metallicRoughnessTexture?: Image;
    roughnessTexture?: Image;
    normalTexture?: Image;
    normalScale?: number;
    occlusionTexture?: Image;
    occlusionStrength?: number;
    displacement?: Image;
    displacementMid?: number;
    displacementScale?: number;
}

/** 材质贴图的四个通道：决定进哪个 texture_2d_array，也决定要不要 sRGB 解码。 */
export type TextureKind = 'albedo' | 'normal' | 'roughness' | 'occlusion';

// ══════════════════════════ 实现层（不导出） ══════════════════════════

class PBRMaterialImpl implements PBRMaterial {
    public static async buildByBlend(url: string, meters?: number): Promise<PBRMaterial> {
        return new PBRMaterialImpl(await parseBlendMaterial(url, meters));
    }

    public readonly baseColor: Color;
    public readonly metallic: number;
    public readonly roughness: number;
    public readonly baseColorTexture?: Image;
    public readonly metallicRoughnessTexture?: Image;
    public readonly roughnessTexture?: Image;
    public readonly normalTexture?: Image;
    public readonly normalScale: number;
    public readonly occlusionTexture?: Image;
    public readonly occlusionStrength: number;
    public readonly displacement?: Image;
    public readonly displacementMid?: number;
    public readonly displacementScale?: number;

    constructor(options: PBRMaterialOptions = {}) {
        this.baseColor = options.baseColor ?? { r: 0.85, g: 0.85, b: 0.85, a: 1 };
        this.metallic = options.metallic ?? 0;
        this.roughness = options.roughness ?? 0.5;
        this.baseColorTexture = options.baseColorTexture;
        this.metallicRoughnessTexture = options.metallicRoughnessTexture;
        this.roughnessTexture = options.roughnessTexture;
        this.normalTexture = options.normalTexture;
        this.normalScale = options.normalScale ?? 1;
        this.occlusionTexture = options.occlusionTexture;
        this.occlusionStrength = options.occlusionStrength ?? 1;
        this.displacement = options.displacement;
        this.displacementMid = options.displacementMid;
        this.displacementScale = options.displacementScale;
    }

    public withoutDisplacement(): PBRMaterial {
        if (!this.displacement) return this;
        const { displacement: _image, displacementMid: _mid, displacementScale: _scale, ...rest } = this;
        return new PBRMaterialImpl(rest);
    }
}

/** 契约层类绑定的赋值。 */
PBRMaterial = PBRMaterialImpl;
