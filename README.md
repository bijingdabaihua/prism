# Prism

一个从零写的 WebGPU 路径追踪器。不依赖 three.js 或第三方 BVH。
渲染页的画面由浏览器实时计算；画廊封面是静态预览图。

![荒地](assets/gallery/vase-on-dirt.png)

## 看什么

首页是画廊，每个场景一张卡片，点进去就是一页实时渲染的画面。访客不需要调任何参数 ——
构图、光照、材质都在场景里定好了。

| | |
|---|---|
| **荒地** | 一只青花瓷罐搁在夕阳下的砂土上，右下角躺着一根枯枝。地面由 3 × 3 块带位移的地砖拼成，覆盖约 6.33 m 见方；地表起伏是真几何，会参与求交和阴影。当前场景暂时关闭景深，以降低焦外噪点。 |

渲染页左上角返回画廊。渲染跑在 Worker 里，主线程只负责显示；左键拖拽旋转、
滚轮缩放、右键拖拽平移。

## 跑起来

```bash
npm install
npm run dev        # http://localhost:5173
```

需要支持 WebGPU 的浏览器：**Chrome / Edge 113+**，或 **Safari 18+**。
Firefox 需要 `dom.webgpu.enabled`。不支持时会落到静态页 —— 渲染出来的图和
一段说明，至少不会白屏。

## 实现了什么

**求交**
- 两级 BVH（TLAS / BLAS），CPU 构建后扁平化成一维 buffer，GPU 上迭代遍历
- Möller–Trumbore 三角形求交，局部空间求交保持 `t` 在实例缩放时不变

**光照**
- NEE：环境图按亮度 CDF 重要性采样，阴影射线可见就直接取该方向的辐亮度
- MIS：BSDF 采样与光源采样用 power heuristic 混合
- 俄罗斯轮盘赌：第 4 次弹射后按存活概率终止
- 渐进累积：每帧累加 1 spp，相机和参数不动就不重置

**其它**
- 环境图重要性采样（2K EXR）；项目内解码器支持单部分 scanline EXR 的未压缩、ZIPS、ZIP 格式及 half / float RGB 通道
- glTF 2.0 导入（`.gltf` 外链资源、`.glb` 单文件都行），
  画面里那只瓷罐和枯枝就是这么进来的
- 薄透镜景深：光圈半径 + 对焦距离，都是米；主射线从光瞳圆盘上一点出发，
  指向焦平面上的同一个点；当前场景暂时关闭这一效果
- 位移贴图烘成真几何（细分 + 顶点位移），进 BVH 参与求交和阴影；演化后的材质
  不带位移字段，着色器不知道它存在
- 材质贴图走 `texture_2d_array`：漫反射 / 法线 / 粗糙度 / 环境光遮蔽四个数组，
  各自带完整 mip 链；glTF 的 metallicRoughness 打包图原样进粗糙度数组，
  G 通道读粗糙度、B 通道读金属度（Poly Haven 的 arm 图同一套排列）
- 每帧只采屏幕的一部分像素，dispatch 量由 GPU 实测耗时自适应；顺序表保证
  每一轮正好覆盖每个像素一次，所以累加平均的分母不需要逐像素存储

## 没做什么

写清楚边界比列功能表有用：

- **glTF 只吃三角形和四种贴图**。Draco / meshopt 压缩、蒙皮、动画、morph 都不支持
  （Draco 会明确报错）
- **不支持透明 / alpha 切边**。树叶、纱帘这类靠 alpha 裁形的模型会渲染成实心片
  （材质 IR 里连 alpha 字段都不留，要做得从 glTF 导入一路加回来）
- **没有透射 / 玻璃**。这是路径追踪最能打的画面，排在下一个
- **BVH 是 16 桶 × 三条轴的分桶 SAH**；深度超过 40 层时退回中位数切分，
  给 shader 的 64 层遍历栈留余量。叶子最多 2 个图元
- **没有自发光 / 显式光源类型**。照明只有环境图，模型自带的 emissive 材质会被忽略

## 结构

依赖方向单向，越往下越不认识上面的东西：

```
src/app/               路由、画廊、渲染页、降级页、场景清单和装配
src/core/api/          Prism 门面、主线程 RPC 和三个句柄
src/core/worker/       Worker 入口及渲染实现
  camera/              帧循环、路径追踪、显示
  background/          环境图和亮度 CDF
  model/               glTF、材质、图像解码、网格和 BVH
src/core/math/         向量、矩阵、颜色
```

光追完全跑在 Worker 里：`Prism.create()` 起一个 Worker、要一个独立 `GPUDevice`，
主线程不碰 GPU，只通过 `camera.listen()` 接收 `ImageBitmap`。`src/core/worker/worker.ts`
负责装配相机、背景和模型；场景定义与加载顺序留在 app 层。

约 8800 行 TypeScript + WGSL + CSS。

### 加一个场景

在 `src/app/catalog.ts` 的 `SCENE_CARDS` 中添加机位、环境图、几何清单和卡片信息；
按需扩展 `src/app/assemble.ts` 中的装配逻辑，再放一张 `assets/gallery/<key>.png` 当封面。
路由、画廊、降级页都会自动带上它。

## 场景从哪来

模型、贴图、环境图都来自 **Poly Haven**（CC0）：模型和材质下载后放进
`assets/model/`、`assets/textures/`，环境图（EXR）放进 `assets/env/`。
模型摆放和取景写在 `src/app/catalog.ts`；渲染器不认识具体场景。

地面这类大面积几何由 `src/app/ground.ts` 生成；位移贴图挂在材质上，网格在编译时细分成真几何，
顶点抬起、法线按高度场重算，那张图进 GPU 之前就被消耗掉了。

## 许可

MIT
