# Prism

从零写的 WebGPU 路径追踪器。前台是零操作的画廊；v1 的目标是加一层命令接口，让人（工作台页面）和 AI（MCP）都能调用。
不使用 three.js 渲染管线或第三方 BVH；渲染页画面在浏览器里计算，画廊封面是静态预览图。

- 在线 demo：https://bijingdabaihua.github.io/prism/
- 技术栈：TypeScript、WGSL、WebGPU、Web Workers、Vite、glTF 2.0、EXR
- 资产：Poly Haven（CC0），新增资产保持同样的许可
## 当前阶段与优先级（8 周，从 2026 年 12 月起）

这个项目首先是求职作品。判断一个改动值不值得做，看它能否在**效果、速度（要有数字）、可用性**里至少提升一项。

1. **第 1–2 周，定接口**：写出场景 JSON 和命令集的类型，在 `src/core/api/prism.ts` 门面和消息协议上扩展；把现有的场景清单接入新接口；用画廊场景搭参考图回归测试。
2. **第 3–4 周，补渲染短板**：透射/玻璃 BSDF；自发光材质，并纳入 NEE；画廊新增一个玻璃场景。
3. **第 5–6 周，工作台与 AI**：工作台页面（载入模型、材质库、环境预设、导出 PNG、性能面板）；MCP server 加页面桥接。
4. **第 7–8 周，包装**：README 第一屏放性能数字和 AI 录屏，写一页技术说明。

**v1 不做**：场景编辑器、动画/蒙皮/Draco/meshopt、体积渲染、光谱渲染、账号/云端、npm 包。
**进度不够时**：先砍材质库和预设的数量，玻璃和 AI 这两块不砍。

## 架构

```
src/
  app/          业务层：路由、画廊、渲染页、降级页、场景清单和装配
  core/api/     Prism 门面：主线程 RPC、消息协议、句柄
  core/worker/  Worker 入口、相机、环境光、模型、图像解码和路径追踪
  core/math/    向量、矩阵、颜色，app 和 core 共用
```

### 硬规则（不要违反）

- **依赖单向向下**：app → core → math。core 不 import app，math 不 import 任何上层。
- **主线程不碰 GPU**：所有 WebGPU 调用都在 Worker 里。主线程只通过 `prism.ts` 门面通信，用 `camera.listen()` 接收 `ImageBitmap` 帧。
- **`src/core/worker/worker.ts` 装配渲染模块**；场景清单保留在 app 层，不要让 core 依赖 app。
- **新能力一律走命令**：在 `prism.ts` 门面和消息协议上加带类型的命令。UI 和 AI 用同一套命令，不给任何一方开后门。
- **不引入** three.js，也不引入第三方 BVH 或渲染库。
- 代码风格以 `CODE_STYLE.md` 为准，格式化用 Prettier。

## 渲染现状（改渲染代码前先读）

- 求交：两级 BVH（TLAS/BLAS），在 CPU 构建，压平成一维 buffer，在 GPU 上迭代遍历。构建用 16 桶分桶 SAH，深度超过 40 时退回中位切分，叶子最多 2 个图元。三角形求交用局部空间的 Möller–Trumbore。
- 光照：NEE 配合按亮度 CDF 对环境贴图做重要性采样；用 power heuristic 做 MIS，结合 BSDF 采样和光源采样；第 4 次反弹之后用俄罗斯轮盘终止。
- 累积：每帧 1 spp 渐进累积，只在相机或参数变化时重置。每帧采样的像素数根据实测 GPU 耗时自适应调整，采样顺序保证每轮覆盖每个像素一次。
- 其他：2K EXR 环境贴图；薄透镜景深（光圈和对焦距离单位为米）；位移贴图通过细分和顶点位移烘焙成真实几何再放进 BVH；材质纹理用带完整 mip 链的 `texture_2d_array`（diffuse、normal、roughness、AO）。
- 已知限制：glTF 只支持三角形和上面 4 种纹理；没有透明和 alpha 镂空；没有透射/玻璃；没有自发光和显式光源，光照只来自环境贴图。

## 常见任务

**加一个场景**：在 `src/app/catalog.ts` 的 `SCENE_CARDS` 中加机位、环境图、几何与卡片信息；按需扩展 `src/app/assemble.ts`，再放封面图到 `assets/gallery/<key>.png`。

**命令**：`npm install` 安装依赖，`npm run dev` 开发，`npm run build` 构建，`npm run preview` 预览构建产物。当前没有测试脚本。

## 工作方式

- 自上而下：先确认问题和期望效果，再定接口，最后实现。动架构之前，先写出方案等确认，不要直接大改。
- 改渲染代码后跑参考图回归测试。测试还没建好之前，就在画廊场景里截图做前后对比。
- 性能相关的改动要给出前后数字：GPU 帧耗时、每秒采样数、收敛到可展示所需的时间。
- 改完后保持画廊所有场景能正常打开，不支持 WebGPU 的浏览器仍然能看到降级的静态图。

## 待决问题

- [ ] 玻璃和自发光先做哪个（倾向先做玻璃）
- [ ] 场景从代码 builder 改成 JSON 描述的重构是否确认
- [ ] AI 接入方式：MCP + WebSocket 连接页面，还是用无头浏览器跑渲染
- [ ] Wasteland 场景收敛到可展示需要多少秒（用来定验收数字）
- [ ] 工作台做成画廊里的一个入口，还是单独一个页面
