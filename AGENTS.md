# Prism

从零写的 WebGPU 路径追踪器。前台是零操作的画廊；v1 的目标是加一层命令接口，让人（工作台页面）和 AI（MCP）都能调用。
不使用 three.js 渲染管线或第三方 BVH；渲染页画面在浏览器里计算，画廊封面是静态预览图。

- 在线 demo：https://bijingdabaihua.github.io/prism/
- 技术栈：TypeScript、WGSL、WebGPU、Web Workers、Vite、glTF 2.0、EXR
- 资产：Poly Haven（CC0），新增资产保持同样的许可

## 当前阶段与优先级（约 6 周）

这个项目首先是求职作品。判断一个改动值不值得做，看它能否在**效果、速度（要有数字）、可用性**里至少提升一项。

场景继续由应用层通过现有 API 装配。工作台和 AI 接入时按实际需要补充操作。等有保存、分享或批量加载场景的需求，再考虑 JSON 格式。

1. **第 1–2 周，补渲染短板（当前）**：光滑玻璃透射已完成，「玻璃球」场景已验证。自发光材质和发光几何直接采样已接入，由「发光面板」场景验证。
2. **第 3–4 周，工作台与 AI**：工作台页面（载入模型、材质库、环境预设、导出 PNG、性能面板）；MCP server 加页面桥接。
3. **第 5–6 周，包装**：README 第一屏放性能数字和 AI 录屏，写一页技术说明。

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

## 渲染现状（改渲染代码前先读）

- 求交：两级 BVH（TLAS/BLAS），在 CPU 构建，压平成一维 buffer，在 GPU 上迭代遍历。构建用 16 桶分桶 SAH，深度超过 40 时退回中位切分，叶子最多 2 个图元。三角形求交用局部空间的 Möller–Trumbore。
- 光照：NEE 配合按亮度 CDF 对环境贴图做重要性采样；用 power heuristic 做 MIS，结合 BSDF 采样和光源采样；第 4 次反弹之后用俄罗斯轮盘终止。
- 累积：每帧 1 spp 渐进累积，只在相机或参数变化时重置。每帧采样的像素数根据实测 GPU 耗时自适应调整，采样顺序保证每轮覆盖每个像素一次。
- 其他：2K EXR 环境贴图；薄透镜景深（光圈和对焦距离单位为米）；位移贴图通过细分和顶点位移烘焙成真实几何再放进 BVH；材质纹理用带完整 mip 链的 `texture_2d_array`（diffuse、normal、roughness、AO）。
- 已知限制：glTF 只支持三角形和上面 4 种纹理；没有透明和 alpha 镂空；支持光滑玻璃透射，尚不支持粗糙透射或体吸收；支持双面发光几何，尚无独立的点光源、聚光灯等类型。

## 常见任务

**加一个场景**：在 `src/app/catalog.ts` 的 `SCENE_CARDS` 中加机位、环境图、几何与卡片信息；按需扩展 `src/app/assemble.ts`，再放封面图到 `assets/gallery/<key>.png`。

**命令**：`npm install` 安装依赖，`npm run dev` 开发，`npm run build` 构建，`npm run preview` 预览构建产物。当前没有测试脚本。

## 工作方式

- 自上而下：先确认问题和期望效果，再定接口，最后实现。动架构之前，先写出方案等确认，不要直接大改。
- 改渲染代码后跑参考图回归测试。测试还没建好之前，就在画廊场景里截图做前后对比。
- 性能相关的改动要给出前后数字：GPU 帧耗时、每秒采样数、收敛到可展示所需的时间。
- 改完后保持画廊所有场景能正常打开，不支持 WebGPU 的浏览器仍然能看到降级的静态图。

## 待决问题

- [ ] AI 接入方式：MCP + WebSocket 连接页面，还是用无头浏览器跑渲染
- [ ] Wasteland 场景收敛到可展示需要多少秒（用来定验收数字）
- [ ] 工作台做成画廊里的一个入口，还是单独一个页面

## 代码规范

Prism 项目 TypeScript / WGSL 代码规范。所有 AI 生成的代码必须遵守。

### TypeScript

#### 文件结构：两层划分

每个模块文件固定分两层，层间用分隔线隔开：

```typescript
// ═══════════════════════ 契约层（导出） ═══════════════════════

/** 构造入口：只有声明（类型 + 文档），不含实现 */
export let createXxx: (deps: XxxDeps) => XxxHandle;

export interface XxxHandle { ... }      // 对外契约
export interface XxxDeps { ... }        // 依赖面
export type XxxRequest = ...;           // 被框架引用的形状（如消息报文）

// ═══════════════════════ 实现层（不导出） ═══════════════════════

class XxxImpl implements XxxHandle { ... }   // 私有类、私有类型

createXxx = (deps) => new XxxImpl(deps);     // 契约层变量的赋值
```

- **契约层 = 全部导出。** 它是模块的全部承诺：读完这一层即可掌握整个模块，
  不需要读实现层。
  - 值：只出现「类型 + 文档」的声明，**不含任何实现**（连一行委托也不行）；
  - 类型：对外契约、依赖面、被框架引用的形状（如消息报文）。
- **实现层 = 全部不导出。** 它是模块的全部细节：私有类、私有类型，以及
  契约层变量的赋值语句。契约不变，本层可任意重构。
- 一个符号归哪层只看一条：**它导出吗**。导出就契约层，不导出就实现层，
  没有例外。
- 引用方向：实现层可以引用契约层；契约层不得引用实现层的任何符号。
- 导出值的标准形态是 `export let` + 实现层赋值。对外只读由 ESM 保证
  （import 绑定不可重新赋值）；模块内不重新赋值契约层变量，靠约定遵守。
  不用 `const` 的原因：`const` 必须在声明处初始化，而初始化式就是实现，
  与「契约层零实现」互斥。
- 参考样例：`src/core/api/background.ts`。

**以导出类为主体的模块**（如 IR 数据类）：类不直接导出，同样拆两层——

- 契约层：`export interface Xxx`（全部字段 + 方法签名，文档只写在这里）+
  `export let Xxx`（类绑定，只写构造签名和静态方法签名，无实现）+ 其余导出类型；
- 实现层：`class XxxImpl implements Xxx`（字段不带文档，与 interface 的一致性
  由 `implements` 在编译期强制）+ 底部 `Xxx = XxxImpl` 赋值；
- interface 与类绑定同名共存（类型空间 + 值空间），调用方的 `new Xxx()`、
  `Xxx.static()`、`: Xxx` 写法全部不变；
- 参考样例：`src/core/worker/model/material/material.ts`。

#### 导入

- 项目内全部用相对路径
- type-only import 放最前面

#### 类设计

- 所有成员显式标注 `public` / `private`，不允许省略
- 禁止 getter/setter，全部用普通方法（`width()` 而非 `get width()`）

#### 类成员顺序

越开放越靠前，严格按以下顺序排列：

```
public static
public 属性
public readonly
private 属性
private readonly
constructor
public 方法
private 方法
```

**属性区规则：**
- 属性之间不允许空行
- 不同可见性分组（public / public readonly / private / private readonly）之间也不允许空行
- 整个属性区从上到下是连续的，中间不插入空行

**方法区规则：**
- 方法体必须展开换行，即使只有一行 return 也不允许单行写法
  ```typescript
  // ❌ 错误
  public width(): number { return this.w; }

  // ✅ 正确
  public width(): number {
    return this.w;
  }
  ```
- 方法之间必须有空行分隔

#### 注释

- 只在关键逻辑处写简短注释
- 层分隔线使用「文件结构」规定的格式；除此之外仍禁止装饰性分隔线、ASCII 结构框
- 禁止重复代码已表达内容的注释
- 改完自查

### WGSL

- 变量/函数名使用 `snake_case`
- 结构体名使用 `PascalCase`
