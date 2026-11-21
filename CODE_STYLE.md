# 代码规范

Prism 项目 TypeScript / WGSL 代码规范。所有 AI 生成的代码必须遵守。

## TypeScript

### 文件结构：两层划分

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
- 参考样例：`src/core/background.ts`。

**以导出类为主体的模块**（如 IR 数据类）按以下口径：

- 类声明整体放契约层：类的成员签名和字段文档就是契约本身，构造函数的默认值、
  小方法体视为契约的一部分，不拆到实现层；
- 模块级的私有辅助（私有函数、私有类型）放实现层；模块没有私有辅助时
  只有契约层，不强行加分隔线；
- 数据类不要为了统一而硬拆成「工厂 + 接口」——那会产生两份字段清单
  （接口一份、类一份），长期互相漂移。「工厂 + 接口」只属于句柄/服务型模块。

### 导入

- 项目内全部用相对路径
- type-only import 放最前面

### 类设计

- 所有成员显式标注 `public` / `private`，不允许省略
- 禁止 getter/setter，全部用普通方法（`width()` 而非 `get width()`）

### 类成员顺序

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

### 注释

- 只在关键逻辑处写简短注释
- 层分隔线使用「文件结构」规定的格式；除此之外仍禁止装饰性分隔线、ASCII 结构框
- 禁止重复代码已表达内容的注释
- 改完自查

## WGSL

- 变量/函数名使用 `snake_case`
- 结构体名使用 `PascalCase`
