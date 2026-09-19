---
description: AI-Infra Odyssey 生产级 LaTeX 数学公式排版与 GitHub 原生渲染规范
globs: **/*.md
---

# AI-Infra Odyssey 生产级 LaTeX 数学公式排版规范

所有在本项目中编写、修改、生成的 Markdown 教学文档与技术 RFC，必须严格遵守以下 **LaTeX 数学公式排版规范**，确保在 **GitHub 原生 Web (GFM + MathJax)**、**VS Code**、**Obsidian** 与各类静态站点中 100% 优雅渲染。

## 核心准则一：文本模式零下划线（Zero Underscore in Text Mode）
- **【绝对禁止】** 严禁在 `\text{...}`、`\mathrm{...}`、`\mathbf{...}` 内部出现裸下划线 `_` 或转义下划线 `\_` / `\\_`。
  - 错误：`\text{total\_bytes}`、`\text{active\\_banks}`、`\text{driver\\_overhead}`
  - 原因：在 LaTeX 文本模式下，`\\` 被解析为换行符，紧随其后的 `_` 缺少合法数学基底，直接导致语法崩溃。
- **【推荐规范】**
  - **单一实体修饰**：使用正体简写，如 $N_{\text{total}}$、 $I_{\text{tiled}}$、 $M_{\text{weights}}$、 $M_{\text{opt}}$。
  - **复合短语修饰**：多词短语统一使用连字符（Hyphen），如 $N_{\text{in-flight}}$、 $N_{\text{per-SM}}$、 $M_{\text{driver-overhead}}$、 $M_{\text{safety-buffer}}$、 $T_{\text{comm-inter-node}}$。

## 核心准则二：代码标识符与数学符号严格分离（Code vs Math Separation）
- **【配置项与引擎变量】**：如果是表达 Python/C++/CUDA 的配置参数、命令行参数或结构体成员，必须使用 Markdown 行内代码语法（Inline Code），严禁放在 `$...$` 数学环境中伪装为变量：
  - 错误：`$\text{max\\_num\\_batched\\_tokens}$` $\to$ **正确**：`` `max_num_batched_tokens` ``
  - 错误：`$\text{gpus\\_per\\_node}$` $\to$ **正确**：`` `gpus_per_node` ``（或数学符号 $N_{\text{gpu/node}}$）
  - 错误：`$\text{block\\_table}[b]$` $\to$ **正确**：`` `block_table[b]` ``
- **【代码数组与局部变量】**：在公式中表达数组或切片时，采用标准数学下标，严禁加反斜杠：
  - 错误：`r\\_a[i] \times r\\_b[j]` $\to$ **正确**：`r_a[i] \times r_b[j]`
- **【数学函数算子】**：函数名使用 `\operatorname{...}` 配连字符，严禁下划线：
  - 错误：`\text{rotate\\_half}(x)` $\to$ **正确**：`\operatorname{rotate-half}(x)`

## 核心准则三：禁止在 `\mathbf{}` 内嵌套中文（No Chinese in MathBF）
- **【禁止】** `\mathbf{16\text{ 字节}}`、`\mathbf{20\text{ 步}}`、`\mathbf{0.31\text{ 秒}}`
  - 原因：部分 Markdown 渲染引擎对 `\mathbf{}` 内多字节中文字符的字体回退支持不一致，会导致排版错位或字体缺失。
- **【推荐规范】** 公式内保留国际标准英制单位，中文量词置于公式外部正文说明：
  - 错误：`= \mathbf{16\text{ 字节}}` $\to$ **正确**：`= \mathbf{16 \text{ B}}`（即 16 字节）
  - 错误：`= \mathbf{20\text{ 步}}` $\to$ **正确**：`= \mathbf{20 \text{ steps}}`（20 步）
  - 错误：`\approx \mathbf{0.31\text{ 秒}}` $\to$ **正确**：`\approx \mathbf{0.31 \text{ s}}`（约 0.31 秒）

## 核心准则四：GitHub GFM 全角标点空格避让（Leading Space Rule）
- **【标点空格避让】**：行内公式紧贴中文全角标点（如冒号 `：`、括号 `（`、逗号 `，`）时，GFM 词法分析器会因缺乏空白字符将 `$` 当作普通字符忽略。中文标点与 `$` 之间必须保留半角空格：
  - 错误：`：$2 \times P$` $\to$ **正确**：`： $2 \times P$`
  - 错误：`（$r \ll M$）` $\to$ **正确**：`（ $r \ll M$ ）`
- **【列表项内块级公式隔离】**：`$$` 块严禁无空行直接挂在 `- 列表项：` 后面或同行，必须采用独立小标题配合段落级 `$$`（前后留空行），或者使用 ```` ```math ```` 语法块。

## 核心准则五：统一体系结构与排队论量纲符号（Unified System Symbols）
在推导性能与系统模型时，保持全库符号链条统一：
- **数据量 / 字节量**：$N$ 或 $D$（$N_{\text{total}}$、 $D_{\text{useful}}$、 $b_{\text{thread}} = 16 \text{ B}$、 $b_{\text{warp}} = 512 \text{ B}$）
- **算力与带宽**：$C_{\text{peak}}$（峰值 FLOPs/s）、 $B_{\text{mem}}$（显存带宽）、 $B_{\text{net}}$（网络带宽）
- **硬件拓扑与规模**：$N_{\text{cards}}$（卡数）、 $N_{\text{nodes}}$（节点数）、 $S_{\text{count}}$（SM 总数）、 $W_{\text{needed}}$（所需 Warp 数）
- **显存账本（Memory Ledger）**：
  - 静态权重：$M_{\text{weights}}$
  - 梯度：$M_{\text{grads}}$
  - 优化器状态：$M_{\text{opt}}$
  - 激活值：$M_{\text{act}}$
  - 动态 KV Cache 池：$M_{\text{kv-pool}}$
  - 驱动与运行时开销：$M_{\text{driver-overhead}}$
  - 防波堤安全缓冲：$M_{\text{safety-buffer}}$
