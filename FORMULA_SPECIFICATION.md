# AI-Infra Odyssey 数学公式排版与 KaTeX 兼容性规范指南

> 本规范为全套教程（从基础篇到 Capstone 实战）的通用数学排版与渲染基线，旨在保证全库在 VS Code、Obsidian、GitHub Web 及各种 Markdown 静态站点（VitePress / Docusaurus）中实现 **100% 语法零报错** 与 **顶级计算机系统学术出版物级别的一致性美感**。

---

## 1. 核心排版准则

### 准则 1：文本模式零下划线（Zero Underscore in Text Mode）
- **致命反模式**：
  - `\text{total\_bytes}`、`\text{total\\_bytes}`
  - `\text{driver\\_overhead}`、`\text{active\\_banks}`
- **技术原理**：
  在 KaTeX / LaTeX 文本模式（`\text{...}`）下，`\\` 被解析为断行符（Linebreak）。随后的 `_` 成为失去数学基底的非法标记，直接触发：
  ```text
  ParseError: KaTeX parse error: Expected 'EOF', got '_'
  ```
- **治理标准**：
  - **单一修饰**：使用简写正体，如 $N_{\text{total}}$、$I_{\text{tiled}}$、$M_{\text{weights}}$；
  - **复合短语**：使用连字符（Hyphen）连接，如 $N_{\text{in-flight}}$、$N_{\text{per-SM}}$、$M_{\text{driver-overhead}}$、$M_{\text{safety-buffer}}$。

### 准则 2：代码标识符与数学符号隔离（Code vs Math Separation）
- **系统/框架配置参数**：
  - 严禁包裹在 `$...$` 中伪装为数学变量。
  - 必须使用 Markdown 行内代码语法（Inline Code）：
    - ❌ `$\text{max\\_num\\_batched\\_tokens}$` $\to$ ✅ `` `max_num_batched_tokens` ``
    - ❌ `$\text{gpus\\_per\\_node}$` $\to$ ✅ `` `gpus_per_node` ``
    - ❌ `$\text{block\\_table}[b]$` $\to$ ✅ `` `block_table[b]` ``
- **片上寄存器与数组访问**：
  - 在公式中表达数组或切片时，采用标准数学下标，严禁加反斜杠转义：
    - ❌ `r\\_a[i] \times r\\_b[j]` $\to$ ✅ `r_a[i] \times r_b[j]`
- **函数与变换算子**：
  - 使用 `\operatorname{...}` 结合连字符定义：
    - ❌ `\text{rotate\\_half}(x)` $\to$ ✅ `\operatorname{rotate-half}(x)`

### 准则 3：禁止在 `\mathbf{}` 内裸嵌中文（No Chinese in MathBF）
- **反模式**：`\mathbf{16\text{ 字节}}`、`\mathbf{20\text{ 步}}`、`\mathbf{0.31\text{ 秒}}`
- **治理标准**：
  - 公式内部统一使用标准英文计量单位（Bytes, steps, s, GB, TFLOPS）；
  - 中文修饰与量词置于公式外部的正文说明中：
    - ❌ `= \mathbf{16\text{ 字节}}` $\to$ ✅ `= \mathbf{16 \text{ B}}`（即 16 字节）
    - ❌ `= \mathbf{20\text{ 步}}` $\to$ ✅ `= \mathbf{20 \text{ steps}}`（20 步）
### 准则 4：GitHub GFM 解析器特异性避坑（标点空格避让与列表块隔离）

GitHub Web 页面采用 **GFM (GitHub Flavored Markdown) + MathJax** 渲染引擎，具有以下特定的词法解析规则与陷阱：

1. **行内公式与中文全角标点避让（Leading Space Rule）**：
   - **致命反模式**：`：$2 \times P \times B \times S$`、`：$\text{FLOPs}$`
   - **技术原理**：GFM 为了避免将普通文本或货币符号（如 `$100`）误判为 LaTeX，要求行内开头的 `$` 前面必须为**空白字符或行首**。如果前面紧贴全角冒号 `：` 或括号 `（`，GFM 词法分析器将直接放弃匹配，导致整段公式退化为原始文本！
   - **治理标准**：
     - 标点与公式之间**必须保留一个半角空格**：
       - ❌ `前向算力：$2 \times P$` $\to$ ✅ `前向算力： $2 \times P$`
     - 或使用 GitHub 官方推荐的带反引号转义定界符：
       - ✅ `前向算力：$`2 \times P`$`

2. **块级公式严禁紧贴/单行嵌在列表项中（List Item Block Conflict）**：
   - **致命反模式**：
     ```markdown
     - **模型 FLOPs 利用率 (MFU)**：$$ \text{MFU} = \frac{6P}{C} $$
     ```
     或列表项下没有空行直接缩进 2 空格：
     ```markdown
     - **模型 FLOPs 利用率 (MFU)**：
       $$
       \text{MFU} = \frac{6P}{C}
       $$
     ```
   - **技术原理**：GFM 会先执行 Markdown 块级切分（List Parser），再执行段落软换行（插入 `<br>`）与斜体强调（将公式中的下划线 `_` 误匹配为 `<em>...</em>`），最后才由前端渲染引擎处理 MathJax。塞在无空行的列表项内会直接生成 `<p>$$<br>...</p>`，彻底摧毁 MathJax 块级匹配，导致整块公式原样吐出！
   - **治理标准**：
     - **最佳实践 A（顶级段落独立块，推荐）**：小标题使用加粗文本段落，公式块前后保留干净空行：
       ```markdown
       **模型 FLOPs 利用率 (MFU)**：

       $$
       \text{MFU} = \frac{6P}{C}
       $$
       ```
     - **最佳实践 B（GitHub 官方 Math 代码块）**：
       ````markdown
       ```math
       \text{MFU} = \frac{6P}{C}
       ```
       ````
     - **最佳实践 C（若必须在无序列表中）**：必须保留空行，且公式块必须严格使用 **4 空格** 缩进：
       ```markdown
       - **模型 FLOPs 利用率 (MFU)**：

           $$
           \text{MFU} = \frac{6P}{C}
           $$
       ```

---

## 2. 全体系结构核心量纲与符号对照表

| 概念分类 | 推荐数学符号 | 示例与场景 | 禁止写法 |
| :--- | :--- | :--- | :--- |
| **算力 (Compute)** | $C_{\text{peak}}$, $C_{\text{eff}}$ | A100 FP16 峰值 $C_{\text{peak}} = 312 \text{ TFLOPS}$ | `\text{Compute}_{\text{peak}}` |
| **带宽 (Bandwidth)** | $B_{\text{mem}}$, $B_{\text{net}}$ | HBM 带宽 $B_{\text{mem}} = 2039 \text{ GB/s}$ | `\text{BW}_{\text{HBM}}` |
| **延迟 (Latency)** | $L_{\text{hbm}}$, $L_{\text{net}}$, $W$ | 访问延迟 $L \approx 400 \text{ ns}$ | `\text{Latency}_{\text{avg}}` |
| **数据量 (Data Volume)** | $N_{\text{total}}$, $b_{\text{warp}}$, $D_{\text{useful}}$ | 在途数据量 $N_{\text{total}} = 816 \text{ KB}$ | `N_{\text{total\_bytes}}` |
| **算术强度 (Intensity)** | $I$, $I_{\text{knee}}$, $I_{\text{tiled}}$ | 硬件拐点 $I_{\text{knee}} = 153.0 \text{ FLOPs/Byte}$ | `I_{\text{block\_tiled}}` |
| **硬件计数 (Count)** | $N_{\text{cards}}$, $N_{\text{nodes}}$, $S_{\text{count}}$ | 集群规模 $N_{\text{cards}} = 128$ | `\text{Num\_GPUs}` |
| **显存账本 (Memory Ledger)** | $M_{\text{weights}}$, $M_{\text{kv-pool}}$, $M_{\text{opt}}$ | 动态 KV 池容量 $M_{\text{kv-pool}} = 50 \text{ GB}$ | `M_{\text{kv\_pool}}` |
| **SLO 核心指标** | $\text{TTFT}$, $\text{TPOT}$, $\text{QPS}$ | 首字延迟 $\text{TTFT} \le 400 \text{ ms}$ | `\text{Target\_TTFT}` |

---

## 3. 自动化合规检查

本项目提供了全自动化公式扫描工具，可随时执行全库合规性校验：

```bash
python3 scripts/verify_katex_formulas.py
```
若存在任何语法错误或反模式，脚本将精准输出文件、行号及违规上下文并给出退出码 `1`。
