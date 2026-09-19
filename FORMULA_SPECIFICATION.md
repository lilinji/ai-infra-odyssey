# AI-Infra Odyssey 生产级 LaTeX 数学公式排版与 GitHub 原生渲染规范指南

> 🔔 **权威文档**：本规范已全面升级为 **[LATEX_FORMULA_SPECIFICATION.md](./LATEX_FORMULA_SPECIFICATION.md)**。  
> 本专栏全书的所有数学公式统一采用 **100% 标准 LaTeX 数学语法**，并在底层完全适配 **GitHub 原生 MathJax**、**KaTeX**、**VS Code**、**Obsidian** 与 **VitePress**。

---

## 1. 核心排版准则

### 准则 1：文本模式零下划线（Zero Underscore in Text Mode）
- **致命反模式**：
  - `\text{total\_bytes}`、`\text{total\\_bytes}`
  - `\text{driver\\_overhead}`、`\text{active\\_banks}`
- **技术原理**：
  在 LaTeX / MathJax / KaTeX 文本模式（`\text{...}`）下，`\\` 被解析为换行符。随后的 `_` 成为失去数学基底的非法标记，直接触发语法崩溃。
- **治理标准**：
  - **单一修饰**：使用简写正体，如 $N_{\text{total}}$、 $I_{\text{tiled}}$、 $M_{\text{weights}}$；
  - **复合短语**：使用连字符（Hyphen）连接，如 $N_{\text{in-flight}}$、 $N_{\text{per-SM}}$、 $M_{\text{driver-overhead}}$、 $M_{\text{safety-buffer}}$。

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
  - ❌ `\approx \mathbf{0.31\text{ 秒}}` $\to$ ✅ `\approx \mathbf{0.31 \text{ s}}`（约 0.31 秒）

### 准则 4：GitHub GFM 全角标点空格避让（Leading Space Rule）
GitHub 网页端采用 **GFM (GitHub Flavored Markdown) + MathJax** 管道：
- **致命反模式**：`： $2 \times P$`、`（ $r \ll M$ ）`、`， $X$`、`、 $Y$`
- **技术原理**：
  GFM 词法分析器为了防止将货币符号（如 `$100`）或变量名误识别为公式，要求开头的 `$` 必须紧随空白字符、ASCII 标点或处于行首。中文全角字符（`：`、`（`、`，`、`、`、`；`、`！`、`？`）属于 Unicode 非 ASCII 字符，会导致 GFM 词法分析器直接忽略公式定界符，整行退化为死文本！
- **治理标准**：
  - 中文全角标点后与 `$` 之间**必须保留一个半角空格**：
  - ❌ `算力上限： $2 \times P$` $\to$ ✅ `算力上限： $2 \times P$`
  - ❌ `矩阵低秩（ $r \ll M$ ）` $\to$ ✅ `矩阵低秩（ $r \ll M$ ）`
  - ❌ `包含特征值， $A$` $\to$ ✅ `包含特征值， $A$`
  - 或使用 GitHub 官方带反引号转义定界符：
  - ✅ `算力上限： $`2 \times P`$`

### 准则 5：块级公式独立成段（Block Math Isolation）
- **致命反模式**：
  ```markdown
  - **模型 MFU**：$$ \text{MFU} = \frac{6P}{C} $$
  ```
  或列表项下没有空行直接缩进 2 空格紧贴：
  ```markdown
  - **模型 MFU**：
    $$
    \text{MFU} = \frac{6P}{C}
    $$
  ```
- **技术原理**：
  GFM 管道中 Markdown 结构解析先于公式渲染。紧贴在列表项内的 `$$` 会被切分成普通段落并插入 `<br>` 标签，且公式内的下划线 `_` 会被 Markdown 优先拆成斜体 `<em>...</em>`，彻底破坏 MathJax 公式块识别。
- **治理标准**：
  - **最佳实践（独立段落块）**：使用加粗小标题，公式块前后保留干净空行：
    ```markdown
    **模型 FLOPs 利用率 (MFU)**：

    $$
    \text{MFU} = \frac{\text{实测吞吐 Tokens/s} \times 6P}{\text{集群卡数} \times \text{单卡峰值 FLOPs/s}}
    $$
    ```
  - **或使用 GitHub 官方 `math` 围栏代码块**：
    ````markdown
    ```math
    \text{MFU} = \frac{\text{实测吞吐 Tokens/s} \times 6P}{\text{集群卡数} \times \text{单卡峰值 FLOPs/s}}
    ```
    ````

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

## 3. 自动化合规检查与格式化工具链

本项目提供了全自动化的 GitHub 原生 LaTeX 扫描与格式化工具链：

```bash
# 1. 运行自动化合规性校验（支持 CI/CD 流水线，全面检测标点、下划线与缩进）
python3 scripts/verify_latex_formulas.py

# 2. 一键自动化规范全库行内公式标点空格（解决中文字符粘连问题）
python3 scripts/fix_github_latex_spacing.py

# 3. 一键自动化规范全库块级公式顶格与空行隔离（解决列表嵌套与斜体撕裂问题）
python3 scripts/normalize_latex_blocks.py
```
若存在任何语法错误或 GFM 兼容性反模式，脚本将精准输出文件、行号及违规上下文并给出退出码 `1`。
