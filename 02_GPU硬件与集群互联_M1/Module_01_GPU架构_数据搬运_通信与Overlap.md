# 🚀 Module 01: GPU 架构、数据搬运、通信与 Compute-Communication Overlap

> **讲师 / 作者**：👓 **Ringi**（AI Infrastructure 工程师）  
> **模块定位**：必修 ｜ **建议时长**：14–18h ｜ **先修要求**：Python、PyTorch 基础、基本线性代数  
> **核心关键词**：SM / Warp / HBM / Roofline / PCIe / NVLink / NVSwitch / RDMA / GPUDirect RDMA / InfiniBand / RoCE / NCCL / NVSHMEM / Collective / TMA / mbarrier / Warp Specialization / Communication-Compute Overlap
>
> **模块目标**：不是“认识一堆通信名词”，而是建立一套可以从 Tensor Shape 一路追到真实硬件 Data Path，并能定量判断瓶颈、设计 Overlap、做性能实验和故障定位的 AI Infra 心智模型。

---

## 🎯 0. 为什么 Module 01 必须这样学？

大模型训练与推理性能问题，最终几乎都可以还原成两个动作：

1. **计算（Compute）**：Tensor 在 CUDA Core / Tensor Core 上完成运算；
2. **搬运（Movement / Communication）**：Tensor 在 Register、SRAM、L2、HBM、GPU、NIC、节点之间移动。

因此，本模块统一采用下面的第一性原理：

> **能不搬就不搬；能少搬就少搬；必须搬，就走最近、最宽、最适合当前消息规模的路径；最后再考虑如何把搬运隐藏在计算后面。**

这里的“通信”采用广义定义：

```text
Register ↔ Shared Memory
       ↓
      L2
       ↓
      HBM
       ↓
GPU ↔ GPU
       ↓
GPU ↔ NIC
       ↓
Node ↔ Node
```

只要数据从一个物理位置移动到另一个物理位置，都可以视为 **Data Movement**。

这也是后续学习 DDP、FSDP、ZeRO、Tensor Parallel、Pipeline Parallel、Context Parallel、Expert Parallel、MoE、Distributed Inference 的统一基础。

---

# 🧭 1. Ringi 四问教学协议

学习任何一个算子、并行策略或通信优化时，必须回答以下四个问题。

## 1.1 📐 Shape 是什么？

先追 Tensor，不先背名词。

例如：

```text
[B, S, H]
   ↓
Linear / QKV
   ↓
[B, S, 3H]
   ↓
TP Shard
   ↓
[B, S, H / TP]
   ↓
AllGather / ReduceScatter / AllReduce
```

必须明确：

- Tensor Shape；
- dtype；
- shard 维度；
- Reshape / Transpose；
- Broadcast / Reduction；
- 每个 rank 持有什么；
- 通信前后 Shape 如何变化。

---

## 1.2 💰 钱花在哪里？

任何性能结论都要尽量量化。

至少估算：

```text
Parameters
FLOPs
HBM Bytes
Communication Volume
Message Size
Latency
Bandwidth
Step Time
Exposed Communication
```

并判断：

```text
Compute-bound
Memory-bound
Communication-bound
Latency-bound
Resource-contention-bound
```

---

## 1.3 ⚙️ 真正在机器上怎么跑？

不能停留在：

```python
dist.all_reduce(x)
```

要继续追：

```text
PyTorch
  ↓
ProcessGroupNCCL
  ↓
NCCL Algorithm / Protocol
  ↓
CUDA Kernel / Copy Engine / NIC DMA
  ↓
NVLink / PCIe / RDMA
  ↓
Remote GPU Memory
```

必须知道：

- 谁发起；
- 谁搬；
- 谁同步；
- 谁等待；
- 占不占 SM；
- 是否占 HBM/L2/PCIe/NIC；
- 数据走哪条物理路径。

---

## 1.4 🚚 数据到底怎么搬？

看到：

```text
GPU0 → GPU1
```

必须继续问：

```text
GPU0 HBM
   ↓
LSU / TMA / Copy Engine / NIC DMA ?
   ↓
NVLink / PCIe ?
   ↓
NVSwitch / PCIe Switch / Root Complex ?
   ↓
NIC ?
   ↓
IB / RoCE ?
   ↓
Remote NIC
   ↓
Remote GPU HBM
```

> **Module 01 的最终能力标准：看到一个通信算子，脑中能自动展开 Data Path。**

---

# 🧠 2. 本模块统一性能模型

## 2.1 第一条公式：`t = α + S / β`

几乎所有通信问题都可以先用：

\[
T = \alpha + \frac{S}{\beta}
\]

其中：

- \(\alpha\)：固定启动开销；
- \(S\)：工作规模 / Message Size；
- \(\beta\)：有效吞吐能力。

因此：

### 小消息

```text
α >> S / β
```

→ **Latency-bound**

### 大消息

```text
S / β >> α
```

→ **Bandwidth-bound**

这个模型会贯穿：

- TMA vs LSU；
- NVLink；
- RDMA；
- NCCL Collective；
- MoE All-to-All；
- Decode 小包通信；
- Training 大包通信。

---

## 2.2 第二条公式：Roofline

单个 Kernel 的基本性能模型：

\[
P = \min(P_{peak}, BW_{HBM} \times AI)
\]

其中：

\[
AI = \frac{FLOPs}{Bytes}
\]

由此判断：

```text
低 Arithmetic Intensity → Memory-bound
高 Arithmetic Intensity → Compute-bound
```

---

## 2.3 第三条公式：Exposed Communication

通信时间不等于通信真正增加的 Step Time。

定义：

\[
T_{exposed} = T_{comm} - T_{hidden}
\]

于是：

\[
T_{step} \approx T_{compute} + T_{exposed} + T_{other}
\]

真正的优化目标是：

> **最小化 Exposed Communication，而不只是最小化 Communication Duration。**

---

## 2.4 第四条模型：Overlap 竞争倍率 `k`

通信与计算同时执行，不代表互不干扰。

可定义：

\[
T_{comm,overlap} = k \cdot T_{comm,solo}, \quad k \ge 1
\]

`k` 来自：

- SM 竞争；
- Register File；
- L2；
- HBM；
- Warp Scheduler；
- PCIe / NVLink / NIC 流量叠加。

因此：

> **Timeline 上重叠 ≠ 性能上免费。**

---

# 🔺 3. 通信的不可能三角

Data Movement 通常需要在三个目标之间取舍：

```text
             高带宽
               ▲
              / \
             /   \
            /     \
           /       \
          ▼─────────▼
       低延迟      少占 SM
```

典型策略：

| 搬运方式 | 带宽 | 延迟 | SM 占用 | 典型用途 |
|---|---|---|---|---|
| SM Load/Store | 中/高 | 低 | 高 | 需要计算/归约的搬运 |
| TMA | 高 | 有启动开销 | 很低 | Kernel 内大块异步搬运 |
| Copy Engine | 高 | 有调度开销 | 近零 | Kernel 外纯搬运 |
| RDMA + CPU 控制 | 高 | 中 | 数据面低 | 大包跨机 |
| GPU-initiated RDMA | 高 | 更低 | 短暂占用 | 小包 / 低延迟跨机 |

因此不存在“所有场景都最好的通信方式”。

训练通常更关心：

```text
Bandwidth + Overlap
```

Decode 通常更关心：

```text
Latency
```

---

# 📑 4. 本模块章节目录

| 序号 | 章节 | 核心问题 | 关键产出 | 建议时长 |
|---:|---|---|---|---:|
| **01** | **GPU 执行与存储体系** | Tensor 在单卡上如何被执行和搬运？ | Roofline 分析 + Kernel 分类 | 2h |
| **02** | **通信第一性原理与性能模型** | 为什么需要搬？什么时候 Latency-bound / Bandwidth-bound？ | α+S/β 手算模型 | 1.5h |
| **03** | **GPU 节点与集群硬件拓扑** | NVLink / PCIe / NIC / Rail / Clos 如何决定路径？ | Topology 图 + Rank Placement | 2h |
| **04** | **RDMA 与 GPUDirect RDMA** | 如何把 CPU/Host 从数据路径移出去？ | QP/WQE/CQ/MR 时序图 | 2h |
| **05** | **机内数据搬运** | LSU / TMA / CE 谁搬、谁同步、谁占 SM？ | SM-free Data Path 对比 | 2h |
| **06** | **机间数据搬运** | IBRC / GPU-initiated 控制面、两跳/一跳如何选择？ | Inter-node Data Path | 1.5h |
| **07** | **NCCL / NVSHMEM / Collective** | Collective 如何映射到底层 transport？ | 通信量推导 + nccl-tests | 2h |
| **08** | **Compute-Communication Overlap** | 如何真正隐藏通信，又不把计算拖慢？ | Nsight Timeline + overlap 指标 | 2h |
| **09** | **Benchmark、监控与故障诊断** | 如何判断慢在 GPU、PCIe、NVLink、NIC、网络还是某个 Rank？ | AI 集群排障决策树 | 2h |

---

# 01｜GPU 执行与存储体系
## SM / Warp / Tensor Core / Register / Shared Memory / L2 / HBM

### 学习目标

完成本讲后，应能够解释：

- GPU 为什么适合大规模矩阵计算；
- SM、Warp、Thread Block、Grid 的关系；
- Tensor Core / CUDA Core / LSU 分别负责什么；
- Register / Shared Memory / L2 / HBM 的层级；
- Occupancy、Memory Coalescing、Warp Divergence；
- Arithmetic Intensity 与 Roofline。

---

## 01.1 GPU 执行层级

```text
GPU
 ├── GPC / Processing Cluster
 │    └── SM
 │         ├── Warp Scheduler
 │         ├── CUDA Core
 │         ├── Tensor Core
 │         ├── LSU
 │         ├── Register File
 │         └── Shared Memory / L1
 │
 ├── L2 Cache
 └── HBM
```

软件执行层级：

```text
Thread
  ↓
Warp
  ↓
Thread Block / CTA
  ↓
Grid
  ↓
Kernel
```

重点理解：

> **Warp 是调度基本粒度，SM 是资源承载与调度核心单位。**

---

## 01.2 SM 不只是“算力”

SM 既承担：

```text
Compute
Control
Memory Instruction Issue
Synchronization
```

因此在通信问题中，SM 可能扮演：

- 搬运者；
- WQE 构建者；
- Flag Poller；
- Producer；
- Consumer；
- Reduce Operator。

这也是为什么通信会与 GEMM 竞争 GPU 资源。

---

## 01.3 存储层级

```text
Register
   ↓
Shared Memory / L1
   ↓
L2
   ↓
HBM
```

越靠近计算单元通常：

```text
Latency ↓
Capacity ↓
Cost / Byte ↑
```

越远：

```text
Capacity ↑
Latency ↑
```

因此高性能 Kernel 本质上是在做：

> **数据复用 + 减少远距离搬运。**

---

## 01.4 Roofline 分析模板

对任何 Kernel：

### Step 1：算 FLOPs

```text
FLOPs = ?
```

### Step 2：算 HBM Bytes

```text
Read Bytes + Write Bytes = ?
```

### Step 3：算 Arithmetic Intensity

\[
AI = FLOPs / Bytes
\]

### Step 4：判断瓶颈

```text
低 AI → Memory-bound
高 AI → Compute-bound
```

---

## 🧪 Lab 01：Kernel Roofline

测试：

- GEMM；
- Element-wise Add；
- Softmax；
- LayerNorm。

工具：

```text
PyTorch Profiler
Nsight Systems
Nsight Compute
```

输出：

| Kernel | FLOPs | HBM Bytes | AI | 判断 | Profiler 证据 |
|---|---:|---:|---:|---|---|
| GEMM | | | | | |
| Add | | | | | |
| Softmax | | | | | |
| LayerNorm | | | | | |

---

# 02｜通信第一性原理与性能模型
## Data Movement / α+S/β / Latency / Bandwidth

### 学习目标

- 理解为什么并行必然产生通信；
- 建立“能不搬 / 少搬 / 走近路”的优化顺序；
- 区分 latency-bound 与 bandwidth-bound；
- 区分规格带宽、实测带宽、算法带宽；
- 理解训练与 Decode 的通信目标为什么不同。

---

## 02.1 为什么并行必然产生通信？

并行本质上是在多个设备之间重新分配：

```text
Compute Capacity
Memory Capacity
Memory Bandwidth
```

但分块之后必须交换部分信息。

| 并行 | 换来了什么 | 通信代价 |
|---|---|---|
| DP | 吞吐 | Gradient Sync |
| TP | 单层算力/显存扩展 | Activation / Partial Result |
| PP | 跨节点模型切分 | Activation / Gradient P2P |
| EP | Expert 扩展 | Token Dispatch / Combine |
| FSDP | 参数/梯度/优化器状态分片 | AllGather / ReduceScatter |

---

## 02.2 通信优化的正确顺序

### Level 0：不搬

例如直接改变算法，使大 Tensor 不需要被 Gather。

### Level 1：少搬

典型：

- Quantization；
- Compression；
- Top-k；
- Sharding；
- Hierarchical Reduction；
- Recomputation。

### Level 2：走最近路径

```text
SRAM < HBM < NVLink < PCIe / NIC < Network
```

### Level 3：提高搬运效率

```text
大包
多通道
合理拓扑
RDMA
GPUDirect
```

### Level 4：Overlap

让搬运退出关键路径。

---

## 02.3 带宽必须分层理解

看到一个“900 GB/s”“400 Gb/s”数字，必须问：

1. **Peak / Spec Bandwidth？**
2. **单向还是双向？**
3. **Achieved Bandwidth？**
4. **Algorithm Bandwidth？**
5. **Application Effective Bandwidth？**

> **禁止把规格书峰值直接当训练通信速度。**

---

## 🧪 Lab 02：α + S/β

测试 Message Size：

```text
1 KB
16 KB
1 MB
64 MB
1 GB
```

绘制：

```text
Latency vs Message Size
Bandwidth vs Message Size
```

回答：

- 从哪个消息规模开始进入 bandwidth-bound？
- 为什么 Decode 与 Training 的最优通信实现不同？

---

# 03｜GPU 节点与集群硬件拓扑
## PCIe / NUMA / NVLink / NVSwitch / NIC / Clos / Rail

### 学习目标

建立从单节点到 Pod 的物理拓扑视图。

---

## 03.1 典型节点

```text
CPU / NUMA
   │
PCIe Root Complex
   │
 ┌─┴────────────┐
GPU            NIC
 │
NVLink / NVSwitch
 │
GPU
```

关键问题：

- GPU 与 GPU 是 NVLink 还是 PCIe？
- GPU 与 NIC 是否同一 PCIe / NUMA 域？
- 一张 GPU 对应哪张 NIC？

---

## 03.2 NVLink 与 NVSwitch

NVLink 解决：

```text
GPU ↔ GPU 高带宽互联
```

NVSwitch 解决：

```text
多 GPU 高带宽交换 / 全互联问题
```

要明确：

```text
NVLink = Link
NVSwitch = Switch Fabric
```

不能混为一个概念。

---

## 03.3 PCIe 的四个角色

PCIe 虽不是大模型机内通信的理想主路径，但始终是关键桥梁：

1. GPU ↔ NIC；
2. GPU ↔ CPU D2H/H2D；
3. Host 控制路径；
4. 无高速 P2P 时 GPU ↔ GPU fallback。

PCIe 是否成为瓶颈，必须看：

> **同一条 PCIe Path 上所有并发流量之和，而不是整机所有设备带宽简单相加。**

---

## 03.4 NUMA / Affinity

错误路径：

```text
GPU0
 ↓
PCIe
 ↓
CPU Socket 0
 ↓
UPI
 ↓
CPU Socket 1
 ↓
NIC1
```

优选：

```text
GPU0
 ↓
Local PCIe Root
 ↓
NIC0
```

因此需要理解：

- CPU affinity；
- NUMA memory placement；
- GPU-NIC affinity。

---

## 03.5 Clos / Fat-tree

典型：

```text
GPU Node
   ↓
Leaf / ToR
   ↓
Spine
   ↓
Leaf / ToR
   ↓
GPU Node
```

关键指标：

```text
Oversubscription Ratio
Non-blocking
Incast
Congestion
```

---

## 03.6 Rail-Optimized

8 GPU + 8 NIC 节点可抽象成：

```text
GPU0-NIC0 ── Rail0
GPU1-NIC1 ── Rail1
...
GPU7-NIC7 ── Rail7
```

区分：

```text
Rail-parallel
Cross-rail
Incast
```

典型映射：

| 并行 | 拓扑倾向 |
|---|---|
| TP | NVLink Domain |
| DP / FSDP | Rail-parallel / Pod |
| PP | 较灵活 |
| EP | Cross-rail 敏感 |

---

## 03.7 Rank Placement 也是性能问题

硬件拓扑正确，不代表软件一定走对路径。

错误的：

```text
Rank Mapping
Hostfile
DeviceMesh
```

可能把本应“近邻”的通信映射成跨 Rail / 跨 Pod。

因此：

> **Topology + Rank Mapping 才构成真实通信拓扑。**

---

## 🧪 Lab 03：Topology Mapping

执行：

```bash
nvidia-smi topo -m
```

并结合：

```text
lspci
numactl --hardware
ibdev2netdev
```

输出一张节点图，标出：

- GPU；
- NVLink；
- PCIe Root；
- CPU NUMA；
- NIC；
- 推荐 GPU-NIC 配对。

---

# 04｜RDMA 与 GPUDirect RDMA
## DMA / QP / WQE / CQ / MR / One-sided / Two-sided

### 学习目标

理解 RDMA 不是“更快的 Socket”，而是一整套 **控制面 / 数据面分离** 的系统设计。

---

## 04.1 TCP/IP 的问题在哪里？

传统网络路径中 CPU 可能参与：

```text
Protocol Processing
Kernel Buffer
Memory Copy
Interrupt
Scheduling
```

RDMA 的核心方向：

```text
Zero-copy
Kernel Bypass
CPU Bypass in Data Path
```

硬件基础：

```text
DMA Engine
```

---

## 04.2 控制面 vs 数据面

### 控制面

回答：

```text
搬什么？
从哪搬？
搬到哪？
搬多少？
什么时候开始？
```

### 数据面

真正执行：

```text
Memory → NIC → Network → NIC → Memory
```

这是后面理解 IBRC / GPU-initiated RDMA 的基础。

---

## 04.3 RDMA 五个核心对象

| 对象 | 解决的问题 | 心智模型 |
|---|---|---|
| QP | 从哪条通信通道走？ | 车道 |
| WQE | 搬什么、搬多少？ | 工作单 |
| CQ / CQE | 搬完了吗？ | 回执 |
| MR | NIC 可以访问哪块内存？ | 注册仓库 |
| PD | 谁有权限访问？ | 权限域 |

典型流程：

```text
Application
   ↓ create WQE
SQ / QP
   ↓ doorbell
NIC
   ↓ DMA
Local MR
   ↓
Network
   ↓
Remote NIC
   ↓ DMA
Remote MR
   ↓
CQE
```

---

## 04.4 单边 vs 双边

### 双边

```text
Sender: Send
Receiver: Recv
```

双方都需要参与控制面。

### 单边

```text
RDMA Write
RDMA Read
```

发送方知道 Remote MR 后即可直接执行 RMA 操作。

关键定义：

> **单边/双边的本质是接收侧是否需要显式参与当前操作的控制面。**

---

## 04.5 Ordering 是正确性问题

下面并不天然正确：

```text
put(data)
put(flag)
```

因为必须证明：

```text
flag 可见 ⇒ data 已经可安全读取
```

可依赖：

- Same-QP ordering；
- Fence；
- Write with Immediate；
- 明确的协议级 ordering guarantee。

> **禁止用“发送端已经发完”替代“接收端数据已经可见”的证明。**

---

## 04.6 GPUDirect RDMA

传统：

```text
GPU
 ↓
Host Memory
 ↓
NIC
 ↓
Network
 ↓
NIC
 ↓
Host Memory
 ↓
GPU
```

GDR：

```text
GPU HBM
 ↓
PCIe
 ↓
NIC DMA
 ↓
Network
 ↓
NIC DMA
 ↓
PCIe
 ↓
Remote GPU HBM
```

核心价值：

```text
消除 Host Staging
减少 Memory Copy
减少 CPU 数据面参与
```

注意：

> **GDR 绕过 Host Memory，不代表绕过 PCIe。**

---

## 🧪 Lab 04：RDMA Benchmark

建议：

```text
ib_write_bw
ib_read_bw
ib_send_bw
```

记录：

- message size；
- QP 数；
- latency；
- bandwidth；
- CPU 使用率；
- NUMA placement。

---

# 05｜机内数据搬运
## LSU / TMA / Copy Engine / mbarrier / IPC

### 核心问题

> **不是只问“搬得多快”，而是问“谁搬、谁等、谁被占用”。**

---

## 05.1 数据面与控制面必须同时看

| 维度 | 问题 |
|---|---|
| 数据面 | 谁真正执行搬运？ |
| 控制面 | 谁通知“搬完了”？ |

只有两者都从 SM 指令管线中尽量解耦，才接近真正的：

```text
SM-free Data Movement
```

---

## 05.2 四种典型搬运方式

| 方式 | 谁搬 | SM 代价 | 适用场景 |
|---|---|---|---|
| LSU Load/Store | SM | 高 | 需要 Load/Store / Reduce |
| TMA | 专用异步搬运单元 | 很低 | Kernel 内块状搬运 |
| Copy Engine | 独立 DMA | 近零 | Kernel 外异步 memcpy |
| Host Staged | Host + DMA | GPU SM 低，但路径远 | P2P 不可用 fallback |

---

## 05.3 LSU vs TMA

### LSU

```text
Global
 ↓
L2/L1
 ↓
Register
 ↓
Shared Memory
```

特点：

- 多条 load/store 指令；
- 地址计算由 SM 完成；
- 数据经过 Register File；
- SM 持续参与。

### TMA

抽象：

```text
SM：描述搬运任务
       ↓
TMA：自主搬运
       ↓
Shared Memory
```

特点：

- 异步；
- 适合块状/多维 Tensor 搬运；
- 降低 Register 与指令压力；
- 存在固定启动开销，因此仍符合 `α + S/β`。

> 具体尺寸阈值必须以架构和 microbenchmark 为准，不能把某一型号的经验值当成永久常数。

---

## 05.4 mbarrier：控制面也要卸载

问题：

```text
TMA 已经在搬
但 Warp 一直 polling
```

那么控制面仍浪费执行资源。

mbarrier 提供：

- 细粒度到达计数；
- phase / generation 语义；
- 等待与唤醒机制；
- 与异步数据搬运形成 producer-consumer pipeline。

典型 Ping-Pong：

```text
Time →

Buffer A:  Compute █████    TMA Write █████
Buffer B:  TMA Write █████  Compute █████
```

---

## 05.5 Copy Engine

Copy Engine 适用于：

```text
cudaMemcpyAsync
P2P Copy
D2H / H2D
Prefetch / Offload
```

优势：

```text
不占主要 SM 执行资源
可与 Compute 并发
```

限制：

```text
纯搬运擅长
复杂归约/算术不适合
Kernel 内细粒度控制能力有限
```

---

## 05.6 Activation Offloading

要真正让 D2H/H2D 搬运有效 overlap，至少检查：

```text
Async Stream
Pinned Memory
NUMA Affinity
```

选择 Offload 还是 Recomputation，本质比较：

```text
Recompute Cost
vs
Transfer Cost
```

---

## 05.7 CUDA IPC

多进程单机训练中，每个进程拥有独立虚拟地址空间。

CUDA IPC 用于：

```text
Process A GPU Allocation
   ↓ cudaIpcGetMemHandle
Handle
   ↓ IPC
Process B
   ↓ cudaIpcOpenMemHandle
Remote GPU Memory Mapping
```

高频场景应尽量避免重复创建/销毁 Handle。

---

## 🧪 Lab 05：机内 Data Movement

测试：

```text
GPU0 ↔ GPU1
GPU0 ↔ GPU4
GPU ↔ Host
```

工具：

```text
p2pBandwidthLatencyTest
nvbandwidth
Nsight Systems
```

回答：

- 实际走 NVLink 还是 PCIe？
- Copy 是否占 SM？
- 与 GEMM 同时运行时谁变慢？

---

# 06｜机间数据搬运
## CPU-controlled vs GPU-initiated RDMA / Two-hop vs One-hop

---

## 06.1 GDR 只统一了数据面

一旦使用 GPUDirect RDMA，主要数据路径都是：

```text
GPU Memory
 ↓
NIC DMA
 ↓
Network
 ↓
Remote NIC DMA
 ↓
Remote GPU Memory
```

但发起 RDMA 操作仍需要：

```text
Build WQE
Ring Doorbell
```

问题变成：

> **谁负责控制面？**

---

## 06.2 CPU-controlled RDMA

典型抽象：

```text
GPU / Application
 ↓
CPU builds WQE
 ↓
NIC
 ↓
RDMA Data Path
```

优势：

- 生态成熟；
- 大包情况下控制面开销容易被传输时间掩盖；
- 稳定。

适合：

```text
Bandwidth-oriented large messages
```

---

## 06.3 GPU-initiated RDMA

抽象：

```text
GPU SM
 ↓ build WQE / doorbell
NIC
 ↓
RDMA
```

优点：

- 缩短控制路径；
- 避免 CPU 调度路径；
- 小包 latency 更有优势；
- 可提高并发发起能力。

代价：

- GPU 侧需要承担部分控制逻辑；
- 实现复杂；
- 对硬件、驱动、NIC 能力依赖更强。

---

## 06.4 两跳聚合 vs 一跳直达

### 两跳聚合

```text
GPU1 ─┐
GPU2 ─┼─ NVLink → Gateway GPU → NIC → Remote Gateway
GPU3 ─┘
```

核心思想：

> **先用更“便宜”的机内高速链路聚合，再让稀缺 NIC 处理少量大包。**

优点：

- 更容易打满 NIC；
- 减少包数量；
- 减少 cross-rail；
- 减少 incast 风险。

适合：

```text
Training
Prefill
Large-batch MoE
Bandwidth-bound
```

### 一跳直达

```text
GPU → NIC → Remote GPU
```

优点：

```text
少一层中转
最低路径延迟
```

适合：

```text
Decode
Small-message MoE
Latency-bound
```

---

## 06.5 Warp 角色分工

高级通信 Kernel 中，不一定所有 Warp 做同一件事。

可以按角色划分：

```text
NVLink Receiver
NVLink Sender
RDMA Sender
RDMA Receiver
Forwarder
Local Handler
```

这是一种 **Warp Specialization**：

> 让不同 Warp 同时驱动不同硬件通道，使 NVLink、NIC、计算资源形成流水，而不是串行执行。

---

## 06.6 “单边通信退化”检查

如果实现逻辑变成：

```text
Put Data
 ↓
等待对端 ACK
 ↓
Put Signal
```

那么从端到端控制语义上已经重新引入 rendezvous / request-response。

检查单边实现时必须问：

> **发送方下一步是否依赖接收方主动响应？**

---

# 07｜NCCL / NVSHMEM / Collective Communication

### 学习目标

将上层 Collective 与底层 Data Path 连接起来。

---

## 07.1 NCCL 在软件栈中的位置

```text
PyTorch
 ↓
torch.distributed
 ↓
ProcessGroupNCCL
 ↓
NCCL
 ↓
Algorithm / Protocol / Channel
 ↓
Transport
 ↓
NVLink / PCIe / RDMA
```

NCCL 不是“网络本身”，而是：

> **面向 GPU 的高性能通信运行时 / Collective Library。**

---

## 07.2 Collective 必会语义

### AllReduce

```text
A  B  C  D
↓  ↓  ↓  ↓
A+B+C+D
```

每个 Rank 都得到结果。

### ReduceScatter

```text
Reduce + Scatter
```

### AllGather

```text
各 Rank 的 shard
 ↓
每个 Rank 得到完整集合
```

### AllToAll

```text
每个 Rank
向每个 Rank
发送不同 shard
```

### Send / Recv

点对点通信。

---

## 07.3 Ring AllReduce 通信量

设：

```text
P = Rank 数
N = Tensor Size
```

Ring AllReduce 每 Rank 近似发送：

\[
2\frac{P-1}{P}N
\]

当 \(P\) 很大：

\[
\approx 2N
\]

必须理解：

```text
AllReduce ≈ ReduceScatter + AllGather
```

是 Ring 的经典结构。

---

## 07.4 Ring vs Tree

### Ring

优势：

```text
Bandwidth Efficient
```

### Tree

优势：

```text
Fewer logical steps
Lower latency for suitable message sizes
```

选择依赖：

```text
Message Size
Rank Count
Topology
Protocol
```

不能形成“Ring 永远更快”的错误记忆。

---

## 07.5 Algorithm BW vs Bus BW

Benchmark 时必须区分：

```text
algBW
busBW
```

它们回答不同问题：

- algBW：上层算法处理有效数据的速度；
- busBW：底层链路承担数据流量的等效速度。

---

## 07.6 NVSHMEM

NVSHMEM 提供：

```text
Symmetric Memory
One-sided Put/Get
Device-side Communication
```

典型：

```text
put(data)
 ↓
ordering / fence
 ↓
put(flag)
```

接收侧：

```text
wait(flag)
 ↓
consume(data)
```

NVSHMEM 的关键价值不是“换一个 API”，而是：

> **让 GPU Kernel 可以直接发起细粒度 RMA，并支持更紧密的通信计算融合。**

---

## 07.7 Parallelism ↔ Collective

| Parallelism | 主要数据 | 典型通信 |
|---|---|---|
| DP | Gradients | AllReduce |
| FSDP | Parameters / Gradients | AllGather / ReduceScatter |
| TP | Activations / Partial Results | AllGather / ReduceScatter / AllReduce |
| PP | Activations / Gradients | Send / Recv |
| EP | Tokens | AllToAll |
| CP | KV / Sequence Shards | P2P / A2A / AG 等，依实现而定 |

这张表必须形成条件反射。

---

## 🧪 Lab 06：nccl-tests

建议测试：

```text
all_reduce_perf
all_gather_perf
reduce_scatter_perf
alltoall_perf
```

测试维度：

```text
Message Size
Rank Count
Single-node / Multi-node
Channels
Protocol
Topology
```

输出：

```text
Message Size → algBW / busBW 曲线
```

---

# 08｜Compute-Communication Overlap
## Stream / Bucket / Chunk / Warp Specialization / Persistent Kernel

---

## 08.1 Overlap 的目标

串行：

```text
Compute      ███████████
Communication           ███████
```

Overlap：

```text
Compute      ███████████████
Communication     ███████
```

真正要优化：

```text
Exposed Communication
```

---

## 08.2 Async ≠ Overlap

下面不一定产生真正 overlap：

```python
async_op=True
```

因为还要满足：

1. Dependency 允许；
2. Communication 足够早启动；
3. 有可覆盖的 Compute Window；
4. 两者资源竞争不能太严重；
5. 同步点不能过早出现。

---

## 08.3 DDP Gradient Bucket

```text
Backward Layer N
      ↓ grad ready
Bucket Ready
      ↓
Async AllReduce

与此同时：
Backward Layer N-1
```

Bucket trade-off：

### 太大

```text
启动晚
Overlap Window 变小
```

### 太小

```text
小消息多
α 开销大
Bandwidth 利用率下降
```

---

## 08.4 FSDP Prefetch

目标：

```text
Compute Layer N
        │
        └── overlap → AllGather Layer N+1
```

关键问题：

- Prefetch 是否足够提前？
- AG 是否与 GEMM 争 SM / HBM / PCIe / NIC？
- Prefetch 太早是否增加显存占用？

---

## 08.5 Chunking / Pipeline

把：

```text
Large Tensor
████████████████████
```

切成：

```text
████ ████ ████ ████
```

形成：

```text
Compute Chunk 1
      ↓
Comm Chunk 1

Compute Chunk 2
      ↓
Comm Chunk 2
```

减少“等整个 Tensor ready”的等待。

---

## 08.6 Warp Specialization

在同一个 Persistent / Fused Kernel 中：

```text
Producer Warp
   ↓ data movement
Shared Buffer
   ↓ barrier
Consumer Warp
   ↓ GEMM / MMA
```

也可以扩展为：

```text
Communication Warp
Compute Warp
Synchronization Warp
Forwarding Warp
```

优势：

- 更细粒度 overlap；
- 减少 Kernel Launch / Global Sync；
- 能显式控制角色和资源。

代价：

- 强硬件相关；
- 同步复杂；
- Register / SM / Shared Memory 预算更困难。

> 现代 GPU 可以并发执行多个 Kernel，但跨 Kernel 并发的资源分配和同步粒度通常不如融合 Kernel 内的角色划分可控。因此“融合”不是因为多 Kernel 完全无法并发，而是为了获得更细粒度、可预测的流水和资源控制。

---

## 08.7 Persistent Kernel

Persistent Kernel 的核心不是“Kernel 越长越好”，而是：

```text
减少 Launch / Dispatch
保持工作状态
持续消费任务
实现内部流水线
```

需要关注：

- 是否长期占住 SM；
- 是否影响其他 Kernel；
- 是否需要限制 SM 配额；
- 是否存在 polling 浪费。

---

## 08.8 SM 配额与饱和点

通信 SM 数：

```text
少 → 通信打不满
多 → 挤压计算
```

因此性能往往呈：

```text
Throughput
  ▲
  │            ───────────
  │         ──
  │      ──
  │   ──
  │__──────────────────────→ Communication SMs
             saturation
```

工程原则：

> **通信资源的饱和点必须测，不要靠猜。**

可以调：

```text
NCCL Channels
Communication CTA count
SM margin
Dedicated SM quota
```

---

## 08.9 SM-free 是长期方向

理想状态：

```text
Compute → SM / Tensor Core
Communication Data Movement → CE / TMA / NIC DMA
Synchronization → Hardware Event / Barrier
```

这样才能最大程度降低：

```text
k → 1
```

---

# 09｜Benchmark、监控与 AI 集群故障诊断

### 核心目标

看到“训练慢了”，不直接说：

> 网络有问题。

而是逐层建立证据。

---

## 09.1 第一层：Step Time 分解

```text
Step Time
  ├── Compute
  ├── Communication
  ├── Exposed Communication
  ├── Data Loader
  ├── Synchronization
  └── Bubble / Idle
```

---

## 09.2 第二层：通信分类

```text
Communication Slow
       │
       ├── AllReduce
       ├── AllGather
       ├── ReduceScatter
       ├── AllToAll
       └── P2P
```

继续问：

```text
Message Size?
Intra-node / Inter-node?
Which ranks?
Which topology path?
```

---

## 09.3 第三层：Data Path

### Intra-node

检查：

```text
NVLink
NVSwitch
PCIe
P2P
Topology
SM contention
HBM contention
```

### Inter-node

检查：

```text
GPU-NIC affinity
GDR
RDMA
QP
PCIe
NIC
IB / RoCE
Leaf / Spine
Rail
Congestion
```

---

## 09.4 Slow Rank

同步分布式系统：

```text
Step Time ≈ max(rank_i completion time)
```

因此一个 Slow Rank 可以拖慢整个 Job。

来源可能是：

```text
GPU
ECC / Xid
Thermal / Power
CPU
NUMA
NIC
Network
Storage
Background Job
```

---

## 09.5 NCCL Hang 排查顺序

建议从软件正确性到硬件逐层：

```text
Collective sequence mismatch
        ↓
Shape / Count mismatch
        ↓
Rank failure
        ↓
CUDA / GPU error
        ↓
NCCL transport
        ↓
RDMA / NIC
        ↓
Network / Switch
```

不要一上来抓网络包。

---

## 09.6 推荐工具栈

### GPU / CUDA

```text
nvidia-smi
DCGM
Nsight Systems
Nsight Compute
PyTorch Profiler
```

### Topology

```text
nvidia-smi topo -m
lspci
numactl
```

### RDMA / NIC

```text
ibstat
ibv_devinfo
ibdev2netdev
ib_write_bw
ib_read_bw
```

### Collective

```text
nccl-tests
NCCL_DEBUG
```

---

# 🛠️ 5. 本模块六个核心实验

## Experiment A｜单卡 Roofline

目标：

```text
判断 Compute-bound / Memory-bound
```

---

## Experiment B｜通信 α+S/β

目标：

```text
找到 Latency-bound → Bandwidth-bound 转折区间
```

---

## Experiment C｜GPU Topology

目标：

```text
把 nvidia-smi topo -m 转成物理 Data Path
```

---

## Experiment D｜RDMA

目标：

```text
验证 Host / NUMA / QP / Message Size 对网络性能的影响
```

---

## Experiment E｜NCCL

目标：

```text
比较 AR / AG / RS / A2A 的 algBW / busBW
```

---

## Experiment F｜Overlap

至少测四组：

```text
1. Compute Solo
2. Communication Solo
3. Serial Compute + Comm
4. Overlapped Compute + Comm
```

记录：

\[
T_{compute}
\]

\[
T_{comm,solo}
\]

\[
T_{comm,overlap}
\]

\[
k = \frac{T_{comm,overlap}}{T_{comm,solo}}
\]

\[
T_{exposed}
\]

最终回答：

> **这个 overlap 到底隐藏了多少通信，又牺牲了多少计算？**

---

# 📊 6. Ringi 统一性能分析表

以后分析任何 AI Infra 性能问题，优先填写这张表。

| 类别 | 指标 | 数值 / 结论 |
|---|---|---|
| Tensor | Shape | |
| Tensor | dtype | |
| Tensor | Size | |
| Compute | FLOPs | |
| Compute | Kernel Type | |
| Compute | Arithmetic Intensity | |
| Memory | HBM Bytes | |
| Memory | HBM BW | |
| Parallel | DP / TP / PP / EP / CP | |
| Communication | Collective | |
| Communication | Volume / Rank | |
| Communication | Message Size | |
| Communication | α estimate | |
| Communication | Effective BW | |
| Topology | Intra / Inter Node | |
| Topology | NVLink / PCIe / RDMA | |
| Topology | GPU-NIC Affinity | |
| Network | Rail / Clos / Pod | |
| Overlap | Compute Time | |
| Overlap | Comm Solo Time | |
| Overlap | Comm Overlap Time | |
| Overlap | k | |
| Overlap | Hidden Comm | |
| Overlap | Exposed Comm | |
| End-to-End | Step Time | |
| End-to-End | MFU / Throughput | |

---

# 🧩 7. 并行策略与硬件层级映射

```text
                   Transformer
                        │
        ┌───────────────┼────────────────┐
        ▼               ▼                ▼
       TP              DP/FSDP           PP
        │               │                │
  AG / RS / AR      AG / RS / AR      Send / Recv
        │               │                │
        ▼               ▼                ▼
     NVLink        RDMA / Rail        NVLink / RDMA
        │
        └──────────────────────┐
                               ▼
                              EP
                               │
                            AllToAll
                               │
                    Cross-rail / NVLink Domain
```

核心原则：

> **通信越频繁、越处于计算关键路径，就越应该放在更高带宽、更低延迟的局部通信域。**

---

# 🎤 8. 大厂面试 / 口试检验题

## GPU

- SM、Warp、CTA 的关系是什么？
- Tensor Core 和 CUDA Core 分别解决什么问题？
- 为什么 HBM 带宽很高仍可能出现 Memory-bound？
- Arithmetic Intensity 是什么？
- GPU Util 100% 为什么不代表效率高？

## Data Movement

- 为什么说“通信不仅是跨机网络”？
- “能不搬就不搬”有哪些实际例子？
- 为什么 TMA 可以降低 SM 干扰？
- Copy Engine 和 TMA 的使用边界是什么？
- 为什么数据面 SM-free 还不够？

## Topology

- NVLink 与 NVSwitch 的区别？
- PCIe 在 AI 服务器里承担哪些角色？
- 为什么 GPU-NIC Affinity 会影响 RDMA？
- 什么是 Rail-parallel / Cross-rail？
- 为什么错误 rank mapping 可以让 non-blocking 网络表现得像 oversubscription？

## RDMA

- DMA 与 RDMA 什么关系？
- QP / WQE / CQ / MR / PD 分别解决什么问题？
- 为什么需要 Memory Registration？
- 单边与双边通信的本质区别？
- RDMA Write 后为什么不能随便再发一个 flag 就认为安全？
- GPUDirect RDMA 到底绕过了什么？没有绕过什么？

## Collective

- AllReduce、AllGather、ReduceScatter 的 Shape 如何变化？
- Ring AllReduce 每个 Rank 大约传多少数据？
- Ring 和 Tree 的适用场景？
- algBW 与 busBW 有什么区别？
- 为什么 MoE 的 AllToAll 特别容易产生 cross-rail 和 incast？

## Overlap

- Async 为什么不等于 overlap？
- 什么是 Exposed Communication？
- DDP Bucket 太大/太小分别有什么问题？
- 为什么 overlap 后通信和 GEMM 都可能变慢？
- `k` 倍率如何测？
- Warp Specialization 为什么适合通算融合？
- 为什么 Persistent Kernel 不是无条件越多越好？
- 什么叫通信 SM 饱和点？

## Troubleshooting

- 单机快、多机慢，第一轮检查什么？
- NCCL Hang 如何分层排查？
- 如何判断 Slow Rank？
- 如何判断 PCIe 是否成为瓶颈？
- 如何证明问题是网络拥塞而不是 GPU-NIC affinity？

---

# ✅ 9. 模块验收标准

## Level 1｜能解释

- [ ] 能解释 GPU 执行和存储层级。
- [ ] 能解释 NVLink / NVSwitch / PCIe / RDMA。
- [ ] 能解释 QP / WQE / MR / CQ。
- [ ] 能解释 NCCL 常见 Collective。
- [ ] 能解释 Compute-Communication Overlap。

## Level 2｜能计算

- [ ] 能算 Tensor Size。
- [ ] 能算 FLOPs。
- [ ] 能算 Arithmetic Intensity。
- [ ] 能用 `α + S/β` 估通信时间。
- [ ] 能推导 Ring AllReduce 通信量。
- [ ] 能估算 Exposed Communication。

## Level 3｜能测

- [ ] 会使用 `nvidia-smi topo -m`。
- [ ] 会做 P2P Benchmark。
- [ ] 会跑 RDMA Benchmark。
- [ ] 会跑 nccl-tests。
- [ ] 会用 Nsight Systems 看通信/计算 Timeline。

## Level 4｜能诊断

面对：

```text
训练速度突然下降
```

能够完成：

```text
Step decomposition
 ↓
Compute / Communication
 ↓
Collective
 ↓
Message Size
 ↓
Data Path
 ↓
Topology
 ↓
Transport
 ↓
Resource Contention
 ↓
Slow Rank / Network / GPU Root Cause
```

## Level 5｜能设计

- [ ] 能为 TP / DP / PP / EP 选择合理通信域。
- [ ] 能设计 Bucket / Chunk / Prefetch overlap。
- [ ] 能根据训练与 Decode 场景选择 bandwidth-first 或 latency-first 策略。
- [ ] 能设计实验找到通信 SM 饱和点。
- [ ] 能说明某项优化究竟减少了 `α`、减少了 `S`、提高了 `β`，还是减少了 `T_exposed`。

---

# 🧪 10. 课程最终综合题

给定一个：

```text
8 Nodes × 8 GPUs
TP=8
PP=4
DP=16
MoE EP enabled
```

要求完成：

1. 画出并行组；
2. 给每个维度标出主要 Collective；
3. 计算一个指定 Tensor 的通信量；
4. 判断哪些通信应放 NVLink Domain；
5. 判断哪些通信会走 RDMA；
6. 判断哪些流量可能 Cross-rail；
7. 给出 GPU-NIC / Rank Placement 原则；
8. 用 `α + S/β` 分析 Training 和 Decode；
9. 设计至少两种 overlap；
10. 设计 nccl-tests / RDMA / Nsight 验证方案；
11. 构造一个 Slow Rank 并定位；
12. 最后给出完整性能优化报告。

验收报告必须包含：

```text
Shape
FLOPs
Bytes
Topology
Data Path
Collective
Communication Volume
Measured Bandwidth
Overlap Window
Exposed Communication
Bottleneck
Optimization
Before / After Evidence
```

---

# 📚 11. 本模块与五篇通信文章的对应关系

本模块吸收并重组以下材料，但不是按原文章节简单复述，而是按课程学习依赖重新组织。

| 原始材料 | 本课程主要落点 |
|---|---|
| 《从零开始的通信计算 overlap【第一章】》 | GPU/SM/Warp、NVSHMEM、Warp Specialization、通算融合、SM 资源饱和 |
| 《大模型通信基础 2.1：通信硬件拓扑》 | 通信不可能三角、α+S/β、带宽阶梯、NVLink/NVSwitch、PCIe、Rail、Clos、Rank Mapping |
| 《大模型通信基础 2.2：RDMA 核心概念》 | DMA、QP/WQE/CQ/MR/PD、单边/双边、Ordering、GPUDirect RDMA |
| 《大模型通信基础 2.3：机内数据搬运》 | LSU、TMA、Copy Engine、mbarrier、SM-free、k 倍率、IPC、Offload |
| 《大模型通信基础 2.4：机间数据搬运》 | CPU/GPU 控制面、两跳聚合、一跳直达、Warp 角色分工、MoE 通信路径 |

原始链接：

1. https://zhuanlan.zhihu.com/p/2011564057396809841
2. https://zhuanlan.zhihu.com/p/2028907020917449344
3. https://zhuanlan.zhihu.com/p/2028907599861495146
4. https://zhuanlan.zhihu.com/p/2028907936030704604
5. https://zhuanlan.zhihu.com/p/2028908577935336722

---

# ⚠️ 12. 工程阅读注意事项

## 12.1 原理与规格分开

课程中的长期稳定知识：

```text
α + S/β
Roofline
Data / Control Plane
Bandwidth Hierarchy
Topology-aware Placement
One-sided Semantics
SM-free
Overlap / Resource Contention
```

而下面内容属于架构/版本相关信息：

```text
SM 数量
HBM 带宽
NVLink 带宽
NIC 线速
TMA 具体能力
NCCL 默认算法
IBGDA / RMA 特性
```

实验时必须以当前：

```text
GPU Architecture
CUDA Version
Driver
NCCL Version
NIC Firmware
Network Topology
```

为准。

---

## 12.2 不背 Benchmark 常数

例如：

```text
某实现 20 SM 最优
某消息 2 KB 后 TMA 更快
某集群 50 GB/s
```

这些只能作为 Case Study。

正确学习方式：

> **理解为什么存在 crossover / saturation，然后在自己的机器上测出 crossover / saturation。**

---

# 🔗 13. 与后续 Module 的接口

```text
Module 01
GPU + Data Movement + Communication
                │
                ▼
Module 02
Transformer Compute / Memory Cost
                │
                ▼
Module 03
Data Parallel / DDP / FSDP / ZeRO
                │
                ▼
Module 04
Tensor / Sequence / Context Parallel
                │
                ▼
Module 05
Pipeline Parallel
                │
                ▼
Module 06
MoE / Expert Parallel / DeepEP
                │
                ▼
Module 07+
Megatron / DeepSpeed / Distributed Inference
```

之后遇到任何新并行技术，都重新套 Module 01 四问：

```text
Shape 是什么？
钱花在哪里？
机器怎么跑？
数据怎么搬？
```

---

# 🏁 14. Module 01 最终一句话

> **AI Infra 性能优化，本质上是在管理 Tensor 的“计算”和“搬运”：决定哪些数据根本不该搬、哪些必须搬、搬多少、从哪里到哪里、由谁搬、占什么硬件资源，以及这些搬运最终有多少真正暴露在关键路径上。**

---

*返回主目录：[《Ringi AI Infra 学习体系全景地图》](../README.md)*
