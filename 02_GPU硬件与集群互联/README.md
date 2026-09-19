# 🚀 第二篇：GPU 硬件微架构、数据搬运与集群互联 (GPU Architecture & Interconnect)

> **主讲人**：👓 **Ringi**（大厂 AI Infrastructure 工程师）  
> **模块定位**：必修 ｜ **建议时长**：16–20h ｜ **先修要求**：[Module 00: 性能工程与系统前置](../01_性能工程与系统前置/README.md)  
> **核心关键词**：SM / Warp / Tensor Core / HBM / Roofline / PCIe / NVLink / NVSwitch / RDMA / GPUDirect RDMA / InfiniBand / RoCE / NCCL / NVSHMEM / Collective / TMA / mbarrier / Warp Specialization / Compute-Communication Overlap
>
> **模块愿景**：告别死记硬背通信名词，建立一套从 **Tensor Shape 一路追踪到硬件物理 Data Path**、能定量计算瓶颈、精细设计 Overlap、熟练进行压测与大集群故障定位的 AI Infra 工业级心智模型！

---

## 🎯 0. 为什么必须这样学 GPU 与集群通信？

大模型分布式训练与在线推理的性能瓶颈，在系统底层最终都可以精确归结为两个物理动作：

1. **计算（Compute）**：Tensor 在 CUDA Core / Tensor Core 运算单元上执行浮点运算；
2. **搬运（Data Movement / Communication）**：Tensor 在 Register、Shared Memory、L2 Cache、HBM、GPU 之间、NIC 以及跨节点网络之间移动。

```
                    ┌────────────────────────────────────────────────────────┐
                    │               AI Infra 广义数据搬运层级                │
                    └───────────────────────────┬────────────────────────────┘
                                                │
       ┌──────────────────┬─────────────────────┼─────────────────────┬──────────────────┐
       ▼                  ▼                     ▼                     ▼                  ▼
【SM 内部寄存器/SRAM】  【卡内存储层次】      【机内卡间互连】      【节点与网卡直通】    【跨机集群网络】
Register ↔ Shared Mem   Shared Mem ↔ L2 ↔ HBM  GPU ↔ NVLink ↔ GPU    GPU ↔ PCIe ↔ NIC     NIC ↔ Clos ↔ NIC
(0.5ns, 几十 TB/s)      (1~2ns, 3~8 TB/s)      (100ns, 900~1800GB/s) (微秒级, 64~128GB/s) (微秒级, 400~800Gbps)
```

本模块坚守的**通信第一性原理**：

> 💡 **“能不搬就不搬；能少搬就少搬；必须搬，就走最近、最宽、最适合当前消息规模的路径；最后再考虑如何把搬运彻底隐藏在计算后面（Overlap）。”**

---

## 🧭 1. Ringi 四问教学协议

学习本模块的每一个硬件结构、通信原语与并行策略时，必须贯穿以下四个问题：

```text
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ 1. 📐           │     │ 2. 💰           │     │ 3. ⚙️           │     │ 4. 🚚           │
│ Shape 是什么？  │ ──► │ 钱花在哪里？    │ ──► │ 机器上怎么跑？  │ ──► │ 数据到底怎么搬？│
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
 追踪 Tensor 形状变化    量化 FLOPs/带宽/显存    分析指令/Kernel/同步   绘制物理 Data Path
```

1. **📐 Shape 是什么？**：追踪 Tensor 的物理维度变化（ $[B, S, H] \to [B, S, H/TP]$ ），明确切分轴与通信前后的语义。
2. **💰 钱花在哪里？**：定量手算参数量、FLOPs、HBM 访存量、通信数据量，判断是 Compute-bound、Memory-bound 还是 Communication-bound。
3. **⚙️ 机器上怎么跑？**：穿透 PyTorch、NCCL、CUDA Kernel、Copy Engine 到 DMA 引擎，明确谁发起、谁等待、占不占 SM。
4. **🚚 数据到底怎么搬？**：在脑海中展开完整的硬件物理路径（HBM $\to$ LSU/TMA $\to$ PCIe/NVLink $\to$ NVSwitch $\to$ NIC $\to$ IB/RoCE $\to$ 远端 HBM）。

---

## 🧠 2. 本模块统一性能模型与四大核心公式

### 1. 通信耗时基础模型： $T = \alpha + \frac{S}{\beta}$
- $\alpha$（固定时延/握手开销）：小消息（ $S \to 0$ ）主导，属于 **Latency-bound**（如 Decode 逐字生成、MoE 稀疏 Dispatch）；
- $\frac{S}{\beta}$（传输带宽开销）：大消息（ $S \gg 1\text{MB}$ ）主导，属于 **Bandwidth-bound**（如 DDP 反向梯度 AllReduce、FSDP AllGather）。

### 2. 单算子算力与访存边界：Roofline 模型
$$
P = \min\left(P_{\text{peak}}, \text{BW}_{\text{HBM}} \times \text{AI}\right), \quad \text{Arithmetic Intensity (AI)} = \frac{\text{FLOPs}}{\text{Bytes}}
$$

### 3. 真实暴露通信耗时：Exposed Communication
$$
T_{\text{step}} = T_{\text{compute}} + T_{\text{exposed-comm}} = T_{\text{compute}} + \max(0, T_{\text{comm}} - T_{\text{compute-overlap}})
$$

### 4. Overlap 惩罚因子模型： $k \ge 1.0$
$$
\text{实际总耗时 } T_{\text{total}} = \max\left(k_{\text{comp}} \cdot T_{\text{compute}}, \, k_{\text{comm}} \cdot T_{\text{comm}}\right)
$$

当通信与计算并发争抢 SM、L2 Cache 或 HBM 带宽时，两者速度都会下降（ $k > 1$ ）。

---

## 🔺 3. 通信的不可能三角

```text
                  高带宽 (Bandwidth)
                         ▲
                        / \
                       /   \
                      /     \
                     /       \
                    ▼─────────▼
       低延迟 (Latency)      少占 SM (SM-free)
```

- **训练任务（Training）**：追求 **高带宽 + Overlap 隐藏**；
- **推理任务（Decode）**：追求 **极致低延迟（<10μs）**；
- **现代架构演进（Hopper/Blackwell）**：通过 **TMA、mbarrier、GPU-initiated RDMA** 走向 **SM-free** 卸载。

---

## 📑 4. 本模块章节全景导航（10 大核心篇章）

| 序号 | 章节名称 | 核心知识点与第一性原理 | 关键产出 | 建议时长 | 文档链接 |
|:---:|:---|:---|:---|:---:|:---:|
| **12** | **GPU 执行与存储体系架构** | SM、Warp 调度、Tensor Core、Memory Hierarchy、Arithmetic Intensity 与 Roofline 边界 | Roofline 手算 + Kernel 分类 | 2h | [`12_GPU执行与存储体系...md`](./12_GPU执行与存储体系_SM_Warp_TensorCore_HBM_Roofline.md) |
| **13** | **通信第一性原理与性能模型** | 并行通信根源、 $\alpha+S/\beta$ 模型、通信优化五层级、分层物理带宽瓶颈 | 通信耗时估算模型 | 1.5h | [`13_通信第一性原理...md`](./13_通信第一性原理与性能模型_AlphaBeta_Latency_Bandwidth.md) |
| **14** | **GPU 节点与集群硬件拓扑** | PCIe 拓扑、NUMA 亲和性、NVLink/NVSwitch、Rail-Optimized 组网与 Rank Placement | 拓扑图 + Rank 映射表 | 2h | [`14_GPU节点与集群硬件拓扑...md`](./14_GPU节点与集群硬件拓扑_PCIe_NVLink_NVSwitch_Clos_Rail.md) |
| **15** | **RDMA 与 GPUDirect RDMA** | TCP/IP 缺陷、控制面与数据面分离、QP/WQE/CQ/MR 原理、Zero-Copy 旁路 CPU | RDMA 时序图与抓包剖析 | 2h | [`15_RDMA与GPUDirect_RDMA...md`](./15_RDMA与GPUDirect_RDMA_QP_WQE_CQ_MR_ZeroCopy.md) |
| **16** | **机内数据搬运机制与硬件卸载** | LSU vs TMA、Copy Engine、mbarrier 硬件同步、CUDA IPC 与 Activation Offloading | SM-free 路径比对表 | 2h | [`16_机内数据搬运...md`](./16_机内数据搬运_LSU_TMA_CopyEngine_mbarrier_IPC.md) |
| **17** | **机间数据搬运与网络架构** | CPU-controlled vs GPU-initiated RDMA、两跳聚合 vs 一跳直达、IBRC 调优 | 跨机数据流决策树 | 1.5h | [`17_机间数据搬运...md`](./17_机间数据搬运_GPU_Initiated_RDMA_IBRC_OneHop.md) |
| **18** | **NCCL 通信库与 Collective 算子** | Ring/Tree 算法推导、NVLS 硬件加速、Algorithm BW vs Bus BW 换算与并行映射 | nccl-tests 压测报告 | 2h | [`18_NCCL通信库与Collective算子...md`](./18_NCCL通信库与Collective算子_Ring_Tree_NVLS_NVSHMEM.md) |
| **19** | **Compute-Communication Overlap** | CUDA Stream、DDP Bucket、FSDP Prefetch、Warp Specialization 与 Overlap 惩罚因子 | Nsight Timeline 分析 | 2h | [`19_Compute_Communication_Overlap...md`](./19_Compute_Communication_Overlap_Stream_Bucket_Chunk_WarpSpec.md) |
| **20** | **Benchmark、监控与 AI 集群故障诊断** | Step Time 三层分解、慢节点（Slow Node）定位、NCCL Hang 排查四步法与网络监控 | 生产故障排查决策树 | 2h | [`20_Benchmark_监控与AI集群故障诊断...md`](./20_Benchmark_监控与AI集群故障诊断_Xid_SlowNode_NCCL_Hang.md) |
| **21** | **GPU 硬件全景与多元异构算力** | NVIDIA V100/A100/H100/B200 代际演进、华为昇腾/海光/寒武纪等国产生态全景与一云多芯选型 | 芯片规格大横评 + 迁移五步法 | 2.5h | [`21_GPU硬件全景与多元异构算力...md`](./21_GPU硬件全景与多元异构算力_NVIDIA代际演进_国产芯片与生态选型.md) |

---

## 🛠️ 5. 本模块六大动手实战实验（Labs）

```text
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ 🧪 Lab 01    │   │ 🧪 Lab 02    │   │ 🧪 Lab 03    │   │ 🧪 Lab 04    │   │ 🧪 Lab 05    │   │ 🧪 Lab 06    │
│ Roofline 分类│──►│ α+S/β 估算   │──►│ 拓扑 Rank 映射│──►│ RDMA 性能压测 │──►│ 机内数据搬运  │──►│ nccl-tests   │
└──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
```

1. **Lab 01（Kernel Roofline 分析）**：手算 GEMM 与 Softmax 的 AI 值，绘制 Roofline 曲线定位瓶颈。
2. **Lab 02（ $\alpha + S/\beta$ 通信模型实测）**：使用不同 Message Size 拟合单机与跨机 $\alpha$ 和 $\beta$ 参数。
3. **Lab 03（Topology Mapping 与 Rank 映射）**：通过 `nvidia-smi topo -m` 探测拓扑，设计 TP/DP/PP 的最优 Rank Placement。
4. **Lab 04（RDMA 基准测试）**：运行 `ibv_rc_pingpong`，比对 RDMA Write vs Send 的带宽与延迟差异。
5. **Lab 05（机内 Data Movement 评测）**：使用 Nsight Compute 观察 LSU vs TMA 搬运对 SM Occupancy 的影响。
6. **Lab 06（nccl-tests 深度实战）**：在单机 8 卡与跨机集群运行 `all_reduce_perf`，计算 Bus Bandwidth 并定位网络瓶颈。

---

## 🧩 6. 并行策略与硬件层级映射矩阵

| 并行策略 | 产生通信的切分轴 | 核心通信算子 | 推荐物理承载层级 | 核心瓶颈特征 |
|---|---|---|---|---|
| **TP（张量并行）** | 隐藏层维度 / 头维度 | `AllReduce` / `ReduceScatter + AllGather` | **NVLink 机内高速通道** | Bandwidth-bound，严禁跨机 |
| **PP（流水线并行）** | 模型层数维度 | 点对点 `P2P (Send/Recv)` | **PCIe / 跨机 RDMA** | Bubble 气泡率与 P2P 延迟 |
| **DP / DDP** | Batch 维度 | 梯度 `AllReduce` | **跨机 RDMA / IB / RoCE** | 可与反向计算完全 Overlap |
| **ZeRO-3 / FSDP** | 权重/梯度/优化器分片 | 前向/反向 `AllGather + ReduceScatter` | **跨机 RDMA 高带宽网络** | 通信量为 DDP 1.5 倍，极度依赖 Overlap |
| **CP（上下文并行）** | 序列长度维度 (Sequence) | `AllGather / P2P Ring Attention` | **机内 NVLink 或同 Rail 跨机** | 随上下文增长通信频次激增 |
| **EP（专家并行/MoE）** | Token 路由分配维度 | `All-to-All` (Dispatch & Combine) | **全网跨机 bisection 带宽** | 多对多 Incast 拥塞，Latency-bound |

---

## 🎯 7. 模块通关验收标准（Checklist）

完成本模块学习后，你必须能够独立达成以下 9 项硬核能力：

- [ ] **L1**：能在白板上画出 GPU 存储层级与 SM 执行流水线，推导任意算子的 Roofline 瓶颈。
- [ ] **L2**：熟练运用 $\alpha + S/\beta$ 公式，准确手算不同消息体量下的理论通信耗时。
- [ ] **L3**：拿到真实服务器拓扑图，能在 5 分钟内规划出 TP/DP/PP/EP 的最优 Rank 亲和性。
- [ ] **L4**：能清晰向他人讲透 GPUDirect RDMA 如何通过 P2P DMA 彻底旁路 Host CPU 与内存拷贝。
- [ ] **L5**：能说出 TMA、mbarrier 与 Copy Engine 相较于传统 LSU 搬运在 SM 资源占用上的本质优势。
- [ ] **L6**：能推导出 Ring AllReduce 通信量公式 $2\frac{P-1}{P}M$，并解释为什么 Bus Bandwidth 能够消除节点数影响。
- [ ] **L7**：能手画 DDP Bucket 与 FSDP Prefetch 的时序流水线，推导 Overlap 惩罚因子 $k$ 的物理成因。
- [ ] **L8**：熟练解读 Nsight Systems Timeline，一眼看出 Exposed Communication 发生在何处。
- [ ] **L9**：面对线上千卡集群 NCCL Hang 与 Slow Node 报警，能依据排障决策树在 15 分钟内锁定根因。

---

*返回前序模块：[《Module 00: 性能工程与系统前置》](../01_性能工程与系统前置/README.md)*  
*进入下一模块：[《Module 02: 大模型分布式并行算法与框架》](../03_CUDA与算子性能优化/README.md)*
