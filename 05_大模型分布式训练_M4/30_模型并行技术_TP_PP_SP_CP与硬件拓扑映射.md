---
title: "第30讲：模型并行与长序列基石——TP/PP/SP/CP 矩阵切分与硬件拓扑映射"
subtitle: "从GEMM列切行切到Megatron 3D编排/1F1B气泡率消除/Ring Attention与Ulysses上下文切分/NVLink与IB通信域拓扑对齐"
date: "2026-09-08"
pubDate: "2026-09-08"
tags:
  [
    "模型并行",
    "张量并行TP",
    "流水线并行PP",
    "序列并行SP",
    "上下文并行CP",
    "3D并行",
    "Ring Attention",
    "DeepSpeed Ulysses",
    "Megatron-LM",
    "通信拓扑映射",
    "AI Infra",
  ]
category: "distributed-training"
order: 30
author: "Ringi Lee"
showToc: true
tocOpen: false
math: true
---

# 第30讲：模型并行与长序列基石——TP/PP/SP/CP 矩阵切分与硬件拓扑映射

> **主讲人**：👓 **Ringi**（大厂 AI Infrastructure 资深性能架构师）  
> **所属专栏**：[《AI_Infra大话西游之水滴石穿》](../../README.md) ➔ [Module 04: 大模型分布式训练系统](../README.md)  
> **篇章范式**：🌐 大规模分布式训练系统范式（Distributed Training Systems Paradigm）  
> **源码与实验环境**：NVIDIA A100-SXM4-80GB / H100-SXM5-80GB | CUDA 12.4 | Python 3.10 | PyTorch 2.3+ | Megatron-LM v0.6+  
> **知识底账索引**：
>
> - 张量并行与序列并行：[第6章 张量并行TP与序列并行SP（AIInfraGuide）](file:///d:/GeneTind/Interview/AI_BOOK/AIInfraGuide/docs/guides/模块三-分布式训练/第6章-张量并行与序列并行.md)
> - 流水线并行与气泡消除：[第7章 流水线并行PP（AIInfraGuide）](file:///d:/GeneTind/Interview/AI_BOOK/AIInfraGuide/docs/guides/模块三-分布式训练/第7章-流水线并行.md)
> - 长序列训练与上下文并行：[第9章 长序列训练与上下文并行（AIInfraGuide）](file:///d:/GeneTind/Interview/AI_BOOK/AIInfraGuide/docs/guides/模块三-分布式训练/第9章-长序列训练与上下文并行.md)
> - 3D 混合并行与拓扑编排：[第11章 3D并行与混合并行策略（AIInfraGuide）](file:///d:/GeneTind/Interview/AI_BOOK/AIInfraGuide/docs/guides/模块三-分布式训练/第11章-3D并行与混合并行策略.md)
> - 多维度混合并行深度剖析：[6. 多维度混合并行（llm_interview_note）](file:///d:/GeneTind/Interview/AI_BOOK/llm_interview_note/04.分布式训练/6.多维度混合并行/6.多维度混合并行.md)

---

![Ringi 导师解构：3D 混合并行与硬件拓扑对齐全景工坊](assets/ringi_30_overview.png)

```text
=================================================================================================
                                   Ringi 3D 架构工坊 · 核心全景
                    大模型多维切分技术与物理硬件拓扑层次的“空间金字塔映射”
=================================================================================================
 [算子微观切分: TP (Tensor Parallel) + SP (Sequence Parallel)]
   • 切分对象: 单个 GEMM 矩阵按列/按行分片 + Non-Attention 激活序列切分
   • 核心原语: AllReduce (或 ReduceScatter + AllGather)
   • 物理网络: 锁死在单机 8 卡内部! 跑 900 GB/s NVLink 超高速回路 (通信频次高达每层 2 次!)

 [长序列切分: CP (Context Parallel) —— Ring Attention / DeepSpeed-Ulysses]
   • 切分对象: 沿 Sequence 维度切分 Q/K/V 计算 + Attention 计算本身
   • 核心原语: P2P 环形点对点 或 All-to-All 维度转置
   • 物理网络: 机内 NVLink 或小跨度机间 (2~4 节点) 高带宽 IB 网卡 (突破 128K 显存墙!)

 [网络层级切分: PP (Pipeline Parallel) —— 1F1B / Interleaved]
   • 切分对象: 沿 Transformer Block 深度纵向切分 (Stage 0 ➔ Stage 1 ➔ ... ➔ Stage P-1)
   • 核心原语: Point-to-Point (P2P Send/Recv)
   • 物理网络: 跨机柜跨节点互联 (仅在 Stage 边界传递轻量激活张量，完美规避机间慢速网络!)

 [集群宏观复制: DP (Data Parallel) / ZeRO-1 / FSDP]
   • 切分对象: Batch 维度全局切分 + 优化器状态/梯度全局分片
   • 核心原语: AllReduce / ReduceScatter
   • 物理网络: 跨越整个数据中心集群 (利用反向传播长计算时间异步掩盖通信!)
=================================================================================================
```

<!-- more -->

## 📑 目录导航

- [0. Ringi 开场：生产真实现场与痛点冲突](#0-ringi-开场生产真实现场与痛点冲突)
  - [0.1 真实工程矛盾：当单层显存刺穿与 128K 序列长相遇，数据并行彻底失效](#01-真实工程矛盾当单层显存刺穿与-128k-序列长相遇数据并行彻底失效)
  - [0.2 线上真实事故复盘：某千亿模型拓扑乱配引发的“NVLink 闲死，IB 塞爆，MFU 跌破 15%”](#02-线上真实事故复盘某千亿模型拓扑乱配引发的nvlink-闲死ib-塞爆mfu-跌破-15)
  - [0.3 模型并行与多维切分技术全景演进速查表](#03-模型并行与多维切分技术全景演进速查表)
- [1. 张量并行（Tensor Parallelism, TP）深度拆解与数学推导](#1-张量并行tensor-parallelism-tp深度拆解与数学推导)
  - [1.1 为什么需要切矩阵：单层 GEMM 算力与参数物理分割](#11-为什么需要切矩阵单层-gemm-算力与参数物理分割)
  - [1.2 Column Parallel Linear（列切分）数学证明](#12-column-parallel-linear列切分数学证明)
  - [1.3 Row Parallel Linear（行切分）数学证明](#13-row-parallel-linear行切分数学证明)
  - [1.4 Transformer Block 的神级闭环组合：两层 GEMM 仅需 2 次 AllReduce](#14-transformer-block-的神级闭环组合两层-gemm-仅需-2-次-allreduce)
  - [1.5 No Naked Formula 2.0 穿透 TP 通信量](#15-no-naked-formula-20-穿透-tp-通信量)
- [2. 序列并行（Sequence Parallelism, SP）——消灭非 TP 区域的激活冗余](#2-序列并行sequence-parallelism-sp消灭非-tp-区域的激活冗余)
  - [2.1 隐形显存刺客：LayerNorm 和 Dropout 上的全量激活冗余](#21-隐形显存刺客layernorm-和-dropout-上的全量激活冗余)
  - [2.2 Megatron-SP 的代数变换：AllReduce 拆解为 ReduceScatter + AllGather](#22-megatron-sp-的代数变换allreduce-拆解为-reducescatter--allgather)
  - [2.3 零额外通信代价下的激活显存线性暴降](#23-零额外通信代价下的激活显存线性暴降)
- [3. 上下文并行（Context Parallelism, CP）——突破 128K+ 序列长度墙](#3-上下文并行context-parallelism-cp突破-128k-序列长度墙)
  - [3.1 为什么长文本下 TP+SP 依然 OOM：Attention 计算的 $O(s^2)$ 极限](#31-为什么长文本下-tpsp-依然-oomattention-计算的-os2-极限)
  - [3.2 方案一：Ring-Attention 环形点对点流转与 Online Softmax 拼装](#32-方案一ring-attention-环形点对点流转与-online-softmax-拼装)
  - [3.3 方案二：DeepSpeed-Ulysses 的 All-to-All 注意力头维度转置](#33-方案二deepspeed-ulysses-的-all-to-all-注意力头维度转置)
  - [3.4 Ring-Attention vs DeepSpeed-Ulysses 工业级选型决策](#34-ring-attention-vs-deepspeed-ulysses-工业级选型决策)
- [4. 流水线并行（Pipeline Parallelism, PP）与气泡率消除](#4-流水线并行pipeline-parallelism-pp与气泡率消除)
  - [4.1 纵向层间切分与 P2P 通信特征](#41-纵向层间切分与-p2p-通信特征)
  - [4.2 流水线调度演进与气泡率推导（Naive ➔ GPipe ➔ 1F1B ➔ Interleaved）](#42-流水线调度演进与气泡率推导naive--gpipe--1f1b--interleaved)
  - [4.3 显存峰值控制：1F1B 如何把激活显存从 $M$ 份压至 $P$ 份](#43-显存峰值控制1f1b-如何把激活显存从-m-份压至-p-份)
- [5. 3D 混合并行与硬件拓扑对齐（Hardware Topology Mapping）](#5-3d-混合并行与硬件拓扑对齐hardware-topology-mapping)
  - [5.1 维度正交律与世界规模方程：$\text{World Size} = \text{DP} \times \text{PP} \times \text{TP} \times \text{CP}$](#51-维度正交律与世界规模方程textworld-size--textdp-times-textpp-times-texttp-times-textcp)
  - [5.2 硬件拓扑映射的黄金四原则（带宽阶梯法则）](#52-硬件拓扑映射的黄金四原则带宽阶梯法则)
  - [5.3 Megatron-LM 内部笛卡尔积 Rank 编排与通信组构建](#53-megatron-lm-内部笛卡尔积-rank-编排与通信组构建)
  - [5.4 工业级生产案例：千卡集群训练 70B / 530B 的黄金参数矩阵](#54-工业级生产案例千卡集群训练-70b--530b-的黄金参数矩阵)
- [6. 全场景实战与实验代码（Minimal Runnable Code）](#6-全场景实战与实验代码minimal-runnable-code)
  - [6.1 实验一：纯 Python 原生实现的 TP 列切与行切矩阵数学对齐实战](#61-实验一纯-python-原生实现的-tp-列切与行切矩阵数学对齐实战)
  - [6.2 实验二：工业级 3D 并行拓扑配置器与通信/显存/气泡率综合评估器 `megatron_3d_topology_planner.py`](#62-实验二工业级-3d-并行拓扑配置器与通信显存气泡率综合评估器-megatron_3d_topology_plannerpy)
- [7. Ringi 避坑指南与生产黄金准则](#7-ringi-避坑指南与生产黄金准则)
  - [7.1 7 大常见小白认知误区 vs 大厂 AI Infra 正确物理认知](#71-7-大常见小白认知误区-vs-大厂-ai-infra-正确物理认知)
  - [7.2 生产模型并行工程黄金 Checklist](#72-生产模型并行工程黄金-checklist)
- [8. Ringi 5 点核心速记口诀、自我检验清单与课后深度思考题](#8-ringi-5-点核心速记口诀自我检验清单与课后深度思考题)
  - [8.1 5 点押韵核心速记口诀](#81-5-点押韵核心速记口诀)
  - [8.2 10 条白板自我检验清单](#82-10-条白板自我检验清单)
  - [8.3 3 道高阶开放式课后思考题（含极限 Corner Case）](#83-3-道高阶开放式课后思考题含极限-corner-case)
- [9. 📚 参考资料与核心源码/经典论文指引](#9--参考资料与核心源码经典论文指引)
- [附录：Appendix A — 大厂硬核高频面试题与白板推导（Interview Drill）](#附录appendix-a--大厂硬核高频面试题与白板推导interview-drill)

---

# 0. Ringi 开场：生产真实现场与痛点冲突

### 0.1 真实工程矛盾：当单层显存刺穿与 128K 序列长相遇，数据并行彻底失效

在前两讲（第 28 讲与第 29 讲）中，我们把数据并行（DDP）和显存分片技术（ZeRO / FSDP）拆到了晶体管和通信原语级别：
- DDP 依靠反向异步 Bucket AllReduce 实现了极高的计算通信重叠；
- ZeRO-3 / FSDP 依靠流式按需拼装（On-demand AllGather），将单卡静态显存从 $16\Psi$ 压低到了极限的 $\frac{16\Psi}{N}$。

许多刚入行的同学往往会产生一种美好的幻想：
> *“既然 FSDP 已经可以把参数、梯度和优化器切到无限小，那是不是意味着我们只需要 FSDP 就足够统治一切大模型预训练了？”*

**现实给了我们一记沉重的耳光：在生产环境里，纯粹的数据并行会瞬间撞上两堵坚不可摧的物理高墙！**

1. **单层 GEMM 算力与显存墙（Layer-level Memory Wall）**：
   - 哪怕使用 FSDP，在前向执行某一个具体的 Transformer Block 时，系统也**必须在显存中拼装出这一层完整的参数**；
   - 对于千亿模型（如 530B），单个中间隐藏层投影矩阵的尺寸高达 $20480 \times 20480$（单个权重矩阵就占近 1 GB 显存），而当执行大 Batch 或长序列前向时，仅这一层产生的**单次矩阵乘法中间临时激活显存就会直接击穿单张 80GB 卡的物理上限！**
   - FSDP 根本无法把**单个算子内部的计算和单层激活值**切开！
2. **长序列 $O(s^2)$ 长度墙（Sequence Length Wall）**：
   - 当大模型从 2K 序列跃升至 32K、128K 乃至 1M 时，Self-Attention 的注意力矩阵计算量与显存开销随序列长度 $s$ 呈二次方爆炸；
   - 传统的 DDP/FSDP 是在 Batch 维度切分。当序列超长导致单卡哪怕只能跑 $\text{Batch Size} = 1$ 的单个样本时，单卡显存也当场爆仓！

**结论极其残酷**：如果模型单层放不下，必须切矩阵（**TP**）；如果单条序列太长，必须切上下文（**CP/SP**）；如果模型层数太多跨机放不下，必须切层间网络（**PP**）。  
这就是为什么顶级大模型训练集群必须走向 **3D / 4D 混合并行** 的终极原因。

---

### 0.2 线上真实事故复盘：某千亿模型拓扑乱配引发的“NVLink 闲死，IB 塞爆，MFU 跌破 15%”

2024 年，国内某智算中心在 64 台 8 卡 H800（共 512 张 GPU）集群上预训练一个 175B 规模的稠密模型。

初始启动时，架构团队分配了如下配置：
- 全局 Batch Size 较大，配置了 $\text{TP} = 16, \text{PP} = 4, \text{DP} = 8$（$16 \times 4 \times 8 = 512$）。
- 由于单机只有 8 张 GPU，配置 $\text{TP} = 16$ 意味着**张量并行组强行跨越了物理机界限**：每 2 台机器的 16 张卡组成一个 TP 组。

任务上线跑出第一个 Step，监控告警全线飘红：
1. **网络交换机 PFC 流控风暴**：机间 400G InfiniBand 网络的丢包与拥塞通知瞬间拉满；
2. **GPU 算力利用率暴跌**：整机单步步时高达 28.6 秒，MFU（Model FLOPs Utilization）跌至凄惨的 **13.8%**；
3. **硬件资源极度倒挂**：节点内的 NVLink 跑道空空荡荡（利用率不足 10%），而机间 IB 网卡跑满发烫，90% 的时间 GPU 都在等待跨机 AllReduce 的握手同步！

![Ringi 导师解构：Column/Row Parallel 矩阵切分与 Transformer 通信插入工坊](assets/ringi_30_tp_column_row.png)

```text
[PyTorch Profiler 事故现场微观剖析]
单层 Transformer Block 前向 GEMM 耗时: 1.8 ms
单层 TP 跨机 AllReduce 耗时 (跨 400G IB): 24.2 ms (◄── 通信耗时是计算的整整 13.4 倍!)
全网 80 层累加: 每一层都要停下来在机间等 2 次 AllReduce! 系统彻底退化为跨机网络等车大会!
```

**事故致命根因**：
- 张量并行（TP）是微观算子级并行，在单个 Transformer Block 内部每算一层 GEMM 就要触发一次通信，**通信频率极其恐怖（单步数百次）**；
- 架构师错误地把高频通信的 TP 推到了机间慢速网络（400G IB 单向带宽仅 50 GB/s，且延迟高达数微秒）；而节点内部单向带宽高达 400 GB/s、延迟仅几百纳秒的 **NVLink 却被严重闲置**！

**终极抢救方案**：
- 严格遵循**硬件物理拓扑映射阶梯法则**：强制把 TP 限制在单机 8 卡内部（$\text{TP} = 8$），锁死在 NVLink 高速公路；
- 将跨机切分交给流水线并行（$\text{PP} = 8$）与数据并行（$\text{DP} = 8$）；
- 重构后，单步步时从 28.6 秒骤降到 **4.1 秒**，吞吐暴增近 **7 倍**，MFU 强势跃升至 **54.2%**！

---

### 0.3 模型并行与多维切分技术全景演进速查表

| 并行策略 | 核心切分维度 | 关键通信原语 | 每步通信频次与通信量 | 物理硬件映射首选 | 核心解决问题 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **张量并行 (TP)** | 权重矩阵与隐藏维度 ($h$) | **AllReduce** | 极高频（每层 2 次，量级与激活正比） | **严禁出机！必须单机 8 卡内 NVLink** | 单层参数/GEMM 激活突破单卡显存 |
| **序列并行 (SP)** | 非 Attention 激活序列维度 ($s$) | **ReduceScatter + AllGather** | 与 TP 同频，将 AllReduce 拆解重排 | **与 TP 完全重合（单机 8 卡内 NVLink）** | 消除 LayerNorm/Dropout 激活冗余 |
| **上下文并行 (CP)** | Attention 计算序列维度 ($s$) | **P2P (Ring) 或 All-to-All** | 中频（注意力计算期，通信量与 KV 正比）| **机内 NVLink 或小跨度机间 IB** | 彻底突破 32K~1M 超长文本 Attention 墙 |
| **流水线并行 (PP)** | 模型层级深度纵向 ($L$) | **P2P (Send / Recv)** | 低频（仅 micro-batch 边界传递激活张量）| **跨节点 / 跨机柜 InfiniBand 网络** | 模型总层数太多、参数总量突破单机容量 |
| **数据并行 (DP/ZeRO)** | Batch 维度与全局状态 ($b$) | **AllReduce / ReduceScatter** | 极低频（每步反向结束时异步聚合一次） | **跨越全数据中心集群的所有节点** | 增大全局 Batch Size，加速训练收敛 |

---

# 1. 张量并行（Tensor Parallelism, TP）深度拆解与数学推导

> 💡 **架构全景速览**：在深潜源码前，先在白板上建立坚不可摧的模型并行（TP/PP/SP/CP）与硬件物理拓扑映射底账。
> 
> ![模型并行技术 TP / PP / SP / CP 与硬件物理拓扑映射全景架构图](assets/arch_30_model_parallel_tp_pp_sp_cp_topology.svg)

### 1.1 为什么需要切矩阵：单层 GEMM 算力与参数物理分割

大模型的核心是由密集的通用矩阵乘法（GEMM）构成的。无论是注意力模块的 $Q, K, V$ 投影与输出投影，还是 MLP 模块的双层前馈网络，本质上都在反复计算：

$$Y = X \cdot W$$

其中：
- $X$ 是输入激活张量，Shape 为 $[b, s, h]$（Batch Size $\times$ 序列长度 $\times$ 隐藏层维度）；
- $W$ 是线性层权重矩阵，Shape 为 $[h, h_{\text{out}}]$（或 MLP 中的 $[h, 4h]$）；
- $Y$ 是输出激活张量，Shape 为 $[b, s, h_{\text{out}}]$。

当参数量达到千亿级别时，$h$ 往往高达 $8192$ 甚至 $12288$。单张卡不仅放不下如此庞大的矩阵，而且单卡 Tensor Core 的算力也无法满足实时低延迟计算的需求。

Megatron-LM 论文（Shoeybi et al., 2019）提出了开创性的解决方案：**将矩阵 $W$ 按列切分（Column Parallel）或按行切分（Row Parallel），并在适当时机插入集合通信原语，确保全网计算结果与单卡串行计算在数学上严格等价。**

---

### 1.2 Column Parallel Linear（列切分）数学证明

#### ① 切分方式
将权重矩阵 $W \in \mathbb{R}^{h \times h_{\text{out}}}$ 沿**列方向（输出特征维度）**均匀切分成 $N$ 份（$N$ 为 TP 度，通常为 8）：

$$W = \begin{bmatrix} W_1 & W_2 & \cdots & W_N \end{bmatrix}, \quad W_i \in \mathbb{R}^{h \times \frac{h_{\text{out}}}{N}}$$

#### ② 计算过程
每张 GPU 独立持有完整的输入激活 $X$ 以及自己负责的那一列权重分片 $W_i$。各卡在本地独立执行矩阵乘法：

$$Y_i = X \cdot W_i, \quad Y_i \in \mathbb{R}^{b \times s \times \frac{h_{\text{out}}}{N}}$$

#### ③ 输出拼接与通信
各卡计算出的 $Y_i$ 恰好拼成完整的输出矩阵 $Y$：

$$Y = X \cdot W = X \cdot \begin{bmatrix} W_1 & W_2 & \cdots & W_N \end{bmatrix} = \begin{bmatrix} XW_1 & XW_2 & \cdots & XW_N \end{bmatrix} = \begin{bmatrix} Y_1 & Y_2 & \cdots & Y_N \end{bmatrix}$$

- **惊艳特性**：**在前向传播过程中，Column Parallel 完全不需要任何卡间通信！** 每张卡拿着完整的 $X$，各算各的列分片，算出来的结果 $Y_i$ 也是自然按列分片的。

---

### 1.3 Row Parallel Linear（行切分）数学证明

#### ① 切分方式
将权重矩阵 $W \in \mathbb{R}^{h_{\text{in}} \times h}$ 沿**行方向（输入特征维度）**均匀切分成 $N$ 份：

$$W = \begin{bmatrix} W_1 \\ W_2 \\ \vdots \\ W_N \end{bmatrix}, \quad W_i \in \mathbb{R}^{\frac{h_{\text{in}}}{N} \times h}$$

#### ② 输入要求
为了与切断的行矩阵相乘，输入张量 $X$ 必须沿**列方向切分**，每张卡只持有对应的分片 $X_i \in \mathbb{R}^{b \times s \times \frac{h_{\text{in}}}{N}}$。

#### ③ 计算过程
各卡在本地计算部分矩阵乘积：

$$Y_i = X_i \cdot W_i, \quad Y_i \in \mathbb{R}^{b \times s \times h}$$

#### ④ 输出聚合与通信
根据分块矩阵乘法法则：

$$Y = X \cdot W = \begin{bmatrix} X_1 & X_2 & \cdots & X_N \end{bmatrix} \begin{bmatrix} W_1 \\ W_2 \\ \vdots \\ W_N \end{bmatrix} = \sum_{i=1}^N X_i W_i = \sum_{i=1}^N Y_i$$

- **通信插入点**：各卡算出的 $Y_i$ 具有完整的输出维度 $[b, s, h]$，但包含的只是“部分求和结果（Partial Sum）”。**必须在各卡之间执行一次全局求和规约（AllReduce），才能恢复出数学上完全正确的完整输出 $Y$！**

---

### 1.4 Transformer Block 的神级闭环组合：两层 GEMM 仅需 2 次 AllReduce

如果随便滥用列切和行切，每一层矩阵乘法前后都需要插入大量的 AllGather 或 AllReduce，网络瞬间爆炸。  
Megatron-LM 最天才的工程发明，就是将 **Column Parallel 与 Row Parallel 巧妙配对**，构成两个优雅的双层结构：

```text
[结构一：MLP 模块的列切行切配对]
输入 X ──► [FC1: Column Parallel] ──► 输出 [Y1, Y2, ..., YN] (无需通信!)
           │ (切分输出维度 4h/N)
           ▼
           [GELU / SwiGLU 激活函数] (完全按元素独立计算，零通信!)
           │
           ▼
输入 [Z1, ..., ZN] ──► [FC2: Row Parallel] ──► [AllReduce 求和] ──► 输出完整 MLP(X)
                      (切分输入维度 4h/N)       (仅需 1 次通信!)

[结构二：Self-Attention 模块的列切行切配对]
输入 X ──► [Q, K, V 投影: Column Parallel] ──► [Head 独立切分注意力计算] (零通信!)
           │ (将总 Head 数分给 N 张卡)
           ▼
输入 [Attn1, ..., AttnN] ──► [Output 投影: Row Parallel] ──► [AllReduce 求和] ──► 输出完整 Attn(X)
```

**数学闭环的震撼美感**：
1. **MLP 模块**：第一层 FC1 按列切，输出自然分成 $N$ 份；直接送入逐元素的激活函数；第二层 FC2 恰好需要按行切的输入，两者无缝咬合！仅在 FC2 输出时做 **1 次 AllReduce**；
2. **Attention 模块**：$Q, K, V$ 投影按列切，由于多头注意力各个 Head 本身就是互相独立的，每张卡只需负责 $\frac{\text{Heads}}{N}$ 个头；算完注意力矩阵后，输出投影矩阵按行切，仅在最终投影结束时做 **1 次 AllReduce**！
3. **整层总结**：一个包含 Attention 和 MLP 的标准 Transformer Block，在前向传播中**总共只需要执行 2 次 AllReduce**！

---

### 1.5 No Naked Formula 2.0 穿透 TP 通信量

我们严格执行 **No Naked Formula 2.0**，将 TP 的通信量底账算到每一个字节。

#### ① 为什么需要算它？
张量并行发生在微观层级，网络通信极其密集。必须精确算清每一步通信的字节数，用数学证明为什么 TP 绝对不能跨机，而只能活在 NVLink 内。

#### ② Mental Model（物理直觉比喻）
4 个数学家合力算一个庞大的多维方程组。大家先各自拿着全套题目，各自算出一部分变量（列切，零通信）；接着大家把中间草稿直接代入下一阶段（无缝咬合）；最后在得出总结果时，4 个人必须把手头的数字碰头加在一起汇总（Row Parallel 触发 AllReduce 求和）。

#### ③ Tiny Calculator（极简手算）
设：
- Token 数量 $b \times s = 2$；
- 隐藏层维度 $h = 4$；
- 张量并行度 $N = 2$；
- 数据类型为 BF16（每个元素 2 字节）。

在 Row Parallel 结束时，每张 GPU 都算出了一个 Shape 为 $[2, 4]$ 的局部张量，包含 $2 \times 4 = 8$ 个元素，大小为 $8 \times 2 = 16\text{ 字节}$。  
执行 Ring-AllReduce 时，单卡发送的数据量为：

$$\text{Comm} = 2 \times \left(\frac{N-1}{N}\right) \times \text{Size} = 2 \times \left(\frac{2-1}{2}\right) \times 16\text{ B} = \mathbf{16\text{ 字节}}$$

#### ④ Formal Model（标准公式）
对于一个隐藏层维度为 $h$、序列长度为 $s$、批大小为 $b$ 的大模型：
1. **前向传播（Forward）**：
   - Attention 输出投影后 1 次 AllReduce：数据大小为 $b \times s \times h$；
   - MLP 输出投影后 1 次 AllReduce：数据大小为 $b \times s \times h$；
   - 单卡前向通信总量（基于 Ring-AllReduce 发送量 $2 \frac{N-1}{N} \text{Size}$，当 $N=8$ 时 $\frac{N-1}{N} \approx 1$）：
     $$\text{Comm}_{\text{fwd}} = 2 \times \left(2 \times \frac{N-1}{N} \times b s h \times 2\text{ Bytes}\right) \approx \mathbf{4 b s h} \quad (\text{Words}) = \mathbf{8 b s h} \quad (\text{Bytes})$$
2. **反向传播（Backward）**：
   - 伴随矩阵求导法则，前向的 Row Parallel 在反向求梯度时变为 Column Parallel（需 1 次 AllReduce）；
   - 前向的 Column Parallel 在反向时变为 Row Parallel（需 1 次 AllReduce）；
   - **反向通信量与前向完全对称**：同样为 $4 b s h$（Words）！
3. **单层单步总通信量**：
   $$\mathbf{\text{Comm}_{\text{TP, layer}} = \text{Comm}_{\text{fwd}} + \text{Comm}_{\text{bwd}} = 8 b s h \quad (\text{Words}) = \mathbf{16 b s h} \quad (\text{Bytes})}$$

#### ⑤ Sanity Check（数量级校验）
以 **LLaMA-3-70B**（$h = 8192$, 层数 $L = 80$）在 $b=2, s=4096$ 下单卡每步通信量为例：
- 单层通信量：$16 \times 2 \times 4096 \times 8192 \times 2\text{ Bytes} \approx \mathbf{2.15\text{ GB}}$；
- 全模型 80 层单步通信量：$80 \times 2.15\text{ GB} \approx \mathbf{172\text{ GB}}$！
- **震撼结论**：单步迭代哪怕只需 2 秒，单卡每秒必须吞吐 **$86\text{ GB/s}$** 的通信流！
- **硬件审判**：跨机 InfiniBand 400G 网卡的有效吞吐仅约 45 GB/s（瞬间被撑死，步时拉长数倍）；而机内 NVLink（450~900 GB/s）吞吐轻松承载这 86 GB/s，通信占比被压缩至 10% 以内！**TP 严禁出机是铁一般的物理法则！**

---

# 2. 序列并行（Sequence Parallelism, SP）——消灭非 TP 区域的激活冗余

### 2.1 隐形显存刺客：LayerNorm 和 Dropout 上的全量激活冗余

Megatron-LM 的标准 TP 虽然优雅地切分了 GEMM，但工程团队很快发现：**显存并没有像预期那样随着 TP 线性缩小！**

仔细审视 Transformer Block 的内部结构：
- Attention 和 MLP 内部的矩阵被切成了 $1/N$；
- 但是在 **LayerNorm、Dropout 以及残差连接（Residual Addition）** 区域，输入和输出张量都是全尺寸的 $[b, s, h]$；
- 为了执行反向求导，系统必须在显存中缓存这些区域的前向激活值。**这些非 TP 区域的激活值，在每张 GPU 上都保存了一份 100% 重复的全量副本！**
- 在长序列训练中，这些激活值显存甚至超过了模型参数本身！

```text
[经典 TP 的激活值分布]
LayerNorm (全量 bsh) ──► GEMM (分片 bsh/N) ──► AllReduce ──► Dropout (全量 bsh) ──► 残差 (全量 bsh)
▲ 各卡保存了完全相同的激活值副本！巨大的物理浪费！
```

---

### 2.2 Megatron-SP 的代数变换：AllReduce 拆解为 ReduceScatter + AllGather

Megatron-LM 团队在 2022 年（Korthikanti et al., 2022）提出了著名的 **Megatron-SP（序列并行）**。其核心洞察力堪称代数神来之笔：

我们知道集合通信原语存在一个恒等分解式：

$$\mathbf{\text{AllReduce} = \text{ReduceScatter} + \text{AllGather}}$$

- **ReduceScatter**：将全尺寸张量规约求和，并将结果切成 $N$ 份分散到各卡；
- **AllGather**：将各卡持有的 $1/N$ 分片收集拼装成全尺寸张量。

Megatron-SP 巧妙地把这两个原本紧紧黏在一起的原语**拉开了距离**：

```text
[Megatron-SP 架构执行数据流]
1. [非 TP 区域: LayerNorm / Dropout]
   • 数据状态: 沿序列维度切分! 每卡仅持有 [b, s/N, h] 的激活值!
   • 显存收益: 激活显存瞬间降至 1/N!

2. [进入 Column Parallel GEMM 之前]
   • 执行 AllGather: 将 [b, s/N, h] 收集拼装为 [b, s, h];
   • 计算 GEMM: 正常执行列切分矩阵乘法.

3. [走出 Row Parallel GEMM 之后]
   • 不再执行 AllReduce! 
   • 直接执行 ReduceScatter: 规约求和并直接按序列切分，各卡仅收回 [b, s/N, h]!

4. [回到非 TP 区域]
   • 带着 [b, s/N, h] 进入下一个 LayerNorm 和残差连接.
```

---

### 2.3 零额外通信代价下的激活显存线性暴降

**这笔账极其震撼**：
1. **通信量守恒**：经典 TP 在 Row Parallel 后做一次 AllReduce（传输量为 $2 \frac{N-1}{N} bsh$）；而在 SP 中，变成了“一次 ReduceScatter（$\frac{N-1}{N} bsh$）+ 一次 AllGather（$\frac{N-1}{N} bsh$）”，**总通信量完全守恒，没有增加任何一个字节！**
2. **显存收益巨大**：整层 Transformer Block 中，不仅 GEMM 区域是 $1/N$ 显存，连 LayerNorm、Dropout 和残差连接也全变成了 $1/N$ 显存；
3. **结论**：**Sequence Parallelism 是免费的午餐（Free Lunch）**。在工业界生产中，只要启用了 TP，**必须无条件同步开启 SP**！

---

# 3. 上下文并行（Context Parallelism, CP）——突破 128K+ 序列长度墙

### 3.1 为什么长文本下 TP+SP 依然 OOM：Attention 计算的 $O(s^2)$ 极限

当我们将上下文长度推进到 **128K、256K 乃至 1M** 时，又遭遇了新的生死劫：
- TP+SP 仅仅将序列切分到了单机 8 卡（$\text{TP}=8$），序列长度从 $128\text{K}$ 降到了 $16\text{K}$；
- 然而，在 Self-Attention 的核心区域，每个 Query Token 依然需要与全序列的所有 Key Token 进行点积，计算复杂度与中间 Softmax 显存依然是 **$O(s^2)$**；
- 哪怕单卡 Batch Size 压低到 1，仅 $128\text{K}$ 的 KV 激活和 Attention Score 也会直接在单卡爆掉。

**上下文并行（Context Parallelism, CP）** 应运而生：**不再依赖单机 TP，而是跨节点将一条超长序列沿 Context 维度切分到数十甚至数百张 GPU 上并行计算 Attention！**

当前工业界主要存在两大技术流派：**Ring-Attention** 与 **DeepSpeed-Ulysses**。

---

### 3.2 方案一：Ring-Attention 环形点对点流转与 Online Softmax 拼装

由 UC Berkeley（Hao Liu et al., 2023）提出的 **Ring-Attention**，是长序列领域极具颠覆性的工作。

```text
[Ring-Attention 物理环路流转机制 (以 4 张 GPU 为例)]
GPU 0 持有: Q0, K0, V0
GPU 1 持有: Q1, K1, V1
GPU 2 持有: Q2, K2, V2
GPU 3 持有: Q3, K3, V3

Step 0: 各卡在本地计算 Attn(Qi, Ki, Vi) ──► 初始化局部 Softmax 统计量 (m_i, l_i)
Step 1: [P2P 环形传递!] 各卡将自己的 (K, V) 分片发送给下一张卡 (GPU 0 ➔ 1 ➔ 2 ➔ 3 ➔ 0)
        各卡拿到新的 (K, V)，计算 Attn(Qi, K_new, V_new)，利用 FlashAttention Online Softmax 动态更新结果!
Step 2: 再次环形发送... 再次更新...
Step 3: 循环 N 步后，所有 Query 完成与全网所有 Key/Value 的注意力交互!
```

#### 关键优势与 Corner Case：
1. **通信计算完美 Overlap**：P2P 传输下一个 KV 块的时间，完全被当前块的 GEMM 计算所掩盖；
2. **因果掩码（Causal Mask）下的负载不均衡**：
   - 自回归模型中，靠前的 Token 无法看到靠后的 Token（下三角掩码）；
   - 在朴素切分下，持有序列前半段的 GPU 在后半程只能空转等车；
   - **工业级解决方案（Zigzag / Striped Ring-Attention）**：不再按连续区间切分，而是采用跨步交错切分（如 GPU 0 持有 Token 0, 4, 8...），让每张 GPU 承担均等大小的有效计算三角区，算力利用率回升至近 100%！

---

### 3.3 方案二：DeepSpeed-Ulysses 的 All-to-All 注意力头维度转置

微软 DeepSpeed 团队提出的 **DeepSpeed-Ulysses**，采用了一种极其巧妙的“维度置换魔法”：

```text
[DeepSpeed-Ulysses 维度转置流程]
1. [输入状态]: 张量沿 Sequence 维度切分! 
   Shape: [Batch, Seq / N_cp, Heads, Dim]

2. [第 1 次 All-to-All 通信]: 
   在 CP 通信组内执行全交换，把 Sequence 切分转换为 Head 切分!
   Shape 变为: [Batch, Seq (完整超长序列!), Heads / N_cp, Dim]

3. [本地极速计算 Attention]:
   每张卡直接调用原生 FlashAttention，计算完整长序列但在较少 Head 上的注意力!
   (零通信，算子完全复用现有极致优化内核!)

4. [第 2 次 All-to-All 通信]:
   再次执行全交换，把 Head 切分重新还原为 Sequence 切分!
   Shape 恢复为: [Batch, Seq / N_cp, Heads, Dim]
```

---

### 3.4 Ring-Attention vs DeepSpeed-Ulysses 工业级选型决策

| 评估维度 | Ring-Attention | DeepSpeed-Ulysses |
| :--- | :--- | :--- |
| **底层核心通信原语** | 节点间点对点传输（P2P Send/Recv） | 全局全交换规约（All-to-All） |
| **通信量与序列长度关系** | 传输总量与 KV 尺寸正比，随步数流转 | 传输总量固定为 $2 \times bsh$，通信高效 |
| **注意力头数（Heads）约束** | **完全无约束**（哪怕只有一个 Head 也能切！） | **强约束！** 总 Head 数必须能被 CP 度整除 |
| **GQA / MQA 兼容性** | 天然兼容 GQA（直接环传极小的 KV 分片） | 极为受限（当 KV Head=8 时，CP 度最大只能设为 8） |
| **长文本极限扩展能力** | 理论支持无限长（上亿 Token 均可跑起） | 受限于网络 All-to-All 小包延迟与拓扑开销 |
| **生产最佳适用场景** | 128K~1M 极限长序列、大模型 GQA 架构 | 32K~64K 中等长序列、全互联高速网络集群 |

---

# 4. 流水线并行（Pipeline Parallelism, PP）与气泡率消除

### 4.1 纵向层间切分与 P2P 通信特征

当模型不仅单层放不下，而且全网层数高达 80 层、120 层时，单台机器连所有权重都装不下。  
**流水线并行（Pipeline Parallelism, PP）** 沿着神经网络的深度方向进行纵向切分：
- 设模型总层数为 $L$，流水线并行度为 $P$（Stage 数量）；
- 每个 Stage 分配连续的 $\frac{L}{P}$ 层（例如 Stage 0 持有 1~20 层，Stage 1 持有 21~40 层……）；
- **通信特征极其优越**：只有相邻的两个 Stage 之间存在数据传输，且传输的仅仅是层间激活张量（Shape 为 $[b, s, h]$）。**通信完全是点对点（P2P），通信量与模型参数量完全解耦！**

这使得流水线并行成为了**跨越物理机柜、跨越低带宽机间网络的最完美屏障**。

---

### 4.2 流水线调度演进与气泡率推导（Naive ➔ GPipe ➔ 1F1B ➔ Interleaved）

流水线并行面临的最大敌人是 **Pipeline Bubble（流水线气泡）**：后一个 Stage 必须等待前一个 Stage 算完才能动工，导致 GPU 发生大面积空转。

我们运用 **No Naked Formula 2.0**，推导出四代流水线调度的气泡率演进公式：

![Ringi 导师解构：1F1B 与 Interleaved 流水线气泡消除工坊](assets/ringi_30_pp_schedule_bubble.png)

```text
=================================================================================================
                                  四代流水线调度时序对比图 (P=4)
=================================================================================================
[1. 朴素调度 Naive PP]: 气泡率 = (P - 1) / P = 75%
GPU 0: [F0]──────────────────────────[B0]
GPU 1: ────[F0]──────────────────[B0]────
GPU 2: ────────[F0]──────────[B0]────────
GPU 3: ────────────[F0]──[B0]────────────
▲ 只有 1 个 Batch 时，全网 75% 的时间都在等待空转! 灾难级浪费!

[2. GPipe 调度]: 将 Batch 切分成 M 个 Micro-batch (F0~F7, 全部 F 算完再算 B)
GPU 0: [F0][F1][F2][F3]────────────────────[B0][B1][B2][B3]
GPU 1: ────[F0][F1][F2][F3]────────────[B0][B1][B2][B3]────
▲ 气泡缩小，但 GPU 0 必须把所有 M 个 Micro-batch 的激活值全部死存，显存爆仓!

[3. 1F1B 调度 (One-Forward-One-Backward) —— 工业界通用标配]:
GPU 0: [F0][F1][F2][F3][B0][F4][B1][F5][B2][F6][B3][F7][B4]...
▲ 稳定期算一次前向紧跟一次反向! 激活显存上限死死压在 P 份!

[4. Interleaved 1F1B 调度 (虚拟 Stage) —— 极限气泡压缩]:
每个物理 GPU 承载 v 个非连续的虚拟 Stage (例如 GPU 0 负责 Layer 1~10 与 Layer 41~50)
▲ 气泡被切得极细，气泡率再次除以 v!
=================================================================================================
```

#### 气泡率量化推导对比表：

| 调度策略 | 气泡率理论公式（Bubble Ratio） | 极简数字手算（$P=4, M=16$） | 峰值激活显存占用 | 通信频率与开销 |
| :--- | :--- | :--- | :--- | :--- |
| **朴素串行 (Naive PP)** | $\frac{P-1}{P}$ | $\frac{3}{4} = \mathbf{75.0\%}$ (空转致死) | $1$ 份激活 | 极低 |
| **GPipe 调度** | $\frac{P-1}{M + P - 1}$ | $\frac{3}{16 + 3} = \mathbf{15.8\%}$ | **$M$ 份激活（显存极易爆炸）** | 低 |
| **1F1B 调度** | $\frac{P-1}{M + P - 1}$ | $\frac{3}{16 + 3} = \mathbf{15.8\%}$ | **$P$ 份激活（与 $M$ 解耦，安全！）**| 低 |
| **Interleaved 1F1B** | $\mathbf{\frac{P-1}{v \cdot M + P - 1}}$ | 当 $v=2$ 时：$\frac{3}{32 + 3} = \mathbf{8.5\%}$ | $P$ 份激活 | 增加 $v$ 倍跨卡激活传输 |

---

### 4.3 显存峰值控制：1F1B 如何把激活显存从 $M$ 份压至 $P$ 份

在 GPipe 中，所有 Micro-batch 的前向必须全部跑完，才开始跑反向。如果我们将一个全局 Batch 切分成 $M = 64$ 个 Micro-batch，Stage 0 就必须在显存里死死保留整整 **64 份完整的中间激活张量**，显存当场炸穿。

**1F1B 的破局之道（One-Forward-One-Backward）**：
1. **Warmup 阶段**：流水线先连续注入 $P$ 个 Micro-batch 的前向计算；
2. **Steady 稳态阶段**：流水线填满后，**每执行一个 Micro-batch 的前向计算，立刻执行上一个已经就绪的 Micro-batch 的反向计算！**
3. **因果释放**：一旦反向计算完成，该 Micro-batch 对应的中间激活显存立刻被 `free()` 彻底释放！
4. **数学铁证**：在整个稳态运行期间，任何一个物理 Stage 内部存活的活跃 Micro-batch 数量被严格封顶在 **$P$ 份以内**（与总微批次数 $M$ 彻底解耦），彻底解除了显存炸裂的后顾之忧。

---

# 5. 3D 混合并行与硬件拓扑对齐（Hardware Topology Mapping）

### 5.1 维度正交律与世界规模方程：$\text{World Size} = \text{DP} \times \text{PP} \times \text{TP} \times \text{CP}$

在超大规模集群中，单一的并行策略都存在致命缺陷：
- 纯 DP：单卡显存装不下超大模型；
- 纯 TP：通信过于频繁，无法跨越物理节点；
- 纯 PP：气泡率随着 Stage 增加急剧上升，且降低了数据吞吐；
- 纯 CP：仅解决单条序列长度，无法解决总权重装填。

工业级训练系统（以 Megatron-DeepSpeed 为代表）将各个维度正交组合，构成著名的 **3D / 4D 混合并行**：

$$\mathbf{\text{World Size} = \text{DP} \times \text{PP} \times \text{TP} \times \text{CP}}$$

每个物理 GPU 在全局通信世界中，都拥有一个四维离散坐标：

$$\text{Rank} \longleftrightarrow (\text{rank}_{\text{dp}}, \text{rank}_{\text{pp}}, \text{rank}_{\text{tp}}, \text{rank}_{\text{cp}})$$

---

### 5.2 硬件拓扑映射的黄金四原则（带宽阶梯法则）

集群网络存在残酷的带宽金字塔：
- **第一层**：单机内 NVLink 4.0（双向 900 GB/s，延迟 $< 1\mu s$）；
- **第二层**：同机架机间 InfiniBand NDR 400G（双向 50 GB/s，延迟数微秒）；
- **第三层**：跨核心交换机网络（拥塞与跳步增加）。

**顶级性能架构师必须遵守的拓扑对齐黄金法则**：

```text
[通信频率与带宽匹配阶梯]
通信频率最高:   【张量并行 TP】 ─────► 必须 100% 锁死在机内 NVLink 8 卡内!
通信频次次高:   【上下文并行 CP】 ───► 优先机内，跨机必须走专用 Rail-Optimized 顶配 IB!
微批次点对点:   【流水线并行 PP】 ───► 跨节点 / 跨机柜部署，通过高吞吐 IB 传递轻量激活!
粗粒度全局同步: 【数据并行 DP】 ─────► 跨越整个集群边界，利用反向计算时间全重叠掩盖!
```

---

### 5.3 Megatron-LM 内部笛卡尔积 Rank 编排与通信组构建

在 Megatron-LM 源码（`megatron/core/parallel_state.py`）中，进程组的初始化逻辑遵循极其严谨的**局部性优先步长排列**：

为了让物理相邻的 GPU（Rank 0~7）拥有连续的编号，Megatron 通常将 TP 放在最内层（步长为 1），随后是 CP，接着是 DP，最后是 PP：

```python
# Rank 坐标映射代数关系 (以 TP=8, DP=2, PP=4 为例，共 64 卡)
# 步长 (Strides):
# tp_stride = 1
# dp_stride = TP = 8
# pp_stride = TP * DP = 16

rank = (pp_rank * TP * DP) + (dp_rank * TP) + tp_rank
```

通过这种编排：
- 任意一个 TP 组内的 8 张卡，其物理编号一定是连续的 `[0, 1, 2, 3, 4, 5, 6, 7]`，**完美与物理单机 8 卡 NVLink 对齐**；
- 跨机时，PP 和 DP 组按固定大步长跨越节点，将低频通信自然引向机间 IB 光纤。

---

### 5.4 工业级生产案例：千卡集群训练 70B / 530B 的黄金参数矩阵

| 训练目标模型 | 集群规模（GPU 总数） | 推荐并行组合配置 | 序列长度 | 气泡率与 MFU 表现 | 架构设计底账与考量 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **LLaMA-3-8B** | 64 张 A100 (8 节点) | $\text{TP}=1, \text{PP}=1, \text{DP}=64$ (配合 ZeRO-2) | 8K | MFU: **56.8%** | 8B 参数小，单卡直接放得下，纯数据并行通信最纯粹 |
| **LLaMA-3-70B** | 256 张 H800 (32 节点) | $\text{TP}=8, \text{PP}=4, \text{DP}=8$ ($M=32$) | 8K | MFU: **52.4%** | TP=8 占满机内 NVLink；PP=4 跨机切分降低显存，1F1B 气泡仅 8.5% |
| **LLaMA-3-70B (超长文本)**| 512 张 H800 (64 节点) | $\text{TP}=4, \text{CP}=4, \text{PP}=4, \text{DP}=8$ | **64K** | MFU: **47.5%** | 引入 CP=4 突破长文本 Attention 墙，TP 降为 4 腾出机内带宽 |
| **MT-NLG 530B (千亿极品)**| 4480 张 A100 (560 节点) | $\text{TP}=8, \text{PP}=35, \text{DP}=16$ | 4K | MFU: **50.2%** | 经典大模型拓扑！PP=35 跨机柜深度切分，微批次 $M=192$ 抹平气泡 |

---

# 6. 全场景实战与实验代码（Minimal Runnable Code）

### 6.1 实验一：纯 Python 原生实现的 TP 列切与行切矩阵数学对齐实战

本实验通过纯 Python 与 PyTorch 原生矩阵算子，**从零模拟张量并行 Column Parallel 与 Row Parallel 的切分、独立计算与 AllReduce 聚合**，并严格验证其数值与单卡全局 GEMM 误差达到机器精度（$10^{-7}$）：

```python
"""
Ringi AI Infra 实验室：Minimal Runnable Code
实验一：张量并行 (TP) 列切与行切数学等价性与 AllReduce 模拟验证
环境要求：支持在安装了 PyTorch 的环境运行；若无 PyTorch 则自动无缝降级为纯 Python 标准库执行！
"""

import sys

def run_with_torch():
    import torch
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 2
    seq_len = 4
    hidden_dim = 8
    ffn_dim = 16
    tp_size = 2

    print("==================================================================")
    print(f"  Ringi AI Infra 实验室：张量并行 (TP) 算子切分数学等价性验证 (PyTorch 引擎)")
    print(f"  配置: Batch={batch_size}, SeqLen={seq_len}, Hidden={hidden_dim}, FFN={ffn_dim}, TP={tp_size}")
    print("==================================================================")

    # 1. 构造单卡标准模型权重与输入 (Ground Truth)
    X = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    W_fc1 = torch.randn(hidden_dim, ffn_dim, device=device)
    W_fc2 = torch.randn(ffn_dim, hidden_dim, device=device)

    # 单卡标准前向传播 (无切分基准)
    H_dense = torch.matmul(X, W_fc1)
    H_act = torch.nn.functional.gelu(H_dense)
    Y_ground_truth = torch.matmul(H_act, W_fc2)

    # 2. 模拟张量并行切分
    # ① FC1 列切分 (Column Parallel)
    col_splits = ffn_dim // tp_size
    W_fc1_rank0 = W_fc1[:, :col_splits]
    W_fc1_rank1 = W_fc1[:, col_splits:]

    # ② FC2 行切分 (Row Parallel)
    W_fc2_rank0 = W_fc2[:col_splits, :]
    W_fc2_rank1 = W_fc2[col_splits:, :]

    # 3. 各 Rank 本地独立执行计算 (零通信执行列切 + 激活)
    H_rank0 = torch.matmul(X, W_fc1_rank0)
    H_act_rank0 = torch.nn.functional.gelu(H_rank0)
    Partial_Y_rank0 = torch.matmul(H_act_rank0, W_fc2_rank0)

    H_rank1 = torch.matmul(X, W_fc1_rank1)
    H_act_rank1 = torch.nn.functional.gelu(H_rank1)
    Partial_Y_rank1 = torch.matmul(H_act_rank1, W_fc2_rank1)

    # 4. 执行行切分后的核心通信：模拟 AllReduce 求和
    Y_tp_result = Partial_Y_rank0 + Partial_Y_rank1

    # 5. 数值精度误差校验
    max_abs_diff = torch.max(torch.abs(Y_ground_truth - Y_tp_result)).item()
    print(f"• 单卡标准输出与 TP 切分输出的最大绝对误差: {max_abs_diff:.2e}")
    assert max_abs_diff < 1e-6, "[FAIL] 误差超标，张量并行数学推导存在缺陷！"
    print(">>> 校验结果: [PASS] TP 列切与行切组合输出与单卡全局输出 100% 严格一致！\n")

def run_pure_python():
    import random
    random.seed(42)

    def matmul_2d(A, B):
        M, K = len(A), len(A[0])
        K2, N = len(B), len(B[0])
        res = [[0.0 for _ in range(N)] for _ in range(M)]
        for i in range(M):
            for j in range(N):
                for k in range(K):
                    res[i][j] += A[i][k] * B[k][j]
        return res

    M, K, N_dim = 8, 8, 16 # 8 个 Token, Hidden=8, FFN=16
    tp_size = 2

    print("==================================================================")
    print(f"  Ringi AI Infra 实验室：张量并行 (TP) 算子切分数学等价性验证 (Python 原生轻量引擎)")
    print(f"  配置: Tokens={M}, Hidden={K}, FFN={N_dim}, TP={tp_size}")
    print("==================================================================")

    # 1. 生成随机输入与权重
    X = [[random.gauss(0, 1) for _ in range(K)] for _ in range(M)]
    W_fc1 = [[random.gauss(0, 1) for _ in range(N_dim)] for _ in range(K)]
    W_fc2 = [[random.gauss(0, 1) for _ in range(K)] for _ in range(N_dim)]

    # 单卡标准前向传播
    H = matmul_2d(X, W_fc1)
    Y_ground_truth = matmul_2d(H, W_fc2)

    # 2. 模拟 TP 切分
    mid = N_dim // tp_size
    # 列切分
    W1_r0 = [row[:mid] for row in W1_fc1] if 'W1_fc1' in locals() else [row[:mid] for row in W_fc1]
    W1_r1 = [row[mid:] for row in W_fc1]
    # 行切分
    W2_r0 = W_fc2[:mid]
    W2_r1 = W_fc2[mid:]

    # 3. 本地独立计算
    H_r0 = matmul_2d(X, W1_r0)
    H_r1 = matmul_2d(X, W1_r1)
    Part_r0 = matmul_2d(H_r0, W2_r0)
    Part_r1 = matmul_2d(H_r1, W2_r1)

    # 4. AllReduce 求和
    Y_tp = [[Part_r0[i][j] + Part_r1[i][j] for j in range(K)] for i in range(M)]

    # 5. 校验误差
    max_diff = max(abs(Y_ground_truth[i][j] - Y_tp[i][j]) for i in range(M) for j in range(K))
    print(f"• 单卡标准输出与 TP 切分输出的最大绝对误差: {max_diff:.2e}")
    assert max_diff < 1e-6, "[FAIL] 误差超标，张量并行数学推导存在缺陷！"
    print(">>> 校验结果: [PASS] TP 列切与行切组合输出与单卡全局输出 100% 严格一致！\n")

if __name__ == "__main__":
    try:
        import torch
        run_with_torch()
    except ImportError:
        print("[提示: 本机未安装 PyTorch，无缝自动启用纯 Python 标准库验证引擎]")
        run_pure_python()
```

---

### 6.2 实验二：工业级 3D 并行拓扑配置器与通信/显存/气泡率综合评估器 `megatron_3d_topology_planner.py`

面对百卡、千卡集群，严禁拍脑袋配置 TP、PP、DP。本脚本实现了**工业级 3D 并行拓扑推演评估器**，精确手算显存分布、流水线气泡率、通信总负载并输出决策结论：

```python
"""
Ringi AI Infra 实验室：Minimal Runnable Code
实验二：工业级 3D 并行拓扑配置器与通信/显存/气泡率综合评估器
纯 Python 标准库编写，零外部依赖，控制台安全输出！
"""

class Megatron3DTopologyPlanner:
    def __init__(
        self,
        model_name: str,
        num_layers: int,
        hidden_dim: int,
        vocab_size: int,
        total_params_billion: float,
        total_gpus: int = 512,
        gpus_per_node: int = 8,
        single_gpu_mem_gb: float = 80.0
    ):
        self.model_name = model_name
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.params_b = total_params_billion
        self.total_gpus = total_gpus
        self.gpus_per_node = gpus_per_node
        self.gpu_mem_gb = single_gpu_mem_gb

    def evaluate_plan(self, tp: int, pp: int, dp: int, micro_batches_m: int, v_virtual_stages: int = 1):
        # 1. 基础世界规模约束校验
        allocated_gpus = tp * pp * dp
        is_valid_world = (allocated_gpus == self.total_gpus)

        # 2. 硬件拓扑匹配检查
        tp_exceeds_node = (tp > self.gpus_per_node)

        # 3. 流水线气泡率推导 (1F1B vs Interleaved)
        # 公式: Bubble = (PP - 1) / (v * M + PP - 1)
        bubble_ratio = (pp - 1) / (v_virtual_stages * micro_batches_m + pp - 1)

        # 4. 单卡静态显存粗略估算 (BF16 参数 2B + 梯度 2B + AdamW 12B = 16B)
        # 在 TP * PP 切分下，静态参数被切分为 1 / (TP * PP)
        total_static_gb = (self.params_b * 1e9 * 16) / (1024 ** 3)
        static_mem_per_gpu = total_static_gb / (tp * pp)

        # 5. 单层 TP 单步通信量估算 (以 b=1, s=4096 为例，每层 16 * bsh 字节)
        # 单卡单步 TP 裸通信量 (GB)
        layers_per_stage = self.num_layers / pp
        tp_comm_per_step_gb = (layers_per_stage * 16 * 1 * 4096 * self.hidden_dim * 2) / (1024 ** 3)

        return {
            "valid_world": is_valid_world,
            "tp_exceeds_node": tp_exceeds_node,
            "bubble_pct": bubble_ratio * 100,
            "static_mem_gb": static_mem_per_gpu,
            "tp_comm_gb": tp_comm_per_step_gb,
            "can_fit_mem": static_mem_per_gpu < (self.gpu_mem_gb * 0.7) # 预留 30% 给激活与临时缓存
        }

if __name__ == "__main__":
    planner = Megatron3DTopologyPlanner(
        model_name="LLaMA-3-70B",
        num_layers=80,
        hidden_dim=8192,
        vocab_size=128256,
        total_params_billion=70.6,
        total_gpus=512,
        gpus_per_node=8,
        single_gpu_mem_gb=80.0
    )

    plans = [
        {"name": "方案 A (盲开跨机大TP)", "tp": 16, "pp": 4, "dp": 8, "m": 32, "v": 1},
        {"name": "方案 B (经典机内满血TP)", "tp": 8, "pp": 4, "dp": 16, "m": 32, "v": 1},
        {"name": "方案 C (深度PP+虚拟Stage)", "tp": 8, "pp": 8, "dp": 8, "m": 32, "v": 2},
        {"name": "方案 D (纯数据并行FSDP)", "tp": 1, "pp": 1, "dp": 512, "m": 1, "v": 1},
    ]

    print("=========================================================================================================")
    print(f"  Ringi AI Infra 决策推演：{planner.model_name} 在 {planner.total_gpus} 卡集群上的 3D 并行拓扑全景对比")
    print("=========================================================================================================")
    print(f"{'方案名称':<22} | {'TP-PP-DP':<10} | {'静态显存':<10} | {'PP气泡率':<9} | {'拓扑安全性':<14} | {'综合评估'}")
    print("---------------------------------------------------------------------------------------------------------")

    for p in plans:
        res = planner.evaluate_plan(p["tp"], p["pp"], p["dp"], p["m"], p["v"])
        if not res["valid_world"]:
            status = "[FAIL] 节点算力未打满"
            eval_str = "卡数不匹配"
        elif res["tp_exceeds_node"]:
            status = "[WARN] TP出机塞爆IB"
            eval_str = "吞吐雪崩 (禁止上线!)"
        elif not res["can_fit_mem"]:
            status = "[OOM] 静态显存超标"
            eval_str = "无法启动"
        else:
            status = "[PASS] 拓扑严密对齐"
            eval_str = f"推荐生产使用 (气泡{res['bubble_pct']:.1f}%)"

        print(f"{p['name']:<22} | {p['tp']}x{p['pp']}x{p['dp']:<6} | {res['static_mem_gb']:6.1f} GB  | {res['bubble_pct']:6.1f}%  | {status:<14} | {eval_str}")

    print("=========================================================================================================")
    print(">>> 架构师选型建议：")
    print("1. 方案 A 虽然把参数切得很细，但 TP=16 强行跨越两台机器，机间 IB 网卡被高频 AllReduce 瞬间击穿；")
    print("2. 方案 B 是工业界最经典最稳健的黄金配置：TP=8 占满机内 NVLink，PP=4 气泡仅 8.6%，静态显存仅 32.9 GB；")
    print("3. 方案 C 适用于更大参数或显存极度紧张场景，配合 Interleaved (v=2) 成功将气泡压制在 9.9% 高位！")
```

---

# 7. Ringi 避坑指南与生产黄金准则

### 7.1 7 大常见小白认知误区 vs 大厂 AI Infra 正确物理认知

| 序号 | ❌ 常见小白错误理解 | ✅ 大厂 AI Infra 正确物理认知 | 体系结构本质与底层机理解析 |
| :--- | :--- | :--- | :--- |
| **01** | “TP 越大越好，既然有 64 张卡，直接开 TP=64 加速比最高。” | **TP 必须严格锁死在单机 8 卡内部（$\text{TP} \le 8$），绝对不能跨机！** | TP 每层都在触发 AllReduce，跨机延迟比 NVLink 高一个数量级，跨机 TP 会让网络瞬间沦为全系统绝对瓶颈。 |
| **02** | “序列并行（SP）多了一次 AllGather 和 ReduceScatter，所以比普通 TP 慢。” | **SP 与普通 TP 的通信总量完全守恒（均为 $2bsh$），通信耗时几乎完全相同！** | $\text{AllReduce} = \text{ReduceScatter} + \text{AllGather}$，SP 只是把通信原语拆开重排，零额外开销省下数倍显存。 |
| **03** | “流水线并行（PP）不管切多少个 Stage，算力利用率都是一样的。” | **PP 存在固有的流水线气泡，Stage 数 $P$ 越多，必须配置更多的微批次 $M$ 才能冲淡气泡。** | 气泡率严格正比于 $\frac{P-1}{M+P-1}$，盲目拉大 $P$ 而没有足够的 Batch 注入会导致一半以上的 GPU 白白空转。 |
| **04** | “长序列训练只要开 FlashAttention 就能跑 128K，不需要搞上下文并行（CP）。” | **FlashAttention 只能把显存从 $O(s^2)$ 降到 $O(s)$，当 $s=128\text{K}$ 时，$O(s)$ 的 KV 激活依然会打穿单卡 80GB！** | 突破 64K~1M 超长序列必须配合 CP（Ring Attention 或 Ulysses），把序列本身跨卡切碎。 |
| **05** | “1F1B 调度比 GPipe 跑得更快，气泡率更小。” | **1F1B 与 GPipe 的稳态气泡率在数学上完全相同，1F1B 的核心革命在于大幅压缩峰值激活显存！** | 1F1B 及时用反向释放了前向激活，把存活微批次从 $M$ 压低到了 $P$，避免了显存爆仓。 |
| **06** | “GQA 架构（如 8 个 KV Head）下，张量并行度 TP 可以随意设为 16。” | **在标准 Megatron-TP 下，TP 必须能被注意力头数整除；当 KV Head < TP 时，必须显式复制或分组共享 KV！** | 否则单个 GPU 连一个完整的 KV Head 都分不到，矩阵乘法在维度对齐上直接崩盘。 |
| **07** | “3D 并行配好后，只要看整体 Step Time 就能判断网络有没有问题。” | **必须深入 NCCL Profiler 查看不同通信组的耗时分布，防止出现‘慢卡拖垮全流水线’的 Straggler 效应。** | PP 的前向和反向存在级联依赖，只要其中一个 Stage 稍有抖动，气泡就会沿整条流水线成倍扩散放大。 |

---

### 7.2 生产模型并行工程黄金 Checklist

- [ ] 1. **【TP 单机闭环铁律】**：张量并行度必须满足 $\text{TP} \le \text{gpus\_per\_node}$（通常 $\text{TP} \le 8$），严禁分配超出物理节点的高频 TP。
- [ ] 2. **【SP 无条件协同】**：只要在 Megatron-LM 或框架中开启了 `--tensor-model-parallel-size > 1`，必须显式加上 `--sequence-parallel`，享受免费的激活显存压降。
- [ ] 3. **【微批次倍数约束】**：在配置流水线并行时，微批次数量 $M$ 必须至少满足 $M \ge 4 \times P$，确保流水线气泡率严格控制在 20% 以下。
- [ ] 4. **【Interleaved 虚拟 Stage 权衡】**：仅在机间 InfiniBand 带宽极其充裕（如 8x400G IB）时才开启 `v_virtual_stages >= 2`，防止激活值跨机传输翻倍抵消算力收益。
- [ ] 5. **【GQA 头数整除性审查】**：审查模型架构参数，确保 Query Head 与 KV Head 均能被 `TP` 或 `TP / Group` 整除，杜绝隐式填充和非均匀切分。
- [ ] 6. **【Rank 笛卡尔积拓扑对齐】**：启动脚本前必须打印并核验 `parallel_state` 中的通信组 Rank 映射，确保 TP 进程严格绑定在同一 PCIe/NVSwitch 拓扑树下。
- [ ] 7. **【CP 方案根据 Head 与序列选型】**：文本长度 $\le 64\text{K}$ 且 Head 数充裕首选 DeepSpeed-Ulysses；文本长度 $\ge 128\text{K}$ 或 GQA 极度紧凑首选 Zigzag Ring-Attention。

---

# 8. Ringi 5 点核心速记口诀、自我检验清单与课后深度思考题

### 8.1 5 点押韵核心速记口诀

```text
算子切分看两层，列切行切两聚合；
张量并行锁机内，九百带宽快如梭；
序列并行免单费，全规拆开省激活；
流水气泡调微批，一前一后稳如铁；
三维并行排四海，万卡集群破天界！
```

---

### 8.2 10 条白板自我检验清单

1. 能否在白板上手画一个 Transformer Block，标明 FC1 列切、FC2 行切的矩阵维度变化以及 AllReduce 的具体插入位置？
2. 为什么在 Column Parallel 中，前向传播不需要任何卡间通信，而通信压力全部转移到了反向传播？
3. 阐明序列并行（SP）是如何利用 $\text{AllReduce} = \text{ReduceScatter} + \text{AllGather}$ 的代数恒等式，在零增加通信量的前提下砍掉非 TP 区域显存的？
4. 闭卷推导标准 1F1B 流水线调度的气泡率公式：为什么微批次数 $M$ 越大，气泡率越小？
5. 为什么说 1F1B 并没有在数学上减少 GPipe 的稳态气泡，但它却是工业界唯一的救命稻草？
6. 在 Interleaved 1F1B（虚拟 Stage）中，为什么将每个 GPU 切分为 $v$ 个虚拟阶段能够将气泡率再除以 $v$？它付出的硬件代价是什么？
7. 对比 Ring-Attention 与 DeepSpeed-Ulysses：它们分别使用了哪种集合通信原语？在 GQA 架构下各有什么限制？
8. 为什么在 3D 并行拓扑映射中，必须严格遵循“TP 在机内、PP 跨节点、DP 在最外层”的硬件阶梯原则？
9. 当训练长文本（如 64K）时，为什么不能简单地无脑把 TP 从 8 开到 16？
10. Megatron-LM 的 `parallel_state.py` 是如何利用步长乘积构建多维通信组 NCCL Communicator 的？

---

### 8.3 3 道高阶开放式课后思考题（含极限 Corner Case）

1. **【GQA 架构下的 TP 极度非均匀切分 Corner Case】**：在 LLaMA-3-70B 中，Query 头数为 64，但 KV 头数仅为 8（GQA 比率为 8:1）。如果我们设置 $\text{TP} = 8$，则每张 GPU 分配到 8 个 Q 头和 1 个 KV 头，切分完美均衡。但是如果我们为了追求更小的单卡显存，强行在跨节点多机上设置 $\text{TP} = 16$：此时 8 个 KV 头无法均分给 16 张 GPU。工程上通常有两种处理方案：① 强制在每 2 张 GPU 之间复制一份相同的 KV Head；② 将 KV Head 重组为分组切分。请从计算冗余、显存浪费与 NCCL 通信原语变动三个维度，推导这两种方案对端到端 MFU 的致命冲击。
2. **【超长上下文 CP 的因果掩码踩踏事故】**：在自回归语言模型中，因果掩码（Causal Mask）呈现严格的下三角矩阵结构。假设采用朴素的 Ring Attention 沿序列均匀切分为 8 个 Chunk（GPU 0~7）：GPU 0 负责第 1 块，GPU 7 负责最后 1 块。请画图推导演讲：在第 0 步和最后一步计算中，GPU 0 的计算量与 GPU 7 的计算量相差多少倍？为什么这种极度的算力倾斜会导致全网严重的 Straggler 阻塞？Zigzag Ring-Attention 是通过何种映射置换消除这一气泡的？
3. **【跨机 PP 慢节点引发的“毒性扩散”】**：在一个包含 8 个 Stage 的流水线并行集群中，假设位于 Stage 3 的某张 GPU 由于 PCIe 降速（如退化为 Gen3 x4）导致计算耗时拉长了 30%。请从 1F1B 调度的时间线推演：这个局部的微小延迟是如何像滚雪球一样向前反向传播（阻碍 Stage 2 的反向求导）、向后正向传播（延迟 Stage 4 的前向激活），最终导致整个千卡集群的 GPU 利用率全部腰斩的？针对这种跨机抖动，现代智算平台在调度与自动容灾上应如何设计心跳探测？

---

# 9. 📚 参考资料与核心源码/经典论文指引

### 权威学术论文：
1. **Megatron-LM TP 奠基之作**：Shoeybi et al., *"Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism"*, 2019. [arXiv:1909.08053](https://arxiv.org/abs/1909.08053)
2. **Megatron-SP 序列并行**：Korthikanti et al., *"Reducing Activation Recomputation in Large Transformer Models"*, MLSys 2023. [arXiv:2205.05198](https://arxiv.org/abs/2205.05198)
3. **GPipe 微批次流水线**：Huang et al., *"GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism"*, NeurIPS 2019. [arXiv:1811.06965](https://arxiv.org/abs/1811.06965)
4. **PipeDream 1F1B 调度**：Narayanan et al., *"Memory-Efficient Pipeline-Parallel DNN Training"*, ICML 2021. [arXiv:2006.09503](https://arxiv.org/abs/2006.09503)
5. **Ring-Attention 长序列**：Liu et al., *"Ring Attention with Block Paged Memory for Exceedingly Long Sequences"*, 2023. [arXiv:2310.01889](https://arxiv.org/abs/2310.01889)
6. **DeepSpeed-Ulysses 序列转置**：Jacobs et al., *"DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models"*, 2023. [arXiv:2309.14509](https://arxiv.org/abs/2309.14509)

### 工业级开源源码指引：
1. **Megatron-LM 张量切分源码**：`megatron/core/tensor_parallel/layers.py`（包含 `ColumnParallelLinear` 与 `RowParallelLinear`）
2. **Megatron-LM 序列并行通信**：`megatron/core/tensor_parallel/mappings.py`（包含 `reduce_scatter_to_sequence_parallel_region` 与 `gather_from_sequence_parallel_region`）
3. **Megatron-LM 3D 拓扑构建**：`megatron/core/parallel_state.py`（包含 `initialize_model_parallel` 笛卡尔积通信域构建）

### 本地 AI_BOOK 知识库精准映射：
- 张量并行与序列并行详解：[第6章 张量并行TP与序列并行SP.md](file:///d:/GeneTind/Interview/AI_BOOK/AIInfraGuide/docs/guides/模块三-分布式训练/第6章-张量并行与序列并行.md)
- 流水线并行与气泡调度：[第7章 流水线并行PP.md](file:///d:/GeneTind/Interview/AI_BOOK/AIInfraGuide/docs/guides/模块三-分布式训练/第7章-流水线并行.md)
- 长序列训练与上下文并行：[第9章 长序列训练与上下文并行.md](file:///d:/GeneTind/Interview/AI_BOOK/AIInfraGuide/docs/guides/模块三-分布式训练/第9章-长序列训练与上下文并行.md)
- 3D 混合并行与拓扑映射：[第11章 3D并行与混合并行策略.md](file:///d:/GeneTind/Interview/AI_BOOK/AIInfraGuide/docs/guides/模块三-分布式训练/第11章-3D并行与混合并行策略.md)
- 多维混合并行大厂面试总结：[6. 多维度混合并行.md](file:///d:/GeneTind/Interview/AI_BOOK/llm_interview_note/04.分布式训练/6.多维度混合并行/6.多维度混合并行.md)

---

# 附录：Appendix A — 大厂硬核高频面试题与白板推导（Interview Drill）

### 面试真题 1：请白板画图推导 Transformer Block 在 TP 下前向与反向通信量为什么是 $4bsh$？为什么 TP 严禁出单机？

#### 考察维度：张量并行微观通信机理、矩阵分块相乘法则、算力与网络延迟边界。
#### 标准推导路径：
1. **前向传播（Forward Pass）分析**：
   - **Self-Attention 模块**：$Q, K, V$ 投影按列切分，无通信；输出投影矩阵按行切分，根据分块矩阵乘法 $Y = \sum X_i W_i$，必须在各卡之间对大小为 $[b, s, h]$ 的局部求和张量执行一次 **AllReduce**；
   - **MLP 模块**：FC1 矩阵按列切分，逐元素激活函数无通信；FC2 矩阵按行切分，再次对大小为 $[b, s, h]$ 的局部张量执行一次 **AllReduce**；
   - **单卡发送量**：每次 Ring-AllReduce 单卡发送数据量为 $2 \frac{N-1}{N} \times \text{Size} \approx 2 bsh$（Words）；
   - **前向总发送量**：$2 \text{ 次 AllReduce} \times 2 bsh = \mathbf{4 bsh} \quad (\text{Words})$。
2. **反向传播（Backward Pass）对称性分析**：
   - 对行切分层求输入梯度时，依据伴随转置，反向计算变为按列切分；
   - 对列切分层求输入梯度时，反向计算变为按行切分，必须再次插入一次 **AllReduce**；
   - 因此反向传播同样需要精确触发 2 次 AllReduce，单卡发送量严格等于：$\mathbf{4 bsh} \quad (\text{Words})$。
3. **单层单步总通信量累加**：
   $$\text{Total Comm Per Layer} = \text{Comm}_{\text{fwd}} + \text{Comm}_{\text{bwd}} = 4bsh + 4bsh = \mathbf{8bsh} \quad (\text{Words}) = \mathbf{16bsh} \quad (\text{Bytes})$$
4. **为什么严禁出机**：
   - 设单层前向 GEMM 耗时仅 1~2 毫秒；
   - 若在机内 NVLink（900 GB/s，延迟 $< 1\mu s$），传输几十兆数据仅需数十微秒，完全被计算掩盖；
   - 若跨机走 InfiniBand（50 GB/s，跨交换机延迟 $3\sim 5\mu s$），小包排队与协议栈延迟直接飙升到数毫秒，网络耗时反超计算耗时数倍，全集群 MFU 当场跌破 15%！

---

### 面试真题 2：白板推导 1F1B 流水线气泡率公式，Interleaved 1F1B 是如何通过虚拟 Stage 压缩气泡的？代价是什么？

#### 考察维度：流水线并行调度原理、时间片甘特图推导、通信计算 Trade-off。
#### 标准参考答案：
1. **1F1B 稳态气泡率数学推导**：
   - 设流水线包含 $P$ 个 Stage，全局批次被切分成 $M$ 个微批次（Micro-batches）；
   - 设单个 Micro-batch 在单个 Stage 上的前向耗时为 $t_f$，反向耗时为 $t_b$（通常 $t_b \approx 2 t_f$），此处为简化推导设理想均匀时间片为 $t_{\text{step}}$；
   - **充能与排空空转时间（Bubble Time）**：
     - 在第 0 个微批次从 Stage 0 到达 Stage $P-1$ 的过程中，后序节点处于空等，共有 $P - 1$ 个时间步的空转；
     - 在反向传播全部结束排空时，前序节点在等待后序节点反向，又有 $P - 1$ 个时间步的空转；
     - 全流水线单个物理周期的总空转时间为：$t_{\text{bubble}} = (P - 1) \times t_{\text{step}}$；
   - **全流程有效计算时间**：
     - 每个微批次必须完整跑完前向与反向，总有效微批次步数为 $M \times t_{\text{step}}$；
   - **端到端总执行时间**：$T_{\text{total}} = (M + P - 1) \times t_{\text{step}}$；
   - **稳态气泡率公式**：
     $$\text{Bubble Ratio} = \frac{t_{\text{bubble}}}{T_{\text{total}}} = \mathbf{\frac{P - 1}{M + P - 1}}$$
2. **Interleaved 1F1B 虚拟阶段压缩机理**：
   - 每个物理 GPU 不再只管一个大阶段，而是将其细化为 $v$ 个交错的虚拟阶段（Virtual Stages）；
   - 单个微批次在每个虚拟阶段的计算耗时缩小为 $\frac{t_{\text{step}}}{v}$；
   - 充能与排空等待时间缩短为 $(P - 1) \times \frac{t_{\text{step}}}{v}$；
   - **压缩后气泡率**：
     $$\text{Bubble Ratio}_{\text{interleaved}} = \mathbf{\frac{P - 1}{v \cdot M + P - 1}}$$
   - 气泡率被**等效扩大了 $v$ 倍的微批次数所稀释**，气泡面积直接削减近 $1/v$！
3. **付出的代价（Trade-off）**：
   - 物理 Stage 数量虽然不变，但层间切断的边界增加了 $v$ 倍；
   - 相邻虚拟 Stage 跨物理设备传递中间激活张量的**通信频次与传输总量直接翻了 $v$ 倍**；
   - 仅当集群机间网络带宽极其充裕时，这种“用额外通信换低气泡”的策略才能够带来正向收益。

---

### 面试真题 3：Ring Attention 与 DeepSpeed-Ulysses 在长上下文训练中的核心切分机制与通信原语有何本质不同？GQA 下如何选型？

#### 考察维度：长序列并行最新演进、集合通信原语差异、现代 GQA 架构适配。
#### 标准参考答案：
1. **核心切分维度与通信原语本质差异**：
   - **Ring Attention**：
     - **切分维度**：将整条长序列沿 Sequence 维度切成 $N$ 段，每张 GPU 拥有局部 $Q_i, K_i, V_i$；
     - **通信原语**：使用环形点对点通信（**P2P Send/Recv**）。$Q_i$ 留在原地，各卡将 $K_i, V_i$ 块沿环逐跳传递，配合 FlashAttention Online Softmax 动态累积归一化因子；
   - **DeepSpeed-Ulysses**：
     - **切分维度**：输入时沿 Sequence 维度切分，但在执行 Attention 计算前，通过 **All-to-All** 通信原语将序列切分转置为注意力头（Head）维度切分；
     - **通信原语**：注意力计算前后各执行一次全局 **All-to-All**。在注意力内核内部，每张卡直接跑全长度、少头数的标准 FlashAttention。
2. **现代 GQA 架构下的选型决策**：
   - **问题瓶颈**：在现代主流模型（如 LLaMA-3、Mistral）中，普遍采用分组查询注意力（GQA），KV 头数极其稀疏（通常仅 8 个）；
   - **Ulysses 的致命短板**：Ulysses 强制要求注意力头数必须能被 CP 并行度整除。若 KV Head=8，则 Ulysses 的上下文并行度**绝对无法超过 8**！若想扩展到 16 卡或 32 卡，必须引入极其复杂的跨卡 KV 复制；
   - **Ring Attention 的绝对优势**：Ring Attention 沿序列本身切分，对 Head 数量没有任何整除性限制；在 GQA 下传输的 KV 分片尺寸极小，不仅能无缝扩展到数十上百张卡，通信量还获得了数倍的自然压缩；
   - **选型结论**：面对 GQA 稀疏头架构与 128K 以上极限长序列，**必须首选 Zigzag Ring-Attention**；而在 32K 级别、且 Query 与 KV 均为密集多头（MHA）的场景下，**Ulysses 凭借优异的 All-to-All 吞吐具有更高的工程性价比**。
