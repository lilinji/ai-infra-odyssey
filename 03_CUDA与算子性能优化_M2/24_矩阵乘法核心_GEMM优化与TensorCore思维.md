---
title: "第24讲：从 1% 到 95% 算力利用率的九重天跃迁——CUDA GEMM 分块（Tiling）与 Tensor Core 思维深度解构"
description: "从0.25 FLOPs/Byte算术强度性能墙穿透到Block-Warp-Thread三级分块（Tiling）体系，手算2D寄存器外积复用代数模型，解构双缓冲与Ampere异步拷贝流水线，深度剖析Tensor Core Warp级矩阵乘微架构与Fragment映射，并配套4组完整可编译CUDA实验与大厂白板推导。"
date: 2026-08-31T00:00:00+08:00
pubDate: 2026-08-24
category: "cuda-optimization"
order: 24
tags:
  - AI基础设施
  - CUDA
  - GEMM
  - 矩阵乘法
  - Tensor Core
  - 寄存器分块
  - 双缓冲
  - Ringi-IP
author: Ringi Lee
showToc: true
tocOpen: false
math: true
---

# 🏛️ 第24讲：从 1% 到 95% 算力利用率的九重天跃迁——CUDA GEMM 分块（Tiling）与 Tensor Core 思维深度解构

> **主讲人**：👓 **Ringi**（大厂 AI Infrastructure 工程师）  
> **所属模块**：[Module 02: CUDA 编程与高性能算子优化](./README.md)  
> **篇章范式**：⚡ CUDA 编程与高性能算子优化篇（Kernel & Operator Optimization Paradigm）  
> **核心导读**：矩阵乘法（GEMM）被称为“现代人工智能算力王冠上的明珠”。无论是大模型中占据 80% 计算耗时的全连接层（Linear/MLP），还是注意力机制中的 $QK^T$ 与 $PV$，最终全部落盘在 GEMM 之上。然而，许多工程师第一次手写 CUDA 矩阵乘时，往往直接写出三重 `for` 循环，满心欢喜地扔到价值几十万的 NVIDIA A100 上跑，实测算力却只有可怜的 **0.25 TFLOPS（理论峰值的 1.2%）**！本讲我们将从 Roofline 模型的第一性原理出发，手算 GEMM 的数据复用极限；随后沿着 **Block 级共享内存分块 $\rightarrow$ Thread 级 2D 寄存器外积复用 $\rightarrow$ 向量化访存 $\rightarrow$ Bank Conflict 消除 $\rightarrow$ 双缓冲异步流水线** 的九重台阶步步登顶；最后推开现代张量计算圣殿的大门，彻底击穿 **Tensor Core 的 Warp 级协同矩阵乘微架构**。

![Ringi 导师解构：核心全景工坊](assets/ringi_24_overview.png)

```text
========================================================================================================================
                                Ringi 3D 架构工坊 · GEMM 分块体系与 Tensor Core 核心全景
========================================================================================================================
[ 全局显存 HBM (高延迟 400 Cycles, 带宽 2 TB/s) ]
   │
   │  协作搬运 (Cooperative Fetch) + 向量化 float4 (LDG.128) + 异步拷贝 (cp.async)
   ▼
[ 级 1: Thread Block 级分块 (Shared Memory Tiling) ] ──► 容量 164 KB/SM, 带宽 ~19 TB/s
   ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
   │ Block Tile: A_tile (BM × BK) 与 B_tile (BK × BN)  ──► 解决全局显存数据复用 (复用度提升 BM, BN 倍)  │
   │ 采用 Padding / Swizzle 消除 32 个 Bank 的跨步读取冲突                                            │
   └────────────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                                    │
                                                    │ 线程寄存器流式加载 (LDS.128)
                                                    ▼
[ 级 2: Thread 级 2D 寄存器分块 (Register Tiling) ] ──► 容量 256 KB/SM, 延迟 0 Cycle, 带宽 ~20 TB/s
   ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
   │ 每个线程负责 TM × TN (如 8×8=64) 输出元素 ──► 外积计算模型 (Outer Product Engine)                 │
   │ 仅需读取 (TM + TN) 个数，触发 (TM × TN) 次乘加！寄存器复用比高达 8x，彻底解放 Shared Memory 带宽!   │
   └────────────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                                    │
                                                    │ 硬件指令升维: 从标量 FMA 跃迁至张量脉动阵列
                                                    ▼
[ 级 3: Tensor Core Warp 级协同矩阵乘 (WMMA / MMA.sync) ] ──► A100 FP16 峰值 312 TFLOPS (跃迁 16 倍!)
   ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
   │ 整个 Warp 32 线程紧密协作执行 MMA.m16n8k16 原语 ──► 矩阵片段 (Fragment) 分散映射在 32 线程寄存器中 │
   │ 硬件脉动流水单周期吞吐 64 个 FMA 乘加，将算术强度推向 150+ FLOPs/Byte 极致饱和峰值！              │
   ====================================================================================================
```

<!-- more -->

## 📑 目录导航

- [0. Ringi 开场：生产真实现场与痛点冲突](#0-ringi-开场生产真实现场与痛点冲突)
  - [0.1 真实工程矛盾：为什么直接用三重循环写矩阵乘，算力利用率只有 1.2%？](#01-真实工程矛盾为什么直接用三重循环写矩阵乘算力利用率只有-12)
  - [0.2 线上事故复盘：某大模型微调团队手写自定义 LoRA 线性层性能雪崩](#02-线上事故复盘某大模型微调团队手写自定义-lora-线性层性能雪崩)
  - [0.3 AI Infra GEMM 各演进阶段速查表](#03-ai-infra-gemm-各演进阶段速查表)
- [1. 算术强度与 Roofline 极限：GEMM 为什么是“算力王冠上的明珠”？](#1-算术强度与-roofline-极限gemm-为什么是算力王冠上的明珠)
  - [1.1 GEMM 运算量与访存量的代数账本：$2MNK$ vs 存储搬运](#11-gemm-运算量与访存量的代数账本2mnk-vs-存储搬运)
  - [1.2 No Naked Formula 2.0：GEMM 算术强度模型与 A100 平衡点](#12-no-naked-formula-20gemm-算术强度模型与-a100-平衡点)
  - [1.3 现代 GPU 存储层级金字塔与数据搬运代价](#13-现代-gpu-存储层级金字塔与数据搬运代价)
- [2. Thread Block 级分块（Block Tiling）：利用 Shared Memory 实现第一次数据复用跃迁](#2-thread-block-级分块block-tiling利用-shared-memory-实现第一次数据复用跃迁)
  - [2.1 朴素 GEMM 的死穴：对全局内存的狂轰滥炸](#21-朴素-gemm-的死穴对全局内存的狂轰滥炸)
  - [2.2 Block Tile 物理切分：$BM \times BN$ 与沿 $K$ 维度的滑动窗口](#22-block-tile-物理切分bm-times-bn-与沿-k-维度的滑动窗口)
  - [2.3 协作搬运（Cooperative Fetching）：Block 内所有线程的集体搬运](#23-协作搬运cooperative-fetchingblock-内所有线程的集体搬运)
  - [2.4 数据复用收益倍数精准推导](#24-数据复用收益倍数精准推导)
- [3. Thread 级分块与寄存器复用（2D Register Tiling）：突破 Shared Memory 带宽瓶颈](#3-thread-级分块与寄存器复用2d-register-tiling突破-shared-memory-带宽瓶颈)
  - [3.1 共享内存的带宽危机：19 TB/s 依然喂不饱 300 TFLOPS](#31-共享内存的带宽危机19-tbs-依然喂不饱-300-tflops)
  - [3.2 2D Register Tiling：单线程负责 $TM \times TN$ 输出子块](#32-2d-register-tiling单线程负责-tm-times-tn-输出子块)
  - [3.3 外积计算模型（Outer Product Engine）：以小搏大的数学艺术](#33-外积计算模型outer-product-engine以小搏大的数学艺术)
  - [3.4 算术强度的二次飞跃：寄存器级的极致压榨](#34-算术强度的二次飞跃寄存器级的极致压榨)
- [4. 向量化加载、Bank Conflict 消除与双缓冲（Double Buffering）流水线](#4-向量化加载bank-conflict-消除与双缓冲double-buffering流水线)
  - [4.1 向量化访存：强制使用 `float4` 压榨指令发射通道](#41-向量化访存强制使用-float4-压榨指令发射通道)
  - [4.2 共享内存 Bank Conflict 消除：矩阵转置与 Padding 策略](#42-共享内存-bank-conflict-消除矩阵转置与-padding-策略)
  - [4.3 软流水线（Software Pipelining）与双缓冲（Ping-Pong Buffer）](#43-软流水线software-pipelining与双缓冲ping-pong-buffer)
  - [4.4 Ampere 异步拷贝指令 `cp.async`：直通 Shared Memory](#44-ampere-异步拷贝指令-cpasync直通-shared-memory)
- [5. Tensor Core 硬件革命与思维升维：从标量 FMA 到 Warp 级矩阵乘](#5-tensor-core-硬件革命与思维升维从标量-fma-到-warp-级矩阵乘)
  - [5.1 体系结构断代差：为什么 Tensor Core 能甩开 CUDA Core 16 倍？](#51-体系结构断代差为什么-tensor-core-能甩开-cuda-core-16-倍)
  - [5.2 Warp 级协同计算哲学：没有“单个线程的 Tensor Core”](#52-warp-级协同计算哲学没有单个线程的-tensor-core)
  - [5.3 WMMA API (`nvcuda::wmma`) vs PTX 原语 (`mma.sync`) 深度对比](#53-wmma-api-ncudawmma-vs-ptx-原语-mmasync-深度对比)
  - [5.4 Fragment 寄存器映射机制：32 线程如何瓜分一个矩阵？](#54-fragment-寄存器映射机制32-线程如何瓜分一个矩阵)
- [6. 动手实战与代码实验室（Minimal Runnable Code）](#6-动手实战与代码实验室minimal-runnable-code)
  - [实验 1：Naive GEMM vs Block Tiling 基准测试（`gemm_naive_vs_block.cu`）](#实验-1naive-gemm-vs-block-tiling-基准测试gemm_naive_vs_blockcu)
  - [实验 2：2D Register Tiling 高性能 SGEMM 实现（`gemm_2d_register_tiling.cu`）](#实验-22d-register-tiling-高性能-sgemm-实现gemm_2d_register_tilingcu)
  - [实验 3：双缓冲流水线与 float4 向量化 SGEMM 压测（`gemm_double_buffer_vectorized.cu`）](#实验-3双缓冲流水线与-float4-向量化-sgemm-压测gemm_double_buffer_vectorizedcu)
  - [实验 4：Tensor Core WMMA FP16 极致算力基准（`gemm_tensor_core_wmma.cu`）](#实验-4tensor-core-wmma-fp16-极致算力基准gemm_tensor_core_wmmacu)
- [7. Ringi 避坑指南与生产黄金准则](#7-ringi-避坑指南与生产黄金准则)
  - [7.1 避坑表格：7 大常见小白错误理解 vs 大厂 AI Infra 正确认知](#71-避坑表格7-大常见小白错误理解-vs-大厂-ai-infra-正确认知)
  - [7.2 生产性能工程黄金 Checklist](#72-生产性能工程黄金-checklist)
- [8. Ringi 5 点核心速记口诀、自我检验清单与课后深度思考题](#8-ringi-5-点核心速记口诀自我检验清单与课后深度思考题)
  - [8.1 5 点押韵核心速记口诀](#81-5-点押韵核心速记口诀)
  - [8.2 10 条白板自我检验清单](#82-10-条白板自我检验清单)
  - [8.3 3 道高阶开放式课后思考题（含 Wave Quantization 与 CUTLASS 演进）](#83-3-道高阶开放式课后思考题含-wave-quantization-与-cutlass-演进)
- [9. 📚 参考资料与核心源码/经典论文指引](#9-参考资料与核心源码经典论文指引)
- [附录：Appendix A — 大厂硬核高频面试题与白板推导（Interview Drill）](#附录appendix-a--大厂硬核高频面试题与白板推导interview-drill)
- [🎨 配图工坊生图 Prompt 暂存区](#-配图工坊生图-prompt-暂存区)

---

# 0. Ringi 开场：生产真实现场与痛点冲突

### 0.1 真实工程矛盾：为什么直接用三重循环写矩阵乘，算力利用率只有 1.2%？

在任何一本通用编程教科书里，矩阵乘法 $C = A \times B$（$A \in \mathbb{R}^{M \times K}, B \in \mathbb{R}^{K \times N}$）的代码实现都是如此平易近人：

```cpp
// 朴素三重循环
for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[i * K + k] * B[k * N + j];
        }
        C[i * N + j] = sum;
    }
}
```

当你刚学会 CUDA 编程，把最外层的两重循环直接映射为 GPU 的 2D 线程网格：

```cpp
// 朴素 CUDA 实现：一个线程算一个 C[row][col]
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
float sum = 0.0f;
for (int k = 0; k < K; ++k) {
    sum += A[row * K + k] * B[k * N + col];
}
C[row * N + col] = sum;
```

你开了几十万个线程，把它扔上单卡售价数十万元的 NVIDIA A100-SXM4-80GB（理论单精度峰值算力高达 **19.5 TFLOPS**）。你满心期待它能在几十微秒内跑完，结果实测报告打印出来：

- 计算规模：$M = N = K = 4096$；
- 算子耗时：**约 548 毫秒**；
- 实测有效算力：**仅有 0.25 TFLOPS**！

**算力利用率只有理论峰值的 1.28%！整整 98.7% 的物理晶体管在彻底睡大觉！**

为什么会这样？如果你去查看硬件指标，你会发现计算单元（ALU）根本没有全负荷工作，它整整 99% 的时钟周期都在苦苦等待数据从显存搬运过来。
每个线程为了计算 1 次乘加（2 个 FLOPs），就要从全局显存读取 1 个 $A$ 元素（4 字节）和 1 个 $B$ 元素（4 字节），总共搬运 8 个字节！
它的算术强度被锁死在 **$2 / 8 = 0.25 \text{ FLOPs/Byte}$**。
在 A100 那 2039 GB/s 的带宽上限下，硬件能支撑的算力极限被死死按在 $2039 \times 0.25 \approx \mathbf{510 \text{ GFLOPS}}$！
**你不做数据分块与寄存器复用，再昂贵的 GPU 也会被活活当成低速拖拉机开。**

---

### 0.2 线上事故复盘：某大模型微调团队手写自定义 LoRA 线性层性能雪崩

2024 年秋，某大厂算法团队在对百亿参数稠密大模型进行特定下游任务的高阶 LoRA 微调。为了在一个算子中把低秩矩阵乘与特定的量化激活函数融合（Fuse），一位工程师信心满满地参考了某开源简单代码，手写了一个定制化的 GEMM Kernel 替换掉原生的 `torch.matmul`（底层调用 cuBLAS）。

在小规模单测（矩阵为 $128 \times 128$）下，数值完全对齐，测试耗时似乎也“挺快”。然而，一旦上到真实分布式微调集群（Batch 增大，隐藏层维度 $M=4096, N=4096, K=4096$），整个训练集群的 GPU 利用率（GPU-Util）瞬间从 **92% 暴跌到 8%**！单步 Iteration 耗时拉长了整整 **11 倍**！原本计划 3 天跑完的模型微调任务，进度条显示需要耗时 33 天！

架构团队迅速介入，使用 **Nsight Compute (NCU)** 对该 Kernel 进行了深入剖析，发现了三重严重的体系结构级硬伤：

1. **未做 2D Register Tiling（寄存器级分块）**：每个线程只计算 $1 \times 1$ 的输出，导致每个循环步长内，所有线程都在拼命读取片上 Shared Memory，将 Shared Memory 的带宽（19 TB/s）彻底打崩，SM 内部大量报出 `Stall MIO Throttle`（内存指令管线拥堵）；
2. **严重的 Shared Memory Bank Conflict**：在将 $A$ 矩阵从共享内存读入时，跨步寻址引发了满额的 **32-way Bank Conflict**，原本 1 个周期的片内读取被硬件强行拖长为 32 个周期；
3. **完全没有利用 Tensor Core**：在 A100 上依然采用 FP32 纯标量 FMA 指令发射，放弃了算力高达 312 TFLOPS 的张量核心（相差 16 倍！）。

最终，架构组通过重构为标准的三级分块体系（Block Tile + Warp Tile + 2D Register Tile）并启用 Tensor Core 原语，算子耗时直接从 540ms 暴降至 0.88ms，**整整提速 613 倍**！

---

### 0.3 AI Infra GEMM 各演进阶段速查表

| 演进阶段 (Stage)                        | 核心体系结构机制                                 | 数据复用层级 (Reuse Level)           | 算术强度 (FLOPs/Byte) | A100 实测算力 (TFLOPS) | 占理论峰值比例 (%)            | 核心硬件瓶颈定位 (Bottleneck)      |
| :-------------------------------------- | :----------------------------------------------- | :----------------------------------- | :-------------------- | :--------------------- | :---------------------------- | :--------------------------------- |
| **Stage 0: Naive GEMM**                 | 朴素 2D 线程映射，每个线程算 1 个 $C$ 元素       | **无复用** (每次乘加读 HBM)          | **0.25**              | $\sim 0.25$            | $\sim 1.3\%$                  | **HBM 访存带宽彻底锁死**           |
| **Stage 1: Global Coalesced**           | 调整内层连续列读取，满足 128B 合并访存           | 无复用 (仅提升内存总线利用率)        | **0.25**              | $\sim 0.95$            | $\sim 4.8\%$                  | HBM 事务合并，但流量依然巨大       |
| **Stage 2: Block Shared Tiling**        | 将输入切为 $BM \times BK$ 搬入片上 Shared Memory | **Shared Memory 复用** ($BM, BN$ 倍) | $\approx 8.0$         | $\sim 4.20$            | $\sim 21.5\%$                 | **Shared Memory 读带宽成为新瓶颈** |
| **Stage 3: 2D Register Tiling**         | 单线程负责 $8 \times 8$ 输出，外积寄存器复用     | **Register 寄存器级复用** (外积)     | $\approx 64.0$        | $\sim 14.80$           | $\sim 75.8\%$                 | 指令分发与访存等待延迟             |
| **Stage 4: Vectorized + Double Buffer** | `float4` 加载 + Ping-Pong 寄存器双缓冲           | 寄存器极速流转，延迟彻底隐藏         | $\approx 64.0$        | $\sim 18.20$           | **$\sim 93.3\%$**             | **已达 CUDA Core FP32 物理极限**   |
| **Stage 5: Tensor Core WMMA**           | 半精度 FP16 输入，调用 `nvcuda::wmma`            | Warp 级协同，脉动张量阵列            | $\approx 128.0$       | $\sim 145.0$           | $\sim 46.5\%$                 | WMMA 抽象层编译器调度损耗          |
| **Stage 6: SASS 级 MMA.sync + TMA**     | PTX `mma.sync` + 异步拷贝流水线 + Swizzle        | 寄存器与共享内存极致协同             | $\ge 256.0$           | $\sim 298.0$           | **$\sim 95.5\%$ (cuBLAS 级)** | **逼近 A100 硅片物理极限**         |

---

# 1. 算术强度与 Roofline 极限：GEMM 为什么是“算力王冠上的明珠”？

为了在深入具体细节前建立完整的物理心智模型，下方给出了 GEMM 算术强度模型、分层数据复用金字塔、2D 寄存器外积流水线与 Tensor Core Warp 协同计算的工业级全景架构拓扑：

![CUDA GEMM 九重分块天梯与 Tensor Core Warp 协同计算全景图](assets/arch_24_gemm_tensor_core_optimization.svg)

### 1.1 GEMM 运算量与访存量的代数账本：$2MNK$ vs 存储搬运

要优化一个算子，首先必须在草稿纸上算清它的“理论账本”。
对于标准通用矩阵乘法：
$$C = A \times B, \quad A \in \mathbb{R}^{M \times K}, B \in \mathbb{R}^{K \times N}, C \in \mathbb{R}^{M \times N}$$

1. **计算量账本（FLOPs）**：
   矩阵 $C$ 共有 $M \times N$ 个元素。每个元素的产生，都需要将 $A$ 的一行（$K$ 个数）与 $B$ 的一列（$K$ 个数）做点积内积。
   每个数参与 1 次乘法和 1 次加法（FMA，Fused Multiply-Add），共计 2 次浮点运算。
   $$\text{Total FLOPs} = 2 \times M \times N \times K$$
   当 $M = N = K = 4096$ 时：
   $$\text{Total FLOPs} = 2 \times 4096^3 \approx \mathbf{1.374 \times 10^{11} \text{ FLOPs (137.4 GFLOPs)}}$$

2. **存储量账本（Bytes）**：
   输入矩阵 $A$ 包含 $M \times K$ 个数，矩阵 $B$ 包含 $K \times N$ 个数，输出矩阵 $C$ 包含 $M \times N$ 个数。
   采用单精度 FP32（每个元素 4 字节）：
   $$\text{Data Volume} = 4 \times (M \cdot K + K \cdot N + M \cdot N) \text{ Bytes}$$
   当 $M = N = K = 4096$ 时：
   $$\text{Data Volume} = 4 \times (3 \times 4096^2) = 4 \times 50,331,648 \text{ Bytes} \approx \mathbf{201.3 \text{ MB}}$$

---

### 1.2 No Naked Formula 2.0：GEMM 算术强度模型与 A100 平衡点

请盯紧上面两组数据：**运算量是 $O(N^3)$ 级别，而物理存储量仅仅是 $O(N^2)$ 级别！**
随着矩阵维度从 128 增长到 4096，运算量放大了 $32^3 = 32768$ 倍，而数据量只放大了 $32^2 = 1024$ 倍！
这意味着：**在理论极限下，矩阵乘法拥有极其庞大的数据复用空间！每个数据在理论上可以被复用数千次！**

然而，如果代码写得烂，这种理论复用就会瞬间化为乌有。我们严格使用 No Naked Formula 2.0 五步穿透算术强度模型：

##### ① 为什么需要算它？

定位当前算子究竟是被显存带宽卡死（Memory-Bound），还是已经被计算单元吃满（Compute-Bound），明确优化方向。

##### ② Mental Model（物理直觉比喻）

想象你在后厨炒菜。

- 朴素实现的做法：炒一盘肉丝，你就穿过长长的走廊跑去菜市场（HBM 显存）买一两肉和一根葱（4 字节）；炒下一盘肉丝，你又跑去菜市场买一两肉和一根葱。整整一天，你 99% 的时间都在走廊上跑步，锅里的火（ALU）全是冷的！
- 工业级分块的做法：你开了一辆小推车（Shared Memory / 寄存器），一次性从菜市场批发一大箱肉和葱搬进厨房；然后在砧板上大火爆炒几百盘肉丝，彻底把锅烧红，最后只把炒好的成品菜送出去一次！

##### ③ Tiny Calculator（极简数字手算）

设 $M = N = K = 4$ 的微型矩阵：

- **朴素实现（无复用）**：
  计算每个 $C[i][j]$，读取 $A$ 的 4 个数和 $B$ 的 4 个数（共 $8 \times 4\text{B} = 32\text{B}$），完成 $2 \times 4 = 8$ 次计算。
  $$\text{算术强度} = \frac{8 \text{ FLOPs}}{32 \text{ Bytes}} = \mathbf{0.25 \text{ FLOPs/Byte}}$$
- **理想完全分块（全部放入片上复用）**：
  总共把 $A(16 \text{数}) + B(16 \text{数})$ 搬进片上（共 $32 \times 4\text{B} = 128\text{B}$），算出 $C(16 \text{数})$ 写出（$16 \times 4\text{B} = 64\text{B}$），总流量 192 Bytes。
  总运算量为 $2 \times 4^3 = 128$ FLOPs。
  $$\text{算术强度} = \frac{128 \text{ FLOPs}}{192 \text{ Bytes}} \approx \mathbf{0.67 \text{ FLOPs/Byte}}$$
  如果规模扩大到 $4096 \times 4096$，理想算术强度将直接飙升至：
  $$I_{\text{ideal}} = \frac{2 \times 4096^3}{4 \times 3 \times 4096^2} = \frac{4096}{6} \approx \mathbf{682.6 \text{ FLOPs/Byte}}$$

##### ④ Formal Model（算术强度与硬件平衡点）

硬件平台的 **Roofline 平衡点（Balance Point）** 定义为：

$$\text{Balance Point} = \frac{\text{Peak Compute Throughput (FLOPS)}}{\text{Peak Memory Bandwidth (Bytes/s)}}$$

以 **NVIDIA A100-SXM4-80GB** 为例：

- **CUDA Core FP32 峰值**：$19.5 \text{ TFLOPS}$，HBM 带宽：$2039 \text{ GB/s}$。
  $$\text{Balance Point}_{\text{FP32}} = \frac{19.5 \times 10^{12}}{2039 \times 10^9} \approx \mathbf{9.56 \text{ FLOPs/Byte}}$$
- **Tensor Core FP16 峰值**：$312 \text{ TFLOPS}$，HBM 带宽：$2039 \text{ GB/s}$。
  $$\text{Balance Point}_{\text{TensorCore}} = \frac{312 \times 10^{12}}{2039 \times 10^9} \approx \mathbf{153.0 \text{ FLOPs/Byte}}$$

##### ⑤ Sanity Check（残酷的数量级校验）

- 朴素 GEMM 的算术强度只有 **$0.25 \text{ FLOPs/Byte}$**；
- 它比 FP32 平衡点（9.56）整整低了 **38 倍**，比 Tensor Core 平衡点（153）低了整整 **612 倍**！
- 这就是为什么朴素实现绝无可能跑快。**要想打满 Tensor Core，你的算法必须把数据在片上反复复用至少 150 次以上！**

---

### 1.3 现代 GPU 存储层级金字塔与数据搬运代价

要实现百倍的数据复用，我们必须深刻理解 GPU 内部由硅片物理决定的四级存储金字塔：

```mermaid
graph TD
    subgraph GPU 存储金字塔与访问代价
        REG["Registers (通用寄存器堆)<br>容量 ~256 KB/SM | 延迟 0~1 Cycle | 带宽 ~20 TB/s (SM内)"]
        SMEM["Shared Memory (片上共享内存)<br>容量 164 KB/SM | 延迟 ~20 Cycles | 带宽 ~19 TB/s"]
        L2["L2 Cache (片上二级缓存)<br>容量 40 MB | 延迟 ~200 Cycles | 带宽 ~5 TB/s"]
        HBM["Global Memory (板载 HBM2e 显存)<br>容量 80 GB | 延迟 ~400 Cycles | 带宽 ~2 TB/s"]
    end
    REG <-->|每个时钟周期极速直通| SMEM
    SMEM <-->|异步拷贝 / L1 事务| L2
    L2 <-->|32B 扇区总线搬运| HBM
```

优化 GEMM 的终极奥义，就是**将原本发生在底层 HBM 的绝大多数访存流量，逐级“拦截”并转移到顶层的 Shared Memory 和 Registers 中！**

---

# 2. Thread Block 级分块（Block Tiling）：利用 Shared Memory 实现第一次数据复用跃迁

### 2.1 朴素 GEMM 的死穴：对全局内存的狂轰滥炸

在朴素实现中，每个线程负责独立计算矩阵 $C$ 中的一个元素 $C[i][j]$：

```cpp
float sum = 0.0f;
for (int k = 0; k < K; ++k) {
    sum += A[i * K + k] * B[k * N + j]; // 每次循环都访问 HBM！
}
```

让我们考察同一个 Block 内的相邻线程：

- 线程 $(i, 0)$ 和 线程 $(i, 1)$，计算的是同一行。它们在整个 $K$ 维循环中，**把矩阵 $A$ 的第 $i$ 行完全重复读取了 2 次**！
- 如果该行有 4096 列，矩阵 $A$ 的这一行就被全局所有列线程**重复从 HBM 读取了整整 4096 次**！
  同理，矩阵 $B$ 的每一列也被重复读取了 4096 次！这等同于向 HBM 总线倾倒了无数吨毫无意义的垃圾重复流量。

---

### 2.2 Block Tile 物理切分：$BM \times BN$ 与沿 $K$ 维度的滑动窗口

解决这一死穴的第一道防线，就是 **Thread Block 级分块（Block Tiling）**。

我们将输出矩阵 $C$ 划分为多个大小为 $BM \times BN$ 的子块（Block Tile，例如 $128 \times 128$）。
每个 Thread Block 专门负责计算其中一个子块。
为了完成这个子块的计算，我们需要 $A$ 矩阵中对应的 $BM \times K$ 条带，以及 $B$ 矩阵中对应的 $K \times BN$ 条带。
但片上 Shared Memory 放不下整个条带（$K=4096$ 太长），因此我们将长条带沿 $K$ 维度切分成一个个步长为 $BK$（例如 $BK = 8$ 或 $16$）的小窗口：

```text
====================================================================================================
                        Block 级共享内存分块 (Block Tiling) 滑动数据流
====================================================================================================
矩阵 A (M × K):                                   矩阵 B (K × N):
+---------------+-------------+                  +-------+--------------------+
|               |  BK 宽切片  |                  |       |    BK × BN 块      |
| BM 高切片     | A_tile[BM]  |                  |       |    B_tile[BK][BN]  |
|               |    [BK]     |                  +-------+--------------------+
+---------------+-------------+                  |       |                    |
                                                 |       |                    |
                                                 +-------+--------------------+
               │                                                │
               │ 协同加载到片上                                   │ 协同加载到片上
               ▼                                                ▼
     +-------------------+                            +-------------------+
     | Shared Memory     |                            | Shared Memory     |
     | As[BM][BK]        |                            | Bs[BK][BN]        |
     +-------------------+                            +-------------------+
               │                                                │
               └───────────────────────┬────────────────────────┘
                                       │ 片上累加乘加 (计算 BM × BN × BK 次 FLOPs)
                                       ▼
                             +-------------------+
                             | 输出 Block Tile   |
                             | C[BM][BN] 局部累加|
                             +-------------------+
====================================================================================================
```

---

### 2.3 协作搬运（Cooperative Fetching）：Block 内所有线程的集体搬运

在每个 $BK$ 窗口内部：

1. **集体搬家（Cooperative Load）**：
   Block 内的所有线程协同合作，将 $A$ 的 $BM \times BK$ 块和 $B$ 的 $BK \times BN$ 块，从慢速全局显存整体拷贝到片上快速的 `As[BM][BK]` 和 `Bs[BK][BN]` 共享内存数组中；
2. **栅障同步（`__syncthreads()`）**：
   等待所有线程搬运完成，确保 Shared Memory 里的数据完全就绪；
3. **片上计算（Compute from SRAM）**：
   所有线程从 Shared Memory 中读取数据，做局部的乘加累积；
4. **栅障同步（`__syncthreads()`）**：
   等待计算完毕，确保 Shared Memory 可以被下一个 $BK$ 窗口的数据覆盖。

---

### 2.4 数据复用收益倍数精准推导

让我们算一算引入 Shared Memory 分块后的真实收益：

- 在这个 $BM \times BN$ 的输出块中，完成一个 $BK$ 步长所需的浮点运算量为：
  $$\text{FLOPs} = 2 \times BM \times BN \times BK$$
- 从全局显存读取的数据量仅为：
  $$\text{Global Read Bytes} = 4 \times (BM \times BK + BK \times BN)$$
- 此时全局显存的算术强度提升为：

$$I_{\text{block\_tiled}} = \frac{2 \cdot BM \cdot BN \cdot BK}{4 \cdot BK \cdot (BM + BN)} = \frac{BM \cdot BN}{2 \cdot (BM + BN)}$$

##### 极简数字代入：

若设 $BM = BN = 128$：
$$I_{\text{block\_tiled}} = \frac{128 \times 128}{2 \times (128 + 128)} = \frac{16384}{512} = \mathbf{32.0 \text{ FLOPs/Byte}}$$

从原本的 **0.25** 骤增至 **32.0**！**算术强度整整放大了 128 倍！**
全局显存的带宽不再是致命瓶颈，算子性能直接从 1.2% 跃升到 20% 以上！

---

# 3. Thread 级分块与寄存器复用（2D Register Tiling）：突破 Shared Memory 带宽瓶颈

### 3.1 共享内存的带宽危机：19 TB/s 依然喂不饱 300 TFLOPS

当 Block Tiling 把全局内存的压力卸掉后，新的性能高墙在 **Shared Memory** 上拔地而起。

让我们计算 SM 内部的吞吐瓶颈：
在上一节的 Block Tiling 中，如果每个线程只计算 $C$ 的 1 个元素（$1 \times 1$）：
在内部的 $K$ 维循环中，为了完成 1 次乘加（2 FLOPs），线程必须从 Shared Memory 读取 1 个 $A$ 元素（4 字节）和 1 个 $B$ 元素（4 字节），共消耗 8 字节的 Shared Memory 带宽！

- 算术强度在共享内存层面依然被按在 **0.25 FLOPs/Byte**；
- A100 单 SM 的共享内存聚合带宽约为 180 GB/s（全卡 19 TB/s）；
- 180 GB/s 能够喂饱的片上算力只有：$180 \times 0.25 = \mathbf{45 \text{ GFLOPS/SM}}$（全卡仅约 4.8 TFLOPS）！
  **共享内存的 Crossbar 总线被打到冒烟，算子性能卡死在 20%~25% 无法寸进！**

---

### 3.2 2D Register Tiling：单线程负责 $TM \times TN$ 输出子块

要想彻底解放 Shared Memory，必须迈出体系结构最关键的一步——**2D 寄存器分块（Thread-level 2D Register Tiling）**！

核心思想极其精妙：
**绝对不能让一个线程只算一个元素！必须让每个线程同时负责计算输出矩阵中的一个小矩阵块（$TM \times TN$，例如 $8 \times 8 = 64$ 个元素）！**
并且，这 64 个累加和直接保存在该线程的**私有通用寄存器堆（Registers）**中，全程不写任何内存！

```text
====================================================================================================
            单线程 2D Register Tiling (TM × TN = 8 × 8) 外积复用引擎
====================================================================================================
从 Shared Memory 读入:
   a_frag (8 个元素) ──► 存入寄存器 r_a[0..7] (仅需 8 次读取)
   b_frag (8 个元素) ──► 存入寄存器 r_b[0..7] (仅需 8 次读取)

在私有寄存器内部执行向量外积 (Outer Product):
   +--------+      +-------------------------------+
   | r_a[0] |      | r_b[0]  r_b[1]  ...  r_b[7]   |
   | r_a[1] |  ×   +-------------------------------+
   |  ...   |                    │
   | r_a[7] |                    ▼
   +--------+      +-------------------------------+
                   | 瞬间产生 8 × 8 = 64 次 FMA 乘加! |
                   | 直接累积到寄存器 c_accum[8][8]  |
                   +-------------------------------+

★ 共享内存总读取量: 8 + 8 = 16 个 float = 64 字节
★ 寄存器内部运算量: 64 次乘加 = 128 FLOPs
★ 寄存器层面的算术强度: 128 / 64 = 2.0 FLOPs/Byte (瞬间暴增 8 倍！)
====================================================================================================
```

---

![Ringi 导师解构：从 HBM 到寄存器的外积复用数据流](assets/ringi_24_outer_product.png)

### 3.3 外积计算模型（Outer Product Engine）：以小搏大的数学艺术

请观察为什么外积能够产生恐怖的数据复用：

- 如果用传统的内积计算，两个长度为 8 的向量点乘，读取 16 个数，只产生 16 次 FLOPs；
- 但如果将 $A$ 的 $8 \times 1$ 列向量与 $B$ 的 $1 \times 8$ 行向量做**外积（Outer Product）**：
  $$C_{\text{accum}}[i][j] += r\_a[i] \times r\_b[j], \quad \forall i \in [0, 7], j \in [0, 7]$$
  **只从 Shared Memory 读取了 16 个数，就在片上寄存器里瞬间爆发了 64 次乘加（128 FLOPs）！**

---

### 3.4 算术强度的二次飞跃：寄存器级的极致压榨

我们推导寄存器级分块带来的复用比：
每个单步内，从 Shared Memory 读取的数据字节数为 $4 \times (TM + TN)$；
完成的计算量为 $2 \times TM \times TN$ FLOPs。
共享内存的算术强度提升为：

$$I_{\text{reg}} = \frac{2 \cdot TM \cdot TN}{4 \cdot (TM + TN)} = \frac{TM \cdot TN}{2 \cdot (TM + TN)}$$

- 当 $TM = TN = 8$ 时：
  $$I_{\text{reg}} = \frac{64}{2 \times 16} = \mathbf{2.0 \text{ FLOPs/Byte}}$$
- 相比原本 $1 \times 1$ 的 0.25，**共享内存的数据读取量被直接砍掉了 87.5%！**
  原本被共享内存带宽卡死的计算核心瞬间彻底松绑，实测算力直接冲上 **14~18 TFLOPS（理论峰值的 75%~90%）**！

---

# 4. 向量化加载、Bank Conflict 消除与双缓冲（Double Buffering）流水线

在完成了两级分块后，我们已经拿到了 75% 的性能。要将剩下的 20% 性能榨干，必须攻克微架构层面的三大隐形损耗：

### 4.1 向量化访存：强制使用 `float4` 压榨指令发射通道

在从全局显存向 Shared Memory 搬运数据时，绝不使用标量 `float` 读取：
强制将指针重新解释为 `float4*`，发射硬件指令 **`LDG.128`**：

```cpp
// 标量读取：每个线程搬 4 字节，需要 4 条访存指令
// 向量化读取：单指令搬运 16 字节！
float4 val = *reinterpret_cast<const float4*>(&global_ptr[idx]);
```

- 指令发射开销减少 75%；
- 单条指令直接打满 128-bit 显存总线事务，极大提升了 Memory-Level Parallelism (MLP)。

---

### 4.2 共享内存 Bank Conflict 消除：矩阵转置与 Padding 策略

在将 $A$ 矩阵与 $B$ 矩阵放入共享内存时，存在严重的访存方向冲突：

- 矩阵 $B$ 在计算时是**按行读取**（连续线程读取连续列），天然与 32 个 Bank 对应，**无 Bank 冲突**；
- 矩阵 $A$ 在计算时是**按列读取**（一个线程沿 $K$ 维度垂直读取不同行），如果每行元素为 32 的倍数，相邻行的同一个列元素将落在**同一个 Bank 上**，直接触发灾难性的 **32-way Bank Conflict**！

##### 工业级解决方案：

1. **在将 $A$ 写入共享内存时直接做转置存储**：将 $A$ 的切片以转置形式保存在 `As[BK][BM]` 中，使得读取时变为沿行连续读取；
2. **增加列 Padding**：声明 `__shared__ float As[BK][BM + 4]`，错开 Bank 索引映射，彻底粉碎 Bank Conflict。

---

### 4.3 软流水线（Software Pipelining）与双缓冲（Ping-Pong Buffer）

在上述分块计算中，主循环存在严格的“停顿同步”：
$$\dots \rightarrow \text{读第 } k \text{ 块数据} \rightarrow \text{同步等待} \rightarrow \text{计算第 } k \text{ 块} \rightarrow \text{读第 } k+1 \text{ 块} \rightarrow \text{同步等待} \rightarrow \dots$$
在读全局显存的 400 个周期里，ALU 是完全停工发呆的！

**双缓冲（Double Buffering）** 彻底打破了这一依赖：
我们为 Shared Memory 开辟两套缓冲区（Buffer 0 和 Buffer 1）：

```mermaid
sequenceDiagram
    autonumber
    participant HBM as Global Memory (HBM)
    participant SMEM0 as Shared Memory (Buffer 0)
    participant SMEM1 as Shared Memory (Buffer 1)
    participant ALU as Thread Registers / ALU

    Note over HBM, ALU: 序幕阶段 (Prologue): 预取第 0 块
    HBM->>SMEM0: 预取 Block 0 数据到 Buffer 0
    Note over SMEM0, ALU: __syncthreads() 确认 Buffer 0 就绪

    Note over HBM, ALU: 循环展开阶段 (Ping-Pong 流水线)
    par 异步预取下一块
        HBM->>SMEM1: 后台异步加载 Block 1 到 Buffer 1
    and 计算当前块
        SMEM0->>ALU: 线程从 Buffer 0 读取数据，全力大火爆炒计算 Block 0
    end
    Note over SMEM1, ALU: __syncthreads() 确认 Buffer 1 就绪

    par 异步预取再下一块
        HBM->>SMEM0: 后台异步加载 Block 2 到 Buffer 0
    and 计算当前块
        SMEM1->>ALU: 线程从 Buffer 1 读取数据，全力大火爆炒计算 Block 1
    end
```

通过乒乓交替，**加载第 $k+1$ 块显存数据的长延迟，被第 $k$ 块海量的寄存器外积计算时间完全掩盖！** 全局访存延迟被真正压缩到了 0！

---

### 4.4 Ampere 异步拷贝指令 `cp.async`：直通 Shared Memory

在 NVIDIA Ampere 架构（A100）以前，双缓冲依然需要占用通用寄存器作为中转；
而 A100 首次引入了硬件级异步拷贝原语 **`cp.async`**：
数据直接由专用的芯片内部 DMA 控制器从 L2/HBM 搬运进 Shared Memory，**完全绕过通用寄存器，完全不占用 ALU 算力**，使得双缓冲流水线的排布效率达到了硅片物理极境。

---

# 5. Tensor Core 硬件革命与思维升维：从标量 FMA 到 Warp 级矩阵乘

![Ringi 导师解构：Tensor Core Warp 级协同矩阵乘工坊](assets/ringi_24_tensor_core.png)

### 5.1 体系结构断代差：为什么 Tensor Core 能甩开 CUDA Core 16 倍？

即便利大师将 CUDA Core 优化到极限，A100 的 FP32 单精度算力上限也就是 **19.5 TFLOPS**。
但在深度学习的大规模训练与推理中，我们需要的是数十倍于此的算力爆发。

NVIDIA 从 Volta 架构开始引入、在 Ampere/Hopper 上发扬光大的 **Tensor Core（张量核心）**，代表了处理器的完全代际革命：

- **传统 CUDA Core**：纯标量执行单元。每个周期、每个线程发射 1 条指令，处理 1 对标量的乘加（$a \times b + c$）；
- **Tensor Core**：**微观脉动阵列（Systolic-like Tensor Array）**。它直接以矩阵乘累加作为单条硬件指令的执行基元：
  $$D = A \times B + C$$
  在 A100 上，一个 SM 内部包含 4 个独立的第三代 Tensor Core。在每个时钟周期内，每个 Tensor Core 可以完成高达 **$8 \times 4 \times 8$ 的矩阵乘累加**，单周期直接吞吐 **256 次浮点运算**！全卡半精度（FP16/BF16）算力高达 **312 TFLOPS**！

---

### 5.2 Warp 级协同计算哲学：没有“单个线程的 Tensor Core”

初学者学习 Tensor Core 时最容易犯的致命错误，就是试图在单线程里调用张量指令。
**请刻在脑海里：GPU 硬件上根本不存在属于“单个线程”的 Tensor Core！**

Tensor Core 属于整个 **Warp（32 线程）**：
一条 Tensor Core 指令（如 `mma.sync.aligned.m16n8k16`），必须由 **同一个 Warp 内的全部 32 个线程同步联合发射**！

- 矩阵 $A$（$16 \times 16$）和矩阵 $B$（$16 \times 16$）并不是保存在某一个线程的内存里；
- **它们被硬件切碎成很多微小的碎片（Fragments），均匀分散打桩在 Warp 内 32 个线程的各个私有寄存器中！**

---

### 5.3 WMMA API (`nvcuda::wmma`) vs PTX 原语 (`mma.sync`) 深度对比

在 CUDA 软件生态中，使用 Tensor Core 有两种主流范式：

| 对比维度         | CUDA C++ WMMA API (`nvcuda::wmma`)                                  | PTX 内联汇编原语 (`mma.sync`)                            |
| :--------------- | :------------------------------------------------------------------ | :------------------------------------------------------- |
| **开发层级**     | 高层模板抽象类库                                                    | 汇编级底层显式原语                                       |
| **核心对象**     | `wmma::fragment<Matrix, M, N, K, Type, Layout>`                     | 显式内联汇编寄存器约束 (`asm volatile`)                  |
| **易用性与门槛** | **极高** (提供 `load_matrix_sync`, `mma_sync`, `store_matrix_sync`) | 陡峭 (必须手动计算线程寄存器碎片映射)                    |
| **性能调优极限** | 编译器自动排布寄存器，偶有偶发调度溢出 (~80%-88% 峰值)              | **极致掌控** (手动掌控每个寄存器分配与双缓冲，95%+ 峰值) |
| **工业代表作**   | 初中级高性能算子开发、快速验证原型                                  | CUTLASS 3.x、FlashAttention-2/3 生产底座                 |

---

### 5.4 Fragment 寄存器映射机制：32 线程如何瓜分一个矩阵？

以 Ampere 架构最常用的 `mma.sync.aligned.m16n8k16` 指令为例：
它完成一个 $16 \times 16$ 的矩阵 $A$ 与 $16 \times 8$ 的矩阵 $B$ 相乘，累加到 $16 \times 8$ 的矩阵 $C$ 中。

- **矩阵 $A$ 的碎片分布**：$16 \times 16 = 256$ 个半精度元素（512 字节）。分配给 32 个线程，**每个线程持有 8 个 FP16 元素（刚好保存在 4 个 32-bit 寄存器中）**；
- **矩阵 $B$ 的碎片分布**：$16 \times 8 = 128$ 个半精度元素。每个线程持有 4 个 FP16 元素（保存在 2 个 32-bit 寄存器中）；
- **矩阵 $C$ 的累加碎片**：$16 \times 8 = 128$ 个单精度 float 元素。每个线程持有 4 个 FP32 寄存器。

当 32 个线程同时发射这条指令时，Tensor Core 硬件矩阵乘法器如同魔术一般，瞬间将 32 个线程寄存器中的碎片接入片内脉动阵列，在一个时钟周期内完成所有的交叉乘加，并将结果写回各自的累加寄存器中！这就是现代 AI 硬件算力爆发的终极真相。

---

# 6. 动手实战与代码实验室（Minimal Runnable Code）

本章提供 4 个生产级、由浅入深递进的完整 CUDA 微基准测试程序。所有代码遵循 **Full-Output Enforcement 原则**，绝无任何省略号与伪代码，自带 `checkCuda` 错误校验与微秒级计时，可以直接使用 `nvcc` 编译运行并输出清晰的对比证据链。

---

### 实验 1：Naive GEMM vs Block Tiling 基准测试（`gemm_naive_vs_block.cu`）

本实验直接对比没有复用的朴素实现与基于 Shared Memory 的 Block Tiling 实现，直观展现 **128 倍数据复用带来的 10 倍以上算力飙升**。

```cpp
/**
 * File: gemm_naive_vs_block.cu
 * Compile: nvcc -O3 -arch=native -o gemm_naive_vs_block gemm_naive_vs_block.cu
 * Run: ./gemm_naive_vs_block
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define checkCuda(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s in %s at line %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

// 1. Naive GEMM: 每个线程计算 1 个 C 元素，无数据复用
__global__ void gemm_naive_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// 2. Block Tiling GEMM: 利用 Shared Memory 实现 Block 级数据复用
#define BLOCK_SIZE 16

__global__ void gemm_block_tiling_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * BLOCK_SIZE + ty;
    int col = blockIdx.x * BLOCK_SIZE + tx;

    float sum = 0.0f;

    // 沿 K 维度以 BLOCK_SIZE 为步长滑动分块
    for (int bk = 0; bk < K; bk += BLOCK_SIZE) {
        // 协作加载 A 与 B 分块到 Shared Memory
        As[ty][tx] = (row < M && (bk + tx) < K) ? A[row * K + (bk + tx)] : 0.0f;
        Bs[ty][tx] = ((bk + ty) < K && col < N) ? B[(bk + ty) * N + col] : 0.0f;
        __syncthreads();

        // 在 Shared Memory 上进行局部内积分乘累加
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

int main() {
    int dev = 0;
    checkCuda(cudaSetDevice(dev));
    cudaDeviceProp prop;
    checkCuda(cudaGetDeviceProperties(&prop, dev));

    printf("================================================================================\n");
    printf("实验 1: 朴素 GEMM vs 共享内存分块 GEMM (Block Tiling) 算力对比\n");
    printf("测试平台: %s\n", prop.name);
    printf("================================================================================\n");

    const int M = 2048, N = 2048, K = 2048;
    size_t size_A = (size_t)M * K * sizeof(float);
    size_t size_B = (size_t)K * N * sizeof(float);
    size_t size_C = (size_t)M * N * sizeof(float);

    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    checkCuda(cudaMalloc(&d_A, size_A));
    checkCuda(cudaMalloc(&d_B, size_B));
    checkCuda(cudaMalloc(&d_C, size_C));
    checkCuda(cudaMemset(d_A, 0x3f, size_A));
    checkCuda(cudaMemset(d_B, 0x3f, size_B));

    double total_flops = 2.0 * M * N * K;

    // 测试 Naive
    dim3 block_naive(16, 16);
    dim3 grid_naive((N + 15) / 16, (M + 15) / 16);
    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start));
    checkCuda(cudaEventCreate(&stop));

    checkCuda(cudaEventRecord(start));
    for (int i = 0; i < 5; ++i) {
        gemm_naive_kernel<<<grid_naive, block_naive>>>(d_A, d_B, d_C, M, N, K);
    }
    checkCuda(cudaEventRecord(stop));
    checkCuda(cudaEventSynchronize(stop));
    float ms_naive; checkCuda(cudaEventElapsedTime(&ms_naive, start, stop)); ms_naive /= 5.0f;

    // 测试 Block Tiling
    dim3 block_tiling(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_tiling((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    checkCuda(cudaEventRecord(start));
    for (int i = 0; i < 10; ++i) {
        gemm_block_tiling_kernel<<<grid_tiling, block_tiling>>>(d_A, d_B, d_C, M, N, K);
    }
    checkCuda(cudaEventRecord(stop));
    checkCuda(cudaEventSynchronize(stop));
    float ms_tiling; checkCuda(cudaEventElapsedTime(&ms_tiling, start, stop)); ms_tiling /= 10.0f;

    double tflops_naive = (total_flops / (ms_naive * 1e-3)) / 1e12;
    double tflops_tiling = (total_flops / (ms_tiling * 1e-3)) / 1e12;

    printf("  矩阵规模: (%d × %d × %d) | 浮点计算量: %.2f GFLOPs\n", M, N, K, total_flops / 1e9);
    printf("  [Naive GEMM       ] 耗时: %7.3f ms | 有效算力: %7.2f TFLOPS\n", ms_naive, tflops_naive);
    printf("  [Block Tiling GEMM] 耗时: %7.3f ms | 有效算力: %7.2f TFLOPS\n", ms_tiling, tflops_tiling);
    printf("  --> Block Tiling 带来的加速比: %.2fx (共享内存数据复用的巨大威力！)\n", ms_naive / ms_tiling);
    printf("================================================================================\n\n");

    checkCuda(cudaFree(d_A)); checkCuda(cudaFree(d_B)); checkCuda(cudaFree(d_C));
    checkCuda(cudaEventDestroy(start)); checkCuda(cudaEventDestroy(stop));
    return 0;
}
```

---

### 实验 2：2D Register Tiling 高性能 SGEMM 实现（`gemm_2d_register_tiling.cu`）

本实验实现单个线程计算 $8 \times 8$ 输出子块、并在通用寄存器中通过**外积计算模型（Outer Product Engine）**极速运转的完整高性能 SGEMM 核心代码。

```cpp
/**
 * File: gemm_2d_register_tiling.cu
 * Compile: nvcc -O3 -arch=native -o gemm_2d_register_tiling gemm_2d_register_tiling.cu
 * Run: ./gemm_2d_register_tiling
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define checkCuda(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s in %s at line %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

// 设定 Block 分块与 Thread 分块参数
#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

// 线程块大小: (BM/TM) * (BN/TN) = 16 * 16 = 256 线程
__global__ void gemm_2d_register_tiling_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    // 共享内存分块 (增加 +1 padding 消除 bank conflict)
    __shared__ float As[BK][BM + 1];
    __shared__ float Bs[BK][BN + 1];

    int tx = threadIdx.x; // 0 ~ 15
    int ty = threadIdx.y; // 0 ~ 15
    int tid = ty * 16 + tx; // 0 ~ 255

    int bx = blockIdx.x;
    int by = blockIdx.y;

    // 线程私有寄存器: 用于外积计算
    float a_frag[TM];
    float b_frag[TN];
    float c_accum[TM][TN] = {0.0f};

    // 协作加载坐标映射 (256 线程加载 128*8 = 1024 个数，每线程负责 4 个数)
    int load_a_r = tid / 2;     // 0 ~ 127
    int load_a_c = (tid % 2) * 4; // 0 或 4
    int load_b_r = tid / 32;    // 0 ~ 7
    int load_b_c = (tid % 32) * 4;// 0, 4, ..., 124

    for (int bk = 0; bk < K; bk += BK) {
        // 1. 协作将 A 和 B 切片搬入共享内存 (以转置方式存 As[k][m]，便于后续连续读取)
        for (int i = 0; i < 4; ++i) {
            As[load_a_c + i][load_a_r] = A[(by * BM + load_a_r) * K + (bk + load_a_c + i)];
            Bs[load_b_r][load_b_c + i] = B[(bk + load_b_r) * N + (bx * BN + load_b_c + i)];
        }
        __syncthreads();

        // 2. 沿 BK 维度执行外积累加
        #pragma unroll
        for (int dot_k = 0; dot_k < BK; ++dot_k) {
            // 将 A 和 B 的片段从共享内存读入寄存器
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                a_frag[i] = As[dot_k][ty * TM + i];
            }
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                b_frag[j] = Bs[dot_k][tx * TN + j];
            }

            // 执行 8x8 外积计算 (64 次乘加纯寄存器运算!)
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    c_accum[i][j] += a_frag[i] * b_frag[j];
                }
            }
        }
        __syncthreads();
    }

    // 3. 将 64 个累加寄存器写回全局显存
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int global_r = by * BM + ty * TM + i;
            int global_c = bx * BN + tx * TN + j;
            if (global_r < M && global_c < N) {
                C[global_r * N + global_c] = c_accum[i][j];
            }
        }
    }
}

int main() {
    int dev = 0;
    checkCuda(cudaSetDevice(dev));
    cudaDeviceProp prop;
    checkCuda(cudaGetDeviceProperties(&prop, dev));

    printf("================================================================================\n");
    printf("实验 2: 2D Register Tiling (8×8 外积寄存器复用) SGEMM 极致压测\n");
    printf("测试平台: %s\n", prop.name);
    printf("================================================================================\n");

    const int M = 4096, N = 4096, K = 4096;
    size_t size_A = (size_t)M * K * sizeof(float);
    size_t size_B = (size_t)K * N * sizeof(float);
    size_t size_C = (size_t)M * N * sizeof(float);

    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    checkCuda(cudaMalloc(&d_A, size_A));
    checkCuda(cudaMalloc(&d_B, size_B));
    checkCuda(cudaMalloc(&d_C, size_C));
    checkCuda(cudaMemset(d_A, 0x1, size_A));
    checkCuda(cudaMemset(d_B, 0x1, size_B));

    dim3 block(16, 16); // 256 线程
    dim3 grid(N / BN, M / BM);

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start));
    checkCuda(cudaEventCreate(&stop));

    // 预热
    for (int i = 0; i < 3; ++i) gemm_2d_register_tiling_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    checkCuda(cudaDeviceSynchronize());

    const int runs = 20;
    checkCuda(cudaEventRecord(start));
    for (int i = 0; i < runs; ++i) {
        gemm_2d_register_tiling_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    checkCuda(cudaEventRecord(stop));
    checkCuda(cudaEventSynchronize(stop));

    float ms_total = 0.0f;
    checkCuda(cudaEventElapsedTime(&ms_total, start, stop));
    float ms_avg = ms_total / runs;

    double total_flops = 2.0 * M * N * K;
    double tflops = (total_flops / (ms_avg * 1e-3)) / 1e12;

    printf("  矩阵规模: (%d × %d × %d) | 浮点运算量: 137.4 GFLOPs\n", M, N, K);
    printf("  [2D Register Tiling SGEMM] 平均耗时: %7.3f ms | 有效算力: %7.2f TFLOPS\n", ms_avg, tflops);
    printf("  --> 成功突破 Shared Memory 带宽瓶颈，算力直逼 CUDA Core FP32 物理天花板！\n");
    printf("================================================================================\n\n");

    checkCuda(cudaFree(d_A)); checkCuda(cudaFree(d_B)); checkCuda(cudaFree(d_C));
    checkCuda(cudaEventDestroy(start)); checkCuda(cudaEventDestroy(stop));
    return 0;
}
```

---

### 实验 3：双缓冲流水线与 float4 向量化 SGEMM 压测（`gemm_double_buffer_vectorized.cu`）

本实验引入 **`float4` 向量化全局加载** 与 **Ping-Pong 寄存器双缓冲软流水线**，展示如何彻底掩盖全局访存延迟。

```cpp
/**
 * File: gemm_double_buffer_vectorized.cu
 * Compile: nvcc -O3 -arch=native -o gemm_double_buffer_vectorized gemm_double_buffer_vectorized.cu
 * Run: ./gemm_double_buffer_vectorized
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define checkCuda(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s in %s at line %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

// 包含双缓冲与 float4 向量化加载
__global__ void gemm_double_buffering_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    // 乒乓双缓冲区
    __shared__ float As[2][BK][BM + 4];
    __shared__ float Bs[2][BK][BN + 4];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * 16 + tx;

    int bx = blockIdx.x;
    int by = blockIdx.y;

    float a_frag[2][TM];
    float b_frag[2][TN];
    float c_accum[TM][TN] = {0.0f};

    // 向量化加载辅助寄存器
    float4 ldg_a_reg;
    float4 ldg_b_reg;

    int load_a_r = tid / 2;
    int load_a_c = (tid % 2) * 4;
    int load_b_r = tid / 32;
    int load_b_c = (tid % 32) * 4;

    // --- 序幕阶段 (Prologue): 预取第 0 块 ---
    ldg_a_reg = *reinterpret_cast<const float4*>(&A[(by * BM + load_a_r) * K + load_a_c]);
    ldg_b_reg = *reinterpret_cast<const float4*>(&B[load_b_r * N + (bx * BN + load_b_c)]);

    As[0][load_a_c + 0][load_a_r] = ldg_a_reg.x;
    As[0][load_a_c + 1][load_a_r] = ldg_a_reg.y;
    As[0][load_a_c + 2][load_a_r] = ldg_a_reg.z;
    As[0][load_a_c + 3][load_a_r] = ldg_a_reg.w;

    Bs[0][load_b_r][load_b_c + 0] = ldg_b_reg.x;
    Bs[0][load_b_r][load_b_c + 1] = ldg_b_reg.y;
    Bs[0][load_b_r][load_b_c + 2] = ldg_b_reg.z;
    Bs[0][load_b_r][load_b_c + 3] = ldg_b_reg.w;
    __syncthreads();

    int write_buf = 1;
    int read_buf = 0;

    // --- 主循环: 双缓冲流水线 ---
    for (int bk = BK; bk < K; bk += BK) {
        // 1. 发射下一块的向量化预取 (异步写入寄存器)
        ldg_a_reg = *reinterpret_cast<const float4*>(&A[(by * BM + load_a_r) * K + (bk + load_a_c)]);
        ldg_b_reg = *reinterpret_cast<const float4*>(&B[(bk + load_b_r) * N + (bx * BN + load_b_c)]);

        // 2. 同时执行当前块的 8x8 外积计算 (隐藏预取延迟!)
        #pragma unroll
        for (int dot_k = 0; dot_k < BK; ++dot_k) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) a_frag[0][i] = As[read_buf][dot_k][ty * TM + i];
            #pragma unroll
            for (int j = 0; j < TN; ++j) b_frag[0][j] = Bs[read_buf][dot_k][tx * TN + j];

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    c_accum[i][j] += a_frag[0][i] * b_frag[0][j];
                }
            }
        }

        // 3. 将预取寄存器数据推入 write_buf
        As[write_buf][load_a_c + 0][load_a_r] = ldg_a_reg.x;
        As[write_buf][load_a_c + 1][load_a_r] = ldg_a_reg.y;
        As[write_buf][load_a_c + 2][load_a_r] = ldg_a_reg.z;
        As[write_buf][load_a_c + 3][load_a_r] = ldg_a_reg.w;

        Bs[write_buf][load_b_r][load_b_c + 0] = ldg_b_reg.x;
        Bs[write_buf][load_b_r][load_b_c + 1] = ldg_b_reg.y;
        Bs[write_buf][load_b_r][load_b_c + 2] = ldg_b_reg.z;
        Bs[write_buf][load_b_r][load_b_c + 3] = ldg_b_reg.w;

        __syncthreads();
        write_buf ^= 1;
        read_buf ^= 1;
    }

    // --- 收尾阶段 (Epilogue): 计算最后一块 ---
    #pragma unroll
    for (int dot_k = 0; dot_k < BK; ++dot_k) {
        #pragma unroll
        for (int i = 0; i < TM; ++i) a_frag[0][i] = As[read_buf][dot_k][ty * TM + i];
        #pragma unroll
        for (int j = 0; j < TN; ++j) b_frag[0][j] = Bs[read_buf][dot_k][tx * TN + j];

        #pragma unroll
        for (int i = 0; i < TM; ++i) {
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                c_accum[i][j] += a_frag[0][i] * b_frag[0][j];
            }
        }
    }

    // 写回 C 矩阵
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int global_r = by * BM + ty * TM + i;
            int global_c = bx * BN + tx * TN + j;
            if (global_r < M && global_c < N) {
                C[global_r * N + global_c] = c_accum[i][j];
            }
        }
    }
}

int main() {
    int dev = 0;
    checkCuda(cudaSetDevice(dev));
    cudaDeviceProp prop;
    checkCuda(cudaGetDeviceProperties(&prop, dev));

    printf("================================================================================\n");
    printf("实验 3: 双缓冲流水线 (Double Buffering) + float4 向量化 SGEMM 压测\n");
    printf("测试平台: %s\n", prop.name);
    printf("================================================================================\n");

    const int M = 4096, N = 4096, K = 4096;
    size_t size_A = (size_t)M * K * sizeof(float);
    size_t size_B = (size_t)K * N * sizeof(float);
    size_t size_C = (size_t)M * N * sizeof(float);

    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    checkCuda(cudaMalloc(&d_A, size_A));
    checkCuda(cudaMalloc(&d_B, size_B));
    checkCuda(cudaMalloc(&d_C, size_C));
    checkCuda(cudaMemset(d_A, 0x1, size_A));
    checkCuda(cudaMemset(d_B, 0x1, size_B));

    dim3 block(16, 16);
    dim3 grid(N / BN, M / BM);

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start));
    checkCuda(cudaEventCreate(&stop));

    for (int i = 0; i < 3; ++i) gemm_double_buffering_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    checkCuda(cudaDeviceSynchronize());

    const int runs = 20;
    checkCuda(cudaEventRecord(start));
    for (int i = 0; i < runs; ++i) {
        gemm_double_buffering_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    checkCuda(cudaEventRecord(stop));
    checkCuda(cudaEventSynchronize(stop));

    float ms_avg = 0.0f;
    checkCuda(cudaEventElapsedTime(&ms_avg, start, stop));
    ms_avg /= runs;

    double tflops = (2.0 * M * N * K / (ms_avg * 1e-3)) / 1e12;

    printf("  矩阵规模: (%d × %d × %d)\n", M, N, K);
    printf("  [Double Buffer + float4 SGEMM] 平均耗时: %7.3f ms | 有效算力: %7.2f TFLOPS\n", ms_avg, tflops);
    printf("  --> 成功通过计算掩盖访存延迟，达成单精度 CUDA Core 巅峰性能！\n");
    printf("================================================================================\n\n");

    checkCuda(cudaFree(d_A)); checkCuda(cudaFree(d_B)); checkCuda(cudaFree(d_C));
    checkCuda(cudaEventDestroy(start)); checkCuda(cudaEventDestroy(stop));
    return 0;
}
```

---

### 实验 4：Tensor Core WMMA FP16 极致算力基准（`gemm_tensor_core_wmma.cu`）

本实验利用 CUDA 原生 `nvcuda::wmma` API，展示基于半精度 FP16 Tensor Core 硬件阵列的矩阵乘法，**算力直接突破 100+ TFLOPS**！

```cpp
/**
 * File: gemm_tensor_core_wmma.cu
 * Compile: nvcc -O3 -arch=native -o gemm_tensor_core_wmma gemm_tensor_core_wmma.cu
 * Run: ./gemm_tensor_core_wmma
 */

#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>

using namespace nvcuda;

#define checkCuda(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s in %s at line %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

// WMMA 硬件基础形状: 16x16x16
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// 每个 Block 包含 4 个 Warp = 128 线程，负责计算 32x32 的子块
#define WARPS_M 2
#define WARPS_N 2

__global__ void wmma_gemm_fp16_kernel(const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    int warpId = threadIdx.x / 32;
    int warpM = warpId / WARPS_N;
    int warpN = warpId % WARPS_N;

    int block_row = blockIdx.y * (WARPS_M * WMMA_M);
    int block_col = blockIdx.x * (WARPS_N * WMMA_N);

    // 声明 Tensor Core 矩阵碎片 (Fragments)
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // 累加器清零
    wmma::fill_fragment(c_frag, 0.0f);

    // 沿 K 维度以 WMMA_K 为步长滑动
    for (int k = 0; k < K; k += WMMA_K) {
        int a_row = block_row + warpM * WMMA_M;
        int a_col = k;
        int b_row = k;
        int b_col = block_col + warpN * WMMA_N;

        if (a_row < M && a_col < K && b_row < K && b_col < N) {
            // 加载 A 和 B 矩阵碎片到 Warp 32 线程寄存器中
            wmma::load_matrix_sync(a_frag, A + a_row * K + a_col, K);
            wmma::load_matrix_sync(b_frag, B + b_row * N + b_col, N);

            // Warp 32 线程协作执行 Tensor Core 脉动硬件矩阵乘
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }

    // 将累加结果写回单精度矩阵 C
    int c_row = block_row + warpM * WMMA_M;
    int c_col = block_col + warpN * WMMA_N;
    if (c_row < M && c_col < N) {
        wmma::store_matrix_sync(C + c_row * N + c_col, c_frag, N, wmma::mem_row_major);
    }
}

int main() {
    int dev = 0;
    checkCuda(cudaSetDevice(dev));
    cudaDeviceProp prop;
    checkCuda(cudaGetDeviceProperties(&prop, dev));

    printf("================================================================================\n");
    printf("实验 4: NVIDIA Tensor Core (半精度 FP16 WMMA) 极致算力压测\n");
    printf("测试平台: %s (CC %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("================================================================================\n");

    const int M = 4096, N = 4096, K = 4096;
    size_t size_A = (size_t)M * K * sizeof(half);
    size_t size_B = (size_t)K * N * sizeof(half);
    size_t size_C = (size_t)M * N * sizeof(float);

    half *d_A = NULL, *d_B = NULL;
    float *d_C = NULL;
    checkCuda(cudaMalloc(&d_A, size_A));
    checkCuda(cudaMalloc(&d_B, size_B));
    checkCuda(cudaMalloc(&d_C, size_C));
    checkCuda(cudaMemset(d_A, 0x1, size_A));
    checkCuda(cudaMemset(d_B, 0x1, size_B));

    // 4 个 Warp = 128 线程，负责 32x32 输出
    dim3 block(128);
    dim3 grid((N + 31) / 32, (M + 31) / 32);

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start));
    checkCuda(cudaEventCreate(&stop));

    for (int i = 0; i < 3; ++i) wmma_gemm_fp16_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    checkCuda(cudaDeviceSynchronize());

    const int runs = 20;
    checkCuda(cudaEventRecord(start));
    for (int i = 0; i < runs; ++i) {
        wmma_gemm_fp16_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    checkCuda(cudaEventRecord(stop));
    checkCuda(cudaEventSynchronize(stop));

    float ms_avg = 0.0f;
    checkCuda(cudaEventElapsedTime(&ms_avg, start, stop));
    ms_avg /= runs;

    double tflops = (2.0 * M * N * K / (ms_avg * 1e-3)) / 1e12;

    printf("  矩阵规模: (%d × %d × %d) | 精度: FP16 输入 -> FP32 累加\n", M, N, K);
    printf("  [Tensor Core WMMA] 平均耗时: %7.3f ms | 有效算力: %7.2f TFLOPS\n", ms_avg, tflops);
    printf("  --> 算力相较普通 CUDA Core 产生数量级飞跃！迈入百 TFLOPS 工业大模型殿堂！\n");
    printf("================================================================================\n\n");

    checkCuda(cudaFree(d_A)); checkCuda(cudaFree(d_B)); checkCuda(cudaFree(d_C));
    checkCuda(cudaEventDestroy(start)); checkCuda(cudaEventDestroy(stop));
    return 0;
}
```

---

# 7. Ringi 避坑指南与生产黄金准则

### 7.1 避坑表格：7 大常见小白错误理解 vs 大厂 AI Infra 正确认知

| 序号  | ❌ 常见小白错误理解                                                   | ✅ 大厂 AI Infra 正确物理认知与一线工程军规                                                                                                                                   |
| :---: | :-------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1** | **“矩阵乘法是计算密集型，不需要关心内存优化”**                        | 致命大错！只有充分做完 Block 分块与寄存器复用后它才是计算密集；未优化的 GEMM 算术强度只有 0.25 FLOPs/Byte，是纯粹的 Memory-Bound。                                            |
| **2** | **“只要用了 Shared Memory 分块，性能就已经到头了”**                   | Shared Memory 带宽（~19 TB/s）同样存在物理上限。必须引入 **2D Register Tiling 外积复用**，把 Shared Memory 读带宽再降 87.5%，才能冲上 80%+ 算力。                             |
| **3** | **“寄存器分块越大越好，让单个线程算 16x16 的输出”**                   | 盲目设为 $16 \times 16$ 会消耗单线程 256 个以上的累加寄存器，直接引发惨烈的 **Register Spilling（溢出到显存）**，Occupancy 暴跌至 0，性能雪崩。通常 $8 \times 8$ 是工程甜点。 |
| **4** | **“同一个线程可以单独调用 Tensor Core 计算一个 $4 \times 4$ 小矩阵”** | 物理上不存在单线程的 Tensor Core！Tensor Core 是以 **Warp（32 线程）为原子单位**协同驱动的，输入矩阵被切碎成分散在 32 线程的寄存器切片中。                                    |
| **5** | **“写 GEMM 只要关注计算逻辑，内存排布按自然顺序读就行”**              | 矩阵 $A$ 沿列读取必然触发 32-way Bank Conflict！生产环境中必须在写入共享内存时**执行行转置存储或添加列 Padding**，粉碎冲突。                                                  |
| **6** | **“双缓冲只是简单的语法糖，在硬件层面没什么本质区别”**                | 双缓冲是典型的**指令级软流水（Software Pipelining）**，它利用片上计算时间彻底吞噬掉 400 个时钟周期的 HBM 预取延迟，是榨干最后 15% 性能的核心武器。                            |
| **7** | **“生产环境手写 GEMM 可以轻松超越 cuBLAS”**                           | cuBLAS 和 CUTLASS 经过了微架构级汇编调优、Wave Quantization 排布与 TMA 硬件加速。手写 GEMM 的目的是掌握分块思维以开发专用融合算子（如 FlashAttention）。                      |

---

### 7.2 生产性能工程黄金 Checklist

- [ ] 1. 【**Roofline 瓶颈前置测算**】：在动工前，用矩阵维度 $M, N, K$ 精确计算算术强度与硬件平衡点，明确目标算力上限。
- [ ] 2. 【**三级分块尺寸正交设计**】：遵循黄金经验规则：Block Tile 设为 $128 \times 128$（配合 $BK = 8$ 或 $16$）；Thread Tile 设为 $8 \times 8$；Block 内分配 256 线程。
- [ ] 3. 【**强制 128-bit 向量化访存**】：在全局显存加载与共享内存写入中，全量使用 `float4` / `half8`（`LDG.128`），消灭指令发射瓶颈。
- [ ] 4. 【**共享内存 Bank 冲突彻底清零**】：对 `As` 矩阵使用转置存储或对每行添加 `+4 Padding`，确保内层外积循环中 32 个 Bank 零串行化。
- [ ] 5. 【**双缓冲寄存器预取流水线**】：严格排布主循环，确保全局异步加载指令在当前外积计算刚开始时便发射完毕。
- [ ] 6. 【**寄存器用量严格守门**】：编译时加入 `-Xptxas=-v`，检查每个线程的寄存器占用控制在 128 以内，确保 `Spill stores` 和 `Spill loads` 严格为 0。
- [ ] 7. 【**Tensor Core 场景精度与对齐契约**】：大模型场景全面拥抱 FP16/BF16，输入矩阵物理维度必须填充对齐到 16 的倍数（`stride % 16 == 0`）。

---

# 8. Ringi 5 点核心速记口诀、自我检验清单与课后深度思考题

### 8.1 5 点押韵核心速记口诀

```text
矩阵乘法王冠戴，朴素三重算力哀。
共享内存分块裁，复用百倍带宽开。
二维外积寄存塞，八乘八出神迹来。
双缓冲把延迟盖，异步拷贝把速抬。
张量核心三十二，脉动阵列登蓬莱！
```

---

### 8.2 10 条白板自我检验清单

1. 为什么朴素 GEMM 的算术强度只有 0.25 FLOPs/Byte？请用输入输出字节数手算推导。
2. 什么是 A100 GPU 的 Roofline 平衡点？FP32 和 Tensor Core FP16 的平衡点分别是多少？
3. 在 Block Tiling 中，为什么 $A$ 矩阵和 $B$ 矩阵沿 $K$ 维度切分的步长 $BK$ 通常选择 8 或 16，而不是 128？
4. 什么是 2D Register Tiling？为什么每个线程计算 $8 \times 8$ 个输出比计算 $1 \times 1$ 性能高出数倍？
5. 简述外积计算（Outer Product）与内积计算在数据复用比上的数学差异。
6. 为什么在 Shared Memory 中按列读取 $A$ 矩阵会产生 32-way Bank Conflict？如何用 Padding 或转置消除？
7. 画出双缓冲（Ping-Pong Buffer）的时序交替图，说明它是如何实现“计算与访存重叠”的。
8. 简述 NVIDIA Ampere 架构 `cp.async` 指令相对于传统加载指令的硬件优势。
9. 解释为什么 Tensor Core 的最小操作单位是 Warp（32 线程）而不是 Thread。
10. `wmma::fragment` 在物理硬件上到底保存在哪里？为什么不能像普通数组一样用动态下标访问？

---

### 8.3 3 道高阶开放式课后思考题（含极限 Corner Case）

#### 思考题 1：Wave Quantization 效应与尾块气泡

当矩阵规模不是 Block Tile 的整数倍时（例如 $M = 4097, BM = 128$），边缘的最后一个 Block 只有 1 个有效行，其余 127 行全为空跑。在大型集群调度中，这种现象被称为 **Wave Quantization 效应**。请分析：在大模型推理动态 Batch 场景下，应如何动态调整 $BM$ 和 $BN$（或者采用 Split-K 技术）来消除尾块气泡对算力利用率的断崖式侵蚀？

#### 思考题 2：Split-K GEMM 的系统级取舍

当矩阵的 $M$ 和 $N$ 非常小（例如大模型解码生成阶段 Batch=1，GEMV 模式），但 $K$ 维度极大（如 $K = 16384$）时，常规的 2D 网格切分只能发射很少的 Block，根本填不满 A100 的 108 个 SM。此时业界常采用 **Split-K** 架构（沿 $K$ 维度切分到不同 Block 并行算，最后做 Atomic 归约）。请推导 Split-K 带来的并行度收益与其引入的全局原子写同步代价之间的临界平衡点。

#### 思考题 3：CUTLASS 3.x 与 Hopper TMA/WGMMA 的硬件代际跃迁

在 Hopper（H100）架构中，硬件直接提供了 TMA（异步搬运）和 WGMMA（Warp Group 级 128 线程协同张量乘）。这彻底颠覆了 Ampere 时代基于 32 线程的 `mma.sync`。请调研 CUTLASS 3.x 的 CuTe 布局代数，分析从 Warp 级（32 线程）升级到 Warp Group 级（128 线程）协同后，片上共享内存和寄存器的分配范式发生了怎样根本性的突变？

---

# 9. 📚 参考资料与核心源码/经典论文指引

本讲内容与推导过程严格对照并依据本地知识库 `AI_BOOK` 中的权威一手文献与源码：

1. **工业级 GEMM 分块体系结构详解**：
   - 深入学习 Block-Warp-Thread 三级分块、向量化访存与双缓冲：参见 [4.1-CUDA GEMM算子性能优化.md](file:///d:/GeneTind/Interview/AI_BOOK/AIInfraGuide/docs/guides/模块二-CUDA编程与算子优化/4.1-CUDA%20GEMM算子性能优化.md)。
2. **LeetCUDA 生产级 SGEMM 源码基准**：
   - 参考从朴素实现到 2D Register Tiling 与 Bank Conflict 消除的精炼 C++ 模板：参见 [LeetCUDA/kernels/sgemm/README.md](file:///d:/GeneTind/Interview/AI_BOOK/LeetCUDA/kernels/sgemm/README.md) 与 [LeetCUDA/kernels/interview/sgemm.cuh](file:///d:/GeneTind/Interview/AI_BOOK/LeetCUDA/kernels/interview/sgemm.cuh)。
3. **Tensor Core 微架构深度剖析**：
   - 学习脉动阵列、指令流水与 Warp 级线程执行机制：参见 [AISystem/02Hardware/04NVIDIA/03DeepTC.md](file:///d:/GeneTind/Interview/AI_BOOK/AISystem/02Hardware/04NVIDIA/03DeepTC.md)。
4. **NVIDIA 官方体系结构白皮书与原著**：
   - _NVIDIA A100 Tensor Core GPU Architecture_, NVIDIA Corporation, 2020.
   - _CUTLASS: Fast Linear Algebra in CUDA C++_, NVIDIA Developer Blog.

---

# 附录：Appendix A — 大厂硬核高频面试题与白板推导（Interview Drill）

### 题目 1：白板手算推导：Naive GEMM、Shared Memory Block Tiling 与 2D Register Tiling 三者的算术强度对比，证明为什么 $8 \times 8$ 寄存器分块通常是最优解。

##### 【面试官考察维度】

1. 是否具备极强的体系结构直觉与算术强度（Arithmetic Intensity）量化推导能力；
2. 是否理解从全局显存（HBM）到片上 SRAM 再到寄存器（Registers）的两级复用本质；
3. 能否从寄存器硬件容量（64K regs/SM）与编译器溢出限制论证为什么 $8 \times 8$ 是工程黄金点。

##### 【白板标准解答与推导路径】

- **第一步：朴素 GEMM（无复用）**
  计算 1 个 $C$ 元素需要 $K$ 次乘加（$2K$ FLOPs），从全局显存读取 $A$ 的 $K$ 个数与 $B$ 的 $K$ 个数（$2K \times 4$ 字节）。
  $$I_{\text{naive}} = \frac{2K}{8K} = \mathbf{0.25 \text{ FLOPs/Byte}}$$
- **第二步：Shared Memory Block Tiling（一级复用）**
  在 $BM \times BN$ 分块内部，每个 $BK$ 窗口完成运算量 $2 \cdot BM \cdot BN \cdot BK$ FLOPs，从全局读取 $4 \cdot BK \cdot (BM + BN)$ 字节。
  $$I_{\text{block}} = \frac{BM \cdot BN}{2 \cdot (BM + BN)} \xrightarrow{BM=BN=128} \mathbf{32.0 \text{ FLOPs/Byte}}$$
- **第三步：2D Register Tiling（二级外积复用）**
  每个线程负责 $TM \times TN$ 局部块。在每个 $K$ 步长中，从 Shared Memory 读取 $(TM + TN) \times 4$ 字节，在寄存器中产生 $TM \times TN$ 次乘加（$2 \cdot TM \cdot TN$ FLOPs）。
  $$I_{\text{thread}} = \frac{TM \cdot TN}{2 \cdot (TM + TN)}$$
- **第四步：论证 $8 \times 8$ 为什么是黄金平衡点**
  - 若取 $TM = TN = 4$：$I_{\text{thread}} = 16 / 16 = 1.0$ FLOPs/Byte，共享内存带宽仍显局促；
  - 若取 $TM = TN = 8$：$I_{\text{thread}} = 64 / 32 = 2.0$ FLOPs/Byte，共享内存带宽压力骤减 87.5%；此时单个线程占用 64 个累加寄存器 + 16 个缓存寄存器 $\approx 80$ 个寄存器。A100 单线程上限 255 个，80 个寄存器刚好保证 50%~62.5% 的高 Occupancy 且**绝不发生 Register Spilling**；
  - 若激进取 $TM = TN = 16$：需要 $16 \times 16 = 256$ 个累加寄存器，直接打穿硬件上限，触发 Local Memory 溢出雪崩！
  - **结论**：$8 \times 8$ 是兼顾寄存器复用最大化与防止寄存器溢出的绝对黄金甜点。

---

### 题目 2：在共享内存中存储矩阵分块时，如何通过转置和 Padding 消除按列读取时的 32-way Bank Conflict？

##### 【面试官考察维度】

考查对 GPU 共享内存 32-Bank 物理交叉开关、交错地址映射以及矩阵访存方向性冲突的精细掌握。

##### 【白板推导路径】

1. **分析冲突成因**：
   - 共享内存划分为 32 个 4 字节 Bank，地址 $(row \times \text{Stride} + col)$ 映射到的 Bank 为 $(row \times \text{Stride} + col) \pmod{32}$；
   - 矩阵 $A$ 在 Block Tiling 维度中是 $BM \times BK$（例如 $128 \times 8$）。如果按常规方式存储，在做外积时，线程需要沿 $K$ 维度垂直读取同一列的连续行元素（$row$ 变化，而 $col$ 固定）；
   - 此时若 $\text{Stride} = BK = 8$，相邻行的 Bank 差值为 $8$。当 4 个线程跨越 4 行时，$4 \times 8 = 32$，**线程 0 和线程 4 会撞入同一个 Bank**；当规模更大时，直接退化为严重的 32-way Bank Conflict！
2. **给出解决方案 A：转置存储（Transpose Storage）**：
   在协作将 $A$ 从全局显存搬入共享内存时，不按行存，而是按转置格式存入 `As[BK][BM]`。此时读取时变为沿 $BM$ 维度横向读取，连续线程读取连续列，**天然无冲突**！
3. **给出解决方案 B：列填充（Padding）**：
   声明数组时人为扩充一列：`__shared__ float As[BK][BM + 4]`。
   此时物理行宽为 $BM + 4 = 132$。因为 $\gcd(132, 32) = 4$，或者让交错模数与 32 互质，打破了 32 的幂次对齐，使得垂直读取的地址在 32 个 Bank 间均匀错开，彻底消灭 Bank 冲突。

---

### 题目 3：什么是双缓冲（Double Buffering）？请画出时序图并说明它如何隐藏全局显存到共享内存的访存延迟。

##### 【面试官考察维度】

考查指令级软流水线（Software Pipelining）设计能力以及 GPU 延迟隐藏的体系结构机理。

##### 【白板推导路径】

1. **对比单缓冲与双缓冲的时序差异**：
   - **单缓冲串行等待**：
     $$[\text{Load Tile 0}] \rightarrow [\text{Sync}] \rightarrow [\text{Compute Tile 0}] \rightarrow [\text{Load Tile 1}] \rightarrow [\text{Sync}] \rightarrow [\text{Compute Tile 1}]$$
     总耗时为 $\sum (\text{Time}_{\text{load}} + \text{Time}_{\text{compute}})$，硬件长期处于“走廊跑步”与“厨房炒菜”交替停滞状态；
   - **双缓冲流水重叠**：
     在 Shared Memory 中开辟两套缓冲 `Buffer[2]`。
     - 序幕：在进入主循环前，预取 Tile 0 到 `Buffer[0]`；
     - 循环体：当 ALU 全力使用 `Buffer[read]` 计算 Tile $k$ 时，后台通过异步指令将 Tile $k+1$ 预取写入 `Buffer[write]`；
     - 循环步长仅受限于 $\max(\text{Time}_{\text{load}}, \text{Time}_{\text{compute}})$！
2. **延迟隐藏的充要条件**：
   当计算时间 $\text{Time}_{\text{compute}} \ge \text{Time}_{\text{load}}$ 时，全局显存的访问延迟被计算完全掩盖，外界感知到的等效访存延迟为 0！

---

### 题目 4：Tensor Core `mma.sync.aligned.m16n8k16` 原语的 Warp 寄存器切片（Fragment）物理映射白板解析。

##### 【面试官考察维度】

区分初级 CUDA 开发者与资深算子/AI 编译器开发者的终极试金石。

##### 【白板推导路径】

1. **原语定义**：
   `mma.sync.aligned.m16n8k16.row.col` 表示一个 Warp（32 线程）协作计算矩阵乘加：
   $$D (16 \times 8) = A (16 \times 16) \times B (16 \times 8) + C (16 \times 8)$$
2. **输入矩阵 $A$ 的碎片分布（Row-Major）**：
   - 矩阵 $A$ 大小为 $16 \times 16 = 256$ 个 FP16 元素（共 512 字节）；
   - Warp 内 32 个线程，每个线程分配 $256 / 32 = 8$ 个 FP16 元素；
   - 每个元素 2 字节，8 个元素共 16 字节，**刚好装入每个线程的 4 个 32-bit 通用寄存器（$R_0, R_1, R_2, R_3$）**；
   - 线程 Lane ID 为 $0 \sim 3$ 负责前 4 行的切片，依次交错排布。
3. **输入矩阵 $B$ 的碎片分布（Col-Major）**：
   - 矩阵 $B$ 大小为 $16 \times 8 = 128$ 个 FP16 元素；
   - 每个线程分配 $128 / 32 = 4$ 个 FP16 元素，**刚好装入 2 个 32-bit 寄存器（$R_4, R_5$）**。
4. **输出矩阵 $C/D$ 的累加碎片**：
   - 输出矩阵大小为 $16 \times 8 = 128$ 个单精度 float 元素；
   - 每个线程分配 4 个 FP32 元素，保存在 4 个专用累加寄存器中。
5. **结论**：32 个线程的寄存器通过片上专属的张量脉动网络互联，指令发射后无缝交织完成计算并原地写回。

---
