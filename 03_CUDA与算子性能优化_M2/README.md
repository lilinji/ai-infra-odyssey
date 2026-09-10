# 🚀 Module 02: CUDA 编程与高性能算子优化 (CUDA & Operator Performance Engineering)

> **讲师 / 作者**：👓 **Ringi**（大厂 AI Infrastructure 工程师）  
> **模块定位**：必修 ｜ **建议时长**：16h  
> **核心目标**：从线程层次模型到 Softmax/GEMM/FlashAttention 算子手写与极致优化。

---

## 🎯 模块设计理念与工程师视角

在这一模块中，我们将坚持 **Ringi 三问教学协议**：
1. 📐 **Shape 是什么？**（Tensor 形状在每一步算子中到底怎么流转？有哪些 Reshape/Broadcast/Reduction？）
2. 💰 **钱花在哪里？**（估算参数量、FLOPs、显存占用、通信量，明确判断 Compute-bound、Memory-bound 还是 Communication-bound？）
3. ⚙️ **真正在机器上怎么跑？**（在 GPU 的 SM、HBM、SRAM、PCIe/NVLink、NCCL 上对应什么具体指令和 Kernel？）

---

## 📑 本模块章节目录与实战任务

| 序号 | 章节名称 | 核心知识点与实战产出 | 文档链接 |
| :--- | :--- | :--- | :--- |
| **22** | **第22讲：CUDA 编程模型与内存层次（Grid/Block/Thread/Coalescing/Shared Memory）** | 掌握线程索引映射、全局内存合并访问（Memory Coalescing）、Shared Memory 冲突与 Occupancy 调优。 | [01_CUDA编程模型与内存层次_Thread_Block_Coalescing.md](./22_CUDA编程模型与内存层次_Thread_Block_Coalescing.md) |
| **23** | **第23讲：经典算子优化——从 Naive Reduce 到 Fused Online Softmax** | 掌握 Reduction Tree、Warp Shuffle 原语、数值稳定性保护与 Online Softmax 算子融合技术。 | [02_经典算子优化_Reduce与Softmax_OnlineSoftmax_Fused.md](./23_经典算子优化_Reduce与Softmax_OnlineSoftmax_Fused.md) |
| **24** | **第24讲：矩阵乘法核心——GEMM 分块（Tiling）与 Tensor Core 思维** | 深入 Shared Memory Staging、Register Reuse、Arithmetic Intensity 与 cuBLAS 对标调优。 | [03_矩阵乘法核心_GEMM优化与TensorCore思维.md](./24_矩阵乘法核心_GEMM优化与TensorCore思维.md) |
| **25** | **第25讲：注意力算子深度优化——FlashAttention 原理剖析与 Triton 实战** | 揭秘 IO-Aware Attention 分块算法，解析 HBM 与 SRAM 间数据搬运优化，结合 Nsight Compute 剖析 Profiling 证据。 | [04_注意力算子深度优化_TiledAttention_FlashAttention_Triton.md](./25_注意力算子深度优化_TiledAttention_FlashAttention_Triton.md) |

---

## 🛠️ 推荐实验与检验标准

- [ ] 完成本模块对应的手算推导与代码验证。
- [ ] 能结合 Profiler（PyTorch Profiler / Nsight / nccl-tests）给出定量性能证据。
- [ ] 能够回答本模块对应的经典大厂面试题与故障诊断题。

---
*返回主目录：[《Ringi AI Infra 学习体系全景地图》](../README.md)*
