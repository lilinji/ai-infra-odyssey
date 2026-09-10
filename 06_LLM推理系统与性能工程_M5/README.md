# 🚀 Module 05: LLM 在线推理系统与性能工程 (LLM Serving & Inference Optimization)

> **讲师 / 作者**：👓 **Ringi**（大厂 AI Infrastructure 工程师）  
> **模块定位**：必修 ｜ **建议时长**：18h  
> **核心目标**：深入 Prefill/Decode 特性差异、PagedAttention、Continuous Batching、vLLM 架构、PD 分离与容量规划。

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
| **32** | **第32讲：LLM 在线推理两阶段（Prefill vs Decode）与核心 SLO 指标** | 区分 Compute-bound 与 Memory-bound，建立 TTFT、TPOT/ITL、E2E Latency、吞吐量与 P99 评估模型。 | [01_LLM推理两阶段_Prefill_Decode与核心SLO指标.md](./32_LLM推理两阶段_Prefill_Decode与核心SLO指标.md) |
| **33** | **第33讲：KV Cache 内存管理演进——PagedAttention、Continuous Batching 与 Chunked Prefill** | 剖析虚拟内存分页思想在 KV 缓存中的应用，彻底解决显存碎片与 Headroom 浪费问题。 | [02_KV_Cache管理演进_PagedAttention与ContinuousBatching.md](./33_KV_Cache管理演进_PagedAttention与ContinuousBatching.md) |
| **34** | **第34讲：vLLM 系统架构剖析与高阶加速（Prefix Caching / CUDA Graph / 量化 / 投机采样）** | 解析 Scheduler、Block Manager、Attention Backend、INT8/FP8/AWQ 量化与 Speculative Decoding。 | [03_vLLM系统架构剖析与高阶加速技术.md](./34_vLLM系统架构剖析与高阶加速技术.md) |
| **35** | **第35讲：分布式推理与 PD 分离架构（Prefill-Decode Disaggregation）** | 分析 TP/PP/DP 并行 Serving 拓扑，深入 PD 解耦架构设计、KV 跨节点传输与分层缓存体系。 | [04_分布式推理与PD分离架构_Prefill_Decode_Disaggregation.md](./35_分布式推理与PD分离架构_Prefill_Decode_Disaggregation.md) |
| **36** | **第36讲：在线推理容量规划、SLO 治理与生产故障排障实战** | 根据 24 小时真实流量曲线计算 GPU 数量需求，治理长尾延迟（Tail Latency）与突发 Overload。 | [05_在线推理容量规划_SLO治理与生产故障诊断.md](./36_在线推理容量规划_SLO治理与生产故障诊断.md) |

---

## 🛠️ 推荐实验与检验标准

- [ ] 完成本模块对应的手算推导与代码验证。
- [ ] 能结合 Profiler（PyTorch Profiler / Nsight / nccl-tests）给出定量性能证据。
- [ ] 能够回答本模块对应的经典大厂面试题与故障诊断题。

---
*返回主目录：[《Ringi AI Infra 学习体系全景地图》](../README.md)*
