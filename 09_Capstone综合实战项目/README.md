# 🚀 Module 09: Capstone 综合实战项目库 (Production Capstone Projects & Architecture RFCs)

> **讲师 / 作者**：👓 **Ringi**（大厂 AI Infrastructure 工程师）  
> **模块定位**：综合实战 ｜ **建议时长**：综合考评  
> **核心目标**：包含 5 大面向大厂生产与晋升标准的综合实战项目与架构设计 RFC。

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
| **01** | **项目一：分布式 LLM 训练 Scaling Study 综合实战（DDP → FSDP → 64 GPU 架构推导）** | 基于 TinyStories 完成多卡 Benchmark，绘制 Scaling Efficiency 曲线并归因通信与计算瓶颈。 | [01_Capstone_分布式训练Scaling_Study实战.md](./01_Capstone_分布式训练Scaling_Study实战.md) |
| **02** | **项目二：生产级 LLM Serving 吞吐与延迟极限优化（vLLM + 复杂工作负载）** | A/B 测试不同 Serving 参数，绘制 TTFT vs Throughput 的 Pareto 边界，输出容量规划表。 | [02_Capstone_生产级LLM_Serving吞吐与延迟优化实战.md](./02_Capstone_生产级LLM_Serving吞吐与延迟优化实战.md) |
| **03** | **项目三：CUDA Softmax 与 Attention 算子极致性能调优（从 Naive 到 Tiled/Fused）** | 手写 GPU Kernel，运用 Tiling、Shared Memory 与 Warp 原语，输出 Nsight Compute 性能答辩报告。 | [03_Capstone_CUDA_Attention算子极致性能优化实战.md](./03_Capstone_CUDA_Attention算子极致性能优化实战.md) |
| **04** | **项目四：企业级 AI Platform 统一调度与集群系统架构设计 RFC** | 设计同时支持 SFT、离线 Batch 与在线 P99 服务的千卡 Kubernetes GPU 集群架构方案与故障 Runbook。 | [04_Capstone_企业级AI_Platform统一调度与集群架构RFC.md](./04_Capstone_企业级AI_Platform统一调度与集群架构RFC.md) |
| **05** | **项目五：从单算子到端到端——FlashAttention 全景演进与大模型 Token 生成全链路性能优化实战** | FA-1/2/3 微架构演进 · Online Softmax 递推证明 · Prefill/Decode 冲突治理 · 连续批处理 · 投机采样 · PD 分离。 | [05_Capstone_FlashAttention全景与Token生成端到端全链路优化实战.md](./05_Capstone_FlashAttention全景与Token生成端到端全链路优化实战.md) |

---

## 🛠️ 推荐实验与检验标准

- [ ] 完成本模块对应的手算推导与代码验证。
- [ ] 能结合 Profiler（PyTorch Profiler / Nsight / nccl-tests）给出定量性能证据。
- [ ] 能够回答本模块对应的经典大厂面试题与故障诊断题。

---
*返回主目录：[《Ringi AI Infra 学习体系全景地图》](../README.md)*
