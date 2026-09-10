# 🚀 Module 07: Post-Training 与相邻 AI Infra（选修）(Adjacent Workloads & Heterogeneous Computing)

> **讲师 / 作者**：👓 **Ringi**（大厂 AI Infrastructure 工程师）  
> **模块定位**：选修 ｜ **建议时长**：12h  
> **核心目标**：理解 SFT/RLHF、RAG/Agent 基础设施、NPU/DPU 异构硬件对底层 Infra 的系统需求。

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
| **41** | **第41讲：SFT、LoRA 与 RLHF 系统负载特性与算力显存评估** | 从 Infra 视角评估 Rollout、Reference/Reward Model、LoRA 权重合并对显存与通信的冲击。 | [01_SFT_LoRA与RLHF系统负载与资源评估.md](./41_SFT_LoRA与RLHF系统负载与资源评估.md) |
| **42** | **第42讲：RAG 与 Agent 基础设施（向量检索服务 / Context 缓存 / 代码沙箱）** | 分析高并发向量检索、动态长上下文缓存管理与多租户代码执行沙箱环境的 Infra 支撑。 | [02_RAG与Agent基础设施_VectorDB_Context_Sandbox.md](./42_RAG与Agent基础设施_VectorDB_Context_Sandbox.md) |
| **43** | **第43讲：NPU、DPU 与异构加速器硬件抽象架构** | 打破单一 CUDA 绑定思维，理解昇腾/TPU 等非 CUDA 芯片的编译器抽象与 DPU 网络卸载。 | [03_NPU_DPU与异构加速器硬件抽象.md](./43_NPU_DPU与异构加速器硬件抽象.md) |
| **44** | **第44讲：AI Infra 大厂全真面试演练与系统设计真题集** | 从基础定义题升级到定量推导、故障定位与千卡生产级 System Design 白板答辩。 | [04_AI_Infra大厂全真面试演练与系统设计题库.md](./44_AI_Infra大厂全真面试演练与系统设计题库.md) |

---

## 🛠️ 推荐实验与检验标准

- [ ] 完成本模块对应的手算推导与代码验证。
- [ ] 能结合 Profiler（PyTorch Profiler / Nsight / nccl-tests）给出定量性能证据。
- [ ] 能够回答本模块对应的经典大厂面试题与故障诊断题。

---
*返回主目录：[《Ringi AI Infra 学习体系全景地图》](../README.md)*
