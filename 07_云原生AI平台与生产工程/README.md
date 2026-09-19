# 🚀 第七篇：云原生 AI 平台与生产工程 (Cloud-Native AI Platform & Production Engineering)

> **讲师 / 作者**：👓 **Ringi**（大厂 AI Infrastructure 工程师）  
> **模块定位**：必修 ｜ **建议时长**：10h  
> **核心目标**：从 Kubernetes Device Plugin / DRA 到 GPU 调度、多租户共享、高可用与可观测性。

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
| **37** | **第37讲：GPU 容器化技术与 Kubernetes Device Plugin / DRA 架构** | 解析 NVIDIA Container Toolkit 栈，理解从驱动、Hook 到 Kubelet 资源暴露与动态资源分配（DRA）。 | [01_GPU容器化与Kubernetes_Device_Plugin_DRA.md](./37_GPU容器化与Kubernetes_Device_Plugin_DRA.md) |
| **38** | **第38讲：AI 任务调度与多租户隔离（Gang Scheduling / Kueue / HAMi / GPU 虚拟化）** | 掌握排队优先级、拓扑感知调度、GPU 显存/算力切分共享与训练/推理混部方案。 | [02_AI任务调度与多租户隔离_Gang_Kueue_HAMi_GPUSharing.md](./38_AI任务调度与多租户隔离_Gang_Kueue_HAMi_GPUSharing.md) |
| **39** | **第39讲：生产级 AI Serving 高可用、可观测性与自动扩缩容** | 搭建 Prometheus + Grafana GPU 监控体系，设计 HPA/KEDA 扩缩容策略与背压（Backpressure）机制。 | [03_生产级AI_Serving高可用_可观测与SLO治理.md](./39_生产级AI_Serving高可用_可观测与SLO治理.md) |
| **40** | **第40讲：大模型分布式存储、快速 Checkpoint 与多级缓存体系架构** | 剖析高吞吐权重加载、分布式文件系统（POSIX/S3）、内存级 KV 共享与跨节点缓存优化。 | [04_大模型分布式存储_Checkpoint与缓存体系架构.md](./40_大模型分布式存储_Checkpoint与缓存体系架构.md) |

---

## 🛠️ 推荐实验与检验标准

- [ ] 完成本模块对应的手算推导与代码验证。
- [ ] 能结合 Profiler（PyTorch Profiler / Nsight / nccl-tests）给出定量性能证据。
- [ ] 能够回答本模块对应的经典大厂面试题与故障诊断题。

---
*返回主目录：[《Ringi AI Infra 学习体系全景地图》](../README.md)*
