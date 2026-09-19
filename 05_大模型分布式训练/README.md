# 🚀 第五篇：大模型大规模分布式训练系统 (Distributed Training Systems)

> **讲师 / 作者**：👓 **Ringi**（大厂 AI Infrastructure 工程师）  
> **模块定位**：必修 ｜ **建议时长**：16h  
> **核心目标**：掌握 DDP、FSDP、ZeRO-1/2/3、TP、PP、SP、CP、MoE 及 3D 混合并行训练体系与故障复盘。

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
| **28** | **第28讲：DDP 分布式数据并行与通信/计算重叠（Bucket Overlap）** | 深入 Process Group、Gradient Sync、Bucket 聚合机制与多卡 Scaling Efficiency 性能瓶颈诊断。 | [01_DDP分布式数据并行与通信计算重叠.md](./28_DDP分布式数据并行与通信计算重叠.md) |
| **29** | **第29讲：显存分片技术——ZeRO-1/2/3 与 PyTorch FSDP 深度剖析** | 解析 Optimizer/Gradient/Parameter 状态分片机制，权衡 Communication 与 Memory Trade-off。 | [02_显存分片技术_ZeRO_1_2_3与PyTorch_FSDP深度剖析.md](./29_显存分片技术_ZeRO_1_2_3与PyTorch_FSDP深度剖析.md) |
| **30** | **第30讲：模型并行技术——TP/PP/SP/CP 架构切分与通信拓扑映射** | 掌握 Column/Row Parallel 矩阵切分、Pipeline Bubble 消除、Sequence/Context 并行设计。 | [03_模型并行技术_TP_PP_SP_CP与硬件拓扑映射.md](./30_模型并行技术_TP_PP_SP_CP与硬件拓扑映射.md) |
| **31** | **第31讲：3D 混合并行 (DP×TP×PP)、MoE 专家并行与训练故障 Debug** | 设计千卡集群混合并行拓扑，解决 AllToAll 通信瓶颈、负载不均、Loss 异常与 NCCL Hang 排查。 | [04_3D混合并行_MoE专家并行与大规模训练故障排查.md](./31_3D混合并行_MoE专家并行与大规模训练故障排查.md) |

---

## 🛠️ 推荐实验与检验标准

- [ ] 完成本模块对应的手算推导与代码验证。
- [ ] 能结合 Profiler（PyTorch Profiler / Nsight / nccl-tests）给出定量性能证据。
- [ ] 能够回答本模块对应的经典大厂面试题与故障诊断题。

---
*返回主目录：[《Ringi AI Infra 学习体系全景地图》](../README.md)*
