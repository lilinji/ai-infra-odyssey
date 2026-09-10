# 🚀 Module 03: LLM 架构、FLOPs 与显存建模 (LLM Architecture & Memory Ledger)

> **讲师 / 作者**：👓 **Ringi**（大厂 AI Infrastructure 工程师）  
> **模块定位**：必修 ｜ **建议时长**：6h  
> **核心目标**：从模型 Config 精确手算参数量、计算量 FLOPs、训练静态显存与动态 KV Cache 显存。

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
| **26** | **第26讲：Decoder-only 大模型结构深度拆解与算子映射** | 深度剖析 MHA/MQA/GQA 演进、RoPE 几何旋转位置编码、RMSNorm 与 SwiGLU 门控前馈网络。 | [01_Decoder_only架构深度拆解_MHA_MQA_GQA_RoPE_SwiGLU.md](./26_Decoder_only架构深度拆解_MHA_MQA_GQA_RoPE_SwiGLU.md) |
| **27** | **第27讲：训练与推理显存账本——参数/梯度/优化器/激活值/KV Cache 精确推导** | 白板手算 7B/70B 显存分布，编写 Memory Estimator 脚本，预测 OOM 临界点与 KV 增长曲线。 | [02_训练与推理显存账本_FLOPs推导与KV_Cache容量规划.md](./27_训练与推理显存账本_FLOPs推导与KV_Cache容量规划.md) |

---

## 🛠️ 推荐实验与检验标准

- [ ] 完成本模块对应的手算推导与代码验证。
- [ ] 能结合 Profiler（PyTorch Profiler / Nsight / nccl-tests）给出定量性能证据。
- [ ] 能够回答本模块对应的经典大厂面试题与故障诊断题。

---
*返回主目录：[《Ringi AI Infra 学习体系全景地图》](../README.md)*
