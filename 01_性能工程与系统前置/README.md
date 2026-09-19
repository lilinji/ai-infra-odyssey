# 🚀 第一篇：性能工程与系统前置基石 (Performance Engineering & Systems Baseline)

> **主讲人**：👓 **Ringi**（大厂 AI Infrastructure 工程师）  
> **模块定位**：必修基础 ｜ **建议时长**：10~12h  
> **适用对象**：会 Python/懂一点 PyTorch、希望系统建立 AI Infra 底层硬件、C++、操作系统与性能建模心智模型的初学者。

---

## 🎯 为什么设置系统前置模块？

在深入写 CUDA Kernel 或千卡分布式训练之前，我们必须打通从 **编程语言 (C++/Python) $	o$ 数学与算子 $	o$ 操作系统/CPU体系结构 $	o$ PyTorch内部机制 $	o$ 集合通信 $	o$ 性能工程三账本** 的完整知识链路。

本模块参考了 `caomaolufei/AIInfraGuide` 模块一前置知识体系，并注入了 Ringi 工程师的实战视角，帮助零基础与初学者扫清所有底层认知障碍！

---

## 📑 模块全景章节目录（11 讲系统递进）

| 序号 | 章节名称 | 核心知识点与实战产出 | 文档直达 |
| :--- | :--- | :--- | :--- |
| **01** | **第01讲：AI Infra 工程师的 C++ 底层修炼——从物理内存指针、RAII 显存安全到 PyTorch C++ 扩展实战** | 物理内存模型、RAII 智能指针显存池、std::move 零拷贝、PyBind11 跨语言桥接、ATen 与 JIT/AOT C++ 扩展 | [01_AI_Infra工程师的C++与Python底层必修课.md](./01_AI_Infra工程师的C++与Python底层必修课.md) |
| **02** | **第02讲：AI Infra 数学第一性原理——从张量物理寻址、GEMM 算力账本到自动微分与数值防爆** | 张量物理 Strides、GEMM 分块算术强度、反向 2 次 GEMM 推导、Stable Softmax 数值防爆 | [02_AI_Infra数学基础.md](./02_AI_Infra数学基础.md) |
| **03** | **第03讲：计算机体系结构前置——CPU 存储层次、Cache Line 与 NUMA 架构** | 寄存器/L1/L2/L3/DRAM 延迟阶梯、Cache Line 伪共享、内存对齐、NUMA 跨插槽瓶颈 | [03_计算机体系结构前置_CPU存储层次_CacheLine与NUMA架构.md](./03_计算机体系结构前置_CPU存储层次_CacheLine与NUMA架构.md) |
| **04** | **第04讲：Linux 系统基线与性能分析核心工具箱** | top/vmstat/iostat/sar、进程线程模型、perf 性能分析、CPU 火焰图 FlameGraph | [04_Linux系统基线与性能分析核心工具箱.md](./04_Linux系统基线与性能分析核心工具箱.md) |
| **05** | **第05讲：PyTorch 底层系统机制——Tensor 内存布局、Stride 与 Contiguous** | Tensor Storage、Strides 步长推导、零拷贝转置、View vs Reshape 连续性避坑 | [05_PyTorch底层系统机制_Tensor内存布局_Stride与Contiguous.md](./05_PyTorch底层系统机制_Tensor内存布局_Stride与Contiguous.md) |
| **06** | **第06讲：PyTorch 计算图与 Autograd 显存生命周期** | 动态 DAG 构建、`grad_fn`、Forward/Backward 显存保留与释放、Activation Checkpointing | [06_PyTorch计算图与Autograd显存生命周期.md](./06_PyTorch计算图与Autograd显存生命周期.md) |
| **07** | **第07讲：PyTorch 异步执行流——CUDA Stream 与 CUDA Graph 原理** | CPU-GPU 异步流水线、Stream 并发与 Event 同步、CPU 发射瓶颈与 CUDA Graph 录制 | [07_PyTorch异步执行流_CUDA_Stream与CUDA_Graph原理.md](./07_PyTorch异步执行流_CUDA_Stream与CUDA_Graph原理.md) |
| **08** | **第08讲：Transformer 系统全景与算子执行图深度拆解** | QKV 投影、Softmax、FFN、Norm 物理执行图、算力/访存密集型算子分类、Prefill vs Decode | [08_Transformer系统全景与算子执行图深度拆解.md](./08_Transformer系统全景与算子执行图深度拆解.md) |
| **09** | **第09讲：网络通信基础与集合通信 Collective 原语直觉** | 网络带宽 Gbps 换算、P2P vs Collective、8 大集合通信原语（AllReduce/AllGather 等）图解 | [09_网络通信基础与集合通信Collective原语直觉.md](./09_网络通信基础与集合通信Collective原语直觉.md) |
| **10** | **第10讲：AI Infra 性能工程方法论与三账本（FLOPs/显存/通信）** | Throughput/Latency/MFU 指标、计算/显存/通信三账本手算、Roofline 瓶颈定位决策树 | [10_AI_Infra性能工程方法论与三账本_FLOPs_显存_通信.md](./10_AI_Infra性能工程方法论与三账本_FLOPs_显存_通信.md) |
| **11** | **第11讲：PyTorch Profiler 实战——手把手带你做一次系统性能体检** | `torch.profiler` 深度配置、Chrome Trace / Perfetto 时间线分析、CPU 气泡与算子瓶颈排查 | [11_PyTorch_Profiler实战_手把手带你做一次性能体检.md](./11_PyTorch_Profiler实战_手把手带你做一次性能体检.md) |

---

## 🛠️ 模块学习建议与检验标准

1. **动手跑代码**：每一讲均配有最小可运行验证代码，请在本地 Python / PyTorch 环境中亲自运行并观察输出。
2. **白板手算**：合上代码，能够手算 GEMM 计算量、Tensor Strides 物理偏移量与模型三账本。
3. **完成 Profiling 实验**：掌握通过 `chrome://tracing` 分析性能瓶颈的核心工程方法。

---
*返回主目录：[《Ringi AI Infra 学习体系全景地图》](../README.md)*
