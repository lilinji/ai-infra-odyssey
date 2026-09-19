---
title: "项目四：万卡熔炉的定海神针——企业级 AI Platform 统一调度与集群系统架构设计 RFC（K8s + Kueue + HAMi + 双轨无损算网）"
description: "直面千卡超算中心的资源撕裂绝境：深拆 Gang 调度死锁规避与 Kueue 弹性配额借用算法，解构 HAMi 动态显存拦截与算力切分，手撕 8x400G 双轨网络与四级存储流水线，输出一套生产级万卡集群统一平台架构设计 RFC 与自愈调度系统。"
date: 2026-09-08T17:45:00+08:00
pubDate: 2026-09-08
category: "ai-platform"
order: 4
tags:
  - AI基础设施
  - AI平台
  - Kubernetes
  - Kueue
  - HAMi
  - 集群调度
  - 双轨网络
  - 架构RFC
  - Capstone实战
  - Ringi-IP
author: Ringi Lee
showToc: true
tocOpen: false
math: true
---

# 🏛️ 项目四：万卡熔炉的定海神针——企业级 AI Platform 统一调度与集群系统架构设计 RFC（K8s + Kueue + HAMi + 双轨无损算网）

> **主讲人**：👓 **Ringi**（大厂 AI Infrastructure 工程师）  
> **所属模块**：[Module 08: Capstone 综合实战项目库](./README.md)  
> **篇章范式**：☁️ 云原生 AI 平台与超算集群架构设计篇（Cloud-Native AI Platform & Supercomputing Architecture RFC Paradigm）  
> **核心导读**：千卡预训练的大粒度 Gang 独占、在线 Serving 的毫秒级 P99 刚性保障、离线评估任务的边角料算力抢占——当几十个业务团队在同一座万卡集群争夺资源时，如何打破资源孤岛与死锁泥潭？本文带你以一线顶级架构师视角，手撕一套支撑万卡超算中枢的完整系统设计 RFC。

![Ringi 导师解构：核心全景工坊——多租户争抢与死锁排队的算力熔炉](assets/ringi_capstone04_overview.png)

```text
=================================================================================================
                                   Ringi 3D 架构工坊 · 核心全景
                   【企业级万卡统一 AI 算力平台全景指挥中枢与控制拓扑】

     [多租户复杂业务流量入口] (大模型预训练、微调 SFT、在线 Serving、离线评估 Batch)
            │
            ▼
    ┌────────────────────────────────────────────────────────────────────────┐
    │ 统一云原生调度控制平面 (Kubernetes Control Plane + Kueue / Volcano)   │
    │   • 多层级弹性配额队列 (LocalQueue / ClusterQueue / Cohort 动态借用)    │
    │   • 严格 Gang Scheduling (全上或全不上，彻底杜绝多机相互等待死锁)     │
    │   • 拓扑感知调度器 (Topology-Aware: TP=8 严守单机 NVLink 物理拓扑)    │
    └────────────────────────────────────────────────────────────────────────┘
            │                                                │
            ▼ (训练专网: 8x400G 计算导轨)                     ▼ (混部与切分: HAMi 显存硬隔离)
    ┌────────────────────────────────────────┐   ┌────────────────────────────────────────┐
    │  大规模预训练集群 (Pretrain / SFT)     │   │  在线推理与离线评测混部集群 (Serving)   │
    │  • 8000 卡纯物理独占, 消除虚拟化损耗   │   │  • 2000 卡 HAMi 算力时间片与显存截断   │
    │  • 无损 RoCEv2 双轨导轨网 (Rail-Opt)   │   │  • 在线请求突发时，秒级优雅抢占离线任务│
    │  • 3FS / NVMe 四级并行 Checkpoint 流水线│  │  • 动态保障 P99 TTFT < 400ms SLO 契约  │
    └────────────────────────────────────────┘   └────────────────────────────────────────┘
            │                                                │
            └───────────────────────┬────────────────────────┘
                                    ▼
    ┌────────────────────────────────────────────────────────────────────────┐
    │ 智能高可用与自愈闭环守护进程 (Auto-Healing Sentinel & Runbook):        │
    │   500ms 周期嗅探 GPU XID 故障 ──> 自动打污点隔离 ──> 注入热备节点 ──>   │
    │   拉取最近一次分布式 DCP 切片 ──> 算力回滚损耗严格控制在 5 分钟以内！   │
    └────────────────────────────────────────────────────────────────────────┘
=================================================================================================
```

<!-- more -->

## 📑 目录导航
- [0. Ringi 现场复盘：价值三亿元的万卡算力死锁事故](#0-ringi-现场复盘价值三亿元的万卡算力死锁事故)
- [1. 第一部分：万卡集群调度四大硬核支柱（Mental Model）](#1-第一部分万卡集群调度四大硬核支柱mental-model)
  - [1.1 传统 Kube-Scheduler 在大模型算力面前的三大死穴](#11-传统-kube-scheduler-在大模型算力面前的三大死穴)
  - [1.2 No Naked Formula 2.0：死锁概率与集群碎片率手算模型](#12-no-naked-formula-20死锁概率与集群碎片率手算模型)
- [2. 第二部分：现代云原生 AI 调度中枢——Kueue 弹性配额借用机制](#2-第二部分现代云原生-ai-调度中枢kueue-弹性配额借用机制)
  - [2.1 LocalQueue、ClusterQueue 与 Cohort 共享池的数学模型](#21-localqueueclusterqueue-与-cohort-共享池的数学模型)
  - [2.2 严格 Gang Scheduling 状态流转时序](#22-严格-gang-scheduling-状态流转时序)
- [3. 第三部分：算力虚拟化与多租户切分——HAMi 核心原理解析](#3-第三部分算力虚拟化与多租户切分hami-核心原理解析)
  - [3.1 为什么 vGPU 昂贵而原生 MPS 缺乏安全隔离？](#31-为什么-vgpu-昂贵而原生-mps-缺乏安全隔离)
  - [3.2 基于动态劫持（CUDA Hook）的显存硬隔离与算力时间片配额](#32-基于动态劫持cuda-hook的显存硬隔离与算力时间片配额)
- [4. 第四部分：双轨算网拓扑与四级存储流水线](#4-第四部分双轨算网拓扑与四级存储流水线)
  - [4.1 Rail-Optimized 导轨网如何节约 40% 的核心交换机成本](#41-rail-optimized-导轨网如何节约-40-的核心交换机成本)
  - [4.2 毫秒级故障感知与 DCP Checkpoint 快速自愈流水线](#42-毫秒级故障感知与-dcp-checkpoint-快速自愈流水线)
- [5. 第五部分：企业级架构设计规范正文——RFC-AI-0045](#5-第五部分企业级架构设计规范正文rfc-ai-0045)
- [6. 第六部分：动手实战代码实验室（100% 完整可运行代码）](#6-第六部分动手实战代码实验室100-完整可运行代码)
  - [实战 1: 多租户弹性队列与 Gang 调度防死锁仿真器](#实战-1-多租户弹性队列与-gang-调度防死锁仿真器)
  - [实战 2: 生产级 GPU 显存拦截与配额超限熔断控制器](#实战-2-生产级-gpu-显存拦截与配额超限熔断控制器)
  - [实战 3: 毫秒级 GPU XID 故障嗅探与热备自愈执行器](#实战-3-毫秒级-gpu-xid-故障嗅探与热备自愈执行器)
- [7. 第七部分：生产落地避坑指南与黄金准则](#7-第七部分生产落地避坑指南与黄金准则)
  - [7.1 企业级 AI Platform 核心避坑矩阵](#71-企业级-ai-platform-核心避坑矩阵)
  - [7.2 万卡超算平台运维工程黄金 Checklist](#72-万卡超算平台运维工程黄金-checklist)
- [8. 第八部分：Ringi 5 点口诀、自我检验清单与课后深度思考题](#8-第八部分ringi-5-点口诀自我检验清单与课后深度思考题)
  - [8.1 5 点押韵核心速记口诀](#8.1-5-点押韵核心速记口诀)
  - [8.2 10 条白板自我检验清单](#82-10-条白板自我检验清单)
  - [8.3 3 道高阶开放式课后思考题](#83-3-道高阶开放式课后思考题)
- [9. 第九部分：知识库与权威论文证据溯源](#9-第九部分知识库与权威论文证据溯源)
- [附录：Appendix A — 大厂架构师级面试真题深度破局](#附录appendix-a--大厂架构师级面试真题深度破局)

---

## 0. Ringi 现场复盘：价值三亿元的万卡算力死锁事故

某国内头部互联网大厂曾为其千亿多模态模型重金组建了一座拥有 8,000 张 H100 GPU 的智算数据中心。然而在平台上线运营的第二周，监控大屏便亮起了全网级红色告警：

- **集群宏观指标**：
  - 全网 8,000 张 GPU 的分配率（Allocation Rate）高达 **97.5%**（几乎所有卡都被 Pod 占据）；
  - 但令人吐血的是，实际运行的 GPU 利用率（GPU Util %）长期徘徊在 **1.2%** 左右；
  - 核心训练任务的整体进度（Progress）**完全停滞达 48 小时之久**。

查看调度器日志，发现了令人啼笑皆非的**经典哲学家就餐死锁（Dining Philosophers Deadlock）**：
- 预训练团队 A 提交了一个需要 4,096 卡（512 台 8 卡节点）的万亿模型任务，K8s 默认调度器（Kube-Scheduler）逐个 Pod 发放资源，为任务 A 成功抢占到了 3,500 张卡，还差 596 张卡才能启动；
- 与此同时，SFT 团队 B 提交了一个需要 2,048 卡的任务，抢占到了 1,500 张卡；微调团队 C 又抢占了 1,000 张卡；
- **死锁形成**：任务 A 在苦苦等待 B 和 C 释放资源，任务 B 在等待 A 释放资源，任务 C 也在等待 A 释放资源！由于谁都凑不齐完整的分布式算力，所有的 PyTorch 进程都在 `torch.distributed.init_process_group` 阶段处于超时死等状态！

```text
三亿元集群陷入全面瘫痪的死锁闭环:
┌──────────────────────────────┐          ┌──────────────────────────────┐
│ 任务 A (需 4096 卡, 已占 3500) │ ──等待──> │ 任务 B (需 2048 卡, 已占 1500) │
└──────────────────────────────┘          └──────────────────────────────┘
               ▲                                         │
               │                  等待                   │
               └─────────────────────────────────────────┘
 ▲ 价值数亿元的 8000 张顶级 GPU，竟全在空转死等握手，每小时直接烧掉近十万元电费！
```

更为雪上加霜的是：
- 运维团队尝试手动杀任务，但由于没有拓扑感知，新拉起的任务被随机打散在不同的机架上，同一个 Tensor Parallel（TP=8）组跨越了 3 台交换机，通信耗时激增 20 倍；
- 生产在线 Serving 服务突遭晚间流量洪峰，向 K8s 申请弹性扩容，但所有节点已被处于死锁挂起状态的训练任务强占，导致在线推理服务 P99 彻底穿底，业务直接大面积瘫痪。

这就是缺乏企业级 AI 平台调度中枢的惨痛代价。大模型集群不是普通的微服务集群，它需要的是具备**大粒度 Gang 协调、跨租户借用、拓扑感知与智能故障自愈**的顶级调度神经中枢。

---

## 1. 第一部分：万卡集群调度四大硬核支柱（Mental Model）

> 💡 **架构全景速览**：在深潜 RFC 设计与代码前，先在白板上建立坚不可摧的企业级 AI Platform 统一调度与集群架构全景底账。
> 
> ![企业级 AI Platform 统一调度与集群架构全景图：Kube-Scheduler 三大死穴、Kueue 弹性借调、严格 Gang 调度与 HAMi 双轨网络 RFC](assets/arch_capstone_04_ai_platform_unified_scheduling_rfc.svg)

### 1.1 传统 Kube-Scheduler 在大模型算力面前的三大死穴

标准的 Kubernetes 是为 Web 无状态微服务设计的。将它生搬硬套到大模型 AI 超算中，会立刻暴露出致命的缺陷：

```mermaid
graph TD
    A["传统 Kube-Scheduler"] --> B["死穴 1: 逐 Pod 独立调度 (缺少 Gang 原语)"]
    A --> C["死穴 2: 扁平拓扑视图 (缺少 NVLink/导轨感知)"]
    A --> D["死穴 3: 静态配额死板 (缺少弹性借用与优雅抢占)"]
    B --> B1["千卡任务部分拿到卡, 湊不齐全集引发全局死锁"]
    C --> C1["同一个 TP 组打散在不同物理机, 算力被通信腰斩"]
    D --> D1["忙碌团队排队上百小时, 闲置团队资源常年空转"]
```

| 调度能力维度 | 传统 Web 容器调度（K8s Default） | 企业级 AI Platform 统一调度 | 架构底层差异 |
| :--- | :--- | :--- | :--- |
| **调度粒度** | **单 Pod 独立判定**（Individual Pod） | **作业级全量原子判定（Gang / All-or-Nothing）** | 作业所需 $N$ 张卡必须一次性全部就绪，否则 1 张卡都不占，杜绝半死锁 |
| **拓扑亲和性** | 弱亲和性（仅感知 Node、Zone） | **微架构级拓扑感知（Topology-Aware Scheduling）** | 识别单机 8 卡 NVLink 环形拓扑、机架内同一 Leaf 交换机导轨（Rail） |
| **配额管理** | 静态 ResourceQuota（死板硬限制） | **多层级弹性借用（Hierarchical Cohort Sharing）** | 闲置配额自动借给兄弟部门，一旦主人发起作业，秒级触发优雅驱逐归还 |
| **混部隔离** | 依靠 cgroups 限制 CPU 和内存 | **HAMi 显存硬拦截 + 时间片动态限速** | 拦截 CUDA Driver API，防止动态显存膨胀导致同机在线服务 OOM |

---

### 1.2 No Naked Formula 2.0：死锁概率与集群碎片率手算模型

我们用数学公式定量证明：**为什么在没有 Gang 调度的大集群中，死锁几乎是 100% 必然发生的物理规律？**

#### 步骤 1：为什么算它？
评估集群死锁风险与碎片损耗，为管理层提供引入云原生高级调度的定量决策依据。

#### 步骤 2：Mental Model（物理直觉比喻）
如果有 8 个人去餐厅吃火锅，必须集齐 8 双筷子才能开吃。服务员每次只随机给每个人发 1 根筷子（逐 Pod 调度）。当桌上坐满了人，每个人手里都只拿着 1 根筷子，谁也不肯放手，所有人都会被活活饿死。

#### 步骤 3：Tiny Calculator（极简数字小算盘）
假设集群总共有 8 张 GPU，现有 2 个团队各自提交了一个需要 4 张卡的分布式作业：
- 调度器随机逐卡分配：
  - 作业 1 拿到 4 张卡、作业 2 拿到 4 张卡的概率（成功）：

$$
P_{\text{success}} = \frac{\binom{8}{4} \text{ 且连续分配}}{\text{所有可能全排列}}
$$

  - 但若作业 1 拿到了 3 张卡，作业 2 抢到了剩余的 5 张卡里的 3 张卡，剩余 2 张卡未分配；
  - 此时作业 1 缺 1 张卡，作业 2 也缺 1 张卡，**两者同时卡死**！
在只有 8 张卡的小系统里，死锁概率就已经高达 35% 以上；而在拥有数千张卡、几十个并发任务的大集群中，死锁概率无限逼近于 **99.99%**！

#### 步骤 4：Formal Model（集群算力碎片率方程）
定义集群不可用碎片率为：

$$
\text{FragRatio} = 1 - \frac{\sum_{i \in \text{RunningJobs}} \text{GPUs}(i)}{N_{\text{total}}}
$$

在缺乏拓扑感知的调度下，单机 8 卡中如果被小任务零散占用了 1 张卡，剩下的 7 张卡就无法被任何需要完整 TP=8 的大模型任务使用，此时单机的有效算力碎片率高达：

$$
\text{NodeFrag} = \frac{7}{8} = 87.5\%
$$

---

## 2. 第二部分：现代云原生 AI 调度中枢——Kueue 弹性配额借用机制

![Ringi 导师解构：Kueue 弹性配额借用与 Gang 全上全不上大转盘](assets/ringi_capstone04_kueue_gang_scheduling.png)

为了消灭死锁并实现资源弹性，云原生社区与大厂广泛采用 **Kueue（Kubernetes-native Job Queueing）** 作为控制面调度大脑。

### 2.1 LocalQueue、ClusterQueue 与 Cohort 共享池的数学模型

Kueue 构建了一套严密的三层队列配额映射模型：

```mermaid
graph TD
    subgraph Users ["业务团队租户层"]
        U1["预训练组 (Team Pretrain)"]
        U2["搜索推荐组 (Team Search)"]
        U3["大模型微调组 (Team SFT)"]
    end

    subgraph LocalQueues ["本地队列 (LocalQueue / 命名空间隔离)"]
        LQ1["LocalQueue: pretrain-queue"]
        LQ2["LocalQueue: search-queue"]
        LQ3["LocalQueue: sft-queue"]
    end

    subgraph ClusterQueues ["全局集群队列 (ClusterQueue / 核心配额控制)"]
        CQ1["ClusterQueue: cq-pretrain<br/>Guaranteed: 4096 GPUs"]
        CQ2["ClusterQueue: cq-search<br/>Guaranteed: 2048 GPUs"]
        CQ3["ClusterQueue: cq-sft<br/>Guaranteed: 1024 GPUs"]
    end

    subgraph CohortPool ["跨部门弹性共享池 (Cohort: 'enterprise-all-gpus')"]
        Pool["全集群总算力资源池 (Total 7168 GPUs)<br/>闲置配额允许跨队列无缝弹性借用 (Borrowing)"]
    end

    U1 --> LQ1 --> CQ1 --> Pool
    U2 --> LQ2 --> CQ2 --> Pool
    U3 --> LQ3 --> CQ3 --> Pool
```

#### 动态借用数学规则（Lending & Borrowing Policy）：
设一个 ClusterQueue 的保底配额为 $Q_{\text{nominal}}$，当前借出上限为 $Q_{\text{borrow-limit}}$：
1. **优先满足保底（Nominal Capacity）**：每个租户在其保底额度内提交的任务，享有**最高调度优先级**，立即可被准入调度；
2. **闲置借用（Borrowing from Cohort）**：若搜索组白天只用了 500 卡，剩余的 1,548 卡自动借给预训练组跑大规模任务；
3. **主人归还与优雅抢占（Reclaim & Preemption）**：当搜索组突然发起高优任务时，Kueue 立即触发 **Workload Reclaim**：
   - 向预训练借用配额的作业发送 `SIGTERM` 信号；
   - 给予 **3 分钟保存 Checkpoint 的宽限期（Grace Period）**；
   - 宽限期结束后优雅驱逐并回收 GPU，还给真正的主人。

---

### 2.2 严格 Gang Scheduling 状态流转时序

在 Kueue 体系中，一个分布式 Job（例如 PyTorchJob）不再被直接创建为活跃 Pod，而是首先包装为 **Workload CRD**。

```mermaid
sequenceDiagram
    autonumber
    participant D as 数据科学家 (kubectl / 平台)
    participant K as Kueue 调度大脑
    participant T as 拓扑感知分配器
    participant P as PyTorch Operator (Pod 生产线)
    participant G as 物理 GPU 节点

    D->>K: 提交 512 节点 (4096 卡) 训练作业
    Note over K: 状态: PENDING (进入排队队列)
    K->>K: 评估 ClusterQueue 配额是否满足?
    
    alt 算力不足 (仅有 3000 卡可用)
        Note over K: 状态: SUSPENDED (挂起)<br/>严禁创建哪怕 1 个 Pod! 0 占用 0 浪费!
    else 算力充足 (满足 4096 卡)
        K->>T: 请求 512 台整机连续物理拓扑
        T->>K: 返回完美避开机架拥塞的节点列表
        Note over K: 状态: ADMITTED (原子准入)
        K->>P: 激活 Workload -> 允许创建 Pod
        P->>G: 512 台机器的 4096 个 Pod 同时瞬间拉起 (All-at-Once)
        G-->>D: 分布式初始化秒级完成，立即开始计算！
    end
```

---

## 3. 第三部分：算力虚拟化与多租户切分——HAMi 核心原理解析

解决了调度排队后，面对在线推理与低算力开发场景，如何将昂贵的高端 GPU（如 A100/H100）安全地切分给多个轻量级容器？

### 3.1 为什么 vGPU 昂贵而原生 MPS 缺乏安全隔离？

在生产实践中，存在三大 GPU 共享技术路线：

| 技术路线 | 实现机制 | 显存隔离能力 | 故障隔离安全性 | 商业授权与开销 | 生产评价 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **NVIDIA vGPU (GRID)** | 硬件级虚拟机切分（Hypervisor） | **硬件绝对物理隔离** | 极高（单虚机崩溃不影响宿主机） | **极其昂贵**（需额外购买高昂 License） | 成本过高，云原生容器场景难以大规模落地 |
| **NVIDIA MPS (多进程服务)** | 共享同一个 CUDA Context 上下文 | **无显存硬限制**（极易互相越界） | **致命**：单个容器 SegFault 会带崩全卡所有租户！ | 免费 | 严禁用于不可信多租户生产环境 |
| **HAMi (异构算力虚拟化)** | **用户态 CUDA Driver 动态符号劫持（Hook）** | **软件级显存硬限制与 OOM 拦截** | 高（单容器 OOM 仅自裁，零影响相邻 Pod） | **开源免费** | **大厂云原生 AI 平台切分与混部的标准答案** |

---

### 3.2 基于动态劫持（CUDA Hook）的显存硬隔离与算力时间片配额

HAMi 的核心实现机制是在容器启动时，通过 `LD_PRELOAD` 强行注入一个自研的高性能动态链接库 `libvgpu.so`：

```text
容器内业务进程 (PyTorch / TensorRT)
            │
            ▼ (调用底层 CUDA API)
    ┌────────────────────────────────────────────────────────┐
    │ HAMi 拦截层 (libvgpu.so - 用户态动态 Hook)             │
    │   • 拦截 cuMemAlloc() / cudaMalloc():                  │
    │     检查: (当前已用显存 + 请求显存) <= Pod 硬配额?     │
    │     - 若超过: 立即返回 CUDA_ERROR_OUT_OF_MEMORY 熔断!  │
    │     - 绝不让多余申请触达物理驱动，保护宿主机邻居安全   │
    │                                                        │
    │   • 拦截 cuLaunchKernel():                             │
    │     基于令牌桶（Token Bucket）算法进行时间片节流:       │
    │     根据 Pod 声明的算力百分比 (如 cores: 30%) 限制发射 │
    └────────────────────────────────────────────────────────┘
            │
            ▼ (通过安全校验的指令)
真实物理驱动 (libcuda.so ➔ NVIDIA Driver ➔ 物理 GPU 芯片)
```

通过这一轻量级劫持，容器完全不需要修改任何一行业务代码，就能实现：
- 单卡同时运行 4 个 20GB 显存的推理微服务；
- 任意一个服务突发 Memory Leak 时，仅仅自身抛出 OOM 退出，其他 3 个微服务纹丝不动！

---

## 4. 第四部分：双轨算网拓扑与四级存储流水线

![Ringi 导师解构：双轨无损网络与四级存储金字塔中枢](assets/ringi_capstone04_network_storage_pyramid.png)

### 4.1 Rail-Optimized 导轨网如何节约 40% 的核心交换机成本

在万卡大规模超算中，网络布线成本占据整机房投资的 20% 以上。

传统的 Fat-Tree 架构假设任意两台机器之间都需要均等的全互联带宽，导致顶层 Spine/Core 交换机数量发生二次方爆炸。

**Rail-Optimized（导轨优化网络）** 的破局点在于充分利用了大模型分布式并行的因果律：
- 在主流 3D 并行中，Tensor Parallel（TP=8）全部在机器内部走 900 GB/s 的 NVLink 消化完毕；
- 跨机流转的通信流量几乎全是 Data Parallel（DP）或 Pipeline Parallel（PP）；
- **因此，将所有物理机的相同卡号（如所有主机的 GPU 0）连入同一个 Leaf 交换机集群，构成独立的导轨 Rail 0；将所有 GPU 1 连入 Rail 1...**

```mermaid
graph TD
    subgraph Server0 ["Node 0 (8x H100)"]
        G0_0["GPU 0"]
        G0_1["GPU 1"]
        G0_7["GPU 7"]
    end

    subgraph Server1 ["Node 1 (8x H100)"]
        G1_0["GPU 0"]
        G1_1["GPU 1"]
        G1_7["GPU 7"]
    end

    subgraph RailSwitches ["独立导轨交换机集群 (Rail-Optimized Spine)"]
        R0["Rail 0 专享网络 (仅 GPU 0 互通)"]
        R1["Rail 1 专享网络 (仅 GPU 1 互通)"]
        R7["Rail 7 专享网络 (仅 GPU 7 互通)"]
    end

    G0_0 --- R0
    G1_0 --- R0
    G0_1 --- R1
    G1_1 --- R1
    G0_7 --- R7
    G1_7 --- R7
```

**工程收益**：
- 集合通信被严格约束在同轨内流转，绝大部分通信根本不需要跨越昂贵的 Core 交换机；
- 全网核心交换机数量削减 **40% 以上**，网络布线清爽规整，网络延迟大幅收敛！

---

### 4.2 毫秒级故障感知与 DCP Checkpoint 快速自愈流水线

万卡集群预训练平均每 12 小时就会发生一次硬件故障（XID 错误、光纤误码、显存 ECC 翻转）。

传统的故障恢复流程：
`故障发生 ➔ Watchdog 30分钟超时 ➔ 任务挂死 ➔ 人工介入 ➔ 重新排队调度 ➔ 从慢速 S3 加载 Checkpoint ➔ 恢复计算`  
**单次故障损失高达 1~2 小时的宝贵算力！**

**企业级自动巡航自愈流水线**：
1. **毫秒级探测 DaemonSet**：节点级守护进程以 500ms 周期直读 GPU NVML 寄存器与网卡 FEC 计数器；
2. **瞬时污点与隔离（Instant Cordon）**：一旦捕获 XID 79（掉卡）或 XID 48（Double-bit ECC 错误），立即给宿主机打上 `gpu-hardware-fault=true:NoSchedule` 污点，防止污染新任务；
3. **热备节点直接置换（Hot-Spare Swap）**：预留 2% 的健康热备节点池（Hot-standby Pool），调度器直接将故障节点下线，将热备节点无缝注入原任务拓扑；
4. **DCP（Distributed Checkpoint）本地闪存恢复**：利用单机 1.5TB NVMe 本地磁盘缓存的最近一个 Step 切片，百秒内完成状态重载。**全链路故障自愈 MTTR 压制在 5 分钟以内！**

---

## 5. 第五部分：企业级架构设计规范正文——RFC-AI-0045

以下为可直接提交企业技术委员会审核的标准系统架构设计提案（Request for Comments）。

```markdown
# [RFC-AI-0045] 企业级万卡统一算力调度与 AI 平台系统设计规范

- **提案状态**：APPROVED (已核准生产执行)
- **目标版本**：Production v2.4
- **主导团队**：AI Platform Core Architecture Team
- **责任架构师**：Ringi Lee

## 1. 业务背景与系统目标 (Context & Goals)
### 1.1 核心目标 (Goals)
1. **超大规模调度**：单集群稳定支撑 10,000 张 GPU 统一编排，预训练平均 MFU 保持在 45% 以上。
2. **死锁彻底清零**：在 100% 满载高竞争场景下，作业级 Gang 调度死锁发生率为 0。
3. **多租户混部与弹性借用**：通过 Kueue 实现跨部门多层级配额弹性借用，全集群综合利用率从 45% 提升至 85% 以上。
4. **高可用 SLA**：硬件单点故障自动检测隔离与热备自愈时间（MTTR）< 5 分钟，单次训练中断算力回滚损耗 < 10 分钟。

### 1.2 非目标 (Non-Goals)
- 本平台不负责具体模型代码内部算法与超参数的自动调优。

## 2. 总体架构拓扑 (System Topology)
系统划分为三大物理/逻辑平面：
- **控制与调度中枢**：Kube-Apiserver + Kueue Controller + Topology-Aware Scheduler Plugin；
- **计算网络数据面**：双轨分离架构。计算专网采用 8x400G 无损 RoCEv2（Rail-Optimized），存储/管理采用双 100G 独立上行；
- **四级存储金字塔**：HBM3 (GPU) ➔ 1.5TB 本地 NVMe (Scratch Cache) ➔ 3FS 并行分布式文件系统 ➔ 对象存储 (S3 冷备)。

## 3. 核心子系统详细规范
### 3.1 调度准入准则 (Admission Policy)
- 任何卡数规模 $\ge 8$ 的作业必须配置 `queue.x-k8s.io/gang-scheduling: strict`；
- 同一个 Tensor Parallel（TP）通信组必须调度在同一个物理 Node（基于 Topology Hints 匹配）；
- 跨机作业必须调度在同一个 Leaf 交换机网络导轨组内，严禁跨 Spine 交换机产生不规则链路跳跃。

### 3.2 多租户隔离与 HAMi 切分规则
- 预训练生产任务一律分配完整物理卡（`nvidia.com/gpu: 8`），严禁任何虚拟化拦截；
- 在线推理与开发调试任务统一采用 HAMi 切分，通过资源声明：
  - `hami.io/gpu-mem: 24000` (硬限 24GB 显存)
  - `hami.io/gpu-cores: 50` (硬限 50% 核心时间片)

## 4. 容灾与故障自愈规程 (Fault-Tolerance Runbook)
- 节点 Daemon 捕获 Xid 错误代码 [31, 43, 45, 48, 62, 79, 92] 时，立即触发 P0 级自愈：
  1. 调用 Kubernetes API 将本节点打上污点并踢出调度池；
  2. 调度器从 Hot-Spare 备用池挑选拓扑对齐的节点加入作业；
  3. 作业 Controller 重载最新 Checkpoint 切片并恢复计算；
  4. 向 On-Call 工程师钉钉/飞书群发送硬件返修工单。
```

---

## 6. 第六部分：动手实战代码实验室（100% 完整可运行代码）

本节交付 3 套工业级生产仿真脚本。代码零省略、零伪代码，在本地即可验证调度、显存拦截与高可用自愈逻辑。

### 实战 1: 多租户弹性队列与 Gang 调度防死锁仿真器

本脚本模拟当多个大作业并发争抢集群资源时，传统逐 Pod 调度（引发死锁）与 Kueue 严格 Gang 调度（零死锁、高效借用）的直观对比。

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: kueue_cluster_simulator.py
Description: 云原生 AI 调度中枢：Gang Scheduling 防死锁与多租户配额借用仿真器
Author: Ringi (AI Infra Architect)
Standards: Full-Output Enforcement, Zero-Omission, Production-Ready
"""

from typing import List, Dict, Any

class Job:
    def __init__(self, name: str, required_gpus: int, tenant: str, duration_steps: int):
        self.name = name
        self.required_gpus = required_gpus
        self.tenant = tenant
        self.duration_steps = duration_steps
        self.allocated_gpus = 0
        self.status = "PENDING" # PENDING, RUNNING, COMPLETED

class NaiveScheduler:
    """传统逐 Pod 贪心分配调度器 (极易引发死锁)"""
    def __init__(self, total_gpus: int = 16):
        self.total_gpus = total_gpus
        self.available_gpus = total_gpus

    def step(self, jobs: List[Job]) -> str:
        for job in jobs:
            if job.status == "PENDING":
                # 贪心分配：有多少给多少
                can_allocate = min(self.available_gpus, job.required_gpus - job.allocated_gpus)
                if can_allocate > 0:
                    job.allocated_gpus += can_allocate
                    self.available_gpus -= can_allocate
                    if job.allocated_gpus == job.required_gpus:
                        job.status = "RUNNING"
        
        # 检查死锁: 资源用光但没有任何作业处于 RUNNING
        if self.available_gpus == 0 and not any(j.status == "RUNNING" for j in jobs if j.status != "COMPLETED"):
            return "🚨 全局死锁爆发！所有可用 GPU 被瓜分，但没有一个作业凑齐完整算力！"
        return "正常运转中"


class GangKueueScheduler:
    """严格 Gang 调度与原子准入控制调度器 (彻底杜绝死锁)"""
    def __init__(self, total_gpus: int = 16):
        self.total_gpus = total_gpus
        self.available_gpus = total_gpus

    def step(self, jobs: List[Job]):
        for job in jobs:
            if job.status == "PENDING":
                # All-or-Nothing 原子判定
                if self.available_gpus >= job.required_gpus:
                    # 一次性完整分配
                    job.allocated_gpus = job.required_gpus
                    self.available_gpus -= job.required_gpus
                    job.status = "RUNNING"
                else:
                    # 凑不齐则 1 张卡都不占，继续挂起排队
                    job.allocated_gpus = 0


def run_scheduler_demo():
    print("=" * 80)
    print(">> 实战 1：集群调度防死锁对决：传统逐卡调度 vs Kueue 严格 Gang 调度")
    print("=" * 80)

    # 场景：集群共有 16 张 GPU，两个大作业 A(需12卡) 和 B(需8卡) 同时提交争抢
    print("\n【场景 A: 传统逐卡调度 (Naive Kube-Scheduler)】:")
    naive_jobs = [Job("Pretrain-A", required_gpus=12, tenant="LLM", duration_steps=5),
                  Job("FineTune-B", required_gpus=8, tenant="Search", duration_steps=3)]
    
    naive_sched = NaiveScheduler(total_gpus=16)
    result_msg = naive_sched.step(naive_jobs)
    
    for j in naive_jobs:
        print(f"  * 作业 {j.name} (需求 {j.required_gpus} 卡) ──> 已抢占: {j.allocated_gpus} 卡 | 状态: {j.status}")
    print(f"  -> 调度诊断: {result_msg}")

    print("\n【场景 B: Kueue 严格 Gang 调度 (All-or-Nothing)】:")
    gang_jobs = [Job("Pretrain-A", required_gpus=12, tenant="LLM", duration_steps=5),
                 Job("FineTune-B", required_gpus=8, tenant="Search", duration_steps=3)]
    
    gang_sched = GangKueueScheduler(total_gpus=16)
    gang_sched.step(gang_jobs)

    for j in gang_jobs:
        print(f"  * 作业 {j.name} (需求 {j.required_gpus} 卡) ──> 已分配: {j.allocated_gpus} 卡 | 状态: {j.status}")
    print(f"  -> 调度诊断: ✅ 彻底避免半死锁！作业 A 完整拿到 12 卡全速计算，作业 B 优雅排队等待！")
    print("=" * 80)


if __name__ == "__main__":
    run_scheduler_demo()
```

---

### 实战 2: 生产级 GPU 显存拦截与配额超限熔断控制器

本脚本模拟 HAMi 用户态 Hook 拦截显存申请的核心逻辑：在单卡 80GB 上虚拟化多租户容器，动态阻止恶意或超额显存申请。

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: hami_memory_interceptor.py
Description: HAMi GPU 显存虚拟化与动态 Hook 拦截熔断仿真器
Author: Ringi (AI Infra Architect)
Standards: Full-Output Enforcement, Zero-Omission, Production-Ready
"""

from typing import Dict, Tuple

class HAMiMemoryController:
    """
    HAMi 容器显存配额硬拦截控制器
    """
    def __init__(self, physical_gpu_mem_mb: float = 81920.0):
        self.physical_gpu_mem_mb = physical_gpu_mem_mb
        self.used_physical_mb = 0.0
        # 记录每个容器的配额与当前用量 {container_id: {"limit": mb, "used": mb}}
        self.containers: Dict[str, Dict[str, float]] = {}

    def register_container(self, container_id: str, limit_mb: float):
        """容器初始化注册配额"""
        self.containers[container_id] = {"limit": limit_mb, "used": 0.0}

    def hook_cu_mem_alloc(self, container_id: str, request_bytes: int) -> Tuple[bool, str]:
        """
        拦截业务层的 cuMemAlloc() 调用
        """
        request_mb = request_bytes / (1024 * 1024)
        c_info = self.containers.get(container_id)
        if not c_info:
            return False, "未知容器上下文！"

        # 1. 容器级别硬配额检查 (Cgroup-like)
        if c_info["used"] + request_mb > c_info["limit"]:
            return False, f"触发 HAMi 显存配额硬熔断！当前已用 {c_info['used']:.1f}MB + 申请 {request_mb:.1f}MB > 限制 {c_info['limit']:.1f}MB"

        # 2. 物理显存余量检查 (保护宿主机)
        if self.used_physical_mb + request_mb > self.physical_gpu_mem_mb:
            return False, f"物理 GPU 显存耗尽 (OOM)！"

        # 通过安全校验，放行底层申请
        c_info["used"] += request_mb
        self.used_physical_mb += request_mb
        return True, f"成功分配 {request_mb:.1f}MB | 容器当前占用: {c_info['used']:.1f}MB"


def run_hami_demo():
    print("=" * 80)
    print(">> 实战 2：HAMi 显存虚拟化：多租户动态 Hook 拦截与安全隔离实测")
    print("=" * 80)

    # 模拟在一张 A100 (80GB) 上混部两个在线推理 Pod
    controller = HAMiMemoryController(physical_gpu_mem_mb=81920.0)
    controller.register_container("Serving-Pod-1", limit_mb=24576.0) # 24GB 配额
    controller.register_container("Serving-Pod-2", limit_mb=24576.0) # 24GB 配额

    print("[阶段 1: 正常加载模型权重 (各申请 14GB)]")
    ok1, msg1 = controller.hook_cu_mem_alloc("Serving-Pod-1", 14 * 1024 * 1024 * 1024)
    ok2, msg2 = controller.hook_cu_mem_alloc("Serving-Pod-2", 14 * 1024 * 1024 * 1024)
    print(f"  - Serving-Pod-1: {'✅' if ok1 else '❌'} {msg1}")
    print(f"  - Serving-Pod-2: {'✅' if ok2 else '❌'} {msg2}")

    print("\n[阶段 2: Pod-1 突发内存泄露，试图恶意申请超额 15GB 显存]")
    ok_leak, msg_leak = controller.hook_cu_mem_alloc("Serving-Pod-1", 15 * 1024 * 1024 * 1024)
    print(f"  - Serving-Pod-1 越界申请: {'✅' if ok_leak else '🚨 成功拦截'} {msg_leak}")

    print("\n[阶段 3: 检查被拦截后相邻 Pod-2 的安全状态]")
    ok_safe, msg_safe = controller.hook_cu_mem_alloc("Serving-Pod-2", 2 * 1024 * 1024 * 1024)
    print(f"  - Serving-Pod-2 正常业务请求: {'✅ 安全放行' if ok_safe else '❌'} {msg_safe}")
    print("=" * 80)


if __name__ == "__main__":
    run_hami_demo()
```

---

### 实战 3: 毫秒级 GPU XID 故障嗅探与热备自愈执行器

本脚本模拟生产级健康巡检守护进程（DaemonSet），毫秒级捕获硬件 Xid 错误，自动对故障节点执行 Cordon 隔离，并调度热备主机注入自愈。

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: cluster_auto_healing_daemon.py
Description: 生产级万卡集群 XID 故障嗅探与热备自愈控制器
Author: Ringi (AI Infra Architect)
Standards: Full-Output Enforcement, Zero-Omission, Production-Ready
"""

from typing import List, Dict, Any

class ClusterAutoHealingDaemon:
    """
    万卡集群高可用故障自愈调度引擎
    """
    def __init__(self):
        # 记录集群节点状态
        self.active_nodes = ["node-01", "node-02", "node-03", "node-04"]
        self.hot_spare_nodes = ["spare-node-01", "spare-node-02"]
        self.tainted_nodes: List[str] = []
        
        # 致命硬件 Xid 错误代码黑名单
        self.critical_xid_codes = {
            31: "GPU 内存页面错误 (Memory Page Fault)",
            48: "不可纠正的 Double-bit ECC 校验翻转",
            79: "GPU 掉卡 (Fallen off the Bus)",
            92: "高频过热硬件降频告警 (Thermal Throttle)"
        }

    def inspect_node_telemetry(self, node_id: str, reported_xid: int) -> Dict[str, Any]:
        """
        嗅探节点健康上报
        """
        if reported_xid in self.critical_xid_codes:
            diag_reason = self.critical_xid_codes[reported_xid]
            # 1. 触发节点驱逐与隔离 (Cordon & Taint)
            self.tainted_nodes.append(node_id)
            if node_id in self.active_nodes:
                self.active_nodes.remove(node_id)

            # 2. 从热备节点池唤醒新节点置换
            if self.hot_spare_nodes:
                new_spare = self.hot_spare_nodes.pop(0)
                self.active_nodes.append(new_spare)
                action_plan = f"隔离故障机 [{node_id}] 并打上 NoSchedule 污点，热调配 [{new_spare}] 替换入网并重载 Checkpoint！"
            else:
                action_plan = f"隔离故障机 [{node_id}]，但热备池枯竭，作业被迫进入挂起等待！"

            return {
                "healthy": False,
                "node_id": node_id,
                "xid": reported_xid,
                "reason": diag_reason,
                "action": action_plan
            }
        return {"healthy": True, "node_id": node_id, "reason": "节点运行健康"}


def run_healing_demo():
    print("=" * 80)
    print(">> 实战 3：万卡超算平台毫秒级 XID 故障嗅探与热备自愈恢复测试")
    print("=" * 80)

    daemon = ClusterAutoHealingDaemon()
    print(f"初始活跃节点: {daemon.active_nodes} | 热备池节点: {daemon.hot_spare_nodes}")

    # 模拟 node-02 突发 XID 79 (掉卡)
    print("\n[突发事件]: node-02 发生 PCIe 链路重置，抛出 XID 79 致命硬件错误！")
    event = daemon.inspect_node_telemetry(node_id="node-02", reported_xid=79)

    print(f"  - 诊断结论: [🚨 致命告警] {event['reason']} (Xid={event['xid']})")
    print(f"  - 平台自愈行动: {event['action']}")
    print(f"\n自愈后活跃节点: {daemon.active_nodes} (成功踢出故障卡，注入 spare-node-01！)")
    print(f"被隔离污点节点: {daemon.tainted_nodes}")
    print("=" * 80)


if __name__ == "__main__":
    run_healing_demo()
```

---

## 7. 第七部分：生产落地避坑指南与黄金准则

结合国内外顶尖 AI 超算平台架构与运维事故复盘，提炼出如下避坑矩阵与 Checklist。

### 7.1 企业级 AI Platform 核心避坑矩阵

| 陷阱分类 | 典型错误做法与直觉认知 | 生产灾难与恶果 | 正确架构级处理方案 |
| :--- | :--- | :--- | :--- |
| **原生 K8s 裸跑** | 直接用原生 Deployment/Job 调度大模型多机多卡任务 | 缺乏 Gang 机制，导致部分 Pod 抢到卡，全集群长期陷入死锁停滞 | 必须引入 Kueue 或 Volcano，强制配置全上或全不上的原子准入 |
| **忽视物理拓扑** | 调度器盲目随机在同一机架内挑卡，把同一个 TP 组打散 | TP 通信被迫经过慢速 PCIe 交换或跨机网络，训练速度暴跌 90% | 部署 Topology-Aware Scheduler，保证 TP=8 严守单机 NVLink 物理域 |
| **共享未加硬限** | 认为设置了 PyTorch 显存上限就够了，多个 Pod 裸跑 MPS | 动态中间变量突发溢出，导致整张卡全部 Pod 连锁 OOM 崩溃 | 使用 HAMi 等内核级 Hook 工具，在用户态实施显存配额硬拦截 |
| **缺乏热备节点** | 机房 100% 满配分配给业务，不预留任何热备节点池 | 哪怕只坏了一张卡，整个千卡作业必须停机数小时等待人工换卡维修 | 常态化预留 2% 的热备节点池，配合 Daemon 实现百秒级故障自愈 |

---

### 7.2 万卡超算平台运维工程黄金 Checklist

- [ ] 1. **【Gang 调度核验】** 检查所有多机分布式任务，确认已关联带有 `All-or-Nothing` 属性的 Workload 准入控制。
- [ ] 2. **【拓扑标签对准】** 宿主机打上清晰的 NVLink、NUMA 拓扑标签，确保调度器能够精准匹配单机 8 卡物理边界。
- [ ] 3. **【Kueue 配额借用】** 为各部门配置 LocalQueue，并在 ClusterQueue 层级设定合理的 Borrowing Limit 与抢占宽限期（通常 3 分钟）。
- [ ] 4. **【HAMi 驱动注入】** 针对共享推理节点，验证容器内 `LD_PRELOAD` 是否成功挂载 `libvgpu.so`，并执行越界内存熔断测试。
- [ ] 5. **【双轨网络隔离】** 物理核验计算专网（8x400G RoCEv2）与存储网络（双 100G）是否处于不同的交换机平面。
- [ ] 6. **【PFC/ECN 阈值】** 在所有计算导轨交换机上配置严格的 DSCP 优先级映射与 PFC 队列流控，严防丢包重传风暴。
- [ ] 7. **【四级存储缓存】** 检查训练节点本地 NVMe SSD 是否挂载为分布式 Checkpoint 的异步暂存盘，避免直写 S3 阻塞主训练流。
- [ ] 8. **【XID 秒级探针】** 部署 GPU 健康探测 DaemonSet，对 Xid [31, 48, 79, 92] 等硬件致命错误配置自动 Cordon 策略。
- [ ] 9. **【热备池水位监控】** 监控平台热备节点池水位，低于 1% 时立即向机房运维发出巡检补货告警。
- [ ] 10. **【全链路演练复盘】** 每季度组织一次非预期断电与坏卡拔出逃生演练，实测验证 MTTR 是否稳定在 5 分钟以内。

---

## 8. 第八部分：Ringi 5 点口诀、自我检验清单与课后深度思考题

![Ringi 导师解构：毫秒级 XID 探针与集群热备自愈指挥舱](assets/ringi_capstone04_xid_fault_healing_cockpit.png)

### 8.1 5 点押韵核心速记口诀

```text
万卡调度莫乱投，原子准入消死囚。
部门配额可借用，主人回眸优雅收。
拓扑感知机内守，双轨导轨畅无忧。
显存拦截硬如铁，虚实隔离不发愁。
故障探针毫秒嗅，热备自愈立神舟！
```

---

### 8.2 10 条白板自我检验清单

- [ ] 1. 什么是哲学家就餐死锁？为什么传统 K8s 逐 Pod 调度在大模型分布式训练中极易诱发死锁？
- [ ] 2. 详细阐述 Kueue 中 LocalQueue、ClusterQueue 与 Cohort 的关系与弹性借用逻辑。
- [ ] 3. 为什么在分布式训练中，同一个 Tensor Parallel（TP=8）组绝不允许跨越单机物理边界？
- [ ] 4. 什么是拓扑感知调度（Topology-Aware Scheduling）？它如何感知单机内的 NVLink 连接矩阵？
- [ ] 5. 比较 NVIDIA vGPU、MPS 与 HAMi 三种切分方案在显存隔离与故障隔离上的本质区别。
- [ ] 6. 说明 HAMi 是如何通过用户态 `LD_PRELOAD` 劫持 `cuMemAlloc` 实现显存硬限制的。
- [ ] 7. 什么是 Rail-Optimized（导轨网络）？它相比传统无阻断 Fat-Tree 架构能节省多少核心交换机成本？
- [ ] 8. 详细画出万卡集群在捕获 GPU XID 79（掉卡）错误后的自动化故障自愈时序图。
- [ ] 9. 为什么必须为分布式训练设置优雅驱逐宽限期（Grace Period）？3 分钟宽限期通常用来做什么？
- [ ] 10. 解释四级存储金字塔（HBM ➔ NVMe ➔ 3FS ➔ S3）如何将万卡 Checkpoint 保存耗时从数十分钟压缩到数秒。

---

### 8.3 3 道高阶开放式课后思考题

1. **抢占雪崩防范（Cascading Preemption Trap）**：在复杂的跨部门弹性借用体系中，如果高优先级作业频繁发起抢占，可能会导致低优先级作业不断被中断、重试，最终算力全部浪费在 Checkpoint 保存与恢复的开销上（Thrashing）。系统设计层面应引入哪些滞后惩罚（Hysteresis）与冷却机制？
2. **异构集群混部统一调度（Heterogeneous Co-scheduling）**：若机房内同时存在 NVIDIA H100、A100 与华为昇腾 NPU 节点，如何基于同一套 Kubernetes 控制面设计统一的算力抽象层，使上层模型训练框架感知不到底层硬件差异？
3. **网络拥塞扩散（Congestion Spreading）根因定位**：在万卡 RoCEv2 导轨网络中，若某一台机器的光纤因弯折衰减发生持续丢包，PFC 会向上游逐级发送暂停帧（PAUSE Frame），导致整个交换机端口被大面积堵死。如何在平台侧秒级揪出产生拥塞的最初源头节点？

---

## 9. 第九部分：知识库与权威论文证据溯源

本章所有架构设计、调度状态机与自愈规程均严格溯源于以下权威文献与本地实测证据库：

1. **云原生 AI 平台与调度体系**：
   - 参考 **AI_BOOK/AI-fundamentals/03_ai_cluster_ops/** 与 **AI_BOOK/AI-fundamentals/04_cloud_native_ai_platform/**：核对生产级集群运维与调度体系。
2. **Kueue 官方架构原著与源码规范**：
   - Kubernetes SIGs: *Kueue: Kubernetes-native Job Queueing System Architecture RFC*。
3. **GPU 虚拟化与 HAMi 架构**：
   - Project HAMi (Heterogeneous AI Computing Virtualization) 官方规范与驱动 Hook 规范。
4. **超算中心拓扑与网络白皮书**：
   - NVIDIA SuperPOD: *Next-Generation AI Infrastructure Architecture Guide*。

---

## 附录：Appendix A — 大厂架构师级面试真题深度破局

### Q1: 在万卡规模集群中，如果让你作为总架构师设计一套混合承载万卡预训练与千卡在线 Serving 的基础设施平台，请在白板上画出你的架构核心拓扑，并说明网络与存储的关键设计。
**Ringi 考官拆解与满分回答**：
1. **控制与调度层（软隔离与弹性借用）**：
   - 采用 Kubernetes + Kueue 构建控制中枢，使用 Cohort 将全集群算力划分为保底配额与弹性共享池；
   - 预训练任务实施严格 Gang 调度（All-or-Nothing），在线 Serving 配置基于排队时延的 HPA 与自适应背压；
   - 白天 Serving 高峰时，自动抢占并优雅驱逐离线批处理任务，归还算力。
2. **双轨算网拓扑（硬物理隔离）**：
   - **计算专网**：万卡预训练采用独立无损 8x400G RoCEv2，采用 Rail-Optimized 导轨拓扑，保证同卡号直连同一 Leaf 组；
   - **存储专网**：独立的双 100G 网络平面，负责 Checkpoint 读写与镜像拉取，绝不与计算 RoCE 流量争抢带宽。
3. **四级分层存储流水线**：
   - 训练节点本地配置 1.5TB NVMe 本地盘作为高速缓存（Scratch Cache）；
   - Checkpoint 保存时，主进程百毫秒内将 DCP 写入本地 NVMe 立即恢复计算，后台 Daemon 异步同步至 3FS / Ceph 高性能并行存储，并定期归档至 S3。
4. **全自动容灾 SLA**：
   - 部署轻量级 DaemonSet 500ms 周期嗅探 Xid 故障，预留 2% 热备节点池，做到单点故障 5 分钟内自愈回滚。

---

### Q2: 为什么在多租户 GPU 共享场景中，基于用户态 Hook 的显存拦截方案（如 HAMi）比原生 NVIDIA MPS 更加安全稳定？
**Ringi 考官拆解与满分回答**：
1. **原生 MPS 的阿喀琉斯之踵（共享上下文故障级联）**：
   - MPS 的核心设计初衷是提升小 Kernel 的并发利用率，其原理是将多个进程的指令合并到同一个由 Control Daemon 维护的单一 CUDA Context 中；
   - **致命安全缺陷**：由于共享同一个硬件上下文空间，一旦其中一个租户的进程发生了内存越界、非法指令或段错误（Segmentation Fault），整个 CUDA Context 会被驱动强制重置；
   - 这会导致同一张 GPU 上运行的所有其他合法租户的业务进程**在一瞬间全部被连带杀死**！
2. **HAMi 用户态动态 Hook 的沙箱隔离**：
   - 每个容器依然拥有完全独立、隔离的操作系统进程与独立的 CUDA Context 上下文；
   - HAMi 仅在用户态（`libvgpu.so`）对 `cuMemAlloc` 等显存申请 API 进行无侵入拦截；
   - 当某个容器试图申请超过声明配额的显存时，拦截层在进入内核驱动前直接返回标准错误码 `CUDA_ERROR_OUT_OF_MEMORY`；
   - 只有违规的这一个容器抛出 OOM 异常或自行重启，宿主机物理驱动与相邻的租户容器完全不受任何影响，实现了高可靠的安全生产隔离！
