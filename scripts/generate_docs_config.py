import os
import glob
import json

REPO_ROOT = "/Users/ringi/ai-infra-odyssey"
DOCS_DIR = os.path.join(REPO_ROOT, "docs")

MODULE_TITLES = {
    "01_性能工程与系统前置": "第一卷：性能工程与系统前置",
    "02_GPU硬件与集群互联": "第二卷：GPU硬件微架构与集群互联",
    "03_CUDA与算子性能优化": "第三卷：CUDA与算子性能优化实战",
    "04_大模型架构与显存建模": "第四卷：大模型架构与显存精确建模",
    "05_大模型分布式训练": "第五卷：大模型分布式并行训练",
    "06_LLM推理系统与性能工程": "第六卷：LLM高并发推理系统与性能工程",
    "07_云原生AI平台与生产工程": "第七卷：云原生AI算力平台与调度治理",
    "08_Post-Training与相邻Infra": "第八卷：Post-Training强化学习与相邻Infra",
    "09_Capstone综合实战项目": "第九卷：5 大企业级 Capstone 综合实战"
}

MODULE_SHORT_DESCS = {
    "01_性能工程与系统前置": "C++17/Python 底层模型、NUMA、Linux 性能分析、PyTorch 内存布局与算子图。",
    "02_GPU硬件与集群互联": "SM 调度、Tensor Core、Roofline 模型、RDMA、NVLink 与 NCCL 算法机理。",
    "03_CUDA与算子性能优化": "CUDA 编程模型、Memory Coalescing、Online Softmax、GEMM 分块与 FlashAttention 实战。",
    "04_大模型架构与显存建模": "Decoder-only 架构、MHA/GQA、RoPE、SwiGLU 与训练/推理显存全景账本推导。",
    "05_大模型分布式训练": "DDP 分桶重叠、ZeRO-1/2/3、FSDP、TP/PP/SP 混合并行与大规模故障排查。",
    "06_LLM推理系统与性能工程": "Prefill/Decode 特性、PagedAttention、vLLM 架构源码、PD 分离架构与 SLO 治理。",
    "07_云原生AI平台与生产工程": "K8s Device Plugin/DRA、Gang 调度/Kueue、HAMi 虚拟化、DeepSeek 3FS 与多级缓存。",
    "08_Post-Training与相邻Infra": "SFT/LoRA/PPO 显存负载、RAG 向量检索沙箱、异构加速芯片抽象与系统设计题库。",
    "09_Capstone综合实战项目": "5 大工业级顶石项目实战：Scaling Study、Serving 极限调优、CUDA 算子与全链路优化。"
}

MODULE_ICONS = {
    "01_性能工程与系统前置": "microchip",
    "02_GPU硬件与集群互联": "network-wired",
    "03_CUDA与算子性能优化": "bolt",
    "04_大模型架构与显存建模": "calculator",
    "05_大模型分布式训练": "cubes",
    "06_LLM推理系统与性能工程": "gauge-high",
    "07_云原生AI平台与生产工程": "server",
    "08_Post-Training与相邻Infra": "brain",
    "09_Capstone综合实战项目": "trophy"
}

# Define 3 major tabs
SECTIONS = [
    {
        "group": "全书导读与全景总览",
        "pages": ["index", "style-guide", "content-types"]
    },
    {
        "group": "硬件微架构与算子 (卷一~三)",
        "modules": ["01_性能工程与系统前置", "02_GPU硬件与集群互联", "03_CUDA与算子性能优化"]
    },
    {
        "group": "大模型与分布式系统 (卷四~六)",
        "modules": ["04_大模型架构与显存建模", "05_大模型分布式训练", "06_LLM推理系统与性能工程"]
    },
    {
        "group": "平台治理与顶石战役 (卷七~九)",
        "modules": ["07_云原生AI平台与生产工程", "08_Post-Training与相邻Infra", "09_Capstone综合实战项目"]
    }
]

groups_config = [SECTIONS[0]]
all_modules_info = []

for sec in SECTIONS[1:]:
    sec_pages = []
    for mod in sec["modules"]:
        mod_dir = os.path.join(DOCS_DIR, mod)
        mdx_files = sorted([f for f in os.listdir(mod_dir) if f.endswith(".mdx") and f != "README.mdx"])
        pages = [f"{mod}/{f.replace('.mdx', '')}" for f in mdx_files]
        sec_pages.append({
            "group": MODULE_TITLES[mod],
            "root": f"{mod}/README",
            "expanded": False,
            "pages": pages
        })
        all_modules_info.append({
            "dir": mod,
            "title": MODULE_TITLES[mod],
            "desc": MODULE_SHORT_DESCS[mod],
            "icon": MODULE_ICONS[mod],
            "root": f"{mod}/README",
            "pages_count": len(pages),
            "pages": pages
        })
    groups_config.append({
        "group": sec["group"],
        "pages": sec_pages
    })

docs_json = {
    "$schema": "https://mintlify.com/docs.json",
    "name": "AI Infra 大话西游之水滴石穿",
    "theme": "maple",
    "colors": {
        "primary": "#DA7756",
        "light": "#E07A5F",
        "dark": "#C15F3D"
    },
    "background": {
        "color": {
            "light": "#FAF9F5",
            "dark": "#181816"
        }
    },
    "navbar": {
        "links": [
            {
                "type": "github",
                "href": "https://github.com/lilinji/ai-infra-odyssey"
            }
        ]
    },
    "footer": {
        "socials": {
            "github": "https://github.com/lilinji/ai-infra-odyssey"
        }
    },
    "favicon": "/favicon.svg",
    "navigation": {
        "groups": groups_config
    }
}

with open(os.path.join(DOCS_DIR, "docs.json"), "w", encoding="utf-8") as f:
    json.dump(docs_json, f, ensure_ascii=False, indent=2)

print("Generated docs/docs.json successfully with full sidebar groups!")

# Now build docs/index.mdx
card_group_items = ""
for m in all_modules_info:
    card_group_items += f"""  <Card title="{m['title']}" icon="{m['icon']}" href="/{m['root']}">
    {m['desc']} ({m['pages_count']} 篇核心讲义)
  </Card>\n"""

accordion_catalog_items = ""
for m in all_modules_info:
    accordion_catalog_items += f"### [{m['title']}](/{m['root']})\n\n"
    for p in m["pages"]:
        fname = os.path.basename(p)
        accordion_catalog_items += f"- [{fname}](/{p})\n"
    accordion_catalog_items += "\n"

index_mdx = f"""---
title: "AI Infra 大话西游之水滴石穿"
description: "从基础设施到智能未来的工程哲学与实战指南：涵盖从底层物理硅基、编译算子到万卡超算调度中枢的完整 8 层工业级 AI Infra 技术栈。"
---

# 🌊 《AI Infra 大话西游之水滴石穿》
### —— 从基础设施到智能未来的工程哲学与实战指南

> **主讲人 / IP 定位**：👓 **Ringi**（大厂 AI Infrastructure 架构师）  
> **核心宗旨**：*“我是 Ringi，我也在系统学习 AI Infra。这是我把一个原本很难、很理论的底层知识真正搞懂之后，重新用工程师视角讲给你的实战版本。”*  
> **课程定位**：面向准备进入大厂 AI Infra、GPU 计算、大模型分布式训练/在线推理优化与云原生平台的研发工程师与架构师。  
> **内容规模**：**全书九大分卷 · 44 篇万字核心系统讲义 + 5 大生产级 Capstone 顶石项目（全套 49 篇已 100% 全部完结交付！）**  
> **学习时长**：**80 小时核心必修 + 12 小时选修扩展 + 5 大企业级实战战役**

---

<p align="center">
  <img src="/assets/book_cover.png" alt="AI Infra 大话西游之水滴石穿 官方封面" width="480" />
</p>

---

## 🗺️ 全景技术体系总架构图 (Master Architecture)

本专栏历时沉淀，构建了从底层物理硅基、编译算子到万卡超算调度中枢的完整 8 层工业级技术栈：

<Accordion title="点击展开查看超清技术全景总架构大图 (ai_infra_master_architecture.svg)" defaultOpen>

![AI Infra 大话西游之水滴石穿 全景技术体系总架构图](/assets/ai_infra_master_architecture.svg)

</Accordion>

---

## 🏛️ 全书九大分卷速览与核心入口

<CardGroup cols={3}>
{card_group_items}</CardGroup>

---

## 📊 企业级大模型算力平台交付汇报 (Enterprise Delivery Report)

面向万卡大模型算力平台的生产级交付与验收体系，全套 9 页瑞士极简风超清矢量图示：

<CardGroup cols={3}>
  <Card title="01 交付总览与验收" icon="clipboard-check" href="/assets/slides/P01.svg">
    全栈验收标准、指标基线与交付里程碑规划。
  </Card>
  <Card title="02 物理微架构与集群互联" icon="microchip" href="/assets/slides/P02.svg">
    GPU 节点拓扑、RoCE/IB 双轨无损算网与 Clos 架构。
  </Card>
  <Card title="03 算力显存通信三账本" icon="calculator" href="/assets/slides/P03.svg">
    FLOPs 利用率、参数/梯度/激活显存与集合通信拓扑。
  </Card>
  <Card title="04 算子熔炉与片上 SRAM" icon="bolt" href="/assets/slides/P04.svg">
    Memory Coalescing、Online Softmax 与 FlashAttention Tiling。
  </Card>
  <Card title="05 分布式 3D+MoE 编排" icon="cubes" href="/assets/slides/P05.svg">
    TP/PP/DP/SP 混合切分、ZeRO 分片与负载均衡调度。
  </Card>
  <Card title="06 在线推理与 PD 分离" icon="gauge-high" href="/assets/slides/P06.svg">
    PagedAttention、连续批处理与 Prefill-Decode 分离架构。
  </Card>
  <Card title="07 智算调度与资源虚拟化" icon="server" href="/assets/slides/P07.svg">
    K8s Device Plugin、Kueue 队列与 HAMi 多租户隔离。
  </Card>
  <Card title="08 分布式存储与容灾救援" icon="hard-drive" href="/assets/slides/P08.svg">
    DeepSeek 3FS、异步快照与多级 KV Cache 缓存体系。
  </Card>
  <Card title="09 故障图谱与根因决策树" icon="shield-halved" href="/assets/slides/P09.svg">
    NCCL Hang、慢节点定位、Xid 告警与毫秒级自愈。
  </Card>
</CardGroup>

---

<Accordion title="展开查看全书 9 大分卷 49 篇核心讲义完整目录明细">

{accordion_catalog_items}
</Accordion>

---

## 🛠️ 本地交互式阅读与开发指令

<CodeGroup>
```bash 启动本地实时热重载预览
npm run dev
# 或直接执行
npx mintlify dev --dir docs
```

```bash 执行 MDX 语法与 AST 编译校验
npm run validate
# 或直接执行
npx mintlify validate --dir docs
```

```bash 检查全站内部链接与死链
npm run broken-links
# 或直接执行
npx mintlify broken-links --dir docs
```
</CodeGroup>
"""

with open(os.path.join(DOCS_DIR, "index.mdx"), "w", encoding="utf-8") as f:
    f.write(index_mdx)

print("Generated docs/index.mdx successfully!")
