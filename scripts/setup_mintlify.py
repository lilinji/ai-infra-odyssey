import os
import re
import glob
import json

REPO_ROOT = "/Users/ringi/ai-infra-odyssey"
DOCS_DIR = os.path.join(REPO_ROOT, "docs")

MODULE_TITLES = {
    "01_性能工程与系统前置": "第一卷：性能工程与系统前置 (Module 01: Prerequisites)",
    "02_GPU硬件与集群互联": "第二卷：GPU硬件微架构与集群互联 (Module 02: Hardware & Interconnect)",
    "03_CUDA与算子性能优化": "第三卷：CUDA与算子性能优化实战 (Module 03: CUDA & Kernel Optimization)",
    "04_大模型架构与显存建模": "第四卷：大模型架构与显存精确建模 (Module 04: Architecture & Memory Modeling)",
    "05_大模型分布式训练": "第五卷：大模型分布式并行训练 (Module 05: Distributed Training & 3D Parallelism)",
    "06_LLM推理系统与性能工程": "第六卷：LLM高并发推理系统与性能工程 (Module 06: LLM Inference & Serving)",
    "07_云原生AI平台与生产工程": "第七卷：云原生AI算力平台与调度治理 (Module 07: Cloud-Native AI Platform)",
    "08_Post-Training与相邻Infra": "第八卷：Post-Training强化学习与相邻Infra (Module 08: Post-Training & Adjacent Infra)",
    "09_Capstone综合实战项目": "第九卷：5 大企业级 Capstone 综合实战战役 (Module 09: Capstone Projects)"
}

MODULE_DESCRIPTIONS = {
    "01_性能工程与系统前置": "扎实掌握 C++17/Python 底层模型、计算机体系结构、Linux 性能分析、PyTorch 内存布局与算子执行图。",
    "02_GPU硬件与集群互联": "深度解密 SM 调度、Tensor Core、Roofline 模型、RDMA/GPUDirect、跨机通信与 NCCL 算法实现。",
    "03_CUDA与算子性能优化": "深入 CUDA 编程模型、Memory Coalescing、Online Softmax、GEMM 分块与 FlashAttention Triton 实战。",
    "04_大模型架构与显存建模": "系统剖析 Decoder-only 结构、MHA/GQA、RoPE、SwiGLU 与训练/推理白板显存全景账本推导。",
    "05_大模型分布式训练": "全栈攻坚 DDP 分桶重叠、ZeRO-1/2/3、FSDP、TP/PP/SP 混合并行与千卡训练故障排查治理。",
    "06_LLM推理系统与性能工程": "解构 Prefill/Decode 特性、PagedAttention、vLLM 架构源码、PD 分离架构与长尾 SLO 治理。",
    "07_云原生AI平台与生产工程": "实战 K8s Device Plugin/DRA、Gang 调度/Kueue、HAMi 虚拟化、DeepSeek 3FS 与多级缓存体系。",
    "08_Post-Training与相邻Infra": "评估 SFT/LoRA/PPO 显存负载、RAG 向量检索沙箱、异构加速芯片抽象与系统设计面试题库。",
    "09_Capstone综合实战项目": "5 大工业级顶石项目实战：Scaling Study、Serving 极致调优、CUDA 算子、集群 RFC 与 Token 全链路。"
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

def clean_title(raw_title):
    # Remove markdown formatting, leading numbers
    t = re.sub(r'^[#\s]+', '', raw_title).strip()
    t = t.replace('`', '').replace('"', '').replace("'", "")
    # If title is very long with a dash separator, make it readable
    if "——" in t:
        parts = t.split("——")
        # Keep lecture prefix + main topic
        t = parts[0].strip() + "：" + parts[1].split("（")[0].strip()
    return t[:80]

def process_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Determine title & description
    title = ""
    description = ""

    if content.startswith("---"):
        parts = content.split("---", 2)
        frontmatter_text = parts[1]
        body = parts[2] if len(parts) > 2 else ""
        
        # parse title
        m_t = re.search(r'^title:\s*["\']?(.*?)["\']?$', frontmatter_text, re.MULTILINE)
        if m_t:
            title = m_t.group(1).strip()
        m_d = re.search(r'^description:\s*["\']?(.*?)["\']?$', frontmatter_text, re.MULTILINE)
        if m_d:
            description = m_d.group(1).strip()
    else:
        body = content

    # If title missing, extract from first H1
    if not title:
        m_h1 = re.search(r'^#\s+(.+)$', body, re.MULTILINE)
        if m_h1:
            title = m_h1.group(1).strip()
        else:
            base = os.path.basename(file_path).replace(".md", "").replace(".mdx", "")
            title = base

    title = clean_title(title)

    if not description:
        # Try blockquote core guide
        m_desc = re.search(r'>\s*\*\*核心导读\*\*：\s*(.+)', body)
        if not m_desc:
            m_desc = re.search(r'>\s*\*\*学习目标\*\*：\s*(.+)', body)
        if m_desc:
            description = m_desc.group(1).strip()[:160]
        else:
            description = f"系统学习 {title} 的核心技术原理与工业级落地实战。"

    description = description.replace('"', '\\"').replace('\n', ' ')

    # Format new frontmatter
    new_frontmatter = f"""---
title: "{title}"
description: "{description}"
---
"""
    # Clean body: if body previously had frontmatter, strip it
    if body.startswith("\n"):
        body = body.lstrip("\n")

    # Replace relative link extensions: [text](path.md) -> [text](path)
    body = re.sub(r'(\]\([^\)]+?)\.md(\))', r'\1\2', body)

    # Rename file to .mdx
    new_path = os.path.splitext(file_path)[0] + ".mdx"
    
    with open(new_path, "w", encoding="utf-8") as f:
        f.write(new_frontmatter + "\n" + body)

    if new_path != file_path and os.path.exists(file_path):
        os.remove(file_path)

    return title, description, new_path

print("Starting scan and conversion...")

modules = sorted([d for d in os.listdir(DOCS_DIR) if os.path.isdir(os.path.join(DOCS_DIR, d)) and d[0].isdigit()])

module_data = {}

for mod in modules:
    mod_path = os.path.join(DOCS_DIR, mod)
    md_files = sorted(glob.glob(os.path.join(mod_path, "*.md")) + glob.glob(os.path.join(mod_path, "*.mdx")))
    
    # Process README first if exists
    readme_path = os.path.join(mod_path, "README.md")
    if not os.path.exists(readme_path):
        readme_path = os.path.join(mod_path, "README.mdx")
    
    articles = []
    
    for f in md_files:
        base = os.path.basename(f)
        t, d, new_f = process_file(f)
        rel_no_ext = os.path.relpath(new_f, DOCS_DIR).replace(".mdx", "")
        if "README" in base:
            readme_rel = rel_no_ext
        else:
            articles.append({
                "title": t,
                "description": d,
                "path": rel_no_ext,
                "filename": os.path.basename(new_f)
            })
            
    module_data[mod] = {
        "title": MODULE_TITLES.get(mod, mod),
        "desc": MODULE_DESCRIPTIONS.get(mod, ""),
        "icon": MODULE_ICONS.get(mod, "book"),
        "root": f"{mod}/README",
        "articles": articles
    }
    print(f"Processed {mod}: {len(articles)} articles + README")

print("All files processed!")
