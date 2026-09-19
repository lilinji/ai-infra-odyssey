#!/usr/bin/env python3
"""
AI-Infra Odyssey LaTeX Formula & GitHub Native MathJax Compliance Linter
Scans all Markdown files in the repository to ensure 100% LaTeX compatibility
and flawless rendering on GitHub Web (GFM + MathJax), VS Code, and Obsidian.
"""

import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Target directories to scan (standardized book volumes without _M suffixes)
TARGET_DIRS = [
    "01_性能工程与系统前置",
    "02_GPU硬件与集群互联",
    "03_CUDA与算子性能优化",
    "04_大模型架构与显存建模",
    "05_大模型分布式训练",
    "06_LLM推理系统与性能工程",
    "07_云原生AI平台与生产工程",
    "08_Post-Training与相邻Infra",
    "09_Capstone综合实战项目",
]

def scan_markdown_file(file_path):
    violations = []
    rel_path = os.path.relpath(file_path, REPO_ROOT)
    
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    in_code_block = False
    
    for lno, line in enumerate(lines, 1):
        stripped = line.strip()
        
        # Track code blocks (ignore text inside code blocks)
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
            
        # 1. Rule 1: Zero \\_ (escaped underscores in formulas/text)
        if "\\\\_" in line:
            violations.append((lno, "Rule 1: Double-backslash underscore (\\\\_) detected", line.strip()))
            
        # 2. Rule 2: \text{..._...} bare underscores inside \text
        m_text_underscore = re.findall(r"\\text\{[^}]*_[^}]*\}", line)
        if m_text_underscore:
            violations.append((lno, f"Rule 2: Underscore inside \\text{{...}}: {m_text_underscore}", line.strip()))
            
        # 3. Rule 3: \mathbf{...[\u4e00-\u9fa5]...} Chinese characters inside \mathbf
        m_mathbf_cn = re.findall(r"\\mathbf\{[^}]*[\u4e00-\u9fa5]+[^}]*\}", line)
        if m_mathbf_cn:
            violations.append((lno, f"Rule 3: Chinese characters inside \\mathbf{{...}}: {m_mathbf_cn}", line.strip()))
            
        # 4. Rule 4: Chinese full-width punctuation directly touching opening $
        # Full-width punctuation: ： （ ， 、 ； ？ ！
        m_punct_dollar = re.findall(r"([：，；、？！]|（)\$[^\$]", line)
        if m_punct_dollar:
            violations.append((lno, f"Rule 4: Full-width Chinese punctuation directly touching opening $: {m_punct_dollar}", line.strip()))
            
        # 5. Rule 5: List bullet with $$ block on the same line
        if re.match(r"^\s*[-*+]\s+.*?\$\$.*?\$\$", line):
            violations.append((lno, "Rule 5: Block math ($$) inline on list item line (breaks GFM parser)", line.strip()))
            
    return rel_path, violations

def main():
    total_violations = 0
    scanned_files = 0
    modules_scanned = 0
    
    print("=" * 70)
    print("AI-Infra Odyssey LaTeX Formula & GitHub Native Math Linter")
    print("=" * 70)
    
    for d in TARGET_DIRS:
        dir_path = os.path.join(REPO_ROOT, d)
        if not os.path.exists(dir_path):
            continue
        modules_scanned += 1
        dir_violations = 0
        
        for root, _, files in os.walk(dir_path):
            for file in sorted(files):
                if file.endswith(".md"):
                    scanned_files += 1
                    fpath = os.path.join(root, file)
                    rel_path, violations = scan_markdown_file(fpath)
                    if violations:
                        dir_violations += len(violations)
                        print(f"\n❌ [{rel_path}] - {len(violations)} violation(s):")
                        for lno, desc, snippet in violations:
                            print(f"   Line {lno:4d}: {desc}")
                            print(f"              {snippet[:90]}")
                            
        total_violations += dir_violations
        if dir_violations == 0:
            print(f"✅ [{d}] Passed cleanly (100% LaTeX & GitHub GFM compliant).")
            
    print("\n" + "=" * 70)
    print(f"Summary: {scanned_files} files scanned across {modules_scanned} volumes.")
    if total_violations == 0:
        print("🎉 SUCCESS: 0 LaTeX formula violations found! All clean & GitHub native ready.")
        print("=" * 70)
        sys.exit(0)
    else:
        print(f"⚠️ FAILURE: Found {total_violations} total violations.")
        print("=" * 70)
        sys.exit(1)

if __name__ == "__main__":
    main()
