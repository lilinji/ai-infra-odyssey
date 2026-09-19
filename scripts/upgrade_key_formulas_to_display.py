#!/usr/bin/env python3
"""
Upgrade key mathematical derivations and system equations from cramped inline
formulas in bullet points into standalone, HD, elegantly centered display MathJax blocks ($$ ... $$).
"""

import os
import glob
import re

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SUBSTANTIAL_KEYWORDS = [
    r"\frac", r"\sum", r"\prod", r"\approx", r"\times", r"\cdot",
    "FLOPs", "TFLOPs", "TFLOPS", r"\text{",
    r"P_{\text{peak}}", r"C_{\text{peak}}", r"B_{\text{peak}}",
    r"T_{\text{ring}}", r"T_{\text{tree}}", r"t_{\text{bubble}}", r"T_{\text{total}}",
    r"M_{\text{cluster}}", r"M_{\text{per-gpu}}", r"M_{\text{kv", r"\|\mathbf"
]

def is_substantial_formula(form: str) -> bool:
    if form.count("=") >= 2:
        return True
    for kw in SUBSTANTIAL_KEYWORDS:
        if kw in form:
            return True
    return False

def upgrade_file(fpath: str) -> int:
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.split("\n")
    new_lines = []
    in_fence = False
    upgrades = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            new_lines.append(line)
            continue

        if in_fence:
            new_lines.append(line)
            continue

        # Check for bullet item with single substantial equation:
        # e.g. "- **标准总计算量**： $\text{FLOPs} = 2P + 4P = 6P$；"
        # or "1. **前向总发送量**： $2 \text{ 次 AllReduce} \times 2 bsh = 4 bsh$；"
        m = re.match(r"^([ \t]*[-\*\d\.]+\s+.*?[:：])\s*\$([^\$]+=[^\$]+)\$\s*([；;。]?)$", line)
        if m:
            prefix, formula, suffix = m.groups()
            formula_clean = formula.strip()
            
            # Avoid breaking simple short assignments like $i = 0$ or $x = 1$
            if is_substantial_formula(formula_clean) and len(formula_clean) > 15:
                # Add trailing punctuation into formula if appropriate or drop markdown punctuation
                upgrades += 1
                # Format prefix cleanly, ensure blank line before $$
                pref_clean = prefix.rstrip()
                new_lines.append(pref_clean)
                new_lines.append("")
                new_lines.append("$$")
                new_lines.append(formula_clean)
                new_lines.append("$$")
                continue

        new_lines.append(line)

    if upgrades > 0:
        with open(fpath, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines))

    return upgrades

def main():
    md_files = sorted(glob.glob(f"{REPO_ROOT}/**/*.md", recursive=True))
    total_upgrades = 0
    modified_files = 0

    for fpath in md_files:
        if any(ignored in fpath for ignored in ["node_modules", ".git", "scratch", ".system_generated", "SPECIFICATION", "README"]):
            continue
        rel = os.path.relpath(fpath, REPO_ROOT)
        count = upgrade_file(fpath)
        if count > 0:
            print(f"✨ Upgraded {count:2d} formulas in {rel}")
            total_upgrades += count
            modified_files += 1

    print("=" * 60)
    print(f"Total equations upgraded to display MathJax: {total_upgrades} across {modified_files} files.")

if __name__ == "__main__":
    main()
