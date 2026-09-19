#!/usr/bin/env python3
"""
AI-Infra Odyssey - GitHub Native LaTeX Auto-Formatter
Cleans up Chinese full-width punctuation collisions with LaTeX math delimiters ($...$)
across all Markdown files to ensure 100% native rendering on GitHub Web (GFM + MathJax).
"""

import glob
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def fix_content(content):
    lines = content.splitlines(keepends=True)
    in_code_block = False
    mod_lines = []
    changes = 0
    
    for line in lines:
        stripped = line.strip()
        # Toggle code block
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            mod_lines.append(line)
            continue
            
        if in_code_block:
            mod_lines.append(line)
            continue
            
        orig = line
        
        # 1. Chinese/ASCII colon touching $: ：$ -> ： $
        line = re.sub(r'([：:])\$(?!\$)', r'\1 $', line)
        
        # 2. Chinese open paren touching $: （$ -> （ $
        line = re.sub(r'([（])\$(?!\$)', r'\1 $', line)
        
        # 3. Closing $ touching Chinese close paren: $） -> $ ）
        line = re.sub(r'(?<!\$)\$([）])', r'$ \1', line)
        
        # 4. Chinese comma/semicolon/dunhao touching $: ，$ -> ， $
        line = re.sub(r'([，；、])\$(?!\$)', r'\1 $', line)
        
        # 5. Chinese question mark / exclamation touching $: ？$ -> ？ $
        line = re.sub(r'([？！])\$(?!\$)', r'\1 $', line)
        
        if line != orig:
            changes += 1
            
        mod_lines.append(line)
        
    return "".join(mod_lines), changes

def main():
    target_pattern = os.path.join(REPO_ROOT, "**", "*.md")
    all_files = sorted(glob.glob(target_pattern, recursive=True))
    
    # Filter out node_modules, .git, etc.
    md_files = [f for f in all_files if not any(x in f for x in [".git", "node_modules", ".gemini"])]
    
    total_files_changed = 0
    total_lines_changed = 0
    
    print(f"Scanning {len(md_files)} markdown files for GitHub Native LaTeX compatibility...")
    
    for fpath in md_files:
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()
            
        new_content, changes = fix_content(content)
        if changes > 0:
            total_files_changed += 1
            total_lines_changed += changes
            rel_path = os.path.relpath(fpath, REPO_ROOT)
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(new_content)
            print(f"  ✅ Fixed {changes:2d} line(s) in {rel_path}")
            
    print("-" * 60)
    print(f"🎉 Complete! Updated {total_lines_changed} lines across {total_files_changed} files.")
    print("All formulas now comply with GitHub Native LaTeX (GFM + MathJax) formatting rules.")

if __name__ == "__main__":
    main()
