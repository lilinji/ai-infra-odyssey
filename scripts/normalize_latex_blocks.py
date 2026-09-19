#!/usr/bin/env python3
"""
AI-Infra Odyssey - Universal LaTeX Block Isolator & Normalizer
Ensures all LaTeX display math blocks ($$...$$) across all Markdown files:
1. Are flush left (column 0, zero indentation) to avoid GFM list paragraph & italic (_ -> <em>) mangling;
2. Have blank lines before and after;
3. Converts single-line "$$ formula $$" into standard 3-line blocks;
4. Clamps over-indented list items following formulas to prevent accidental indented code block conversion.
"""

import glob
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def normalize_blocks(content):
    lines = content.splitlines(keepends=True)
    in_code = False
    output_lines = []
    changes = 0
    
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        stripped = line.strip()
        
        # Code fence tracking
        if stripped.startswith("```"):
            in_code = not in_code
            output_lines.append(line)
            idx += 1
            continue
            
        if in_code:
            output_lines.append(line)
            idx += 1
            continue
            
        # Case 1: Single line block math on its own line: e.g. "   $$ formula $$" or "$$ formula $$"
        if stripped.startswith("$$") and stripped.endswith("$$") and len(stripped) > 2:
            formula = stripped[2:-2].strip()
            # Ensure blank line before
            if output_lines and output_lines[-1].strip() != "":
                output_lines.append("\n")
            output_lines.append("$$\n")
            output_lines.append(f"{formula}\n")
            output_lines.append("$$\n")
            if idx + 1 < len(lines) and lines[idx+1].strip() != "":
                output_lines.append("\n")
            changes += 1
            idx += 1
            continue
            
        # Case 2: Multiline $$ opening
        if stripped == "$$":
            had_indent = line.startswith(" ") or line.startswith("\t")
            # Collect lines of the math block
            math_lines = []
            idx += 1
            while idx < len(lines):
                cur_line = lines[idx]
                cur_stripped = cur_line.strip()
                if cur_stripped == "$$":
                    idx += 1
                    break
                else:
                    math_lines.append(cur_stripped + "\n")
                    idx += 1
                    
            need_before_blank = output_lines and output_lines[-1].strip() != ""
            need_after_blank = idx < len(lines) and lines[idx].strip() != ""
            
            if had_indent or need_before_blank:
                changes += 1
                
            if need_before_blank:
                output_lines.append("\n")
            output_lines.append("$$\n")
            output_lines.extend(math_lines)
            output_lines.append("$$\n")
            if need_after_blank:
                output_lines.append("\n")
            continue
            
        # Case 3: Over-indented list item following a formula block (>= 4 spaces before bullet)
        m_over_indent = re.match(r'^[ ]{4,}([-*+]|\d+\.)\s+(.*)$', line)
        if m_over_indent:
            # Check if previous output was a blank line following math or other content
            bullet = m_over_indent.group(1)
            rest = m_over_indent.group(2)
            # Reclamp to 2 spaces indentation to avoid markdown indented code block parsing
            new_line = f"  {bullet} {rest}\n"
            if new_line != line:
                changes += 1
                output_lines.append(new_line)
                idx += 1
                continue
                
        output_lines.append(line)
        idx += 1
        
    return "".join(output_lines), changes

def main():
    target_pattern = os.path.join(REPO_ROOT, "**", "*.md")
    all_files = sorted(glob.glob(target_pattern, recursive=True))
    md_files = [f for f in all_files if not any(x in f for x in [".git", "node_modules", ".gemini"])]
    
    total_files_changed = 0
    total_blocks_normalized = 0
    
    print(f"Normalizing LaTeX display blocks across {len(md_files)} markdown files...")
    
    for fpath in md_files:
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()
        new_content, changes = normalize_blocks(content)
        if changes > 0:
            total_files_changed += 1
            total_blocks_normalized += changes
            rel_path = os.path.relpath(fpath, REPO_ROOT)
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(new_content)
            print(f"  ✅ Normalized {changes:2d} block(s) in {rel_path}")
            
    print("-" * 60)
    print(f"🎉 Complete! Normalized {total_blocks_normalized} math blocks across {total_files_changed} files.")
    print("All LaTeX blocks are now 100% flush-left isolated, preventing GFM italic & list mangling.")

if __name__ == "__main__":
    main()
