#!/usr/bin/env python3
"""
AI-Infra Odyssey LaTeX / KaTeX Formula Linter Proxy
Delegates to scripts/verify_latex_formulas.py to maintain backward compatibility.
"""
import os
import sys
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_SCRIPT = os.path.join(SCRIPT_DIR, "verify_latex_formulas.py")

if __name__ == "__main__":
    result = subprocess.run([sys.executable, TARGET_SCRIPT] + sys.argv[1:])
    sys.exit(result.returncode)
