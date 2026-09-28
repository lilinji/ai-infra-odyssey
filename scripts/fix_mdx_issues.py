import os
import glob
import re

DOCS_DIR = "/Users/ringi/ai-infra-odyssey/docs"

mdx_files = glob.glob(os.path.join(DOCS_DIR, "**", "*.mdx"), recursive=True)

print(f"Checking {len(mdx_files)} MDX files...")

fixed_count = 0

for file_path in mdx_files:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    orig = content

    # 1. Remove `<!-- more -->` or similar hexo comments
    content = re.sub(r'<!--\s*more\s*-->', '', content)

    # 2. Convert remaining HTML comments to JSX comments (outside of codeblocks)
    def comment_replacer(match):
        return "{/* " + match.group(1).strip() + " */}"
    content = re.sub(r'<!--(.*?)-->', comment_replacer, content, flags=re.DOTALL)

    # 3. Escape < followed by a number or space outside codeblocks, e.g. <10 -> &lt;10
    # Split by code blocks ```
    parts = content.split("```")
    for i in range(0, len(parts), 2): # even index is outside code fence
        # Replace < followed by a number, like <10 or < 10
        parts[i] = re.sub(r'<(\d)', r'&lt;\1', parts[i])
        parts[i] = re.sub(r'<\s*(\d)', r'&lt; \1', parts[i])
        # Replace cols=3 in JSX to cols={3}
        parts[i] = re.sub(r'cols=(\d+)', r'cols={\1}', parts[i])

    content = "```".join(parts)

    if content != orig:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        fixed_count += 1

print(f"Fixed {fixed_count} files!")
