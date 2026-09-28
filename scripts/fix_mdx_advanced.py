import os
import glob
import re

DOCS_DIR = "/Users/ringi/ai-infra-odyssey/docs"
mdx_files = glob.glob(os.path.join(DOCS_DIR, "**", "*.mdx"), recursive=True)

print(f"Scanning {len(mdx_files)} MDX files...")

fixed_count = 0

for file_path in mdx_files:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    orig = content

    # 1. Fix autolinks: <https://...> -> [https://...](https://...)
    content = re.sub(r'<(https?://[^>]+)>', r'[\1](\1)', content)

    # 2. Fix <br> -> <br />
    content = re.sub(r'<br\s*>', r'<br />', content, flags=re.IGNORECASE)

    # Split by code blocks
    parts = content.split("```")
    for i in range(0, len(parts), 2):
        # 3. Fix << to &lt;&lt;
        parts[i] = re.sub(r'<<', r'&lt;&lt;', parts[i])
        
        # 4. Fix lone < not starting a known component or HTML tag
        # Known tags: Accordion, CardGroup, Card, Steps, Step, Note, Tip, Warning, Check, Info, CodeGroup, Tabs, Tab, br, img, p, a, div, span, b, i, strong, em, table, thead, tbody, tr, th, td
        # If < is followed by something like <T> or <typename> or <value> outside code, escape it
        def tag_escaper(match):
            full = match.group(0)
            tag_name = match.group(1)
            known = {
                "Accordion", "AccordionGroup", "CardGroup", "Card", "Steps", "Step",
                "Note", "Tip", "Warning", "Check", "Info", "CodeGroup", "Tabs", "Tab",
                "br", "img", "p", "a", "div", "span", "b", "i", "strong", "em",
                "table", "thead", "tbody", "tr", "th", "td", "details", "summary",
                "Frame", "Icon"
            }
            if tag_name in known or tag_name.startswith("/"):
                return full
            # escape
            return "&lt;" + full[1:]

        parts[i] = re.sub(r'<([a-zA-Z0-9_\-]+)[^>]*>', tag_escaper, parts[i])

    content = "```".join(parts)

    if content != orig:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        fixed_count += 1

print(f"Fixed {fixed_count} files!")
