import os
import glob
import re

DOCS_DIR = "/Users/ringi/ai-infra-odyssey/docs"
mdx_files = glob.glob(os.path.join(DOCS_DIR, "**", "*.mdx"), recursive=True)

print(f"Normalizing links & images across {len(mdx_files)} files...")

fixed_count = 0

for file_path in mdx_files:
    if file_path == os.path.join(DOCS_DIR, "index.mdx"):
        continue

    rel_dir = os.path.relpath(os.path.dirname(file_path), DOCS_DIR)
    # rel_dir is e.g. "01_性能工程与系统前置"
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    orig = content

    # 1. Fix markdown image links:
    # ![...](assets/...) -> ![...](/{rel_dir}/assets/...)
    # ![...](./assets/...) -> ![...](/{rel_dir}/assets/...)
    content = re.sub(r'!\[([^\]]*)\]\((\.?/)?assets/([^)]+)\)', rf'![\1](/{rel_dir}/assets/\3)', content)

    # 2. Fix HTML img src:
    # src="assets/..." or src="./assets/..." -> src="/{rel_dir}/assets/..."
    content = re.sub(r'src=["\'](\.?/)?assets/([^"\']+)["\']', rf'src="/{rel_dir}/assets/\2"', content)

    # 3. Fix links to README:
    # ../README or ../../README -> /
    content = re.sub(r'\]\(\.\./(\.\./)?README(\.md)?\)', r'](/)', content)
    # ./README -> /{rel_dir}/README
    content = re.sub(r'\]\(\./README(\.md)?\)', rf'](/{rel_dir}/README)', content)

    # 4. Fix relative links between modules:
    # ../0X_xxx/README -> /0X_xxx/README
    # ../0X_xxx/YY_yyy -> /0X_xxx/YY_yyy
    content = re.sub(r'\]\(\.\./([0-9][0-9]_[^)]+)\)', r'](/\1)', content)

    if content != orig:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        fixed_count += 1

print(f"Updated links and images in {fixed_count} files!")
