import os
import re

def fix_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    block_list_pattern = re.compile(r'^\s*>\s*([-*]|[0-9]+\.)\s+')
    block_empty_pattern = re.compile(r'^\s*>\s*$')
    block_header_pattern = re.compile(r'^\s*>\s*#+')
    block_text_pattern = re.compile(r'^\s*>\s+')

    new_lines = []
    in_fence = False
    fence_pattern = re.compile(r'^\s*```')

    for i in range(len(lines)):
        if fence_pattern.match(lines[i]):
            in_fence = not in_fence
            new_lines.append(lines[i])
            continue

        if in_fence:
            new_lines.append(lines[i])
            continue

        if i > 0 and block_list_pattern.match(lines[i]):
            prev_line = lines[i-1]
            # Check if prev_line is a blockquote line but NOT a list, NOT empty blockquote, NOT block header
            if block_text_pattern.match(prev_line) and \
               not block_list_pattern.match(prev_line) and \
               not block_empty_pattern.match(prev_line) and \
               not block_header_pattern.match(prev_line):
                new_lines.append('> \n')

        new_lines.append(lines[i])

    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'parts')
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith('.md'):
            full_path = os.path.join(root, file)
            fix_file(full_path)

print("Markdown formatting fixed successfully.")
