import os
import re

def fix_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    list_pattern = re.compile(r'^\s*([-*]|[0-9]+\.)\s+')
    header_pattern = re.compile(r'^\s*#+')
    empty_pattern = re.compile(r'^\s*$')
    block_pattern = re.compile(r'^\s*>')
    fence_pattern = re.compile(r'^\s*```')

    new_lines = []
    in_fence = False

    for i in range(len(lines)):
        if fence_pattern.match(lines[i]):
            in_fence = not in_fence
            new_lines.append(lines[i])
            continue

        if in_fence:
            new_lines.append(lines[i])
            continue

        if i > 0 and list_pattern.match(lines[i]):
            prev_line = lines[i-1]
            if not empty_pattern.match(prev_line) and \
               not list_pattern.match(prev_line) and \
               not header_pattern.match(prev_line) and \
               not block_pattern.match(prev_line) and \
               not fence_pattern.match(prev_line) and \
               '---' not in prev_line:
                new_lines.append('\n')

        new_lines.append(lines[i])

    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'parts')
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith('.md'):
            full_path = os.path.join(root, file)
            fix_file(full_path)
