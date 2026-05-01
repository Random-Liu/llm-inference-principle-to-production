import os
import re

def check_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    list_pattern = re.compile(r'^\s*([-*]|[0-9]+\.)\s+')
    header_pattern = re.compile(r'^\s*#+')
    empty_pattern = re.compile(r'^\s*$')
    block_pattern = re.compile(r'^\s*>')
    fence_pattern = re.compile(r'^\s*```')

    violations = []
    in_fence = False

    for i in range(1, len(lines)):
        curr_line = lines[i]
        prev_line = lines[i-1]

        if fence_pattern.match(curr_line):
            in_fence = not in_fence
            continue

        if in_fence:
            continue

        if list_pattern.match(curr_line):
            if not empty_pattern.match(prev_line) and \
               not list_pattern.match(prev_line) and \
               not header_pattern.match(prev_line) and \
               not block_pattern.match(prev_line) and \
               not fence_pattern.match(prev_line) and \
               '---' not in prev_line:
                violations.append((i + 1, prev_line.strip(), curr_line.strip()))

    return violations

base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'parts')
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith('.md'):
            full_path = os.path.join(root, file)
            errors = check_file(full_path)
            if errors:
                print(f"\n=== {file} ===")
                for line_no, prev, curr in errors:
                    print(f"Line {line_no}: '{prev}' -> '{curr}'")
