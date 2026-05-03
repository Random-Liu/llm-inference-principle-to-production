import os
import re

root_dir = "/Users/lantaol/Documents/blog"
parts_dir = os.path.join(root_dir, "parts")

files_to_scan = []

# Find files in root
for f in os.listdir(root_dir):
    if f.endswith(".md"):
        files_to_scan.append(os.path.join(root_dir, f))

# Find files in parts
for f in os.listdir(parts_dir):
    if f.endswith(".md"):
        files_to_scan.append(os.path.join(parts_dir, f))

# Regex for links
link_regex = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')

issues = {
    "spacing": [],
    "links": []
}

def file_exists(path):
    return os.path.exists(path)

for file_path in files_to_scan:
    rel_path = os.path.relpath(file_path, root_dir)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
        # Check spacing
        blank_count = 0
        for i, line in enumerate(lines):
            if line.strip() == "":
                blank_count += 1
            else:
                if blank_count > 1:
                    issues["spacing"].append({
                        "file": rel_path,
                        "line": i - blank_count + 1,
                        "count": blank_count
                    })
                blank_count = 0
                
        # Check links
        for i, line in enumerate(lines):
            matches = link_regex.findall(line)
            for match in matches:
                text, url = match
                if url.startswith(".") or url.startswith("#"):
                    parts = url.split("#")
                    target_file = parts[0]
                    
                    if target_file:
                        target_path = os.path.abspath(os.path.join(os.path.dirname(file_path), target_file))
                        if not file_exists(target_path):
                            issues["links"].append({
                                "file": rel_path,
                                "line": i + 1,
                                "link": url,
                                "reason": "File missing"
                            })

# Output Markdown Table
print("| File | Line Number | Issue |")
print("| :--- | :--- | :--- |")

for issue in issues["spacing"]:
    print(f"| [{issue['file']}](file://{os.path.join(root_dir, issue['file'])}) | {issue['line']} | {issue['count']} consecutive blank lines |")

for issue in issues["links"]:
    print(f"| [{issue['file']}](file://{os.path.join(root_dir, issue['file'])}) | {issue['line']} | Broken Link: {issue['link']} ({issue['reason']}) |")
