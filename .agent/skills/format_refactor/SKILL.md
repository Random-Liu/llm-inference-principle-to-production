# Skill: Repository-Wide Document Formatting and Cleanup Workflow

## Purpose
This skill provides a safe, reliable, and multi-step workflow for auditing and cleaning up document formatting issues across the repository. It addresses broken links, redundant spacing, and Table of Contents synchronization without causing data loss or broken structures.

## Applicability
Use this skill when:
- You need to ensure adherence to structural cleanliness rules in `GEMINI.md`.
- Large files or many files have been modified, risking broken internal links or inconsistent spacing.
- Headers have been changed, requiring TOC updates.

## Workflow Steps

### Step 1: Diagnostic and Audit Phase
1. **Identify Target Files**: Gather all Markdown files in the repository (e.g., `README.md`, `README.cn.md`, `parts/*.md`, `parts/*.cn.md`).
2. **Audit Issues**: You can use the included Python script `format_auditor.py` in the `scripts/` directory to scan for broken links and consecutive blank lines automatically. Run it from the root directory: `python3 .agent/skills/format_refactor/scripts/format_auditor.py`.
   - **Dead Links**: Verified by the script for file existence.
   - **Redundant Spacing**: Verified by the script for consecutive blank lines.
   - **TOC Mismatch**: Must still be checked manually by comparing file headers against the TOC in `README.md` and `README.cn.md`.
3. **Generate Diagnostic Report**: Create a Markdown artifact summarizing all findings:
   - **Broken Links Table**: File, Line, Broken Link, Reason.
   - **Formatting Violations**: File, Line Range, Issue (e.g., "3 consecutive blank lines").
   - **TOC Mismatches**: Header modified but not reflected in TOC.

### Step 2: User Review and Confirmation
1. **Present the Report**: Share the artifact with the user.
2. **Wait for Approval**: Do not proceed with modifications until the user confirms the plan or provides specific corrections.

### Step 3: Execution and Repair Phase
1. **Targeted Fixes**: Use `multi_replace_file_content` to apply changes chunk by chunk to avoid token exhaustion or file corruption.
2. **Specific Repair Rules**:
   - **Links**: Fix to correct valid paths or anchors.
   - **Spacing**: Clean up blank lines to ensure exactly one blank line between sections or paragraph blocks.
   - **TOC**: Update nested lists in TOC with strict multiples of 4 spaces indentation.
3. **Bilingual Synchronization**: Ensure changes are applied to both Chinese (`*.cn.md`) and English (`*.md`) versions.
4. **Final Verification**: Re-scan edited lines to ensure violations are resolved.
