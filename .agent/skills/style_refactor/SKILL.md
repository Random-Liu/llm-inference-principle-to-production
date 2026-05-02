# Skill: Large-Scale Style Refactoring Workflow

## Purpose
This skill provides a reliable and efficient workflow for updating the writing style or refactoring large files without causing generation failures, token exhaustion, or missing complex constraints. It avoids full-file rewrites in favor of a diagnostic and targeted modification approach.

## Applicability
Use this skill when:
- A file is too large to be safely rewritten in a single turn (typically > 200 lines).
- There are complex, multi-dimensional style rules to apply (e.g., maintaining technical accuracy, specific spacing rules, and a neutral tone simultaneously).
- You need to maintain synchronization between bilingual versions.

## Workflow Steps

### Step 1: Scan and Report (Diagnostic Phase)
1. **Read and Scan in Chunks**: Use `view_file` to read the target file. For large files, you **MUST** read and scan the file in smaller chunks or by chapters (e.g., 100-200 lines per chunk) to avoid attention degradation ("Lost in the Middle").
2. **Analyze for Violations**: Scan each chunk against the specific style rules defined in `GEMINI.md` or provided in the user request. Look for non-objective language, exaggerated metaphors, passive voice, or structural issues.
3. **Intermediate Results**: You may store intermediate results for each chunk in scratch files.
4. **Generate Consolidated Report**: Consolidate all findings into a single total diagnostic report as a Markdown artifact. Do NOT modify the file yet. The report MUST include a table with the following columns:
   - **Line Number**: The specific line where the issue occurs.
   - **Current Text**: The problematic string.
   - **Issue Type**: A brief categorization (e.g., "Exaggerated Metaphor", "Subjective Evaluation").
   - **Suggested Fix**: The proposed objective and neutral replacement.
5. **Highlight Open Questions**: At the end of the report, list specific design decisions or edge cases that require user input (e.g., whether to keep certain pedagogical analogies).

### Step 2: User Review and Confirmation
1. **Present the Report**: Notify the user of the generated report and highlight the key open questions.
2. **Wait for Feedback**: Solicit user input on the suggestions and choices. Do not proceed until the user confirms the plan or specifies adjustments.

### Step 3: Targeted Fix (Execution Phase)
1. **Apply Non-Contiguous Edits**: Use the `multi_replace_file_content` tool to apply the modifications confirmed by the user in a single call. This avoids full rewrites and saves tokens.
2. **Verify Matches**: Ensure the `TargetContent` in each ReplacementChunk matches the file content exactly, including whitespace.
3. **Synchronize Translations**: If the file has a corresponding bilingual version (e.g., translating `*.cn.md` to `*.md`), apply the same semantic changes to the other file, maintaining the style rules specific to that language (e.g., conciseness for English).
4. **Update TOC**: If headers or titles were modified, review and update the global Table of Contents in both `README.md` and `README.cn.md` to ensure consistency.
