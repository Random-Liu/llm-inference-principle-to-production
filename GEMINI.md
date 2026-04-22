# Agent Instructions

This file contains instructions and conventions for AI agents working on this repository.

## Git Commit Convention
- All commits made by AI agents should include the following co-author signature in the commit message body (separated by a blank line from the main message):
  ```
  Co-authored-by: Gemini <gemini@google.com>
  ```

## Bilingual Maintenance and Translation Workflow
- **Source Language**: The user will continue writing and updating the book in Chinese, using files marked as `*.cn.md` (e.g., `README.cn.md`, `parts/part1_principles.cn.md`).
- **Automatic Translation**: Every time the user updates a Chinese file (`*.cn.md`), the AI agent MUST automatically translate the updates and update the corresponding English file (`*.md`) BEFORE committing the changes. The English version serves as the default version.
- **Link Consistency**:
  - `README.md` (English) must link to English files (`*.md`).
  - `README.cn.md` (Chinese) must link to Chinese files (`*.cn.md`).
- **Verification**: Ensure that technical terms are translated accurately and consistently across both versions.