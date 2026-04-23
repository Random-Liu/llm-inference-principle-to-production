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

## Writing Style Conventions for Language Versions
- **Chinese Files (`*.cn.md`)**: Follow the user's style. The user maintains the primary content, and AI agents should respect the existing tone and style when making edits.
- **Non-Chinese Files (`*.md`, primarily English)**: Must follow a **strictly concise and direct** style to avoid verbosity. Follow these principles:
  - **Principle**: Adopt an "idiomatic + concise" (意译+精简) strategy rather than word-for-word translation.
  - **Content Preservation**: Only simplify sentences and phrasing; never delete or alter technical content.
  - **Directness**: Use short sentences and avoid complex clauses (e.g., avoid stacking "is, that, which").
  - **Action-Oriented**: Use active verbs instead of abstract noun phrases (e.g., avoid "the essence of... is...").
  - **Clarity**: Remove redundant phrasing and filler words.
  - **Causal Mask Awareness**: In explanations of Transformer mechanics, ensure the language strictly respects the causal mask (i.e., queries attend to past/current tokens, not future ones).
  - **Heading Format**: Maintain a consistent "Title: Subtitle" format for section headers (e.g., `### Section X: Title: Subtitle`). Keep headings as concise as possible while maintaining structural clarity.