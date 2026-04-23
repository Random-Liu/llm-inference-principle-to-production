# Agent Instructions

This file contains instructions and conventions for AI agents working on this repository.

## Git Commit Convention
- All commits made by AI agents should include the following co-author signature in the commit message body (separated by a blank line from the main message):
  ```
  Co-authored-by: Gemini <gemini@google.com>
  ```
- **Review and Squash Before Push**: Before pushing, the AI agent MUST review all local commits. Appropriately squash related commits together to maintain a clean history.
- **Pre-Push Review Summary**: Before pushing, the AI agent MUST send a summary of the pending commits to the user for review. Do not push until the user confirms.
- **Detailed Commit Message**: AI agents SHOULD include a detailed summary of the changes in the commit message body (in English), listing the main modifications, before the co-author signature.

## Bilingual Maintenance and Translation Workflow
- **Source Language**: The user will continue writing and updating the book in Chinese, using files marked as `*.cn.md` (e.g., `README.cn.md`, `parts/part1_principles.cn.md`).
- **Automatic Translation**: Every time the user updates a Chinese file (`*.cn.md`), the AI agent MUST automatically translate the updates and update the corresponding English file (`*.md`) BEFORE committing the changes. The English version serves as the default version.
- **Link Consistency**:
  - `README.md` (English) must link to English files (`*.md`).
  - `README.cn.md` (Chinese) must link to Chinese files (`*.cn.md`).
- **Verification**: Ensure that technical terms are translated accurately and consistently across both versions.

## Writing Style Conventions for Language Versions
- **Chinese Files (`*.cn.md`)**: Follow the user's style. The user maintains the primary content, and AI agents should respect the existing tone and style when making edits.
  - **LaTeX Spacing**: Ensure all inline math formulas `$formula$` are separated from adjacent CJK characters or Chinese punctuation by a space (e.g., `文本 $formula$ 文本` or `公式： $formula$`) to prevent GitHub from failing to render the LaTeX.
- **Non-Chinese Files (`*.md`, primarily English)**: Must follow a **strictly concise and direct** style to avoid verbosity. Follow these principles:
  - **Principle**: Adopt an "idiomatic + concise" (意译+精简) strategy rather than word-for-word translation.
  - **Content Preservation**: Only simplify sentences and phrasing; never delete or alter technical content.
  - **Directness**: Use short sentences and avoid complex clauses (e.g., avoid stacking "is, that, which").
  - **Action-Oriented**: Use active verbs instead of abstract noun phrases (e.g., avoid "the essence of... is...").
  - **Clarity**: Remove redundant phrasing and filler words.
  - **Causal Mask Awareness**: In explanations of Transformer mechanics, ensure the language strictly respects the causal mask (i.e., queries attend to past/current tokens, not future ones).
  - **Heading Format**: Maintain a consistent "Title: Subtitle" format for section headers (e.g., `### Section X: Title: Subtitle`). Keep headings as concise as possible while maintaining structural clarity.

## Mermaid Diagram Conventions
- **Use Emojis in Labels**: Add relevant emojis to node labels to make diagrams more engaging and visually clear (e.g., 🧠 for CPU, 📟 for GPU, 📦 for Pod, 🚀 for high-speed links).
- **Quoting for Safety**: When using emojis or spaces in labels, always enclose the label in double quotes inside the brackets (e.g., `NodeID["🧠 Label"]`) to prevent rendering syntax errors.