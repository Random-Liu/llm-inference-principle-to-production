# Agent Instructions

This file contains instructions and conventions for AI agents working on this repository.

## 1. Writing Style & Tone (Core Rules)

### Objective and Neutral Tone
- Maintain an evidence-based, qualitative, and neutral tone. Focus on logical first principles rather than speculative performance outcomes.
- **Forbidden Terms (English)**: Avoid absolute or exaggerated terms like "blazingly fast", "infinite", "uncompromising", "absolute necessity".
- **Forbidden Terms (Chinese)**: Prohibit exaggerated metaphors (e.g., "海啸", "终极武器", "雷霆手段") and overly emotional or absolute words without verification (e.g., "消灭", "最大的不稳定因素"). Also avoid conversational meta-commentary in formal text (e.g., "简短一提"). Pedagogical analogies are permitted if not flowery.

### Spacing & Formatting
- **LaTeX Spacing**: Separate inline math formulas `$formula$` from adjacent CJK characters by a space (e.g., `文本 $formula$ 文本`).
- **Bolding Spacing**: Ensure spaces around bold markers if adjacent to CJK characters (e.g., `文本 **内容** 文本` or `文本 **“内容”** 文本`).

### English Translation Principles
- **Idiomatic + Concise**: Adopt an "idiomatic + concise" (意译+精炼) strategy. "Concise" refers only to simplifying sentences, never dropping paragraphs or details.
- **100% Mapping**: Every chapter, section, bullet point, and bolded terminology in the English version MUST be a 100% precise, 1:1 structural map of the Chinese source text.
- **Directness**: Use short sentences; avoid complex clauses, stack of "is, that, which".
- **Action-Oriented**: Use active verbs instead of abstract noun phrases.
- **Causal Mask Awareness**: Respect the causal mask in explanations of Transformer mechanics.
- **Heading Format**: Maintain consistent "Title: Subtitle" format (e.g., `### Section X: Title: Subtitle`).

## 2. Bilingual Maintenance and Translation Workflow
- **Source Language**: The user writes and updates in Chinese (`*.cn.md`).
- **Automatic Translation**: Every time a Chinese file is updated, automatically translate and update the corresponding English file (`*.md`) BEFORE committing. English is the default version.
- **Link Consistency**: `README.md` links to `*.md`; `README.cn.md` links to `*.cn.md`.
- **Verification**: Ensure technical terms are translated accurately and consistently.
- **Non-Bilingual Files**: Files without a corresponding `*.cn.md` file (e.g., `TODO.md`, `GEMINI.md`) MUST be in English only.

## 3. Agent Self-Correction
- **Self-Verification**: When generating new content or making significant modifications, perform a self-verification step before presenting or committing changes. Check against all rules in this file and explicitly confirm adherence in the response.
- **Self-Correcting Instructions**: When questioned or corrected by the USER on style violations not listed here, automatically update this file to add the instance as a new "Bad Case" in the corresponding section before completing the turn.

## 4. Document Structure & Cleanliness
- **Header Level Hierarchy**: Part Title (`#`), Chapter Title (`##`), Section Title (`###`), Sub-sections (`####`).
- **No Stray Blank Lines**: Clean up redundant, consecutive blank lines when deleting content.
- **Block-Level Spacing**: Block-level elements (lists, quotes, tables) MUST have exactly one blank line preceding and following them.
- **Structure Synchronization**: When modifying titles or headers, update the Table of Contents in both `README.md` and `README.cn.md`, and update all internal hyperlinks.
- **TOC Indentation**: Nested lists in TOC MUST be strictly indented by multiples of 4 spaces.

## 5. Mermaid Diagram Conventions
- **Use Emojis in Labels**: Add relevant emojis to node labels (e.g., 🧠 CPU, 📟 GPU, 📦 Pod).
- **Quoting for Safety**: Enclose labels in double quotes inside brackets (e.g., `NodeID["🧠 Label"]`).

## 6. Git Commit Convention
- Include co-author signature separated by a blank line from the main message:
  ```
  Co-authored-by: Gemini <gemini@google.com>
  ```
- **Review and Squash**: Review all local commits and appropriately squash related commits before pushing.
- **Pre-Push Review**: Send a summary of pending commits to the user for review before pushing. Do not push until the user confirms.
- **Detailed Message**: Include a detailed summary of changes in the commit message body in English before the signature.