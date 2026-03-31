---
name: summarize
description: Summarize files and directories into a concise report.
---

## Summarize

When asked to summarize, follow this workflow:

1. Use Glob to discover all files matching the user's scope
2. Read each file (use offset/limit for large files)
3. Produce a structured summary with:
   - **File count** and total size
   - **Key findings** — one bullet per file
   - **Recommendations** — actionable next steps

Write the summary to a markdown file unless told otherwise.
