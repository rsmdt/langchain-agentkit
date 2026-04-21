"""CoreBehaviorExtension — universal, domain-neutral agent guidance."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_agentkit.extension import Extension

if TYPE_CHECKING:
    from langgraph.prebuilt import ToolRuntime

_CORE_BEHAVIOR_BODY = (
    (Path(__file__).parent / "prompts" / "core_behavior.md").read_text(encoding="utf-8").rstrip()
)

# File-inspection tools that replace raw shell equivalents when present.
_SPECIALIZED_FS_TOOLS: frozenset[str] = frozenset({"Read", "Write", "Edit", "Glob", "Grep"})

_BASH_WITH_SPECIALIZED_APPENDIX = """
## Shell vs. dedicated tools

Dedicated file tools are available — prefer them over raw shell equivalents:

- File search: use `Glob` (not `find`, `ls`, or `fd`).
- Content search: use `Grep` (not `grep` or `rg`).
- Read files: use `Read` (not `cat`, `head`, `tail`, or `less`).
- Edit files: use `Edit` (not `sed`, `awk`, or heredoc redirection).
- Write files: use `Write` (not `echo >` or `cat <<EOF`).

Reserve `Bash` for operations with no dedicated tool — running tests,
building, git, package managers, and process control.
""".rstrip()

_BASH_ONLY_APPENDIX = """
## Shell-only environment

No dedicated file tools are registered — use `Bash` for reads, writes,
edits, and searches. Quote paths with spaces, keep output bounded, and
prefer ripgrep/find over wildcard expansion for recursive operations.
""".rstrip()


class CoreBehaviorExtension(Extension):
    """Contributes universal, domain-neutral agent guidance to the prompt.

    The guidance adapts to the composed toolset: when specialized file
    tools (``Read``, ``Grep``, etc.) are registered alongside ``Bash``,
    the prompt steers the model toward the dedicated tools; when only
    ``Bash`` is available, it pivots to shell-based conventions.
    """

    def prompt(
        self,
        state: dict[str, Any],
        runtime: ToolRuntime | None = None,
        *,
        tools: frozenset[str] = frozenset(),
    ) -> str:
        sections = [_CORE_BEHAVIOR_BODY]
        has_bash = "Bash" in tools
        has_specialized = bool(tools & _SPECIALIZED_FS_TOOLS)
        if has_bash and has_specialized:
            sections.append(_BASH_WITH_SPECIALIZED_APPENDIX)
        elif has_bash and not has_specialized:
            sections.append(_BASH_ONLY_APPENDIX)
        return "\n\n".join(sections)
