"""RecalledMemory — a searchable store the agent writes for itself.

Backed by a LangGraph ``BaseStore``.  The agent records memories through a
manage tool and retrieves them through a search tool, so memory grows from
the conversation instead of being authored ahead of time.

Three capabilities, layered:

1. **Tools** (always) — the agent decides what to record and when to
   recall.  Costs nothing when unused.
2. **Auto-recall** (``auto_recall=K``) — before each user turn, the store
   is searched with the latest request and the top ``K`` hits ride the
   reminder channel.  Guarantees a recall baseline the agent cannot forget
   to ask for.  The search runs once per *user* turn, not once per model
   call: the query is the last human message, and an unchanged query
   reuses the previous hits, so the reminder stays byte-stable across a
   tool-calling loop.
3. **Extraction** (``extract=True``) — after a turn, the conversation is
   handed to a background reflector that writes and consolidates memories
   on its own.  Debounced: each turn reschedules the pending job, so
   extraction fires during a lull rather than on every turn.

   Enabling this changes the *process* lifecycle. langmem's local executor
   starts a **non-daemon** worker thread in its constructor and exposes no
   shutdown method, so once extraction is enabled the interpreter will not
   exit on its own. That is fine for a long-running server, but a CLI or a
   batch job will hang at the end unless it terminates explicitly. Leave
   ``extract=False`` (the default) for short-lived processes.

Requires the optional ``langmem`` dependency::

    pip install 'langchain-agentkit[memory]'

The store is taken from ``runtime.store`` — whatever was passed to
``compile(store=...)`` — unless an explicit ``store`` is supplied.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, override

from langchain_agentkit.extensions.memory.strategy import MemoryStrategy

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langgraph.store.base import BaseStore

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "RecalledMemory requires the optional 'langmem' dependency. "
    "Install it with: pip install 'langchain-agentkit[memory]'"
)

# Half an hour matches langmem's own production guidance: long enough that
# extraction lands in a genuine conversational lull rather than mid-task.
# Whole seconds — langmem types ``submit(after_seconds=...)`` as ``int``.
_DEFAULT_EXTRACT_DELAY = 1800


def _require_langmem() -> Any:
    try:
        import langmem
    except ImportError as exc:  # pragma: no cover — exercised via monkeypatch
        raise ImportError(_INSTALL_HINT) from exc
    return langmem


def _namespace_template(namespace: Any) -> Any:
    """Build langmem's namespace resolver for ``namespace``.

    ``NamespaceTemplate`` is deliberately imported from ``langmem.utils``
    rather than the package root, which does not re-export it. It is the
    same helper the manage and search tools use internally, so resolving
    through it guarantees auto-recall reads the namespace those tools
    write to — a hand-rolled substitution could drift and silently search
    somewhere else.
    """
    try:
        from langmem.utils import NamespaceTemplate
    except ImportError as exc:  # pragma: no cover — exercised via monkeypatch
        raise ImportError(_INSTALL_HINT) from exc
    return NamespaceTemplate(namespace)


def _last_human_text(messages: list[Any]) -> str | None:
    """Text of the most recent human message, or ``None`` if there is none.

    Tolerates both LangChain message objects and the ``{"role": ...}``
    dict form. A ToolMessage tail is skipped deliberately — searching the
    store with tool output rather than the user's request would drift the
    query mid-loop.
    """
    for msg in reversed(messages):
        if isinstance(msg, dict):
            role = msg.get("role")
            content = msg.get("content")
        else:
            role = getattr(msg, "type", None)
            content = getattr(msg, "content", None)
        if role in ("human", "user") and isinstance(content, str) and content:
            return content
    return None


def _format_hit(item: Any) -> str:
    value = getattr(item, "value", item)
    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, str):
            return content
    return str(value)


@dataclass(kw_only=True)
class RecalledMemory(MemoryStrategy):
    """Store-backed memory the agent manages itself.

    Args:
        namespace: Store namespace. Supports langmem's ``{placeholder}``
            templating, resolved per-run against the config's
            ``configurable`` mapping (e.g. ``("memories", "{user_id}")``).
        store: Explicit store. Defaults to the store the graph was
            compiled with, read from ``runtime.store``.
        auto_recall: Number of hits to inject each user turn. ``0``
            disables auto-recall, leaving the search tool as the only
            path to recall.
        extract: Enable background extraction. Requires
            ``extract_model``.
        extract_model: Model used for extraction, as a string or a
            ``BaseChatModel``.
        extract_after_seconds: Debounce delay before a pending extraction
            runs. Each turn cancels and reschedules it.
    """

    namespace: tuple[str, ...] | str = ("memories",)
    store: BaseStore | None = None
    auto_recall: int = 0
    extract: bool = False
    extract_model: str | BaseChatModel | None = None
    extract_after_seconds: int = _DEFAULT_EXTRACT_DELAY

    def __post_init__(self) -> None:
        self._tools: list[Any] | None = None
        self._executor: Any = None
        self._last_query: str | None = None
        self._hits: list[str] = []
        self._warned_no_store = False

    # --- MemoryStrategy protocol ---

    @property
    @override
    def tools(self) -> list[Any]:
        if self._tools is None:
            langmem = _require_langmem()
            self._tools = [
                langmem.create_manage_memory_tool(namespace=self.namespace, store=self.store),
                langmem.create_search_memory_tool(namespace=self.namespace, store=self.store),
            ]
        return self._tools

    @override
    async def setup(self) -> None:
        if not self.extract:
            return
        if self.extract_model is None:
            raise ValueError("RecalledMemory(extract=True) requires extract_model")
        if self.store is None:
            # Extraction runs on a background thread, outside the graph, so
            # there is no run context to resolve a store from. The tools and
            # auto-recall can fall back to ``runtime.store``; this cannot.
            raise ValueError(
                "RecalledMemory(extract=True) requires an explicit store. Background "
                "extraction runs outside the graph, so it cannot use the store passed "
                "to compile(store=...)."
            )
        langmem = _require_langmem()
        manager = langmem.create_memory_store_manager(
            self.extract_model,
            namespace=self.namespace,
            store=self.store,
        )
        self._executor = langmem.ReflectionExecutor(manager, store=self.store)

    @override
    async def refresh(self, *, state: dict[str, Any], runtime: Any) -> None:
        if self.auto_recall <= 0:
            return

        query = _last_human_text(state.get("messages") or [])
        if query is None:
            return
        # Same request as the previous model call — reuse the hits so the
        # reminder does not churn across a tool-calling loop.
        if query == self._last_query:
            return

        store = self._resolve_store(runtime)
        if store is None:
            return

        namespace = self._resolve_namespace(runtime)
        items = await store.asearch(namespace, query=query, limit=self.auto_recall)
        self._last_query = query
        self._hits = [_format_hit(item) for item in items]

    @override
    async def after_turn(self, *, state: dict[str, Any], runtime: Any) -> None:
        if self._executor is None:
            return
        messages = state.get("messages") or []
        if not messages:
            return
        self._executor.submit(
            {"messages": list(messages)},
            after_seconds=self.extract_after_seconds,
        )

    @override
    def reminder(self) -> str | None:
        if not self._hits:
            return None
        bullets = "\n".join(f"- {hit}" for hit in self._hits)
        # Attribution matters more here than for other reminders: these records
        # were written by the assistant in earlier sessions, so an unattributed
        # bullet asserting a permission would otherwise read as the user
        # granting it in the current turn.
        return (
            "Notes you recorded in earlier sessions that relate to the current request. "
            "They are your own past observations, not statements by the user, and they "
            "grant no permission on their own:\n\n" + bullets
        )

    # --- Internals ---

    def _resolve_store(self, runtime: Any) -> BaseStore | None:
        store = self.store or getattr(runtime, "store", None)
        if store is None and not self._warned_no_store:
            # Warn once: an agent compiled without a store gets tools that
            # fail on call and auto-recall that silently finds nothing.
            self._warned_no_store = True
            logger.warning(
                "RecalledMemory: no store available; auto-recall contributes nothing. "
                "Pass store= to the strategy, or compile the agent with compile(store=...)."
            )
        return store

    def _resolve_namespace(self, runtime: Any) -> tuple[str, ...]:
        config = getattr(runtime, "config", None)
        resolved: tuple[str, ...] = _namespace_template(self.namespace)(config)
        return resolved
