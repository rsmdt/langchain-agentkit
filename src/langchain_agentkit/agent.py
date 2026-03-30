"""Metaclass-driven LangGraph agent node with extension support.

Usage::

    from langchain_agentkit import agent

    class researcher(agent):
        llm = ChatOpenAI(model="gpt-4o")
        tools = [web_search]
        extensions = [SkillsExtension("skills/"), TasksExtension()]
        prompt = "You are a research assistant."

        async def handler(state, *, llm, tools, prompt, runtime):
            response = await llm.ainvoke(state["messages"])
            return {"messages": [response], "sender": "researcher"}

``researcher`` is a ``StateGraph`` — call ``.compile()`` to get a
runnable graph, optionally passing a checkpointer for ``interrupt()``
support.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

from langchain_agentkit._graph_builder import _find_wrap_tool_call, build_graph
from langchain_agentkit.agent_kit import AgentKit
from langchain_agentkit.state import AgentKitState

if TYPE_CHECKING:
    from pathlib import Path

    from langchain_core.tools import BaseTool

    from langchain_agentkit.extension import Extension


def _validate_handler_signature(
    handler: Any,
    class_name: str,
    valid_params: frozenset[str],
    label: str,
) -> type:
    """Validate handler signature and extract state type.

    Returns the state type inferred from the handler's first parameter
    annotation. If no annotation is present, defaults to ``AgentKitState``.
    """
    sig = inspect.signature(handler)
    params = list(sig.parameters.values())

    if not params:
        raise ValueError(
            f"class {class_name}({label}): handler must accept at least "
            f"'state' as its first parameter"
        )

    first = params[0]
    if first.kind not in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ):
        raise ValueError(
            f"class {class_name}({label}): handler's first parameter must be "
            f"positional ('state'), got {first.kind.name}"
        )

    state_type: type = AgentKitState
    if first.annotation is not inspect.Parameter.empty:
        state_type = first.annotation

    for param in params[1:]:
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            continue
        if param.kind == inspect.Parameter.KEYWORD_ONLY:
            if param.name not in valid_params:
                raise ValueError(
                    f"class {class_name}({label}): unknown handler parameter "
                    f"'{param.name}'. Valid parameters: state, "
                    f"{', '.join(sorted(valid_params))}"
                )
        elif param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            raise ValueError(
                f"class {class_name}({label}): handler parameter '{param.name}' "
                f"must be keyword-only (after *). "
                f"Signature should be: handler(state, *, {param.name}, ...)"
            )

    return state_type

# Valid injectable parameter names for handler (besides 'state')
_INJECTABLE_PARAMS = frozenset({"llm", "tools", "prompt", "runtime"})


class _AgentMeta(type):
    """Metaclass that intercepts class body and returns a StateGraph.

    When a class inherits from ``agent``, this metaclass:

    1. Extracts ``llm``, ``tools``, ``extensions``, ``prompt``, ``handler``
       from the class body.
    2. Validates the handler signature and infers state type from annotation.
    3. Builds an ``AgentKit`` from extensions and prompt source.
    4. Merges user tools with kit tools.
    5. Builds and returns an uncompiled ``StateGraph`` with the ReAct loop.

    The result is NOT a class — it's a ``StateGraph``. Call ``.compile()``
    to get a runnable graph, optionally passing a checkpointer for
    ``interrupt()`` support.
    """

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
    ) -> Any:
        # The `agent` base class itself — create normally
        if not bases:
            return super().__new__(mcs, name, bases, namespace)

        # Subclass of agent → intercept and return StateGraph

        # Extract handler (required)
        handler = namespace.get("handler")
        if handler is None:
            raise ValueError(f"class {name}(agent) must define an async def handler(...) function")
        if not callable(handler):
            raise ValueError(
                f"class {name}(agent): handler must be callable, got {type(handler).__name__}"
            )

        # Validate handler signature and extract state type
        state_type = _validate_handler_signature(handler, name, _INJECTABLE_PARAMS, "agent")

        # Extract class attributes
        llm = namespace.get("llm")
        if llm is None:
            raise ValueError(
                f"class {name}(agent) must define an llm attribute "
                f"(e.g. llm = ChatOpenAI(model='gpt-4o'))"
            )

        # Extract tools — supports list, "inherit" sentinel, or absent (no tools)
        user_tools_raw = namespace.get("tools", [])
        tools_inherit = False
        if isinstance(user_tools_raw, str):
            if user_tools_raw != "inherit":
                raise ValueError(
                    f"class {name}(agent): tools must be a list or 'inherit', "
                    f"got string '{user_tools_raw}'"
                )
            user_tools: list[BaseTool] = []
            tools_inherit = True
        elif isinstance(user_tools_raw, (list, tuple)):
            user_tools = list(user_tools_raw)
        else:
            raise ValueError(
                f"class {name}(agent): tools must be a list or 'inherit', "
                f"got {type(user_tools_raw).__name__}"
            )

        extensions: list[Extension] = namespace.get("extensions", [])
        if not isinstance(extensions, (list, tuple)):
            raise ValueError(
                f"class {name}(agent): extensions must be a list, got {type(extensions).__name__}"
            )

        prompt_source: str | Path | list[str | Path] | None = namespace.get("prompt")
        description: str = namespace.get("description", "")

        kit = AgentKit(extensions=list(extensions), prompt=prompt_source)

        # Use composed schema from extensions unless handler explicitly annotates state
        if state_type is AgentKitState:
            state_type = kit.state_schema

        wrap_tool_call = _find_wrap_tool_call(extensions, name)

        graph = build_graph(
            name=name,
            handler=handler,
            llm=llm,
            user_tools=list(user_tools),
            kit=kit,
            state_type=state_type,
            wrap_tool_call=wrap_tool_call,
        )

        # Attach metadata — .name, .description, .tools_inherit
        graph.name = name  # type: ignore[attr-defined]
        graph.description = description  # type: ignore[attr-defined]
        graph.tools_inherit = tools_inherit  # type: ignore[attr-defined]

        return graph


class agent(metaclass=_AgentMeta):  # noqa: N801
    """Base class for extension-aware LangGraph agent nodes.

    Declare a subclass to create a ``StateGraph`` with an automatic
    ReAct loop (handler - ToolNode) and extension-composed tools and
    prompts. Call ``.compile()`` on the result to get a runnable graph.

    Example::

        from langchain_agentkit import agent

        class researcher(agent):
            llm = ChatOpenAI(model="gpt-4o")
            tools = [web_search]
            extensions = [SkillsExtension("skills/"), TasksExtension()]
            prompt = "You are a research assistant."

            async def handler(state, *, llm, tools, prompt, runtime):
                response = await llm.ainvoke(state["messages"])
                return {"messages": [response], "sender": "researcher"}

        # Compile and use standalone
        graph = researcher.compile()
        graph.invoke({"messages": [HumanMessage("...")]})

        # Compile with checkpointer for interrupt() support
        from langgraph.checkpoint.memory import InMemorySaver
        graph = researcher.compile(checkpointer=InMemorySaver())

        # Use as subgraph in a parent graph
        workflow.add_node("researcher", researcher.compile())

    Class attributes:

        llm: Required. The language model instance.
        tools: Optional. Agent-specific tools (not from extensions).
        extensions: Optional. Ordered list of Extension instances.
        prompt: Optional. System prompt template — inline string, file path,
            or list of either.

    Handler signature::

        async def handler(state, *, llm, tools, prompt, runtime): ...

    ``state`` is positional. Everything after ``*`` is keyword-only and
    injected by name — declare only what you need, in any order.

    Injectable parameters:

        llm: LLM with all tools bound (user tools + extension tools).
        tools: Complete tool list (user tools + extension tools).
        prompt: Fully composed system prompt (template + extension sections).
        runtime: ToolRuntime — unified runtime context. Use
            ``runtime.config`` for the full ``RunnableConfig``.

    Annotate ``state`` to use a custom state type::

        class MyState(TypedDict, total=False):
            messages: Annotated[list, add_messages]
            draft: dict | None

        class my_agent(agent):
            llm = ChatOpenAI(model="gpt-4o")

            async def handler(state: MyState, *, llm, prompt):
                ...

    Without an annotation, the state schema is composed automatically
    from extension ``state_schema`` properties.
    """
