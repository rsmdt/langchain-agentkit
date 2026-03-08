"""Metaclass-driven LangGraph agent node with middleware support.

Usage::

    from langchain_agentkit import agent

    class researcher(agent):
        llm = ChatOpenAI(model="gpt-4o")
        tools = [web_search]
        middleware = [SkillsMiddleware("skills/"), TasksMiddleware()]
        prompt = "You are a research assistant."

        async def handler(state, *, llm, tools, prompt, config, runtime):
            response = await llm.ainvoke(state["messages"])
            return {"messages": [response], "sender": "researcher"}

``researcher`` is a ``StateGraph`` — call ``.compile()`` to get a
runnable graph, optionally passing a checkpointer for ``interrupt()``
support.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from langchain_agentkit._handler_validation import validate_handler_signature
from langchain_agentkit.agent_kit import AgentKit
from langchain_agentkit.state import AgentState

if TYPE_CHECKING:
    from pathlib import Path

    from langchain_core.language_models import BaseChatModel
    from langchain_core.runnables import RunnableConfig
    from langchain_core.tools import BaseTool

    from langchain_agentkit.middleware import Middleware

# Valid injectable parameter names for handler (besides 'state')
_INJECTABLE_PARAMS = frozenset({"llm", "tools", "prompt", "config", "runtime"})


def _validate_handler_signature(handler: Any, class_name: str) -> tuple[set[str], type]:
    """Validate handler signature, extract injectables and state type.

    Delegates to the shared implementation in ``_handler_validation``.
    """
    return validate_handler_signature(handler, class_name, _INJECTABLE_PARAMS, "agent")


def _build_inject(
    injectable: set[str],
    bound_llm: Any,
    all_tools: list[Any],
    composed_prompt: str,
    config: RunnableConfig,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Build the injection dict for the handler based on requested params."""
    inject: dict[str, Any] = {}
    if "llm" in injectable:
        inject["llm"] = bound_llm
    if "tools" in injectable:
        inject["tools"] = list(all_tools)
    if "prompt" in injectable:
        inject["prompt"] = composed_prompt
    if "config" in injectable:
        inject["config"] = config
    if "runtime" in injectable:
        inject["runtime"] = kwargs.get("runtime")
    return inject


def _build_graph(
    name: str,
    handler: Any,
    llm: BaseChatModel,
    user_tools: list[BaseTool],
    kit: AgentKit,
    all_tools: list[BaseTool],
    injectable: set[str],
    state_type: type = AgentState,
) -> Any:
    """Build the ReAct subgraph.

    Returns an uncompiled ``StateGraph``. Call ``.compile()`` on the
    result to get a runnable graph, optionally passing a checkpointer
    for ``interrupt()`` support.
    """
    node_name = name

    async def _agent_node(
        state: dict[str, Any], config: RunnableConfig, **kwargs: Any
    ) -> dict[str, Any]:
        composed_prompt = kit.prompt(state, config)
        bound_llm = llm.bind_tools(all_tools) if all_tools else llm
        inject = _build_inject(injectable, bound_llm, all_tools, composed_prompt, config, kwargs)

        result = handler(state, **inject)
        if inspect.isawaitable(result):
            result = await result

        return result  # type: ignore[no-any-return]

    _agent_node.__name__ = node_name
    _agent_node.__qualname__ = f"agent.<locals>.{node_name}"

    workflow: StateGraph[Any] = StateGraph(state_type)
    workflow.add_node(node_name, _agent_node)

    if all_tools:
        tool_node = ToolNode(all_tools)
        workflow.add_node("tools", tool_node)

        def _should_continue(state: dict[str, Any]) -> str:
            last = state["messages"][-1]
            if hasattr(last, "tool_calls") and last.tool_calls:
                return "tools"
            return END

        workflow.set_entry_point(node_name)
        workflow.add_conditional_edges(
            node_name,
            _should_continue,
            {"tools": "tools", END: END},
        )
        workflow.add_edge("tools", node_name)
    else:
        workflow.set_entry_point(node_name)
        workflow.add_edge(node_name, END)

    return workflow


class _AgentMeta(type):
    """Metaclass that intercepts class body and returns a StateGraph.

    When a class inherits from ``agent``, this metaclass:

    1. Extracts ``llm``, ``tools``, ``middleware``, ``prompt``, ``handler``
       from the class body.
    2. Validates the handler signature and infers state type from annotation.
    3. Builds an ``AgentKit`` from middleware and prompt source.
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
        injectable, state_type = _validate_handler_signature(handler, name)

        # Extract class attributes
        llm = namespace.get("llm")
        if llm is None:
            raise ValueError(
                f"class {name}(agent) must define an llm attribute "
                f"(e.g. llm = ChatOpenAI(model='gpt-4o'))"
            )

        user_tools: list[BaseTool] = namespace.get("tools", [])
        if not isinstance(user_tools, (list, tuple)):
            raise ValueError(
                f"class {name}(agent): tools must be a list, got {type(user_tools).__name__}"
            )

        middleware: list[Middleware] = namespace.get("middleware", [])
        if not isinstance(middleware, (list, tuple)):
            raise ValueError(
                f"class {name}(agent): middleware must be a list, got {type(middleware).__name__}"
            )

        prompt_source: str | Path | list[str | Path] | None = namespace.get("prompt")

        kit = AgentKit(list(middleware), prompt=prompt_source)
        all_tools: list[BaseTool] = list(user_tools) + kit.tools

        return _build_graph(
            name=name,
            handler=handler,
            llm=llm,
            user_tools=list(user_tools),
            kit=kit,
            all_tools=all_tools,
            injectable=injectable,
            state_type=state_type,
        )


class agent(metaclass=_AgentMeta):  # noqa: N801
    """Base class for middleware-aware LangGraph agent nodes.

    Declare a subclass to create a ``StateGraph`` with an automatic
    ReAct loop (handler ⇄ ToolNode) and middleware-composed tools and
    prompts. Call ``.compile()`` on the result to get a runnable graph.

    Example::

        from langchain_agentkit import agent

        class researcher(agent):
            llm = ChatOpenAI(model="gpt-4o")
            tools = [web_search]
            middleware = [SkillsMiddleware("skills/"), TasksMiddleware()]
            prompt = "You are a research assistant."

            async def handler(state, *, llm, tools, prompt, config, runtime):
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
        tools: Optional. Agent-specific tools (not from middleware).
        middleware: Optional. Ordered list of Middleware instances.
        prompt: Optional. System prompt template — inline string, file path,
            or list of either.

    Handler signature::

        async def handler(state, *, llm, tools, prompt, config, runtime): ...

    ``state`` is positional. Everything after ``*`` is keyword-only and
    injected by name — declare only what you need, in any order.

    Injectable parameters:

        llm: LLM with all tools bound (user tools + middleware tools).
        tools: Complete tool list (user tools + middleware tools).
        prompt: Fully composed system prompt (template + middleware sections).
        config: LangGraph RunnableConfig for the current invocation.
        runtime: LangGraph runtime context.

    Annotate ``state`` to use a custom state type::

        class MyState(TypedDict, total=False):
            messages: Annotated[list, add_messages]
            draft: dict | None

        class my_agent(agent):
            llm = ChatOpenAI(model="gpt-4o")

            async def handler(state: MyState, *, llm, prompt):
                ...

    Without an annotation, ``AgentState`` is used by default.
    """
