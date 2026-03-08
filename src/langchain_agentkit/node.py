"""Metaclass-driven LangGraph node with skill support.

Usage::

    from langchain_agentkit import node

    class researcher(node):
        llm = ChatOpenAI(model="gpt-4o")
        tools = [web_search, calculate]
        skills = "skills/"

        async def handler(state, *, llm, tools, runtime):
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
from langchain_agentkit.runtime import ToolRuntime
from langchain_agentkit.skill_registry import SkillRegistry
from langchain_agentkit.state import AgentState

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.runnables import RunnableConfig
    from langchain_core.tools import BaseTool

# Valid injectable parameter names for handler (besides 'state')
_INJECTABLE_PARAMS = frozenset({"llm", "tools", "runtime"})


def _validate_handler_signature(handler: Any, class_name: str) -> tuple[set[str], type]:
    """Validate handler signature, extract injectables and state type.

    Delegates to the shared implementation in ``_handler_validation``.
    """
    return validate_handler_signature(handler, class_name, _INJECTABLE_PARAMS, "node")


def _normalize_skills(
    skills: str | list[str] | SkillRegistry | None,
) -> SkillRegistry | None:
    """Normalize the skills parameter into a SkillRegistry instance or None."""
    if skills is None:
        return None
    if isinstance(skills, SkillRegistry):
        return skills
    if isinstance(skills, str):
        return SkillRegistry(skills)
    if isinstance(skills, list):
        return SkillRegistry(skills)
    raise TypeError(
        f"skills must be str, list[str], SkillRegistry, or None, got {type(skills).__name__}"
    )


def _build_inject(
    injectable: set[str],
    bound_llm: Any,
    all_tools: list[Any],
    runtime: ToolRuntime,
) -> dict[str, Any]:
    """Build the injection dict for the handler based on requested params."""
    inject: dict[str, Any] = {}
    if "llm" in injectable:
        inject["llm"] = bound_llm
    if "tools" in injectable:
        inject["tools"] = list(all_tools)
    if "runtime" in injectable:
        inject["runtime"] = runtime
    return inject


def _build_graph(
    name: str,
    handler: Any,
    llm: BaseChatModel,
    user_tools: list[BaseTool],
    skill_registry: SkillRegistry | None,
    injectable: set[str],
    state_type: type = AgentState,
) -> Any:
    """Build the ReAct subgraph.

    Returns an uncompiled ``StateGraph``. Call ``.compile()`` on the
    result to get a runnable graph, optionally passing a checkpointer
    for ``interrupt()`` support.
    """
    # Build complete tool list
    skill_tools: list[BaseTool] = []
    if skill_registry is not None:
        skill_tools = skill_registry.tools

    all_tools = list(user_tools) + skill_tools
    node_name = name

    # Build the handler wrapper as a LangGraph node
    async def _agent_node(
        state: dict[str, Any], config: RunnableConfig, **kwargs: Any
    ) -> dict[str, Any]:
        runtime = ToolRuntime(config, **kwargs)
        bound_llm = llm.bind_tools(all_tools) if all_tools else llm
        inject = _build_inject(injectable, bound_llm, all_tools, runtime)

        result = handler(state, **inject)
        if inspect.isawaitable(result):
            result = await result

        return result  # type: ignore[no-any-return]

    _agent_node.__name__ = node_name
    _agent_node.__qualname__ = f"node.<locals>.{node_name}"

    # Build the graph
    workflow: StateGraph[Any] = StateGraph(state_type)
    workflow.add_node(node_name, _agent_node)  # type: ignore[type-var]

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


class _NodeMeta(type):
    """Metaclass that intercepts class body and returns a StateGraph.

    When a class inherits from ``node``, this metaclass:

    1. Extracts ``llm``, ``tools``, ``skills``, ``handler`` from the class body.
    2. Validates the handler signature and infers state type from annotation.
    3. Builds and returns an uncompiled ``StateGraph`` with the ReAct loop.

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
        # The `node` base class itself — create normally
        if not bases:
            return super().__new__(mcs, name, bases, namespace)

        # Subclass of node → intercept and return StateGraph

        # Extract handler (required)
        handler = namespace.get("handler")
        if handler is None:
            raise ValueError(f"class {name}(node) must define an async def handler(...) function")
        if not callable(handler):
            raise ValueError(
                f"class {name}(node): handler must be callable, got {type(handler).__name__}"
            )

        # Validate handler signature and extract state type
        injectable, state_type = _validate_handler_signature(handler, name)

        # Extract class attributes
        llm = namespace.get("llm")
        if llm is None:
            raise ValueError(
                f"class {name}(node) must define an llm attribute "
                f"(e.g. llm = ChatOpenAI(model='gpt-4o'))"
            )

        user_tools = namespace.get("tools", [])
        if not isinstance(user_tools, (list, tuple)):
            raise ValueError(
                f"class {name}(node): tools must be a list, got {type(user_tools).__name__}"
            )

        skills_raw = namespace.get("skills")
        skill_registry = _normalize_skills(skills_raw)

        return _build_graph(
            name=name,
            handler=handler,
            llm=llm,
            user_tools=list(user_tools),
            skill_registry=skill_registry,
            injectable=injectable,
            state_type=state_type,
        )


class node(metaclass=_NodeMeta):  # noqa: N801
    """Base class for skill-aware LangGraph agent nodes.

    Declare a subclass to create a ``StateGraph`` with an automatic
    ReAct loop (handler ⇄ ToolNode). Call ``.compile()`` on the result
    to get a runnable graph.

    Example::

        from langchain_agentkit import node

        class researcher(node):
            llm = ChatOpenAI(model="gpt-4o")
            tools = [web_search, calculate]
            skills = "skills/"

            async def handler(state, *, llm, tools, runtime):
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
        tools: Optional. List of LangChain tools available to the agent.
        skills: Optional. Path(s) to skill directories or a SkillRegistry instance.

    Handler signature::

        async def handler(state, *, llm, tools, runtime): ...

    ``state`` is positional. Everything after ``*`` is keyword-only and
    injected by name — declare only what you need, in any order.

    Annotate ``state`` to use a custom state type::

        class MyState(TypedDict, total=False):
            messages: Annotated[list, add_messages]
            draft: dict | None

        class my_agent(node):
            llm = ChatOpenAI(model="gpt-4o")

            async def handler(state: MyState, *, llm):
                ...

    Without an annotation, ``AgentState`` is used by default.
    """
