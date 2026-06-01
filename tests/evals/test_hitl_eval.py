# ruff: noqa: N801, N805
"""HITL extension evals — AskUser tool selection and tool approval flow.

Tests exercise:
1. **AskUser trajectory**: LLM selects AskUser for ambiguous requests
2. **Direct action trajectory**: LLM acts directly for clear requests
3. **Tool approval flow**: Full interrupt → resume cycle with checkpointer

Requires:
- A valid OPENAI_API_KEY in the environment (loaded from .env by conftest)
- The ``langchain-openai`` package

Run::

    uv run pytest tests/evals/test_hitl_eval.py -x -v -m eval
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Annotated, Any, TypedDict
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

from langchain_agentkit import (
    Agent,
    AgentKit,
    CountStrategy,
    FilesystemExtension,
    HistoryExtension,
    HITLExtension,
    ResilienceExtension,
)
from langchain_agentkit.backends.os import OSBackend
from langchain_agentkit.extensions.hitl import InterruptConfig
from tests.evals.conftest import EVAL_MODEL
from tests.evals.datasets import (
    HITL_ASK_USER_DATASET,
    HITL_DIRECT_ACTION_DATASET,
)
from tests.evals.eval_runner import run_eval

pytestmark = [
    pytest.mark.eval,
    pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set",
    ),
]

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None  # type: ignore[assignment,misc]

_MODEL = EVAL_MODEL


class _SupervisorState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def _get_llm():
    """Return a deterministic ChatOpenAI instance."""
    return ChatOpenAI(model=_MODEL, temperature=0)


# ------------------------------------------------------------------
# Mock interrupt handler — auto-selects the first option
# ------------------------------------------------------------------


def _mock_interrupt_handler(payload: dict[str, Any]) -> dict[str, Any]:
    """Auto-select the first option for each question (index-keyed answers)."""
    answers: dict[str, str] = {}
    for i, q in enumerate(payload.get("questions", [])):
        if q.get("options"):
            answers[str(i)] = q["options"][0]["label"]
    return {"answers": answers}


# ------------------------------------------------------------------
# Agent builders
# ------------------------------------------------------------------


def _build_ask_user_agent():
    """Build agent with AskUser + filesystem tools for trajectory evals."""
    tmpdir = tempfile.mkdtemp(prefix="eval_hitl_")
    workspace = Path(tmpdir) / "workspace"
    workspace.mkdir()
    (workspace / "config.json").write_text('{"debug": true, "port": 3000}')
    (workspace / "app.py").write_text("print('hello')")

    import asyncio

    kit = AgentKit(
        extensions=[
            HITLExtension(),
            FilesystemExtension(backend=OSBackend(root=tmpdir)),
        ]
    )
    from langchain_agentkit.agent_kit import run_extension_setup

    asyncio.run(run_extension_setup(kit))

    llm = _get_llm()

    prompt_text = (
        "You are a helpful project assistant with filesystem tools and "
        "the AskUser tool.\n\n"
        "RULES:\n"
        "- When a request is ambiguous or involves choosing between "
        "multiple valid approaches, you MUST use the AskUser tool to "
        "present 2-4 options and let the user decide. Do NOT make "
        "assumptions about user preferences.\n"
        "- For clear, specific requests (read a file, search for files, "
        "write specific content), act directly without asking.\n"
        "- Each AskUser question needs a short header (max 12 chars), "
        "a question string, and 2-4 options with labels and descriptions.\n"
    )

    async def agent_node(state: dict) -> dict:
        bound = llm.bind_tools(kit.tools)
        system = SystemMessage(content=prompt_text + kit.compose(state).prompt)
        msgs = [system] + state["messages"]
        return {"messages": [await bound.ainvoke(msgs)]}

    def should_continue(state: dict) -> str:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(kit.state_schema)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(kit.tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "agent")

    return graph.compile()


async def _build_tool_approval_agent():
    """Build agent with HITL tool approval for interrupt/resume evals.

    Uses the ``Agent`` base class so the wrap_tool hook is properly wired
    into the ToolNode via the HookRunner.
    """
    from langchain_core.tools import tool

    @tool
    def write_file(path: str, content: str) -> str:
        """Write content to a file at the given path."""
        return f"Successfully wrote to {path}"

    _llm = _get_llm()

    class ApprovalAgent(Agent):
        model = _llm
        tools = [write_file]
        extensions = [
            HITLExtension(
                interrupt_on={
                    "write_file": InterruptConfig(
                        options=["approve", "reject"],
                        question="Allow writing to the file?",
                    ),
                },
            ),
        ]
        prompt = (
            "You are a helpful assistant. When asked to write content "
            "to a file, use the write_file tool."
        )

        async def handler(state, *, llm, tools, prompt, runtime):
            bound = llm.bind_tools(tools)
            messages = [SystemMessage(content=prompt)] + state["messages"]
            response = await bound.ainvoke(messages)
            return {"messages": [response]}

    return await ApprovalAgent().compile(checkpointer=InMemorySaver())


async def _build_supervisor_with_agentkit_subgraph():
    """Embed a real-LLM AgentKit approval agent as a subgraph of a supervisor.

    The AgentKit graph is compiled WITHOUT a checkpointer so it inherits the
    supervisor's. Proves the interrupt bubbles up to the supervisor and that
    ``Command(resume=...)`` on the supervisor routes back down into the
    subgraph's paused ``interrupt()``.
    """
    from langchain_core.tools import tool

    @tool
    def write_file(path: str, content: str) -> str:
        """Write content to a file at the given path."""
        return f"Successfully wrote to {path}"

    _llm = _get_llm()

    class ApprovalAgent(Agent):
        model = _llm
        tools = [write_file]
        extensions = [
            HITLExtension(
                interrupt_on={
                    "write_file": InterruptConfig(
                        options=["approve", "reject"],
                        question="Allow writing to the file?",
                    ),
                },
            ),
        ]
        prompt = (
            "You are a helpful assistant. When asked to write content "
            "to a file, use the write_file tool."
        )

        async def handler(state, *, llm, tools, prompt, runtime):
            bound = llm.bind_tools(tools)
            messages = [SystemMessage(content=prompt)] + state["messages"]
            return {"messages": [await bound.ainvoke(messages)]}

    subgraph = await ApprovalAgent().compile()  # no checkpointer — inherits parent's

    supervisor = StateGraph(_SupervisorState)
    supervisor.add_node("worker", subgraph)
    supervisor.add_edge(START, "worker")
    supervisor.add_edge("worker", END)
    return supervisor.compile(checkpointer=InMemorySaver())


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _fail_msg(r: dict) -> str:
    return (
        f"FAILED: {r['description']}\n"
        f"  Comment: {r['comment']}\n"
        f"  Expected: {r['expected_tool_calls']}\n"
        f"  Actual: {r['actual_tool_calls']}"
    )


# ------------------------------------------------------------------
# AskUser trajectory evals
# ------------------------------------------------------------------


class TestAskUserSelectionEval:
    """LLM should use AskUser for ambiguous requests with multiple approaches."""

    @pytest.fixture(autouse=True)
    def _agent(self):
        self.agent = _build_ask_user_agent()

    @pytest.mark.parametrize(
        "entry",
        HITL_ASK_USER_DATASET,
        ids=lambda e: e["description"],
    )
    def test_uses_ask_user(self, entry):
        with patch(
            "langchain_agentkit.extensions.hitl.tools.ask_user.interrupt",
            side_effect=_mock_interrupt_handler,
        ):
            results = run_eval(self.agent, [entry])
        for r in results:
            assert r["score"], _fail_msg(r)


class TestDirectActionEval:
    """LLM should act directly without AskUser for clear, specific requests."""

    @pytest.fixture(autouse=True)
    def _agent(self):
        self.agent = _build_ask_user_agent()

    @pytest.mark.parametrize(
        "entry",
        HITL_DIRECT_ACTION_DATASET,
        ids=lambda e: e["description"],
    )
    def test_acts_directly(self, entry):
        with patch(
            "langchain_agentkit.extensions.hitl.tools.ask_user.interrupt",
            side_effect=_mock_interrupt_handler,
        ):
            results = run_eval(self.agent, [entry])
        for r in results:
            assert r["score"], _fail_msg(r)
            # Verify AskUser was NOT called for clear requests
            ask_user_calls = [tc for tc in r["actual_tool_calls"] if tc["name"] == "AskUser"]
            assert not ask_user_calls, (
                f"AskUser should not be called for clear requests, but was called: {ask_user_calls}"
            )


# ------------------------------------------------------------------
# Tool approval interrupt/resume evals
# ------------------------------------------------------------------


class TestToolApprovalApproveFlow:
    """Full interrupt → approve → execute cycle with real LLM."""

    @pytest.mark.asyncio
    async def test_approve_allows_tool_execution(self):
        """When user approves, the tool should execute and return a result."""
        graph = await _build_tool_approval_agent()
        config = {"configurable": {"thread_id": "approve-test"}}

        # Step 1: Invoke — LLM should call write_file, HITL interrupts
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="Write 'hello world' to /tmp/test.txt")]},
            config,
        )

        # Verify graph paused (has AIMessage with tool_calls but no ToolMessage)
        messages = result["messages"]
        ai_messages = [m for m in messages if isinstance(m, AIMessage) and m.tool_calls]
        assert ai_messages, "LLM should have made a write_file tool call"
        tool_name = ai_messages[-1].tool_calls[0]["name"]
        assert tool_name == "write_file", f"Expected write_file, got {tool_name}"

        # Step 2: Resume with approve
        result = await graph.ainvoke(
            Command(resume={"answers": {"0": "Approve"}}),
            config,
        )

        # Verify tool executed — should have a ToolMessage with success content
        messages = result["messages"]
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        assert tool_messages, "Tool should have executed after approval"
        assert any("Successfully wrote" in m.content for m in tool_messages), (
            f"Expected success message, got: {[m.content for m in tool_messages]}"
        )


class TestToolApprovalRejectFlow:
    """Full interrupt → reject → error message cycle with real LLM."""

    @pytest.mark.asyncio
    async def test_reject_prevents_tool_execution(self):
        """When user rejects, the tool should NOT execute."""
        graph = await _build_tool_approval_agent()
        config = {"configurable": {"thread_id": "reject-test"}}

        # Step 1: Invoke — triggers interrupt
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="Write 'hello world' to /tmp/test.txt")]},
            config,
        )

        # Verify graph paused with write_file call
        messages = result["messages"]
        ai_messages = [m for m in messages if isinstance(m, AIMessage) and m.tool_calls]
        assert ai_messages, "LLM should have made a write_file tool call"

        # Step 2: Resume with reject
        result = await graph.ainvoke(
            Command(resume={"answers": {"0": "Reject"}}),
            config,
        )

        # Verify tool was rejected — ToolMessage with error status
        messages = result["messages"]
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        assert tool_messages, "Should have a ToolMessage with rejection"
        reject_msgs = [m for m in tool_messages if m.status == "error"]
        assert reject_msgs, (
            f"Should have error-status ToolMessage, got: "
            f"{[(m.content, m.status) for m in tool_messages]}"
        )
        # Tool should NOT have "Successfully wrote" in any message
        assert not any("Successfully wrote" in m.content for m in tool_messages), (
            "Tool should not have executed after rejection"
        )


class TestSubgraphApprovalFlow:
    """AgentKit-as-subgraph: interrupt bubbles to the supervisor, resume routes down."""

    @pytest.mark.asyncio
    async def test_subgraph_interrupt_bubbles_and_resume_executes(self):
        graph = await _build_supervisor_with_agentkit_subgraph()
        config = {"configurable": {"thread_id": "subgraph-approve"}}

        # Step 1: invoke the supervisor — the subgraph's HITL interrupt must
        # surface on the supervisor's state.
        await graph.ainvoke(
            {"messages": [HumanMessage(content="Write 'hello world' to /tmp/test.txt")]},
            config,
        )
        state = await graph.aget_state(config)
        assert state.interrupts, "Subgraph interrupt should bubble up to the supervisor"
        payload = state.interrupts[0].value
        assert payload["questions"][0]["context"]["tool"] == "write_file"

        # Step 2: resume on the supervisor — routes back down into the subgraph.
        result = await graph.ainvoke(
            Command(resume={"answers": {"0": "Approve"}}),
            config,
        )
        tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert any("Successfully wrote" in m.content for m in tool_messages), (
            f"Tool should execute after approval, got: {[m.content for m in tool_messages]}"
        )


async def _build_full_stack_agent(*, with_history: bool, checkpointer: bool):
    """A real-LLM AgentKit assembled with Resilience + (History) + Filesystem + HITL.

    Gates the filesystem ``Write`` tool for approval. ``with_history`` is left off
    for the subgraph case — ``HistoryExtension`` does not yet compose with
    ``interrupt()`` resume across a subgraph boundary (see the xfail in
    ``tests/integration/test_hitl_full_stack.py``).
    """
    tmpdir = tempfile.mkdtemp(prefix="eval_fullstack_")
    _llm = _get_llm()

    extensions: list[Any] = [ResilienceExtension()]
    if with_history:
        extensions.append(HistoryExtension(strategy=CountStrategy(max_messages=100)))
    extensions.append(FilesystemExtension(backend=OSBackend(root=tmpdir)))
    extensions.append(
        HITLExtension(
            interrupt_on={
                "Write": InterruptConfig(
                    options=["approve", "reject"],
                    question="Allow writing to the file?",
                )
            },
        )
    )

    class FullStackAgent(Agent):
        model = _llm
        prompt = (
            "You are a helpful assistant with filesystem tools. When asked to "
            "write a file, use the Write tool."
        )

        async def handler(state, *, llm, tools, prompt, runtime):
            bound = llm.bind_tools(tools)
            return {
                "messages": [
                    await bound.ainvoke([SystemMessage(content=prompt)] + state["messages"])
                ]
            }

    FullStackAgent.extensions = extensions

    agent = FullStackAgent()
    if checkpointer:
        return await agent.compile(checkpointer=InMemorySaver())
    return await agent.compile()


class TestFullStackApprovalFlow:
    """Approval through a fully-assembled stack (Resilience + History + Filesystem + HITL)."""

    @pytest.mark.asyncio
    async def test_standalone_full_stack(self):
        graph = await _build_full_stack_agent(with_history=True, checkpointer=True)
        config = {"configurable": {"thread_id": "fullstack-approve"}}

        await graph.ainvoke(
            {"messages": [HumanMessage(content="Write the text 'hello world' to notes.txt")]},
            config,
        )
        state = await graph.aget_state(config)
        assert state.interrupts, "Write should be gated for approval"
        assert state.interrupts[0].value["questions"][0]["context"]["tool"] == "Write"

        result = await graph.ainvoke(Command(resume={"answers": {"0": "Approve"}}), config)
        tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert any(m.status != "error" for m in tool_messages), (
            f"Write should execute after approval, got: "
            f"{[(m.content, m.status) for m in tool_messages]}"
        )

    @pytest.mark.asyncio
    async def test_full_stack_as_subgraph(self):
        # History excluded for the subgraph case (known limitation).
        subgraph = await _build_full_stack_agent(with_history=False, checkpointer=False)
        supervisor = StateGraph(_SupervisorState)
        supervisor.add_node("worker", subgraph)
        supervisor.add_edge(START, "worker")
        supervisor.add_edge("worker", END)
        graph = supervisor.compile(checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": "fullstack-subgraph"}}

        await graph.ainvoke(
            {"messages": [HumanMessage(content="Write the text 'hello world' to notes.txt")]},
            config,
        )
        state = await graph.aget_state(config)
        assert state.interrupts, "Subgraph Write interrupt should bubble to the supervisor"

        result = await graph.ainvoke(Command(resume={"answers": {"0": "Approve"}}), config)
        tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert any(m.status != "error" for m in tool_messages), (
            "Write should execute after approval through the subgraph"
        )
