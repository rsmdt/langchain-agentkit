# ruff: noqa: N801, N805
"""HITL extension evals — ask_user tool selection and tool approval flow.

Tests exercise:
1. **ask_user trajectory**: LLM selects ask_user for ambiguous requests
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
from typing import Any
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

from langchain_agentkit import AgentKit, FilesystemExtension, HITLExtension
from langchain_agentkit.agent import agent
from langchain_agentkit.backends import OSBackend
from langchain_agentkit.extensions.hitl import InterruptConfig
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

_MODEL = os.environ.get("AGENTKIT_EVAL_MODEL", "gpt-4o-mini")


def _get_llm():
    """Return a deterministic ChatOpenAI instance."""
    return ChatOpenAI(model=_MODEL, temperature=0)


# ------------------------------------------------------------------
# Mock interrupt handler — auto-selects the first option
# ------------------------------------------------------------------


def _mock_interrupt_handler(payload: dict[str, Any]) -> dict[str, Any]:
    """Auto-select the first option for each question in an interrupt payload."""
    answers: dict[str, str] = {}
    for q in payload.get("questions", []):
        if q.get("options"):
            answers[q["question"]] = q["options"][0]["label"]
    return {"answers": answers}


# ------------------------------------------------------------------
# Agent builders
# ------------------------------------------------------------------


def _build_ask_user_agent():
    """Build agent with ask_user + filesystem tools for trajectory evals."""
    tmpdir = tempfile.mkdtemp(prefix="eval_hitl_")
    workspace = Path(tmpdir) / "workspace"
    workspace.mkdir()
    (workspace / "config.json").write_text('{"debug": true, "port": 3000}')
    (workspace / "app.py").write_text("print('hello')")

    import asyncio

    kit = AgentKit(
        extensions=[
            HITLExtension(tools=True),
            FilesystemExtension(backend=OSBackend(root=tmpdir)),
        ]
    )
    from langchain_agentkit.agent_kit import run_extension_setup

    asyncio.run(run_extension_setup(kit))

    llm = _get_llm()

    prompt_text = (
        "You are a helpful project assistant with filesystem tools and "
        "the ask_user tool.\n\n"
        "RULES:\n"
        "- When a request is ambiguous or involves choosing between "
        "multiple valid approaches, you MUST use the ask_user tool to "
        "present 2-4 options and let the user decide. Do NOT make "
        "assumptions about user preferences.\n"
        "- For clear, specific requests (read a file, search for files, "
        "write specific content), act directly without asking.\n"
        "- Each ask_user question needs a short header (max 12 chars), "
        "a question string, and 2-4 options with labels and descriptions.\n"
    )

    async def agent_node(state: dict) -> dict:
        bound = llm.bind_tools(kit.tools)
        system = SystemMessage(content=prompt_text + kit.compose(state).joined)
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


def _build_tool_approval_agent():
    """Build agent with HITL tool approval for interrupt/resume evals.

    Uses the ``agent`` metaclass so the wrap_tool hook is properly wired
    into the ToolNode via the HookRunner.
    """
    from langchain_core.tools import tool

    @tool
    def write_file(path: str, content: str) -> str:
        """Write content to a file at the given path."""
        return f"Successfully wrote to {path}"

    _llm = _get_llm()

    class approval_agent(agent):
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

    return approval_agent.compile(checkpointer=InMemorySaver())


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
# ask_user trajectory evals
# ------------------------------------------------------------------


class TestAskUserSelectionEval:
    """LLM should use ask_user for ambiguous requests with multiple approaches."""

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
    """LLM should act directly without ask_user for clear, specific requests."""

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
            # Verify ask_user was NOT called for clear requests
            ask_user_calls = [tc for tc in r["actual_tool_calls"] if tc["name"] == "ask_user"]
            assert not ask_user_calls, (
                f"ask_user should not be called for clear requests, "
                f"but was called: {ask_user_calls}"
            )


# ------------------------------------------------------------------
# Tool approval interrupt/resume evals
# ------------------------------------------------------------------


class TestToolApprovalApproveFlow:
    """Full interrupt → approve → execute cycle with real LLM."""

    @pytest.mark.asyncio
    async def test_approve_allows_tool_execution(self):
        """When user approves, the tool should execute and return a result."""
        graph = _build_tool_approval_agent()
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
            Command(resume={"answers": {"Allow writing to the file?": "Approve"}}),
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
        graph = _build_tool_approval_agent()
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
            Command(
                resume={
                    "answers": {"Allow writing to the file?": "Reject"},
                    "message": "Not allowed",
                },
            ),
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
