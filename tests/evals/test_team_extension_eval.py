# ruff: noqa: N801, N805
"""Real-LLM integration evals for TeamExtension coordination.

Tests exercise the FULL compiled graph flow: lead agent receives a message,
uses team coordination tools (TeamCreate, TeamMessage, TeamStatus,
TeamDissolve) via LangGraph's ReAct loop, teammates process with real LLM
calls as asyncio.Tasks, and results propagate back through the lead.

Requires:
- A valid OPENAI_API_KEY in the environment (loaded from .env by conftest)
- The ``langchain-openai`` package

Run::

    uv run pytest tests/evals/test_team_extension_eval.py -x -v -m eval
"""

from __future__ import annotations

import os

import pytest
from langchain_core.messages import HumanMessage

from langchain_agentkit.agent import agent
from langchain_agentkit.extensions.tasks import TasksExtension
from langchain_agentkit.extensions.teams import TeamExtension

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

_MODEL = os.environ.get("AGENTKIT_EVAL_MODEL", "gpt-5.4-mini")


def _get_llm():
    """Return a deterministic ChatOpenAI instance."""
    return ChatOpenAI(model=_MODEL, temperature=0)


# ---------------------------------------------------------------------------
# Worker agent definitions
# ---------------------------------------------------------------------------


def _build_worker():
    """General-purpose worker that answers questions concisely."""
    _llm = _get_llm()

    class worker(agent):
        model = _llm
        description = "General-purpose worker that answers questions concisely"
        prompt = (
            "You are a helpful assistant. Answer questions concisely "
            "in one sentence. Be direct and factual."
        )

        async def handler(state, *, llm, prompt):
            from langchain_core.messages import SystemMessage

            messages = [SystemMessage(content=prompt)] + state["messages"]
            response = await llm.ainvoke(messages)
            return {"messages": [response], "sender": "worker"}

    return worker


def _build_math_worker():
    """Math-focused worker that answers with just the number."""
    _llm = _get_llm()

    class math_worker(agent):
        model = _llm
        description = "Answers math questions with just the numeric result"
        prompt = (
            "You are a calculator. When asked a math question, respond with "
            "ONLY the numeric answer. No words, no explanation."
        )

        async def handler(state, *, llm, prompt):
            from langchain_core.messages import SystemMessage

            messages = [SystemMessage(content=prompt)] + state["messages"]
            response = await llm.ainvoke(messages)
            return {"messages": [response], "sender": "math_worker"}

    return math_worker


# ---------------------------------------------------------------------------
# Lead agent builder
# ---------------------------------------------------------------------------

_TEAM_LEAD_PROMPT = """\
You are a team lead. Follow these steps EXACTLY in order:

1. Use TeamCreate to create a team with the workers you need.
   - Each member needs a unique "name" and an "agent_type" matching a registered agent.
2. Use TeamMessage to give each worker their task.
3. Use TeamStatus to collect results. If no pending messages yet, call \
TeamStatus again after a moment.
4. Once you have all results, use TeamDissolve to shut down the team.
5. Report the final results to the user clearly.

IMPORTANT: Always complete ALL steps. Never skip TeamDissolve.
IMPORTANT: You MUST call the tools — never answer questions yourself.\
"""


def _build_team_lead(mw_team, mw_tasks):
    """Build a lead agent with team and task extensions."""
    _llm = _get_llm()

    class team_lead(agent):
        model = _llm
        extensions = [mw_tasks, mw_team]
        prompt = _TEAM_LEAD_PROMPT

        async def handler(state, *, llm, tools, prompt):
            from langchain_core.messages import SystemMessage

            messages = [SystemMessage(content=prompt)] + state["messages"]
            response = await llm.bind_tools(tools).ainvoke(messages)
            return {"messages": [response], "sender": "team_lead"}

    return team_lead


# ---------------------------------------------------------------------------
# Test 1: Single worker full lifecycle via compiled graph
# ---------------------------------------------------------------------------


class TestTeamSingleWorkerLifecycle:
    """Lead spawns 1 worker, assigns task, checks, dissolves — full graph flow."""

    async def test_team_single_worker_lifecycle(self):
        """Lead creates team, assigns 'Capital of Japan?', reports 'Tokyo'."""
        worker = _build_worker()
        mw_team = TeamExtension(agents=[worker])
        mw_tasks = TasksExtension()
        lead = _build_team_lead(mw_team, mw_tasks)

        graph = lead.compile()
        result = await graph.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Create a team called 'geo-team' with one worker "
                            "named 'alice' of agent_type 'worker'. "
                            "Assign alice the task: 'What is the capital of Japan?' "
                            "Then check for her answer and dissolve the team. "
                            "Report alice's answer."
                        )
                    )
                ]
            },
            {"recursion_limit": 40},
        )

        # Team should be dissolved (team metadata reset to None)
        assert result.get("team") is None, (
            f"Expected team dissolved (team=None), got: {result.get('team')}"
        )

        # Final response should contain "Tokyo"
        final = result["messages"][-1].content.lower()
        assert "tokyo" in final, f"Expected 'tokyo' in final response, got: {final}"


# ---------------------------------------------------------------------------
# Test 2: Multi-agent team via compiled graph
# ---------------------------------------------------------------------------


class TestTeamMultiAgent:
    """Lead spawns 2 workers of different types, assigns parallel tasks."""

    async def test_team_multi_agent(self):
        """Worker answers factual question, math_worker answers math."""
        worker = _build_worker()
        math_worker = _build_math_worker()
        mw_team = TeamExtension(agents=[worker, math_worker])
        mw_tasks = TasksExtension()
        lead = _build_team_lead(mw_team, mw_tasks)

        graph = lead.compile()
        result = await graph.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Create a team called 'mixed-team' with two members:\n"
                            "- 'alice' of agent_type 'worker'\n"
                            "- 'bob' of agent_type 'math_worker'\n"
                            "Assign alice: 'What color is the sky on a clear day?'\n"
                            "Assign bob: 'What is 15 + 27?'\n"
                            "Check for their answers, dissolve the team, and "
                            "report both results."
                        )
                    )
                ]
            },
            {"recursion_limit": 50},
        )

        # Team should be dissolved
        assert result.get("team") is None, f"Expected team dissolved, got: {result.get('team')}"

        # Final response should contain both answers
        final = result["messages"][-1].content.lower()
        assert "blue" in final, f"Expected 'blue' in final response, got: {final}"
        assert "42" in final, f"Expected '42' in final response, got: {final}"


# ---------------------------------------------------------------------------
# Test 3: Tools exposed (unit-style, no LLM needed)
# ---------------------------------------------------------------------------


class TestTeamToolsExposed:
    """Verify TeamExtension exposes exactly 4 tools."""

    def test_team_tools_exposed(self):
        worker = _build_worker()
        mw = TeamExtension(agents=[worker])
        tool_names = sorted(t.name for t in mw.tools)

        assert tool_names == [
            "TeamCreate",
            "TeamDissolve",
            "TeamMessage",
            "TeamStatus",
        ]


# ---------------------------------------------------------------------------
# Test 4: State schema (unit-style, no LLM needed)
# ---------------------------------------------------------------------------


class TestTeamStateSchema:
    """Verify state_schema returns TeamState."""

    def test_team_state_schema(self):
        from langchain_agentkit.extensions.teams.state import TeamState

        worker = _build_worker()
        mw = TeamExtension(agents=[worker])
        assert mw.state_schema is TeamState


# ---------------------------------------------------------------------------
# Test 5: Prompt includes agent roster (unit-style, no LLM needed)
# ---------------------------------------------------------------------------


class TestTeamPromptRoster:
    """Verify prompt() includes registered agent names."""

    def test_team_prompt_roster(self):
        worker = _build_worker()
        mw = TeamExtension(agents=[worker])
        prompt = mw.prompt(state={"messages": []})

        assert "worker" in prompt


# ---------------------------------------------------------------------------
# Cross-turn rehydration prompt + builder
# ---------------------------------------------------------------------------

_MULTI_TURN_LEAD_PROMPT = """\
You are a team lead. You handle multi-turn conversations.

If NO team is active:
1. Use TeamCreate to create a team with the workers you need.
2. Use TeamMessage to assign tasks.
3. Use TeamStatus to collect results.
4. Report results. Do NOT dissolve the team unless the user asks you to.

If a team IS already active from a previous turn:
1. Do NOT call TeamCreate — reuse the existing team.
2. Use TeamMessage to give new tasks or follow-up instructions.
3. Use TeamStatus to collect results.
4. If the user asks to dissolve, use TeamDissolve. Otherwise leave the team.
5. Report results.

IMPORTANT: You MUST call the tools — never answer questions yourself.\
"""


def _build_multi_turn_lead(mw_team, mw_tasks):
    """Lead that handles both fresh creation and continuation across turns."""
    _llm = _get_llm()

    class team_lead_mt(agent):
        model = _llm
        extensions = [mw_tasks, mw_team]
        prompt = _MULTI_TURN_LEAD_PROMPT

        async def handler(state, *, llm, tools, prompt):
            from langchain_core.messages import SystemMessage

            messages = [SystemMessage(content=prompt)] + state["messages"]
            response = await llm.bind_tools(tools).ainvoke(messages)
            return {"messages": [response], "sender": "team_lead_mt"}

    return team_lead_mt


# ---------------------------------------------------------------------------
# Test 6: Cross-turn rehydration — teammate remembers prior work
# ---------------------------------------------------------------------------


class TestTeamCrossTurnRehydration:
    """Team state and teammate history survive across ainvoke boundaries.

    Uses an InMemorySaver checkpointer and thread_id to simulate the
    multi-container / multi-request pattern: Turn 1 creates a team and
    assigns work without dissolving; Turn 2 sends a follow-up that
    requires the teammate to recall Turn 1's answer.

    IMPORTANT: both turns use the same compiled graph (same class name
    → same node names) so the checkpointer can resume correctly.
    """

    async def test_teammate_remembers_prior_turn(self):
        """Turn 1: Ask capital of Japan. Turn 2: Ask population of that capital."""
        from langgraph.checkpoint.memory import InMemorySaver

        worker = _build_worker()
        mw_team = TeamExtension(agents=[worker])
        mw_tasks = TasksExtension()
        lead = _build_multi_turn_lead(mw_team, mw_tasks)
        checkpointer = InMemorySaver()
        graph = lead.compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "rehydrate-test"}}

        # --- Turn 1 ---
        result1 = await graph.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Create a team called 'geo-team' with one worker "
                            "named 'alice' of agent_type 'worker'. "
                            "Ask alice: 'What is the capital of Japan?'"
                        )
                    )
                ]
            },
            {**config, "recursion_limit": 40},
        )

        # Team should still be active (not dissolved)
        assert result1.get("team") is not None, "Team should persist after turn 1"
        assert result1["team"]["name"] == "geo-team"

        # Tokyo should be in the conversation somewhere
        all_content = " ".join(
            str(m.content).lower() for m in result1.get("messages", []) if hasattr(m, "content")
        )
        assert "tokyo" in all_content, f"Expected 'tokyo' in turn 1 output: {all_content[:500]}"

        # --- Turn 2 — same graph, same thread, new user message ---
        result2 = await graph.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Send alice a follow-up: 'What is the approximate "
                            "population of that capital city you mentioned?' "
                            "Then dissolve the team and report."
                        )
                    )
                ]
            },
            {**config, "recursion_limit": 50},
        )

        # Team should now be dissolved after turn 2
        assert result2.get("team") is None, (
            f"Expected team dissolved after turn 2, got: {result2.get('team')}"
        )

        # Turn 2 conversation should contain a population figure (millions).
        # The data may appear in the subagent's response rather than the
        # lead's final summary, so search all turn-2 messages.
        all_turn2 = " ".join(
            str(m.content).lower() for m in result2.get("messages", []) if hasattr(m, "content")
        )
        assert any(kw in all_turn2 for kw in ["million", "14", "13", "tokyo", "population"]), (
            f"Expected population info in turn 2 messages: {all_turn2[:500]}"
        )


# ---------------------------------------------------------------------------
# Test 7: Tagged messages survive checkpointer round-trip
# ---------------------------------------------------------------------------


class TestTeamMessageHistoryPersistence:
    """Teammate-tagged messages in state['messages'] survive checkpointer storage.

    Verifies the capture-buffer → before_model flush → add_messages
    reducer → checkpointer write → checkpointer read path preserves
    the ``additional_kwargs["team"]["member"]`` tags.
    """

    async def test_tagged_messages_in_state_after_turn(self):
        """After Turn 1, state['messages'] contains team-tagged entries."""
        from langgraph.checkpoint.memory import InMemorySaver

        from langchain_agentkit.extensions.teams.filter import is_team_tagged, team_member_of

        worker = _build_worker()
        mw_team = TeamExtension(agents=[worker])
        mw_tasks = TasksExtension()
        lead = _build_multi_turn_lead(mw_team, mw_tasks)
        checkpointer = InMemorySaver()
        graph = lead.compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "history-persist-test"}}

        result = await graph.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Create a team called 'test-team' with one worker "
                            "named 'alice' of agent_type 'worker'. "
                            "Ask alice: 'What color is the sky on a clear day?'"
                        )
                    )
                ]
            },
            {**config, "recursion_limit": 40},
        )

        # Verify team-tagged messages exist in the checkpointed state
        messages = result.get("messages", [])
        tagged = [m for m in messages if is_team_tagged(m)]
        assert len(tagged) > 0, (
            "Expected at least one team-tagged message in state after teammate interaction"
        )

        # All tagged messages should belong to 'alice'
        for m in tagged:
            assert team_member_of(m) == "alice", (
                f"Expected team_member='alice', got '{team_member_of(m)}' on {m}"
            )

        # The lead's own messages should NOT be team-tagged
        from langchain_core.messages import AIMessage

        lead_ais = [
            m
            for m in messages
            if isinstance(m, AIMessage) and not is_team_tagged(m) and getattr(m, "tool_calls", None)
        ]
        assert len(lead_ais) > 0, "Expected at least one untagged lead AIMessage with tool_calls"


# ---------------------------------------------------------------------------
# Test 8: Dissolve after rehydration
# ---------------------------------------------------------------------------


class TestTeamDissolveAfterRehydration:
    """Dissolve runs correctly against a rehydrated (not freshly-created) team.

    Turn 1 creates and uses the team without dissolving.
    Turn 2 immediately dissolves.  Verifies runtime teardown works on a
    team that was reconstructed from state, not from the original
    TeamCreate call.
    """

    async def test_dissolve_rehydrated_team(self):
        """Turn 1: create + work. Turn 2: dissolve only."""
        from langgraph.checkpoint.memory import InMemorySaver

        worker = _build_worker()
        mw_team = TeamExtension(agents=[worker])
        mw_tasks = TasksExtension()
        lead = _build_multi_turn_lead(mw_team, mw_tasks)
        checkpointer = InMemorySaver()
        graph = lead.compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "dissolve-rehydrate-test"}}

        # --- Turn 1: create + work ---
        await graph.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Create a team called 'temp-team' with one worker "
                            "named 'alice' of agent_type 'worker'. "
                            "Ask alice: 'Say hello.'"
                        )
                    )
                ]
            },
            {**config, "recursion_limit": 40},
        )

        # --- Turn 2: dissolve only ---
        result2 = await graph.ainvoke(
            {
                "messages": [
                    HumanMessage(content="Dissolve the team immediately and confirm it's done.")
                ]
            },
            {**config, "recursion_limit": 30},
        )

        assert result2.get("team") is None, f"Expected team dissolved, got: {result2.get('team')}"


# ---------------------------------------------------------------------------
# Test 9: Internal dialogue structure verification
# ---------------------------------------------------------------------------


class TestTeamInternalDialogueStructure:
    """Verify the shape of teammate-internal messages in persisted state.

    After a team turn, ``state["messages"]`` contains both the lead's
    conversation (untagged) and each teammate's internal conversation
    (tagged with ``team_member``).  This test inspects the tagged slice
    to verify:

    - The teammate received an instruction (tagged ``HumanMessage``)
    - The teammate produced a response (tagged ``AIMessage``)
    - The internal conversation alternates correctly
    - The lead's filtered view (what the LLM sees) contains no tagged
      messages and has valid tool-call pairs
    - The router-wrapped reply exists in the lead's view
    """

    async def test_internal_dialogue_shape(self):
        """Create team, assign task, inspect the raw and filtered views."""
        from langgraph.checkpoint.memory import InMemorySaver

        from langchain_agentkit.extensions.teams.filter import (
            filter_out_team_messages,
            filter_team_messages,
            is_team_tagged,
        )

        worker = _build_worker()
        mw_team = TeamExtension(agents=[worker])
        mw_tasks = TasksExtension()
        lead = _build_multi_turn_lead(mw_team, mw_tasks)
        checkpointer = InMemorySaver()
        graph = lead.compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "dialogue-structure-test"}}

        result = await graph.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Create a team called 'qa-team' with one worker "
                            "named 'alice' of agent_type 'worker'. "
                            "Ask alice: 'What is 2 + 2?'"
                        )
                    )
                ]
            },
            {**config, "recursion_limit": 40},
        )

        all_messages = result.get("messages", [])

        # --- Teammate's internal view ---
        alice_messages = filter_team_messages(all_messages, "alice")
        assert len(alice_messages) >= 2, (
            f"Expected at least HumanMessage + AIMessage for alice, "
            f"got {len(alice_messages)} messages"
        )

        # First alice message should be a HumanMessage (the instruction
        # she received from the bus, constructed by _teammate_loop).
        from langchain_core.messages import AIMessage as _AIMessage
        from langchain_core.messages import HumanMessage as _HumanMessage

        human_msgs = [m for m in alice_messages if isinstance(m, _HumanMessage)]
        ai_msgs = [m for m in alice_messages if isinstance(m, _AIMessage)]

        assert len(human_msgs) >= 1, "Alice should have received at least one instruction"
        assert len(ai_msgs) >= 1, "Alice should have produced at least one response"

        # The instruction should contain the question asked via TeamMessage.
        instruction_contents = " ".join(m.content for m in human_msgs).lower()
        assert "2 + 2" in instruction_contents or "2+2" in instruction_contents, (
            f"Expected '2 + 2' in alice's instruction, got: {instruction_contents}"
        )

        # Alice's reply should contain "4".
        reply_contents = " ".join(str(m.content) for m in ai_msgs).lower()
        assert "4" in reply_contents, f"Expected '4' in alice's reply, got: {reply_contents}"

        # All of alice's messages should have the team_member tag.
        for m in alice_messages:
            assert is_team_tagged(m), f"Alice message missing team_member tag: {m}"

        # --- Lead's filtered view ---
        lead_view = filter_out_team_messages(all_messages)

        # No tagged messages should leak into the lead's view.
        for m in lead_view:
            assert not is_team_tagged(m), f"Tagged message leaked into lead view: {m}"

        # Lead's view should contain the router-wrapped reply: a plain
        # HumanMessage (no team metadata) injected by the router node.
        # The first HumanMessage is the user's input; any subsequent ones
        # are router-wrapped teammate replies.
        router_wraps = [m for m in lead_view[1:] if isinstance(m, _HumanMessage)]
        assert len(router_wraps) >= 1, (
            "Expected at least one router-wrapped teammate reply in the lead's view"
        )
        # The router wrap should contain alice's answer.
        wrap_content = " ".join(m.content for m in router_wraps).lower()
        assert "4" in wrap_content, f"Expected '4' in router-wrapped reply, got: {wrap_content}"

        # Lead's view should have valid tool-call pairing: every
        # AIMessage(tool_calls=[{id: X}]) should have a matching
        # ToolMessage(tool_call_id=X) somewhere later in the list.
        from langchain_core.messages import ToolMessage as _ToolMessage

        tool_call_ids_emitted: set[str] = set()
        tool_call_ids_resolved: set[str] = set()
        for m in lead_view:
            if isinstance(m, _AIMessage) and getattr(m, "tool_calls", None):
                for tc in m.tool_calls:
                    tool_call_ids_emitted.add(tc["id"])
            if isinstance(m, _ToolMessage):
                tool_call_ids_resolved.add(m.tool_call_id)

        orphaned = tool_call_ids_emitted - tool_call_ids_resolved
        assert not orphaned, (
            f"Lead view has orphaned tool calls (no matching ToolMessage): {orphaned}"
        )

    async def test_rehydrated_teammate_sees_prior_internal_dialogue(self):
        """After rehydration, teammate's tagged history from Turn 1 is present in Turn 2."""
        from langgraph.checkpoint.memory import InMemorySaver

        from langchain_agentkit.extensions.teams.filter import filter_team_messages

        worker = _build_worker()
        mw_team = TeamExtension(agents=[worker])
        mw_tasks = TasksExtension()
        lead = _build_multi_turn_lead(mw_team, mw_tasks)
        checkpointer = InMemorySaver()
        graph = lead.compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "rehydrate-dialogue-test"}}

        # --- Turn 1 ---
        result1 = await graph.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Create a team called 'math-team' with one worker "
                            "named 'bob' of agent_type 'worker'. "
                            "Ask bob: 'What is 10 * 5?'"
                        )
                    )
                ]
            },
            {**config, "recursion_limit": 40},
        )

        turn1_bob = filter_team_messages(result1["messages"], "bob")
        assert len(turn1_bob) >= 2, "Turn 1 should have bob's instruction + reply"

        # --- Turn 2 ---
        result2 = await graph.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Ask bob: 'Now double your previous answer.' "
                            "Then dissolve the team and report."
                        )
                    )
                ]
            },
            {**config, "recursion_limit": 50},
        )

        # Turn 2's state should contain BOTH turn 1 and turn 2's bob messages.
        turn2_bob = filter_team_messages(result2["messages"], "bob")
        assert len(turn2_bob) > len(turn1_bob), (
            f"Turn 2 should have more bob messages than turn 1. "
            f"Turn 1: {len(turn1_bob)}, Turn 2: {len(turn2_bob)}"
        )

        # The old messages from turn 1 should still be in the list
        # (they aren't deleted — only the teammate's rehydration filters
        # them as input; they stay in persisted state).
        turn1_contents = {str(m.content) for m in turn1_bob}
        turn2_contents = {str(m.content) for m in turn2_bob}
        assert turn1_contents.issubset(turn2_contents), (
            "Turn 1 bob messages should be a subset of turn 2 bob messages"
        )

        # The final answer should contain "100" (50 * 2).
        final = result2["messages"][-1].content.lower()
        assert "100" in final, f"Expected '100' in final response, got: {final}"
