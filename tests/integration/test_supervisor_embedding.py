# ruff: noqa: N805
"""Embedding an AgentKit agent as a *node* inside a supervisor/router graph.

Proves the "supervisor pattern" works with AgentKit agents as first-class graph
nodes — not as imperatively-invoked wrappers whose completion signal the
supervisor must sniff out of a message tag. The embedded agent shares the
parent's ``messages`` channel, and a tool passed to it can return
``Command(goto=..., graph=Command.PARENT)`` to navigate the *parent* supervisor
directly. The hand-off signal is a real edge (the ``goto`` target), not a
sniffed message.

No LLM: the embedded agent's model is a scripted fake, so this runs in CI with
no API key and isolates the *wiring* (Command.PARENT escaping the agent
subgraph) from any model behaviour. The companion LLM-behaviour eval lives in
``tests/evals/test_supervisor_embedding_eval.py``.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from langchain_agentkit import Agent
from tests._supervisor_embedding import (
    SETUP_PROMPT,
    build_supervisor,
    do_setup_work,
    handoff_tools,
    orphaned_tool_messages,
)


class _ScriptedToolModel(BaseChatModel):
    """Deterministic chat model that emits a queued list of AIMessages.

    ``bind_tools`` is a no-op passthrough — the script, not the bound tools,
    decides what the model "says" — so tests stay deterministic and need no
    API key while still exercising real tool-call execution downstream.
    """

    responses: list[Any]
    model_config = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:
        return "scripted-tool"

    def bind_tools(self, tools: Any, **kwargs: Any) -> _ScriptedToolModel:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Any = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        message = self.responses.pop(0)
        return ChatResult(generations=[ChatGeneration(message=message)])


def _scripted_model(*responses: AIMessage) -> _ScriptedToolModel:
    """A deterministic chat model that yields *responses* in order."""
    return _ScriptedToolModel(responses=list(responses))


async def _build_embedded_agent(model_obj: _ScriptedToolModel) -> Any:
    """Compile an AgentKit agent carrying the two hand-off tools."""

    class SetupAgent(Agent):
        model = model_obj
        prompt = SETUP_PROMPT
        tools = [do_setup_work, *handoff_tools()]

        async def handler(state, *, llm, tools):
            bound = llm.bind_tools(tools)
            response = await bound.ainvoke(state["messages"])
            return {"messages": [response]}

    return await SetupAgent().compile()


class TestSupervisorEmbedding:
    """Command(graph=PARENT) from an embedded AgentKit tool navigates the parent."""

    async def test_complete_phase_navigates_supervisor_to_advance(self) -> None:
        """A tool returning Command(goto='advance', PARENT) routes the parent to advance."""
        model = _scripted_model(
            AIMessage(
                content="",
                tool_calls=[{"name": "complete_phase", "args": {}, "id": "call_1"}],
            ),
        )
        agent = await _build_embedded_agent(model)
        supervisor = build_supervisor(agent)

        result = await supervisor.ainvoke({"messages": [HumanMessage(content="set up the repo")]})

        # The hand-off edge fired: control reached the parent's 'advance' node...
        assert "advance" in result["visited"]
        # ...and never bounced into the wrong hand-back target.
        assert "classify_route" not in result["visited"]
        # The hand-off ToolMessage is in the shared messages channel.
        assert any(
            isinstance(m, ToolMessage) and "complete_phase->advance" in m.content
            for m in result["messages"]
        )
        # The forwarded history is well-formed (no orphaned tool messages).
        assert orphaned_tool_messages(result["messages"]) == []

    async def test_return_to_supervisor_navigates_to_classify_route(self) -> None:
        """A tool returning Command(goto='classify_route', PARENT) hands back to the supervisor."""
        model = _scripted_model(
            AIMessage(
                content="",
                tool_calls=[{"name": "return_to_supervisor", "args": {}, "id": "call_1"}],
            ),
        )
        agent = await _build_embedded_agent(model)
        supervisor = build_supervisor(agent)

        result = await supervisor.ainvoke(
            {"messages": [HumanMessage(content="actually, what's the weather?")]}
        )

        assert "classify_route" in result["visited"]
        assert "advance" not in result["visited"]
        assert any(
            isinstance(m, ToolMessage) and "return_to_supervisor->classify_route" in m.content
            for m in result["messages"]
        )

    async def test_work_then_handoff_loops_internally_then_escapes(self) -> None:
        """Multi-step: the agent runs its own ReAct turns, THEN hands off.

        Guards the realistic path where the agent loops internally
        (work tool -> back to model) before the hand-off tool fires — the
        internal ``tools -> agent`` loop must NOT swallow the eventual
        Command(graph=PARENT). The hand-off still reaches the parent.
        """
        model = _scripted_model(
            AIMessage(
                content="",
                tool_calls=[{"name": "do_setup_work", "args": {"detail": "scaffold"}, "id": "w1"}],
            ),
            AIMessage(
                content="",
                tool_calls=[{"name": "complete_phase", "args": {}, "id": "call_2"}],
            ),
        )
        agent = await _build_embedded_agent(model)
        supervisor = build_supervisor(agent)

        result = await supervisor.ainvoke({"messages": [HumanMessage(content="set up")]})

        # The hand-off fired despite the internal work loop.
        assert "advance" in result["visited"]

    async def test_handoff_preserves_full_valid_message_history(self) -> None:
        """The hand-off forwards the agent's full turn trace, valid and well-formed.

        A bare ``Command(graph=PARENT)`` escape would propagate only its own
        ``update``, losing the agent's internal (AIMessage, ToolMessage) pairs
        and orphaning the closing ToolMessage. The hand-off tool instead forwards
        the agent's full trace (via ``InjectedState``) so the supervisor sees the
        complete history AND it stays valid for a downstream LLM call: every
        ToolMessage is preceded by an AIMessage bearing its tool_call_id.
        """
        model = _scripted_model(
            AIMessage(
                content="",
                tool_calls=[{"name": "do_setup_work", "args": {"detail": "scaffold"}, "id": "w1"}],
            ),
            AIMessage(
                content="",
                tool_calls=[{"name": "complete_phase", "args": {}, "id": "call_2"}],
            ),
        )
        agent = await _build_embedded_agent(model)
        supervisor = build_supervisor(agent)

        result = await supervisor.ainvoke({"messages": [HumanMessage(content="set up")]})
        msgs = result["messages"]

        # The agent's internal work IS preserved in the parent history.
        assert any(isinstance(m, ToolMessage) and "did:scaffold" in m.content for m in msgs)
        # The hand-off closing ToolMessage is present (the Command demonstrably fired).
        assert any(
            isinstance(m, ToolMessage) and "complete_phase->advance" in m.content for m in msgs
        )
        # History is well-formed: no orphaned ToolMessages -> a downstream
        # supervisor LLM call over these messages would be valid.
        assert orphaned_tool_messages(msgs) == []
        # The original user message is not duplicated by the forward.
        assert sum(1 for m in msgs if isinstance(m, HumanMessage)) == 1

    async def test_handoff_history_persists_in_parent_checkpointer(self) -> None:
        """The forwarded history lands in the parent checkpointer's thread state.

        With a checkpointer on the parent only (the embedded subgraph inherits
        it), the worker's full forwarded trace is persisted under the parent
        thread and retrievable via ``get_state`` — so a resumed supervisor turn
        sees the complete, well-formed history.
        """
        from langgraph.checkpoint.memory import InMemorySaver

        model = _scripted_model(
            AIMessage(
                content="",
                tool_calls=[{"name": "do_setup_work", "args": {"detail": "scaffold"}, "id": "w1"}],
            ),
            AIMessage(
                content="",
                tool_calls=[{"name": "complete_phase", "args": {}, "id": "call_2"}],
            ),
        )
        agent = await _build_embedded_agent(model)
        supervisor = build_supervisor(agent, checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": "t1"}}

        await supervisor.ainvoke({"messages": [HumanMessage(content="set up")]}, config)

        # Inspect the PERSISTED parent thread state, not just the return value.
        persisted = supervisor.get_state(config).values["messages"]
        assert any(isinstance(m, ToolMessage) and "did:scaffold" in m.content for m in persisted)
        assert any(
            isinstance(m, ToolMessage) and "complete_phase->advance" in m.content for m in persisted
        )
        assert orphaned_tool_messages(persisted) == []

    async def test_plain_answer_takes_static_end_edge(self) -> None:
        """When no hand-off tool fires, the agent answers and takes the static END edge.

        The diagram's path (a): a static ``setup -> END`` edge and the dynamic
        ``Command(goto=...)`` hand-off edges coexist on the same node — the
        Command overrides when present (tests above), the static edge applies
        when absent. The agent's answer reaches the parent via normal subgraph
        completion.
        """
        model = _scripted_model(AIMessage(content="setup acknowledged"))
        agent = await _build_embedded_agent(model)
        supervisor = build_supervisor(agent)

        result = await supervisor.ainvoke({"messages": [HumanMessage(content="status?")]})

        # No hand-off node was reached...
        assert "advance" not in result["visited"]
        assert "classify_route" not in result["visited"]
        # ...and a normal completion DOES sync the agent's answer upward.
        assert any(
            isinstance(m, AIMessage) and m.content == "setup acknowledged"
            for m in result["messages"]
        )
