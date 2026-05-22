# ruff: noqa: N801, N805
"""Tests for TasksExtension."""

from unittest.mock import MagicMock

from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolRuntime

from langchain_agentkit.agent_kit import run_extension_setup
from langchain_agentkit.extensions.tasks import (
    TASK_MANAGEMENT_PROMPT,
    TasksExtension,
    format_task_context,
)

_TEST_RUNTIME = ToolRuntime(
    state={},
    context=None,
    config={},
    stream_writer=lambda _: None,
    tool_call_id=None,
    store=None,
)


class TestTasksExtensionTools:
    def test_default_constructor_creates_task_tools(self):
        mw = TasksExtension()

        tool_names = [t.name for t in mw.tools]
        assert "TaskCreate" in tool_names
        assert "TaskUpdate" in tool_names
        assert "TaskList" in tool_names
        assert "TaskGet" in tool_names
        assert "TaskStop" in tool_names

    def test_constructor_with_custom_tools_uses_those(self):
        tool_a = MagicMock(spec=BaseTool)
        tool_b = MagicMock(spec=BaseTool)

        mw = TasksExtension(tools=[tool_a, tool_b])

        assert mw.tools == (tool_a, tool_b)

    def test_tools_returns_immutable_tuple(self):
        tool_a = MagicMock(spec=BaseTool)
        mw = TasksExtension(tools=[tool_a])

        first_call = mw.tools
        second_call = mw.tools

        assert first_call == second_call
        assert isinstance(first_call, tuple)


class TestTasksExtensionPrompt:
    def test_prompt_with_no_tasks_includes_task_management(self):
        mw = TasksExtension()

        result = mw.prompt({"tasks": []}, _TEST_RUNTIME)

        assert TASK_MANAGEMENT_PROMPT in result

    def test_prompt_with_tasks_splits_guidance_and_live_list(self):
        mw = TasksExtension()
        tasks = [{"subject": "Write tests", "status": "in_progress"}]

        result = mw.prompt({"tasks": tasks}, _TEST_RUNTIME)

        # Static guidance stays in the cacheable system prompt; the live
        # list rides the per-turn reminder channel.
        assert isinstance(result, dict)
        assert TASK_MANAGEMENT_PROMPT in result["prompt"]
        assert "Write tests" in result["reminder"]

    def test_prompt_without_tasks_key_includes_task_management(self):
        mw = TasksExtension()

        result = mw.prompt({}, _TEST_RUNTIME)

        assert TASK_MANAGEMENT_PROMPT in result

    def test_prompt_channels_by_task_presence(self):
        mw = TasksExtension()

        # No active task: static guidance only, as a plain string.
        assert mw.prompt({"tasks": []}, _TEST_RUNTIME) == TASK_MANAGEMENT_PROMPT
        assert mw.prompt({}, _TEST_RUNTIME) == TASK_MANAGEMENT_PROMPT
        # Pending-only also yields guidance only — nothing is in progress yet.
        pending = {"tasks": [{"subject": "P", "status": "pending"}]}
        assert mw.prompt(pending, _TEST_RUNTIME) == TASK_MANAGEMENT_PROMPT

        # An in-progress task adds the grounding reminder.
        result_with = mw.prompt(
            {"tasks": [{"subject": "Active", "status": "in_progress"}]},
            _TEST_RUNTIME,
        )
        assert isinstance(result_with, dict)
        assert result_with["prompt"] == TASK_MANAGEMENT_PROMPT
        assert "Active" in result_with["reminder"]

    def test_custom_formatter_is_used(self):
        def my_formatter(tasks):
            return f"Custom: {len(tasks)} tasks"

        mw = TasksExtension(formatter=my_formatter)
        result = mw.prompt({"tasks": [{"subject": "A"}]}, _TEST_RUNTIME)

        assert result["reminder"] == "Custom: 1 tasks"


class TestFormatTaskContext:
    def test_empty_list_returns_empty(self):
        # Static guidance moved to the system prompt; the formatter now
        # renders only the dynamic list, which is empty when there are no tasks.
        assert format_task_context([]) == ""

    def test_no_in_progress_returns_empty(self):
        # Only an in-progress task grounds the reminder; pending/completed
        # tasks are already visible in the conversation via the tool calls.
        tasks = [
            {"subject": "Pending", "status": "pending"},
            {"subject": "Done", "status": "completed"},
        ]
        assert format_task_context(tasks) == ""

    def test_in_progress_task_grounds_the_reminder(self):
        tasks = [
            {"subject": "Done", "status": "completed"},
            {"subject": "Write tests", "status": "in_progress"},
            {"subject": "Later", "status": "pending"},
        ]

        result = format_task_context(tasks)

        # Only the active task is named; the others are not repeated.
        assert result == "Keep your focus on:\n- Write tests"
        assert "Done" not in result
        assert "Later" not in result

    def test_multiple_in_progress_listed(self):
        tasks = [
            {"subject": "Alpha", "status": "in_progress"},
            {"subject": "Beta", "status": "in_progress"},
        ]

        result = format_task_context(tasks)

        assert result.startswith("Keep your focus on:")
        assert "- Alpha" in result
        assert "- Beta" in result

    def test_all_deleted_returns_empty(self):
        tasks = [{"subject": "Gone", "status": "deleted"}]

        assert format_task_context(tasks) == ""


class TestConditionalTeamTips:
    def test_default_no_team_tips(self):
        mw = TasksExtension()

        tool_descriptions = [t.description for t in mw.tools if t.name == "TaskCreate"]
        assert len(tool_descriptions) == 1
        assert "Team tips" not in tool_descriptions[0]

    async def test_team_active_adds_tips(self):
        from langchain_agentkit import Agent, AgentKit, TeamExtension

        class Teammate(Agent):
            model = MagicMock()

            async def handler(state, *, llm):  # noqa: N805
                return {"messages": [], "sender": "teammate"}

        teammate = await Teammate().compile()
        kit = AgentKit(extensions=[TasksExtension(), TeamExtension(agents=[teammate])])
        await run_extension_setup(kit)
        tasks_ext = next(e for e in kit._extensions if isinstance(e, TasksExtension))

        tool_descriptions = [t.description for t in tasks_ext.tools if t.name == "TaskCreate"]
        assert len(tool_descriptions) == 1
        assert "Team tips" in tool_descriptions[0]

    async def test_team_active_adds_teammate_context_inline(self):
        from langchain_agentkit import Agent, AgentKit, TeamExtension

        class Teammate(Agent):
            model = MagicMock()

            async def handler(state, *, llm):  # noqa: N805
                return {"messages": [], "sender": "teammate"}

        teammate = await Teammate().compile()
        kit = AgentKit(extensions=[TasksExtension(), TeamExtension(agents=[teammate])])
        await run_extension_setup(kit)
        tasks_ext = next(e for e in kit._extensions if isinstance(e, TasksExtension))

        tool_descriptions = [t.description for t in tasks_ext.tools if t.name == "TaskCreate"]
        desc = tool_descriptions[0]
        # Team-active TaskCreate points at an existing teammate via the
        # TaskUpdate(owner) path — the boundary that keeps it distinct from TeamCreate.
        assert "teammate" in desc
        assert "TaskUpdate" in desc


class TestTasksExtensionProtocol:
    def test_satisfies_extension_protocol_structurally(self):
        """TasksExtension has the tools property and prompt method required by Extension."""
        mw = TasksExtension()

        assert hasattr(mw, "tools")
        assert callable(mw.prompt)

        # Verify the tools property returns a sequence
        assert isinstance(mw.tools, (list, tuple))

        # Verify prompt accepts (state, config) and returns str
        result = mw.prompt({}, _TEST_RUNTIME)
        assert isinstance(result, str)
