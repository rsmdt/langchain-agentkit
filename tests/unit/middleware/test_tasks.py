"""Tests for TasksMiddleware."""

from unittest.mock import MagicMock

from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolRuntime

from langchain_agentkit.middleware.tasks import (
    BASE_AGENT_PROMPT,
    TASK_MANAGEMENT_PROMPT,
    TasksMiddleware,
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


class TestTasksMiddlewareTools:
    def test_default_constructor_creates_task_tools(self):
        mw = TasksMiddleware()

        tool_names = [t.name for t in mw.tools]
        assert "TaskCreate" in tool_names
        assert "TaskUpdate" in tool_names
        assert "TaskList" in tool_names
        assert "TaskGet" in tool_names
        assert "TaskStop" in tool_names

    def test_constructor_with_custom_tools_uses_those(self):
        tool_a = MagicMock(spec=BaseTool)
        tool_b = MagicMock(spec=BaseTool)

        mw = TasksMiddleware(task_tools=[tool_a, tool_b])

        assert mw.tools == (tool_a, tool_b)

    def test_tools_returns_immutable_tuple(self):
        tool_a = MagicMock(spec=BaseTool)
        mw = TasksMiddleware(task_tools=[tool_a])

        first_call = mw.tools
        second_call = mw.tools

        assert first_call == second_call
        assert isinstance(first_call, tuple)


class TestTasksMiddlewarePrompt:
    def test_prompt_with_no_tasks_includes_base_and_task_management(self):
        mw = TasksMiddleware()

        result = mw.prompt({"tasks": []}, _TEST_RUNTIME)

        assert BASE_AGENT_PROMPT in result
        assert TASK_MANAGEMENT_PROMPT in result

    def test_prompt_with_tasks_includes_base_and_formatted_context(self):
        mw = TasksMiddleware()
        tasks = [{"subject": "Write tests", "status": "in_progress"}]

        result = mw.prompt({"tasks": tasks}, _TEST_RUNTIME)

        assert BASE_AGENT_PROMPT in result
        assert "Write tests" in result
        assert TASK_MANAGEMENT_PROMPT not in result

    def test_prompt_without_tasks_key_includes_task_management(self):
        mw = TasksMiddleware()

        result = mw.prompt({}, _TEST_RUNTIME)

        assert BASE_AGENT_PROMPT in result
        assert TASK_MANAGEMENT_PROMPT in result

    def test_prompt_always_returns_string(self):
        mw = TasksMiddleware()

        result_empty = mw.prompt({"tasks": []}, _TEST_RUNTIME)
        result_with = mw.prompt(
            {"tasks": [{"subject": "Task", "status": "pending"}]},
            _TEST_RUNTIME,
        )
        result_none = mw.prompt({}, _TEST_RUNTIME)

        assert isinstance(result_empty, str)
        assert isinstance(result_with, str)
        assert isinstance(result_none, str)

    def test_custom_formatter_is_used(self):
        def my_formatter(tasks):
            return f"Custom: {len(tasks)} tasks"

        mw = TasksMiddleware(formatter=my_formatter)
        result = mw.prompt({"tasks": [{"subject": "A"}]}, _TEST_RUNTIME)

        assert "Custom: 1 tasks" in result


class TestFormatTaskContext:
    def test_empty_list_returns_task_management_prompt(self):
        result = format_task_context([])

        assert result == TASK_MANAGEMENT_PROMPT

    def test_single_task_formatted_with_checkbox(self):
        result = format_task_context([{"subject": "Deploy app", "status": "pending"}])

        assert "Deploy app" in result
        assert "[ ]" in result

    def test_completed_task_has_x_marker(self):
        result = format_task_context([{"subject": "Done task", "status": "completed"}])

        assert "[x]" in result
        assert "Done task" in result

    def test_pending_task_has_empty_marker(self):
        result = format_task_context([{"subject": "Pending task", "status": "pending"}])

        assert "[ ]" in result
        assert "Pending task" in result

    def test_in_progress_task_has_arrow_marker(self):
        result = format_task_context([{"subject": "Active task", "status": "in_progress"}])

        assert "[>]" in result
        assert "Active task" in result

    def test_in_progress_with_active_form_shows_spinner_text(self):
        tasks = [{"subject": "Analyzing", "status": "in_progress", "active_form": "Analyzing..."}]

        result = format_task_context(tasks)

        assert '[>] Analyzing -- "Analyzing..."' in result

    def test_deleted_tasks_filtered_out(self):
        tasks = [
            {"subject": "Visible", "status": "pending"},
            {"subject": "Hidden", "status": "deleted"},
        ]

        result = format_task_context(tasks)

        assert "Visible" in result
        assert "Hidden" not in result

    def test_blocked_by_shown_for_pending_tasks(self):
        tasks = [
            {"subject": "Blocked task", "status": "pending", "blocked_by": ["task-1", "task-2"]},
        ]

        result = format_task_context(tasks)

        assert "blocked by: task-1, task-2" in result

    def test_multiple_tasks_all_formatted(self):
        tasks = [
            {"subject": "First", "status": "completed"},
            {"subject": "Second", "status": "pending"},
            {"subject": "Third", "status": "in_progress"},
        ]

        result = format_task_context(tasks)

        assert "[x] First" in result
        assert "[ ] Second" in result
        assert "[>] Third" in result

    def test_numbered_list_format(self):
        tasks = [
            {"subject": "First", "status": "pending"},
            {"subject": "Second", "status": "pending"},
        ]

        result = format_task_context(tasks)

        assert "1. [ ] First" in result
        assert "2. [ ] Second" in result

    def test_xml_wrapping(self):
        tasks = [{"subject": "Task", "status": "pending"}]

        result = format_task_context(tasks)

        assert "<tasks>" in result
        assert "</tasks>" in result

    def test_missing_status_defaults_to_pending(self):
        result = format_task_context([{"subject": "No status"}])

        assert "[ ]" in result

    def test_all_deleted_returns_task_management_prompt(self):
        tasks = [{"subject": "Gone", "status": "deleted"}]

        result = format_task_context(tasks)

        assert result == TASK_MANAGEMENT_PROMPT


class TestTasksMiddlewareProtocol:
    def test_satisfies_middleware_protocol_structurally(self):
        """TasksMiddleware has the tools property and prompt method required by Middleware."""
        mw = TasksMiddleware()

        assert hasattr(mw, "tools")
        assert callable(mw.prompt)

        # Verify the tools property returns a sequence
        assert isinstance(mw.tools, (list, tuple))

        # Verify prompt accepts (state, config) and returns str
        result = mw.prompt({}, _TEST_RUNTIME)
        assert isinstance(result, str)
