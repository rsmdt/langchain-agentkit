"""Tests for TasksMiddleware."""

from unittest.mock import MagicMock

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from langchain_agentkit.tasks_middleware import (
    BASE_AGENT_PROMPT,
    TASK_MANAGEMENT_PROMPT,
    TasksMiddleware,
    format_task_context,
)


class TestTasksMiddlewareTools:
    def test_default_constructor_has_empty_tools(self):
        mw = TasksMiddleware()

        assert mw.tools == []

    def test_constructor_with_tools_returns_those_tools(self):
        tool_a = MagicMock(spec=BaseTool)
        tool_b = MagicMock(spec=BaseTool)

        mw = TasksMiddleware(task_tools=[tool_a, tool_b])

        assert mw.tools == [tool_a, tool_b]

    def test_tools_returns_defensive_copy(self):
        tool_a = MagicMock(spec=BaseTool)
        mw = TasksMiddleware(task_tools=[tool_a])

        first_call = mw.tools
        second_call = mw.tools

        assert first_call == second_call
        assert first_call is not second_call


class TestTasksMiddlewarePrompt:
    def test_prompt_with_no_tasks_includes_base_and_task_management(self):
        mw = TasksMiddleware()

        result = mw.prompt({"tasks": []}, RunnableConfig())

        assert BASE_AGENT_PROMPT in result
        assert TASK_MANAGEMENT_PROMPT in result

    def test_prompt_with_tasks_includes_base_and_formatted_context(self):
        mw = TasksMiddleware()
        tasks = [{"subject": "Write tests", "status": "in_progress"}]

        result = mw.prompt({"tasks": tasks}, RunnableConfig())

        assert BASE_AGENT_PROMPT in result
        assert "Write tests" in result
        assert TASK_MANAGEMENT_PROMPT not in result

    def test_prompt_without_tasks_key_includes_task_management(self):
        mw = TasksMiddleware()

        result = mw.prompt({}, RunnableConfig())

        assert BASE_AGENT_PROMPT in result
        assert TASK_MANAGEMENT_PROMPT in result

    def test_prompt_always_returns_string(self):
        mw = TasksMiddleware()

        result_empty = mw.prompt({"tasks": []}, RunnableConfig())
        result_with = mw.prompt(
            {"tasks": [{"subject": "Task", "status": "pending"}]},
            RunnableConfig(),
        )
        result_none = mw.prompt({}, RunnableConfig())

        assert isinstance(result_empty, str)
        assert isinstance(result_with, str)
        assert isinstance(result_none, str)


class TestFormatTaskContext:
    def test_empty_list_returns_empty_string(self):
        result = format_task_context([])

        assert result == ""

    def test_single_task_formatted_with_checkbox(self):
        result = format_task_context([{"subject": "Deploy app", "status": "pending"}])

        assert "Deploy app" in result
        assert "[ ]" in result or "[x]" in result

    def test_completed_task_has_x_marker(self):
        result = format_task_context([{"subject": "Done task", "status": "completed"}])

        assert "[x]" in result
        assert "Done task" in result

    def test_pending_task_has_empty_marker(self):
        result = format_task_context([{"subject": "Pending task", "status": "pending"}])

        assert "[ ]" in result
        assert "Pending task" in result

    def test_multiple_tasks_all_formatted(self):
        tasks = [
            {"subject": "First", "status": "completed"},
            {"subject": "Second", "status": "pending"},
            {"subject": "Third", "status": "in_progress"},
        ]

        result = format_task_context(tasks)

        assert "[x] First" in result
        assert "[ ] Second" in result
        assert "[ ] Third" in result

    def test_missing_status_defaults_to_pending(self):
        result = format_task_context([{"subject": "No status"}])

        assert "[ ]" in result

    def test_missing_subject_defaults_to_untitled(self):
        result = format_task_context([{"status": "pending"}])

        assert "Untitled" in result


class TestTasksMiddlewareProtocol:
    def test_satisfies_middleware_protocol_structurally(self):
        """TasksMiddleware has the tools property and prompt method required by Middleware."""
        mw = TasksMiddleware()

        assert hasattr(mw, "tools")
        assert callable(mw.prompt)

        # Verify the tools property returns a list
        assert isinstance(mw.tools, list)

        # Verify prompt accepts (state, config) and returns str
        result = mw.prompt({}, RunnableConfig())
        assert isinstance(result, str)
