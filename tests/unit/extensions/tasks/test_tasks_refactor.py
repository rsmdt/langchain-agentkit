"""Tests for tx1 refactor: TasksExtension no longer emits universal guidance."""

from __future__ import annotations

from langgraph.prebuilt import ToolRuntime

from langchain_agentkit.extensions.tasks import (
    TASK_MANAGEMENT_PROMPT,
    TasksExtension,
)

_RUNTIME = ToolRuntime(
    state={},
    context=None,
    config={},
    stream_writer=lambda _: None,
    tool_call_id=None,
    store=None,
)


class TestTasksExtensionPromptContent:
    def test_prompt_with_no_tasks_is_task_management_only(self):
        ext = TasksExtension()
        result = ext.prompt({"tasks": []}, _RUNTIME)

        assert result == TASK_MANAGEMENT_PROMPT

    def test_prompt_without_tasks_key_is_task_management_only(self):
        ext = TasksExtension()
        result = ext.prompt({}, _RUNTIME)

        assert result == TASK_MANAGEMENT_PROMPT

    def test_prompt_with_tasks_renders_task_list(self):
        ext = TasksExtension()
        tasks = [{"subject": "Write tests", "status": "in_progress"}]
        result = ext.prompt({"tasks": tasks}, _RUNTIME)

        # Static guidance -> system prompt; live list -> reminder channel.
        assert isinstance(result, dict)
        assert "Write tests" in result["reminder"]
