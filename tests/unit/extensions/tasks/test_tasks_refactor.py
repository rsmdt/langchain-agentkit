"""Tests for tx1 refactor: TasksExtension no longer emits universal guidance."""

from __future__ import annotations

import pytest
from langgraph.prebuilt import ToolRuntime

from langchain_agentkit.extensions.tasks import (
    TASK_MANAGEMENT_PROMPT,
    TasksExtension,
)
from langchain_agentkit.extensions.tasks import extension as tasks_ext_module

_RUNTIME = ToolRuntime(
    state={},
    context=None,
    config={},
    stream_writer=lambda _: None,
    tool_call_id=None,
    store=None,
)


class TestBaseAgentPromptRemoved:
    def test_base_agent_prompt_constant_removed_from_module(self):
        assert not hasattr(tasks_ext_module, "BASE_AGENT_PROMPT")

    def test_base_agent_prompt_not_reexported(self):
        from langchain_agentkit.extensions import tasks as tasks_pkg

        assert not hasattr(tasks_pkg, "BASE_AGENT_PROMPT")
        assert "BASE_AGENT_PROMPT" not in tasks_pkg.__all__

    def test_base_agent_prompt_file_deleted(self):
        from pathlib import Path

        path = Path(tasks_ext_module.__file__).parent / "base_agent_prompt.md"
        assert not path.exists()


class TestTasksExtensionPromptContent:
    def test_prompt_with_no_tasks_is_task_management_only(self):
        ext = TasksExtension()
        result = ext.prompt({"tasks": []}, _RUNTIME)

        assert result == TASK_MANAGEMENT_PROMPT

    def test_prompt_without_tasks_key_is_task_management_only(self):
        ext = TasksExtension()
        result = ext.prompt({}, _RUNTIME)

        assert result == TASK_MANAGEMENT_PROMPT

    def test_prompt_does_not_contain_core_behavior_guidance(self):
        ext = TasksExtension()
        result = ext.prompt({"tasks": []}, _RUNTIME)

        # Universal guidance now lives in CoreBehaviorExtension.
        assert "Core Behavior" not in result
        assert "NEVER add unnecessary preamble" not in result

    def test_prompt_with_tasks_renders_task_list(self):
        ext = TasksExtension()
        tasks = [{"subject": "Write tests", "status": "in_progress"}]
        result = ext.prompt({"tasks": tasks}, _RUNTIME)

        assert "Write tests" in result
        assert "Core Behavior" not in result


class TestTaskManagementPromptIsDomainNeutral:
    @pytest.mark.parametrize(
        "forbidden",
        [
            "React",
            "Vue",
            "Svelte",
            "GitHub Actions",
            "CI/CD",
            "navbar",
            "logout button",
        ],
    )
    def test_no_software_engineering_specific_examples(self, forbidden):
        assert forbidden not in TASK_MANAGEMENT_PROMPT
