"""Tests for Command-based task tools."""

import json

import pytest
from langgraph.types import Command

from langchain_agentkit.task_tools import (
    _task_create,
    _task_get,
    _task_list,
    _task_update,
    create_task_tools,
)


class TestCreateTaskTools:
    def test_returns_four_tools(self):
        tools = create_task_tools()

        assert len(tools) == 4

    def test_tool_names(self):
        tools = create_task_tools()
        names = [t.name for t in tools]

        assert names == ["TaskCreate", "TaskUpdate", "TaskList", "TaskGet"]

    def test_tools_have_descriptions(self):
        tools = create_task_tools()

        for tool in tools:
            assert tool.description, f"{tool.name} has no description"


class TestTaskCreate:
    def test_returns_command(self):
        result = _task_create(
            subject="Test task",
            description="A test",
            state={"tasks": []},
        )

        assert isinstance(result, Command)

    def test_appends_task_to_state(self):
        result = _task_create(
            subject="New task",
            description="Do something",
            state={"tasks": []},
        )

        tasks = result.update["tasks"]
        assert len(tasks) == 1
        assert tasks[0]["subject"] == "New task"
        assert tasks[0]["description"] == "Do something"
        assert tasks[0]["status"] == "pending"

    def test_preserves_existing_tasks(self):
        existing = [{"id": "old", "subject": "Old task", "status": "completed"}]

        result = _task_create(
            subject="New task",
            description="Another",
            state={"tasks": existing},
        )

        tasks = result.update["tasks"]
        assert len(tasks) == 2
        assert tasks[0]["id"] == "old"

    def test_generates_uuid_id(self):
        result = _task_create(
            subject="Task",
            description="Desc",
            state={},
        )

        task_id = result.update["tasks"][0]["id"]
        assert len(task_id) == 36  # UUID format

    def test_stores_active_form(self):
        result = _task_create(
            subject="Analyze",
            description="Deep analysis",
            active_form="Analyzing...",
            state={},
        )

        assert result.update["tasks"][0]["active_form"] == "Analyzing..."

    def test_stores_metadata(self):
        result = _task_create(
            subject="Task",
            description="Desc",
            metadata={"priority": "high"},
            state={},
        )

        assert result.update["tasks"][0]["metadata"] == {"priority": "high"}

    def test_does_not_mutate_original_state(self):
        original_tasks = [{"id": "1", "subject": "Original", "status": "pending"}]
        state = {"tasks": original_tasks}

        _task_create(subject="New", description="Desc", state=state)

        assert len(original_tasks) == 1  # Original list unchanged


class TestTaskUpdate:
    @pytest.fixture()
    def state_with_task(self):
        return {
            "tasks": [
                {
                    "id": "task-1",
                    "subject": "Original",
                    "description": "Original desc",
                    "status": "pending",
                    "active_form": "",
                    "owner": None,
                    "blocks": [],
                    "blocked_by": [],
                    "metadata": {},
                }
            ]
        }

    def test_updates_status(self, state_with_task):
        result = _task_update(task_id="task-1", state=state_with_task, status="in_progress")

        task = result.update["tasks"][0]
        assert task["status"] == "in_progress"

    def test_updates_subject(self, state_with_task):
        result = _task_update(task_id="task-1", state=state_with_task, subject="Updated")

        assert result.update["tasks"][0]["subject"] == "Updated"

    def test_updates_owner(self, state_with_task):
        result = _task_update(task_id="task-1", state=state_with_task, owner="researcher")

        assert result.update["tasks"][0]["owner"] == "researcher"

    def test_adds_blocked_by(self, state_with_task):
        result = _task_update(task_id="task-1", state=state_with_task, add_blocked_by=["task-2"])

        assert "task-2" in result.update["tasks"][0]["blocked_by"]

    def test_adds_blocks(self, state_with_task):
        result = _task_update(task_id="task-1", state=state_with_task, add_blocks=["task-3"])

        assert "task-3" in result.update["tasks"][0]["blocks"]

    def test_merges_metadata(self, state_with_task):
        state_with_task["tasks"][0]["metadata"] = {"existing": "value"}

        result = _task_update(
            task_id="task-1", state=state_with_task, metadata={"new_key": "new_value"}
        )

        meta = result.update["tasks"][0]["metadata"]
        assert meta["existing"] == "value"
        assert meta["new_key"] == "new_value"

    def test_deletes_metadata_keys_with_none(self, state_with_task):
        state_with_task["tasks"][0]["metadata"] = {"keep": "yes", "remove": "this"}

        result = _task_update(task_id="task-1", state=state_with_task, metadata={"remove": None})

        meta = result.update["tasks"][0]["metadata"]
        assert "keep" in meta
        assert "remove" not in meta

    def test_raises_on_missing_task(self, state_with_task):
        from langchain_core.tools import ToolException

        with pytest.raises(ToolException, match="not found"):
            _task_update(task_id="nonexistent", state=state_with_task)

    def test_returns_command(self, state_with_task):
        result = _task_update(task_id="task-1", state=state_with_task, status="completed")

        assert isinstance(result, Command)

    def test_does_not_mutate_original_state(self, state_with_task):
        original_task = state_with_task["tasks"][0]

        _task_update(task_id="task-1", state=state_with_task, status="completed")

        assert original_task["status"] == "pending"  # Unchanged


class TestTaskList:
    def test_returns_json_string(self):
        state = {"tasks": [{"id": "1", "subject": "Task", "status": "pending"}]}

        result = _task_list(state=state)
        parsed = json.loads(result)

        assert len(parsed) == 1
        assert parsed[0]["subject"] == "Task"

    def test_filters_deleted_tasks(self):
        state = {
            "tasks": [
                {"id": "1", "subject": "Visible", "status": "pending"},
                {"id": "2", "subject": "Hidden", "status": "deleted"},
            ]
        }

        result = _task_list(state=state)
        parsed = json.loads(result)

        assert len(parsed) == 1
        assert parsed[0]["subject"] == "Visible"

    def test_empty_tasks_returns_empty_list(self):
        result = _task_list(state={})
        parsed = json.loads(result)

        assert parsed == []

    def test_includes_blocked_by(self):
        state = {
            "tasks": [
                {"id": "1", "subject": "Task", "status": "pending", "blocked_by": ["other"]},
            ]
        }

        result = _task_list(state=state)
        parsed = json.loads(result)

        assert parsed[0]["blocked_by"] == ["other"]


class TestTaskGet:
    def test_returns_full_task_json(self):
        task = {"id": "1", "subject": "Full task", "status": "pending", "description": "Details"}
        state = {"tasks": [task]}

        result = _task_get(task_id="1", state=state)
        parsed = json.loads(result)

        assert parsed["subject"] == "Full task"
        assert parsed["description"] == "Details"

    def test_raises_on_missing_task(self):
        from langchain_core.tools import ToolException

        with pytest.raises(ToolException, match="not found"):
            _task_get(task_id="nonexistent", state={"tasks": []})
