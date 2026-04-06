"""Tests for Command-based task tools."""

import json

import pytest
from langchain_core.messages import ToolMessage
from langchain_core.tools import ToolException
from langgraph.types import Command

from langchain_agentkit.extensions.tasks.state import _merge_task_pair, _merge_tasks
from langchain_agentkit.extensions.tasks.tools import (
    Task,
    TaskStatus,
    _compute_blocks,
    _task_create,
    _task_get,
    _task_list,
    _task_stop,
    _task_update,
    create_task_tools,
)

FAKE_TOOL_CALL_ID = "call_abc123"


class TestTaskType:
    def test_task_is_typed_dict(self):
        assert hasattr(Task, "__required_keys__") or hasattr(Task, "__annotations__")

    def test_task_has_all_expected_required_keys(self):
        assert Task.__required_keys__ == {
            "id",
            "subject",
            "description",
            "status",
            "active_form",
        }

    def test_task_has_all_expected_optional_keys(self):
        optional = Task.__optional_keys__
        assert "blocked_by" in optional
        assert "blocks" in optional
        assert "owner" in optional
        assert "metadata" in optional

    def test_task_status_is_literal(self):
        assert "pending" in TaskStatus.__args__
        assert "in_progress" in TaskStatus.__args__
        assert "completed" in TaskStatus.__args__
        assert "deleted" in TaskStatus.__args__

    def test_task_create_returns_task_shaped_dict(self):
        result = _task_create(
            subject="Test",
            description="Desc",
            state={"tasks": []},
            tool_call_id=FAKE_TOOL_CALL_ID,
        )
        task = result.update["tasks"][0]
        for key in Task.__required_keys__:
            assert key in task, f"Missing required key: {key}"

    def test_task_importable_from_package(self):
        from langchain_agentkit import Task, TaskStatus

        assert Task is not None
        assert TaskStatus is not None


class TestCreateTaskTools:
    def test_returns_five_tools(self):
        tools = create_task_tools()

        assert len(tools) == 5

    def test_tool_names(self):
        tools = create_task_tools()
        names = [t.name for t in tools]

        assert names == ["TaskCreate", "TaskUpdate", "TaskList", "TaskGet", "TaskStop"]

    def test_tools_have_descriptions(self):
        tools = create_task_tools()

        for tool in tools:
            assert tool.description, f"{tool.name} has no description"

    def test_descriptions_are_detailed(self):
        tools = create_task_tools()
        by_name = {t.name: t for t in tools}

        assert "proactively" in by_name["TaskCreate"].description
        assert "ONLY mark completed" in by_name["TaskUpdate"].description
        assert "Teammate workflow" in by_name["TaskList"].description
        assert "dependencies" in by_name["TaskGet"].description
        assert "running" in by_name["TaskStop"].description.lower()


class TestTaskCreate:
    def test_returns_command(self):
        result = _task_create(
            subject="Test task",
            description="A test",
            state={"tasks": []},
            tool_call_id=FAKE_TOOL_CALL_ID,
        )

        assert isinstance(result, Command)

    def test_appends_task_to_state(self):
        result = _task_create(
            subject="New task",
            description="Do something",
            state={"tasks": []},
            tool_call_id=FAKE_TOOL_CALL_ID,
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
            tool_call_id=FAKE_TOOL_CALL_ID,
        )

        tasks = result.update["tasks"]
        assert len(tasks) == 2
        assert tasks[0]["id"] == "old"

    def test_generates_uuid_id(self):
        result = _task_create(
            subject="Task",
            description="Desc",
            state={},
            tool_call_id=FAKE_TOOL_CALL_ID,
        )

        task_id = result.update["tasks"][0]["id"]
        assert len(task_id) == 36  # UUID format

    def test_stores_active_form(self):
        result = _task_create(
            subject="Analyze",
            description="Deep analysis",
            active_form="Analyzing...",
            state={},
            tool_call_id=FAKE_TOOL_CALL_ID,
        )

        assert result.update["tasks"][0]["active_form"] == "Analyzing..."

    def test_does_not_mutate_original_state(self):
        original_tasks = [{"id": "1", "subject": "Original", "status": "pending"}]
        state = {"tasks": original_tasks}

        _task_create(subject="New", description="Desc", state=state, tool_call_id=FAKE_TOOL_CALL_ID)

        assert len(original_tasks) == 1  # Original list unchanged

    def test_includes_tool_message_in_command(self):
        result = _task_create(
            subject="Test task",
            description="A test",
            state={"tasks": []},
            tool_call_id=FAKE_TOOL_CALL_ID,
        )

        messages = result.update["messages"]
        assert len(messages) == 1
        assert isinstance(messages[0], ToolMessage)
        assert messages[0].tool_call_id == FAKE_TOOL_CALL_ID
        content = json.loads(messages[0].content)
        assert content["subject"] == "Test task"


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
                }
            ]
        }

    @pytest.fixture()
    def state_with_two_tasks(self):
        return {
            "tasks": [
                {"id": "task-1", "subject": "First", "status": "pending", "active_form": ""},
                {"id": "task-2", "subject": "Second", "status": "pending", "active_form": ""},
            ]
        }

    def test_updates_status(self, state_with_task):
        result = _task_update(
            task_id="task-1",
            state=state_with_task,
            tool_call_id=FAKE_TOOL_CALL_ID,
            status="in_progress",
        )

        task = result.update["tasks"][0]
        assert task["status"] == "in_progress"

    def test_updates_subject(self, state_with_task):
        result = _task_update(
            task_id="task-1",
            state=state_with_task,
            tool_call_id=FAKE_TOOL_CALL_ID,
            subject="Updated",
        )

        assert result.update["tasks"][0]["subject"] == "Updated"

    def test_adds_blocked_by(self, state_with_task):
        result = _task_update(
            task_id="task-1",
            state=state_with_task,
            tool_call_id=FAKE_TOOL_CALL_ID,
            add_blocked_by=["task-2"],
        )

        assert "task-2" in result.update["tasks"][0]["blocked_by"]

    def test_adds_blocks(self, state_with_two_tasks):
        result = _task_update(
            task_id="task-1",
            state=state_with_two_tasks,
            tool_call_id=FAKE_TOOL_CALL_ID,
            add_blocks=["task-2"],
        )

        # task-2 should now have task-1 in its blocked_by
        task_2 = next(t for t in result.update["tasks"] if t["id"] == "task-2")
        assert "task-1" in task_2.get("blocked_by", [])

    def test_sets_owner(self, state_with_task):
        result = _task_update(
            task_id="task-1",
            state=state_with_task,
            tool_call_id=FAKE_TOOL_CALL_ID,
            owner="agent-1",
        )

        assert result.update["tasks"][0]["owner"] == "agent-1"

    def test_merges_metadata(self, state_with_task):
        result = _task_update(
            task_id="task-1",
            state=state_with_task,
            tool_call_id=FAKE_TOOL_CALL_ID,
            metadata={"priority": "high", "category": "bug"},
        )

        meta = result.update["tasks"][0]["metadata"]
        assert meta["priority"] == "high"
        assert meta["category"] == "bug"

    def test_metadata_merge_preserves_existing(self, state_with_task):
        # First update: set initial metadata
        state_with_task["tasks"][0]["metadata"] = {"existing": "value"}

        result = _task_update(
            task_id="task-1",
            state=state_with_task,
            tool_call_id=FAKE_TOOL_CALL_ID,
            metadata={"new_key": "new_value"},
        )

        meta = result.update["tasks"][0]["metadata"]
        assert meta["existing"] == "value"
        assert meta["new_key"] == "new_value"

    def test_metadata_null_deletes_key(self, state_with_task):
        state_with_task["tasks"][0]["metadata"] = {"keep": "yes", "remove": "me"}

        result = _task_update(
            task_id="task-1",
            state=state_with_task,
            tool_call_id=FAKE_TOOL_CALL_ID,
            metadata={"remove": None},
        )

        meta = result.update["tasks"][0]["metadata"]
        assert "remove" not in meta
        assert meta["keep"] == "yes"

    def test_raises_on_missing_task(self, state_with_task):
        with pytest.raises(ToolException, match="not found"):
            _task_update(
                task_id="nonexistent",
                state=state_with_task,
                tool_call_id=FAKE_TOOL_CALL_ID,
            )

    def test_returns_command(self, state_with_task):
        result = _task_update(
            task_id="task-1",
            state=state_with_task,
            tool_call_id=FAKE_TOOL_CALL_ID,
            status="completed",
        )

        assert isinstance(result, Command)

    def test_does_not_mutate_original_state(self, state_with_task):
        original_task = state_with_task["tasks"][0]

        _task_update(
            task_id="task-1",
            state=state_with_task,
            tool_call_id=FAKE_TOOL_CALL_ID,
            status="completed",
        )

        assert original_task["status"] == "pending"  # Unchanged

    def test_includes_tool_message_in_command(self, state_with_task):
        result = _task_update(
            task_id="task-1",
            state=state_with_task,
            tool_call_id=FAKE_TOOL_CALL_ID,
            status="completed",
        )

        messages = result.update["messages"]
        assert len(messages) == 1
        assert isinstance(messages[0], ToolMessage)
        assert messages[0].tool_call_id == FAKE_TOOL_CALL_ID
        content = json.loads(messages[0].content)
        assert content["status"] == "completed"

    def test_add_blocks_ignores_missing_target(self, state_with_task):
        # Should not raise when target task doesn't exist
        result = _task_update(
            task_id="task-1",
            state=state_with_task,
            tool_call_id=FAKE_TOOL_CALL_ID,
            add_blocks=["nonexistent"],
        )

        assert isinstance(result, Command)

    def test_blocked_by_deduplicates(self, state_with_task):
        state_with_task["tasks"][0]["blocked_by"] = ["task-2"]

        result = _task_update(
            task_id="task-1",
            state=state_with_task,
            tool_call_id=FAKE_TOOL_CALL_ID,
            add_blocked_by=["task-2", "task-3"],
        )

        blocked = result.update["tasks"][0]["blocked_by"]
        assert blocked == ["task-2", "task-3"]  # no duplicate task-2

    # -- Dependency enforcement (gap #1) --

    def test_rejects_in_progress_with_unresolved_blockers(self):
        state = {
            "tasks": [
                {"id": "task-1", "subject": "Blocker", "status": "pending", "active_form": ""},
                {
                    "id": "task-2",
                    "subject": "Blocked",
                    "status": "pending",
                    "active_form": "",
                    "blocked_by": ["task-1"],
                },
            ]
        }

        with pytest.raises(ToolException, match="blocked"):
            _task_update(
                task_id="task-2",
                state=state,
                tool_call_id=FAKE_TOOL_CALL_ID,
                status="in_progress",
            )

    def test_allows_in_progress_when_blockers_completed(self):
        state = {
            "tasks": [
                {"id": "task-1", "subject": "Blocker", "status": "completed", "active_form": ""},
                {
                    "id": "task-2",
                    "subject": "Was blocked",
                    "status": "pending",
                    "active_form": "",
                    "blocked_by": ["task-1"],
                },
            ]
        }

        result = _task_update(
            task_id="task-2",
            state=state,
            tool_call_id=FAKE_TOOL_CALL_ID,
            status="in_progress",
        )

        assert result.update["tasks"][1]["status"] == "in_progress"

    def test_allows_non_status_update_on_blocked_task(self):
        """Updating subject/description on a blocked task should work."""
        state = {
            "tasks": [
                {"id": "task-1", "subject": "Blocker", "status": "pending", "active_form": ""},
                {
                    "id": "task-2",
                    "subject": "Blocked",
                    "status": "pending",
                    "active_form": "",
                    "blocked_by": ["task-1"],
                },
            ]
        }

        result = _task_update(
            task_id="task-2",
            state=state,
            tool_call_id=FAKE_TOOL_CALL_ID,
            subject="Updated subject",
        )

        assert result.update["tasks"][1]["subject"] == "Updated subject"

    # -- Claim validation (gap #2) --

    def test_rejects_claim_when_already_owned(self):
        state = {
            "tasks": [
                {
                    "id": "task-1",
                    "subject": "Claimed",
                    "status": "in_progress",
                    "active_form": "",
                    "owner": "agent-a",
                },
            ]
        }

        with pytest.raises(ToolException, match="already claimed"):
            _task_update(
                task_id="task-1",
                state=state,
                tool_call_id=FAKE_TOOL_CALL_ID,
                status="in_progress",
                owner="agent-b",
            )

    def test_allows_same_owner_to_reclaim(self):
        state = {
            "tasks": [
                {
                    "id": "task-1",
                    "subject": "Mine",
                    "status": "pending",
                    "active_form": "",
                    "owner": "agent-a",
                },
            ]
        }

        result = _task_update(
            task_id="task-1",
            state=state,
            tool_call_id=FAKE_TOOL_CALL_ID,
            status="in_progress",
            owner="agent-a",
        )

        assert result.update["tasks"][0]["status"] == "in_progress"

    # -- Deletion cascade (gap #5) --

    def test_deletion_cascades_blocked_by_references(self):
        state = {
            "tasks": [
                {"id": "task-1", "subject": "To delete", "status": "pending", "active_form": ""},
                {
                    "id": "task-2",
                    "subject": "Was blocked",
                    "status": "pending",
                    "active_form": "",
                    "blocked_by": ["task-1"],
                },
            ]
        }

        result = _task_update(
            task_id="task-1",
            state=state,
            tool_call_id=FAKE_TOOL_CALL_ID,
            status="deleted",
        )

        task_2 = next(t for t in result.update["tasks"] if t["id"] == "task-2")
        assert "task-1" not in task_2.get("blocked_by", [])

    def test_deletion_cascades_both_blocks_and_blocked_by(self):
        """Task with both blocks and blocked_by refs should cascade both."""
        state = {
            "tasks": [
                {
                    "id": "task-a",
                    "subject": "Upstream",
                    "status": "pending",
                    "active_form": "",
                    "blocks": ["task-1"],
                },
                {
                    "id": "task-1",
                    "subject": "To delete",
                    "status": "pending",
                    "active_form": "",
                    "blocked_by": ["task-a"],
                    "blocks": ["task-b"],
                },
                {
                    "id": "task-b",
                    "subject": "Downstream",
                    "status": "pending",
                    "active_form": "",
                    "blocked_by": ["task-1"],
                },
            ]
        }

        result = _task_update(
            task_id="task-1",
            state=state,
            tool_call_id=FAKE_TOOL_CALL_ID,
            status="deleted",
        )

        task_a = next(t for t in result.update["tasks"] if t["id"] == "task-a")
        task_b = next(t for t in result.update["tasks"] if t["id"] == "task-b")
        assert "task-1" not in task_a.get("blocks", [])
        assert "task-1" not in task_b.get("blocked_by", [])

    def test_deletion_cascades_blocks_references(self):
        state = {
            "tasks": [
                {
                    "id": "task-1",
                    "subject": "To delete",
                    "status": "pending",
                    "active_form": "",
                    "blocks": ["task-2"],
                },
                {
                    "id": "task-2",
                    "subject": "Was blocking",
                    "status": "pending",
                    "active_form": "",
                    "blocked_by": ["task-1"],
                },
            ]
        }

        result = _task_update(
            task_id="task-1",
            state=state,
            tool_call_id=FAKE_TOOL_CALL_ID,
            status="deleted",
        )

        task_2 = next(t for t in result.update["tasks"] if t["id"] == "task-2")
        assert "task-1" not in task_2.get("blocked_by", [])


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

    def test_includes_owner(self):
        state = {
            "tasks": [
                {"id": "1", "subject": "Task", "status": "pending", "owner": "agent-1"},
            ]
        }

        result = _task_list(state=state)
        parsed = json.loads(result)

        assert parsed[0]["owner"] == "agent-1"

    def test_owner_defaults_to_empty_string(self):
        state = {"tasks": [{"id": "1", "subject": "Task", "status": "pending"}]}

        result = _task_list(state=state)
        parsed = json.loads(result)

        assert parsed[0]["owner"] == ""

    # -- Filter resolved blockers (gap #6) --

    def test_filters_resolved_blockers(self):
        state = {
            "tasks": [
                {"id": "1", "subject": "Done", "status": "completed"},
                {"id": "2", "subject": "Still blocking", "status": "pending"},
                {
                    "id": "3",
                    "subject": "Mixed blockers",
                    "status": "pending",
                    "blocked_by": ["1", "2"],
                },
            ]
        }

        result = _task_list(state=state)
        parsed = json.loads(result)

        task_3 = next(t for t in parsed if t["id"] == "3")
        assert task_3["blocked_by"] == ["2"]  # "1" filtered (completed)

    def test_filters_all_resolved_blockers(self):
        """When all blockers completed, blocked_by should be empty."""
        state = {
            "tasks": [
                {"id": "1", "subject": "Done A", "status": "completed"},
                {"id": "2", "subject": "Done B", "status": "completed"},
                {
                    "id": "3",
                    "subject": "Unblocked",
                    "status": "pending",
                    "blocked_by": ["1", "2"],
                },
            ]
        }

        result = _task_list(state=state)
        parsed = json.loads(result)

        task_3 = next(t for t in parsed if t["id"] == "3")
        assert task_3["blocked_by"] == []

    # -- Filter internal tasks (gap #7) --

    def test_filters_internal_tasks(self):
        state = {
            "tasks": [
                {"id": "1", "subject": "Visible", "status": "pending"},
                {
                    "id": "2",
                    "subject": "Internal",
                    "status": "pending",
                    "metadata": {"_internal": True},
                },
            ]
        }

        result = _task_list(state=state)
        parsed = json.loads(result)

        assert len(parsed) == 1
        assert parsed[0]["id"] == "1"


class TestTaskGet:
    def test_returns_full_task_json(self):
        task = {"id": "1", "subject": "Full task", "status": "pending", "description": "Details"}
        state = {"tasks": [task]}

        result = _task_get(task_id="1", state=state)
        parsed = json.loads(result)

        assert parsed["subject"] == "Full task"
        assert parsed["description"] == "Details"

    def test_raises_on_missing_task(self):
        with pytest.raises(ToolException, match="not found"):
            _task_get(task_id="nonexistent", state={"tasks": []})

    def test_includes_computed_blocks(self):
        state = {
            "tasks": [
                {"id": "1", "subject": "Blocker", "status": "pending"},
                {"id": "2", "subject": "Blocked", "status": "pending", "blocked_by": ["1"]},
                {"id": "3", "subject": "Also blocked", "status": "pending", "blocked_by": ["1"]},
            ]
        }

        result = _task_get(task_id="1", state=state)
        parsed = json.loads(result)

        assert "2" in parsed["blocks"]
        assert "3" in parsed["blocks"]

    def test_blocks_excludes_completed_tasks(self):
        state = {
            "tasks": [
                {"id": "1", "subject": "Blocker", "status": "pending"},
                {"id": "2", "subject": "Done", "status": "completed", "blocked_by": ["1"]},
                {"id": "3", "subject": "Still blocked", "status": "pending", "blocked_by": ["1"]},
            ]
        }

        result = _task_get(task_id="1", state=state)
        parsed = json.loads(result)

        assert parsed["blocks"] == ["3"]

    def test_blocks_empty_when_nothing_blocked(self):
        state = {"tasks": [{"id": "1", "subject": "Solo", "status": "pending"}]}

        result = _task_get(task_id="1", state=state)
        parsed = json.loads(result)

        assert parsed["blocks"] == []

    def test_includes_owner_and_metadata(self):
        state = {
            "tasks": [
                {
                    "id": "1",
                    "subject": "Task",
                    "status": "pending",
                    "owner": "agent-1",
                    "metadata": {"priority": "high"},
                },
            ]
        }

        result = _task_get(task_id="1", state=state)
        parsed = json.loads(result)

        assert parsed["owner"] == "agent-1"
        assert parsed["metadata"]["priority"] == "high"


class TestTaskStop:
    def test_stops_in_progress_task(self):
        state = {
            "tasks": [
                {"id": "task-1", "subject": "Running", "status": "in_progress", "active_form": ""},
            ]
        }

        result = _task_stop(task_id="task-1", state=state, tool_call_id=FAKE_TOOL_CALL_ID)

        assert isinstance(result, Command)
        task = result.update["tasks"][0]
        assert task["status"] == "pending"

    def test_returns_stopped_flag_in_message(self):
        state = {
            "tasks": [
                {"id": "task-1", "subject": "Running", "status": "in_progress", "active_form": ""},
            ]
        }

        result = _task_stop(task_id="task-1", state=state, tool_call_id=FAKE_TOOL_CALL_ID)

        content = json.loads(result.update["messages"][0].content)
        assert content["stopped"] is True

    def test_raises_on_missing_task(self):
        with pytest.raises(ToolException, match="not found"):
            _task_stop(task_id="nonexistent", state={"tasks": []}, tool_call_id=FAKE_TOOL_CALL_ID)

    def test_raises_on_non_in_progress_task(self):
        state = {
            "tasks": [
                {"id": "task-1", "subject": "Pending", "status": "pending", "active_form": ""},
            ]
        }

        with pytest.raises(ToolException, match="not in_progress"):
            _task_stop(task_id="task-1", state=state, tool_call_id=FAKE_TOOL_CALL_ID)

    def test_raises_on_completed_task(self):
        state = {
            "tasks": [
                {"id": "task-1", "subject": "Done", "status": "completed", "active_form": ""},
            ]
        }

        with pytest.raises(ToolException, match="not in_progress"):
            _task_stop(task_id="task-1", state=state, tool_call_id=FAKE_TOOL_CALL_ID)


class TestComputeBlocks:
    def test_finds_blocked_tasks(self):
        tasks = [
            {"id": "1", "status": "pending"},
            {"id": "2", "status": "pending", "blocked_by": ["1"]},
        ]

        assert _compute_blocks("1", tasks) == ["2"]

    def test_excludes_completed(self):
        tasks = [
            {"id": "1", "status": "pending"},
            {"id": "2", "status": "completed", "blocked_by": ["1"]},
        ]

        assert _compute_blocks("1", tasks) == []

    def test_excludes_deleted(self):
        tasks = [
            {"id": "1", "status": "pending"},
            {"id": "2", "status": "deleted", "blocked_by": ["1"]},
        ]

        assert _compute_blocks("1", tasks) == []

    def test_empty_when_no_blockers(self):
        tasks = [{"id": "1", "status": "pending"}]

        assert _compute_blocks("1", tasks) == []


class TestMergeTaskPair:
    def test_incoming_scalar_wins(self):
        existing = {"id": "1", "subject": "Old", "status": "pending"}
        incoming = {"id": "1", "subject": "New", "status": "in_progress"}

        result = _merge_task_pair(existing, incoming)

        assert result["subject"] == "New"
        assert result["status"] == "in_progress"

    def test_blocked_by_unioned_and_deduped(self):
        existing = {"id": "1", "blocked_by": ["a", "b"]}
        incoming = {"id": "1", "blocked_by": ["b", "c"]}

        result = _merge_task_pair(existing, incoming)

        assert result["blocked_by"] == ["a", "b", "c"]

    def test_blocks_unioned_and_deduped(self):
        existing = {"id": "1", "blocks": ["x"]}
        incoming = {"id": "1", "blocks": ["x", "y"]}

        result = _merge_task_pair(existing, incoming)

        assert result["blocks"] == ["x", "y"]

    def test_blocked_by_from_empty(self):
        existing = {"id": "1"}
        incoming = {"id": "1", "blocked_by": ["a"]}

        result = _merge_task_pair(existing, incoming)

        assert result["blocked_by"] == ["a"]

    def test_metadata_merged(self):
        existing = {"id": "1", "metadata": {"a": "1", "b": "2"}}
        incoming = {"id": "1", "metadata": {"b": "3", "c": "4"}}

        result = _merge_task_pair(existing, incoming)

        assert result["metadata"] == {"a": "1", "b": "3", "c": "4"}

    def test_metadata_null_deletes(self):
        existing = {"id": "1", "metadata": {"keep": "yes", "remove": "me"}}
        incoming = {"id": "1", "metadata": {"remove": None}}

        result = _merge_task_pair(existing, incoming)

        assert result["metadata"] == {"keep": "yes"}

    def test_metadata_from_empty(self):
        existing = {"id": "1"}
        incoming = {"id": "1", "metadata": {"key": "val"}}

        result = _merge_task_pair(existing, incoming)

        assert result["metadata"] == {"key": "val"}

    def test_does_not_mutate_existing(self):
        existing = {"id": "1", "blocked_by": ["a"]}
        incoming = {"id": "1", "blocked_by": ["b"]}
        original_blocked = list(existing["blocked_by"])

        _merge_task_pair(existing, incoming)

        assert existing["blocked_by"] == original_blocked


class TestMergeTasks:
    def test_new_tasks_appended(self):
        left = [{"id": "1", "subject": "First"}]
        right = [{"id": "2", "subject": "Second"}]

        result = _merge_tasks(left, right)

        assert len(result) == 2
        assert result[0]["id"] == "1"
        assert result[1]["id"] == "2"

    def test_existing_tasks_merged(self):
        left = [{"id": "1", "subject": "Old", "status": "pending"}]
        right = [{"id": "1", "subject": "New", "status": "in_progress"}]

        result = _merge_tasks(left, right)

        assert len(result) == 1
        assert result[0]["subject"] == "New"
        assert result[0]["status"] == "in_progress"

    def test_blocked_by_merged_on_update(self):
        left = [{"id": "1", "blocked_by": ["a"]}]
        right = [{"id": "1", "blocked_by": ["b"]}]

        result = _merge_tasks(left, right)

        assert result[0]["blocked_by"] == ["a", "b"]

    def test_parallel_creates_all_preserved(self):
        left = []
        right = [
            {"id": "1", "subject": "Task A"},
            {"id": "2", "subject": "Task B"},
            {"id": "3", "subject": "Task C"},
        ]

        result = _merge_tasks(left, right)

        assert len(result) == 3

    def test_preserves_insertion_order(self):
        left = [
            {"id": "a", "subject": "Alpha"},
            {"id": "b", "subject": "Beta"},
        ]
        right = [{"id": "c", "subject": "Gamma"}]

        result = _merge_tasks(left, right)

        assert [t["id"] for t in result] == ["a", "b", "c"]

    def test_update_preserves_order(self):
        left = [
            {"id": "a", "subject": "Alpha"},
            {"id": "b", "subject": "Beta"},
        ]
        right = [{"id": "a", "subject": "Updated Alpha"}]

        result = _merge_tasks(left, right)

        assert [t["id"] for t in result] == ["a", "b"]
        assert result[0]["subject"] == "Updated Alpha"

    def test_empty_left(self):
        result = _merge_tasks([], [{"id": "1", "subject": "New"}])

        assert len(result) == 1

    def test_empty_right(self):
        result = _merge_tasks([{"id": "1", "subject": "Existing"}], [])

        assert len(result) == 1

    def test_both_empty(self):
        assert _merge_tasks([], []) == []

    def test_none_left(self):
        result = _merge_tasks(None, [{"id": "1", "subject": "New"}])

        assert len(result) == 1

    def test_none_right(self):
        result = _merge_tasks([{"id": "1", "subject": "Existing"}], None)

        assert len(result) == 1

    def test_concurrent_blocked_by_updates_merged(self):
        """Simulates two parallel TaskUpdate calls adding different blocked_by."""
        left = [{"id": "1", "status": "pending", "blocked_by": ["a"]}]
        # First update adds "b"
        right_1 = [{"id": "1", "status": "pending", "blocked_by": ["a", "b"]}]
        # Second update adds "c"
        right_2 = [{"id": "1", "status": "pending", "blocked_by": ["a", "c"]}]

        # Simulate: left + right_1, then result + right_2
        intermediate = _merge_tasks(left, right_1)
        final = _merge_tasks(intermediate, right_2)

        assert set(final[0]["blocked_by"]) == {"a", "b", "c"}

    def test_skips_tasks_without_id(self):
        result = _merge_tasks(
            [{"subject": "No ID"}],
            [{"id": "1", "subject": "Has ID"}],
        )

        assert len(result) == 1
        assert result[0]["id"] == "1"
