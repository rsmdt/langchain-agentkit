"""Tests for team task proxy system.

Covers:
- task_router: try_parse_task_op, process_task_op, classify_and_process
- task_proxy: proxy tools via bus round-trip
- bus: request_response method
"""

import asyncio
import json

import pytest

from langchain_agentkit.extensions.teams.bus import TeamMessageBus
from langchain_agentkit.extensions.teams.task_proxy import (
    TASK_OP_TYPE,
    _proxy_task_create,
    _proxy_task_get,
    _proxy_task_list,
    _proxy_task_update,
    create_task_proxy_tools,
)
from langchain_agentkit.extensions.teams.task_router import (
    classify_and_process,
    process_task_op,
    try_parse_task_op,
)

# ---------------------------------------------------------------------------
# try_parse_task_op
# ---------------------------------------------------------------------------


class TestTryParseTaskOp:
    def test_valid_task_op(self):
        content = json.dumps({"type": TASK_OP_TYPE, "op": "create", "request_id": "r1"})
        result = try_parse_task_op(content)
        assert result is not None
        assert result["op"] == "create"

    def test_non_json(self):
        assert try_parse_task_op("hello world") is None

    def test_wrong_type(self):
        content = json.dumps({"type": "not_task_op", "op": "create", "request_id": "r1"})
        assert try_parse_task_op(content) is None

    def test_missing_op(self):
        content = json.dumps({"type": TASK_OP_TYPE, "request_id": "r1"})
        assert try_parse_task_op(content) is None

    def test_missing_request_id(self):
        content = json.dumps({"type": TASK_OP_TYPE, "op": "create"})
        assert try_parse_task_op(content) is None

    def test_non_dict_json(self):
        assert try_parse_task_op(json.dumps([1, 2, 3])) is None

    def test_empty_string(self):
        assert try_parse_task_op("") is None

    def test_shutdown_request_not_task_op(self):
        content = json.dumps({"type": "shutdown_request"})
        assert try_parse_task_op(content) is None


# ---------------------------------------------------------------------------
# process_task_op — create
# ---------------------------------------------------------------------------


class TestProcessTaskOpCreate:
    def test_create_task(self):
        op = {
            "type": TASK_OP_TYPE,
            "op": "create",
            "request_id": "r1",
            "subject": "Fix bug",
            "description": "In login flow",
        }
        ack_json, tasks = process_task_op(op, [])
        ack = json.loads(ack_json)

        assert ack["request_id"] == "r1"
        assert "task" in ack
        assert ack["task"]["subject"] == "Fix bug"
        assert ack["task"]["status"] == "pending"
        assert len(tasks) == 1
        assert tasks[0]["id"] == ack["task"]["id"]

    def test_create_preserves_existing_tasks(self):
        existing = [{"id": "t1", "subject": "Old", "status": "completed"}]
        op = {"type": TASK_OP_TYPE, "op": "create", "request_id": "r2", "subject": "New"}
        _, tasks = process_task_op(op, existing)
        assert len(tasks) == 2
        assert tasks[0]["id"] == "t1"


# ---------------------------------------------------------------------------
# process_task_op — update
# ---------------------------------------------------------------------------


class TestProcessTaskOpUpdate:
    def test_update_status(self):
        tasks = [{"id": "t1", "subject": "Fix bug", "status": "pending"}]
        op = {
            "type": TASK_OP_TYPE,
            "op": "update",
            "request_id": "r1",
            "task_id": "t1",
            "status": "in_progress",
        }
        ack_json, updated = process_task_op(op, tasks)
        ack = json.loads(ack_json)

        assert ack["task"]["status"] == "in_progress"
        assert updated[0]["status"] == "in_progress"

    def test_update_owner(self):
        tasks = [{"id": "t1", "subject": "Fix bug", "status": "pending"}]
        op = {
            "type": TASK_OP_TYPE,
            "op": "update",
            "request_id": "r1",
            "task_id": "t1",
            "owner": "researcher",
        }
        _, updated = process_task_op(op, tasks)
        assert updated[0]["owner"] == "researcher"

    def test_update_missing_task(self):
        op = {
            "type": TASK_OP_TYPE,
            "op": "update",
            "request_id": "r1",
            "task_id": "nonexistent",
            "status": "completed",
        }
        ack_json, _ = process_task_op(op, [])
        ack = json.loads(ack_json)
        assert "error" in ack

    def test_update_missing_task_id(self):
        op = {"type": TASK_OP_TYPE, "op": "update", "request_id": "r1"}
        ack_json, _ = process_task_op(op, [])
        ack = json.loads(ack_json)
        assert "error" in ack

    def test_update_metadata_merge(self):
        tasks = [{"id": "t1", "subject": "X", "status": "pending", "metadata": {"a": 1}}]
        op = {
            "type": TASK_OP_TYPE,
            "op": "update",
            "request_id": "r1",
            "task_id": "t1",
            "metadata": {"b": 2, "a": None},
        }
        _, updated = process_task_op(op, tasks)
        assert updated[0]["metadata"] == {"b": 2}

    def test_update_add_blocked_by(self):
        tasks = [
            {"id": "t1", "subject": "A", "status": "pending"},
            {"id": "t2", "subject": "B", "status": "pending"},
        ]
        op = {
            "type": TASK_OP_TYPE,
            "op": "update",
            "request_id": "r1",
            "task_id": "t2",
            "add_blocked_by": ["t1"],
        }
        _, updated = process_task_op(op, tasks)
        assert "t1" in updated[1]["blocked_by"]

    def test_update_add_blocks(self):
        tasks = [
            {"id": "t1", "subject": "A", "status": "pending"},
            {"id": "t2", "subject": "B", "status": "pending"},
        ]
        op = {
            "type": TASK_OP_TYPE,
            "op": "update",
            "request_id": "r1",
            "task_id": "t1",
            "add_blocks": ["t2"],
        }
        _, updated = process_task_op(op, tasks)
        assert "t1" in updated[1].get("blocked_by", [])


# ---------------------------------------------------------------------------
# process_task_op — list
# ---------------------------------------------------------------------------


class TestProcessTaskOpList:
    def test_list_tasks(self):
        tasks = [
            {"id": "t1", "subject": "A", "status": "pending"},
            {"id": "t2", "subject": "B", "status": "completed"},
            {"id": "t3", "subject": "C", "status": "deleted"},
        ]
        ack_json, _ = process_task_op(
            {"type": TASK_OP_TYPE, "op": "list", "request_id": "r1"}, tasks,
        )
        ack = json.loads(ack_json)
        assert len(ack["tasks"]) == 2  # excludes deleted

    def test_list_empty(self):
        ack_json, _ = process_task_op(
            {"type": TASK_OP_TYPE, "op": "list", "request_id": "r1"}, [],
        )
        ack = json.loads(ack_json)
        assert ack["tasks"] == []


# ---------------------------------------------------------------------------
# process_task_op — get
# ---------------------------------------------------------------------------


class TestProcessTaskOpGet:
    def test_get_task(self):
        tasks = [{"id": "t1", "subject": "A", "status": "pending"}]
        ack_json, _ = process_task_op(
            {"type": TASK_OP_TYPE, "op": "get", "request_id": "r1", "task_id": "t1"}, tasks,
        )
        ack = json.loads(ack_json)
        assert ack["task"]["id"] == "t1"
        assert "blocks" in ack["task"]

    def test_get_missing_task(self):
        ack_json, _ = process_task_op(
            {"type": TASK_OP_TYPE, "op": "get", "request_id": "r1", "task_id": "nope"}, [],
        )
        ack = json.loads(ack_json)
        assert "error" in ack

    def test_get_computes_blocks(self):
        tasks = [
            {"id": "t1", "subject": "A", "status": "pending"},
            {"id": "t2", "subject": "B", "status": "pending", "blocked_by": ["t1"]},
        ]
        ack_json, _ = process_task_op(
            {"type": TASK_OP_TYPE, "op": "get", "request_id": "r1", "task_id": "t1"}, tasks,
        )
        ack = json.loads(ack_json)
        assert "t2" in ack["task"]["blocks"]


# ---------------------------------------------------------------------------
# process_task_op — unknown
# ---------------------------------------------------------------------------


class TestProcessTaskOpUpdateValidation:
    """Tests for dependency enforcement and claim validation in the router."""

    def test_rejects_in_progress_with_unresolved_blockers(self):
        tasks = [
            {"id": "t1", "subject": "Blocker", "status": "pending"},
            {"id": "t2", "subject": "Blocked", "status": "pending", "blocked_by": ["t1"]},
        ]
        op = {
            "type": TASK_OP_TYPE,
            "op": "update",
            "request_id": "r1",
            "task_id": "t2",
            "status": "in_progress",
        }
        ack_json, _ = process_task_op(op, tasks)
        ack = json.loads(ack_json)
        assert "error" in ack
        assert "blocked" in ack["error"].lower()

    def test_allows_in_progress_when_blockers_completed(self):
        tasks = [
            {"id": "t1", "subject": "Blocker", "status": "completed"},
            {"id": "t2", "subject": "Unblocked", "status": "pending", "blocked_by": ["t1"]},
        ]
        op = {
            "type": TASK_OP_TYPE,
            "op": "update",
            "request_id": "r1",
            "task_id": "t2",
            "status": "in_progress",
        }
        ack_json, updated = process_task_op(op, tasks)
        ack = json.loads(ack_json)
        assert "error" not in ack
        assert updated[1]["status"] == "in_progress"

    def test_rejects_claim_when_already_owned(self):
        tasks = [
            {"id": "t1", "subject": "Claimed", "status": "in_progress", "owner": "alice"},
        ]
        op = {
            "type": TASK_OP_TYPE,
            "op": "update",
            "request_id": "r1",
            "task_id": "t1",
            "status": "in_progress",
            "owner": "bob",
        }
        ack_json, _ = process_task_op(op, tasks)
        ack = json.loads(ack_json)
        assert "error" in ack
        assert "claimed" in ack["error"].lower()

    def test_allows_same_owner_to_reclaim(self):
        tasks = [
            {"id": "t1", "subject": "Mine", "status": "pending", "owner": "alice"},
        ]
        op = {
            "type": TASK_OP_TYPE,
            "op": "update",
            "request_id": "r1",
            "task_id": "t1",
            "status": "in_progress",
            "owner": "alice",
        }
        ack_json, updated = process_task_op(op, tasks)
        ack = json.loads(ack_json)
        assert "error" not in ack
        assert updated[0]["status"] == "in_progress"

    def test_auto_owner_on_in_progress_via_process_task_op(self):
        """Auto-owner works when sender is injected by classify_and_process."""
        tasks = [{"id": "t1", "subject": "Unclaimed", "status": "pending"}]
        op = {
            "type": TASK_OP_TYPE,
            "op": "update",
            "request_id": "r1",
            "task_id": "t1",
            "status": "in_progress",
            "sender": "alice",  # Injected by classify_and_process from m.sender
        }
        _, updated = process_task_op(op, tasks)
        assert updated[0]["owner"] == "alice"

    @pytest.mark.asyncio
    async def test_auto_owner_via_classify_and_process(self):
        """classify_and_process injects sender so auto-owner works end-to-end."""
        bus = TeamMessageBus()
        bus.register("lead")
        bus.register("alice")

        from langchain_agentkit.extensions.teams.bus import TeamMessage

        msg = TeamMessage(
            id="m1",
            sender="alice",
            receiver="lead",
            content=json.dumps({
                "type": TASK_OP_TYPE,
                "op": "update",
                "request_id": "r1",
                "task_id": "t1",
                "status": "in_progress",
            }),
            timestamp=1.0,
        )

        tasks = [{"id": "t1", "subject": "Task", "status": "pending"}]
        result = await classify_and_process([msg], tasks, bus)

        assert result["tasks"][0]["owner"] == "alice"

    def test_deletion_cascades_references(self):
        tasks = [
            {"id": "t1", "subject": "To delete", "status": "pending"},
            {"id": "t2", "subject": "Was blocked", "status": "pending", "blocked_by": ["t1"]},
        ]
        op = {
            "type": TASK_OP_TYPE,
            "op": "update",
            "request_id": "r1",
            "task_id": "t1",
            "status": "deleted",
        }
        _, updated = process_task_op(op, tasks)
        t2 = next(t for t in updated if t["id"] == "t2")
        assert "t1" not in t2.get("blocked_by", [])


class TestProcessTaskOpListFiltering:
    """Tests for resolved blocker filtering and internal task hiding."""

    def test_filters_resolved_blockers(self):
        tasks = [
            {"id": "t1", "subject": "Done", "status": "completed"},
            {"id": "t2", "subject": "Still blocking", "status": "pending"},
            {"id": "t3", "subject": "Mixed", "status": "pending", "blocked_by": ["t1", "t2"]},
        ]
        ack_json, _ = process_task_op(
            {"type": TASK_OP_TYPE, "op": "list", "request_id": "r1"}, tasks,
        )
        ack = json.loads(ack_json)
        t3 = next(t for t in ack["tasks"] if t["id"] == "t3")
        assert t3["blocked_by"] == ["t2"]

    def test_filters_internal_tasks(self):
        tasks = [
            {"id": "t1", "subject": "Visible", "status": "pending"},
            {
                "id": "t2", "subject": "Internal", "status": "pending",
                "metadata": {"_internal": True},
            },
        ]
        ack_json, _ = process_task_op(
            {"type": TASK_OP_TYPE, "op": "list", "request_id": "r1"}, tasks,
        )
        ack = json.loads(ack_json)
        assert len(ack["tasks"]) == 1
        assert ack["tasks"][0]["id"] == "t1"


class TestProcessTaskOpCompletionNudge:
    """Test that completing a task returns a work discovery nudge."""

    def test_completion_includes_nudge(self):
        tasks = [{"id": "t1", "subject": "Do thing", "status": "in_progress", "owner": "alice"}]
        op = {
            "type": TASK_OP_TYPE,
            "op": "update",
            "request_id": "r1",
            "task_id": "t1",
            "status": "completed",
            "sender": "alice",
        }
        ack_json, _ = process_task_op(op, tasks)
        ack = json.loads(ack_json)
        assert "nudge" in ack or "TaskList" in ack.get("message", "")


class TestAssignmentNotification:
    """Test that owner changes trigger notifications via bus."""

    @pytest.mark.asyncio
    async def test_assignment_sends_notification(self):
        bus = TeamMessageBus()
        bus.register("lead")
        bus.register("alice")

        from langchain_agentkit.extensions.teams.bus import TeamMessage

        task_msg = TeamMessage(
            id="m1",
            sender="lead",
            receiver="lead",
            content=json.dumps({
                "type": TASK_OP_TYPE,
                "op": "update",
                "request_id": "r1",
                "task_id": "t1",
                "owner": "alice",
            }),
            timestamp=1.0,
        )

        tasks = [{"id": "t1", "subject": "Do X", "status": "pending"}]
        await classify_and_process([task_msg], tasks, bus)

        # Alice should have received a task_assignment notification
        msg = await bus.receive("alice", timeout=1.0)
        assert msg is not None
        parsed = json.loads(msg.content)
        assert parsed["type"] == "task_assignment"
        assert parsed["task_id"] == "t1"


class TestAssignmentNotificationEdgeCases:
    @pytest.mark.asyncio
    async def test_notification_to_unregistered_member_does_not_crash(self):
        """Assigning to a name not on the bus should not raise."""
        bus = TeamMessageBus()
        bus.register("lead")
        # "ghost" is NOT registered

        from langchain_agentkit.extensions.teams.bus import TeamMessage

        task_msg = TeamMessage(
            id="m1",
            sender="lead",
            receiver="lead",
            content=json.dumps({
                "type": TASK_OP_TYPE,
                "op": "update",
                "request_id": "r1",
                "task_id": "t1",
                "owner": "ghost",
            }),
            timestamp=1.0,
        )

        tasks = [{"id": "t1", "subject": "Task", "status": "pending"}]
        result = await classify_and_process([task_msg], tasks, bus)

        # Update should succeed even though notification failed
        assert result["tasks"][0]["owner"] == "ghost"


class TestProcessTaskOpUnknown:
    def test_unknown_op(self):
        ack_json, _ = process_task_op(
            {"type": TASK_OP_TYPE, "op": "explode", "request_id": "r1"}, [],
        )
        ack = json.loads(ack_json)
        assert "error" in ack


# ---------------------------------------------------------------------------
# TeamMessageBus.request_response
# ---------------------------------------------------------------------------


class TestBusRequestResponse:
    @pytest.mark.asyncio
    async def test_matching_response(self):
        bus = TeamMessageBus()
        bus.register("alice")
        bus.register("lead")

        req_id = "req-123"

        # Simulate router responding after a short delay
        async def _respond():
            await asyncio.sleep(0.05)
            msg = await bus.receive("lead", timeout=2.0)
            assert msg is not None
            response = json.dumps({"request_id": req_id, "task": {"id": "t1"}})
            await bus.send("lead", "alice", response)

        asyncio.create_task(_respond())

        payload = json.dumps({"type": TASK_OP_TYPE, "op": "create", "request_id": req_id})
        result = await bus.request_response(
            "alice", "lead", payload, request_id=req_id, timeout=5.0,
        )

        assert result is not None
        parsed = json.loads(result.content)
        assert parsed["request_id"] == req_id

    @pytest.mark.asyncio
    async def test_timeout(self):
        bus = TeamMessageBus()
        bus.register("alice")
        bus.register("lead")

        payload = json.dumps({"type": TASK_OP_TYPE, "op": "list", "request_id": "r1"})
        result = await bus.request_response("alice", "lead", payload, request_id="r1", timeout=0.1)
        assert result is None

    @pytest.mark.asyncio
    async def test_stashed_messages_requeued(self):
        bus = TeamMessageBus()
        bus.register("alice")
        bus.register("lead")

        req_id = "req-456"

        async def _respond():
            await asyncio.sleep(0.02)
            # Drain the task op from lead's queue
            await bus.receive("lead", timeout=1.0)
            # Send a non-matching message first
            await bus.send("lead", "alice", "hello from bob")
            # Then the matching response
            await asyncio.sleep(0.02)
            await bus.send("lead", "alice", json.dumps({"request_id": req_id, "ok": True}))

        asyncio.create_task(_respond())

        payload = json.dumps({"type": TASK_OP_TYPE, "op": "list", "request_id": req_id})
        result = await bus.request_response(
            "alice", "lead", payload, request_id=req_id, timeout=5.0,
        )

        assert result is not None
        # The non-matching message should be re-queued
        stashed = await bus.receive("alice", timeout=0.5)
        assert stashed is not None
        assert stashed.content == "hello from bob"


# ---------------------------------------------------------------------------
# classify_and_process
# ---------------------------------------------------------------------------


class TestClassifyAndProcess:
    @pytest.mark.asyncio
    async def test_mixed_messages(self):
        bus = TeamMessageBus()
        bus.register("lead")
        bus.register("alice")

        # Build fake TeamMessages
        from langchain_agentkit.extensions.teams.bus import TeamMessage

        task_msg = TeamMessage(
            id="m1",
            sender="alice",
            receiver="lead",
            content=json.dumps({
                "type": TASK_OP_TYPE,
                "op": "create",
                "request_id": "r1",
                "subject": "Do thing",
            }),
            timestamp=1.0,
        )
        text_msg = TeamMessage(
            id="m2",
            sender="alice",
            receiver="lead",
            content="I finished the analysis",
            timestamp=2.0,
        )

        result = await classify_and_process([task_msg, text_msg], [], bus)

        # Task op should produce a tasks update
        assert "tasks" in result
        assert len(result["tasks"]) == 1
        assert result["tasks"][0]["subject"] == "Do thing"

        # Text message should become a HumanMessage
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert "I finished the analysis" in result["messages"][0].content

        # Ack should have been sent to alice
        ack = await bus.receive("alice", timeout=1.0)
        assert ack is not None
        parsed = json.loads(ack.content)
        assert parsed["request_id"] == "r1"

    @pytest.mark.asyncio
    async def test_only_task_ops(self):
        bus = TeamMessageBus()
        bus.register("lead")
        bus.register("bob")

        from langchain_agentkit.extensions.teams.bus import TeamMessage

        msg = TeamMessage(
            id="m1",
            sender="bob",
            receiver="lead",
            content=json.dumps({
                "type": TASK_OP_TYPE,
                "op": "list",
                "request_id": "r2",
            }),
            timestamp=1.0,
        )

        result = await classify_and_process([msg], [], bus)
        assert "tasks" not in result or result["tasks"] == []
        assert "messages" not in result

    @pytest.mark.asyncio
    async def test_only_text_messages(self):
        bus = TeamMessageBus()
        bus.register("lead")

        from langchain_agentkit.extensions.teams.bus import TeamMessage

        msg = TeamMessage(
            id="m1", sender="alice", receiver="lead",
            content="just a normal message", timestamp=1.0,
        )

        result = await classify_and_process([msg], [], bus)
        assert "messages" in result
        assert "tasks" not in result


# ---------------------------------------------------------------------------
# create_task_proxy_tools
# ---------------------------------------------------------------------------


class TestCreateTaskProxyTools:
    def test_returns_four_tools(self):
        bus = TeamMessageBus()
        bus.register("alice")
        bus.register("lead")

        tools = create_task_proxy_tools(bus, "alice")
        names = [t.name for t in tools]

        assert len(tools) == 4
        assert "TaskCreate" in names
        assert "TaskUpdate" in names
        assert "TaskList" in names
        assert "TaskGet" in names


# ---------------------------------------------------------------------------
# Proxy tool integration (proxy → bus → router processing)
# ---------------------------------------------------------------------------


class TestProxyToolIntegration:
    @pytest.mark.asyncio
    async def test_proxy_create_roundtrip(self):
        """Proxy sends create op, simulated router processes and responds."""
        bus = TeamMessageBus()
        bus.register("alice")
        bus.register("lead")

        async def _router_sim():
            msg = await bus.receive("lead", timeout=5.0)
            assert msg is not None
            parsed = json.loads(msg.content)
            ack, _ = process_task_op(parsed, [])
            await bus.send("lead", msg.sender, ack)

        asyncio.create_task(_router_sim())

        result = await _proxy_task_create(
            subject="Write tests",
            description="For the proxy system",
            bus=bus,
            member_name="alice",
        )
        parsed = json.loads(result)
        assert "task" in parsed
        assert parsed["task"]["subject"] == "Write tests"

    @pytest.mark.asyncio
    async def test_proxy_update_roundtrip(self):
        bus = TeamMessageBus()
        bus.register("alice")
        bus.register("lead")

        existing_tasks = [{"id": "t1", "subject": "Do X", "status": "pending"}]

        async def _router_sim():
            msg = await bus.receive("lead", timeout=5.0)
            parsed = json.loads(msg.content)
            ack, _ = process_task_op(parsed, existing_tasks)
            await bus.send("lead", msg.sender, ack)

        asyncio.create_task(_router_sim())

        result = await _proxy_task_update(
            task_id="t1",
            status="completed",
            bus=bus,
            member_name="alice",
        )
        parsed = json.loads(result)
        assert parsed["task"]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_proxy_list_roundtrip(self):
        bus = TeamMessageBus()
        bus.register("alice")
        bus.register("lead")

        existing = [{"id": "t1", "subject": "A", "status": "pending"}]

        async def _router_sim():
            msg = await bus.receive("lead", timeout=5.0)
            parsed = json.loads(msg.content)
            ack, _ = process_task_op(parsed, existing)
            await bus.send("lead", msg.sender, ack)

        asyncio.create_task(_router_sim())

        result = await _proxy_task_list(bus=bus, member_name="alice")
        parsed = json.loads(result)
        assert len(parsed["tasks"]) == 1

    @pytest.mark.asyncio
    async def test_proxy_get_roundtrip(self):
        bus = TeamMessageBus()
        bus.register("alice")
        bus.register("lead")

        existing = [{"id": "t1", "subject": "A", "status": "pending"}]

        async def _router_sim():
            msg = await bus.receive("lead", timeout=5.0)
            parsed = json.loads(msg.content)
            ack, _ = process_task_op(parsed, existing)
            await bus.send("lead", msg.sender, ack)

        asyncio.create_task(_router_sim())

        result = await _proxy_task_get(task_id="t1", bus=bus, member_name="alice")
        parsed = json.loads(result)
        assert parsed["task"]["id"] == "t1"

    @pytest.mark.asyncio
    async def test_proxy_timeout(self):
        """Proxy times out when no router responds."""
        bus = TeamMessageBus()
        bus.register("alice")
        bus.register("lead")

        result = await _proxy_task_list(bus=bus, member_name="alice")
        parsed = json.loads(result)
        assert "error" in parsed
