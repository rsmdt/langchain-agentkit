"""Tests for progressive state schema composition."""

import typing
from pathlib import Path

from langchain_agentkit.agent_kit import AgentKit
from langchain_agentkit.middleware.filesystem import FilesystemMiddleware
from langchain_agentkit.middleware.skills import SkillsMiddleware
from langchain_agentkit.middleware.tasks import TasksMiddleware
from langchain_agentkit.state import AgentKitState, TasksState

FIXTURES = Path(__file__).parent.parent / "fixtures"


class TestAgentKitState:
    def test_has_messages(self):
        assert "messages" in AgentKitState.__annotations__

    def test_has_sender(self):
        assert "sender" in AgentKitState.__annotations__

    def test_does_not_have_tasks(self):
        assert "tasks" not in AgentKitState.__annotations__


class TestTasksState:
    def test_has_tasks(self):
        assert "tasks" in TasksState.__annotations__

    def test_tasks_has_reducer(self):
        hints = typing.get_type_hints(TasksState, include_extras=True)
        assert hasattr(hints["tasks"], "__metadata__")


class TestAgentKitStateSchema:
    def test_no_middleware_returns_base(self):
        kit = AgentKit([])

        assert kit.state_schema is AgentKitState

    def test_skills_only_returns_base(self):
        kit = AgentKit([SkillsMiddleware(skills=str(FIXTURES / "skills"))])

        schema = kit.state_schema
        annotations = typing.get_type_hints(schema, include_extras=True)

        assert "messages" in annotations
        assert "tasks" not in annotations

    def test_tasks_adds_tasks_key(self):
        kit = AgentKit([TasksMiddleware()])

        schema = kit.state_schema
        annotations = typing.get_type_hints(schema, include_extras=True)

        assert "messages" in annotations
        assert "tasks" in annotations

    def test_tasks_reducer_preserved(self):
        kit = AgentKit([TasksMiddleware()])

        schema = kit.state_schema
        hints = typing.get_type_hints(schema, include_extras=True)

        assert hasattr(hints["tasks"], "__metadata__")

    def test_messages_reducer_preserved(self):
        kit = AgentKit([TasksMiddleware()])

        schema = kit.state_schema
        hints = typing.get_type_hints(schema, include_extras=True)

        assert hasattr(hints["messages"], "__metadata__")

    def test_multiple_middleware_compose(self):
        kit = AgentKit(
            [
                SkillsMiddleware(skills=str(FIXTURES / "skills")),
                TasksMiddleware(),
                FilesystemMiddleware(),
            ]
        )

        schema = kit.state_schema
        annotations = typing.get_type_hints(schema, include_extras=True)

        assert "messages" in annotations
        assert "tasks" in annotations

    def test_duplicate_middleware_deduplicates(self):
        kit = AgentKit([TasksMiddleware(), TasksMiddleware()])

        schema = kit.state_schema
        annotations = typing.get_type_hints(schema, include_extras=True)

        assert "messages" in annotations
        assert "tasks" in annotations

    def test_filesystem_only_returns_base(self):
        kit = AgentKit([FilesystemMiddleware()])

        schema = kit.state_schema

        assert schema is AgentKitState
