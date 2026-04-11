"""Tests for progressive state schema composition."""

import typing

from langchain_agentkit.agent_kit import AgentKit
from langchain_agentkit.extensions.filesystem import FilesystemExtension
from langchain_agentkit.extensions.skills import SkillsExtension
from langchain_agentkit.extensions.skills.types import SkillConfig
from langchain_agentkit.extensions.tasks import TasksExtension
from langchain_agentkit.extensions.tasks.state import TasksState
from langchain_agentkit.state import AgentKitState


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
    def test_no_extensions_returns_base(self):
        kit = AgentKit(extensions=[])

        assert kit.state_schema is AgentKitState

    def test_skills_only_returns_base(self):
        kit = AgentKit(
            extensions=[
                SkillsExtension(
                    skills=[
                        SkillConfig(name="test", description="test", prompt="test"),
                    ]
                )
            ]
        )

        schema = kit.state_schema
        annotations = typing.get_type_hints(schema, include_extras=True)

        assert "messages" in annotations
        assert "tasks" not in annotations

    def test_tasks_adds_tasks_key(self):
        kit = AgentKit(extensions=[TasksExtension()])

        schema = kit.state_schema
        annotations = typing.get_type_hints(schema, include_extras=True)

        assert "messages" in annotations
        assert "tasks" in annotations

    def test_tasks_reducer_preserved(self):
        kit = AgentKit(extensions=[TasksExtension()])

        schema = kit.state_schema
        hints = typing.get_type_hints(schema, include_extras=True)

        assert hasattr(hints["tasks"], "__metadata__")

    def test_messages_reducer_preserved(self):
        kit = AgentKit(extensions=[TasksExtension()])

        schema = kit.state_schema
        hints = typing.get_type_hints(schema, include_extras=True)

        assert hasattr(hints["messages"], "__metadata__")

    def test_multiple_extensions_compose(self):
        kit = AgentKit(
            extensions=[
                SkillsExtension(
                    skills=[
                        SkillConfig(name="test", description="test", prompt="test"),
                    ]
                ),
                TasksExtension(),
                FilesystemExtension(),
            ]
        )

        schema = kit.state_schema
        annotations = typing.get_type_hints(schema, include_extras=True)

        assert "messages" in annotations
        assert "tasks" in annotations

    def test_duplicate_extensions_deduplicates(self):
        kit = AgentKit(extensions=[TasksExtension(), TasksExtension()])

        schema = kit.state_schema
        annotations = typing.get_type_hints(schema, include_extras=True)

        assert "messages" in annotations
        assert "tasks" in annotations

    def test_filesystem_only_returns_base(self):
        kit = AgentKit(extensions=[FilesystemExtension()])

        schema = kit.state_schema

        assert schema is AgentKitState
