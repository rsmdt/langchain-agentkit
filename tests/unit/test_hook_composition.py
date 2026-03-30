"""Tests for hook composition and wiring in the graph builder."""

from __future__ import annotations

import pytest

from langchain_agentkit.extension import Extension
from langchain_agentkit.hook_runner import HookRunner
from langchain_agentkit.hooks import after, before, wrap

# --- Test extensions ---

class LoggingExtension(Extension):
    def __init__(self, name: str, log: list):
        self._name = name
        self._log = log

    async def before_model(self, state, runtime):
        self._log.append(f"{self._name}:before_model")
        return None

    async def after_model(self, state, runtime):
        self._log.append(f"{self._name}:after_model")
        return None

    async def wrap_model(self, request, handler):
        self._log.append(f"{self._name}:wrap_model:enter")
        result = await handler(request)
        self._log.append(f"{self._name}:wrap_model:exit")
        return result

    async def before_run(self, state, runtime):
        self._log.append(f"{self._name}:before_run")
        return None

    async def after_run(self, state, runtime):
        self._log.append(f"{self._name}:after_run")
        return None

    async def wrap_tool(self, request, handler):
        self._log.append(f"{self._name}:wrap_tool:enter")
        result = await handler(request)
        self._log.append(f"{self._name}:wrap_tool:exit")
        return result


class ToolFilterExtension(Extension):
    def __init__(self, log: list):
        self._log = log

    @wrap("tool", tools=["delete_file"])
    async def gate_delete(self, request, handler):
        self._log.append("gate_delete")
        return await handler(request)

    @wrap("tool", tools=["send_email"])
    async def gate_email(self, request, handler):
        self._log.append("gate_email")
        return await handler(request)


class HistoryExtension(Extension):
    def __init__(self, prefix: str, log: list):
        self._prefix = prefix
        self._log = log

    def process_history(self, messages):
        self._log.append(f"{self._prefix}:process_history")
        return [f"[{self._prefix}]{m}" for m in messages]


class ErrorExtension(Extension):
    def __init__(self, log: list):
        self._log = log

    async def on_error(self, error, state, runtime):
        self._log.append(f"on_error:{type(error).__name__}")
        return None


# --- HookRunner tests ---


class TestHookRunnerBeforeAfterOrdering:
    """before_* runs in declaration order, after_* in reverse order."""

    @pytest.mark.asyncio
    async def test_before_model_runs_in_order(self):
        log = []
        runner = HookRunner([
            LoggingExtension("A", log),
            LoggingExtension("B", log),
        ])

        await runner.run_before("model", state={}, runtime=None)

        assert log == ["A:before_model", "B:before_model"]

    @pytest.mark.asyncio
    async def test_after_model_runs_in_reverse_order(self):
        log = []
        runner = HookRunner([
            LoggingExtension("A", log),
            LoggingExtension("B", log),
        ])

        await runner.run_after("model", state={}, runtime=None)

        assert log == ["B:after_model", "A:after_model"]

    @pytest.mark.asyncio
    async def test_before_run_runs_in_order(self):
        log = []
        runner = HookRunner([
            LoggingExtension("A", log),
            LoggingExtension("B", log),
        ])

        await runner.run_before("run", state={}, runtime=None)

        assert log == ["A:before_run", "B:before_run"]

    @pytest.mark.asyncio
    async def test_after_run_runs_in_reverse_order(self):
        log = []
        runner = HookRunner([
            LoggingExtension("A", log),
            LoggingExtension("B", log),
        ])

        await runner.run_after("run", state={}, runtime=None)

        assert log == ["B:after_run", "A:after_run"]


class TestHookRunnerWrapComposition:
    """wrap_* composes as onion layers (first extension = outermost)."""

    @pytest.mark.asyncio
    async def test_wrap_model_onion_order(self):
        log = []
        runner = HookRunner([
            LoggingExtension("A", log),
            LoggingExtension("B", log),
        ])

        async def actual_model(request):
            log.append("model_call")
            return "response"

        result = await runner.run_wrap("model", request="input", handler=actual_model)

        assert log == [
            "A:wrap_model:enter",
            "B:wrap_model:enter",
            "model_call",
            "B:wrap_model:exit",
            "A:wrap_model:exit",
        ]
        assert result == "response"

    @pytest.mark.asyncio
    async def test_wrap_tool_onion_order(self):
        log = []
        runner = HookRunner([
            LoggingExtension("A", log),
            LoggingExtension("B", log),
        ])

        async def actual_tool(request):
            log.append("tool_call")
            return "tool_result"

        result = await runner.run_wrap("tool", request="input", handler=actual_tool)

        assert log == [
            "A:wrap_tool:enter",
            "B:wrap_tool:enter",
            "tool_call",
            "B:wrap_tool:exit",
            "A:wrap_tool:exit",
        ]
        assert result == "tool_result"

    @pytest.mark.asyncio
    async def test_no_wrap_hooks_calls_handler_directly(self):
        runner = HookRunner([])

        async def actual_model(request):
            return "direct"

        result = await runner.run_wrap("model", request="input", handler=actual_model)
        assert result == "direct"


class TestHookRunnerToolFiltering:
    """Per-tool filtering on wrap("tool", tools=[...])."""

    @pytest.mark.asyncio
    async def test_matching_tool_name_fires_hook(self):
        log = []
        runner = HookRunner([ToolFilterExtension(log)])

        async def handler(request):
            return "ok"

        await runner.run_wrap("tool", request="input", handler=handler, tool_name="delete_file")

        assert "gate_delete" in log
        assert "gate_email" not in log

    @pytest.mark.asyncio
    async def test_non_matching_tool_name_skips_hook(self):
        log = []
        runner = HookRunner([ToolFilterExtension(log)])

        async def handler(request):
            return "ok"

        await runner.run_wrap("tool", request="input", handler=handler, tool_name="read_file")

        assert log == []

    @pytest.mark.asyncio
    async def test_no_filter_matches_all_tools(self):
        log = []
        runner = HookRunner([LoggingExtension("A", log)])

        async def handler(request):
            return "ok"

        await runner.run_wrap("tool", request="input", handler=handler, tool_name="anything")

        assert "A:wrap_tool:enter" in log


class TestHookRunnerProcessHistory:
    """process_history composes as a pipeline in declaration order."""

    def test_single_processor(self):
        log = []
        runner = HookRunner([HistoryExtension("A", log)])

        result = runner.run_process_history(["msg1", "msg2"])

        assert result == ["[A]msg1", "[A]msg2"]
        assert log == ["A:process_history"]

    def test_pipeline_order(self):
        log = []
        runner = HookRunner([
            HistoryExtension("A", log),
            HistoryExtension("B", log),
        ])

        result = runner.run_process_history(["msg"])

        assert log == ["A:process_history", "B:process_history"]
        # B processes A's output
        assert result == ["[B][A]msg"]

    def test_no_processors_returns_unchanged(self):
        runner = HookRunner([])

        messages = ["msg1", "msg2"]
        result = runner.run_process_history(messages)

        assert result == messages


class TestHookRunnerOnError:
    """on_error hook fires on unhandled errors."""

    @pytest.mark.asyncio
    async def test_on_error_fires(self):
        log = []
        runner = HookRunner([ErrorExtension(log)])

        await runner.run_on_error(ValueError("test"), state={}, runtime=None)

        assert log == ["on_error:ValueError"]

    @pytest.mark.asyncio
    async def test_no_error_hooks_is_noop(self):
        runner = HookRunner([])

        # Should not raise
        await runner.run_on_error(ValueError("test"), state={}, runtime=None)


class TestHookRunnerStateUpdates:
    """before_* hooks can return state updates."""

    @pytest.mark.asyncio
    async def test_before_returns_state_update(self):
        class UpdateExtension(Extension):
            async def before_model(self, state, runtime):
                return {"extra_key": "injected"}

        runner = HookRunner([UpdateExtension()])

        updates = await runner.run_before("model", state={}, runtime=None)

        assert updates == [{"extra_key": "injected"}]

    @pytest.mark.asyncio
    async def test_before_returns_none_excluded(self):
        class NoopExtension(Extension):
            async def before_model(self, state, runtime):
                return None

        runner = HookRunner([NoopExtension()])

        updates = await runner.run_before("model", state={}, runtime=None)

        assert updates == []


class TestRunLifecycleHooksWired:
    """before_run/after_run hooks are wired into the graph via _run_entry/_run_exit nodes."""

    def test_graph_has_run_entry_exit_when_hooks_present(self):
        """When extensions define before_run/after_run, the graph has lifecycle nodes."""
        from unittest.mock import MagicMock

        from langchain_agentkit import agent

        log = []

        class LifecycleExtension(Extension):
            async def before_run(self, state, runtime):
                log.append("before_run")
                return None

            async def after_run(self, state, runtime):
                log.append("after_run")
                return None

        mock_llm = MagicMock()
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)

        class my_agent(agent):
            llm = mock_llm
            extensions = [LifecycleExtension()]

            async def handler(state, *, llm, prompt):
                pass

        # my_agent is a StateGraph — check it has the lifecycle nodes
        assert "_run_entry" in my_agent.nodes
        assert "_run_exit" in my_agent.nodes

    def test_graph_no_lifecycle_nodes_without_hooks(self):
        """Without run lifecycle hooks, no extra nodes are added."""
        from unittest.mock import MagicMock

        from langchain_agentkit import agent

        mock_llm = MagicMock()
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)

        class my_agent(agent):
            llm = mock_llm

            async def handler(state, *, llm, prompt):
                pass

        assert "_run_entry" not in my_agent.nodes
        assert "_run_exit" not in my_agent.nodes


class TestBeforeAfterToolFiltering:
    """Per-tool filtering works on before/after hooks, not just wrap."""

    @pytest.mark.asyncio
    async def test_before_tool_with_matching_filter(self):
        log = []

        class FilteredExt(Extension):
            @before("tool", tools=["delete_file"])
            async def gate(self, state, runtime):
                log.append("before:delete_file")
                return None

        runner = HookRunner([FilteredExt()])

        await runner.run_before("tool", state={}, runtime=None, tool_name="delete_file")
        assert log == ["before:delete_file"]

    @pytest.mark.asyncio
    async def test_before_tool_with_non_matching_filter(self):
        log = []

        class FilteredExt(Extension):
            @before("tool", tools=["delete_file"])
            async def gate(self, state, runtime):
                log.append("before:delete_file")
                return None

        runner = HookRunner([FilteredExt()])

        await runner.run_before("tool", state={}, runtime=None, tool_name="read_file")
        assert log == []

    @pytest.mark.asyncio
    async def test_after_tool_with_matching_filter(self):
        log = []

        class FilteredExt(Extension):
            @after("tool", tools=["send_email"])
            async def audit(self, state, runtime):
                log.append("after:send_email")
                return None

        runner = HookRunner([FilteredExt()])

        await runner.run_after("tool", state={}, runtime=None, tool_name="send_email")
        assert log == ["after:send_email"]

    @pytest.mark.asyncio
    async def test_after_tool_with_non_matching_filter(self):
        log = []

        class FilteredExt(Extension):
            @after("tool", tools=["send_email"])
            async def audit(self, state, runtime):
                log.append("after:send_email")
                return None

        runner = HookRunner([FilteredExt()])

        await runner.run_after("tool", state={}, runtime=None, tool_name="read_file")
        assert log == []


class TestJumpToRouting:
    """jump_to routing from before_model/after_model hooks."""

    @pytest.mark.asyncio
    async def test_jump_to_end_from_before_model(self):
        class LimitExtension(Extension):
            async def before_model(self, state, runtime):
                if len(state.get("messages", [])) > 5:
                    return {"jump_to": "end"}
                return None

        runner = HookRunner([LimitExtension()])

        updates = await runner.run_before(
            "model", state={"messages": list(range(10))}, runtime=None
        )
        assert updates == [{"jump_to": "end"}]

    @pytest.mark.asyncio
    async def test_jump_to_model_from_after_model(self):
        class RetryExtension(Extension):
            async def after_model(self, state, runtime):
                return {"jump_to": "model"}

        runner = HookRunner([RetryExtension()])

        updates = await runner.run_after("model", state={}, runtime=None)
        assert updates == [{"jump_to": "model"}]

    @pytest.mark.asyncio
    async def test_no_jump_when_none_returned(self):
        class PassthroughExtension(Extension):
            async def before_model(self, state, runtime):
                return None

        runner = HookRunner([PassthroughExtension()])

        updates = await runner.run_before("model", state={}, runtime=None)
        assert updates == []

    @pytest.mark.asyncio
    async def test_invalid_jump_to_raises(self):
        class BadExtension(Extension):
            async def before_model(self, state, runtime):
                return {"jump_to": "invalid_target"}

        runner = HookRunner([BadExtension()])

        with pytest.raises(ValueError, match="Invalid jump_to target"):
            await runner.run_before("model", state={}, runtime=None)

    @pytest.mark.asyncio
    async def test_valid_jump_to_targets(self):
        """All three valid targets should work."""
        for target in ("model", "tools", "end"):

            class JumpExt(Extension):
                _target = target

                async def before_model(self, state, runtime):
                    return {"jump_to": self._target}

            runner = HookRunner([JumpExt()])
            updates = await runner.run_before("model", state={}, runtime=None)
            assert updates == [{"jump_to": target}]
