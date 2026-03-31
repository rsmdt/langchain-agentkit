"""Tests for hook decorators and Extension base class."""

from __future__ import annotations

from langchain_agentkit.extension import Extension
from langchain_agentkit.hooks import after, before, wrap

# --- Decorator metadata tests ---


class TestDecorators:
    """Test that before/after/wrap decorators attach correct metadata."""

    def test_before_sets_metadata(self):
        @before("model")
        async def my_hook(self, state, runtime):
            pass

        assert my_hook._hook_phase == "before"
        assert my_hook._hook_point == "model"
        assert my_hook._hook_tool_filter is None

    def test_after_sets_metadata(self):
        @after("tool")
        async def my_hook(self, request, result, runtime):
            pass

        assert my_hook._hook_phase == "after"
        assert my_hook._hook_point == "tool"
        assert my_hook._hook_tool_filter is None

    def test_wrap_sets_metadata(self):
        @wrap("run")
        async def my_hook(self, request, handler):
            pass

        assert my_hook._hook_phase == "wrap"
        assert my_hook._hook_point == "run"
        assert my_hook._hook_tool_filter is None

    def test_per_tool_filtering(self):
        @wrap("tool", tools=["delete_file", "send_email"])
        async def my_hook(self, request, handler):
            pass

        assert my_hook._hook_tool_filter == ["delete_file", "send_email"]

    def test_before_tool_with_filter(self):
        @before("tool", tools=["web_search"])
        async def my_hook(self, request, runtime):
            pass

        assert my_hook._hook_phase == "before"
        assert my_hook._hook_point == "tool"
        assert my_hook._hook_tool_filter == ["web_search"]

    def test_decorator_preserves_function(self):
        @before("model")
        async def my_named_hook(self, state, runtime):
            """My docstring."""

        assert my_named_hook.__name__ == "my_named_hook"
        assert my_named_hook.__doc__ == "My docstring."

    def test_valid_points(self):
        """All valid hook points should work without error."""
        for point in ("run", "model", "tool"):

            @before(point)
            async def hook(self, state, runtime):
                pass

            assert hook._hook_point == point


# --- Extension base class tests ---


class TestExtensionHookCollection:
    """Test that Extension.__init_subclass__ collects decorated hooks."""

    def test_collects_single_decorated_hook(self):
        class MyExt(Extension):
            @before("model")
            async def check_input(self, state, runtime):
                pass

        assert ("before", "model") in MyExt._decorated_hooks
        assert len(MyExt._decorated_hooks[("before", "model")]) == 1

    def test_collects_multiple_hooks_same_point(self):
        class MyExt(Extension):
            @before("model")
            async def check_a(self, state, runtime):
                pass

            @before("model")
            async def check_b(self, state, runtime):
                pass

        hooks = MyExt._decorated_hooks[("before", "model")]
        assert len(hooks) == 2

    def test_collects_hooks_across_points(self):
        class MyExt(Extension):
            @before("model")
            async def pre(self, state, runtime):
                pass

            @after("model")
            async def post(self, state, runtime):
                pass

            @wrap("tool", tools=["delete"])
            async def gate(self, request, handler):
                pass

        assert ("before", "model") in MyExt._decorated_hooks
        assert ("after", "model") in MyExt._decorated_hooks
        assert ("wrap", "tool") in MyExt._decorated_hooks

    def test_no_decorated_hooks_yields_empty(self):
        class PlainExt(Extension):
            pass

        assert PlainExt._decorated_hooks == {}

    def test_preserves_tool_filter_on_collected_hooks(self):
        class MyExt(Extension):
            @wrap("tool", tools=["send_email"])
            async def gate(self, request, handler):
                pass

        hook = MyExt._decorated_hooks[("wrap", "tool")][0]
        assert hook._hook_tool_filter == ["send_email"]

    def test_inheritance_does_not_leak_between_subclasses(self):
        class ExtA(Extension):
            @before("model")
            async def hook_a(self, state, runtime):
                pass

        class ExtB(Extension):
            @after("tool")
            async def hook_b(self, request, result, runtime):
                pass

        assert ("before", "model") in ExtA._decorated_hooks
        assert ("before", "model") not in ExtB._decorated_hooks
        assert ("after", "tool") in ExtB._decorated_hooks
        assert ("after", "tool") not in ExtA._decorated_hooks


class TestExtensionNamedMethods:
    """Test that Extension discovers named methods (before_model, wrap_tool, etc.)."""

    def test_discovers_named_before_model(self):
        class MyExt(Extension):
            async def before_model(self, state, runtime):
                return None

        hooks = MyExt._get_named_hooks()
        assert ("before", "model") in hooks

    def test_discovers_named_wrap_tool(self):
        class MyExt(Extension):
            async def wrap_tool(self, request, handler):
                return await handler(request)

        hooks = MyExt._get_named_hooks()
        assert ("wrap", "tool") in hooks

    def test_discovers_all_named_hooks(self):  # noqa: C901
        class FullExt(Extension):
            async def before_run(self, state, runtime):
                return None

            async def after_run(self, state, runtime):
                return None

            async def on_error(self, error, state, runtime):
                return None

            async def before_model(self, state, runtime):
                return None

            async def after_model(self, state, runtime):
                return None

            async def wrap_model(self, request, handler):
                return await handler(request)

            async def before_tool(self, request, runtime):
                return None

            async def after_tool(self, request, result, runtime):
                return result

            async def wrap_tool(self, request, handler):
                return await handler(request)

        hooks = FullExt._get_named_hooks()
        expected = [
            ("before", "run"),
            ("after", "run"),
            ("on_error", "run"),
            ("before", "model"),
            ("after", "model"),
            ("wrap", "model"),
            ("before", "tool"),
            ("after", "tool"),
            ("wrap", "tool"),
        ]
        for key in expected:
            assert key in hooks, f"Missing named hook: {key}"

    def test_no_named_hooks_on_plain_extension(self):
        class PlainExt(Extension):
            pass

        hooks = PlainExt._get_named_hooks()
        assert hooks == {}


class TestExtensionGetAllHooks:
    """Test that get_all_hooks merges named methods + decorated hooks."""

    def test_merges_named_and_decorated(self):
        class MyExt(Extension):
            async def before_model(self, state, runtime):
                return None

            @before("model")
            async def extra_check(self, state, runtime):
                return None

        instance = MyExt()
        all_hooks = instance.get_all_hooks()
        # Should have both hooks under ("before", "model")
        assert len(all_hooks[("before", "model")]) == 2

    def test_decorated_only(self):
        class MyExt(Extension):
            @wrap("tool", tools=["delete"])
            async def gate(self, request, handler):
                pass

        instance = MyExt()
        all_hooks = instance.get_all_hooks()
        assert len(all_hooks[("wrap", "tool")]) == 1

    def test_named_only(self):
        class MyExt(Extension):
            async def after_run(self, state, runtime):
                return None

        instance = MyExt()
        all_hooks = instance.get_all_hooks()
        assert len(all_hooks[("after", "run")]) == 1


class TestExtensionProtocol:
    """Test Extension satisfies the required protocol surface."""

    def test_default_tools_returns_empty_list(self):
        class MyExt(Extension):
            pass

        ext = MyExt()
        assert ext.tools == []

    def test_default_prompt_returns_none(self):
        class MyExt(Extension):
            pass

        ext = MyExt()
        assert ext.prompt({}) is None

    def test_default_state_schema_returns_none(self):
        class MyExt(Extension):
            pass

        ext = MyExt()
        assert ext.state_schema is None

    def test_default_dependencies_returns_empty(self):
        class MyExt(Extension):
            pass

        ext = MyExt()
        assert ext.dependencies() == []

    def test_overridden_tools(self):
        class MyExt(Extension):
            @property
            def tools(self):
                return ["tool_a", "tool_b"]

        ext = MyExt()
        assert ext.tools == ["tool_a", "tool_b"]

    def test_overridden_prompt(self):
        class MyExt(Extension):
            def prompt(self, state, runtime=None):
                return "Custom prompt"

        ext = MyExt()
        assert ext.prompt({}) == "Custom prompt"

    def test_overridden_state_schema(self):
        class MyExt(Extension):
            @property
            def state_schema(self):
                return dict  # placeholder

        ext = MyExt()
        assert ext.state_schema is dict

    def test_process_history_optional(self):
        """process_history is optional — not present by default."""

        class MyExt(Extension):
            pass

        ext = MyExt()
        assert (
            not hasattr(ext.__class__, "process_history")
            or getattr(ext.__class__, "process_history", None) is None
        )
