"""Shared handler signature validation for node and agent metaclasses."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


def validate_handler_signature(
    handler: Any,
    class_name: str,
    valid_params: frozenset[str],
    label: str,
) -> tuple[set[str], type]:
    """Validate handler signature, extract injectables and state type.

    Returns a tuple of (injectable_params, state_type).

    The state type is inferred from the handler's first parameter annotation.
    If no annotation is present, defaults to ``AgentState``.

    Args:
        handler: The handler callable to inspect.
        class_name: The class name used in error messages.
        valid_params: The set of valid injectable parameter names.
        label: A label for error messages (e.g. "node" or "agent").

    Raises:
        ValueError: If handler has invalid parameters.
    """
    from langchain_agentkit.state import AgentState

    sig = inspect.signature(handler)
    params = list(sig.parameters.values())

    if not params:
        raise ValueError(
            f"class {class_name}({label}): handler must accept at least "
            f"'state' as its first parameter"
        )

    first = params[0]
    if first.kind not in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ):
        raise ValueError(
            f"class {class_name}({label}): handler's first parameter must be "
            f"positional ('state'), got {first.kind.name}"
        )

    state_type: type = AgentState
    if first.annotation is not inspect.Parameter.empty:
        state_type = first.annotation

    injectable = set()
    for param in params[1:]:
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            continue
        if param.kind == inspect.Parameter.KEYWORD_ONLY:
            if param.name not in valid_params:
                raise ValueError(
                    f"class {class_name}({label}): unknown handler parameter "
                    f"'{param.name}'. Valid parameters: state, "
                    f"{', '.join(sorted(valid_params))}"
                )
            injectable.add(param.name)
        elif param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            raise ValueError(
                f"class {class_name}({label}): handler parameter '{param.name}' "
                f"must be keyword-only (after *). "
                f"Signature should be: handler(state, *, {param.name}, ...)"
            )

    return injectable, state_type
