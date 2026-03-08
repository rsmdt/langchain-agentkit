"""Backwards-compatibility shim — use ``langchain_agentkit`` instead.

This module re-exports the original ``langchain_skillkit`` public API from
``langchain_agentkit`` and emits a deprecation warning on first import.

.. deprecated:: 0.4.0
    ``langchain_skillkit`` has been renamed to ``langchain_agentkit``.
    Update your imports::

        # Before
        from langchain_skillkit import node, SkillKit, AgentState

        # After
        from langchain_agentkit import node, SkillKit, AgentState
"""

# ruff: noqa: E402
import warnings

warnings.warn(
    "langchain_skillkit has been renamed to langchain_agentkit. "
    "Update your imports: 'from langchain_agentkit import ...'",
    DeprecationWarning,
    stacklevel=2,
)

from langchain_agentkit.node import node
from langchain_agentkit.skill_kit import SkillKit
from langchain_agentkit.state import AgentState

__all__ = [
    "AgentState",
    "SkillKit",
    "node",
]
