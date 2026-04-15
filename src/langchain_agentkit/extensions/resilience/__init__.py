"""ResilienceExtension — self-healing tool execution and message invariants.

Layer 1 (shipped): write-time prevention. Wraps every tool call so that
any unhandled exception becomes a ``ToolMessage`` paired to the
originating ``tool_call_id``. Prevents orphan ``AIMessage(tool_calls=...)``
from ever being written to the checkpoint, which is what causes the
OpenAI Responses API error ``"No tool output found for function call"``.

Future layers (not yet shipped):
    - Layer 2: read-time repair of orphan tool calls already in state
    - Layer 3: richer telemetry, persistence of repairs
"""

from langchain_agentkit.extensions.resilience.extension import ResilienceExtension
from langchain_agentkit.extensions.resilience.types import ToolErrorEvent

__all__ = ["ResilienceExtension", "ToolErrorEvent"]
