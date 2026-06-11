"""ResilienceExtension — self-healing tool execution and message invariants.

Layer 1 (shipped): write-time prevention. Wraps every tool call so that
any unhandled exception becomes a ``ToolMessage`` paired to the
originating ``tool_call_id``. Prevents orphan ``AIMessage(tool_calls=...)``
from ever being written to the checkpoint, which is what causes the
OpenAI Responses API error ``"No tool output found for function call"``.

Layer 2 (shipped): read-time repair. Before each LLM call, scans the
message list for orphan tool calls already in state (from earlier
crashes, infrastructure failures, or manual edits) and injects
synthetic ``ToolMessage``\\ s so the model request is always well-formed.

Future layers (not yet shipped):
    - Layer 3: richer telemetry, persistence of repairs
"""

from langchain_agentkit.extensions.resilience.extension import ResilienceExtension
from langchain_agentkit.extensions.resilience.types import (
    OrphanRepairEvent,
    ToolErrorEvent,
)

__all__ = ["OrphanRepairEvent", "ResilienceExtension", "ToolErrorEvent"]
