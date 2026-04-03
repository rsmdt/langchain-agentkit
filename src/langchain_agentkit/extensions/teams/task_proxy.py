"""Task proxy tools for teammates.

Teammates cannot access the lead's graph state directly. These proxy
tools serialize task operations into structured JSON messages, send them
through the :class:`TeamMessageBus`, and wait for the router to process
the operation and return the result.

The router node in :mod:`~langchain_agentkit.extensions.teams.extension`
detects these structured messages and applies them to
``state["tasks"]`` before sending an acknowledgment back.

Protocol::

    Teammate                       Bus                     Router (lead graph)
       |                            |                            |
       |-- {type: task_op, ...} --> |                            |
       |                            |-- TeamMessage ----------> |
       |                            |                    applies to state["tasks"]
       |                            | <-- {request_id, ...} --- |
       | <-- ack with result ------ |                            |
"""

from __future__ import annotations

import json
import uuid
from functools import partial
from typing import TYPE_CHECKING, Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

    from langchain_agentkit.extensions.teams.bus import TeamMessageBus

# ---------------------------------------------------------------------------
# Message type constant
# ---------------------------------------------------------------------------

TASK_OP_TYPE = "task_op"

# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------


class _ProxyTaskCreateInput(BaseModel):
    subject: str = Field(
        description=(
            "A brief, actionable title in imperative form "
            '(e.g., "Fix authentication bug in login flow").'
        ),
    )
    description: str = Field(
        description="Detailed description of what needs to be done.",
    )
    active_form: str = Field(
        default="",
        description='Present continuous form shown while in_progress (e.g., "Fixing bug").',
    )


class _ProxyTaskUpdateInput(BaseModel):
    task_id: str = Field(description="Task ID to update.")
    status: str | None = Field(
        default=None,
        description="New status: pending, in_progress, completed, or deleted.",
    )
    subject: str | None = Field(default=None, description="Updated title.")
    description: str | None = Field(default=None, description="Updated requirements.")
    active_form: str | None = Field(default=None, description="Updated spinner text.")
    owner: str | None = Field(default=None, description="Set the task owner (agent name).")
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Metadata keys to merge. Set a key to null to delete it.",
    )
    add_blocked_by: list[str] | None = Field(
        default=None,
        description="Task IDs that must complete before this one can start.",
    )
    add_blocks: list[str] | None = Field(
        default=None,
        description="Task IDs that cannot start until this one completes.",
    )


class _ProxyTaskListInput(BaseModel):
    pass


class _ProxyTaskGetInput(BaseModel):
    task_id: str = Field(description="Task ID to retrieve.")


# ---------------------------------------------------------------------------
# Proxy implementations
# ---------------------------------------------------------------------------


async def _proxy_task_create(
    subject: str,
    description: str,
    active_form: str = "",
    *,
    bus: TeamMessageBus,
    member_name: str,
) -> str:
    """Create a task via the lead's state."""
    request_id = str(uuid.uuid4())
    payload = json.dumps({
        "type": TASK_OP_TYPE,
        "op": "create",
        "request_id": request_id,
        "subject": subject,
        "description": description,
        "active_form": active_form,
    })

    response = await bus.request_response(
        member_name, "lead", payload, request_id=request_id,
    )
    if response is None:
        return json.dumps({"error": "Timeout waiting for task creation acknowledgment."})
    return response.content


async def _proxy_task_update(
    task_id: str,
    status: str | None = None,
    subject: str | None = None,
    description: str | None = None,
    active_form: str | None = None,
    owner: str | None = None,
    metadata: dict[str, Any] | None = None,
    add_blocked_by: list[str] | None = None,
    add_blocks: list[str] | None = None,
    *,
    bus: TeamMessageBus,
    member_name: str,
) -> str:
    """Update a task via the lead's state."""
    request_id = str(uuid.uuid4())
    fields: dict[str, Any] = {"task_id": task_id}
    if status is not None:
        fields["status"] = status
    if subject is not None:
        fields["subject"] = subject
    if description is not None:
        fields["description"] = description
    if active_form is not None:
        fields["active_form"] = active_form
    if owner is not None:
        fields["owner"] = owner
    if metadata is not None:
        fields["metadata"] = metadata
    if add_blocked_by is not None:
        fields["add_blocked_by"] = add_blocked_by
    if add_blocks is not None:
        fields["add_blocks"] = add_blocks

    payload = json.dumps({
        "type": TASK_OP_TYPE,
        "op": "update",
        "request_id": request_id,
        **fields,
    })

    response = await bus.request_response(
        member_name, "lead", payload, request_id=request_id,
    )
    if response is None:
        return json.dumps({"error": "Timeout waiting for task update acknowledgment."})
    return response.content


async def _proxy_task_list(
    *,
    bus: TeamMessageBus,
    member_name: str,
) -> str:
    """List all tasks from the lead's state."""
    request_id = str(uuid.uuid4())
    payload = json.dumps({
        "type": TASK_OP_TYPE,
        "op": "list",
        "request_id": request_id,
    })

    response = await bus.request_response(
        member_name, "lead", payload, request_id=request_id,
    )
    if response is None:
        return json.dumps({"error": "Timeout waiting for task list."})
    return response.content


async def _proxy_task_get(
    task_id: str,
    *,
    bus: TeamMessageBus,
    member_name: str,
) -> str:
    """Get full details of a task from the lead's state."""
    request_id = str(uuid.uuid4())
    payload = json.dumps({
        "type": TASK_OP_TYPE,
        "op": "get",
        "request_id": request_id,
        "task_id": task_id,
    })

    response = await bus.request_response(
        member_name, "lead", payload, request_id=request_id,
    )
    if response is None:
        return json.dumps({"error": "Timeout waiting for task details."})
    return response.content


# ---------------------------------------------------------------------------
# Descriptions
# ---------------------------------------------------------------------------

_PROXY_TASK_CREATE_DESC = """\
Create a new task in the shared team task list.

Use to track work items visible to all team members. Tasks are created \
with status 'pending'. Use TaskUpdate to assign an owner or set dependencies.\
"""

_PROXY_TASK_UPDATE_DESC = """\
Update an existing task in the shared team task list.

Use to change status, assign ownership, or set dependencies. \
Mark tasks as 'completed' when work is done, then check TaskList for next work.\
"""

_PROXY_TASK_LIST_DESC = """\
List all tasks in the shared team task list.

Use to find available work (status: pending, no owner, not blocked). \
Prefer tasks in ID order when multiple are available.\
"""

_PROXY_TASK_GET_DESC = """\
Get full details of a task by ID from the shared team task list.

Returns subject, description, status, owner, blocks, blockedBy, and metadata.\
"""


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_task_proxy_tools(
    bus: TeamMessageBus,
    member_name: str,
) -> list[BaseTool]:
    """Create task proxy tools bound to a specific teammate and bus.

    These tools have the same interface as the regular TaskCreate/TaskUpdate
    tools but route operations through the message bus to the lead's graph
    state instead of reading/writing local state.
    """
    return [
        StructuredTool.from_function(
            coroutine=partial(_proxy_task_create, bus=bus, member_name=member_name),
            name="TaskCreate",
            description=_PROXY_TASK_CREATE_DESC,
            args_schema=_ProxyTaskCreateInput,
            handle_tool_error=True,
        ),
        StructuredTool.from_function(
            coroutine=partial(_proxy_task_update, bus=bus, member_name=member_name),
            name="TaskUpdate",
            description=_PROXY_TASK_UPDATE_DESC,
            args_schema=_ProxyTaskUpdateInput,
            handle_tool_error=True,
        ),
        StructuredTool.from_function(
            coroutine=partial(_proxy_task_list, bus=bus, member_name=member_name),
            name="TaskList",
            description=_PROXY_TASK_LIST_DESC,
            args_schema=_ProxyTaskListInput,
            handle_tool_error=True,
        ),
        StructuredTool.from_function(
            coroutine=partial(_proxy_task_get, bus=bus, member_name=member_name),
            name="TaskGet",
            description=_PROXY_TASK_GET_DESC,
            args_schema=_ProxyTaskGetInput,
            handle_tool_error=True,
        ),
    ]
