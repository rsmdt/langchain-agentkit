"""Eval datasets for tool usage verification.

Each dataset entry defines:
- ``inputs``: User message triggering tool usage
- ``reference_trajectory``: Expected tool call sequence (OpenAI message format)

Trajectory messages use OpenAI format with ``tool_calls``::

    {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "Skill", "arguments": '{"skill_name": "web-research"}'}}
    ]}
"""

from __future__ import annotations

import json
from typing import Any


def _tool_call(name: str, **kwargs: Any) -> dict:
    """Helper to build a tool call dict in OpenAI format."""
    return {
        "id": "",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(kwargs),
        },
    }


def _assistant_with_tools(*tool_calls: dict) -> dict:
    """Helper to build an assistant message with tool calls."""
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": list(tool_calls),
    }


# --- Skill Loading Scenarios ---

SKILL_LOADING_DATASET = [
    {
        "description": "Agent loads skill when user request matches skill domain",
        "inputs": "Can you help me estimate the market size for electric scooters?",
        "reference_trajectory": [
            _assistant_with_tools(
                _tool_call("Skill", skill_name="market-sizing"),
            ),
        ],
    },
    {
        "description": "Agent does NOT load skill for simple question",
        "inputs": "What is 2 + 2?",
        "reference_trajectory": [],  # No tool calls expected
    },
]


# --- Read Tool Scenarios ---

READ_TOOL_DATASET = [
    {
        "description": "Agent reads a specific file at a given path",
        "inputs": "Read the file at /workspace/notes.txt",
        "reference_trajectory": [
            _assistant_with_tools(
                _tool_call("Read", file_path="/workspace/notes.txt"),
            ),
        ],
    },
    {
        "description": "Agent reads config file when asked about configuration",
        "inputs": "Show me the contents of /workspace/config.json",
        "reference_trajectory": [
            _assistant_with_tools(
                _tool_call("Read", file_path="/workspace/config.json"),
            ),
        ],
    },
]


# --- Write Tool Scenarios ---

WRITE_TOOL_DATASET = [
    {
        "description": "Agent writes output to a file",
        "inputs": "Save the text 'Hello World' to /output/hello.txt",
        "reference_trajectory": [
            _assistant_with_tools(
                _tool_call("Write", file_path="/output/hello.txt", content="Hello World"),
            ),
        ],
    },
]


# --- Edit Tool Scenarios ---

EDIT_TOOL_DATASET = [
    {
        "description": "Agent edits a file with exact string replacement",
        "inputs": ("In /workspace/config.json, replace 'debug: true' with 'debug: false'"),
        "reference_trajectory": [
            _assistant_with_tools(
                _tool_call(
                    "Edit",
                    file_path="/workspace/config.json",
                    old_string="debug: true",
                    new_string="debug: false",
                ),
            ),
        ],
    },
]


# --- Glob Tool Scenarios ---

GLOB_TOOL_DATASET = [
    {
        "description": "Agent uses Glob to find all text files",
        "inputs": "Find all .txt files in the workspace",
        "reference_trajectory": [
            _assistant_with_tools(
                _tool_call("Glob", pattern="**/*.txt"),
            ),
        ],
    },
]


# --- Grep Tool Scenarios ---

GREP_TOOL_DATASET = [
    {
        "description": "Agent uses Grep to search for a pattern",
        "inputs": "Search for 'TODO' in all files",
        "reference_trajectory": [
            _assistant_with_tools(
                _tool_call("Grep", pattern="TODO"),
            ),
        ],
    },
]


# --- Multi-Step Scenarios ---

MULTI_STEP_DATASET = [
    {
        "description": "Agent loads skill when asked about market sizing",
        "inputs": "I need to do market sizing. Load the relevant skill.",
        "reference_trajectory": [
            _assistant_with_tools(
                _tool_call("Skill", skill_name="market-sizing"),
            ),
        ],
    },
    {
        "description": "Agent discovers files with Glob, then reads one",
        "inputs": "What files are in /workspace? Then read the notes file.",
        "reference_trajectory": [
            _assistant_with_tools(
                _tool_call("Glob"),
            ),
            _assistant_with_tools(
                _tool_call("Read"),
            ),
        ],
    },
]


# --- Task Tool Scenarios ---

TASK_CREATE_DATASET = [
    {
        "description": "Agent creates tasks for a multi-step request",
        "inputs": (
            "I need you to: 1) research competitor pricing, "
            "2) draft a pricing proposal, 3) review the proposal"
        ),
        "reference_trajectory": [
            _assistant_with_tools(
                _tool_call("TaskCreate", subject="Research competitor pricing", description=""),
            ),
            _assistant_with_tools(
                _tool_call("TaskCreate", subject="Draft a pricing proposal", description=""),
            ),
            _assistant_with_tools(
                _tool_call("TaskCreate", subject="Review the proposal", description=""),
            ),
        ],
    },
    {
        "description": "Agent does NOT create tasks for a trivial request",
        "inputs": "What time is it?",
        "reference_trajectory": [],  # No tool calls expected
    },
]


TASK_LIFECYCLE_DATASET = [
    {
        "description": "Agent creates task and marks it completed",
        "inputs": ("Create a task 'Write unit tests' and then mark it done."),
        "reference_trajectory": [
            _assistant_with_tools(
                _tool_call("TaskCreate", subject="Write unit tests", description=""),
            ),
            _assistant_with_tools(
                _tool_call("TaskUpdate", status="completed"),
            ),
        ],
    },
]


TASK_DEPENDENCIES_DATASET = [
    {
        "description": "Agent creates tasks with dependencies for sequential work",
        "inputs": (
            "Plan this work: first set up the database schema, then write the API "
            "endpoints (depends on schema), then write integration tests (depends on API). "
            "Create all tasks with proper dependencies."
        ),
        "reference_trajectory": [
            _assistant_with_tools(
                _tool_call("TaskCreate", subject="Set up database schema", description=""),
            ),
            _assistant_with_tools(
                _tool_call("TaskCreate", subject="Write API endpoints", description=""),
            ),
            _assistant_with_tools(
                _tool_call("TaskCreate", subject="Write integration tests", description=""),
            ),
            _assistant_with_tools(
                _tool_call("TaskUpdate"),
            ),
        ],
    },
]


TASK_LIST_DATASET = [
    {
        "description": "Agent lists tasks when asked for status",
        "inputs": "What tasks do we have? Show me the current status.",
        "reference_trajectory": [
            _assistant_with_tools(
                _tool_call("TaskList"),
            ),
        ],
    },
]


TASK_STOP_DATASET = [
    {
        "description": "Agent stops a running task when asked",
        "inputs": "Stop the task that is currently running.",
        "reference_trajectory": [
            _assistant_with_tools(
                _tool_call("TaskList"),
            ),
            _assistant_with_tools(
                _tool_call("TaskStop"),
            ),
        ],
    },
]


# --- LS Tool Scenarios ---

LS_TOOL_DATASET = [
    {
        "description": "Agent lists directory contents when asked",
        "inputs": "What files are in the /data directory?",
        # LS tool removed — agent uses Glob or Bash to list files
        "reference_trajectory": [
            _assistant_with_tools(
                _tool_call("Glob"),
            ),
        ],
    },
]


# --- MultiEdit Tool Scenarios ---

MULTI_EDIT_TOOL_DATASET = [
    {
        "description": "Agent applies multiple edits to one file",
        "inputs": (
            "In /workspace/config.json, make these changes: "
            "replace 'debug' with 'verbose' and replace 'true' with 'false'"
        ),
        # Accept either MultiEdit once or Edit twice — both are correct
        # strategies. Subset mode: at least one Edit-family call on the file.
        "reference_trajectory": [
            _assistant_with_tools(
                _tool_call("Edit", file_path="/workspace/config.json"),
            ),
        ],
    },
]


# --- Filesystem Multi-Step Scenarios ---

FILESYSTEM_MULTI_STEP_DATASET = [
    {
        "description": "Agent discovers files, reads one, then writes analysis",
        "inputs": (
            "Find all .txt files in the workspace, read each one, "
            "and write a summary to /workspace/summary.md"
        ),
        "reference_trajectory": [
            _assistant_with_tools(
                _tool_call("Glob", pattern="**/*.txt"),
            ),
            _assistant_with_tools(
                _tool_call("Read"),
            ),
            _assistant_with_tools(
                _tool_call("Write", file_path="/workspace/summary.md"),
            ),
        ],
    },
    {
        "description": "Agent searches for pattern then reads matching file",
        "inputs": "Search for 'TODO' in the workspace files and show me the matches",
        "reference_trajectory": [
            _assistant_with_tools(
                _tool_call("Grep", pattern="TODO"),
            ),
        ],
    },
]


# --- HITL Ask-User Scenarios ---

HITL_ASK_USER_DATASET = [
    {
        "description": "Agent uses AskUser when choosing between databases",
        "inputs": (
            "We need to add a database to this project. What type of database should I set up?"
        ),
        "reference_trajectory": [
            _assistant_with_tools(_tool_call("AskUser")),
        ],
    },
    {
        "description": "Agent uses AskUser when configuring logging",
        "inputs": (
            "Configure the logging system for this application. "
            "There are several approaches we could take."
        ),
        "reference_trajectory": [
            _assistant_with_tools(_tool_call("AskUser")),
        ],
    },
]


HITL_DIRECT_ACTION_DATASET = [
    {
        "description": "Agent reads file directly without asking",
        "inputs": "Read the contents of /workspace/config.json",
        "reference_trajectory": [
            _assistant_with_tools(_tool_call("Read")),
        ],
    },
    {
        "description": "Agent searches files directly without asking",
        "inputs": "Find all Python files in the workspace",
        "reference_trajectory": [
            _assistant_with_tools(_tool_call("Glob")),
        ],
    },
]


# --- Combined dataset for full eval run ---

ALL_DATASETS = {
    "skill_loading": SKILL_LOADING_DATASET,
    "read_tool": READ_TOOL_DATASET,
    "write_tool": WRITE_TOOL_DATASET,
    "edit_tool": EDIT_TOOL_DATASET,
    "glob_tool": GLOB_TOOL_DATASET,
    "grep_tool": GREP_TOOL_DATASET,
    "ls_tool": LS_TOOL_DATASET,
    "multi_edit_tool": MULTI_EDIT_TOOL_DATASET,
    "multi_step": MULTI_STEP_DATASET,
    "filesystem_multi_step": FILESYSTEM_MULTI_STEP_DATASET,
    "task_create": TASK_CREATE_DATASET,
    "task_lifecycle": TASK_LIFECYCLE_DATASET,
    "task_dependencies": TASK_DEPENDENCIES_DATASET,
    "task_list": TASK_LIST_DATASET,
    "task_stop": TASK_STOP_DATASET,
    "hitl_ask_user": HITL_ASK_USER_DATASET,
    "hitl_direct_action": HITL_DIRECT_ACTION_DATASET,
}
