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
        "description": "Agent reads a skill reference file via Read tool",
        "inputs": "Read the calculator.py reference file from the market-sizing skill",
        "reference_trajectory": [
            _assistant_with_tools(
                _tool_call("Read", file_path="/skills/market-sizing/calculator.py"),
            ),
        ],
    },
    {
        "description": "Agent reads a specific file at a given path",
        "inputs": "Read the file at /workspace/notes.txt",
        "reference_trajectory": [
            _assistant_with_tools(
                _tool_call("Read", file_path="/workspace/notes.txt"),
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
        "inputs": (
            "In /workspace/config.json, replace 'debug: true' with 'debug: false'"
        ),
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
        "description": "Agent uses Glob to find all markdown files",
        "inputs": "Find all markdown files under /skills/",
        "reference_trajectory": [
            _assistant_with_tools(
                _tool_call("Glob", pattern="/skills/**/*.md"),
            ),
        ],
    },
]


# --- Grep Tool Scenarios ---

GREP_TOOL_DATASET = [
    {
        "description": "Agent uses Grep to search for a pattern",
        "inputs": "Search for 'TODO' in all Python files",
        "reference_trajectory": [
            _assistant_with_tools(
                _tool_call("Grep", pattern="TODO", glob="/skills/**/*.py"),
            ),
        ],
    },
]


# --- Multi-Step Scenarios ---

MULTI_STEP_DATASET = [
    {
        "description": "Agent loads skill, then reads reference file",
        "inputs": "I need to do market sizing. Load the skill and show me the calculator.",
        "reference_trajectory": [
            _assistant_with_tools(
                _tool_call("Skill", skill_name="market-sizing"),
            ),
            _assistant_with_tools(
                _tool_call("Read", file_path="/skills/market-sizing/calculator.py"),
            ),
        ],
    },
    {
        "description": "Agent discovers files with Glob, then reads one",
        "inputs": "What skill files are available? Then read the first one.",
        "reference_trajectory": [
            _assistant_with_tools(
                _tool_call("Glob", pattern="/skills/**/*"),
            ),
            _assistant_with_tools(
                _tool_call("Read", file_path="/skills/market-sizing/SKILL.md"),
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
        "description": "Agent creates task, marks in_progress, then completed",
        "inputs": (
            "Create a task 'Write unit tests' and then immediately start working on it "
            "and mark it done."
        ),
        "reference_trajectory": [
            _assistant_with_tools(
                _tool_call("TaskCreate", subject="Write unit tests", description=""),
            ),
            _assistant_with_tools(
                _tool_call("TaskUpdate", status="in_progress"),
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


# --- Combined dataset for full eval run ---

ALL_DATASETS = {
    "skill_loading": SKILL_LOADING_DATASET,
    "read_tool": READ_TOOL_DATASET,
    "write_tool": WRITE_TOOL_DATASET,
    "edit_tool": EDIT_TOOL_DATASET,
    "glob_tool": GLOB_TOOL_DATASET,
    "grep_tool": GREP_TOOL_DATASET,
    "multi_step": MULTI_STEP_DATASET,
    "task_create": TASK_CREATE_DATASET,
    "task_lifecycle": TASK_LIFECYCLE_DATASET,
    "task_dependencies": TASK_DEPENDENCIES_DATASET,
    "task_list": TASK_LIST_DATASET,
    "task_stop": TASK_STOP_DATASET,
}
