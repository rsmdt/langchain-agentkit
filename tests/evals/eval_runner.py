"""Eval runner for tool usage trajectory evaluation.

Runs an agent against eval datasets and scores tool call trajectories
using ``agentevals`` trajectory matching.

Usage::

    from tests.evals.eval_runner import run_eval, EvalResult

    results = run_eval(
        agent=my_compiled_graph,
        dataset=SKILL_LOADING_DATASET,
        trajectory_mode="subset",
    )
    for r in results:
        print(f"{r['description']}: {'PASS' if r['score'] else 'FAIL'}")
"""

from __future__ import annotations

import json
from typing import Any, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage

TrajectoryMode = Literal["strict", "unordered", "subset", "superset"]


class EvalResult(TypedDict):
    """Result of evaluating a single dataset entry."""

    description: str
    score: bool
    actual_tool_calls: list[dict[str, Any]]
    expected_tool_calls: list[dict[str, Any]]
    comment: str


def extract_tool_calls_from_messages(
    messages: list[BaseMessage],
) -> list[dict[str, Any]]:
    """Extract tool call names and args from LangChain messages.

    Returns list of ``{"name": str, "args": dict}`` dicts.
    """
    tool_calls: list[dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(
                    {
                        "name": tc["name"],
                        "args": tc["args"],
                    }
                )
    return tool_calls


def extract_tool_calls_from_openai_trajectory(
    trajectory: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extract tool call names and args from OpenAI-format trajectory."""
    tool_calls: list[dict[str, Any]] = []
    for msg in trajectory:
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                func = tc.get("function", {})
                tool_calls.append(
                    {
                        "name": func.get("name", ""),
                        "args": json.loads(func.get("arguments", "{}")),
                    }
                )
    return tool_calls


def _find_in_remaining(
    needle: dict[str, Any],
    haystack: list[dict[str, Any]],
    tool_args_mode: str,
) -> int | None:
    """Find first match for *needle* in *haystack*, return index or None."""
    for j, candidate in enumerate(haystack):
        if candidate["name"] != needle["name"]:
            continue
        if tool_args_mode == "ignore" or _args_match(
            candidate["args"],
            needle["args"],
            tool_args_mode,
        ):
            return j
    return None


def _match_strict(
    actual: list[dict[str, Any]],
    expected: list[dict[str, Any]],
    tool_args_mode: str,
) -> tuple[bool, str]:
    if len(actual) != len(expected):
        return False, (
            f"Expected {len(expected)} tool calls, got {len(actual)}. "
            f"Expected: {[e['name'] for e in expected]}, "
            f"Got: {[a['name'] for a in actual]}"
        )
    for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
        if a["name"] != e["name"]:
            return False, f"Step {i}: expected {e['name']}, got {a['name']}"
        if not _args_match(a["args"], e["args"], tool_args_mode):
            return False, (
                f"Step {i} ({a['name']}): args mismatch. Expected {e['args']}, got {a['args']}"
            )
    return True, "Strict match"


def _match_find_all(
    needles: list[dict[str, Any]],
    haystack: list[dict[str, Any]],
    tool_args_mode: str,
    label: str,
) -> tuple[bool, str]:
    """Check every needle has a match in haystack (consuming matches)."""
    remaining = list(haystack)
    for needle in needles:
        idx = _find_in_remaining(needle, remaining, tool_args_mode)
        if idx is None:
            return False, (
                f"{label}: {needle['name']} not found. Remaining: {[r['name'] for r in remaining]}"
            )
        remaining.pop(idx)
    return True, f"{label} match"


def match_tool_calls(
    actual: list[dict[str, Any]],
    expected: list[dict[str, Any]],
    mode: TrajectoryMode = "subset",
    tool_args_mode: Literal["exact", "ignore", "subset"] = "ignore",
) -> tuple[bool, str]:
    """Compare actual tool calls against expected trajectory.

    Args:
        actual: Tool calls the agent actually made.
        expected: Tool calls the agent should have made.
        mode: How to compare trajectories.
        tool_args_mode: How to compare tool arguments.

    Returns:
        (passed, comment) tuple.
    """
    if not expected and not actual:
        return True, "No tool calls expected or made"
    if not expected and actual:
        return False, f"Expected no tool calls but got {len(actual)}"
    if not actual and expected:
        return False, f"Expected {len(expected)} tool calls but got none"

    if mode == "strict":
        return _match_strict(actual, expected, tool_args_mode)
    elif mode == "subset":
        return _match_find_all(expected, actual, tool_args_mode, "Subset")
    elif mode == "unordered":
        if len(actual) != len(expected):
            return False, (f"Expected {len(expected)} tool calls, got {len(actual)}")
        return _match_find_all(expected, actual, tool_args_mode, "Unordered")
    elif mode == "superset":
        return _match_find_all(actual, expected, tool_args_mode, "Superset")

    return False, f"Unknown mode: {mode}"


def _args_match(
    actual: dict,
    expected: dict,
    mode: str,
) -> bool:
    if mode == "exact":
        return actual == expected
    elif mode == "subset":
        return all(k in actual and actual[k] == v for k, v in expected.items())
    return True


def run_eval(
    agent: Any,
    dataset: list[dict[str, Any]],
    *,
    trajectory_mode: TrajectoryMode = "subset",
    tool_args_mode: Literal["exact", "ignore", "subset"] = "ignore",
    state_factory: Any | None = None,
) -> list[EvalResult]:
    """Run evaluation dataset against a compiled LangGraph agent.

    Args:
        agent: Compiled LangGraph ``StateGraph`` with ``.invoke()``.
        dataset: List of eval entries with ``inputs``, ``reference_trajectory``.
        trajectory_mode: How to compare tool call sequences.
        tool_args_mode: How to compare tool arguments.
        state_factory: Optional callable returning initial state dict.

    Returns:
        List of ``EvalResult`` dicts with scores and diagnostics.
    """
    results: list[EvalResult] = []

    for entry in dataset:
        description = entry["description"]
        user_input = entry["inputs"]
        reference = entry["reference_trajectory"]

        # Build initial state
        if state_factory:
            state = state_factory()
        else:
            state = {"messages": [{"role": "user", "content": user_input}]}
        # Ensure messages always contains the user input
        if "messages" not in state:
            state["messages"] = [{"role": "user", "content": user_input}]

        # Run agent (async tools require ainvoke)
        try:
            import asyncio

            final_state = asyncio.run(agent.ainvoke(state))
            actual_messages = final_state.get("messages", [])
            actual_calls = extract_tool_calls_from_messages(actual_messages)
        except Exception as exc:
            results.append(
                EvalResult(
                    description=description,
                    score=False,
                    actual_tool_calls=[],
                    expected_tool_calls=extract_tool_calls_from_openai_trajectory(
                        reference,
                    ),
                    comment=f"Agent error: {exc}",
                )
            )
            continue

        expected_calls = extract_tool_calls_from_openai_trajectory(reference)

        passed, comment = match_tool_calls(
            actual_calls,
            expected_calls,
            trajectory_mode,
            tool_args_mode,
        )

        results.append(
            EvalResult(
                description=description,
                score=passed,
                actual_tool_calls=actual_calls,
                expected_tool_calls=expected_calls,
                comment=comment,
            )
        )

    return results


def print_eval_results(results: list[EvalResult]) -> None:
    """Pretty-print eval results to stdout."""
    total = len(results)
    passed = sum(1 for r in results if r["score"])

    print(f"\n{'=' * 60}")
    print(f"EVAL RESULTS: {passed}/{total} passed")
    print(f"{'=' * 60}\n")

    for r in results:
        status = "PASS" if r["score"] else "FAIL"
        icon = "✓" if r["score"] else "✗"
        print(f"  {icon} [{status}] {r['description']}")
        if not r["score"]:
            print(f"    Comment: {r['comment']}")
            if r["expected_tool_calls"]:
                names = [tc["name"] for tc in r["expected_tool_calls"]]
                print(f"    Expected: {names}")
            if r["actual_tool_calls"]:
                names = [tc["name"] for tc in r["actual_tool_calls"]]
                print(f"    Actual:   {names}")
        print()
