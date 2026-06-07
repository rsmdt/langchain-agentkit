# ruff: noqa: N805
"""Rubric-gated iteration — grade the agent's work and revise until it's done.

RubricExtension reviews the agent's output against a caller-supplied rubric
using a separate (cheaper) grader sub-agent. In ``auto`` mode the agent
self-revises in a closed loop until every criterion is satisfied or the review
budget runs out — no human in the loop.

This example uses a *forcing function*: the rubric demands an exact marker the
agent has no way to know up front (only the grader's rubric contains it), so the
first draft cannot pass and the loop must run for the marker to appear — making
the iteration visible when you run it.

The grader's bookkeeping is private state, so the verdict is observed here via
the ``on_evaluation`` callback (no checkpointer needed); ``get_state`` and the
``rubric_evaluation_*`` stream events are the other channels.

A trivial ``scratch`` tool is registered because the ``auto`` revision loop
routes via ``jump_to``, which the graph only honors when the kit has ≥1 tool.
"""

import asyncio

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langchain_agentkit import Agent
from langchain_agentkit.extensions.rubric import (
    ReviewPolicy,
    RubricEvaluation,
    RubricExtension,
)

# Each grading round is appended here so we can print the loop afterwards.
evaluations: list[RubricEvaluation] = []


@tool
def scratch(note: str) -> str:
    """Jot a private working note. Has no side effects."""
    return "noted"


class Writer(Agent):
    model = ChatOpenAI(model="gpt-4o")
    tools = [scratch]
    extensions = [
        RubricExtension(
            model=ChatOpenAI(model="gpt-4o-mini"),  # cheaper grader model
            mode="auto",
            review=ReviewPolicy(stop=4),  # up to 4 review rounds, then give up
            on_evaluation=evaluations.append,
        ),
    ]
    prompt = "You are a concise assistant. Answer the user directly."

    async def handler(state, *, llm, tools, prompt):
        bound = llm.bind_tools(tools) if tools else llm
        messages = [SystemMessage(content=prompt)] + state["messages"]
        return {"messages": [await bound.ainvoke(messages)]}


async def main():
    graph = await Writer().compile()
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage("In one sentence, what is the capital of France?")],
            # The agent is NOT told these criteria — only the grader is. The
            # marker is unguessable, so the loop must revise for it to appear.
            "rubric": (
                "- The answer states that the capital of France is Paris\n"
                "- The response ends with this exact marker on its own line: DONE-7Q2"
            ),
        }
    )

    print("Grading rounds:")
    for e in evaluations:
        open_gaps = [c["name"] for c in e["criteria"] if not c["passed"]]
        suffix = f" (open: {', '.join(open_gaps)})" if open_gaps else ""
        print(f"  [{e['iteration']}] {e['result']}{suffix}")

    print(f"\nFinal answer:\n{result['messages'][-1].content}")


if __name__ == "__main__":
    asyncio.run(main())
