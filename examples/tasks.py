# ruff: noqa: N805
"""Task management — break complex objectives into tracked steps.

TasksExtension provides TaskCreate, TaskUpdate, TaskList, TaskGet, and
TaskStop tools. The agent decomposes work into tasks, tracks progress,
and manages dependencies via blocked_by.

The state schema is composed automatically — TasksExtension adds a
`tasks` key with a merge-by-ID reducer that handles parallel tool calls.
"""

import asyncio

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from langchain_agentkit import Agent, TasksExtension


class Planner(Agent):
    model = ChatOpenAI(model="gpt-4o")
    extensions = [TasksExtension()]
    prompt = """\
You are a project planner. When given a complex objective:
1. Break it into 3-7 tasks using TaskCreate
2. Set up dependencies with TaskUpdate (add_blocked_by) where needed
3. Work through tasks in order, marking each in_progress then completed
"""

    async def handler(state, *, llm, tools, prompt):
        messages = [SystemMessage(content=prompt)] + state["messages"]
        response = await llm.bind_tools(tools).ainvoke(messages)
        return {"messages": [response]}


async def main():
    graph = Planner().compile()
    result = await graph.ainvoke(
        {
            "messages": [
                HumanMessage(
                    "Set up a new Python project with: "
                    "1) project structure, 2) CI/CD pipeline, 3) documentation"
                )
            ]
        }
    )

    # Print the final task list from state
    for task in result.get("tasks", []):
        status = task.get("status", "pending")
        icon = {"completed": "x", "in_progress": ">", "deleted": "-"}.get(status, " ")
        blocked = task.get("blocked_by", [])
        suffix = f" (blocked by: {', '.join(blocked)})" if blocked else ""
        print(f"  [{icon}] {task['subject']}{suffix}")

    print(f"\nAgent: {result['messages'][-1].content[:200]}")


if __name__ == "__main__":
    asyncio.run(main())
