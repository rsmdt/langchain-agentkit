# ruff: noqa: N801, N805
"""Agent teams — coordinate concurrent specialists.

AgentTeamMiddleware enables a lead agent to spawn a team of concurrent
workers, assign tasks, exchange messages, and synthesize results. This
is the most powerful multi-agent pattern — use it when work requires
back-and-forth coordination between specialists.

Key concepts:

- **Message-driven**: Teammates run as asyncio.Tasks. When a teammate
  finishes, its result is delivered to the lead automatically via the
  Router Node — the lead reacts, it doesn't poll.
- **Conversation history**: Each teammate has its own checkpointer.
  Multiple messages to the same teammate accumulate context (the teammate
  remembers previous interactions within the team session).
- **Shared task list**: Teams reuse TasksMiddleware for coordination.
  AssignTask creates a tracked task with an owner. The lead sees
  progress via the standard task tools.
- **Lifecycle**: SpawnTeam → AssignTask → (react to messages) → DissolveTeam.

Tools provided:

| Tool | Purpose |
|------|---------|
| SpawnTeam | Create team with named members |
| AssignTask | Assign work to a specific member |
| MessageTeammate | Send guidance or follow-up to a member |
| CheckTeammates | See statuses and collect pending messages |
| DissolveTeam | Shut down team, collect final results |

Run::

    export OPENAI_API_KEY=...
    uv run python examples/team.py
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langchain_agentkit import AgentTeamMiddleware, TasksMiddleware, agent


# ---------------------------------------------------------------------------
# 1. Define worker agents
# ---------------------------------------------------------------------------

@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    return f"[Search results for '{query}']: Token bucket is the standard approach..."


class researcher(agent):
    """Worker: researches topics and reports findings."""

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    description = "Research worker — investigates topics and reports concise findings"
    tools = [web_search]
    prompt = (
        "You are a research worker on a team. Answer the task you're given "
        "concisely. Use web_search when needed. Keep responses under 200 words."
    )

    async def handler(state, *, llm, tools, prompt):
        messages = [SystemMessage(content=prompt)] + state["messages"]
        return {"messages": [await llm.bind_tools(tools).ainvoke(messages)]}


class coder(agent):
    """Worker: writes code and implementation plans."""

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    description = "Coding worker — writes code, designs APIs, creates implementation plans"
    prompt = (
        "You are a coding worker on a team. Write concise code or plans "
        "for the task you're given. Keep responses focused and under 200 words."
    )

    async def handler(state, *, llm, prompt):
        messages = [SystemMessage(content=prompt)] + state["messages"]
        return {"messages": [await llm.ainvoke(messages)]}


# ---------------------------------------------------------------------------
# 2. Create the lead agent with team coordination
# ---------------------------------------------------------------------------

class lead(agent):
    """Team lead that coordinates researcher and coder."""

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    middleware = [
        TasksMiddleware(),
        AgentTeamMiddleware(
            [researcher, coder],
            max_team_size=5,
            router_timeout=30.0,
        ),
    ]
    prompt = (
        "You are a project lead. Follow these steps:\n"
        "1. SpawnTeam with the workers you need\n"
        "2. AssignTask to each worker\n"
        "3. React to teammate messages as they arrive\n"
        "4. Use MessageTeammate to share info between workers\n"
        "5. DissolveTeam when all work is done\n"
        "6. Synthesize results and report to the user\n\n"
        "IMPORTANT: Always complete all steps. Never skip DissolveTeam."
    )

    async def handler(state, *, llm, tools, prompt):
        messages = [SystemMessage(content=prompt)] + state["messages"]
        return {"messages": [await llm.bind_tools(tools).ainvoke(messages)]}


# ---------------------------------------------------------------------------
# 3. Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio

    async def main():
        graph = lead.compile()
        result = await graph.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        "Build a rate-limited API. Have the researcher investigate "
                        "rate limiting best practices, and the coder design the API. "
                        "Share the researcher's findings with the coder."
                    )
                ]
            },
            {"recursion_limit": 40},
        )

        print("=== Team Members (final) ===")
        for member in result.get("team_members", []):
            print(f"  {member['name']} ({member.get('agent_type', '?')}): {member.get('status', '?')}")

        print("\n=== Tasks ===")
        for task in result.get("tasks", []):
            print(f"  [{task.get('status', '?')}] {task.get('subject', '?')} (owner: {task.get('owner', '?')})")

        print("\n=== Final Response ===")
        print(result["messages"][-1].content)

    asyncio.run(main())
