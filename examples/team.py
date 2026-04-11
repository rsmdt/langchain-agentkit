# ruff: noqa: N805
"""Agent teams — coordinate concurrent specialists.

TeamExtension enables a lead agent to spawn a team of concurrent
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
- **Shared task list**: Teams reuse TasksExtension for coordination.
  TeamMessage sends work to members. The lead sees progress via the
  standard task tools.
- **Lifecycle**: TeamCreate → TeamMessage → (react to messages) → TeamDissolve.

Tools provided:

| Tool | Purpose |
|------|---------|
| TeamCreate | Create team with named members |
| TeamMessage | Send work, guidance, or follow-ups to a member |
| TeamStatus | See statuses and collect pending messages |
| TeamDissolve | Shut down team, collect final results |

Run::

    export OPENAI_API_KEY=...
    uv run python examples/team.py
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langchain_agentkit import Agent, TasksExtension, TeamExtension

# ---------------------------------------------------------------------------
# 1. Define worker agents
# ---------------------------------------------------------------------------


@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    return f"[Search results for '{query}']: Token bucket is the standard approach..."


class ResearchWorker(Agent):
    """Worker: researches topics and reports findings."""

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    description = "Research worker — investigates topics and reports concise findings"
    tools = [web_search]
    prompt = (
        "You are a research worker on a team. Answer the task you're given "
        "concisely. Use web_search when needed. Keep responses under 200 words."
    )

    async def handler(state, *, llm, tools, prompt):
        messages = [SystemMessage(content=prompt)] + state["messages"]
        return {"messages": [await llm.bind_tools(tools).ainvoke(messages)]}


class Coder(Agent):
    """Worker: writes code and implementation plans."""

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    description = "Coding worker — writes code, designs APIs, creates implementation plans"
    prompt = (
        "You are a coding worker on a team. Write concise code or plans "
        "for the task you're given. Keep responses focused and under 200 words."
    )

    async def handler(state, *, llm, prompt):
        messages = [SystemMessage(content=prompt)] + state["messages"]
        return {"messages": [await llm.ainvoke(messages)]}


# Build worker StateGraphs for use as team members
research_worker = ResearchWorker().graph()
coder = Coder().graph()

# ---------------------------------------------------------------------------
# 2. Create the lead agent with team coordination
# ---------------------------------------------------------------------------


class Lead(Agent):
    """Team lead that coordinates researcher and coder."""

    model = ChatOpenAI(model="gpt-4o", temperature=0)
    extensions = [
        TasksExtension(),
        TeamExtension(
            agents=[research_worker, coder],
            max_team_size=5,
            router_timeout=30.0,
        ),
    ]
    prompt = (
        "You are a project lead. Follow these steps:\n"
        "1. TeamCreate to create the workers you need\n"
        "2. TeamMessage to assign work to each worker\n"
        "3. React to teammate messages as they arrive\n"
        "4. Use TeamMessage to share info between workers\n"
        "5. TeamDissolve when all work is done\n"
        "6. Synthesize results and report to the user\n\n"
        "IMPORTANT: Always complete all steps. Never skip TeamDissolve."
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
        graph = Lead().compile()
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
        team = result.get("team") or {}
        for member in team.get("members", []):
            kind = member.get("kind", "?")
            descriptor = member.get("agent_id") or member.get("system_prompt", "?")[:40]
            print(f"  {member['member_name']} ({kind}): {descriptor}")

        print("\n=== Tasks ===")
        for task in result.get("tasks", []):
            status = task.get("status", "?")
            subject = task.get("subject", "?")
            owner = task.get("owner", "?")
            print(f"  [{status}] {subject} (owner: {owner})")

        print("\n=== Final Response ===")
        print(result["messages"][-1].content)

    asyncio.run(main())
