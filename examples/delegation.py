# ruff: noqa: N801, N805
"""Agent delegation — delegate tasks to specialist subagents.

AgentExtension enables a lead agent to delegate work to specialist
subagents at runtime. The lead decides when and to whom to delegate
via the Agent tool. Subagents run in isolation (scoped context)
and return concise results.

Key concepts:

- **Blocking delegation**: The lead waits for the subagent to finish.
  The tool-calling loop stays alive naturally.
- **Parallel delegation**: The LLM can call Agent multiple times
  in one turn — LangGraph's ToolNode runs them concurrently.
- **Scoped context**: Subagents receive only the task message, not
  the lead's full conversation history.
- **tools="inherit"**: Subagents can opt into receiving the parent's
  tools at delegation time.
- **Dynamic agents**: With ``ephemeral=True``, the lead can create
  one-shot agents with custom instructions at runtime.

Run::

    export OPENAI_API_KEY=...
    uv run python examples/delegation.py
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langchain_agentkit import AgentExtension, TasksExtension, agent


# ---------------------------------------------------------------------------
# 1. Define specialist agents
# ---------------------------------------------------------------------------

@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    return f"[Search results for '{query}']: The global SaaS market is $300B..."


@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))  # noqa: S307


class researcher(agent):
    """Specialist: information gathering."""

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    description = "Research specialist — gathers information from the web"
    tools = [web_search]
    prompt = "You are a research specialist. Answer questions using web_search. Be concise."

    async def handler(state, *, llm, tools, prompt):
        messages = [SystemMessage(content=prompt)] + state["messages"]
        return {"messages": [await llm.bind_tools(tools).ainvoke(messages)]}


class analyst(agent):
    """Specialist: data analysis and calculations."""

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    description = "Data analyst — performs calculations and analysis"
    tools = [calculator]
    prompt = "You are a data analyst. Use the calculator for any math. Be concise."

    async def handler(state, *, llm, tools, prompt):
        messages = [SystemMessage(content=prompt)] + state["messages"]
        return {"messages": [await llm.bind_tools(tools).ainvoke(messages)]}


# ---------------------------------------------------------------------------
# 2. Create the lead agent with delegation
# ---------------------------------------------------------------------------

class lead(agent):
    """Lead agent that delegates to specialists."""

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    extensions = [
        TasksExtension(),
        AgentExtension(
            [researcher, analyst],
            ephemeral=True,           # enable dynamic agents
            delegation_timeout=60.0,  # 60s max per delegation
        ),
    ]
    prompt = (
        "You are a project lead. Delegate research to the researcher "
        "and analysis to the analyst. Synthesize their results."
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
            {"messages": [HumanMessage("What is the global SaaS market size? Calculate 15% growth.")]}
        )

        print("=== Final Response ===")
        print(result["messages"][-1].content)

    asyncio.run(main())
