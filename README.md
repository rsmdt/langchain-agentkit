# langchain-agentkit

Composable extension framework for LangGraph agents.

[![PyPI](https://img.shields.io/pypi/v/langchain-agentkit)](https://pypi.org/project/langchain-agentkit/)
[![Python 3.11+](https://img.shields.io/pypi/pyversions/langchain-agentkit)](https://pypi.org/project/langchain-agentkit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Installation

```bash
pip install langchain-agentkit
```

Requires Python 3.11+.

## Quick Start

### The `agent` metaclass

Declare a class, get a complete ReAct agent with extension-composed tools and prompts:

```python
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_agentkit import agent, SkillsExtension, TasksExtension

class researcher(agent):
    model = ChatOpenAI(model="gpt-4o")
    extensions = [
        SkillsExtension(skills="skills/"),
        TasksExtension(),
    ]
    prompt = "You are a research assistant."

    async def handler(state, *, llm, prompt):
        messages = [SystemMessage(content=prompt)] + state["messages"]
        return {"messages": [await llm.ainvoke(messages)]}

graph = researcher.compile()
result = graph.invoke({"messages": [HumanMessage("Size the B2B SaaS market")]})
```

The `model` attribute accepts a `BaseChatModel` instance (used as-is) or a string resolved via `AgentKit.model_resolver`:

```python
kit = AgentKit(
    extensions=[...],
    model_resolver=lambda name: ChatOpenAI(model=name),
)

class fast_agent(agent):
    model = "gpt-4o-mini"  # resolved via model_resolver
    ...
```

The state schema is composed automatically from extensions — `TasksExtension` adds a `tasks` key, `SkillsExtension` adds nothing. No need to define state manually.

### `AgentKit` for manual graph wiring

Use `AgentKit` when you need full control over graph topology — custom routing, multi-node graphs, or a shared `ToolNode`:

```python
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from langchain_agentkit import AgentKit, SkillsExtension, TasksExtension

kit = AgentKit([
    SkillsExtension(skills="skills/"),
    TasksExtension(),
])

llm = ChatOpenAI(model="gpt-4o")
all_tools = kit.tools
bound_llm = llm.bind_tools(all_tools)

def agent_node(state):
    prompt = kit.prompt(state)
    messages = [SystemMessage(content=prompt)] + state["messages"]
    return {"messages": [bound_llm.invoke(messages)]}

def should_continue(state):
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END

# State schema composed automatically from extensions
graph = StateGraph(kit.state_schema)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(all_tools))
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

app = graph.compile()
result = app.invoke({"messages": [HumanMessage("Size the B2B SaaS market")]})
```

## Extensions

Each extension provides tools, a prompt section, and optional state requirements. Compose them in any combination:

```python
extensions = [
    SkillsExtension(skills="skills/"),
    TasksExtension(),
    FilesystemExtension(),
    WebSearchExtension(),
    HITLExtension(interrupt_on={"send_email": True}, tools=True),
    AgentExtension(agents=[researcher, coder]),
    TeamExtension(agents=[researcher, coder]),
]
```

### SkillsExtension

Loads skills and provides progressive disclosure — the agent sees skill names and descriptions, then loads full content on demand via the `Skill` tool.

Two input modes:

```python
from langchain_agentkit import SkillsExtension, SkillConfig

# Programmatic — pass SkillConfig objects directly
ext = SkillsExtension(skills=[
    SkillConfig(name="market-sizing", description="Calculate TAM/SAM/SOM", prompt="..."),
])

# Directory discovery — scan a directory for SKILL.md files
ext = SkillsExtension(skills="skills/")

# With a custom backend (e.g. Daytona sandbox)
ext = SkillsExtension(skills="/skills", backend=my_backend)
```

Always provides exactly one tool: `Skill`. Filesystem tools (Read, Write, etc.) come from `FilesystemExtension`.

**Tools:**

| Tool | Description |
|------|-------------|
| `Skill(skill_name)` | Load a skill's prompt content |

Skill directories follow the [AgentSkills.io](https://agentskills.io/specification) format:

```
skills/
└── market-sizing/
    ├── SKILL.md          # YAML frontmatter (name, description) + prompt body
    └── calculator.py     # Reference files accessible via Read tool
```

### AgentExtension

Delegate tasks to specialist subagents at runtime. Accepts compiled StateGraphs, `AgentConfig` definitions, or discovers agents from a directory of markdown files.

```python
from langchain_agentkit import agent, AgentExtension, AgentConfig

class researcher(agent):
    model = ChatOpenAI(model="gpt-4o-mini")
    description = "Research specialist for information gathering"
    tools = [web_search]
    prompt = "You are a research specialist."
    async def handler(state, *, llm, tools, prompt): ...

# Programmatic — mix compiled graphs and AgentConfig definitions
ext = AgentExtension(agents=[
    researcher,                                          # compiled StateGraph
    AgentConfig(name="coder", description="Code expert", prompt="You code."),
])

# Directory discovery — scan for .md files with frontmatter
ext = AgentExtension(agents="agents/")

# With a custom backend
ext = AgentExtension(agents="/agents", backend=my_backend)
```

**AgentConfig** supports the same frontmatter fields as file-based agents:

```python
AgentConfig(
    name="researcher",
    description="Research specialist",
    prompt="You are a research assistant.",
    model="gpt-4o-mini",            # resolved via model_resolver
    tools=["WebSearch", "Read"],     # filtered from parent's tools
    skills=["api-conventions"],      # preloaded into prompt at delegation time
    max_turns=10,                    # recursion limit
)
```

**File-based agent** (`agents/researcher.md`):

```yaml
---
name: researcher
description: Research specialist
model: gpt-4o-mini
tools: WebSearch, Read
skills: api-conventions, error-handling
maxTurns: 10
---
You are a research assistant.
```

**The `Agent` tool uses shape-based discrimination** — the LLM provides either `{id: "<name>"}` for a pre-defined agent or `{prompt: "..."}` for a dynamic one:

```json
{"agent": {"id": "researcher"}, "message": "Find info on X"}
{"agent": {"prompt": "You are a legal expert..."}, "message": "Analyze this contract"}
```

**Key features:**
- `description` — used in the prompt roster so the LLM knows what each specialist does
- `tools="inherit"` — subagent receives the parent's tools at delegation time
- `ephemeral=True` — enables dynamic (on-the-fly) reasoning agents
- `skills` preloading — full skill content injected into agent's prompt at startup
- `model` override — per-agent model selection via `model_resolver`
- `delegation_timeout` — max seconds per delegation (default 300s)

See [`examples/delegation.py`](examples/delegation.py) for a complete example.

### TasksExtension

Task management for complex multi-step objectives. The agent creates, tracks, and completes tasks with dependency ordering.

```python
ext = TasksExtension()
ext.tools  # [TaskCreate, TaskUpdate, TaskList, TaskGet, TaskStop]
```

**Tools:**

| Tool | Description |
|------|-------------|
| `TaskCreate` | Create a task with subject, description, and optional spinner text |
| `TaskUpdate` | Update status, owner, metadata, or dependencies |
| `TaskList` | List all non-deleted tasks with status and dependencies |
| `TaskGet` | Get full task details including computed `blocks` |
| `TaskStop` | Stop a running task |

Tasks support `blocked_by` dependencies, `owner` assignment, and arbitrary `metadata`. Parallel `TaskCreate` calls are handled by a merge-by-ID reducer.

### FilesystemExtension

File tools operating on the OS filesystem via `OSBackend`:

```python
from langchain_agentkit import FilesystemExtension

# Current working directory
ext = FilesystemExtension()

# Scoped to a specific directory (with path traversal prevention)
ext = FilesystemExtension(root="./workspace")
```

**Tools:**

| Tool | Description |
|------|-------------|
| `Read(file_path)` | Read file with line numbers, offset/limit pagination |
| `Write(file_path, content)` | Create or overwrite a file |
| `Edit(file_path, old_string, new_string)` | Exact string replacement |
| `Glob(pattern)` | Find files by pattern (supports `*`, `**`, `?`) |
| `Grep(pattern)` | Search file contents by regex |
| `Bash(command)` | Execute shell commands (when backend supports `execute()`) |

### WebSearchExtension

Multi-provider web search. Fans out queries to all providers in parallel. Ships with two built-in providers (no API key needed):

```python
from langchain_agentkit import WebSearchExtension, DuckDuckGoSearchProvider

# Zero config (defaults to Qwant)
ext = WebSearchExtension()

# DuckDuckGo (recommended — more reliable)
ext = WebSearchExtension(providers=[DuckDuckGoSearchProvider()])

# Custom providers
from langchain_tavily import TavilySearch

ext = WebSearchExtension(providers=[TavilySearch(max_results=5)])
```

### HITLExtension

Human-in-the-loop via a unified Question protocol. Two capabilities:

**Tool approval** — gate sensitive tools with human review:

```python
hitl = HITLExtension(interrupt_on={
    "send_email": True,           # approve / edit / reject
    "delete_file": {"options": ["approve", "reject"]},
})
# Tools not listed in interrupt_on execute normally without interruption.
```

**ask_user tool** — let the LLM ask structured questions:

```python
hitl = HITLExtension(tools=True)

# Or combine both:
hitl = HITLExtension(
    interrupt_on={"send_email": True},
    tools=True,
)
```

Both use the same interrupt payload (`Question` objects) and resume format.
Requires a checkpointer. Resume with `Command(resume={"answers": {"<question>": "<answer>"}})`.

### TeamExtension

Coordinate a team of concurrent agents for complex, multi-step work that requires back-and-forth communication. The lead spawns teammates, assigns tasks, reacts to their results, and can forward information between team members.

```python
from langchain_agentkit import agent, TeamExtension, TasksExtension

class lead(agent):
    model = ChatOpenAI(model="gpt-4o")
    extensions = [TasksExtension(), TeamExtension(agents=[researcher, coder])]
    prompt = "You are a project lead. Coordinate your team."
    async def handler(state, *, llm, tools, prompt): ...
```

**How it works:** Teammates run as `asyncio.Task`s with their own checkpointers (conversation history persists across messages). A **Router Node** in the graph checks for teammate messages after each tool execution — when a teammate sends a result, the lead is automatically re-invoked with the message.

**Tools:**

| Tool | Description |
|------|-------------|
| `AgentTeam(team_name, members)` | Create a team with named members |
| `SendMessage(member_name, message)` | Send work, guidance, or follow-ups to a member |
| `CheckTeammates()` | See statuses and collect pending messages |
| `DissolveTeam()` | Graceful shutdown |

**When to use Teams vs Agent:**

| | Agent | Team |
|---|---|---|
| Interaction | Single request → result | Multi-turn conversation |
| Lead during execution | Blocked waiting | Active (coordinating) |
| Communication | One-way | Bidirectional (messages) |
| Use case | "Do this and report back" | "Let's work on this together" |

See [`examples/team.py`](examples/team.py) for a complete example.

## Custom Extensions

Any class with `tools`, `prompt()`, and `state_schema` satisfies the protocol:

```python
from langchain_agentkit import Extension

class MyExtension(Extension):
    @property
    def tools(self):
        return [my_tool]

    def prompt(self, state, runtime=None):
        return "You have access to my_tool."

    @property
    def state_schema(self):
        return None  # or a TypedDict mixin
```

## Contributing

```bash
git clone https://github.com/rsmdt/langchain-agentkit.git
cd langchain-agentkit
uv sync --extra dev
uv run pytest tests/unit/ -q
uv run ruff check src/ tests/
uv run mypy src/

# LLM integration evals (requires OPENAI_API_KEY in .env)
uv sync --extra eval
uv run pytest tests/evals/ -m eval -v
```
