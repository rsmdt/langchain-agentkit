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
    llm = ChatOpenAI(model="gpt-4o")
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

## Extension

Each extension provides tools, a prompt section, and optional state requirements. Compose them in any combination:

```python
extensions = [
    SkillsExtension(skills="skills/"),
    TasksExtension(),
    FilesystemExtension(),
    WebSearchExtension(),
    HITLExtension(interrupt_on={"send_email": True}),
    AgentExtension([researcher, coder]),
    TeamExtension([researcher, coder]),
]
```

### SkillsExtension

Loads skills from directories containing `SKILL.md` files. Provides progressive disclosure — the agent sees skill names and descriptions, then loads full instructions on demand.

```python
# Convenience: includes filesystem tools for reading skill reference files
mw = SkillsExtension(skills="skills/")
mw.tools  # [Skill, Read, Write, Edit, Glob, Grep]

# Explicit: provide a shared VFS, manage filesystem tools separately
from langchain_agentkit import VirtualFilesystem, FilesystemExtension

vfs = VirtualFilesystem()
skills_mw = SkillsExtension(skills="skills/", filesystem=vfs)
fs_mw = FilesystemExtension(filesystem=vfs)
skills_mw.tools  # [Skill]
fs_mw.tools      # [Read, Write, Edit, Glob, Grep]
```

**Tools:**

| Tool | Description |
|------|-------------|
| `Skill(skill_name)` | Load a skill's instructions |

Skill directories follow the [AgentSkills.io](https://agentskills.io/specification) format:

```
skills/
└── market-sizing/
    ├── SKILL.md          # YAML frontmatter (name, description) + instructions
    └── calculator.py     # Reference files accessible via Read tool
```

### TasksExtension

Task management for complex multi-step objectives. The agent creates, tracks, and completes tasks with dependency ordering.

```python
mw = TasksExtension()
mw.tools  # [TaskCreate, TaskUpdate, TaskList, TaskGet, TaskStop]
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

Claude Code-aligned file tools operating on an in-memory virtual filesystem:

```python
from langchain_agentkit import FilesystemExtension, VirtualFilesystem

vfs = VirtualFilesystem()
vfs.write("/data/config.json", '{"key": "value"}')

mw = FilesystemExtension(filesystem=vfs)
mw.tools  # [Read, Write, Edit, Glob, Grep]
```

**Tools:**

| Tool | Description |
|------|-------------|
| `Read(file_path)` | Read file with line numbers, offset/limit pagination |
| `Write(file_path, content)` | Create or overwrite a file |
| `Edit(file_path, old_string, new_string)` | Exact string replacement |
| `Glob(pattern)` | Find files by pattern (supports `*`, `**`, `?`) |
| `Grep(pattern)` | Search file contents by regex |

### WebSearchExtension

Multi-provider web search. Fans out queries to all providers in parallel. Works out of the box with built-in Qwant search (no API key needed):

```python
# Zero config
mw = WebSearchExtension()

# Custom providers
from langchain_tavily import TavilySearch

mw = WebSearchExtension(providers=[TavilySearch(max_results=5)])
```

### HITLExtension

Human-in-the-loop approval for sensitive tool calls via LangGraph `interrupt()`:

```python
mw = HITLExtension(interrupt_on={
    "send_email": True,           # requires approval
    "search": False,              # auto-approved
    "delete_file": {"allowed_decisions": ["approve", "reject"]},
})
```

Requires a checkpointer. Resume with `Command(resume={"type": "approve"})`.

### AgentExtension

Delegate tasks to specialist subagents at runtime. The lead agent decides when and to whom to delegate via the `Agent` tool. Subagents run in isolation — they receive only the task message (not the lead's full conversation history) and return a concise result.

```python
from langchain_agentkit import agent, AgentExtension

class researcher(agent):
    llm = ChatOpenAI(model="gpt-4o-mini")
    description = "Research specialist for information gathering"
    tools = [web_search]
    prompt = "You are a research specialist."
    async def handler(state, *, llm, tools, prompt): ...

class coder(agent):
    llm = ChatOpenAI(model="gpt-4o")
    description = "Code implementation and debugging"
    tools = [file_read, file_write]
    prompt = "You are a coding specialist."
    async def handler(state, *, llm, tools, prompt): ...

class lead(agent):
    llm = ChatOpenAI(model="gpt-4o")
    extensions = [AgentExtension([researcher, coder], ephemeral=True)]
    prompt = "Delegate research to the researcher and coding to the coder."
    async def handler(state, *, llm, tools, prompt): ...
```

**How it works:** `Agent` is a blocking tool call. The lead's ReAct loop stays alive because it's waiting for the tool result. For parallelism, the LLM calls multiple `Agent` tools in one turn — LangGraph's `ToolNode` executes them concurrently.

**The `Agent` tool uses shape-based discrimination** — the LLM provides either `{id: "<name>"}` for a pre-defined agent or `{prompt: "..."}` for a dynamic one:

```json
// Pre-defined agent from the roster
{"agent": {"id": "researcher"}, "message": "Find info on X"}

// Dynamic agent (ephemeral=True required)
{"agent": {"prompt": "You are a legal expert..."}, "message": "Analyze this contract"}
```

**Key features:**
- `description` attribute on agents — used in the prompt roster so the LLM knows what each specialist does
- `tools="inherit"` — subagent receives the parent's tools at delegation time instead of its own
- `ephemeral=True` — enables dynamic (on-the-fly) reasoning agents in the `Agent` tool schema
- `delegation_timeout` — max seconds per delegation (default 300s)

See [`examples/delegation.py`](examples/delegation.py) for a complete example.

### TeamExtension

Coordinate a team of concurrent agents for complex, multi-step work that requires back-and-forth communication. The lead spawns teammates, assigns tasks, reacts to their results, and can forward information between team members.

```python
from langchain_agentkit import agent, TeamExtension, TasksExtension

class lead(agent):
    llm = ChatOpenAI(model="gpt-4o")
    extensions = [TasksExtension(), TeamExtension([researcher, coder])]
    prompt = "You are a project lead. Coordinate your team."
    async def handler(state, *, llm, tools, prompt): ...
```

There is **one shared task list** — the same one `TasksExtension` provides. `AssignTask` writes to it, `TaskList` reads from it. No separate task system for teams. `TeamExtension` always includes `TasksState` in the state schema so `AssignTask` works even without explicit `TasksExtension`:

```python
# Minimal — AssignTask creates tasks, but lead has no TaskList/TaskCreate tools
extensions = [TeamExtension([researcher, coder])]

# Full — lead also has task management tools (recommended)
extensions = [TasksExtension(), TeamExtension([researcher, coder])]

# The LLM drives the lifecycle:
# 1. AgentTeam("dev-team", [{"name": "alice", "agent_type": "researcher"}, ...])
# 2. AssignTask("alice", "Research rate limiting best practices")
# 3. [alice works, sends result back via message bus]
# 4. Lead receives alice's result automatically (Router Node)
# 5. MessageTeammate("bob", "Alice found: use token bucket")
# 6. DissolveTeam() → synthesize → respond to user
```

**How it works:** Teammates run as `asyncio.Task`s with their own checkpointers (conversation history persists across messages). A **Router Node** in the graph checks for teammate messages after each tool execution — when a teammate sends a result, the lead is automatically re-invoked with the message. The lead reacts to messages, it doesn't poll.

**Tools:**

| Tool | Description |
|------|-------------|
| `AgentTeam(team_name, members)` | Create a team. Each member gets an asyncio.Task + checkpointer. |
| `AssignTask(member_name, task_description)` | Assign work — creates a tracked task and sends it to the member. |
| `MessageTeammate(member_name, message)` | Send guidance, follow-ups, or information from other members. |
| `CheckTeammates()` | See member statuses, collect pending messages, view task progress. |
| `DissolveTeam()` | Graceful shutdown — sends shutdown signals, waits, cleans up. |

**Key features:**
- **Router Node** — automatic message delivery from teammates to the lead (no polling)
- **Conversation history** — each teammate has its own `InMemorySaver` checkpointer, so multiple messages accumulate context
- **Shared task list** — one task list, shared with `TasksExtension`. `AssignTask` writes to it, `TaskList` reads from it. Add `TasksExtension` for full task management tools.
- `max_team_size` — limit concurrent members (default 5)
- `router_timeout` — how long the Router waits for messages (default 30s)
- `max_iterations` — safety limit on Router re-invocations (default 50)

**When to use Teams vs Agent:**

| | Agent | Team |
|---|---|---|
| Interaction | Single request → result | Multi-turn conversation |
| Lead during execution | Blocked waiting | Active (coordinating) |
| Communication | One-way | Bidirectional (messages) |
| Use case | "Do this and report back" | "Let's work on this together" |

See [`examples/team.py`](examples/team.py) for a complete example.

## Custom Extension

Any class with `tools`, `prompt()`, and `state_schema` satisfies the protocol:

```python
from langchain_agentkit import Extension

class MyExtension:
    @property
    def tools(self):
        return [my_tool]

    def prompt(self, state, runtime=None):
        return "You have access to my_tool."

    @property
    def state_schema(self):
        return None  # or a TypedDict mixin if you need state keys
```

If extension needs a custom state key, return a TypedDict mixin from `state_schema`:

```python
from typing import TypedDict

class MyState(TypedDict, total=False):
    my_data: list[str]

class MyExtension:
    @property
    def state_schema(self):
        return MyState

    # ... tools and prompt
```

The state key will be automatically included when the extension is composed via `AgentKit`.

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
