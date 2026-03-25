# langchain-agentkit

Composable middleware framework for LangGraph agents.

[![Python](https://img.shields.io/pypi/pyversions/langchain-agentkit.svg)](https://pypi.org/project/langchain-agentkit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Installation

```bash
pip install langchain-agentkit
```

Requires Python 3.11+.

## Quick Start

### The `agent` metaclass

Declare a class, get a complete ReAct agent with middleware-composed tools and prompts:

```python
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_agentkit import agent, SkillsMiddleware, TasksMiddleware

class researcher(agent):
    llm = ChatOpenAI(model="gpt-4o")
    middleware = [
        SkillsMiddleware(skills="skills/"),
        TasksMiddleware(),
    ]
    prompt = "You are a research assistant."

    async def handler(state, *, llm, prompt):
        messages = [SystemMessage(content=prompt)] + state["messages"]
        return {"messages": [await llm.ainvoke(messages)]}

graph = researcher.compile()
result = graph.invoke({"messages": [HumanMessage("Size the B2B SaaS market")]})
```

The state schema is composed automatically from middleware — `TasksMiddleware` adds a `tasks` key, `SkillsMiddleware` adds nothing. No need to define state manually.

### `AgentKit` for manual graph wiring

Use `AgentKit` when you need full control over graph topology:

```python
from langchain_agentkit import AgentKit, SkillsMiddleware, TasksMiddleware

kit = AgentKit([
    SkillsMiddleware(skills="skills/"),
    TasksMiddleware(),
])

all_tools = my_tools + kit.tools
system_prompt = kit.prompt(state, runtime)
state_schema = kit.state_schema  # composed from middleware
```

## Middleware

Each middleware provides tools, a prompt section, and optional state requirements. Compose them in any combination:

```python
middleware = [
    SkillsMiddleware(skills="skills/"),
    TasksMiddleware(),
    FilesystemMiddleware(),
    WebSearchMiddleware(),
    HITLMiddleware(interrupt_on={"send_email": True}),
]
```

### SkillsMiddleware

Loads skills from directories containing `SKILL.md` files. Provides progressive disclosure — the agent sees skill names and descriptions, then loads full instructions on demand.

```python
# Convenience: includes filesystem tools for reading skill reference files
mw = SkillsMiddleware(skills="skills/")
mw.tools  # [Skill, Read, Write, Edit, Glob, Grep]

# Explicit: provide a shared VFS, manage filesystem tools separately
from langchain_agentkit import VirtualFilesystem, FilesystemMiddleware

vfs = VirtualFilesystem()
skills_mw = SkillsMiddleware(skills="skills/", filesystem=vfs)
fs_mw = FilesystemMiddleware(filesystem=vfs)
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

### TasksMiddleware

Task management for complex multi-step objectives. The agent creates, tracks, and completes tasks with dependency ordering.

```python
mw = TasksMiddleware()
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

### FilesystemMiddleware

Claude Code-aligned file tools operating on an in-memory virtual filesystem:

```python
from langchain_agentkit import FilesystemMiddleware, VirtualFilesystem

vfs = VirtualFilesystem()
vfs.write("/data/config.json", '{"key": "value"}')

mw = FilesystemMiddleware(filesystem=vfs)
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

### WebSearchMiddleware

Multi-provider web search. Fans out queries to all providers in parallel. Works out of the box with built-in Qwant search (no API key needed):

```python
# Zero config
mw = WebSearchMiddleware()

# Custom providers
from langchain_tavily import TavilySearch

mw = WebSearchMiddleware(providers=[TavilySearch(max_results=5)])
```

### HITLMiddleware

Human-in-the-loop approval for sensitive tool calls via LangGraph `interrupt()`:

```python
mw = HITLMiddleware(interrupt_on={
    "send_email": True,           # requires approval
    "search": False,              # auto-approved
    "delete_file": {"allowed_decisions": ["approve", "reject"]},
})
```

Requires a checkpointer. Resume with `Command(resume={"type": "approve"})`.

## Custom Middleware

Any class with `tools`, `prompt()`, and `state_schema` satisfies the protocol:

```python
from langchain_agentkit import Middleware

class MyMiddleware:
    @property
    def tools(self):
        return [my_tool]

    def prompt(self, state, runtime=None):
        return "You have access to my_tool."

    @property
    def state_schema(self):
        return None  # or a TypedDict mixin if you need state keys
```

If your middleware needs a custom state key, return a TypedDict mixin from `state_schema`:

```python
from typing import TypedDict

class MyState(TypedDict, total=False):
    my_data: list[str]

class MyMiddleware:
    @property
    def state_schema(self):
        return MyState

    # ... tools and prompt
```

The state key will be automatically included when the middleware is composed via `AgentKit`.

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
