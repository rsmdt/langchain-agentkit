# langchain-agentkit

Composable middleware framework for LangGraph agents.

[![Python](https://img.shields.io/pypi/pyversions/langchain-agentkit.svg)](https://pypi.org/project/langchain-agentkit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Build LangGraph agents with reusable middleware that composes tools and prompts. Two layers to choose from:

- **`agent` metaclass** — declare a class, get a complete ReAct agent with middleware-composed tools and prompts
- **`AgentKit`** — primitive composition engine for full control over graph topology

> **Migrating from `langchain-skillkit`?** See [Migration](#migrating-from-langchain-skillkit) below. The old import path still works with a deprecation warning.

## Installation

Requires **Python 3.11+**, `langchain-core>=0.3`, `langgraph>=0.4`.

```bash
pip install langchain-agentkit
```

## Quick Start

### The `agent` metaclass (recommended)

Declare a class that inherits from `agent` to get a `StateGraph` with an automatic ReAct loop:

```python
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_agentkit import agent, SkillsMiddleware, TasksMiddleware

class researcher(agent):
    llm = ChatOpenAI(model="gpt-4o")
    tools = [web_search]
    middleware = [SkillsMiddleware("skills/"), TasksMiddleware()]
    prompt = "You are a research assistant."

    async def handler(state, *, llm, prompt):
        messages = [SystemMessage(content=prompt)] + state["messages"]
        response = await llm.ainvoke(messages)
        return {"messages": [response], "sender": "researcher"}

# Compile and use
graph = researcher.compile()
result = graph.invoke({"messages": [HumanMessage("Size the B2B SaaS market")]})

# With checkpointer for interrupt() support
from langgraph.checkpoint.memory import InMemorySaver
graph = researcher.compile(checkpointer=InMemorySaver())
```

### `AgentKit` for manual graph wiring

Use `AgentKit` when you need full control over graph topology — multi-node graphs, shared `ToolNode`, custom routing:

```python
from langchain_agentkit import AgentKit, SkillsMiddleware, TasksMiddleware

kit = AgentKit([
    SkillsMiddleware("skills/"),
    TasksMiddleware(),
])

# In any graph node:
all_tools = my_tools + kit.tools
system_prompt = kit.prompt(state, runtime)
```

### Standalone `SkillKit`

Use `SkillKit` directly for skill discovery without the middleware layer:

```python
from langchain_agentkit import SkillKit

kit = SkillKit("skills/")
tools = kit.tools  # [Skill, SkillRead]
```

## Examples

See [`examples/`](examples/) for complete working code:

- **[`standalone_node.py`](examples/standalone_node.py)** — Simplest usage: declare a node class, compile, invoke
- **[`manual_wiring.py`](examples/manual_wiring.py)** — Use `SkillKit` as a standalone toolkit with full graph control
- **[`multi_agent.py`](examples/multi_agent.py)** — Compose multiple agents in a parent graph
- **[`root_with_checkpointer.py`](examples/root_with_checkpointer.py)** — Multi-turn conversations with `interrupt()` and `Command(resume=...)`
- **[`subgraph_with_checkpointer.py`](examples/subgraph_with_checkpointer.py)** — Subgraph inherits parent's checkpointer automatically
- **[`custom_state_type.py`](examples/custom_state_type.py)** — Custom state shape via handler annotation + subgraph schema translation

## API Reference

### `agent`

Declarative agent builder. Subclassing produces a `StateGraph`. Call `.compile()` to get a runnable graph.

**Class attributes:**

| Attribute | Required | Description |
|-----------|----------|-------------|
| `llm` | Yes | Language model instance |
| `tools` | No | List of LangChain tools |
| `middleware` | No | Ordered list of `Middleware` instances |
| `prompt` | No | System prompt — inline string, file path, or list of either |

**Handler signature:**

```python
async def handler(state, *, llm, tools, prompt, runtime): ...
```

`state` is positional. Everything after `*` is keyword-only and injected by name — declare only what you need:

| Parameter | Type | Description |
|-----------|------|-------------|
| `state` | `dict` | LangGraph state (positional, required) |
| `llm` | `BaseChatModel` | LLM pre-bound with all tools via `bind_tools()` |
| `tools` | `list[BaseTool]` | All tools (user tools + middleware tools) |
| `prompt` | `str` | Fully composed prompt (template + middleware sections) |
| `runtime` | `ToolRuntime` | Unified runtime context. Use `runtime.config` for the full `RunnableConfig` |

Both sync and async handlers are supported — sync handlers are detected via `inspect.isawaitable` and awaited automatically.

**Custom state types** — annotate the handler's `state` parameter:

```python
class MyState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    draft: dict | None

class my_agent(agent):
    llm = ChatOpenAI(model="gpt-4o")

    async def handler(state: MyState, *, llm):
        ...
```

Without an annotation, `AgentState` is used by default.

### `Middleware` protocol

Any class with `tools` (property) and `prompt(state, runtime)` (method) satisfies the protocol via structural subtyping — no base class needed:

```python
class MyMiddleware:
    @property
    def tools(self) -> list[BaseTool]:
        return [my_tool]

    def prompt(self, state: dict, runtime: ToolRuntime) -> str | None:
        return "You have access to my_tool."
```

**Built-in middleware:**

| Middleware | Tools | Prompt |
|-----------|-------|--------|
| `SkillsMiddleware(skills_dirs)` | `Skill`, `SkillRead` | Progressive disclosure skill list with load instructions |
| `TasksMiddleware()` | `TaskCreate`, `TaskUpdate`, `TaskList`, `TaskGet` | Base agent behavior + task context with status icons |

### `TasksMiddleware`

Task management middleware with `Command`-based tools that update graph state via LangGraph's `ToolNode`.

```python
# Default — auto-creates task tools
mw = TasksMiddleware()

# Custom tools
mw = TasksMiddleware(task_tools=[my_create, my_update])

# Custom task formatter
mw = TasksMiddleware(formatter=my_format_function)
```

Task tools use `InjectedState` to read tasks from state and return `Command(update={"tasks": [...]})` to apply changes. Task state is updated locally within the agent's graph, visible to the prompt on every ReAct loop iteration. When used as a subgraph, state flows back to the parent graph on completion.

You can also create task tools directly:

```python
from langchain_agentkit import create_task_tools

tools = create_task_tools()  # [TaskCreate, TaskUpdate, TaskList, TaskGet]
```

### `AgentKit(middleware, prompt=None)`

Composition engine that merges tools and prompts from middleware.

- **`tools`** — All tools from all middleware, deduplicated by name (first middleware wins, cached)
- **`prompt(state, config)`** — Template + middleware sections, joined with double newline (dynamic per call)

Prompt templates can be inline strings, file paths, or a list of either:

```python
kit = AgentKit(middleware, prompt="You are helpful.")
kit = AgentKit(middleware, prompt=Path("prompts/system.txt"))
kit = AgentKit(middleware, prompt=["prompts/base.txt", "Extra instructions"])
```

### `node`

The original skill-aware metaclass. Uses `skills` attribute instead of `middleware`. Consider migrating to `agent` for the full middleware composition model — `agent` adds `middleware`, `prompt`, and `config` injection.

```python
class my_agent(node):
    llm = ChatOpenAI(model="gpt-4o")
    tools = [web_search]
    skills = "skills/"  # str, list[str], or SkillKit instance

    async def handler(state, *, llm, tools, runtime): ...
```

### `AgentState`

Minimal LangGraph state type with task support:

| Field | Type | Description |
|-------|------|-------------|
| `messages` | `Annotated[list, add_messages]` | Conversation history with LangGraph message reducer |
| `sender` | `str` | Name of the last node that produced output |
| `tasks` | `list[dict[str, Any]]` | Task list managed by `TasksMiddleware` tools |

Extend with your own fields:

```python
class MyState(AgentState):
    current_project: str
    iteration_count: int
```

## Security

- **Path traversal prevention**: Skill file paths resolved to absolute and checked against skill directories. Reference file names reject `.` and `..` patterns.
- **Name validation**: Skill names validated per [AgentSkills.io spec](https://agentskills.io/specification) — lowercase alphanumeric + hyphens, 1-64 chars.
- **Tool scoping**: Each agent only has access to the tools declared in its `tools` attribute plus middleware-provided tools.
- **Prompt trust boundary**: Prompt templates and middleware prompt sections are set by the developer at construction time, not by end-user input.

## Migrating from `langchain-skillkit`

`langchain-agentkit` v0.4.0 is the successor to `langchain-skillkit`. The old import path works with a deprecation warning:

```python
# Still works — emits DeprecationWarning
from langchain_skillkit import node, SkillKit, AgentState

# Update to:
from langchain_agentkit import node, SkillKit, AgentState
```

**What changed:**

| Before (`langchain-skillkit`) | After (`langchain-agentkit`) |
|------------------------------|------------------------------|
| `from langchain_skillkit import ...` | `from langchain_agentkit import ...` |
| `node` with `skills` attribute | `node` (unchanged) + new `agent` with `middleware` |
| `SkillKit` only | `SkillKit` + `SkillsMiddleware` + `TasksMiddleware` |
| No middleware system | `Middleware` protocol + `AgentKit` composition |
| Handler injectables: `llm`, `tools`, `runtime` | `agent` adds: `prompt`. `runtime` is now `ToolRuntime` |

**Migration steps:**

1. Update imports from `langchain_skillkit` to `langchain_agentkit`
2. Existing `node` subclasses work unchanged
3. Optionally migrate `node` to `agent` for middleware support:
   - Replace `skills = "skills/"` with `middleware = [SkillsMiddleware("skills/")]`
   - Add `prompt` and `config` injectables as needed

## Contributing

```bash
git clone https://github.com/rsmdt/langchain-agentkit.git
cd langchain-agentkit
uv sync --extra dev
uv run pytest --tb=short -q
uv run ruff check src/ tests/
uv run mypy src/
```
