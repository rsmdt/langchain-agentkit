# langchain-agentkit

Composable middleware framework for LangGraph agents.

[![Python](https://img.shields.io/pypi/pyversions/langchain-agentkit.svg)](https://pypi.org/project/langchain-agentkit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Build LangGraph agents with reusable middleware that composes tools and prompts. Two layers to choose from:

- **`agent` metaclass** — declare a class, get a complete ReAct agent with middleware-composed tools and prompts
- **`AgentKit`** — primitive composition engine for full control over graph topology

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
from langchain_agentkit import agent, SkillsMiddleware, TasksMiddleware, WebSearchMiddleware

class researcher(agent):
    llm = ChatOpenAI(model="gpt-4o")
    middleware = [
        SkillsMiddleware("skills/"),
        TasksMiddleware(),
        WebSearchMiddleware(),  # Built-in Qwant search, no API key needed
    ]
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

### Standalone `SkillRegistry`

Use `SkillRegistry` directly for skill discovery without the middleware layer:

```python
from langchain_agentkit import SkillRegistry

registry = SkillRegistry("skills/")
tools = registry.tools  # [Skill, SkillRead]
```

## Examples

See [`examples/`](examples/) for complete working code:

- **[`standalone_node.py`](examples/standalone_node.py)** — Simplest usage: declare a node class, compile, invoke
- **[`manual_wiring.py`](examples/manual_wiring.py)** — Use `SkillRegistry` as a standalone toolkit with full graph control
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
| `WebSearchMiddleware(providers?)` | `web_search` | Search guidance with provider names |

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

### `WebSearchMiddleware`

Multi-provider web search middleware. Fans out queries to all configured search providers in parallel via `asyncio.gather`, returning results attributed per provider.

Works out of the box with zero configuration — uses built-in Qwant search (no API key required). Add your own providers for more comprehensive results:

```python
# Zero config — uses built-in Qwant search
mw = WebSearchMiddleware()

# Custom providers — any BaseTool or callable
from langchain_tavily import TavilySearch

mw = WebSearchMiddleware(providers=[
    TavilySearch(max_results=5),
])

# Mix built-in with custom
from langchain_community.tools import DuckDuckGoSearchRun

mw = WebSearchMiddleware(providers=[
    DuckDuckGoSearchRun(),
    my_custom_search_function,  # auto-wrapped into BaseTool
])

# Custom prompt template
mw = WebSearchMiddleware(prompt_template="Use {provider_names} to search.")
```

Providers can be any LangChain `BaseTool` or a callable with signature `(query: str) -> str`. Callables are auto-wrapped into tools. Provider errors are captured per-provider — one failing provider doesn't break the search.

### `QwantSearchTool`

Built-in web search tool using [Qwant's](https://www.qwant.com/) search API. No API key required. Works as a standalone LangChain tool or as the default provider for `WebSearchMiddleware`:

```python
from langchain_agentkit import QwantSearchTool

# Standalone usage — like any other LangChain tool
tool = QwantSearchTool()
result = tool.invoke("latest AI news")

# Configurable
tool = QwantSearchTool(max_results=3, locale="fr_FR", safesearch=2)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_results` | `5` | Number of results to return (max 10) |
| `locale` | `"en_US"` | Search locale |
| `safesearch` | `1` | Safe search level: 0=off, 1=moderate, 2=strict |

### `AgentKit(middleware, prompt=None)`

Composition engine that merges tools and prompts from middleware.

- **`tools`** — All tools from all middleware, deduplicated by name (first middleware wins, cached)
- **`prompt(state, runtime)`** — Template + middleware sections, joined with double newline (dynamic per call)

Prompt templates can be inline strings, file paths, or a list of either:

```python
kit = AgentKit(middleware, prompt="You are helpful.")
kit = AgentKit(middleware, prompt=Path("prompts/system.txt"))
kit = AgentKit(middleware, prompt=["prompts/base.txt", "Extra instructions"])
```

### `node`

Skill-aware metaclass. Uses `skills` attribute instead of `middleware`. Consider using `agent` for the full middleware composition model.

```python
class my_agent(node):
    llm = ChatOpenAI(model="gpt-4o")
    tools = [web_search]
    skills = "skills/"  # str, list[str], or SkillRegistry instance

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

## Patterns

### Reasoning: ReAct and Chain of Thought

The `agent` metaclass builds a [ReAct](https://arxiv.org/abs/2210.03629) loop — the LLM reasons, calls tools, observes results, and reasons again. This is Chain of Thought with tool use built in:

```
handler → LLM reasons → tool calls? → ToolNode executes → handler → LLM reasons → ... → END
```

All middleware tools (search, tasks, skills) live in a **single shared ToolNode**. No special routing or if/else logic — `ToolNode` dispatches by tool name.

#### Prompt-based Chain of Thought

Add reasoning instructions via the `prompt` attribute — no code changes needed:

```python
class analyst(agent):
    llm = ChatOpenAI(model="gpt-4o")
    middleware = [WebSearchMiddleware()]
    prompt = """You are a research analyst. Think step by step:
1. Identify what information you need
2. Search for evidence using web_search
3. Synthesize findings into a clear answer"""

    async def handler(state, *, llm, prompt):
        messages = [SystemMessage(content=prompt)] + state["messages"]
        return {"messages": [await llm.ainvoke(messages)]}
```

#### Multi-node reasoning pipeline

For explicit Reason → Act → Synthesize stages, use `AgentKit` with manual graph wiring:

```python
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from langchain_agentkit import AgentKit, WebSearchMiddleware

kit = AgentKit([WebSearchMiddleware()])

async def reason(state, config, **kw):
    """Analyze the question and plan what to search for."""
    ...

async def act(state, config, **kw):
    """Execute searches based on the reasoning step."""
    ...

async def synthesize(state, config, **kw):
    """Combine findings into a final answer."""
    ...

workflow = StateGraph(MyState)
workflow.add_node("reason", reason)
workflow.add_node("act", act)
workflow.add_node("tools", ToolNode(kit.tools))
workflow.add_node("synthesize", synthesize)

workflow.set_entry_point("reason")
workflow.add_edge("reason", "act")
workflow.add_conditional_edges("act", should_continue, {"tools": "tools", "synthesize": "synthesize"})
workflow.add_edge("tools", "act")
workflow.add_edge("synthesize", END)
```

#### Choosing a pattern

| Pattern | When to use | How |
|---------|------------|-----|
| **ReAct** (default) | Most agents — LLM decides when to use tools | `agent` metaclass, automatic |
| **Prompt CoT** | Step-by-step reasoning without changing architecture | Add instructions to `prompt` |
| **Multi-node pipeline** | Explicit reasoning stages, different LLMs per stage | `AgentKit` + manual `StateGraph` |

## Security

- **Path traversal prevention**: Skill file paths resolved to absolute and checked against skill directories. Reference file names reject `.` and `..` patterns.
- **Name validation**: Skill names validated per [AgentSkills.io spec](https://agentskills.io/specification) — lowercase alphanumeric + hyphens, 1-64 chars.
- **Tool scoping**: Each agent only has access to the tools declared in its `tools` attribute plus middleware-provided tools.
- **Prompt trust boundary**: Prompt templates and middleware prompt sections are set by the developer at construction time, not by end-user input.

## Contributing

```bash
git clone https://github.com/rsmdt/langchain-agentkit.git
cd langchain-agentkit
uv sync --extra dev
uv run pytest --tb=short -q
uv run ruff check src/ tests/
uv run mypy src/
```
