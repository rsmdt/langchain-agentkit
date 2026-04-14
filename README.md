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

### The `Agent` class

Declare a class, get a complete ReAct agent with extension-composed tools and prompts:

```python
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_agentkit import Agent, SkillsExtension, TasksExtension

class Researcher(Agent):
    model = ChatOpenAI(model="gpt-4o")
    extensions = [
        SkillsExtension(skills="skills/"),
        TasksExtension(),
    ]
    prompt = "You are a research assistant."

    async def handler(state, *, llm, tools, prompt):
        messages = [SystemMessage(content=prompt)] + state["messages"]
        return {"messages": [await llm.bind_tools(tools).ainvoke(messages)]}

app = Researcher().compile()
result = app.invoke({"messages": [HumanMessage("Size the B2B SaaS market")]})
```

The `model` attribute accepts a `BaseChatModel` instance (used as-is) or a string resolved via `model_resolver`:

```python
class FastAgent(Agent):
    model = "gpt-4o-mini"
    model_resolver = staticmethod(lambda name: ChatOpenAI(model=name))
    extensions = [AgentsExtension(agents=[Researcher, Coder])]
    ...
```

The state schema is composed automatically from extensions — `TasksExtension` adds a `tasks` key, `SkillsExtension` adds nothing. No need to define state manually.

### Dynamic properties

Each property (`model`, `prompt`, `extensions`, `tools`, `model_resolver`) can be a static attribute, a sync method, or an async method — `compile()` resolves them uniformly. Pass any per-request configuration via `__init__(**kwargs)`:

```python
from langchain_agentkit import Agent, FilesystemExtension

class Researcher(Agent):
    model = ChatOpenAI(model="gpt-4o")

    async def prompt(self):
        return await self.backend.read(".agentkit/AGENTS.md")

    def extensions(self):
        return [FilesystemExtension(backend=self.backend)]

    async def handler(state, *, llm, tools, prompt, runtime):
        messages = [SystemMessage(content=prompt)] + state["messages"]
        return {"messages": [await llm.bind_tools(tools).ainvoke(messages)]}

app = Researcher(backend=my_backend).compile()
```

Use `graph()` instead of `compile()` when you need the uncompiled `StateGraph` for composition — e.g. passing to `AgentsExtension(agents=[...])` or embedding as a subgraph.

### `AgentKit` for managed or manual graph wiring

`AgentKit` accepts extensions, user tools, model, and prompt — then `compile(handler)` builds a complete ReAct graph with hooks wired:

```python
from langchain_openai import ChatOpenAI
from langchain_agentkit import AgentKit, SkillsExtension, TasksExtension

kit = AgentKit(
    extensions=[SkillsExtension(skills="skills/"), TasksExtension()],
    tools=[web_search],
    model=ChatOpenAI(model="gpt-4o"),
    prompt="You are a research assistant.",
)

async def handler(state, *, llm, tools, prompt, runtime):
    from langchain_core.messages import SystemMessage
    messages = [SystemMessage(content=prompt)] + state["messages"]
    return {"messages": [await llm.bind_tools(tools).ainvoke(messages)]}

graph = kit.compile(handler)          # uncompiled StateGraph
app = graph.compile()                 # compiled, ready to invoke
```

For **full manual control** over graph topology (custom routing, multi-node graphs, shared `ToolNode`), access `kit.tools`, `kit.compose(state)`, `kit.model`, `kit.state_schema`, and `kit.hooks` directly:

```python
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from langchain_agentkit import AgentKit, SkillsExtension, TasksExtension

kit = AgentKit(extensions=[
    SkillsExtension(skills="skills/"),
    TasksExtension(),
])

llm = ChatOpenAI(model="gpt-4o")
all_tools = kit.tools
bound_llm = llm.bind_tools(all_tools)

def agent_node(state):
    prompt = kit.compose(state).joined
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
    HistoryExtension(strategy="count", max_messages=50),
    HITLExtension(interrupt_on={"send_email": True}, tools=True),
    AgentsExtension(agents=[researcher, coder]),
    TeamExtension(agents=[researcher, coder]),
]
```

### Recommended ordering

Declaration order in `extensions=[...]` determines prompt-section order in the composed system prompt. For best results, declare extensions in the following order:

```python
extensions = [
    # --- Prompt-contributing layer (order shapes system prompt) ---
    CoreBehaviorExtension(),                        # universal behavior
    MemoryExtension(),                              # persistent user/project memory
    EnvExtension(),                                 # auto-detected runtime env block
    FilesystemExtension(root="..."),                # workspace root + file tools
    SkillsExtension(skills="skills/"),              # progressive-disclosure skill catalog
    TasksExtension(),                               # task tracking
    WebSearchExtension(),                           # external research tool
    TeamExtension(agents=[...]),                    # peer-to-peer coordination
    AgentsExtension(agents=[...]),                  # delegate-to-specialist tool

    # --- Hook-only layer (order shapes wrap onion, not prompt) ---
    HITLExtension(interrupt_on={"send_email": True}, tools=True),
    HistoryExtension(strategy="count", max_messages=50),
    ContextCompactionExtension(keep_recent=5),
    MessagePersistenceExtension(persist=write_to_db),
]
```

The stack has two layers. The **prompt-contributing layer** produces the system prompt in declaration order — placement here is visible to the model. The **hook-only layer** contributes no prompt text; declaration order here only determines wrap-hook nesting and run-lifecycle call order.

Rationale for each slot:

| Slot | Extension | Why it sits here |
|------|-----------|------------------|
| 1 | `CoreBehaviorExtension` | Domain-neutral behavior — terseness, tool-use discipline, action safety, tool-result summarization. Applies to everything that follows. |
| 2 | `MemoryExtension` | User/project memory context should be available before any runtime environment or tool-specific guidance. |
| 3 | `EnvExtension` | Runtime env (cwd, git, platform) is read-mostly context; placed ahead of tool blocks so tool guidance can reference the environment. |
| 4 | `FilesystemExtension` | Workspace-root guidance anchors later filesystem-adjacent tools. |
| 5 | `SkillsExtension` | Progressive-disclosure skill catalog; read before any task or coordination layer so the agent knows what capabilities it has. |
| 6 | `TasksExtension` | Task tracking is a coordination primitive consumed by teams/agents — must appear before them. |
| 7 | `WebSearchExtension` | Leaf research tool; grouped with other first-party tool blocks, before coordination layers. |
| 8 | `TeamExtension` | Peer coordination; depends on `TasksExtension` (auto-prepended if omitted). |
| 9 | `AgentsExtension` | Delegation to specialists; placed last among tool blocks so the agent has full context of its own capabilities before deciding to hand off. |
| 10 | `HITLExtension` | `wrap_tool` interceptor — declared outermost in the hook layer so approval gating wraps every tool call before other tool-layer hooks run. |
| 11 | `HistoryExtension` | `wrap_model` message truncation; runs before compaction so truncation decides *which* messages survive, then compaction redacts within the survivors. |
| 12 | `ContextCompactionExtension` | `wrap_model` eviction of old tool results, composing inside the history window. |
| 13 | `MessagePersistenceExtension` | `before_run`/`after_run` — declared last so its snapshot captures state produced by all prior before_run hooks, and its after_run (which runs in reverse order) fires first to persist the full turn delta. |

Notes:

- **Dependencies are auto-resolved.** Missing dependencies are prepended automatically — e.g. omitting `TasksExtension` while using `TeamExtension` still produces a valid stack.
- **`preset="full"` seeds the first two slots.** Passing `preset="full"` to `AgentKit(...)` prepends `CoreBehaviorExtension` and `TasksExtension` ahead of any user-declared extensions.
- **Every extension listed is opt-in.** Include only what your agent needs. Hook-layer extensions (`HITL`, `History`, `ContextCompaction`, `MessagePersistence`) pay off in different scenarios: use `ContextCompaction` for sessions with many tool calls, `History` when raw message count grows faster than tool results, `HITL` when specific tools need approval, and `MessagePersistence` when turns must be mirrored to an external store.
- **Wrap-hook ordering is an onion.** First declaration = outermost layer. If you need compaction to run before truncation (redact first, then drop), swap slots 11 and 12.

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

### AgentsExtension

Delegate tasks to specialist subagents at runtime. Accepts compiled StateGraphs, `AgentConfig` definitions, or discovers agents from a directory of markdown files.

```python
from langchain_agentkit import Agent, AgentsExtension, AgentConfig

class Researcher(Agent):
    model = ChatOpenAI(model="gpt-4o-mini")
    description = "Research specialist for information gathering"
    tools = [web_search]
    prompt = "You are a research specialist."
    async def handler(state, *, llm, tools, prompt): ...

researcher = Researcher().compile()  # StateGraph

# Programmatic — mix compiled graphs and AgentConfig definitions
ext = AgentsExtension(agents=[
    researcher,                                          # compiled StateGraph
    AgentConfig(name="coder", description="Code expert", prompt="You code."),
])

# Directory discovery — scan for .md files with frontmatter
ext = AgentsExtension(agents="agents/")

# With a custom backend
ext = AgentsExtension(agents="/agents", backend=my_backend)
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

### HistoryExtension

Manage conversation history to keep the LLM context window lean. Truncated messages are removed from graph state via `ReplaceMessages` so the checkpointer stays compact.

```python
from langchain_agentkit import HistoryExtension

# Keep the last 50 messages
ext = HistoryExtension(strategy="count", max_messages=50)

# Keep messages within a token budget
ext = HistoryExtension(strategy="tokens", max_tokens=4000)

# Custom token counter
ext = HistoryExtension(strategy="tokens", max_tokens=4000, token_counter=my_fn)

# Custom strategy — any object with transform(messages) -> messages
ext = HistoryExtension(strategy=MySummarizationStrategy())
```

Both built-in strategies preserve a leading `SystemMessage` when truncating. Dropped messages are bulk-replaced in graph state using LangGraph's `REMOVE_ALL_MESSAGES` sentinel (wrapped in `ReplaceMessages` for convenience).

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
from langchain_agentkit import Agent, TeamExtension, TasksExtension

class Lead(Agent):
    model = ChatOpenAI(model="gpt-4o")
    extensions = [TasksExtension(), TeamExtension(agents=[researcher, coder])]
    prompt = "You are a project lead. Coordinate your team."
    async def handler(state, *, llm, tools, prompt): ...

graph = Lead().compile()
```

**How it works:** Teammates run as `asyncio.Task`s with their own checkpointers (conversation history persists across messages). A **Router Node** in the graph checks for teammate messages after each tool execution — when a teammate sends a result, the lead is automatically re-invoked with the message.

**Tools:**

| Tool | Description |
|------|-------------|
| `TeamCreate(name, agents)` | Create a team with named members |
| `TeamMessage(to, message)` | Send work, guidance, or follow-ups to a member |
| `TeamStatus()` | See statuses and collect pending messages |
| `TeamDissolve()` | Graceful shutdown |

**When to use Teams vs Agent:**

| | Agent | Team |
|---|---|---|
| Interaction | Single request → result | Multi-turn conversation |
| Lead during execution | Blocked waiting | Active (coordinating) |
| Communication | One-way | Bidirectional (messages) |
| Use case | "Do this and report back" | "Let's work on this together" |

See [`examples/team.py`](examples/team.py) for a complete example.

## Custom Extensions

Any subclass of `Extension` can contribute tools, a prompt section, state schema, lifecycle hooks, and graph modifications:

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

### Lifecycle hooks

Extensions intercept the run/model/tool lifecycle via three decorators:

| Decorator | `run` | `model` | `tool` |
|---|---|---|---|
| `@before(point)` | ✓ | ✓ | ✓ |
| `@after(point)` | ✓ | ✓ | ✓ |
| `@wrap(point)` | — | ✓ | ✓ |

`wrap_run` is intentionally not supported. `before_run` and `after_run` are implemented as langgraph nodes (not Runnable-boundary wrappers), so they participate in checkpointing: on resume from a mid-graph checkpoint, `before_run` does not re-fire and `after_run` only fires at real completion. This is the behavior snapshot-style hooks (e.g. `MessagePersistenceExtension`) depend on, and it falls out naturally from reducer-based state updates. An onion wrap cannot be expressed as a graph node — so rather than adopting divergent semantics that would re-fire on every resume, run-scoped wrapping is omitted. Use `before_run` + `after_run` for setup/teardown; use `wrap_model` or `wrap_tool` when try/finally-shaped control flow is required.

### Sibling-aware configuration via `setup()`

When an extension needs to react to other extensions in the kit (e.g. enabling a feature only when a particular sibling is present), override `setup()`:

```python
from langchain_agentkit import Extension
from langchain_agentkit.extensions.hitl import HITLExtension

class MyExtension(Extension):
    def __init__(self):
        self._hitl_enabled = False

    def setup(self, *, extensions, **_):
        # Inspect the assembled kit and configure self accordingly.
        self._hitl_enabled = any(isinstance(e, HITLExtension) for e in extensions)
```

`setup()` is called once after dependency resolution, before the graph is built. Each extension declares only the kwargs it needs — the framework uses signature introspection to pass only what's requested. Available kwargs:

| Kwarg | Type | Meaning |
|---|---|---|
| `extensions` | `list[Extension]` | All extensions in the kit, including `self` |
| `prompt` | `str` | The base prompt configured on `AgentKit` (empty if none) |
| `model_resolver` | `Callable` or `None` | The kit-level model resolver, if configured |

**Contract — inspect presence, not state**: `setup()` runs in declaration order, so another extension's `setup()` may not have run yet when yours executes. Only inspect sibling *presence* via `isinstance()` checks — never read mutable state that another extension's `setup()` might populate. Anything that depends on a sibling being fully configured should happen lazily at runtime.

### Declaring dependencies

If your extension requires another extension to function, declare it via `dependencies()` — AgentKit will auto-add it if missing:

```python
class MyExtension(Extension):
    def dependencies(self):
        return [TasksExtension()]  # auto-added if user didn't include one
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
