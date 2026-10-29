<p align="center">
  <img src="agentkit.png" alt="langchain-agentkit — composable extension framework for LangGraph agents" width="640">
</p>

# langchain-agentkit

Composable extension framework for LangGraph agents.

[![PyPI](https://img.shields.io/pypi/v/langchain-agentkit)](https://pypi.org/project/langchain-agentkit/)
[![Python 3.12+](https://img.shields.io/pypi/pyversions/langchain-agentkit)](https://pypi.org/project/langchain-agentkit/)
[![CI](https://github.com/rsmdt/langchain-agentkit/actions/workflows/ci.yml/badge.svg)](https://github.com/rsmdt/langchain-agentkit/actions/workflows/ci.yml)
[![Downloads](https://img.shields.io/pypi/dm/langchain-agentkit)](https://pypi.org/project/langchain-agentkit/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
  - [The `Agent` class](#the-agent-class)
  - [Dynamic properties](#dynamic-properties)
  - [`AgentKit` for managed or manual graph wiring](#agentkit-for-managed-or-manual-graph-wiring)
- [Extensions](#extensions)
  - [Recommended ordering](#recommended-ordering)
  - [SkillsExtension](#skillsextension)
  - [AgentsExtension](#agentsextension)
  - [TasksExtension](#tasksextension)
  - [FilesystemExtension](#filesystemextension)
  - [WebExtension](#webextension)
  - [HistoryExtension](#historyextension)
  - [MemoryExtension](#memoryextension)
  - [HITLExtension](#hitlextension)
  - [TeamExtension](#teamextension)
  - [RubricExtension](#rubricextension)
- [FilesystemExtension Backends](#filesystemextension-backends)
- [Custom Extensions](#custom-extensions)
  - [Lifecycle hooks](#lifecycle-hooks)
  - [Sibling-aware configuration via `setup()`](#sibling-aware-configuration-via-setup)
  - [Declaring dependencies](#declaring-dependencies)
- [Contributing](#contributing)

## Installation

```bash
pip install langchain-agentkit
```

Requires Python 3.12+.

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

A compiled agent also drops straight into an external supervisor/router graph as a node: it shares the parent's `messages` channel, and a tool you pass it can return `Command(goto=..., graph=Command.PARENT)` to hand control back to the supervisor — no framework change required. See [`examples/supervisor.py`](examples/supervisor.py) for a complete example.

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
    WebExtension(),
    HistoryExtension(strategy=CountStrategy(max_messages=50)),
    HITLExtension(interrupt_on={"send_email": True}),
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
    WebExtension(providers=[...]),                  # external research tools
    TeamExtension(agents=[...]),                    # peer-to-peer coordination
    AgentsExtension(agents=[...]),                  # delegate-to-specialist tool

    # --- Hook-only layer (order shapes wrap onion, not prompt) ---
    ResilienceExtension(),                          # outermost wrap_tool — catches unhandled tool errors
    HITLExtension(interrupt_on={"send_email": True}),
    HistoryExtension(strategy=CompactionStrategy()),
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
| 7 | `WebExtension` | Leaf research tool; grouped with other first-party tool blocks, before coordination layers. |
| 8 | `TeamExtension` | Peer coordination; depends on `TasksExtension` (auto-prepended if omitted). |
| 9 | `AgentsExtension` | Delegation to specialists; placed last among tool blocks so the agent has full context of its own capabilities before deciding to hand off. |
| 10 | `ResilienceExtension` | `wrap_tool` safety net — declared outermost so it wraps every other tool-layer hook (including HITL). Converts any unhandled tool exception into a synthetic error `ToolMessage`, preserving the `AIMessage`↔`ToolMessage` pairing invariant and keeping the ReAct loop alive. |
| 11 | `HITLExtension` | `wrap_tool` approval gate — sits inside resilience so approvals apply to real tool invocations, but a raising approval flow still can't orphan a tool call. |
| 12 | `HistoryExtension` | `wrap_model` history rewrite (truncation, compaction or handoff); also retries once on a context-overflow error with a smaller history. |
| 13 | `MessagePersistenceExtension` | `before_run`/`after_run` — declared last so its snapshot captures state produced by all prior before_run hooks, and its after_run (which runs in reverse order) fires first to persist the full turn delta. |

Notes:

- **Dependencies are auto-resolved.** Missing dependencies are prepended automatically — e.g. omitting `TasksExtension` while using `TeamExtension` still produces a valid stack.
- **`preset="full"` seeds the first two slots.** Passing `preset="full"` to `AgentKit(...)` prepends `CoreBehaviorExtension` and `TasksExtension` ahead of any user-declared extensions. Passing your own instance of a seeded type replaces the seeded one in its slot.
- **Every extension listed is opt-in.** Include only what your agent needs. Hook-layer extensions (`HITL`, `History`, `MessagePersistence`) pay off in different scenarios: use `History` for long sessions, `HITL` when specific tools need approval, and `MessagePersistence` when turns must be mirrored to an external store.
- **Wrap-hook ordering is an onion.** First declaration = outermost layer. `ResilienceExtension` must stay ahead of any other `wrap_tool` extension so it can catch exceptions raised inside them. It also repairs tool calls left without a result, including calls whose arguments were malformed or truncated.

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

# With a custom backend (e.g. Daytona sandbox or AgentFS database)
ext = SkillsExtension(skills="/skills", backend=my_backend)
```

Always provides exactly one tool: `Skill`. Filesystem tools (Read, Write, etc.) come from `FilesystemExtension`.

**Tools:**

| Tool | Description |
|------|-------------|
| `Skill(skill_name)` | Load a skill's prompt content |

**Pinned skills** — an application can load skills deterministically instead of waiting for the model to call `Skill`. Name them in the `pinned_skills` state key; before the next model call each skill's content is added to the conversation once and the key is cleared:

```python
await graph.ainvoke({"messages": [HumanMessage("Size this market")], "pinned_skills": ["market-sizing"]})
```

Unknown names are logged and skipped. Descriptions longer than the roster cap log a warning at discovery.

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
    max_turns=10,                    # hard recursion ceiling; unset → inherit parent's budget
    mode="isolated",                 # "isolated" (default) or "fork"
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

Unknown argument keys are rejected with a tool error that names the key, so a task placed under an invented key never reaches the subagent as an empty request.

**Context modes:**

| `mode` | The subagent starts with |
|---|---|
| `isolated` (default) | Only the delegated message |
| `fork` | The parent's conversation so far, plus the delegated instruction. A fork cannot delegate again. |

**Resuming a subagent** — each result carries an agent id. Passing it back continues that subagent's conversation with its full history:

```json
{"agent": {"id": "researcher"}, "resume": "<agent id>", "message": "Now check the 2025 figures"}
```

Child conversations are saved through the parent's checkpointer, scoped to the parent thread, so one conversation cannot resume another's children. Without a checkpointer, results carry no id and `resume` returns an error.

**Approvals inside subagents** — an `interrupt()` raised in a subagent (for example by a `HITLExtension` on the subagent) surfaces at the parent graph. Resuming the parent with `Command(resume=...)` finishes the subagent.

**Provenance** — the task a subagent receives names the delegating agent, and the `Agent` tool result is the subagent's report, so neither is mistaken for a message from the user.

**Key features:**
- `description` — used in the prompt roster so the LLM knows what each specialist does
- `tools="inherit"` — subagent receives the parent's tools at delegation time
- `ephemeral=True` — enables dynamic (on-the-fly) reasoning agents
- `skills` preloading — full skill content injected into agent's prompt at startup
- `model` override — per-agent model selection via `model_resolver`
- `delegation_timeout` — max seconds per delegation (default 300s)
- `max_turns` inheritance — a subagent that sets no `max_turns`/`maxTurns` inherits the
  outer agent's turn budget: a parent `TurnBudgetExtension` (soft, graceful wrap-up) and/or
  the parent's hard ceiling (`Agent`/`AgentKit` `max_turns`). An explicit per-subagent value
  is a hard ceiling and overrides inheritance. A subagent that hits its hard ceiling returns a
  distinct "reached its turn limit" result rather than a generic error.

See [`examples/delegation.py`](examples/delegation.py) for a complete example.

### TasksExtension

Task tracking for complex multi-step objectives. The agent creates, tracks, and completes tasks with dependency ordering. Tasks are a passive todo list; they do not execute work.

```python
ext = TasksExtension()
ext.tools  # [TaskCreate, TaskUpdate, TaskList, TaskGet]
```

**Tools:**

| Tool | Description |
|------|-------------|
| `TaskCreate` | Create a task with subject, description, and optional spinner text |
| `TaskUpdate` | Update status, owner, metadata, or dependencies |
| `TaskList` | List all non-deleted tasks with status and dependencies |
| `TaskGet` | Get full task details including computed `blocks` |

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
| `Read(file_path)` | Read a file with line numbers and offset/limit pagination; reports the total line count and the offset to continue from. Images and PDFs come back as content blocks the model can see. |
| `Write(file_path, content)` | Create or overwrite a file |
| `Edit(file_path, old_string, new_string)` | Exact string replacement |
| `Glob(pattern)` | Find files by pattern (supports `*`, `**`, `?`). A pattern without `/` matches file names at any depth. |
| `Grep(pattern)` | Search file contents by regex |
| `Bash(command)` | Execute shell commands (when backend supports `execute()`) |

Glob and Grep stop at a result cap and tell the model when results are partial; search failures are reported instead of looking like "no matches". Writes and edits to the same file within one step run one after another. When the active model's profile says it cannot accept images or PDFs, those blocks are replaced with a short placeholder for that request only; saved state keeps the originals.

**Alternative backends:** `FilesystemExtension` accepts any object satisfying `FilesystemProtocol`. The framework ships four optional remote/sandboxed/unified-VFS backends alongside the default `OSBackend` — see the [FilesystemExtension Backends](#filesystemextension-backends) section for the full catalog and runnable examples.

### WebExtension

The web capability: a `WebFetch` tool that reads one URL, plus a `WebSearch`
tool when you pass search `providers`. Without providers only `WebFetch` is
exposed. Both need the `web` extra (`pip install langchain-agentkit[web]`).

```python
from langchain_agentkit import WebExtension, DuckDuckGoSearchProvider

# WebFetch only
ext = WebExtension()

# WebSearch (DuckDuckGo HTML mode, no API key) + WebFetch
ext = WebExtension(providers=[DuckDuckGoSearchProvider(use_html=True)])

# Multiple providers, queried concurrently and merged
from langchain_agentkit import QwantSearchProvider

ext = WebExtension(
    providers=[DuckDuckGoSearchProvider(), QwantSearchProvider()],
    fan_out=True,
)
```

`WebSearch` returns typed `SearchResult` records. Providers are queried in
fallback order by default — the first to return results wins. Set
`fan_out=True` to query every provider concurrently and merge, dedupe, and rank
the results. A provider is any `SearchProvider` returning `list[SearchResult]`.
To use a backend (Tavily, Exa, Brave, …), subclass and map its response:

```python
from langchain_agentkit import SearchProvider, SearchResult

class TavilyProvider(SearchProvider):
    name = "Tavily"

    def __init__(self, client):
        self._client = client

    def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        hits = self._client.search(query, max_results=max_results)["results"]
        return [
            SearchResult(url=h["url"], title=h["title"], snippet=h["content"], score=h.get("score"))
            for h in hits
        ]
```

`WebFetch(url, format="markdown" | "text" | "html", max_content_tokens=…)` reads
one URL. It asks for markdown first via `Accept`, converts HTML to the requested
format (markdown by default), and truncates to a token budget. Non-text content
types (images, PDFs, archives) are refused. A Cloudflare bot challenge is retried
once with a plain agent User-Agent before failing. Fetch is SSRF-guarded (scheme
allowlist, private/loopback/metadata IP blocking with IPv4-mapped and 6to4
unwrapping, DNS pinning against rebinding, per-hop redirect re-validation). For
human approval on each fetch, compose `HITLExtension(interrupt_on=["WebFetch"])`.

### HistoryExtension

Manage conversation history to keep the LLM context window lean. Rewritten history replaces graph state via `ReplaceMessages` so the checkpointer stays compact.

```python
from langchain_agentkit import (
    CompactionStrategy,
    CountStrategy,
    HandoffStrategy,
    HistoryExtension,
    TokenStrategy,
)

# Keep the last 50 messages
ext = HistoryExtension(strategy=CountStrategy(max_messages=50))

# Keep messages within a token budget
ext = HistoryExtension(strategy=TokenStrategy(max_tokens=4000))

# Summarize older history when the context window fills up
ext = HistoryExtension(strategy=CompactionStrategy(reserve_tokens=16_384))

# Replace older history with a brief written toward the current request
ext = HistoryExtension(strategy=HandoffStrategy(reserve_tokens=16_384))

# Custom strategy — any object with async transform(messages, *, runtime)
ext = HistoryExtension(strategy=MyStrategy())
```

| Strategy | What it keeps |
|---|---|
| `CountStrategy` / `TokenStrategy` | The most recent messages within the limit |
| `CompactionStrategy` | A running summary of older history plus the recent messages |
| `HandoffStrategy` | A fresh brief (facts, decisions, current state, open items, relevant files) conditioned on the latest user request, plus that request verbatim |

All built-in strategies preserve a leading `SystemMessage`. Compaction and handoff derive their previous summary from the conversation itself, so one strategy instance is safe to share across threads and subagents.

**Overflow recovery** — when the model call raises `ContextOverflowError`, a strategy that implements `compact()` (compaction and handoff do) shrinks the history and the call is retried once, only if the new request is smaller. Otherwise the original error propagates.

### MemoryExtension

Persistent memory across sessions. The extension frames the content; a strategy decides where it comes from.

```python
from langchain_agentkit import CuratedMemory, MemoryExtension, RecalledMemory

# An operator-authored MEMORY.md, always in the system prompt (default)
ext = MemoryExtension()  # ~/.agents/memory/<project-key>/MEMORY.md
ext = MemoryExtension(strategy=CuratedMemory(path="./memory", project_discovery=False))

# A store the agent writes and searches itself (needs the [memory] extra)
ext = MemoryExtension(strategy=RecalledMemory(namespace=("memories", "{user_id}"), auto_recall=3))
```

Declare the extension twice to combine both: a curated document that is always present, alongside a store the agent queries on demand.

**`RecalledMemory`** is backed by the LangGraph store passed to `compile(store=...)`:

- The agent records and searches memories through tools; `auto_recall=K` adds the top `K` hits for the latest request to each user turn.
- `extract=True` runs a background reflector that writes memories from the conversation after a lull.
- To review the agent's own writes, gate its memory tool with `HITLExtension(interrupt_on={"manage_memory": True})`.
- To review background extraction, set `staging_namespace`. Extracted memories land there, invisible to recall, until you call `approve_staged(key)` or `reject_staged(key)`; `list_staged()` shows what is pending.
- `consolidate(source=..., target=...)` merges duplicates and resolves contradictions into a separate namespace, leaving the source untouched so the result can be reviewed before it is swapped in. Run it from your own scheduler.

```bash
pip install 'langchain-agentkit[memory]'
```

### HITLExtension

Human-in-the-loop via a unified Question protocol. Two capabilities:

**Tool approval** — gate sensitive tools with human review:

```python
hitl = HITLExtension(interrupt_on={
    "send_email": True,           # approve / reject
    "delete_file": {"options": ["approve", "reject"]},
})
# Tools not listed in interrupt_on execute normally without interruption.
```

**AskUser tool** — the `AskUser` tool lets the LLM ask structured questions. It is provided by default:

```python
hitl = HITLExtension()  # provides [AskUser]

# Combine with approval gating (AskUser is still included):
hitl = HITLExtension(interrupt_on={"send_email": True})

# Drop AskUser entirely:
hitl = HITLExtension(interrupt_on={"send_email": True}, tools=[])

# Replace AskUser with a custom tool:
hitl = HITLExtension(tools=[my_custom_tool])
```

Both use the same interrupt payload (`Question` objects) and resume format. Approval requests raised inside a subagent surface at the parent graph and resume the same way.
Requires a checkpointer. Answers are keyed by each question's position in the
interrupt `questions` array (as a string), so resume with
`Command(resume={"answers": {"0": "Approve"}})`. For tool approval, `"Approve"`
runs the tool and `"Reject"` denies it; any other value is forwarded to the LLM.

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

**How it works:** Teammates run as `asyncio.Task`s with their own checkpointers (conversation history persists across messages). A **Router Node** in the graph checks for teammate messages after each tool execution — when a teammate sends a result, the lead is automatically re-invoked with the message. Messages between the lead and teammates name their sender and are marked as not coming from the user.

Teammates run outside the lead's graph step, so an approval request raised inside a teammate does not reach the lead; it is reported back as an error message.

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

### RubricExtension

Rubric-gated iteration: a separate (often cheaper) grader sub-agent reviews the agent's work against a caller-supplied `rubric` and drives revision until every criterion is met. Pass the rubric on invocation state; with no rubric the extension is a no-op, so it's safe to include unconditionally.

```python
from langchain_anthropic import ChatAnthropic
from langchain_agentkit.extensions.rubric import RubricExtension, ReviewPolicy

ext = RubricExtension(
    model=ChatAnthropic(model="claude-haiku-4-5"),  # grader (separate from the agent's model)
    mode="auto",                                     # auto | user
    review=ReviewPolicy(min=1, max=5, stop=7),
)
# invoke with:  {"messages": [...], "rubric": "- criterion A\n- criterion B"}
```

**Modes** — what happens when the grader returns `needs_revision`:

| `mode` | Action | Who's in the loop |
|--------|--------|-------------------|
| `auto` | Grader feedback is injected; the model re-drafts autonomously until satisfied | Nobody — closed loop |
| `user` | The turn yields to the user; the agent sees the rubric + open gaps and converses (prose, or a HITL tool of its choosing) toward the criteria | The user, across turns |

**`review`** — when the grader runs (an `int`, a dict, or a `ReviewPolicy`):

| Field | Unit | Meaning |
|-------|------|---------|
| `min` | turns | earliest assistant turn to grade |
| `max` | turns | force a grade by this turn; the agent may call `request_review()` earlier |
| `stop` | reviews | give up after this many reviews (graceful terminal `review_limit_reached`) |

`review=1` grades every turn; `review=5` expands to `(min=1, max=5)`. A `request_review` tool is offered in `user` mode when a discretionary window exists (`max > min`).

The grader runs off-graph and uses tool-calling structured output, so it works across providers. Its `_rubric_*` bookkeeping is marked `PrivateStateAttr` — kept out of the `invoke` result; observe the verdict via `get_state`, the `on_evaluation` callback, or `rubric_evaluation_*` stream events. Terminal statuses: `satisfied`, `failed`, `review_limit_reached`, `grader_error`.

In `auto` mode the revision loop routes via `jump_to`, honored only when the kit exposes at least one tool; the rubric's `review.stop` should sit within the agent's `max_turns` ceiling (see [Recommended ordering](#recommended-ordering)).

See [`examples/rubric.py`](examples/rubric.py) for a complete example.

## FilesystemExtension Backends

The [`FilesystemExtension`](#filesystemextension) above operates on top of a backend that implements `FilesystemProtocol` (file ops + bulk transfer) and optionally `SandboxProtocol` (adds `execute` and `environment`). Capability gating is structural: the `Bash` tool registers automatically when the backend implements `SandboxProtocol`. Different backends address different deployment shapes — host-local development, cloud sandboxes, persistent virtual filesystems, multi-tenant Linux hosts, or unified multi-service VFS.

| Backend | Install | Capability | Use when |
|---------|---------|------------|----------|
| [`OSBackend`](src/langchain_agentkit/backends/os.py) | built-in | `FilesystemProtocol` + `SandboxProtocol` | Local development against the host filesystem with path-traversal prevention. |
| [`DaytonaBackend`](examples/daytona/README.md) | `pip install langchain-agentkit[daytona]` | `FilesystemProtocol` + `SandboxProtocol` | Cloud sandbox per agent run; a fresh isolated container with full Linux tooling. |
| [`AgentFSBackend`](examples/agentfs/README.md) | `pip install langchain-agentkit[agentfs]` | `FilesystemProtocol` only | Agent state persists in a single content-addressed `.db` file. Pair with a sandbox backend if you also need shell exec. |
| [`BubblewrapBackend`](examples/bubblewrap/README.md) | `apt-get install bubblewrap` + `pip install langchain-agentkit[bubblewrap]` | `FilesystemProtocol` + `SandboxProtocol` | Linux multi-tenant hosts; per-call `bwrap` namespace + cgroup isolation, no remote service. |
| [`MirageBackend`](examples/mirage/README.md) | `pip install langchain-agentkit[mirage]` | `FilesystemProtocol` + `SandboxProtocol` | Multi-mount unified VFS — S3, Google Drive, Slack, GitHub, etc. side-by-side under one tree, with cross-mount pipelines. |

```python
# Pick any backend that satisfies the protocol you need
from langchain_agentkit.backends.daytona import DaytonaBackend     # cloud sandbox
# from langchain_agentkit.backends.agentfs import AgentFSBackend    # SQLite-backed VFS
# from langchain_agentkit.backends.bubblewrap import BubblewrapBackend  # Linux sandbox
# from langchain_agentkit.backends.mirage import MirageBackend      # multi-mount VFS

ext = FilesystemExtension(backend=...)
```

`glob()` and `grep()` return `GlobResult` / `GrepResult` with `truncated` and `error_message`, and every backend applies the same glob rules. Remote backends run searches with `python3` inside the sandbox.

Each example under [`examples/<backend>/`](examples/) ships a runnable end-to-end agent against that backend; see the per-example README for upstream links, prerequisites, and the recommended composition pattern.

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
