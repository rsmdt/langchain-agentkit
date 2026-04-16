# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Entries are added only when a release is cut. Work in progress is not tracked here — see the git history between the latest tag and `HEAD` for unreleased changes.

This file retains detailed entries for the last 10 minor releases plus their patch revisions. Older release notes can be found in the git history and on each version's [GitHub release page](https://github.com/rsmdt/langchain-agentkit/releases).

## [0.23.1] — 2026-04-16

### Changed

- **AgentsExtension owns subagent trace filtering end-to-end.** The filter is now a `wrap_model` hook on `AgentsExtension` itself (active only when `output_mode` tags messages hidden-from-LLM — the built-in `trace_hidden` strategy, or custom strategies that set `_tags_hidden_from_llm=True`). Consumers who use `trace_hidden` must declare `AgentsExtension` **after** `HistoryExtension` in the extensions list; `setup()` raises `ValueError` at kit-construction time when the ordering is violated. Mirrors the ordering check already used by `TeamExtension`.

### Removed

- **`HideSubagentTraceExtension`** — removed. Its functionality is now provided by `AgentsExtension.wrap_model`. The low-level `strip_hidden_from_llm()` helper remains exported for advanced users writing custom strategies that bypass the standard onion.

## [0.23.0] — 2026-04-16

### Added

- **AgentsExtension `output_mode`** — subagent results now flow through a pluggable strategy. Three built-ins: `"last_message"` (a single ToolMessage with the final text, matches langgraph-supervisor/deepagents), `"full_history"` (all subagent AIMessages + final ToolMessage, matches supervisor's full-history mode), and `"trace_hidden"` *(new default)* which persists every subagent AIMessage tagged `{prefix}_hidden_from_llm=True` plus the terminal ToolMessage — so UIs reading `AIMessage.content` blocks render reasoning as thinking on reload while the parent LLM context stays lean. Consumers can pass a custom callable with signature `(SubagentOutput, StrategyContext) -> list[BaseMessage]`.
- **Metadata tag namespace** — `AgentsExtension(metadata_prefix="agentkit")` controls the `response_metadata` keys written by the strategy (`{prefix}_subagent_tool_call_id`, `{prefix}_subagent_name`, `{prefix}_subagent_final`, `{prefix}_hidden_from_llm`). Overridable.
- **HideSubagentTraceExtension** — *(superseded in 0.23.1; functionality folded into `AgentsExtension`)*. Paired `wrap_model` extension that strips messages flagged `{prefix}_hidden_from_llm=True` from the per-request message list.
- **Public API**: `SubagentOutput`, `StrategyContext`, `SubagentOutputStrategy`, `last_message_strategy`, `full_history_strategy`, `trace_hidden_strategy`, `resolve_output_strategy`, `strip_hidden_from_llm`, `DEFAULT_METADATA_PREFIX`.

### Changed

- **BREAKING (behavior)**: default output of `AgentsExtension` is now `"trace_hidden"` instead of a single JSON-stringified `ToolMessage`. Consumers who relied on the old JSON-dump shape should set `output_mode="last_message"` to restore a single plain-text `ToolMessage`. The private helper `_extract_final_response` has been removed; use `last_message_strategy` or call `BaseMessage.content` directly.

## [0.22.0] — 2026-04-15

### Added

- **ResilienceExtension** gains a `wrap_model` hook (Layer 2 — read-time repair). Before every LLM call, scans the message list for `AIMessage(tool_calls=[...])` whose `tool_call_id`s have no paired `ToolMessage` and injects synthetic ones so the request is always well-formed. Repairs orphans that pre-date the `wrap_tool` layer — e.g. from crashes before `0.21.2`, pod kills between checkpoint writes, transient checkpointer failures, or manual state edits. Only fires when entering the model node, so legitimate in-flight tool calls paused by HITL interrupts are never misidentified. Controlled via `repair_orphan_tool_calls` (default `True`), `orphan_repair_message`, and `on_orphan_repaired` callback. New public type: `OrphanRepairEvent`.

## [0.21.2] — 2026-04-15

### Added

- **ResilienceExtension** (Layer 1 — write-time prevention). A new extension that wraps every tool call and converts any unhandled exception into a synthetic `ToolMessage` paired to the originating `tool_call_id`. Prevents the orphan-tool-call checkpoint state that causes the OpenAI Responses API to reject subsequent turns with `"No tool output found for function call"`. `ToolException` and `asyncio.CancelledError` are re-raised so typed error contracts and cancellation semantics are preserved. Emits `ToolErrorEvent` via a WARN log and an optional `on_tool_error_caught` callback for forwarding to metrics or incident sinks.

## [0.21.1] — 2026-04-15

### Fixed

- **AgentsExtension / TeamExtension** now pick up kit-level `llm_getter` and `tools_getter` during `setup()`. Config-based agents without an explicit `model` previously crashed at tool-call time with `TypeError: 'NoneType' object is not callable` because `_parent_llm_getter` was never wired. Explicit `set_parent_*_getter()` calls still win.

## [0.21.0] — 2026-04-15

### Added

- **MemoryExtension** gains a `backend` field for remote-filesystem reads. When set, async `setup()` primes the cached body and `before_model` refreshes it each turn; local-filesystem mode keeps its sync-per-turn semantics.
- **TeamExtension** accepts `agents` as a programmatic list, a directory path, or a path + `BackendProtocol` — mirroring `AgentsExtension`. `AgentConfig`-based teammates are compiled on demand with resolved model, filtered parent tools, merged skills, and bus-proxied task tools.

### Changed

- **BREAKING:** `MemoryExtension` no longer accepts a `loader` callable. Use `backend` for remote reads; local reads continue to use `path`.
- **BREAKING:** `TeamExtension.setup()` is now `async`. Callers composing kits via `AgentKit` are unaffected (setup is awaited through `run_extension_setup`); direct callers must `await team.setup(...)`.

### Fixed

- Agent tool coroutines are now introspectable by LangGraph's `ToolNode`. `functools.partial` wrappers caused `typing.get_type_hints` to reject injected tools; they are replaced with a plain async closure that preserves the injectable-argument contract.

## [0.20.0] — 2026-04-14

### Added

- **ContextCompactionExtension** for long-running agents — automatically manages context window size to prevent token limit issues during extended conversations.
- **EnvExtension** extracted as a standalone extension for environment detection, available for opt-in use outside presets.

### Changed

- **BREAKING:** Prompt composition simplified to two channels: `prompt` and `reminder`. The `prompt_cache_scope` and static-vs-dynamic routing layers have been removed — extensions should target one of the two channels directly.
- **BREAKING:** The `"full"` preset now seeds only `CoreBehavior` and `Tasks`. Memory and environment detection are no longer included by default — add `MemoryExtension` and `EnvExtension` explicitly if needed.

## [0.19.0] — 2026-04-14

### Added
- `CoreBehaviorExtension` contributes universal domain-neutral agent guidance plus a per-turn `<env>` block with cwd, platform, shell, and git detection.
- `MemoryExtension` surfaces a `MEMORY.md` file each turn, with configurable path, project-key sanitization, and size caps.
- `MessagePersistenceExtension` forwards per-turn generated messages to a caller-supplied async callback, computing turn deltas by diffing message IDs so it remains correct across HITL resume and history truncation.
- `preset="full"` constructor parameter seeds `AgentKit` with `CoreBehavior`, `Tasks`, and `Memory`.
- Built-in always-on `currentDate` reminder; extensions can contribute reminders via `{"prompt", "reminder"}` dict returns.
- LLM behavioral eval coverage (25 new evals) exercising tool-description preferences, CoreBehavior terseness/parallelism, task workflow, skill disambiguation, memory steering, reminders, and `ask_user` flows.

### Changed
- **BREAKING**: `compose()` now returns a frozen `PromptComposition` dataclass with `static`/`dynamic`/`reminder` channels instead of a tuple. Extensions declare `prompt_cache_scope` and may return `{"prompt", "reminder"}` dicts.
- **BREAKING**: Universal agent guidance moved from `TasksExtension`'s `BASE_AGENT_PROMPT` to `CoreBehaviorExtension`; `BASE_AGENT_PROMPT` is no longer exported from the tasks module.
- **BREAKING**: `SkillsExtension` now returns the skills roster via the reminder channel rather than inline in the prompt.
- Tool descriptions now live as module-level docstrings in `extensions/<ext>/tools/<tool>.py`, replacing the prior fixture-file + loader pattern. Single-file `tools.py` modules expanded into packages with one module per tool. Public factory imports are preserved; some internal private symbols moved paths.

### Removed
- **BREAKING**: `ReminderConfig`, `AgentKit.full()`, the tuple `compose()` return, and the `{"static", "dynamic"}` dict keys. Use `preset="full"` and the `PromptComposition` dataclass instead.

## [0.18.0] — 2026-04-11

### Added

- **Agent base class** for declarative agent definition — supports properties as class attributes, sync methods, or async methods, all resolved uniformly at `compile()`. The legacy `agent` metaclass is retained but undocumented.
- **Cross-turn team rehydration** — teams now survive HTTP request/response boundaries. `TeamMetadata` in graph state is sufficient to rebuild the bus, asyncio tasks, and compiled graphs on any pod, with graceful degradation for missing roster agents.
- **`AgentKit.compile(handler)`** as the primary graph-building path — absorbs the full ReAct loop construction (agent node, tool node, hooks, routing) so callers no longer wire graphs manually.
- Authoritative team-message filtering via a single `additional_kwargs["team"]` key, replacing fragile name-based heuristics across message types.

### Changed

- **BREAKING:** `BackendProtocol` is now fully async — all methods (`execute`, `read`, `write`, `edit`, `glob`, `grep`) are coroutines. `OSBackend` uses `asyncio.create_subprocess_shell`; `DaytonaBackend` awaits its shell calls. All tools and extensions have been migrated to `await` backend calls.
- **BREAKING:** `AgentKit` constructor now takes keyword-only arguments (`tools`, `model`, `model_resolver`, `name`). `asetup()` is replaced by the standalone `run_extension_setup(kit)` function.
- **BREAKING:** `AgentExtension` renamed to `AgentsExtension`. The extension no longer accepts its own `model_resolver` — it picks up the kit-level resolver during `setup()`.
- **BREAKING:** `BackendProtocol.read()` returns raw text instead of line-numbered output. Line numbering moves to the Read tool as a presentation concern.
- `before_model` hook returns now merge into the agent node's output so they reach the checkpointer, enabling hook-based message capture such as the teammate message flush.
- Team state consolidated into a typed `TeamMetadata` reducer (name, members as `TeammateSpec`, created\_at) with replace-wins semantics, replacing the untyped `team_members`/`team_name` channels.
- Public API surface expanded: `AgentKit.extensions`, `base_prompt`, and `compose()` are now documented public methods; `HookRunner.has_run_hooks` replaces private `_hooks` access.

### Fixed

- `build_ephemeral_graph` now calls `bind_tools` so ephemeral agents can emit tool calls.
- Variable shadowing in `_check_teammates` resolved with hoisted assignment.

### Removed

- Unused `team_messages` list reducer and `_append_messages` helper — team communication routes through the bus, not graph state.
- `tools_inherit` attribute on `_AgentConfigProxy` — definition-based agents now route through `_agent_config` before any inherit check.
- `resolve_agent_fn` injection hook — `AgentExtension` and `TeamExtension` share a single resolver via the extracted `agents/refs` module.

## [0.17.0] — 2026-04-09

### Changed
- **BREAKING**: Handlers now own tool binding. The framework injects the raw `llm` and the composed `tools` list instead of pre-binding, so handlers must call `llm.bind_tools(tools, ...)` themselves before invocation. This unlocks full control over provider-specific kwargs (`strict`, `parallel_tool_calls`, `tool_choice`) and enables dynamic per-step tool filtering. Handlers that relied on a pre-bound `llm` will silently stop seeing tool calls — update them to accept `tools` in the keyword-only parameters and wrap invocation as `await llm.bind_tools(tools).ainvoke(messages)`.

## [0.16.0] — 2026-04-09

### Added
- `Extension.setup()` lifecycle hook, called once after dependency resolution and before graph building. Extensions declare only the kwargs they need (`extensions`, `prompt`) and AgentKit dispatches via signature introspection, letting extensions self-wire without leaky AgentKit plumbing.
- `CHANGELOG.md` with backfilled entries for v0.6.0 through v0.15.0, plus a reusable `scripts/changelog-entry.sh` generator that pipes commits to `claude -p` at release time. Release flow is documented in `CLAUDE.md`.
- README coverage for `HistoryExtension` and the `setup()` lifecycle hook.

### Changed
- **BREAKING**: `model_resolver` moved from `AgentKit.__init__` to `AgentsExtension`, where it's actually used. `AgentKit.resolve_model()` now scans extensions for any with a `model_resolver` attribute, so string-based model references in the agent metaclass pattern continue to work — but callers passing `AgentKit(model_resolver=...)` must migrate.
- **BREAKING**: `TasksExtension`, `FilesystemExtension`, and `AgentsExtension` now derive team awareness, HITL availability, and skills resolver discovery inside their own `setup()` implementations. The following setters and constructor arguments have been removed: `TasksExtension(team_active=...)`, `AgentsExtension.set_model_resolver()`, `AgentsExtension.set_skills_resolver()`, and `FilesystemExtension.set_hitl_available()`.
- `CLAUDE.md` eval test documentation now distinguishes dataset validation (`test_eval_runner.py`, no API calls) from LLM integration evals (`-m eval`, requires `OPENAI_API_KEY`), and warns that `pytest -m eval` skips the dataset validation tests.

## [0.15.0] — 2026-04-09

### Added
- New `HistoryExtension` for context window management, providing pluggable strategies to truncate messages before each LLM call. Built-in strategies include `count` (keep last N messages) and `tokens` (keep within a token budget), plus support for custom strategies via any object implementing `transform(messages)`. Uses `ReplaceMessages` to bulk-replace graph state so the checkpointer stays lean while preserving `SystemMessage` entries.

### Changed
- **BREAKING**: Hook API unified to keyword-only arguments. All hooks (`before`, `after`, `wrap`, `on_error`) now receive `state` and `runtime` as keyword args. `wrap_model` hooks now consistently receive `(state, handler, runtime)`, matching the other hook types. Update any custom hooks to use keyword-only parameters.

### Removed
- **BREAKING**: Removed the `process_history` pipeline from `HookRunner`. It had no consumers and is superseded by `HistoryExtension` for context window management. Migrate any history-processing logic to a custom `HistoryExtension` strategy.

## [0.14.0] — 2026-04-08

Version bump only — no code changes.

## [0.13.1] — 2026-04-08

### Fixed
- Restored Python 3.11 and 3.13 compatibility for tool registration — async functions are now wrapped in closures instead of `functools.partial`, which `inspect.iscoroutinefunction` failed to recognize on Python < 3.14 and caused `StructuredTool.from_function` to reject them.
- Prevented `OSError` when loading long or multi-line inline prompt strings on Linux by skipping filesystem existence checks for content that clearly isn't a file path (newlines or >255 chars).
- Fixed subagent graph resolution when stubs satisfy `AgentLike` via `runtime_checkable` Protocol auto-matching, ensuring `_compile_or_resolve` and `wrap_if_needed` correctly identify and compile raw graphs.

### Changed
- Added `ruff format --check` to the local development workflow and documented commands to keep formatting in sync with CI and prevent drift.

## [0.13.0] — 2026-04-08

### Added
- **System-reminder protocol**: extensions can now return a `{prompt, reminder}` dict, with reminders injected as ephemeral `HumanMessage`s at LLM call time so dynamic per-step context (task lists, team status) never pollutes conversation history.
- **Task proxy protocol** for teammate-to-lead task operations: teammates create/update/list/get shared tasks via structured JSON messages routed over the `TeamMessageBus`, with proxies wired into both ephemeral and predefined teammates.
- **Task coordination v2**: dependency enforcement, claim validation, auto-owner assignment, deletion cascade, resolved-blocker filtering, internal task hiding, assignment notifications, completion nudge, task-based agent status, and task cleanup on team dissolution.
- **Type-aware `Read` tool**: dispatches by extension to return multimodal image blocks (PNG/JPG/GIF/WebP), extract PDF pages, render notebook cells with outputs, and reject unknown binaries with a clear message. Adds `read_bytes` to `BackendProtocol`.
- **Structured output on all file tools** via `content_and_artifact`, aligning `Read`, `Write`, `Edit`, `Glob`, `Grep`, and `Bash` input schemas, output messages, and behavior (file-unchanged dedup, structured patches, quote normalization, context precedence, pagination, multiline/type filters, etc.).
- **Extension enhancements**: skills budget controls (`budget_percent`, `max_description_chars`, `context_window`) with plain-text description format; agent roster showing per-agent tool access; owner display in task context; richer HITL `ask_user` with `Option.preview`.
- **`stderr` field on `ExecuteResponse`** so downstream consumers can distinguish error output from stdout.
- **`BackendProtocol` integration conformance suite**: parameterized matrix runs the full protocol contract against `OSBackend` always, and `DaytonaBackend` when `DAYTONA_API_URL` is set.

### Changed
- **BREAKING**: Team tools renamed to a consistent `Team*` prefix — `TeamCreate`, `TeamMessage`, `TeamStatus`, `TeamDissolve`.
- **BREAKING**: `AssignTask` and `MessageTeammate` consolidated into a single `SendMessage` tool with broadcast support (`to: "*"`). Raw `SHUTDOWN_SIGNAL` replaced with a structured JSON shutdown handshake for safer team dissolution.
- **BREAKING**: All extensions now use keyword-only `__init__` arguments.
- **BREAKING**: `BaseSandbox` ABC removed in favor of a standalone `DaytonaBackend`; all shell-based file operations are inlined per backend, and the deprecated `backend.py` compatibility shim is gone. `DaytonaBackend.execute()` gains path-traversal protection and proper error handling.
- `OSBackend.execute()` now returns stdout only in the `output` field, matching `DaytonaBackend`.
- `tools.py` split into a per-tool package (`read`, `write`, `edit`, `glob`, `grep`, `common`) with backend-scoped read-state cache to eliminate redundant file reads.

### Fixed
- `OSBackend._resolve()` no longer doubles paths when the LLM constructs absolute paths from the prompt-exposed root directory.
- Skill and agent discovery gracefully skips files with malformed or missing YAML frontmatter, logging a warning instead of crashing.
- `_BashInput.timeout` description corrected from "milliseconds" to "seconds".
- `_wrap_with_permission_check` now preserves `response_format` from the wrapped tool.

## Older release history

- [0.20.0]: https://github.com/rsmdt/langchain-agentkit/releases/tag/v0.20.0
- [0.19.1]: https://github.com/rsmdt/langchain-agentkit/releases/tag/v0.19.1
- [0.19.0]: https://github.com/rsmdt/langchain-agentkit/releases/tag/v0.19.0
- [0.18.0]: https://github.com/rsmdt/langchain-agentkit/releases/tag/v0.18.0
- [0.17.0]: https://github.com/rsmdt/langchain-agentkit/releases/tag/v0.17.0
- [0.16.0]: https://github.com/rsmdt/langchain-agentkit/releases/tag/v0.16.0
- [0.15.0]: https://github.com/rsmdt/langchain-agentkit/releases/tag/v0.15.0
- [0.14.0]: https://github.com/rsmdt/langchain-agentkit/releases/tag/v0.14.0
- [0.13.1]: https://github.com/rsmdt/langchain-agentkit/releases/tag/v0.13.1
- [0.13.0]: https://github.com/rsmdt/langchain-agentkit/releases/tag/v0.13.0
- [0.12.0]: https://github.com/rsmdt/langchain-agentkit/releases/tag/v0.12.0
- [0.11.0]: https://github.com/rsmdt/langchain-agentkit/releases/tag/v0.11.0
- [0.10.0]: https://github.com/rsmdt/langchain-agentkit/releases/tag/v0.10.0
- [0.9.0]: https://github.com/rsmdt/langchain-agentkit/releases/tag/v0.9.0
- [0.8.0]: https://github.com/rsmdt/langchain-agentkit/releases/tag/v0.8.0
- [0.7.0]: https://github.com/rsmdt/langchain-agentkit/releases/tag/v0.7.0
- [0.6.0]: https://github.com/rsmdt/langchain-agentkit/releases/tag/v0.6.0
- [0.5.0]: https://github.com/rsmdt/langchain-agentkit/releases/tag/v0.5.0
- [0.4.1]: https://github.com/rsmdt/langchain-agentkit/releases/tag/v0.4.1
- [0.4.0]: https://github.com/rsmdt/langchain-agentkit/releases/tag/v0.4.0
- [0.3.0]: https://github.com/rsmdt/langchain-agentkit/releases/tag/v0.3.0
- [0.2.1]: https://github.com/rsmdt/langchain-agentkit/releases/tag/v0.2.1
- [0.2.0]: https://github.com/rsmdt/langchain-agentkit/releases/tag/v0.2.0
