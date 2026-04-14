# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Entries are added only when a release is cut. Work in progress is not tracked here — see the git history between the latest tag and `HEAD` for unreleased changes.

This file retains detailed entries for the last 10 minor releases plus their patch revisions. Older release notes can be found in the git history and on each version's [GitHub release page](https://github.com/rsmdt/langchain-agentkit/releases).

## [Unreleased]

### Changed

- Tool descriptions now live on each tool function as module-level constants / docstrings, matching LangChain's convention. The per-tool `.md` fixture files under `src/langchain_agentkit/extensions/<ext>/descriptions/` and the `_descriptions.py` loader have been deleted. Description content is byte-identical to the previous fixtures.
- The `Skill` tool description is now static. The per-skill roster is delivered through AgentKit's built-in reminder channel — `SkillsExtension.prompt()` returns `{"prompt": <static framing>, "reminder": <roster>}` and AgentKit appends the `reminder` value under a `# SkillsExtension` block inside the `<system-reminder>` envelope.
- **BREAKING:** Reminders are now a built-in, non-configurable capability. `AgentKit.compose(state, runtime).reminder` always emits today's date (`# currentDate`). Extensions contribute additional reminder content by returning `{"reminder": <text>}` from `prompt()`; AgentKit concatenates them under a per-extension `# <ClassName>` header inside the envelope.
- **BREAKING:** Removed the built-in `AGENTS.md` walker. AgentKit no longer auto-discovers `AGENTS.md` files from `cwd` upward or from `~/.agents/AGENTS.md`. Inject project context explicitly via `AgentKit(prompt=Path("AGENTS.md"))` or a list such as `prompt=[Path("AGENTS.md"), Path("~/.agents/AGENTS.md").expanduser()]` — `prompt` already accepts `str | Path | list[str | Path] | None` and loads file paths transparently.
- **BREAKING:** `AgentKit.compose(state, runtime)` now returns a frozen `PromptComposition(static, dynamic, reminder)` dataclass instead of a `(prompt, system_reminder)` tuple. Use `composition.joined` to obtain the combined `static + "\n\n" + dynamic` system prompt. Callers that tuple-unpacked the previous return must be updated.
- **BREAKING:** Extension `prompt()` may now return `str | None | dict[str, str]`. The recognized dict keys are `"prompt"` (routed by `prompt_cache_scope`) and `"reminder"` (appended to the built-in reminder channel). The previous `{"static", "dynamic"}` dict shape is no longer recognized; unknown keys are silently ignored.
- **BREAKING:** Existing extensions' `prompt()` output has been trimmed and realigned. Callers that asserted on specific prompt strings must update:
  - `FilesystemExtension` now emits only a minimal `Filesystem root: <path>` line when the backend root differs from `cwd`, otherwise `None`. Tool-name guidance (Read/Write/Edit/Glob/Grep/Bash) moved to each tool's `description`. Declared `prompt_cache_scope = "static"`.
  - `SkillsExtension` keeps the dynamic skill roster plus progressive-disclosure framing but drops generic per-step guidance (migrated to the `Skill` tool description). Declared `prompt_cache_scope = "static"`.
  - `AgentsExtension` keeps the dynamic agent roster and ephemeral/conciseness notes but drops the "Delegation Guidelines" / "When (Not) to Delegate" prose (migrated to the `Agent` tool description). Declared `prompt_cache_scope = "static"`.
  - `TeamExtension` retains its "Team Coordination" framing; declared `prompt_cache_scope = "dynamic"`.
  - `WebSearchExtension.prompt()` now returns `None` — all guidance migrates to the `WebSearch` tool description.
  - `TasksExtension` declared `prompt_cache_scope = "dynamic"`.
- Every AgentKit tool's `description` is now loaded at import time from a frozen fixture under `src/langchain_agentkit/extensions/<ext>/descriptions/<tool_name>.md`. Each fixture begins with an HTML provenance comment declaring source and adaptation notes. Verbatim adoptions from the reference harness: `Read`, `Write`, `Edit`, `Glob`, `Grep`, `Skill`, `WebSearch`, `ask_user`. Reference text adapted for provider- and domain-neutrality: `Bash` (shell safety and tool-preference hints kept; git-commit and PR-creation protocols dropped), `Agent` (delegation rationale and prompt-writing guidance kept; SE-specific examples replaced with research-triage and document-analysis examples), `TaskCreate`/`TaskUpdate`/`TaskList`/`TaskGet`/`TaskStop`, `TeamCreate`/`TeamMessage`/`TeamStatus`/`TeamDissolve`.

### Added

- `AgentKit(preset="full", ...)` — batteries-included preset that prepends `CoreBehaviorExtension`, `TasksExtension`, and `MemoryExtension(path="~/.agents/memory")` ahead of any user-supplied `extensions=[...]`. `preset=None` (default) leaves the extension list untouched; unknown preset strings raise `ValueError`.
- `Extension.prompt_cache_scope: Literal["static", "dynamic"]` class attribute (default `"dynamic"`) that routes plain-string `prompt()` returns into the matching scope of `PromptComposition` for prompt-cache reuse.
- `CoreBehaviorExtension` (`langchain_agentkit.extensions.core_behavior`) — contributes domain-neutral universal agent guidance (static) plus an auto-detected `<env>` block (dynamic) covering cwd, git status, worktree, platform, shell, and OS version. Configurable via `cwd` callable and `include_env` toggle; no other config surface.
- `MemoryExtension` (`langchain_agentkit.extensions.memory`) — surfaces persistent memory content to the agent each turn. Defaults to a per-project layout at `~/.agents/memory/<sanitized-cwd>/MEMORY.md` via a ported `sanitize_path()` helper, with full escape hatches (`project_key_fn`, `loader`, `extra_sources`) for custom layouts. Applies line- and byte-caps, expands `~` and env vars on each call, and never performs filesystem I/O during construction.

### Removed

- **BREAKING:** `AgentKit.prompt(state, runtime)` helper. Use `AgentKit.compose(state, runtime).joined` instead.
- **BREAKING:** `BASE_AGENT_PROMPT` constant and `base_agent_prompt.md` file removed from `langchain_agentkit.extensions.tasks`. Universal agent guidance now lives in `CoreBehaviorExtension`. `TasksExtension.prompt()` now emits only task-specific guidance and the current task list. The task-management prompt wording has been made domain-neutral (no coding-specific examples).
- **BREAKING:** `ReminderConfig` and the `reminders=` parameter on `AgentKit.__init__` have been removed. The reminder channel is now a built-in capability of `compose()` — AGENTS.md discovery and today's date always fire, and extensions contribute by returning a `"reminder"` key from `prompt()`. `SkillsExtension.build_reminder_source()` is gone.
- **BREAKING:** `AgentKit.full(...)` classmethod has been removed. Use `AgentKit(preset="full", ...)` instead — the preset prepends `[CoreBehaviorExtension(), TasksExtension(), MemoryExtension(path="~/.agents/memory")]` and leaves all other optional extensions as explicit user-supplied entries.

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

## [0.12.0] — 2026-03-31

Version bump only — no code changes.

## [0.11.0] — 2026-03-31

### Added
- **Extension hook system**: new `Extension` base class with `before`/`after`/`wrap` decorators for model, tool, and run lifecycle events. `HookRunner` composes hooks onion-style and supports `jump_to` routing and `process_history` pipelines, wired end-to-end into graph execution.
- **Backends package**: 6-method `BackendProtocol` (Read, Write, Edit, Glob, Grep, Execute). `BaseSandbox` ABC implements file ops via shell so new providers only implement `execute()`. First provider is `DaytonaSandbox` (~75 lines), with a full integration example covering filesystem, skills, and agent discovery on a remote sandbox.
- **Permission system** with registration and per-call gates (allow/deny/ask), four presets (DEFAULT, READONLY, PERMISSIVE, STRICT), and HITL-driven approval prompts on `ask`.
- **Unified `Agent` tool** with shape-based discrimination (`{id}` for predefined, `{prompt}` for ephemeral) and a matching `AgentTeam` schema, plus `TeamAgent` (SocietyOfMind-style) that exposes a lead + teammates as a single `AgentLike`.
- **`AgentConfig`** carrying `tools`, `model`, `skills`, and `max_turns` parsed from frontmatter, accepted directly in `agents=` lists alongside compiled graphs. `AgentKit` auto-wires `model_resolver` and `skills_resolver` to sibling extensions.
- **Unified `Question` protocol** for `HITLExtension` with an opt-in `ask_user` tool that lets the LLM present structured choices during execution.
- **Router Node** for message-driven team coordination, teammate checkpointing via `InMemorySaver`, and automatic extension dependency resolution (`AgentTeamExtension` pulls in `TasksExtension` for a shared task list).
- New filesystem tool evals (Write, Edit, Glob, Grep, LS, MultiEdit) running against real temp directories.

### Changed
- **BREAKING**: `Middleware` renamed to `Extension` across the entire API — `*Middleware` classes, the `middleware=` parameter on `AgentKit` and the `agent` metaclass, and the `middleware/` package are all gone with no backward-compat shims.
- **BREAKING**: `agent` metaclass renames `llm` → `model`; `model` now accepts a `BaseChatModel` directly or a string resolved via `AgentKit.model_resolver`. Adds `skills` and `max_turns` class attributes.
- **BREAKING**: `SkillConfig.instructions` renamed to `.prompt` to align with `AgentConfig`; the `SKILL_NAME_PATTERN` alias has been removed.
- **BREAKING**: Filesystem stack collapsed onto the OS by default — `VirtualFilesystem`, `MemoryBackend`, and `CompositeBackend` are removed, `LocalBackend` is renamed to `OSBackend`, and `FilesystemExtension` defaults to `OSBackend(root=".")`. External backends (e.g. Daytona) plug in via `BackendProtocol`.
- **BREAKING**: `SpawnTeam` renamed to `AgentTeam`; web search and Qwant tools renamed to CamelCase (`WebSearch`, `QwantSearch`); agent graphs now expose `.name`/`.description` directly with the `agentkit_*` prefix removed from all consumers.
- Skills and agents now discover from the real filesystem via `pathlib`, with an optional `backend` parameter for remote discovery through `BackendProtocol`.
- Each extension (agents, filesystem, hitl, skills, tasks, teams, web_search) is now a cohesive package with dedicated modules for tools, logic, types, discovery, state, and prompts.
- Graph builder extracted to `_graph_builder.py`, prompt templates cached at module level (previously re-read from disk every LLM turn), tool factories use `functools.partial`, and delegation observability metadata has moved out of graph state into the tracing layer.
- README and examples rewritten to reflect the new `model` / `AgentConfig` / `SkillConfig` / `OSBackend` API; `CLAUDE.md` added with project commands and design decisions.

### Fixed
- **Qwant provider replaced with DuckDuckGo** as the default web search backend — Qwant's API was returning 403 for all programmatic requests. `DuckDuckGoSearchTool` is now the default; `QwantSearchTool` remains as a compatibility alias with a configurable `user_agent`.
- `TeamExtension` is now safe under concurrent tool calls via an `asyncio.Lock` around create/dissolve, a non-blocking lead queue drain, and a dedicated shutdown sentinel.
- `jump_to` routing moved from a shared mutable closure to per-invocation state, eliminating races across concurrent graph invocations.
- `OSBackend` hardened with explicit UTF-8 encoding on all I/O, streamed reads via `itertools.islice`, and rejection of ambiguous single-replacement edits when multiple matches exist.
- Delegation and extension error messages are now sanitized before reaching the LLM to avoid leaking internal exceptions.

### Removed
- 21 dead files from the hexagonal reorg: extension shims shadowed by package directories, orphaned `prompts/` duplicates, `tools/` shim modules, and the root `types.py` / `validate.py` shims. `SandboxProtocol` alias, redundant state re-exports, and `delegation_log` / `iteration_count` graph state fields also removed.

## [0.10.0] — 2026-03-25

### Added
- New examples showcasing the middleware layer: `filesystem`, `shared_filesystem`, `tasks`, and `web_search`.
- `Glob` tool now accepts a `path` parameter to scope searches to a specific directory.
- `Grep` tool gains aligned options: `output_mode` (`files_with_matches`, `content`, `count`), context lines, and `head_limit`.

### Changed
- **BREAKING**: `Grep` default `output_mode` is now `files_with_matches` (previously returned matching content). Callers relying on content output must pass `output_mode="content"` explicitly.
- `Read`, `Write`, and `Edit` tool descriptions for consistent agent behavior.
- Existing examples modernized to use the `agent` decorator with simplified state handling.
- Filesystem middleware and VFS expanded with broader test coverage.

[0.18.0]: https://github.com/rsmdt/langchain-agentkit/releases/tag/v0.18.0
[0.17.0]: https://github.com/rsmdt/langchain-agentkit/releases/tag/v0.17.0
[0.16.0]: https://github.com/rsmdt/langchain-agentkit/releases/tag/v0.16.0
[0.15.0]: https://github.com/rsmdt/langchain-agentkit/releases/tag/v0.15.0
[0.14.0]: https://github.com/rsmdt/langchain-agentkit/releases/tag/v0.14.0
[0.13.1]: https://github.com/rsmdt/langchain-agentkit/releases/tag/v0.13.1
[0.13.0]: https://github.com/rsmdt/langchain-agentkit/releases/tag/v0.13.0
[0.12.0]: https://github.com/rsmdt/langchain-agentkit/releases/tag/v0.12.0
[0.11.0]: https://github.com/rsmdt/langchain-agentkit/releases/tag/v0.11.0
[0.10.0]: https://github.com/rsmdt/langchain-agentkit/releases/tag/v0.10.0
[0.9.0]: https://github.com/rsmdt/langchain-agentkit/releases/tag/v0.9.0
[0.8.0]: https://github.com/rsmdt/langchain-agentkit/releases/tag/v0.8.0
[0.7.0]: https://github.com/rsmdt/langchain-agentkit/releases/tag/v0.7.0
[0.6.0]: https://github.com/rsmdt/langchain-agentkit/releases/tag/v0.6.0
[0.5.0]: https://github.com/rsmdt/langchain-agentkit/releases/tag/v0.5.0
[0.4.1]: https://github.com/rsmdt/langchain-agentkit/releases/tag/v0.4.1
[0.4.0]: https://github.com/rsmdt/langchain-agentkit/releases/tag/v0.4.0
[0.3.0]: https://github.com/rsmdt/langchain-agentkit/releases/tag/v0.3.0
[0.2.1]: https://github.com/rsmdt/langchain-agentkit/releases/tag/v0.2.1
[0.2.0]: https://github.com/rsmdt/langchain-agentkit/releases/tag/v0.2.0
