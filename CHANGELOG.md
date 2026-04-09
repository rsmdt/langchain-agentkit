# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Entries are added only when a release is cut. Work in progress is not tracked here — see the git history between the latest tag and `HEAD` for unreleased changes.

This file retains detailed entries for the last 10 minor releases plus their patch revisions. Older release notes can be found in the git history and on each version's [GitHub release page](https://github.com/rsmdt/langchain-agentkit/releases).

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
- **Structured output on all file tools** via `content_and_artifact`, aligning `Read`, `Write`, `Edit`, `Glob`, `Grep`, and `Bash` input schemas, output messages, and behavior with the Claude Code reference (file-unchanged dedup, structured patches, quote normalization, context precedence, pagination, multiline/type filters, etc.).
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
- **Backends package**: 6-method `BackendProtocol` (Read, Write, Edit, Glob, Grep, Execute) mirroring the Claude Code tool surface. `BaseSandbox` ABC implements file ops via shell so new providers only implement `execute()`. First provider is `DaytonaSandbox` (~75 lines), with a full integration example covering filesystem, skills, and agent discovery on a remote sandbox.
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
- `Grep` tool gains Claude Code-aligned options: `output_mode` (`files_with_matches`, `content`, `count`), context lines, and `head_limit`.

### Changed
- **BREAKING**: `Grep` default `output_mode` is now `files_with_matches` (previously returned matching content). Callers relying on content output must pass `output_mode="content"` explicitly.
- `Read`, `Write`, and `Edit` tool descriptions realigned with the Claude Code tool spec for consistent agent behavior.
- Existing examples modernized to use the `agent` decorator with simplified state handling.
- Filesystem middleware and VFS expanded with broader test coverage.

## [0.9.0] — 2026-03-25

### Added
- New `VirtualFilesystem` with Claude Code-aligned file tools (`Read`, `Write`, `Edit`, `Glob`, `Grep`) supporting POSIX paths, unicode, glob wildcards, and regex search. Exposed via the new `FilesystemMiddleware` for standalone use.
- Trajectory-based eval framework for tool usage with 4 match modes (strict, unordered, subset, superset) and 3 tool-args modes. Ships with 12 datasets and 8 LLM integration evals (`pytest -m eval`) covering filesystem, task, and skill workflows. `.env` is auto-loaded for API keys, and an `eval` optional dependency group is available.
- Task tools gain `owner`, `metadata`, and `blocks` fields, plus a new `TaskStop` tool for halting in-progress tasks. `TaskUpdate` supports owner/metadata merges (with null-delete) and `add_blocks`; `TaskGet` returns computed `blocks` as the reverse of `blocked_by`. Parallel `Command` updates are safely merged via a new `_merge_tasks` reducer (union-dedup for lists, deep-merge for metadata).
- Task management prompt rewritten with structured decision criteria, decomposition heuristics (prefer 3-7 tasks), lifecycle rules, and concrete worked examples.

### Changed
- **BREAKING**: State is now composed progressively from middleware. `AgentState` is split into `AgentKitState` (messages + sender) and `TasksState` (tasks with reducer). Middleware declares state requirements via a `state_schema` property, and `AgentKit.state_schema` composes `TypedDict`s dynamically. The `agent` metaclass uses the composed schema when no handler annotation is provided.
- **BREAKING**: `SkillsMiddleware` is decoupled from the filesystem. `SkillsMiddleware(skills="path/")` owns its own VFS and includes file tools; pass an existing `filesystem=vfs` to expose only the `Skill` tool. Skills are loaded from the real filesystem into the VFS at `/skills/{name}/`, and `SkillRead` has been removed in favor of `Read("/skills/name/file")`. Reference files are now listed in the available-skills XML.
- Source layout reorganized into layered `tools/`, `middleware/`, `vfs/`, and `prompts/` packages, with tests mirroring the new structure under `tests/unit/` and `tests/evals/`. Public API is unchanged — all existing imports continue to work via `__init__.py` re-exports.

### Removed
- **BREAKING**: `SkillRead` tool removed. Use `Read` against the `/skills/{name}/` VFS path instead.

## [0.8.0] — 2026-03-16

### Changed
- **BREAKING**: Simplified the `Task` type by removing `owner`, `blocks`, `metadata`, and `created_at` fields. Tasks are ephemeral graph state, not database records — these fields added complexity without concrete use cases. The `blocked_by` field is now optional and only present when a task has actual dependencies.

### Added
- Exported `Task` (TypedDict) and `TaskStatus` (Literal) from the public API for use in type annotations and handlers.

## [0.7.0] — 2026-03-15

### Added
- New `WebSearchMiddleware` with multi-provider fan-out via `asyncio.gather`, shipping with a zero-config `QwantSearchTool` default (no API key required) and accepting any `BaseTool` or callable as a provider.
- `SkillRegistry` and `SkillsMiddleware` now accept `Path` objects (or lists mixing `str` and `Path`) for `skills_dirs`, matching the common `Path(__file__).parent / "skills"` pattern.

### Changed
- **BREAKING**: The `runtime` parameter in the `Middleware` protocol is now optional and defaults to `None` across the protocol, `AgentKit`, and all built-in middleware implementations, supporting callers without a `ToolRuntime` (e.g. prompt composition outside `ToolNode`). Custom middleware implementations should update their signatures accordingly.

### Fixed
- `TaskCreate` and `TaskUpdate` now return a proper `ToolMessage` alongside their `Command` update, resolving LangGraph runtime errors caused by `ToolNode` requiring every tool call to produce a corresponding `ToolMessage`.

## [0.6.0] — 2026-03-08

### Changed
- **BREAKING**: Migrated tool runtime to `langgraph.prebuilt.ToolRuntime`. Custom tool execution paths have been replaced with the upstream LangGraph primitive, which may affect code depending on the previous internal runtime behavior.
- **BREAKING**: Removed the `node` metaclass. Code relying on this construct must be updated to use the current graph-building APIs.

### Removed
- Dead code paths eliminated as part of the runtime migration, reducing surface area and simplifying the extension framework internals.

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
