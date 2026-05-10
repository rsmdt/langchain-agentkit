# CLAUDE.md

## Commands

```bash
uv run pytest tests/unit/ -q                          # Unit tests only
uv run pytest tests/integration/ -v                   # Integration tests
uv run pytest tests/evals/test_eval_runner.py -q      # Eval dataset validation
uv run pytest tests/evals/ -m eval -v                 # LLM evals (needs OPENAI_API_KEY)
uv run ruff check src/ tests/                         # Lint
uv run ruff format --check src/ tests/                # Format check (must match CI)
uv run mypy src/                                      # Type check
```

## Architecture

Composable extension framework for LangGraph agents. Python 3.12+, src layout (`src/langchain_agentkit/`). See [`docs/architecture.md`](docs/architecture.md) for the full design.

**Invariants** — must hold for every change:

- **Handler owns `llm.bind_tools()`.** The framework injects raw `llm` and a composed `tools` list and never binds. This preserves provider-specific kwargs (`strict`, `parallel_tool_calls`, `tool_choice`) and dynamic per-step tool filtering.
- **`kit.tools` dedupes by name, user tools first.** Don't shadow extension tools accidentally; name collisions resolve to the user-provided tool.
- **`kit.compile(handler)` is the primary graph-building path.** Manual wiring via `kit.tools`, `kit.prompt()`, `kit.model`, `kit.hooks` is supported, but hook ordering and attachment become the caller's responsibility.

## Release

See [`docs/release.md`](docs/release.md) for the changelog + tag flow.
