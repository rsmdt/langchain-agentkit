# CLAUDE.md

## Commands

```bash
uv run pytest tests/              # All tests (unit + evals)
uv run pytest tests/unit/ -q      # Unit tests only
uv run pytest tests/evals/ -m eval -v  # LLM evals only (needs OPENAI_API_KEY)
uv run ruff check src/ tests/     # Lint
uv run mypy src/                  # Type check
```

## Tests

- **Unit tests** (`tests/unit/`) — fast, no external deps, ~600 tests in <1s
- **Evals** (`tests/evals/`) — LLM integration evals that make real API calls. Require `OPENAI_API_KEY` in `.env` or environment. Marked with `@pytest.mark.eval`. ~25 evals, ~4 min.
- Run everything with `pytest tests/` — no need to split runs.

## Project

Composable extension framework for LangGraph agents. Python 3.11+, src layout (`src/langchain_agentkit/`).

### Key design decisions

- `llm.bind_tools()` is called **per-step** intentionally (not cached at build time) to support dynamic tool binding in handlers.
- `kit.prompt()` is called **per-step** intentionally — extension prompts render current state (task list, team status).
- The `agent` metaclass returns a `StateGraph`, not a class. This is deliberate.
