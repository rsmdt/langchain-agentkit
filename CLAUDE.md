# CLAUDE.md

## Commands

```bash
uv run pytest tests/unit/ -q                          # Unit tests only
uv run pytest tests/integration/ -v                   # Integration tests (optional: DAYTONA_API_URL)
uv run pytest tests/evals/test_eval_runner.py -q      # Eval dataset validation
uv run pytest tests/evals/ -m eval -v                 # LLM evals (needs OPENAI_API_KEY)
uv run ruff check src/ tests/                         # Lint
uv run ruff format --check src/ tests/                # Format check (must match CI)
uv run mypy src/                                      # Type check
```

## Tests

- **Unit tests** (`tests/unit/`) — fast, no external deps, no network calls.
- **Integration tests** (`tests/integration/`) — BackendProtocol conformance matrix. Runs against OSBackend always; also against DaytonaBackend when `DAYTONA_API_URL` is set and `daytona-sdk` is installed.
- **Evals** (`tests/evals/`) — two categories:
  - `test_eval_runner.py` — dataset format validation (no API calls, runs in <1s, not marked `@pytest.mark.eval`).
  - Everything else — LLM integration evals that make real API calls. Require `OPENAI_API_KEY` in `.env` or environment. Marked with `@pytest.mark.eval`.
  - **Important**: `pytest tests/evals/ -m eval` selects only the LLM evals and skips the dataset validation tests. Run both commands for full eval coverage, or `pytest tests/evals/ -v` to run everything at once.

## Project

Composable extension framework for LangGraph agents. Python 3.11+, src layout (`src/langchain_agentkit/`).

### Key design decisions

- `llm.bind_tools()` is called **per-step** intentionally (not cached at build time) to support dynamic tool binding in handlers.
- `kit.prompt()` is called **per-step** intentionally — extension prompts render current state (task list, team status).
- The `agent` metaclass returns a `StateGraph`, not a class. This is deliberate.
