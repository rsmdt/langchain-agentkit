"""Evaluation framework for langchain-agentkit tool usage.

Uses trajectory matching to verify agents make correct tool calls.

Run evals::

    # Unit tests for eval framework (no LLM needed)
    pytest tests/evals/ -m "not eval" -v

    # Integration evals (require API key)
    OPENAI_API_KEY=... pytest tests/evals/ -m eval -v
"""
