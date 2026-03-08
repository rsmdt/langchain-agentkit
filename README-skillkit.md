# langchain-skillkit

> **This package has been renamed to [`langchain-agentkit`](https://pypi.org/project/langchain-agentkit/).**

## Migration

```bash
pip install langchain-agentkit
```

```python
# Before
from langchain_skillkit import node, SkillKit, AgentState

# After
from langchain_agentkit import node, SkillKit, AgentState
```

The old import path (`from langchain_skillkit import ...`) still works via a compatibility shim that emits a `DeprecationWarning`. It will be removed in a future release.

For full documentation, see [langchain-agentkit](https://pypi.org/project/langchain-agentkit/).
