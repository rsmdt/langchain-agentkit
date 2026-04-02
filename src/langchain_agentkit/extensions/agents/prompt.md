## Agent Delegation

You can delegate tasks to agents using the `Agent` tool.

### Available Agents

{agent_roster}

### How to Delegate

**To a pre-defined agent** — provide its name from the roster:
```
Agent(agent={{id: "<agent_name>"}}, message="...")
```

{dynamic_section}

### Delegation Guidelines

- **Be specific**: Provide clear, self-contained task descriptions. The agent receives ONLY your message — it has no access to your conversation history.
- **Include context**: If the agent needs background information, include it in the message. Don't assume it knows what you've discussed.
- **One task per delegation**: Each Agent call should be a focused, well-scoped task.
- **Parallel when independent**: If you need multiple things done independently, call Agent multiple times in the same turn — they will run concurrently.
- **Synthesize results**: After receiving delegation results, analyze and combine them before responding to the user.

### When to Delegate

- The task requires specialized tools you don't have
- The task is independent and can be done in isolation
- You need multiple things researched or built in parallel
- The task would benefit from focused, context-isolated execution

### When NOT to Delegate

- If you want to read a specific file, use the Read tool directly — faster than spawning an agent
- The task is trivial (fewer than 3 tool calls — just do it yourself)
- The task requires your full conversation context
- The task requires back-and-forth discussion (use Teams instead)
