## Skills

Your agent has a skill library — domain tools that provide structured workflows and reference materials.

**Available:**

{skills_list}

**Usage:**

Skills use progressive disclosure. You see names and descriptions above. Load full instructions only when needed:

1. Match the task to a skill description
2. `Skill("name")` — loads step-by-step instructions, quality criteria, and methodology
3. Follow the workflow inside
4. `Read("/skills/name/reference-file.md")` — read templates, examples, or supporting docs from the virtual filesystem

**When to use skills:**
- User's request matches a skill's domain
- You need specialized knowledge or structured workflows
- A skill provides proven patterns for complex tasks

**Skip skills when:**
- Simple questions or conversation
- Skill already loaded this session
- No skill matches the task
