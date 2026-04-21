"""Prompt template extension — parameterized markdown commands.

Users drop markdown files in ``.agentkit/commands/*.md`` (or pass
:class:`PromptTemplate` objects directly) and invoke them by name through
the ``RunCommand`` tool or ``kit.templates.render(name, args)``.
"""

from langchain_agentkit.extensions.prompt_templates.extension import (
    PromptTemplateExtension,
)
from langchain_agentkit.extensions.prompt_templates.parser import (
    ParsedArgs,
    parse_args,
)
from langchain_agentkit.extensions.prompt_templates.render import (
    expand_template,
)
from langchain_agentkit.extensions.prompt_templates.types import (
    PromptTemplate,
    PromptTemplateError,
)

__all__ = [
    "ParsedArgs",
    "PromptTemplate",
    "PromptTemplateError",
    "PromptTemplateExtension",
    "expand_template",
    "parse_args",
]
