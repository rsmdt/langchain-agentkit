"""ReplaceMessages — bulk message replacement for the add_messages reducer.

Wraps LangGraph's built-in ``REMOVE_ALL_MESSAGES`` sentinel so callers
don't need to know about the underlying mechanism.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import RemoveMessage


class ReplaceMessages(list[Any]):
    """Drop-in list replacement for the ``messages`` return value.

    When passed through the ``add_messages`` reducer, this clears all
    existing messages and replaces them with the new list.

    Uses LangGraph's built-in ``RemoveMessage(id=REMOVE_ALL_MESSAGES)``
    sentinel internally.

    Example::

        return {"messages": ReplaceMessages(kept_messages)}
    """

    def __init__(self, messages: list[Any]) -> None:
        # RemoveMessage("__remove_all__") tells add_messages to clear
        # everything before it — then the remaining items are appended.
        super().__init__(
            [RemoveMessage(id="__remove_all__"), *messages],
        )
