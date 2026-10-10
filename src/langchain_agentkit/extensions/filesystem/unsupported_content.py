from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from langchain_core.messages import HumanMessage, ToolMessage

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_INPUT_FIELD_BY_BLOCK_TYPE: Final[Mapping[str, str]] = {
    "image": "image_inputs",
    "file": "pdf_inputs",
}

_TOOL_MESSAGE_FIELD_BY_BLOCK_TYPE: Final[Mapping[str, str]] = {
    "image": "image_tool_message",
    "file": "pdf_tool_message",
}

_PDF_MIME_TYPE: Final = "application/pdf"

_KIND_LABEL: Final[Mapping[str, str]] = {"image": "image", "file": "PDF"}


def _profile_accepts(
    block: Mapping[str, Any], profile: Mapping[str, Any], *, in_tool_message: bool
) -> bool:
    block_type = block.get("type")
    input_field = _INPUT_FIELD_BY_BLOCK_TYPE.get(str(block_type))
    if input_field is None:
        return True
    if block_type == "file" and block.get("mime_type") != _PDF_MIME_TYPE:
        return True
    if in_tool_message and profile.get(_TOOL_MESSAGE_FIELD_BY_BLOCK_TYPE[str(block_type)]) is False:
        return False
    return profile.get(input_field) is not False


def _placeholder(block: Mapping[str, Any], message: Any) -> dict[str, Any]:
    artifact = getattr(message, "artifact", None)
    file_path = artifact.get("filePath") if isinstance(artifact, dict) else None
    subject = file_path or "This attachment"
    kind = _KIND_LABEL[str(block["type"])]
    return {
        "type": "text",
        "text": f"[{subject} is not attached: the current model does not accept {kind} input.]",
    }


def _filter_message(message: Any, profile: Mapping[str, Any]) -> Any:
    in_tool_message = isinstance(message, ToolMessage)
    blocks = message.content_blocks
    replaced = [
        block
        if _profile_accepts(block, profile, in_tool_message=in_tool_message)
        else _placeholder(block, message)
        for block in blocks
    ]
    if all(new is old for new, old in zip(replaced, blocks, strict=True)):
        return message
    return message.model_copy(update={"content": replaced})


def strip_unsupported_content(
    messages: Sequence[Any], profile: Mapping[str, Any]
) -> tuple[list[Any], dict[int, Any]]:
    filtered: list[Any] = []
    originals: dict[int, Any] = {}
    for message in messages:
        if not isinstance(message, (HumanMessage, ToolMessage)) or isinstance(message.content, str):
            filtered.append(message)
            continue
        replacement = _filter_message(message, profile)
        if replacement is not message:
            originals[id(replacement)] = message
        filtered.append(replacement)
    return filtered, originals


def restore_original_messages(result: Any, originals: Mapping[int, Any]) -> None:
    messages = result.get("messages") if isinstance(result, dict) else None
    if not isinstance(messages, list):
        return
    for index, message in enumerate(messages):
        original = originals.get(id(message))
        if original is not None:
            messages[index] = original
