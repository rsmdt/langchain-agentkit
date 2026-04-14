"""Read tool — type-aware file reading with dedup caching."""

from __future__ import annotations

import base64
import json
import os
from typing import TYPE_CHECKING, Any

from langchain_core.tools import StructuredTool, ToolException

from langchain_agentkit.extensions.filesystem.tools.common import (
    _IMAGE_MIME_MAP,
    BINARY_EXTENSIONS,
    FILE_UNCHANGED_STUB,
    IMAGE_EXTENSIONS,
    NOTEBOOK_EXTENSIONS,
    PDF_EXTENSIONS,
    _format_file_size,
    _ReadInput,
)

_READ_DESCRIPTION = """Reads a file from the local filesystem. You can access any file directly by using this tool.
Assume this tool is able to read all files on the machine. If the User provides a path to a file assume that path is valid. It is okay to read a file that does not exist; an error will be returned.

Usage:
- The file_path parameter must be an absolute path, not a relative path
- By default, it reads up to 2000 lines starting from the beginning of the file
- When you already know which part of the file you need, only read that part. This can be important for larger files.
- Results are returned using cat -n format, with line numbers starting at 1
- This tool allows the agent to read images (eg PNG, JPG, etc). When reading an image file the contents are presented visually as the agent is multimodal.
- This tool can read PDF files (.pdf). For large PDFs (more than 10 pages), you MUST provide the pages parameter to read specific page ranges (e.g., pages: "1-5"). Reading a large PDF without the pages parameter will fail. Maximum 20 pages per request.
- This tool can read Jupyter notebooks (.ipynb files) and returns all cells with their outputs, combining code, text, and visualizations.
- This tool can only read files, not directories. To read a directory, use an ls command via the Bash tool.
- You will regularly be asked to read screenshots. If the user provides a path to a screenshot, ALWAYS use this tool to view the file at the path. This tool will work with all temporary file paths.
- If you read a file that exists but has empty contents you will receive a system reminder warning in place of file contents."""

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool


# ---------------------------------------------------------------------------
# Notebook
# ---------------------------------------------------------------------------


async def _read_notebook(backend: Any, file_path: str) -> tuple[str, dict[str, Any]]:
    """Read a Jupyter notebook and return formatted cell contents."""
    data = await backend.read_bytes(file_path)
    notebook = json.loads(data.decode("utf-8"))
    cells = notebook.get("cells", [])
    parts: list[str] = []
    structured_cells: list[dict[str, Any]] = []
    for i, cell in enumerate(cells):
        cell_type = cell.get("cell_type", "unknown")
        source_lines = cell.get("source", [])
        source = "".join(source_lines) if isinstance(source_lines, list) else source_lines
        parts.append(f"\n--- Cell {i + 1} ({cell_type}) ---")
        parts.append(source)
        cell_outputs: list[str] = []
        for output in cell.get("outputs", []):
            _format_notebook_output(output, parts, cell_outputs)
        structured_cells.append(
            {
                "cell_type": cell_type,
                "source": source,
                "outputs": cell_outputs,
            }
        )
    content = "\n".join(parts)
    return content, {
        "type": "notebook",
        "filePath": file_path,
        "cells": structured_cells,
    }


def _format_notebook_output(
    output: dict[str, Any],
    parts: list[str],
    cell_outputs: list[str],
) -> None:
    """Format a single notebook cell output and append to parts."""
    output_type = output.get("output_type", "")
    if output_type == "stream":
        text = "".join(output.get("text", []))
        parts.append(f"[Output]\n{text}")
        cell_outputs.append(text)
    elif output_type in ("execute_result", "display_data"):
        text_data = output.get("data", {}).get("text/plain", [])
        text = "".join(text_data) if isinstance(text_data, list) else text_data
        if text:
            parts.append(f"[Output]\n{text}")
            cell_outputs.append(text)
    elif output_type == "error":
        ename = output.get("ename", "Error")
        evalue = output.get("evalue", "")
        parts.append(f"[Error: {ename}] {evalue}")
        cell_outputs.append(f"{ename}: {evalue}")


# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------


def _detect_image_dimensions(data: bytes) -> dict[str, int] | None:
    """Detect image dimensions. Returns dict with width/height or None."""
    try:
        import io

        from PIL import Image as PILImage  # type: ignore[import-not-found]

        img = PILImage.open(io.BytesIO(data))
        w, h = img.size
        img.close()
        return {"originalWidth": w, "originalHeight": h}
    except Exception:  # noqa: BLE001
        return None


async def _read_image(backend: Any, file_path: str, ext: str) -> tuple[str, dict[str, Any]]:
    """Read an image file and return as multimodal content block."""
    data = await backend.read_bytes(file_path)
    mime_type = _IMAGE_MIME_MAP.get(ext, "application/octet-stream")
    encoded = base64.b64encode(data).decode("ascii")

    content = json.dumps(
        [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "data": encoded,
                    "media_type": mime_type,
                },
            }
        ]
    )

    dimensions = _detect_image_dimensions(data)
    artifact: dict[str, Any] = {
        "type": "image",
        "filePath": file_path,
        "base64": encoded,
        "mediaType": mime_type,
        "originalSize": len(data),
    }
    if dimensions:
        artifact["dimensions"] = dimensions
    return content, artifact


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------


def _parse_page_range(pages: str) -> tuple[int, int]:
    """Parse a page range string like '1-5', '3', '10-20'. Returns (start, end) 0-indexed."""
    pages = pages.strip()
    if "-" in pages:
        parts = pages.split("-", 1)
        start = int(parts[0]) - 1
        end = int(parts[1])
    else:
        start = int(pages) - 1
        end = int(pages)
    return max(0, start), end


async def _read_pdf(
    backend: Any,
    file_path: str,
    pages: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Read a PDF file and return base64 content, optionally extracting pages."""
    data = await backend.read_bytes(file_path)

    if pages is not None:
        return _read_pdf_pages(data, file_path, pages)

    encoded = base64.b64encode(data).decode("ascii")
    size_str = _format_file_size(len(data))
    content = f"PDF file read: {file_path} ({size_str})"
    artifact: dict[str, Any] = {
        "type": "pdf",
        "filePath": file_path,
        "base64": encoded,
        "originalSize": len(data),
    }
    return content, artifact


def _read_pdf_pages(
    data: bytes,
    file_path: str,
    pages: str,
) -> tuple[str, dict[str, Any]]:
    """Extract specific pages from a PDF. Requires pymupdf."""
    try:
        import fitz  # type: ignore[import-not-found]  # pymupdf (optional)
    except ImportError:
        encoded = base64.b64encode(data).decode("ascii")
        size_str = _format_file_size(len(data))
        content = (
            f"PDF file read: {file_path} ({size_str}). "
            f"Page extraction requires 'pip install pymupdf'."
        )
        return content, {
            "type": "pdf",
            "filePath": file_path,
            "base64": encoded,
            "originalSize": len(data),
        }

    start, end = _parse_page_range(pages)
    doc = fitz.open(stream=data, filetype="pdf")
    page_count = min(end, len(doc)) - start
    page_texts: list[str] = []
    for i in range(start, min(end, len(doc))):
        page_texts.append(f"--- Page {i + 1} ---\n{doc[i].get_text()}")
    doc.close()

    size_str = _format_file_size(len(data))
    content = (
        f"PDF pages extracted: {page_count} page(s) from {file_path} ({size_str})\n"
        + "\n".join(page_texts)
    )
    return content, {
        "type": "parts",
        "filePath": file_path,
        "originalSize": len(data),
        "count": page_count,
    }


# ---------------------------------------------------------------------------
# Text (with dedup caching)
# ---------------------------------------------------------------------------


async def _get_content_hash(backend: Any, file_path: str) -> str:
    """Get a content hash for dedup. Uses file content since mtime isn't in protocol."""
    import hashlib

    try:
        data = await backend.read_bytes(file_path)
        return hashlib.md5(data).hexdigest()  # noqa: S324
    except (FileNotFoundError, OSError):
        return ""


def _check_file_unchanged(
    read_state: dict[str, str],
    file_path: str,
    offset: int,
    limit: int,
    content_hash: str,
) -> bool:
    """Check if a file is unchanged since last read with same offset/limit."""
    cache_key = f"{file_path}:{offset}:{limit}"
    if cache_key not in read_state:
        return False
    return bool(content_hash and content_hash == read_state[cache_key])


def _add_line_numbers(text: str, offset: int) -> str:
    """Add line-number prefixes for LLM display (tool-level presentation)."""
    lines = text.splitlines(keepends=True)
    return "".join(f"{offset + i + 1}\t{line}" for i, line in enumerate(lines))


async def _read_text(
    backend: Any,
    file_path: str,
    offset: int,
    limit: int,
    read_state: dict[str, str],
) -> tuple[str, dict[str, Any]]:
    """Read a text file with line numbers."""
    content_hash = await _get_content_hash(backend, file_path)

    if _check_file_unchanged(read_state, file_path, offset, limit, content_hash):
        return FILE_UNCHANGED_STUB, {
            "type": "file_unchanged",
            "filePath": file_path,
        }

    text = await backend.read(file_path, offset=offset, limit=limit)
    num_lines = len(text.splitlines()) if text else 0

    if not text or offset > 0:
        total_text = await backend.read(file_path, limit=100_000)
        total_lines = len(total_text.splitlines()) if total_text else 0
    else:
        total_lines = num_lines

    if not text:
        if total_lines == 0:
            content = (
                "<system-reminder>Warning: the file exists but "
                "the contents are empty.</system-reminder>"
            )
        else:
            content = (
                f"<system-reminder>Warning: the file exists but is shorter "
                f"than the provided offset ({offset + 1}). "
                f"The file has {total_lines} lines.</system-reminder>"
            )
    else:
        content = _add_line_numbers(text, offset)

    cache_key = f"{file_path}:{offset}:{limit}"
    if content_hash:
        read_state[cache_key] = content_hash

    artifact: dict[str, Any] = {
        "type": "text",
        "filePath": file_path,
        "content": text,
        "numLines": num_lines,
        "startLine": offset + 1,
        "totalLines": total_lines,
    }
    return content, artifact


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def _build_read(backend: Any, read_state: dict[str, str]) -> BaseTool:
    async def read(
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
        pages: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Read a file with type-aware dispatch."""
        ext = os.path.splitext(file_path)[1].lower()

        if ext in NOTEBOOK_EXTENSIONS:
            return await _read_notebook(backend, file_path)

        if ext in BINARY_EXTENSIONS:
            if ext in IMAGE_EXTENSIONS:
                return await _read_image(backend, file_path, ext)
            if ext in PDF_EXTENSIONS:
                return await _read_pdf(backend, file_path, pages=pages)
            raise ToolException(
                f"Cannot read binary {ext} file. Use appropriate tools for binary file analysis."
            )

        return await _read_text(backend, file_path, offset, limit, read_state)

    return StructuredTool.from_function(
        coroutine=read,
        name="Read",
        description=_READ_DESCRIPTION,
        args_schema=_ReadInput,
        response_format="content_and_artifact",
        handle_tool_error=True,
    )
