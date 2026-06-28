"""CLI utilities for TTY detection, output handling, and progress indicators.

Provides Unix-friendly utilities that respect pipeline patterns.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import IO, Any

from tqdm import tqdm


def is_tty(file: IO = sys.stdout) -> bool:
    """Check if file descriptor is a TTY (terminal).

    Args:
        file: File object to check (default: stdout)

    Returns:
        True if file is a TTY, False if piped/redirected

    Example:
        >>> is_tty(sys.stdout)  # True in terminal
        >>> is_tty(sys.stdout)  # False when piped: armature discover log.xes | jq
    """
    return file.isatty() if hasattr(file, "isatty") else False


def write_output(content: str, output_path: Path | None, format: str = "yaml") -> None:
    """Write content to file or stdout based on path parameter.

    Args:
        content: String content to write
        output_path: File path to write to, or None for stdout
        format: Format type for validation (yaml or json)

    Raises:
        ValueError: If format is invalid

    Example:
        >>> write_output(yaml_str, None, "yaml")  # Prints to stdout
        >>> write_output(yaml_str, Path("out.yaml"), "yaml")  # Writes to file
    """
    if format not in ("yaml", "json"):
        raise ValueError(f"Invalid format: {format}")

    if output_path is None:
        # Write to stdout
        print(content)
    else:
        # Write to file
        output_path.write_text(content, encoding="utf-8")


def show_progress(
    description: str,
    iterable: Iterable | None = None,
    total: int | None = None,
    file: IO = sys.stderr,
) -> tqdm | Iterable:
    """Create progress bar that respects TTY detection.

    Only shows progress bar if stderr is a TTY. When piped, returns silent
    iterator to avoid polluting output streams.

    Args:
        description: Progress bar description
        iterable: Optional iterable to wrap
        total: Optional total count for manual updates
        file: File to write progress to (default: stderr to avoid stdout pollution)

    Returns:
        tqdm progress bar if TTY, else silent iterator/context

    Example:
        >>> for item in show_progress("Processing", items):
        ...     process(item)

        >>> with show_progress("Loading", total=100) as pbar:
        ...     for i in range(100):
        ...         pbar.update(1)
    """
    # Disable progress bar if not TTY
    disabled = not is_tty(file)

    if iterable is not None:
        return tqdm(
            iterable,
            desc=description,
            disable=disabled,
            file=file,
        )
    else:
        return tqdm(
            total=total,
            desc=description,
            disable=disabled,
            file=file,
        )


def format_matrix_as_json(matrix: Any) -> str:
    """Format Matrix as JSON string.

    Converts Matrix object to JSON representation. Handles Pydantic models
    using model_dump with JSON mode for enum serialization.

    Args:
        matrix: Matrix object to serialize

    Returns:
        JSON string with 2-space indentation
    """
    # Use Pydantic's model_dump for JSON-compatible dict
    data = matrix.model_dump(mode="json", exclude_none=True, by_alias=False)
    return json.dumps(data, indent=2)
