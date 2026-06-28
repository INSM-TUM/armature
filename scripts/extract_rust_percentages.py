#!/usr/bin/env python3.12
"""Extract Rust percentage outputs for comparison with Python implementation."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


def extract_rust_percentages(log_name: str) -> dict | None:
    """Extract Rust percentage outputs for test log.

    Args:
        log_name: Log name (e.g., "Log06", "Log08", "Log10", "Log12")

    Returns:
        Dict with 9 percentage fields or None if Rust binary unavailable

    Raises:
        FileNotFoundError: If .xes file not found for log_name
        RuntimeError: If Rust binary execution fails
    """
    rust_binary = Path("/tmp/automated-process-classification/target/release/matrix_classifier")
    if not rust_binary.exists():
        print(f"Rust binary not found: {rust_binary}", file=sys.stderr)
        return None

    # Find .xes file in Test Data/Classification/
    test_data = Path("/home/choky/kerstin/armature/Test Data/Classification")
    xes_files = list(test_data.rglob(f"{log_name}*.xes"))
    if not xes_files:
        raise FileNotFoundError(f"No .xes file found for {log_name}")

    xes_path = xes_files[0]

    # Run Rust binary with --print-ratios flag
    result = subprocess.run(
        [str(rust_binary), "-f", str(xes_path), "--print-ratios"],
        capture_output=True,
        text=True,
        timeout=30,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Rust binary failed: {result.stderr}")

    # Parse output for CalculatedPercentages
    # Expected format: "CalculatedPercentages { none_none: 0.123, ... }"
    percentages_line = None
    for line in result.stdout.splitlines():
        if "CalculatedPercentages {" in line:
            percentages_line = line
            break

    if not percentages_line:
        raise RuntimeError("CalculatedPercentages not found in Rust output")

    # Extract percentages using regex
    percentages = {}
    field_pattern = r"(\w+):\s*([\d.]+)"
    matches = re.findall(field_pattern, percentages_line)

    for field_name, value in matches:
        percentages[field_name] = float(value)

    return percentages


if __name__ == "__main__":
    log = sys.argv[1] if len(sys.argv) > 1 else "Log06"
    pct = extract_rust_percentages(log)
    if pct:
        print(json.dumps(pct, indent=2))
    else:
        print("Rust binary not available", file=sys.stderr)
        sys.exit(1)
