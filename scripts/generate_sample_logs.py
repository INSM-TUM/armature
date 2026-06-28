#!/usr/bin/env python3
"""Write a set of synthetic XES logs to disk for use with compare_log_dependencies.py.

Usage:
    python scripts/generate_sample_logs.py [--output-dir DIR]

Default output directory: /tmp/armature_sample_logs/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.compute_synthetic_ratios import build_xes

SAMPLE_LOGS: list[tuple[str, list[tuple[list[str], int]]]] = [
    (
        "S1_strict_sequence",
        [
            (["a", "b", "c", "d", "e"], 50),
        ],
    ),
    (
        "S2_parallel_block",
        [
            (["a", "b", "c", "d", "e"], 25),
            (["a", "c", "b", "d", "e"], 25),
        ],
    ),
    (
        "S3_xor_two_paths",
        [
            (["a", "b", "c", "e"], 25),
            (["a", "b", "d", "e"], 25),
        ],
    ),
    (
        "S5_sese_loop",
        [
            (["a", "b", "d"], 20),
            (["a", "b", "c", "b", "d"], 20),
            (["a", "b", "c", "b", "c", "b", "d"], 10),
        ],
    ),
    (
        "U1_unstructured_3act",
        [
            (["a", "b", "c"], 10),
            (["a", "c", "b"], 10),
            (["b", "a", "c"], 10),
            (["b", "c", "a"], 10),
            (["c", "a", "b"], 10),
            (["c", "b", "a"], 10),
        ],
    ),
    (
        "U2_unstructured_5act",
        [
            ([a, b, c, d, e], 1)
            for a in ["a", "b", "c", "d", "e"]
            for b in ["a", "b", "c", "d", "e"]
            if b != a
            for c in ["a", "b", "c", "d", "e"]
            if c not in (a, b)
            for d in ["a", "b", "c", "d", "e"]
            if d not in (a, b, c)
            for e in ["a", "b", "c", "d", "e"]
            if e not in (a, b, c, d)
        ],
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic XES sample logs.")
    parser.add_argument(
        "--output-dir",
        default="/tmp/armature_sample_logs",
        help="Directory to write XES files into (default: /tmp/armature_sample_logs)",
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name, variants in SAMPLE_LOGS:
        path = out / f"{name}.xes"
        path.write_text(build_xes(variants))
        n_traces = sum(c for _, c in variants)
        print(f"  wrote {path}  ({n_traces} traces, {len(variants)} variants)")

    print(f"\nRun comparison:")
    print(f"  python scripts/compare_log_dependencies.py {out}/*.xes --no-zeros --format markdown")


if __name__ == "__main__":
    main()
