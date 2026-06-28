#!/usr/bin/env python3
"""Compare activity dependency distributions across multiple XES logs.

For each log, counts every directed (A, B) pair where A != B and tallies
the temporal and existential dependency type assigned by discovery.
Frequencies are normalized by n*(n-1) — the total number of directed pairs
for a log with n activities.

Output: table with columns = logs, rows = dependency types
(temporal and existential in separate sections).

Usage:
    python scripts/compare_log_dependencies.py log1.xes log2.xes [...]
    python scripts/compare_log_dependencies.py logs/*.xes --format csv
    python scripts/compare_log_dependencies.py logs/*.xes --no-zeros --format markdown
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from itertools import permutations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from armature.core.dependencies import ExistentialDependency, TemporalDependency
from armature.discovery.discover import discover


TEMPORAL_ORDER = list(TemporalDependency)
EXISTENTIAL_ORDER = list(ExistentialDependency)


def count_dependencies(matrix) -> dict[tuple[str, str], float]:
    """Return normalized frequency for each dependency type in a matrix.

    Enumerates all directed (A, B) pairs where A != B (n*(n-1) total).
    Pairs absent from the sparse matrix default to NO_ORDERING / INDEPENDENCE.
    Each count is divided by n*(n-1) so all temporal values sum to 1.0
    and all existential values sum to 1.0.
    """
    activities = matrix.activities
    n = len(activities)
    total_pairs = n * (n - 1)
    if total_pairs == 0:
        return {}

    temporal_counts: dict[TemporalDependency, int] = defaultdict(int)
    existential_counts: dict[ExistentialDependency, int] = defaultdict(int)

    stored = matrix.dependencies

    for a, b in permutations(activities, 2):
        cell = stored.get(a, {}).get(b)
        if cell is not None:
            temporal_counts[cell.temporal] += 1
            existential_counts[cell.existential] += 1
        else:
            temporal_counts[TemporalDependency.NO_ORDERING] += 1
            existential_counts[ExistentialDependency.INDEPENDENCE] += 1

    result: dict[tuple[str, str], float] = {}
    for t in TEMPORAL_ORDER:
        result[("temporal", t.value)] = temporal_counts[t] / total_pairs
    for e in EXISTENTIAL_ORDER:
        result[("existential", e.value)] = existential_counts[e] / total_pairs

    return result


def build_table(log_paths: list[str], decimals: int = 4) -> pd.DataFrame:
    data: dict[str, dict[tuple[str, str], float]] = {}
    names: list[str] = []
    for path in log_paths:
        name = Path(path).name
        print(f"Processing {name}...", file=sys.stderr)
        matrix = discover(Path(path))
        data[name] = count_dependencies(matrix)
        names.append(name)

    index = pd.MultiIndex.from_tuples(
        [("temporal", t.value) for t in TEMPORAL_ORDER]
        + [("existential", e.value) for e in EXISTENTIAL_ORDER],
        names=["kind", "type"],
    )
    df = pd.DataFrame(data, index=index).fillna(0.0).round(decimals)
    df = df.reset_index()

    if len(names) == 1:
        df = df.rename(columns={names[0]: "normalized count"})

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare dependency distributions across XES logs."
    )
    parser.add_argument("logs", nargs="+", help="XES log files")
    parser.add_argument(
        "--format",
        choices=["table", "csv", "latex", "markdown"],
        default="table",
        help="Output format (default: table)",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=4,
        help="Decimal places to round to (default: 4)",
    )
    parser.add_argument(
        "--no-zeros",
        action="store_true",
        help="Drop rows where all values are zero across all logs",
    )
    args = parser.parse_args()

    df = build_table(args.logs, decimals=args.decimals)

    if args.no_zeros:
        df = df.loc[(df != 0).any(axis=1)]

    if args.format == "csv":
        print(df.to_csv())
    elif args.format == "latex":
        print(df.to_latex())
    elif args.format == "markdown":
        print(df.to_markdown())
    else:
        print(df.to_string())


if __name__ == "__main__":
    main()
