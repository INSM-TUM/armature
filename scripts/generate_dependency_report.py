#!/usr/bin/env python3
"""Generate a visual HTML dependency-ratio report for synthetic logs.

Produces five heatmap tables:
  1. Structured logs (s1–s12)
  2. Semi-structured logs (ss1–ss3)
  3. Loosely structured logs (ls1–ls4)
  4. Unstructured logs (u1–u2)
  5. All logs together

Usage:
    python scripts/generate_dependency_report.py
    python scripts/generate_dependency_report.py --output my_report.html
    python scripts/generate_dependency_report.py --data-dir "/path/to/Synthetic Log Data"
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

# ---------------------------------------------------------------------------
# Dependency counting (same logic as compare_log_dependencies.py)
# ---------------------------------------------------------------------------

def count_dependencies(matrix) -> dict[tuple[str, str], float]:
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


def build_df(log_paths: list[Path], decimals: int = 4) -> pd.DataFrame:
    data: dict[str, dict[tuple[str, str], float]] = {}
    for path in log_paths:
        name = path.name
        print(f"  Processing {name}...", file=sys.stderr)
        matrix = discover(path)
        data[name] = count_dependencies(matrix)

    index = pd.MultiIndex.from_tuples(
        [("temporal", t.value) for t in TEMPORAL_ORDER]
        + [("existential", e.value) for e in EXISTENTIAL_ORDER],
        names=["kind", "type"],
    )
    df = pd.DataFrame(data, index=index).fillna(0.0).round(decimals)
    return df


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

CSS = """
<style>
  body {
    font-family: 'Segoe UI', Arial, sans-serif;
    background: #f5f6fa;
    color: #1a1a2e;
    margin: 0;
    padding: 2rem;
  }
  h1 {
    font-size: 1.6rem;
    margin-bottom: 0.3rem;
    color: #0f3460;
  }
  .subtitle {
    color: #555;
    font-size: 0.9rem;
    margin-bottom: 2.5rem;
  }
  .section {
    margin-bottom: 3rem;
  }
  h2 {
    font-size: 1.15rem;
    color: #16213e;
    border-left: 4px solid #0f3460;
    padding-left: 0.6rem;
    margin-bottom: 0.8rem;
  }
  .section-meta {
    font-size: 0.78rem;
    color: #888;
    margin-bottom: 0.6rem;
  }
  .table-wrap {
    overflow-x: auto;
  }
  table {
    border-collapse: collapse;
    font-size: 0.82rem;
    white-space: nowrap;
    min-width: 100%;
  }
  thead tr th {
    background: #0f3460;
    color: #fff;
    padding: 6px 12px;
    text-align: center;
    font-weight: 600;
    position: sticky;
    top: 0;
  }
  thead tr th.col-kind { text-align: left; background: #16213e; }
  thead tr th.col-type { text-align: left; background: #16213e; }
  tbody tr td {
    padding: 5px 12px;
    border-bottom: 1px solid #e8e8ee;
    text-align: right;
  }
  tbody tr td.col-kind {
    text-align: left;
    font-weight: 700;
    font-size: 0.72rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: #0f3460;
    border-right: 2px solid #ccd;
    padding-left: 8px;
    background: #f0f2fa;
  }
  tbody tr td.col-type {
    text-align: left;
    font-family: monospace;
    font-size: 0.82rem;
    color: #333;
    border-right: 2px solid #e0e0ee;
    padding-right: 14px;
    background: #fafbff;
  }
  tbody tr:hover td { filter: brightness(0.96); }
  .zero { color: #ccc !important; }
  .group-divider td {
    border-top: 2px solid #b0b8d0 !important;
  }
  .legend {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.75rem;
    color: #666;
    margin-bottom: 0.4rem;
  }
  .legend-bar {
    width: 120px;
    height: 10px;
    border-radius: 3px;
  }
</style>
"""

# Color palettes per kind: (r_low,g_low,b_low) -> (r_high,g_high,b_high)
PALETTE = {
    "temporal":    ((235, 245, 255), (15, 52, 96)),   # white -> deep blue
    "existential": ((255, 243, 235), (150, 60, 10)),   # white -> deep orange
}


def _lerp_color(t: float, lo: tuple, hi: tuple) -> str:
    r = int(lo[0] + t * (hi[0] - lo[0]))
    g = int(lo[1] + t * (hi[1] - lo[1]))
    b = int(lo[2] + t * (hi[2] - lo[2]))
    return f"rgb({r},{g},{b})"


def _text_color(t: float) -> str:
    return "#fff" if t > 0.65 else "#1a1a2e"


def render_table(df: pd.DataFrame, title: str, log_count: int, pair_note: str = "") -> str:
    col_names = list(df.columns)

    # Per-kind max for color scaling
    kind_max: dict[str, float] = {}
    for kind in ("temporal", "existential"):
        sub = df.loc[kind] if kind in df.index.get_level_values("kind") else pd.DataFrame()
        kind_max[kind] = float(sub.values.max()) if not sub.empty else 1.0

    lines: list[str] = []
    lines.append(f'<div class="section">')
    lines.append(f'<h2>{title}</h2>')
    lines.append(f'<div class="section-meta">{log_count} log(s) &mdash; {len(col_names)} column(s){" &mdash; " + pair_note if pair_note else ""}</div>')

    # Legend
    t_lo, t_hi = PALETTE["temporal"]
    e_lo, e_hi = PALETTE["existential"]
    lines.append('<div class="legend">')
    lines.append(f'<span>Temporal scale:</span>')
    lines.append(f'<span class="legend-bar" style="background:linear-gradient(to right,rgb{t_lo},rgb{t_hi})"></span>')
    lines.append(f'<span style="margin-left:1rem">Existential scale:</span>')
    lines.append(f'<span class="legend-bar" style="background:linear-gradient(to right,rgb{e_lo},rgb{e_hi})"></span>')
    lines.append('</div>')

    lines.append('<div class="table-wrap"><table>')

    # Header
    lines.append('<thead><tr>')
    lines.append('<th class="col-kind">Kind</th>')
    lines.append('<th class="col-type">Type</th>')
    for col in col_names:
        lines.append(f'<th>{col}</th>')
    lines.append('</tr></thead>')

    lines.append('<tbody>')
    prev_kind = None
    for (kind, dep_type), row in df.iterrows():
        divider_class = ' class="group-divider"' if kind != prev_kind and prev_kind is not None else ''
        lines.append(f'<tr{divider_class}>')

        # kind cell — only show on kind change
        if kind != prev_kind:
            # count rows in this kind
            count_kind = df.index.get_level_values("kind").tolist().count(kind)
            lines.append(f'<td class="col-kind" rowspan="{count_kind}">{kind}</td>')
        lines.append(f'<td class="col-type">{dep_type}</td>')

        kmax = kind_max.get(kind, 1.0) or 1.0
        lo, hi = PALETTE[kind]
        for col in col_names:
            val = row[col]
            if val == 0.0:
                lines.append('<td class="zero">—</td>')
            else:
                t = min(val / kmax, 1.0)
                bg = _lerp_color(t, lo, hi)
                fg = _text_color(t)
                lines.append(
                    f'<td style="background:{bg};color:{fg};font-weight:{"600" if t>0.4 else "400"}">'
                    f'{val:.4f}</td>'
                )
        lines.append('</tr>')
        prev_kind = kind

    lines.append('</tbody></table></div>')
    lines.append('</div>')
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate visual HTML dependency-ratio report.")
    parser.add_argument(
        "--data-dir",
        default="Synthetic Log Data",
        help="Path to 'Synthetic Log Data' directory (default: ./Synthetic Log Data)",
    )
    parser.add_argument(
        "--output",
        default="dependency_report.html",
        help="Output HTML file (default: dependency_report.html)",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=4,
    )
    args = parser.parse_args()

    base = Path(args.data_dir)

    groups: list[tuple[str, list[Path]]] = [
        ("Structured logs", sorted((base / "structuredLogs").glob("*.xes"))),
        ("Semi-structured logs", sorted((base / "semiStructuredLogs").glob("*.xes"))),
        ("Loosely structured logs", sorted((base / "looselyStructuredLogs").glob("*.xes"))),
        ("Unstructured logs", sorted((base / "unstructuredLogs").glob("*.xes"))),
    ]

    dfs: list[pd.DataFrame] = []
    sections: list[str] = []

    for title, paths in groups:
        if not paths:
            print(f"[warn] No XES files found for: {title}", file=sys.stderr)
            continue
        print(f"\n[{title}]", file=sys.stderr)
        df = build_df(paths, decimals=args.decimals)
        dfs.append(df)
        sections.append(render_table(df, title, len(paths)))

    # All together
    all_paths = [p for _, paths in groups for p in paths]
    print("\n[All logs together]", file=sys.stderr)
    df_all = build_df(all_paths, decimals=args.decimals)
    sections.append(render_table(df_all, "All logs together", len(all_paths)))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dependency Ratio Report — Synthetic Logs</title>
{CSS}
</head>
<body>
<h1>Dependency Ratio Report &mdash; Synthetic Logs</h1>
<p class="subtitle">
  Normalized frequency of each dependency type per directed activity pair (n&times;(n&minus;1) denominator).
  Color intensity = ratio relative to the per-table maximum for that kind.
  &mdash; = zero.
</p>
{"".join(sections)}
</body>
</html>"""

    out_path = Path(args.output)
    out_path.write_text(html)
    print(f"\nReport written to: {out_path.resolve()}", file=sys.stderr)


if __name__ == "__main__":
    main()
