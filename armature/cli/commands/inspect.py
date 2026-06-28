"""inspect command - Debug dump of ARM matrix with detailed views.

Shows full matrix summary, grid rendering, and cell details for transparency.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from armature.cli.utils import is_tty, show_progress, write_output
from armature.core.matrix import Matrix
from armature.discovery import discover
from armature.serialization.yaml_codec import YAMLCodec


@click.command()
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.argument("activities", nargs=-1, type=str)
@click.option(
    "--format",
    type=click.Choice(["human", "json"], case_sensitive=False),
    default="human",
    help="Output format",
)
@click.pass_context
def inspect(
    ctx: click.Context,
    input_file: Path,
    activities: tuple[str, ...],
    format: str,
) -> None:
    """Inspect ARM matrix with full transparency.

    INPUT_FILE can be a YAML matrix or XES log (auto-discovers).

    Optional ACTIVITIES arguments filter the view:
    - No args: Show full matrix summary + grid
    - One arg (A): Show all cells involving activity A
    - Two args (A B): Show cell (A,B) details

    Examples:
      armature inspect matrix.yaml
      armature inspect log.xes
      armature inspect matrix.yaml A
      armature inspect matrix.yaml A B
    """
    verbose = ctx.obj.get("verbose", False)

    try:
        # Load or discover matrix
        suffix = input_file.suffix.lower()
        second_suffix = input_file.suffixes[-2].lower() if len(input_file.suffixes) > 1 else ""

        if suffix in (".xes", ".gz") or second_suffix == ".xes":
            # XES input - auto-discover
            if not ctx.obj.get("quiet", False) and is_tty(sys.stderr):
                with show_progress("Discovering matrix", total=None) as pbar:
                    matrix = discover(input_file)
                    pbar.update(1)
            else:
                matrix = discover(input_file)

            if verbose:
                click.echo(
                    f"Discovered matrix: {len(matrix.activities)} activities",
                    err=True,
                )

        elif suffix in (".yaml", ".yml"):
            # YAML input - load directly
            try:
                matrix = YAMLCodec.load(input_file)
            except Exception as e:
                click.echo(f"Parse failed: {e}", err=True)
                sys.exit(1)

            if verbose:
                click.echo(f"Loaded matrix: {len(matrix.activities)} activities", err=True)

        else:
            click.echo(
                f"Unsupported file type: {suffix}. " f"Expected .yaml, .yml, .xes, or .xes.gz",
                err=True,
            )
            sys.exit(1)

        # Validate activities
        for activity in activities:
            if activity not in matrix.activities:
                click.echo(f"Activity '{activity}' not in matrix", err=True)
                sys.exit(1)

        # Generate output based on filter
        if len(activities) == 0:
            # Full matrix summary
            output_text = _format_matrix_summary(matrix, format)
        elif len(activities) == 1:
            # Single activity - show all cells involving it
            output_text = _format_activity_cells(matrix, activities[0], format)
        elif len(activities) == 2:
            # Two activities - show specific cell
            output_text = _format_cell_details(matrix, activities[0], activities[1], format)
        else:
            click.echo(
                "Too many activities. Specify 0 (full matrix), "
                "1 (activity filter), or 2 (cell details)",
                err=True,
            )
            sys.exit(1)

        # Write output
        write_output(output_text, None, format="json" if format.lower() == "json" else "yaml")

    except FileNotFoundError as e:
        click.echo(f"File not found: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        if verbose:
            raise
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


def _format_matrix_summary(matrix: Matrix, format: str) -> str:
    """Format full matrix summary with stats and grid.

    Args:
        matrix: Matrix to inspect
        format: Output format (human or json)

    Returns:
        Formatted string
    """
    if format.lower() == "json":
        import json

        # JSON format - structured data
        non_neutral_count = sum(
            1
            for src_deps in matrix.dependencies.values()
            for cell in src_deps.values()
            if not cell.is_neutral()
        )

        n = len(matrix.activities)
        density = non_neutral_count / (n * n) if n > 0 else 0.0

        data = {
            "activity_count": n,
            "activities": sorted(matrix.activities),
            "dependency_count": non_neutral_count,
            "density": density,
            "source": matrix.source,
        }
        return json.dumps(data, indent=2)

    # Human-readable format
    lines = []

    # Stats
    n = len(matrix.activities)
    non_neutral_count = sum(
        1
        for src_deps in matrix.dependencies.values()
        for cell in src_deps.values()
        if not cell.is_neutral()
    )
    density = non_neutral_count / (n * n) if n > 0 else 0.0

    lines.append(f"Activity count: {n}")
    lines.append(f"Dependency count: {non_neutral_count}")
    lines.append(f"Density: {density:.2f}")
    if matrix.source:
        lines.append(f"Source: {matrix.source}")
    lines.append("")

    # Grid
    lines.append("Matrix Grid:")
    lines.append("")
    lines.append(_render_grid(matrix))
    lines.append("")
    lines.append("Legend:")
    lines.append("  Temporal: ≺d=DIRECT, ≺=EVENTUAL, ≺t=TRUE_EVENTUAL")
    lines.append("           ≻d=DIRECT_BACK, ≻=EVENTUAL_BACK, ≻t=TRUE_EVENTUAL_BACK")
    lines.append("           ∥=INDEPENDENCE")
    lines.append("  Existential: =>=IMPLICATION, <==IMPLICATION_BACK, ⇔=EQUIVALENCE")
    lines.append("              ⇎=NEGATED_EQUIVALENCE, ⊼=NAND, ∨=OR, ⊕=INDEPENDENCE")
    lines.append("  Format: temporal,existential (e.g., ≺d,<= or ∥,⊼)")
    lines.append("  Special: None=diagonal, -=neutral")

    return "\n".join(lines)


def _render_grid(matrix: Matrix) -> str:
    """Render visual grid of matrix dependencies using Unicode symbols.

    Args:
        matrix: Matrix to render

    Returns:
        Grid string with activity labels and combined temporal+existential symbols
    """
    from armature.core.dependencies import ExistentialDependency, TemporalDependency

    activities = sorted(matrix.activities)
    lines = []

    # Header row - adjust spacing for Unicode symbols
    header = "        " + "  ".join(f"{a:5}" for a in activities)
    lines.append(header)

    # Data rows
    for src in activities:
        row = f"{src:5}  "
        for tgt in activities:
            cell = matrix.get_cell(src, tgt)
            if cell and not cell.is_neutral():
                # Map temporal to Unicode symbol
                temporal_map = {
                    TemporalDependency.DIRECT: "≺d",
                    TemporalDependency.DIRECT_BACKWARD: "≻d",
                    TemporalDependency.EVENTUAL: "≺",
                    TemporalDependency.EVENTUAL_BACKWARD: "≻",
                    TemporalDependency.TRUE_EVENTUAL: "≺t",
                    TemporalDependency.TRUE_EVENTUAL_BACKWARD: "≻t",
                    TemporalDependency.INDEPENDENCE: "∥",
                }
                temporal_symbol = temporal_map.get(cell.temporal, "-")

                # Map existential to Unicode symbol
                existential_map = {
                    ExistentialDependency.IMPLICATION: "=>",
                    ExistentialDependency.IMPLICATION_BACKWARD: "<=",
                    ExistentialDependency.EQUIVALENCE: "⇔",
                    ExistentialDependency.NEGATED_EQUIVALENCE: "⇎",
                    ExistentialDependency.NAND: "⊼",
                    ExistentialDependency.OR: "∨",
                    ExistentialDependency.INDEPENDENCE: "⊕",
                }
                existential_symbol = existential_map.get(cell.existential, "")

                # Combine: "temporal,existential" or just "temporal" if no existential
                if existential_symbol:
                    symbol = f"{temporal_symbol},{existential_symbol}"
                else:
                    symbol = temporal_symbol
            elif src == tgt:
                # Diagonal - no self-dependency shown
                symbol = "None"
            else:
                symbol = "-"
            row += f" {symbol:5}"
        lines.append(row)

    return "\n".join(lines)


def _format_activity_cells(matrix: Matrix, activity: str, format: str) -> str:
    """Format all cells involving a specific activity.

    Args:
        matrix: Matrix to inspect
        activity: Activity name
        format: Output format (human or json)

    Returns:
        Formatted string
    """
    if format.lower() == "json":
        import json

        # Collect all cells where activity is source or target
        cells = {}

        # As source
        for tgt in sorted(matrix.activities):
            if tgt == activity:
                continue
            cell = matrix.get_cell(activity, tgt)
            if not cell.is_neutral():
                cells[f"{activity},{tgt}"] = {
                    "temporal": cell.temporal.value,
                    "existential": cell.existential.value if cell.existential else None,
                }

        # As target
        for src in sorted(matrix.activities):
            if src == activity:
                continue
            cell = matrix.get_cell(src, activity)
            if not cell.is_neutral():
                cells[f"{src},{activity}"] = {
                    "temporal": cell.temporal.value,
                    "existential": cell.existential.value if cell.existential else None,
                }

        return json.dumps({"activity": activity, "cells": cells}, indent=2)

    # Human-readable format
    lines = []
    lines.append(f"Cells involving '{activity}':")
    lines.append("")

    # As source
    lines.append(f"As source ({activity} → ...):")
    found_source = False
    for tgt in sorted(matrix.activities):
        if tgt == activity:
            continue
        cell = matrix.get_cell(activity, tgt)
        if not cell.is_neutral():
            found_source = True
            lines.append(
                f"  {activity} → {tgt}: "
                f"temporal={cell.temporal.value}, "
                f"existential={cell.existential.value}"
            )
    if not found_source:
        lines.append("  (none)")
    lines.append("")

    # As target
    lines.append(f"As target (... → {activity}):")
    found_target = False
    for src in sorted(matrix.activities):
        if src == activity:
            continue
        cell = matrix.get_cell(src, activity)
        if not cell.is_neutral():
            found_target = True
            lines.append(
                f"  {src} → {activity}: "
                f"temporal={cell.temporal.value}, "
                f"existential={cell.existential.value}"
            )
    if not found_target:
        lines.append("  (none)")

    return "\n".join(lines)


def _format_cell_details(matrix: Matrix, source: str, target: str, format: str) -> str:
    """Format detailed view of a specific cell.

    Args:
        matrix: Matrix to inspect
        source: Source activity
        target: Target activity
        format: Output format (human or json)

    Returns:
        Formatted string
    """
    cell = matrix.get_cell(source, target)

    if format.lower() == "json":
        import json

        data = {
            "source": source,
            "target": target,
            "temporal": cell.temporal.value,
            "existential": cell.existential.value,
            "is_neutral": cell.is_neutral(),
        }
        return json.dumps(data, indent=2)

    # Human-readable format
    lines = []
    lines.append(f"Cell ({source}, {target}):")
    lines.append(f"  Temporal: {cell.temporal.value}")
    lines.append(f"  Existential: {cell.existential.value}")
    lines.append("")

    if cell.is_neutral():
        lines.append("  (Neutral - no dependency)")
    else:
        lines.append("  Note: Discovery metadata (counts/scores) not stored in matrix.")
        lines.append("        Re-run discovery with --debug flag for detailed metrics.")

    return "\n".join(lines)
