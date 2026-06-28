"""Discover command - extract ARM matrix from XES event log."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from armature.classification.config import ConfigLoader
from armature.cli.utils import format_matrix_as_json, is_tty, show_progress, write_output
from armature.core.matrix import Matrix
from armature.discovery.discover import discover
from armature.serialization.yaml_codec import YAMLCodec


def _render_grid(matrix: Matrix) -> str:
    """Render visual grid of matrix dependencies.

    Args:
        matrix: Matrix to render

    Returns:
        Grid string with activity labels and dependency symbols
    """
    activities = sorted(matrix.activities)
    lines = []

    # Header row
    header = "     " + "  ".join(f"{a:3}" for a in activities)
    lines.append(header)

    # Data rows
    for src in activities:
        row = f"{src:3}  "
        for tgt in activities:
            cell = matrix.get_cell(src, tgt)
            if cell and not cell.is_neutral():
                # Show first letter of temporal type
                symbol = cell.temporal.value[0].upper()
            else:
                symbol = "-"
            row += f" {symbol:3}"
        lines.append(row)

    return "\n".join(lines)


@click.command()
@click.argument("input_file", type=click.Path(exists=True), required=False)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    help="Output file path (default: stdout)",
)
@click.option(
    "--format",
    type=click.Choice(["yaml", "json"], case_sensitive=False),
    default="yaml",
    help="Output format (default: yaml)",
)
@click.option(
    "--source",
    type=str,
    help="Source identifier for matrix metadata",
)
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="YAML config file for discovery thresholds",
)
@click.option(
    "--threshold-eventual",
    type=float,
    help="Override eventual threshold from config",
)
@click.option(
    "--threshold-direct",
    type=float,
    help="Override direct threshold from config",
)
@click.option(
    "--grid/--no-grid",
    default=False,
    help="Show visual grid before YAML output (default: no-grid)",
)
@click.option(
    "--no-true-eventuals",
    "no_true_eventuals",
    is_flag=True,
    default=False,
    help="Downgrade TRUE_EVENTUAL → EVENTUAL and TRUE_EVENTUAL_BACKWARD → EVENTUAL_BACKWARD",
)
def discover_cmd(
    input_file: str | None,
    output: str | None,
    format: str,
    source: str | None,
    config: str | None,
    threshold_eventual: float | None,
    threshold_direct: float | None,
    grid: bool,
    no_true_eventuals: bool,
) -> None:
    r"""Extract ARM matrix from XES event log.

    INPUT_FILE: Path to XES event log (or '-' for stdin)

    Examples:
      \b
      # Output YAML to stdout
      armature discover log.xes

      \b
      # Save to file
      armature discover log.xes -o matrix.yaml

      \b
      # JSON output
      armature discover log.xes --format json

      \b
      # With config file
      armature discover log.xes --config thresholds.yaml

      \b
      # Override specific thresholds
      armature discover log.xes --threshold-eventual 0.8 --threshold-direct 0.6
    """
    try:
        # Handle stdin detection
        if input_file is None or input_file == "-":
            if not sys.stdin.isatty():
                # Stdin is piped, read from it
                click.echo("Error: stdin input not yet implemented", err=True)
                sys.exit(1)
            else:
                # No input file and no piped stdin
                click.echo("Error: INPUT_FILE required", err=True)
                sys.exit(2)

        input_path = Path(input_file)

        # Validate input file exists
        if not input_path.exists():
            click.echo(f"Error: File not found: {input_path}", err=True)
            sys.exit(1)

        # Load and validate config if provided
        if config or threshold_eventual is not None or threshold_direct is not None:
            try:
                # Load config file or defaults
                config_obj = ConfigLoader.load(Path(config) if config else None)

                # Apply flag overrides
                if threshold_eventual is not None:
                    config_obj.eventual_ratio_structured = threshold_eventual
                if threshold_direct is not None:
                    config_obj.direct_ratio_structured = threshold_direct

                # Warn user that discovery engine doesn't use thresholds yet
                click.echo(
                    "Warning: Discovery thresholds not yet implemented in engine, config ignored",
                    err=True,
                )

            except (FileNotFoundError, ValueError) as e:
                click.echo(f"Error: Config validation failed: {e}", err=True)
                sys.exit(1)

        # Show progress only if stderr is TTY (don't pollute stdout)
        show_progress_bar = is_tty(sys.stderr)

        # Run discovery
        try:
            if show_progress_bar:
                with show_progress("Discovering ARM matrix", total=None) as pbar:
                    matrix = discover(input_path, source, no_true_eventuals=no_true_eventuals)
                    pbar.update(1)
            else:
                matrix = discover(input_path, source, no_true_eventuals=no_true_eventuals)

        except Exception as e:
            # Terse technical error messages
            click.echo(f"Error: XES parse failed: {e}", err=True)
            sys.exit(1)

        # Format output
        try:
            if format.lower() == "json":
                if grid:
                    click.echo("Warning: --grid incompatible with JSON format, ignoring", err=True)
                output_content = format_matrix_as_json(matrix)
            else:
                # YAML format - use temporary file then read
                from tempfile import NamedTemporaryFile

                with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
                    YAMLCodec.save(matrix, tmp.name)
                    tmp_path = Path(tmp.name)

                output_content = tmp_path.read_text()
                tmp_path.unlink()  # Clean up temp file

                # Prepend grid if requested
                if grid:
                    grid_lines = [
                        "# Matrix Grid:",
                        "#",
                    ]
                    for line in _render_grid(matrix).split("\n"):
                        grid_lines.append(f"# {line}")
                    grid_lines.extend(
                        [
                            "#",
                            "# Legend: D=DIRECT, E=EVENTUAL, T=TRUE_EVENTUAL, " "I=INDEPENDENCE, -=neutral",
                            "",
                        ]
                    )
                    output_content = "\n".join(grid_lines) + "\n" + output_content

        except Exception as e:
            click.echo(f"Error: Output formatting failed: {e}", err=True)
            sys.exit(1)

        # Write output
        output_path = Path(output) if output else None
        write_output(output_content, output_path, format.lower())

    except KeyboardInterrupt:
        click.echo("\nInterrupted", err=True)
        sys.exit(130)
