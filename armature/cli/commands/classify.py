"""classify command - Classify ARM matrix or XES log.

Auto-discovers from XES if needed. Outputs human-readable or JSON.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from armature.classification import classify as classify_matrix
from armature.cli.utils import is_tty, show_progress, write_output
from armature.discovery import discover
from armature.serialization.yaml_codec import YAMLCodec


@click.command()
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    help="Output file (default: stdout)",
)
@click.option(
    "--format",
    type=click.Choice(["human", "json"], case_sensitive=False),
    default="human",
    help="Output format",
)
@click.pass_context
def classify(
    ctx: click.Context,
    input_file: Path,
    output: Path | None,
    format: str,
) -> None:
    """Classify process structure from ARM matrix or XES log.

    INPUT_FILE can be:
    - .yaml or .yml: ARM matrix file
    - .xes or .xes.gz: Event log (auto-discovers then classifies)

    Examples:
      armature classify matrix.yaml
      armature classify log.xes --format json
    """
    verbose = ctx.obj.get("verbose", False)
    quiet = ctx.obj.get("quiet", False)

    try:
        # Step 1: Load or discover matrix
        suffix = input_file.suffix.lower()
        second_suffix = input_file.suffixes[-2].lower() if len(input_file.suffixes) > 1 else ""

        if suffix in (".xes", ".gz") or second_suffix == ".xes":
            # XES input - auto-discover
            if not quiet and is_tty(sys.stderr):
                with show_progress("Discovering matrix", total=None) as pbar:
                    matrix = discover(input_file)
                    pbar.update(1)
            else:
                matrix = discover(input_file)

            if verbose:
                click.echo(f"Discovered matrix: {len(matrix.activities)} activities", err=True)

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
                f"Unsupported file type: {suffix}. Expected .yaml, .yml, .xes, or .xes.gz",
                err=True,
            )
            sys.exit(1)

        # Step 2: Classify
        try:
            result = classify_matrix(matrix)
        except ValueError as e:
            click.echo(f"Classification failed: {e}", err=True)
            sys.exit(1)

        # Step 3: Format output
        if format.lower() == "json":
            output_text = result.to_json(indent=2)
        else:
            # Human-readable format
            output_text = _format_human(result)

        # Step 4: Write output
        write_output(output_text, output, format="json" if format.lower() == "json" else "yaml")

    except FileNotFoundError as e:
        click.echo(f"File not found: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        if verbose:
            raise
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


_LABEL_FULL = {
    "S": "Structured",
    "SS": "Semi-Structured",
    "LS": "Loosely Structured",
    "U": "Unstructured",
}

_CATEGORY_TO_SHORT = {
    "structured": "S",
    "semi_structured": "SS",
    "loosely_structured": "LS",
    "unstructured": "U",
}


def _format_human(result) -> str:
    """Format ClassificationResult (weighted engine) as human-readable text."""
    lines = []

    short = _CATEGORY_TO_SHORT.get(result.category.value, result.category.value)
    label = _LABEL_FULL.get(short, result.category.value.replace("_", " ").title())
    confidence_tag = " [borderline]" if result.confidence == "boundary" else ""
    lines.append(f"Classification: {label} ({short}){confidence_tag}")
    lines.append("")

    # Scores
    scores: dict = result.metadata.get("scores", {})
    method: str = result.metadata.get("method", "unknown")

    if scores:
        lines.append("Scores:")
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for cls, s in ranked:
            marker = " <-- predicted" if cls == short else ""
            lines.append(f"  {cls:2s}  {s:+.4f}{marker}")
        lines.append("")

        if len(ranked) >= 2:
            margin = ranked[0][1] - ranked[1][1]
            lines.append(f"Margin (top-2): {margin:.4f}")
            lines.append("")

    lines.append(f"Decision: {method}")
    if result.rule_trace:
        last = result.rule_trace[-1]
        reason = last.get("reason") or last.get("ranked") or ""
        if reason:
            if isinstance(reason, list):
                reason = " > ".join(reason)
            lines.append(f"Reason:   {reason}")
    lines.append("")

    # Feature ratios (non-zero, grouped)
    ratios = result.dependency_ratios
    temporal_feats = [(k, v) for k, v in ratios.items() if k.startswith("temporal.") and v > 0]
    existential_feats = [(k, v) for k, v in ratios.items() if k.startswith("existential.") and v > 0]

    temporal_feats.sort(key=lambda x: x[1], reverse=True)
    existential_feats.sort(key=lambda x: x[1], reverse=True)

    if temporal_feats:
        lines.append("Temporal ratios (non-zero):")
        for k, v in temporal_feats:
            dep = k.split(".", 1)[1]
            lines.append(f"  {dep:<28s} {v:.4f}")
        lines.append("")

    if existential_feats:
        lines.append("Existential ratios (non-zero):")
        for k, v in existential_feats:
            dep = k.split(".", 1)[1]
            lines.append(f"  {dep:<28s} {v:.4f}")

    return "\n".join(lines)
