"""HTML report generator for discovery validation."""

from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader


def temporal_symbol(value: str) -> str:
    """Map temporal dependency to Unicode symbol."""
    symbols = {
        "no_ordering": "×",
        "independence": "−",
        "direct": "≺d",
        "eventual": "≺",
        "true_eventual": "≺t",
        "direct_backward": "≻d",
        "eventual_backward": "≻",
        "true_eventual_backward": "≻t",
    }
    return symbols.get(value, value)


def existential_symbol(value: str) -> str:
    """Map existential dependency to Unicode symbol."""
    symbols = {
        "independence": "−",
        "implication": "⇒",
        "implication_backward": "⇐",
        "equivalence": "⇔",
        "xor": "⇎",
        "nand": "⊼",
        "or": "∨",
    }
    return symbols.get(value, value)


def generate_report(results: dict[str, Any], output_path: Path) -> Path:
    """Generate HTML validation report from collected results.

    Args:
        results: Dict with files, errors, noisy_comparisons, start_time, end_time
        output_path: Path to write HTML report

    Returns:
        Path to generated report
    """
    # Ensure end_time is set
    if results.get("end_time") is None:
        import time

        results["end_time"] = time.time()

    # Setup Jinja2
    templates_dir = Path(__file__).parent / "templates"
    env = Environment(
        loader=FileSystemLoader(templates_dir),
        autoescape=True,
    )
    # Add custom filters
    env.filters["temporal_symbol"] = temporal_symbol
    env.filters["existential_symbol"] = existential_symbol
    template = env.get_template("report.html.j2")

    # Render
    html = template.render(
        timestamp=datetime.now().isoformat(),
        results=results,
    )

    # Write
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")

    return output_path


def compare_matrices(clean_matrix, noisy_matrix) -> dict[tuple[str, str], str]:
    """Compare two matrices and return dict of changes.

    Args:
        clean_matrix: Baseline Matrix
        noisy_matrix: Noisy variant Matrix

    Returns:
        Dict mapping (source, target) -> change_type ('changed', 'added', 'removed')
    """
    changes = {}

    # Get all activities from both
    all_activities = set(clean_matrix.activities) | set(noisy_matrix.activities)

    for source in all_activities:
        for target in all_activities:
            clean_in = source in clean_matrix.activities and target in clean_matrix.activities
            noisy_in = source in noisy_matrix.activities and target in noisy_matrix.activities

            if clean_in and noisy_in:
                clean_cell = clean_matrix.get_cell(source, target)
                noisy_cell = noisy_matrix.get_cell(source, target)

                temporal_changed = clean_cell.temporal != noisy_cell.temporal
                existential_changed = clean_cell.existential != noisy_cell.existential
                if temporal_changed or existential_changed:
                    changes[(source, target)] = "changed"
            elif clean_in and not noisy_in:
                changes[(source, target)] = "removed"
            elif not clean_in and noisy_in:
                changes[(source, target)] = "added"

    return changes
