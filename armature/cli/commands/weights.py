"""Weights command — classification debugging tool for XES logs.

Helps researchers understand WHY a log was classified as a particular structuredness class.
Shows the discovered relationship matrix, ratio distribution, classification scores,
and the weights table that drives scoring.

Views:
  1. Header with log stats and classification result
  2. Relationship distribution (what % of pairs are each type)
  3. Classification scores with weights table
  4. Deduplicated pair matrix (A↔B shown once)
  5. Optional: raw weights for specific pairs (deep debugging)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from armature.classification import classify as classify_matrix
from armature.classification.weighted_engine import WEIGHTS, CLASSES
from armature.core.dependencies import ExistentialDependency, TemporalDependency
from armature.core.matrix import Matrix
from armature.discovery import discover
from armature.discovery.weights import compute_weights
from armature.discovery.xes_parser import parse_xes
from armature.serialization.yaml_codec import YAMLCodec


# ── dependency metadata ───────────────────────────────────────────────────────

TEMPORAL_TYPES = [
    ("direct", "≺d", "Direct follows"),
    ("direct_backward", "≻d", "Direct backward"),
    ("eventual", "≺", "Eventual"),
    ("eventual_backward", "≻", "Eventual backward"),
    ("true_eventual", "≺t", "True eventual"),
    ("true_eventual_backward", "≻t", "True eventual backward"),
    ("independence", "∥", "Independence"),
    ("no_ordering", "⊘", "No ordering"),
]

EXISTENTIAL_TYPES = [
    ("equivalence", "⇔", "Co-occurrence"),
    ("negated_equivalence", "⇎", "Exclusive OR"),
    ("nand", "⊼", "Not both"),
    ("implication", "⇒", "Implication"),
    ("implication_backward", "⇐", "Implication backward"),
    ("or_dep", "∨", "Inclusive OR"),
    ("independence", "⊕", "Independence"),
]

# Symbol lookups
TM_SYM = {f: s for f, s, _ in TEMPORAL_TYPES}
EX_SYM = {f: s for f, s, _ in EXISTENTIAL_TYPES}
TM_DESC = {f: d for f, _, d in TEMPORAL_TYPES}
EX_DESC = {f: d for f, _, d in EXISTENTIAL_TYPES}


# ── color helpers ─────────────────────────────────────────────────────────────

def _weight_color(weight: float) -> str:
    if weight >= 0.9:
        return "green"
    elif weight >= 0.5:
        return "yellow"
    else:
        return "red"


def _score_color(score: float, max_score: float) -> str:
    if max_score > 0:
        ratio = score / max_score
        if ratio >= 0.9:
            return "bold green"
        elif ratio >= 0.7:
            return "yellow"
        elif ratio > 0:
            return "red"
    return "dim"


# ── section renderers ─────────────────────────────────────────────────────────

def _render_header(
    console: Console,
    filename: str,
    matrix: Matrix,
    result,
) -> None:
    """Header panel with log stats and classification result."""
    n = len(matrix.activities)
    total_pairs = n * (n - 1)
    
    # Classification info
    from armature.classification.result import CategoryEnum
    cat_map = {
        CategoryEnum.STRUCTURED: ("Structured", "S"),
        CategoryEnum.SEMI_STRUCTURED: ("Semi-Structured", "SS"),
        CategoryEnum.LOOSELY_STRUCTURED: ("Loosely Structured", "LS"),
        CategoryEnum.UNSTRUCTURED: ("Unstructured", "U"),
    }
    cat_name, cat_short = cat_map.get(result.category, (str(result.category), "?"))
    confidence = " [borderline]" if result.confidence == "boundary" else ""
    method = result.metadata.get("method", "unknown")
    
    # Scores
    scores = result.metadata.get("scores", {})
    score_parts = []
    for cls in ["S", "SS", "LS"]:
        s = scores.get(cls, 0)
        score_parts.append(f"{cls}={s:+.2f}")
    
    lines = [
        f"[bold]File:[/bold]         {filename}",
        f"[bold]Activities:[/bold]   {n}  [dim]({', '.join(sorted(matrix.activities))})[/dim]",
        f"[bold]Pairs:[/bold]        {total_pairs} (ordered), {total_pairs // 2} (unique)",
        "",
        f"[bold]Classification:[/bold] [bold cyan]{cat_name} ({cat_short})[/bold cyan]{confidence}",
        f"[bold]Method:[/bold]       {method}",
        f"[bold]Scores:[/bold]       {' | '.join(score_parts)}",
    ]
    
    panel = Panel(
        "\n".join(lines),
        title="[bold white]Armature Weights — Classification Debug[/bold white]",
        border_style="bright_blue",
        padding=(0, 1),
    )
    console.print(panel)
    console.print()


def _render_ratio_distribution(
    console: Console,
    ratios: dict[str, float],
) -> None:
    """Show what % of pairs have each relationship type."""
    # Temporal ratios
    table = Table(
        title="Temporal Relationship Distribution",
        box=box.ROUNDED,
        border_style="bright_magenta",
        title_style="bold white",
    )
    table.add_column("Type", style="bold")
    table.add_column("Symbol", justify="center")
    table.add_column("Ratio", justify="right")
    table.add_column("Bar", min_width=30)
    table.add_column("Description", style="dim")
    
    tm_ratios = [(f, ratios.get(f"temporal.{f}", 0.0)) for f, _, _ in TEMPORAL_TYPES]
    tm_ratios.sort(key=lambda x: x[1], reverse=True)
    
    for field, ratio in tm_ratios:
        sym = TM_SYM[field]
        desc = TM_DESC[field]
        pct = f"{ratio:.1%}"
        bar_len = int(ratio * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        color = "green" if ratio >= 0.2 else "yellow" if ratio >= 0.1 else "dim"
        table.add_row(field, sym, pct, f"[{color}]{bar}[/{color}]", desc)
    
    console.print(table)
    console.print()
    
    # Existential ratios
    table = Table(
        title="Existential Relationship Distribution",
        box=box.ROUNDED,
        border_style="bright_cyan",
        title_style="bold white",
    )
    table.add_column("Type", style="bold")
    table.add_column("Symbol", justify="center")
    table.add_column("Ratio", justify="right")
    table.add_column("Bar", min_width=30)
    table.add_column("Description", style="dim")
    
    ex_ratios = [(f, ratios.get(f"existential.{f}", 0.0)) for f, _, _ in EXISTENTIAL_TYPES]
    ex_ratios.sort(key=lambda x: x[1], reverse=True)
    
    for field, ratio in ex_ratios:
        sym = EX_SYM[field]
        desc = EX_DESC[field]
        pct = f"{ratio:.1%}"
        bar_len = int(ratio * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        color = "green" if ratio >= 0.2 else "yellow" if ratio >= 0.1 else "dim"
        table.add_row(field, sym, pct, f"[{color}]{bar}[/{color}]", desc)
    
    console.print(table)
    console.print()


def _render_weights_table(
    console: Console,
    scores: dict[str, float],
) -> None:
    """Show the classification weights table with current scores."""
    table = Table(
        title="Classification Weights (per relationship type)",
        box=box.ROUNDED,
        border_style="bright_green",
        title_style="bold white",
    )
    table.add_column("Domain", style="bold")
    table.add_column("Type", style="bold")
    table.add_column("Symbol", justify="center")
    for cls in CLASSES:
        s = scores.get(cls, 0)
        color = "bold green" if s == max(scores.values()) else ""
        table.add_column(f"{cls}", justify="right", header_style=color)
    
    # Temporal weights
    first_tm = True
    for field, sym, _ in TEMPORAL_TYPES:
        key = f"temporal.{field}"
        row = []
        if first_tm:
            row.append("Temporal")
            first_tm = False
        else:
            row.append("")
        row.append(field)
        row.append(sym)
        for cls in CLASSES:
            w = WEIGHTS[cls].get(key, 0.0)
            if w > 0:
                row.append(f"[green]+{w:.1f}[/green]")
            elif w < 0:
                row.append(f"[red]{w:.1f}[/red]")
            else:
                row.append(f"[dim]{w:.1f}[/dim]")
        table.add_row(*row)
    
    # Separator
    table.add_row("─" * 8, "─" * 20, "─" * 4, "─" * 6, "─" * 6, "─" * 6)
    
    # Existential weights
    first_ex = True
    for field, sym, _ in EXISTENTIAL_TYPES:
        key = f"existential.{field}" if field != "or_dep" else "existential.or"
        row = []
        if first_ex:
            row.append("Existential")
            first_ex = False
        else:
            row.append("")
        row.append(field)
        row.append(sym)
        for cls in CLASSES:
            w = WEIGHTS[cls].get(key, 0.0)
            if w > 0:
                row.append(f"[green]+{w:.1f}[/green]")
            elif w < 0:
                row.append(f"[red]{w:.1f}[/red]")
            else:
                row.append(f"[dim]{w:.1f}[/dim]")
        table.add_row(*row)
    
    console.print(table)
    console.print()
    
    # Score summary
    max_s = max(scores.values()) if scores else 1
    parts = []
    for cls in CLASSES:
        s = scores.get(cls, 0)
        color = "bold green" if s == max_s else "dim"
        parts.append(f"[{color}]{cls}={s:+.4f}[/{color}]")
    
    margin = max(scores.values()) - sorted(scores.values())[-2] if len(scores) >= 2 else 0
    borderline = "[yellow]BORDERLINE[/yellow]" if margin < 0.2 else "[green]clear[/green]"
    
    console.print(f"  [bold]Scores:[/bold] {' | '.join(parts)}")
    console.print(f"  [bold]Margin:[/bold] {margin:.4f} ({borderline})")
    console.print()


def _render_pair_matrix(
    console: Console,
    matrix: Matrix,
) -> None:
    """Deduplicated pair matrix — A↔B shown once."""
    activities = sorted(matrix.activities)
    n = len(activities)
    
    if n == 0:
        return
    
    # Calculate column width
    max_name = max(len(a) for a in activities)
    col_w = max(max_name + 1, 6)
    
    table = Table(
        title="Discovered Relationship Matrix (deduplicated: A↔B shown once)",
        box=box.SIMPLE_HEAVY,
        border_style="bright_blue",
        title_style="bold white",
    )
    
    # Header
    table.add_column("", style="bold", no_wrap=True, width=col_w)
    for tgt in activities:
        table.add_column(tgt, justify="center", no_wrap=True, width=col_w)
    
    # Rows — only show upper triangle (i < j)
    for i, src in enumerate(activities):
        row = [src]
        for j, tgt in enumerate(activities):
            if j <= i:
                # Lower triangle or diagonal — empty
                if j == i:
                    row.append("[dim]·[/dim]")
                else:
                    row.append("")
                continue
            
            # Upper triangle — show relationship
            cell = matrix.get_cell(src, tgt)
            if cell and not cell.is_neutral():
                tm_sym = TM_SYM.get(cell.temporal.value, "?")
                ex_sym = EX_SYM.get(cell.existential.value, "") if cell.existential else ""
                # Color based on "structuredness" — direct/equivalence = green, independence = dim
                if cell.temporal in (TemporalDependency.DIRECT, TemporalDependency.DIRECT_BACKWARD):
                    color = "green"
                elif cell.temporal in (TemporalDependency.EVENTUAL, TemporalDependency.TRUE_EVENTUAL,
                                       TemporalDependency.EVENTUAL_BACKWARD, TemporalDependency.TRUE_EVENTUAL_BACKWARD):
                    color = "yellow"
                else:
                    color = "dim"
                cell_str = f"[{color}]{tm_sym}[/{color}]"
                if ex_sym and cell.existential != ExistentialDependency.INDEPENDENCE:
                    cell_str += f",{ex_sym}"
            else:
                cell_str = "[dim]–[/dim]"
            row.append(cell_str)
        table.add_row(*row)
    
    console.print(table)
    console.print()
    
    # Legend
    console.print("  [bold]Legend:[/bold] temporal[,existential]")
    console.print("  Temporal: ≺d=direct  ≻d=direct_back  ≺=eventual  ≻=eventual_back  ≺t=true_eventual  ≻t=true_eventual_back  ∥=independence  ⊘=no_ordering")
    console.print("  Existential: ⇔=equivalence  ⇎=neg_equiv  ⊼=nand  ⇒=implication  ⇐=implication_back  ∨=or  ⊕=independence")
    console.print()


def _render_score_breakdown(
    console: Console,
    ratios: dict[str, float],
    scores: dict[str, float],
) -> None:
    """Show which relationship types contribute most to each class score."""
    table = Table(
        title="Score Breakdown (ratio × weight = contribution)",
        box=box.ROUNDED,
        border_style="bright_yellow",
        title_style="bold white",
    )
    table.add_column("Type", style="bold")
    table.add_column("Symbol", justify="center")
    table.add_column("Ratio", justify="right")
    for cls in CLASSES:
        table.add_column(f"{cls}", justify="right")
    table.add_column("Top Driver", justify="center")
    
    def _top_drivers(contribs: dict[str, float]) -> str:
        """Find top driver class(es), joining ties with '/'."""
        max_abs = max(abs(v) for v in contribs.values())
        tied = [c for c in CLASSES if abs(abs(contribs[c]) - max_abs) < 1e-9]
        return "/".join(tied)

    # Collect all contributions
    contributions = []
    for field, sym, _ in TEMPORAL_TYPES:
        key = f"temporal.{field}"
        ratio = ratios.get(key, 0.0)
        if ratio == 0:
            continue
        contribs = {}
        for cls in CLASSES:
            w = WEIGHTS[cls].get(key, 0.0)
            contribs[cls] = ratio * w
        top_cls = _top_drivers(contribs)
        contributions.append((field, sym, ratio, contribs, top_cls))
    
    for field, sym, _ in EXISTENTIAL_TYPES:
        key = f"existential.{'or' if field == 'or_dep' else field}"
        
        ratio = ratios.get(key, 0.0)
        if ratio == 0:
            continue
        
        contribs = {}
        for cls in CLASSES:
            w = WEIGHTS[cls].get(key, 0.0)
            contribs[cls] = ratio * w
        
        # Find top driver class(es)
        top_cls = _top_drivers(contribs)
        contributions.append((field, sym, ratio, contribs, top_cls))
    
    # Sort by absolute total contribution
    contributions.sort(key=lambda x: sum(abs(v) for v in x[3].values()), reverse=True)
    
    for field, sym, ratio, contribs, top_cls in contributions[:10]:  # Top 10
        row = [field, sym, f"{ratio:.1%}"]
        for cls in CLASSES:
            c = contribs[cls]
            if c > 0.01:
                row.append(f"[green]+{c:.3f}[/green]")
            elif c < -0.01:
                row.append(f"[red]{c:.3f}[/red]")
            else:
                row.append(f"[dim]{c:+.3f}[/dim]")
        
        # Top driver indicator
        classes = top_cls.split("/")
        if len(classes) == 1:
            # Single top driver — use its sign for color
            top_val = contribs[classes[0]]
            if abs(top_val) > 0.05:
                color = "green" if top_val > 0 else "red"
                row.append(f"[{color}]{top_cls}[/{color}]")
            else:
                row.append("[dim]–[/dim]")
        else:
            # Tied — green if any tied class positive, red if all negative
            vals = [contribs[c] for c in classes]
            any_pos = any(v > 0.05 for v in vals)
            all_neg = all(v < -0.05 for v in vals)
            if any_pos:
                row.append(f"[green]{top_cls}[/green]")
            elif all_neg:
                row.append(f"[red]{top_cls}[/red]")
            else:
                row.append(f"[dim]{top_cls}[/dim]")
        table.add_row(*row)
    
    console.print(table)
    console.print()


# ── raw weights for specific pair ─────────────────────────────────────────────

def _render_raw_weights(
    console: Console,
    pair_weights,
    source: str,
    target: str,
) -> None:
    """Show raw weights (W = N_D/T) for a specific pair."""
    pw = pair_weights.get((source, target))
    if not pw:
        console.print(f"[yellow]No weight data for pair ({source}, {target})[/yellow]")
        return
    
    ew = pw.existential
    tw = pw.temporal
    T = ew.total
    
    console.print(Panel(
        f"[bold]{source}[/bold] ↔ [bold]{target}[/bold]  [dim](T={T} traces)[/dim]",
        title="Raw Weights (W = N_D / T)",
        border_style="bright_magenta",
    ))
    
    # Co-occurrence
    console.print(f"  [bold]Co-occurrence:[/bold]  Both={ew.count_both}  Only A={ew.count_only_a}  Only B={ew.count_only_b}  Neither={ew.count_neither}")
    console.print()
    
    # Temporal weights
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
    table.add_column("Type", style="bold")
    table.add_column("Symbol", justify="center")
    table.add_column("Weight", justify="right")
    table.add_column("Count", justify="right")
    
    for field, sym, desc in TEMPORAL_TYPES:
        w = getattr(tw, field)
        n = round(w * T)
        color = _weight_color(w)
        table.add_row(field, sym, f"[{color}]{w:.3f}[/{color}]", f"{n}/{T}")
    
    console.print(table)
    console.print()
    
    # Existential weights
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
    table.add_column("Type", style="bold")
    table.add_column("Symbol", justify="center")
    table.add_column("Weight", justify="right")
    table.add_column("Count", justify="right")
    
    for field, sym, desc in EXISTENTIAL_TYPES:
        if field == "independence":
            w = 1.0
            n = T
        else:
            # Compute numerator based on co-occurrence counts
            if field == "equivalence":
                n = ew.count_both + ew.count_neither
            elif field == "negated_equivalence":
                n = ew.count_only_a + ew.count_only_b
            elif field == "nand":
                n = ew.count_only_a + ew.count_only_b + ew.count_neither
            elif field == "implication_backward":
                n = T - ew.count_only_b
            elif field == "implication":
                n = T - ew.count_only_a
            elif field == "or_dep":
                n = T - ew.count_neither
            else:
                n = 0
            w = n / T if T > 0 else 0.0
        
        color = _weight_color(w)
        table.add_row(field, sym, f"[{color}]{w:.3f}[/{color}]", f"{n}/{T}")
    
    console.print(table)
    console.print()


# ── CLI command ───────────────────────────────────────────────────────────────

@click.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json"], case_sensitive=False),
    default="table",
    show_default=True,
    help="Output format.",
)
@click.option(
    "-o", "--output",
    type=click.Path(),
    help="Write output to file instead of stdout.",
)
@click.option(
    "--no-color",
    is_flag=True,
    default=False,
    help="Disable color coding.",
)
@click.option(
    "--pair",
    "pair_detail",
    nargs=2,
    type=str,
    default=None,
    help="Show raw weights for a specific pair (SOURCE TARGET).",
)
@click.option(
    "--sections",
    type=str,
    default=None,
    help="Comma-separated sections: header,distribution,scores,matrix,breakdown",
)
@click.option(
    "--no-matrix",
    is_flag=True,
    default=False,
    help="Skip the pair matrix.",
)
def weights_cmd(
    input_file: str,
    output_format: str,
    output: str | None,
    no_color: bool,
    pair_detail: tuple[str, str] | None,
    sections: str | None,
    no_matrix: bool,
) -> None:
    r"""Classification debugging tool — understand WHY a log was classified as it was.

    Shows the discovered relationship matrix, ratio distribution, classification
    scores, and weights table. Helps researchers debug misclassifications.

    \b
    Examples:
      armature weights log.xes                    # Full debugging view
      armature weights log.xes --pair a b         # Raw weights for pair a↔b
      armature weights log.xes --sections header,scores  # Just classification info
      armature weights log.xes --format json      # JSON output
    """
    path = Path(input_file)
    
    # Determine sections to show
    if sections:
        active_sections = set(s.strip().lower() for s in sections.split(","))
    elif pair_detail:
        active_sections = {"header", "raw_weights"}
    else:
        active_sections = {"header", "distribution", "scores", "matrix", "breakdown"}
    
    # ── Load or discover matrix ──────────────────────────────────────────
    suffix = path.suffix.lower()
    second_suffix = path.suffixes[-2].lower() if len(path.suffixes) > 1 else ""
    
    try:
        if suffix in (".xes", ".gz") or second_suffix == ".xes":
            matrix = discover(path)
            # Also compute raw weights for pair detail
            traces = parse_xes(path)
            raw_weights = compute_weights(traces) if traces else {}
        elif suffix in (".yaml", ".yml"):
            matrix = YAMLCodec.load(path)
            raw_weights = {}
        else:
            click.echo(f"Error: Unsupported file type: {suffix}", err=True)
            sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    
    # ── Classify ─────────────────────────────────────────────────────────
    try:
        result = classify_matrix(matrix)
    except ValueError as e:
        click.echo(f"Error: Classification failed: {e}", err=True)
        sys.exit(1)
    
    ratios = result.dependency_ratios
    scores = result.metadata.get("scores", {})
    
    # ── JSON output ──────────────────────────────────────────────────────
    if output_format == "json":
        out = {
            "file": path.name,
            "activities": sorted(matrix.activities),
            "activity_count": len(matrix.activities),
            "classification": {
                "category": result.category.value,
                "confidence": result.confidence,
                "scores": scores,
                "method": result.metadata.get("method", "unknown"),
                "rule_trace": result.rule_trace,
            },
            "ratios": ratios,
            "weights_table": WEIGHTS,
        }
        
        if pair_detail:
            src, tgt = pair_detail
            pw = raw_weights.get((src, tgt)) or raw_weights.get((tgt, src))
            if pw:
                ew = pw.existential
                tw = pw.temporal
                T = ew.total
                out["pair_detail"] = {
                    "source": src,
                    "target": tgt,
                    "co_occurrence": {
                        "both": ew.count_both,
                        "only_a": ew.count_only_a,
                        "only_b": ew.count_only_b,
                        "neither": ew.count_neither,
                        "total": T,
                    },
                    "temporal_weights": {f: round(getattr(tw, f), 6) for f, _, _ in TEMPORAL_TYPES},
                    "existential_weights": {f: round(getattr(ew, f) if f != "independence" else 1.0, 6) for f, _, _ in EXISTENTIAL_TYPES},
                }
        
        content = json.dumps(out, indent=2)
        if output:
            Path(output).write_text(content, encoding="utf-8")
        else:
            click.echo(content)
        return
    
    # ── Table output ─────────────────────────────────────────────────────
    use_color = not no_color
    
    if output:
        from io import StringIO
        use_color = False
        buf = StringIO()
        console = Console(file=buf, highlight=False, markup=True, no_color=True, width=180)
    else:
        import shutil
        term_width = shutil.get_terminal_size(fallback=(180, 40)).columns
        console = Console(highlight=False, markup=True, width=max(term_width, 120))
    
    # Header
    if "header" in active_sections:
        _render_header(console, path.name, matrix, result)
    
    # Raw weights for specific pair
    if "raw_weights" in active_sections and pair_detail:
        src, tgt = pair_detail
        _render_raw_weights(console, raw_weights, src, tgt)
        if output:
            Path(output).write_text(buf.getvalue(), encoding="utf-8")
        return
    
    # Ratio distribution
    if "distribution" in active_sections:
        _render_ratio_distribution(console, ratios)
    
    # Classification scores + weights table
    if "scores" in active_sections:
        _render_weights_table(console, scores)
    
    # Score breakdown
    if "breakdown" in active_sections:
        _render_score_breakdown(console, ratios, scores)
    
    # Pair matrix
    if "matrix" in active_sections and not no_matrix:
        _render_pair_matrix(console, matrix)
    
    if output:
        Path(output).write_text(buf.getvalue(), encoding="utf-8")
