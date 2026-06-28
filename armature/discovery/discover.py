"""Main discovery entry point - full ARM matrix discovery from XES."""

from __future__ import annotations

from pathlib import Path

from armature.core.dependencies import (
    EXISTENTIAL_INVERSE,
    TEMPORAL_INVERSE,
    DependencyCell,
    ExistentialDependency,
    TemporalDependency,
)
from armature.core.matrix import Matrix
from armature.discovery.dfg import build_dfg
from armature.discovery.existential import extract_existential_dependencies
from armature.discovery.scc import find_sccs
from armature.discovery.temporal import extract_temporal_dependencies
from armature.discovery.xes_parser import parse_xes


def set_symmetric_cell(
    matrix: Matrix,
    source: str,
    target: str,
    temporal: TemporalDependency,
    existential: ExistentialDependency,
    temporal_deps: dict,
    existential_deps: dict,
) -> None:
    """Set cell with automatic symmetric inverse for non-diagonal pairs.

    For (a,b) where a < b alphabetically, sets:
    - matrix[a,b] = (temporal, existential)
    - matrix[b,a] = (inverse_temporal, inverse_existential)

    Exception: If (b,a) has an explicit value in extractors AND it's the SAME
    as (a,b) (bidirectional relationship like loops), use that explicit value.

    Diagonal cells (a,a) represent self-loops and use backward dependencies.

    This enforces matrix symmetry while preserving true bidirectional relationships.
    """
    # For self-loops, symmetric dependencies stay as-is (don't invert)
    # Only invert asymmetric dependencies (IMPLICATION, TRUE_EVENTUAL, etc.)
    if source == target:
        # Temporal: DIRECT, EVENTUAL, INDEPENDENCE are symmetric for self-loops
        if temporal not in {TemporalDependency.DIRECT, TemporalDependency.DIRECT_BACKWARD,
                             TemporalDependency.EVENTUAL, TemporalDependency.INDEPENDENCE,
                             TemporalDependency.NO_ORDERING}:
            temporal = TEMPORAL_INVERSE.get(temporal, temporal)
        # Existential: symmetric dependencies stay as-is
        if existential not in {
            ExistentialDependency.EQUIVALENCE,
            ExistentialDependency.NEGATED_EQUIVALENCE,
            ExistentialDependency.OR,
            ExistentialDependency.NAND,
            ExistentialDependency.INDEPENDENCE,
        }:
            existential = EXISTENTIAL_INVERSE.get(existential, existential)

    cell = DependencyCell(temporal=temporal, existential=existential)
    matrix.set_cell(source, target, cell)

    # Set inverse for non-diagonal
    if source != target:
        reverse_pair = (target, source)

        # Check if reverse has explicit values
        reverse_temporal_explicit = temporal_deps.get(reverse_pair)
        reverse_existential_explicit = existential_deps.get(reverse_pair)

        # For temporal: use explicit if it's a symmetric dependency (DIRECT, EVENTUAL, etc)
        # These indicate true bidirectional relationships (loops)
        symmetric_temporal = {
            TemporalDependency.DIRECT,
            TemporalDependency.EVENTUAL,
            TemporalDependency.TRUE_EVENTUAL,
            TemporalDependency.INDEPENDENCE,
        }

        if (
            reverse_temporal_explicit is not None
            and temporal in symmetric_temporal
            and reverse_temporal_explicit in symmetric_temporal
        ):
            # Both directions have explicit symmetric relationships - use explicit value
            reverse_temporal = reverse_temporal_explicit
        else:
            # Use inverse mapping
            reverse_temporal = TEMPORAL_INVERSE.get(temporal, temporal)

        # For existential: symmetric dependencies map to themselves
        symmetric_existential = {
            ExistentialDependency.EQUIVALENCE,
            ExistentialDependency.NEGATED_EQUIVALENCE,
            ExistentialDependency.NAND,
            ExistentialDependency.OR,
            ExistentialDependency.INDEPENDENCE,
        }

        if (
            reverse_existential_explicit is not None
            and existential in symmetric_existential
            and reverse_existential_explicit in symmetric_existential
        ):
            # Use explicit value
            reverse_existential = reverse_existential_explicit
        else:
            # Use inverse mapping
            reverse_existential = EXISTENTIAL_INVERSE.get(existential, existential)

        inverse_cell = DependencyCell(
            temporal=reverse_temporal,
            existential=reverse_existential,
        )
        matrix.set_cell(target, source, inverse_cell)


def discover(
    path: Path | str,
    source: str | None = None,
    threshold: float = 1.0,
    no_true_eventuals: bool = False,
) -> Matrix:
    """Discover ARM matrix from XES event log.

    This is the main entry point for the discovery pipeline. It executes
    all discovery steps in sequence:
    1. Parse XES file into traces
    2. Build directly-follows graph (DFG)
    3. Find strongly connected components (loops)
    4. Extract temporal dependencies (ordering)
    5. Extract existential dependencies (co-occurrence)
    6. Construct Matrix with all relationships

    Adds metadata for ML feature extraction (trace count, variant diversity, filename).

    Args:
        path: Path to XES event log file
        source: Optional custom source identifier (defaults to file path)
        threshold: Minimum fraction of traces that must satisfy patterns (0.0-1.0).
                   Default 1.0 means patterns must hold in 100% of traces (deterministic).
                   Lower values allow noise/exceptions in the data.
                   Applies to both temporal (ordering) and existential (co-occurrence) patterns.
                   Examples:
                   - threshold=1.0: Strict, 100% of traces must match (default)
                   - threshold=0.9: 90% of traces must match (10% noise allowed)
                   - threshold=0.8: 80% of traces must match (20% noise allowed)
        no_true_eventuals: If True, downgrade TRUE_EVENTUAL → EVENTUAL and
                           TRUE_EVENTUAL_BACKWARD → EVENTUAL_BACKWARD in the resulting matrix.

    Returns:
        Matrix populated with temporal and existential dependencies
    """
    # Convert path to Path object if string
    if isinstance(path, str):
        path = Path(path)

    # Step 1: Parse XES file
    traces = parse_xes(path)

    # Calculate trace metadata for ML features
    num_traces = len(traces)
    # Count unique variants (by activity sequence)
    from collections import Counter

    variants = Counter()
    for trace in traces:
        variant = tuple(event.activity for event in trace.events)
        variants[variant] += 1
    num_variants = len(variants)

    # Step 2: Build DFG
    dfg = build_dfg(traces)

    # Step 3: Find SCCs (loop detection)
    loop_ctx = find_sccs(dfg)

    # Step 4: Extract temporal dependencies
    temporal_deps = extract_temporal_dependencies(traces, dfg, loop_ctx, threshold)

    # Step 5: Extract existential dependencies
    existential_deps = extract_existential_dependencies(traces, threshold)

    # Step 6: Construct Matrix with metadata
    matrix = Matrix(
        source=source or str(path),
        num_traces=num_traces,
        num_variants=num_variants,
    )

    # Add all activities
    for activity in sorted(dfg.activities):
        matrix.add_activity(activity)

    # Set all dependencies - process upper triangle + diagonal only
    sorted_activities = sorted(dfg.activities)
    for i, source_activity in enumerate(sorted_activities):
        for target_activity in sorted_activities[i:]:  # Upper triangle + diagonal
            pair = (source_activity, target_activity)
            reverse_pair = (target_activity, source_activity)

            # Get dependencies from extractors - check BOTH directions
            temporal = temporal_deps.get(pair, DependencyCell().temporal)
            existential = existential_deps.get(pair, DependencyCell().existential)

            # Also check reverse pair
            reverse_temporal = temporal_deps.get(reverse_pair, DependencyCell().temporal)
            reverse_existential = existential_deps.get(reverse_pair, DependencyCell().existential)

            # Prefer non-default values
            default_cell = DependencyCell()

            # If forward is default but reverse is not, use reverse (with inverse)
            if temporal == default_cell.temporal and reverse_temporal != default_cell.temporal:
                temporal = TEMPORAL_INVERSE.get(reverse_temporal, reverse_temporal)

            if existential == default_cell.existential and reverse_existential != default_cell.existential:
                existential = EXISTENTIAL_INVERSE.get(reverse_existential, reverse_existential)

            # Check if non-default
            if temporal != default_cell.temporal or existential != default_cell.existential:
                set_symmetric_cell(
                    matrix,
                    source_activity,
                    target_activity,
                    temporal,
                    existential,
                    temporal_deps,
                    existential_deps,
                )

    if no_true_eventuals:
        _downgrade_true_eventuals(matrix)

    return matrix


_TRUE_EVENTUAL_MAP = {
    TemporalDependency.TRUE_EVENTUAL: TemporalDependency.EVENTUAL,
    TemporalDependency.TRUE_EVENTUAL_BACKWARD: TemporalDependency.EVENTUAL_BACKWARD,
}


def _downgrade_true_eventuals(matrix: Matrix) -> None:
    """Replace TRUE_EVENTUAL/TRUE_EVENTUAL_BACKWARD with EVENTUAL/EVENTUAL_BACKWARD in-place."""
    for source, targets in matrix.dependencies.items():
        for target, cell in list(targets.items()):
            replacement = _TRUE_EVENTUAL_MAP.get(cell.temporal)
            if replacement is not None:
                targets[target] = DependencyCell(temporal=replacement, existential=cell.existential)
