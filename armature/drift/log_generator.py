"""XES log generation utilities for drift detection testing.

Creates synthetic event logs with controlled concept drifts for
validating ARM's superiority over Bose S/N/A detection.
"""
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Callable, Optional
from lxml import etree


class DriftType(Enum):
    """Types of concept drifts to generate."""
    EXISTENTIAL = "existential"           # IMPLICATION -> INDEPENDENCE
    TEMPORAL_DIRECTNESS = "directness"    # DIRECT -> EVENTUAL
    TEMPORAL_REACHABILITY = "reachability" # EVENTUAL -> TRUE_EVENTUAL
    COMBINED = "combined"                  # Multiple changes
    STRUCTURAL = "structural"              # Activity added/removed


@dataclass
class DriftScenario:
    """Configuration for a drift scenario."""
    name: str
    drift_type: DriftType
    drift_point: int  # Trace index where drift occurs
    traces_before: int  # Number of traces before drift
    traces_after: int   # Number of traces after drift
    description: str
    expected_arm_detection: Optional[int] = None  # Expected trace index
    expected_bose_detection: Optional[int] = None  # None = not detected


def _create_xes_log() -> etree.Element:
    """Create XES log root element with proper namespaces."""
    log = etree.Element('log')
    log.set('xes.version', '1849-2016')
    log.set('xes.features', 'nested-attributes')

    # Global trace attributes
    trace_globals = etree.SubElement(log, 'global')
    trace_globals.set('scope', 'trace')
    name_attr = etree.SubElement(trace_globals, 'string')
    name_attr.set('key', 'concept:name')
    name_attr.set('value', '__INVALID__')

    # Global event attributes
    event_globals = etree.SubElement(log, 'global')
    event_globals.set('scope', 'event')
    name_attr = etree.SubElement(event_globals, 'string')
    name_attr.set('key', 'concept:name')
    name_attr.set('value', '__INVALID__')

    return log


def _add_trace(log: etree.Element, case_id: str, activities: List[str]) -> None:
    """Add a trace with given activities to the log."""
    trace = etree.SubElement(log, 'trace')
    name = etree.SubElement(trace, 'string')
    name.set('key', 'concept:name')
    name.set('value', case_id)

    for activity in activities:
        event = etree.SubElement(trace, 'event')
        name_elem = etree.SubElement(event, 'string')
        name_elem.set('key', 'concept:name')
        name_elem.set('value', activity)


def _write_xes(log: etree.Element, path: Path) -> None:
    """Write XES log to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tree = etree.ElementTree(log)
    tree.write(str(path), encoding='utf-8', xml_declaration=True, pretty_print=True)


def generate_existential_drift_log(
    path: Path,
    traces_before: int = 50,
    traces_after: int = 50,
) -> DriftScenario:
    """Generate log with IMPLICATION -> INDEPENDENCE drift.

    Before drift: A always has B (IMPLICATION: A => B)
    After drift: A sometimes without B (INDEPENDENCE)

    This targets Bose's weakness: S/N/A only tracks succession,
    not co-occurrence patterns. Both before and after have
    "A sometimes followed by B" (S), so Bose sees no change.
    ARM detects the existential dependency change.

    Returns DriftScenario with metadata.
    """
    log = _create_xes_log()

    # Before drift: Every trace has A -> B -> C (IMPLICATION: A => B, A => C)
    for i in range(traces_before):
        _add_trace(log, str(i), ["A", "B", "C"])

    # After drift: 50% of traces have A without B (INDEPENDENCE for A-B)
    for i in range(traces_after):
        trace_idx = traces_before + i
        if i % 2 == 0:
            _add_trace(log, str(trace_idx), ["A", "C"])  # No B!
        else:
            _add_trace(log, str(trace_idx), ["A", "B", "C"])

    _write_xes(log, path)

    return DriftScenario(
        name="existential_implication_to_independence",
        drift_type=DriftType.EXISTENTIAL,
        drift_point=traces_before,
        traces_before=traces_before,
        traces_after=traces_after,
        description=(
            "Before: A always followed by B (IMPLICATION). "
            "After: A sometimes without B (INDEPENDENCE). "
            "Bose sees 'S' (sometimes follows) in both cases. "
            "ARM detects IMPLICATION -> INDEPENDENCE change."
        ),
        expected_arm_detection=traces_before,
        expected_bose_detection=None,  # Bose should NOT detect this
    )


def generate_temporal_directness_drift_log(
    path: Path,
    traces_before: int = 50,
    traces_after: int = 50,
) -> DriftScenario:
    """Generate log with DIRECT -> EVENTUAL drift.

    Before drift: A -> B (DIRECT, consecutive)
    After drift: A -> C -> B (EVENTUAL, not consecutive)

    Bose sees "A followed by B" in both cases (succession exists).
    ARM distinguishes DIRECT (consecutive) from EVENTUAL (gap).
    """
    log = _create_xes_log()

    # Before drift: A directly followed by B
    for i in range(traces_before):
        _add_trace(log, str(i), ["A", "B", "C"])

    # After drift: A followed by B with C in between (EVENTUAL)
    for i in range(traces_after):
        trace_idx = traces_before + i
        _add_trace(log, str(trace_idx), ["A", "C", "B"])  # C between A and B

    _write_xes(log, path)

    return DriftScenario(
        name="temporal_direct_to_eventual",
        drift_type=DriftType.TEMPORAL_DIRECTNESS,
        drift_point=traces_before,
        traces_before=traces_before,
        traces_after=traces_after,
        description=(
            "Before: A directly followed by B (DIRECT). "
            "After: A followed by C then B (EVENTUAL). "
            "Bose sees succession in both. "
            "ARM detects DIRECT -> EVENTUAL change."
        ),
        expected_arm_detection=traces_before,
        expected_bose_detection=None,  # Subtle change, Bose may miss
    )


def generate_combined_drift_log(
    path: Path,
    traces_before: int = 50,
    traces_after: int = 50,
) -> DriftScenario:
    """Generate log with multiple dependency changes.

    Before: A -> B -> C (linear, DIRECT transitions, IMPLICATION)
    After: A -> (B or C) -> D (parallel choice, EVENTUAL, NAND)

    Multiple ARM dimensions change:
    - Temporal: DIRECT -> EVENTUAL (A to B/C)
    - Existential: IMPLICATION -> NAND (B vs C mutually exclusive)
    - Structure: D added
    """
    log = _create_xes_log()

    # Before drift: Linear sequence A -> B -> C
    for i in range(traces_before):
        _add_trace(log, str(i), ["A", "B", "C"])

    # After drift: A -> choice(B, C) -> D
    for i in range(traces_after):
        trace_idx = traces_before + i
        if i % 2 == 0:
            _add_trace(log, str(trace_idx), ["A", "B", "D"])
        else:
            _add_trace(log, str(trace_idx), ["A", "C", "D"])

    _write_xes(log, path)

    return DriftScenario(
        name="combined_multiple_changes",
        drift_type=DriftType.COMBINED,
        drift_point=traces_before,
        traces_before=traces_before,
        traces_after=traces_after,
        description=(
            "Before: A -> B -> C (linear). "
            "After: A -> choice(B,C) -> D. "
            "Multiple ARM changes: temporal, existential, structure. "
            "ARM should detect earlier due to richer signals."
        ),
        expected_arm_detection=traces_before,
        expected_bose_detection=traces_before + 10,  # Bose detects late
    )


def generate_subtle_implication_drift_log(
    path: Path,
    traces_before: int = 50,
    traces_after: int = 50,
) -> DriftScenario:
    """Generate log with subtle implication pattern change.

    Before: A always implies B AND C (EQUIVALENCE-like)
    After: A implies B OR C but not both (XOR)

    Both have similar succession patterns, but existential differs.
    """
    log = _create_xes_log()

    # Before drift: A always with both B and C
    for i in range(traces_before):
        _add_trace(log, str(i), ["A", "B", "C"])

    # After drift: A with exactly one of B or C (XOR pattern)
    for i in range(traces_after):
        trace_idx = traces_before + i
        if i % 2 == 0:
            _add_trace(log, str(trace_idx), ["A", "B"])  # Only B
        else:
            _add_trace(log, str(trace_idx), ["A", "C"])  # Only C

    _write_xes(log, path)

    return DriftScenario(
        name="subtle_implication_to_xor",
        drift_type=DriftType.EXISTENTIAL,
        drift_point=traces_before,
        traces_before=traces_before,
        traces_after=traces_after,
        description=(
            "Before: A always with both B and C. "
            "After: A with exactly one of B or C (XOR). "
            "Succession patterns similar, existential changes. "
            "ARM detects OR/NAND pattern shift."
        ),
        expected_arm_detection=traces_before,
        expected_bose_detection=None,
    )


def generate_drift_log(
    drift_type: DriftType,
    output_path: Path,
    traces_before: int = 50,
    traces_after: int = 50,
) -> DriftScenario:
    """Generate drift log based on type.

    Args:
        drift_type: Type of drift to generate
        output_path: Path to write XES file
        traces_before: Traces before drift point
        traces_after: Traces after drift point

    Returns:
        DriftScenario with metadata
    """
    generators = {
        DriftType.EXISTENTIAL: generate_existential_drift_log,
        DriftType.TEMPORAL_DIRECTNESS: generate_temporal_directness_drift_log,
        DriftType.COMBINED: generate_combined_drift_log,
    }

    generator = generators.get(drift_type)
    if generator is None:
        raise ValueError(f"Unknown drift type: {drift_type}")

    return generator(output_path, traces_before, traces_after)
