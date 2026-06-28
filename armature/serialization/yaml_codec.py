"""YAML codec for Matrix serialization and deserialization.

Format (v2.0):
    metadata:
      format_version: '2.0'
      source: ...
      description: ...
      ...
    activities:
    - A
    - B
    dependencies:
    - from: A
      to: B
      temporal:
        type: direct
        symbol: ≺_d
        direction: forward
      existential:
        type: implication
        symbol: ⇒
        direction: forward
"""

import yaml
from pathlib import Path
from typing import Union

from armature.core.dependencies import DependencyCell, ExistentialDependency, TemporalDependency
from armature.core.matrix import Matrix

# ---------------------------------------------------------------------------
# Symbol / direction tables
# ---------------------------------------------------------------------------

_TEMPORAL_INFO: dict[TemporalDependency, tuple[str, str, str]] = {
    TemporalDependency.DIRECT:                   ("direct",        "≺_d", "forward"),
    TemporalDependency.DIRECT_BACKWARD:          ("direct",        "≺_d", "backward"),
    TemporalDependency.TRUE_EVENTUAL:            ("true_eventual", "≺",   "forward"),
    TemporalDependency.TRUE_EVENTUAL_BACKWARD:   ("true_eventual", "≺",   "backward"),
    TemporalDependency.EVENTUAL:                 ("eventual",      "≺",   "forward"),
    TemporalDependency.EVENTUAL_BACKWARD:        ("eventual",      "≺",   "backward"),
    TemporalDependency.INDEPENDENCE:             ("independence",  "↔",   "both"),
    TemporalDependency.NO_ORDERING:              ("no_ordering",   "⊘",   "none"),
}

_EXISTENTIAL_INFO: dict[ExistentialDependency, tuple[str, str, str]] = {
    ExistentialDependency.IMPLICATION:           ("implication",          "⇒", "forward"),
    ExistentialDependency.IMPLICATION_BACKWARD:  ("implication",          "⇒", "backward"),
    ExistentialDependency.EQUIVALENCE:           ("equivalence",          "⇔", "both"),
    ExistentialDependency.NEGATED_EQUIVALENCE:   ("negated_equivalence",  "⊕", "both"),
    ExistentialDependency.OR:                    ("or",                   "∨", "both"),
    ExistentialDependency.NAND:                  ("nand",                 "↑", "both"),
    ExistentialDependency.INDEPENDENCE:          ("independence",         "−", "both"),
}

# Reverse lookups for load()
_TEMPORAL_FROM_TYPE_DIR: dict[tuple[str, str], TemporalDependency] = {
    ("direct",        "forward"):  TemporalDependency.DIRECT,
    ("direct",        "backward"): TemporalDependency.DIRECT_BACKWARD,
    ("true_eventual", "forward"):  TemporalDependency.TRUE_EVENTUAL,
    ("true_eventual", "backward"): TemporalDependency.TRUE_EVENTUAL_BACKWARD,
    ("eventual",      "forward"):  TemporalDependency.EVENTUAL,
    ("eventual",      "backward"): TemporalDependency.EVENTUAL_BACKWARD,
    ("independence",  "both"):     TemporalDependency.INDEPENDENCE,
    ("no_ordering",   "none"):     TemporalDependency.NO_ORDERING,
}

_EXISTENTIAL_FROM_TYPE_DIR: dict[tuple[str, str], ExistentialDependency] = {
    ("implication",         "forward"):  ExistentialDependency.IMPLICATION,
    ("implication",         "backward"): ExistentialDependency.IMPLICATION_BACKWARD,
    ("equivalence",         "both"):     ExistentialDependency.EQUIVALENCE,
    ("negated_equivalence", "both"):     ExistentialDependency.NEGATED_EQUIVALENCE,
    ("or",                  "both"):     ExistentialDependency.OR,
    ("nand",                "both"):     ExistentialDependency.NAND,
    ("independence",        "both"):     ExistentialDependency.INDEPENDENCE,
}


def _serialize_temporal(t: TemporalDependency) -> dict:
    typ, symbol, direction = _TEMPORAL_INFO[t]
    return {"type": typ, "symbol": symbol, "direction": direction}


def _serialize_existential(e: ExistentialDependency) -> dict:
    typ, symbol, direction = _EXISTENTIAL_INFO[e]
    return {"type": typ, "symbol": symbol, "direction": direction}


def _parse_temporal(data: dict) -> TemporalDependency:
    key = (data["type"], data["direction"])
    t = _TEMPORAL_FROM_TYPE_DIR.get(key)
    if t is None:
        raise ValueError(f"Unknown temporal (type={data['type']!r}, direction={data['direction']!r})")
    return t


def _parse_existential(data: dict) -> ExistentialDependency:
    key = (data["type"], data["direction"])
    e = _EXISTENTIAL_FROM_TYPE_DIR.get(key)
    if e is None:
        raise ValueError(f"Unknown existential (type={data['type']!r}, direction={data['direction']!r})")
    return e


class YAMLCodec:
    """Matrix ↔ YAML serialization with round-trip compatibility.

    Produces human-readable block-style YAML with explicit type, symbol, and
    direction fields for each dependency — matching the format used by the
    INSM business-process-redesign sample matrices.
    """

    @staticmethod
    def save(matrix: Matrix, path: Union[str, Path]) -> None:
        """Save Matrix to YAML file.

        Args:
            matrix: Matrix instance to serialize
            path: File path for YAML output
        """
        # Build metadata block (only include non-empty fields)
        metadata: dict = {"format_version": matrix.format_version}
        for field in ("description", "source", "created_at", "classification",
                      "num_traces", "num_variants"):
            val = getattr(matrix, field, None)
            if val is not None and val != "":
                metadata[field] = val

        # Build flat dependency list
        dep_list = []
        for source, targets in matrix.dependencies.items():
            for target, cell in targets.items():
                dep_list.append({
                    "from": source,
                    "to": target,
                    "temporal": _serialize_temporal(cell.temporal),
                    "existential": _serialize_existential(cell.existential),
                })

        data = {
            "metadata": metadata,
            "activities": list(matrix.activities),
            "dependencies": dep_list,
        }

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(
                data,
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
                indent=2,
            )

    @staticmethod
    def load(path: Union[str, Path]) -> Matrix:
        """Load Matrix from YAML file.

        Supports both the current format (metadata wrapper + flat dependency list)
        and the legacy format (flat top-level + nested dependency dict).

        Args:
            path: File path to YAML file

        Returns:
            Matrix instance

        Raises:
            FileNotFoundError: If file doesn't exist
            yaml.YAMLError: If YAML parsing fails
            ValidationError: If structure doesn't match Matrix schema
        """
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if "metadata" in data:
            return YAMLCodec._load_new_format(data)
        else:
            return YAMLCodec._load_legacy_format(data)

    @staticmethod
    def _load_new_format(data: dict) -> Matrix:
        """Parse new format: metadata wrapper + flat dependency list."""
        meta = data.get("metadata", {})

        matrix = Matrix(
            format_version=meta.get("format_version", "2.0"),
            description=meta.get("description", ""),
            source=meta.get("source"),
            created_at=meta.get("created_at"),
            classification=meta.get("classification"),
            num_traces=meta.get("num_traces"),
            num_variants=meta.get("num_variants"),
            activities=list(data.get("activities") or []),
        )

        for dep in data.get("dependencies") or []:
            source = dep["from"]
            target = dep["to"]
            temporal = _parse_temporal(dep["temporal"])
            existential = _parse_existential(dep["existential"])
            matrix.set_cell(source, target, DependencyCell(temporal=temporal, existential=existential))

        return matrix

    @staticmethod
    def _load_legacy_format(data: dict) -> Matrix:
        """Parse legacy format: flat top-level + nested dependency dict."""
        # Use model_validate for the old pydantic-dumped format
        return Matrix.model_validate(data)
