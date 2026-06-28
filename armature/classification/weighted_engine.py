"""Definition-based weighted scoring classifier for ARM matrices.

Implements the CLASSIFICATION_WEIGHTS approach: compute per-feature ratios over all
ordered activity pairs, apply pre-filters for structural signatures, then resolve via
weighted dot-product scores among S / SS / LS.

Unstructured is only returned via pre-filter. It is never a candidate in weighted scoring.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import permutations

from armature.classification.result import CategoryEnum, ClassificationResult
from armature.core.dependencies import ExistentialDependency, TemporalDependency
from armature.core.matrix import Matrix

# ---------------------------------------------------------------------------
# Weights  (S / SS / LS only — U is handled exclusively by pre-filter)
# ---------------------------------------------------------------------------

WEIGHTS: dict[str, dict[str, float]] = {
    "S": {
        "temporal.direct": +3.0,
        "temporal.direct_backward": +2.0,
        "temporal.no_ordering": +2.5,
        "temporal.independence": +0.5,
        "temporal.true_eventual": +2.0,
        "temporal.true_eventual_backward": +2.0,
        "temporal.eventual": +1.0,
        "temporal.eventual_backward": +1.0,
        "existential.equivalence": +3.0,
        "existential.negated_equivalence": +2.5,
        "existential.nand": +1.5,
        "existential.implication": +1.5,
        "existential.implication_backward": +1.5,
        "existential.or": +1.0,
        "existential.independence": -3.0,
    },
    "SS": {
        "temporal.true_eventual": +2.0,
        "temporal.true_eventual_backward": +2.0,
        "temporal.eventual": +2.5,
        "temporal.eventual_backward": +2.5,
        "temporal.direct": +1.5,
        "temporal.direct_backward": +1.0,
        "temporal.independence": +2.0,
        "temporal.no_ordering": +0.5,
        "existential.independence": +3.0,
        "existential.or": +1.5,
        "existential.negated_equivalence": +1.5,
        "existential.implication": +1.5,
        "existential.implication_backward": +1.5,
        "existential.equivalence": +1.0,
        "existential.nand": +0.5,
    },
    "LS": {
        "temporal.independence": +3.0,
        "temporal.eventual": +1.5,
        "temporal.eventual_backward": +1.5,
        "temporal.true_eventual": +1.0,
        "temporal.true_eventual_backward": +1.0,
        "temporal.direct": -1.5,
        "temporal.direct_backward": -1.0,
        "temporal.no_ordering": 0.0,
        "existential.implication": +3.0,
        "existential.implication_backward": +3.0,
        "existential.independence": +2.5,
        "existential.or": +1.5,
        "existential.nand": +1.0,
        "existential.negated_equivalence": +1.0,
        "existential.equivalence": -1.0,
    },
}

CLASSES = ["S", "SS", "LS"]

# Pre-filter thresholds
_LS_MIN_INDEPENDENCE = 0.75
_LS_MIN_EQUIVALENCE = 0.45
_LS_MAX_DIRECT = 0.02

_S_MIN_NO_ORDERING = 0.15
_S_MIN_NEG_EQUIV = 0.05

_BORDERLINE_MARGIN = 0.2

# ---------------------------------------------------------------------------
# Ratio computation
# ---------------------------------------------------------------------------

def _compute_ratios(matrix: Matrix) -> dict[str, float]:
    """Compute per-feature ratios over all ordered activity pairs.

    Missing pairs (not stored) default to NO_ORDERING + INDEPENDENCE.
    Total denominator = n*(n-1).
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

    ratios: dict[str, float] = {}
    for t in TemporalDependency:
        ratios[f"temporal.{t.value}"] = temporal_counts[t] / total_pairs
    for e in ExistentialDependency:
        ratios[f"existential.{e.value}"] = existential_counts[e] / total_pairs
    return ratios


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def _score(ratios: dict[str, float]) -> dict[str, float]:
    return {
        cls: sum(WEIGHTS[cls].get(k, 0.0) * v for k, v in ratios.items())
        for cls in CLASSES
    }


def classify_matrix(matrix: Matrix) -> ClassificationResult:
    """Classify ARM matrix using definition-based weighted scoring.

    Args:
        matrix: ARM matrix to classify

    Returns:
        ClassificationResult with category, scores, ratios, and trace

    Raises:
        ValueError: If matrix has no activity pairs
    """
    activities = matrix.activities
    n = len(activities)

    if n == 0:
        raise ValueError("Cannot classify empty matrix - no activities found")

    ratios = _compute_ratios(matrix)

    if not ratios:
        raise ValueError("Cannot classify matrix with a single activity - no pairs")

    scores = _score(ratios)
    rule_trace: list[dict] = []

    t_direct = ratios.get("temporal.direct", 0.0)

    # U pre-filter (degenerate case): all pairs have the exact same relationship.
    # When temporal AND existential are each 100% one type, no structure is definable.
    # Covers: (t_indep=1, e_equiv=1), (t_indep=1, e_indep=1), (t_no_ordering=1, e_indep=1), etc.
    max_temp = max(v for k, v in ratios.items() if k.startswith("temporal."))
    max_exist2 = max(v for k, v in ratios.items() if k.startswith("existential."))
    if max_temp == 1.0 and max_exist2 == 1.0 and t_direct == 0.0:
        # Find which types dominate
        dom_temp = next(k.split(".",1)[1] for k, v in ratios.items() if k.startswith("temporal.") and v == 1.0)
        dom_exist = next(k.split(".",1)[1] for k, v in ratios.items() if k.startswith("existential.") and v == 1.0)
        rule_trace.append({
            "decision": "prefilter_U",
            "reason": f"degenerate uniform pairs: temporal.{dom_temp}=1.0, existential.{dom_exist}=1.0",
        })
        return _make_result(CategoryEnum.UNSTRUCTURED, "exact", ratios, scores, rule_trace, n, "prefilter_U")

    t_no_ord = ratios.get("temporal.no_ordering", 0.0)

    # S pre-filter (disabled — was occasionally misattributing SS as S)
    # e_neg_equiv = ratios.get("existential.negated_equivalence", 0.0)
    # if t_no_ord > _S_MIN_NO_ORDERING and e_neg_equiv > _S_MIN_NEG_EQUIV:
    #     rule_trace.append({
    #         "decision": "prefilter_S",
    #         "reason": f"BPMN XOR signature: no_ordering ({t_no_ord:.4f}>{_S_MIN_NO_ORDERING}), "
    #                   f"negated_equivalence ({e_neg_equiv:.4f}>{_S_MIN_NEG_EQUIV})",
    #     })
    #     return _make_result(CategoryEnum.STRUCTURED, "exact", ratios, scores, rule_trace, n, "prefilter_S")

    # Weighted score among S / SS / LS
    ranked = sorted(CLASSES, key=lambda c: scores[c], reverse=True)
    predicted = ranked[0]
    margin = scores[ranked[0]] - scores[ranked[1]]
    borderline = margin < _BORDERLINE_MARGIN

    rule_trace.append({
        "decision": "weighted_score",
        "scores": {c: round(scores[c], 6) for c in CLASSES},
        "ranked": ranked,
        "margin": round(margin, 6),
        "borderline": borderline,
    })

    category_map = {
        "S": CategoryEnum.STRUCTURED,
        "SS": CategoryEnum.SEMI_STRUCTURED,
        "LS": CategoryEnum.LOOSELY_STRUCTURED,
    }
    confidence = "boundary" if borderline else "exact"
    return _make_result(category_map[predicted], confidence, ratios, scores, rule_trace, n, "weighted_score")


def _make_result(
    category: CategoryEnum,
    confidence: str,
    ratios: dict[str, float],
    scores: dict[str, float],
    rule_trace: list[dict],
    n: int,
    method: str,
) -> ClassificationResult:
    return ClassificationResult(
        category=category,
        confidence=confidence,  # type: ignore[arg-type]
        dependency_counts={},
        dependency_ratios=ratios,
        thresholds_applied={},
        rule_trace=rule_trace,
        activity_count=n,
        total_dependencies=0,
        density=0.0,
        metadata={"scores": {k: round(v, 6) for k, v in scores.items()}, "method": method},
    )
