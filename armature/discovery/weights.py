"""Weighted dependency discovery — compute W = N_D / T for every dependency type.

For each activity pair (A, B) and each dependency type D:
  - N_D = number of traces that do NOT violate the rule for D
  - T   = total traces in event log
  - W   = N_D / T  (1.0 = fully consistent, 0.0 = always violated)

See .planning/quick/008-weighted-discovery-violation-rules for the full
violation rule table for both existential and temporal types.
"""

from __future__ import annotations

from dataclasses import dataclass

from armature.discovery.models import Trace


@dataclass(frozen=True)
class ExistentialWeights:
    """Per-pair existential dependency weights.

    Each field holds N_D / T for its dependency type, where N_D is the count
    of traces not violating that type's rule.

    Violation rules (see planning doc for full table):
      equivalence          : violated when only A or only B present
      negated_equivalence  : violated when both or neither present
      nand                 : violated when both present
      implication_backward : violated when B present without A
      implication          : violated when A present without B
      or_dep               : violated when neither present
      independence         : never violated (always 1.0)
    """

    equivalence: float
    negated_equivalence: float
    nand: float
    implication_backward: float
    implication: float
    or_dep: float
    independence: float
    # Raw counts (useful for display as fractions)
    count_both: int
    count_only_a: int
    count_only_b: int
    count_neither: int
    total: int


@dataclass(frozen=True)
class TemporalWeights:
    """Per-pair temporal dependency weights.

    Each field holds N_D / T.  Violations are only counted for traces where
    BOTH activities are present; traces where neither appears are never
    considered violations.

    Violation rules (see planning doc for full table):
      direct                    : both present AND A never directly precedes B
      direct_backward           : both present AND B never directly precedes A
      eventual                  : both present AND A never appears before B
      eventual_backward         : both present AND B never appears before A
      true_eventual             : same per-trace check as eventual (structural distinction)
      true_eventual_backward    : same per-trace check as eventual_backward
      independence              : both present AND NOT (a_before_b AND b_before_a)
      no_ordering               : both present (co-occurrence itself = violation)
    """

    direct: float
    direct_backward: float
    eventual: float
    eventual_backward: float
    true_eventual: float
    true_eventual_backward: float
    independence: float
    no_ordering: float
    total: int


@dataclass(frozen=True)
class PairWeights:
    """All dependency weights for a single activity pair (source, target)."""

    source: str
    target: str
    existential: ExistentialWeights
    temporal: TemporalWeights


def compute_weights(traces: list[Trace]) -> dict[tuple[str, str], PairWeights]:
    """Compute dependency weights for all non-diagonal activity pairs.

    Args:
        traces: List of process traces parsed from an XES event log.

    Returns:
        Dict mapping (source, target) -> PairWeights for every pair where
        source != target.  Diagonal (self) pairs are excluded.
    """
    # Collect all activities
    all_activities: set[str] = set()
    for trace in traces:
        for event in trace.events:
            all_activities.add(event.activity)

    activities = sorted(all_activities)
    T = len(traces)

    if T == 0:
        return {}

    result: dict[tuple[str, str], PairWeights] = {}

    for act_a in activities:
        for act_b in activities:
            if act_a == act_b:
                continue

            # ── existential counters ──────────────────────────────────────
            count_both = 0
            count_only_a = 0
            count_only_b = 0
            count_neither = 0

            # ── temporal violation counters (traces where rule IS violated) ─
            v_direct = 0
            v_direct_bk = 0
            v_eventual = 0
            v_eventual_bk = 0
            v_true_eventual = 0
            v_true_eventual_bk = 0
            v_independence = 0
            v_no_ordering = 0

            for trace in traces:
                acts = [e.activity for e in trace.events]
                act_set = set(acts)

                has_a = act_a in act_set
                has_b = act_b in act_set

                # ── existential ──────────────────────────────────────────
                if has_a and has_b:
                    count_both += 1
                elif has_a:
                    count_only_a += 1
                elif has_b:
                    count_only_b += 1
                else:
                    count_neither += 1

                # ── temporal (only when both present) ────────────────────
                if not (has_a and has_b):
                    continue

                # Positions of each activity in this trace
                a_pos = [i for i, e in enumerate(acts) if e == act_a]
                b_pos = [i for i, e in enumerate(acts) if e == act_b]

                # Any A before any B?
                a_before_b = min(a_pos) < max(b_pos)
                # Any B before any A?
                b_before_a = min(b_pos) < max(a_pos)

                # A directly precedes B (consecutive)?
                a_direct_b = any(
                    acts[k] == act_a and acts[k + 1] == act_b
                    for k in range(len(acts) - 1)
                )
                # B directly precedes A (consecutive)?
                b_direct_a = any(
                    acts[k] == act_b and acts[k + 1] == act_a
                    for k in range(len(acts) - 1)
                )

                if not a_direct_b:
                    v_direct += 1
                if not b_direct_a:
                    v_direct_bk += 1
                if not a_before_b:
                    v_eventual += 1
                    v_true_eventual += 1
                if not b_before_a:
                    v_eventual_bk += 1
                    v_true_eventual_bk += 1
                if not (a_before_b and b_before_a):
                    v_independence += 1
                # NO_ORDERING is violated whenever both are present
                v_no_ordering += 1

            # ── build weight objects ──────────────────────────────────────
            ew = ExistentialWeights(
                equivalence=(count_both + count_neither) / T,
                negated_equivalence=(count_only_a + count_only_b) / T,
                nand=(count_only_a + count_only_b + count_neither) / T,
                implication_backward=(T - count_only_b) / T,
                implication=(T - count_only_a) / T,
                or_dep=(T - count_neither) / T,
                independence=1.0,
                count_both=count_both,
                count_only_a=count_only_a,
                count_only_b=count_only_b,
                count_neither=count_neither,
                total=T,
            )

            tw = TemporalWeights(
                direct=(T - v_direct) / T,
                direct_backward=(T - v_direct_bk) / T,
                eventual=(T - v_eventual) / T,
                eventual_backward=(T - v_eventual_bk) / T,
                true_eventual=(T - v_true_eventual) / T,
                true_eventual_backward=(T - v_true_eventual_bk) / T,
                independence=(T - v_independence) / T,
                no_ordering=(T - v_no_ordering) / T,
                total=T,
            )

            result[(act_a, act_b)] = PairWeights(
                source=act_a,
                target=act_b,
                existential=ew,
                temporal=tw,
            )

    return result
