"""Compute ARM percentages for synthetic logs defined as variant strings.

Usage: python scripts/compute_synthetic_ratios.py

Each log is defined as a list of (variant_activities, count) tuples where
variant_activities is a list of single-letter activity names.
Generates a temporary XES file, runs discovery, computes percentages,
and prints a formatted summary.
"""

from __future__ import annotations

import sys
import tempfile
import textwrap
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from armature.classification.percentages import CalculatedPercentages
from armature.discovery.discover import discover


# ---------------------------------------------------------------------------
# XES generation from variant list
# ---------------------------------------------------------------------------


def build_xes(variants: list[tuple[list[str], int]]) -> str:
    """Generate a minimal XES string from a list of (activities, count) variants."""
    base_time = datetime(2024, 1, 1, 8, 0, 0)
    lines = [
        '<?xml version="1.0" encoding="UTF-8" ?>',
        '<log xes.version="1.0" xmlns="http://www.xes-standard.org">',
    ]
    case_id = 0
    for activities, count in variants:
        for _ in range(count):
            lines.append("  <trace>")
            lines.append(f'    <string key="concept:name" value="case_{case_id}"/>')
            t = base_time
            for act in activities:
                ts = t.strftime("%Y-%m-%dT%H:%M:%S.000+00:00")
                lines.append("    <event>")
                lines.append(f'      <string key="concept:name" value="{act}"/>')
                lines.append(f'      <date key="time:timestamp" value="{ts}"/>')
                lines.append("    </event>")
                t += timedelta(minutes=5)
            lines.append("  </trace>")
            case_id += 1
    lines.append("</log>")
    return "\n".join(lines)


def compute(name: str, variants: list[tuple[list[str], int]]) -> dict:
    """Run full pipeline on a variant set; return percentages dict."""
    xes_content = build_xes(variants)
    with tempfile.NamedTemporaryFile(suffix=".xes", mode="w", delete=False) as f:
        f.write(xes_content)
        tmp_path = Path(f.name)
    try:
        matrix = discover(tmp_path)
        pct = CalculatedPercentages.from_matrix(matrix)
        total_traces = sum(c for _, c in variants)
        total_variants = len(variants)
        return {
            "name": name,
            "total_traces": total_traces,
            "total_variants": total_variants,
            "activities": sorted(matrix.activities),
            "independence_none": round(pct.independence_none, 4),
            "no_ordering_none": round(pct.no_ordering_none, 4),
            "none_none": round(pct.none_none, 4),
            "none_implication": round(pct.none_implication, 4),
            "none_equivalence": round(pct.none_equivalence, 4),
            "none_negated_equivalence": round(pct.none_negated_equivalence, 4),
            "eventual_equivalence": round(pct.eventual_equivalence, 4),
            "eventual_implication": round(pct.eventual_implication, 4),
            "eventual_any_existential": round(pct.eventual_any_existential, 4),
            "eventual_or": round(pct.eventual_or, 4),
            "direct_any_existential": round(pct.direct_any_existential, 4),
            "direct_none": round(pct.direct_none, 4),
            "true_eventual_ratio": round(pct.true_eventual_ratio, 4),
        }
    finally:
        tmp_path.unlink(missing_ok=True)


def fmt(r: dict) -> str:
    return textwrap.dedent(f"""\
        ### {r["name"]}
        - activities      : {", ".join(r["activities"])}
        - total_traces    : {r["total_traces"]}
        - total_variants  : {r["total_variants"]}
        - independence_none        : {r["independence_none"]}
        - no_ordering_none         : {r["no_ordering_none"]}
        - none_none (sum)          : {r["none_none"]}
        - none_implication         : {r["none_implication"]}
        - none_equivalence         : {r["none_equivalence"]}
        - none_negated_equivalence : {r["none_negated_equivalence"]}
        - eventual_equivalence     : {r["eventual_equivalence"]}
        - eventual_implication     : {r["eventual_implication"]}
        - eventual_any_existential : {r["eventual_any_existential"]}
        - eventual_or              : {r["eventual_or"]}
        - direct_any_existential   : {r["direct_any_existential"]}
        - direct_none              : {r["direct_none"]}
        - true_eventual_ratio      : {r["true_eventual_ratio"]}
    """)


# ---------------------------------------------------------------------------
# Log definitions — designed completely independently, no reference to
# existing granular_rules thresholds or previous logs
# ---------------------------------------------------------------------------

LOGS: list[tuple[str, list[tuple[list[str], int]]]] = [
    # -----------------------------------------------------------------------
    # UNSTRUCTURED LOGS
    # All permutations of activities appear equally often.
    # Zero ordering constraints — every pair seen in both orders.
    # -----------------------------------------------------------------------
    # U1: 3 activities — all 6 permutations × 10 traces
    (
        "U1_unstructured_3act",
        [
            (["a", "b", "c"], 10),
            (["a", "c", "b"], 10),
            (["b", "a", "c"], 10),
            (["b", "c", "a"], 10),
            (["c", "a", "b"], 10),
            (["c", "b", "a"], 10),
        ],
    ),
    # U2: 5 activities — all 120 permutations × 1 trace each
    # (all 5! = 120 permutations)
    (
        "U2_unstructured_5act",
        [
            ([a, b, c, d, e], 1)
            for a in ["a", "b", "c", "d", "e"]
            for b in ["a", "b", "c", "d", "e"]
            if b != a
            for c in ["a", "b", "c", "d", "e"]
            if c not in (a, b)
            for d in ["a", "b", "c", "d", "e"]
            if d not in (a, b, c)
            for e in ["a", "b", "c", "d", "e"]
            if e not in (a, b, c, d)
        ],
    ),
    # -----------------------------------------------------------------------
    # STRUCTURED LOGS
    # -----------------------------------------------------------------------
    # S1: Pure strict sequence — a→b→c→d→e, all traces identical
    (
        "S1_strict_sequence",
        [
            (["a", "b", "c", "d", "e"], 50),
        ],
    ),
    # S2: Sequence with AND-parallel middle block
    # Flow: a → (b ‖ c) → d → e
    # b and c always both happen, can interleave: aba-c or a-c-b
    (
        "S2_parallel_block",
        [
            (["a", "b", "c", "d", "e"], 25),
            (["a", "c", "b", "d", "e"], 25),
        ],
    ),
    # S3: XOR split — two mutually exclusive paths
    # Flow: a → b → (c XOR d) → e
    # c and d never co-occur
    (
        "S3_xor_two_paths",
        [
            (["a", "b", "c", "e"], 25),
            (["a", "b", "d", "e"], 25),
        ],
    ),
    # S4: Three-way XOR — three exclusive paths of different lengths
    # Flow: a → (b→c XOR d XOR e→f→g) → h
    (
        "S4_xor_three_paths",
        [
            (["a", "b", "c", "h"], 20),
            (["a", "d", "h"], 20),
            (["a", "e", "f", "g", "h"], 20),
        ],
    ),
    # S5: Single-entry single-exit loop
    # Flow: a → [b → (c → back-to-b | exit)] → d
    # b and c form the loop body; loop repeats 0–2 extra times
    (
        "S5_sese_loop",
        [
            (["a", "b", "d"], 20),  # loop body once, exit immediately
            (["a", "b", "c", "b", "d"], 20),  # one extra iteration
            (["a", "b", "c", "b", "c", "b", "d"], 10),  # two extra iterations
        ],
    ),
    # S6: AND-parallel with one optional branch via XOR skip
    # Flow: a → (b ‖ c) → (d XOR skip) → e
    # d appears in ~60% of traces
    (
        "S6_parallel_optional",
        [
            (["a", "b", "c", "d", "e"], 15),
            (["a", "c", "b", "d", "e"], 15),
            (["a", "b", "c", "e"], 10),
            (["a", "c", "b", "e"], 10),
        ],
    ),
    # S7: Multiple instances of one activity (self-loop)
    # Flow: a → b(×1-3) → c → d
    # b appears 1, 2, or 3 times consecutively
    (
        "S7_multi_instance",
        [
            (["a", "b", "c", "d"], 20),
            (["a", "b", "b", "c", "d"], 20),
            (["a", "b", "b", "b", "c", "d"], 10),
        ],
    ),
    # S8: Nested parallel inside XOR
    # Flow: a → b → (XOR: path1=(c→d) | path2=(e‖f)→g) → h
    # path2's e and f are concurrent (AND)
    (
        "S8_nested_parallel_xor",
        [
            (["a", "b", "c", "d", "h"], 20),  # XOR path 1
            (["a", "b", "e", "f", "g", "h"], 20),  # XOR path 2, e first
            (["a", "b", "f", "e", "g", "h"], 20),  # XOR path 2, f first
        ],
    ),
    # S9: Loop with AND-parallel body (single-entry single-exit)
    # Flow: a → [b ‖ c → (exit | back)] → d
    # Each iteration: b and c run concurrently; loop repeats 1–2 times
    (
        "S9_loop_with_parallel_body",
        [
            (["a", "b", "c", "d"], 10),  # one iteration, b first
            (["a", "c", "b", "d"], 10),  # one iteration, c first
            (["a", "b", "c", "b", "c", "d"], 10),  # two iterations, b first both
            (["a", "b", "c", "c", "b", "d"], 10),  # two iterations mixed
            (["a", "c", "b", "b", "c", "d"], 10),  # two iterations mixed
            (["a", "c", "b", "c", "b", "d"], 10),  # two iterations, c first both
        ],
    ),
    # S10: Everything combined — AND + XOR + loop + optional
    # Flow: a → (b ‖ c) → d → (e→[f→(g | back-to-f)] XOR h) → (i XOR skip) → j
    # b,c parallel; d always; XOR to loop path (e,f,g with f looping) or h;
    # optional i; j always at end
    (
        "S10_full_coverage_1",
        [
            # loop path, no loop repeat, with optional i
            (["a", "b", "c", "d", "e", "f", "g", "i", "j"], 10),
            (["a", "c", "b", "d", "e", "f", "g", "i", "j"], 10),
            # loop path, one loop repeat, with optional i
            (["a", "b", "c", "d", "e", "f", "f", "g", "i", "j"], 8),
            (["a", "c", "b", "d", "e", "f", "f", "g", "i", "j"], 8),
            # loop path, no loop repeat, skip i
            (["a", "b", "c", "d", "e", "f", "g", "j"], 8),
            (["a", "c", "b", "d", "e", "f", "g", "j"], 8),
            # h path (XOR alternative), with optional i
            (["a", "b", "c", "d", "h", "i", "j"], 10),
            (["a", "c", "b", "d", "h", "i", "j"], 10),
            # h path, skip i
            (["a", "b", "c", "d", "h", "j"], 10),
            (["a", "c", "b", "d", "h", "j"], 10),
        ],
    ),
    # S11: Everything combined v2 — longer, more activities, multi-instance + loop
    # Flow: a → (b→c ‖ d) → e → [f(×1-2) → (g | back)] → (h ‖ i) → (j XOR skip) → k
    # b,c sequential in one branch; d in the other (parallel); e always;
    # f multi-instance loop with g exit; h,i parallel; optional j; k always
    (
        "S11_full_coverage_2",
        [
            # f once, g exit, h/i both, with j
            (["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"], 8),
            (["a", "b", "c", "d", "e", "f", "g", "i", "h", "j", "k"], 8),
            (["a", "d", "b", "c", "e", "f", "g", "h", "i", "j", "k"], 8),
            (["a", "d", "b", "c", "e", "f", "g", "i", "h", "j", "k"], 8),
            # f twice, g exit, h/i both, with j
            (["a", "b", "c", "d", "e", "f", "f", "g", "h", "i", "j", "k"], 5),
            (["a", "b", "c", "d", "e", "f", "f", "g", "i", "h", "j", "k"], 5),
            (["a", "d", "b", "c", "e", "f", "f", "g", "h", "i", "j", "k"], 5),
            (["a", "d", "b", "c", "e", "f", "f", "g", "i", "h", "j", "k"], 5),
            # f once, g exit, h/i both, skip j
            (["a", "b", "c", "d", "e", "f", "g", "h", "i", "k"], 6),
            (["a", "b", "c", "d", "e", "f", "g", "i", "h", "k"], 6),
            (["a", "d", "b", "c", "e", "f", "g", "h", "i", "k"], 6),
            (["a", "d", "b", "c", "e", "f", "g", "i", "h", "k"], 6),
        ],
    ),
    # S12: XOR with nested AND on one path, self-loop on the other, plus optional
    # Flow: a → b → XOR[ (c‖d → e) | (f⁺ → g) ] → optional(h) → i
    # XOR path 1: c and d concurrent, then e
    # XOR path 2: f repeated 1-2 times (self-loop), then g
    # optional h before i
    (
        "S12_xor_nested_and_selfloop_optional",
        [
            (["a", "b", "c", "d", "e", "h", "i"], 10),  # path1, c first, with h
            (["a", "b", "d", "c", "e", "h", "i"], 10),  # path1, d first, with h
            (["a", "b", "c", "d", "e", "i"], 8),  # path1, c first, skip h
            (["a", "b", "d", "c", "e", "i"], 8),  # path1, d first, skip h
            (["a", "b", "f", "g", "h", "i"], 10),  # path2, f×1, with h
            (["a", "b", "f", "f", "g", "h", "i"], 8),  # path2, f×2, with h
            (["a", "b", "f", "g", "i"], 8),  # path2, f×1, skip h
            (["a", "b", "f", "f", "g", "i"], 8),  # path2, f×2, skip h
        ],
    ),
]


if __name__ == "__main__":
    results = []
    for name, variants in LOGS:
        print(f"Processing {name}...", file=sys.stderr)
        r = compute(name, variants)
        results.append(r)
        print(fmt(r))
