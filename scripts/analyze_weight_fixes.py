#!/usr/bin/env python3
"""Analyze misclassifications and suggest weight changes to fix them.

For each misclassified log: computes per-feature delta contributions (what's
pushing toward wrong class vs. truth), groups by confusion pair, and suggests
minimal weight adjustments with justification.

Also analyzes unstructured pre-filter candidates.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from itertools import permutations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from armature.core.dependencies import ExistentialDependency, TemporalDependency
from armature.discovery.discover import discover

WEIGHTS: dict[str, dict[str, float]] = {
    "S": {
        "temporal.direct": +3.0, "temporal.direct_backward": +2.0,
        "temporal.no_ordering": +2.5, "temporal.independence": +0.5,
        "temporal.true_eventual": +1.0, "temporal.true_eventual_backward": +1.0,
        "temporal.eventual": +0.5, "temporal.eventual_backward": +0.5,
        "existential.equivalence": +3.0, "existential.negated_equivalence": +2.5,
        "existential.nand": +1.5, "existential.implication": +1.5,
        "existential.implication_backward": +1.5, "existential.or": +1.0,
        "existential.independence": -3.0,
    },
    "SS": {
        "temporal.true_eventual": +2.5, "temporal.true_eventual_backward": +2.5,
        "temporal.eventual": +2.5, "temporal.eventual_backward": +2.5,
        "temporal.direct": +1.5, "temporal.direct_backward": +1.0,
        "temporal.independence": +2.0, "temporal.no_ordering": +0.5,
        "existential.independence": +3.0,
        "existential.or": +1.5, "existential.negated_equivalence": +1.5,
        "existential.implication": +1.5, "existential.implication_backward": +1.5,
        "existential.equivalence": +1.0, "existential.nand": +0.5,
    },
    "LS": {
        "temporal.independence": +3.0, "temporal.eventual": +1.5,
        "temporal.true_eventual": +1.0, "temporal.direct": -1.5,
        "temporal.direct_backward": -1.0, "temporal.no_ordering": 0.0,
        "existential.implication": +3.0, "existential.implication_backward": +3.0,
        "existential.independence": +2.5, "existential.or": +1.5,
        "existential.nand": +1.0, "existential.negated_equivalence": +1.0,
        "existential.equivalence": -1.0,
    },
    "U": {
        "temporal.no_ordering": +3.0, "temporal.independence": +2.0,
        "temporal.direct": -2.0, "temporal.true_eventual": -1.5,
        "temporal.eventual": -1.0,
        "existential.equivalence": -2.0, "existential.implication": -1.5,
        "existential.implication_backward": -1.5, "existential.independence": -1.0,
        "existential.nand": -0.5, "existential.negated_equivalence": -0.5,
        "existential.or": -0.5,
    },
}

CLASSES = ["S", "SS", "LS", "U"]
FEAT_ORDER = (
    [f"temporal.{t.value}" for t in TemporalDependency]
    + [f"existential.{e.value}" for e in ExistentialDependency]
)


def compute_ratios(path: Path) -> dict[str, float]:
    matrix = discover(path)
    activities = matrix.activities
    n = len(activities)
    total_pairs = n * (n - 1)
    if total_pairs == 0:
        return {}
    tc: dict[TemporalDependency, int] = defaultdict(int)
    ec: dict[ExistentialDependency, int] = defaultdict(int)
    stored = matrix.dependencies
    for a, b in permutations(activities, 2):
        cell = stored.get(a, {}).get(b)
        if cell is not None:
            tc[cell.temporal] += 1
            ec[cell.existential] += 1
        else:
            tc[TemporalDependency.NO_ORDERING] += 1
            ec[ExistentialDependency.INDEPENDENCE] += 1
    ratios: dict[str, float] = {}
    for t in TemporalDependency:
        ratios[f"temporal.{t.value}"] = tc[t] / total_pairs
    for e in ExistentialDependency:
        ratios[f"existential.{e.value}"] = ec[e] / total_pairs
    return ratios


def classify(ratios: dict[str, float]) -> tuple[dict[str, float], str]:
    scores = {c: sum(WEIGHTS[c].get(k, 0.0) * v for k, v in ratios.items()) for c in CLASSES}
    return scores, max(scores, key=lambda c: scores[c])


def score_gap(scores: dict[str, float], truth: str, pred: str) -> float:
    return scores[pred] - scores[truth]


def feature_deltas(ratios: dict[str, float], pred: str, truth: str) -> list[tuple[str, float, float, float, float]]:
    """
    Returns list of (feat, ratio, contrib_pred, contrib_truth, delta=contrib_pred-contrib_truth)
    sorted by abs(delta) desc. Only nonzero ratio features.
    """
    rows = []
    for feat in FEAT_ORDER:
        r = ratios.get(feat, 0.0)
        if r == 0.0:
            continue
        cp = WEIGHTS[pred].get(feat, 0.0) * r
        ct = WEIGHTS[truth].get(feat, 0.0) * r
        rows.append((feat, r, cp, ct, cp - ct))
    rows.sort(key=lambda x: abs(x[4]), reverse=True)
    return rows


def main():
    base = Path("Synthetic Log Data")
    groups = [
        ("structuredLogs", "S"),
        ("semiStructuredLogs", "SS"),
        ("looselyStructuredLogs", "LS"),
        ("unstructuredLogs", "U"),
    ]

    all_results: list[dict] = []
    for dir_name, label in groups:
        for path in sorted((base / dir_name).glob("*.xes")):
            ratios = compute_ratios(path)
            scores, predicted = classify(ratios)
            all_results.append({
                "name": path.stem, "truth": label, "predicted": predicted,
                "correct": predicted == label, "scores": scores, "ratios": ratios,
            })

    correct = [r for r in all_results if r["correct"]]
    wrong   = [r for r in all_results if not r["correct"]]

    print(f"=== ACCURACY: {len(correct)}/{len(all_results)} ({len(correct)/len(all_results)*100:.1f}%) ===\n")

    # -----------------------------------------------------------------------
    # Section 1: Unstructured pre-filter analysis
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("SECTION 1: UNSTRUCTURED PRE-FILTER ANALYSIS")
    print("=" * 70)

    print("\n--- Feature signatures for ALL logs (key U-relevant features) ---")
    print(f"{'Log':<12} {'truth':<5} {'pred':<5} {'t.no_ord':>9} {'t.indep':>9} {'t.direct':>9} {'e.indep':>9} {'e.equiv':>9} {'e.std':>7}")
    print("-" * 80)

    for r in sorted(all_results, key=lambda x: (x["truth"], x["name"])):
        ra = r["ratios"]
        e_vals = [ra.get(f"existential.{e.value}", 0.0) for e in ExistentialDependency]
        import statistics
        e_std = statistics.stdev(e_vals) if len(e_vals) > 1 else 0
        e_max = max(e_vals)
        marker = " <-- U" if r["truth"] == "U" else (" WRONG" if not r["correct"] else "")
        print(
            f"{r['name']:<12} {r['truth']:<5} {r['predicted']:<5}"
            f" {ra.get('temporal.no_ordering', 0):>9.4f}"
            f" {ra.get('temporal.independence', 0):>9.4f}"
            f" {ra.get('temporal.direct', 0):>9.4f}"
            f" {ra.get('existential.independence', 0):>9.4f}"
            f" {ra.get('existential.equivalence', 0):>9.4f}"
            f" {e_std:>7.4f}"
            f"{marker}"
        )

    print("\n--- Max existential ratio per log (U should have low max = flat distribution) ---")
    print(f"{'Log':<12} {'truth':<5} {'pred':<5} {'max_exist':>10} {'max_exist_feat':<30} {'e.std':>7}")
    print("-" * 70)
    for r in sorted(all_results, key=lambda x: (x["truth"], x["name"])):
        ra = r["ratios"]
        e_vals = {f"existential.{e.value}": ra.get(f"existential.{e.value}", 0.0) for e in ExistentialDependency}
        max_feat = max(e_vals, key=e_vals.get)
        max_val = e_vals[max_feat]
        e_std = statistics.stdev(list(e_vals.values()))
        marker = " <-- U" if r["truth"] == "U" else (" WRONG" if not r["correct"] else "")
        print(f"{r['name']:<12} {r['truth']:<5} {r['predicted']:<5} {max_val:>10.4f} {max_feat:<30} {e_std:>7.4f}{marker}")

    print("\n--- U scores breakdown ---")
    for r in [r for r in all_results if r["truth"] == "U"]:
        print(f"\n{r['name']} (truth=U, pred={r['predicted']})")
        print(f"  Scores: " + " | ".join(f"{c}={r['scores'][c]:+.4f}" for c in CLASSES))
        print(f"  U score gap (U - pred): {r['scores']['U'] - r['scores'][r['predicted']]:+.4f}")
        print("  Feature contributions (feat, ratio, contrib_U, contrib_pred, delta):")
        deltas = feature_deltas(r["ratios"], r["predicted"], "U")
        for feat, ratio, cp, ct, delta in deltas[:10]:
            sign = ">>> pushes WRONG" if delta > 0.005 else ("<<< pushes RIGHT" if delta < -0.005 else "")
            print(f"    {feat:<40} ratio={ratio:.4f}  c({r['predicted']})={cp:+.4f}  c(U)={ct:+.4f}  Δ={delta:+.4f}  {sign}")

    # -----------------------------------------------------------------------
    # Section 2: S → SS misclassifications
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SECTION 2: S → SS MISCLASSIFICATIONS")
    print("=" * 70)

    s_to_ss = [r for r in wrong if r["truth"] == "S" and r["predicted"] == "SS"]
    print(f"\n{len(s_to_ss)} logs: {[r['name'] for r in s_to_ss]}")

    # Compare ratios: correctly classified S vs misclassified S
    correct_s = [r for r in correct if r["truth"] == "S"]
    print(f"\nCorrectly classified S: {[r['name'] for r in correct_s]}")

    print("\n--- Score gaps (SS_score - S_score) ---")
    for r in s_to_ss:
        gap = r["scores"]["SS"] - r["scores"]["S"]
        print(f"  {r['name']}: SS={r['scores']['SS']:+.4f}, S={r['scores']['S']:+.4f}, gap={gap:+.4f}")

    print("\n--- Common features pushing S→SS (feature, ratio_in_wrong, delta=c(SS)-c(S)) ---")
    feat_impact: dict[str, list[float]] = defaultdict(list)
    for r in s_to_ss:
        for feat, ratio, cp, ct, delta in feature_deltas(r["ratios"], "SS", "S"):
            feat_impact[feat].append((ratio, delta))

    # Aggregate: which features CONSISTENTLY push toward SS across all S→SS logs
    print(f"\n{'Feature':<42} {'avg_ratio':>10} {'avg_delta':>10} {'n_logs':>7} {'consistency':>12}")
    print("-" * 85)
    agg = []
    for feat, entries in feat_impact.items():
        avg_ratio = sum(r for r, _ in entries) / len(entries)
        avg_delta = sum(d for _, d in entries) / len(entries)
        agg.append((feat, avg_ratio, avg_delta, len(entries)))
    agg.sort(key=lambda x: abs(x[2]), reverse=True)
    for feat, avg_r, avg_d, n in agg[:15]:
        direction = "→SS" if avg_d > 0 else "→S"
        print(f"  {feat:<42} {avg_r:>10.4f} {avg_d:>10.4f} {n:>7} {direction}")

    print("\n--- Per-log detail: top 8 features pushing toward SS ---")
    for r in s_to_ss:
        print(f"\n  {r['name']}  (gap={r['scores']['SS']-r['scores']['S']:+.4f})")
        print(f"  {'Feature':<42} {'ratio':>7} {'c(SS)':>8} {'c(S)':>8} {'delta':>8}")
        for feat, ratio, cp, ct, delta in feature_deltas(r["ratios"], "SS", "S")[:8]:
            print(f"    {feat:<42} {ratio:>7.4f} {cp:>8.4f} {ct:>8.4f} {delta:>8.4f}")

    # Compare ratios of key features between correct_S and wrong_S
    key_feats = [
        "temporal.true_eventual", "temporal.eventual",
        "temporal.independence", "temporal.direct",
        "existential.independence", "existential.equivalence",
        "existential.negated_equivalence",
    ]
    print("\n--- Key feature ratios: correct S vs misclassified S ---")
    print(f"{'Feature':<42} {'correct_S_avg':>14} {'wrong_S_avg':>13}")
    print("-" * 72)
    for feat in key_feats:
        avg_c = sum(r["ratios"].get(feat, 0) for r in correct_s) / max(len(correct_s), 1)
        avg_w = sum(r["ratios"].get(feat, 0) for r in s_to_ss) / max(len(s_to_ss), 1)
        diff = avg_w - avg_c
        marker = "  <<<" if abs(diff) > 0.03 else ""
        print(f"  {feat:<42} {avg_c:>14.4f} {avg_w:>13.4f}  Δ={diff:+.4f}{marker}")

    # -----------------------------------------------------------------------
    # Section 3: SS → S and LS → SS misclassifications
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SECTION 3: OTHER MISCLASSIFICATIONS (SS→S, LS→SS)")
    print("=" * 70)

    for truth_cls, pred_cls in [("SS", "S"), ("LS", "SS")]:
        subset = [r for r in wrong if r["truth"] == truth_cls and r["predicted"] == pred_cls]
        if not subset:
            continue
        print(f"\n--- {truth_cls}→{pred_cls}: {[r['name'] for r in subset]} ---")
        for r in subset:
            gap = r["scores"][pred_cls] - r["scores"][truth_cls]
            print(f"  {r['name']}  gap={gap:+.4f}")
            print(f"  Scores: " + " | ".join(f"{c}={r['scores'][c]:+.4f}" for c in CLASSES))
            print(f"  Top features pushing toward {pred_cls}:")
            for feat, ratio, cp, ct, delta in feature_deltas(r["ratios"], pred_cls, truth_cls)[:10]:
                direction = f"→{pred_cls}" if delta > 0.001 else (f"→{truth_cls}" if delta < -0.001 else "~")
                print(f"    {feat:<42} ratio={ratio:.4f}  c({pred_cls})={cp:+.4f}  c({truth_cls})={ct:+.4f}  Δ={delta:+.4f}  {direction}")

    # -----------------------------------------------------------------------
    # Section 4: Suggested weight changes
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SECTION 4: WEIGHT CHANGE SUGGESTIONS (programmatic)")
    print("=" * 70)

    print("""
Analysis approach:
  For each confusion pair, identify which weight adjustments would close the score gap.
  A weight change on feature F for class C changes the score gap by:
    Δgap = Δweight(C, F) × ratio(F)
  We want to find changes that fix ALL misclassifications in the group
  without breaking correctly classified logs.
""")

    # For S→SS: find what weight changes close the SS-S gap
    print("--- Suggested weight changes for S→SS confusion ---")
    print("  Primary question: which features are HIGH in misclassified-S but LOW in correct-S?")
    print("  These are the 'discriminating features' within the S category itself.\n")

    # Already computed above - reference key_feats comparison
    print("  Re-check: what weight changes on SS weights would reduce SS score for these logs?")
    print("  AND/OR what weight changes on S weights would increase S score?")

    # For each wrong S: what minimum weight change on the most impactful feature fixes it?
    print(f"\n  {'Log':<12} {'gap':>8}  {'top_feature_pushing_SS':<40} {'ratio':>7}  {'delta_contribution':>18}  {'w_change_needed':>15}")
    print(f"  {'-'*110}")
    for r in s_to_ss:
        gap = r["scores"]["SS"] - r["scores"]["S"]
        top = feature_deltas(r["ratios"], "SS", "S")[0]
        feat, ratio_val, cp, ct, delta = top
        # To close gap by changing weight of feat in SS: gap / ratio_val = needed weight decrease
        needed_ss_decrease = -gap / ratio_val if ratio_val > 0 else float("inf")
        # To close gap by changing weight of feat in S: same
        needed_s_increase = gap / ratio_val if ratio_val > 0 else float("inf")
        print(f"  {r['name']:<12} {gap:>8.4f}  {feat:<40} {ratio_val:>7.4f}  {delta:>18.4f}  SS-={needed_ss_decrease:.2f} or S+={needed_s_increase:.2f}")

    print("\n  BUT: single-feature fixes are fragile. Better to look at which feature adjustments")
    print("  fix ALL S→SS logs simultaneously without breaking correct-S logs.\n")

    # Find features where adjusting the weight in SS (decrease) or S (increase) helps ALL wrong S
    # and doesn't hurt correct S (i.e., the feature is NOT high in correct-S either)
    print("  Features that CONSISTENTLY push S→SS (avg_delta > 0.02) across all misclassified S:")
    for feat, avg_r, avg_d, n in agg:
        if avg_d > 0.02 and n == len(s_to_ss):
            # Check if this feature is also high in correct S (would hurt correct S if we change S weight)
            avg_correct = sum(r["ratios"].get(feat, 0) for r in correct_s) / max(len(correct_s), 1)
            avg_wrong = sum(r["ratios"].get(feat, 0) for r in s_to_ss) / max(len(s_to_ss), 1)
            w_ss = WEIGHTS["SS"].get(feat, 0)
            w_s  = WEIGHTS["S"].get(feat, 0)
            print(f"    {feat:<42} w(SS)={w_ss:+.1f}  w(S)={w_s:+.1f}  avg_ratio_wrong={avg_wrong:.4f}  avg_ratio_correct={avg_correct:.4f}")
            if avg_correct < avg_wrong * 0.7:
                print(f"      ^ Feature is LOW in correct S ({avg_correct:.4f}) but HIGH in wrong S ({avg_wrong:.4f})")
                print(f"        → Adjusting w(SS, {feat}) DOWN or w(S, {feat}) UP would discriminate well.")
            else:
                print(f"      ^ Feature is SIMILAR in correct ({avg_correct:.4f}) and wrong ({avg_wrong:.4f}) S")
                print(f"        → Weight change risky — would also affect correctly classified S logs.")

    print("\n  Features that CONSISTENTLY push →S in correct-S but NOT in wrong-S (reinforcement direction):")
    correct_s_high: dict[str, float] = {}
    wrong_s_high: dict[str, float] = {}
    for feat in FEAT_ORDER:
        correct_s_high[feat] = sum(r["ratios"].get(feat, 0) for r in correct_s) / max(len(correct_s), 1)
        wrong_s_high[feat]   = sum(r["ratios"].get(feat, 0) for r in s_to_ss) / max(len(s_to_ss), 1)
    diffs_s = [(f, correct_s_high[f] - wrong_s_high[f]) for f in FEAT_ORDER]
    diffs_s.sort(key=lambda x: x[1], reverse=True)
    print(f"  {'Feature':<42} {'correct_S_avg':>14} {'wrong_S_avg':>12} {'diff':>8}")
    for feat, diff in diffs_s[:8]:
        if diff > 0.01:
            w_s = WEIGHTS["S"].get(feat, 0)
            print(f"    {feat:<42} {correct_s_high[feat]:>14.4f} {wrong_s_high[feat]:>12.4f} {diff:>8.4f}  w(S)={w_s:+.1f}")


if __name__ == "__main__":
    main()
