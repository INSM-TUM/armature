#!/usr/bin/env python3
"""Generate a comprehensive HTML classification report for structuredness scoring.

Applies the definition-based weighted scoring function from CLASSIFICATION_WEIGHTS.md
to all XES logs in Synthetic Log Data/, compares predictions to ground-truth labels
derived from directory names, and produces a detailed diagnostic report.

Usage:
    python scripts/generate_classification_report.py
    python scripts/generate_classification_report.py --output report.html
    python scripts/generate_classification_report.py --data-dir "/path/to/Synthetic Log Data"
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from itertools import permutations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from armature.core.dependencies import ExistentialDependency, TemporalDependency
from armature.discovery.discover import discover

# ---------------------------------------------------------------------------
# Classification weights (verbatim from CLASSIFICATION_WEIGHTS.md)
# ---------------------------------------------------------------------------

WEIGHTS: dict[str, dict[str, float]] = {
    "S": {
        "temporal.direct": +3.0,
        "temporal.direct_backward": +2.0,
        "temporal.no_ordering": +2.5,
        "temporal.independence": +0.5,
        "temporal.true_eventual": +1.0,
        "temporal.true_eventual_backward": +1.0,
        "temporal.eventual": +0.5,
        "temporal.eventual_backward": +0.5,
        "existential.equivalence": +3.0,
        "existential.negated_equivalence": +2.5,
        "existential.nand": +1.5,
        "existential.implication": +1.5,
        "existential.implication_backward": +1.5,
        "existential.or": +1.0,
        "existential.independence": -1.5,  # v2: relaxed from -3.0; complex S with optional+XOR generates small independence
    },
    "SS": {
        "temporal.true_eventual": +2.5,
        "temporal.true_eventual_backward": +2.5,
        "temporal.eventual": +1.0,         # v2: reduced from +2.5; eventual also arises in long structured sequences
        "temporal.eventual_backward": +1.0, # v2: reduced from +2.5
        "temporal.direct": +1.5,
        "temporal.direct_backward": +1.0,
        "temporal.independence": +2.5,  # v2: raised from +2.0; SS inter-segment ordering creates substantial independence
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
        "temporal.eventual": 0.0,           # v2: reduced from +1.5; eventual also arises in S/SS, neutral for LS
        "temporal.eventual_backward": 0.0,  # v2: reduced from +1.5
        "temporal.true_eventual": +1.0,
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
    "U": {
        "temporal.no_ordering": +3.0,
        "temporal.independence": +2.0,
        "temporal.direct": -2.0,
        "temporal.true_eventual": -1.5,
        "temporal.eventual": -1.0,
        "existential.equivalence": -2.0,
        "existential.implication": -1.5,
        "existential.implication_backward": -1.5,
        "existential.independence": -1.0,
        "existential.nand": -0.5,
        "existential.negated_equivalence": -0.5,
        "existential.or": -0.5,
    },
}

CLASSES = ["S", "SS", "LS", "U"]

FEAT_ORDER = (
    [f"temporal.{t.value}" for t in TemporalDependency]
    + [f"existential.{e.value}" for e in ExistentialDependency]
)

DIR_TO_LABEL = {
    "structuredLogs": "S",
    "semiStructuredLogs": "SS",
    "looselyStructuredLogs": "LS",
    "unstructuredLogs": "U",
}

LABEL_FULL = {"S": "Structured", "SS": "Semi-Structured", "LS": "Loosely Structured", "U": "Unstructured"}
LABEL_COLOR = {"S": "#1565c0", "SS": "#2e7d32", "LS": "#e65100", "U": "#6a1b9a"}
LABEL_BG    = {"S": "#e3f2fd", "SS": "#e8f5e9", "LS": "#fff3e0", "U": "#f3e5f5"}

# ---------------------------------------------------------------------------
# Ratio computation
# ---------------------------------------------------------------------------

def compute_ratios(path: Path) -> dict[str, float]:
    matrix = discover(path)
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


BORDERLINE_MARGIN = 0.2             # score gap below which top-2 are flagged as borderline

# LS pre-filter thresholds: ConDec all-required any-order pattern
LS_PREFILTER_MIN_INDEPENDENCE = 0.75
LS_PREFILTER_MIN_EQUIVALENCE = 0.45
LS_PREFILTER_MAX_DIRECT = 0.02

# S pre-filter thresholds: BPMN XOR exclusive-gateway signature
S_PREFILTER_MIN_NO_ORDERING = 0.15
S_PREFILTER_MIN_NEG_EQUIV = 0.05


def classify(ratios: dict[str, float]) -> tuple[dict[str, float], str, str | None]:
    """Classify a ratio profile.

    Returns (scores, predicted, borderline) where borderline is e.g. "SS|LS" when
    the top-2 scoring classes are within BORDERLINE_MARGIN of each other, else None.
    Prefilter decisions are never marked borderline — they are structural rule matches.
    """
    scores = {
        cls: sum(WEIGHTS[cls].get(k, 0.0) * v for k, v in ratios.items())
        for cls in CLASSES
    }

    # Degenerate guard: no pairs at all (single-activity or empty log)
    if all(v == 0.0 for v in scores.values()):
        return scores, "UNDETERMINED", None

    # LS pre-filter: high temporal independence + high existential equivalence + no direct-follows.
    # Captures ConDec models where all activities are required (hence high equivalence) but can
    # execute in any order (hence full independence). The linear scorer is confused by the high
    # equivalence, which normally favours S/SS; this pattern is definitionally LS (knowledge-worker
    # driven, no temporal constraints specified, all activities present).
    if (t_indep >= LS_PREFILTER_MIN_INDEPENDENCE
            and ratios.get("existential.equivalence", 0.0) >= LS_PREFILTER_MIN_EQUIVALENCE
            and ratios.get("temporal.direct", 0.0) < LS_PREFILTER_MAX_DIRECT):
        return scores, "LS", None

    # S pre-filter: strong BPMN XOR exclusive-gateway signature.
    # XOR gateways produce temporal.no_ordering (activities on different branches never co-occur in
    # any consistent direction) paired with existential.negated_equivalence (exactly one of the two
    # exclusive branch activities occurs per trace). ConDec nand constraints produce a similar
    # temporal.no_ordering but WITHOUT negated_equivalence (both activities are optional, not
    # mutually-required). The conjunction of both signals is structurally unique to BPMN S logs.
    if (ratios.get("temporal.no_ordering", 0.0) > S_PREFILTER_MIN_NO_ORDERING
            and ratios.get("existential.negated_equivalence", 0.0) > S_PREFILTER_MIN_NEG_EQUIV):
        return scores, "S", None

    # Among S / SS / LS only
    ranked = sorted(("S", "SS", "LS"), key=lambda c: scores[c], reverse=True)
    predicted = ranked[0]
    margin = scores[ranked[0]] - scores[ranked[1]]
    borderline = f"{ranked[0]}|{ranked[1]}" if margin < BORDERLINE_MARGIN else None
    return scores, predicted, borderline


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

CSS = """
<style>
:root {
  --blue:   #1565c0; --blue-bg:   #e3f2fd;
  --green:  #2e7d32; --green-bg:  #e8f5e9;
  --orange: #e65100; --orange-bg: #fff3e0;
  --purple: #6a1b9a; --purple-bg: #f3e5f5;
  --text: #1a1a2e; --surface: #f5f6fa; --card: #fff;
  --border: #dee2ec;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'Segoe UI', system-ui, sans-serif;
  background: var(--surface);
  color: var(--text);
  padding: 2rem;
  line-height: 1.5;
}
h1 { font-size: 1.7rem; color: #0f3460; margin-bottom: 0.25rem; }
.subtitle { color: #666; font-size: 0.88rem; margin-bottom: 2.5rem; }

/* ---- Cards ---- */
.card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1.4rem 1.6rem;
  margin-bottom: 2rem;
}
.card h2 {
  font-size: 1.05rem;
  color: #0f3460;
  border-left: 4px solid #0f3460;
  padding-left: 0.55rem;
  margin-bottom: 1rem;
}
.card h3 { font-size: 0.95rem; color: #333; margin-bottom: 0.6rem; }

/* ---- Metrics row ---- */
.metrics-row {
  display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.5rem;
}
.metric-box {
  flex: 1; min-width: 120px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 0.8rem 1rem;
  text-align: center;
}
.metric-box .val { font-size: 2rem; font-weight: 700; color: #0f3460; }
.metric-box .lbl { font-size: 0.75rem; color: #666; text-transform: uppercase; letter-spacing: 0.06em; }

/* ---- Per-class metrics ---- */
.class-metrics {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 0.8rem;
  margin-bottom: 1.5rem;
}
.class-card {
  border-radius: 6px;
  padding: 0.7rem 0.9rem;
  border-left: 4px solid;
}
.class-card .cls-name { font-weight: 700; font-size: 0.82rem; margin-bottom: 0.35rem; }
.class-card .cls-stat { font-size: 0.78rem; color: #444; }
.class-card .cls-stat span { font-weight: 600; }

/* ---- Confusion matrix ---- */
.confusion-wrap { overflow-x: auto; }
table.confusion {
  border-collapse: collapse;
  font-size: 0.84rem;
  margin-bottom: 0;
}
table.confusion th, table.confusion td {
  border: 1px solid var(--border);
  padding: 7px 14px;
  text-align: center;
}
table.confusion thead th { background: #0f3460; color: #fff; font-weight: 600; }
table.confusion tbody th {
  background: var(--surface);
  font-weight: 700;
  text-align: right;
  padding-right: 10px;
}
table.confusion .diag { font-weight: 700; }
table.confusion .corner { background: #0f3460; color: #fff; }

/* ---- Main results table ---- */
.table-wrap { overflow-x: auto; }
table.results {
  border-collapse: collapse;
  font-size: 0.82rem;
  width: 100%;
  white-space: nowrap;
}
table.results thead tr th {
  background: #0f3460; color: #fff;
  padding: 7px 12px;
  text-align: center;
  font-weight: 600;
  position: sticky; top: 0;
}
table.results thead tr th.left { text-align: left; }
table.results tbody tr td {
  padding: 5px 10px;
  border-bottom: 1px solid var(--border);
  text-align: center;
}
table.results tbody tr td.left { text-align: left; font-family: monospace; font-size: 0.8rem; }
table.results tbody tr.wrong { background: #fff8f8; }
table.results tbody tr.wrong:hover { background: #ffefef; }
table.results tbody tr.correct:hover { background: #f0f7f0; }

.badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 12px;
  font-size: 0.74rem;
  font-weight: 700;
  letter-spacing: 0.03em;
}
.badge-correct { background: #e8f5e9; color: #2e7d32; border: 1px solid #a5d6a7; }
.badge-wrong   { background: #ffebee; color: #c62828; border: 1px solid #ef9a9a; }

.score-bar {
  display: inline-block;
  height: 12px;
  border-radius: 3px;
  vertical-align: middle;
  margin-right: 4px;
}
.score-cell { min-width: 100px; text-align: left !important; }

/* ---- Feature heatmap ---- */
table.feats {
  border-collapse: collapse;
  font-size: 0.8rem;
  width: 100%;
  white-space: nowrap;
}
table.feats thead th {
  background: #16213e; color: #fff;
  padding: 5px 10px;
  text-align: center;
  font-weight: 600;
  position: sticky; top: 0;
}
table.feats thead th.left { text-align: left; }
table.feats tbody td {
  padding: 4px 10px;
  border-bottom: 1px solid var(--border);
  text-align: right;
}
table.feats tbody td.left { text-align: left; font-family: monospace; font-size: 0.78rem; }
table.feats tbody td.kind {
  text-align: left;
  font-size: 0.7rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: #0f3460;
  background: #f0f2fa;
  border-right: 2px solid #ccd;
}
.group-div td { border-top: 2px solid #b0b8d0 !important; }
.zero { color: #ccc !important; }

/* ---- Misclassification analysis ---- */
.misclass-block {
  margin-bottom: 1.5rem;
  border: 1px solid var(--border);
  border-radius: 6px;
  overflow: hidden;
}
.misclass-header {
  padding: 0.6rem 1rem;
  font-weight: 700;
  font-size: 0.88rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  background: #fff8f8;
  border-bottom: 1px solid var(--border);
}
.misclass-body { padding: 0.8rem 1rem; }
table.contrib {
  border-collapse: collapse;
  font-size: 0.8rem;
  width: 100%;
}
table.contrib th {
  padding: 4px 10px;
  background: var(--surface);
  border: 1px solid var(--border);
  text-align: center;
  font-weight: 600;
}
table.contrib th.left { text-align: left; }
table.contrib td {
  padding: 3px 10px;
  border: 1px solid var(--border);
  text-align: right;
}
table.contrib td.left { text-align: left; font-family: monospace; font-size: 0.79rem; }
.pos { color: #1b5e20; }
.neg { color: #b71c1c; }

/* ---- Tabs ---- */
.tab-bar { display: flex; gap: 4px; margin-bottom: 1rem; }
.tab-btn {
  padding: 5px 14px;
  border: 1px solid var(--border);
  border-radius: 4px 4px 0 0;
  background: var(--surface);
  font-size: 0.82rem;
  cursor: pointer;
  font-weight: 600;
}
.tab-btn.active { background: #0f3460; color: #fff; border-color: #0f3460; }
.tab-pane { display: none; }
.tab-pane.active { display: block; }

/* ---- Collapsible ---- */
details summary {
  cursor: pointer;
  font-size: 0.84rem;
  color: #0f3460;
  font-weight: 600;
  padding: 6px 4px;
}
details summary:hover { text-decoration: underline; }
details[open] summary { margin-bottom: 0.6rem; }

</style>
"""

JS = """
<script>
function switchTab(group, idx) {
  const panes  = document.querySelectorAll('.' + group + '-pane');
  const btns   = document.querySelectorAll('.' + group + '-btn');
  panes.forEach((p, i) => p.classList.toggle('active', i === idx));
  btns .forEach((b, i) => b.classList.toggle('active', i === idx));
}
</script>
"""


def _label_badge(label: str) -> str:
    if label == "UNDETERMINED":
        return '<span class="badge" style="background:#f5f5f5;color:#777;border:1px solid #bbb">?</span>'
    c = LABEL_COLOR[label]
    bg = LABEL_BG[label]
    return f'<span class="badge" style="background:{bg};color:{c};border:1px solid {c}40">{label}</span>'


def _score_bar(score: float, max_abs: float, color: str) -> str:
    if max_abs == 0:
        return ""
    pct = min(abs(score) / max_abs * 100, 100)
    opacity = "1" if score >= 0 else "0.4"
    return f'<span class="score-bar" style="width:{pct:.0f}px;background:{color};opacity:{opacity}"></span>'


def _heat(val: float, vmax: float, palette: tuple) -> str:
    lo, hi = palette
    if val == 0.0:
        return '<td class="zero">—</td>'
    t = min(val / vmax, 1.0) if vmax > 0 else 0
    r = int(lo[0] + t * (hi[0] - lo[0]))
    g = int(lo[1] + t * (hi[1] - lo[1]))
    b = int(lo[2] + t * (hi[2] - lo[2]))
    fg = "#fff" if t > 0.6 else "#1a1a2e"
    fw = "600" if t > 0.35 else "400"
    return f'<td style="background:rgb({r},{g},{b});color:{fg};font-weight:{fw}">{val:.4f}</td>'


T_PAL = ((235, 245, 255), (15, 52, 96))
E_PAL = ((255, 243, 235), (150, 60, 10))

# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def section_summary(results: list[dict]) -> str:
    total = len(results)
    correct_n = sum(1 for r in results if r["correct"])
    acc = correct_n / total * 100 if total else 0

    # Per-class stats (exclude UNDETERMINED from FP counts — degenerate logs)
    per_class: dict[str, dict] = {c: {"tp": 0, "fp": 0, "fn": 0, "n": 0} for c in CLASSES}
    for r in results:
        gt, pred = r["truth"], r["predicted"]
        per_class[gt]["n"] += 1
        if gt == pred:
            per_class[gt]["tp"] += 1
        else:
            per_class[gt]["fn"] += 1
            if pred in per_class:  # skip UNDETERMINED
                per_class[pred]["fp"] += 1

    def safe_div(a, b):
        return a / b if b else 0.0

    lines = ['<div class="card">']
    lines.append('<h2>Overall Results</h2>')

    # Top metrics
    lines.append('<div class="metrics-row">')
    lines.append(f'<div class="metric-box"><div class="val">{total}</div><div class="lbl">Logs Tested</div></div>')
    lines.append(f'<div class="metric-box"><div class="val">{correct_n}</div><div class="lbl">Correct</div></div>')
    lines.append(f'<div class="metric-box"><div class="val">{total - correct_n}</div><div class="lbl">Wrong</div></div>')
    lines.append(f'<div class="metric-box"><div class="val">{acc:.1f}%</div><div class="lbl">Accuracy</div></div>')
    lines.append('</div>')

    # Per-class metrics grid
    lines.append('<h3>Per-Class Metrics</h3>')
    lines.append('<div class="class-metrics">')
    for cls in CLASSES:
        s = per_class[cls]
        tp, fp, fn, n = s["tp"], s["fp"], s["fn"], s["n"]
        prec = safe_div(tp, tp + fp)
        rec  = safe_div(tp, tp + fn)
        f1   = safe_div(2 * prec * rec, prec + rec)
        c, bg = LABEL_COLOR[cls], LABEL_BG[cls]
        lines.append(
            f'<div class="class-card" style="background:{bg};border-left-color:{c}">'
            f'<div class="cls-name" style="color:{c}">{cls} — {LABEL_FULL[cls]}</div>'
            f'<div class="cls-stat">N: <span>{n}</span> &nbsp; TP: <span>{tp}</span> &nbsp; FP: <span>{fp}</span> &nbsp; FN: <span>{fn}</span></div>'
            f'<div class="cls-stat">Precision: <span>{prec:.2f}</span> &nbsp; Recall: <span>{rec:.2f}</span> &nbsp; F1: <span>{f1:.2f}</span></div>'
            f'</div>'
        )
    lines.append('</div>')

    # Confusion matrix
    lines.append('<h3>Confusion Matrix &nbsp;<small style="font-weight:400;color:#888">(rows = ground truth, cols = predicted)</small></h3>')
    lines.append('<div class="confusion-wrap"><table class="confusion">')
    lines.append('<thead><tr><th class="corner">Truth \\ Pred</th>')
    for c in CLASSES:
        lines.append(f'<th style="background:{LABEL_COLOR[c]}">{c}</th>')
    lines.append('</tr></thead><tbody>')

    pred_cols = CLASSES + ["UNDETERMINED"]
    confusion: dict[str, dict[str, int]] = {gt: {pred: 0 for pred in pred_cols} for gt in CLASSES}
    for r in results:
        confusion[r["truth"]][r["predicted"]] += 1

    for gt in CLASSES:
        c = LABEL_COLOR[gt]
        lines.append(f'<tr><th style="color:{c}">{gt}</th>')
        for pred in pred_cols:
            v = confusion[gt][pred]
            if pred == "UNDETERMINED":
                if v > 0:
                    lines.append(f'<td style="background:#f5f5f5;color:#777">{v}</td>')
                else:
                    lines.append('<td style="color:#ccc">—</td>')
            elif gt == pred:
                pct = safe_div(v, per_class[gt]["n"]) * 100
                lines.append(f'<td class="diag" style="background:{LABEL_BG[gt]};color:{LABEL_COLOR[gt]}">{v} <small>({pct:.0f}%)</small></td>')
            elif v > 0:
                lines.append(f'<td style="background:#ffebee;color:#c62828">{v}</td>')
            else:
                lines.append('<td style="color:#ccc">—</td>')
        lines.append('</tr>')

    lines.append('</tbody></table></div>')
    lines.append('</div>')
    return "\n".join(lines)


def section_results_table(results: list[dict]) -> str:
    max_score = max(
        abs(s)
        for r in results
        for s in r["scores"].values()
    )

    lines = ['<div class="card">']
    lines.append('<h2>Per-Log Classification Results</h2>')
    lines.append('<div class="table-wrap"><table class="results">')

    # Header
    lines.append('<thead><tr>')
    lines.append('<th class="left">Log</th>')
    lines.append('<th>Truth</th><th>Predicted</th><th>Result</th><th>Margin</th>')
    for c in CLASSES:
        lines.append(f'<th class="score-cell" style="color:{LABEL_COLOR[c]}60;background:{"#0f3460"}">{c} score</th>')
    lines.append('</tr></thead><tbody>')

    # Sort: wrong first, then by log name
    sorted_results = sorted(results, key=lambda r: (r["correct"], r["name"]))

    for r in sorted_results:
        row_cls = "correct" if r["correct"] else "wrong"
        badge = '<span class="badge badge-correct">✓ correct</span>' if r["correct"] \
                else '<span class="badge badge-wrong">✗ wrong</span>'

        scores = r["scores"]
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        margin = sorted_scores[0][1] - sorted_scores[1][1]

        lines.append(f'<tr class="{row_cls}">')
        lines.append(f'<td class="left">{r["name"]}</td>')
        lines.append(f'<td>{_label_badge(r["truth"])}</td>')
        lines.append(f'<td>{_label_badge(r["predicted"])}</td>')
        lines.append(f'<td>{badge}</td>')
        lines.append(f'<td style="font-weight:{"700" if margin < 0.05 else "400"};color:{"#c62828" if margin < 0.05 else "#333"}">{margin:.4f}</td>')

        for c in CLASSES:
            s = scores[c]
            bar = _score_bar(s, max_score, LABEL_COLOR[c])
            sign = "+" if s >= 0 else ""
            fw = "700" if c == r["predicted"] else "400"
            lines.append(f'<td class="score-cell" style="font-weight:{fw}">{bar}{sign}{s:.4f}</td>')

        lines.append('</tr>')

    lines.append('</tbody></table></div>')
    lines.append('</div>')
    return "\n".join(lines)


def section_feature_heatmap(results: list[dict]) -> str:
    """Feature ratio heatmaps split by ground-truth class (tabbed)."""
    lines = ['<div class="card">']
    lines.append('<h2>Feature Ratio Heatmaps (by Ground-Truth Class)</h2>')

    tab_btns = []
    tab_panes = []

    for idx, cls in enumerate(CLASSES):
        cls_results = [r for r in results if r["truth"] == cls]
        if not cls_results:
            continue

        log_names = [r["name"] for r in cls_results]
        active = "active" if idx == 0 else ""
        tab_btns.append(
            f'<button class="tab-btn feat-btn {active}" onclick="switchTab(\'feat\', {idx})">'
            f'{cls} ({len(cls_results)})</button>'
        )

        feat_rows = []
        prev_kind = None
        for feat in FEAT_ORDER:
            kind, _ = feat.split(".", 1)
            feat_vals = [r["ratios"].get(feat, 0.0) for r in cls_results]
            feat_rows.append((feat, kind, feat_vals))

        kind_max = {}
        for feat, kind, vals in feat_rows:
            kind_max[kind] = max(kind_max.get(kind, 0), max(vals) if vals else 0)

        pane_lines = [f'<div class="tab-pane feat-pane {active}"><div class="table-wrap"><table class="feats">']
        pane_lines.append('<thead><tr><th class="left" colspan="2">Feature</th>')
        for name in log_names:
            pane_lines.append(f'<th>{name}</th>')
        pane_lines.append('</tr></thead><tbody>')

        for feat, kind, vals in feat_rows:
            div_class = ' class="group-div"' if kind != prev_kind and prev_kind is not None else ''
            pane_lines.append(f'<tr{div_class}>')
            if kind != prev_kind:
                count_kind = sum(1 for _, k, _ in feat_rows if k == kind)
                pane_lines.append(f'<td class="kind" rowspan="{count_kind}">{kind}</td>')
            _, dep_type = feat.split(".", 1)
            pane_lines.append(f'<td class="left">{dep_type}</td>')
            vmax = kind_max.get(kind, 1.0) or 1.0
            pal = T_PAL if kind == "temporal" else E_PAL
            for v in vals:
                pane_lines.append(_heat(v, vmax, pal))
            pane_lines.append('</tr>')
            prev_kind = kind

        pane_lines.append('</tbody></table></div></div>')
        tab_panes.append("\n".join(pane_lines))

    lines.append('<div class="tab-bar">' + "".join(tab_btns) + '</div>')
    lines.extend(tab_panes)
    lines.append('</div>')
    return "\n".join(lines)


def section_misclassifications(results: list[dict]) -> str:
    wrong = [r for r in results if not r["correct"]]
    if not wrong:
        return '<div class="card"><h2>Misclassifications</h2><p style="color:#2e7d32;font-weight:600">All logs classified correctly! ✓</p></div>'

    lines = ['<div class="card">']
    lines.append(f'<h2>Misclassification Analysis ({len(wrong)} logs)</h2>')
    lines.append('<p style="font-size:0.83rem;color:#666;margin-bottom:1rem">'
                 'For each misclassified log: feature contributions (weight × ratio) that pushed the score toward '
                 'the wrong prediction vs. the correct class. Sorted by absolute contribution magnitude. '
                 'Positive = supports that class, Negative = opposes it.</p>')

    for r in sorted(wrong, key=lambda x: x["name"]):
        truth = r["truth"]
        pred = r["predicted"]
        tc = LABEL_COLOR[truth]
        pc = LABEL_COLOR.get(pred, "#777")

        lines.append('<div class="misclass-block">')
        lines.append(
            f'<div class="misclass-header">'
            f'<span style="font-family:monospace">{r["name"]}</span>'
            f'<span style="color:#888;font-weight:400">— truth:</span> {_label_badge(truth)}'
            f'<span style="color:#888;font-weight:400">predicted:</span> {_label_badge(pred)}'
            f'<span style="color:#888;font-weight:400;margin-left:auto">scores: '
            + " / ".join(f'{c}={r["scores"][c]:+.4f}' for c in CLASSES)
            + '</span></div>'
        )

        # UNDETERMINED has no weight vector — show a note instead of contrib table
        if pred == "UNDETERMINED":
            lines.append('<div class="misclass-body"><p style="font-size:0.82rem;color:#888">'
                         'Log has no activity pairs (single-activity or empty). All scores = 0. '
                         'Cannot classify — insufficient data.</p></div></div>')
            continue

        lines.append('<div class="misclass-body">')
        lines.append(
            f'<table class="contrib"><thead><tr>'
            f'<th class="left">Feature</th>'
            f'<th>Ratio</th>'
            f'<th style="color:{pc}">w({pred})</th>'
            f'<th style="color:{pc}">contrib({pred})</th>'
            f'<th style="color:{tc}">w({truth})</th>'
            f'<th style="color:{tc}">contrib({truth})</th>'
            f'<th>Δ (pred − truth)</th>'
            f'</tr></thead><tbody>'
        )

        contribs = []
        for feat in FEAT_ORDER:
            ratio = r["ratios"].get(feat, 0.0)
            if ratio == 0.0:
                continue
            w_pred  = WEIGHTS[pred].get(feat, 0.0)
            w_truth = WEIGHTS[truth].get(feat, 0.0)
            c_pred  = w_pred  * ratio
            c_truth = w_truth * ratio
            delta   = c_pred - c_truth
            contribs.append((feat, ratio, w_pred, c_pred, w_truth, c_truth, delta))

        contribs.sort(key=lambda x: abs(x[6]), reverse=True)

        for feat, ratio, w_pred, c_pred, w_truth, c_truth, delta in contribs:
            delta_cls = "pos" if delta > 0 else "neg"
            lines.append(
                f'<tr>'
                f'<td class="left">{feat}</td>'
                f'<td>{ratio:.4f}</td>'
                f'<td>{w_pred:+.1f}</td>'
                f'<td class="{"pos" if c_pred >= 0 else "neg"}">{c_pred:+.4f}</td>'
                f'<td>{w_truth:+.1f}</td>'
                f'<td class="{"pos" if c_truth >= 0 else "neg"}">{c_truth:+.4f}</td>'
                f'<td class="{delta_cls}" style="font-weight:{"700" if abs(delta) > 0.05 else "400"}">{delta:+.4f}</td>'
                f'</tr>'
            )

        lines.append('</tbody></table>')
        lines.append('</div></div>')

    lines.append('</div>')
    return "\n".join(lines)


def section_weight_reference() -> str:
    lines = ['<div class="card">']
    lines.append('<h2>Weight Reference</h2>')
    lines.append('<details><summary>Show full weight table (all classes × all features)</summary>')
    lines.append('<div class="table-wrap"><table class="feats" style="margin-top:0.6rem">')
    lines.append('<thead><tr><th class="left" colspan="2">Feature</th>')
    for c in CLASSES:
        lines.append(f'<th style="color:{LABEL_COLOR[c]}80;background:#0f3460">{c}</th>')
    lines.append('</tr></thead><tbody>')

    prev_kind = None
    for feat in FEAT_ORDER:
        kind, dep_type = feat.split(".", 1)
        div_class = ' class="group-div"' if kind != prev_kind and prev_kind is not None else ''
        lines.append(f'<tr{div_class}>')
        if kind != prev_kind:
            count_kind = sum(1 for f in FEAT_ORDER if f.startswith(kind + "."))
            lines.append(f'<td class="kind" rowspan="{count_kind}">{kind}</td>')
        lines.append(f'<td class="left">{dep_type}</td>')
        for c in CLASSES:
            w = WEIGHTS[c].get(feat, 0.0)
            if w == 0.0:
                lines.append('<td class="zero">0</td>')
            else:
                fg = "#1b5e20" if w > 0 else "#b71c1c"
                bg = "#e8f5e9" if w > 0 else "#ffebee"
                fw = "700" if abs(w) >= 2.5 else "400"
                lines.append(f'<td style="background:{bg};color:{fg};font-weight:{fw}">{w:+.1f}</td>')
        lines.append('</tr>')
        prev_kind = kind

    lines.append('</tbody></table></div></details>')
    lines.append('</div>')
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="Synthetic Log Data")
    parser.add_argument("--output", default="classification_report_v2.html")
    args = parser.parse_args()

    base = Path(args.data_dir)
    groups = [
        ("structuredLogs",       "S"),
        ("semiStructuredLogs",   "SS"),
        ("looselyStructuredLogs","LS"),
        ("unstructuredLogs",     "U"),
    ]

    results: list[dict] = []

    for dir_name, label in groups:
        xes_paths = sorted((base / dir_name).glob("*.xes"))
        if not xes_paths:
            print(f"[warn] No XES files in {dir_name}", file=sys.stderr)
            continue
        print(f"\n[{dir_name}] ({label})", file=sys.stderr)
        for path in xes_paths:
            print(f"  {path.name} ...", file=sys.stderr, end=" ")
            try:
                ratios = compute_ratios(path)
                scores, predicted, borderline = classify(ratios)
                correct = predicted == label
                results.append({
                    "name": path.stem,
                    "truth": label,
                    "predicted": predicted,
                    "correct": correct,
                    "scores": scores,
                    "ratios": ratios,
                })
                mark = "✓" if correct else f"✗ (→{predicted})"
                print(mark, file=sys.stderr)
            except Exception as exc:
                print(f"ERROR: {exc}", file=sys.stderr)

    if not results:
        print("No results — nothing to report.", file=sys.stderr)
        sys.exit(1)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Structuredness Classification Report</title>
{CSS}
</head>
<body>
<h1>Structuredness Classification Report</h1>
<p class="subtitle">
  Definition-based weighted scoring (CLASSIFICATION_WEIGHTS.md) &mdash; no prior, no ML &mdash;
  applied to {len(results)} XES logs &mdash; ground truth from directory labels.
</p>
{section_summary(results)}
{section_results_table(results)}
{section_misclassifications(results)}
{section_feature_heatmap(results)}
{section_weight_reference()}
{JS}
</body>
</html>"""

    out = Path(args.output)
    out.write_text(html)
    total = len(results)
    correct_n = sum(1 for r in results if r["correct"])
    print(f"\n[done] {correct_n}/{total} correct ({correct_n/total*100:.1f}%) → {out.resolve()}", file=sys.stderr)


if __name__ == "__main__":
    main()
