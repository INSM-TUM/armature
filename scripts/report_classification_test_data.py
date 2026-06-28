#!/usr/bin/env python3
"""Generate HTML classification report for Test Data/Classification/.

Uses the current weighted_engine.classify_matrix (post-prefilter-redesign).
Output: Evaluation/classification_test_data_report.html

Usage:
    python scripts/report_classification_test_data.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from armature.classification.weighted_engine import classify_matrix
from armature.core.dependencies import ExistentialDependency, TemporalDependency
from armature.discovery.discover import discover

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "Test Data" / "Classification"
OUT_PATH = ROOT / "Evaluation" / "classification_test_data_report.html"

FOLDERS = [
    ("structured",        "S"),
    ("semi-structured",   "SS"),
    ("loosely-structured","LS"),
    ("unstructured",      "U"),
    ("edge-cases",        None),
]

CAT_SHORT = {
    "structured":        "S",
    "semi_structured":   "SS",
    "loosely_structured":"LS",
    "unstructured":      "U",
}
CLASSES = ["S", "SS", "LS", "U"]
LABEL_FULL  = {"S": "Structured", "SS": "Semi-Structured", "LS": "Loosely Structured", "U": "Unstructured"}
LABEL_COLOR = {"S": "#1565c0",    "SS": "#2e7d32",         "LS": "#e65100",            "U": "#6a1b9a"}
LABEL_BG    = {"S": "#e3f2fd",    "SS": "#e8f5e9",         "LS": "#fff3e0",            "U": "#f3e5f5"}

FEAT_ORDER = (
    [f"temporal.{t.value}" for t in TemporalDependency]
    + [f"existential.{e.value}" for e in ExistentialDependency]
)


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect() -> list[dict]:
    results = []
    for folder, expected in FOLDERS:
        folder_path = DATA_DIR / folder
        if not folder_path.exists():
            continue
        for xes in sorted(folder_path.glob("*.xes")):
            print(f"  {folder}/{xes.name} ...", end=" ", flush=True)
            try:
                matrix = discover(xes, threshold=1.0)
                r = classify_matrix(matrix)
            except Exception as exc:
                print(f"ERROR: {exc}")
                results.append({
                    "folder": folder, "file": xes.name, "stem": xes.stem,
                    "expected": expected or "?", "predicted": "ERR",
                    "confidence": "—", "correct": False, "borderline": False,
                    "scores": {c: 0.0 for c in CLASSES}, "ratios": {},
                    "activity_count": 0, "method": "error", "error": str(exc),
                })
                continue

            pred = CAT_SHORT.get(r.category.value, r.category.value)
            scores = r.metadata.get("scores", {c: 0.0 for c in CLASSES if c != "U"})
            # U comes from prefilter, not scoring; fill 0 for display
            for c in CLASSES:
                scores.setdefault(c, 0.0)

            borderline = r.confidence == "boundary"
            # top-2 classes by score (borderline check uses 0.2 margin from engine)
            ranked = sorted((c for c in CLASSES if c != "U"), key=lambda c: scores.get(c, 0.0), reverse=True)
            in_top2 = expected is not None and expected in ranked[:2]
            correct = (expected is None) or (pred == expected) or (borderline and in_top2)
            mark = "✓" if pred == expected else ("~" if correct else "✗")
            print(f"{mark} {pred} (exp={expected})")
            results.append({
                "folder": folder, "file": xes.name, "stem": xes.stem,
                "expected": expected or "?", "predicted": pred,
                "confidence": r.confidence,
                "correct": correct,
                "exact_correct": pred == expected,
                "borderline": borderline,
                "scores": {c: round(scores.get(c, 0.0), 4) for c in CLASSES},
                "ratios": {k: round(v, 6) for k, v in r.dependency_ratios.items()},
                "activity_count": r.activity_count,
                "method": r.metadata.get("method", "unknown"),
                "error": None,
            })
    return results


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

CSS = """
<style>
:root {
  --blue:#1565c0; --blue-bg:#e3f2fd;
  --green:#2e7d32; --green-bg:#e8f5e9;
  --orange:#e65100; --orange-bg:#fff3e0;
  --purple:#6a1b9a; --purple-bg:#f3e5f5;
  --text:#1a1a2e; --surface:#f5f6fa; --card:#fff; --border:#dee2ec;
}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',system-ui,sans-serif;background:var(--surface);color:var(--text);padding:2rem;line-height:1.5}
h1{font-size:1.7rem;color:#0f3460;margin-bottom:.25rem}
.subtitle{color:#666;font-size:.88rem;margin-bottom:2.5rem}
.card{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:1.4rem 1.6rem;margin-bottom:2rem}
.card h2{font-size:1.05rem;color:#0f3460;border-left:4px solid #0f3460;padding-left:.55rem;margin-bottom:1rem}
.card h3{font-size:.95rem;color:#333;margin-bottom:.6rem}
.metrics-row{display:flex;gap:1rem;flex-wrap:wrap;margin-bottom:1.5rem}
.metric-box{flex:1;min-width:120px;background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:.8rem 1rem;text-align:center}
.metric-box .val{font-size:2rem;font-weight:700;color:#0f3460}
.metric-box .lbl{font-size:.75rem;color:#666;text-transform:uppercase;letter-spacing:.06em}
.class-metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:.8rem;margin-bottom:1.5rem}
.class-card{border-radius:6px;padding:.7rem .9rem;border-left:4px solid}
.class-card .cls-name{font-weight:700;font-size:.82rem;margin-bottom:.35rem}
.class-card .cls-stat{font-size:.78rem;color:#444}
.class-card .cls-stat span{font-weight:600}
.confusion-wrap{overflow-x:auto}
table.confusion{border-collapse:collapse;font-size:.84rem;margin-bottom:0}
table.confusion th,table.confusion td{border:1px solid var(--border);padding:7px 14px;text-align:center}
table.confusion thead th{background:#0f3460;color:#fff;font-weight:600}
table.confusion tbody th{background:var(--surface);font-weight:700;text-align:right;padding-right:10px}
table.confusion .corner{background:#0f3460;color:#fff}
.table-wrap{overflow-x:auto}
table.results{border-collapse:collapse;font-size:.82rem;width:100%;white-space:nowrap}
table.results thead tr th{background:#0f3460;color:#fff;padding:7px 12px;text-align:center;font-weight:600;position:sticky;top:0}
table.results thead tr th.left{text-align:left}
table.results tbody tr td{padding:5px 10px;border-bottom:1px solid var(--border);text-align:center}
table.results tbody tr td.left{text-align:left;font-family:monospace;font-size:.8rem}
table.results tbody tr.wrong{background:#fff8f8}
table.results tbody tr.wrong:hover{background:#ffefef}
table.results tbody tr.correct:hover{background:#f0f7f0}
.badge{display:inline-block;padding:2px 8px;border-radius:12px;font-size:.74rem;font-weight:700;letter-spacing:.03em}
.badge-correct{background:#e8f5e9;color:#2e7d32;border:1px solid #a5d6a7}
.badge-wrong{background:#ffebee;color:#c62828;border:1px solid #ef9a9a}
.badge-borderline{background:#fff8e1;color:#e65100;border:1px solid #ffcc80}
.score-bar{display:inline-block;height:12px;border-radius:3px;vertical-align:middle;margin-right:4px}
.score-cell{min-width:100px;text-align:left!important}
table.feats{border-collapse:collapse;font-size:.8rem;width:100%;white-space:nowrap}
table.feats thead th{background:#16213e;color:#fff;padding:5px 10px;text-align:center;font-weight:600;position:sticky;top:0}
table.feats thead th.left{text-align:left}
table.feats tbody td{padding:4px 10px;border-bottom:1px solid var(--border);text-align:right}
table.feats tbody td.left{text-align:left;font-family:monospace;font-size:.78rem}
table.feats tbody td.kind{text-align:left;font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.05em;color:#0f3460;background:#f0f2fa;border-right:2px solid #ccd}
.group-div td{border-top:2px solid #b0b8d0!important}
.zero{color:#ccc!important}
.misclass-block{margin-bottom:1.5rem;border:1px solid var(--border);border-radius:6px;overflow:hidden}
.misclass-header{padding:.6rem 1rem;font-weight:700;font-size:.88rem;display:flex;align-items:center;gap:.5rem;background:#fff8f8;border-bottom:1px solid var(--border);flex-wrap:wrap}
.misclass-body{padding:.8rem 1rem}
table.contrib{border-collapse:collapse;font-size:.8rem;width:100%}
table.contrib th{padding:4px 10px;background:var(--surface);border:1px solid var(--border);text-align:center;font-weight:600}
table.contrib th.left{text-align:left}
table.contrib td{padding:3px 10px;border:1px solid var(--border);text-align:right}
table.contrib td.left{text-align:left;font-family:monospace;font-size:.79rem}
.pos{color:#1b5e20}.neg{color:#b71c1c}
.tab-bar{display:flex;gap:4px;margin-bottom:1rem;flex-wrap:wrap}
.tab-btn{padding:5px 14px;border:1px solid var(--border);border-radius:4px 4px 0 0;background:var(--surface);font-size:.82rem;cursor:pointer;font-weight:600}
.tab-btn.active{background:#0f3460;color:#fff;border-color:#0f3460}
.tab-pane{display:none}
.tab-pane.active{display:block}
details summary{cursor:pointer;font-size:.84rem;color:#0f3460;font-weight:600;padding:6px 4px}
details summary:hover{text-decoration:underline}
details[open] summary{margin-bottom:.6rem}
.folder-tag{font-size:.72rem;font-family:monospace;background:#eef;color:#446;padding:1px 6px;border-radius:4px;margin-left:4px}
</style>"""

JS = """
<script>
function switchTab(group, idx) {
  document.querySelectorAll('.'+group+'-pane').forEach((p,i)=>p.classList.toggle('active',i===idx));
  document.querySelectorAll('.'+group+'-btn').forEach((b,i)=>b.classList.toggle('active',i===idx));
}
</script>"""


def _label_badge(label: str) -> str:
    if label == "?" or label not in LABEL_COLOR:
        return f'<span class="badge" style="background:#f5f5f5;color:#777;border:1px solid #bbb">{label}</span>'
    c, bg = LABEL_COLOR[label], LABEL_BG[label]
    return f'<span class="badge" style="background:{bg};color:{c};border:1px solid {c}40">{label}</span>'


def _score_bar(score: float, max_abs: float, color: str) -> str:
    if max_abs == 0:
        return ""
    pct = min(abs(score) / max_abs * 100, 100)
    op = "1" if score >= 0 else "0.35"
    return f'<span class="score-bar" style="width:{pct:.0f}px;background:{color};opacity:{op}"></span>'


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
    # Exclude unstructured + unlabeled from accuracy
    labeled = [r for r in results if r["expected"] != "?" and r["folder"] != "unstructured"]
    total = len(labeled)
    correct_n = sum(1 for r in labeled if r["correct"])
    exact_n = sum(1 for r in labeled if r.get("exact_correct", r["correct"]))
    acc = correct_n / total * 100 if total else 0

    per_class: dict[str, dict] = {c: {"tp": 0, "fp": 0, "fn": 0, "n": 0} for c in CLASSES}
    for r in labeled:
        gt, pred = r["expected"], r["predicted"]
        if gt not in per_class:
            continue
        per_class[gt]["n"] += 1
        if gt == pred:
            per_class[gt]["tp"] += 1
        else:
            per_class[gt]["fn"] += 1
            if pred in per_class:
                per_class[pred]["fp"] += 1

    def safe_div(a, b): return a / b if b else 0.0

    lines = ['<div class="card">']
    lines.append('<h2>Overall Results</h2>')
    borderline_correct_n = correct_n - exact_n
    lines.append('<div class="metrics-row">')
    lines.append(f'<div class="metric-box"><div class="val">{total}</div><div class="lbl">Labeled Logs<br><small style="font-size:.6rem;color:#999">(excl. unstructured)</small></div></div>')
    lines.append(f'<div class="metric-box"><div class="val">{correct_n}</div><div class="lbl">Correct<br><small style="font-size:.6rem;color:#999">(incl. borderline adj.)</small></div></div>')
    lines.append(f'<div class="metric-box"><div class="val">{exact_n}</div><div class="lbl">Exact Correct</div></div>')
    lines.append(f'<div class="metric-box"><div class="val">{borderline_correct_n}</div><div class="lbl">Borderline Correct</div></div>')
    lines.append(f'<div class="metric-box"><div class="val">{total - correct_n}</div><div class="lbl">Wrong</div></div>')
    lines.append(f'<div class="metric-box"><div class="val">{acc:.1f}%</div><div class="lbl">Accuracy</div></div>')
    lines.append('</div>')

    lines.append('<h3>Per-Class Metrics</h3><div class="class-metrics">')
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

    lines.append('<h3>Confusion Matrix &nbsp;<small style="font-weight:400;color:#888">(rows = ground truth, cols = predicted)</small></h3>')
    lines.append('<div class="confusion-wrap"><table class="confusion"><thead><tr><th class="corner">Truth \\ Pred</th>')
    for c in CLASSES:
        lines.append(f'<th style="background:{LABEL_COLOR[c]}">{c}</th>')
    lines.append('</tr></thead><tbody>')

    confusion: dict[str, dict[str, int]] = {gt: {pred: 0 for pred in CLASSES} for gt in CLASSES}
    for r in labeled:
        gt, pred = r["expected"], r["predicted"]
        if gt in confusion and pred in confusion[gt]:
            confusion[gt][pred] += 1

    for gt in CLASSES:
        c = LABEL_COLOR[gt]
        n = per_class[gt]["n"]
        lines.append(f'<tr><th style="color:{c}">{gt}</th>')
        for pred in CLASSES:
            v = confusion[gt][pred]
            if gt == pred:
                pct = safe_div(v, n) * 100
                lines.append(f'<td style="background:{LABEL_BG[gt]};color:{LABEL_COLOR[gt]};font-weight:700">{v} <small>({pct:.0f}%)</small></td>')
            elif v > 0:
                lines.append(f'<td style="background:#ffebee;color:#c62828">{v}</td>')
            else:
                lines.append('<td style="color:#ccc">—</td>')
        lines.append('</tr>')
    lines.append('</tbody></table></div></div>')
    return "\n".join(lines)


def section_results_table(results: list[dict]) -> str:
    max_score = max(
        (abs(s) for r in results for s in r["scores"].values()),
        default=1.0
    )

    lines = ['<div class="card"><h2>Per-Log Classification Results</h2>']
    lines.append('<div class="table-wrap"><table class="results"><thead><tr>')
    lines.append('<th class="left">File</th><th>Folder</th><th>Expected</th><th>Predicted</th><th>Confidence</th><th>Result</th>')
    for c in CLASSES:
        lines.append(f'<th class="score-cell" style="background:#0f3460;color:{LABEL_COLOR[c]}90">{c} score</th>')
    lines.append('<th>Activities</th></tr></thead><tbody>')

    sorted_res = sorted(results, key=lambda r: (r["correct"], r["folder"], r["file"]))
    for r in sorted_res:
        row_cls = "correct" if r["correct"] else "wrong"
        exact_ok = r.get("exact_correct", r["correct"])
        if exact_ok:
            badge = '<span class="badge badge-correct">✓ correct</span>'
        elif r["correct"]:
            badge = '<span class="badge badge-borderline">~ borderline adj.</span>'
        else:
            badge = '<span class="badge badge-wrong">✗ wrong</span>'

        conf_badge = (
            '<span style="background:#ffc107;color:#000;padding:1px 5px;border-radius:8px;font-size:10px">boundary</span>'
            if r["borderline"]
            else '<span style="background:#198754;color:#fff;padding:1px 5px;border-radius:8px;font-size:10px">exact</span>'
        )

        lines.append(f'<tr class="{row_cls}">')
        lines.append(f'<td class="left">{r["file"]}</td>')
        lines.append(f'<td><span class="folder-tag">{r["folder"]}</span></td>')
        lines.append(f'<td>{_label_badge(r["expected"])}</td>')
        lines.append(f'<td>{_label_badge(r["predicted"])}</td>')
        lines.append(f'<td>{conf_badge}</td>')
        lines.append(f'<td>{badge}</td>')
        for c in CLASSES:
            s = r["scores"].get(c, 0.0)
            bar = _score_bar(s, max_score, LABEL_COLOR[c])
            sign = "+" if s >= 0 else ""
            fw = "700" if c == r["predicted"] else "400"
            lines.append(f'<td class="score-cell" style="font-weight:{fw}">{bar}{sign}{s:.4f}</td>')
        lines.append(f'<td style="text-align:center">{r["activity_count"]}</td>')
        lines.append('</tr>')

    lines.append('</tbody></table></div></div>')
    return "\n".join(lines)


def section_misclassifications(results: list[dict], weights: dict) -> str:
    wrong = [r for r in results if not r["correct"] and r["expected"] != "?"]
    if not wrong:
        return '<div class="card"><h2>Misclassifications</h2><p style="color:#2e7d32;font-weight:600">All labeled logs classified correctly! ✓</p></div>'

    lines = [f'<div class="card"><h2>Misclassification Analysis ({len(wrong)} logs)</h2>']
    lines.append('<p style="font-size:.83rem;color:#666;margin-bottom:1rem">'
                 'Feature contributions (weight × ratio) sorted by |Δ| = contribution toward wrong class minus correct class.</p>')

    for r in sorted(wrong, key=lambda x: x["file"]):
        truth, pred = r["expected"], r["predicted"]
        if truth not in LABEL_COLOR or pred not in LABEL_COLOR:
            continue
        tc, pc = LABEL_COLOR[truth], LABEL_COLOR[pred]
        lines.append('<div class="misclass-block">')
        lines.append(
            f'<div class="misclass-header">'
            f'<span style="font-family:monospace">{r["file"]}</span>'
            f'<span class="folder-tag">{r["folder"]}</span>'
            f'<span style="color:#888;font-weight:400">truth:</span> {_label_badge(truth)}'
            f'<span style="color:#888;font-weight:400">predicted:</span> {_label_badge(pred)}'
            f'<span style="color:#888;font-weight:400;margin-left:auto">scores: '
            + " / ".join(f'{c}={r["scores"].get(c,0):+.4f}' for c in CLASSES)
            + '</span></div>'
        )
        lines.append('<div class="misclass-body">')
        lines.append(
            f'<table class="contrib"><thead><tr>'
            f'<th class="left">Feature</th><th>Ratio</th>'
            f'<th style="color:{pc}">w({pred})</th><th style="color:{pc}">contrib({pred})</th>'
            f'<th style="color:{tc}">w({truth})</th><th style="color:{tc}">contrib({truth})</th>'
            f'<th>Δ (pred−truth)</th>'
            f'</tr></thead><tbody>'
        )
        contribs = []
        for feat in FEAT_ORDER:
            ratio = r["ratios"].get(feat, 0.0)
            if ratio == 0.0:
                continue
            wp = weights.get(pred, {}).get(feat, 0.0)
            wt = weights.get(truth, {}).get(feat, 0.0)
            cp, ct = wp * ratio, wt * ratio
            delta = cp - ct
            contribs.append((feat, ratio, wp, cp, wt, ct, delta))
        contribs.sort(key=lambda x: abs(x[6]), reverse=True)
        for feat, ratio, wp, cp, wt, ct, delta in contribs:
            lines.append(
                f'<tr><td class="left">{feat}</td><td>{ratio:.4f}</td>'
                f'<td>{wp:+.1f}</td><td class="{"pos" if cp>=0 else "neg"}">{cp:+.4f}</td>'
                f'<td>{wt:+.1f}</td><td class="{"pos" if ct>=0 else "neg"}">{ct:+.4f}</td>'
                f'<td class="{"pos" if delta>0 else "neg"}" style="font-weight:{"700" if abs(delta)>.05 else "400"}">{delta:+.4f}</td>'
                f'</tr>'
            )
        lines.append('</tbody></table></div></div>')

    lines.append('</div>')
    return "\n".join(lines)


def section_feature_heatmap(results: list[dict]) -> str:
    lines = ['<div class="card"><h2>Feature Ratio Heatmaps (by Ground-Truth Class)</h2>']
    tab_btns, tab_panes = [], []
    tab_idx = 0
    for folder, expected in FOLDERS:
        cls_results = [r for r in results if r["folder"] == folder]
        if not cls_results:
            continue
        active = "active" if tab_idx == 0 else ""
        label = expected or "?"
        tab_btns.append(
            f'<button class="tab-btn feat-btn {active}" onclick="switchTab(\'feat\', {tab_idx})">'
            f'{folder} ({len(cls_results)})</button>'
        )
        log_names = [r["stem"] for r in cls_results]
        feat_rows = [(feat, feat.split(".")[0], [r["ratios"].get(feat, 0.0) for r in cls_results]) for feat in FEAT_ORDER]
        kind_max: dict[str, float] = {}
        for _, kind, vals in feat_rows:
            kind_max[kind] = max(kind_max.get(kind, 0), max(vals) if vals else 0)

        pane_lines = [f'<div class="tab-pane feat-pane {active}"><div class="table-wrap"><table class="feats">']
        pane_lines.append('<thead><tr><th class="left" colspan="2">Feature</th>')
        for name in log_names:
            pane_lines.append(f'<th style="font-size:.72rem">{name}</th>')
        pane_lines.append('</tr></thead><tbody>')
        prev_kind = None
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
        tab_idx += 1

    lines.append('<div class="tab-bar">' + "".join(tab_btns) + '</div>')
    lines.extend(tab_panes)
    lines.append('</div>')
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def section_loop_problem(results: list[dict]) -> str:
    loop_logs = [r for r in results if r["folder"] == "loosely-structured" and "loop" in r["file"].lower()]
    loop_names = [r["file"] for r in loop_logs]

    # Gather representative ratios from first loop log
    rep = loop_logs[0] if loop_logs else None
    ratio_rows = ""
    if rep:
        key_feats = [
            ("temporal.eventual",           "Eventual ordering (A sometimes before B)",    "SS +0.5",  "LS 0.0"),
            ("temporal.eventual_backward",  "Eventual backward (B sometimes before A)",    "SS +0.5",  "LS 0.0"),
            ("temporal.independence",       "Temporal independence (no ordering observed)", "SS +2.5", "LS +3.0"),
            ("existential.independence",    "Existential independence (co-occurrence free)","SS +2.5", "LS +2.5"),
            ("temporal.direct",             "Direct-follows (A always immediately before B)","SS +1.5","LS −1.5"),
            ("temporal.direct_backward",    "Direct backward (loop indicator)",             "SS +1.0", "LS −1.0"),
        ]
        for feat, desc, w_ss, w_ls in key_feats:
            v = rep["ratios"].get(feat, 0.0)
            ratio_rows += (
                f'<tr><td style="font-family:monospace;font-size:11px">{feat}</td>'
                f'<td>{desc}</td>'
                f'<td style="text-align:center">{v:.4f}</td>'
                f'<td style="text-align:center;color:{LABEL_COLOR["SS"]}">{w_ss}</td>'
                f'<td style="text-align:center;color:{LABEL_COLOR["LS"]}">{w_ls}</td>'
                f'</tr>'
            )

    loop_list = ", ".join(f'<code>{n}</code>' for n in loop_names) if loop_names else "none"

    return f"""<div class="card">
<h2>Loop LS Problem — Why Weighted Scoring Fails</h2>
<p style="font-size:.85rem;color:#444;margin-bottom:1rem">
  Affected logs: {loop_list}
</p>

<h3>Root Cause: Joint Feature Not Representable</h3>
<p style="font-size:.84rem;margin-bottom:.8rem">
  Loop loosely-structured logs contain process loops — activities cycle through multiple times per
  case, producing a distinctive ratio profile that is <strong>indistinguishable from semi-structured
  in independent feature space</strong>.
</p>
<p style="font-size:.84rem;margin-bottom:.8rem">
  The granular classifier (replaced by the weighted engine in <code>b7c6226</code>) caught these
  logs via its <strong>LS1 rule</strong>, which checks <code>none_none &gt; 0.13</code> — the
  fraction of activity pairs that are <em>simultaneously</em> temporally and existentially
  independent. This is a <strong>joint (cross-domain) feature</strong>: it only fires when a pair
  is BOTH temporally AND existentially unrelated. The weighted engine computes
  <code>temporal.independence</code> and <code>existential.independence</code> as <em>separate
  marginal ratios</em>, so a pair contributing to both gets counted twice independently. There is
  no weight combination over independent features that reproduces the joint condition.
</p>

<h3>Why the Loop Logs Look Like SS</h3>
<p style="font-size:.84rem;margin-bottom:.8rem">
  Process loops inflate <code>temporal.eventual</code> and
  <code>temporal.eventual_backward</code>: because activity A can appear before B in one
  iteration and B before A in the next, both eventual ordering directions get recorded. High
  eventual ordering in both directions is also the primary signal for semi-structured processes
  (activities within each SS phase are always eventually ordered). The weighted scorer cannot
  tell whether high eventual ordering originates from SS phase structure or from process looping.
</p>

<table class="main" style="font-size:.82rem;margin-bottom:1rem">
  <thead><tr>
    <th class="left">Feature</th><th class="left">Meaning</th>
    <th>Ratio (loop LS)</th>
    <th style="color:{LABEL_COLOR['SS']}80;background:#0f3460">SS weight</th>
    <th style="color:{LABEL_COLOR['LS']}80;background:#0f3460">LS weight</th>
  </tr></thead>
  <tbody>{ratio_rows}</tbody>
</table>

<p style="font-size:.84rem;margin-bottom:.8rem">
  The loop logs score <strong>SS ≈ 3.4, LS ≈ 3.0</strong> (margin ≈ 0.4 after the
  <code>existential.independence</code> weight fix). The SS advantage comes almost entirely
  from <code>temporal.eventual</code> and <code>temporal.eventual_backward</code>, which together
  contribute ~0.59 to SS over LS. Raising LS weights for these features to neutralise the gap
  would break correctly-classified SS logs that also have high eventual ordering (e.g. Log17).
</p>

<h3>What Would Fix It</h3>
<p style="font-size:.84rem;margin-bottom:.8rem">
  Two viable approaches:
</p>
<ol style="font-size:.84rem;padding-left:1.4rem;line-height:1.8">
  <li>
    <strong>Pre-filter using <code>none_none</code>:</strong> compute the fraction of pairs that
    are BOTH temporally and existentially independent (replicating the granular LS1 condition).
    A pre-filter <code>none_none &gt; 0.13 ∧ r_{{≡}} &lt; 0.10 ∧ r_{{⊳}} &lt; 0.05</code>
    would catch the loop LS logs without touching the scoring weights.
  </li>
  <li>
    <strong>Reintroduce the granular engine as a fallback:</strong> run the weighted scorer first;
    if the top-2 margin is below a threshold, fall back to the granular rule engine for a
    second opinion. Effective but adds complexity.
  </li>
</ol>
<p style="font-size:.84rem;color:#888">
  Note: the <code>direct_backward</code> feature (loop indicator) alone is insufficient —
  some SS logs also contain backward dependencies from intra-phase loops.
</p>
</div>"""


def main() -> None:
    # Pull weights from the current engine for misclassification analysis
    from armature.classification import weighted_engine as _we
    # The engine exposes WEIGHTS at module level
    weights: dict = getattr(_we, "WEIGHTS", {})

    print("Collecting results...")
    results = collect()
    if not results:
        print("No results.", file=sys.stderr)
        sys.exit(1)

    labeled = [r for r in results if r["expected"] != "?" and r["folder"] != "unstructured"]
    correct_n = sum(1 for r in labeled if r["correct"])
    total = len(labeled)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Armature — Classification Test Data Report</title>
{CSS}
</head>
<body>
<h1>Armature — Classification Test Data Report</h1>
<p class="subtitle">
  Test Data/Classification — weighted_engine.classify_matrix — threshold=1.0 — {total} labeled logs
</p>
{section_summary(results)}
{section_results_table(results)}
{section_misclassifications(results, weights)}
{section_loop_problem(results)}
{section_feature_heatmap(results)}
{JS}
</body>
</html>"""

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(html)
    print(f"\nSaved: {OUT_PATH}")
    print(f"Accuracy: {correct_n}/{total} ({100*correct_n/total:.1f}%)")


if __name__ == "__main__":
    main()
