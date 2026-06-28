#!/usr/bin/env python3
"""Batch classify all evaluation + synthetic + s13 logs.
Uses updated weights: S.true_eventual=+2.0, S.eventual=+1.0.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from armature.classification import classify
from armature.discovery import discover

SHORT = {"structured":"S","semi_structured":"SS","loosely_structured":"LS","unstructured":"U"}

# --- EVALUATION LOGS (survey ground truth + per-log thresholds) ---
EVAL_LOGS = [
    ("Evaluation/p1_s_BPI19.xes",               "SS", 0.8),
    ("Evaluation/Process_2_synthetic.xes",       "SS", 1.0),
    ("Evaluation/Process_3_Invoice before GR (Standard).xes", "S", 0.6),
    ("Evaluation/Process_4_Augur.xes",           "S",  0.7),
    ("Evaluation/Process_5_MIMIC_Dataset.xes",   "LS", 0.9),
    ("Evaluation/Process_6_synthetic.xes",       "LS", 1.0),
    ("Evaluation/Process_7_chickenhunt.xes",     "LS", 0.9),
    ("Evaluation/p8_us_synthetic.xes",           "U",  1.0),
]

# --- SYNTHETIC LOG DATA (ground truth from directory name) ---
SYNTH_BASE = Path("Synthetic Log Data")
SYNTH_DIRS = [
    ("structuredLogs", "S"),
    ("semiStructuredLogs", "SS"),
    ("looselyStructuredLogs", "LS"),
    ("unstructuredLogs", "U"),
]

def classify_one(path: Path, threshold: float = 1.0) -> dict:
    """Classify a single XES log and return result dict."""
    try:
        matrix = discover(path, threshold=threshold)
        result = classify(matrix)
        short = SHORT.get(result.category.value, "?")
        scores = result.metadata.get("scores", {})
        method = result.metadata.get("method", "?")
        return {
            "log": str(path),
            "Pred": short,
            "Conf": result.confidence,
            "S": scores.get("S", 0),
            "SS": scores.get("SS", 0),
            "LS": scores.get("LS", 0),
            "U": scores.get("U", 0),
            "Method": method,
        }
    except Exception as e:
        return {
            "log": str(path),
            "Pred": f"ERR: {e}",
            "Conf": "",
            "S": 0, "SS": 0, "LS": 0, "U": 0,
            "Method": "error",
        }

# --- RUN ALL ---
all_results = []

# 1. Evaluation logs
for fname, gt, t in EVAL_LOGS:
    r = classify_one(Path(fname), threshold=t)
    r["GT"] = gt
    r["T"] = t
    all_results.append(r)

# 2. Synthetic Log Data
for dirname, gt in SYNTH_DIRS:
    log_dir = SYNTH_BASE / dirname
    if not log_dir.exists():
        continue
    for xes_file in sorted(log_dir.glob("*.xes")):
        r = classify_one(xes_file, threshold=1.0)
        r["GT"] = gt
        r["T"] = 1.0
        all_results.append(r)

# 3. s13.xes
s13 = Path.home() / "Downloads/s13.xes"
if s13.exists():
    r = classify_one(s13, threshold=1.0)
    r["GT"] = "?"
    r["T"] = 1.0
    all_results.append(r)

# --- PRINT TABLE ---
header = f"{'Log':<42s} {'GT':>2s} {'T':>4s} {'Pred':>4s} {'Conf':<8s} {'S':>7s} {'SS':>7s} {'LS':>7s} {'U':>7s} {'Method'}"
sep = "=" * len(header)
print(sep)
print(header)
print(sep)

eval_correct = 0
eval_total = 0
synth_correct = 0
synth_total = 0

for r in all_results:
    gt = r.get("GT", "?")
    pred = r.get("Pred", "?")
    ok = "✓" if gt != "?" and pred == gt else (" " if gt == "?" else "✗")
    t = r.get("T", 1.0)
    conf = r.get("Conf", "")
    s = f"{r.get('S',0):+.2f}"
    ss = f"{r.get('SS',0):+.2f}"
    ls = f"{r.get('LS',0):+.2f}"
    u = f"{r.get('U',0):+.2f}"
    short_name = r.get("log", "").split("/")[-1]
    print(f"{short_name:<42s} {gt:>2s} {t:>4.1f} {pred:>4s} {conf:<8s} {s:>7s} {ss:>7s} {ls:>7s} {u:>7s} {r.get('Method',''):<20s} {ok}")

    if "Evaluation/" in r.get("log", ""):
        eval_total += 1
        if ok == "✓":
            eval_correct += 1
    elif gt != "?":
        synth_total += 1
        if ok == "✓":
            synth_correct += 1

print(sep)
print(f"\nEvaluation: {eval_correct}/{eval_total} ({eval_correct/eval_total*100:.1f}%)" if eval_total > 0 else "")
print(f"Synthetic : {synth_correct}/{synth_total} ({synth_correct/synth_total*100:.1f}%)" if synth_total > 0 else "")

# Save JSON
out = Path("Evaluation/classification_results_full.json")
with open(out, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved to {out}")
