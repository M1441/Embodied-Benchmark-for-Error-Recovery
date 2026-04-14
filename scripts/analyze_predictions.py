#!/usr/bin/env python3
"""
analyze_predictions.py
Generate confusion matrices and prediction distribution analysis.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.parser import parse_prediction, parse_action

MODES = ["full", "after-only", "swap-test", "text-only"]
DATA_PATH = PROJECT_ROOT / "data" / "samples.jsonl"
ABLATION_DIR = PROJECT_ROOT / "outputs" / "ablations"


def load_samples():
    samples = {}
    with open(DATA_PATH) as f:
        for line in f:
            s = json.loads(line)
            samples[s["sample_id"]] = s
    return samples


def load_predictions(mode):
    path = ABLATION_DIR / f"predictions_{mode}.jsonl"
    preds = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if row.get("status") == "ok" and row.get("parsed"):
                preds[row["sample_id"]] = row["parsed"]["recovery_action"]
    return preds


def extract_action_type(action_str):
    """Extract just the action name: Navigate(x) -> Navigate"""
    if action_str is None:
        return "INVALID"
    name, _ = parse_action(action_str)
    return name if name else "INVALID"


def main():
    samples = load_samples()

    for mode in MODES:
        print(f"\n{'='*60}")
        print(f"MODE: {mode}")
        print(f"{'='*60}")

        preds = load_predictions(mode)

        # 1. Action type distribution: gold vs predicted
        gold_types = Counter()
        pred_types = Counter()
        gold_actions = Counter()
        pred_actions = Counter()

        # Per failure type breakdown
        per_ft = defaultdict(lambda: {"gold_types": Counter(), "pred_types": Counter(),
                                       "correct": 0, "total": 0})

        for sid, sample in samples.items():
            gold = sample["gold_recovery_action"]
            pred = preds.get(sid, "LookAround()")

            gt = extract_action_type(gold)
            pt = extract_action_type(pred)
            ft = sample["failure_type"]

            gold_types[gt] += 1
            pred_types[pt] += 1
            gold_actions[gold] += 1
            pred_actions[pred] += 1

            per_ft[ft]["gold_types"][gt] += 1
            per_ft[ft]["pred_types"][pt] += 1
            per_ft[ft]["total"] += 1
            if gold == pred:
                per_ft[ft]["correct"] += 1

        # Print action type distribution
        all_types = sorted(set(gold_types.keys()) | set(pred_types.keys()))
        print(f"\n  Action Type Distribution:")
        print(f"  {'Type':<15} {'Gold':>6} {'Pred':>6} {'Diff':>6}")
        print(f"  {'-'*35}")
        for t in all_types:
            g = gold_types[t]
            p = pred_types[t]
            print(f"  {t:<15} {g:>6} {p:>6} {p-g:>+6}")

        # Print top-15 most predicted actions
        print(f"\n  Top 15 Predicted Actions:")
        for action, count in pred_actions.most_common(15):
            in_gold = gold_actions.get(action, 0)
            print(f"    {action:<40} pred={count:>3}  gold={in_gold:>3}")

        # Print per-failure-type confusion
        print(f"\n  Per Failure Type:")
        print(f"  {'FT':<5} {'Acc':>7} {'N':>4}  Top Predicted Types")
        print(f"  {'-'*60}")
        for ft in sorted(per_ft.keys()):
            info = per_ft[ft]
            acc = info["correct"] / info["total"] if info["total"] > 0 else 0
            top_pred = info["pred_types"].most_common(3)
            top_str = ", ".join(f"{t}:{c}" for t, c in top_pred)
            print(f"  {ft:<5} {acc:>6.1%} {info['total']:>4}  {top_str}")

        # Confusion matrix: gold action type vs predicted action type
        print(f"\n  Confusion Matrix (Gold rows x Pred cols, action types):")
        types_order = ["LookAround", "Navigate", "Pick", "Place",
                       "Open", "Close", "Retry", "INVALID"]
        types_present = [t for t in types_order if t in all_types]

        header = f"  {'':>12}" + "".join(f"{t:>12}" for t in types_present)
        print(header)
        for gt in types_present:
            row_counts = Counter()
            for sid, sample in samples.items():
                gold = sample["gold_recovery_action"]
                pred = preds.get(sid, "LookAround()")
                if extract_action_type(gold) == gt:
                    row_counts[extract_action_type(pred)] += 1
            row = f"  {gt:>12}" + "".join(
                f"{row_counts.get(pt, 0):>12}" for pt in types_present
            )
            print(row)

    # F5-only swap analysis
    print(f"\n{'='*60}")
    print("F5-ONLY SWAP ANALYSIS")
    print(f"{'='*60}")

    for mode in ["full", "swap-test"]:
        preds = load_predictions(mode)
        correct = 0
        total = 0
        for sid, sample in samples.items():
            if sample["failure_type"] != "F5":
                continue
            gold = sample["gold_recovery_action"]
            pred = preds.get(sid, "LookAround()")
            total += 1
            if gold == pred:
                correct += 1
        acc = correct / total if total > 0 else 0
        print(f"  {mode:<15} F5 accuracy: {correct}/{total} = {acc:.1%}")


if __name__ == "__main__":
    main()