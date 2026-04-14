#!/usr/bin/env python3
"""Evaluate all baselines on 300 samples — using the SAME evaluator as VLM."""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.baseline import heuristic_recovery_action, blind_heuristic_recovery_action
from src.evaluator import (
    evaluate_with_details,
    load_jsonl,
    bootstrap_mean_ci95,
    bootstrap_macro_f1_ci95,
)


def load_samples(path="data/samples.jsonl"):
    return load_jsonl(path)


def build_predictions_dict(samples, predict_fn):
    """Build {sample_id: raw_json_text} dict, same format VLM pipeline uses."""
    preds = {}
    for s in samples:
        action = predict_fn(s)
        preds[s["sample_id"]] = json.dumps({"recovery_action": action})
    return preds


def main():
    samples = load_samples()
    print(f"Loaded {len(samples)} samples\n")

    methods = [
        ("Oracle",           lambda s: s["gold_recovery_action"]),
        ("LookAround",       lambda s: "LookAround()"),
        ("Heuristic-Oracle", lambda s: heuristic_recovery_action(s)),
        ("Blind-Heuristic",  lambda s: blind_heuristic_recovery_action(s)),
    ]

    all_reports = {}
    os.makedirs("outputs/baselines", exist_ok=True)

    for name, fn in methods:
        preds = build_predictions_dict(samples, fn)
        report, details = evaluate_with_details(samples, preds, max_steps=1)

        report["ci95"] = {
            "action_accuracy": bootstrap_mean_ci95(
                details["action_correct_flags"], n_boot=1000
            ),
            "macro_f1": bootstrap_macro_f1_ci95(
                details["y_true"], details["y_pred"], n_boot=1000
            ),
        }
        report["method"] = name
        all_reports[name] = report

        # Per-sample predictions for debugging
        pred_path = f"outputs/baselines/predictions_{name.lower().replace('-', '_')}.jsonl"
        with open(pred_path, "w") as f:
            for s in samples:
                action = fn(s)
                gold = s["gold_recovery_action"]
                f.write(
                    json.dumps(
                        {
                            "sample_id": s["sample_id"],
                            "predicted": action,
                            "gold": gold,
                            "correct": action == gold,
                            "failure_type": s["failure_type"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    # ── Print summary table ──
    fts = ["F1", "F2", "F3", "F4", "F5"]
    print("=" * 105)
    print(
        f"{'Method':<20} {'N':>5} {'Acc':>8} {'M-F1':>8} {'FmtRaw':>8}  ",
        end="",
    )
    for ft in fts:
        print(f"{ft:>8}", end="")
    print()
    print("-" * 105)

    for name, _ in methods:
        r = all_reports[name]
        print(
            f"{name:<20} {r['num_samples']:>5} "
            f"{r['action_accuracy']:>7.2%} "
            f"{r['macro_f1']:>8.4f} "
            f"{r.get('format_compliance_rate_raw', 1.0):>7.2%}  ",
            end="",
        )
        for ft in fts:
            a = r["failure_breakdown"].get(ft, {}).get("action_acc", 0)
            print(f"{a:>7.1%}", end=" ")
        print()

    print("=" * 105)
    print("\nNOTE:")
    print("  All methods evaluated with src.evaluator.evaluate_with_details().")
    print("  Macro-F1 is computed per unique action-string label (same as VLM eval).")
    print("  Heuristic-Oracle uses ground-truth failure_type (oracle info).")
    print("  Blind-Heuristic  uses only attempted_action (fair VLM comparison).")

    summary_path = "outputs/baselines/baseline_summary_300.json"
    with open(summary_path, "w") as f:
        json.dump(all_reports, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {summary_path}")


if __name__ == "__main__":
    main()