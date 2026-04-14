#!/usr/bin/env python3
"""
generate_unified_table.py
=========================
Reads baseline_summary + ablation_summary (and optional strategy summaries),
generates a single paper-ready Markdown / LaTeX table with consistent metrics.

Usage:
    python scripts/generate_unified_table.py
    python scripts/generate_unified_table.py --strategies counterfactual,mental_simulation
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _pct(x, digits=1):
    return f"{100.0 * float(x):.{digits}f}"


def _load_json(path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_row(label, report, is_vlm=False):
    """Extract a uniform row dict from a report dict."""
    return {
        "label": label,
        "n": report.get("num_samples", 0),
        "acc": report.get("action_accuracy", 0.0),
        "macro_f1": report.get("macro_f1", 0.0),
        "fmt_raw": report.get("format_compliance_rate_raw", 1.0),
        "recovery": report.get("recovery_rate_at_k", 0.0),
        "completion": report.get("semi_loop_task_completion_rate", 0.0),
        "F1": report.get("failure_breakdown", {}).get("F1", {}).get("action_acc", 0.0),
        "F2": report.get("failure_breakdown", {}).get("F2", {}).get("action_acc", 0.0),
        "F3": report.get("failure_breakdown", {}).get("F3", {}).get("action_acc", 0.0),
        "F4": report.get("failure_breakdown", {}).get("F4", {}).get("action_acc", 0.0),
        "F5": report.get("failure_breakdown", {}).get("F5", {}).get("action_acc", 0.0),
        "is_vlm": is_vlm,
        "ci_acc": report.get("ci95", {}).get("action_accuracy", {}),
        "ci_f1": report.get("ci95", {}).get("macro_f1", {}),
    }


def _ci_str(ci_dict):
    """Format CI as [low, high]."""
    if not ci_dict:
        return ""
    low = ci_dict.get("low")
    high = ci_dict.get("high")
    if low is None or high is None:
        return ""
    return f"[{_pct(low)}, {_pct(high)}]"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--baseline-summary",
        default="outputs/baselines/baseline_summary_300.json",
    )
    p.add_argument(
        "--ablation-summary",
        default="outputs/ablations/ablation_summary.json",
    )
    p.add_argument(
        "--strategies",
        default="",
        help="Comma-separated extra strategies, e.g. counterfactual,mental_simulation",
    )
    p.add_argument("--out-dir", default="outputs/tables")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    # ── Baselines ──
    bl_data = _load_json(Path(args.baseline_summary))
    if bl_data:
        for method_name in ["LookAround", "Blind-Heuristic", "Heuristic-Oracle", "Oracle"]:
            report = bl_data.get(method_name)
            if report:
                rows.append(_extract_row(f"B: {method_name}", report, is_vlm=False))
    else:
        print(f"⚠️  Baseline summary not found: {args.baseline_summary}")
        print("   Run: python scripts/eval_baselines_full.py  (with the FIXED version)")

    # ── Vanilla VLM ablations ──
    abl_data = _load_json(Path(args.ablation_summary))
    if abl_data:
        model = abl_data.get("model", "?")
        reports = abl_data.get("reports", {})
        mode_labels = {
            "text-only": f"V: {model} text-only",
            "after-only": f"V: {model} after-only",
            "full": f"V: {model} full (B+A)",
            "swap-test": f"V: {model} swap-test",
        }
        for mode in ["text-only", "after-only", "full", "swap-test"]:
            report = reports.get(mode)
            if report:
                rows.append(_extract_row(mode_labels[mode], report, is_vlm=True))
    else:
        print(f"⚠️  Ablation summary not found: {args.ablation_summary}")

    # ── Extra strategies ──
    if args.strategies:
        abl_dir = Path(args.ablation_summary).parent
        for strat in args.strategies.split(","):
            strat = strat.strip()
            if not strat:
                continue
            strat_path = abl_dir / f"ablation_summary_{strat}.json"
            strat_data = _load_json(strat_path)
            if not strat_data:
                print(f"⚠️  Strategy summary not found: {strat_path}")
                continue
            model = strat_data.get("model", "?")
            reports = strat_data.get("reports", {})
            full_report = reports.get("full")
            if full_report:
                rows.append(
                    _extract_row(f"V: {model} full+{strat}", full_report, is_vlm=True)
                )

    if not rows:
        print("❌ No data to generate table.")
        return

    # ── Markdown table ──
    fts = ["F1", "F2", "F3", "F4", "F5"]
    md_lines = [
        "# Unified Results Table",
        "",
        "| Method | N | Acc (%) | 95% CI | Macro-F1 | Fmt% |"
        + "".join(f" {ft} |" for ft in fts),
        "|---|---:|---:|---|---:|---:|" + "---:|" * len(fts),
    ]

    for r in rows:
        ci_str = _ci_str(r["ci_acc"])
        line = (
            f"| {r['label']} "
            f"| {r['n']} "
            f"| {_pct(r['acc'])} "
            f"| {ci_str} "
            f"| {float(r['macro_f1']):.4f} "
            f"| {_pct(r['fmt_raw'])} "
        )
        for ft in fts:
            line += f"| {_pct(r[ft])} "
        line += "|"
        md_lines.append(line)

    # Recovery info (separate small table)
    md_lines.extend([
        "",
        "## Semi-Loop Recovery (VLM only)",
        "",
        "| Method | Recovery@K | Completion | Note |",
        "|---|---:|---:|---|",
    ])
    for r in rows:
        if r["is_vlm"]:
            note = ""
            # F1+F4 (120 samples) have empty goal_state → recovery ceiling ~60%
            md_lines.append(
                f"| {r['label']} "
                f"| {_pct(r['recovery'])} "
                f"| {_pct(r['completion'])} "
                f"| {note} |"
            )
    md_lines.append("")
    md_lines.append(
        "> **Note**: Recovery Rate ceiling is ~60% because F1 (LookAround) and "
        "F4 (Retry) have empty goal states that cannot be verified by the symbolic executor."
    )

    md_path = out_dir / "unified_table.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"✅ {md_path}")

    # ── LaTeX table ──
    tex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Recovery action prediction results. "
        r"Macro-F1 computed per action-string label. "
        r"B = baseline, V = VLM.}",
        r"\label{tab:main}",
        r"\small",
        r"\begin{tabular}{l r r r r " + "r " * len(fts) + r"}",
        r"\toprule",
        r"Method & N & Acc\% & M-F1 & Fmt\% & "
        + " & ".join(fts)
        + r" \\",
        r"\midrule",
    ]

    prev_vlm = None
    for r in rows:
        if prev_vlm is not None and r["is_vlm"] != prev_vlm:
            tex_lines.append(r"\midrule")
        prev_vlm = r["is_vlm"]

        ft_cells = " & ".join(_pct(r[ft]) for ft in fts)
        tex_lines.append(
            f"{r['label']} & {r['n']} & {_pct(r['acc'])} & "
            f"{float(r['macro_f1']):.3f} & {_pct(r['fmt_raw'])} & "
            f"{ft_cells} \\\\"
        )

    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    tex_path = out_dir / "unified_table.tex"
    tex_path.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")
    print(f"✅ {tex_path}")

    # ── JSON dump for programmatic use ──
    json_path = out_dir / "unified_rows.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"✅ {json_path}")


if __name__ == "__main__":
    main()