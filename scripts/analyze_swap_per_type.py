#!/usr/bin/env python3
"""
analyze_swap_per_type.py
========================
Per-failure-type swap sensitivity analysis with McNemar test.
Addresses the issue that aggregate swap-test = full (28.3% = 28.3%)
hides meaningful per-type differences.

Usage:
    python scripts/analyze_swap_per_type.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DATA = PROJECT_ROOT / "data" / "samples.jsonl"
ABLATION_DIR = PROJECT_ROOT / "outputs" / "ablations"

# Load samples
samples = {}
with open(DATA) as f:
    for line in f:
        s = json.loads(line)
        samples[s["sample_id"]] = s

# Load predictions
def load_preds(mode):
    preds = {}
    path = ABLATION_DIR / f"predictions_{mode}.jsonl"
    if not path.exists():
        print(f"❌ {path} not found")
        sys.exit(1)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("status") == "ok" and r.get("parsed"):
                preds[r["sample_id"]] = r["parsed"]["recovery_action"]
    return preds

full_preds = load_preds("full")
swap_preds = load_preds("swap-test")

# Load manifest to check same_frame
manifest_path = PROJECT_ROOT / "data" / "image_manifest.json"
manifest_map = {}
if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text())
    manifest_map = {m["sample_id"]: m for m in manifest}

# Per-type analysis
fts = ["F1", "F2", "F3", "F4", "F5"]
print("=" * 75)
print("PER-FAILURE-TYPE SWAP SENSITIVITY ANALYSIS")
print("=" * 75)
print()

total_discordant = 0
for ft in fts:
    ft_samples = {sid: s for sid, s in samples.items() if s["failure_type"] == ft}
    n = len(ft_samples)

    # Count same_frame ratio
    same_frame_count = sum(
        1 for sid in ft_samples if manifest_map.get(sid, {}).get("same_frame", True)
    )

    both_correct = 0
    full_only = 0
    swap_only = 0
    both_wrong = 0

    for sid, s in ft_samples.items():
        gold = s["gold_recovery_action"]
        f_correct = full_preds.get(sid) == gold
        s_correct = swap_preds.get(sid) == gold

        if f_correct and s_correct:
            both_correct += 1
        elif f_correct and not s_correct:
            full_only += 1
        elif not f_correct and s_correct:
            swap_only += 1
        else:
            both_wrong += 1

    full_acc = (both_correct + full_only) / n if n > 0 else 0
    swap_acc = (both_correct + swap_only) / n if n > 0 else 0
    delta = full_acc - swap_acc

    b, c = full_only, swap_only
    total_discordant += b + c

    print(f"  {ft} (n={n}, same_frame={same_frame_count}/{n})")
    print(f"    Full acc:  {full_acc:.1%}  ({both_correct + full_only}/{n})")
    print(f"    Swap acc:  {swap_acc:.1%}  ({both_correct + swap_only}/{n})")
    print(f"    Delta:     {delta:+.1%}")
    print(f"    McNemar:   b={b} (full-only), c={c} (swap-only)")

    if b + c > 0:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        sig = "p<0.05 ✓" if chi2 > 3.84 else "p≥0.05 ✗"
        print(f"    chi²={chi2:.3f}  ({sig})")

        # Key insight annotation
        if same_frame_count == n:
            print(f"    ⚠️  ALL samples have same before/after — swap is a no-op for this type")
        elif delta > 0.05:
            print(f"    ✅ Full > Swap: model uses temporal direction for {ft}")
        elif delta < -0.05:
            print(f"    ⚡ Swap > Full: swapping helps (unexpected)")
    else:
        print(f"    No discordant pairs")

    print()

# Summary
print("-" * 75)
print("SUMMARY FOR PAPER")
print("-" * 75)
print()
print("Swap test is only meaningful for failure types with different before/after images.")
print("In our dataset:")
print(f"  F1-F4: before == after (same_frame=True) → swap is identity, Δ reflects noise only")
print(f"  F5:    before ≠ after (same_frame=False) → swap is meaningful")
print()
print("Recommendation: Report swap results per failure type. For aggregate,")
print("note that the null result is expected since 80% of samples are swap-invariant.")
print()
print(f"Total discordant pairs across all types: {total_discordant}")