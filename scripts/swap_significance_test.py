#!/usr/bin/env python3
"""scripts/swap_significance_test.py — McNemar test on F5 subset."""
import json
from pathlib import Path

DATA = Path("data/samples.jsonl")
ABLATION_DIR = Path("outputs/ablations")

samples = {}
with open(DATA) as f:
    for line in f:
        s = json.loads(line)
        if s["failure_type"] == "F5":
            samples[s["sample_id"]] = s

def load_preds(mode):
    preds = {}
    path = ABLATION_DIR / f"predictions_{mode}.jsonl"
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("status") == "ok" and r.get("parsed"):
                preds[r["sample_id"]] = r["parsed"]["recovery_action"]
    return preds

full = load_preds("full")
swap = load_preds("swap-test")

# McNemar's test: count discordant pairs
both_correct = 0
full_only = 0
swap_only = 0
both_wrong = 0

for sid, s in samples.items():
    gold = s["gold_recovery_action"]
    f_correct = full.get(sid) == gold
    s_correct = swap.get(sid) == gold

    if f_correct and s_correct:
        both_correct += 1
    elif f_correct and not s_correct:
        full_only += 1
    elif not f_correct and s_correct:
        swap_only += 1
    else:
        both_wrong += 1

n = len(samples)
print(f"F5 subset: n={n}")
print(f"  Both correct:  {both_correct}")
print(f"  Full only:     {full_only}")
print(f"  Swap only:     {swap_only}")
print(f"  Both wrong:    {both_wrong}")
print(f"  Full acc:      {(both_correct+full_only)/n:.1%}")
print(f"  Swap acc:      {(both_correct+swap_only)/n:.1%}")

# McNemar chi-squared (with continuity correction)
b = full_only
c = swap_only
if b + c > 0:
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    print(f"\n  McNemar chi2 = {chi2:.3f}")
    print(f"  (b={b}, c={c})")
    if chi2 > 3.84:
        print("  => p < 0.05, significant")
    else:
        print("  => p >= 0.05, not significant")
else:
    print("\n  No discordant pairs, test not applicable")