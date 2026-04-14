#!/usr/bin/env python3
"""scripts/extract_case_studies.py — Find interesting success/failure pairs."""
import json
from pathlib import Path

DATA = Path("data/samples.jsonl")
FULL = Path("outputs/ablations/predictions_full.jsonl")
TEXT = Path("outputs/ablations/predictions_text-only.jsonl")

samples = {}
with open(DATA) as f:
    for line in f:
        s = json.loads(line)
        samples[s["sample_id"]] = s

def load_preds(path):
    preds = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("status") == "ok":
                preds[r["sample_id"]] = r
    return preds

full_preds = load_preds(FULL)
text_preds = load_preds(TEXT)

print("=== CASE TYPE A: Full correct, Text wrong (image helped) ===\n")
count = 0
for sid, s in samples.items():
    gold = s["gold_recovery_action"]
    fp = full_preds.get(sid, {}).get("parsed", {}).get("recovery_action")
    tp = text_preds.get(sid, {}).get("parsed", {}).get("recovery_action")
    if fp == gold and tp != gold:
        fe = full_preds[sid].get("parsed", {}).get("evidence", "")
        te = text_preds[sid].get("parsed", {}).get("evidence", "")
        print(f"  {sid} [{s['failure_type']}]")
        print(f"    Gold:      {gold}")
        print(f"    Full pred: {fp}  evidence: {fe}")
        print(f"    Text pred: {tp}  evidence: {te}")
        print(f"    Instruction: {s['instruction'][:80]}")
        print()
        count += 1
        if count >= 10:
            break

print(f"\n=== CASE TYPE B: Both wrong, but Full uses visual reasoning ===\n")
count = 0
for sid, s in samples.items():
    gold = s["gold_recovery_action"]
    fp = full_preds.get(sid, {}).get("parsed", {}).get("recovery_action")
    tp = text_preds.get(sid, {}).get("parsed", {}).get("recovery_action")
    fe = full_preds.get(sid, {}).get("parsed", {}).get("evidence", "")
    if fp != gold and tp != gold and fp != tp:
        if any(kw in fe.lower() for kw in ["visible", "not visible", "in view",
                                              "scene", "image", "unchanged"]):
            print(f"  {sid} [{s['failure_type']}]")
            print(f"    Gold:      {gold}")
            print(f"    Full pred: {fp}  evidence: {fe}")
            print(f"    Text pred: {tp}")
            print()
            count += 1
            if count >= 8:
                break

print(f"\n=== CASE TYPE C: Text correct, Full wrong (image confused model) ===\n")
count = 0
for sid, s in samples.items():
    gold = s["gold_recovery_action"]
    fp = full_preds.get(sid, {}).get("parsed", {}).get("recovery_action")
    tp = text_preds.get(sid, {}).get("parsed", {}).get("recovery_action")
    if tp == gold and fp != gold:
        fe = full_preds[sid].get("parsed", {}).get("evidence", "")
        print(f"  {sid} [{s['failure_type']}]")
        print(f"    Gold:      {gold}")
        print(f"    Full pred: {fp}  evidence: {fe}")
        print(f"    Text pred: {tp}")
        print()
        count += 1
        if count >= 8:
            break