#!/usr/bin/env python
"""Second pass over ONLY the rows currently labelled `other`.

Many first-pass `other` labels are really a specific intent buried in venting.
This walks you through just those rows so you can reclassify the real ones.
Typo-proof: type a number, the full label, or any unambiguous prefix.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trustagent.config import load_config

CSV = Path("golden/golden_set.csv")


def resolve_intent(raw: str, labels: list[str], current: str) -> str:
    raw = raw.strip().lower()
    if not raw:
        return current
    if raw.isdigit() and 0 <= int(raw) < len(labels):
        return labels[int(raw)]
    hits = [l for l in labels if l == raw] or [l for l in labels if l.startswith(raw)] \
        or [l for l in labels if raw in l]
    if len(hits) == 1:
        return hits[0]
    print(f"  ?? '{raw}' matched {hits or 'nothing'} — keeping {current}")
    return current


def main():
    cfg = load_config()
    labels = cfg.intent_labels
    d = pd.read_csv(CSV, dtype=str).fillna("")
    todo = d[d["gold_intent"] == "other"].index.tolist()
    print(f"{len(todo)} rows labelled `other`. For each: type a number/label to reclassify, "
          f"or Enter to keep `other`.  Intents:")
    for i, l in enumerate(labels):
        print(f"   {i} = {l}")
    print("action: a=auto_handle  e=escalate  (Enter = keep current).  q = save & quit\n")

    changed = 0
    for n, idx in enumerate(todo, 1):
        r = d.loc[idx]
        print("=" * 90)
        print(f"({n}/{len(todo)})  CUSTOMER: {r['customer_text']}")
        print(f"           BRAND: {r['brand_text'][:200]}")
        print(f"           now: intent=other  action={r['gold_action']}")
        gi = input("   new intent > ").strip()
        if gi.lower() == "q":
            break
        new_i = resolve_intent(gi, labels, "other")
        ga = input(f"   action (a/e) [{r['gold_action'][0]}] > ").strip().lower()
        new_a = {"a": "auto_handle", "e": "escalate", "": r["gold_action"]}.get(ga, r["gold_action"])
        if new_i != "other" or new_a != r["gold_action"]:
            changed += 1
            note = (r["gold_notes"] + " [refined from other]").strip()
            d.loc[idx, ["gold_intent", "gold_action", "gold_notes"]] = [new_i, new_a, note]

    d.to_csv(CSV, index=False)
    print(f"\nsaved. reclassified {changed} of {len(todo)} rows.")
    print("intent distribution now:", d["gold_intent"].value_counts().to_dict())


if __name__ == "__main__":
    main()
