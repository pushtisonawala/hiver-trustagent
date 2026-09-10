#!/usr/bin/env python
"""Cluster the agent's worst cases into failure modes with real examples.

Reads results/failure_pool.csv (worst 40 by combined judge score, written by
`make eval`), tags each with a failure mode, and writes a ranked
results/failure_analysis.md. The hypotheses are seeded but you are expected to
sharpen them — the assignment wants *your* reasoning, not the tool's.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trustagent import llm
from trustagent.config import load_config

MODES = {
    "hallucinated_fix": "Reply states steps/policy not present in the evidence.",
    "unsafe_auto_handle": "Auto-handled a message that needed a human (billing/account/legal/anger).",
    "over_escalation": "Escalated something the agent could clearly have answered.",
    "intent_confusion": "Wrong intent, which then routed the whole response wrong.",
    "generic_nonanswer": "Vague 'we're looking into it' / 'DM us' with no actual help.",
    "tone_off": "Correct content but off-brand: robotic, preachy, or too long.",
    "retrieval_irrelevant": "Top precedent was not actually similar; grounding is spurious.",
    "multiturn_blind": "Needed earlier conversation context the single-message agent never saw.",
}
SEED_HYPOTHESIS = {
    "hallucinated_fix": "Drafter treats low-similarity hits as usable. Fix: hard gate reply generation "
                        "on max similarity, and add a self-check pass that deletes unsupported sentences.",
    "unsafe_auto_handle": "Keyword list misses paraphrases ('charged me twice' has no keyword). "
                          "Fix: move the high-risk check to the intent classifier's probability mass "
                          "on {billing_subscription, cancel_or_refund, account_access}.",
    "over_escalation": "min_retrieval_support is tuned too high for common intents. "
                       "Fix: per-intent thresholds; low-risk intents get a lower bar.",
    "intent_confusion": "Adjacent labels (billing vs cancel; playback vs device) leak. "
                        "Fix: merge or add contrastive few-shot examples for the confused pair.",
    "generic_nonanswer": "Model hedges when evidence is thin instead of escalating. "
                         "Fix: forbid 'DM us' as a *reply* — that path must escalate.",
    "tone_off": "Voice guide not enforced. Fix: add 2 golden style exemplars + a length cap.",
    "retrieval_irrelevant": "TF-IDF/embedding retrieves lexically-close but semantically-different cases. "
                            "Fix: re-rank top-20 with the LLM; drop hits it flags as off-topic.",
    "multiturn_blind": "We evaluate on the first message only. Fix: feed the last 3 turns as context.",
}


def tag(row, cfg) -> str:
    m = cfg.model("judge_a")
    sys_p = ("Classify one customer-support failure into exactly one mode key. Modes:\n" +
             "\n".join(f"- {k}: {v}" for k, v in MODES.items()) +
             '\nReturn JSON {"mode": key, "why": "<12 words"}.')
    user = (f"customer: {row['customer_text']}\nintent(pred/gold): {row.get('intent')}/{row.get('gold_intent')}\n"
            f"action(pred/gold): {row.get('action')}/{row.get('gold_action')}\n"
            f"reply: {row['draft_reply']}\njudge_a: {row.get('judge_a_overall')}")
    try:
        d = llm.parse_json(llm.complete(m["provider"], m["model"], sys_p, user,
                                        max_tokens=400, json_mode=True, reasoning="none"))
        return d.get("mode", "generic_nonanswer") if d.get("mode") in MODES else "generic_nonanswer"
    except Exception:  # noqa: BLE001
        r = str(row["draft_reply"]).lower()
        if row.get("action") == "auto_handle" and row.get("gold_action") == "escalate":
            return "unsafe_auto_handle"
        if row.get("action") == "escalate" and row.get("gold_action") == "auto_handle":
            return "over_escalation"
        if "dm" in r or "looking into" in r:
            return "generic_nonanswer"
        if row.get("intent") != row.get("gold_intent"):
            return "intent_confusion"
        return "tone_off"


def main():
    cfg = load_config()
    res = cfg.path("results")
    fp = res / "failure_pool.csv"
    if not fp.exists():
        raise SystemExit("run `make eval` first")
    df = pd.read_csv(fp).fillna("")
    df["mode"] = [tag(r, cfg) for _, r in df.iterrows()]
    counts = Counter(df["mode"])

    out = ["# Failure analysis — agent\n",
           f"Pool = {len(df)} worst-scoring golden cases (by combined Judge A+B overall).\n",
           "| rank | failure mode | count | share |", "|---|---|---|---|"]
    top = counts.most_common(5)
    for i, (mode, n) in enumerate(top, 1):
        out.append(f"| {i} | `{mode}` | {n} | {n/len(df):.0%} |")

    for i, (mode, n) in enumerate(top, 1):
        ex = df[df["mode"] == mode].head(2)
        out += [f"\n## {i}. `{mode}` — {MODES[mode]}  ({n} cases)\n"]
        for _, r in ex.iterrows():
            out += [f"> **customer:** {r['customer_text']}",
                    f"> **agent ({r['intent']}, {r['action']}):** {r['draft_reply']}",
                    f"> **gold:** intent={r['gold_intent']} action={r['gold_action']}  "
                    f"· judgeA={r['judge_a_overall']}\n"]
        out += [f"**Seed hypothesis.** {SEED_HYPOTHESIS[mode]}\n",
                "**Author note.** _<add your sharper hypothesis + what you actually saw here>_\n"]

    (res / "failure_analysis.md").write_text("\n".join(out) + "\n")
    print("wrote", res / "failure_analysis.md")


if __name__ == "__main__":
    main()
