#!/usr/bin/env python
"""Group the agent's worst cases into failure modes with real examples.

Reads results/failure_pool.csv (written by `make eval`), tags each row with a
mode, and writes results/failure_analysis.md. My analysis of each mode lives in
results/failure_notes.md, keyed by mode name; this script only pulls it in, so
regenerating the analysis never overwrites what I wrote. If a note is missing it
falls back to a one-line stub.
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

    notes = _load_notes(res, [m for m, _ in counts.most_common()])

    top = counts.most_common(5)
    out = [f"Worst {len(df)} of the golden set by combined judge score, grouped. "
           f"Counts are how many of those {len(df)} fell in each mode.\n"]
    for mode, n in top:
        ex = df[df["mode"] == mode].head(2)
        out += [f"### {mode} ({n})\n", f"{MODES[mode]}\n"]
        for _, r in ex.iterrows():
            out += [f"> customer: {r['customer_text']}",
                    f"> agent ({r['intent']}, {r['action']}): {r['draft_reply']}",
                    f"> gold: {r['gold_intent']} / {r['gold_action']}  (judge A {r['judge_a_overall']})\n"]
        out += [notes.get(mode, SEED_HYPOTHESIS[mode]), ""]

    (res / "failure_analysis.md").write_text("\n".join(out).rstrip() + "\n")
    print("wrote", res / "failure_analysis.md")


def _load_notes(res: Path, modes: list[str]) -> dict:
    """results/failure_notes.md: `## <mode>` headers, prose under each."""
    fp = res / "failure_notes.md"
    if not fp.exists():
        stub = ["# Failure-mode notes",
                "",
                "One paragraph per mode. Edit these; `make failures` reads them and won't",
                "overwrite them.", ""]
        for m in modes:
            stub += [f"## {m}", "", SEED_HYPOTHESIS.get(m, ""), ""]
        fp.write_text("\n".join(stub))
        return {}
    out, cur, buf = {}, None, []
    for line in fp.read_text().splitlines():
        if line.startswith("## "):
            if cur:
                out[cur] = "\n".join(buf).strip()
            cur, buf = line[3:].strip(), []
        elif cur is not None:
            buf.append(line)
    if cur:
        out[cur] = "\n".join(buf).strip()
    return {k: v for k, v in out.items() if v}


if __name__ == "__main__":
    main()
