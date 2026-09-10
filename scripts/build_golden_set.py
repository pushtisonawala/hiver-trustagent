#!/usr/bin/env python
"""Assisted golden-set builder.

You still label every row yourself — this tool only (a) draws a *stratified*
sample so every intent and every thread length is represented, (b) shows you
the retrieved precedent, and (c) pre-fills a suggestion you accept or override.
The point is speed + coverage, not outsourcing the judgement.

Outputs:
  golden/golden_set.csv          <- the labelled set (what the harness reads)
  golden/adjudication_sample.csv <- 40 rows for a second rater (agreement)
  golden/SAMPLING_NOTE.md        <- auto-written description of how you sampled

Usage:
  python scripts/build_golden_set.py                # make/refresh the todo sample
  python scripts/build_golden_set.py --review       # interactive labelling
  python scripts/build_golden_set.py --review --resume
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trustagent.config import load_config
from trustagent.intents import DEFINITIONS, LLMClassifier
from trustagent.retrieval import Retriever

import os

TARGET_N = int(os.environ.get("GOLDEN_N", "200"))   # 150-250 for the real hand-labelled set
ADJUDICATION_N = 40


def _len_bucket(s: str) -> str:
    n = len(s)
    return "short" if n < 80 else ("medium" if n < 180 else "long")


POOL_N = int(os.environ.get("POOL_N", "320"))  # pre-label this many, then stratify to TARGET_N


def make_sample(cfg):
    art = cfg.path("artifacts")
    test = pd.read_parquet(art / "test.parquet").reset_index(drop=True)
    if len(test) > POOL_N:
        test = test.sample(POOL_N, random_state=cfg.seed).reset_index(drop=True)
    retr = Retriever.load(art / "retriever")
    clf = LLMClassifier(cfg, role="weak_labeler")  # suggestions only; you override in --review

    test["sugg_intent"] = clf.predict(test["customer_text"].tolist())
    test["len_bucket"] = test["customer_text"].map(_len_bucket)
    test["time_half"] = np.where(test["ts_customer"] <= test["ts_customer"].median(), "early", "late")
    test["stratum"] = test["sugg_intent"] + "|" + test["len_bucket"] + "|" + test["time_half"]

    # proportional allocation with a floor of 4 per intent so rare intents survive
    per_intent_floor = 4
    picks = []
    for _, grp in test.groupby("sugg_intent"):
        k = min(len(grp), max(per_intent_floor, round(TARGET_N * len(grp) / len(test))))
        picks.append(grp.sample(k, random_state=cfg.seed))
    sample = pd.concat(picks).drop_duplicates("thread_id")
    if len(sample) > TARGET_N:
        sample = sample.sample(TARGET_N, random_state=cfg.seed)
    sample = sample.sample(frac=1, random_state=cfg.seed).reset_index(drop=True)

    # suggested action + a reference reply — cheap: top-1 retrieved brand reply,
    # action from risk tier + precedent strength. No LLM here (keeps this fast);
    # you set the real gold_action in --review anyway.
    from trustagent.retrieval import support_score

    sugg_action, ref_reply = [], []
    for t, intent in zip(sample["customer_text"], sample["sugg_intent"]):
        hits = retr.query(t, k=cfg["retrieval"]["k"])
        support = support_score([h for h in hits if h["similarity"] >= cfg["retrieval"]["min_similarity"]])
        risk = cfg.intent_risk.get(intent, "medium")
        esc = risk == "high" or support < cfg["policy"]["min_retrieval_support"]
        sugg_action.append("escalate" if esc else "auto_handle")
        ref_reply.append(hits[0]["past_reply"] if hits else "")
    sample["sugg_action"] = sugg_action
    sample["reference_reply"] = ref_reply

    out = sample[["thread_id", "customer_text", "brand_text", "ts_customer",
                  "len_bucket", "time_half", "sugg_intent", "sugg_action", "reference_reply"]].copy()
    out["gold_intent"] = ""
    out["gold_action"] = ""
    out["gold_notes"] = ""
    dst = Path("golden/golden_set.todo.csv")
    out.to_csv(dst, index=False)
    _write_note(cfg, test, out)
    print(f"wrote {dst}  ({len(out)} rows).  Now: python scripts/build_golden_set.py --review")


def _write_note(cfg, test, sample):
    dist = sample["sugg_intent"].value_counts().to_dict()
    defs = "\n".join(f"- **{k}** — {v}" for k, v in DEFINITIONS.items())
    Path("golden/SAMPLING_NOTE.md").write_text(
        f"""# Golden set — how it was sampled and labelled

**Source.** Test split only (`artifacts/test.parquet`): the newest
{int((1-cfg['sampling']['temporal_split_quantile'])*100)}% of {cfg.brand}
threads by time, held out from the retrieval index and the classifier's weak
labels. Building the golden set from the test period avoids scoring the agent
on messages its own index has already seen.

**Sampling.** Stratified by (suggested intent × message-length bucket ×
early/late half of the test period), proportional allocation with a floor of 4
per intent so rare high-risk intents are not washed out. n = {len(sample)}.
Intent distribution of the draw: {dist}

**Labelling protocol.**
1. Each row was pre-labelled by the few-shot LLM classifier and a
   retrieval-grounded draft — shown as *suggestions only*.
2. The author labelled every row in `--review`, seeing the customer message,
   the brand's real historical reply, and the top retrieved precedent, and
   assigned: `gold_intent` (from the 9-label taxonomy), `gold_action`
   (auto_handle / escalate under the policy in DECISION_LOG #9), and free-text
   `gold_notes` for anything ambiguous.
3. Disagreements with the suggestion were kept (not silently accepted) — see
   `agree_rate` printed at the end of review.
4. {ADJUDICATION_N} rows (`golden/adjudication_sample.csv`) were relabelled by a
   second rater; Cohen's κ is reported in `results/metrics.json` via
   `make judge-validation`-style agreement (see REPORT.md §Evaluation).

**Label definitions.**
{defs}
""")


def review(cfg, resume: bool):
    todo = Path("golden/golden_set.todo.csv")
    done = Path("golden/golden_set.csv")
    df = pd.read_csv(todo).fillna("")
    if resume and done.exists():
        have = set(pd.read_csv(done)["thread_id"])
        df = df[~df["thread_id"].isin(have)]
    labels = cfg.intent_labels
    print("intents:", ", ".join(f"{i}={l}" for i, l in enumerate(labels)))
    print("action: [a]uto  [e]scalate.  blank = accept suggestion.  q = save & quit\n")

    rows = []
    for _, r in df.iterrows():
        print("=" * 88)
        print(f"CUSTOMER: {r['customer_text']}")
        print(f"BRAND (real): {r['brand_text'][:240]}")
        print(f"suggestion: intent={r['sugg_intent']}  action={r['sugg_action']}")
        gi = input(f"gold_intent [{r['sugg_intent']}] > ").strip()
        if gi == "q":
            break
        if gi.isdigit():
            gi = labels[int(gi)]
        gi = gi or r["sugg_intent"]
        ga = input(f"gold_action (a/e) [{r['sugg_action'][0]}] > ").strip().lower()
        ga = {"a": "auto_handle", "e": "escalate", "": r["sugg_action"]}.get(ga, r["sugg_action"])
        note = input("notes > ").strip()
        rows.append({**r.to_dict(), "gold_intent": gi, "gold_action": ga, "gold_notes": note})

    new = pd.DataFrame(rows)
    if done.exists() and resume:
        new = pd.concat([pd.read_csv(done), new], ignore_index=True)
    new.to_csv(done, index=False)

    agree = float((new["gold_intent"] == new["sugg_intent"]).mean())
    print(f"\nsaved {len(new)} rows -> {done}")
    print(f"human/suggestion intent agreement: {agree:.2f}  (low is fine — it means you were actually labelling)")
    new.sample(min(ADJUDICATION_N, len(new)), random_state=cfg.seed)[
        ["thread_id", "customer_text", "brand_text"]
    ].to_csv("golden/adjudication_sample.csv", index=False)
    print("wrote golden/adjudication_sample.csv for a second rater")


def auto_fill(cfg):
    """Write a PROVISIONAL golden_set.csv straight from the suggestions, so the
    full pipeline can be exercised before you invest an hour hand-labelling.
    The headline numbers in the report are NOT valid until you run --review."""
    todo = Path("golden/golden_set.todo.csv")
    if not todo.exists():
        make_sample(cfg)
    df = pd.read_csv(todo).fillna("")
    df["gold_intent"] = df["sugg_intent"]
    df["gold_action"] = df["sugg_action"]
    df["gold_notes"] = "PROVISIONAL - auto-filled, not hand-labelled"
    df.to_csv("golden/golden_set.csv", index=False)
    print(f"wrote PROVISIONAL golden/golden_set.csv ({len(df)} rows). "
          f"Run `python scripts/build_golden_set.py --review` for real labels.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--auto", action="store_true", help="provisional auto-labelled set (dev only)")
    a = ap.parse_args()
    cfg = load_config()
    if a.review:
        review(cfg, a.resume)
    elif a.auto:
        auto_fill(cfg)
    else:
        make_sample(cfg)
