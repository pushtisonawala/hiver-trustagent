from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import load_config, offline


def _scan_brands(args):
    """Rank brands by how many replies-to-customers they posted — a proxy for
    'how much resolvable support conversation is there for this brand'."""
    from collections import Counter

    cfg = load_config()
    src = Path(args.input or cfg.raw["paths"]["raw_csv"])
    replies: Counter = Counter()
    for chunk in pd.read_csv(src, dtype=str, chunksize=400_000,
                             usecols=["author_id", "inbound", "in_response_to_tweet_id"]):
        chunk = chunk.fillna("")
        out = chunk[(chunk["inbound"].str.lower() == "false")
                    & (~chunk["author_id"].str.isnumeric())
                    & (chunk["in_response_to_tweet_id"] != "")]
        replies.update(out["author_id"])
    print(f"{'brand':<22} replies_to_customers")
    for handle, n in replies.most_common(25):
        print(f"{handle:<22} {n}")


def _build(args):
    from .pipeline import build

    cfg = load_config()
    build(cfg, input_csv=args.input, smoke=args.smoke)


def _eval(args):
    from .pipeline import evaluate

    evaluate(load_config(), smoke=args.smoke)


def _demo(args):
    from .agent import TrustAgent
    from .intents import LLMClassifier
    from .retrieval import Retriever

    cfg = load_config()
    retr = Retriever.load(cfg.path("artifacts") / "retriever")
    agent = TrustAgent(cfg, retr, LLMClassifier(cfg))
    print(f"TrustAgent for {cfg.brand}  (offline={offline()}).  Ctrl-C to quit.\n")
    while True:
        try:
            msg = input("customer> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not msg:
            continue
        r = agent.handle(msg)
        print(f"\n  intent      : {r['intent']}  (conf {r['intent_confidence']:.2f}, "
              f"risk {r['risk_tier']})")
        print(f"  precedent   : support {r['retrieval_support']:.2f}")
        for h in r["evidence"][:3]:
            print(f"     - [{h['similarity']:.2f}] {h['past_customer'][:70]!r} -> {h['past_reply'][:70]!r}")
        print(f"  draft       : {r['draft_reply']}")
        print(f"  >>> DECISION: {r['action'].upper()}  — {r['reason']}\n")


def main():
    p = argparse.ArgumentParser("trustagent")
    sub = p.add_subparsers(required=True)

    s = sub.add_parser("scan-brands"); s.add_argument("--input"); s.set_defaults(fn=_scan_brands)
    s = sub.add_parser("build"); s.add_argument("--input"); s.add_argument("--smoke", action="store_true")
    s.set_defaults(fn=_build)
    s = sub.add_parser("eval"); s.add_argument("--smoke", action="store_true"); s.set_defaults(fn=_eval)
    s = sub.add_parser("demo"); s.set_defaults(fn=_demo)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
