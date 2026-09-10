#!/usr/bin/env python
"""How much can we trust the LLM judge?

Step 1:  python scripts/judge_validation.py --make-sheet
         -> golden/human_reply_ratings.todo.csv  (60 replies, agent + simple, shuffled, blind)
Step 2:  you rate every row: human_overall 1-5, human_acceptable 0/1
         save as golden/human_reply_ratings.csv
Step 3:  python scripts/judge_validation.py
         -> agreement stats (Spearman, MAE, Cohen's kappa, judge-vs-judge)
            merged into results/metrics.json under "judge_validation"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trustagent.config import load_config
from trustagent.judge import _judge_one
from trustagent.metrics import judge_agreement
from trustagent.retrieval import Retriever

N = 60


def make_sheet(cfg):
    res = cfg.path("results")
    frames = []
    for name in ("agent", "simple"):
        fp = res / f"predictions_{name}.csv"
        if fp.exists():
            d = pd.read_csv(fp)[["customer_text", "draft_reply"]].copy()
            d["source"] = name
            frames.append(d)
    pool = pd.concat(frames, ignore_index=True).drop_duplicates("draft_reply")
    sheet = pool.sample(min(N, len(pool)), random_state=cfg.seed).sample(frac=1, random_state=cfg.seed + 1)
    sheet = sheet.rename(columns={"draft_reply": "reply"})
    sheet["human_overall"] = ""
    sheet["human_acceptable"] = ""
    sheet = sheet.drop(columns=["source"])  # keep the human blind to which system wrote it
    dst = Path("golden/human_reply_ratings.todo.csv")
    sheet.to_csv(dst, index=False)
    print(f"wrote {dst} ({len(sheet)} rows). Rate human_overall (1-5) + human_acceptable (0/1), "
          f"save as golden/human_reply_ratings.csv")


def run(cfg):
    hp = Path("golden/human_reply_ratings.csv")
    if not hp.exists():
        raise SystemExit("golden/human_reply_ratings.csv not found — run --make-sheet first.")
    h = pd.read_csv(hp)
    h = h[pd.to_numeric(h["human_overall"], errors="coerce").notna()].reset_index(drop=True)
    h["human_overall"] = h["human_overall"].astype(float)

    retr = Retriever.load(cfg.path("artifacts") / "retriever")
    accept_thr = cfg["judge"]["quality_accept_threshold"]

    def judge(which):
        m = cfg.model("judge_a" if which == "a" else "judge_b")
        scores = []
        for t, reply in zip(h["customer_text"], h["reply"]):
            hits = retr.query(str(t), k=cfg["retrieval"]["k"])
            scores.append(_judge_one(str(t), str(reply), hits, m["provider"], m["model"])["overall"])
        return np.array(scores, float)

    ja, jb = judge("a"), judge("b")
    from scipy.stats import spearmanr

    out = {
        "n": int(len(h)),
        "judge_a": {"model": cfg.model("judge_a")["model"],
                    **judge_agreement(ja, h["human_overall"], accept_thr)},
        "judge_b": {"model": cfg.model("judge_b")["model"],
                    **judge_agreement(jb, h["human_overall"], accept_thr)},
        "judge_a_vs_judge_b": {
            "spearman": float(spearmanr(ja, jb).statistic),
            "mae": float(np.abs(ja - jb).mean()),
            "mean_a": float(ja.mean()), "mean_b": float(jb.mean()),
        },
        "interpretation": "Use judge_a/judge_b spearman + cohen_kappa_accept to decide how much "
                          "weight the headline judge number can carry. If kappa < ~0.4 the judge "
                          "is not a reliable stand-in for a human and the headline must be caveated.",
    }
    mp = cfg.path("results") / "metrics.json"
    data = json.load(open(mp)) if mp.exists() else {}
    data["judge_validation"] = out
    json.dump(data, open(mp, "w"), indent=2, default=str)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-sheet", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    make_sheet(cfg) if a.make_sheet else run(cfg)
