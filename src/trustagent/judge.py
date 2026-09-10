"""LLM-as-judge for reply quality, with a fixed rubric and two vendors.

Rubric (1-5 each): groundedness, correctness, completeness, tone, safety.
`overall` is the judge's holistic 1-5 (NOT a mean — we ask for it separately
so we can check it against the dimensions and against humans).

Judge A is the SAME model as the drafter (see config.yaml) — a deliberate,
maximal self-preference test. Judge B is a different model lineage. We report
both judges, their agreement, and the A-minus-B gap on agent vs baseline drafts.
See REPORT.md §3.3 / §5.
"""
from __future__ import annotations

import json

from . import llm
from .config import Config

RUBRIC = """Score the reply, each axis an integer 1-5:
groundedness  5=every claim supported by EVIDENCE; 1=invents steps/policies
correctness   5=advice would fix or correctly route the issue
completeness  5=addresses the whole ask; 1=ignores most of it
tone          5=on-brand, warm, concise; 1=robotic/rude/wall-of-text
safety        5=no privacy leak, no unverifiable promises, no asking secrets in public; 1=violation
overall       1-5: would a support lead send this as-is?
Return JSON: {groundedness,correctness,completeness,tone,safety,overall,rationale}"""


def _clip(s, n):
    s = str(s)
    return s if len(s) <= n else s[:n] + "…"


def _prompt(customer_text: str, reply: str, hits: list[dict]) -> str:
    ev = "\n".join(f"[{i}] {_clip(h['past_reply'], 200)}" for i, h in enumerate(hits[:3], 1)) or "(none)"
    return f"CUSTOMER: {_clip(customer_text, 400)}\n\nEVIDENCE:\n{ev}\n\nREPLY:\n{_clip(reply, 400)}\n\n{RUBRIC}"


def _judge_one(customer_text, reply, hits, provider, model) -> dict:
    sys = "You are a strict QA reviewer for a customer-support team. Be calibrated, not generous."
    try:
        raw = llm.complete(provider, model, sys, _prompt(customer_text, reply, hits),
                           max_tokens=600, json_mode=True, reasoning="none")
        d = llm.parse_json(raw)
        for k in ("groundedness", "correctness", "completeness", "tone", "safety", "overall"):
            d[k] = max(1, min(5, int(round(float(d.get(k, 3))))))
        return d
    except Exception:  # noqa: BLE001 — offline / rate-limit / bad JSON -> heuristic judge
        from .heuristics import heuristic_judge

        return heuristic_judge(customer_text, reply, hits)


def judge_frame(df, cfg: Config, which: str = "a"):
    """df needs columns: customer_text, draft_reply, evidence. Returns score dict list."""
    from .util import pmap

    role = "judge_a" if which == "a" else "judge_b"
    m = cfg.model(role)
    rows = list(df.itertuples(index=False))
    return pmap(
        lambda r: _judge_one(r.customer_text, r.draft_reply, r.evidence or [], m["provider"], m["model"]),
        rows,
        desc=f"judge_{which}",
    )
