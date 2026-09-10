"""Baselines. Two trivial, one simple. See REPORT.md "Results vs baselines"."""
from __future__ import annotations

import re

import pandas as pd

from .config import Config
from .intents import MajorityClassifier, TfidfLogReg
from .retrieval import Retriever, support_score

CANNED = ("Thanks for the message! One of our team will take a look and get back to you shortly.")


class TrivialAlwaysEscalate:
    """Majority intent, canned reply, escalate everything. The 'do nothing' floor."""

    name = "trivial_always_escalate"

    def __init__(self, cfg: Config, majority_intent: str):
        self.mi = majority_intent

    def run_batch(self, texts):
        return pd.DataFrame(
            [{"intent": self.mi, "intent_confidence": 1.0, "retrieval_support": 0.0,
              "evidence": [], "draft_reply": CANNED, "grounded": False, "missing_info": [],
              "action": "escalate", "reason": "trivial policy: escalate all", "risk_tier": "medium"}
             for _ in texts]
        )


class TrivialAlwaysAuto(TrivialAlwaysEscalate):
    """Same, but auto-handle everything. Shows what 'maximise automation' costs."""

    name = "trivial_always_auto"

    def run_batch(self, texts):
        df = super().run_batch(texts)
        df["action"] = "auto_handle"
        df["reason"] = "trivial policy: auto-handle all"
        return df


class SimpleBaseline:
    """TF-IDF+LogReg intent (distilled from weak labels) + verbatim top-1 past
    reply + rule-based escalation. No generative model in the loop."""

    name = "simple"

    def __init__(self, cfg: Config, retriever: Retriever, train_texts, train_weak_labels):
        self.cfg = cfg
        self.retriever = retriever
        self.clf = TfidfLogReg(seed=cfg.seed).fit(train_texts, train_weak_labels)

    def _escalate(self, text, intent, support, conf) -> tuple[bool, str]:
        pol = self.cfg["policy"]
        tl = text.lower()
        kw = [k for k in pol["hard_escalate_keywords"] if k in tl]
        if kw:
            return True, f"keyword: {kw[0]}"
        if self.cfg.intent_risk.get(intent) == "high":
            return True, f"high-risk intent {intent}"
        if support < pol["min_retrieval_support"]:
            return True, "no precedent"
        if conf < pol["min_intent_confidence"]:
            return True, "low confidence"
        return False, "auto"

    def run_batch(self, texts):
        probas = self.clf.predict_proba(texts)
        rows = []
        for t, p in zip(texts, probas):
            intent = max(p, key=p.get)
            conf = float(p[intent])
            hits = [h for h in self.retriever.query(t, k=self.cfg["retrieval"]["k"])
                    if h["similarity"] >= self.cfg["retrieval"]["min_similarity"]]
            support = support_score(hits)
            reply = hits[0]["past_reply"] if hits else CANNED
            esc, why = self._escalate(t, intent, support, conf)
            rows.append({
                "intent": intent, "intent_confidence": conf, "retrieval_support": support,
                "evidence": hits, "draft_reply": reply, "grounded": bool(hits),
                "missing_info": [], "action": "escalate" if esc else "auto_handle",
                "reason": why, "risk_tier": self.cfg.intent_risk.get(intent, "medium"),
            })
        return pd.DataFrame(rows)
