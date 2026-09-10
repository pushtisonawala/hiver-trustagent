"""End-to-end agent: classify -> retrieve -> draft -> decide."""
from __future__ import annotations

import pandas as pd

from .config import Config
from .drafting import draft_reply
from .intents import LLMClassifier
from .policy import decide
from .retrieval import Retriever, support_score


class TrustAgent:
    def __init__(self, cfg: Config, retriever: Retriever, classifier=None):
        self.cfg = cfg
        self.retriever = retriever
        self.classifier = classifier or LLMClassifier(cfg)

    def handle(self, customer_text: str) -> dict:
        proba = self.classifier.predict_proba([customer_text])[0]
        intent = max(proba, key=proba.get)
        confidence = float(proba[intent])

        hits = self.retriever.query(customer_text, k=self.cfg["retrieval"]["k"])
        hits = [h for h in hits if h["similarity"] >= self.cfg["retrieval"]["min_similarity"]]
        support = support_score(hits)

        draft = draft_reply(customer_text, intent, hits, self.cfg)
        decision = decide(
            customer_text=customer_text,
            intent=intent,
            intent_confidence=confidence,
            retrieval_support=support,
            draft=draft,
            cfg=self.cfg,
        )
        return {
            "intent": intent,
            "intent_confidence": confidence,
            "retrieval_support": support,
            "evidence": hits,
            "draft_reply": draft.get("reply", ""),
            "grounded": draft.get("grounded", False),
            "missing_info": draft.get("missing_info", []),
            "action": decision["action"],
            "reason": decision["reason"],
            "risk_tier": decision["risk_tier"],
        }

    def run_batch(self, texts: list[str]) -> pd.DataFrame:
        from .util import pmap

        return pd.DataFrame(pmap(self.handle, texts, desc="agent"))
