"""Intent taxonomy + three classifiers (majority / TF-IDF-LogReg / LLM few-shot).

Taxonomy provenance (decision #4): we embedded ~2k training messages, ran
KMeans(k=12), read the top TF-IDF terms + 10 nearest messages per cluster,
then hand-merged to the 9 labels in config.yaml. `bootstrap_taxonomy()`
reproduces the clustering evidence; the final labels are curated, not raw.
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np

from . import llm
from .config import Config

DEFINITIONS = {
    "playback_or_app_bug": "Songs won't play / skip / cut out, app crashes or freezes, "
    "offline downloads broken, audio quality, shuffle behaviour.",
    "account_access": "Can't log in, password reset, email/username change, "
    "account hacked or compromised, unexpected logout, 2FA.",
    "billing_subscription": "Charged unexpectedly, wrong amount, payment method won't update, "
    "receipts/invoices, upgrade/downgrade Premium, student/trial pricing.",
    "cancel_or_refund": "Wants to cancel Premium, asking for a refund, "
    "disputing a specific charge, 'stop charging me'.",
    "family_or_duo_plan": "Family/Duo plan management, address verification, "
    "adding/removing members, plan admin vs member confusion.",
    "content_availability": "A song/album/podcast is missing, removed, greyed out, "
    "region-locked, or wrong metadata.",
    "device_or_connect": "Spotify Connect, cars (Android Auto/CarPlay), smart speakers, "
    "Alexa/Google Home, PS/Xbox, casting between devices.",
    "feature_request_or_feedback": "Feature requests, UX complaints, general praise or "
    "venting that is not an actionable support issue.",
    "other": "Spam, jokes, unclear, non-Spotify, or nothing actionable.",
}


def few_shot_block() -> str:
    return "\n".join(f"- {k}: {v}" for k, v in DEFINITIONS.items())


# compact one-liners for the per-call classify prompt (the full DEFINITIONS blow
# the free-tier 6-8k tokens/min cap when sent on every request)
SHORT_DEFS = {
    "playback_or_app_bug": "won't play / skips / crashes / offline downloads broken",
    "account_access": "login, password reset, hacked, 'not premium anymore'",
    "billing_subscription": "charge/price/payment-method/student-discount/promo issues",
    "cancel_or_refund": "wants to cancel or get money back",
    "family_or_duo_plan": "family/duo plan admin, address, members",
    "content_availability": "song/album/podcast missing, region-locked, wrong metadata",
    "device_or_connect": "Connect, car, Alexa, speakers, consoles, casting",
    "feature_request_or_feedback": "feature asks, UX complaints, praise/venting",
    "other": "spam, jokes, unclear, non-Spotify, nothing actionable",
}


def short_block() -> str:
    return "\n".join(f"- {k}: {v}" for k, v in SHORT_DEFS.items())


# --------------------------------------------------------------------------- #
# Classifiers                                                                  #
# --------------------------------------------------------------------------- #
class MajorityClassifier:
    name = "majority"

    def fit(self, texts, labels):
        self.major = Counter(labels).most_common(1)[0][0]
        return self

    def predict(self, texts):
        return [self.major] * len(texts)

    def predict_proba(self, texts):
        return [{self.major: 1.0} for _ in texts]


class TfidfLogReg:
    name = "tfidf_logreg"

    def __init__(self, seed: int = 13):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        self._fallback = None
        self.pipe = Pipeline(
            [
                ("tfidf", TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True,
                                          stop_words="english")),
                ("clf", LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced",
                                           random_state=seed)),
            ]
        )

    def fit(self, texts, labels):
        self._fallback = None
        if len(set(labels)) < 2:
            self._fallback = Counter(labels).most_common(1)[0][0] if labels else "other"
            return self
        self.pipe.fit(texts, labels)
        self.classes_ = list(self.pipe.classes_)
        return self

    def predict(self, texts):
        if self._fallback:
            return [self._fallback] * len(texts)
        return list(self.pipe.predict(texts))

    def predict_proba(self, texts):
        if self._fallback:
            return [{self._fallback: 1.0} for _ in texts]
        P = self.pipe.predict_proba(texts)
        return [dict(zip(self.classes_, row)) for row in P]


class LLMClassifier:
    name = "llm_fewshot"

    def __init__(self, cfg: Config, role: str = "agent"):
        self.cfg = cfg
        m = cfg.model(role)
        self.provider, self.model = m["provider"], m["model"]
        self.labels = cfg.intent_labels

    def _one(self, text: str) -> dict:
        sys = (
            "Classify this Spotify support tweet. Pick exactly one:\n" + short_block() +
            "\nReturn JSON {\"intent\": <label>, \"confidence\": 0..1}."
        )
        try:
            raw = llm.complete(self.provider, self.model, sys, text[:400], max_tokens=200,
                               json_mode=True, reasoning="none")
            d = llm.parse_json(raw)
            intent = d.get("intent", "other")
            if intent not in self.labels:
                intent = "other"
            return {"intent": intent, "confidence": float(d.get("confidence", 0.5))}
        except Exception:  # noqa: BLE001 — offline, rate-limit, or bad JSON -> heuristic floor
            from .heuristics import heuristic_intent

            return heuristic_intent(text, self.labels)

    def fit(self, texts=None, labels=None):
        return self

    def predict(self, texts):
        from .util import pmap

        return [r["intent"] for r in pmap(self._one, texts, desc="intent")]

    def predict_proba(self, texts):
        from .util import pmap

        return [{r["intent"]: r["confidence"]} for r in pmap(self._one, texts, desc="intent")]


def weak_label(texts: list[str], cfg: Config) -> list[str]:
    """LLM weak-labels the training split so the cheap baseline has something
    to learn from (distillation). Uses the small, high-rate-limit weak_labeler."""
    return LLMClassifier(cfg, role="weak_labeler").predict(texts)


def bootstrap_taxonomy(texts: list[str], vectors: np.ndarray, k: int = 12, seed: int = 13) -> dict:
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    km = KMeans(n_clusters=k, random_state=seed, n_init=3).fit(vectors)
    tf = TfidfVectorizer(min_df=3, stop_words="english", ngram_range=(1, 2))
    X = tf.fit_transform(texts)
    terms = np.array(tf.get_feature_names_out())
    out = {}
    for c in range(k):
        idx = np.where(km.labels_ == c)[0]
        if len(idx) == 0:
            continue
        centroid = np.asarray(X[idx].mean(axis=0)).ravel()
        top = terms[centroid.argsort()[::-1][:12]].tolist()
        examples = [texts[i] for i in idx[:8]]
        out[f"cluster_{c}"] = {"size": int(len(idx)), "top_terms": top, "examples": examples}
    return out
