"""Text -> vector. OpenAI embeddings when available, TF-IDF otherwise.

TF-IDF is not just a fallback — it is also the retrieval backend for the
`simple` baseline, so we keep it first-class. See DECISION_LOG.md #6.
"""
from __future__ import annotations

import os

import numpy as np


def openai_available() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY")) and os.environ.get("TRUSTAGENT_OFFLINE", "0") != "1"


def embed_openai(texts: list[str], model: str = "text-embedding-3-small", batch: int = 256) -> np.ndarray:
    from openai import OpenAI

    client = OpenAI()
    out: list[list[float]] = []
    for i in range(0, len(texts), batch):
        chunk = [t.replace("\n", " ")[:8000] or " " for t in texts[i : i + batch]]
        resp = client.embeddings.create(model=model, input=chunk)
        out.extend(d.embedding for d in resp.data)
    arr = np.asarray(out, dtype=np.float32)
    return arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)


class TfidfEmbedder:
    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vec = TfidfVectorizer(min_df=2, max_features=12000, ngram_range=(1, 2),
                                   sublinear_tf=True, stop_words="english")
        self._fitted = False

    def fit(self, texts: list[str]):
        self.vec.fit(texts)
        self._fitted = True
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        m = self.vec.transform(texts).astype(np.float32).toarray()
        return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)
