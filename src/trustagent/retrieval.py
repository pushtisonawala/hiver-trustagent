"""Retrieve historically-similar resolved cases for grounding a draft reply.

Index = training-split pairs only (temporal cutoff already applied upstream).
Each hit carries the past customer message, the brand's reply, and the
resolution score from resolution.py so the drafter can prefer replies that
actually worked.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from . import embeddings
from .resolution import resolution_score


class Retriever:
    def __init__(self, backend: str = "auto", emb_model: str = "text-embedding-3-small"):
        self.backend = backend
        self.emb_model = emb_model

    def fit(self, train: pd.DataFrame):
        self.df = train.reset_index(drop=True).copy()
        self.df["resolution_score"] = [
            resolution_score(b, f) for b, f in zip(self.df.brand_text, self.df.customer_followup)
        ]
        corpus = self.df["customer_text"].tolist()

        use_openai = self.backend == "openai" or (self.backend == "auto" and embeddings.openai_available())
        if use_openai:
            self.mode = "openai"
            self.M = embeddings.embed_openai(corpus, self.emb_model)
        else:
            self.mode = "tfidf"
            self.embedder = embeddings.TfidfEmbedder().fit(corpus)
            self.M = self.embedder.transform(corpus)
        return self

    def _vec(self, texts: list[str]) -> np.ndarray:
        if self.mode == "openai":
            return embeddings.embed_openai(texts, self.emb_model)
        return self.embedder.transform(texts)

    def query(self, text: str, k: int = 4) -> list[dict]:
        q = self._vec([text])[0]
        sims = self.M @ q
        order = np.argsort(sims)[::-1][:k]
        hits = []
        for i in order:
            r = self.df.iloc[int(i)]
            hits.append(
                {
                    "thread_id": r.thread_id,
                    "similarity": float(sims[i]),
                    "past_customer": r.customer_text,
                    "past_reply": r.brand_text,
                    "resolution_score": float(r.resolution_score),
                }
            )
        return hits


    def save(self, path: str | Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "matrix.npy", self.M)
        self.df.to_parquet(path / "pairs.parquet")
        meta = {"mode": self.mode, "emb_model": self.emb_model}
        (path / "meta.pkl").write_bytes(pickle.dumps(meta))
        if self.mode == "tfidf":
            (path / "embedder.pkl").write_bytes(pickle.dumps(self.embedder))

    @classmethod
    def load(cls, path: str | Path) -> "Retriever":
        path = Path(path)
        meta = pickle.loads((path / "meta.pkl").read_bytes())
        r = cls(backend=meta["mode"], emb_model=meta["emb_model"])
        r.mode = meta["mode"]
        r.M = np.load(path / "matrix.npy")
        r.df = pd.read_parquet(path / "pairs.parquet")
        if r.mode == "tfidf":
            r.embedder = pickle.loads((path / "embedder.pkl").read_bytes())
        return r


def support_score(hits: list[dict]) -> float:
    """How much precedent do we actually have? similarity x resolution, top-weighted."""
    if not hits:
        return 0.0
    vals = [h["similarity"] * (0.5 + 0.5 * h["resolution_score"]) for h in hits]
    return float(0.6 * vals[0] + 0.4 * (np.mean(vals[1:]) if len(vals) > 1 else vals[0]))
