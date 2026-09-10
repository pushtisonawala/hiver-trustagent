"""Ingest twcs.csv, reconstruct threads, emit (customer message -> brand reply) pairs.

The Kaggle schema is:
  tweet_id, author_id, inbound, created_at, text, response_tweet_id, in_response_to_tweet_id

`author_id` is a brand handle (e.g. "SpotifyCares") for company tweets and an
opaque integer for customers. `inbound=True` means customer->company.
"""
from __future__ import annotations

import html
import re
from pathlib import Path

import pandas as pd

_URL = re.compile(r"https?://\S+|pic\.twitter\.com/\S+")
_MENTION = re.compile(r"@\w+")
_WS = re.compile(r"\s+")
_TW_FMT = "%a %b %d %H:%M:%S %z %Y"


def clean_text(t: str) -> str:
    if not isinstance(t, str):
        return ""
    t = html.unescape(t)
    t = _URL.sub(" ", t)
    t = _MENTION.sub(" ", t)          # drop @handles; they carry no intent signal
    t = t.replace("\n", " ")
    t = _WS.sub(" ", t).strip()
    return t


def _read_brand_rows(csv_path: Path, brand: str, chunksize: int = 400_000) -> pd.DataFrame:
    """Two passes over the big CSV so we never lose a customer tweet that sits
    in a different chunk from the brand's reply.

    Pass 1 collects: brand tweet ids, and the parent ids the brand replied to.
    Pass 2 keeps: brand rows, those parents (customer first messages), and any
    row replying to a brand tweet (customer follow-ups).
    """
    brand_lower = brand.lower()
    brand_ids: set[str] = set()
    parent_ids: set[str] = set()
    for chunk in pd.read_csv(csv_path, dtype=str, chunksize=chunksize,
                             usecols=["tweet_id", "author_id", "in_response_to_tweet_id"]):
        m = chunk["author_id"].str.lower() == brand_lower
        brand_ids.update(chunk.loc[m, "tweet_id"])
        parent_ids.update(chunk.loc[m, "in_response_to_tweet_id"].dropna())

    wanted = brand_ids | parent_ids
    keep = []
    for chunk in pd.read_csv(csv_path, dtype=str, chunksize=chunksize):
        chunk.columns = [c.strip() for c in chunk.columns]
        near = (
            (chunk["author_id"].str.lower() == brand_lower)
            | chunk["tweet_id"].isin(wanted)
            | chunk["in_response_to_tweet_id"].isin(brand_ids)
        )
        keep.append(chunk.loc[near])
    return pd.concat(keep, ignore_index=True) if keep else pd.DataFrame()


def load_pairs(csv_path: str | Path, brand: str, *, min_customer_chars: int = 15) -> pd.DataFrame:
    """Return one row per (first customer message, first brand reply) pair."""
    csv_path = Path(csv_path)
    raw = _read_brand_rows(csv_path, brand)
    if raw.empty:
        raise SystemExit(f"No rows for brand '{brand}'. Run `make scan-brands` to see valid names.")

    raw["inbound"] = raw["inbound"].astype(str).str.lower().map({"true": True, "false": False})
    raw["created_at"] = pd.to_datetime(raw["created_at"], format=_TW_FMT, errors="coerce", utc=True)
    by_id = {r.tweet_id: r for r in raw.itertuples(index=False)}

    rows = []
    brand_lower = brand.lower()
    for br in raw.itertuples(index=False):
        if str(br.author_id).lower() != brand_lower:
            continue
        parent_id = br.in_response_to_tweet_id
        cust = by_id.get(parent_id)
        if cust is None or not cust.inbound:
            continue
        cust_clean = clean_text(cust.text)
        if len(cust_clean) < min_customer_chars:
            continue

        # customer follow-up after the brand reply -> resolution signal
        follow_ids = [x for x in str(br.response_tweet_id or "").split(",") if x]
        follow_txt = ""
        for fid in follow_ids:
            f = by_id.get(fid.strip())
            if f is not None and f.inbound:
                follow_txt = clean_text(f.text)
                break

        rows.append(
            {
                "thread_id": cust.tweet_id,
                "customer_id": cust.author_id,
                "customer_raw": cust.text,
                "customer_text": cust_clean,
                "brand_raw": br.text,
                "brand_text": clean_text(br.text),
                "customer_followup": follow_txt,
                "ts_customer": cust.created_at,
                "ts_brand": br.created_at,
            }
        )

    df = pd.DataFrame(rows).dropna(subset=["ts_customer"]).drop_duplicates("thread_id")
    # deterministic order (thread_id tiebreaker) so re-runs hit the response cache
    df = df.sort_values(["ts_customer", "thread_id"], kind="stable").reset_index(drop=True)
    return df


def temporal_split(df: pd.DataFrame, quantile: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on the oldest `quantile` of threads, test on the newest.

    A random split would let the retrieval index see the future and inflate
    every number. See REPORT.md "What is misleading about my headline number".
    """
    keys = ["ts_customer", "thread_id"] if "thread_id" in df.columns else ["ts_customer"]
    df = df.sort_values(keys, kind="stable").reset_index(drop=True)
    k = int(len(df) * quantile)
    return df.iloc[:k].reset_index(drop=True), df.iloc[k:].reset_index(drop=True)
