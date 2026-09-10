from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

WORKERS = int(os.environ.get("TRUSTAGENT_WORKERS", "8"))


def pmap(fn, items, desc: str | None = None, workers: int | None = None):
    items = list(items)
    w = workers or WORKERS
    if w <= 1 or len(items) <= 1:
        return [fn(x) for x in tqdm(items, desc=desc, disable=not desc)]
    out = [None] * len(items)
    with ThreadPoolExecutor(max_workers=w) as ex:
        futs = {ex.submit(fn, x): i for i, x in enumerate(items)}
        for f in tqdm(futs, total=len(items), desc=desc, disable=not desc):
            out[futs[f]] = f.result()
    return out
