"""Weak-supervision heuristic: did the brand's reply actually resolve the issue?

We only want to ground drafts in replies that *worked*. There is no label for
this in the data, so we infer it from the customer's follow-up. This heuristic
is deliberately called out as a limitation in REPORT.md — "thanks" is not the
same as "resolved", and silent churn looks identical to success.
"""
from __future__ import annotations

import re

_POS = re.compile(
    r"\b(thanks?|thank you|thx|ty|appreciate|perfect|great|awesome|worked|working now|"
    r"sorted|solved|fixed|got it|legend|cheers|love you)\b",
    re.I,
)
_NEG = re.compile(
    r"\b(still|not working|doesn'?t work|didn'?t work|won'?t|same (issue|problem|thing)|"
    r"useless|unhelpful|ridiculous|again|nothing works|already (tried|did)|no help)\b",
    re.I,
)
_DEFLECT = re.compile(r"\b(dm|direct message|shoot us|send us)\b", re.I)


def resolution_score(brand_text: str, followup: str) -> float:
    s = 0.5
    bt, fu = brand_text or "", followup or ""

    if _DEFLECT.search(bt) and len(bt) < 120:
        s -= 0.15  # pure "please DM us" — we cannot learn a resolution from it

    if not fu:
        s += 0.15  # customer did not come back to complain
    else:
        if _POS.search(fu):
            s += 0.35
        if _NEG.search(fu):
            s -= 0.4

    # a concrete instruction (steps, links, settings) is more groundable
    if re.search(r"\b(settings|toggle|reinstall|log ?out|update the app|offline mode|"
                 r"restart|clear cache|tap|go to)\b", bt, re.I):
        s += 0.1

    return max(0.0, min(1.0, s))


def is_resolved(brand_text: str, followup: str, threshold: float = 0.6) -> bool:
    return resolution_score(brand_text, followup) >= threshold
