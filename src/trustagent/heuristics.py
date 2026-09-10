"""Offline fallbacks so the whole pipeline runs (and the harness reproduces)
with zero API keys. These are intentionally weak — they exist to keep the
system runnable and to serve as an honest floor, not to be good.
"""
from __future__ import annotations

import re

_RULES = [
    ("account_access", r"\b(log ?in|login|password|hack|compromis|can'?t access|locked out|2fa|verification code)\b"),
    ("cancel_or_refund", r"\b(cancel|refund|money back|stop charging|unsubscribe)\b"),
    ("billing_subscription", r"\b(charg|payment|card|invoice|receipt|billed|price|subscription|premium cost)\b"),
    ("family_or_duo_plan", r"\b(family plan|duo|family member|address (verif|confirm))\b"),
    ("device_or_connect", r"\b(connect|alexa|google home|car ?play|android auto|speaker|chromecast|xbox|playstation|ps5|sonos)\b"),
    ("content_availability", r"\b(missing|removed|can'?t find|greyed out|not available|region|podcast episode|taken down)\b"),
    ("playback_or_app_bug", r"\b(won'?t play|not playing|skip|crash|freez|offline|download|buffer|stops? playing|error code)\b"),
    ("feature_request_or_feedback", r"\b(wish|should add|feature|please add|hate the new|bring back|ui|redesign)\b"),
]


def heuristic_intent(text: str, labels: list[str]) -> dict:
    t = text.lower()
    for label, pat in _RULES:
        if label in labels and re.search(pat, t):
            return {"intent": label, "confidence": 0.45}
    return {"intent": "other", "confidence": 0.3}


def heuristic_draft(customer_text: str, hits: list[dict]) -> dict:
    if hits and hits[0]["similarity"] > 0.3:
        reply = hits[0]["past_reply"]
        return {"reply": reply, "used_evidence": [hits[0]["thread_id"]],
                "grounded": True, "missing_info": []}
    return {
        "reply": "Thanks for reaching out — so we can dig into this, could you DM us your "
        "account email and a bit more detail on what you're seeing?",
        "used_evidence": [],
        "grounded": False,
        "missing_info": ["account-specific detail"],
    }


def heuristic_judge(customer_text: str, reply: str, hits: list[dict]) -> dict:
    words = len(reply.split())
    grounded = any(h["past_reply"][:40].lower() in reply.lower() for h in hits) if hits else False
    length_ok = 8 <= words <= 90
    asks_dm = "dm" in reply.lower() or "direct message" in reply.lower()
    score = 2 + int(length_ok) + int(grounded) + int(not (asks_dm and words < 20))
    score = max(1, min(5, score))
    return {
        "groundedness": 4 if grounded else 2,
        "correctness": score,
        "completeness": score,
        "tone": 4 if length_ok else 3,
        "safety": 5,
        "overall": score,
        "rationale": "heuristic judge (offline mode)",
    }
