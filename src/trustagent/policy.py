"""Auto-handle vs escalate — a transparent rule stack with a stated reason.

Philosophy (decision #9): the gate is deliberately conservative and rule-based,
not a learned model. A reviewer can read exactly why any message escalated, and
each rule maps to a real operational risk. The asymmetric cost matrix
(cost_false_autohandle >> cost_false_escalate) is applied when we *tune* the
thresholds, not here.
"""
from __future__ import annotations

import re

from .config import Config


def decide(
    *,
    customer_text: str,
    intent: str,
    intent_confidence: float,
    retrieval_support: float,
    draft: dict,
    cfg: Config,
) -> dict:
    pol = cfg["policy"]
    risk = cfg.intent_risk.get(intent, "medium")
    reasons: list[str] = []

    text_l = customer_text.lower()
    hard_kw = [k for k in pol["hard_escalate_keywords"] if k in text_l]
    if hard_kw:
        reasons.append(f"contains high-risk phrase(s): {', '.join(hard_kw)}")

    if intent_confidence < pol["min_intent_confidence"]:
        reasons.append(f"low intent confidence ({intent_confidence:.2f} < {pol['min_intent_confidence']})")

    if retrieval_support < pol["min_retrieval_support"]:
        reasons.append(f"weak historical precedent (support {retrieval_support:.2f} < {pol['min_retrieval_support']})")

    if risk == "high" and retrieval_support < 0.55:
        reasons.append(f"high-risk intent '{intent}' without strong precedent")

    if not draft.get("grounded", False):
        reasons.append("drafter could not ground the reply in evidence")

    if draft.get("missing_info"):
        reasons.append(f"needs account-specific info: {', '.join(draft['missing_info'])}")

    if re.search(r"\b(angry|furious|disgusting|worst|never again|cancel(l)?ing|switching to)\b", text_l):
        reasons.append("churn / strong-negative sentiment signal")

    action = "escalate" if reasons else "auto_handle"
    return {
        "action": action,
        "reason": "; ".join(reasons) if reasons else "clear intent, strong precedent, grounded low-risk reply",
        "risk_tier": risk,
    }
