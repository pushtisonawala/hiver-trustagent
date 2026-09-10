from trustagent.config import load_config
from trustagent.policy import decide

CFG = load_config()


def _d(**kw):
    base = dict(customer_text="my playlist won't load", intent="playback_or_app_bug",
               intent_confidence=0.9, retrieval_support=0.8,
               draft={"grounded": True, "missing_info": []}, cfg=CFG)
    base.update(kw)
    return decide(**base)


def test_clean_low_risk_case_auto_handles():
    assert _d()["action"] == "auto_handle"


def test_hard_keyword_forces_escalation():
    r = _d(customer_text="my account was hacked and someone is playing music")
    assert r["action"] == "escalate" and "high-risk phrase" in r["reason"]


def test_high_risk_intent_without_precedent_escalates():
    r = _d(intent="cancel_or_refund", retrieval_support=0.2)
    assert r["action"] == "escalate"


def test_ungrounded_draft_escalates():
    r = _d(draft={"grounded": False, "missing_info": ["account email"]})
    assert r["action"] == "escalate"
