import numpy as np
import pandas as pd

from trustagent.data import clean_text, temporal_split
from trustagent.metrics import bootstrap_ci, escalation_report, judge_agreement
from trustagent.resolution import resolution_score


def test_clean_text_strips_mentions_urls_entities():
    assert clean_text("@SpotifyCares my app won't play &amp; https://t.co/x") == "my app won't play &"
    assert clean_text(None) == ""


def test_temporal_split_is_time_ordered_not_random():
    ts = pd.to_datetime(pd.date_range("2017-10-01", periods=100, freq="h", tz="UTC"))
    df = pd.DataFrame({"ts_customer": ts, "customer_text": list("x" * 100)})
    train, test = temporal_split(df, 0.8)
    assert train["ts_customer"].max() <= test["ts_customer"].min()
    assert 0 < len(test) < len(train)


def test_resolution_score_reads_followup_sentiment():
    assert resolution_score("go to settings and toggle offline mode", "thank you, worked!") > 0.7
    assert resolution_score("please DM us", "still not working, same problem") < 0.3


def test_escalation_cost_is_asymmetric():
    gold = [True, True, False, False]
    # one false auto-handle (missed escalation) vs one needless escalation
    r = escalation_report(gold, ["escalate", "auto_handle", "escalate", "auto_handle"],
                          cost_fa=5.0, cost_fe=1.0)
    assert r["false_auto_handle"] == 1 and r["needless_escalation"] == 1
    assert abs(r["cost_per_msg"] - (5.0 + 1.0) / 4) < 1e-9


def test_bootstrap_ci_brackets_mean():
    mean, lo, hi = bootstrap_ci([1, 2, 3, 4, 5] * 20, iters=500)
    assert lo < mean < hi


def test_judge_agreement_perfect():
    a = judge_agreement([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    assert a["mae"] == 0 and a["spearman"] > 0.99
