from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Intent classification                                                        #
# --------------------------------------------------------------------------- #
def intent_report(y_true, y_pred, labels):
    from sklearn.metrics import classification_report, confusion_matrix, f1_score

    rep = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    return {
        "accuracy": float(np.mean([a == b for a, b in zip(y_true, y_pred)])),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "per_class": {k: rep[k] for k in labels if k in rep},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
    }


def expected_calibration_error(confidences, correct, n_bins: int = 10) -> float:
    conf = np.asarray(confidences, float)
    correct = np.asarray(correct, float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


# --------------------------------------------------------------------------- #
# Escalation gate                                                              #
# --------------------------------------------------------------------------- #
def escalation_report(gold_should_escalate, pred_action, *, cost_fa=5.0, cost_fe=1.0):
    """positive class = 'should escalate'."""
    y = np.asarray([bool(x) for x in gold_should_escalate])
    p = np.asarray([a == "escalate" for a in pred_action])
    tp = int((y & p).sum()); fp = int((~y & p).sum())
    fn = int((y & ~p).sum()); tn = int((~y & ~p).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    n = len(y)
    return {
        "precision": prec, "recall": rec, "f1": f1,
        "auto_handle_rate": float((~p).mean()),
        "false_auto_handle": fn,              # shipped a bad auto-reply
        "needless_escalation": fp,
        "cost_per_msg": float((cost_fa * fn + cost_fe * fp) / n) if n else 0.0,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


# --------------------------------------------------------------------------- #
# Reply quality (judge)                                                        #
# --------------------------------------------------------------------------- #
def quality_report(scores: list[dict], actions: list[str], accept_threshold: int = 4):
    ov = np.array([s["overall"] for s in scores], float)
    acc = ov >= accept_threshold
    auto = np.array([a == "auto_handle" for a in actions])
    dims = ("groundedness", "correctness", "completeness", "tone", "safety")
    out = {
        "mean_overall_all": float(ov.mean()),
        "acceptable_rate_all": float(acc.mean()),
        "mean_overall_auto_only": float(ov[auto].mean()) if auto.any() else None,
        "acceptable_rate_auto_only": float(acc[auto].mean()) if auto.any() else None,
        "safe_automation_rate": float((auto & acc).mean()),   # headline candidate
        "unsafe_auto_rate": float((auto & ~acc).mean()),      # the number that matters
        "per_dimension_mean": {d: float(np.mean([s[d] for s in scores])) for d in dims},
        "n": len(scores),
    }
    return out


# --------------------------------------------------------------------------- #
# Uncertainty                                                                  #
# --------------------------------------------------------------------------- #
def bootstrap_ci(values, statistic=np.mean, iters=2000, seed=13, alpha=0.05):
    rng = np.random.default_rng(seed)
    v = np.asarray(values, float)
    if len(v) == 0:
        return (float("nan"), float("nan"), float("nan"))
    boot = [statistic(rng.choice(v, size=len(v), replace=True)) for _ in range(iters)]
    return (float(statistic(v)), float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2)))


def deferral_curve(confidence, quality_overall, n_points=15):
    """Sweep an auto-handle confidence threshold; report (coverage, mean quality
    on the auto-handled subset). This is selective prediction — the headline
    quality number is meaningless without it."""
    c = np.asarray(confidence, float)
    q = np.asarray(quality_overall, float)
    pts = []
    for thr in np.linspace(c.min(), c.max(), n_points):
        auto = c >= thr
        pts.append({
            "threshold": float(thr),
            "coverage": float(auto.mean()),
            "mean_quality_auto": float(q[auto].mean()) if auto.any() else None,
        })
    return pts


# --------------------------------------------------------------------------- #
# Judge validation vs humans                                                   #
# --------------------------------------------------------------------------- #
def judge_agreement(judge_overall, human_overall, accept_threshold=4):
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import cohen_kappa_score

    j = np.asarray(judge_overall, float); h = np.asarray(human_overall, float)
    ja = (j >= accept_threshold).astype(int); ha = (h >= accept_threshold).astype(int)
    return {
        "n": int(len(j)),
        "spearman": float(spearmanr(j, h).statistic),
        "pearson": float(pearsonr(j, h).statistic),
        "mae": float(np.abs(j - h).mean()),
        "within_1": float((np.abs(j - h) <= 1).mean()),
        "cohen_kappa_accept": float(cohen_kappa_score(ja, ha)) if ha.std() and ja.std() else None,
        "judge_mean_minus_human_mean": float(j.mean() - h.mean()),
    }
