"""Orchestration: `build` (data -> index) and `evaluate` (baselines + agent + judge)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import intents, metrics
from .agent import TrustAgent
from .baselines import SimpleBaseline, TrivialAlwaysAuto, TrivialAlwaysEscalate
from .config import Config, load_config
from .data import load_pairs, temporal_split
from .intents import LLMClassifier, MajorityClassifier, TfidfLogReg
from .judge import judge_frame
from .resolution import resolution_score
from .retrieval import Retriever


# --------------------------------------------------------------------------- #
def _art_dir(cfg: Config, smoke: bool) -> Path:
    # smoke runs write to a sibling dir so they never clobber a real `make all`
    p = cfg.path("artifacts") / ("_smoke" if smoke else "")
    p.mkdir(parents=True, exist_ok=True)
    return p


def _res_dir(cfg: Config, smoke: bool) -> Path:
    p = cfg.path("results") / ("_smoke" if smoke else "")
    p.mkdir(parents=True, exist_ok=True)
    return p


def build(cfg: Config, input_csv: str | None = None, smoke: bool = False):
    art = _art_dir(cfg, smoke)
    src = input_csv or (cfg.raw["paths"]["sample_csv"] if smoke else cfg.raw["paths"]["raw_csv"])

    print(f"[build] loading pairs for {cfg.brand} from {src}")
    df = load_pairs(src, cfg.brand, min_customer_chars=cfg["sampling"]["min_customer_chars"])
    print(f"[build] {len(df)} (customer -> brand) pairs")

    if not smoke and len(df) > cfg["sampling"]["max_threads"]:
        df = df.sample(cfg["sampling"]["max_threads"], random_state=cfg.seed)
        df = df.sort_values(["ts_customer", "thread_id"], kind="stable").reset_index(drop=True)

    train, test = temporal_split(df, cfg["sampling"]["temporal_split_quantile"])
    print(f"[build] temporal split: train={len(train)}  test={len(test)}  cutoff={test.ts_customer.min()}")
    train.to_parquet(art / "train.parquet")
    test.to_parquet(art / "test.parquet")

    print("[build] fitting retrieval index")
    emb = cfg.model("embeddings")
    retr = Retriever(backend=emb["provider"], emb_model=emb.get("model", "")).fit(train)
    retr.save(art / "retriever")
    print(f"[build] retriever mode = {retr.mode}")

    # weak labels for the distilled baseline (sample to keep the run fast)
    n = min(len(train), 220 if not smoke else len(train))
    sub = train.sample(n, random_state=cfg.seed) if n < len(train) else train
    print(f"[build] weak-labelling {len(sub)} training messages for the simple baseline")
    wl = intents.weak_label(sub["customer_text"].tolist(), cfg)
    json.dump(
        {"texts": sub["customer_text"].tolist(), "labels": wl},
        open(art / "weak_labels.json", "w"),
    )

    # taxonomy bootstrap evidence (for the report / decision log)
    try:
        vecs = retr.M[: min(1500, len(train))]
        boot = intents.bootstrap_taxonomy(train["customer_text"].tolist()[: len(vecs)], np.asarray(vecs), seed=cfg.seed)
        json.dump(boot, open(art / "taxonomy_bootstrap.json", "w"), indent=2)
    except Exception as e:  # noqa: BLE001
        print(f"[build] taxonomy bootstrap skipped: {e}")

    print("[build] done ->", art)


# --------------------------------------------------------------------------- #
def _load_golden(cfg: Config, smoke: bool) -> pd.DataFrame:
    if smoke:
        # derive a throwaway 'golden' set from the sample so smoke test is self-contained
        test = pd.read_parquet(_art_dir(cfg, True) / "test.parquet").copy()
        test["gold_intent"] = intents.LLMClassifier(cfg).predict(test["customer_text"].tolist())
        test["gold_action"] = np.where(
            test["customer_text"].str.contains("refund|cancel|hack", case=False), "escalate", "auto_handle"
        )
        test["reference_reply"] = test["brand_text"]
        return test
    g = pd.read_csv(cfg.path("golden"))
    need = {"customer_text", "gold_intent", "gold_action"}
    if not need.issubset(g.columns):
        raise SystemExit(f"{cfg.path('golden')} missing columns {need - set(g.columns)}. Run `make golden`.")
    return g


def evaluate(cfg: Config, smoke: bool = False):
    art = _art_dir(cfg, smoke)
    res = _res_dir(cfg, smoke)

    train = pd.read_parquet(art / "train.parquet")
    retr = Retriever.load(art / "retriever")
    golden = _load_golden(cfg, smoke).reset_index(drop=True)
    texts = golden["customer_text"].tolist()
    labels = cfg.intent_labels
    print(f"[eval] {len(golden)} golden rows")

    wl = json.load(open(art / "weak_labels.json"))
    majority_intent = pd.Series(wl["labels"]).value_counts().index[0]

    models = {
        "trivial_always_escalate": TrivialAlwaysEscalate(cfg, majority_intent),
        "trivial_always_auto": TrivialAlwaysAuto(cfg, majority_intent),
        "simple": SimpleBaseline(cfg, retr, wl["texts"], wl["labels"]),
        "agent": TrustAgent(cfg, retr, LLMClassifier(cfg)),
    }

    out: dict = {"brand": cfg.brand, "n_golden": len(golden), "seed": cfg.seed,
                 "retriever_mode": retr.mode, "models": {}}
    per_model_frames = {}

    for name, model in models.items():
        print(f"[eval] running {name}")
        pred = model.run_batch(texts).reset_index(drop=True)
        pred["customer_text"] = texts
        pred["gold_intent"] = golden["gold_intent"].values
        pred["gold_action"] = golden["gold_action"].values

        gold_should_escalate = golden["gold_action"].eq("escalate").tolist()
        m = {
            "intent": metrics.intent_report(golden["gold_intent"].tolist(), pred["intent"].tolist(), labels),
            "escalation": metrics.escalation_report(
                gold_should_escalate, pred["action"].tolist(),
                cost_fa=cfg["policy"]["cost_false_autohandle"], cost_fe=cfg["policy"]["cost_false_escalate"],
            ),
        }

        # judge reply quality for the systems whose reply is non-trivial.
        # (trivial baselines emit a constant canned string — no signal in judging it.)
        if name in ("simple", "agent"):
            cap = int(cfg["judge"].get("max_rows", 120))
            jpred = pred if len(pred) <= cap else pred.sample(cap, random_state=cfg.seed).reset_index(drop=True)
            ja = judge_frame(jpred, cfg, "a")
            jb = judge_frame(jpred, cfg, "b")
            pred = jpred  # quality metrics below are over the judged subset
            pred["judge_a_overall"] = [s["overall"] for s in ja]
            pred["judge_b_overall"] = [s["overall"] for s in jb]
            pred["judge_a_full"] = [json.dumps(s) for s in ja]
            m["quality_judge_a"] = metrics.quality_report(ja, pred["action"].tolist(),
                                                          cfg["judge"]["quality_accept_threshold"])
            m["quality_judge_b"] = metrics.quality_report(jb, pred["action"].tolist(),
                                                          cfg["judge"]["quality_accept_threshold"])
            j = np.array([s["overall"] for s in ja], float)
            mean_j, lo, hi = metrics.bootstrap_ci(j, iters=cfg["eval"]["bootstrap_iterations"], seed=cfg.seed)
            m["quality_judge_a"]["mean_overall_ci95"] = [mean_j, lo, hi]

        out["models"][name] = m
        pred.to_csv(res / f"predictions_{name}.csv", index=False)
        per_model_frames[name] = pred

    # --- selective prediction / deferral curve for the agent -----------------
    ag = per_model_frames["agent"]
    if "judge_a_overall" in ag:
        out["deferral_curve"] = metrics.deferral_curve(
            ag["intent_confidence"].tolist(), ag["judge_a_overall"].tolist(),
            n_points=cfg["eval"]["deferral_curve_points"],
        )
        correct = (ag["intent"] == ag["gold_intent"]).astype(int).tolist()
        out["models"]["agent"]["intent"]["ece"] = metrics.expected_calibration_error(
            ag["intent_confidence"].tolist(), correct
        )

    # --- self-preference / cross-vendor judge bias --------------------------
    if "judge_a_overall" in per_model_frames["agent"]:
        a_on_agent = per_model_frames["agent"]["judge_a_overall"].mean()
        b_on_agent = per_model_frames["agent"]["judge_b_overall"].mean()
        a_on_simple = per_model_frames["simple"]["judge_a_overall"].mean()
        b_on_simple = per_model_frames["simple"]["judge_b_overall"].mean()
        out["judge_cross_vendor"] = {
            "judge_a_model": cfg.model("judge_a")["model"],
            "judge_b_model": cfg.model("judge_b")["model"],
            "agent_drafts_by_": cfg.model("agent")["model"],
            "A_scores_agent": float(a_on_agent), "B_scores_agent": float(b_on_agent),
            "A_scores_simple": float(a_on_simple), "B_scores_simple": float(b_on_simple),
            "A_minus_B_on_agent": float(a_on_agent - b_on_agent),
            "note": "Agent drafts are written by the same vendor as Judge A. "
                    "A positive A_minus_B_on_agent that is larger than A_minus_B_on_simple "
                    "is a self-preference red flag.",
            "A_minus_B_on_simple": float(a_on_simple - b_on_simple),
        }

    json.dump(out, open(res / "metrics.json", "w"), indent=2, default=str)
    _write_results_md(out, res)
    _plots(out, per_model_frames, cfg, res)
    _failure_pool(per_model_frames.get("agent"), res)
    print("[eval] wrote", res / "metrics.json")
    return out


# --------------------------------------------------------------------------- #
def _write_results_md(out: dict, res: Path):
    L = [f"_{out['brand']}, golden n={out['n_golden']}, seed {out['seed']}, "
         f"retriever={out['retriever_mode']}. Regenerate with `make eval`._\n",
         "| model | intent acc | intent macroF1 | escal. P | escal. R | auto-handle rate | "
         "unsafe-auto rate | safe-automation rate | judgeA mean |",
         "|---|---|---|---|---|---|---|---|---|"]
    for name, m in out["models"].items():
        q = m.get("quality_judge_a", {})
        L.append(
            f"| {name} | {m['intent']['accuracy']:.2f} | {m['intent']['macro_f1']:.2f} | "
            f"{m['escalation']['precision']:.2f} | {m['escalation']['recall']:.2f} | "
            f"{m['escalation']['auto_handle_rate']:.2f} | "
            f"{q.get('unsafe_auto_rate', float('nan')):.3f} | "
            f"{q.get('safe_automation_rate', float('nan')):.2f} | "
            f"{q.get('mean_overall_all', float('nan')):.2f} |"
        )
    if "judge_cross_vendor" in out:
        cv = out["judge_cross_vendor"]
        L += ["\n**Judge A vs Judge B on the agent's replies:** "
              f"{cv['judge_a_model']} gives {cv['A_scores_agent']:.2f}, "
              f"{cv['judge_b_model']} gives {cv['B_scores_agent']:.2f} "
              f"(gap {cv['A_minus_B_on_agent']:+.2f}; on the simple baseline the gap is "
              f"{cv['A_minus_B_on_simple']:+.2f}). Neither passes the human check in §3.3."]
    (res / "RESULTS.md").write_text("\n".join(L) + "\n")


def _plots(out, frames, cfg, res: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # confusion matrix for the agent
    m = out["models"]["agent"]["intent"]
    cm = np.array(m["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(m["labels"]))); ax.set_xticklabels(m["labels"], rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(m["labels"]))); ax.set_yticklabels(m["labels"], fontsize=8)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7)
    ax.set_title("Agent intent confusion (rows=gold)")
    fig.tight_layout(); fig.savefig(res / "confusion_agent.png", dpi=120); plt.close(fig)

    if "deferral_curve" in out:
        dc = out["deferral_curve"]
        cov = [p["coverage"] for p in dc]
        q = [p["mean_quality_auto"] or np.nan for p in dc]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(cov, q, "o-")
        ax.set_xlabel("coverage (auto-handle rate)"); ax.set_ylabel("mean judge quality on auto subset")
        ax.set_title("Deferral curve — quality vs how much you automate")
        fig.tight_layout(); fig.savefig(res / "deferral_curve.png", dpi=120); plt.close(fig)


def _failure_pool(agent_df, res: Path):
    if agent_df is None or "judge_a_overall" not in agent_df:
        return
    df = agent_df.copy()
    df["worst"] = df["judge_a_overall"] + df.get("judge_b_overall", df["judge_a_overall"])
    df = df.sort_values("worst")
    cols = [c for c in ["customer_text", "intent", "gold_intent", "action", "gold_action",
                        "reason", "draft_reply", "judge_a_overall", "judge_b_overall", "judge_a_full"]
            if c in df.columns]
    df[cols].head(40).to_csv(res / "failure_pool.csv", index=False)
