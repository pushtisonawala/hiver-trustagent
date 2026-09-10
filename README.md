# TrustAgent — an auditable AI support agent for **@SpotifyCares**

> Built for the Hiver SDE-Intern take-home. The agent is small on purpose. The
> evaluation around it is the actual submission.

TrustAgent takes one incoming Spotify support tweet and produces:

1. **Intent** — one of 9 labels bootstrapped from the data (`src/trustagent/intents.py`).
2. **A grounded draft reply** — generated only from *retrieved past cases where
   Spotify actually resolved a similar issue* (`src/trustagent/retrieval.py`,
   `drafting.py`).
3. **A decision + reason** — `auto_handle` or `escalate`, from a transparent
   rule stack tied to real operational risk (`src/trustagent/policy.py`).

Then it tries to talk me out of trusting it. See
[**REPORT.md → "What is misleading about my headline number"**](REPORT.md#5-what-is-misleading-about-my-headline-number).

---

## TL;DR results

<!-- INCLUDE:results/RESULTS.md -->

Full write-up: [REPORT.md](REPORT.md) · decisions: [DECISION_LOG.md](DECISION_LOG.md)
· failure modes: [results/failure_analysis.md](results/failure_analysis.md)

---

## Reproduce the headline numbers (target: < 15 min)

New machine? Follow [SETUP.md](SETUP.md) first (installs Python etc.).

**Everything below is free.** LLM calls go to Groq's free tier
(`qwen/qwen3.8-27b` for the agent + Judge A, `allam-2-7b` for Judge B) — no
credit card, one key.
Retrieval is local TF-IDF (no embedding API). The response cache is committed,
so a reviewer reproduces the exact numbers with **no key at all**:

```bash
make setup                       # venv + deps                            (~3 min)
make data                        # Kaggle download -> data/raw/twcs.csv    (~3 min)
make all                         # build (fresh) + eval/failures/report replayed from
                                 # the committed .cache/llm/ -> results/   (~6 min, $0, no keys)
open REPORT_GENERATED.md results/RESULTS.md results/*.png
```

To regenerate against the live APIs (also free):

```bash
cp .env.example .env && $EDITOR .env    # just GROQ_API_KEY (1 min, free, no card)
rm -rf .cache/llm                       # drop the cache
make build
make provisional-golden && make all     # first run ~15-20 min (free-tier rate limits), then cached
python scripts/build_golden_set.py --review   # you hand-label ~200 rows (~45 min)
make all
```

> `make all` needs `golden/golden_set.csv`. `make provisional-golden` writes a
> throwaway one so the pipeline runs; the **committed** `golden_set.csv` is the
> hand-labelled set and the report's numbers are only valid against that.

No Kaggle account? This still runs end-to-end on a tiny committed synthetic
sample in offline heuristic mode — proves the wiring, numbers are not meaningful:

```bash
make setup && make smoke
```

### What `make all` does

| step | command | output |
|---|---|---|
| ingest + threads + temporal split | `trustagent build` | `artifacts/{train,test}.parquet` |
| retrieval index (local TF-IDF) | ″ | `artifacts/retriever/` |
| weak-label ~220 msgs for the distilled baseline | ″ | `artifacts/weak_labels.json` |
| baselines + agent + **cross-vendor LLM judge** | `trustagent eval` | `results/metrics.json`, `predictions_*.csv` |
| calibration, bootstrap CIs, deferral curve | ″ | `results/*.png` |
| failure-mode clustering | `failure_analysis.py` | `results/failure_analysis.md` |
| fill the report with real numbers | `render_report.py` | `REPORT_GENERATED.md` |

### The golden set is built by hand (assisted)

```bash
make golden                              # draws a stratified sample -> golden/golden_set.todo.csv
python scripts/build_golden_set.py --review   # you label every row in the terminal
```

`golden/golden_set.csv` (≈200 rows) and `golden/SAMPLING_NOTE.md` are committed.
Judge-vs-human agreement:

```bash
python scripts/judge_validation.py --make-sheet   # 60 replies to rate blind
# ...you fill golden/human_reply_ratings.csv...
make judge-validation                             # Spearman / kappa / judge-vs-judge -> metrics.json
```

---

## Repo map

```
src/trustagent/
  data.py         twcs.csv -> (customer msg -> brand reply) pairs, TEMPORAL split
  resolution.py   weak-supervision: did the brand's reply actually work?
  intents.py      taxonomy bootstrap + 3 classifiers (majority / tfidf-logreg / LLM)
  retrieval.py    embed + nearest-neighbour over resolved past cases
  drafting.py     grounded reply generation (cites evidence thread ids)
  policy.py       auto_handle vs escalate  — readable rule stack + reason string
  agent.py        classify -> retrieve -> draft -> decide
  baselines.py    2 trivial + 1 simple (no generative model)
  judge.py        LLM-as-judge rubric, Judge A = qwen3.8-27b, Judge B = allam-2-7b
  metrics.py      intent / escalation(cost-weighted) / quality / ECE / bootstrap / deferral
  pipeline.py     build() and evaluate() orchestration
scripts/          data download, golden-set builder, judge validation, failure analysis, report render
golden/           the hand-labelled set + sampling note + labelling guide
results/          committed so reviewers see numbers without running anything
```

## Standout bits (why this isn't a generic RAG demo)

- **Temporal split, not random** — the retrieval index and weak labels only see
  tweets older than the test period. A random split silently inflates every number.
- **Cost-weighted escalation** — shipping a bad auto-reply is treated as 5× worse
  than a needless escalation; thresholds are tuned against that, not F1.
- **Selective-prediction deferral curve** — headline reply-quality is only
  meaningful *alongside* the auto-handle rate it was measured at.
- **Cross-vendor judge + self-preference check** — the drafter is `qwen3.8-27b`,
  so Judge A (the *same* `qwen3.8-27b`) *and* Judge B (`allam-2-7b`, a
  different lineage) both score, and we measure whether the model flatters itself.
- **Judge validated against humans** — Spearman + Cohen's κ before the judge
  number is allowed to carry any weight.
- **Everything cached + seeded, cache committed** — reviewers reproduce the exact
  numbers with $0 and no keys; `rm -rf .cache/llm` to re-run live.

## Provenance

AI coding assistant (Claude) used throughout, per the assignment rules. Dataset:
Kaggle `thoughtvector/customer-support-on-twitter` (pulled from the no-login HF
mirror `SunidhiSriram/twcs`). LLMs: Groq `qwen/qwen3.8-27b` + `allam-2-7b`,
free tier, via the OpenAI client. Libraries:
scikit-learn, pandas, matplotlib, openai. Borrowed ideas cited in `DECISION_LOG.md`.
