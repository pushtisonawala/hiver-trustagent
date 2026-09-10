# TrustAgent

A support agent for `@SpotifyCares`, for the Hiver SDE-intern take-home. It
takes one incoming support tweet, picks an intent, drafts a reply from retrieved
past cases, and decides whether to auto-handle it or send it to a human with a
reason attached.

The agent is small. Most of the work is the evaluation around it. Full write-up
is [REPORT.md](REPORT.md) (or [REPORT_GENERATED.md](REPORT_GENERATED.md) with the
numbers filled in).

## Results

<!-- INCLUDE:results/RESULTS.md -->

`docs/demo_transcript.md` has five example messages run through the pipeline.
`make demo` is the interactive version.

## Running it

The LLM responses are committed as `llm-cache.tgz`, so this reproduces the exact
numbers with no API key:

```bash
make setup      # uv + a Python 3.12 venv + deps, and unpacks the response cache
make data       # downloads twcs.csv (no-login HF mirror, falls back to Kaggle)
make all        # build + eval + failure analysis + report, replaying the cache
open REPORT_GENERATED.md results/RESULTS.md results/*.png
```

To hit the live (free) APIs instead:

```bash
cp .env.example .env         # GROQ_API_KEY, free, no card, console.groq.com/keys
rm -rf .cache/llm
make build
make provisional-golden      # a throwaway golden set so the pipeline runs
make all
```

Groq's free tier is rate-limited, so the first live run takes 20-30 minutes;
after that the cache makes it fast. `make smoke` runs everything offline on a
tiny synthetic sample.

The golden set is hand-labelled: `build_golden_set.py --review` in two passes,
notes in `golden/SAMPLING_NOTE.md`. Judge-vs-human agreement:
`judge_validation.py --rate` then `make judge-validation`.

## Layout

```
src/trustagent/
  data.py        twcs.csv -> (message, brand reply) pairs, split by time
  resolution.py  heuristic: did the brand's reply look like it worked?
  intents.py     taxonomy + 3 classifiers (majority / tfidf-logreg / LLM few-shot)
  retrieval.py   TF-IDF nearest-neighbour over past resolved cases
  drafting.py    grounded reply generation
  policy.py      auto-handle vs escalate: a rule stack with a reason string
  agent.py       classify -> retrieve -> draft -> decide
  baselines.py   two trivial baselines + one distilled non-generative one
  judge.py       LLM-as-judge rubric, two judges
  metrics.py     intent / escalation (cost-weighted) / quality / ECE / bootstrap / deferral
  pipeline.py    build() and evaluate()
scripts/         data download, golden-set builder, judge validation, failure analysis, report render
golden/          the hand-labelled set + sampling note + labelling guide
results/         committed so the numbers are visible without running anything
```

## Notes

The train/test split is by time so the retrieval index never sees the future.
Escalation is scored with an asymmetric cost, a bad auto-reply counting 5x a
needless escalation. Reply quality is always reported next to the coverage it was
measured at. Both judges were checked against my own ratings first; the results
are in the report.

## Provenance

I used an AI coding assistant (Claude) throughout, which the brief allows.
Dataset: Kaggle `thoughtvector/customer-support-on-twitter`, from the no-login
Hugging Face mirror `SunidhiSriram/twcs`. Models: Groq free tier, `allam-2-7b`
(agent and Judge A) and `openai/gpt-oss-120b` (Judge B), through the OpenAI
client. Libraries: scikit-learn, pandas, matplotlib, openai. Design choices are
in [DECISION_LOG.md](DECISION_LOG.md).
