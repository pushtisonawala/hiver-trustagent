# TrustAgent

A small AI support agent for one Twitter support brand (`@SpotifyCares`), built
for the Hiver SDE-intern take-home. The agent is deliberately simple. The work is
in the evaluation and in being honest about what the numbers don't say.

Given one incoming support tweet it produces:

1. an **intent** (one of 9 labels I derived from the data),
2. a **draft reply** built only from retrieved past cases where Spotify seemed to
   resolve a similar issue, and
3. an **auto-handle / escalate decision** with a plain-English reason.

The write-up is in [REPORT.md](REPORT.md). The part that matters most is
§5, "what's misleading about my headline number".

## Results

<!-- INCLUDE:results/RESULTS.md -->

There's a live-ish trace of 5 example messages in
[docs/demo_transcript.md](docs/demo_transcript.md) (`make demo-run` regenerates
it; `make demo` is interactive).

## Reproducing this

The LLM responses are committed as `llm-cache.tgz`, so you can reproduce the
exact numbers with no API key and no cost:

```bash
make setup      # installs uv, a Python 3.12 venv, deps; unpacks the response cache
make data       # downloads twcs.csv (tries a no-login HF mirror, falls back to Kaggle)
make all        # build + eval + failure analysis + report, replaying the cache (~6 min)
open REPORT_GENERATED.md results/RESULTS.md results/*.png
```

To run it against the live (free) APIs instead:

```bash
cp .env.example .env         # add GROQ_API_KEY (free, no card, console.groq.com/keys)
rm -rf .cache/llm            # llm-cache.tgz stays as a backup
make build
make provisional-golden      # writes a throwaway golden set so the pipeline runs
make all
```

Free-tier rate limits make the first live run slow (~20-30 min); after that the
cache makes it instant. `make smoke` runs the whole thing offline on a tiny
synthetic sample if you just want to check the wiring.

The golden set is hand-labelled (`build_golden_set.py --review`, two passes,
notes in `golden/SAMPLING_NOTE.md`). Judge-vs-human agreement is
`judge_validation.py --rate` (rate ~25 replies) then `make judge-validation`.

## Layout

```
src/trustagent/
  data.py        twcs.csv -> (customer message -> brand reply) pairs, split by time
  resolution.py  heuristic: did the brand's reply look like it worked?
  intents.py     taxonomy + 3 classifiers (majority / tfidf-logreg / LLM few-shot)
  retrieval.py   TF-IDF nearest-neighbour over past resolved cases
  drafting.py    grounded reply generation
  policy.py      auto-handle vs escalate: a readable rule stack + reason
  agent.py       classify -> retrieve -> draft -> decide
  baselines.py   two trivial baselines + one distilled non-generative one
  judge.py       LLM-as-judge rubric, two judges
  metrics.py     intent / escalation (cost-weighted) / quality / ECE / bootstrap / deferral
  pipeline.py    build() and evaluate()
scripts/         data download, golden-set builder, judge validation, failure analysis, report render
golden/          the hand-labelled set + sampling note + labelling guide
results/         committed so the numbers are visible without running anything
```

A few things I'd point out: the train/test split is by time so the retrieval
index never sees the future; escalation is scored with an asymmetric cost (a bad
auto-reply is treated as 5x worse than a needless escalation); the reply-quality
headline is always reported next to the coverage it was measured at; and both
judges were checked against my own ratings before any judge number was trusted
(spoiler: they failed, see the report).

## Provenance

I used an AI coding assistant (Claude) throughout, which the brief allows.
Dataset: Kaggle `thoughtvector/customer-support-on-twitter`, pulled from the
no-login Hugging Face mirror `SunidhiSriram/twcs`. Models: Groq free tier,
`allam-2-7b` (agent + Judge A) and `openai/gpt-oss-120b` (Judge B), via the
OpenAI client. Libraries: scikit-learn, pandas, matplotlib, openai. Design
choices and what I borrowed are in [DECISION_LOG.md](DECISION_LOG.md).
