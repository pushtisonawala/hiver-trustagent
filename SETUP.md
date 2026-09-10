# Setup (macOS, from a clean machine)

This machine had no dev tools. Here is the whole path from zero to `make all`.
Every command is copy-pasteable. Nothing here needs `sudo` except where noted.

---

## 1. Xcode Command Line Tools (gives you `git`, `clang`, headers)

```bash
xcode-select --install
```

A GUI dialog pops up → **Install** → wait ~5–10 min. Verify:

```bash
xcode-select -p          # should print /Library/Developer/CommandLineTools
git --version            # should print something
```

## 2. Python (handled for you by `uv`)

You don't need to install Python yourself. `make setup` (step 4) installs
[`uv`](https://docs.astral.sh/uv/) — a single self-contained binary, no `sudo` —
and uv downloads a clean Python 3.12 and builds the venv. The system `python3`
stub is only needed for step 1's Xcode check.

If you'd rather install uv up front:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 3. Get the project onto the machine

The project already lives at `~/Desktop/hiver-assignment`. Turn it into a git
repo (the assignment wants a repo link):

```bash
cd ~/Desktop/hiver-assignment
git init && git add -A && git commit -m "TrustAgent: initial commit"
```

To push: create an **empty** repo on GitHub (no README), then:

```bash
git remote add origin https://github.com/<you>/hiver-trustagent.git
git branch -M main && git push -u origin main
```

## 4. Python environment + dependencies

```bash
cd ~/Desktop/hiver-assignment
make setup
```

This installs `uv` if needed, then `uv venv --python 3.12 .venv` and
`uv pip install -e . pytest`. If `uv` isn't on your PATH afterwards, restart the
shell or `source $HOME/.local/bin/env`.

Sanity check:

```bash
make test        # unit tests, no network
```

## 5. API key — ONE, free, no credit card

You only need this to regenerate results against the live API. The repo ships a
committed response cache, so `make all` reproduces the exact numbers with **no
key at all**. To re-run live:

```bash
cp .env.example .env
```

Edit `.env`:

- `GROQ_API_KEY` — <https://console.groq.com/keys>. Sign in with Google, click
  "Create API Key". Free, no card. That's the **only** key you need — it powers
  the drafter + classifier + weak-labeller + Judge A (`qwen/qwen3.8-27b`) and
  Judge B (`allam-2-7b`, a different lineage for the cross-check).

Model names drift on free tiers. If a call 404s with "model not available",
check <https://console.groq.com/docs/models> and update `config.yaml → models:`.

Cost of a full live `make all`: **$0**. Groq's free tier caps each model at ~1000
requests/min but 8000 tokens/min, so the first live run is ~20–30 min; every run
after that is instant (cache). Without a key, `make all` uses the committed
cache; `make smoke` uses offline heuristics. Both are free.

## 6. Dataset

```bash
make data
```

`make data` first tries a **no-login Hugging Face mirror**
(`SunidhiSriram/twcs`, same file, ~470 MB) and only falls back to Kaggle if that
fails. For the Kaggle path you need a free token:

1. <https://www.kaggle.com/settings> → **Create New Token** → `kaggle.json`
2. `mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json`

Either way you end up with `data/raw/twcs.csv` (~2.8M rows).

## 7. Run it

```bash
make all
```

Targets `< 15 min` on a laptop. Then open:

- `REPORT_GENERATED.md` — the report with real numbers filled in
- `results/RESULTS.md`, `results/metrics.json`
- `results/confusion_agent.png`, `results/deferral_curve.png`
- `results/failure_analysis.md`

## 8. Build the golden set (do this once, before `make all` for real numbers)

```bash
make golden
python scripts/build_golden_set.py --review     # ~40–60 min of labelling
```

Commit `golden/golden_set.csv`. Re-run `make all`.

---

## Troubleshooting

| symptom | fix |
|---|---|
| `xcode-select: note: No developer tools were found` | step 1 not finished — run it, wait for the GUI installer |
| `uv: command not found` after `make setup` | `source $HOME/.local/bin/env` or restart the shell |
| `kaggle: command not found` | `.venv/bin/pip install kaggle` (the Makefile does this automatically) |
| first live run is slow | free-tier rate limits (~30/min). It's working — leave it. Re-runs hit the cache and finish in minutes. |
| `RateLimitError` / 429 spam | expected on free tiers; the client backs off and retries. If it stalls for good, just re-run — finished calls are cached. |
| want the fast path | don't set a key — `make all` unpacks `llm-cache.tgz` and replays it in <5 min for $0 |
