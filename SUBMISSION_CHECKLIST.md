# Submission checklist — maps the brief to this repo

| Deliverable (from the brief) | Where | Status |
|---|---|---|
| **1. Repo with runnable pipeline**, README reproduces headline in < 15 min | [README.md](README.md), `Makefile`, `src/trustagent/` | ✅ code · ⏳ run `make all` on your machine to populate `results/` |
| **2. Golden set 150–250 hand-labelled** + sampling/labelling note | `golden/golden_set.csv`, [`golden/SAMPLING_NOTE.md`](golden/SAMPLING_NOTE.md), [`golden/labelling_guide.md`](golden/labelling_guide.md) | ⏳ run `python scripts/build_golden_set.py --review` (~45 min) |
| **3. Eval harness**: automated metrics + LLM-judge rubric + human-agreement evidence | `src/trustagent/metrics.py`, `judge.py`, `scripts/judge_validation.py` | ✅ code · ⏳ `make eval` + `make judge-validation` |
| **4. Report** (≤6 pages) | [REPORT.md](REPORT.md) → `REPORT_GENERATED.md` after `make report` | ✅ written · ⏳ numbers auto-fill on run |
| &nbsp;&nbsp;· problem framing + what you chose not to build | REPORT §1 | ✅ |
| &nbsp;&nbsp;· results vs ≥2 baselines (trivial + simple) | REPORT §4, `src/trustagent/baselines.py` | ✅ code · ⏳ numbers |
| &nbsp;&nbsp;· failure analysis: top 5 with real examples + hypotheses | REPORT §6, `results/failure_analysis.md`, `scripts/failure_analysis.py` | ✅ code · ⏳ + your own notes |
| &nbsp;&nbsp;· **"what is misleading about my headline number"** | REPORT §5 | ✅ written (10 points) |
| &nbsp;&nbsp;· what you'd do next with one more week | REPORT §7 | ✅ |
| **5. Decision log** (10–15 non-obvious decisions) | [DECISION_LOG.md](DECISION_LOG.md) | ✅ 16 entries |

## Before you submit

1. `make setup && make data`
2. `cp .env.example .env` → add **GROQ_API_KEY** (free, no card, ~1 min)
3. `make build`
4. `python scripts/build_golden_set.py --review`  → label ~200 rows → commit `golden/golden_set.csv`
5. `make all`  → first run ~15-20 min (free-tier limits). Then `make bundle-cache`
   to repack `llm-cache.tgz` — that's what lets reviewers reproduce your exact
   numbers for $0 with no keys.
6. `python scripts/judge_validation.py --make-sheet` → rate 60 replies → save as
   `golden/human_reply_ratings.csv` → `make judge-validation`
7. `make report` again (picks up judge-validation numbers) → `make bundle-cache` →
   **commit `results/`, `llm-cache.tgz`, `REPORT_GENERATED.md`**
8. Fill the "Author note" lines in `results/failure_analysis.md` — reviewers
   explicitly want *your* hypotheses, not the tool's seed text.
9. `git add -A && git commit && git push`; paste the repo link + the contents of
   `REPORT_GENERATED.md` into the Notion form.

> Sanity check before pushing, from a clean clone with **no `.env`**:
> `make setup && make data && make all` should reproduce `results/` from the
> committed cache in well under 15 min. If it does, a reviewer's will too.
> (`artifacts/` is gitignored — too large — so `build` re-runs; it's fast, free
> and deterministic.)

## What is NOT done for you (on purpose)

- The **actual labelling** of the golden set and the **60 human reply ratings** —
  these must be yours; the tools only assist.
- The **"Author note"** hypotheses in the failure analysis.
- Any claim in the report that depends on a number is `??` until you run it.
