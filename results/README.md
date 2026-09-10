# results/

Committed so reviewers see the headline numbers without running anything.
Regenerate with `make all`.

| file | what |
|---|---|
| `metrics.json` | every number, machine-readable — the source of truth |
| `RESULTS.md` | headline table + cross-vendor judge check (spliced into README + REPORT) |
| `predictions_<model>.csv` | per-golden-row output for each of the 4 systems, incl. judge scores |
| `failure_pool.csv` | worst agent cases, input to `failure_analysis.py` |
| `failure_analysis.md` | top failure modes with real examples + hypotheses |
| `confusion_agent.png` | intent confusion matrix (rows = gold) |
| `deferral_curve.png` | reply quality vs auto-handle coverage — read this before the headline |

`metrics.json` layout: `models.<name>.{intent,escalation,quality_judge_a,quality_judge_b}`,
plus top-level `deferral_curve`, `judge_cross_vendor`, and (after
`make judge-validation`) `judge_validation`.

The committed set is from the **provisional** (auto-labelled) golden set — see the
banner in `REPORT_GENERATED.md`. Replace `golden/golden_set.csv` with the
hand-labelled one (`build_golden_set.py --review`) and rerun for submission numbers.
