_SpotifyCares, golden n=148, seed 13, retriever=tfidf. Regenerate with `make eval`._

| model | intent acc | intent macroF1 | escal. P | escal. R | auto-handle rate | unsafe-auto rate | safe-automation rate | judgeA mean |
|---|---|---|---|---|---|---|---|---|
| trivial_always_escalate | 0.19 | 0.04 | 0.56 | 1.00 | 0.00 | nan | nan | nan |
| trivial_always_auto | 0.19 | 0.04 | 0.00 | 0.00 | 1.00 | nan | nan | nan |
| simple | 0.36 | 0.32 | 0.57 | 0.98 | 0.03 | 0.000 | 0.03 | 4.63 |
| agent | 0.65 | 0.63 | 0.74 | 0.94 | 0.28 | 0.000 | 0.37 | 4.90 |

On the agent's replies, allam-2-7b averages 4.90 and openai/gpt-oss-120b averages 3.43 (gap +1.47; the gap on the simple baseline is +2.00).
