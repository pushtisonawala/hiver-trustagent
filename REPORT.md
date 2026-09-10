# TrustAgent — Report

**Brand:** `@SpotifyCares` · **Dataset:** Kaggle `customer-support-on-twitter`
· **Golden set:** {{ n_golden }} held-out test-period messages (see §3.1 for how they were labelled)
· Numbers below are spliced from `results/metrics.json` by `scripts/render_report.py`.

---

## 1. Problem framing

### What "good" means for @SpotifyCares

Spotify's public support is high-volume, low-severity, and *mostly
self-contained*: playback bugs, offline sync, Connect/device pairing, "where did
this song go". A meaningful slice can be resolved with a known instruction the
brand has already given dozens of times. That is the opportunity.

But a hard core is **account- and money-shaped**: double charges, "I cancelled
and you billed me", compromised accounts, student-pricing reverification. Here
the *right* public reply is almost always "we'll take this to DM" — the
resolution happens off the transcript, needs identity verification, and a wrong
confident answer is actively harmful (false refund promises, privacy).

So "good" for this brand is not "answer everything". It is:

1. **Route correctly.** Never auto-handle an account/billing/safety message.
   A missed escalation is far worse than a needless one.
2. **When it does answer, be grounded.** Every step in the reply should trace to
   something Spotify actually told a customer before, in a thread that looked
   resolved.
3. **Sound like @SpotifyCares.** Short, warm, lowercase-friendly, ≤1 emoji.
4. **Be honest about coverage.** A system that safely handles 25% of tweets well
   beats one that "handles" 90% and ships nonsense on a fifth of them.

The headline metric I optimise for is therefore **safe-automation rate** =
fraction of all traffic that is *both* auto-handled *and* judged acceptable
(overall ≥ 4) — reported next to its dangerous twin, **unsafe-auto rate**
(auto-handled but not acceptable).

### What I deliberately did **not** build

| Not built | Why |
|---|---|
| Multi-turn dialogue agent | The dataset's public threads mostly truncate at "DM us". Single inbound message → decision is the realistic, evaluable unit. Multi-turn is [§7](#7-what-id-do-with-one-more-week). |
| Fine-tuned classifier / fine-tuned generator | 200 gold labels is too few to fine-tune honestly; few-shot + a distilled TF-IDF baseline shows the ceiling and floor. |
| A real "resolved?" label | No ground truth exists. I use a weak heuristic (`resolution.py`) and treat it as a known error source, not truth. |
| Retrieval over the full history | Temporal leakage. Index is train-period only. |
| Actually sending replies / a UI | Out of scope; `trustagent demo` is the inspection surface. |
| Tuning to beat the LLM judge | The judge is a measuring instrument; optimising against it corrupts it. |

---

## 2. System (one paragraph)

`classify → retrieve → draft → decide`. Intent = 9-label few-shot classifier
(taxonomy bootstrapped by KMeans over embeddings, then hand-curated — see
`artifacts/taxonomy_bootstrap.json` and DECISION_LOG #4). Retrieval = nearest
neighbours over *train-period* `(customer message → brand reply)` pairs, each
weighted by a resolution-likelihood score. Draft = an open model
(`{{ judge_cross_vendor.agent_drafts_by_ }}`, Groq free tier) constrained to the
retrieved evidence, emitting `{reply, used_evidence, grounded, missing_info}`.
Decide =
a readable rule stack (`policy.py`): hard-escalate keywords, low intent
confidence, weak precedent, high-risk intent without strong precedent,
ungrounded draft, account-specific info required, churn sentiment — any hit ⇒
`escalate`, with the triggered rules as the human-readable reason.

---

## 3. Evaluation design

### 3.1 Golden set ({{ n_golden }} examples)

- **Source:** test split only — the newest 20% of `@SpotifyCares` threads by
  time, unseen by the retrieval index and the classifier's weak labels.
- **Sampling:** stratified by (suggested intent × message-length bucket ×
  early/late half of the test window), proportional with a floor of 4 per intent
  so rare high-risk intents survive the draw.
- **Labelling:** every row labelled by hand in `build_golden_set.py --review`
  with the real historical reply and top precedent visible; LLM suggestions
  shown but overridable (and overridden — see `agree_rate` in the note).
  Fields: `gold_intent`, `gold_action`, free-text `gold_notes`.
- **Second rater:** {{ judge_validation.n }} rows (`adjudication_sample.csv`)
  relabelled independently; agreement in `results/metrics.json`.
- Full protocol: [`golden/SAMPLING_NOTE.md`](golden/SAMPLING_NOTE.md),
  [`golden/labelling_guide.md`](golden/labelling_guide.md).

### 3.2 Metrics

| Axis | Metric | Why this one |
|---|---|---|
| Intent | accuracy, **macro-F1**, per-class F1, confusion, **ECE** | macro-F1 because high-risk intents are rare; ECE because the policy trusts the confidence number |
| Escalation | precision/recall on "should escalate", **cost/msg** (5× / 1×) | F1 treats both errors equally; operations do not |
| Reply quality | LLM-judge rubric ×5 (groundedness, correctness, completeness, tone, safety) + holistic `overall` | one number hides *why* a reply is bad |
| Coverage vs quality | **deferral curve**: mean quality on the auto-handled subset as the confidence threshold sweeps | the headline quality number is meaningless without the coverage it was measured at |
| Uncertainty | 2000× **bootstrap 95% CI** on headline quality | n≈200 → wide intervals; say so |

### 3.3 LLM-as-judge + human agreement

Rubric in `judge.py`. **Judge A = {{ judge_cross_vendor.judge_a_model }}** (the
same model as the drafter — a deliberate self-preference test), **Judge B =
{{ judge_cross_vendor.judge_b_model }}** (a different model lineage).

Validation (`scripts/judge_validation.py`, n = {{ judge_validation.n }} replies
rated blind by a human):

| | Judge A | Judge B |
|---|---|---|
| Spearman vs human | {{ judge_validation.judge_a.spearman | .2f }} | {{ judge_validation.judge_b.spearman | .2f }} |
| MAE vs human (1–5) | {{ judge_validation.judge_a.mae | .2f }} | {{ judge_validation.judge_b.mae | .2f }} |
| within ±1 of human | {{ judge_validation.judge_a.within_1 | .0% }} | {{ judge_validation.judge_b.within_1 | .0% }} |
| Cohen's κ on "acceptable" | {{ judge_validation.judge_a.cohen_kappa_accept | .2f }} | {{ judge_validation.judge_b.cohen_kappa_accept | .2f }} |
| judge mean − human mean | {{ judge_validation.judge_a.judge_mean_minus_human_mean | +.2f }} | {{ judge_validation.judge_b.judge_mean_minus_human_mean | +.2f }} |

Judge-vs-judge: Spearman {{ judge_validation.judge_a_vs_judge_b.spearman | .2f }},
MAE {{ judge_validation.judge_a_vs_judge_b.mae | .2f }}.

**Rule I set myself:** if κ on "acceptable" is below ~0.4 for both judges, the
headline quality number is not allowed to stand without the caveat in §5.

---

## 4. Results vs baselines

Baselines: **trivial-1** majority-intent + canned reply + escalate-all;
**trivial-2** same but auto-handle-all; **simple** TF-IDF+LogReg intent
(distilled from LLM weak labels) + verbatim top-1 past reply + rule-based gate,
*no generative model*.

<!-- INCLUDE:results/RESULTS.md -->

**How to read this table** (direction depends on your run — the analysis does not):

- **Intent:** the gap between `simple` and `agent` macro-F1 is the value the LLM
  adds over cheap distillation. If it is small, the taxonomy is mostly
  lexically separable and the LLM classifier is not worth its latency.
- **Escalation:** `trivial_always_auto` has auto-handle rate 1.0 and the worst
  cost/msg — that is the number a naïve "automation %" headline would celebrate.
  `trivial_always_escalate` has cost driven purely by needless escalations and
  is the safety ceiling. The agent should sit between them on cost, closer to
  escalate-all.
- **Quality:** compare `mean_overall_all` (every draft) with
  `mean_overall_auto_only` (only what got shipped). The second should be higher —
  if it is not, the gate is not selecting the replies the drafter is good at.
- **`unsafe_auto_rate` is the real headline.** It is the fraction of *all*
  traffic where the agent shipped a reply a human would reject. Everything else
  is secondary to keeping this near zero.
- CI on agent quality: {{ models.agent.quality_judge_a.mean_overall_ci95 }}.
- Agent intent ECE: {{ models.agent.intent.ece | .3f }} (lower = confidence is
  trustworthy; the policy leans on it).

See `results/deferral_curve.png` for quality-vs-coverage and
`results/confusion_agent.png` for where intent errors concentrate.

---

## 5. What is misleading about my headline number

Take the headline as *"the agent's mean judged reply quality is
{{ models.agent.quality_judge_a.mean_overall_all | .2f }}/5 (Judge A) and it
safely automates {{ models.agent.quality_judge_a.safe_automation_rate | .0% }} of
traffic."* Here is why you should not believe it as stated.

0. **The judges do not agree with a human — so the quality number is barely
   measuring quality.** I validated both against {{ judge_validation.n }} of my
   own blind human ratings (§3.3). Spearman with the human:
   **{{ judge_validation.judge_a.spearman | .2f }}** (Judge A) and
   **{{ judge_validation.judge_b.spearman | .2f }}** (Judge B); Cohen's κ on
   "acceptable" is {{ judge_validation.judge_a.cohen_kappa_accept | .2f }} and
   {{ judge_validation.judge_b.cohen_kappa_accept | .2f }} — essentially zero.
   Judge A runs {{ judge_validation.judge_a.judge_mean_minus_human_mean | +.2f }}
   above my ratings, Judge B {{ judge_validation.judge_b.judge_mean_minus_human_mean | +.2f }}
   below. **The self-imposed rule (κ < 0.4 ⇒ caveat) fires hard: the reply-quality
   headline is model-specific bias plus noise, not a trustworthy score.** With
   these free models I cannot report a defensible automated quality number at all
   — the honest output is the *intent* and *escalation* metrics plus this
   admission. This is the single most misleading thing in the report.

1. **The quality number is conditioned on the agent's own routing (selection
   bias).** `mean_overall_auto_only` is measured only on messages the agent
   *chose* to answer — the easy ones. It is not the quality you would get on a
   random tweet. The honest figure is the deferral curve, not a point.

2. **Judge A is the same model as the drafter.** The agent drafts with
   `{{ judge_cross_vendor.agent_drafts_by_ }}` and Judge A is
   `{{ judge_cross_vendor.judge_a_model }}` — literally the same weights grading
   their own output, and it scores the agent
   {{ judge_cross_vendor.A_minus_B_on_agent | +.2f }} above Judge B
   (`{{ judge_cross_vendor.judge_b_model }}`, a different lineage). The gap is
   actually *larger* on the simple baseline
   ({{ judge_cross_vendor.A_minus_B_on_simple | +.2f }}), so this looks less like
   targeted self-flattery and more like Judge A being uniformly generous — which
   point 0 confirms.

3. **"Resolved" is a heuristic, and it is optimistic.** Retrieval grounding
   prefers past replies where the customer said "thanks" or didn't reply again.
   Silent churn — customer gives up and leaves — looks *identical* to success.
   So the evidence pool is biased toward replies that pacified people, not
   necessarily replies that fixed things.

4. **The golden labels were made by the person who designed the taxonomy, in
   two passes.** `gold_intent`, the 9-label taxonomy, and the pass-2
   `other`→specific reclassification (`golden/SAMPLING_NOTE.md`) all came from
   one head. Intent accuracy partly measures "does the classifier think like its
   author", not "is the taxonomy right". A 40-row adjudication sample is set
   aside for a second rater (`golden/adjudication_sample.csv`) but that
   inter-annotator κ has not been collected yet — it is the missing guard.

5. **No distribution shift is tested.** Train and test are both Oct–Dec 2017,
   same product, same support team. Real deployment faces new features, new bug
   waves, policy changes. The temporal split removes *leakage*; it does not
   simulate *drift*. Expect every number to decay in production.

6. **Macro-F1 hides that the worst class is the one that matters.** Look at
   per-class F1 in `results/metrics.json` for `account_access` /
   `billing_subscription` / `cancel_or_refund` — they are rare, so a mediocre
   score there barely moves macro-F1 but directly drives `unsafe_auto_rate`.

7. **n ≈ {{ n_golden }} → wide intervals.** The bootstrap CI on agent quality is
   {{ models.agent.quality_judge_a.mean_overall_ci95 }}. Differences between the
   agent and the simple baseline that are smaller than that interval are noise.

8. **Single-message evaluation flatters the agent.** Real threads have context
   ("still not working after I did that"). Scoring the first message only means
   the agent is never penalised for the multi-turn derailments it would have.

9. **Escalations are graded as free.** Cost/msg charges a needless escalation
   1×, but in reality a 90% escalation rate means you have not reduced human
   load at all — the project's actual goal. The metric rewards a coward.

10. **Judge and gold both from English, mostly-US tweets.** Non-English and
    code-switched messages (present in the raw data) are under-sampled in the
    golden set and the judge is weakest there.

**What would make me trust it:** a judge that actually passes human validation
(κ > 0.6) — likely a frontier model, or an ensemble with a rationale audit —
quoted *with* its CI, *at a fixed coverage* from the deferral curve, with
`unsafe_auto_rate` on the high-risk intents reported separately and required to
be ~0, plus a fresh 100-message sample from a *later* time slice. Until then the
intent and escalation numbers carry the report; the quality number is a
directional signal at best.

---

## 6. Failure analysis (top 5)

<!-- INCLUDE:results/failure_analysis.md -->

---

## 7. What I'd do with one more week

1. **Multi-turn.** Feed the last 3 turns; re-label ~60 golden rows as
   conversations; measure derailment rate.
2. **Per-intent thresholds** tuned against the cost matrix by grid search on a
   dev split, instead of one global `min_retrieval_support`.
3. **Groundedness verifier.** A second pass that deletes any reply sentence an
   NLI check can't entail from the evidence — turn "grounded" from a
   self-reported bool into a measured rate.
4. **Better "resolved" signal.** Train a small classifier on ~300 hand-labelled
   (thread → resolved?) examples to replace the regex heuristic; propagate its
   uncertainty into retrieval weighting.
5. **Judge ensemble + rationale audit.** 3 judges, median score, and spot-check
   50 rationales for the failure modes judges are known to have (length bias,
   politeness bias).
6. **Distribution-shift harness.** Slice the test window into months; report the
   metric trajectory, not a single value.
7. **Human-in-the-loop cost model.** Replace the 5×/1× guess with real handle
   times from Hiver-style data if available.

---

## Appendix

- Non-obvious decisions: [DECISION_LOG.md](DECISION_LOG.md)
- Reproduce: [README.md](README.md) · environment: [SETUP.md](SETUP.md)
- Raw numbers: [`results/metrics.json`](results/metrics.json)
