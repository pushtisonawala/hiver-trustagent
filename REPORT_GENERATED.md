# TrustAgent — Report

**Brand:** `@SpotifyCares` · **Dataset:** Kaggle `customer-support-on-twitter`
· **Golden set:** 148 held-out test-period messages (see §3.1 for how they were labelled)
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
(`allam-2-7b`, Groq free tier) constrained to the
retrieved evidence, emitting `{reply, used_evidence, grounded, missing_info}`.
Decide =
a readable rule stack (`policy.py`): hard-escalate keywords, low intent
confidence, weak precedent, high-risk intent without strong precedent,
ungrounded draft, account-specific info required, churn sentiment — any hit ⇒
`escalate`, with the triggered rules as the human-readable reason.

---

## 3. Evaluation design

### 3.1 Golden set (148 examples)

- **Source:** test split only — the newest 20% of `@SpotifyCares` threads by
  time, unseen by the retrieval index and the classifier's weak labels.
- **Sampling:** stratified by (suggested intent × message-length bucket ×
  early/late half of the test window), proportional with a floor of 4 per intent
  so rare high-risk intents survive the draw.
- **Labelling:** every row labelled by hand in `build_golden_set.py --review`
  with the real historical reply and top precedent visible; LLM suggestions
  shown but overridable (and overridden — see `agree_rate` in the note).
  Fields: `gold_intent`, `gold_action`, free-text `gold_notes`.
- **Second rater:** `??` rows (`adjudication_sample.csv`)
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

Rubric in `judge.py`. **Judge A = allam-2-7b**,
**Judge B = openai/gpt-oss-120b** (cross-vendor on purpose).

Validation (`scripts/judge_validation.py`, n = `??` replies
rated blind by a human):

| | Judge A | Judge B |
|---|---|---|
| Spearman vs human | `??` | `??` |
| MAE vs human (1–5) | `??` | `??` |
| within ±1 of human | `??` | `??` |
| Cohen's κ on "acceptable" | `??` | `??` |
| judge mean − human mean | `??` | `??` |

Judge-vs-judge: Spearman `??`,
MAE `??`.

**Rule I set myself:** if κ on "acceptable" is below ~0.4 for both judges, the
headline quality number is not allowed to stand without the caveat in §5.

---

## 4. Results vs baselines

Baselines: **trivial-1** majority-intent + canned reply + escalate-all;
**trivial-2** same but auto-handle-all; **simple** TF-IDF+LogReg intent
(distilled from LLM weak labels) + verbatim top-1 past reply + rule-based gate,
*no generative model*.

# Results (generated by `make eval`)

Brand: **SpotifyCares**  ·  golden n = 148  ·  seed 13  ·  retriever = tfidf

## Headline table

| model | intent acc | intent macroF1 | escal. P | escal. R | auto-handle rate | unsafe-auto rate | safe-automation rate | judgeA mean |
|---|---|---|---|---|---|---|---|---|
| trivial_always_escalate | 0.19 | 0.04 | 0.56 | 1.00 | 0.00 | nan | nan | nan |
| trivial_always_auto | 0.19 | 0.04 | 0.00 | 0.00 | 1.00 | nan | nan | nan |
| simple | 0.36 | 0.32 | 0.57 | 0.98 | 0.03 | 0.000 | 0.03 | 4.63 |
| agent | 0.65 | 0.63 | 0.75 | 0.94 | 0.30 | 0.000 | 0.37 | 4.83 |

## Cross-vendor judge check

- Judge A (allam-2-7b) on agent: **4.83**
- Judge B (openai/gpt-oss-120b) on agent: **3.50**
- A−B on agent: +1.33  vs  A−B on simple: +1.93


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
- CI on agent quality: 4.83  (95% CI 4.67–4.97).
- Agent intent ECE: 0.287 (lower = confidence is
  trustworthy; the policy leans on it).

See `results/deferral_curve.png` for quality-vs-coverage and
`results/confusion_agent.png` for where intent errors concentrate.

---

## 5. What is misleading about my headline number

Take the headline as *"the agent's mean judged reply quality is
4.83/5 and it safely
automates 37% of
traffic."* Here is why you should not believe it as stated.

1. **The quality number is conditioned on the agent's own routing (selection
   bias).** `mean_overall_auto_only` is measured only on messages the agent
   *chose* to answer — the easy ones. It is not the quality you would get on a
   random tweet. The honest figure is the deferral curve, not a point.

2. **Judge A is the same model as the drafter.** The agent drafts with
   `allam-2-7b` and Judge A is
   `allam-2-7b` — literally the same weights grading
   their own output. Self-preference check: A−B on the agent's drafts =
   +1.33, A−B on the simple
   baseline's drafts = +1.93.
   If the first is meaningfully larger, the headline is inflated by a judge that
   recognises and rewards its own style. **Judge B (Gemini, cross-vendor) is the
   number I actually stand behind.**

3. **"Resolved" is a heuristic, and it is optimistic.** Retrieval grounding
   prefers past replies where the customer said "thanks" or didn't reply again.
   Silent churn — customer gives up and leaves — looks *identical* to success.
   So the evidence pool is biased toward replies that pacified people, not
   necessarily replies that fixed things.

4. **The golden labels were made by the person who designed the taxonomy.**
   `gold_intent` and the 9-label taxonomy came from the same head. Intent
   accuracy partly measures "does the classifier think like its author", not
   "is the taxonomy right". The second-rater κ is the only guard, and it is on
   `??` rows.

5. **No distribution shift is tested.** Train and test are both Oct–Dec 2017,
   same product, same support team. Real deployment faces new features, new bug
   waves, policy changes. The temporal split removes *leakage*; it does not
   simulate *drift*. Expect every number to decay in production.

6. **Macro-F1 hides that the worst class is the one that matters.** Look at
   per-class F1 in `results/metrics.json` for `account_access` /
   `billing_subscription` / `cancel_or_refund` — they are rare, so a mediocre
   score there barely moves macro-F1 but directly drives `unsafe_auto_rate`.

7. **n ≈ 148 → wide intervals.** The bootstrap CI on agent quality is
   4.83  (95% CI 4.67–4.97). Differences between the
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

**What would make me trust it:** Judge B (cross-vendor) as the headline, quoted
*with* its CI, *at a fixed coverage* from the deferral curve, with
`unsafe_auto_rate` on the high-risk intents reported separately and required to
be ~0, plus a fresh 100-message sample from a *later* time slice.

---

## 6. Failure analysis (top 5)

# Failure analysis — agent

Pool = 30 worst-scoring golden cases (by combined Judge A+B overall).

| rank | failure mode | count | share |
|---|---|---|---|
| 1 | `multiturn_blind` | 15 | 50% |
| 2 | `generic_nonanswer` | 8 | 27% |
| 3 | `retrieval_irrelevant` | 3 | 10% |
| 4 | `tone_off` | 3 | 10% |
| 5 | `intent_confusion` | 1 | 3% |

## 1. `multiturn_blind` — Needed earlier conversation context the single-message agent never saw.  (15 cases)

> **customer:** what’s going on I can’t do anything but listen to my playlists
> **agent (playback_or_app_bug, escalate):** Hey Megan, help's here! Just to check, are you trying to sync local files from your desktop to your phone? /DV
> **gold:** intent=other action=escalate  · judgeA=5

> **customer:** when is the Top Songs of 2017 playlist coming?
> **agent (feature_request_or_feedback, escalate):** Hey! We're launching regularly in countries around the world. Sign up here to be first to hear: /DF
> **gold:** intent=content_availability action=auto_handle  · judgeA=5

**Seed hypothesis.** We evaluate on the first message only. Fix: feed the last 3 turns as context.

**Author note.** _<add your sharper hypothesis + what you actually saw here>_


## 2. `generic_nonanswer` — Vague 'we're looking into it' / 'DM us' with no actual help.  (8 cases)

> **customer:** Fam why is “Rain On Me” NOT on Chapter II ??
> **agent (content_availability, auto_handle):** Don't fret! It's being updated regularly & we'll be sure to let the right team know of your observations. Any questions, give us a shout /GN
> **gold:** intent=content_availability action=auto_handle  · judgeA=5

> **customer:** when im listening to music it states that i can click the thing to get 30 minutes of ad free listening but when i go to click the part that has been there in the past, its not there so i cant do anything. it has done this about three times since i started listening today
> **agent (playback_or_app_bug, auto_handle):** That doesn't sound good! Did the app crash at any point during the 30 minutes? That can cause the ad-free time to reset /FR
> **gold:** intent=playback_or_app_bug action=auto_handle  · judgeA=5

**Seed hypothesis.** Model hedges when evidence is thin instead of escalating. Fix: forbid 'DM us' as a *reply* — that path must escalate.

**Author note.** _<add your sharper hypothesis + what you actually saw here>_


## 3. `retrieval_irrelevant` — Top precedent was not actually similar; grounding is spurious.  (3 cases)

> **customer:** How to request a song / album (origional album or artist) to upload in your library which is still not there . Please let me know via direct message or send me link. Waiting reply.
> **agent (feature_request_or_feedback, escalate):** Thanks for getting back to us. We do our best! But sometimes agreements are ongoing, or can't be reached with the content provider /GT
> **gold:** intent=content_availability action=auto_handle  · judgeA=3

> **customer:** So keeps doing this thing where all my downloaded songs disappear and then when I go to listen to music offline I can't. Pretty awesome. any idea why? I've reinstalled, cleared out the offline devices list, redownloaded music numerous times.
> **agent (playback_or_app_bug, escalate):** Hi there! We understand how frustrating this must be. Please DM us your account's email address so we can take a closer look and see if we can resolve this issue for you. Stay tuned for updates!
> **gold:** intent=playback_or_app_bug action=escalate  · judgeA=5

**Seed hypothesis.** TF-IDF/embedding retrieves lexically-close but semantically-different cases. Fix: re-rank top-20 with the LLM; drop hits it flags as off-topic.

**Author note.** _<add your sharper hypothesis + what you actually saw here>_


## 4. `tone_off` — Correct content but off-brand: robotic, preachy, or too long.  (3 cases)

> **customer:** what the fuq with your web player ? #spotifyth it keeps getting crashed all the freaking time !
> **agent (playback_or_app_bug, escalate):** Hey there! We understand how frustrating this must be. Please try clearing your Spotify cache and cookies, then restart the web player. If the issue persists, kindly provide us with the browser you're using and its version. We'll do our best to assist you. /JH
> **gold:** intent=playback_or_app_bug action=escalate  · judgeA=4

> **customer:** still upset that got rid of messaging a while ago lol
> **agent (other, auto_handle):** Hey there! We understand your concerns. Please DM us your Spotify account's email or username, and we'll investigate the issues you mentioned. You can also check our content information here: /PB
> **gold:** intent=feature_request_or_feedback action=auto_handle  · judgeA=5

**Seed hypothesis.** Voice guide not enforced. Fix: add 2 golden style exemplars + a length cap.

**Author note.** _<add your sharper hypothesis + what you actually saw here>_


## 5. `intent_confusion` — Wrong intent, which then routed the whole response wrong.  (1 cases)

> **customer:** I know, but your UI is saying that I’m at 1k+ of 4k+ downloads. It seems like it should say 1k+ of 3,333. ;)
> **agent (other, escalate):** Hi there! We understand your concern about the download progress. Based on the evidence provided, it seems like the UI is still in testing phases. Your account is indeed downloading correctly. If you have any further feedback, please feel free to share it. Grounded: True, Missing Info: None
> **gold:** intent=feature_request_or_feedback action=escalate  · judgeA=5

**Seed hypothesis.** Adjacent labels (billing vs cancel; playback vs device) leak. Fix: merge or add contrastive few-shot examples for the confused pair.

**Author note.** _<add your sharper hypothesis + what you actually saw here>_



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
