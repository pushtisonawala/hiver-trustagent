# TrustAgent

Brand: `@SpotifyCares`. Dataset: Kaggle `customer-support-on-twitter`.
Golden set: {{ n_golden }} held-out messages I labelled myself (§3).

The numbers in this doc are pulled straight from `results/metrics.json` by
`scripts/render_report.py`, so they can't drift from what the code actually
produced.

---

## 1. Problem framing

### What "good" looks like for this brand

I read a few hundred @SpotifyCares threads before picking a target. Most of the
volume is low-severity and self-contained: songs won't play, offline downloads
vanished, Connect can't see the speaker, "where did this album go". Spotify has
answered these hundreds of times with the same handful of fixes, so there's a
real slice a bot could take.

The rest is account- and money-shaped: double charges, "I cancelled and you
billed me again", hacked accounts, student pricing that stopped applying. For
those the correct public reply is almost always "let's move to DM". The actual
resolution happens off the transcript and needs identity checks, and a confident
wrong answer here is genuinely harmful (a false refund promise, leaking that an
email is on an account).

So "good" here isn't "answer everything". It's:

- Route safely. Never auto-answer an account/billing/security message. Missing an
  escalation costs a lot more than a needless one.
- When it does answer, don't make things up. Every step should trace back to
  something Spotify actually told a customer, in a thread that looked resolved.
- Sound like the brand. Short, warm, lowercase is fine, one emoji at most.
- Be upfront about coverage. Safely handling 25% well beats "handling" 90% and
  shipping nonsense on a chunk of it.

The number I care about most is **safe-automation rate**: share of all traffic
that gets auto-handled *and* would pass a human. I always report it next to
**unsafe-auto rate**, the share that got auto-handled but shouldn't have.

### What I didn't build, on purpose

- **Multi-turn dialogue.** The public threads mostly cut off at "DM us", so I
  can't evaluate a conversation honestly on this data. One inbound message ->
  decision is the unit I can actually score. (This came back to bite me, see §6.)
- **Any fine-tuning.** {{ n_golden }} labels isn't enough to fine-tune without
  fooling myself. Few-shot plus a distilled linear baseline shows the ceiling and
  the floor.
- **A real "was this resolved" label.** There isn't one in the data. I use a
  regex heuristic over the customer's follow-up and treat it as a known source of
  error, not truth.
- **Retrieval over the whole history.** That leaks the future into the index. It
  only sees the training period.
- **A UI or actually sending replies.** Out of scope. `make demo` is the way to
  poke at it (see README).
- **Tuning anything to please the LLM judge.** The judge is a ruler. If I bend
  the system to score well on it, I've broken the ruler.

---

## 2. How it works

Four steps: classify, retrieve, draft, decide.

Intent is a 9-label few-shot classifier. I got to the 9 labels by clustering
~2k training messages (KMeans over embeddings), reading the top terms and nearest
examples per cluster, then merging by hand. The raw clusters split "playback"
three ways and lumped every money problem together, so the final taxonomy is a
judgement call (`artifacts/taxonomy_bootstrap.json` has the evidence).

Retrieval is nearest-neighbour over training-period `(customer message -> brand
reply)` pairs, each weighted by how likely that reply looked resolved. Drafting
is a small open model on Groq's free tier
(`{{ judge_cross_vendor.agent_drafts_by_ }}`), told to only use fixes that appear
in the retrieved evidence and to hand back
`{reply, used_evidence, grounded, missing_info}`.

The decide step is a plain rule stack in `policy.py`, not a model. It escalates
on: hard keywords (fraud, chargeback, hacked, GDPR...), low classifier
confidence, no similar precedent, a high-risk intent without strong precedent, a
draft the model couldn't ground, a draft that needs account-specific info, or
churn language. Any hit escalates, and the triggered rules are the reason string
a human sees. I kept this readable on purpose, a support lead should be able to
audit why anything escalated.

---

## 3. Evaluation

### 3.1 Golden set ({{ n_golden }} rows)

Drawn from the **test split only**: the newest slice of @SpotifyCares threads by
time, which the retrieval index and the classifier's weak labels never see.
Stratified by (suggested intent x message length x early/late half of the test
window), with a floor per intent so the rare high-risk ones don't disappear.

I labelled every row myself in `build_golden_set.py --review`, with the real
historical reply and the top retrieved precedent visible. First pass agreed with
the model's suggestion ~95% of the time on intent, which honestly is because the
suggestions were mostly fine. That first pass also dumped 84 of 148 rows into the
catch-all `other` bucket, which is what the model does when it's unsure. `other`
is supposed to mean "spam / joke / nothing actionable", so I did a second pass
(`refine_other.py`) over just those rows and moved 58 to their real intent.
Details and the honest caveats are in `golden/SAMPLING_NOTE.md`.

There's a 40-row adjudication sample set aside for a second rater. I didn't get
that inter-annotator number, so single-annotator bias is real here (§5).

### 3.2 Metrics

| What | Metric | Why |
|---|---|---|
| Intent | accuracy, macro-F1, per-class F1, confusion, ECE | macro-F1 because the risky intents are rare; ECE because the policy trusts the confidence score |
| Escalation | precision/recall on "should escalate", cost/msg at 5x/1x | plain F1 treats both mistakes the same, ops doesn't |
| Reply quality | 5-axis LLM-judge rubric + a holistic score | one number hides *why* a reply is bad |
| Coverage vs quality | deferral curve: quality on the auto-handled subset as the confidence bar moves | a quality number means nothing without the coverage it was measured at |
| Uncertainty | bootstrap 95% CI on the headline | n is small, the intervals are wide, better to show it |

### 3.3 LLM-as-judge, and checking it against a human

Rubric is in `judge.py`. Judge A is `{{ judge_cross_vendor.judge_a_model }}`, the
same model that writes the drafts (a deliberate worst-case for self-preference).
Judge B is `{{ judge_cross_vendor.judge_b_model }}`, a different model family.

I rated {{ judge_validation.n }} replies blind myself and compared:

| | Judge A | Judge B |
|---|---|---|
| Spearman vs me | {{ judge_validation.judge_a.spearman | .2f }} | {{ judge_validation.judge_b.spearman | .2f }} |
| MAE vs me (1-5) | {{ judge_validation.judge_a.mae | .2f }} | {{ judge_validation.judge_b.mae | .2f }} |
| within 1 point of me | {{ judge_validation.judge_a.within_1 | .0% }} | {{ judge_validation.judge_b.within_1 | .0% }} |
| Cohen's κ on "acceptable" | {{ judge_validation.judge_a.cohen_kappa_accept | .2f }} | {{ judge_validation.judge_b.cohen_kappa_accept | .2f }} |
| judge mean − my mean | {{ judge_validation.judge_a.judge_mean_minus_human_mean | +.2f }} | {{ judge_validation.judge_b.judge_mean_minus_human_mean | +.2f }} |

The rule I set before running this: if κ is under ~0.4 for both judges, the
quality headline doesn't get to stand without the caveat in §5. It's under 0.4.
More on that below.

---

## 4. Results

Baselines: two trivial ones (majority intent + canned reply, one escalates
everything, one auto-handles everything), and a "simple" one that's a TF-IDF +
logistic-regression classifier distilled from LLM weak labels, plus the top
retrieved reply verbatim, plus the same rule-based gate. No generative model in
the simple baseline.

<!-- INCLUDE:results/RESULTS.md -->

Reading the table:

The `simple` -> `agent` gap on macro-F1 is what the LLM buys over cheap
distillation. `trivial_always_auto` auto-handles 100% and has the worst cost per
message, which is exactly the number a naive "automation %" pitch would brag
about. `unsafe_auto_rate` is the one I'd actually defend: the fraction of all
traffic where the agent shipped something a human would reject.

Bootstrap CI on agent quality: {{ models.agent.quality_judge_a.mean_overall_ci95 }}.
Agent intent calibration (ECE): {{ models.agent.intent.ece | .3f }}. That's not
great, the confidence scores the policy leans on aren't very trustworthy, which
is a problem given the gate uses a confidence threshold.

`results/deferral_curve.png` is quality vs coverage; `results/confusion_agent.png`
shows where the intent errors cluster (mostly `other` and `device_or_connect`,
both of which are small).

---

## 5. What's misleading about my headline number

Say the headline is *"{{ models.agent.quality_judge_a.mean_overall_all | .2f }}/5
mean reply quality, {{ models.agent.quality_judge_a.safe_automation_rate | .0% }}
safely automated"*. Reasons not to trust that as written:

**The judge doesn't track a human, so the quality number is mostly noise.**
Against my {{ judge_validation.n }} blind ratings, Judge A correlates at
{{ judge_validation.judge_a.spearman | .2f }} and Judge B at
{{ judge_validation.judge_b.spearman | .2f }}. Both κ ≈ 0. Judge A sits
{{ judge_validation.judge_a.judge_mean_minus_human_mean | +.2f }} above me, Judge
B {{ judge_validation.judge_b.judge_mean_minus_human_mean | +.2f }} below. With
these free models I can't put a defensible number on reply quality at all. The
intent and escalation metrics are what carry this report; the quality score is a
direction, not a measurement. This is the biggest problem with the headline and
it's one I built the tooling to find.

**The quality number only covers the messages the agent chose to answer.**
`mean_overall_auto_only` is measured on the easy subset the gate picked, not on a
random tweet. The deferral curve is the honest version.

**Judge A is the drafter's own model.** It rates the agent
{{ judge_cross_vendor.A_minus_B_on_agent | +.2f }} above Judge B. The gap is
actually bigger on the simple baseline
({{ judge_cross_vendor.A_minus_B_on_simple | +.2f }}), so this reads less like
targeted self-flattery and more like Judge A just being soft on everything.

**"Resolved" is a guess, and an optimistic one.** Grounding prefers past replies
where the customer said thanks or went quiet. Someone giving up and churning
looks the same as someone who got helped, so the evidence pool leans toward
replies that calmed people down, not necessarily ones that fixed the problem.

**I designed the taxonomy and I labelled the golden set, twice.** So intent
accuracy is partly measuring "does the classifier think like me". The
second-rater check exists on paper (`adjudication_sample.csv`) but I didn't run
it.

**Nothing here tests distribution shift.** Train and test are both late 2017,
same product, same support team. Real deployment hits new features and new bug
waves. The temporal split kills leakage, it doesn't simulate drift. Expect every
number to sag in production.

**Macro-F1 averages away the classes that matter.** `account_access`,
`billing_subscription`, `cancel_or_refund` are rare, so a weak score there barely
moves macro-F1 but is exactly what would drive unsafe auto-handling.

**n = {{ n_golden }}.** The CI on quality is
{{ models.agent.quality_judge_a.mean_overall_ci95 }}. Any agent-vs-simple gap
smaller than that is noise.

**Escalating is treated as almost free.** Cost/msg only charges a needless
escalation 1x, but a system that escalates 90% of the time hasn't reduced human
load, which was the whole point.

To actually trust a quality number I'd want a judge that passes human validation
(κ > 0.6, probably a frontier model or an ensemble with a rationale audit),
quoted with its CI at a fixed point on the deferral curve, with unsafe-auto
broken out for the high-risk intents and held near zero, checked against a fresh
sample from a later time slice.

---

## 6. Failure analysis

<!-- INCLUDE:results/failure_analysis.md -->

---

## 7. With another week

1. Multi-turn. Feed the last few turns as context, re-label ~60 golden rows as
   conversations, measure how often it derails. This is failure mode #1.
2. Per-intent escalation thresholds instead of one global bar, tuned against the
   cost matrix on a dev split.
3. A real groundedness check: a second pass that drops any sentence an NLI model
   can't entail from the evidence, so "grounded" becomes a measured rate instead
   of the drafter's own say-so.
4. A trained "was this resolved" classifier (~300 hand labels) to replace the
   regex, and carry its uncertainty into retrieval weighting.
5. A judge ensemble (3 models, median) plus spot-checking 50 rationales for the
   biases judges are known to have (length, politeness).
6. Slice the test window by month and report the metric trajectory, not one
   number.
7. Swap the 5x/1x cost guess for real handle times if that data exists.

---

## Appendix

- Decisions and trade-offs: [DECISION_LOG.md](DECISION_LOG.md)
- Setup and reproduction: [README.md](README.md), [SETUP.md](SETUP.md)
- Raw numbers: [`results/metrics.json`](results/metrics.json)
