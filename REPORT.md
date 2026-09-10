<!-- This file has {{...}} placeholders. The filled version with real numbers is
     REPORT_GENERATED.md (run `make report` to regenerate it). -->

# TrustAgent

I built a small support agent for `@SpotifyCares` and spent most of the time
trying to figure out whether it actually works. It mostly doesn't, in ways that
took a real evaluation to see, which is the interesting part.

Given one incoming tweet the agent picks an intent, drafts a reply from
retrieved past cases, and decides whether to auto-handle it or send it to a
human. Numbers in this doc come straight out of `results/metrics.json`.

## Picking the brand and the goal

I skimmed a few brands first. Amazon and Apple have more volume but they push
almost everything to DM, so the public thread never contains the actual fix, and
grounding a reply in "how they resolved it before" falls apart. Airlines are the
same. Spotify keeps a lot of fixes in the open (toggle offline mode, reinstall,
"that album is region-locked") while still having a hard core of billing and
account cases where a confident wrong answer is genuinely bad. So Spotify.

For this brand "good" isn't "answer everything". It's mostly about routing
safely: never auto-answer an account, billing or security message, because
missing one of those costs far more than a needless escalation. When it does
answer, it shouldn't invent steps, it should sound like the brand, and I should
be honest about how little it actually covers. The number I care about is the
share of traffic that gets auto-handled *and* would pass a human, reported next
to the share that got auto-handled and shouldn't have.

A few things I left out on purpose. No multi-turn, because the public threads cut
off at "DM us" and I can't score a conversation fairly on this data (this turned
out to be the agent's biggest weakness anyway). No fine-tuning, since a
{{ n_golden }}-row set isn't enough to do it honestly. No dense retrieval, no UI.
And nothing tuned to please the LLM judge, because that just breaks the judge.

## How it works

Classify, retrieve, draft, decide.

Intent is a 9-label few-shot classifier. I got the 9 labels by clustering ~2k
training messages, reading the top terms and nearest examples per cluster, then
merging by hand. The raw clusters split playback three ways and lumped every
money problem together, so the taxonomy is a judgement call and I treat it as a
bias source later.

Retrieval is TF-IDF nearest-neighbour over training-period `(message, brand
reply)` pairs, each weighted by whether that old reply looked like it resolved
the issue. Drafting is a small open model on Groq's free tier (`{{ judge_cross_vendor.agent_drafts_by_ }}`), told to only use fixes that appear in the retrieved evidence.

The decide step is a plain rule stack, not a model: it escalates on hard
keywords, low classifier confidence, no precedent, a high-risk intent without
strong precedent, an ungrounded draft, a draft that needs account data, or churn
language. Whatever fired is the reason a human sees. I kept it readable so a
support lead could audit why anything escalated.

## Evaluation

**Golden set.** {{ n_golden }} messages from the test split only (the newest
slice by time, which the retrieval index never sees). Stratified by intent,
message length and time bucket. I labelled every row myself with the real
historical reply visible. First pass agreed with the model's suggestion ~95% of
the time on intent, which honestly is because the suggestions were decent. That
pass also dumped 84 rows into `other`, so I did a second pass over just those and
moved 58 to a real intent. Full details and the caveats are in
`golden/SAMPLING_NOTE.md`. There's a 40-row adjudication sample for a second
rater that I didn't get to.

**Metrics.** For intent: accuracy, macro-F1 (the risky intents are rare so the
average matters), per-class F1, and ECE, because the policy trusts the
confidence score. For escalation: precision and recall on "should escalate",
plus a cost per message that charges a bad auto-reply 5x a needless escalation.
For reply quality: a five-axis LLM-judge rubric plus a holistic score, and a
deferral curve, because a quality number means nothing without the coverage it
was measured at. Bootstrap 95% CIs on the headline because n is small.

**Checking the judge against a human.** Judge A is the same model that writes the
drafts (worst case for self-preference, on purpose). Judge B is a different
family. I rated {{ judge_validation.n }} replies blind and compared:

| | Judge A | Judge B |
|---|---|---|
| Spearman vs me | {{ judge_validation.judge_a.spearman | .2f }} | {{ judge_validation.judge_b.spearman | .2f }} |
| MAE vs me (1-5) | {{ judge_validation.judge_a.mae | .2f }} | {{ judge_validation.judge_b.mae | .2f }} |
| Cohen's κ on "acceptable" | {{ judge_validation.judge_a.cohen_kappa_accept | .2f }} | {{ judge_validation.judge_b.cohen_kappa_accept | .2f }} |
| judge mean − my mean | {{ judge_validation.judge_a.judge_mean_minus_human_mean | +.2f }} | {{ judge_validation.judge_b.judge_mean_minus_human_mean | +.2f }} |

I decided up front that if Cohen's κ came in under about 0.4 for both judges, the
quality headline wasn't allowed to stand on its own. It did. There's a whole
section on that below.

## Results

Baselines: two trivial ones (majority intent plus a canned reply, one escalating
everything and one auto-handling everything), and a "simple" one that's a
TF-IDF + logistic-regression classifier distilled from LLM weak labels plus the
top retrieved reply verbatim. No generative model in the simple baseline.

<!-- INCLUDE:results/RESULTS.md -->

The agent roughly doubles the simple baseline's intent accuracy, which is the
value the LLM buys over cheap distillation. `trivial_always_auto` auto-handles
100% and has the worst cost per message, which is exactly the number a naive
"automation %" pitch would celebrate. The one I'd defend is unsafe-auto rate:
the fraction of all traffic where the agent shipped something a human would
reject.

The bootstrap CI on agent quality is {{ models.agent.quality_judge_a.mean_overall_ci95 }}, and intent calibration is poor (ECE {{ models.agent.intent.ece | .3f }}), which matters because the escalation gate keys off a confidence threshold.

## What's misleading about the headline

The headline reads roughly "{{ models.agent.quality_judge_a.mean_overall_all | .2f }}/5 reply quality, {{ models.agent.quality_judge_a.safe_automation_rate | .0% }} safely automated". A few reasons not to take that at face value.

The big one is that the judges don't agree with me. Judge A's Spearman correlation against my {{ judge_validation.n }} ratings is {{ judge_validation.judge_a.spearman | .2f }} and Judge B's is {{ judge_validation.judge_b.spearman | .2f }}, and both κ are near zero. On average Judge A scores {{ judge_validation.judge_a.judge_mean_minus_human_mean | +.2f }} higher than me and Judge B {{ judge_validation.judge_b.judge_mean_minus_human_mean | +.2f }} lower. So with these free models I can't put a defensible number on reply quality at all. The intent and escalation metrics are what carry this report. The quality score is a direction, not a measurement, and I'm glad I built the validation harness to find that out rather than just quoting the number.

That quality number is also only measured on the messages the agent chose to
answer, which are the easy ones, so it's not what you'd see on a random tweet.
The deferral curve is the honest version.

Judge A is the drafter's own model, and it scores the agent {{ judge_cross_vendor.A_minus_B_on_agent | +.2f }} higher than Judge B does. The gap is actually wider on the simple baseline ({{ judge_cross_vendor.A_minus_B_on_simple | +.2f }}), so this looks less like targeted self-flattery and more like Judge A being soft across the board.

"Resolved" is a guess. Grounding prefers past replies where the customer said
thanks or went quiet, and someone giving up looks identical to someone helped,
so the evidence pool leans toward replies that calmed people down rather than
fixed things.

I designed the taxonomy and labelled the set myself, in two passes, so intent
accuracy partly measures whether the classifier thinks like me. Nothing tests
distribution shift; train and test are both late 2017. Macro-F1 averages away the
rare high-risk classes that are exactly what would drive unsafe auto-handling.
The CI is wide because n is only {{ n_golden }}, so any agent-vs-simple gap smaller than it is noise. And escalating is treated as nearly free by the cost matrix, when a system that escalates most of the time hasn't actually reduced anyone's workload.

To actually trust a quality number I'd want a judge that passes human validation
(κ > 0.6, probably a frontier model or an ensemble), quoted with its CI at a
fixed point on the deferral curve, with unsafe-auto broken out for the high-risk
intents and held near zero, checked against a fresh sample from a later time.

## Failure analysis

<!-- INCLUDE:results/failure_analysis.md -->

## With another week

Multi-turn first, since it's failure mode number one: feed the last few turns,
re-label ~60 rows as conversations, measure how often it derails. Then per-intent
escalation thresholds instead of one global bar, tuned against the cost matrix.
A real groundedness check that drops any sentence an NLI model can't entail from
the evidence, so "grounded" stops being the drafter's own say-so. A trained
"was this resolved" classifier to replace the regex. A judge ensemble with a
rationale spot-check. And slicing the test window by month to see the metric
trajectory instead of one number.

---

Decisions and trade-offs: [DECISION_LOG.md](DECISION_LOG.md). Setup and
reproduction: [README.md](README.md). Raw numbers:
[`results/metrics.json`](results/metrics.json).
