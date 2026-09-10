# TrustAgent

Brand: `@SpotifyCares`. Dataset: Kaggle `customer-support-on-twitter`.
Golden set: 148 held-out messages I labelled myself (§3).

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
- **Any fine-tuning.** 148 labels isn't enough to fine-tune without
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
(`allam-2-7b`), told to only use fixes that appear
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

### 3.1 Golden set (148 rows)

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

Rubric is in `judge.py`. Judge A is `allam-2-7b`, the
same model that writes the drafts (a deliberate worst-case for self-preference).
Judge B is `openai/gpt-oss-120b`, a different model family.

I rated 48 replies blind myself and compared:

| | Judge A | Judge B |
|---|---|---|
| Spearman vs me | -0.03 | 0.08 |
| MAE vs me (1-5) | 1.06 | 1.44 |
| within 1 point of me | 73% | 56% |
| Cohen's κ on "acceptable" | 0.11 | -0.00 |
| judge mean − my mean | +0.77 | -0.94 |

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

_SpotifyCares, golden n=148, seed 13, retriever=tfidf. Regenerate with `make eval`._

| model | intent acc | intent macroF1 | escal. P | escal. R | auto-handle rate | unsafe-auto rate | safe-automation rate | judgeA mean |
|---|---|---|---|---|---|---|---|---|
| trivial_always_escalate | 0.19 | 0.04 | 0.56 | 1.00 | 0.00 | nan | nan | nan |
| trivial_always_auto | 0.19 | 0.04 | 0.00 | 0.00 | 1.00 | nan | nan | nan |
| simple | 0.36 | 0.32 | 0.57 | 0.98 | 0.03 | 0.000 | 0.03 | 4.63 |
| agent | 0.65 | 0.63 | 0.74 | 0.94 | 0.28 | 0.000 | 0.37 | 4.90 |

**Judge A vs Judge B on the agent's replies:** allam-2-7b gives 4.90, openai/gpt-oss-120b gives 3.43 (gap +1.47; on the simple baseline the gap is +2.00). Neither passes the human check in §3.3.


Reading the table:

The `simple` -> `agent` gap on macro-F1 is what the LLM buys over cheap
distillation. `trivial_always_auto` auto-handles 100% and has the worst cost per
message, which is exactly the number a naive "automation %" pitch would brag
about. `unsafe_auto_rate` is the one I'd actually defend: the fraction of all
traffic where the agent shipped something a human would reject.

Bootstrap CI on agent quality: 4.90  (95% CI 4.77–5.00).
Agent intent calibration (ECE): 0.287. That's not
great, the confidence scores the policy leans on aren't very trustworthy, which
is a problem given the gate uses a confidence threshold.

`results/deferral_curve.png` is quality vs coverage; `results/confusion_agent.png`
shows where the intent errors cluster (mostly `other` and `device_or_connect`,
both of which are small).

---

## 5. What's misleading about my headline number

Say the headline is *"4.90/5
mean reply quality, 37%
safely automated"*. Reasons not to trust that as written:

**The judge doesn't track a human, so the quality number is mostly noise.**
Against my 48 blind ratings, Judge A correlates at
-0.03 and Judge B at
0.08. Both κ ≈ 0. Judge A sits
+0.77 above me, Judge
B -0.94 below. With
these free models I can't put a defensible number on reply quality at all. The
intent and escalation metrics are what carry this report; the quality score is a
direction, not a measurement. This is the biggest problem with the headline and
it's one I built the tooling to find.

**The quality number only covers the messages the agent chose to answer.**
`mean_overall_auto_only` is measured on the easy subset the gate picked, not on a
random tweet. The deferral curve is the honest version.

**Judge A is the drafter's own model.** It rates the agent
+1.47 above Judge B. The gap is
actually bigger on the simple baseline
(+2.00), so this reads less like
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

**n = 148.** The CI on quality is
4.90  (95% CI 4.77–5.00). Any agent-vs-simple gap
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

# Failure analysis — agent

Pool = 30 worst-scoring golden cases (by combined Judge A+B overall).

| rank | failure mode | count | share |
|---|---|---|---|
| 1 | `multiturn_blind` | 15 | 50% |
| 2 | `generic_nonanswer` | 10 | 33% |
| 3 | `tone_off` | 3 | 10% |
| 4 | `intent_confusion` | 1 | 3% |
| 5 | `retrieval_irrelevant` | 1 | 3% |

## 1. `multiturn_blind` — Needed earlier conversation context the single-message agent never saw.  (15 cases)

> **customer:** what’s going on I can’t do anything but listen to my playlists
> **agent (playback_or_app_bug, escalate):** Hey Megan, help's here! Just to check, are you trying to sync local files from your desktop to your phone? /DV
> **gold:** intent=other action=escalate  · judgeA=5

> **customer:** Hey How come you have no #blackfriday or #CyberMonday discounts... For instance on family premium packages?
> **agent (billing_subscription, escalate):** Hey there! To assist you with your subscription concerns
> **gold:** intent=feature_request_or_feedback action=auto_handle  · judgeA=5

**Seed hypothesis.** We evaluate on the first message only. Fix: feed the last 3 turns as context.

**Author note.** The agent has no way to tell it's replying to turn 3 of a thread, so it just invents a plausible turn 1. "I can't do anything but listen to my playlists" gets a question about syncing local files that has nothing to do with anything, and the Black Friday one trails off mid-sentence. Worth noting: judgeA scored both a 5. The same-model judge can't see the problem either, which is really the §5 point showing up in the failure pool. Cheapest fix short of real multi-turn is a "is this a standalone message or a fragment" check that routes fragments to escalate.


## 2. `generic_nonanswer` — Vague 'we're looking into it' / 'DM us' with no actual help.  (10 cases)

> **customer:** How to request a song / album (origional album or artist) to upload in your library which is still not there . Please let me know via direct message or send me link. Waiting reply.
> **agent (feature_request_or_feedback, escalate):** Hi there! To request a specific song or album (including original albums or artists), you can use the 'Request Song' feature. To do so, follow these steps: 1. Click on the three dots next to the song or album you want. 2. Select 'Request Song'. 3. Choose the country you'd like the song to be available in. If the song isn't available in your country yet, we'll let you know. If you can't find the feature, make sure you have the latest version of Spotify. If you still can't find it, please provide your device and Spotify version, and we'll assist further
> **gold:** intent=content_availability action=auto_handle  · judgeA=5

> **customer:** Fam why is “Rain On Me” NOT on Chapter II ??
> **agent (content_availability, auto_handle):** Don't fret! It's being updated regularly & we'll be sure to let the right team know of your observations. Any questions, give us a shout /GN
> **gold:** intent=content_availability action=auto_handle  · judgeA=5

**Seed hypothesis.** Model hedges when evidence is thin instead of escalating. Fix: forbid 'DM us' as a *reply* — that path must escalate.

**Author note.** Two different failures got bucketed together here. The Rain On Me reply is the classic vague hedge ("it's being updated regularly"). But the "request a song" reply is worse: allam invented a whole "Request Song" feature with numbered steps that does not exist. When the retrieved evidence is thin the small model either hedges or confidently makes something up, and the self-reported `grounded` flag caught neither. Fix is an external groundedness check (does the reply contain a fact or step that's actually in the evidence?), and if not, escalate instead of letting the drafter improvise.


## 3. `tone_off` — Correct content but off-brand: robotic, preachy, or too long.  (3 cases)

> **customer:** what the fuq with your web player ? #spotifyth it keeps getting crashed all the freaking time !
> **agent (playback_or_app_bug, escalate):** Hey there! We understand how frustrating this must be. Please try clearing your Spotify cache and cookies, then restart the web player. If the issue persists, kindly provide us with the browser you're using and its version. We'll do our best to assist you. /JH
> **gold:** intent=playback_or_app_bug action=escalate  · judgeA=4

> **customer:** still upset that got rid of messaging a while ago lol
> **agent (other, auto_handle):** Hey there! We understand your concerns. Please DM us your Spotify account's email or username, and we'll investigate the issues you mentioned. You can also check our content information here: /PB
> **gold:** intent=feature_request_or_feedback action=auto_handle  · judgeA=5

**Seed hypothesis.** Voice guide not enforced. Fix: add 2 golden style exemplars + a length cap.

**Author note.** allam does not hold the voice spec. The web-player reply is a support-macro wall ending "kindly provide us with the browser you're using and its version", nothing like how @SpotifyCares actually writes. The "still upset about messaging" one is pure nostalgia venting and the agent asks for an account email to "investigate the issues" when there are no issues. This is a model-size ceiling, not a prompt bug. A 7B model in JSON mode won't reliably follow "1-3 sentences, lowercase ok". Real fix is a bigger drafter or a cheap rewrite pass.


## 4. `intent_confusion` — Wrong intent, which then routed the whole response wrong.  (1 cases)

> **customer:** I know, but your UI is saying that I’m at 1k+ of 4k+ downloads. It seems like it should say 1k+ of 3,333. ;)
> **agent (other, escalate):** Hi there! We understand your concern about the download progress. Based on the evidence provided, it seems like the UI is still in testing phases. Your account is indeed downloading correctly. If you have any further feedback, please feel free to share it
> **gold:** intent=feature_request_or_feedback action=escalate  · judgeA=5

**Seed hypothesis.** Adjacent labels (billing vs cancel; playback vs device) leak. Fix: merge or add contrastive few-shot examples for the confused pair.

**Author note.** The download-counter nitpick ("should say 1k+ of 3,333 ;)") is playful feature feedback, but the emoji and the lack of an explicit ask make it read as "unclear" to the classifier, so it lands in `other` and gets escalated. The reply then invents that "the UI is still in testing phases", another small hallucination. Fix: add feature-feedback few-shot examples that include sarcasm and emoji so the classifier stops defaulting these to `other`.


## 5. `retrieval_irrelevant` — Top precedent was not actually similar; grounding is spurious.  (1 cases)

> **customer:** So keeps doing this thing where all my downloaded songs disappear and then when I go to listen to music offline I can't. Pretty awesome. any idea why? I've reinstalled, cleared out the offline devices list, redownloaded music numerous times.
> **agent (playback_or_app_bug, escalate):** Hi there! We understand how frustrating this must be. Please DM us your account's email address so we can take a closer look and see if we can resolve this issue for you. Stay tuned for updates!
> **gold:** intent=playback_or_app_bug action=escalate  · judgeA=5

**Seed hypothesis.** TF-IDF/embedding retrieves lexically-close but semantically-different cases. Fix: re-rank top-20 with the LLM; drop hits it flags as off-topic.

**Author note.** The customer explicitly lists everything they already tried (reinstall, clear offline list, redownload) and the agent still replies "please DM us your account's email", i.e. start over. TF-IDF matched on the shared words (download, offline, music) and pulled a generic escalation template; nothing in the pipeline notices that the obvious steps are already exhausted. Two fixes: embedding retrieval plus an LLM re-rank to drop off-topic hits, and a pass that extracts "already tried X" from the message and down-weights precedents that just re-suggest it.



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
