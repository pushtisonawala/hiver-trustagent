<!-- Template with unfilled placeholders. Read REPORT_GENERATED.md for the
     version with real numbers (produced by `make report`). -->

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
148-row set isn't enough to do it honestly. No dense retrieval, no UI.
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
the issue. Drafting is a small open model on Groq's free tier (`allam-2-7b`), told to only use fixes that appear in the retrieved evidence.

The decide step is a plain rule stack, not a model: it escalates on hard
keywords, low classifier confidence, no precedent, a high-risk intent without
strong precedent, an ungrounded draft, a draft that needs account data, or churn
language. Whatever fired is the reason a human sees. I kept it readable so a
support lead could audit why anything escalated.

## Evaluation

**Golden set.** 148 messages from the test split only (the newest
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
family. I rated 48 replies blind and compared:

| | Judge A | Judge B |
|---|---|---|
| Spearman vs me | -0.03 | 0.08 |
| MAE vs me (1-5) | 1.06 | 1.44 |
| Cohen's κ on "acceptable" | 0.11 | -0.00 |
| judge mean − my mean | +0.77 | -0.94 |

I decided up front that if Cohen's κ came in under about 0.4 for both judges, the
quality headline wasn't allowed to stand on its own. It did. There's a whole
section on that below.

## Results

Baselines: two trivial ones (majority intent plus a canned reply, one escalating
everything and one auto-handling everything), and a "simple" one that's a
TF-IDF + logistic-regression classifier distilled from LLM weak labels plus the
top retrieved reply verbatim. No generative model in the simple baseline.

_SpotifyCares, golden n=148, seed 13, retriever=tfidf. Regenerate with `make eval`._

| model | intent acc | intent macroF1 | escal. P | escal. R | auto-handle rate | unsafe-auto rate | safe-automation rate | judgeA mean |
|---|---|---|---|---|---|---|---|---|
| trivial_always_escalate | 0.19 | 0.04 | 0.56 | 1.00 | 0.00 | nan | nan | nan |
| trivial_always_auto | 0.19 | 0.04 | 0.00 | 0.00 | 1.00 | nan | nan | nan |
| simple | 0.36 | 0.32 | 0.57 | 0.98 | 0.03 | 0.000 | 0.03 | 4.63 |
| agent | 0.65 | 0.63 | 0.74 | 0.94 | 0.28 | 0.000 | 0.37 | 4.90 |

On the agent's replies, allam-2-7b averages 4.90 and openai/gpt-oss-120b averages 3.43 (gap +1.47; the gap on the simple baseline is +2.00).


The agent roughly doubles the simple baseline's intent accuracy, which is the
value the LLM buys over cheap distillation. `trivial_always_auto` auto-handles
100% and has the worst cost per message, which is exactly the number a naive
"automation %" pitch would celebrate. The one I'd defend is unsafe-auto rate:
the fraction of all traffic where the agent shipped something a human would
reject.

The bootstrap CI on agent quality is 4.90  (95% CI 4.77–5.00), and intent calibration is poor (ECE 0.287), which matters because the escalation gate keys off a confidence threshold.

## What's misleading about the headline

The headline reads roughly "4.90/5 reply quality, 37% safely automated". A few reasons not to take that at face value.

The big one is that the judges don't agree with me. Judge A's Spearman correlation against my 48 ratings is -0.03 and Judge B's is 0.08, and both κ are near zero. On average Judge A scores +0.77 higher than me and Judge B -0.94 lower. So with these free models I can't put a defensible number on reply quality at all. The intent and escalation metrics are what carry this report. The quality score is a direction, not a measurement, and I'm glad I built the validation harness to find that out rather than just quoting the number.

That quality number is also only measured on the messages the agent chose to
answer, which are the easy ones, so it's not what you'd see on a random tweet.
The deferral curve is the honest version.

Judge A is the drafter's own model, and it scores the agent +1.47 higher than Judge B does. The gap is actually wider on the simple baseline (+2.00), so this looks less like targeted self-flattery and more like Judge A being soft across the board.

"Resolved" is a guess. Grounding prefers past replies where the customer said
thanks or went quiet, and someone giving up looks identical to someone helped,
so the evidence pool leans toward replies that calmed people down rather than
fixed things.

I designed the taxonomy and labelled the set myself, in two passes, so intent
accuracy partly measures whether the classifier thinks like me. Nothing tests
distribution shift; train and test are both late 2017. Macro-F1 averages away the
rare high-risk classes that are exactly what would drive unsafe auto-handling.
The CI is wide because n is only 148, so any agent-vs-simple gap smaller than it is noise. And escalating is treated as nearly free by the cost matrix, when a system that escalates most of the time hasn't actually reduced anyone's workload.

To actually trust a quality number I'd want a judge that passes human validation
(κ > 0.6, probably a frontier model or an ensemble), quoted with its CI at a
fixed point on the deferral curve, with unsafe-auto broken out for the high-risk
intents and held near zero, checked against a fresh sample from a later time.

## Failure analysis

Worst 30 of the golden set by combined judge score, grouped. Counts are how many of those 30 fell in each mode.

### multiturn_blind (15)

Needed earlier conversation context the single-message agent never saw.

> customer: what’s going on I can’t do anything but listen to my playlists
> agent (playback_or_app_bug, escalate): Hey Megan, help's here! Just to check, are you trying to sync local files from your desktop to your phone? /DV
> gold: other / escalate  (judge A 5)

> customer: Hey How come you have no #blackfriday or #CyberMonday discounts... For instance on family premium packages?
> agent (billing_subscription, escalate): Hey there! To assist you with your subscription concerns
> gold: feature_request_or_feedback / auto_handle  (judge A 5)

The agent can't tell it's replying to turn 3 of a thread, so it invents a
plausible turn 1. "I can't do anything but listen to my playlists" gets a
question about syncing local files that relates to nothing, and the Black Friday
one just trails off. Note the judge scored both a 5, so the same-model judge
can't see the problem any better than the drafter can. Short of real multi-turn,
the cheap fix is a check for "is this a standalone message or a fragment" that
routes fragments straight to escalate.

### generic_nonanswer (10)

Vague 'we're looking into it' / 'DM us' with no actual help.

> customer: How to request a song / album (origional album or artist) to upload in your library which is still not there . Please let me know via direct message or send me link. Waiting reply.
> agent (feature_request_or_feedback, escalate): Hi there! To request a specific song or album (including original albums or artists), you can use the 'Request Song' feature. To do so, follow these steps: 1. Click on the three dots next to the song or album you want. 2. Select 'Request Song'. 3. Choose the country you'd like the song to be available in. If the song isn't available in your country yet, we'll let you know. If you can't find the feature, make sure you have the latest version of Spotify. If you still can't find it, please provide your device and Spotify version, and we'll assist further
> gold: content_availability / auto_handle  (judge A 5)

> customer: Fam why is “Rain On Me” NOT on Chapter II ??
> agent (content_availability, auto_handle): Don't fret! It's being updated regularly & we'll be sure to let the right team know of your observations. Any questions, give us a shout /GN
> gold: content_availability / auto_handle  (judge A 5)

Two different failures got lumped here. The Rain On Me reply is the vague hedge
("it's being updated regularly"). The "request a song" reply is worse: the model
invented a whole "Request Song" feature with numbered steps that doesn't exist.
When the retrieved evidence is thin the small model either hedges or confidently
makes something up, and the self-reported `grounded` flag caught neither. Fix is
an external groundedness check, and if it fails, escalate instead of letting the
drafter improvise.

### tone_off (3)

Correct content but off-brand: robotic, preachy, or too long.

> customer: what the fuq with your web player ? #spotifyth it keeps getting crashed all the freaking time !
> agent (playback_or_app_bug, escalate): Hey there! We understand how frustrating this must be. Please try clearing your Spotify cache and cookies, then restart the web player. If the issue persists, kindly provide us with the browser you're using and its version. We'll do our best to assist you. /JH
> gold: playback_or_app_bug / escalate  (judge A 4)

> customer: still upset that got rid of messaging a while ago lol
> agent (other, auto_handle): Hey there! We understand your concerns. Please DM us your Spotify account's email or username, and we'll investigate the issues you mentioned. You can also check our content information here: /PB
> gold: feature_request_or_feedback / auto_handle  (judge A 5)

The model doesn't hold the voice spec. The web-player reply is a support-macro
wall ending "kindly provide us with the browser you're using and its version",
nothing like how the brand writes. The "still upset about messaging" one is pure
venting and the agent asks for an account email to "investigate the issues" when
there are none. This is a model-size ceiling, not a prompt bug. A 7B model in
JSON mode won't reliably follow "1-3 sentences, lowercase ok". Real fix is a
bigger drafter or a cheap rewrite pass.

### intent_confusion (1)

Wrong intent, which then routed the whole response wrong.

> customer: I know, but your UI is saying that I’m at 1k+ of 4k+ downloads. It seems like it should say 1k+ of 3,333. ;)
> agent (other, escalate): Hi there! We understand your concern about the download progress. Based on the evidence provided, it seems like the UI is still in testing phases. Your account is indeed downloading correctly. If you have any further feedback, please feel free to share it
> gold: feature_request_or_feedback / escalate  (judge A 5)

The download-counter nitpick ("should say 1k+ of 3,333 ;)") is playful feature
feedback, but the emoji and the missing explicit ask make it read as unclear, so
it lands in `other` and gets escalated. The reply then invents that "the UI is
still in testing phases". Fix: add feature-feedback few-shot examples with
sarcasm and emoji so the classifier stops defaulting these to `other`.

### retrieval_irrelevant (1)

Top precedent was not actually similar; grounding is spurious.

> customer: So keeps doing this thing where all my downloaded songs disappear and then when I go to listen to music offline I can't. Pretty awesome. any idea why? I've reinstalled, cleared out the offline devices list, redownloaded music numerous times.
> agent (playback_or_app_bug, escalate): Hi there! We understand how frustrating this must be. Please DM us your account's email address so we can take a closer look and see if we can resolve this issue for you. Stay tuned for updates!
> gold: playback_or_app_bug / escalate  (judge A 5)

The customer lists everything they already tried (reinstall, clear offline list,
redownload) and the agent still says "please DM us your email", i.e. start over.
TF-IDF matched on the shared words and pulled a generic escalation template;
nothing notices the obvious steps are exhausted. Two fixes: embedding retrieval
plus an LLM re-rank to drop off-topic hits, and a pass that pulls "already tried
X" out of the message and down-weights precedents that just re-suggest it.


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
