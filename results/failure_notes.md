# Failure-mode notes

One paragraph per mode. `make failures` reads these and won't overwrite them.

## multiturn_blind

The agent can't tell it's replying to turn 3 of a thread, so it invents a
plausible turn 1. "I can't do anything but listen to my playlists" gets a
question about syncing local files that relates to nothing, and the Black Friday
one just trails off. Note the judge scored both a 5, so the same-model judge
can't see the problem any better than the drafter can. Short of real multi-turn,
the cheap fix is a check for "is this a standalone message or a fragment" that
routes fragments straight to escalate.

## generic_nonanswer

Two different failures got lumped here. The Rain On Me reply is the vague hedge
("it's being updated regularly"). The "request a song" reply is worse: the model
invented a whole "Request Song" feature with numbered steps that doesn't exist.
When the retrieved evidence is thin the small model either hedges or confidently
makes something up, and the self-reported `grounded` flag caught neither. Fix is
an external groundedness check, and if it fails, escalate instead of letting the
drafter improvise.

## tone_off

The model doesn't hold the voice spec. The web-player reply is a support-macro
wall ending "kindly provide us with the browser you're using and its version",
nothing like how the brand writes. The "still upset about messaging" one is pure
venting and the agent asks for an account email to "investigate the issues" when
there are none. This is a model-size ceiling, not a prompt bug. A 7B model in
JSON mode won't reliably follow "1-3 sentences, lowercase ok". Real fix is a
bigger drafter or a cheap rewrite pass.

## intent_confusion

The download-counter nitpick ("should say 1k+ of 3,333 ;)") is playful feature
feedback, but the emoji and the missing explicit ask make it read as unclear, so
it lands in `other` and gets escalated. The reply then invents that "the UI is
still in testing phases". Fix: add feature-feedback few-shot examples with
sarcasm and emoji so the classifier stops defaulting these to `other`.

## retrieval_irrelevant

The customer lists everything they already tried (reinstall, clear offline list,
redownload) and the agent still says "please DM us your email", i.e. start over.
TF-IDF matched on the shared words and pulled a generic escalation template;
nothing notices the obvious steps are exhausted. Two fixes: embedding retrieval
plus an LLM re-rank to drop off-topic hits, and a pass that pulls "already tried
X" out of the message and down-weights precedents that just re-suggest it.

## hallucinated_fix

Same root cause as generic_nonanswer above: thin evidence, small model fills the
gap with a confident invented procedure.

## unsafe_auto_handle

Didn't show up in this run (unsafe-auto rate is 0), but the risk is a
paraphrased billing complaint ("charged me twice") that has no keyword hit and a
retrieved precedent that looks answerable. The mitigation is routing on the
classifier's probability mass over the high-risk intents, not just keywords.

## over_escalation

The flip side: one global `min_retrieval_support` bar is too high for the common
low-risk intents. Per-intent thresholds would let playback and content questions
through at a lower bar.
