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

**Author note.** The real problem is not just missing context, it is that the agent never *notices* it is replying to a fragment, so it confabulates a confident first turn (asking about "syncing local files" to someone who said "I can't do anything but listen to my playlists"). Both examples are clearly turn 2+ of a thread. A cheap fix ahead of full multi-turn: a binary "standalone message vs reply-fragment" check, and route fragments straight to escalate. Note the judgeA=5 on both, the same-model judge rewards fluent nonsense, which is why these only surfaced via the combined A+B ranking.


## 2. `generic_nonanswer` — Vague 'we're looking into it' / 'DM us' with no actual help.  (8 cases)

> **customer:** Fam why is “Rain On Me” NOT on Chapter II ??
> **agent (content_availability, auto_handle):** Don't fret! It's being updated regularly & we'll be sure to let the right team know of your observations. Any questions, give us a shout /GN
> **gold:** intent=content_availability action=auto_handle  · judgeA=5

> **customer:** when im listening to music it states that i can click the thing to get 30 minutes of ad free listening but when i go to click the part that has been there in the past, its not there so i cant do anything. it has done this about three times since i started listening today
> **agent (playback_or_app_bug, auto_handle):** That doesn't sound good! Did the app crash at any point during the 30 minutes? That can cause the ad-free time to reset /FR
> **gold:** intent=playback_or_app_bug action=auto_handle  · judgeA=5

**Seed hypothesis.** Model hedges when evidence is thin instead of escalating. Fix: forbid 'DM us' as a *reply* — that path must escalate.

**Author note.** The drafter falls back to corporate filler ("we'll let the right team know", "give us a shout") whenever the retrieved evidence lacks a crisp fix, and because allam-2-7b is small, that filler register is its default. The gate then auto-handles these because the drafter self-reported grounded=true. Fix: stop trusting the drafter's own flag, check externally whether the reply contains a concrete instruction or fact, and if not, escalate. The Rain On Me case actually has a real answer (album licensing differs by market); the agent just does not know it and will not admit that.


## 3. `retrieval_irrelevant` — Top precedent was not actually similar; grounding is spurious.  (3 cases)

> **customer:** How to request a song / album (origional album or artist) to upload in your library which is still not there . Please let me know via direct message or send me link. Waiting reply.
> **agent (feature_request_or_feedback, escalate):** Thanks for getting back to us. We do our best! But sometimes agreements are ongoing, or can't be reached with the content provider /GT
> **gold:** intent=content_availability action=auto_handle  · judgeA=3

> **customer:** So keeps doing this thing where all my downloaded songs disappear and then when I go to listen to music offline I can't. Pretty awesome. any idea why? I've reinstalled, cleared out the offline devices list, redownloaded music numerous times.
> **agent (playback_or_app_bug, escalate):** Hi there! We understand how frustrating this must be. Please DM us your account's email address so we can take a closer look and see if we can resolve this issue for you. Stay tuned for updates!
> **gold:** intent=playback_or_app_bug action=escalate  · judgeA=5

**Seed hypothesis.** TF-IDF/embedding retrieves lexically-close but semantically-different cases. Fix: re-rank top-20 with the LLM; drop hits it flags as off-topic.

**Author note.** TF-IDF matches on shared surface words ("song", "library", "download") rather than on what the customer needs, and it has no concept of "already tried". In the second example the customer explicitly lists reinstall + cache-clear + re-download, and the agent still replies "please DM us" because the retrieved precedent was a generic escalation template. Two-part fix: (a) embedding retrieval plus an LLM re-rank that drops off-topic hits, and (b) parse the "already tried X" steps out of the message and down-weight precedents that only re-suggest them.


## 4. `tone_off` — Correct content but off-brand: robotic, preachy, or too long.  (3 cases)

> **customer:** what the fuq with your web player ? #spotifyth it keeps getting crashed all the freaking time !
> **agent (playback_or_app_bug, escalate):** Hey there! We understand how frustrating this must be. Please try clearing your Spotify cache and cookies, then restart the web player. If the issue persists, kindly provide us with the browser you're using and its version. We'll do our best to assist you. /JH
> **gold:** intent=playback_or_app_bug action=escalate  · judgeA=4

> **customer:** still upset that got rid of messaging a while ago lol
> **agent (other, auto_handle):** Hey there! We understand your concerns. Please DM us your Spotify account's email or username, and we'll investigate the issues you mentioned. You can also check our content information here: /PB
> **gold:** intent=feature_request_or_feedback action=auto_handle  · judgeA=5

**Seed hypothesis.** Voice guide not enforced. Fix: add 2 golden style exemplars + a length cap.

**Author note.** allam-2-7b does not reliably obey the voice spec ("1-3 sentences, lowercase ok, <=1 emoji") even with it in the system prompt, the web-player reply is a 45-word macro ending "kindly provide us with the browser you're using and its version". This is a model-capability ceiling, not a prompt bug: a 7B model in JSON mode will not hold style constraints. Real fixes are a stronger drafter or a cheap post-edit pass that rewrites to the house voice; prompt tweaking will not close it.


## 5. `intent_confusion` — Wrong intent, which then routed the whole response wrong.  (1 cases)

> **customer:** I know, but your UI is saying that I’m at 1k+ of 4k+ downloads. It seems like it should say 1k+ of 3,333. ;)
> **agent (other, escalate):** Hi there! We understand your concern about the download progress. Based on the evidence provided, it seems like the UI is still in testing phases. Your account is indeed downloading correctly. If you have any further feedback, please feel free to share it. Grounded: True, Missing Info: None
> **gold:** intent=feature_request_or_feedback action=escalate  · judgeA=5

**Seed hypothesis.** Adjacent labels (billing vs cancel; playback vs device) leak. Fix: merge or add contrastive few-shot examples for the confused pair.

**Author note.** Two bugs stacked. (a) The playful UI-math nitpick (emoji, ";)", no explicit ask) reads as unclear to the classifier, so it lands in `other` and gets escalated instead of logged as feature feedback. (b) The drafter serialised its own control fields into the customer-facing text: "Grounded: True, Missing Info: None" appears in the reply. Fix (a): add feature-feedback few-shot examples with sarcasm/emoji. Fix (b): strict JSON parsing plus a guard that rejects any reply containing "Grounded:" or "Missing Info:".

