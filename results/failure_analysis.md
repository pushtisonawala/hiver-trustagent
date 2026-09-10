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

