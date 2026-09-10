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

