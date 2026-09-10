# Failure analysis — agent

Pool = 40 worst-scoring golden cases (by combined Judge A+B overall).

| rank | failure mode | count | share |
|---|---|---|---|
| 1 | `generic_nonanswer` | 37 | 92% |
| 2 | `over_escalation` | 1 | 2% |
| 3 | `hallucinated_fix` | 1 | 2% |
| 4 | `intent_confusion` | 1 | 2% |

## 1. `generic_nonanswer` — Vague 'we're looking into it' / 'DM us' with no actual help.  (37 cases)

> **customer:** Hello , please make it possible for us to change a playlist cover on the mobile app. I edit all my covers on my iPhone and it’s always a hassle to then transfer it to my pc to upload it there. Thanks in advance.
> **agent (feature_request_or_feedback, escalate):** we can see how that’d be handy! you can vote for the official idea here: /AG
> **gold:** intent=feature_request_or_feedback action=escalate  · judgeA=2

> **customer:** It doesn’t because you’re basically telling me to buy another device that does what the roku does just to listen to Spotify. Are there any plans on making Spotify work well and not look like trash on Rokus?
> **agent (feature_request_or_feedback, escalate):** we hear you on the frustration with the current experience. since we can't confirm specific roadmap details here, we'd love to continue this conversation in DMs to share any available updates.
> **gold:** intent=device_or_connect action=escalate  · judgeA=2

**Seed hypothesis.** Model hedges when evidence is thin instead of escalating. Fix: forbid 'DM us' as a *reply* — that path must escalate.

**Author note.** _<add your sharper hypothesis + what you actually saw here>_


## 2. `over_escalation` — Escalated something the agent could clearly have answered.  (1 cases)

> **customer:** it looks like somebody brought this up 5 years ago, but i don't see the feature yet: is it on the road map yet?
> **agent (feature_request_or_feedback, escalate):** thanks for the nudge! i can't share specific roadmap details here, but i'm passing your feedback along to the team so they know it's still on people's minds. 🎧
> **gold:** intent=feature_request_or_feedback action=auto_handle  · judgeA=2

**Seed hypothesis.** min_retrieval_support is tuned too high for common intents. Fix: per-intent thresholds; low-risk intents get a lower bar.

**Author note.** _<add your sharper hypothesis + what you actually saw here>_


## 3. `hallucinated_fix` — Reply states steps/policy not present in the evidence.  (1 cases)

> **customer:** Uh : Why does Daft Punk's "One More Time" fade out, then go into "Aerodymanic" on the album Discovery? "One More Time" leads into "Aerodymanic".
> **agent (other, escalate):** that's just how the album was mastered! the tracks are designed to flow into each other, so the fade out is part of the original recording.
> **gold:** intent=content_availability action=auto_handle  · judgeA=2

**Seed hypothesis.** Drafter treats low-similarity hits as usable. Fix: hard gate reply generation on max similarity, and add a self-check pass that deletes unsupported sentences.

**Author note.** _<add your sharper hypothesis + what you actually saw here>_


## 4. `intent_confusion` — Wrong intent, which then routed the whole response wrong.  (1 cases)

> **customer:** i watched the video 174892 times, when do i get my 30 minutes of ad free music?
> **agent (other, auto_handle):** that sounds frustrating! did the app crash at any point during the 30 minutes? that can cause the ad-free time to reset /FR
> **gold:** intent=feature_request_or_feedback action=auto_handle  · judgeA=3

**Seed hypothesis.** Adjacent labels (billing vs cancel; playback vs device) leak. Fix: merge or add contrastive few-shot examples for the confused pair.

**Author note.** _<add your sharper hypothesis + what you actually saw here>_

