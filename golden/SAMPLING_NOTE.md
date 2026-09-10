# How the golden set was built

**Where it comes from.** The test split only: the newest ~19% of @SpotifyCares
threads by time. The retrieval index and the classifier's weak labels never see
these, so the agent isn't being scored on messages it already trained on.

**How rows were picked.** Stratified by (suggested intent x message length x
early or late half of the test window), with a floor of 4 per intent so the rare
high-risk ones don't vanish from the sample. n = 148.

**Final distribution:** billing_subscription 30, feature_request_or_feedback 28,
other 26, playback_or_app_bug 23, content_availability 15, account_access 14,
cancel_or_refund 4, device_or_connect 4, family_or_duo_plan 4. Action split: 83
escalate, 65 auto-handle.

**Labelling, two passes.**

Pass 1 (`build_golden_set.py --review`): I went through every row with the
customer message, Spotify's real reply, and the top retrieved precedent visible,
and set intent, action, and a free-text note. The model pre-fills a suggestion; I
agreed with it about 95% of the time on intent, mostly because the suggestions
were reasonable.

That first pass left 84 of 148 rows as `other`, which is what the model picks
when it's unsure. `other` is meant to be "spam / joke / nothing actionable", so
I did a second pass (`refine_other.py`) over just those rows and moved 58 of them
to a real intent (billing, feature feedback, playback, content, account). 26
stayed `other` because they genuinely are banter or one-word follow-ups. Those
rows have `[2nd pass: reclassified from other]` in their notes.

Action follows the policy in DECISION_LOG #9: escalate whenever the issue needs
account / email / payment info, or is security or legal.

**What's weak about this.** One person designed the 9-way taxonomy and did both
labelling passes, so the labels partly encode my own reading. There's a 40-row
adjudication sample (`adjudication_sample.csv`) for a second rater but I didn't
get that number. The pass-2 `other` reclassification in particular is a judgement
call a second rater should push back on. This is in REPORT §5.

**Label definitions.**

- **playback_or_app_bug** — won't play, skips, cuts out, app crashes or freezes,
  offline downloads broken, audio quality, shuffle behaviour
- **account_access** — can't log in, password reset, email/username change,
  hacked or compromised, unexpected logout, 2FA
- **billing_subscription** — charged wrong, payment method won't update,
  receipts, upgrade/downgrade, student or trial pricing
- **cancel_or_refund** — wants to cancel Premium, asking for a refund, disputing
  a specific charge
- **family_or_duo_plan** — plan admin, address verification, adding/removing
  members, manager vs member confusion
- **content_availability** — a song/album/podcast missing, removed, greyed out,
  region-locked, or wrong metadata
- **device_or_connect** — Connect, cars, smart speakers, Alexa/Google Home,
  consoles, casting between devices
- **feature_request_or_feedback** — feature asks, UX complaints, praise or
  venting that isn't an actionable support issue
- **other** — spam, jokes, unclear, non-Spotify, nothing actionable
