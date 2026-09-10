# Golden set — how it was sampled and labelled

**Source.** Test split only (`artifacts/test.parquet`): the newest
19% of SpotifyCares
threads by time, held out from the retrieval index and the classifier's weak
labels. Building the golden set from the test period avoids scoring the agent
on messages its own index has already seen.

**Sampling.** Stratified by (suggested intent × message-length bucket ×
early/late half of the test period), proportional allocation with a floor of 4
per intent so rare high-risk intents are not washed out. **n = 148.**

**Final intent distribution:** billing_subscription 30 · feature_request_or_feedback 28
· other 26 · playback_or_app_bug 23 · content_availability 15 · account_access 14
· cancel_or_refund 4 · device_or_connect 4 · family_or_duo_plan 4.
Action: 83 escalate / 65 auto_handle.

**Labelling protocol (two passes).**
1. Each row was pre-labelled by the few-shot LLM classifier — shown as a
   *suggestion only*.
2. **Pass 1 (`scripts/build_golden_set.py --review`):** every row was reviewed
   against the customer message + the brand's real historical reply + the top
   retrieved precedent, and `gold_intent` / `gold_action` / free-text
   `gold_notes` were set. First-pass agreement with the suggestion was ~0.95 on
   intent — high, because the LLM's guesses were mostly reasonable.
3. **Pass 2 (`scripts/refine_other.py`):** the first pass left 84/148 rows in
   the catch-all `other` bucket — the LLM's default when unsure. The definition
   of `other` is "spam, jokes, unclear, nothing actionable", so any row with a
   discernible topic was re-examined and moved to its real intent. 58 of the 84
   were reclassified (billing, feature-feedback, playback, content, account);
   26 stayed `other` (genuine banter / one-word follow-ups / non-support). Rows
   touched in pass 2 carry `[2nd pass: reclassified from other]` in `gold_notes`.
4. `gold_action` follows the policy in DECISION_LOG #9: escalate whenever the
   issue needs account/email/payment info or is a security/legal matter.
5. 40 rows (`golden/adjudication_sample.csv`) are set aside for a second rater;
   Cohen's κ on their labels is the inter-annotator check.

**Known limitation:** the two-pass process means the labels are anchored to one
person's reading of a 9-way taxonomy the same person designed — see REPORT.md
§5.4. The `other` → specific reclassification in pass 2 is a judgement call that
a second rater should stress-test.

**Label definitions.**
- **playback_or_app_bug** — Songs won't play / skip / cut out, app crashes or freezes, offline downloads broken, audio quality, shuffle behaviour.
- **account_access** — Can't log in, password reset, email/username change, account hacked or compromised, unexpected logout, 2FA.
- **billing_subscription** — Charged unexpectedly, wrong amount, payment method won't update, receipts/invoices, upgrade/downgrade Premium, student/trial pricing.
- **cancel_or_refund** — Wants to cancel Premium, asking for a refund, disputing a specific charge, 'stop charging me'.
- **family_or_duo_plan** — Family/Duo plan management, address verification, adding/removing members, plan admin vs member confusion.
- **content_availability** — A song/album/podcast is missing, removed, greyed out, region-locked, or wrong metadata.
- **device_or_connect** — Spotify Connect, cars (Android Auto/CarPlay), smart speakers, Alexa/Google Home, PS/Xbox, casting between devices.
- **feature_request_or_feedback** — Feature requests, UX complaints, general praise or venting that is not an actionable support issue.
- **other** — Spam, jokes, unclear, non-Spotify, or nothing actionable.
