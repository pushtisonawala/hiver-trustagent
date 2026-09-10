# Golden set — how it was sampled and labelled

**Source.** Test split only (`artifacts/test.parquet`): the newest
19% of SpotifyCares
threads by time, held out from the retrieval index and the classifier's weak
labels. Building the golden set from the test period avoids scoring the agent
on messages its own index has already seen.

**Sampling.** Stratified by (suggested intent × message-length bucket ×
early/late half of the test period), proportional allocation with a floor of 4
per intent so rare high-risk intents are not washed out. n = 148.
Intent distribution of the draw: {'other': 84, 'billing_subscription': 20, 'feature_request_or_feedback': 14, 'account_access': 9, 'playback_or_app_bug': 8, 'cancel_or_refund': 4, 'content_availability': 4, 'family_or_duo_plan': 4, 'device_or_connect': 1}

**Labelling protocol.**
1. Each row was pre-labelled by the few-shot LLM classifier and a
   retrieval-grounded draft — shown as *suggestions only*.
2. The author labelled every row in `--review`, seeing the customer message,
   the brand's real historical reply, and the top retrieved precedent, and
   assigned: `gold_intent` (from the 9-label taxonomy), `gold_action`
   (auto_handle / escalate under the policy in DECISION_LOG #9), and free-text
   `gold_notes` for anything ambiguous.
3. Disagreements with the suggestion were kept (not silently accepted) — see
   `agree_rate` printed at the end of review.
4. 40 rows (`golden/adjudication_sample.csv`) were relabelled by a
   second rater; Cohen's κ is reported in `results/metrics.json` via
   `make judge-validation`-style agreement (see REPORT.md §Evaluation).

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
