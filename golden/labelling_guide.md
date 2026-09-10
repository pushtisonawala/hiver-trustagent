# Golden-set labelling guide

Rater instructions for `scripts/build_golden_set.py --review`. Keep this open
while labelling. The goal is *consistency*, not agreement with the LLM
suggestion.

## `gold_intent` — pick exactly one

| label | pick when… | not when… |
|---|---|---|
| `playback_or_app_bug` | song won't play/skips/cuts out, app crash/freeze, offline download broken, shuffle/queue wrong | it's a *device* handoff problem (→ `device_or_connect`) |
| `account_access` | can't log in, password reset broken, email/username change, **hacked/compromised**, unexpected logout | they're logged in but billing is wrong (→ `billing_subscription`) |
| `billing_subscription` | charged wrong amount, card won't update, needs receipt/invoice, student/trial pricing, upgrade/downgrade | they explicitly want to **cancel or get money back** (→ `cancel_or_refund`) |
| `cancel_or_refund` | "cancel my Premium", "refund me", "stop charging me", disputing a specific charge | just *asking how* cancellation works with no intent to act now — still `cancel_or_refund` (intent, not action) |
| `family_or_duo_plan` | plan admin, address verification, add/remove member, manager-vs-member confusion | generic billing on a family plan with no plan-structure issue (→ `billing_subscription`) |
| `content_availability` | a song/album/podcast is missing, greyed out, region-locked, wrong metadata | the content plays but *sounds* broken (→ `playback_or_app_bug`) |
| `device_or_connect` | Connect, CarPlay/Android Auto, Alexa/Google Home, consoles, casting, speakers | the app itself is broken on the phone (→ `playback_or_app_bug`) |
| `feature_request_or_feedback` | feature ask, UX complaint, "bring back X", praise, venting with no actionable issue | there's a real bug underneath the venting (label the bug) |
| `other` | spam, joke, unclear, not about Spotify, no actionable content | you can *reasonably* infer a support intent — then use it |

Tie-breakers: (a) **what does the customer want done**, not what topic they
mention. (b) If two apply, pick the one that determines the *right reply*.
(c) Put genuine ambiguity in `gold_notes` and pick the higher-risk label.

## `gold_action` — `auto_handle` or `escalate`

Label it as **what a good support ops policy should do**, independent of what the
agent did.

**Escalate if ANY of:**
- touches money movement or a specific disputed charge (refund, double charge, "you billed me after I cancelled")
- account identity / security (hacked, "someone's on my account", locked out with reset broken)
- legal / regulatory language (lawyer, GDPR, data deletion, chargeback)
- needs data only the customer's account can provide to answer correctly
- strong churn / anger ("cancelling", "switching to Apple Music", "worst company")
- you, the rater, are not confident a canned-style reply fully resolves it

**Auto-handle if:** it's a known, self-contained fix (settings toggle, reinstall,
"licensing changes by region", Connect re-pair) that Spotify has clearly given
many times, with no account-specific dependency and no money/security angle.

When unsure → `escalate`. A needless escalation costs a glance; a wrong
auto-reply costs trust.

## `gold_notes`

One line. Anything that made this hard, any assumption, any "the LLM suggested X
but it's actually Y because…". This is what we grep when the numbers look weird.

## Second rater

`golden/adjudication_sample.csv` (40 rows) is relabelled by a different person
with only the customer message + brand reply visible (no suggestions). We report
Cohen's κ on both `gold_intent` and `gold_action`.
