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
