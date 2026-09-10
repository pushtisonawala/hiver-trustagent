# Decision log

The non-obvious calls I made and why. Referenced by number from REPORT.md.

1. **Brand: @SpotifyCares.** `make scan-brands` ranks brands by how many replies
   they sent to customers. AmazonHelp and AppleSupport are bigger but resolve
   almost everything in DM, so the public thread rarely contains the actual fix,
   which kills the "draft a reply grounded in past resolutions" part of the task.
   Airlines are the same ("DM us your record locator"). Spotify has the best mix:
   lots of self-contained public fixes (settings toggles, reinstall steps,
   licensing explanations) plus a hard core of account/billing cases that make
   the escalate/auto decision actually matter.

2. **One inbound message in, one (intent, draft, decision) out. No multi-turn.**
   The public threads mostly stop at "DM us", so I can't score a conversation
   fairly on this data. The cost is that the agent never gets marked down for
   derailing across turns, which turned out to be its top failure mode anyway.

3. **Split train/test by time, not randomly.** 80% oldest for training and the
   retrieval index, newest 20% for the golden set. A random split lets
   near-duplicate tweets from the same bug wave land on both sides and inflates
   everything. Downside: the test set is one later slice of time, so I can show
   there's no leakage but I can't show robustness to drift.

4. **Taxonomy: cluster first, then curate by hand.** Embedded ~2k training
   messages, ran KMeans (k=12), read the top terms and nearest examples per
   cluster, then merged down to 9 labels plus `other`. The raw clusters were
   unstable: they split playback three ways and lumped all money problems
   together. So the final 9 are a judgement call, and that's a bias source
   (REPORT §5).

5. **`other` is a real label, not a dumping ground.** If a message is genuinely
   unclear it should be flagged as such, not forced into a support intent and
   then confidently auto-answered. (In practice the LLM over-used it, hence the
   second labelling pass, see #8 below and SAMPLING_NOTE.)

6. **Retrieval is local TF-IDF, no embedding API.** Support tweets are short and
   lexical ("won't play", "charged twice", "can't log in"), and TF-IDF bigrams
   find the near-duplicates well enough. It also keeps the whole thing free and
   runnable offline. It does miss paraphrases a dense embedder would catch, which
   shows up as the `retrieval_irrelevant` failure mode. The code still takes
   OpenAI embeddings via `config.yaml` if a key is present.

7. **The "simple" baseline is distilled, not hand-written rules.** The LLM weak-
   labels ~220 training messages and a TF-IDF + logistic-regression model learns
   from those. I wanted a strong simple baseline: if the full agent can't beat a
   linear model plus verbatim retrieval, that's worth knowing. (Kept the count
   low so a live re-run fits the free-tier limits.)

8. **"Resolved" is a regex heuristic over the customer's follow-up.** Thanks or
   silence counts as resolved, "still not working" counts as not, a bare "DM us"
   counts as unknown. There's no real label for this and the heuristic is
   optimistic: someone giving up looks the same as someone helped. Flagged in the
   report. Training a proper classifier on ~300 hand labels is week-2 work.

9. **The escalation gate is a rule stack, not a model.** Anyone can read exactly
   why a message escalated. The rules, in order: hard keywords, low classifier
   confidence, no precedent, high-risk intent without strong precedent, a draft
   the model couldn't ground, a draft that needs account data, churn language. A
   learned gate on ~150 labels would be higher variance and impossible to audit.

10. **Cost matrix: a bad auto-reply is 5x worse than a needless escalation.** A
    wrong confident public reply can leak info or promise a refund that isn't
    coming; a needless escalation just wastes a human glance. The 5x is a guess
    and I call it out as one. Thresholds are set against it, not against F1.

11. **Headline metric is safe-automation rate, always shown with unsafe-auto
    rate.** Not "automation %", because a system that auto-handles everything
    scores 100% on that and is useless. Unsafe-auto (auto-handled and a human
    would reject it) is the number I actually stand behind.

12. **All models are Groq's free tier.** `allam-2-7b` for the agent and Judge A,
    `openai/gpt-oss-120b` for Judge B. One free key, no card. The free tier caps
    each model at roughly 1000 requests/day and 6-8k tokens/minute, which I only
    figured out by reading the response headers after a few slow runs. So
    `llm.py` throttles per model, trims every prompt hard, turns reasoning off
    for classify/judge, backs off on 429, and retries once bigger if a reasoning
    model returns an empty answer. Model names on free tiers change; `config.yaml`
    is the one place to swap them.

13. **Two judges: one is the drafter's own model, one isn't.** Judge A is the
    same `allam-2-7b` that wrote the reply, which is the worst case for
    self-preference and the point. Judge B is a different family. I validated
    both against 50 of my own blind ratings before trusting either. Both failed
    (Spearman near zero), which is the main finding in REPORT §5.

14. **The judge gives a holistic 1-5 separately from the five rubric axes,** not
    a mean of them. Lets me check the overall score against the parts and catch a
    judge that's just averaging instead of judging.

15. **Every LLM call is cached by an input hash (seed 13), and the cache is
    committed** as `llm-cache.tgz` (~900 KB, unpacked by `make setup` / `make
    all`, repacked by `make bundle-cache`). So a reviewer reproduces the exact
    numbers with no key and no cost. Editing a prompt or `config.yaml` misses the
    cache, which is noted in the README.

16. **Subsample cap of 6000 threads.** Enough for stable retrieval and metrics,
    small enough to fit the time budget. The brief encourages subsampling.
    Larger caps didn't move the headline in spot checks.

17. **Small models leak their control fields into the reply text**
    ("Grounded: True, Missing info: None"). `drafting.py` strips those with a
    regex before the reply is used or judged. A stronger drafter wouldn't need
    it; this is a patch, not a fix.
