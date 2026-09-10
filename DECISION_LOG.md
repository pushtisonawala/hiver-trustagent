# Decision log

Non-obvious choices, why I made them, and what I traded away. Numbered for
reference from REPORT.md.

1. **Brand = @SpotifyCares.** `make scan-brands` ranks brands by
   replies-to-customers. AmazonHelp / AppleSupport are bigger but resolve almost
   everything in DM, so the public transcript rarely contains the resolution —
   which guts the "draft a reply grounded in past resolutions" task. Airlines
   (Delta, etc.) are the same ("please DM your record locator"). Spotify has the
   best ratio of *in-thread, self-contained* resolutions (settings toggles,
   reinstall steps, licensing explanations) while still having a hard core of
   account/billing cases to make the escalation decision non-trivial.

2. **Unit of work = one inbound customer message → (intent, draft, decision).**
   Not multi-turn. The public threads mostly stop at "DM us", so a multi-turn
   agent can't be evaluated honestly on this data. Trade: the agent never gets
   penalised for multi-turn derailment (noted as a headline caveat).

3. **Temporal split (oldest 80% train / newest 20% test), not random.** The
   retrieval index, the weak labels, and the golden set must not see the future.
   A random split lets near-duplicate tweets (same bug wave, same week) land on
   both sides and inflates retrieval + intent metrics. Cost: the test set is a
   single later time-slice, so I can measure leakage-freeness but not drift.

4. **Taxonomy: bootstrap then curate.** Embedded ~2k training messages, KMeans
   k=12, read top TF-IDF terms + nearest examples per cluster
   (`artifacts/taxonomy_bootstrap.json`), then hand-merged to 9 labels + `other`.
   Pure data-driven clusters were unstable and split "playback" three ways while
   collapsing all money issues into one; pure top-down risked missing real
   categories. The 9 labels are a judgement call and a known bias source (REPORT §5.4).

5. **`other` is a real label with medium risk.** Unclear messages should not be
   force-fit into a support intent and then confidently auto-answered.

6. **Retrieval = local TF-IDF + cosine, no embedding API.** Support tweets are
   short and lexical ("won't play", "charged twice", "can't log in"); TF-IDF
   bigrams retrieve near-duplicates well and cost nothing. It also keeps the
   whole pipeline free and offline-capable. Trade: misses paraphrase matches a
   dense embedder would catch — listed as a week-2 upgrade and a failure mode
   (`retrieval_irrelevant`). The code still supports OpenAI embeddings via
   `config.yaml` if you have a key.

7. **`simple` baseline is distilled, not rule-based.** The LLM weak-labels ~220
   training messages, a TF-IDF+LogReg learns from those. This is a *strong*
   simple baseline on purpose — if the full agent can't beat a distilled linear
   model + verbatim retrieval, that's the finding. (Count kept low so a live
   re-run fits the free-tier rate limit; bump `pipeline.build`'s `n` for a
   stronger baseline if you have headroom.)

8. **"Resolved" = weak heuristic over the customer's follow-up** (gratitude /
   no-reply = good; "still not working" = bad; pure "DM us" = unknown). There is
   no label for this. It is optimistic (silent churn looks like success) and is
   flagged as such. Alternative (hand-label 300 threads) is week-2 work.

9. **Escalation gate is a readable rule stack, not a learned model.** A reviewer
   (and a support lead) can read exactly why any message escalated. Rules:
   hard keywords → low intent confidence → weak precedent → high-risk intent w/o
   strong precedent → ungrounded draft → needs account info → churn sentiment.
   A learned gate on 200 labels would be higher-variance and unauditable.

10. **Asymmetric cost: false auto-handle = 5 × needless escalation.** Shipping a
    wrong confident public reply (privacy leak, false refund promise, bad advice)
    is much worse than asking a human to glance at a ticket. The 5× is a
    deliberate guess, called out in REPORT §5.9; thresholds are tuned to it.

11. **Headline metric = safe-automation rate, reported with unsafe-auto rate.**
    Not "automation %". A system that auto-handles everything scores 100% on the
    naïve metric and is useless. `unsafe_auto_rate` (auto-handled AND a human
    would reject it) is the number I actually defend.

12. **LLM stack is all Groq free tier: `qwen/qwen3.8-27b` (agent + Judge A),
    `allam-2-7b` (Judge B, different lineage).** One free key, no card,
    `make all` costs $0. Free-tier limits discovered from the response headers:
    `qwen`/`allam` are ~1000 req/**min** (good); the `openai/gpt-oss-*` models
    are 1000 req/**day** (avoid — we burned a day's quota finding this out).
    All are reasoning-capable, and reasoning tokens count against the 8k
    tokens/min cap, so `llm.py` (a) throttles per-`provider:model` (separate
    buckets → concurrent), (b) trims every prompt and sets
    `reasoning_effort=none` for classify/judge (`low` only for drafting),
    (c) retries once with a bigger budget on an empty answer, (d) backs off on
    429. A live `make all` takes ~20-30 min first time — hence the committed
    cache. Model names on free tiers drift; `config.yaml` is the one place to fix.

13. **LLM-as-judge: same-model self-check + a cross-vendor check.** Judge A is
    the *exact same* `qwen/qwen3.8-27b` that wrote the draft (maximal
    self-preference risk, on purpose), Judge B is `allam-2-7b` (different
    lineage). `A_minus_B` on the
    agent vs on the simple baseline is the bias probe; **Judge B is the
    conservative headline**. Judge validated against a human on 60 blind ratings
    (Spearman + Cohen's κ); self-imposed rule: κ < ~0.4 ⇒ the headline quality
    number must carry the caveat in REPORT §5.

14. **`overall` is asked of the judge separately, not computed as a mean of the
    5 rubric dims.** Lets me check the holistic score against the dimensions and
    catch judges that rubric-average instead of judging.

15. **Every LLM call cached by input-hash, seeded (seed=13), and the cache is
    committed to the repo.** A reviewer runs `make all` and reproduces the exact
    headline numbers in minutes with **no API keys and $0** — the cache replays.
    It's committed as a single `llm-cache.tgz` (596 KB) that `make setup` /
    `make all` unpack; `make bundle-cache` repacks it after a fresh run.
    `rm -rf .cache/llm` + a key re-runs against the live free APIs. Cost: editing a
    prompt or `config.yaml` silently misses the cache (documented in README).

16. **Subsample cap = 6000 threads.** Enough for stable retrieval + metrics,
    small enough for the time budget. The assignment explicitly encourages
    subsampling. Larger caps didn't move the headline in spot checks.
