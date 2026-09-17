# The AI-powered search question

This is the sibling of the set-piece. Nobody has reported Siwen asking it, so
treat the wording below as constructed rather than verbatim — but it is the
question her round turns into the moment you say the word "LLM" out loud, and
every org running a 10M-product search stack in 2026 is being asked some
version of it internally.

> **Take the same search stack and make it AI-powered.** Same catalogue and
> traffic as before — 10M products, 200 QPS, 100 ms end-to-end p99, six months
> of logs — but now you have an LLM budget. Where does the LLM go: query
> understanding, retrieval, ranking, relevance judgement, or the answer
> surface? What do you refuse to let it touch, and how do you prove any of it
> worked?

## What is actually being tested

Not whether you can name a model. Four things:

**Can you place an LLM by latency tier rather than by enthusiasm.** The single
discriminating question in this round is *where the call happens*. Offline,
near-line, online-from-cache, and online-blocking are four different systems
with four different cost models, and most candidates collapse them into one.

**Do you know that the highest-leverage LLM use in search is not in the serving
path at all.** It is the catalogue and the judgement set. Both are offline,
both are unbounded in latency, and both move tail recall more than any reranker
will.

**Do you know what not to do.** Generative retrieval over a 10M-item catalogue
that changes daily, a listwise LLM reranker inside a 100 ms budget, an answer
surface on a transactional query — each of these is a plausible-sounding answer
that a senior person rules out in one sentence with a number attached.

**Can you prove it worked.** "The LLM judge says relevance is up 8%" is not a
result, and knowing why is most of the difference between an L5 and an L6
answer here.

!trap The phrase "AI-powered search" tempts you into putting the model in the serving path, because that is where it is visible. The best answers spend the first twenty minutes offline -- on the catalogue and on labels -- and only then ask what is left that has to happen at query time.

## The first five minutes: what to ask

1. **"Is the 100 ms budget unchanged?"** If yes, then no LLM call happens in the
   blocking path except as a cache lookup, and I will design around that.
   *Assume unchanged unless told otherwise, and say you are assuming it.*
2. **"What is the cost envelope — per query, or per month?"** At 200 QPS a
   per-query call is ~520M calls a year. A per-item offline pass is 10M calls,
   once. Those are different businesses and the answer changes with it.
3. **"Is the deliverable better products, or an answer?"** A ranked grid and a
   generated answer are different surfaces with different metrics, and in
   commerce the answer surface can cannibalise the grid.
4. **"Do we own the catalogue text, and can we write back to it?"** This decides
   whether the biggest lever is available at all.

If you get a fifth: ask what languages. Multilingual query understanding is
where LLMs are most clearly better than the pipeline they replace, and it is
the easiest win to under-claim.

## Numbers you derive from the prompt, out loud

Say the token math before anyone asks for it. Rates vary by model and vendor and
change often, so carry the *formula* and one worked example at a stated
assumption rather than a memorised price.

| Workload | Derivation | Volume |
|---|---|---|
| Catalogue enrichment, one pass | 10M items, ~600 in + 150 out | 6B in, 1.5B out |
| Re-enrichment, steady state | ~1% of catalogue changes daily | 60M in, 15M out per day |
| Judgement set, 50k pairs | 50k, ~400 in + 80 out | 20M in, 4M out |
| Distillation labels, 5M pairs | 5M, ~400 in + 20 out | 2B in, 100M out |
| Query rewrite, every query | 200 QPS, ~150 in + 60 out | 2.6B in per 6 months |
| Query rewrite, cached head+torso | Precompute top 200k queries once | 30M in, 12M out |

The last two rows are the whole argument. Rewriting every query costs roughly a
thousand times what precomputing the queries that matter costs, for a gain that
lands almost entirely on the tail — which is exactly the part a cache misses.
The resolution is not "cache or not", it is a tiered policy, below.

!num A per-query LLM call at 200 QPS is ~17M calls a day. A one-time catalogue pass is 10M calls, ever. If the per-query call is not measurably better than a distilled student model, it does not ship.

## The sixty minutes

| Minutes | What | Why |
|---|---|---|
| 0-5 | Clarify, state the latency tier framing | Establishes the organising idea immediately |
| 5-12 | The tier table and the system diagram | Everything later is "which tier is this in" |
| 12-25 | **Offline: catalogue enrichment and the judgement flywheel** | The two biggest levers, and the ones most candidates skip |
| 25-35 | Query understanding, cached, with the fallback | Where the prompt's tail example actually gets fixed |
| 35-45 | **Retrieval and reranking: teacher offline, student online** | The distillation answer is the whole point |
| 45-52 | The answer surface, and when not to build it | Product judgement, not modelling |
| 52-60 | Evaluation, cost, what you would build first | Staff signal |

# The system: five tiers, and everything goes in one of them

Draw this table before you draw any boxes. It is the answer to the question,
and every later section is an instance of it.

| Tier | Budget | What belongs here | What it costs |
|---|---|---|---|
| Offline batch | Hours to days | Catalogue enrichment, synthetic queries, judgement sets, distillation labels | Pipelines and versioning |
| Near-line | Seconds to minutes | New-item enrichment, warming the rewrite cache for emerging queries | A queue and a staleness budget |
| Online, cached | Under 1 ms | Rewrites and expansions for head and torso queries, looked up | Cache size, and a fallback |
| Online, blocking | 10-30 ms | A distilled cross-encoder. Not an LLM API call | GPU or a quantised CPU model |
| Never | — | Ungrounded generation about price, stock, safety or sizing | Trust, and in commerce, liability |

@fig:funnel Figure 1. The funnel, unchanged. The point of this round is that almost nothing in the serving path becomes an LLM call. What changes is the quality of the inputs each box receives.

@decide Where does the LLM go first?
- chose: Offline -- catalogue enrichment and the judgement set
- over: Online, as a query rewriter or a reranker, which is what the phrase "AI-powered search" tempts you into
- why: both are unbounded in latency, both are paid once per item rather than once per query, and both move tail recall -- which is the stated problem -- while a reranker only reorders a candidate set that was already missing the item
- cost: enrichment pipelines, a re-enrichment story every time the prompt or the model version changes, and a human audit sample you have to keep funding
- drop it if: the catalogue text is already clean and structured, in which case the labels are the only offline lever left and you skip straight to the judgement flywheel
@end

Three things to say while you draw it.

**The serving path barely changes.** Query understanding, three retrieval arms,
fusion, L1, L2, slate policy. What changes is that every box gets better inputs:
cleaner item text, better labels, better teachers.

**Recall is still set at the top and never recovered.** An LLM reranker cannot
retrieve an item the retriever did not return. If the tail query in the prompt
fails because the word "lightweight" appears nowhere in the item's text, no
amount of reranking fixes it — writing that attribute into the item does.

**Every online LLM call must beat its own distilled student.** That is the bar,
and stating it early is what stops the rest of the answer becoming a wish list.

# Offline, part one: the catalogue is the biggest lever

Go back to the prompt's own example: *"durable lightweight winter boots for
toddlers."* Four constraints. Ask why the current system fails on it, and the
honest answer is usually not a modelling failure at all. The item that should
win is a toddler snow boot whose title is *"Kids Winter Snow Boots Waterproof
Non-Slip 2024 New"* — it is 340 grams, it is durable, and neither fact appears
anywhere a retrieval system can see. The seller wrote a title for a keyword
search engine in 2016.

**So extract the attributes.** For each item, run one pass over title,
description, structured seller fields, images and the review text, and emit a
constrained JSON record against a fixed, closed attribute vocabulary: age band,
weight class, material, water resistance, sole type, closure, and so on.

Three places the output lands, and all three matter:

- **Fielded lexical index.** New BM25F fields with their own weights. This is
  what fixes "toddlers" as a facet rather than a token.
- **Item text for the dense encoder.** The two-tower item tower now reads a
  normalised, attribute-bearing document instead of seller keyword soup, which
  is a larger quality change to the embedding than any encoder swap.
- **Facets and filters in the product surface.** Free, and often the thing the
  PM notices first.

**Then generate queries for the items nobody searches for.** Synthetic query
generation — doc2query, in the old naming — asks the model for the 10 to 20
queries a shopper would plausibly type to find this item, and indexes them as an
extra field. This is aimed squarely at the zero-engagement tail: the items that
have no clicks, therefore no behavioural signal, therefore no presence in any
query-to-item table, and, as the set-piece document notes, are never even
sampled as negatives. Synthetic queries give the lexical arm something to match
and the dense arm something to train on.

@decide How far do you trust extracted attributes?
- chose: Closed vocabulary, confidence-thresholded, verified against the seller's structured fields where they exist, with a human-audited sample every batch
- over: Free-text extraction written straight into the index
- why: in commerce a hallucinated attribute is not a relevance bug -- "waterproof", "flame retardant", "BPA free" and a size chart are claims, and a wrong one is a return, a complaint, and sometimes a regulator
- cost: coverage is lower than free extraction, and the vocabulary needs maintenance
- fallback: when confidence is below threshold, emit nothing rather than a guess, and let the item keep its old behaviour
@end

!say The cheapest way to improve tail retrieval on a marketplace is usually not a better retriever. It is writing the attributes the shopper searched for into the item, because on a third-party catalogue they are genuinely absent, not merely hard to match.

**How you measure it, so it is not vibes:** attribute coverage (share of items
with a non-null value per attribute), precision on a human-audited sample per
attribute, and then the only one that counts — recall on tail queries, measured
by frequency decile, before and after. If coverage goes up and tail recall does
not move, the attributes you extracted are not the ones shoppers search with,
and the fix is to mine the vocabulary from the query logs rather than from the
taxonomy.

# Offline, part two: the judgement flywheel

The bottleneck in search quality is almost never model capacity. It is labels.
Human relevance judgement costs dollars per pair and takes days; a search team
with 5,000 judged pairs makes decisions with error bars wider than every change
it ships. This is the part of "AI-powered search" that actually compounds.

**The mechanism.** Graded relevance rubric, three or four points, written down
and versioned. The model sees the query, the item's enriched text, and the
rubric, and returns a grade with a one-line justification. You now have 500,000
judged pairs for what 5,000 used to cost.

**Calibrate it before you believe it.** Keep a human gold set — a few thousand
pairs, stratified by query frequency decile and by grade — that the judge never
sees during prompt development. Report the judge's agreement with human on that
set against the right ceiling, which is *human-human agreement*, not perfect.
On graded commerce relevance, human-human Cohen's kappa in the 0.6 to 0.7 range
is normal, and a judge at 0.6 is therefore at the ceiling, not failing.

**What the judgements buy you, in order of value:**

- **Training data for the cross-encoder**, which is the model that actually
  serves. This is the distillation path, below.
- **False-negative filtering for the retriever.** The set-piece answer calls
  this out: a mined hard negative that is actually relevant teaches the model
  the opposite of what you want. A judge over mined negatives removes them at
  a scale a human panel never could.
- **A regression set you can run on every change**, segmented by query
  frequency, which is what lets you see that a launch won the tail and lost the
  head.

**Where the judge is not allowed.** Two places, and naming them unprompted is
worth a lot:

- **Shipping decisions.** Offline judged relevance is a leading indicator, not
  the outcome. Interleaving and A/B decide, because the judge cannot see
  whether the customer bought anything.
- **Any comparison where the judge shares a failure mode with the thing judged.**
  If an LLM rewrote the query and an LLM judges whether the results match the
  rewritten intent, the loop is closed and the number means nothing. Judge
  against the *original* user query, always.

!trap "Our LLM judge says relevance improved 8%" is not a result. The questions are: which prompt and model version, what was its agreement with the human gold set, was the gold set refreshed after the change, and what did interleaving say. A metric whose definition includes a model version is a metric that silently drifts when someone upgrades the model.

**Version the judge as part of the metric.** Pin model and prompt; a judge
upgrade is a metric redefinition and needs a re-baseline of every historical
number, exactly like changing the NDCG cutoff. Teams that skip this discover
six months of trend line that reflects model releases.

# Online: query understanding, cached, with an honest fallback

Now the serving path. The query in the prompt is ambiguous and long-tail, which
is precisely the population a cache cannot help and precisely where an LLM helps
most. That tension is the content of this section.

**The tiered policy.**

| Segment | Share of traffic | Policy | Latency |
|---|---|---|---|
| Head, top ~10k queries | Roughly half | Precomputed rewrite, reviewed, served from cache | Lookup |
| Torso, next ~200k | A further quarter or so | Precomputed offline, refreshed weekly | Lookup |
| Tail, everything else | The remainder, and where the money is | Deterministic pipeline now; LLM enrichment queued near-line | No LLM in path |
| Emerging, rising fast | Small but urgent | Near-line: detected by frequency delta, enriched within minutes | No LLM in path |

The tail row is the one to defend. You do *not* make a blocking LLM call for a
tail query. You serve it with the existing pipeline — spell correction,
segmentation, the attribute gazetteer that the catalogue enrichment just made
much better — and you enqueue the query. The second person to type it, minutes
later, gets the enriched version. Tail queries are unique per session but not
unique per week, and the repeat rate is high enough that near-line enrichment
captures most of the value at none of the latency cost.

**What the model actually emits.** Constrained JSON against a schema, not prose:
normalised tokens, extracted attributes mapped to the closed vocabulary, an
intent label, a category distribution with confidence. A validator parses it; if
it does not parse, you fall back, and the fallback is the current pipeline, not
an error.

@decide Does the rewrite replace the query or join it?
- chose: The rewrite becomes an additional retrieval arm, fused
- over: Replacing the user's query with the rewrite
- why: a rewrite that drops a constraint is invisible and catastrophic -- "boots for toddlers" retrieved for "winter boots for toddlers size 8" looks fine in a demo and is a wrong answer. Fusing keeps the original arm's recall as a floor
- cost: an extra arm to serve and a fusion layer that has to be learned, not constant
- drop it if: measurement shows the rewrite arm's unique contribution is negligible, which does happen on head queries where the original is already precise
@end

**Guard the head.** Almost every LLM query-understanding launch wins the tail
and loses the head, because head queries are short, precise, and already
handled, so any rewrite is at best neutral and at worst adds a constraint the
shopper did not ask for. Make "no regression on the top 10k queries" a launch
gate, and note that an aggregate metric can sit flat while both segments moved a
lot in opposite directions.

# Retrieval: what "AI-powered" does and does not change

**Hybrid stays.** Three arms — lexical, dense, behavioural — is still the answer,
and an LLM does not retire BM25. What changes is that the lexical arm now
indexes extracted attributes and synthetic queries, and the dense arm now trains
on enriched item text and judge-filtered negatives. Both arms got better without
either becoming an LLM.

**Learned sparse is the most underrated option in this round.** SPLADE-style
models expand a query and a document into weighted term distributions over the
vocabulary and serve out of an ordinary inverted index. You get semantic
matching with lexical infrastructure, exact-match behaviour preserved, and — the
part that matters in an interview — full interpretability, because you can read
which expanded terms fired. The cost is real and you should name it: posting
lists get much denser, so index size and query latency both rise, which is why
the models carry a FLOPS regulariser to keep expansion sparse. On a catalogue
with heavy identifier traffic (part numbers, model codes, sizes), this is often
the highest ratio of quality gained to infrastructure changed.

**Generative retrieval and semantic IDs: know it, and decline it here.** The
framing is seductive — quantise each item into a short code, then have a
sequence model emit the code directly, so retrieval becomes decoding and the ANN
index disappears. Three reasons it is not the first thing to ship on this
system:

- **New items.** 10M products on a marketplace with continuous listings. A new
  item's identifier has no presence in the decoder until you retrain or graft it
  in; the cold-start story is exactly the population you most need to serve.
- **Cost of decoding.** Beam search per query against a 100 ms budget is a much
  harder sell than one HNSW traversal.
- **Evaluation.** Recall at a fixed latency against a strong hybrid baseline, at
  this corpus size, is not a comparison the published results have settled.

Say where it *does* earn its place — recommendation over a stable corpus, and as
a complementary head-query arm — and move on. Knowing the boundary is the signal.

**Multimodal is a real fourth arm, and it is optional.** Product images carry
attributes the text never will, and commerce tail queries are full of visual
language. Add a vision-embedding arm only after you have measured the dense
arm's unique contribution, and hold it to the same bar: clicked items that only
this arm retrieved.

# Reranking: the teacher is offline, the student serves

This is where the round is won, and the answer is one sentence: **an LLM makes
the labels, a cross-encoder makes the latency.**

The three forms of LLM reranking, and what each costs:

| Form | What it does | Quality | Cost per query |
|---|---|---|---|
| Pointwise | Score each item for relevance independently | Good, calibratable | 50 calls, parallelisable |
| Pairwise | Compare two at a time | Better, quadratic | Impractical online |
| Listwise permutation | Emit a reordering of a window, sliding over the list | Best | Hundreds of ms to seconds |

Listwise is the strongest and it has two properties people forget. It is
**position-biased inside its own prompt** — the model over-favours items placed
early in the list it is shown — which is mitigated by permuting and aggregating,
and that multiplies an already large cost. And it is **non-deterministic**,
which makes an A/B harder to reason about and makes caching partial.

Against a 100 ms end-to-end budget where the cross-encoder gets roughly 10 to 15
ms for 50 candidates, a listwise LLM pass is one to two orders of magnitude out.
It does not go in the blocking path. It goes in the teacher seat:

1. Teacher scores a large sample of (query, candidate) pairs offline, drawn from
   real logged candidate sets so the distribution matches serving.
2. Distil into the **cross-encoder** — train on the teacher's scores or its
   pairwise preferences rather than on clicks, which removes position bias from
   the label at the source.
3. Distil again into the **bi-encoder**, which is how teacher quality reaches
   recall rather than only order.
4. Serve the students. Refresh the teacher's labels quarterly, or whenever the
   candidate distribution shifts — which, per the seam section, is every time
   retrieval changes.

@decide Any LLM call at all in the blocking path?
- chose: No. Students only, with an asynchronous refresh for a narrow slice
- over: A listwise LLM reranker on the top 20
- why: 100 ms p99 end to end, and the distilled cross-encoder recovers most of the teacher's ordering at a fiftieth of the latency
- cost: a distillation pipeline, and a ceiling set by the teacher rather than by the student
- fallback: for low-confidence tail queries, serve the student immediately and stream a re-ordered grid a second later if the async teacher disagrees -- but only if the surface can tolerate re-ordering, which on mobile it usually cannot
@end

!push "What if I give you 500 ms?" Then the listwise reranker becomes arguable for the tail only -- route by L1 confidence, cap the window at 20, cache the result keyed on the normalised query so the second occurrence is free, and keep the student as the fallback on timeout. I would still spend the first 500 ms of budget on retrieval breadth rather than reordering, because reordering cannot recover recall.

# The answer surface: RAG over your own catalogue

Product judgement, not modelling, and a senior answer starts by refusing most of
the surface area.

**When it helps.** Informational and comparison intent — *"which toddler winter
boots are warmest?"*, *"is this jacket machine washable?"* — where the shopper is
doing research the product grid answers badly. Grounded in the retrieved items
and their reviews, it is genuinely better than ten blue products.

**When it hurts.** Navigational and transactional intent, which on a commerce
search bar is most of the traffic. Someone typing a brand and a model number
wants the item, and putting a paragraph above it costs them a scroll.

**How it plugs in.** It consumes the funnel's output; it is a *rendering* of the
retrieved set, not a second retrieval stack. Every claim cites the item or the
review it came from. When the retrieved set does not support an answer, the
surface says so and shows products instead — a refusal path that exists and is
measured is the difference between a demo and a system.

**The commerce-specific constraints**, which are the part that shows you have
shipped one:

- **Never generate price, stock or delivery claims.** They change between
  generation and impression, and a cached answer asserting a price that no
  longer holds is a consumer-protection problem, not a quality problem.
- **Never generate safety, medical, or compliance claims** — flame retardancy,
  allergen content, age safety ratings. Quote the seller's structured field or
  say nothing.
- **Never block the grid on the answer.** Render products at 100 ms and stream
  the answer alongside. The answer's latency budget is a rendering concern, not
  a search concern.

**Measure cannibalisation, not just satisfaction.** The metric set is clicks,
add-to-cart, purchase, *and* reformulation rate. An answer that satisfies the
shopper without a purchase is a loss for this business unless it demonstrably
reduces returns — which is a real and defensible win, and worth saying, because
a well-grounded answer about sizing is one of the few things that does.

# Evaluation: the part that makes it AI-powered rather than AI-flavoured

Three layers, each answering a different question, and the failure mode is using
one to answer another's question.

| Layer | Question it answers | Speed | What it cannot tell you |
|---|---|---|---|
| LLM-judged relevance | Did the results get more relevant | Hours | Whether anyone bought anything |
| Interleaving | Do users prefer B's ordering to A's | Days | Anything about a non-ranking change |
| A/B | Did the business metric move | Weeks | Anything at all on the tail, usually |

**Interleaving is the workhorse for ranking changes** and is dramatically more
sensitive than an A/B for the same traffic, because it compares two rankings
within a single user's result set rather than across populations. Use it for
every reordering change. It does not apply to changes in what is retrieved or
in the surface itself, which is the honest caveat.

**Segment everything by query frequency decile.** An aggregate is dominated by
the head, where none of this work helps. The entire thesis of an AI-powered
search programme is tail quality, and a flat aggregate is the expected result
even from a large win.

**The guardrail set that catches the actual failures:** zero-result rate by
segment, head-query regression, latency p99 including cache misses, and — the
one people forget — the share of queries where the rewrite changed the retrieved
set at all. If the rewrite fires on 3% of traffic, then a 20% improvement on
those queries is a 0.6% improvement, and you should have said so before the
readout rather than after.

# Cost engineering

The arithmetic that decides the design, stated as a formula so it survives
whatever this year's rates are. Let $R_{in}$ and $R_{out}$ be the price per
million input and output tokens.

$$
\text{cost}_{\text{online}} \;=\; \text{QPS}\times 86400 \times p(\text{miss})
\times \left(\frac{t_{in}}{10^6}R_{in} + \frac{t_{out}}{10^6}R_{out}\right)
$$

At 200 QPS, 150 input and 60 output tokens, a 100% miss rate is 17.3M calls a
day and 2.6B input tokens a day. Push $p(\text{miss})$ to 5% with the tiered
cache and it is 130M input tokens a day — the same design, a twentieth of the
bill. That factor, not the choice of model, is where the cost lives.

The levers, in the order they pay:

- **Cache by normalised query.** The largest single factor, by far.
- **Distil.** A student that recovers most of the teacher's quality at a
  fiftieth of the cost is a cost lever before it is a latency lever.
- **Route by segment.** Head queries do not need the big model; they barely need
  a model.
- **Batch the offline work**, and reuse a shared prompt prefix across calls so
  the rubric and the vocabulary are paid for once rather than per call.
- **Small model plus verifier.** Cheap model proposes the extraction, the
  expensive one adjudicates only the low-confidence tail.

# The seam, in this version

The set-piece document's seam is that the ranker is trained on the retriever's
output. This system has a second one, and it is worse because it is invisible.

**The judge's blind spots become the system's blind spots.** The LLM judges
relevance, which trains the cross-encoder, which filters the retriever's
negatives and supplies its distillation targets, which changes what is
retrieved, which changes what the judge is asked about next. It is a closed
loop with no external signal in it. Two things break the loop: a **human gold
set the judge never trains on and that is refreshed on a schedule**, and
**online metrics as the only shipping authority**. Say this unprompted.

**The rewrite moves the training distribution.** Your retriever was trained on
six months of logged raw queries. If the serving path now sends it rewritten,
attribute-normalised queries, it is being evaluated off-distribution. Either
train on rewritten queries — which means rewriting the historical log, offline,
which you can afford — or keep the raw query as an arm so the retriever still
sees what it knows. This is the same class of bug as the ranker's candidate
shift, and almost nobody raises it.

# The follow-up tree

**"Why not just use a big model for everything and cache aggressively?"**
Because the cache hit rate on the segment that needs help is low by
construction. Head queries cache beautifully and do not need the model; tail
queries need the model and do not cache. The tiered policy exists precisely
because those two facts point in opposite directions.

**"How do you know the extracted attributes are right?"** Human-audited sample
per attribute per batch, with an accuracy bar that is higher for claim-like
attributes (waterproof, flame retardant, age rating) than for descriptive ones
(colour family, closure type). Below threshold, emit nothing. And cross-check
against the seller's structured fields where they exist — disagreement is a
signal about the seller as much as about the model.

**"Your judge agrees with humans only 65% of the time. Isn't that too low?"**
It depends entirely on human-human agreement on the same rubric, which on graded
commerce relevance is commonly in that range. If two trained humans agree 68% of
the time, a judge at 65% is at the ceiling. If they agree 90%, it is failing.
Measuring the ceiling first is the whole answer.

**"Would you fine-tune a model for this?"** For the judge, usually not — the
rubric changes and a prompt is cheaper to iterate. For the cross-encoder and the
bi-encoder, absolutely, and that is what distillation is. For query
understanding, fine-tune once the schema stabilises, mostly to cut cost rather
than to raise quality.

**"What breaks first when you ship all of this?"** The head. Every time. The
rewrite fires on a query that was already precise, adds a constraint, and drops
the item everyone clicks. Which is why the launch gate is a head-query
regression check and why the rewrite is an extra arm rather than a replacement.

**"How would you handle a new attribute the taxonomy doesn't have?"** Mine it
from the query logs rather than from the taxonomy: cluster unmatched query
tokens by co-occurrence with clicked items, surface the top clusters, and add
the vocabulary term. The taxonomy is a lagging indicator of what shoppers
search for, and on a marketplace it lags badly.

**"What if latency dropped to 50 ms?"** Nothing in the offline half changes,
which is the point. The cross-encoder budget is the first casualty: drop to
adaptive-depth reranking on the visible slots only, or quantise the student and
cut the candidate set from 50 to 25. The tiered cache and the enrichment are
untouched, and that robustness is the argument for the architecture.

**"Is there anything you would not use an LLM for here at all?"** Ranking. The
ranking objective is expected value over calibrated probabilities estimated from
biased feedback, and that is a job for models trained on your logs with
propensity corrections, not for a language model's judgement. An LLM can supply
a relevance *feature* and a relevance *gate*. It should not supply the score.

# What I would build first

**Q1 — the judgement flywheel and the measurement it unlocks.** Rubric, human
gold set, calibrated judge, 100k judged pairs stratified by query frequency.
Stand up per-decile recall reporting and interleaving. Nothing else on this list
is provable without it, and it is weeks of work rather than quarters.

**Q2 — catalogue enrichment.** Attribute extraction into a closed vocabulary,
fielded BM25F weights, enriched item text for the dense tower, synthetic queries
for the zero-engagement tail. Measure tail recall by decile. This is the biggest
single expected move on the stated problem, and it needs Q1 to be visible.

**Q3 — the serving path, cheaply.** Precomputed rewrites for head and torso,
near-line enrichment for emerging queries, the rewrite as a fused arm with a
head-regression gate. In parallel, distil the teacher into the cross-encoder
using Q1's labels, and use the judge to strip false negatives from the
retriever's hard-negative mining.

**Q4 — the surface, if the data says so.** Learned sparse as a fourth arm if
identifier traffic is hurting, and the answer surface *only* for the
informational query segment, gated behind a cannibalisation readout.

With one engineer for one month instead: the judge and the human gold set, and
the tail-recall-by-decile report. It costs almost nothing and it is what makes
every subsequent argument fundable.

# The whiteboard, in order

| Minute | Draw | Say while drawing |
|---|---|---|
| 5 | **The five tiers** | Every later answer is "which tier is this in" |
| 8 | **The funnel** (Fig 1) | Note how little of it becomes an LLM call |
| 14 | **The enrichment pipeline** | Item text in, closed-vocabulary JSON out, three destinations |
| 20 | **The judgement flywheel** | Judge, gold set, cross-encoder, negatives -- and where the loop closes |
| 30 | **The tiered query policy** | Head cached, tail deferred to near-line, fallback always the old pipeline |
| 38 | **Teacher and student** | The one sentence that answers the reranking question |
| 50 | **The three evaluation layers** | And which question each one cannot answer |

!num 10M items x ~750 tokens is one 7.5B-token offline pass, paid once. 200 QPS is 17M calls a day if you do it per query. Head+torso is ~200k distinct queries covering most traffic. Cross-encoder over 50 candidates is 10-15 ms; a listwise LLM pass over the same 50 is hundreds of ms. Human-human kappa on graded commerce relevance is ~0.6-0.7, and that is your judge's ceiling.

# Ten things that would lose this round

**On placement**

- Putting an LLM call in the blocking path without doing the latency arithmetic.
- Treating "cache it" as a complete answer when the segment that needs the model
  is the segment that does not cache.
- Never mentioning the catalogue, which is where the prompt's own example fails.

**On evidence**

- Quoting an LLM judge's score as a result, with no human gold set and no
  agreement number.
- Not knowing that human-human agreement is the ceiling you measure against.
- Letting the judge evaluate results for a query the same family of model
  rewrote.

**On judgement**

- Proposing generative retrieval for a 10M-item catalogue with daily listings
  and having no answer for new items.
- Building an answer surface over transactional queries and calling the
  resulting drop in clicks a success.
- Generating price or stock into a cached answer.

**On both rounds**

- Forgetting that reranking cannot recover recall, so no amount of LLM at the
  bottom of the funnel fixes a tail miss at the top.
