# The question Siwen asks

This is the ML depth round. A recruiter read out the scorecard from the same
question asked to another candidate, so treat the wording below as close to
verbatim.

> **Design a retrieval and ranking system for an e-commerce mobile app search
> bar.** The goal is to retrieve highly relevant products for ambiguous,
> long-tail queries — for example *"durable lightweight winter boots for
> toddlers"* — while ranking the results to optimise CVR.
>
> Assume a system is already running and has collected logs for six months.
> **10M products. 200 QPS. 100 ms latency.**

## What is actually being tested

The question has two halves and you own both of them. Retrieval and ranking are
named in the prompt, they have different objectives, and a Staff answer holds
the whole funnel — including the seam between them, which is where most of the
interesting failures live.

The scorecard from the other candidate is still worth having, but for a narrower
reason than it first appears. It was written about someone else, and it is not a
prediction about you. What it does tell you is **the vocabulary Siwen probes
with on the retrieval side** — "contrastive learning, embedding models, or
softmax" is her list, not a generic one — which is a genuine signal about where
she considers the bar to be when she digs. It is a reason to be fluent there.
It is not a reason to under-weight ranking.

So: plan for roughly equal time on both, and let her steer. The two halves fail
differently and you should be able to say how.

**Retrieval** is judged on recall, is trained contrastively over sampled
negatives, and is where candidates most often turn out to be thin — most people
have consumed an embedding model without ever having trained one.

**Ranking** is judged on order, is trained on logged feedback that is biased by
the very system that produced it, and is where *this particular prompt* hides a
trap: "optimise CVR" has a specific pathology, and stepping around it carefully
is worth as much as anything you will say about negatives.

The seam between them is its own topic and almost nobody raises it unprompted:
the ranker is trained on whatever candidate distribution the retriever produced,
so changing retrieval silently invalidates the ranker. There is a section on
this at the end, and bringing it up yourself is the clearest signal in the whole
answer that you have shipped one of these.

## The first five minutes: what to ask

Ask four questions, no more. Each one should visibly change your design, and you
should say what it changes — asking a clarifying question and then ignoring the
answer is worse than not asking.

1. **"Is 100 ms the end-to-end server-side p99, or the budget for retrieval
   alone?"** If it is the whole funnel, cross-encoder reranking is tight and I
   will size it carefully. If it is retrieval only, I have room. *Assume
   end-to-end p99 unless told otherwise, and say you are assuming it.*
2. **"Is 200 QPS peak or average?"** It sets replica count and, more
   importantly, tells me how much training data six months buys. At 200 QPS
   average that is roughly 3 billion searches; at 200 peak with a typical
   diurnal curve, closer to one billion. Either is plenty for a retriever.
3. **"CVR of what, over what window?"** Conversion per search session, per
   click, or per impression? Same-session or 7-day attributed? This is the
   single most consequential clarification in the question and I will come
   back to it.
4. **"Is this a marketplace with third-party sellers, and do we control the
   product text?"** It decides whether catalogue quality is a modelling problem
   or a data problem, and in a marketplace it is usually the biggest single
   lever on tail retrieval.

If you get one more, ask whether ads or sponsored placement blend into the same
result set, because that changes the ranking objective from CVR to expected
value and you do not want to discover that at minute 50.

## Numbers you derive from the prompt, out loud

Doing this in the first five minutes is worth a lot — it shows the constraints
are real to you rather than decorative.

| From | Derivation | Number |
|---|---|---|
| 10M products, 256-d fp32 | 10e6 x 256 x 4 | 10.2 GB of raw vectors |
| Same, int8 scalar-quantised | 10e6 x 256 x 1 | 2.6 GB — fits in RAM on one box |
| HNSW graph, M = 32 | 10e6 x 2M x 4 bytes | ~2.6 GB of graph on top |
| 200 QPS at 100 ms | Little's law: 200 x 0.1 | 20 requests in flight |
| 6 months at 200 QPS avg | 200 x 86400 x 180 | ~3.1B searches |
| Clicks at ~40% of searches | | ~1.2B click events |
| Purchases at ~2% of searches | | ~60M purchase events |

!say This is not a scale problem, it is a quality problem. Ten million products and two hundred queries a second fit comfortably on a handful of machines. What is hard is recall on ambiguous tail queries and an honest CVR objective.

10M products and 200 QPS fit comfortably on a handful of machines.
Nothing here forces a distributed index or exotic infrastructure, and a
candidate who spends twenty minutes on sharding has misread the question. What
is hard is recall on ambiguous tail queries and an honest CVR objective.

## The sixty minutes

Her previous round was 15 minutes resume, 60 design, 15 Q&A. Budget the design
block like this, and say the plan out loud at the start — it buys you permission
to defer things.

| Minutes | What | Why |
|---|---|---|
| 0–5 | Clarify, state assumptions, derive the numbers | Establishes the constraints are real |
| 5–10 | Whole-system diagram and the latency budget | Gives her a map; everything later hangs off it |
| 10–14 | Query understanding for the tail example | Shows you understand why the query is hard |
| 14–32 | **Retrieval, and how you train it** | Objective, negatives, geometry, evaluation |
| 32–50 | **Ranking: the objective, the label bias, the model** | Equal weight. The CVR trap lives here. |
| 50–55 | The seam: how the two stages break each other | The part almost nobody raises unprompted |
| 55–60 | Evaluation, experimentation, what you would build first | Staff signal: sequencing under constraint |

Two blocks of eighteen minutes, not one of twenty-five. If she interrupts and
steers, follow her — but know where you were, and say "let me come back to
negatives" rather than losing the thread. If she spends the whole hour on one
half, that is her choice and you should have the depth to fill it; the budget
above is what to do when she leaves it to you.

# The system

Draw this first and leave it on the board. Everything afterwards is a zoom into
one box, and having the map up means you can say "I'm here" instead of
re-explaining context.

@fig:funnel Figure 1. The funnel. Each stage sees roughly ten times fewer items and costs ten times more per item, which is why every stage takes about the same wall-clock.

@decide How many retrieval arms?
- chose: Lexical + dense + behavioural, fused
- over: Dense only, which is what the prompt's wording tempts you into
- why: they fail on disjoint query populations. Dense dies on part numbers and rare identifiers; lexical dies on the paraphrase in the prompt; behavioural is empty on the tail and unbeatable on the head.
- cost: three indexes to keep in sync, and a fusion layer that has to be learned rather than constant
- drop it if: the dense arm's unique contribution -- clicked items only it retrieved -- turns out to be small. Measure, do not assume.
@end

Three things to say while you draw it.

**The three retrieval arms exist because they fail on disjoint query
populations.** Lexical nails exact tokens and part numbers and dies on
paraphrase. Dense nails paraphrase and intent and dies on rare identifiers.
Behavioural — a query-to-item table mined from six months of logs — beats both
on head queries and has nothing at all for the tail. The tail query in the
prompt is precisely the case where only the dense arm helps, which is why the
question is worded the way it is.

**Recall is set here and can never be recovered.** Everything downstream only
reorders. So retrieval gets measured on recall, ranking on order, and confusing
the two produces months of wasted experiments.

**L1 exists to protect L2's budget.** A cheap model over 1000 candidates lets
the expensive model see 150. Without it you either score 1000 with something
cheap, or blow the budget.

## The latency budget, and three ways to spend it

I am assuming 100 ms is end-to-end server-side p99, which is the harder reading.
Fan-out costs are real and candidates routinely forget them: serialisation,
merge, and network between services will eat 15–25 ms before any model runs.

**Option A — all-CPU, no cross-encoder.**

@fig:latency Figure 2. Three ways to spend 100 ms. A already leaves only 12 ms of headroom, which is why the cross-encoder does not fit in the blocking path at all.

Twelve milliseconds of headroom, and everything runs on commodity CPU. This is
what I would ship first.

**Option B — put a cross-encoder reranker over the top 50.** A 6-layer
MiniLM-class model at 128 tokens, batch 50, runs in roughly 10–15 ms on a
T4-class GPU. That is ~103 ms and it is over budget before you have accounted
for anything going wrong. And the way it goes wrong is specific: under load, GPU
batch formation and admission control dominate, and the tail of a GPU service is
far worse behaved than its median — so the number you actually measure at p99
will be worse than 103.

**Option C — progressive ranking, which is the answer I would defend.**

The tempting move is to make the expensive path rare: route tail queries through
the cross-encoder and head queries around it. **That does not work, and knowing
why is the point.** p99 *is* the tail. If 30% of queries take 103 ms, then p99 is
103 ms — making the expensive path rare only helps if "rare" means under one
percent, and by then it is not doing enough work to matter.

The same reasoning kills the other reflex. **Caching the head does not help your
p99 either**, for exactly the same reason: the cached queries are the fast ones,
and p99 is measured on the slow ones. A query cache is a cost and median-latency
win, not an SLO win, and saying so unprompted is a small but real signal.

What does work is taking the expensive model off the blocking path. On a mobile
grid the first screen is four to six items. Serve those from the cheap path
inside budget, then run the cross-encoder over the remaining candidates during
the user's dwell time and deliver the reordered result with the next scroll
page. The blocking p99 stays at 88 ms; the cross-encoder still improves the
session. It costs you a more complicated client contract and a reordering that
must be stable enough not to feel like the page is shuffling under the user.

There is still a role for adaptive depth, but on the retrieval side rather than
the ranking side: give tail queries a larger `ef_search` and deeper candidate
sets, since that cost sits inside the parallel arms where there is slack.

One policy to state without being asked: **never degrade by silently dropping a
retrieval arm on timeout.** That converts a latency problem into an invisible
quality problem that shows up weeks later as an unattributable relevance
regression. Degrade by reducing depth — lower `ef_search`, fewer L2 candidates —
and emit a metric every time you do, so the degradation is measurable.

# Query understanding on the tail

Take the query in the prompt apart on the board, because it demonstrates the
problem better than any description of it.

| Token | What it is | Binds to |
|---|---|---|
| durable | quality attribute, soft | no catalogue field at all |
| lightweight | physical attribute | sometimes in specs, usually only in free text |
| winter | season / use-case | insulation, waterproofing |
| **boots** | **head noun** | **category — the one hard constraint** |
| for toddlers | audience | age 1–3, so a size band — a real filter |

**Hard constraints:** category = boots, audience = toddler.
**Soft preferences:** durable, lightweight, winter-appropriate.
**What BM25 sees:** six tokens, each appearing in thousands of listings and
almost never co-occurring.


!trap Read the example query back to her before you answer. "Durable lightweight winter boots for toddlers" is not decoration -- it was chosen so that lexical retrieval fails, and demonstrating that you noticed is worth more than any architecture you propose in the next five minutes.

Now the point: **BM25 will return either nothing or garbage here.** Require all
six terms and you get near-zero results because no title contains them all.
Relax to OR-with-coordination and the scoring is dominated by whichever term is
rarest — probably "toddlers" — so you retrieve toddler clothing rather than
boots. This is the classic tail failure, and it is why the dense arm is not
optional.

What I would build:

**Head-noun and category prediction, with a confidence gate.** Getting
`category = boots` right is worth more than everything else in query
understanding combined. Predict it, and apply it as a retrieval filter *only*
above a high confidence threshold, with a fallback pass that re-runs unfiltered
if the filtered pass returns too few. Below threshold it goes in as a ranking
feature. The asymmetry matters: a wrong filter deletes results with no recovery,
a wrong feature just gets outvoted.

**Attribute extraction as sequence labelling**, trained on catalogue structured
data rather than hand annotation. You already know each product's category,
audience, colour, size. Reverse that into weak labels over the query side by
mining queries that led to purchases of products with a given attribute. Six
months of logs is exactly what this needs.

**Precompute the head, model the tail.** The top few thousand queries carry a
large share of sessions. Precompute their entire understanding offline, review
them, and serve from a lookup. That gives you editorial control where the
traffic is, costs nothing at serving time, and leaves the live model path for
the tail — which is also where you can afford deeper retrieval, per Option C.

**Where an LLM belongs here.** Offline, generating attribute annotations and
query rewrites for the tail, distilled into a small model or baked into a table.
Not in the 8 ms online path. Say this unprompted; it is a question she will
otherwise ask.

# Retrieval: half the answer

## The architecture, and the asymmetry that defines it

@fig:two-tower Figure 4. The two towers are deliberately different sizes. The item tower can be large precisely because nobody is waiting for it -- that is free capacity most candidates leave on the table by drawing two identical boxes.

State the asymmetry explicitly, because most candidates draw two identical
boxes. **The item tower runs offline over 10M products once a night; the query
tower runs inside your latency budget on every request.** So you can afford a
large item encoder and you cannot afford a large query encoder. That is free
capacity most people leave on the table — and it is the reason two-tower beats
a single shared encoder in production even though sharing weights looks
tidier.

The other consequence of the architecture: query and item never interact before
the dot product. That is what makes ANN possible, and it is also the ceiling on
quality. No amount of training makes a two-tower model reason about the
interaction between "lightweight" and a particular boot's spec sheet the way a
cross-encoder can. So the two-tower's job is recall, and you accept that.

## Where the positives come from

Six months of logs, and the choice of positive is a modelling decision, not a
data-loading detail. Three questions in order: which actions count, how much
each is worth, and what the resulting table looks like.

### The action ladder

| Action | Count, 6 months | Rate | What it is evidence of | Value $v(a)$ |
|---|---|---|---|---|
| Impression | ~62B | ~20 / search | the ranker's opinion, not the user's | baseline, 0 |
| Click | ~1.2B | 0.4 / search | interest **given the thumbnail and price** | 1.0 |
| Add to cart | ~180M | 0.06 / search | genuine consideration | 1.9 |
| Purchase | ~60M | 0.02 / search | intent satisfied | 2.6 |
| Purchase, returned | ~6M | 10% of purchases | relevant, but did not fit | 1.6 |

The naive choice is clicks, because there are 1.2 billion of them. The problem
is that a click in commerce is driven by the thumbnail and the price as much as
by relevance, so a click-trained retriever learns attractiveness. The naive
correction is to train on purchases only — but 60M events spread across 10M
products is far too sparse for the tail, and the tail is the question.

So: **train on clicks for coverage, and weight by the downstream action.**

### How the values are derived, not chosen

This is the part worth being precise about, because "a purchase is worth more"
is not an answer — *how much* more is the answer, and picking it by taste is
how you end up with a model nobody can debug.

The weight should be monotone in **the posterior probability that this item is
genuinely relevant to this query, given that this action happened.** That is
measurable. Take a few thousand query-item pairs, stratified by which action
they received, have humans judge relevance, and compute $P(\text{rel} \mid a)$
per action. Then set the value to the log-odds, with impression as the baseline:

$$
v(a) \;=\; \mathrm{logit}\,P(\text{rel} \mid a) \;-\; \mathrm{logit}\,P(\text{rel} \mid \text{impression})
$$

normalised so a click is 1.0. A few thousand judgements is a day of annotation
and it converts every weight in the pipeline from an opinion into a measurement.

!trap The calibrated values come out far flatter than intuition. Most people reach for a purchase being ten times a click; the likelihood ratio says about two and a half. The click already carries most of the relevance evidence — the purchase adds price, availability and delivery, which are ranking concerns, not retrieval ones. Say this out loud; it is the kind of thing that only comes from having measured it.

The returns row carries the same lesson and is worth volunteering. A returned
purchase is **still good evidence of relevance** — the customer searched, found
it, and bought it. The return usually means fit or quality, not the wrong
product. So returns barely move the retriever and matter enormously to the
ranker.

!say Returns are a ranking signal, not a retrieval signal. The item was relevant; it just did not fit. I would penalise them heavily in the expected-value objective and almost not at all in the retriever's positives.

### Scoring one pair

@fig:positives Figure 4b. The weight for one (query, item) positive. Counts times calibrated values, log-compressed so a pair with ten thousand clicks does not get ten thousand times the weight of one with a single click, then divided by examination propensity, then normalised so the dataset mean is 1.

$$
w(q,d) \;=\; \frac{\log\!\big(1 + \sum_a n_a(q,d)\, v(a)\big)}{\hat{\pi}\big(\overline{\text{rank}}, \text{surface}\big)}
$$

Three choices inside that formula, each of which she can push on.

**The log.** Without it, a pair with 10,000 clicks contributes 10,000 times the
gradient of a pair with one, and your retriever becomes a popularity model.
The log is the same saturation intuition as BM25's $k_1$, applied to training
weight instead of term frequency.

**The propensity divisor.** A click at rank 1 is weak evidence because rank 1 is
examined regardless; a click at rank 20 is strong evidence because the user had
to work for it. Dividing by $\hat{\pi}$ up-weights the deep clicks. **Clip it**
— $\hat{\pi} \geq 0.05$ or so — because a propensity estimate near zero produces
a weight that dominates the entire batch.

**The normalisation.** Scale weights to mean 1 across the dataset so the loss
magnitude, and therefore your learning rate, does not move when you change the
value table.

### Where the weight actually goes

Two mechanically different options, and the choice matters more than it looks.

@decide How do you apply the positive weight?
- chose: Sample positives with probability proportional to $w$, capped, and then train unweighted
- over: Keeping the batch uniform and multiplying each example's loss term by $w$
- why: with in-batch negatives every positive is simultaneously a negative for the other queries in the batch. Loss-weighting only scales its role as a positive, so a heavily weighted pair is a loud positive and a normal-volume negative -- an asymmetry nobody intends. Sampling changes both roles consistently.
- cost: sampling amplifies popularity, so the log compression and the cap are load-bearing, and you lose the ability to tune weights without rebuilding the sampler
- fallback: sample for the bulk of the signal and loss-weight only the returns adjustment, which is small
@end

### Session-level attribution for the tail

For a tail query with a handful of events, a single click is noise. But
reformulation chains are signal, and this is the cheapest tail data you have.

The rule I would write down, because "attribute the session" is not specific
enough to implement:

- A session is one user, events within a 30-minute gap.
- If query $q_1$ produced no click, the user reformulated to $q_2$, and then
  clicked or purchased item $d$ — attribute $d$ as a positive for **both**
  $q_1$ and $q_2$.
- Discount the inferred one: $q_1$ gets $0.5 \times w$, because the attribution
  is an inference and $q_2$ is the observation.
- Cap the chain at two reformulations. Beyond that the user has changed intent.
- Require the two queries to share a token or exceed a similarity threshold,
  or you will attribute "toddler boots" to "phone charger" in the same session.

On this corpus that typically recovers training pairs for a meaningful slice of
tail queries that otherwise have no positive at all — which is exactly the
population the prompt is about.

## The shape of the training data

Two tables, and **they are different shapes on purpose**.

### The event log — what you read from

```
search_event      search_id, user_id, ts, query_raw, query_norm,
                  surface, locale, ranker_version, retriever_version
impression_event  search_id, item_id, position(row,col), above_fold, ts
action_event      search_id, item_id, action, ts, order_id
```

`ranker_version` and `retriever_version` are not bookkeeping — they are what
makes intervention harvesting possible later, because they identify the natural
position randomisation your A/B tests already produced.

### The retriever's table — aggregated to the pair

```
retriever_pairs                                        ~180M rows
  query_norm          string
  item_id             int64
  n_impr, n_click, n_atc, n_purchase, n_return   int
  mean_position       float
  propensity          float        -- clipped
  weight              float        -- the formula above, normalised
  query_freq_bucket   enum(head, torso, tail)
  split               enum(train, val, test)
```

**Why aggregate.** The retriever has no user features — the query tower sees a
string and nothing else — so per-event rows add no information, and they add
popularity skew: a pair shown a million times would contribute a million
gradient steps for a signal that is fully captured by its counts. Aggregating
deduplicates it and costs nothing.

Filters worth naming: drop pairs with zero clicks, drop queries with fewer than
three distinct clicked items (no contrast to learn from), drop bot and scraper
traffic by session-rate heuristics, and drop pairs whose item is no longer in
the catalogue.

### The ranker's table — per impression

```
ranker_impressions                                     ~62B, sampled to ~2B
  search_id, item_id, position, surface, device, ts
  feature_vector      float32[~200]   -- LOGGED at serve time, not recomputed
  label_click, label_atc, label_purchase, label_return  bool
  propensity          float
  split               enum
```

The ranker *does* have user and context features, so it needs per-impression
rows, and it needs the negatives that never got clicked — which the retriever's
table has thrown away. The `feature_vector` is logged rather than recomputed,
for the skew reason in Figure 13.

!say The two tables are different shapes because the retriever has no user context. That is not an implementation detail — it is why the retriever can be trained on an aggregate a thousand times smaller than the ranker's.

### The batch that comes out

| Tensor | Shape | Notes |
|---|---|---|
| `query_ids` | `[4096, 16]` | queries are short; 16 tokens covers almost all |
| `pos_item_ids` | `[4096, 96]` | title + brand + attributes |
| `hard_neg_ids` | `[4096, 4, 96]` | 2 from BM25, 2 from the current ANN index |
| `uni_neg_ids` | `[4096, 2, 96]` | uniform over the 10M index |
| `weights` | `[4096]` | only if you loss-weight rather than sample |
| `log_q` | `[4096, 4102]` | sampling log-prob per candidate, for the logQ correction |

Candidates per query: 1 positive + 4095 in-batch + 4 mined + 2 uniform ≈ **4102**.

!num Item-tower forward passes per step: 4096 positives + 4096 x 6 explicit negatives = 28,672. In-batch negatives are free; every mined or uniform negative is a full encoder pass. That factor of seven is the real cost of the negative strategy, and it is the number to quote if she asks what it costs.

The mitigation, which is what ANCE does in practice: cache item embeddings from
the last index refresh and reuse them for mined negatives, recomputing only on
the refresh cadence. You trade a little staleness for most of the compute.

### The negative sources, side by side

| Source | Per positive | Cost per step | What it fixes | Fails if |
|---|---|---|---|---|
| **In-batch** | ~4095 | free — already encoded | general contrast, cheaply | used alone: popularity-biased, cannot reach the tail |
| **Mined, BM25** | 2 | 2 encoder passes | keyword match vs intent match | not filtered — BM25's top-k is full of false negatives |
| **Mined, ANN** | 2 | 2 passes + re-index | the model's *current* errors | not refreshed — they go stale once the model beats the miner |
| **Uniform** | 2 | 2 passes | the zero-engagement tail | nothing: they are cheap and they are the only source that covers it |

### Splits

**Split by time, never randomly.** The last 14 days are test, the 14 before are
validation, everything earlier is train. A random split leaks: the same
(query, item) pair appears on both sides, and you will serve forward in time, so
a random split measures a task you will never perform.

Hold out a second, **query-disjoint** split as well — whole tail queries the
model has never seen — because time-based splitting still lets a head query's
embedding be learned from the training period and evaluated in the test period.
That second split is the one that tells you about the tail.

## The objective

Retrieval is trained as a classification problem over the corpus: given the
query, pick the right item out of everything. The full softmax over 10M items is
intractable, so you sample.

$$
\mathcal{L} \;=\; -\log
\frac{\exp\!\big(s(q,d^{+})/\tau\big)}
     {\sum_{d \in \{d^{+}\}\cup N} \exp\!\big(s(q,d)/\tau\big)}
$$

where $s(q,d)=\langle f(q), g(d)\rangle$, $\tau$ is the temperature, and $N$ is
the sampled negative set for this query. It is a cross-entropy over a sampled
candidate set: push the positive up, push everything else down, with $\tau$
setting how sharply.

## Which loss, and why not the others

This is worth being properly fluent on, because "contrastive learning … or
softmax" is two thirds of what her scorecard named. The families differ in one
thing that explains everything else: **what a single gradient step is allowed to
see.**

@fig:losses Figure 5b. One gradient step, four ways. Regression needs an absolute target that click data does not provide. Pairwise sees one negative and a margin measured in a space whose scale the model controls. Sampled softmax sees many negatives and has no absolute target at all — only relative order. Listwise sees the whole list and weights each swap by what it does to the metric.

### The one-question rule

Ask what the output is **consumed as**, and the loss follows:

| The output is consumed as | Use | Because |
|---|---|---|
| An ordering over millions of items | **Sampled softmax / InfoNCE** | only relative order matters, and normalising over sampled negatives is the cheapest unbiased way to get it |
| An ordering over a short list where position matters | **Listwise, metric-weighted** (LambdaRank) | you can afford to look at the whole list, and NDCG's discount should drive the gradient |
| A probability you will multiply or threshold | **Pointwise BCE**, then calibrate | the expected-value product is meaningless unless the number is a probability |
| A yes/no decision at a threshold | **Contrastive pair / triplet** | you need an absolute decision boundary in the metric, not a ranking |
| A number with units | **Regression** (Huber over MSE, for outliers) | there is a real target and its magnitude is the answer |
| A teacher's ordering | **Margin-MSE or KL** | the ordering is the signal; the teacher's absolute scale is not |

### Why not MSE

Three reasons, and the first one ends the conversation.

**There is no target.** A click is not a similarity of 1.0. To use MSE you would
have to invent the label, and whatever you invented would be the thing the model
learned. Retrieval supervision is ordinal by nature: this item beat that one for
this query. MSE requires cardinal supervision you do not have.

**It optimises the wrong quantity.** Even granting a target, retrieval only ever
uses the *order* of the scores. MSE spends capacity matching magnitudes that are
never read, and the capacity comes out of the ordering you do care about.

**Nothing forces the positive to win.** MSE has no normalisation across
candidates. A model that outputs 0.3 for every pair has a respectable MSE
against a mostly-negative dataset and precisely zero retrieval utility. The
softmax's denominator is the entire point: the positive can only score well by
*beating the others*.

There is a fourth, quieter one: the gradient is dominated by the easy mass.
Almost every query-item pair in a 10M catalogue is trivially unrelated and
already predicted near zero, so most of the MSE gradient is spent confirming
things the model already knows.

### Why not triplet, and when it is actually right

Triplet is not a silly choice — it is the previous generation's answer, and it
still wins in one setting. But for retrieval the softmax dominates it on four
axes.

**One negative per step.** Triplet's gradient says "be further from this one
thing." The softmax's says "beat all of these at once," which is a far lower
variance estimate of the objective you actually serve.

**The margin is absolute in a space you control.** With unnormalised embeddings
the model can satisfy any margin by inflating norms and learning nothing about
the geometry. That is why triplet effectively forces L2 normalisation — and once
you are on the unit sphere, distances live in a fixed narrow range and the
margin becomes a very tight budget to tune. The softmax's temperature does the
same job *relatively*, so it is scale-free.

**The hinge switches off.** Once a triplet satisfies the margin, its loss is
exactly zero and it contributes no gradient. Late in training most of the batch
contributes nothing, which is why semi-hard mining exists and why it is fiddly.
The softmax is smooth: every negative always contributes something, weighted by
how close it came.

**Collapse.** With easy negatives, triplet can drive everything to a point and
report a happy loss curve. The normalisation makes that much harder to achieve
under a softmax.

!push "So triplet is just worse?" — No. Triplet and contrastive-pair losses give you a *calibrated distance* with a decision boundary; softmax gives you an ordering with no absolute meaning. If the product needs "are these two things the same?" as a yes/no at a threshold, you want the margin. That is verification, not retrieval.

And that case exists in this very system. **Catalogue deduplication** — deciding
whether two seller listings are the same physical product, which a marketplace
has to do constantly and which feeds the slate policy's dedup step — is a
verification problem with a threshold. I would train that with a contrastive
pair loss, not a softmax, and I would say so if she asks whether contrastive
losses have any place here.

### Where MSE is actually the right answer

Two places, both inside this design, and naming them is the cleanest way to show
you are not just reciting "softmax good, MSE bad."

**Distilling the cross-encoder into the bi-encoder.** The teacher produces real
scores, so regression is available. But regress the **margin**, not the raw
score — $s(q,d^{+}) - s(q,d^{-})$ for the student matched to the same difference
for the teacher. The teacher's absolute scale is arbitrary, so fitting it forces
the student to waste capacity on an offset that means nothing; the margin is
what carries the ordering, and it is invariant to that scale. That is Margin-MSE,
and it is the reason plain MSE distillation underperforms.

The alternative is KL over the softmax of the candidate set, which transfers the
whole distribution rather than pairwise gaps. Margin-MSE is simpler and robust;
KL carries more information when your candidate sets are consistent. I would
start with Margin-MSE.

**Training L1 to imitate L2.** L1's job is to not discard anything L2 would have
wanted, so it is a distillation problem, and regression onto L2's scores is a
reasonable objective — with the same caveat, that a ranking-aware distillation
beats plain MSE because only the top-k ordering matters.

### The whole system, loss by loss

If she asks you to put it together — and this is a good thing to volunteer —
every trained component in the design has a different loss, for a reason:

| Component | Loss | Why that one |
|---|---|---|
| Two-tower retriever | Sampled softmax + logQ correction | ordering over 10M, no calibration needed downstream |
| Cross-encoder teacher | Listwise CE on graded labels | short candidate sets, graded judgements available |
| Bi-encoder distillation | Margin-MSE, or KL over candidates | transfers ordering, invariant to teacher scale |
| L1 ranker | Ranking distillation from L2 | measured on recall of L2's top-k, not NDCG |
| pCTR, pCVR, pReturn | BCE, then post-hoc calibration | they get multiplied by price |
| pRel gate | Listwise, or distilled from the cross-encoder | a gate, not a probability |
| Slate re-ranking layer | Listwise over the final top-k | list context is the whole point |
| Catalogue dedup | Contrastive pair with a margin | verification at a threshold |
| Delivery-time feature | Huber regression | a real number with units, and outliers |

@decide Which loss for the retriever?
- chose: Sampled softmax / InfoNCE over the sampled candidate set, with the logQ correction
- over: Triplet with a margin, BPR/RankNet, and MSE regression
- why: retrieval consumes only the ordering, and the softmax is the only one of these that normalises over many candidates at once — so the gradient estimates "beat everything" rather than "beat this one thing", and it needs no absolute target, which click data cannot provide anyway
- cost: large batches to get enough negatives, a temperature to tune, and an output with no probabilistic meaning
- but: keep a contrastive pair loss for catalogue dedup, which is verification at a threshold rather than ranking, and Margin-MSE for distilling the cross-encoder
@end

Three things follow.

**The number of negatives is the difficulty knob.** More negatives means a harder
classification task and a better-conditioned gradient, which is why these models
train at batch sizes in the thousands with cross-device negative sharing.

**Temperature matters more than people expect.** With L2-normalised embeddings
the score lives in [-1, 1], so without a small temperature the softmax is nearly
uniform and there is almost no gradient. Typical values are 0.01–0.05. Push it
too low and the gradient concentrates entirely on the single hardest negative,
which is unstable and can collapse the representation. It can be learned — CLIP
learns log-temperature with a clamp — and I would start at 0.05, tune it, and
watch the positive-score distribution rather than only the loss.

## Negatives: where this model is actually won or lost

This is the sub-topic to be deepest on. It is named in the scorecard and it is
where a candidate with real experience separates from one who has read papers.

### In-batch negatives, and the bias they introduce

In a batch of N pairs, use the other N-1 items as negatives for each query. Free
— no extra forward passes, since you already encoded them.

The catch: **your batch is not a uniform sample of the corpus.** It is sampled
from the click stream, so an item appears as a negative in proportion to how
often it is clicked. Popular items are therefore pushed down constantly, and the
model learns to under-score exactly the items that convert best. In commerce
that is precisely backwards.

The fix is the **logQ correction**, also called sampling-bias correction:

$$
s'(q,d) \;=\; s(q,d) \;-\; \log p(d)
$$

$p(d)$ is the probability that item $d$ is drawn as a sampled negative, which is
essentially its frequency in the training stream. Estimate it online with a
streaming counter: track the average gap in steps between consecutive
occurrences of $d$, and take $p(d)\approx 1/\text{gap}$.

The intuition to say out loud: an item that shows up as a negative ten times as
often should be penalised a tenth as hard each time. Without it, the retriever
has a systematic anti-popularity bias. This is the correction from the YouTube
two-tower work, and naming it is worth doing.

@fig:negatives Figure 6b. Where each negative source can reach. In-batch negatives are drawn from the click stream, so they can only ever be items that appear as a positive somewhere in the batch. At 10M products with a long tail, that leaves most of the corpus untouched.

### The negatives that are never sampled at all

Here is the part most candidates miss, and it matters enormously at 10M
products. **In-batch negatives can only ever be items that appear as a positive
somewhere in the batch** — that is, items with engagement. In a 10M-product
catalogue with a long tail, a large fraction of items have essentially zero
clicks in six months. Those items are never negatives for anyone. The model has
therefore never been trained to push them down, their embeddings sit wherever
initialisation and the item-text encoder put them, and they can surface
spuriously in ANN results for unrelated queries.

@decide Where do negatives come from?
- chose: In-batch with the logQ correction, plus mined hard negatives refreshed from the current model, plus uniform draws from the full index
- over: In-batch only, which is the default in every tutorial
- why: in-batch alone is biased toward popular items and structurally cannot reach the zero-engagement tail; hard negatives teach the distinctions that matter; uniform draws are the only source that covers the whole corpus
- cost: a mining pipeline, a cross-encoder to filter false negatives, and a ratio to tune
@end

The fix is **mixed negative sampling**: in-batch negatives plus a set of
negatives drawn uniformly from the full item index. The uniform ones are easy
and contribute little gradient individually, but they are the only thing
covering the zero-engagement tail, and they are cheap.

### Hard negatives

Random and in-batch negatives are mostly trivially wrong — a laptop for a boots
query. The model saturates on them quickly and stops learning the distinctions
that matter. Hard negatives are items that look right and are not: toddler
*shoes* for a toddler *boots* query; adult winter boots; boots that are neither
lightweight nor durable.

Two sources, in order of sophistication:

- **BM25 top-k, not clicked.** Cheap, no model needed, and a good first pass.
- **The model's own ANN top-k, refreshed periodically** — this is ANCE. As the
  model improves, yesterday's hard negatives stop being hard, so you re-mine
  from the current index every epoch or two. Static hard negatives go stale and
  you silently stop learning from them.

A typical recipe: one positive, a handful of mined hard negatives, the whole
batch as in-batch negatives, plus uniform negatives. The ratio is worth tuning;
all-hard is a known failure mode.

### False negatives, and why they are worse than no negatives

A mined "hard negative" is often just an unclicked relevant item — the user
bought one pair of boots, not all five good ones. Training on it teaches the
model that a correct answer is wrong, which is actively destructive rather than
merely unhelpful.

**Denoise with a cross-encoder.** Score every mined negative with a cross-encoder
teacher and drop any that scores above the positive, or above a threshold. This
is the RocketQA recipe and it is usually worth more than any architecture change
you could make instead. If she asks what to do without a cross-encoder: drop
mined negatives that were clicked by anyone for a similar query, and drop those
whose category matches the query's predicted category exactly.

## Distillation: the highest-leverage step people skip

Train a cross-encoder on the same data. It is far more accurate because query
and item interact. Then use it as a teacher: score a large set of query-item
pairs and train the bi-encoder to match the teacher's *score distribution*
rather than the binary click label — KL on the softmax over the candidate set,
or a margin-MSE on score differences.

Why this is worth saying: the binary label carries one bit. The teacher's scores
carry the full ordering, including how much better the positive is than each
negative. That ordering information is exactly what a retriever needs and what
clicks cannot provide. In my experience this buys more than doubling the
embedding dimension or adding layers.

And it closes the loop nicely: you need the cross-encoder anyway for
false-negative filtering and for the progressive reranker, so you get three uses out
of training it once.

## Embedding geometry

Three decisions, each with a specific consequence.

**Normalise or not.** L2-normalise both towers and use cosine, and every item
lives on the unit sphere so magnitude cannot encode popularity. That is what you
want for a relevance retriever; popularity belongs in the ranker where it can be
traded off explicitly. Leaving embeddings unnormalised lets magnitude carry a
quality prior, which helps in pure recsys retrieval and hurts here.

@decide Normalise the embeddings?
- chose: L2-normalise both towers, cosine similarity, temperature in the loss
- over: Unnormalised dot product, where magnitude carries a popularity prior
- why: popularity belongs in the ranker where it can be traded off explicitly, not smuggled into the retrieval score where you cannot see it
- cost: you need the temperature, since cosine lives in [-1, 1] and the softmax is otherwise nearly flat
@end

**Whatever you choose, serve it the same way.** Training with unnormalised dot
product and then serving a cosine index silently discards the magnitude the
model learned to use. This is a real and common production bug and worth naming
as one.

**Dimension.** 256 for 10M items. Storage, ANN memory and query latency are all
roughly linear in dimension, and the quality gain past ~256 on a catalogue this
size is small. If she pushes: train with Matryoshka representation learning so
the first 64 or 128 dimensions are independently usable, then you can serve a
cheap 64-d first pass and rescore with the full 256 without training twice.

## Putting the training loop together

1. **Build pairs** from six months of logs — clicks, add-to-cart and purchases,
   weighted by action, position-debiased, session-attributed for tail queries.
2. **Mine negatives** — BM25 top-k not clicked, plus ANN top-k from the *current*
   model, refreshed every one to two epochs.
3. **Denoise** — score every mined negative with the cross-encoder teacher and
   drop anything scoring above the positive.
4. **Assemble the batch** — in-batch negatives with the logQ correction, plus the
   mined hard negatives, plus uniform draws from the full index.
5. **Train** — InfoNCE at $\tau\approx 0.05$, large batch, cross-device negative
   sharing, with KL distillation from the cross-encoder.
6. **Embed the corpus** — item tower over 10M products nightly, incremental for
   new and changed listings.
7. **Build the index** — HNSW over int8 vectors, rebuilt nightly and hot-swapped,
   with the model version pinned into the index metadata.


Two operational points that signal you have run one of these.

**The index and the model must be versioned together.** A query embedded by
model v2 against an index built by model v1 produces silent garbage — not an
error, just bad results. Pin the model version into the index metadata and
refuse to serve a mismatch.

**Refresh cadence is a product decision.** Nightly full rebuild plus an
incremental path for new listings, because in commerce a product that cannot be
found for 24 hours after listing is a seller-facing problem. For 10M items a
full item-tower pass is a few GPU-hours, so nightly is comfortable.

## Evaluating the retriever without fooling yourself

Recall@K against held-out clicked or purchased items, where K is the candidate
count the ranker actually receives — Recall@1000, not Recall@10. Measuring
retrieval at 10 is measuring the ranker.

!push "Your Recall@1000 is up six points. Ship it?" -- No, not on that evidence. The held-out positives came from the incumbent's own impressions, so a retriever that finds better items the old one never showed gets no credit and can score worse while being better.

**Now the trap, and I would raise it before she does.** Your held-out positives
are items the *current* system retrieved and displayed. A new retriever that
surfaces a better item the old system never showed gets no credit for it, and
can score *worse* than the incumbent while being better. Offline recall against
logged positives is systematically biased toward the system that generated the
logs.

Four things I would do about it:

- **Pooled judgement.** Take the union of top-K from the incumbent and the
  candidate, judge the pool with humans or an LLM judge calibrated against
  humans, and evaluate both systems against the pooled set. This is the TREC
  method and it exists precisely for this problem.
- **Measure the new-item rate.** What fraction of the candidate's top-100 the
  incumbent never retrieved? Send exactly those to judgement — it is the cheapest
  possible labelling budget and it answers the question directly.
- **Segment by query frequency.** Aggregate recall is dominated by head traffic.
  Report head / torso / tail separately and make tail recall the headline
  number, because that is what the prompt asks for.
- **Let interleaving be the arbiter.** Offline recall chooses what to test;
  online interleaving decides. And interleaving is enormously more sensitive
  than an A/B for a retrieval change, so this is cheap.

Also track **zero-result rate and low-result rate on tail queries** as a direct
product metric. For the query in the prompt, that is the outcome that matters.

## The ANN index

For 10M x 256 the answer is HNSW and I would say so quickly — this is not the
interesting part of the question and spending time here is a mistake.

@decide Which ANN index?
- chose: HNSW over int8-quantised vectors, no product quantisation
- over: IVF-PQ
- why: at 10M x 256 the whole thing is about 5 GB and fits in RAM on one machine. PQ trades recall for memory you do not need; reach for it at a billion vectors, not ten million.
- cost: higher memory than IVF-PQ, and a graph that must be rebuilt rather than updated in place
@end

`M = 32`, `efConstruction = 200`, `efSearch` tuned to hit a recall target
against exact search. Int8 scalar quantisation of the vectors: 2.6 GB of vectors
plus roughly 2.6 GB of graph, so about 5 GB, which fits in RAM on one machine
with room for replicas. **No PQ.** IVF-PQ trades recall for memory you do not
need at this size; reach for it at a billion vectors, not ten million.

The one genuinely interesting ANN issue here is **filtered search**. Search has
hard filters — in stock, ships to this address, category. HNSW degrades badly
under low selectivity because the graph is built over everything and the
traversal wastes its budget on nodes the filter will discard. Post-filtering at
2% selectivity means you need ~500 raw candidates to yield 10. Use in-traversal
filtering, and below some selectivity threshold fall back to brute force over
the filtered subset, because a few hundred thousand exact dot products is faster
than approximate search over ten million. Measure where your crossover is.

## Fusion

Three ranked lists, incomparable score scales. A weighted sum of BM25 and cosine
fails because BM25 is unbounded and query-dependent while cosine is in [-1, 1],
so the effective weight varies per query in a way you did not choose.

Start with **reciprocal rank fusion** — sum of 1/(k + rank) across arms, k ~ 60.
Rank-based, so scale-free, and it is a strong baseline that needs no tuning.

Then replace it with the L1 ranker, which takes each arm's rank and score as
features along with query-understanding signals, and learns the fusion. That is
strictly better than RRF because the right blend is query-dependent — identifier
queries want the lexical arm, tail paraphrase queries want the dense arm — and a
model can condition on that where a fixed constant cannot.

# Ranking: the other half

## "Optimise CVR" is a trap, and saying so is the point

!trap "Optimise CVR" is the trap in this question. Taken literally it produces a ranker that sorts by cheapness, goes blind on the exact tail queries the prompt is about, and compounds through the exposure loop. Name the pathology, then propose expected value.

Do not accept the objective as stated. A ranker trained to maximise conversion
probability and nothing else has three specific pathologies, and naming them is
worth more than any architecture you could propose.

**It ranks by cheapness.** Conversion probability is highest for low-commitment
purchases. Sort by pCVR and a twelve-dollar pair of toddler socks outranks the
hundred-dollar boots the customer actually came for. Conversions go up, GMV goes
down, and the customer did not get what they searched for.

**It is blind to relevance when nothing converts.** For a genuinely rare query,
every candidate has a near-zero and near-identical pCVR, so the ordering is
driven by noise in a model that has no signal. That is exactly the tail query in
the prompt.

**It compounds.** Tomorrow's training data is today's exposure. A ranker that
concentrates on safe converting items sees only those items converting, and the
loop tightens.

@decide What does the ranker actually optimise?
- chose: Expected value -- pCTR x pCVR x value -- with a hard relevance gate and a returns penalty
- over: Pure pCVR, which is what the question literally asks for
- why: conversion probability is highest for low-commitment purchases, so a pure pCVR ranker sorts by cheapness; it is also blind on tail queries where nothing converts, and it compounds through the exposure loop
- cost: you now need calibrated probabilities and a business input for value, and you have to defend the deviation from the stated objective
- fallback: if she wants CVR literally, keep pCVR ordering but make the relevance gate a hard constraint and GMV per session a guardrail that cannot regress
@end

So I would rank on **expected value with a relevance gate**:

$$
\mathrm{score}(q,d) \;=\;
p_{\mathrm{CTR}}\cdot p_{\mathrm{CVR}}\cdot \mathrm{value}(d)\cdot g(p_{\mathrm{rel}})
\;-\; \lambda\, p_{\mathrm{return}}\cdot \mathrm{value}(d)
$$

$\mathrm{value}(d)$ is margin, or price, or contribution -- a business input, not
a model output, and worth asking which one they optimise. $g(p_{\mathrm{rel}})$
is a relevance gate: hard-drop below a threshold, then a mild monotone boost.
$p_{\mathrm{return}}$ matters because returns are negative revenue *and*
negative trust, so a CVR objective that ignores them optimises for regret.

!say CVR is the metric. Expected value is the objective. Those are different sentences and the difference is the whole answer.


Then offer the fallback if she pushes back and wants CVR literally: keep pure
pCVR ordering but make the relevance gate a hard constraint and add GMV per
session as a guardrail that cannot regress. That is a reasonable position and it
shows you can take direction without abandoning judgement.

## What you are actually training on

Before any model choice, two properties of the training set decide how good the
ranker can be, and both are consequences of the fact that the data came from the
ranker you already have.

**The label.** Impressions are free, clicks are plentiful, conversions are
scarce. The natural move is graded relevance — impression 0, click 1,
add-to-cart 2, purchase 3, returned purchase back to 0 — and that is what a
listwise objective wants. But if you are building the expected-value score in
Figure 8 you need *separate calibrated heads*, not one graded label, so the
label design is downstream of the objective decision below. Decide the objective
first.

**The candidate distribution is your own output.** A ranker's training examples
are impressions, and an impression is something that survived retrieval *and*
that this ranker placed high enough to be seen. So the model is trained on the
consequences of its own decisions. Items it wrongly buries generate no data, so
it never learns it was wrong about them, and the error is stable rather than
self-correcting. This is the ranking counterpart of the retrieval evaluation bias
and the reason the exploration section below is not optional.

**Negative downsampling breaks calibration, and you must correct it.** You have
perhaps a hundred impressions per click, so you will downsample negatives to
keep training tractable. That shifts the base rate, and a model trained on the
downsampled data over-predicts. With negatives kept at rate `w`, recover the
true probability with

$$
q \;=\; \frac{p}{\,p + (1-p)/w\,}
$$

$w$ is the negative keep rate and $p$ is the model's output on the downsampled
data. At $w=0.1$ and $p=0.5$ the corrected probability is $0.09$ -- a factor of
five. Forget this and every downstream multiplication by price is wrong by a
constant you did not choose.

Forget this and every downstream multiplication by price is wrong by a constant
factor you did not choose. It is a two-line fix and a very common production bug.

## The objective: pointwise, pairwise, listwise — and the constraint most people miss

Three families, and the usual answer is "listwise, obviously." For this question
that answer is wrong, and being able to say why is worth real credit.

**Pointwise** predicts a number per item independently — pCTR, pCVR — and you
sort by it. Trained with binary cross-entropy. It does not optimise order
directly, and it has no idea that position 1 matters more than position 20. Its
one enormous advantage: the output is a **calibrated probability**.

**Pairwise** — RankNet, BPR — optimises the probability that the better item is
scored above the worse one. Closer to the thing you care about, but it treats
every inversion as equally bad, so it will happily spend capacity fixing a swap
at positions 50 and 51 that no user will ever see.

**Listwise** — LambdaRank and its GBDT form, LambdaMART — fixes exactly that.

$$
\lambda_{ij} \;=\;
\big(\text{pairwise logistic gradient for } (i,j)\big)
\;\times\;
\big|\Delta\mathrm{NDCG}_{ij}\big|
$$

NDCG depends only on the ordering, so it is a step function of the scores: zero
gradient almost everywhere, a jump when two items swap. LambdaRank's move is to
skip defining a loss and write the gradient directly. Each item's gradient is
the sum of $\lambda_{ij}$ over every pair it appears in, and the
$|\Delta\mathrm{NDCG}|$ factor is the whole idea -- a swap at ranks 1 and 2 is
worth far more than one at 50 and 51, so capacity concentrates at the top of the
list. LambdaMART is these gradients plugged into gradient boosting.

!push "Why not just use LambdaMART then?" -- Because a LambdaMART score is not a probability, and the expected-value objective multiplies by price. Calibration is the constraint, not accuracy.

**Now the constraint.** The expected-value score in Figure 8 multiplies pCTR by
pCVR by price. That multiplication is only meaningful if those are
*probabilities*. LambdaMART emits a relevance score with no probabilistic
interpretation at all — you cannot multiply it by a hundred dollars and get
expected revenue. So a pure listwise ranker is incompatible with the objective
this question asks for.

@decide Pointwise, pairwise or listwise?
- chose: Pointwise BCE for the probability heads, listwise only for relevance and for a list-context layer on the final top-k
- over: LambdaMART as the single ranker, which is the reflex answer
- why: the expected-value score multiplies probabilities by price, and that is only meaningful if they are calibrated. A LambdaMART score has no probabilistic interpretation, so you cannot multiply it by a hundred dollars.
- cost: pointwise does not optimise order directly, which is exactly why the list-context layer goes back on top
@end

How I would resolve it, and this is the answer I would defend:

- **Train the probability heads pointwise** with BCE — pCTR, pCVR, pReturn —
  because the EV combination requires calibration, and calibrate them post-hoc.
- **Train a separate listwise relevance model** if you want one, and let its
  score enter the EV formula as the relevance gate `g(pRel)`, not as the score.
- **Add a list-context re-ranking layer on the final top-k**, after the
  pointwise scores. This is where listwise thinking legitimately belongs: a
  small model that sees the whole slate — DLCM, PRM, Seq2Slate in the
  literature — and adjusts for the fact that an item's appeal depends on what it
  is sitting next to. Three near-identical boots compete with each other; a
  pointwise model cannot see that and a slate model can.

That layering also matches the latency budget: the expensive list-context model
only ever runs over the ~50 items you are about to show.

## The CVR model has a data problem before it has a model problem

### Sample selection bias, and ESMM

Conversion is only observed after a click, so the natural CVR training set is
clicked impressions. But at serving time you score every candidate, including
ones that would never be clicked. Train on a selected subset, serve on the full
distribution — and the gap is exactly what your ranker is trying to change.

On top of that, clicked impressions are perhaps 1–5% of all impressions, so the
CVR training set is one to two orders of magnitude smaller than the CTR one,
precisely where you need precision.

ESMM removes both problems by never training CVR directly:

@fig:esmm Figure 11. ESMM. Both supervised tasks live on the full impression space, so the selection bias is gone and CVR is learned as the ratio. The cost: the product is numerically touchy when pCTR is small, and a miscalibrated CTR tower propagates straight through.

Both supervised tasks are defined over all impressions, so the bias is gone, and
the CVR tower inherits representations from the CTR task which has vastly more
data. The caveat to volunteer: the product is numerically touchy when pCTR is
small, and a miscalibrated CTR tower propagates straight into CVR.

### Delayed feedback

A conversion can land hours or days after the click. So at training time a
"negative" might just be a conversion that has not happened yet, and your label
is a function of how long you waited.

Three things I would do. Pick an attribution window from the actual conversion
delay distribution — measure it, do not guess; in most commerce categories the
bulk lands within 24 hours but considered purchases have a long tail. Treat
recent unconverted clicks as censored rather than negative, and either exclude
the most recent window from training or use a delayed-feedback model that
jointly models conversion and delay. And retrain often enough that the censoring
window is a small fraction of your training period.

The failure if you ignore it: your model systematically under-predicts CVR on
exactly the most recent data, which is the data most representative of now.

### Position bias, and how you actually estimate the propensities

Your labels came out of a ranked list, so a click at rank 1 and a click at rank
20 are not equivalent evidence. Rank 1 gets examined regardless. Train on raw
clicks and you learn the ranking you already had — the model's strongest
discovered feature becomes "where did the old ranker put this," and good new
items can never rise. It is a self-fulfilling loop and it is the reason a ranker
can look excellent offline and never improve anything.

The standard correction is inverse propensity scoring: weight each example by
one over the probability that the user examined that position. The interesting
part is where the propensities come from, and there are three answers with very
different costs.

**RandPair — swap two positions at random on a slice of traffic.** Unbiased and
simple. It is also the one that hurts, because you are deliberately showing
worse results. I would run it on a fraction of a percent as a calibration
reference, not as the production mechanism.

**Intervention harvesting — free, and the one I would build.** You are already
running A/B tests, and different rankers put the same query-item pair at
different positions. That is randomisation you have already paid for. Mine the
existing experiment logs for pairs that appeared at multiple positions and take
the click-rate ratio across positions. Zero additional user cost, and at 200 QPS
over six months there is plenty of it.

**Jointly estimate propensity and relevance from clicks** — regression-EM, or a
dual-tower examination model where one tower sees only position and layout
features and the other sees query-item features. No randomisation at all. The
catch is identifiability: without genuine position variation in the logs the
examination tower simply absorbs relevance and you have learned nothing. Check
it, do not assume it.

Three refinements worth naming, because they are where the textbook answer stops
and production begins.

**Position is not one number on a grid.** A mobile app shows a two-column grid,
so examination depends on row, column, device, and whether the item was above
the fold on that screen size. Condition the propensity on layout, not on a rank
index. Getting this wrong on mobile is worse than not correcting at all, because
the correction is confidently wrong.

**Trust bias is separate from examination bias.** Users click top-ranked results
partly *because* they are top-ranked — they trust the ranking — not only because
they looked. IPS on examination alone does not remove that, and the fix is a
click model that has a separate trust term per position.

**IPS has a variance problem.** Small propensities produce enormous weights, a
handful of examples dominate the gradient, and validation metrics oscillate.
Clip the weights, which trades a little bias for a lot of variance; or use
self-normalised IPS; or go doubly robust, combining IPS with a learned reward
model so you are only exposed to variance where the reward model is wrong.
Always report effective sample size alongside anything IPS-weighted.

### Calibration

If you multiply pCVR by price, pCVR must be a probability, not a score. A model
that ranks perfectly but is systematically 3x over-confident produces a
completely wrong expected-value ordering as soon as prices vary.

So: reliability diagrams on held-out data, ECE as a tracked metric, and a
post-hoc calibration map — Platt or isotonic on a held-out set — refit on a
schedule. Report calibration per price decile and per category, because
aggregate calibration hides the segments where it is broken.

## The ranking model

| Head | Loss | Feeds |
|---|---|---|
| pCTR | BCE on clicks over all impressions | the EV product |
| pCVR | none directly — learned through ESMM's product | the EV product |
| pRel | listwise, or distilled from the cross-encoder | the relevance gate |
| pReturn | BCE on returns among purchases | the penalty term |

Shared bottom first. Escalate to MMoE and then PLE only when you have *measured*
negative transfer — each head against a single-task model trained alone. Task
loss weighting usually dominates the architecture choice anyway.


@decide GBDT or neural for the L2?
- chose: Both -- a neural model produces user-history and query-item embeddings offline, and those enter a LambdaMART/LightGBM ranker as features
- over: Pure neural, or pure GBDT
- why: 200 heterogeneous engineered features is tree territory -- scale-invariant, minutes to train on CPU, monotonic constraints available, legible importances. Representation learning over raw IDs and history sequences is not, and trees cannot do it.
- cost: two training pipelines and an embedding-freshness dependency
- but: multi-task goes neural first, because a GBDT does not do multi-task cleanly and ESMM needs shared representations
@end

**Which model.** For an L2 with ~200 engineered tabular features, LambdaMART on
LightGBM is the strong default: minutes to train on CPU, scale-invariant,
monotonic constraints available when you need to guarantee that a higher-rated
product never ranks lower for that reason, and legible feature importance when a
PM asks why something ranked where it did. Neural wins when the signal is in raw
high-cardinality IDs and user history sequences. The honest production answer is
both: a neural model produces user-history and query-item embeddings offline,
and those become features in the GBDT.

Multi-task is the one place I would go neural first, because a GBDT does not do
multi-task cleanly and ESMM needs shared representations.

**Features, in four groups.** Query-item match: BM25 per field, dense cosine, the
arm ranks, category agreement, attribute overlap. Item: price percentile within
category, rating, review count, seller quality, delivery speed, stock, image
count. Query: frequency bucket, predicted category, length, intent. User and
context: history embedding, price sensitivity, device, surface, time.

**The freshness trap.** The behavioural features — historical CTR and CVR for a
query-item pair — will dominate feature importance and are near-empty on tail
queries. Train with explicit missingness rather than imputing zero, so the model
can learn a distinct regime instead of reading "no data" as "bad." Then bucket
your eval by query frequency and check the tail decile specifically, because
aggregate NDCG will look fine while the tail falls off a cliff.

## The L1/L2 split: how you train the cheap model

Everyone draws the two-stage ranker and almost nobody says how L1 is trained,
which is a shame because the answer is interesting and it mirrors retrieval.

L1's job is **not** to rank well. Its job is to not throw away anything L2 would
have wanted. So do not train it on clicks — train it to **imitate L2**. Score a
large sample of candidates with L2 offline, and fit L1 to reproduce L2's
ordering, or at minimum L2's top-k membership. That is distillation, and it is
the right objective because it is literally the thing L1 is for.

Which means the metric for L1 is **recall of L2's top-k**, not NDCG. If L1 keeps
98% of what L2 would have put in the top 100, L1 is doing its job regardless of
how it orders them. Reporting NDCG for L1 is the same category error as
reporting NDCG for retrieval.

Keep it genuinely cheap: a small tree ensemble or a linear model over a feature
subset that avoids anything requiring a remote fetch. The moment L1 needs the
same features as L2, it has stopped being a filter and you have paid for two L2s.

## Features, freshness, and the skew that eats rankers

Four groups, and roughly two hundred of them for a commerce ranker. **Query-item
match**: BM25 per field, dense cosine, each retrieval arm's rank and score,
category agreement, attribute overlap. **Item**: price percentile within
category, rating, review count, seller quality, delivery speed, stock, image
count, age. **Query**: frequency bucket, predicted category, length, intent
class. **User and context**: history embedding, price sensitivity, device,
surface, hour, session depth.

Three failure modes matter more than the feature list.

**Point-in-time correctness.** The single largest source of leakage in ranking
is computing a feature using data from after the impression. A "30-day item CTR"
computed at training time from a table built today includes clicks that happened
*after* the impression you are training on — including the click you are trying
to predict. The model learns a feature it cannot have at serving time and
offline metrics look wonderful. Your feature store must support as-of joins, and
you must test them.

**Log the features, do not recompute them.** The robust fix for training-serving
skew is to log the exact feature vector the model scored at request time, and
train on those logged values. Then skew is impossible by construction: whatever
was wrong at serving is equally wrong at training, so the model learns around it.

@fig:skew Figure 13. Logging the served feature vector removes training-serving skew by construction -- whatever was wrong at serving is equally wrong at training, so the model learns around it. The cost is storage: roughly 200 floats per impression, and you can sample.
The cost is storage — you are logging a 200-float vector per impression — and at
this volume that is real but affordable, and you can sample it.

**Behavioural features dominate and are empty on the tail.** Historical CTR and
CVR for a query-item pair will top your feature importance chart and be missing
for exactly the long-tail queries in the prompt. Train with explicit missingness
rather than imputing zero, so the model can learn a separate regime instead of
reading "no data" as "bad." Then report NDCG by query-frequency decile and make
the tail deciles a shipping gate, not a diagnostic you run after someone
complains.

## Exploration: how a new listing ever gets ranked at all

This is the closed loop from earlier, and it is a real business problem in a
marketplace, not an ML nicety. A new product has no behavioural features, so the
ranker scores it low, so it gets no impressions, so it never acquires the
features that would let it rank. Sellers notice.

Three mechanisms, cheapest first.

**Reserve exposure.** Give new or low-impression items a small guaranteed share
of slots — one slot in the first grid, or a fixed fraction of traffic. Blunt,
trivially implementable, easy to measure, and it works.

**Score by an optimistic bound rather than the mean.** The model's uncertainty
about a cold item is large; UCB-style, add a term proportional to that
uncertainty so unexplored items get a chance in proportion to how little you
know. Thompson sampling from the posterior is the cleaner version. This is
strictly better than a reserved slot because the exploration is targeted, and
strictly harder because you need a calibrated uncertainty estimate.

**Lean on content features.** The same argument as cold-start retrieval: a
ranker whose features are mostly behavioural cannot score a new item at all,
while one with strong content and embedding features can make a reasonable guess
on day zero. ID-feature dropout during training applies here exactly as it does
in the item tower.

The metric to instrument: **time-to-first-hundred-impressions for a new
listing**, and the conversion rate of items in their first week versus steady
state. If new items convert *better* than established ones, your ranker is
under-exploring and you are leaving money on the table.

# The metrics, in full

She asked about CVR; have the whole board ready, and know which one you would
actually ship on.

| Group | Metric | What it is for, and what it hides |
|---|---|---|
| **Engagement** | CTR | Fast, high-volume, the workhorse. Rewards clickbait thumbnails. |
| | Click position / MRR of first click | Are good results near the top, not just present |
| | Add-to-cart rate | Much stronger intent than click, still same-session |
| | Dwell time / long click | Proxy for satisfaction; needs a threshold you must justify |
| **Conversion** | CVR per click | The stated objective. Denominator is clicks, so it is blind to whether anyone clicked. |
| | CVR per search session | The one I would actually ship on. Captures retrieval failure, which per-click CVR cannot see. |
| | CTCVR | Conversion per impression. The ESMM target, and the honest end-to-end rate. |
| **Money** | GMV per session | The guardrail that catches the cheap-item pathology |
| | Revenue / margin per search | If they have margin data, the real objective |
| | Average order value | Moves opposite to CVR under a pure-CVR ranker; that divergence is the tell |
| **Relevance** | NDCG@k | Graded, position-discounted, the offline standard. Needs judgements. |
| | Recall@1000 | The retrieval metric. Measure at the ranker's candidate count. |
| | Precision@k, MAP, MRR | Binary-judgement cousins; MRR when one right answer exists |
| **Search health** | Zero-result rate | The direct tail metric. For this question, a headline number. |
| | Reformulation rate | The user telling you the results were wrong |
| | Query abandonment | Search with no click at all |
| | Clicks below the fold | Did they have to hunt |
| **Long-term** | Return / cancellation rate | A CVR win that raises returns is a loss |
| | Repeat purchase, next-session return | The only real measure of satisfaction |
| | Catalogue coverage, exposure Gini | Marketplace supply health; a concentrating ranker kills sellers |
| **System** | p50 / p99 latency | Latency is a quality metric — measure CVR against it |
| | ECE, calibration ratio | Required if you multiply probabilities by price |

!say I would ship on CVR per search session, with GMV per session and return rate as guardrails that cannot regress, and report tail zero-result rate separately because it is the thing this system is being built to fix.

# Mobile versus web

She specified a mobile app search bar. Being able to say precisely how the web
version differs is cheap signal that you have built both.

@fig:surfaces Figure 14. Four slots above the fold against fifteen to twenty. The consequence to state out loud: estimate propensities *per surface*, because one shared curve is confidently wrong on both.

**Position bias is far steeper on mobile**, because four items are visible
instead of twenty. Practically: you must estimate propensities *per surface* —
one shared propensity curve is simply wrong — and the value of getting rank 1
right is much higher on mobile. If you only fix one thing about your training
labels, fix this.

**Precision beats recall on mobile, and the reverse on web.** With four slots
there is no room for a hedge; showing one wrong item costs a quarter of the
visible page. On web, a broader candidate set lets the user do the comparison
themselves, so slightly lower precision with better coverage is often the better
trade. Concretely, I would run a tighter relevance gate and more aggressive
deduplication on mobile.

**Queries are shorter and dirtier on mobile.** Thumb typing means more typos,
more abbreviations, and heavier reliance on typeahead and voice. So spell
correction and query completion carry more weight on mobile — and the
suggestion ranker becomes a first-class model rather than an afterthought,
because a good suggestion converts an ambiguous tail query into a head query
before retrieval ever runs. That is the cheapest fix for the problem in the
prompt and I would say so.

**Diversity matters more on mobile.** Four near-identical listings — which a
marketplace produces constantly — is the entire visible page. Deduplication by
product identity, not just by listing, is a mobile requirement and a web nicety.

**Latency variance is worse on mobile** because of cellular networks, so the
client-side budget is tighter than the server-side one suggests. Prefetching the
next scroll page and progressive image loading are part of the design.

@decide One model for mobile and web, or two?
- chose: One ranker with surface as a feature; separate propensity curves; separate diversity and dedup policy
- over: Two complete stacks
- why: the model can learn the surface interaction from one pipeline, but the position-bias curves are genuinely different objects and a shared curve is confidently wrong on both
- cost: you must evaluate per surface always, or a mobile regression hides under a web win
@end

**How I would handle it in the model.** Surface as a feature first, not separate
models — you keep one training pipeline and the model learns the interaction.
Separate models only for the things that are genuinely different objects:
propensity curves, definitely; the diversity and dedup policy, probably; the
ranker itself, only if a surface-interaction analysis says the shared model is
leaving real value behind. And evaluate per surface always, because a shared
aggregate metric will hide a mobile regression under a web win.

# The seam: how retrieval and ranking break each other

!say The ranker is trained on whatever candidate distribution the retriever produced. So if I change retrieval, the ranker's training data is stale, and I would not read the experiment until I had retrained it on logs from the new candidates.

Raise this yourself. It is the part of the question that requires having owned
both halves, almost nobody brings it up unprompted, and it is the most direct
evidence you can give that you have shipped one of these rather than read about
it.

@fig:seam Figure 15. Change either box and the other one's training data is stale. A 6-point recall win reads as a flat A/B until the ranker has been retrained on the new candidate distribution. Never change both in one experiment.

**Ranker trained on retriever v1, served with retriever v2.** The ranker has
never seen the newly-surfaced items; they are out of distribution and it scores
them conservatively, precisely because they are unfamiliar. So a genuine 6-point
recall win shows up as a flat A/B. The fix is sequencing, not modelling: ship the
retrieval change to a slice, log the new candidate distribution, retrain the
ranker on it, and only then read the experiment. If you tell her this before she
constructs the scenario as a gotcha, you have answered the hardest follow-up in
the round in advance.

**Retriever trained on positives that the ranker chose to show.** The retriever's
positives are clicks, and clicks only happen on items the ranker put on screen.
So the retriever is being taught to find what the current ranker likes. Two
tightly coupled models, each learning the other's bias. Mitigations: mine
positives from deeper ranks and from reformulated sessions, keep an exploration
slice whose logs are used preferentially for training, and use propensity
weighting on both sides rather than only the ranker.

**Each stage's metric is blind to the other's failure.** Retrieval recall
improves while ranking gets worse, aggregate CTR is flat, and each team reports a
win. The only honest end-to-end offline measure is NDCG computed over the full
pipeline against a judgement set that was *not* pooled from the current system.
Measure stage metrics for diagnosis and pipeline metrics for decisions.

**The practical policy I would state.** Never change retrieval and ranking in the
same experiment — you cannot attribute the result. Retrain the ranker on a
cadence that is a multiple of the retriever's, always downstream of it. And keep
a small always-on randomised slice whose logs are the unbiased sample everything
else gets calibrated against; it costs a fraction of a percent of traffic and it
is the only thing that stops the whole loop drifting somewhere nobody chose.

# The follow-up tree

Everything below is a push she can make. Rehearse the answers out loud; the
failure mode in this round is hesitation, not being wrong.

## On retrieval training

**"How do you set the weight on a purchase versus a click?"** I measure it
rather than pick it. Judge a few thousand query-item pairs stratified by action,
compute the probability the pair is relevant given each action, and take the
log-odds difference against impressions. It comes out far flatter than intuition
— about 2.6 to 1, not 10 to 1 — because the click already carries most of the
relevance evidence and the purchase mostly adds price and availability, which
are ranking concerns.

**"Do you treat a returned purchase as a negative?"** For the retriever, barely
— the customer searched, found it and bought it, so it was relevant; the return
is usually fit or quality. For the ranker it matters enormously, because a
return is negative revenue and negative trust. Returns are a ranking signal, not
a retrieval signal.

**"Why aggregate to (query, item) instead of training per event?"** Because the
retriever has no user features — the query tower sees a string. Per-event rows
therefore add no information and add popularity skew, since a pair shown a
million times would contribute a million gradient steps for a signal its counts
already capture. The ranker is the opposite: it has user and context features,
so it needs per-impression rows and the unclicked negatives the retriever's
table discards.

**"What does your negative strategy cost?"** In-batch negatives are free —
already encoded. Every mined or uniform negative is a full item-tower forward
pass, so six explicit negatives per positive turns 4,096 encoder passes per step
into about 28,000. The mitigation is caching item embeddings from the last index
refresh and recomputing only on the refresh cadence, which is what ANCE does.

**"How do you split?"** By time, never randomly — the last two weeks are test,
because you serve forward in time and a random split leaks the same pair to both
sides. And a second, query-disjoint split of whole tail queries, because a
time-based split still lets a head query's embedding be learned in the training
period and evaluated in the test period.

**"Why a softmax and not MSE?"** Because there is no target. A click is not a
similarity of 1.0, so you would have to invent the label, and the model would
learn whatever you invented. Beyond that, MSE optimises magnitudes nothing ever
reads, and it has no normalisation — a model that outputs 0.3 for everything has
a fine MSE and zero retrieval utility.

**"Why not triplet loss?"** Four things. It sees one negative per step, so the
gradient is a noisy estimate of "beat everything." Its margin is an absolute
quantity in a space whose scale the model controls, so unnormalised it can be
satisfied by inflating norms. The hinge switches off once satisfied, so late in
training most of the batch contributes no gradient — hence semi-hard mining.
And it collapses more readily. The softmax's temperature does the margin's job
relatively rather than absolutely, and it is smooth.

**"Is triplet ever right?"** Yes, for verification rather than retrieval.
Contrastive and triplet losses give you a calibrated distance with a usable
threshold; a softmax gives you an ordering with no absolute meaning. Catalogue
deduplication — is this seller's listing the same physical product as that one —
is exactly that, and it sits inside this system feeding the slate policy's dedup
step. I would train it with a margin loss.

**"Where would you use MSE in this design, then?"** Two places. Distilling the
cross-encoder into the bi-encoder, but on the *margin* rather than the raw score,
because the teacher's absolute scale is arbitrary and only the ordering
transfers. And training L1 to imitate L2, which is the same argument.

**"BPR or RankNet instead?"** They are the pairwise cousins and share triplet's
core limitation: one comparison at a time, and no notion that position 1 matters
more than position 50. BPR is the right tool for implicit-feedback recsys with
small candidate sets; for retrieval over 10M it is strictly dominated.

**"When would you use a listwise loss on the retriever?"** I would not. Listwise
losses need the whole list, and the retriever's "list" is the corpus. Listwise
belongs where the candidate set is small enough to hold — the reranker, and the
slate layer.

**"Why not just use a cross-encoder for retrieval?"** Cost. A cross-encoder is
about a millisecond per pair, and 10M pairs per query is three hours. The
two-tower exists so the item side can be precomputed and the query side reduced
to one dot product against an index. You pay for it in expressiveness — the
towers never interact — which is why the cross-encoder comes back as a reranker
over 50 candidates.

**"Should the two towers share weights?"** No, for two reasons. Query text and
product text are different distributions — three words of intent versus a
hundred words of catalogue copy — and a shared encoder has to compromise. And
sharing throws away the asymmetry: the item tower can be large because it is
offline, the query tower must be small because it is online. Sharing forces both
to the smaller budget.

**"How do you handle a brand new product with no interaction history?"** This is
where the two-tower earns its keep: the item tower is content-based, so a new
listing gets a usable embedding from its title and attributes the moment it is
indexed. That is a strong argument against pure ID-embedding retrieval in a
catalogue with churn.

**"Would you put an item-ID embedding in the item tower?"** Yes, but with ID
dropout during training — randomly mask the ID feature so the model is forced to
produce a good embedding from content alone. Then established items get the
benefit of their learned ID and cold items degrade gracefully instead of falling
off a cliff. Without dropout the model leans entirely on the ID and cold start
breaks.

**"What does batch size actually do here?"** It sets the number of in-batch
negatives, so it sets the difficulty of the classification task and the quality
of the gradient. That is why these models train at thousands and use cross-device
negative sharing. Past a point, more random negatives stop helping and mined hard
negatives take over.

**"How do you choose the temperature?"** Sweep it, and diagnose on the score
distributions rather than the loss — you want clear separation between positive
and negative scores without the positive saturating. Start around 0.05 for
normalised embeddings. Too low and the gradient collapses onto the single
hardest negative; too high and the softmax is flat and there is no signal. It can
be learned with a clamp.

**"What is the ratio of hard to random negatives?"** I would start with a handful
of mined hard negatives per positive on top of the full batch as in-batch
negatives plus uniform corpus negatives, and tune it. All-hard is a known failure
mode — it over-fits the miner's idea of hard, and it amplifies any false
negatives you failed to filter.

**"Your hard negatives are mined from BM25. What happens once the model beats
BM25?"** They stop being hard, and you silently stop learning from them. That is
why you re-mine from the current model's own index every epoch or two — the ANCE
loop. Static negatives go stale and the training curve looks fine while the
model stops improving.

**"How do you know a mined negative is not actually relevant?"** You do not,
which is why you filter. Score every mined negative with the cross-encoder
teacher and drop anything scoring above the positive. Training on a false
negative is worse than training on no negative — you are explicitly teaching the
model that a correct answer is wrong.

**"Why does the logQ correction matter in commerce specifically?"** Because
in-batch negatives are sampled from the click stream, so popular items are
negatives far more often, and the uncorrected model learns an anti-popularity
bias — it under-retrieves exactly the items that convert. The correction
subtracts the log sampling probability from every logit so the penalty is per-
occurrence rather than cumulative.

**"What about items that never appear in any batch?"** They are never negatives
for anyone, so nothing ever pushes them down, and they can surface spuriously.
At 10M products with a long tail this is a large fraction of the catalogue. Fix
it with mixed negative sampling — uniform draws from the full index alongside
the in-batch ones.

**"Recall@1000 improved 6 points and online CTR is flat. What happened?"** Most
likely the ranker cannot exploit the new candidates. It was trained on the
distribution the *old* retriever produced, so the newly-surfaced items are
out-of-distribution for it and it scores them conservatively. Retrieval and
ranking are coupled: after a retrieval change you must retrain the ranker on
logs from the new candidate distribution before you can read the result. The
other candidates are that the new recall is on items that were never going to
convert, or that your offline positives were biased toward the incumbent.

**"How would you prove the dense arm is earning its latency?"** Ablate it in an
interleaving test, and separately measure its unique contribution — the fraction
of clicked or purchased items that only the dense arm retrieved. If that number
is small, the arm is paying for overlap with BM25 and should be cut or retrained.

**"What breaks if you retrain the query tower but not the item index?"** Silent
garbage. The two towers define a joint space; a query embedded by v2 against
items embedded by v1 is meaningless, and nothing errors. Version the model into
the index metadata and refuse to serve a mismatch.

## On the funnel

**"Why three retrieval arms and not two?"** Because they fail on disjoint query
populations. If the behavioural arm's unique contribution is small once you have
a good dense arm, drop it — measure, do not assume.

**"What sets the candidate count out of retrieval?"** The ranker's latency
budget from one side, and the recall curve from the other. Plot recall against
candidate count and find where it flattens; take candidates up to that knee, and
then check the knee is in the same place for tail queries, which it usually is
not.

**"Where does personalisation enter?"** Ranking first, because it is cheap there
and easy to measure. Retrieval only once you have shown that ranking-side
personalisation has saturated, because a personalised retrieval arm multiplies
your index cost and makes caching much harder.

## On ranking

**"Pointwise, pairwise or listwise?"** All three, in different places, and the
constraint is calibration. The expected-value score multiplies probabilities by
price, so pCTR and pCVR must be pointwise and calibrated — a LambdaMART score
cannot be multiplied by a hundred dollars. Listwise belongs in a relevance model
feeding the gate, and in a list-context re-ranking layer over the final fifty.

**"Walk me through LambdaRank."** NDCG is a step function of the ordering so it
has no usable gradient. LambdaRank skips defining a loss and writes the gradient
directly: the pairwise logistic gradient multiplied by the change in NDCG you
would get from swapping that pair. The multiplier is the whole idea — it makes a
swap at ranks 1 and 2 worth far more than one at 50 and 51. Plug those gradients
into gradient boosting and you have LambdaMART.

**"You downsampled negatives 100:1. What did that break?"** Calibration. The
model now predicts in the downsampled base rate, so every probability is inflated
and every multiplication by price is wrong by a constant. Correct it analytically
with the keep rate, then verify on a reliability diagram — do not trust the
formula without checking it.

**"How do you train L1?"** By distilling L2, not on clicks. Its job is to not
discard anything L2 would have wanted, so the objective is to reproduce L2's
ordering and the metric is recall of L2's top-k. Reporting NDCG for L1 is the
same mistake as reporting NDCG for retrieval.

**"Your top feature is 30-day item CTR. What could be wrong with it?"** Two
things. Point-in-time correctness — if it is computed from a table built today it
contains clicks that happened after the impression, including the one you are
predicting, and that is leakage that makes offline metrics look wonderful. And
coverage: it is empty on exactly the tail queries this system is being built for,
so train with explicit missingness rather than imputing zero.

**"How do you stop the ranker from being a self-fulfilling prophecy?"** Two
levers. Propensity-weight the training labels so a click at rank 1 is not treated
as the same evidence as a click at rank 20. And explore: reserved exposure for
cold items, or an optimism bonus proportional to the model's uncertainty. Without
exploration, items the ranker buries generate no data and the error never
corrects.

**"A new seller lists a product. When does it first rank?"** Instrument it —
time-to-first-hundred-impressions is a real metric with a real owner. Without
exploration the honest answer is "possibly never," which in a marketplace is a
supply problem, not an ML problem.

**"Would you use one model for CTR and CVR or two?"** ESMM, which is neither: two
supervised heads on the full impression space, CTR and CTCVR, with CVR learned
implicitly as the ratio and never given a loss of its own. That removes the
sample selection bias and gives the CVR tower a representation trained on far
more data.

**"MMoE or shared-bottom?"** Shared-bottom first, and measure each head against a
single-task model trained alone. A drop is negative transfer and that is your
signal to escalate to MMoE, then PLE. Escalating without the measurement gets you
four times the parameters and no gain. And task loss weighting usually dominates
the architecture choice anyway.

**"Your pCVR is well-ranked but badly calibrated. Does it matter?"** It does the
moment you multiply it by price. A systematic 3x over-confidence produces a
completely wrong expected-value ordering while AUC looks fine. Fix with a
post-hoc monotone map on held-out data, refit on a schedule, and report
calibration per price decile and per category because the aggregate hides the
broken segments.

## On CVR specifically

**"Why not just train on purchases?"** Sixty million purchases across ten million
products is far too sparse, especially on the tail, which is what the question
is about. Train on clicks for volume, weight by the downstream action.

**"How long is your attribution window?"** Measured from the conversion delay
distribution rather than chosen. And whatever it is, recent unconverted clicks
are censored, not negative — either exclude the most recent window or model the
delay.

**"Your pCVR is well-ranked but badly calibrated. Does it matter?"** It does the
moment you multiply it by price, which the expected-value objective does. Then a
systematic 3x over-confidence produces a completely wrong ordering. Fix with a
post-hoc monotone map on held-out data; it barely moves AUC by construction.

**"MMoE or shared-bottom?"** Shared-bottom first, and measure each task head
against a single-task model trained alone. If a task is worse in the joint model
that is negative transfer and you escalate to MMoE, then PLE. Escalating without
that measurement gets you four times the parameters and no gain.

## On serving

**"You are over your latency budget. What do you cut?"** L2 candidate count
first — 1000 to 500 is usually under a point of NDCG for half the ranking cost,
the best exchange rate in the system. Then quantise or distil the ranker. Then
`ef_search`. Retrieval depth last, because that is the only cut that loses recall
permanently. And never a silent timeout that drops an arm.

**"Would you cache?"** Yes, for cost and for median latency — but say clearly
that it does not help p99, because p99 is the tail and the tail is what misses
the cache. And in commerce the cache has a correctness problem: price and stock
change constantly, so cache the ranked ID list and hydrate price and availability
at request time rather than caching rendered results.

**"How do you handle filters at 2% selectivity?"** In-traversal filtering rather
than post-filtering, and below a measured selectivity crossover, brute force over
the filtered subset. A few hundred thousand exact dot products beat approximate
search over ten million.

## On evaluation

**"How do you build a judgement set for tail queries?"** Sample by query
frequency band, not uniformly, or you will get a head-only set. Judge with an
LLM calibrated against a human gold set, benchmarked on human-human agreement
rather than on perfect agreement, with a weekly human audit to catch drift.

**"Offline NDCG is up and the A/B is flat. What is your first move?"** Check that
the judgement set is not stale relative to the catalogue, then check whether the
gain is concentrated in a segment that carries little traffic, then check for a
presentation-layer difference the offline metric cannot see. And check
statistical power before concluding "flat" — most flat results are underpowered.

**"How much traffic do you need to detect a 1% relative CVR lift?"** Ask for the
baseline rate, then size it. At a 2% baseline CVR, detecting a 1% relative lift
at 80% power and alpha 0.05 needs on the order of tens of millions of sessions
per arm. State that you would compute it rather than guess, and that if the
required sample is implausible you switch to interleaving for the ranking
comparison.

# What I would build first

The sequencing question is where Staff shows. Four quarters, and each one ends
with something measurable.

**Q1 — instrumentation and the cheap wins.** You cannot improve what you cannot
measure, and this system has been running for six months with no tail
segmentation. Build the judgement set stratified by query frequency, start
reporting zero-result rate and recall by frequency decile, and stand up
interleaving. In parallel ship the two cheapest quality wins: the category gate
with confidence thresholding and a behavioural synonym layer mined from the
logs. Both are days of work and both move tail queries.

**Q2 — the dense arm, v1.** Two-tower trained with in-batch negatives, the logQ
correction, and mixed negative sampling. RRF fusion with BM25. HNSW over 10M
int8 vectors. Ship behind interleaving and measure unique contribution on tail
queries. This is the single biggest expected move on the stated problem.

**Q3 — make it good.** Train the cross-encoder. Use it three ways: false-negative
filtering, distillation into the bi-encoder, and adaptive-depth reranking for
positions 7+ off the blocking path, per Option C. Add ANCE hard-negative refresh. Replace RRF with a
learned L1 fusion.

**Q4 — the objective.** Multi-task ranker with ESMM, delayed-feedback handling,
calibration, and the expected-value objective with a relevance gate. This is
last not because it is unimportant but because until retrieval is fixed, the
ranker is reordering a bad candidate set.

If she asks what you would do with one engineer for one month instead: the
category gate and the synonym layer, and the tail measurement that proves the
rest is worth funding.

# The whiteboard, in order

What to draw, when, and the numbers to have on the board. Practise the first
three until they come out without thinking.

| Minute | Draw | Say while drawing |
|---|---|---|
| 5 | **The funnel** (Fig 1) | Label each box with its millisecond budget |
| 10 | **Query decomposition** (Fig 3) | Why BM25 fails on the prompt's own example |
| 15 | **The two towers** (Fig 4) | Draw them *different sizes*, and say why |
| 18 | **The weight pipeline** (Fig 4b) | If she asks how positives are chosen — counts, calibrated values, propensity |
| 20 | **The loss** (Fig 5) | Write it out; this is the retrieval assessment |
| 22 | **What each loss sees** (Fig 5b) | Only if she asks why not triplet or MSE — four sketches, thirty seconds |
| 24 | **logQ correction** (Fig 6) | One line under the loss |
| 32 | **The EV objective** (Fig 8) | The pivot from retrieval to ranking |
| 38 | **LambdaRank's lambda** (Fig 10) | And why you still need calibrated pointwise heads |
| 44 | **ESMM** (Fig 11) | Only if she pushes on CVR data |
| 50 | **The seam** (Fig 15) | Raise it yourself; it is the highest-signal moment |

Figures 7, 9, 12, 13 and 14 are worth knowing but not worth board time unless
she asks — the training pipeline, the downsampling correction, log-and-train,
the multi-task head, and the mobile/web comparison.

Numbers to have in your head, not on a slide:

!num 10M x 256-d int8 = 2.6 GB of vectors plus ~2.6 GB of HNSW graph. 20 requests in flight at 200 QPS and 100 ms. Six months = ~3B searches, ~1.2B clicks, ~60M purchases. Cross-encoder over 50 candidates = 10-15 ms on a T4. Clicked impressions are 1-5% of all impressions. ~100 impressions per click. Temperature ~0.05, batch size in the thousands, embedding dim 256.

- 10M x 256-d int8 = **2.6 GB** of vectors, plus ~2.6 GB of HNSW graph at M=32
- **20 requests in flight** at 200 QPS and 100 ms
- 6 months at 200 QPS = **~3B searches, ~1.2B clicks, ~60M purchases**
- Cross-encoder over 50 candidates = **~10–15 ms** on a T4-class GPU
- Clicked impressions are **1–5%** of all impressions — the CVR data problem
- ~**100 impressions per click**, which is why you downsample negatives
- Temperature **~0.05**; batch size **thousands**; embedding dim **256**

# Ten things that would lose this round

**On retrieval**

- Drawing two identically-sized towers, and not knowing why they differ.
- Saying "in-batch negatives" and stopping, without the sampling bias.
- Quoting Recall@10 as the retrieval metric.
- Claiming offline recall against logged clicks is an unbiased evaluation.

**On ranking**

- Accepting "optimise CVR" without naming the cheap-item pathology.
- Answering "listwise, obviously" without noticing that the expected-value
  objective needs calibrated probabilities a LambdaMART score cannot give you.
- Training on raw clicks with no propensity correction, then wondering why the
  model's best feature is the old ranker's output.
- Having no answer for how a brand-new listing ever gets its first impression.

**On both**

- Spending fifteen minutes on sharding a 10M-item index that fits in RAM.
- Treating retrieval and ranking as two independent problems, and having nothing
  to say when a recall win produces a flat A/B.
