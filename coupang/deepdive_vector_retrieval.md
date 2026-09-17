# The vector retrieval question

The third set-piece, and the one that is least about machine learning. Like the
AI-powered search document, this wording is constructed rather than reported —
but it is the question that falls out of the set-piece the moment you say "HNSW
over 10M int8 vectors" and the interviewer decides to find out whether you know
what you just said.

> **Design and size the vector index.** 100M items, 768-dimensional
> embeddings, 2000 QPS, p99 20 ms for the retrieval stage alone. Every query
> carries hard filters — category, in stock, ships to region. The catalogue
> changes continuously, roughly 1M updates a day. Pick the index family, size
> it, defend the recall number you claim, and tell me how you would evaluate
> it.

The numbers are deliberately an order of magnitude above the set-piece. At 10M
and 200 QPS almost any index works and the interesting content is in training
the embeddings; at 100M with filters and continuous updates, the index *is* the
problem, and the answer separates people who have run one from people who have
called one.

## What is actually being tested

**Arithmetic, out loud.** Every claim in this round has a number behind it and
you are expected to produce it unprompted. Memory per representation, candidates
touched per query, bytes scanned, cache misses, requests in flight. A candidate
who says "HNSW, obviously" without sizing it has answered nothing.

**Do you know why any of this works.** The theory has a punchline that changes
what you do: nearest neighbour is provably unstable in high dimensions for
independent data, real embeddings escape that only because they have low
intrinsic dimension and cluster structure, and therefore **your recall numbers
are a property of your data, not of your index**. Everything about evaluation
follows from that one sentence.

**Filters.** This is the part that decides the architecture and the part that
tutorials skip. A design that is correct at 100M and wrong at 2% filter
selectivity is wrong.

**Freshness.** Graph indexes do not really support deletion. Knowing that, and
knowing what you do about it, is a strong signal.

!trap The question looks like "which index would you pick", so people answer with a name. It is actually "show me the recall-latency-memory triangle for this specific corpus, and then tell me what your filters and your update rate do to it." Pick the index last.

## The first five minutes: what to ask

1. **"What recall, at what $k$, against what ground truth?"** Recall@10 at 0.95
   against exact search is a completely different budget from recall@100 at
   0.99. And "against exact search" matters: recall measured against logged
   clicks is not recall, it is a relevance metric wearing recall's clothes.
2. **"What does the filter selectivity distribution look like?"** Not the
   average — the distribution. A filter that keeps 40% of the corpus and one
   that keeps 0.05% need different strategies, and most catalogues have both.
3. **"Is 20 ms the index's own budget, or does it include fan-out and merge?"**
   Across eight shards, serialisation, network and merge are real milliseconds
   before any search runs.
4. **"How often does the embedding model change?"** Item updates are an
   insert-delete problem. A new *model* is a full rebuild, and if that happens
   monthly it dominates the operational design.

## Numbers you derive from the prompt, out loud

Write this table before you choose anything. It is the round.

| Representation | Derivation | Size |
|---|---|---|
| fp32 vectors | 100e6 x 768 x 4 | 307 GB |
| fp16 | 100e6 x 768 x 2 | 154 GB |
| int8, scalar-quantised | 100e6 x 768 x 1 | 76.8 GB |
| PQ, 96 subvectors x 8 bit | 100e6 x 96 | 9.6 GB |
| PQ, 192 subvectors x 8 bit | 100e6 x 192 | 19.2 GB |
| HNSW graph, M = 32 | 100e6 x 64 links x 4 B | 25.6 GB |
| IVF centroids, C = 10,000 | 10e3 x 768 x 4 | 30 MB |
| Ids and payload, 8 B each | 100e6 x 8 | 0.8 GB |

Three consequences, said immediately:

- **fp32 in RAM is off the table** at 307 GB plus a 25.6 GB graph unless you are
  willing to buy very large machines to hold data you have no reason to keep at
  that precision.
- **int8 plus a graph is 102 GB**, which is eight shards of 13 GB — ordinary
  machines, and the design I will defend.
- **PQ plus IVF is 9.6 GB**, which fits on *one* machine with room to spare, and
  buys that with a recall ceiling. Whether that trade is right depends entirely
  on the recall number nobody has told me yet, which is why I asked first.

| From | Derivation | Number |
|---|---|---|
| 2000 QPS at 20 ms | Little's law: 2000 x 0.02 | 40 requests in flight |
| 8 shards | 2000 x 8 | 16,000 shard-queries per second |
| 1M updates a day | 1e6 / 86400 | ~12 writes per second, trivially absorbed |
| 1M updates a day | 1e6 x 365 / 100e6 | the corpus fully turns over in ~9 months |

!num 100M x 768 is 307 GB fp32, 76.8 GB int8, 9.6 GB at 96-byte PQ codes. An HNSW graph at M=32 costs 256 bytes per vector, which is more than the int8 vector for low dimensions and a third of it here. 40 requests in flight. 16,000 shard-queries per second across 8 shards.

## The sixty minutes

| Minutes | What | Why |
|---|---|---|
| 0-5 | Clarify recall, selectivity, budget boundary, model churn | All four change the answer |
| 5-10 | The memory table and the brute-force baseline | Establishes the arithmetic habit |
| 10-18 | What problem this is: NN, cosine, MIPS, and the reductions | Cheap to say, and it is the theory the rest rests on |
| 18-30 | **The index families and the sizing** | The core |
| 30-42 | **Filters and freshness** | Where the design is actually decided |
| 42-52 | Evaluation, and why recall is a property of your data | The senior differentiator |
| 52-60 | Serving, sharding, tail latency, what to build first | |

# First, what problem is this actually?

Thirty seconds, and it earns its place because the wrong answer here is
expensive later.

Three flavours of top-$k$ retrieval, distinguished only by the score:
**$k$-NN** under Euclidean distance, **$k$-MCS** under cosine, and **$k$-MIPS**
under inner product. They reduce to each other, and you should be able to write
the reductions on the board:

**Cosine to inner product.** L2-normalise the corpus once, offline. Then
$\arg\max \langle q,u\rangle / \lVert u \rVert = \arg\max \langle q, u/\lVert u
\rVert\rangle$. The query's norm never matters — it is a positive constant
across all candidates.

**Euclidean to inner product.** Expand $\lVert q-u\rVert^2 = \lVert q\rVert^2 -
2\langle q,u\rangle + \lVert u\rVert^2$, drop the query term, and append one
dimension:

$$
q' = [\,q;\; -\tfrac{1}{2}\,], \qquad u' = [\,u;\; \lVert u\rVert^2\,]
\;\in\; \mathbb{R}^{d+1}
$$

**Inner product to cosine**, which is the direction you need when your vector
database only offers a cosine index and your model was trained with dot product.
Rescale so $\max \lVert u \rVert \leq 1$, then send $u \mapsto [u; \sqrt{1-\lVert
u\rVert^2}]$ and $q \mapsto [q; 0]$. The augmented item vectors are unit norm,
so a cosine index over them ranks by the original inner product. One extra
dimension, no exotic hashing required.

**Why MIPS is genuinely harder, in one line each.** Inner product is not a
metric: no triangle inequality, so every bound a tree or a pivot relies on is
gone. And it has no *coincidence* — a point need not maximise inner product with
itself. If $p = 2v$ then $\langle v,p\rangle > \langle v,v\rangle$. Some items
can therefore never be the answer for any query, which is a statement no
distance metric permits.

!say Before I pick an index I want to know whether the embeddings are normalised, because it decides whether this is a metric problem or not -- and if they are not normalised, whether the norm carries signal. In commerce it usually does: norm correlates with popularity, and normalising quietly deletes it.

The practical consequence people miss: **normalising embeddings changes the
ranking** whenever norms carry information. If your two-tower model learned to
give popular items larger norms, L2-normalising to use a cosine index throws
that away and you will see it as a ranking regression that looks like an index
bug.

# Why high dimensions are hard, and why this works anyway

The theory, compressed to what changes your behaviour.

**The instability result.** If the ratio of the variance of the query-to-point
distance to its squared mean goes to zero as dimension grows, then for any
$\epsilon > 0$ *every* point in the corpus becomes an $\epsilon$-approximate
nearest neighbour. The nearest-neighbour question stops being well posed: the
furthest point is within a whisker of the nearest, and a tiny perturbation of
the query flips the answer. For independent coordinates this ratio is
$d\sigma^2 / (d\mu)^2 \to 0$, so it applies.

**Two operational consequences**, and they are the reason to mention this at all:

- **Never benchmark an ANN index on synthetic independent random vectors.** The
  ground truth there is meaningless, and so is any recall number measured
  against it. This is a real and common evaluation bug.
- **Real embeddings escape the result because they are not independent.** They
  are clustered and they have low *intrinsic* dimension — a ball of radius $2r$
  can be covered by a modest number of balls of radius $r$, far fewer than the
  ambient $2^{768}$. Everything that works in practice, graphs and clustering
  alike, is exploiting that structure and nothing else.

**Why not just project down first?** Because the Johnson-Lindenstrauss bound
says you need $d' = O(\epsilon^{-2}\ln(m/\delta))$ dimensions to preserve
pairwise distances. At $m = 10^8$ and $\epsilon = 0.1$ that is roughly
$18.4/0.01 \approx 1{,}840$ dimensions — *more* than the 768 you started with.
Random projection is not why your 768-d embeddings are searchable. Low intrinsic
dimensionality is.

!push "So what does that mean for my recall target?" That recall is a property of my data, not of the index. I cannot promise 0.95 recall at a latency from a datasheet, because the published curves were measured on SIFT and GloVe. I can promise a measured curve on a sample of our own corpus and our own query distribution, within a week.

# The baseline you must compute before choosing anything

Exhaustive search. Not as a strawman — as the number that tells you whether you
need an index at all, and per shard it often says you do not.

A modern core scans memory for a dot product at roughly 10 GB/s with SIMD. In 10
ms that is about 100 MB, which at 768 int8 bytes per vector is **roughly 130,000
vectors per core per 10 ms**, or about 2M vectors on a 16-core box. So:

- **Under ~1M vectors per shard: do not build an index.** Flat scan, perfect
  recall, no parameters, no rebuild story, no filter problem. Enormous amounts
  of engineering have been spent on indexes for corpora that fit in this box.
- **At 100M: exhaustive is 77 GB per query scanned**, seconds per core. Out of
  the question, and now you know by how much — three orders of magnitude, which
  is what tells you an approximate method with a 100x speedup is not enough on
  its own and you will also be sharding.

@decide Index or no index?
- chose: Index, because 100M is three orders of magnitude past the flat-scan ceiling
- over: Flat scan with more machines, which is cheaper to operate and gives exact answers
- why: recovering 77 GB of scan per query would take roughly 800 cores at 10 ms, which is an absurd fleet for a retrieval stage
- cost: every parameter below, plus a rebuild pipeline and a recall number that is now empirical rather than exact
- drop it if: the corpus is ever partitioned such that a query only ever touches a few million vectors -- a hard category partition can do this, and then flat scan inside the partition is both simpler and exact
@end

# The families, and the one thing to know about each

**Trees.** k-d trees, and their descendants. They die in high dimensions for a
precise reason worth stating: certifying that a cell cannot contain a better
neighbour requires visiting a number of cells exponential in dimension, so exact
tree search degenerates to a full scan somewhere around 20 dimensions. What
survives is *defeatist* search — descend to one leaf, do not backtrack, accept
the miss — repeated over a randomised forest, which is what Annoy is. Verdict:
know the failure mode, because "why not a k-d tree" is a standard probe. Do not
propose one.

**Locality-sensitive hashing.** The one family with real guarantees. A family is
$(r, cr, p_1, p_2)$-sensitive if close points collide with probability at least
$p_1$ and far points at most $p_2$; amplify with $\text{AND}$ over $L$ bands of
$K$ hashes; the exponent $\rho = \ln p_1 / \ln p_2$ then gives query time
$O(d\,m^{\rho})$ and space $O(m^{1+\rho})$. The families to know: **SimHash**
(random hyperplane, collision probability $1 - \theta/\pi$, the one for cosine),
**E2LSH** via $p$-stable distributions for $L_2$, and cross-polytope for the
better exponent. Verdict: the guarantees are real, the constants are bad, and
graphs beat it empirically at every operating point anyone cares about. Its
space bound is the quiet killer — superlinear space at 100M is not a thing you
want.

**Graphs.** The production default, and the theory is worth two sentences. Greedy
traversal on the Delaunay graph is *exact*, but the Delaunay graph's degree
explodes with dimension, so every practical graph is an approximation of it: a
$k$-NN graph plus pruning. Pruning by the relative neighbourhood rule keeps the
graph navigable while cutting degree; **HNSW** adds a hierarchy of long-range
links, which is Kleinberg's small-world construction and buys logarithmic hop
counts; **Vamana/DiskANN** prunes with a slack factor $\alpha > 1$ and is the
only practical graph with worst-case guarantees, plus the only one designed to
live on SSD. Parameters you must be able to explain: $M$ (degree, memory and
recall), $efConstruction$ (build quality, build time), $efSearch$ (the
recall-latency knob at query time, and the only one you can turn without a
rebuild).

**Clustering, i.e. IVF.** Partition with k-means into $C$ clusters, keep
centroids, and at query time scan the $nprobe$ nearest clusters exhaustively.
The cost is $C + nprobe \cdot m / C$ distance computations, which is minimised at
$C = \sqrt{m}$ — so 10,000 clusters at 100M, or about 3,500 per 12.5M shard. The
pathologies, which is the part that shows experience: **cluster imbalance**
(k-means on real catalogue data produces clusters that differ by orders of
magnitude in size, so $nprobe$ buys wildly different amounts of work per query
and your p99 is set by the fat clusters), **boundary queries** (a query near a
Voronoi boundary needs a much larger $nprobe$ than one at a centroid, and
nothing in the index knows which you have), and **norm imbalance under MIPS**,
where plain k-means clusters by direction while the answer depends on magnitude.

**Quantization.** Scalar to int8 is the free 4x — do it always, and calibrate
per dimension, not globally. **Product quantization** splits the vector into $M$
subvectors, k-means each subspace to 256 centroids, and stores $M$ bytes;
scoring uses asymmetric distance computation, precomputing a $M \times 256$
lookup table per query and summing $M$ table lookups per candidate. **OPQ** adds
a learned rotation first so that variance is spread evenly across subspaces,
which is nearly free at query time and worth several points of recall.
**Anisotropic (ScaNN-style)** quantization is the one to name if the objective
is MIPS: the key result is that the component of the quantization error
*parallel* to the vector damages the inner product much more than the orthogonal
component, so the loss should weight them differently rather than minimising
plain reconstruction error.

**Sketching.** JL and random projections. Covered above: at this corpus size the
bound asks for more dimensions than you have, so its role here is as the
argument for why you *do not* project, plus the vocabulary for talking about
sketch variance if pushed.

# The sizing, worked

This is the centre of the round. Do it out loud.

**The design: eight shards, HNSW over int8, flat re-ranking on top.**

| Component | Per shard (12.5M) | Total |
|---|---|---|
| int8 vectors, 768-d | 9.6 GB | 76.8 GB |
| HNSW graph, M = 32 | 3.2 GB | 25.6 GB |
| Ids and filter payload | 0.1 GB | 0.8 GB |
| **Working set** | **~13 GB** | **~103 GB** |

Thirteen gigabytes per shard fits comfortably on a 32 GB machine with the page
cache, the fresh tier and the rebuild headroom that the freshness section needs.
Replicate each shard twice for availability and you have a 16-machine fleet,
which is unremarkable.

**Now the latency, and the important part is *what it is bound by*.** An HNSW
search at $efSearch = 128$ touches a few thousand candidate vectors. Each one is
a **random** memory access:

| Step | Derivation | Time |
|---|---|---|
| Distance computations | ~3,000 candidates x 768 int8 with SIMD | ~30 us of arithmetic |
| Cache misses | ~3,000 random accesses x ~80 ns | ~240 us |
| Per-shard search | dominated by the misses | **~0.3 ms** |
| Fan-out, merge, network | 8 shards in parallel, serialise and merge | ~2-4 ms |
| Re-rank top 200 with int8 exact | 200 x 768 B, sequential | negligible |
| **Total** | | **~5 ms typical** |

!say Graph search is memory-latency bound, not FLOP bound. The arithmetic is thirty microseconds and the pointer chasing is a quarter of a millisecond, which is why quantising to int8 helps more than the 4x memory saving suggests -- it is also a 4x reduction in bytes per cache miss -- and why adding cores stops helping long before it should.

Against a 20 ms p99 budget, 5 ms typical leaves genuine headroom, and you need
it: filters cost over-fetch, the tail is worse than the mean, and the fan-out
amplifies it, per the serving section.

**Capacity.** 16,000 shard-queries per second at ~0.3 ms of CPU each is about 5
core-seconds per second — five busy cores, on a fleet of 16 machines. This is
not a CPU problem. It is a memory-footprint and tail-latency problem, and saying
so out loud is the right framing.

**The alternative, for contrast.** IVF-PQ with 96-byte codes: 9.6 GB total,
single machine, no shards. At $C = 10{,}000$ and $nprobe = 32$ it scans
$32/10{,}000 \times 100\text{M} \approx 320{,}000$ candidates, which at 96 bytes
each is ~31 MB of sequential ADC scan per query — several milliseconds per core,
and worse than the graph. You would buy a tenth of the memory with roughly ten
times the CPU and a recall ceiling set by the quantizer. That is a good trade if
memory is the binding constraint and a bad one here.

| Option | Memory | Typical latency | Recall ceiling | When it wins |
|---|---|---|---|---|
| Flat int8 | 76.8 GB | seconds | exact | Under ~1M per shard |
| HNSW int8, 8 shards | 103 GB | ~5 ms | very high | The default here |
| IVF-PQ, 1 box | 9.6 GB | ~10 ms | capped by PQ | Memory-bound, recall-tolerant |
| DiskANN, PQ in RAM | 9.6 GB + SSD | ~2-5 ms | high | One box, 100M+, SSD available |

@decide Which index family?
- chose: HNSW over int8 vectors, sharded eight ways, with exact re-ranking of the top few hundred
- over: IVF-PQ on a single machine, which is a tenth of the memory
- why: the recall target is the binding constraint, and PQ puts a ceiling on it that no amount of nprobe recovers; 13 GB per shard on commodity machines is not an expensive way to remove a ceiling
- cost: 10x the memory, a fleet instead of a box, and a fan-out tail problem to manage
- drop it if: the recall target turns out to be 0.9 rather than 0.99, or memory is priced such that ten boxes is genuinely painful -- then IVF-PQ with OPQ, or DiskANN if SSDs are available
@end

# Filters: the part that decides the design

This is where most designs are actually wrong, and the interviewer knows it.

The problem in one sentence: **a graph index searches the whole graph, but the
answer must come from a filtered subset, and the two have nothing to do with
each other.** Three strategies, and the right answer is that selectivity picks
between them.

| Selectivity | Strategy | Why |
|---|---|---|
| Above ~10% | Post-filter with over-fetch | Fetch $k/s$ and discard; cheap and simple |
| ~0.1% to ~10% | Filtered traversal, or a partitioned index | Over-fetch factor becomes absurd; the graph must be filter-aware |
| Below ~0.1% | Pre-filter and brute-force the subset | The subset is small enough to scan exactly |

**Post-filtering** fetches $k/s$ candidates for selectivity $s$ and drops the
ones that fail. At $s = 0.4$ you fetch 2.5x — fine. At $s = 0.02$ you fetch 50x,
and at $s = 0.002$ you fetch 500 candidates for every one you keep, which blows
the latency budget and *still* silently returns fewer than $k$ results when the
over-fetch was not enough. That last failure is the one that reaches production:
the system does not error, it just quietly returns eight results instead of
fifty.

**Pre-filtering** resolves the filter first, then scans the surviving set
exactly. This is excellent when the survivor count is small — and note the
baseline arithmetic above says "small" means up to about 130,000 vectors per
core per 10 ms, which covers far more filters than people expect. For *"this
category, in stock, ships to this region"* on a 100M catalogue, the survivor set
is frequently in the tens of thousands, and an exact scan is both faster and
exactly right.

**Filtered traversal** is the middle band: the graph search itself is made
filter-aware, so the traversal keeps expanding through non-matching nodes for
connectivity but only admits matching nodes to the result heap. The failure mode
to name is **graph disconnection** — if you prune the traversal to matching
nodes only, the filtered subgraph is often disconnected and greedy search gets
trapped in one component, which shows up as recall that collapses for certain
filters and is fine for others.

**Partitioning is the pragmatic answer for low-cardinality, high-traffic
filters.** If nearly every query filters by a top-level category and there are
30 of them, build 30 indexes. Each is smaller, each search is exact with respect
to the filter, and the memory cost is the same total. This does not generalise
to conjunctions of many filters, which is why it is a complement to the above
and not a replacement.

@decide How do filters interact with the index?
- chose: A policy chosen per query by estimated selectivity -- pre-filter and exact scan when it is tiny, filtered traversal in the middle, post-filter with over-fetch when it is loose
- over: One strategy for all filters, usually post-filtering, because it is what the library gives you
- why: over-fetch cost scales as 1/selectivity, so a single strategy is guaranteed wrong at one end of the distribution, and the failure at the tight end is silent under-delivery rather than an error
- cost: a cardinality estimator you have to maintain, and three code paths instead of one
- drop it if: the selectivity distribution is genuinely narrow -- if every filter keeps 30-60% of the corpus, post-filter and go home
@end

!push "How do you estimate selectivity?" The same way a query planner does. Maintain per-value counts for the low-cardinality fields and a sketch for the rest, combine assuming independence, and correct the pairs you know are correlated -- in-stock and category are not independent on a marketplace. Then measure the estimator's error, because the cost of under-estimating is a silent short result set and the cost of over-estimating is a wasted exact scan.

# Freshness: inserts, deletes, and the rebuild you will not avoid

1M updates a day is ~12 writes per second, which sounds trivial and is not,
because of what a delete means to a graph.

**Inserts are fine.** HNSW supports incremental insertion by construction — the
insert is itself a search plus a link. Quality degrades slowly as the
distribution drifts away from what the graph was built on, but 12 per second is
nothing.

**Deletes are the problem.** You cannot really remove a node from an HNSW graph:
other nodes route *through* it, so removing it severs paths and degrades recall
for queries that have nothing to do with the deleted item. So everyone
tombstones — mark deleted, keep the node for routing, filter it from results.
The consequences, which are the things to say:

- The index only grows. Memory rises with churn rather than with corpus size.
- Effective $efSearch$ silently falls, because some fraction of the candidates
  you examine are tombstones that cannot be returned. At 20% tombstones you are
  doing 20% more work for the same recall.
- **Therefore you rebuild on a schedule**, sized from the churn rate. At 1M
  updates a day against 100M items, a tombstone share of 10% accumulates in
  roughly ten days, so a rebuild cadence of one to two weeks per shard, rolled
  across shards so no shard rebuild is ever on the serving path.

**The two-tier trick, which is what actually gets you to seconds of freshness.**
A large static index rebuilt on a schedule, plus a small flat (exact) index
holding everything written since the last rebuild. Query both, merge the
results. The fresh tier holds a few hundred thousand vectors at most, so a flat
scan of it is sub-millisecond and exact, and new items are searchable the moment
they are written. This is the standard design and it is worth drawing.

**Embedding model changes are a different animal entirely.** You cannot mix
vectors from two model versions in one index — they are not in the same space,
and the distances between them are meaningless rather than merely inaccurate. A
new model means re-encoding all 100M items and building a whole new index, then
swapping. Which means:

- Plan for **double the memory** during a migration, or migrate shard by shard.
- Re-encoding 100M items is a batch inference job that takes real hours on real
  GPUs, and it is the long pole, not the index build.
- Run the two indexes side by side and A/B them. The one thing you must never do
  is a partial migration that leaves both versions live in the same index — and
  people do this, usually by accident, when a backfill fails halfway.

!trap "We upgraded the embedding model and recall dropped" is, nine times out of ten, an index that contains two model versions. The vectors are all valid, the index is healthy, every monitor is green, and the distances are nonsense.

# Evaluation without fooling yourself

**Recall is measured against exact search, on your own data.** Take a sample of
real queries — stratified, because the head and the tail behave differently —
compute exact top-$k$ by brute force offline, and measure what fraction the index
returns. This is the only definition of recall that means anything here.

Three ways this is done wrong, and naming them is the senior signal:

- **Against logged clicks.** That measures relevance, which the embeddings
  control. It tells you nothing about the index, and it will happily report that
  a broken index is fine because the clicked item was popular enough to be
  returned anyway.
- **On synthetic random vectors.** Per the instability result, the ground truth
  is meaningless. A friend's benchmark on `numpy.random.randn` has told them
  nothing.
- **At a single operating point.** Recall without a latency is not a number.
  Report the **curve**: recall@10 against p99 latency as $efSearch$ sweeps, on
  your corpus, at your filter distribution. Two indexes that both claim "0.95
  recall" can differ 5x in latency at that recall.

**Then check that recall is the thing you should be optimising.** Run the
end-to-end metric at two or three points on the curve. It is extremely common to
find that recall@100 from 0.92 to 0.98 does not move NDCG at all, because the
ranker reorders the candidates anyway and the six items you were missing were
never going to be shown. That finding is worth a lot of money — it is permission
to take the cheaper index — and you only get it by looking.

**What to monitor in production**, which is a different list from what to
measure offline: p99 by shard (a single slow shard sets the fan-out tail),
tombstone share, recall on a fixed golden query set replayed hourly against a
stored exact ground truth, and the share of queries returning fewer than $k$
results, which is the canary for the filter problem.

# Serving: sharding, tails, and the fan-out multiplier

**Shard by hash, not by category.** Category sharding is tempting because it
makes the common filter free, and it is a trap: the load is as skewed as the
category distribution, and a query without a category filter has to hit
everything anyway. Hash sharding gives uniform load and uniform fan-out, and the
filter problem is solved per shard.

**The fan-out tail multiplier is the number to have ready.** A query that waits
for 8 shards experiences the *maximum* of 8 latencies. If a single shard's p99 is
10 ms, then roughly 8% of queries have at least one shard in its p99 tail, so
the *system's* p99 is close to the shard's p99.9 — not its p99. This is why the
5 ms typical figure above needs 20 ms of budget, and it is the single most
common sizing error in this round.

Mitigations, in the order they pay: **hedged requests** (fire to the replica if
the first has not answered by the p95, cancel the loser — a few percent extra
load buys most of the tail back), **fewer shards** if memory permits, and
**pinning memory** so the index never touches swap.

**Warm-up is not optional.** A freshly started replica with a cold page cache
serves its first thousand queries at disk latency. Take replicas out of rotation
until a synthetic query set has walked the index.

# The follow-up tree

**"Why HNSW and not IVF-PQ?"** Recall ceiling. PQ's reconstruction error puts a
cap on achievable recall that more probing does not remove, and at 13 GB per
shard I am not memory-constrained enough to accept the cap. If the target were
0.9 rather than 0.99, or if this had to run on one machine, I would flip.

**"Why not DiskANN, then? One box, 9.6 GB of RAM."** It is the right answer for a
single-machine deployment at this scale and I would take it seriously. PQ codes
in memory for the traversal, full vectors on SSD for re-ranking, beam search so
the SSD reads are batched — a handful of NVMe reads at ~80 us each, so a
millisecond or two. What it costs is operational: the build is much more
expensive, and you are now exposed to SSD tail latency, which is far less
predictable than DRAM.

**"What does $M$ do?"** Graph degree. Memory is $2M \times 4$ bytes per vector at
the base layer. Higher $M$ raises recall at a given $efSearch$ and raises both
memory and build time. 16 to 48 is the usual range; 32 for 768-d data is a
reasonable default and I would tune it on the recall-latency curve rather than
from a blog post.

**"And $efSearch$ versus $efConstruction$?"** $efConstruction$ is build-time
quality — it costs build time only, and you cannot change it without rebuilding.
$efSearch$ is the query-time knob, it is the recall-latency dial, and it can be
changed per query. Which means you can set it *per segment*: a higher
$efSearch$ for tail queries where recall matters and a lower one for head
queries that are easy, spending your latency where it buys something.

**"The recall is 0.99 offline and users complain. What do you check?"** In order:
whether the index has two embedding versions in it; whether queries are being
short-changed by the filter path (share of queries returning fewer than $k$);
whether the offline sample was drawn from the head while the complaints are
about the tail; and only then the index parameters. The first three are more
often the cause than the fourth.

**"What if the embeddings were not normalised?"** Then it is MIPS, not NN, and
three things change: I check whether the library's "inner product" index is
genuinely MIPS-aware or is cosine underneath; I use anisotropic quantization if
I quantize at all, because plain PQ minimises the wrong loss for inner product;
and I watch for norm-imbalanced clusters if I use IVF, since k-means partitions
by direction while the answer depends on magnitude.

**"How would you halve the memory tomorrow?"** OPQ plus PQ on the base vectors,
keeping the graph, and re-rank the top few hundred with int8 vectors fetched
from a second tier. That is the standard two-tier compression trick: cheap
approximate distances to traverse, accurate distances to order. Expect a point
or two of recall and measure it before committing.

**"Would you ever use LSH?"** For this, no — graphs dominate it empirically and
its space bound is superlinear. Where I would: when I need provable guarantees
for a contractual reason, when the data is streaming and I cannot afford a
build, or for near-duplicate detection, where SimHash over shingles is still the
right tool and is not really the same problem.

# What I would build first

**Week 1 — measure before building.** Sample a few million vectors, compute exact
ground truth for a stratified query sample, and produce the recall-latency curve
for three candidates: flat, HNSW, IVF-PQ. On our data, at our filter
distribution. Every argument after this is grounded or it is a preference.

**Weeks 2-4 — the simplest thing that hits the target.** Eight shards, HNSW over
int8, post-filter with over-fetch, no fresh tier, daily rebuilds. Ship it behind
the existing system and compare end-to-end metrics, not just recall.

**Month 2 — the two things production will demand.** The fresh tier, because
"new items are searchable in seconds" is a product requirement disguised as an
infrastructure one; and the selectivity-aware filter policy, because by then you
will have found the tight-filter queries returning short result sets.

**Month 3 — economise.** Now that the target is being hit and measured, ask
whether the end metric actually distinguishes 0.99 from 0.95 recall. If it does
not, move to OPQ-PQ and take the memory back. This is the sequencing that
separates senior from thorough: you buy the expensive version first to establish
the ceiling, then you find out how much of it you actually needed.

# The whiteboard, in order

| Minute | Draw | Say while drawing |
|---|---|---|
| 5 | **The memory table** | Five rows, one per representation. Everything hangs off it |
| 8 | **The flat-scan ceiling** | 130k vectors per core per 10 ms, so under 1M per shard needs no index |
| 12 | **The three reductions** | NN, cosine, MIPS, and the +1 dimension |
| 20 | **HNSW: layers and a greedy path** | Then M, efConstruction, efSearch, and what each one costs |
| 25 | **The latency chain** | 3,000 candidates, 80 ns a miss, 0.3 ms a shard, and why it is memory-bound |
| 32 | **The selectivity bands** | Three strategies and the 1/s over-fetch curve |
| 38 | **The two-tier freshness index** | Static plus fresh, merged, with the rebuild cadence |
| 45 | **The recall-latency curve** | Two indexes at "0.95 recall" differing 5x in latency |

# Ten things that would lose this round

**On arithmetic**

- Naming an index before sizing the corpus.
- Not knowing that 100M x 768 fp32 is 307 GB, or being unable to derive it.
- Forgetting the graph's own memory, which at M = 32 is 256 bytes a vector.
- Sizing to the mean latency and being surprised by the fan-out tail.

**On theory**

- Not knowing that recall is a property of the data, and quoting a datasheet
  number for your corpus.
- Benchmarking on random vectors.
- Treating normalisation as free when the norms carry popularity signal.

**On operations**

- Claiming HNSW supports deletion.
- Having no rebuild cadence, and no answer for where tombstones go.
- Designing a single filter strategy and having nothing to say when a 0.1%
  selective filter silently returns four results.
