# Search / RecSys Depth Track — Four Weeks

For search, recommendation, ads, marketplace-ranking, and retrieval-heavy
platform loops. These loops look different from frontier-lab loops in three
ways: the design round uses named templates (recsys, ads, trust & safety), a
standalone "ML fundamentals" round usually survives, and classical ML — GBDT
internals and experimentation statistics — is load-bearing rather than
insurance. That is why this track starts in Volume IV.

**Budget: 3–4 hours a day.** Five study days, one mock day, one day off, per
week.

## Before day 1 (one hour)

Read `kernel/part0_kernel.tex` once — the 26 principles. Three of them are this
track in general form: **E4, precompute what factorizes and rerank the rest** —
that is the entire retrieval funnel; **A3, learn by comparison when labels run
out** — that is why the two-tower model trains the way it does; and **Group F,
proxies** — that is the offline-online gap before it has a name. Then start
`kernel/core50.tex` at five items a day, spoken, marked red/amber/green.

The quantitative layer runs alongside: `kernel/numbers_card.tex` §5,
"Retrieval, embeddings, and ranking," is this track's home section — storage per
1M 768-d vectors across fp32/fp16/int8/PQ with the quality cost of each, HNSW
graph overhead, funnel and reranker costs, latency budgets. Pair it with §6
("Statistics and evaluation," which works the 2%-CTR / 1%-lift sample size to a
number) in weeks 1 and 3, and §4 ("Inference and serving") in week 4.

## Ordering

This follows the Program Map's **search and recommendation depth** path
exactly: [CML 2] and [CML 6] first, then Volume III cover to cover, then
[DL 6–8] for embeddings and contrastive training, [NLP 9] for the RAG pipeline,
[DL 21] for the context-versus-retrieval decision, and [DL 19] for serving.

One addition: **[NLP 3]** on day 22, because the canonical-home table makes it
the home for embedding theory while [DL 6] keeps production diagnostics. You
want both halves before the two-tower training questions in week 4.

Volume III is the spine and is read cover to cover — all eight chapters, 92
questions. The other volumes are read for the specific sections the path names.

## The standard this track is graded against

The recurring failure in these loops is not ignorance of a method. It is the
offline-online gap: a candidate who can define NDCG but cannot explain why their
reranker won offline and lost online. Four questions in the program are the same
question wearing different clothes —

- [SR 1] "Your new L2 reranker improves offline NDCG by 4%, but online conversions are flat"
- [SR 4] "Your new reranker improved offline NDCG by 4%, but online CTR dropped"
- [SR 7] "Your reranker improved offline NDCG@10 by 3% on the judgment set, but the A/B test shows flat CTR"
- [SR 6] "Your CTR model's offline AUC improved by 0.5% but online revenue dropped 3%"

— plus [NLP 3]'s "+4 points Recall@100 offline but loses the online A/B test"
and [DL 22]'s "online metrics diverge from offline after 2 weeks." If, by day 20,
you cannot give six structurally distinct causes with a discriminating check for
each, the track has not worked. Keep a running list from day 3 and add to it
every time a chapter names a new mechanism.

---

# Week 1 — The classical base and the search funnel

## Day 1 — [CML 2] Trees and Ensembles

The GBDT canonical home. LambdaMART's ranking-specific view is in [SR 5] on day
11; everything underneath it is here.

**Drill:**
- L5 `***` — "Derive the optimal leaf weight and the split-gain formula in XGBoost"
- L5 `***` — "Explain gradient boosting to someone who already understands gradient descent"
- L5 `***` — "Random forest vs. gradient boosting—how do they differ, and when do you reach for each?"
- L6 `***` — "XGBoost vs. LightGBM vs. CatBoost: 600K rows, 45 features of which 14 are categorical including a 40K-cardinality merchant_id. Choose and defend"
- L6 `***` — "Why does a gradient-boosted tree beat your MLP on this tabular dataset—and when would it not?"
- L6 `**` — "Your LambdaMART ranker's training NDCG keeps improving, but validation NDCG peaked at tree 200 and is now degrading. Walk me through your response"
- L6 `**` — "How does CatBoost avoid target leakage with categorical features, and what is 'ordered boosting' fixing?"
- L6 `**` — "Your random forest's impurity-based feature importance says session_id_hash is the top feature. The PM wants to build product strategy around the ranking. What do you tell them?"
- L7 `*` — "Design the serving story for a GBDT fraud model: p99 model latency under 2 ms at 20K QPS, with a compliance requirement that risk never decreases as chargeback count increases"

**Done when.** The leaf weight and split gain are derived on paper from the
second-order objective, from memory, in under five minutes.

## Day 2 — [CML 6] Experimentation and Causal Inference

The A/B statistics canonical home. [SR 7] keeps interleaving and the
ranking-specific protocol; this is everything else.

**Drill:**
- L5 `***` — "We want to detect a 1% relative lift on a 2% click-through rate. How many users do we need, and how long should we run?"
- L6 `***` — "Your test came back flat, but you are confident the feature helps. What do you do?"
- L6 `***` — "Your treatment shows +2% on the primary metric, but the SRM check fired. What now?"
- L6 `***` — "Your marketplace A/B test shows +2% GMV in treatment. Why might the true effect be smaller, or negative?"
- L6 `**` — "Explain CUPED to a PM, and quantify what it buys us"
- L6 `**` — "Thompson sampling, UCB, or epsilon-greedy—mechanics, and when would you use a bandit instead of an A/B test?"
- L6 `**` — "Why not send retention offers to the users with the highest churn probability?"
- L7 `**` — "We cannot randomize prices. How would you estimate price elasticity from historical data?"
- L7 `**` — "Design the experimentation platform for a 200-engineer organization"

**Done when.** The sample-size calculation runs to a number and a duration out
loud, and you can name three distinct interference mechanisms in a marketplace
test.

## Day 3 — [SR 1] Search Systems and Query Understanding

The funnel and the front of it: intent, spell correction, synonyms, query
segmentation, head/torso/tail.

**Drill:**
- L5 `***` — "Walk me through what happens, end to end, when a user types a query into a large-scale search engine"
- L6 `***` — "Design the query understanding system for an e-commerce search engine"
- L6 `***` — "Your new L2 reranker improves offline NDCG by 4%, but online conversions are flat. Walk through your diagnosis"
- L6 `**` — "Why is search a multi-stage funnel at all? Why not run your best model over the whole corpus for every query?"
- L6 `**` — "Design spell correction for a commerce search engine. What breaks a naive dictionary approach?"
- L6 `**` — "How do you mine synonyms for query expansion without a hand-built ontology?"
- L7 `**` — "When should a query understanding signal change retrieval, and when should it only be a ranking feature?"
- L7 `**` — "You inherit a search system that is a single BM25 stage with a 15% zero-result rate. Sequence your first three quarters of investment"
- L5 `**` — "Explain the head/torso/tail decomposition of query traffic. Why must strategy differ by segment?"

**Done when.** You can justify the funnel from a candidate-count × cost-per-doc
budget, not from convention. Start the offline-online cause list today.

## Day 4 — [SR 2] Lexical Retrieval and Inverted Indexes

BM25 mechanically, postings and compression, WAND, index partitioning, SPLADE,
the hybrid split.

**Drill:**
- L5 `***` — "Walk me through, mechanically, how BM25 over an inverted index returns the top-10 results from a billion documents in under 50 ms"
- L5 `***` — "What do BM25's k_1 and b actually control, and when would you change them from the defaults?"
- L6 `***` — "Lexical vs. dense vs. hybrid—argue the retrieval split for a marketplace search engine"
- L6 `**` — "Explain WAND and block-max WAND. Why are they safe, and when does the pruning stop helping?"
- L6 `**` — "Explain SPLADE. What do the log-saturation and the FLOPS regularizer each do, and how does it compare to BM25 and dense retrieval at serving time?"
- L6 `**` — "A search for 'red dress' ranks a product titled 'Red' with 'dress' scattered through a long description above an exact 'Red Dress' title match. Diagnose and fix"
- L6 `**` — "Index-time vs. query-time synonyms—walk through the trade-offs and give your production policy"
- L7 `**` — "You must index 10B documents. Design the partitioning, tiering, and caching. Why did document partitioning beat term partitioning?"

**Done when.** You write the BM25 scoring function from memory and say what each
term does to a long document, and you can explain why block-max WAND is a *safe*
optimization rather than an approximation.

## Day 5 — Consolidation: the two week-1 designs

No new chapters. Two designs, out loud, timed, using the five moves from
[DL 29] §"ML System Design at Staff Level" (constraint discovery, quantify
early, propose then iterate, failure modes, the eval plan):

- 45 min: "You must index 10B documents. Design the partitioning, tiering, and caching."
- 45 min: "You inherit a search system that is a single BM25 stage with a 15% zero-result rate. Sequence your first three quarters of investment."

Then 45 min re-reading the answers you missed on days 1–4, and 30 min on the
XGBoost split-gain derivation until it is automatic.

## Day 6 — Mock

- **60 min.** Design: "Design the query understanding system for an e-commerce
  search engine." Have your partner mutate a constraint at minute 30 — the
  latency budget drops 10× — and re-run the arithmetic rather than defending.
- **45 min.** Rapid-fire: fifteen `***` questions from days 1–4, two minutes
  each, spoken, scored.

## Day 7 — Off

---

# Week 2 — Retrieval

## Day 8 — [SR 3] Vector Retrieval Theory, part 1

Sections through LSH: the NN/cosine/MIPS reductions, why high dimensions are
hard, the brute-force baseline, trees and branch-and-bound, LSH guarantees.

**Drill:**
- L5 `**` — "Why doesn't a k-d tree work for 768-dimensional embeddings, and what minimal changes rescue tree-based indexes?"
- L6 `*` — "You are offered an LSH family with p_1 = 0.8, p_2 = 0.5 at your target radius. Size an index for 100M vectors and decide whether to use it"
- L6 `*` — "Your ANN benchmark on synthetic Gaussian vectors shows terrible recall at any latency, but the same index on real embeddings is fine. Explain"
- L6 `**` — "A recsys team L2-normalizes item embeddings from a dot-product-trained two-tower model so they can reuse a cosine HNSW index. Ranking quality drops. Diagnose and fix"

**Done when.** You can state the MIPS-to-cosine reduction and say exactly what
it costs, and give the intrinsic-dimension argument for why real embeddings
index better than Gaussian noise.

## Day 9 — [SR 3] part 2

Graph indexes (HNSW, DiskANN), IVF, quantization (PQ, OPQ, anisotropic), random
projections, and the recall–latency–memory triangle.

**Drill:**
- L5 `***` — "How does HNSW work, and why is it fast?"
- L6 `***` — "Index 500M × 768-d embeddings and serve top-100 under 15 ms p99 on a machine with 64 GB RAM—walk me through the design"
- L6 `**` — "Your ANN recall dropped after a reindex—same code, same corpus size. Debug it"
- L6 `**` — "Your HNSW recall@10 drops from 0.95 to 0.6 when users apply a metadata filter matching 2% of the corpus. Why, and what are the fixes?"
- L5 `**` — "Walk me through how a PQ index computes distances without decompressing anything, and where the time goes"
- L6 `**` — "Tune an IVF index for 100M vectors: how many clusters? And what do you do when latency headroom remains but recall plateaus as you raise nprobe?"
- L7 `*` — "When and why does ScaNN's anisotropic quantization beat plain PQ?"
- L7 `*` — "Why does greedy search on a proximity graph find the nearest neighbor at all? What breaks in high dimensions, and which practical graph has worst-case guarantees?"

**Done when.** The 500M × 768-d design runs on arithmetic: raw float bytes, the
compression ratio PQ must achieve to fit 64 GB, the resulting recall cost, and
the rerank stage that buys it back. Box-drawing without those numbers is the
archetypal reject.

## Day 10 — [SR 4] Neural Retrieval and Reranking

Bi-encoders and their training, cross-encoders, late interaction, fusion, and
the canonical semantic-search design question.

**Drill:**
- L5 `***` — "Bi-encoder vs. cross-encoder: why not cross-encode everything, and what exactly does the cross-encoder buy?"
- L5 `***` — "You run BM25 and dense retrieval in parallel. How do you combine them—and why does 0.5·BM25 + 0.5·cosine fail?"
- L6 `***` — "Design semantic search over 100 million documents, end to end"
- L6 `***` — "Your new reranker improved offline NDCG by 4%, but online CTR dropped. Walk through the diagnosis"
- L6 `**` — "Why do bi-encoders train with such large batches? Explain in-batch negatives and the logQ correction"
- L6 `**` — "Design the training recipe for a domain bi-encoder from your search logs. Walk through negatives, denoising, and distillation"
- L6 `**` — "When does late interaction (ColBERT) earn its cost? Walk through the storage math"
- L6 `**` — "Dense retrieval tanks on queries containing part numbers. Fix it without abandoning dense retrieval"
- L7 `**` — "A listwise LLM reranker beats your production cross-encoder by 3 NDCG points offline—at 200× the cost. What ships?"

**Done when.** You can write the InfoNCE objective with in-batch negatives *and*
the logQ correction, and say what the correction is fixing. The ColBERT question
should be answered in bytes per document.

## Day 11 — [SR 5] Learning to Rank, part 1

The LTR problem, pointwise/pairwise/listwise, and RankNet → LambdaRank →
LambdaMART.

**Drill:**
- L5 `**` — "Compare pointwise, pairwise, and listwise learning to rank. When is each the right choice?"
- L6 `***` — "Explain LambdaRank to a strong engineer who knows GBDT but has never done ranking"
- L6 `***` — "GBDT or neural network for your L2 ranker? Decide for (a) a commerce search engine with rich engineered features, (b) a feed ranker over user history and item IDs"
- L5 `**` — "When would you insist on a pointwise objective even though the product is a ranked list?"
- L5 `**` — "Walk me through constructing an LTR training set from a marketplace's search logs: labels, negatives, and splits"

**Done when.** You can write RankNet's loss, show the lambda factorization of
its gradient, and say precisely where the |ΔNDCG| weight enters and why that
step is not a gradient of anything.

## Day 12 — [SR 5] part 2

Position bias, propensity estimation, unbiased LTR, and where the loop silently
breaks.

**Drill:**
- L6 `***` — "Your click-trained ranker keeps favoring the items that have historically sat at position 1—better new items never rise. Diagnose and fix"
- L6 `**` — "How would you estimate position-bias propensities without degrading the user experience?"
- L6 `**` — "Your new LTR model improves overall NDCG but tail-query relevance regresses. Why, and what do you change?"
- L6 `**` — "Editorial judgments say document A beats B for this query; click data says B massively outperforms A. Which do you trust, and what do you do?"
- L7 `**` — "Design the end-to-end unbiased-LTR loop for a search product—logging through training through evaluation—and tell me where it silently breaks"
- L7 `*` — "Your IPS-weighted training runs are unstable—a few examples dominate the gradient and validation NDCG oscillates. What is happening and what are your options?"

**Done when.** You can write the IPS estimator for ranking, name its variance
problem, and give two fixes (clipping and the self-normalized form) with what
each costs in bias.

## Day 13 — Mock

- **60 min.** Design: "Design semantic search over 100 million documents, end to
  end." Quantify: corpus bytes, index memory, QPS per replica, the reranker's
  latency budget, and the cost per 1K queries.
- **45 min.** The four offline-online questions from days 3, 10 and week 3,
  answered back to back. Your cause list should be six items by now.
- **30 min.** RankNet/LambdaRank and the IPS estimator on paper.

## Day 14 — Off

---

# Week 3 — Recommendation, evaluation, production

## Day 15 — [SR 6] Recommendation Systems, part 1

Fundamentals, collaborative filtering, deep recommenders and two-tower,
sequential models, generative retrieval with semantic IDs, and LLMs in the
stack. [SR 6] is the largest chapter in Volume III (1,983 lines, 17 questions);
it gets two days.

**Drill:**
- L6 `***` — "Design a recommendation system for a video streaming platform with 100M users and 1M videos"
- L5 `***` — "How do you handle the cold start problem for new users and new items?"
- L5 `***` — "Two-Tower vs. interaction-based models for candidate generation—when would you use each?"
- L6 `**` — "You run retrieval for a marketplace with 200M listings and heavy daily churn. Two-tower + ANN or generative retrieval with semantic IDs?"
- L6 `***` — "Where would you use an LLM in a recommendation stack today—and why won't it replace your ranker?"
- L7 `*` — "Design the embedding infrastructure for a recommendation system with 1B items"
- L6 `***` — "Estimate the memory required for DLRM embedding tables with 100M users, 10M items, and 1000 categorical features"

**Done when.** The DLRM table estimate is done in your head to an order of
magnitude, and you volunteer that in recsys the embeddings *are* the model —
the MLP on top is under 0.1% of the parameters.

## Day 16 — [SR 6] part 2

Multi-task models for ads, pCTR/pCVR and calibration, the ads pipeline, embedding
table systems, system design, bandits in the loop.

**Drill:**
- L6 `***` — "Design the ranking model for an e-commerce product ads system"
- L6 `**` — "Shared-bottom vs. MMoE vs. PLE for multi-task ads modeling—give me a decision framework"
- L6 `**` — "Explain the sample selection bias problem in CVR prediction. How does ESMM solve it?"
- L5 `***` — "Why do recommendation models need calibration? What happens if pCTR is systematically over-confident?"
- L6 `***` — "Your recommendation model has great offline metrics but poor online performance. What went wrong?"
- L6 `***` — "Your CTR model's offline AUC improved by 0.5% but online revenue dropped 3%. Diagnose"
- L6 `**` — "Your ads ranking model needs to score 100 candidates in <50ms. What architecture constraints does this impose?"
- L7 `**` — "Design a real-time personalized notification system"
- L7 `**` — "Your recommendation model shows high engagement but users are churning. What is happening?"
- L7 `*` — "Calculate the daily training data volume for an ads system serving 1B impressions/day"

**Done when.** You can say why a 0.5% AUC gain can cost revenue in at least
three distinct ways — calibration drift, position/selection bias in the training
labels, and a shift in which slice the gain came from.

## Day 17 — [SR 7] Search and RecSys Evaluation, part 1

Binary-relevance metrics, graded relevance and NDCG, the judgment supply chain,
beyond-accuracy metrics, offline protocol design.

**Drill:**
- L5 `***` — "Derive NDCG from first principles, then compute NDCG@3 for a ranking with grades (3, 0, 2), exponential gain, given the ideal available grades are (3, 2, 0)"
- L5 `**` — "You have binary judgments only. When do Precision@k, Recall@k, MRR, and MAP each answer the right question—and construct a case where two of them disagree about which of two systems is better"
- L6 `**` — "A colleague evaluates a new sequential recommender with a random 80/20 interaction split and HR@10 against 100 sampled negatives, and reports a 12% win. What is wrong, and what protocol do you require before believing it?"
- L6 `**` — "Your new recommender lifts NDCG and short-term engagement, but intra-list diversity drops, catalog coverage falls, and exposure Gini rises 6 points. Do you ship it—and how should the evaluation have been set up so this is not a debate?"

**Done when.** You compute that NDCG@3 to a number, on paper, with exponential
gain, and can say why the ideal ranking uses the *available* grades.

## Day 18 — [SR 7] part 2

The offline–online gap, online experimentation for ranking (interleaving,
off-policy estimation), LLM judges, and evaluation-program design.

**Drill:**
- L6 `***` — "Your reranker improved offline NDCG@10 by 3% on the judgment set, but the A/B test shows flat CTR. Walk me through your investigation"
- L6 `**` — "Interleaving vs. A/B testing for a ranking change—how does team-draft interleaving work, why is it more sensitive, and when would it mislead you?"
- L6 `**` — "You replace a lexical retriever with a dense retriever. Offline NDCG on the existing judgment set drops. Is the new system worse?"
- L7 `**` — "Without launching it, estimate what CTR a new ranking policy would achieve, using only logs from the current system. Derive the estimator, its requirements, and its failure modes"
- L6 `**` — "You want to replace most crowd relevance labeling with an LLM judge. Design the program so the labels are trustworthy—and tell me what stays human"
- L7 `*` — "You own relevance for a search org of several teams. Design the evaluation program: what gets measured, at what layer, on what cadence—and how does a change get to ship?"

**Done when.** The off-policy question is answered with the estimator written
down (IPS, then its self-normalized and doubly-robust variants), its support
requirement stated, and its failure mode named.

## Day 19 — [SR 8] Production Retrieval and RAG Integration

Latency budgets, index freshness, deletes in HNSW, embedder migrations, semantic
caching, chunk- vs. doc-level vectors, and cost reduction.

**Drill:**
- L5 `***` — "You own a search endpoint with a 250 ms server-side p99 SLO. Walk me through the latency budget—where does the time go, and what do you cut when you're over?"
- L6 `***` — "Your search index must reflect catalog changes within 60 seconds—design it"
- L6 `***` — "Relevance regressed after last week's embedder upgrade—find it"
- L5 `**` — "Why are deletes and updates hard in HNSW, and what does your vector database actually do when you call delete()?"
- L6 `**` — "You have no labels in production. How do you know your retrieval quality hasn't regressed—and how do you find out fast?"
- L6 `**` — "Traffic to your LLM-answer product is expensive. Design a semantic cache—and tell me how it goes wrong"
- L6 `**` — "You're building the index behind a RAG product over 100M documents. Chunk-level or doc-level vectors—and what does the choice do to your index?"
- L7 `**` — "Cut retrieval serving cost 5× without visible quality loss"
- L7 `**` — "A new embedding model tops the leaderboard. Do you re-embed your 100M-document corpus? Walk through the decision and the cost"

**Done when.** The 250 ms budget is decomposed into named stages with
millisecond allocations, and you volunteer the fan-out tail-amplification
argument for why p99 gets worse as you add shards.

## Day 20 — Mock

- **60 min.** Design: "Design a recommendation system for a video streaming
  platform with 100M users and 1M videos." End with the eval plan; if you did
  not get there, you failed the round.
- **45 min.** Design: "Cut retrieval serving cost 5× without visible quality
  loss" — sequence the levers with an expected win and a risk per lever.
- **30 min.** The offline-online cause list, recited from memory. Six causes,
  six discriminating checks.

## Day 21 — Off

---

# Week 4 — Embeddings, contrastive training, RAG, serving

The Program Map's tail: [DL 6–8], [NLP 9], [DL 21], [DL 19]. These are the
chapters that explain *why* the two-tower model in [SR 4] and [SR 6] trains the
way it does, plus the retrieval-adjacent LLM surface these loops now include.

## Day 22 — [DL 6] Embeddings + [NLP 3] Word Representations

[NLP 3] is the canonical home for embedding theory; [DL 6] keeps the production
diagnostics — collapse, spectral checks, tokenization interaction, memory
arithmetic. Read [NLP 3]'s objective sections and [DL 6] in full (463 lines).

**Drill:**
- [DL 6] L6 `***` — "You are putting 100M item embeddings behind a vector index. Which embedding-side choices do you make—dimension, precision, refresh—and how would you know if one of them silently cost you recall?"
- [DL 6] L6 `**` — "What is embedding collapse and how do you detect/prevent it?"
- [DL 6] L5 `***` — "How does subword tokenization (BPE) interact with embedding quality?"
- [DL 6] L5 `***` — "How would you debug poor embedding quality?"
- [NLP 3] L5 `***` — "Explain the Word2Vec Skip-gram model. What is the training objective? What is negative sampling and why is it needed?"
- [NLP 3] L6 `**` — "Explain the connection between Word2Vec, GloVe, and matrix factorization on co-occurrence statistics. Why does this connection matter?"
- [NLP 3] L6 `**` — "How would you choose an embedding model for a production semantic search system?"
- [NLP 3] L6 `***` — "Your new embedding model wins offline—+4 points Recall@100 on your evaluation set—but loses the online A/B test. Walk me through your investigation"

## Day 23 — [DL 7] Similarity and Metric Learning Architectures

The canonical home for metric-learning training. [SR 4] kept the retrieval-system
recipes; this is the objective-level treatment. Note that ANN indexes are *not*
here — [SR 3] owns them and you covered that on day 9.

**Drill:**
- L5 `***` — "Siamese vs Triplet vs Two-Tower—decision framework for a new retrieval project"
- L5 `***` — "Your two-tower model retrieves documents that share keywords with the query but aren't actually relevant. How do you fix this?"
- L6 `**` — "Your two-tower model's recall@10 is good but precision@10 is poor. What's wrong?"
- L5 `***` — "Cross-encoder vs bi-encoder—when is the quality gap worth the latency cost?"
- L6 `**` — "Hard negative mining strategies—how to choose and when each fails"
- L5 `***` — "Cosine similarity vs dot product vs L2 distance for embeddings—when each?"
- L6 `**` — "Estimate the QPS of a retrieval system using HNSW on 100M documents. Then add a cross-encoder reranker over the top 100—what changes?"
- L7 `*` — "Your contrastive model produces good embeddings for frequent items but poor for rare items. Why?"

**Done when.** The hard-negative answer names the failure mode of each strategy,
including why the hardest negatives are often mislabeled positives.

## Day 24 — [DL 8] Self-Supervised Learning

Contrastive learning at the objective level: InfoNCE and its mutual-information
connection, temperature, why non-contrastive methods do not collapse,
augmentation policy.

**Drill:**
- L6 `**` — "Explain the InfoNCE loss and its connection to mutual information. Why does temperature matter?"
- L5 `***` — "Why do non-contrastive methods like BYOL work without negative samples? Shouldn't they collapse to a trivial solution?"
- L5 `***` — "SimCLR vs BYOL vs MAE—give me a decision framework for choosing between them"
- L5 `**` — "What makes a good augmentation policy for contrastive learning, and how does the choice of augmentations define the learned representation?"
- L6 `***` — "Design an SSL pre-training pipeline for a dataset of 10M unlabeled images and 100K labeled images"
- L6 `**` — "Your SSL model's downstream performance plateaus despite increasing pre-training data from 1M to 10M images. What could be going wrong?"

**Done when.** You can say what temperature does to the gradient's weighting of
hard negatives, in one sentence, without hedging.

## Day 25 — [NLP 9] RAG + [DL 21] context vs. retrieval

[NLP 9] is the RAG pipeline's canonical home; [SR 8] (day 19) owned the
retrieval stage; [DL 21] owns the context-versus-retrieval decision.

**Drill:**
- [NLP 9] L5 `***` — "Design a RAG system for a customer support chatbot that answers questions using your company's knowledge base"
- [NLP 9] L5 `***` — "What chunking strategies exist for RAG? How do you choose chunk size?"
- [NLP 9] L5 `***` — "How would you evaluate a RAG system end-to-end? What metrics would you track?"
- [NLP 9] L5 `***` — "Explain the bi-encoder vs. cross-encoder tradeoff in retrieval. How would you use both in a production system?"
- [NLP 9] L6 `**` — "Your RAG system is returning correct documents but the LLM's answers are still wrong. What would you investigate?"
- [NLP 9] L6 `**` — "What is HyDE (Hypothetical Document Embeddings)? When would it help and when would it fail?"
- [DL 21] L5 `***` — "Long context window vs RAG—when do you choose each?"
- [DL 21] L6 `***` — "Estimate the cost per query for a RAG system vs a long-context LLM (128K)"

## Day 26 — [DL 19] Inference Optimization and LLM Serving

Read for the serving layer these loops need: the latency and cost model behind
a cross-encoder reranker, an LLM-answer product, or a semantic cache. The
frontier-lab depth (parallelism strategies for a 400B model) is optional here.

**Drill:**
- L7 `***` — "Design an LLM serving stack that handles 1,000 queries per second with a p99 latency of 2 seconds for a 70B parameter model"
- L5 `***` — "Explain speculative decoding. When would you use it and when would you not?"
- L6 `***` — "KV cache memory is your bottleneck—you are running out of GPU memory and dropping requests. What do you do?"
- L6 `**` — "Your LLM serving system's p99 latency is 10× the p50. Diagnose"
- L6 `***` — "Tensor parallelism vs pipeline parallelism for inference—what is your decision framework?"

**Done when.** You can price an LLM-answer product per 1,000 queries and say
which term dominates.

## Day 27 — Final mock

- **60 min.** "Design the ranking model for an e-commerce product ads system,"
  with a mid-round constraint mutation.
- **45 min.** "Design semantic search over 100 million documents, end to end,"
  a second time, and compare against your day-13 recording. The delta is your
  progress.
- **30 min.** Derivations on paper: XGBoost split gain; NDCG; RankNet's lambda
  factorization; the IPS estimator.
- **30 min.** Your two questions per interviewer, per [DL 29] §"Your Questions
  Are Data."

## Day 28 — Off, then [`night-before.md`](night-before.md)

---

# Gaps you should know about

- **Volume IV is six-eighths unwritten.** Only [CML 2] and [CML 6] exist. For
  these loops that is a real hole: [CML 5] (leakage, cross-validation,
  interpretability) and [CML 3] (clustering, PCA) are standard platform-loop
  material, and [CML 8] (the from-scratch classical coding canon) would have
  been week 1's coding drill. Chapters 1, 3, 4, 5, 7, and 8 are scaffolds with
  `% TODO` markers — do not send yourself there.
  Use `volumes/conventional-ml/docs/coverage-spec.md` as the checklist for what
  to prepare elsewhere; it lists the must-know subtopics and derivations for all
  eight chapters but teaches none of them. For coding, [DL 28]'s classical
  section (logistic regression, k-means, gradient boosting with stumps) and the
  matching drills do cover the classical implementables.
- **Volume III has no coding round of its own.** The search/recsys
  implementables — BM25 over an inverted index, NDCG@k, two-tower in-batch
  softmax, HNSW-style greedy graph search — live in [DL 28] and
  `drills/canon_b_inference_retrieval.py`. See
  [`coding-week.md`](coding-week.md) day 5.
