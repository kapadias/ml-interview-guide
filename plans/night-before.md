# The Night Before

A 2.5-hour final pass. Not new material — a recall pass over what you already
did, plus the two cards you will keep beside you tomorrow.

The blocks below are counted. Do them in order and stop when they end. If you
have only two hours, the cut is marked at the bottom.

| Block | Minutes | What |
|---|---:|---|
| 1 | 25 | The kernel, spoken |
| 2 | 20 | Four derivations on paper |
| 3 | 45 | The `***` questions for your loop type |
| 4 | 25 | The numbers card |
| 5 | 20 | What to do in the room — write the two cards |
| 6 | 15 | Logistics, your questions, and stop |
| | **150** | **2 h 30 m** |

Do not read a chapter tonight. Nothing you learn in the next three hours will
be load-bearing tomorrow, and displacing sleep to acquire it is a bad trade.

---

## Block 1 — The kernel, spoken (25 min)

Source: `kernel/part0_kernel.tex`, and specifically its closing section, **"The
Kernel on One Page"** — a table of 26 rows, each a principle name and the
half-line that recovers it. That table was written for tonight.

Work down it. Read the name, say the principle aloud in one sentence, *then*
read the half-line to check yourself. About 55 seconds each. Mark — do not
study — any you cannot state; that mark tells you which follow-up to steer away
from tomorrow, which is a legitimate use of one evening.

(The longer form, with an instantiation from each volume per principle, is
`review/first-principles/knowledge-kernel.md`. Do not read it tonight.)

The six groups, so you can recover the list under pressure if the table is not
in front of you:

- **A — Objectives (4).** Read every loss as a likelihood; start the derivation
  at the logits (`p − y`); learn by comparison when labels run out; generate by
  learning a transport.
- **B — Scale (4).** Spend compute as `6ND`; extrapolate the loss, not the
  benchmark; distrust the bias–variance U-curve; spend inductive bias only where
  data is scarce.
- **C — Optimization (3).** `η/B` is the temperature; Adam as a diagonal
  preconditioner; keep the residual highway clean.
- **D — The machine (6).** Ask compute-bound or memory-bound, always; count
  bytes per parameter; do the KV-cache arithmetic; separate prefill from decode;
  buy range before precision; split the model three ways.
- **E — Architecture and adaptation (5).** Attention as a soft lookup; inject
  position deliberately; blame the tokenizer before the model; precompute what
  factorizes and rerank the rest; adapt in a low-rank subspace.
- **F — Proxies (4).** Assume the proxy will be gamed; bound the optimization
  pressure; choose the metric as a modelling decision; assume training and
  serving disagree.

## Block 2 — Four derivations, on paper (20 min)

Five minutes each, from a blank sheet, narrating. These are the ones a "derive
it" follow-up actually reaches for.

1. **Scaled dot-product attention and the 1/sqrt(d_k) variance argument.**
   ([NLP 5] — `Scaled Dot-Product Attention`, `Variance of the Dot Product`)
2. **The softmax cross-entropy gradient reducing to `p − y`.**
   ([DL 1] — `Softmax Cross-Entropy Gradient`)
3. **`C ≈ 6ND` and the Chinchilla allocation** — `N_opt = sqrt(C/120)`,
   `D = 20N`. ([DL 2] — `Chinchilla Optimal Scaling`)
4. **The KV-cache formula `2·L·n_kv·d_head·bytes·s`, and the 16-bytes-per-
   parameter training ledger.** ([DL 19] — `KV Cache Memory`; [DL 15])

If you are on the search/recsys track, swap 3 and 4 for **NDCG** ([SR 7] —
`DCG and NDCG`) and **the XGBoost optimal leaf weight and split gain**
([CML 2] — `The Classic Derivation: Optimal Leaf Weights and Split Gain`).

Any derivation that takes more than five minutes tonight is one to avoid
volunteering tomorrow. Do not try to fix it now.

## Block 3 — The `***` questions for your loop (45 min)

Twenty-two questions, two minutes each, spoken, out loud, no notes, no reading
the answers. Mark misses; do not repair them. This block is a rehearsal of
speaking, not a study session — the round is spoken and writing builds none of
it.

*If you ran the Core Fifty* (`kernel/core50.tex`) during your plan, spend this
block on its closing checklist instead — cover the entries, work down the
names, speak ninety seconds on each, and dwell only on your reds. That is what
the tier was built for and it is a better use of 45 minutes than a fresh list.
Otherwise, pick the one list below that matches your loop.

### Frontier lab (OpenAI/Anthropic-class)

1. [DL 5] Compare MHA, GQA, MQA, and MLA for KV-cache efficiency. Why does MLA need a decoupled RoPE key?
2. [DL 5] Explain Flash Attention. Why is it faster even though it does the same computation?
3. [DL 5] Calculate the KV cache memory for serving LLaMA 70B at 128K context length
4. [DL 5] Estimate the memory required to train a 7B parameter model with batch size 1, sequence length 2048
5. [NLP 5] Why do we scale by d_k? Derive the variance argument
6. [DL 2] Estimate the compute-optimal model size for a training budget of 10^22 FLOPs using Chinchilla scaling laws
7. [DL 15] Data parallelism vs. tensor parallelism vs. pipeline parallelism—what is your decision framework?
8. [DL 15] How would you scale training of a 70B parameter model across 256 GPUs?
9. [DL 15] Your 70B pretraining run diverges at 200B tokens. Give a full differential diagnosis, ordered by prior probability, with the cheapest discriminating experiment for each
10. [DL 26] I give you an op—say, a LayerNorm over a [16,384 × 8,192] BF16 tensor, or a 4096^3 GEMM. Is it compute-bound or bandwidth-bound on an H100? Show your method
11. [DL 26] Your training job shows the GPU at 20% utilization. Diagnose it
12. [DL 26] Estimate the maximum tokens/second for a 70B dense model on 8×H100—for training, and for inference
13. [DL 25] Walk me through what happens between "we have a Common Crawl snapshot" and the first training step
14. [DL 25] You own data for the next frontier pretraining run. Design the pipeline and the ablation program
15. [DL 17] Explain GRPO. Why did it replace PPO's critic for reasoning RL, and what breaks when you use it naively?
16. [DL 17] PPO vs DPO vs RLHF—complete comparison for LLM alignment
17. [DL 17] When is test-time compute cheaper than a bigger model—and when does it stop working?
18. [DL 19] Design an LLM serving stack that handles 1,000 queries per second with a p99 latency of 2 seconds for a 70B parameter model
19. [DL 19] KV cache memory is your bottleneck—you are running out of GPU memory and dropping requests. What do you do?
20. [DL 23] Model B is 1.5 points better than model A on MMLU—do you ship it?
21. [DL 27] When should you NOT build an agent? Your PM wants "an agent" for a document-processing product—how do you decide?
22. [DL 20] Design safety for an agent that can browse the web and execute code

### Search, recommendation, ads, marketplace ranking

1. [SR 1] Walk me through what happens, end to end, when a user types a query into a large-scale search engine
2. [SR 1] Design the query understanding system for an e-commerce search engine
3. [SR 2] Walk me through, mechanically, how BM25 over an inverted index returns the top-10 results from a billion documents in under 50 ms
4. [SR 2] What do BM25's k_1 and b actually control, and when would you change them from the defaults?
5. [SR 2] Lexical vs. dense vs. hybrid—argue the retrieval split for a marketplace search engine
6. [SR 3] How does HNSW work, and why is it fast?
7. [SR 3] Index 500M × 768-d embeddings and serve top-100 under 15 ms p99 on a machine with 64 GB RAM—walk me through the design
8. [SR 4] Bi-encoder vs. cross-encoder: why not cross-encode everything, and what exactly does the cross-encoder buy?
9. [SR 4] You run BM25 and dense retrieval in parallel. How do you combine them—and why does 0.5·BM25 + 0.5·cosine fail?
10. [SR 4] Design semantic search over 100 million documents, end to end
11. [SR 5] Explain LambdaRank to a strong engineer who knows GBDT but has never done ranking
12. [SR 5] Your click-trained ranker keeps favoring the items that have historically sat at position 1—better new items never rise. Diagnose and fix
13. [SR 5] GBDT or neural network for your L2 ranker? Decide for (a) a commerce search engine with rich engineered features, (b) a feed ranker over user history and item IDs
14. [SR 6] Design a recommendation system for a video streaming platform with 100M users and 1M videos
15. [SR 6] How do you handle the cold start problem for new users and new items?
16. [SR 6] Design the ranking model for an e-commerce product ads system
17. [SR 6] Estimate the memory required for DLRM embedding tables with 100M users, 10M items, and 1000 categorical features
18. [SR 6] Why do recommendation models need calibration? What happens if pCTR is systematically over-confident?
19. [SR 7] Derive NDCG from first principles, then compute NDCG@3 for a ranking with grades (3, 0, 2), exponential gain, given the ideal available grades are (3, 2, 0)
20. [SR 8] You own a search endpoint with a 250 ms server-side p99 SLO. Walk me through the latency budget—where does the time go, and what do you cut when you're over?
21. [CML 2] Derive the optimal leaf weight and the split-gain formula in XGBoost
22. [CML 6] We want to detect a 1% relative lift on a 2% click-through rate. How many users do we need, and how long should we run?

### Generalist ML loop (platform company, applied/product org)

1. [DL 23] Your model has 95% accuracy but stakeholders are unhappy. What is going on?
2. [DL 23] Precision is 0.95 but recall is 0.30. What do you do?
3. [DL 23] NDCG vs MAP vs MRR for ranking—when do you use each?
4. [DL 23] How do you know if a model improvement is statistically significant?
5. [DL 23] Your model's AUC is 0.95 but calibration is poor (ECE = 0.15). Why does this matter and how do you fix it?
6. [DL 23] Offline metrics went up but online metrics went down. Give 5 possible reasons
7. [DL 22] How do you handle training-serving skew?
8. [DL 22] Your model's CTR dropped 5% overnight. Walk through your diagnosis
9. [DL 22] Shadow deployment vs A/B test vs multi-armed bandit—when do you use each?
10. [DL 24] Your model needs to run in <10ms. Current latency is 50ms. Walk through the optimization decision tree
11. [DL 24] You have 3 months and a team of 3 ML engineers. Should you fine-tune an LLM or train a custom model?
12. [DL 24] When should you use gradient boosting vs. neural networks for tabular data?
13. [DL 3] When do you use softmax cross-entropy vs. sigmoid BCE?
14. [DL 3] Cross-entropy vs. focal loss vs. class-balanced loss for imbalanced classification—what is your decision framework?
15. [DL 2] Explain the bias-variance tradeoff. Does it apply to deep learning?
16. [DL 4] Why GELU over ReLU? Why SwiGLU over GELU?
17. [DL 16] LoRA vs QLoRA vs full fine-tuning vs prompt tuning—give a complete decision framework
18. [DL 17] Walk me through the RLHF pipeline for aligning an LLM
19. [NLP 9] When should you use RAG vs. fine-tuning vs. long context?
20. [NLP 1] Explain BM25. Why is it still used in production search systems?
21. [CML 2] Random forest vs. gradient boosting—how do they differ, and when do you reach for each?
22. [CML 6] We want to detect a 1% relative lift on a 2% click-through rate. How many users do we need, and how long should we run?

Every question above is `***` (asked constantly) in
`index/master_question_index.md`. If you want more, filter that file for `***`
in your loop's chapters — there are 243 of them program-wide, and you are not
going to do 243 tonight.

## Block 4 — The numbers card (25 min)

**10 min — `kernel/numbers_card.tex` §7, "Orders of magnitude: the night
before," cold.** Fifteen rows, one significant figure each, written for exactly
this hour. Cover the right-hand column and recite: H100 dense BF16 and its
memory bandwidth; the H100 ridge point and where batch-1 decode sits against it;
an H100-hour of useful work at 40% MFU; 6 FLOPs per parameter to train and 2 to
serve; 16 bytes per parameter of training state; Chinchilla's 20:1; the
Llama-2 7B run to check against; a 70B GQA KV cache per token and at 128K; an
8×H100 node's hourly cost and output throughput; 1M 768-d vectors in fp32, int8
and PQ64; the A/B sample size at 2% CTR for a 1% lift. Say the word **"dense"**
out loud with every FLOPs figure — quoting a sparsity-inflated peak is an
instant flag.

**15 min — the four worked examples, spoken end to end**
(`review/first-principles/numbers-fluency.md` §8). About four minutes each:

1. Memory to full fine-tune a 7B with AdamW in mixed precision, then the LoRA
   and QLoRA variants.
2. Serve a 70B at 1K QPS with 500 input and 200 output tokens — sized as two
   separate capacities, prefill and decode, then a KV feasibility check.
3. LoRA parameter count: `r·(d_in + d_out)` per adapted matrix, not
   `r·d_in·d_out`.
4. Embedding table sizing, including the RecSys case where one feature exceeds
   any single GPU.

Use the delivery pattern every time, because it is what is graded: **state the
formula → plug numbers rounded to one significant figure → sanity-check against
a real system → name the binding constraint and one lever that relaxes it.**
The sanity check is what separates L6 from L5.

## Block 5 — What to do in the room (20 min)

Read [DL 29] §"Execution Mechanics" and §"The Five Moves" once — fifteen
minutes — then spend five writing these two cards by hand. Keep them off-camera
tomorrow.

### Card 1 — Design round

**The five moves.**
1. **Constraint discovery** — 2–3 minutes of questions before designing: users
   and traffic, latency and cost budgets, quality bar and error asymmetry, what
   exists already, what "done" means. Say aloud which constraint you believe is
   the expensive one.
2. **Quantify early** — QPS, tokens, bytes, GPUs, dollars, *before* drawing
   boxes. "And then we shard it," with no numbers, is the archetypal
   staff-level reject.
3. **Propose, then iterate** — a complete v1 end to end by minute 25, then
   harden where they steer. Say what you are consciously deferring.
4. **Failure modes as first-class** — per component: how it fails, how you
   detect it, what degrades gracefully. Volunteer these.
5. **The eval plan, always** — metrics, gates, and the experiment that validates
   the riskiest assumption first.

**When they mutate a constraint mid-design** — and they will, deliberately —
go back to move 2. Re-run the arithmetic, name which components survive,
renegotiate what you now cannot meet. Say: *"that changes the binding constraint
from memory to bandwidth; three things break, here is what I'd cut."* Defending
the old design and silently starting over both fail the same probe.

**Clock (60-minute round):** requirements and workload math by 10; complete v1
by 25; steered deep dives to 45; failure modes and eval plan to 55. Still
gathering requirements at 15 means propose and iterate instead.

### Card 2 — Coding round, and the habits that cross every round

**The six checks** you will be graded on: does it run; are the shapes right;
is the softmax stable; is the mask applied *before* the softmax; did you test
without being asked; can you state complexity and memory.

**Clock:** 5 minutes to clarify and plan; working v1 by 20; extensions to 45;
tests and walkthrough to 55. No working v1 by minute 25 → cut scope aloud.

**Ask the AI-assistant rules out loud** at the start. Policies differ per
company and per round; some rounds grade how you direct and verify an assistant.

**Everywhere:**
- **Narrate decisions, not keystrokes.** Silence is the failure mode —
  imperfect narrated work outscores perfect silent work in every rubric,
  because the interviewer can only grade what they can observe.
- **Rent a hint well.** State what you know, what you have ruled out, and the
  specific fork you want input on. Asking well scores positively; flailing
  silently for twenty minutes does not.
- **Keep arithmetic on the board, not in your head**, so it can be audited and
  credited.
- **Deep dive: answers under ~90 seconds before yielding.** Depth comes from
  their next probe, not your monologue.
- **Behavioral: two-minute stories, half the slot left for probes.** Three
  stories probed deeply beats six recited.
- **Say "I don't know, here's how I'd find out"** at your boundary. Bluffing
  there is near-fatal at labs that explicitly calibrate on calibration.
- **A bad round is recoverable if it is a calibrator, not a gate.** Do not carry
  round N's misery into round N+1 — a fresh interviewer starts you at zero. Use
  the break for water, a walk, and two minutes of notes on what comes next.

## Block 6 — Logistics, your questions, and stop (15 min)

- **Test the tools.** Open the actual editor and whiteboard the recruiter named
  and use them for two minutes at speed. Fumbling the canvas taxes your design
  round's clock.
- **Two real questions per interviewer**, written down. They are graded as level
  signal: ask about constraints and decision-making — *"what's the team's
  scarcest resource: compute, people, or decision bandwidth?"*, *"what does the
  last project you killed tell me about how you make that call?"* Questions
  about process and perks read junior.
- **Camera at eye level, notes off-screen, notifications off.** Say explicitly
  when you are pausing to think, so silence is not dead air.
- **Then stop.** Close the laptop. You are not going to learn the thing that
  decides tomorrow in the next hour, and the rounds that decide it — the deep
  dive, the design, the coding — are all worse when you are tired.

---

## The two-hour cut

Drop block 6 (do it in ten minutes over breakfast) and trim block 3 to the
first fifteen questions in your list:

25 + 20 + 30 + 25 + 20 = **120 minutes**.

## The forty-minute cut

You are in a car. Block 1's six groups, spoken (10 min). Card 1's five moves and
card 2's six checks, read twice (10 min). The numbers card's §7 table, recited
(10 min). The first eight questions in your loop's list (16 min). Stop.
