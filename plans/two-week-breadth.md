# Two-Week Breadth Sprint

For a generalist ML loop: a platform company with a named "ML fundamentals"
round, an applied/product org, or a phone screen with a breadth segment. The
target is reliable one-to-two-level coverage of the whole standard map, spoken
in under three minutes per question, with no cliffs.

**Budget: 2–3 hours a day, 14 days.** 109 questions drilled aloud — all but one
of them `***` — spread over 13 days at 5 to 12 a day; day 7 is a pure
self-check. Reading is subordinate to the questions.

## The daily protocol

1. **Read the chapter's `tldr` box** (every chapter has one). 5 minutes.
2. **Answer the day's questions aloud, timed at two minutes, before reading the
   answer.** Record pass/fail. This is 90–120 minutes and it is the whole point:
   the breadth round is graded on spoken organization, and reading builds none
   of it.
3. **Read only the sections your misses point at.** 30–45 minutes.
4. **Write the miss into a running list** with the one sentence you should have
   said. 10 minutes. You will re-drill this list on days 7 and 14.

Do not read chapters cover to cover. The sprint does not have room — [DL 3]
alone is 1,824 lines — and the round does not reward it.

**Run the Core Fifty alongside.** `kernel/core50.tex` is the triage tier between
the kernel and the 543 questions: fifty items with the substance to state, the
pointer to the treatment, and the wrong answer that costs the signal. Its own
cadence is five a day for ten days, spoken, marked red/amber/green — about
fifteen minutes on top of the day's budget, and its closing checklist is your
final pass. If you can only do one thing on a day this sprint goes wrong, do
that day's five.

## Ordering

This follows the Program Map's **breadth-round** path — [DL 1–4], [DL 23],
[DL 24], Volume IV, [NLP 1–3], [SR 1, 6, 7] — with the LLM stack added at one
level of depth in days 5–6 and 8–9. The addition is deliberate:
`review/first-principles/interview-loop.md` finds that the 2026 breadth question
distribution "has shifted hard toward the LLM stack," and a candidate who can
compare LayerNorm variants but not KV-cache growth reads as three years stale.

---

# Week 1 — Foundations and the modern stack

## Day 1 — The kernel and the math

**Read.** `kernel/part0_kernel.tex` — the Part 0 kernel, all 26 principles, one
pass, one hour. Then [DL 1] *Mathematical Foundations*.

**Drill** (5 `***`):
- [DL 1] L5 — "Explain SVD and give three applications in machine learning"
- [DL 1] L5 — "Why does the chain rule matter for backpropagation? Walk through a 3-layer example"
- [DL 1] L5 — "What is KL divergence? Why is it not a true distance metric? When do you use it in practice?"
- [DL 1] L5 — "Estimate the memory for storing a 50K × 768 embedding matrix in FP32, FP16, and INT8"
- [DL 1] L6 — "Derive the gradient of softmax cross-entropy loss with respect to the logits"

**Done when.** You write `p − y` from the softmax cross-entropy loss without
notes, and give the FP32/FP16/INT8 embedding sizes in under 30 seconds.

## Day 2 — Learning theory and the compute budget

**Read.** [DL 2] *Learning Theory Essentials*. Then `kernel/numbers_card.tex`
§3, "Training at scale" — Chinchilla, the GPU-hour constants, and the reference
runs to sanity-check against.

**Drill** (5 `***`):
- [DL 2] L5 — "Explain the bias-variance tradeoff. Does it apply to deep learning?"
- [DL 2] L5 — "Explain double descent. Why does test error decrease again in the over-parameterized regime?"
- [DL 2] L6 — "Your 10B parameter model is clearly overparameterized for your task. Should you reduce it?"
- [DL 2] L6 — "Your team has a fixed compute budget. How do you decide model size vs. training data?"
- [DL 2] L6 — "Estimate the compute-optimal model size for a training budget of 10^22 FLOPs using Chinchilla scaling laws"

**Done when.** `C ≈ 6ND`, `N_opt = sqrt(C/120)`, `D ≈ 20N`, and the 10^22 →
~9B params / ~180B tokens answer come out cold, with the sanity check attached.

## Day 3 — Losses

**Read.** [DL 3] *Loss Functions: The Interview Playbook*. The longest chapter in
Volume I; it ends with a Quick Reference Table, one of only three in the program.
Use the table, not the prose.

**Drill** (10 `***` — the heaviest question day; if you run out of time, cut the
last two):
- [DL 3] L5 — "When do you use softmax cross-entropy vs. sigmoid BCE?"
- [DL 3] L5 — "Cross-entropy vs. focal loss vs. class-balanced loss for imbalanced classification—what is your decision framework?"
- [DL 3] L6 — "Your SFT loss changes when you change batch composition or gradient-accumulation steps—same data, same model, same effective batch size. Why?"
- [DL 3] L5 — "Why does SimCLR need large batch sizes, and what are the alternatives?"
- [DL 3] L6 — "You're building a visual search system for an e-commerce platform with 10M products. What loss function would you use for training the embedding model?"
- [DL 3] L5 — "You need to train an embedding model for semantic search across 100M documents. Which loss function and why?"
- [DL 3] L6 — "Pointwise vs. pairwise vs. listwise ranking losses—when do you use each?"
- [DL 3] L6 — "Design the loss function for a retrieval system where you have click data (implicit feedback) but no explicit relevance labels"
- [DL 3] L5 — "Your classification model gets 95% accuracy but the loss curve shows the model is still decreasing slowly. Should you keep training?"
- [DL 3] L5 — "You're building a product recommendation system with implicit feedback (clicks). What loss would you use and why?"

**Done when.** Cross-entropy vs. focal vs. class-balanced has a stated decision
boundary, not a list; and you can say what InfoNCE's temperature controls.

## Day 4 — Activations, normalization, training mechanics

**Read.** [DL 4] *Activation Functions and Normalization* in full (725 lines,
the shortest of the foundations). Then [DL 15] *Training Optimization*
selectively: optimizers, gradient accumulation, LR schedules, mixed precision,
and the NaN playbook. Skip [DL 15]'s parallelism sections — they are depth-track
material.

**Drill** (4 `***` from [DL 4], 4 selected from [DL 15]):
- [DL 4] L5 — "Explain the dying ReLU problem. How do modern activations solve it?"
- [DL 4] L5 — "Why GELU over ReLU? Why SwiGLU over GELU?"
- [DL 4] L5 — "Training loss is NaN after a few hundred steps. Which activation/normalization issues could cause this?"
- [DL 4] L6 — "Your 70B-parameter run in BF16 starts loss-spiking at 100B tokens. Walk me through the activation- and normalization-level causes and mitigations"
- [DL 15] L5 — "Adam vs SGD with momentum—when would you pick each?"
- [DL 15] L5 — "Your model's loss plateaus after initial descent. Walk through your debugging process"
- [DL 15] L5 — "Training loss is NaN after 1000 steps. Walk through your debugging process"
- [DL 15] L6 — "How much GPU memory is needed to fine-tune a 13B parameter model with LoRA vs. full fine-tuning?"

**Done when.** You can give the pre-norm vs. post-norm gradient-flow argument in
90 seconds, and recite the 16-bytes-per-parameter training memory ledger and
what each byte is.

## Day 5 — Transformers, the derivations

**Read.** [NLP 5] *Transformers and Modern Architectures*. Per the Program Map's
canonical-home contract this is where the transformer derivations live; [DL 5]
tomorrow keeps the quantitative systems view. Do not read them in the other
order.

**Drill** (6 `***`):
- [NLP 5] L5 — "Explain self-attention step by step, including the math"
- [NLP 5] L5 — "Why do we scale by d_k? Derive the variance argument"
- [NLP 5] L5 — "Why do we need multiple attention heads? What happens with just one head?"
- [NLP 5] L5 — "Compare encoder-only, decoder-only, and encoder-decoder transformers. Why did decoder-only dominate the LLM era?"
- [NLP 5] L5 — "What is the computational complexity of self-attention? Why is it a problem?"
- [NLP 5] L5 — "Explain BERT's pretraining objectives. Why was NSP removed in RoBERTa?"

**Done when.** You write scaled dot-product attention and the variance argument
for the 1/sqrt(d_k) factor on paper in four minutes, from memory.

## Day 6 — Transformers, the arithmetic; embeddings

**Read.** [DL 5] *Attention Mechanisms and Transformers* — parameter and FLOP
counting, KV-cache bytes, GQA/MQA/MLA, FlashAttention. Then [DL 6] *Embeddings
and Representation Learning* (463 lines; the production-diagnostics half — the
theory's canonical home is [NLP 3], on day 14). Then `kernel/numbers_card.tex`
§2 ("Model arithmetic") and §4 ("Inference and serving").

**Drill** (6 `***` plus one `**`):
- [DL 5] L6 — "Compare MHA, GQA, MQA, and MLA for KV-cache efficiency. Why does MLA need a decoupled RoPE key?"
- [DL 5] L5 — "Explain Flash Attention. Why is it faster even though it does the same computation?"
- [DL 5] L6 — "Estimate the memory required to train a 7B parameter model with batch size 1, sequence length 2048"
- [DL 5] L6 — "Calculate the KV cache memory for serving LLaMA 70B at 128K context length"
- [DL 5] L5 — "How does RoPE encode position information? Why is it preferred over learned positional embeddings?"
- [DL 5] L5 — "What changed between GPT-2 (2019) and LLaMA (2023) architecturally? Why each change?"
- [DL 6] L5 `**` — "Estimate the memory for an embedding table with 50K vocab, 768 dimensions in FP16 vs INT8"

**Done when.** `N ≈ 12·L·d² + V·d` checked against Llama-7B, and the 70B-at-128K
KV cache (~40 GB) both come out without notes. If you multiplied by `n_heads`
instead of `n_kv` on the GQA model, redo it.

## Day 7 — Week 1 self-check

No new reading.

- **60 min.** Twenty questions drawn at random from days 1–6. Two minutes each,
  spoken, timed, no notes. Score pass/fail as you go.
- **30 min.** On paper, from memory: the softmax cross-entropy gradient; the
  1/sqrt(d_k) variance argument; 6ND and the Chinchilla allocation; the KV-cache
  formula.
- **30 min.** Read the answers to everything you failed, and add the one
  sentence you should have said to your miss list.

**Gate.** More than 5 failures out of 20 means week 1 did not stick. Spend day 8
repeating the two weakest chapters instead of moving on; the second week assumes
the first.

---

# Week 2 — The applied stack and the classical insurance

## Day 8 — Adaptation and alignment

**Read.** [DL 16] *Transfer Learning and Parameter-Efficient Fine-Tuning* and
[DL 17] *Reinforcement Learning and RLHF*, both at breadth level: the pipeline
and the decision boundaries. The derivations are depth-track material.

**Drill** (9 `***`):
- [DL 16] L5 — "Estimate the number of trainable parameters for LoRA with rank 16 on a 7B model"
- [DL 16] L6 — "Your LoRA fine-tuned model performs well on your test set but catastrophically forgets general capabilities. How do you fix it?"
- [DL 16] L6 — "LoRA vs QLoRA vs full fine-tuning vs prompt tuning—give a complete decision framework"
- [DL 16] L5 — "You need to fine-tune a 70B LLM for a customer's specific domain. You have access to a single A100 80GB GPU. What approach would you take?"
- [DL 16] L6 — "You fine-tuned a model and it performs well on your test set but poorly in production. What happened?"
- [DL 17] L5 — "Walk me through the RLHF pipeline for aligning an LLM"
- [DL 17] L6 — "PPO vs DPO vs RLHF—complete comparison for LLM alignment"
- [DL 17] L5 — "What is reward hacking? Give 3 concrete examples and how to mitigate each"
- [DL 17] L5 — "Why does the KL penalty matter in RLHF? What happens without it?"

**Done when.** The fine-tune / LoRA / prompt / RAG decision boundary is stated in
terms of data volume, latency, and update frequency — not as a list of methods.

## Day 9 — Serving and context

**Read.** [DL 19] *Inference Optimization and LLM Serving*, then [DL 21] *Long
Context and RAG Systems* (524 lines, 4 questions).

**Drill** (7 `***`):
- [DL 19] L5 — "Explain speculative decoding. When would you use it and when would you not?"
- [DL 19] L6 — "KV cache memory is your bottleneck—you are running out of GPU memory and dropping requests. What do you do?"
- [DL 19] L6 — "How much memory savings does INT4 quantization give for a 70B model? What is the quality trade-off?"
- [DL 19] L7 — "Design an LLM serving stack that handles 1,000 queries per second with a p99 latency of 2 seconds for a 70B parameter model"
- [DL 21] L6 — "Your 8K-trained model must serve 64K contexts next month. What are your options, and what do they cost?"
- [DL 21] L6 — "Estimate the cost per query for a RAG system vs a long-context LLM (128K)"
- [DL 21] L5 — "Long context window vs RAG—when do you choose each?"

**Done when.** You can say why batch-1 decode is bandwidth-bound *with the
arithmetic-intensity number* (≈1 FLOP/byte against an H100 ridge point near 300),
and give the RAG-vs-long-context boundary with a cost figure attached.

## Day 10 — Evaluation

**Read.** [DL 23] *Evaluation Metrics*. Ten of its twelve questions are `***` —
the densest high-frequency chapter in the program. It also carries a Quick
Reference section.

**Drill** (all 10 `***`):
- [DL 23] L6 — "Model B is 1.5 points better than model A on MMLU—do you ship it?"
- [DL 23] L7 — "Design the eval stack that decides whether tomorrow's checkpoint ships"
- [DL 23] L5 — "Your model has 95% accuracy but stakeholders are unhappy. What is going on?"
- [DL 23] L5 — "How do you evaluate a search ranking system?"
- [DL 23] L5 — "Precision is 0.95 but recall is 0.30. What do you do?"
- [DL 23] L5 — "How do you know if a model improvement is statistically significant?"
- [DL 23] L6 — "Your model's AUC is 0.95 but calibration is poor (ECE = 0.15). Why does this matter and how do you fix it?"
- [DL 23] L5 — "NDCG vs MAP vs MRR for ranking—when do you use each?"
- [DL 23] L6 — "Offline metrics went up but online metrics went down. Give 5 possible reasons"
- [DL 23] L5 — "What is wrong with accuracy as a metric? When is it actually fine?"

**Done when.** ECE, the benchmark standard error, and the A/B sample-size rule
are instant, and the offline-online divergence answer has five distinct causes
rather than three restatements of "distribution shift."

## Day 11 — Production and decisions

**Read.** [DL 22] *Production ML Systems* and [DL 24] *Decision Frameworks: When
to Use What*. These two carry most of the applied/product round's raw material.

**Drill** (12 `***` — the second heaviest day; drop [DL 22]'s last two if short):
- [DL 22] L6 — "Design ML infrastructure for a search ranking system at 10K QPS"
- [DL 22] L6 — "Your model's CTR dropped 5% overnight. Walk through your diagnosis"
- [DL 22] L6 — "How would you set up an A/B test for a new recommendation model?"
- [DL 22] L5 — "How do you handle training-serving skew?"
- [DL 22] L6 — "Your model's online metrics diverge from offline metrics after 2 weeks in production. What is happening?"
- [DL 22] L5 — "Shadow deployment vs A/B test vs multi-armed bandit—when do you use each?"
- [DL 24] L6 — "Design a content recommendation system for a news app"
- [DL 24] L6 — "Design the ML architecture for a content moderation system"
- [DL 24] L6 — "Your model needs to run in <10ms. Current latency is 50ms. Walk through the optimization decision tree"
- [DL 24] L6 — "You have 3 months and a team of 3 ML engineers. Should you fine-tune an LLM or train a custom model?"
- [DL 24] L5 — "Would you use BERT or GPT for sentiment analysis?"
- [DL 24] L5 — "When should you use gradient boosting vs. neural networks for tabular data?"

**Done when.** You can run the sub-10ms decision tree aloud, in order, naming
what each step costs.

## Day 12 — Classical ML

**Read.** [CML 2] *Trees and Ensembles* and [CML 6] *Experimentation and Causal
Inference*.

**Drill** (9 `***`):
- [CML 2] L5 — "Derive the optimal leaf weight and the split-gain formula in XGBoost"
- [CML 2] L5 — "Random forest vs. gradient boosting—how do they differ, and when do you reach for each?"
- [CML 2] L5 — "Explain gradient boosting to someone who already understands gradient descent"
- [CML 2] L6 — "XGBoost vs. LightGBM vs. CatBoost: 600K rows, 45 features of which 14 are categorical including a 40K-cardinality merchant_id. Choose and defend"
- [CML 2] L6 — "Why does a gradient-boosted tree beat your MLP on this tabular dataset—and when would it not?"
- [CML 6] L5 — "We want to detect a 1% relative lift on a 2% click-through rate. How many users do we need, and how long should we run?"
- [CML 6] L6 — "Your test came back flat, but you are confident the feature helps. What do you do?"
- [CML 6] L6 — "Your treatment shows +2% on the primary metric, but the SRM check fired. What now?"
- [CML 6] L6 — "Your marketplace A/B test shows +2% GMV in treatment. Why might the true effect be smaller, or negative?"

**Done when.** You derive the XGBoost leaf weight and split gain on paper without
notes, and run the sample-size calculation for 1% relative lift on a 2% CTR to a
number.

> **Gap — Volume IV is six-eighths unwritten.** The Program Map's breadth path
> says "the whole of Volume IV," but only [CML 2] and [CML 6] exist. Chapters 1
> (linear models, SVM, Naive Bayes, kNN), 3 (clustering, EM, PCA), 4
> (MLE/MAP/Bayes, calibration), 5 (the applied craft: leakage, CV,
> interpretability), 7 (time series), and 8 (the from-scratch coding canon) are
> scaffolds — headings and `% TODO` markers, no content.
>
> A generalist loop *will* probe that surface: the L1-vs-L2 geometry question,
> the logistic-regression gradient, k-means and its failure modes, leakage and
> cross-validation discipline. Prepare it outside this program. The checklist is
> `volumes/conventional-ml/docs/coverage-spec.md` — a 586-line spec listing the
> must-know subtopics, whiteboard derivations, and red-flag answers for all
> eight chapters. It tells you what you need to know; it does not teach it.
> Budget 3–4 hours outside this sprint against Ch. 1, 3, and 5 of that spec.
>
> What the program *does* cover from that surface: bias-variance ([DL 2]),
> calibration and metric choice ([DL 23]), drift and A/B mechanics ([DL 22],
> [CML 6]), and GBDT-vs-NN for tabular data ([DL 24]).

## Day 13 — Search and recommendation

**Read.** [SR 1] *Search Systems and Query Understanding* and [SR 6]
*Recommendation Systems*, at one level of depth — you are buying insurance
against a specialist cliff, not running the depth track. [DL 12] is a two-page
bridge chapter; [SR 6] is the canonical home and the one to read.

**Drill** (12 `***` — the heaviest day of the sprint; cut [SR 6]'s two
estimation questions if you are short):
- [SR 1] L5 — "Walk me through what happens, end to end, when a user types a query into a large-scale search engine"
- [SR 1] L6 — "Design the query understanding system for an e-commerce search engine"
- [SR 1] L6 — "Your new L2 reranker improves offline NDCG by 4%, but online conversions are flat. Walk through your diagnosis"
- [SR 6] L6 — "Design a recommendation system for a video streaming platform with 100M users and 1M videos"
- [SR 6] L5 — "How do you handle the cold start problem for new users and new items?"
- [SR 6] L6 — "Your recommendation model has great offline metrics but poor online performance. What went wrong?"
- [SR 6] L6 — "Design the ranking model for an e-commerce product ads system"
- [SR 6] L6 — "Estimate the memory required for DLRM embedding tables with 100M users, 10M items, and 1000 categorical features"
- [SR 6] L6 — "Your CTR model's offline AUC improved by 0.5% but online revenue dropped 3%. Diagnose"
- [SR 6] L5 — "Two-Tower vs. interaction-based models for candidate generation—when would you use each?"
- [SR 6] L5 — "Why do recommendation models need calibration? What happens if pCTR is systematically over-confident?"
- [SR 6] L6 — "Where would you use an LLM in a recommendation stack today—and why won't it replace your ranker?"

**Done when.** You can draw the retrieval → ranking → re-ranking funnel and say
why each stage exists in terms of a candidate-count and latency budget.

## Day 14 — Classical NLP, ranking metrics, and the final self-check

**Read** (75 min). [NLP 1] *Classical NLP Foundations* and [NLP 3] *Word
Representations and Embeddings* — the `tldr` boxes and the sections behind the
questions below. [SR 7] *Search and RecSys Evaluation* for the two `***` only.
([NLP 2] *Statistical NLP* is optional insurance: take perplexity and Viterbi
from it if your loop touches sequence labeling.)

**Drill** (9 `***`):
- [NLP 1] L5 — "What is TF-IDF? Derive the formula. When would you use it over neural embeddings?"
- [NLP 1] L5 — "Explain BM25. Why is it still used in production search systems?"
- [NLP 1] L5 — "Explain the difference between rule-based, statistical, and neural NLP. When would you still use rule-based methods?"
- [NLP 3] L5 — "Explain the Word2Vec Skip-gram model. What is the training objective? What is negative sampling and why is it needed?"
- [NLP 3] L5 — "What are the differences between Word2Vec, GloVe, and FastText? When would you choose each one?"
- [NLP 3] L5 — "Why are contextual embeddings better than static embeddings? Give a concrete example of when static embeddings fail"
- [NLP 3] L6 — "Your new embedding model wins offline—+4 points Recall@100 on your evaluation set—but loses the online A/B test. Walk me through your investigation"
- [SR 7] L5 — "Derive NDCG from first principles, then compute NDCG@3 for a ranking with grades (3, 0, 2), exponential gain, given the ideal available grades are (3, 2, 0)"
- [SR 7] L6 — "Your reranker improved offline NDCG@10 by 3% on the judgment set, but the A/B test shows flat CTR. Walk me through your investigation"

**Then the week 2 self-check** (60 min): twenty-five questions drawn at random
from days 8–14, two minutes each, spoken, timed.

## The end-of-sprint gate

You are ready for a breadth round when all four hold:

1. **Coverage.** You can answer 9 of 10 randomly drawn `***` questions from any
   day of the sprint at one-to-two levels of depth, in under three minutes each,
   without notes.
2. **Organization.** Your answers give the mechanism, the decision boundary, and
   the exception — in that order — rather than a list of related facts. Rambling
   is the second-most-common breadth failure after cliffs.
3. **No cliffs.** Days 12, 13, and 14 (classical ML, search/recsys, classical
   NLP) are your insurance days. A world-class LLM answer paired with a blank on
   "when does Naive Bayes beat BERT" reads as narrow.
4. **Arithmetic.** You do not refuse to compute. Memory for a 7B fine-tune, KV
   cache at 128K, a sample size for a 1% lift — all to a number, out loud.

Your miss list from 14 days is the night-before reading. Take it to
[`night-before.md`](night-before.md).
