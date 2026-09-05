# LLM Depth Track — Four Weeks

For frontier-lab loops (OpenAI/Anthropic-class): a Socratic depth grill in your
claimed specialty, an LLM training- or serving-infrastructure design round, and
the agent/safety/eval material that shows up in both. Per
`review/first-principles/interview-loop.md`, the depth round is ~20–25% of the
technical decision and the design round ~25–30% — and design is the primary
leveling instrument. This track prepares those two and the coding round's
knowledge layer.

**Budget: 3–4 hours a day.** Structure per week: five study days, one
mock/consolidation day, one day off. The rest day is not optional at this
intensity; four weeks of 3.5-hour days without one is how week 3 collapses.

## What "depth" means here, operationally

The depth round finds your floor. Every answer spawns a *why*, a *what breaks*,
or a *derive it*. So the daily standard is not "I read the chapter." It is:

1. **The derivation is on paper, from memory.** Not recognized — reproduced.
2. **The arithmetic runs to a number, out loud, without a calculator.**
   Interviewers ask for numbers precisely because reciting is easy and computing
   is not. Refusing to compute is a documented staff-level reject.
3. **You can name what breaks.** For each mechanism: the failure mode, its
   symptom, and the cheapest experiment that discriminates it from its nearest
   neighbour.
4. **You have one opinion with receipts.** Something you believe that the
   consensus does not, argued from a mechanism or an experiment, with the
   evidence that would change your mind. This is the research-taste overlay and
   it modifies every round's score.

## Before day 1 (one hour)

Read `kernel/part0_kernel.tex` once — the 26 principles, groups A through F.
Groups B (scale), C (optimization), and D (the machine) are this track's spine,
and every arithmetic question below is one of them discharged into a number.
Then start `kernel/core50.tex` at five items a day, spoken, marked
red/amber/green; only reds earn chapter time, and its closing checklist is your
last-48-hours pass.

The quantitative layer runs alongside: `kernel/numbers_card.tex` §1
(hardware and ridge points) and §2 (model arithmetic) belong to week 1, §3
(training at scale) to week 2, §4 (inference and serving) to week 4, and §6
(statistics and evaluation) to days 18–19.

## Ordering

This follows the Program Map's **LLM / frontier-lab depth** path: [NLP 5]
derivations, [DL 5] quantitative systems, [NLP 6] pretraining and adaptation,
[DL 25] pretraining at scale, [DL 26] GPU performance, [DL 15] optimization,
[DL 14] efficient architectures, [NLP 7] in-context learning and test-time
compute, [DL 17] and [NLP 8] for RL and alignment, [DL 19] inference, [DL 21]
long context, [DL 27] agents, [DL 20] and [NLP 13] safety.

Four additions, each with a reason:

- **[DL 4]** (day 4) — normalization and BF16 training stability. The map's path
  assumes it as foundations; a 2026 depth grill goes straight at loss spikes and
  QK-norm.
- **[DL 9]** (day 5) — where a model's parameters live, the encoder/decoder
  decision, and the mechanistic account of in-context learning. It is the bridge
  between [NLP 5]'s derivations and [DL 5]'s arithmetic, and its
  "where do those 7B parameters live" question is a standard opener.
- **[DL 16]** (day 8) — PEFT *mechanics*. The canonical-home table makes [DL 16]
  the home for LoRA mechanics while [NLP 6] keeps the adaptation-lifecycle
  narrative; you want both.
- **[DL 23]** and **[NLP 10]** (days 18–19) — evaluation. Every design round in
  the loop ends with "how do you know it works," and asking that twice without a
  crisp answer is a documented reject.

Day 9 also re-reads [DL 2]'s scaling material at derivation level, and day 26 is
[DL 29] — which is not an addition but the Program Map's own separate path,
"the loop itself."

---

# The eight derivations that must be fluent on a whiteboard

Not "familiar." Reproducible from a blank sheet in under five minutes each,
while narrating. These are what the "derive it" follow-up reaches for. Drill
them on every mock day until all eight are automatic; they are also the
15-minute block in [`night-before.md`](night-before.md).

| # | Derivation | Where it lives |
|---|---|---|
| 1 | Scaled dot-product attention, and the variance argument for the 1/sqrt(d_k) factor | [NLP 5] — `Scaled Dot-Product Attention` and `Variance of the Dot Product` |
| 2 | The softmax cross-entropy gradient reducing to `p − y` | [DL 1] — `Softmax Cross-Entropy Gradient` |
| 3 | `C ≈ 6ND` (2 forward + 4 backward) and Chinchilla's `N_opt = sqrt(C/120)`, `D = 20N` | [DL 2] — `Chinchilla Optimal Scaling` |
| 4 | The training memory ledger (16 bytes/param under mixed-precision AdamW, and what each byte is) and the KV-cache formula `2·L·n_kv·d_head·bytes·s` | [DL 15]; [DL 19] — `KV Cache Memory`, `Model Memory Estimate` |
| 5 | The KL-constrained RLHF objective → its optimal policy → the DPO loss | [NLP 8] — `RLHF Objective`, `Optimal Policy under KL-Constrained RM Maximization`, `DPO Loss Function` |
| 6 | GRPO's group-relative advantage, and what PPO's critic was doing that it replaces | [DL 17] — `PPO Clipped Objective`, `GRPO Objective and Group-Relative Advantage` |
| 7 | Arithmetic intensity and the roofline ridge point; classifying an op as compute- or bandwidth-bound | [DL 26] §"The Roofline Model", §"Arithmetic Intensity", §"Worked Classifications" |
| 8 | LoRA's reparameterization and its parameter count `r·(d_in + d_out)` per adapted matrix | [DL 16] — `LoRA Reparameterization`, `LoRA Parameter Count` |

The trap on #8 is writing `r·d_in·d_out` — that is the full matrix, and the
whole point is that the count is additive in the dimensions, not multiplicative.
The trap on #4 is multiplying by `n_heads` instead of `n_kv` on a GQA model,
which is an 8× error on Llama-70B.

---

# Week 1 — Architecture and its arithmetic

Theme: the transformer as an object you can both derive and cost.

## Day 1 — [NLP 5] Transformers and Modern Architectures

The canonical home for the transformer derivations. Read it cover to cover;
this is one of the three chapters in the track that earns a full read.

**Drill.** All six `***`, then the L6s: "Explain RoPE (Rotary Position
Embedding). Why is it better than learned absolute positions for long
sequences?"; "What does the FFN layer do in a transformer? Is there evidence it
stores factual knowledge?"; "What is Flash Attention? Why does it help if it
computes the same result as standard attention?"; "Compare Multi-Query
Attention, Grouped-Query Attention, and standard Multi-Head Attention."

**Done when.** Derivations 1 and 2 are on paper from memory, and you can state
the online-softmax rescaling that makes FlashAttention's tiling exact — not just
that "statistics get corrected as tiles arrive."

## Day 2 — [DL 5] Attention Mechanisms and Transformers, part 1

The quantitative systems view. Today: parameter counting, FLOP counting,
KV-cache bytes, and the MHA → MQA → GQA → MLA progression.

**Drill:**
- L6 `***` — "Compare MHA, GQA, MQA, and MLA for KV-cache efficiency. Why does MLA need a decoupled RoPE key?"
- L6 `***` — "Estimate the memory required to train a 7B parameter model with batch size 1, sequence length 2048"
- L6 `***` — "Calculate the FLOPs for a single forward pass through a 7B parameter Transformer"
- L6 `***` — "Calculate the KV cache memory for serving LLaMA 70B at 128K context length"
- L7 `*` — "Estimate the ratio of attention FLOPs to FFN FLOPs at different sequence lengths. When does attention become the bottleneck?"
- [DL 9] L5 `***` — "You're told a model is 'Llama-class, 7B parameters.' Roughly where do those 7B parameters live?"

**Done when.** `12·L·d² + V·d` verified against Llama-7B *and* corrected for
Llama-70B's GQA (attention is ~2.15d², not 4d²), and the attention/FFN crossover
argued in terms of sequence length against model width.

## Day 3 — [DL 5] part 2, plus stability

Today: FlashAttention from the hardware side, RoPE and context extension, long-
context degradation, SSMs, and the attention-logit pathologies.

**Drill:**
- L6 `***` — "Your large BF16 pretraining run shows intermittent loss spikes. Instrumentation shows attention logits growing into the hundreds over training. Diagnose and fix"
- L6 `**` — "Your Transformer model shows degrading perplexity at 32K context despite being trained on 8K. Diagnose and fix"
- L6 `**` — "Compare Transformer vs SSM (Mamba) architectures. When would you choose each?"
- L7 `*` — "Your model's attention patterns show all heads attending to the same positions (typically position 0 or EOS). What's happening?"
- L7 `*` — "Design the attention mechanism for a model that must process 1 million token documents"
- L7 `*` — "You need to modify a standard Transformer for real-time streaming applications (e.g., live transcription). What changes?"

**Done when.** You can name three distinct mechanisms behind BF16 loss spikes
and say which instrumentation distinguishes them.

## Day 4 — [DL 4] Activation Functions and Normalization

Short (725 lines) and load-bearing: pre-norm vs. post-norm gradient flow,
RMSNorm, and why Transformers abandoned BatchNorm.

**Drill.** All four `***` plus: L6 — "Why don't Transformers use BatchNorm?
Derive the reasoning from first principles"; L6 — "Pre-norm vs. post-norm—derive
the gradient-flow difference and explain what modern hybrids fix"; L6 `***` —
"Your 70B-parameter run in BF16 starts loss-spiking at 100B tokens. Walk me
through the activation- and normalization-level causes and mitigations."

**Done when.** The pre-norm/post-norm gradient-flow argument is a derivation on
paper, not a preference.

## Day 5 — [DL 9] NLP Architectures

Where the parameters live, the encoder/decoder decision, why decoder-only won,
and the mechanistic story for in-context learning.

**Drill:**
- L7 `*` — "How does in-context learning work mechanistically? Why can large language models learn from examples in the prompt without any gradient updates?"
- L6 `***` — "Why do decoder-only models dominate modern NLP despite encoder-decoder being theoretically more flexible?"
- L6 `***` — "Your fine-tuned chat model performs worse than the base model it was tuned from. Walk me through your diagnosis"
- L6 `***` — "Your fine-tuned LLM generates fluent but factually wrong answers. Diagnose the issue and propose solutions"
- L6 `**` — "Estimate the inference latency for generating 100 tokens with a 7B parameter model on a single A100 GPU"
- L6 `***` — "Walk me through the DPO loss function. How does it eliminate the need for a separate reward model?" *(preview of week 3; answer it badly today on purpose, then again on day 17)*

## Day 6 — Mock and consolidation

- **45 min.** Depth grill, out loud, with a partner if you have one: attention
  equation → why sqrt(d_k) → prove the variance claim → MHA to GQA to MLA and
  why each exists → KV cache for a 70B at 128K → is decode compute- or
  bandwidth-bound, show the arithmetic → FlashAttention's exact tiling
  correction. This is the canonical opening sequence; it should feel routine.
- **30 min.** Derivations 1, 2, and 4 on paper, timed at five minutes each.
- **45 min.** Re-read every answer you failed this week.
- **30 min.** Start your opinion file: one claim about architecture you believe
  and the consensus does not, with the mechanism and the falsifier.

## Day 7 — Off

---

# Week 2 — Training at scale

Theme: the run — its data, its parallelism, its hardware, and its failures.

## Day 8 — [NLP 6] Pre-training, Fine-tuning, Transfer + [DL 16] LoRA mechanics

[NLP 6] owns the adaptation-lifecycle narrative; [DL 16] owns PEFT mechanics.
Read [NLP 6] in full (1,294 lines) and [DL 16]'s LoRA sections.

**Drill:**
- [NLP 6] L6 `**` — "Compare the major pre-training objectives: MLM, CLM, span corruption, and ELECTRA. When would you choose each?"
- [NLP 6] L6 `**` — "Explain ELECTRA's pre-training approach. Why is it more efficient than MLM?"
- [NLP 6] L5 `**` — "Why is data quality more important than data quantity for SFT?"
- [DL 16] L6 `**` — "Why does LoRA work? What is the low-rank assumption about fine-tuning?"
- [DL 16] L6 `***` — "You have chosen LoRA to adapt a 7B instruction-tuned model to an internal support-and-policy domain: 20K curated examples, one 80 GB GPU, and the adapter will be served alongside the general assistant. Pick rank, alpha, and target modules, and defend the choices"
- [DL 16] L6 `**` — "You have one base model and 100 customers, each with their own LoRA adapter. Design the serving system"

**Done when.** Derivation 8 is fluent, including why the memory saving comes
from the frozen base carrying no gradients, optimizer state, or master weights —
not from "fewer parameters."

## Day 9 — [DL 2] to derivation level + [DL 25] Pretraining at Scale

[DL 2]'s scaling material is a two-hour re-read at depth, not a fresh chapter.
Then [DL 25] — corpus construction, the filter cascade, dedup and
decontamination, mixtures, mid-training, synthetic data, run mechanics. It ends
with a "Numbers Worth Memorizing" subsection; memorize it.

**Drill:**
- [DL 2] L7 `**` — "You get 1% of the target training budget to run scaling experiments that must predict the final run's loss. Design the study"
- [DL 2] L7 `**` — "Neural scaling laws: what scales and what does not? When do they break?"
- [DL 25] L5 `***` — "Walk me through what happens between 'we have a Common Crawl snapshot' and the first training step"
- [DL 25] L7 `***` — "You own data for the next frontier pretraining run. Design the pipeline and the ablation program"
- [DL 25] L6 `***` — "How would you decide the code:web:math mixture for a pretraining run?"
- [DL 25] L7 `***` — "Your data lead proposes raising the synthetic fraction of the next run from 10% to 40%. How do you evaluate the proposal?"
- [DL 25] L6 `**` — "Design the benchmark decontamination strategy for a pretraining run. What does n-gram overlap miss, and what do you do about it?"

**Done when.** You can give the filter cascade in order with a reason for the
ordering, and say what over-deduplication costs.

## Day 10 — [DL 15] Training Optimization

The parallelism triad, ZeRO stages, the memory ledger, throughput, and the
divergence playbook. 1,583 lines; read it in full.

**Drill:**
- L6 `***` — "How would you scale training of a 70B parameter model across 256 GPUs?"
- L6 `***` — "Data parallelism vs. tensor parallelism vs. pipeline parallelism—what is your decision framework?"
- L7 `***` — "Your 70B pretraining run diverges at 200B tokens. Give a full differential diagnosis, ordered by prior probability, with the cheapest discriminating experiment for each"
- L7 `***` — "A node dies every 3 hours on your 16K-GPU run. Design the recovery system"
- L6 `**` — "Your distributed training job is 60% as fast as expected with 8 GPUs. What is wrong?"
- L7 `*` — "Calculate the training throughput (tokens/sec) for a 7B model on 8×A100s"
- L6 `**` — "FP16 vs. BF16 vs. FP8 for training—when would you use each?"

**Done when.** The differential-diagnosis question is answered *as an ordered
list with a discriminating experiment per hypothesis*. An unordered list of
causes is the L5 version of this answer.

## Day 11 — [DL 26] GPU Performance Fundamentals

The hardware layer the design round rewards: memory hierarchy with real numbers,
the H100 anchor table, roofline and arithmetic intensity, tiling and wave
quantization, kernel fusion, NCCL collectives and their price, topology, MFU.

**Drill:**
- L6 `***` — "I give you an op—say, a LayerNorm over a [16,384 × 8,192] BF16 tensor, or a 4096^3 GEMM. Is it compute-bound or bandwidth-bound on an H100? Show your method"
- L6 `***` — "Your training job shows the GPU at 20% utilization. Diagnose it"
- L6 `**` — "How long does it take to all-reduce the gradients of a 70B-parameter model across 8 GPUs? Derive it"
- L6 `*` — "A teammate changed the hidden size from 4096 to 4100 'to add a few features,' and training throughput dropped 35%. What happened?"
- L7 `***` — "Estimate the maximum tokens/second for a 70B dense model on 8×H100—for training, and for inference"
- L7 `**` — "Your 512-GPU pretraining run reports 25% MFU. Walk me through how you find the missing performance"
- L7 `**` — "You have 8 nodes of 8×H100. Map tensor, pipeline, data, and expert parallelism onto this cluster's fabric, justifying the mapping from link characteristics—then tell me what changes on a GB200 NVL72"

**Done when.** Derivation 7 is automatic and you never quote a
sparsity-inflated peak FLOPs number. H100 dense BF16 is ~990 TFLOPS; the 1979
figure on the marketing sheet is the 2:4-sparsity number and quoting it flags
you instantly.

## Day 12 — [DL 14] Efficient Architectures

MoE, quantization, distillation, pruning, and the serving-budget design
question that is a compressed version of the whole design round.

**Drill:**
- L6 `**` — "When would you choose MoE over dense scaling, and vice versa?"
- L6 `**` — "Estimate the compute savings of a MoE model with 8 experts, top-2 routing, vs a dense model of the same quality"
- L6 `**` — "MoE routing collapse: what is it, how do you detect it, and how do you fix it?"
- L6 `***` — "INT8 vs INT4 vs FP8 quantization—what is your decision framework?"
- L6 `**` — "Your quantized model performs well on benchmarks but fails on specific domains (code, math). Why?"
- L7 `***` — "You have a serving budget of 80 GB of GPU memory and 20 ms/token decode latency, with contexts up to 128K. Design the model"
- L6 `**` — "Knowledge distillation: design a pipeline to compress a 70B teacher to a 7B student"

**Done when.** You quote MoE total and active parameters separately without
being asked, and know that compute follows active, memory follows total, and the
KV cache is unaffected by either.

## Day 13 — Mock and consolidation

- **75 min.** Full training-side system design, out loud, timed:
  *"Design the training infrastructure for a 400B-parameter model on 16k H100s:
  parallelism layout, data pipeline, checkpointing, and failure recovery at a
  45-minute MTBF. What is your MFU target and why?"* Impose the five moves from
  [DL 29] §"ML System Design at Staff Level": constraint discovery, quantify
  early, propose then iterate, failure modes as first-class, the eval plan.
  Halfway through, mutate a constraint on yourself — halve the GPU allocation —
  and re-run the arithmetic rather than defending the old design.
- **30 min.** Derivations 3, 4, and 7 on paper.
- **45 min.** Re-read the answers you failed.

## Day 14 — Off

---

# Week 3 — Post-training, reasoning, and evaluation

Theme: everything after the base model, and how you know it worked.

## Day 15 — [NLP 7] Large Language Models and In-Context Learning

Scaling laws in the LLM framing, chain-of-thought, ICL, MoE, test-time compute,
tool calling, prompt injection.

**Drill:**
- L6 `***` — "Explain in-context learning. Why can models learn from examples in the prompt without gradient updates?"
- L5 `***` — "Why does chain-of-thought prompting improve reasoning? What are its limitations?"
- L6 `**` — "Explain test-time compute scaling (o1-style reasoning). How does it differ from simply making a bigger model?"
- L6 `**` — "What are the key factors that affect in-context learning performance? Why do even random labels partially work?"
- L5 `***` — "How does an LLM actually call a tool? Walk me through the full round trip from tool definition to final answer"
- L5 `***` — "What is prompt injection? How would you defend against it in a production system?"
- L5 `***` — "Compare few-shot prompting, fine-tuning, and RAG for adapting an LLM to a new task or domain. When would you use each?"

## Day 16 — [DL 17] Reinforcement Learning and RLHF

The RL mathematics is this chapter's canonical territory; [NLP 8] tomorrow owns
alignment practice. Read in this order.

**Drill:**
- L6 `***` — "Explain GRPO. Why did it replace PPO's critic for reasoning RL, and what breaks when you use it naively?"
- L6 `***` — "PPO vs DPO vs RLHF—complete comparison for LLM alignment"
- L5 `***` — "Why does the KL penalty matter in RLHF? What happens without it?"
- L6 `***` — "Your RLHF-tuned model becomes sycophantic (agrees with everything). Diagnose and fix"
- L6 `**` — "Your GRPO run's mean reward climbs but responses grow unboundedly long and benchmark scores stall—diagnose"
- L7 `*` — "Your RLHF training shows the reward score plateauing while KL divergence keeps climbing. What is happening?"
- L7 `***` — "When is test-time compute cheaper than a bigger model—and when does it stop working?"
- L6 `**` — "Process reward models vs outcome reward models—what are the tradeoffs and when does each win?"
- L6 `**` — "Estimate the memory required to run RLHF (4 models: policy, reference, reward model, critic)"

**Done when.** Derivation 6 is on paper, and you can say precisely what the
critic was estimating that group-relative advantages replace.

## Day 17 — [NLP 8] RLHF and Alignment

Alignment practice and the RLVR recipes; also the home of the DPO derivation.

**Drill:**
- L7 `**` — "Derive the DPO loss from the RLHF objective"
- L7 `***` — "Design the RLHF/RLVR training system for a 70B model"
- L6 `***` — "Explain DPO. How does it relate to RLHF? What are its advantages and limitations?"
- L5 `***` — "What is reward hacking? How does the KL penalty help? When does it fail?"
- L6 `**` — "Compare RLHF, DPO, and KTO. When would you use each?"
- L5 `**` — "What makes a good reward model? How do you evaluate it?"
- L6 `**` — "What is Constitutional AI? How does self-critique enable alignment without human feedback at every step?"

**Done when.** Derivation 5 runs end to end on one sheet: the KL-constrained
objective, its closed-form optimal policy, the substitution that removes the
partition function, and the resulting pairwise logistic loss.

## Day 18 — [DL 23] Evaluation Metrics

The statistics of shipping decisions. Ten of twelve questions are `***`.

**Drill.** All ten `***` (listed in
[`two-week-breadth.md`](two-week-breadth.md) day 10), with two carrying the
depth-round weight:
- L7 `***` — "Design the eval stack that decides whether tomorrow's checkpoint ships"
- L6 `***` — "Model B is 1.5 points better than model A on MMLU—do you ship it?"

**Done when.** You compute a benchmark standard error out loud and can say
whether a 1.5-point MMLU delta clears it at that eval's size.

## Day 19 — [NLP 10] Evaluation, Metrics, and Decoding

The benchmark landscape, LLM-as-judge practice, contamination, decoding
strategies. [DL 23] owns the statistics; this chapter owns the landscape.

**Drill:**
- L6 `**` — "How do you handle benchmark contamination? How would you design an evaluation pipeline that is robust to it?"
- L5 `**` — "What are the problems with LLM-as-judge evaluation? When would you use it and when would you not?"
- L6 `**` — "A model achieves state-of-the-art on MMLU but users report it gives poor answers in practice. What could explain this gap?"
- L6 `***` — "Your model gained 6 points on the internal benchmark suite, but users can't tell the difference—what happened?"
- L5 `**` — "Compare greedy decoding, beam search, and nucleus sampling. When would you use each?"
- L5 `***` — "What is perplexity? How does it relate to cross-entropy? What are its limitations?"

## Day 20 — Mock and consolidation

- **60 min.** Post-training depth grill: "Why does RLHF use PPO with a KL
  penalty? Derive DPO from the RLHF objective. When does DPO fail where PPO does
  not? What changed with GRPO/RLVR?" — then push yourself for the failure mode
  and the discriminating experiment at each step.
- **45 min.** Design round: "Design the eval platform for a frontier lab:
  thousands of evals, contamination control, statistical significance at small
  effect sizes, CI-like gating for training runs."
- **30 min.** Derivations 5 and 6 on paper.
- **30 min.** Update the opinion file: a post-training claim, its mechanism, its
  falsifier.

## Day 21 — Off

---

# Week 4 — Inference, context, agents, safety, and the loop

Theme: what you ship, and how you talk about it.

## Day 22 — [DL 19] Inference Optimization and LLM Serving

The design round's home chapter. Continuous batching, paged KV, speculative
decoding, prefill/decode disaggregation, quantization, capacity planning.

**Drill:**
- L7 `***` — "Design an LLM serving stack that handles 1,000 queries per second with a p99 latency of 2 seconds for a 70B parameter model"
- L7 `***` — "Your capacity plan assumed 500-token outputs. The reasoning model you are now deploying emits ~20K thinking tokens per query. Redo the plan"
- L7 `**` — "Calculate the throughput (tokens/sec) of a 70B model on 8×H100s with tensor parallelism"
- L7 `*` — "Your LLM service needs to handle 500 concurrent users with 2-second time-to-first-token. Size the infrastructure"
- L7 `*` — "Prefill-optimized vs decode-optimized serving—when should you disaggregate them?"
- L6 `***` — "KV cache memory is your bottleneck—you are running out of GPU memory and dropping requests. What do you do?"
- L6 `**` — "Your LLM serving system's p99 latency is 10× the p50. Diagnose"

**Done when.** You size prefill and decode as *two separate capacities* without
being prompted, and check KV feasibility before declaring a node count. Sizing
only decode is the classic failure on input-heavy workloads.

## Day 23 — [DL 21] Long Context + [NLP 9] Retrieval-Augmented Generation

[DL 21] owns the context-versus-retrieval decision; [NLP 9] owns the RAG
pipeline. [DL 21] is short (524 lines, 4 questions) — spend the day's weight on
[NLP 9].

**Drill:**
- [DL 21] L6 `***` — "Your 8K-trained model must serve 64K contexts next month. What are your options, and what do they cost?"
- [DL 21] L6 `***` — "Estimate the cost per query for a RAG system vs a long-context LLM (128K)"
- [DL 21] L6 `**` — "Your model's quality degrades beyond 32K context. Walk me through the diagnosis"
- [NLP 9] L5 `***` — "Design a RAG system for a customer support chatbot that answers questions using your company's knowledge base"
- [NLP 9] L5 `***` — "Compare BM25 vs. dense retrieval. When would you use each? When would you combine them?"
- [NLP 9] L5 `***` — "How would you evaluate a RAG system end-to-end? What metrics would you track?"
- [NLP 9] L6 `**` — "Your RAG system is returning correct documents but the LLM's answers are still wrong. What would you investigate?"
- [NLP 9] L6 `**` — "How would you scale a RAG system to handle 10 million documents and 1000 queries per second?"

## Day 24 — [DL 27] Agents and Tool Use

The 2026 addition to every frontier loop: the agent loop, harness design,
memory and compaction, orchestration, reliability arithmetic, evaluation, and
economics.

**Drill:**
- L5 `***` — "What makes an LLM system an 'agent'? Walk me through the agent loop and where the engineering difficulty actually lives"
- L6 `***` — "When should you NOT build an agent? Your PM wants 'an agent' for a document-processing product—how do you decide?"
- L6 `***` — "Design a code-review agent for your organization's monorepo"
- L7 `***` — "Your agent succeeds on 60% of an internal task suite. Leadership wants 95%. Take it there"
- L7 `**` — "Your agent's tasks run for hours and blow past the context window. Design its memory and state management"
- L5 `**` — "Explain pass@k vs. pass^k. Your agent team reports 92% pass@5—what do you tell the VP who wants to ship?"
- L6 `**` — "Your agent product works but loses money: $4 of inference per task against $1.50 of revenue. Fix the economics without wrecking the success rate"
- L6 `**` — "One agent with 30 tools, or an orchestrator with specialized sub-agents? Argue both sides with numbers"

**Done when.** The compounding-error arithmetic is instant: per-step reliability
raised to the trajectory length, and what that implies about the difference
between a 95% step and a 99% step over 20 steps.

## Day 25 — [DL 20] Safety, Alignment, Interpretability + [NLP 13] Safety, Ethics, Responsible AI

[DL 20] owns technical safety methods and the deployed/agentic framing; [NLP 13]
owns NLP-specific harms, hallucination and faithfulness, and their evaluation.
At Anthropic the values conversation is a genuine veto gate, and this material
is the vocabulary — not the verdict, which comes from your own reading of their
published work.

**Drill:**
- [DL 20] L6 `***` — "Design a content filtering pipeline for an LLM chatbot with <50ms added latency"
- [DL 20] L5 `***` — "Jailbreak attacks: how do they work and how do you defend?"
- [DL 20] L7 `***` — "Design safety for an agent that can browse the web and execute code"
- [DL 20] L7 `**` — "Your LLM refuses too many benign queries. How do you reduce over-refusal without increasing safety risk?"
- [DL 20] L6 `**` — "A jailbreak goes viral against your production model—walk me through the first 24 hours"
- [DL 20] L6 `**` — "What is mechanistic interpretability? Give a concrete example of a discovered circuit"
- [NLP 13] L5 `***` — "How do you detect and mitigate hallucinations in an LLM-based system? Describe a practical defense strategy"
- [NLP 13] L6 `**` — "What is red teaming for LLMs? How would you design a red teaming process for a new model before deployment?"

## Day 26 — [DL 29] The Staff Loop

Not a knowledge chapter. Level calibration (the same question at L5, L6, L7),
the project deep dive, the five design moves, behavioral scenarios, research
taste, and execution mechanics.

**Work, not reading:**
- Read §"Level Calibration" and its three worked pairs. For each of the three
  prompts, say your own L6 answer aloud, then read the L7 delta. The gap you
  find is your leveling gap.
- Write the one-page brief for your two strongest projects using §"The Narrative
  Structure": context, constraints, the decision tree with rejected alternatives
  and the evidence that killed them, metric decomposition, failures, aftermath,
  your specific fingerprints. Rehearse the 90-second opening aloud.
- **Drill:** L6 `***` — "Walk me through your most impactful project. (Then 40
  minutes of drilling.)"; L7 `**` — "If we gave you a year and a small team, what
  would you work on, and why?"; L6 `**` — "Here's a paper claiming a new
  post-training method beats RLHF. Critique its evaluation"; L6 `***` — "You're
  two days from a launch your team has crunched for, and an eval shows a small
  but real regression in a harm category. Walk me through what you actually do";
  L6 `***` — "I'm going to give you a deliberately vague prompt—'design memory
  for our assistant.' Run the first ten minutes."

**Done when.** Your project brief survives a friend asking "whose work was
this?" and "decompose that number" three times.

## Day 27 — Final mock

- **60 min.** Serving design: "Design the serving stack for a ChatGPT-scale
  assistant. Give me p50/p99 TTFT and per-token cost targets and defend them."
- **45 min.** Agent-infrastructure design: "Design the sandboxed execution
  infrastructure for training a coding agent with RL: millions of isolated
  environment rollouts a day, snapshotting, reward extraction, containment."
- **30 min.** All eight derivations on paper, timed. Any that takes more than
  five minutes goes on the night-before list.
- **30 min.** The taste conversation, out loud: your three defended opinions,
  each with the evidence that would flip it.

## Day 28 — Off, then [`night-before.md`](night-before.md)

---

# What this track cannot give you

Stated plainly so you notice it now rather than in the loop:

- **The general SWE round.** Build a bounded work queue, find the deadlock,
  profile the pipeline. Not in any of the four volumes. Budget separate time.
- **Hands-on coding fluency.** This track builds the knowledge layer. The
  execution layer is [`coding-week.md`](coding-week.md) and the runnable
  `drills/`; a weak ML-coding round is close to an auto-reject regardless of
  depth. If you have not hand-written attention in a year, run coding week
  *inside* week 3 of this track.
- **The project deep dive and behavioral scope.** Day 26 gives the structure.
  The content is your career.
