SUMMARY: Complete quantitative-fluency layer for 2026 staff ML interviews in 9 sections: a memorized constant card (GPU FLOPs/memory/bandwidth/prices), model anatomy (12·L·d² + V·d), training compute (6ND, MFU 30–55%, H100-hour ≈ 1.4 exaFLOPs at 40% MFU), training memory (16 bytes/param Adam, ZeRO stages), inference (KV = 2·L·n_kv·d_head·bytes·s; decode = bandwidth/bytes; ~$0.35/1M-token floor for 70B), data (Chinchilla N=sqrt(C/120), FineWeb 15T, 4-epoch rule), hardware (ridge points ~160/300/280 FLOPs/byte for A100/H100/B200), serving economics, and four canonical worked examples with spoken-answer scripts and traps. Book coverage: the book teaches most of this and teaches it well, but scattered across chapters 01/02/05/06/15/22 — ch15 (training memory/throughput) and ch22 (inference) are near-complete; ch02 nails Chinchilla. No consolidated numbers chapter or cheat-sheet appendix exists (the appendix is a question index). Material gaps: corpus token inventories and the multi-epoch/data-wall rules (absent), current GPU spec table (H100 BF16 quoted at a dated 495 TFLOPS in ch22; B200/H200 specs absent), explicit roofline ridge-point numbers, the $/1M-token cost formula, and the LoRA param-count formula r·(d_in+d_out). A one-chapter 'Numbers Everyone Should Know' consolidation would close all gaps.

# 0. The Memorized Constant Card (produce these cold, no derivation)
These are the atoms everything else is built from. In an interview you state them without hesitation; hesitating on any of these reads as not having done the work.

**GPU peak compute (dense, no sparsity — always say 'dense'):**
- A100 (SXM, 80GB): **312 TFLOPS BF16/FP16**, 19.5 TFLOPS FP32, no FP8
- H100 (SXM): **~990 TFLOPS BF16** ("call it 1 PFLOP"), **~2 PFLOPS FP8**. Marketing sheets say 1979/3958 — those are 2:4-sparsity numbers; halve them.
- H200: same compute as H100, more/faster memory
- B200: **~2.25 PFLOPS BF16**, **~4.5 PFLOPS FP8**, ~9 PFLOPS FP4 (dense)

**GPU memory:** A100 40/80 GB HBM2e @ 1.6/2.0 TB/s; H100 80 GB HBM3 @ **3.35 TB/s**; H200 141 GB @ 4.8 TB/s; B200 **192 GB HBM3e @ 8 TB/s**.

**Interconnect:** NVLink4 (A100 600 GB/s, H100 **900 GB/s** per GPU); NVLink5 (B200 1.8 TB/s); InfiniBand NDR **400 Gb/s = 50 GB/s per NIC** (~8 NICs/node → ~400 GB/s/node); PCIe 5.0 x16 = 64 GB/s. Rule: NVLink is ~10x IB — TP inside a node, PP/DP across nodes.

**On-chip SRAM:** A100 ~40 MB L2 + ~20 MB shared/L1 (192 KB × 108 SMs); H100 50 MB L2 + ~30 MB shared (228 KB × 132 SMs). "SRAM is ~20x HBM bandwidth (~19 TB/s effective on A100) and ~2000x smaller" — this is the entire justification for FlashAttention.

**Bytes per parameter:** FP32 = 4, BF16/FP16 = 2, FP8/INT8 = 1, INT4 = 0.5.

**Unit conversions:** 1 token ≈ 4 chars ≈ 0.75 words; 1 GB of text ≈ 250M tokens; 1 exaFLOP = 1e18; a day ≈ 86,400 s ≈ 1e5 s; a month ≈ 730 hours; a year ≈ 8,760 hours.

**Price anchors (2026):** H100 ≈ **$2–3/hr** commodity cloud ($4–10 hyperscaler on-demand); A100 ≈ $1–2/hr; 8×H100 node ≈ **$20–25/hr**; B200 ≈ $5–8/hr early. Frontier API pricing ≈ $3–15 per 1M input / $15–75 per 1M output tokens; small models ≈ $0.10–1.

**MFU:** 30–40% typical, 40–50% good, 55%+ excellent (dense BF16). Never claim above 60%.

BOOK COVERAGE: Absent as a consolidated card. Individual constants scattered: A100 312 TFLOPS in 01/02/05/08/15; H100 3.35 TB/s and NVLink 900 GB/s in 22; IB 50–100 GB/s in 15/22; SRAM/HBM in 05 (FlashAttention). H100 BF16 peak appears only as '495 TFLOPS' in 22_inference_optimization.tex:526 — a dated/halved figure a candidate should not repeat (dense H100 BF16 is ~990). B200/H200 appear only as names (15:1293) with no specs. No cheat-sheet chapter or appendix exists (appendix is a question index).

# 1. Model Anatomy — where the parameters live
**Formulas.**
- Attention block (MHA): W_Q, W_K, W_V, W_O each d×d → **4d² per layer**. With GQA: 2d² (Q,O) + 2·d·(n_kv·d_head) — KV projections shrink by n_heads/n_kv.
- FFN with multiplier f (d_ff = f·d): **2f·d²** (up + down). SwiGLU has 3 matrices with d_ff ≈ (8/3)d → still **≈8d²**.
- **Per layer ≈ 12d²** (4d² attn + 8d² FFN). ~2/3 of params in FFN, ~1/3 in attention.
- **Total: N ≈ 12·L·d² + V·d** (×2 embeddings if untied). Norms and biases are negligible (~2d/layer).
- Shape conventions: d_head = 128, so n_heads = d/128; aspect ratio d/L ≈ 100–130 for dense models.

**Memorized configs:** Llama-7B: L=32, d=4096, V=32K. Llama-70B: L=80, d=8192, n_kv=8 (GQA), V=32K. GPT-3 175B: L=96, d=12288. Modern vocab 128K (Llama-3), embedding at 128K×4096×2B ≈ 1 GB.

**Worked example (as spoken):** "Sanity-check Llama-7B: 12·L·d² = 12 × 32 × 4096². 4096² is 16.8M, times 12 is ~200M per layer, times 32 layers is ~6.4B. Add embeddings: 32K vocab × 4096 ≈ 130M, times two for untied input/output ≈ 260M. Total ≈ 6.7B — matches the nominal 7B. Two-thirds of that is in the MLPs."

**Traps:**
- Forgetting SwiGLU has 3 matrices (people compute 2·d·d_ff and get FFN wrong by 50%).
- Using 12d² for a GQA model's attention (Llama-70B attention is ~2.15d², not 4d²; total comes to ~68B via d_ff=28672, i.e. f=3.5).
- For small models (<1B) embeddings can be 20%+ of params (BERT-base: 23M/110M ≈ 21%); for 70B they're <1%.
- MoE: quote total vs active params separately (Mixtral 8x7B: 47B total, 13B active); compute cost follows active, memory follows total, KV cache unaffected by MoE.

BOOK COVERAGE: Well covered but scattered. 05_attention_transformers.tex:235,854–857 gives 4d² attention and (4+2f)d² per layer with param counting; :1055 and :1442–1444 give FLOPs per layer and attention/FFN crossover at n≈d. Embedding tables: 01_mathematical_foundations.tex:639–641 (BERT/GPT-2 sizes, 21% share) and 06_embeddings_representations.tex:553–565 (50K×768 and 128K×4096 worked examples, weight tying). GQA/KV heads in 22. MoE total-vs-active and 'KV cache unaffected by MoE' in 14_efficient_architectures.tex:593. Missing: a memorized table of famous model configs (L, d, n_kv, d_ff, V) and the SwiGLU 3-matrix trap called out explicitly.

# 2. Training Compute — 6ND and time-to-train
**Formulas.**
- **C ≈ 6ND** FLOPs (N params, D tokens): 2 FLOPs/param/token forward + 4 backward. Forward alone is 2ND — that's your inference-FLOPs formula too.
- Attention adds ~12·L·s·d FLOPs/token (score+value matmuls); ignorable while s ≪ d·(model width terms) — at s=8K on a 7B it's still <15% and usually excluded from the 6ND quote.
- Throughput: **tokens/s = (n_gpus × peak FLOPs × MFU) / 6N**.
- Gradient checkpointing recompute makes it 8ND (one extra forward) — say so if asked why measured FLOPs exceed 6ND.

**Memorized:** FLOPs per GPU-hour at 40% MFU: A100 ≈ **4.5e17**; H100 BF16 ≈ **1.4e18** ("an H100-hour is ~1.4 exaFLOPs of useful work"); H100 FP8 ≈ 2.8e18 at same MFU (FP8 MFU usually a bit lower, 35–45%). Reference runs to sanity-check against: Llama-2 7B = 2T tokens = **184K A100-hours (~3K tok/s/GPU)**; Llama-3 405B = 15.6T tokens ≈ 3.8e25 FLOPs = ~31M H100-hours at ~40% MFU; GPT-4 rumored ~2e25.

**Worked example (as spoken):** "Train a 7B on 2T tokens on H100s. C = 6 × 7e9 × 2e12 = 8.4e22 FLOPs. One H100-hour at 40% MFU gives 990e12 × 0.4 × 3600 ≈ 1.4e18. So 8.4e22 / 1.4e18 = **60K H100-hours**. On 512 GPUs that's ~120 hours — five days. Sanity check: Meta reported 184K A100-hours for exactly this run, and an H100 is ~3x an A100 in BF16, so 60K checks out."

**Traps:**
- Using sparsity-inflated peak FLOPs (1979 for H100) — instantly flags you.
- Assuming 100% utilization; also quoting MFU >60% for real training.
- Forgetting 6ND is per epoch — multiply by epochs.
- Mixing up 2ND (forward/inference) vs 6ND (training); and for MoE, N = active params.
- Not sanity-checking against a published run — the L7 move is always to anchor on Llama-2's 184K A100-hours.

BOOK COVERAGE: Well covered. 02_learning_theory.tex:279–356 has 6ND, its 2+4 derivation, and the 10^22-FLOP Chinchilla worked example converted to GPU-hours (:336). 15_training_optimization.tex:750 and :1124–1166 is a full worked tokens/sec example on 8×A100 with MFU 40–55% and the Llama-2 184K A100-hour sanity check (:1141). 05:1078 has the 7B Chinchilla cost (~30K A100-hours). Gaps: H100/B200 peak numbers for this arithmetic are absent or dated (the 495 TFLOPS figure in ch22); no FLOPs-per-GPU-hour constant is stated as a reusable atom; gradient-checkpointing→8ND not connected to the compute estimate.

# 3. Training Memory — weights, grads, optimizer states, activations, ZeRO
**Formulas.**
- Mixed-precision AdamW: BF16 weights (2) + BF16 grads (2) + FP32 master weights (4) + FP32 m (4) + FP32 v (4) = **16 bytes/param** (quote 16–20; the book's 18P GB rule for P billion params is fine). The 12 bytes of FP32 state dominate.
- Activations (per layer, FP16, no recompute, classic formula): ≈ s·b·d·(34 + 5·a·s/d) bytes; the second term is the attention matrix and vanishes with FlashAttention. Working rule: **activations ≈ 16–34 bytes × b × s × d × L**, cut to ~2 bytes×b×s×d×L with full checkpointing at the cost of ~+1/3 compute.
- ZeRO-1 shards the 12B optimizer bytes across N_dp GPUs; ZeRO-2 adds grads; ZeRO-3/FSDP shards everything → per-GPU ≈ 16P/N_gpus + activations, at higher comms cost.
- LoRA memory: frozen base needs only 2 bytes/param (no grads, no optimizer, no master); the 16-byte tax applies only to adapter params.

**Memorized:** 7B model ≈ **112 GB** of states (does not fit one 80GB GPU); 70B ≈ 1.12 TB states (weights 140 + grads 140 + optimizer 560 + master, per the 16B/param split) → needs ZeRO-3/3D parallelism across ≥16 GPUs before activations.

**Worked example (as spoken):** "Full fine-tune of a 7B with AdamW in BF16: 16 bytes/param → 112 GB for parameter-linked state alone. Add activations — at b=4, s=4096, d=4096, 32 layers with checkpointing, call it 20–40 GB. So ~140–150 GB: two 80GB GPUs minimum with ZeRO-2/3, comfortable on four. With LoRA instead: base at 2 bytes = 14 GB, adapters ~40M params × 16B ≈ 0.6 GB, plus activations — fits one A100-80GB. QLoRA: base in NF4 = 3.5 GB — fits a 24GB consumer card."

**Traps:**
- Saying 'LoRA saves memory because fewer params' — the real saving is no grads/optimizer/master on the frozen base.
- Forgetting activations entirely (they dominate at long sequence/large batch), or forgetting checkpointing costs ~33% more compute.
- ZeRO-1/2 do NOT reduce the 2-byte weight or activation footprint — only 3 shards weights.
- Inference ≠ training: inference needs just 2P bytes (+KV); people apply 16x to serving by mistake.

BOOK COVERAGE: Well covered — the strongest quantitative area of the book. 15_training_optimization.tex:691–699 (bytes/param table, 18P GB rule), :531–624 (ZeRO stages table + DDP-vs-FSDP figure), :1053–1115 (worked 13B: full 240–270 GB vs LoRA 56–86 GB vs QLoRA 37 GB, with the exact 'LoRA misconception' trap at :1106), :811–830 (70B on 256 GPUs with 3D parallelism, 560 GB optimizer states, ZeRO-1 across DP replicas). Activation formula given only as ranges (30–60 GB), not the s·b·d·(34+5as/d) form. Scattered across one chapter but internally consolidated; no gap of substance.

# 4. Inference — KV cache, bandwidth-bound decode, tokens/sec, batching
**Formulas.**
- **KV cache = 2 × L × n_kv × d_head × bytes × seq_len** (2 = K and V). Per-token KV is the number to memorize per model.
- **Decode is memory-bandwidth-bound at low batch: tokens/s per stream ≈ aggregate HBM bandwidth / bytes touched (weights + this stream's KV)**. Prefill is compute-bound: prefill time ≈ 2N·s_prompt / (peak×MFU).
- Batching: batch B reuses the same weight read → throughput ≈ B× until arithmetic intensity hits the roofline crossover (B of ~200–400), then compute-bound; per-token latency starts degrading.
- Cost: **$/1M output tokens = node $/hr ÷ (tok/s × 3600) × 1e6**.

**Memorized:** Llama-70B (GQA, FP16): **~320 KB/token** → 1.3 GB at 4K, **~40 GB at 128K** — one request's KV can exceed weight memory. Llama-8B: ~128 KB/token. Single-stream 70B FP16 on 8×H100: 140 GB / 26.8 TB/s aggregate ≈ 5.2 ms → **~150–190 tok/s theoretical, ~100–150 real**. Full node at high batch: **15–25K output tok/s**. TTFT for a 2K prompt on TP=8 H100: ~70–140 ms. ITL targets: 20–50 ms interactive.

**Worked example (as spoken):** "Why is decode ~120 tok/s but prefill chews 2K tokens in 100 ms? Decode at batch 1 does 2 FLOPs per byte of weight read — arithmetic intensity ~1, versus the H100 roofline crossover at ~300 FLOPs/byte. We're paying for 3.35 TB/s of bandwidth and idling 99% of the FLOPs. That's the whole case for batching: at batch 128, the same weight read serves 128 tokens, so ~15–20K tok/s per node, and it's also why GQA and KV quantization matter — at long context the KV read, not the weight read, becomes the bandwidth bill. Cost check: $25/hr node ÷ 20K tok/s → about **$0.35 per 1M output tokens** at full utilization; at realistic 40% utilization, ~$1."

**Traps:**
- Sizing memory for weights only and forgetting KV cache (the classic fail).
- Treating weight quantization as a fix for KV pressure — different memory pools; quantize the KV cache (FP8 KV) or use GQA/MLA.
- Quoting throughput without stating batch size, or conflating single-stream tok/s with node throughput.
- Forgetting prefill and decode must be capacity-planned separately (input-heavy workloads are compute-bound even at serving time).
- Multiplying KV by n_heads on a GQA model instead of n_kv (8x error on Llama-70B).

BOOK COVERAGE: Well covered — ch22 is effectively an inference-numbers chapter. 22_inference_optimization.tex:27–42 (exact KV formula + Llama-2 70B worked example incl. 40 GB @128K), :282–298 (prefill/decode, arithmetic intensity 100+ vs 1–10 FLOPs/byte table), :512–543 (full 70B-on-8×H100 throughput derivation: 5.2 ms weight read, batch crossover, 15–25K tok/s), :594–606 (500-user capacity plan with $25/hr and $/user), :474–494 (KV-bottleneck debugging incl. the weight-vs-KV-quantization trap at :492). 14:613 has the 7B/100K-context 50GB KV example. Gap: cost-per-1M-tokens is never derived as an explicit formula (only $/hr per user), and no comparison against API list prices.

# 5. Data — corpora sizes, Chinchilla, epochs
**Formulas.**
- Chinchilla: **D ≈ 20N**; with C = 6ND ⇒ C = 120N² ⇒ **N_opt = sqrt(C/120)**, D_opt = 20·N_opt. Both scale as C^0.5.
- Inference-optimal ≠ compute-optimal: overtrain small models (Llama-3 8B at 15T tokens = 1875:1).

**Memorized corpus sizes (tokens):** English Wikipedia ~4–5B; C4 ~170B; The Pile ~300B; FineWeb ~**15T** (filtered CommonCrawl; FineWeb-Edu ~1.3T); RedPajama-v2 ~30T raw; public code (Stack v2) ~1T high-quality; arXiv ~30B; books corpora ~100–200B. Ceiling of deduped, filtered high-quality web text: **~10–20T English tokens** — which is why 15–40T-token frontier runs are at the 'data wall' and lean on multi-epoch, multilingual, synthetic, and code data. Reference points: Chinchilla 70B/1.4T; GPT-3 175B/300B (undertrained); Llama-2 2T; Llama-3 15T.

**Epoch rule (Muennighoff 2023):** up to **~4 epochs** over the same data ≈ nearly as good as fresh tokens; returns decay fast beyond, ~worthless past 16 epochs. 6ND counts total tokens seen, epochs included.

**Worked example (as spoken):** "Compute budget 10^22 FLOPs, compute-optimal model? N = sqrt(1e22/120) ≈ sqrt(8.3e19) ≈ **9B params**, D = 20N ≈ **180B tokens**. Sanity: 6 × 9e9 × 1.8e11 ≈ 1e22 ✓. That's ~18K A100-hours at 50% MFU — 1,000 GPUs for under a day. But if this model serves heavy traffic, I'd train a 3B on 600B+ tokens instead and pocket the inference savings."

**Traps:**
- Quoting 20:1 as a law — it shifts with data quality and is a train-compute-optimum only; every production model since Llama has deliberately violated it.
- Confusing raw CommonCrawl size (petabytes of HTML) with usable tokens (~single-digit trillions per aggressive filter).
- Forgetting tokenizer dependence (~1.3–1.4 tokens/word English; more for code/non-English).

BOOK COVERAGE: Partial. Chinchilla is excellent: 02_learning_theory.tex:244–356 (20:1, C=120N² worked example, training- vs inference-optimal incl. Llama-3 8B 1875:1 at :291) plus a budget table in 19_decision_frameworks.tex:227. Token/byte conversion appears once (08:887, 4 bytes/token). Absent: corpus token inventory (no mention of FineWeb/RedPajama/Pile sizes anywhere), the ~4-epoch repeated-data rule (Muennighoff), and the data-wall framing. This is the biggest pure-content gap in the book's quantitative layer.

# 6. Hardware — memory hierarchy and the roofline crossover
**Formulas.**
- Arithmetic intensity AI = FLOPs / bytes moved. **Compute-bound iff AI > peak FLOPs / bandwidth** (the ridge point).
- Ridge points to memorize: A100 = 312/2.0 ≈ **160 FLOPs/byte**; H100 = 990/3.35 ≈ **300 FLOPs/byte**; B200 = 2250/8 ≈ 280. In BF16 elements, double those. Practical reading: a matmul needs an effective batch/inner dimension of ~300 to saturate an H100 — hence decode (AI≈1–2) is hopeless without batching, and each GPU generation raises the bar (compute grows faster than bandwidth).
- All-reduce time ≈ 2·(message size)/bandwidth (ring, large-message limit); a TP all-reduce per layer is <1 ms on NVLink, ~10x worse on IB — hence 'TP within node, PP/DP across'.

**Memorized hierarchy (H100):** SRAM ~30 MB @ effective ~20+ TB/s → HBM 80 GB @ 3.35 TB/s → NVLink 900 GB/s → IB ~50 GB/s/NIC → PCIe 64 GB/s. Each level ~4–10x slower; FlashAttention, fused kernels, and tiling exist to keep work in the top level.

**Worked example (as spoken):** "Is batch-1 decode of a 70B FP16 on H100 compute- or memory-bound? Per token ~2 FLOPs/param and 2 bytes/param read → AI ≈ 1 FLOP/byte vs a ridge of ~300. Bandwidth-bound by 300x — the GPU delivers <1% of peak FLOPs. To reach the ridge I need ~300 tokens sharing each weight read: batch ~256–512, exactly where vLLM-style continuous batching operates."

**Traps:**
- Using sparsity peaks in the ridge calculation.
- Assuming NVLink bandwidth across nodes (it's IB, 10x less) — cross-node TP is the canonical wrong answer.
- Forgetting that at long context KV reads shift decode's bottleneck from weights to cache.
- FLOPs ≠ latency for low-AI ops (LayerNorm, softmax, depthwise convs): they're bandwidth-priced.

BOOK COVERAGE: Partial and scattered. SRAM/HBM sizes and bandwidths for A100 in 05_attention_transformers.tex:683,717–718 (FlashAttention treatment is good: 20 MB SRAM vs 40 GB HBM, 19 vs 2 TB/s); NVLink 900 GB/s and IB 50–100 GB/s with the TP-within-node rule in 22:354,689–699 and 15:1233–1255; arithmetic-intensity table in 22:296; roofline named at L7 level in 15:1163, 22:543, 01:69–72. Absent: explicit ridge-point numbers (FLOPs/byte crossover per GPU), any B200/H200/NVLink5 specs, all-reduce cost model, and a single memory-hierarchy table.

# 7. Serving Economics — GPU-hours, utilization, $/token
**Formulas.**
- **$/1M output tokens = (cluster $/hr) / (output tok/s × 3600) × 1e6**; divide again by utilization (real fleets run 30–60% useful utilization — traffic diurnality, failover headroom, prefill/decode imbalance).
- Input tokens cost ≈ forward FLOPs: 2N per token; roughly, input tokens are 3–10x cheaper to serve than output tokens (compute-bound and batchable), which is why APIs price them ~3–5x lower.
- Monthly GPU math: 1 GPU-month ≈ 730 GPU-hours; an 8×H100 node ≈ **$15–18K/month** commodity.

**Memorized:** H100 $2–3/hr commodity (own-DC amortized ~$1.5/hr: ~$30K capex over 3–4 yrs + power ~700W ≈ $0.10/hr electricity); A100 $1–2; 8×H100 node $20–25/hr. API anchors: frontier ~$3–15 in / $15–75 out per 1M; mid-tier ~$0.25–3; serving floor for a 70B-class model ≈ **$0.3–1 per 1M output tokens** at scale.

**Worked example (as spoken):** "Does a $10/1M-output API price make money on a 70B? Node cost $25/hr, ~20K output tok/s at high batch → 72M output tokens/hr → hardware floor ≈ $0.35/1M. At 40% utilization, ~$0.90. Against $10 that's a ~90% gross margin — the margin funds the free tier, long-context requests (KV shrinks batch size, cutting throughput 2–5x), and idle capacity. Inverting: any price below ~$0.50/1M for a 70B-class model is at or below cost."

**Traps:**
- Quoting throughput-at-peak-batch as if it holds at production latency SLOs (p99 targets cost you 2–3x throughput).
- Ignoring utilization — the single biggest term; 100% assumed utilization halves-to-thirds your real cost estimate.
- Forgetting long-context requests destroy batch size via KV memory, so $/token is context-length dependent.
- Comparing your serving cost to API prices without matching quality tier and context length.

BOOK COVERAGE: Partial. Concrete anchors exist: 22:606 (8×H100 ≈ $25/hr, $/user/hr sanity check), 07_similarity_metric_learning.tex:1583–1587 ($3/hr A100, QPS→$/month for a reranker), 02:336 (GPU-hours costing for training). 17_production_systems.tex touches cost qualitatively. Absent: the $/1M-token formula and worked derivation, utilization as a first-class economic term, own-DC amortization math, and any API-price anchoring. No section treats serving economics as its own quantitative topic.

# 8. Classic Worked Examples — the four you must nail end-to-end
**(a) Memory to full fine-tune a 7B (Adam, mixed precision).** "16 bytes/param: 2 weights + 2 grads + 12 FP32 optimizer/master → 112 GB before activations. Activations with checkpointing at b=4, s=4K: ~20–40 GB. ~140 GB total → 2×A100-80GB with ZeRO-2 minimum. LoRA: 14 GB frozen base + <1 GB adapter state → one GPU. QLoRA: 3.5 GB base → consumer card." Trap: forgetting activations, or claiming one 80GB GPU suffices.

**(b) Serve a 70B at 1K QPS (500 in / 200 out tokens).** "Two separate capacities. Decode: 1K QPS × 200 = 200K output tok/s; one 8×H100 node does ~20K → **10 nodes**. Prefill: 500K input tok/s × 2N = 2×70e9 FLOPs/token = 7e16 FLOP/s; a node in FP8 at 40% MFU gives ~6e15 → **~11 nodes**. So ~20 nodes / 160 H100s ≈ $500/hr ≈ $0.14 per 1K requests. Check KV: per node ~2K concurrent-stream-equivalents × 0.3 MB/token × ~700 avg tokens ≈ manageable under 640 GB with paging; quantize KV to FP8 for headroom." Trap: sizing only decode (input-heavy workloads flip the bottleneck), forgetting KV feasibility check.

**(c) LoRA parameter count.** "Per adapted W (d_in×d_out): **r·(d_in+d_out)**. Llama-7B, r=16, all attention + MLP: attention 4 matrices, 16×8192 ≈ 131K each → 0.5M; MLP 3 matrices at 4096↔11008, 16×15104 ≈ 242K each → 0.7M; ~1.25M/layer × 32 ≈ **40M ≈ 0.6% of the base**. r=8, q&v only: ~4M." Trap: writing r·d_in·d_out (that's the full matrix — the whole point is it's additive in dims, not multiplicative).

**(d) Embedding table.** "V×d×bytes: 128K vocab × 4096 × 2B ≈ **1 GB** (×2 untied). BERT-base 30K×768×4B ≈ 90 MB ≈ 21% of the model. RecSys DLRM: 1e9 IDs × 128-dim FP32 = **512 GB — one feature exceeds any GPU**, hence sharded/hashed embedding servers; the MLP on top is <0.1% of params." Trap: forgetting the output/LM-head copy, and not flagging that in RecSys embeddings ARE the model.

**Delivery pattern for all of these:** state the formula → plug numbers rounded to one significant figure → sanity-check against a known real system → name the binding constraint and one lever to relax it. The sanity check is what separates L6 from L5.

BOOK COVERAGE: Mostly present, scattered. (a): 15:1053–1115 covers the 13B variant fully (full vs LoRA vs QLoRA table at :1094–1099); 7B states at :521. (b): 22:406–414 and :594–606 are close cousins (400 concurrent requests/node; 500-user sizing) but no QPS-driven dual prefill/decode capacity plan. (c): LoRA memory is thoroughly done (15:1072–1083) but the r·(d_in+d_out) parameter-count formula itself is never stated as a formula; 16_transfer_learning covers LoRA conceptually. (d): 06:553–565 and 01:639–641 cover NLP tables; 12_recommendation_systems.tex:781–793 covers the trillion-param DLRM story qualitatively (no GB-level worked shard math). The answer-delivery pattern (formula → round → sanity-check → lever) is modeled implicitly by many strong answers but never taught explicitly.

