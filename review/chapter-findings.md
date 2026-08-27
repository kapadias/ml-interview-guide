# sections/01_mathematical_foundations.tex (Mathematical Foundations)

VERDICT: Well-structured with genuinely useful estimation questions and L5-L7 ladders, but it contains two real mathematical errors an expert interviewer would pounce on (the attention-rank/softmax claim and the VAE forward-KL claim), the probability/statistics section is interview-question-free filler, and the numerics section is frozen in the A100/BF16 era with no FP8.

STAFF-LEVEL: The interviewq scaffolding is genuinely L6-capable: estimation questions carry real hardware numbers, and L7 ladder bullets name the right depth markers (edge of stability, dynamical isometry, roofline). But the connective 'have you done this' tissue is uneven: the probability/statistics and norms sections are L4 reference tables with zero questions; there is no linear-layer-backward-with-shapes derivation, no dollar-cost endpoint on any estimate, and the deepest section (attention rank, badged L7) is the one containing the chapter's worst error — a Staff candidate who studies it will recite a false claim to exactly the interviewer most likely to catch it. Scale intuition is good for memory but stops at 2021 hardware; add H100/B200 and FP8 or the numbers candidates practice will be the wrong ones.

MISSING (critical/high):
- [critical] Backward pass of a linear layer with explicit shapes (Y = XW: dL/dW = X^T dL/dY, dL/dX = dL/dY W^T) and matrix-calculus layout conventions :: This is the single most common 'derive it on the whiteboard' math question at frontier labs — the chapter teaches the chain rule abstractly but never makes the candidate produce the weight-gradient shapes that interviewers actually ask for.
- [critical] Variance-propagation derivation of Xavier/He init and the 1/sqrt(d_k) attention scaling (Var(q·k) = d_k for unit-variance components) :: He init is name-dropped three times but never derived; 'why divide by sqrt(d_k)?' is arguably the most-asked probability question in transformer interviews and it is pure Var-of-sums math that belongs in this chapter (at minimum a cross-reference).
- [high] FP8 training (E4M3/E5M2, DeepSeek-V3-style FP8 GEMMs with FP32 accumulation) and TF32 in the floating-point table :: DeepSeek-V3 made FP8 pretraining a mainstream interview topic by 2025; a 2026 numerics section that stops at BF16/INT8 reads dated, and 'what breaks when you go below 16 bits' is a live L6/L7 question.
- [high] Closed-form KL between two Gaussians and Jensen's inequality / ELBO-supporting math :: The Gaussian KL is the standard 'derive the VAE KL term' warm-up and Jensen's inequality underlies every ELBO derivation; the info-theory section has two overlapping conceptual KL questions but no derivation candidates can be asked to do.
- [high] Score-function estimator / log-derivative trick (∇E[f] = E[f ∇log p]) and the reparameterization trick as the two ways to differentiate through sampling :: This is the mathematical foundation of REINFORCE, PPO, and GRPO; with RLHF chapters downstream, its absence from 'Calculus for Optimization' leaves the book's most 2026-relevant gradient identity untaught.

CORRECTNESS:
- sec:rank_attention (body and L7 interviewq): 'The softmax does not increase rank—it operates row-wise and preserves the column space' is mathematically false. Softmax is nonlinear; exp of a rank-1 matrix is generically full rank (verified counterexample: softmax of rank-1 [[1,2],[2,4]] has nonzero determinant, rank 2). The red-flag bullet 'Claiming the attention matrix is full-rank because softmax is nonlinear' penalizes the correct answer. The true statement: pre-softmax logits have rank <= d_k; post-softmax the matrix is generically full rank exactly but numerically near-low-rank.
- KL interviewq boxes (both, sec:information_theory): VAE ELBO KL term KL(q||p) is listed under FORWARD KL ('posterior must cover prior' / 'approximate posterior must cover true posterior') while variational inference is simultaneously and correctly listed under REVERSE KL. The VAE's KL term is reverse KL (mode-seeking); the follow-up 'Why does the VAE use forward KL?' is wrong in premise and rationale.
- Numerical Considerations warningbox: 'Use BF16 when you need large dynamic range (gradient accumulation, loss scaling)' — loss scaling is the FP16-specific mitigation that BF16 eliminates; listing it as a reason to use BF16 inverts the logic. 'Use FP16 when you need precision (final inference, metrics computation)' is also dubious guidance (FP32 accumulation is standard for metrics).
- Matmul FLOPs interviewq context: calling one (1024,4096)@(4096,1024) product 'the cost of one feed-forward layer' conflates the weight matrix with a 1024-token activation batch; correct only with the sequence dimension stated.
- L7 rank-attention answer: 'Multi-head attention compensates by summing h rank-d_k matrices' — heads are concatenated and output-projected, not summed as attention matrices; loose as stated.
- Randomized SVD complexity 'O(mnk)' in the SVD follow-up is a rough simplification (standard statement is O(mn log k + (m+n)k^2)); acceptable verbally but risky to recite to a numerical-linear-algebra interviewer.

STALENESS:
- Floating-point table omits FP8 (E4M3/E5M2) and TF32; 'Most modern LLM training uses BF16' needs the DeepSeek-V3 FP8 caveat — as of 2026 mixed FP8 pretraining is production practice.
- A100 is the only GPU referenced; H100/H200/B200 are the 2026 interview baseline for TFLOPS and memory-bandwidth numbers.
- Linformer as the flagship efficient-attention example is 2020-vintage; the low-rank discussion should be reframed around why FlashAttention (exact, IO-aware) won and where linear attention survives in 2026 (Mamba-2/GLA-style linear-attention duality, hybrid models) — the FlashAttention follow-up exists but the section body still centers Linformer.
- Information bottleneck presented uncritically; the compression-phase claim was substantially walked back by 2018-2019 replications and citing it as established would draw pushback.
- No mention of quantization formats that dominate 2026 inference (INT4/NVFP4/AWQ-style groupwise scales) in the numerics discussion — even one sentence pointing to the inference chapter would help.

MUST-KNOW:
- Softmax cross-entropy gradient = y_hat - y, derived from the softmax Jacobian, and why frameworks fuse the two ops
- FLOPs of a matmul = 2mnp; attention O(n^2 d) vs FFN O(n d^2) and where the crossover sits
- Bytes per dtype (FP32/BF16/FP16/FP8/INT8) and the ~16 bytes/param Adam mixed-precision training footprint
- Forward vs reverse KL: mode-covering vs mode-seeking, distillation uses forward, VAE/VI and the RLHF policy penalty use reverse
- Chain rule = backprop; reverse-mode vs forward-mode, activation memory, gradient checkpointing trade-off
- Jacobian/spectral-norm products explain vanishing-exploding gradients; residual connections shift eigenvalues to ~1; He/Xavier variance preservation
- SVD/Eckart-Young and the low-rank story behind LoRA (2dr vs d^2 params)
- BF16 vs FP16: same bits, range-vs-precision trade, why BF16 removed loss scaling; LogSumExp/max-subtraction stability
- Cross-entropy = NLL = forward KL up to constant H(p); perplexity = exp(CE)
- Arithmetic intensity and compute-bound vs memory-bound reasoning (roofline) for any kernel you're asked to estimate

IMPROVEMENTS:
- Fix the attention-rank section (sec:rank_attention): replace the false 'softmax does not increase rank / preserves column space' claim with the correct statement — softmax is nonlinear (exp + row normalization) and generically produces a full-rank matrix from low-rank logits; the honest claim is that trained attention matrices are numerically near-low-rank (small stable rank / fast spectral decay), which is why Linformer-style approximations work. Delete or invert the red-flag bullet that penalizes candidates for saying softmax can raise rank — as written it trains candidates to give a wrong answer at L7.
- Fix the KL question pair: move the VAE ELBO term KL(q(z|x) || p(z)) to the reverse-KL/mode-seeking column (it IS variational inference, which the same list correctly places under reverse KL — the two bullets currently contradict each other), and rewrite the follow-up 'Why does the VAE use forward KL?' which is wrong twice over.
- Merge the two heavily-overlapping KL interviewq boxes (L5 'why not a metric' and L6 'when does asymmetry matter') into one, and spend the freed space on a Gaussian-KL derivation question.
- Rewrite the BF16/FP16 warningbox guidance: 'Use BF16 when you need... loss scaling' is backwards — loss scaling is the FP16 workaround that BF16's range makes unnecessary; and 'use FP16 for metrics computation' is bad advice (metrics/reductions go to FP32). State it as: BF16 for training (range, no loss scaling), FP16 mainly for legacy/inference kernels, FP32 accumulation for reductions.
- In the matmul FLOPs question, state the missing dimension explicitly: the (1024,4096)x(4096,1024) product is one FFN projection for a batch of 1024 tokens, not 'one feed-forward layer' in the abstract — interviewers probe exactly this batch/sequence-dimension sloppiness.
- Update hardware anchors: keep the A100 (312 TFLOPS) but add H100 (~990 TFLOPS BF16 dense) and B200-class numbers, and add a dollar figure (e.g., ~$2-4/H100-hour) so estimation answers can end in cost, which is how frontier labs actually ask these.
- Tighten the multi-head rank claim: heads are concatenated and projected, not summed as n-by-n attention matrices; the defensible statement is that the token-mixing operator is a sum over heads of A_h composed with per-head value/output maps, so effective rank can reach h*d_k.
- Add at least one interviewq to the Probability and Statistics section (currently zero) — e.g., 'MLE vs MAP, and derive why L2 regularization is a Gaussian prior' or the sqrt(d_k) variance question — right now that section is reference tables an interviewer would never ask from.
- Clarify the training-memory follow-up with the standard per-parameter byte accounting: BF16 weights (2) + FP32 master (4) + Adam m,v (8) + BF16 grads (2) = 16 bytes/param, so a 7B model needs ~112 GB before activations — the current '16-20x' multiplier is ambiguous about its base.
- Add gradient noise/straight-through and label smoothing gradient effects are present in follow-ups — good — but the mutual information subsection should note that the Tishby compression-phase story is contested (Saxe et al. 2018) rather than presenting the information bottleneck as settled explanation for 'hourglass' architectures.


# sections/02_learning_theory.tex (Learning Theory Essentials)

VERDICT: Strong 2022-2023 chapter — Chinchilla, double descent, grokking, and the emergence debate are handled with real nuance and good trade-off questions — but it is missing the defining 2024-2026 development in its own scope (test-time-compute scaling and RL-elicited capabilities), mis-frames the p >> n 'paradox' for single-epoch LLM pretraining, and has an internally inconsistent GPU-hours calculation in its flagship estimation question.

STAFF-LEVEL: This chapter is closer to genuine Staff level than most prep material: the trade-off questions (shrink the 10B model, size-vs-data, when laws break) test judgment, and the Chinchilla estimation question is a faithful replica of a real interview exercise, GPU-hours slip aside. Where it falls short of L6/L7: the scale intuition stops at FLOPs and never reaches dollars or exponent-to-benefit conversion ('what does 2x compute buy'); the generalization-theory half over-indexes on academic 2019-2022 phenomena (LTH with no question, grokking, benign overfitting) while mis-framing the regime modern LLMs actually train in; and the complete absence of test-time-compute scaling means the chapter cannot support the scaling conversation as it is actually held in 2026 loops. Fix the framing error, add the test-time axis and data-constrained scaling, and this becomes the strongest chapter-level prep for scaling questions available.

MISSING (critical/high):
- [critical] Test-time compute scaling (o1/R1-style reasoning models): trading train-FLOPs for inference-FLOPs, Snell et al. 2024 compute-optimal test-time scaling, sequential vs parallel (best-of-N, self-consistency) test-time strategies :: This is THE scaling-laws development of 2024-2026 and sits squarely in this chapter's 'what scales' scope; a candidate who discusses scaling in 2026 without the test-time axis will be marked stale at any frontier lab.
- [critical] Data-constrained scaling laws (Muennighoff et al. 2023): repeating data up to ~4 epochs is nearly as good as fresh data, then returns decay :: 'We're out of web data — now what?' is a standard L6/L7 follow-up to any Chinchilla answer, and this paper is the canonical quantitative response; the chapter's current answer ('invest in data quality') is qualitative only.
- [high] The joint parametric Chinchilla loss form L(N,D) = E + A/N^alpha + B/D^beta and how scaling laws are actually fit (multiple small runs, IsoFLOP profiles, extrapolation error) :: Staff candidates get asked to design a scaling-law study; the chapter gives only separated single-variable power laws, so a candidate cannot write down the surface Chinchilla actually fit or discuss the three fitting approaches.
- [high] Whether RLVR/post-training creates capabilities or elicits them (pass@k vs pass@1 debate, distillation of reasoning traces a la R1) :: The emergence section stops at 2023; in 2026 the live version of the emergence question is 'does RL create new capabilities or sharpen existing ones', and interviewers at reasoning-model labs ask it directly.
- [high] Single-epoch/online nature of LLM pretraining: train-test gap is small because data is seen ~once and train ~= test distribution; classic multi-epoch overfitting is largely a fine-tuning-regime concern :: The 'Why Large Models Generalize' section frames LLMs as p >> n interpolators, which misdescribes pretraining (300B tokens > 175B params, loss never reaches zero); a Staff candidate must know which regime each generalization story applies to.

CORRECTNESS:
- Chinchilla interviewq (sec:scaling_laws, Q 'model size vs training data'): 'Chinchilla (70B, 1.4T) matches GPT-3 quality with 4x cheaper inference' — 175B/70B = 2.5x; the 4x figure is vs Gopher 280B. Wrong pairing.
- Compute-estimation interviewq context paragraph: '10^22 FLOPs is roughly 1,000 A100 GPU-hours at 50% MFU' directly contradicts its own subsequent (correct) arithmetic of ~18,000 GPU-hours.
- sec:why_large_generalize opening: 'Modern LLMs have far more parameters than training examples (GPT-3: 175B parameters, ~300B training tokens...)' — 300B > 175B; the parenthetical refutes the sentence. The p >> n interpolation framing misdescribes single-epoch pretraining (loss never hits zero; models are not interpolators of the pretraining set).
- 'What does not scale' answer: 'Larger models are often less calibrated (more overconfident) despite being more accurate' — for LLM base models the opposite is documented; miscalibration is primarily a post-training (RLHF) effect. Would draw expert pushback as stated.
- Bias-variance L7 bullet: 'ensemble methods help less for deep learning... DL already has low effective variance' — deep ensembles are a canonically effective technique; overclaimed.
- Unlabeled-data interviewq: 'validation loss plateaued' is diagnosed as 'classic overfitting'; a plateau is ambiguous (could be capacity/data ceiling) — the overfit signal is rising val loss. Also 'What NOT to do: increase model size — this will widen the train-val gap without improving generalization' is stated as a law and conflicts with the chapter's own double-descent section.
- 'Why SGD finds flat minima' item 1 in sec:why_large_generalize says implicit regularization comes from 'small learning rate and large batches' while the flat-minima item credits small-batch noise — the minimum-norm results are for (full-batch) GD on linear models from small init; the current phrasing muddles two distinct mechanisms.

STALENESS:
- No test-time compute / reasoning-model scaling anywhere — the chapter's scaling story ends at Chinchilla-vs-LLaMA, which is a 2023 conversation; by 2026 'the third axis is inference compute' is table stakes.
- Emergence table thresholds ('in-context learning ~10B', 'instruction following ~10B', 'CoT ~100B') are 2022 folklore invalidated by modern sub-1B instruct models and distilled reasoning models; either delete the table or reframe thresholds as artifacts of that era's data quality.
- No mention of data-constrained scaling (2023), inference-aware/'beyond-Chinchilla' scaling laws by name, precision-aware scaling laws (2024), or distillation scaling laws — the 'when scaling laws break' L7 question is the natural home and cites none of them.
- The semi-supervised toolkit (pseudo-labeling, back-translation) in the unlabeled-data answer reads BERT-era; 2026 answers route through foundation-model continued pretraining and synthetic data generation with verification.
- Grokking follow-up 'does it happen at scale?' is fine but predates 2024-25 work connecting grokking to weight-norm dynamics and double descent; one sentence would modernize it.

MUST-KNOW:
- C = 6ND, where the 6 comes from, and the Chinchilla D = 20N rule — plus the ability to invert it for a given FLOP budget in under a minute
- Training-compute-optimal vs inference-cost-optimal: why LLaMA-class models overtrain far past 20:1 and when that is economically rational
- Test-time compute as the third scaling axis: when spending FLOPs at inference (long CoT, best-of-N) beats spending them on a bigger model
- Double descent: three regimes, the interpolation-threshold peak, and implicit regularization / minimum-norm solutions as the over-parameterized mechanism
- Bias-variance: the decomposition is an identity that always holds; only the U-curve intuition breaks in the p >> n regime
- Emergence debate: Wei vs Schaeffer, discrete-metric artifact argument, and the compositional-threshold arithmetic (0.8^5 vs 0.99^5)
- Loss scales smoothly; downstream benchmark accuracy does not — never promise benchmark gains from a loss extrapolation
- Data-constrained regime: repeating data ~4 epochs is nearly free, then returns decay; data quality shifts effective scaling exponents
- LLM pretraining is ~single-epoch online learning: tiny train-test gap, so classic overfitting reasoning applies to fine-tuning, not pretraining
- How to run a scaling study: fit L(N,D) = E + A/N^a + B/D^b on small runs, IsoFLOP profiles, extrapolate with uncertainty and margin

IMPROVEMENTS:
- Fix the internally inconsistent context paragraph in the 10^22-FLOP estimation question: it opens with 'roughly 1,000 A100 GPU-hours at 50% MFU' and then correctly computes ~18,000 GPU-hours two lines later. Change the opening to '~18,000 A100 GPU-hours (e.g., 1,000 A100s for 18 hours)' and add the dollar endpoint (~$20-70k at cloud prices) — cost-in-dollars is how the question is actually asked.
- Correct the Chinchilla comparison: 70B vs GPT-3 175B is 2.5x cheaper inference; the famous 4x figure is vs Gopher (280B). Either switch the comparison to Gopher or fix the multiplier.
- Rewrite the 'Why Large Models Generalize' framing: lead with the regime distinction (single-epoch pretraining = online SGD with tiny generalization gap; the interpolation/benign-overfitting story applies to multi-epoch fine-tuning and vision-style training), then present the five mechanisms. Also fix the self-contradictory setup sentence — 175B params vs ~300B tokens is not 'more parameters than training examples'.
- Soften or correct the calibration claim in 'what does not scale': for LLMs, larger BASE models are typically better calibrated; the well-documented miscalibration comes from RLHF post-training (GPT-4 system card). As written, an expert interviewer would push back.
- Reconsider the L7 bias-variance bullet claiming ensembles 'help less for deep learning' — deep ensembles are a standard, effective uncertainty tool (Lakshminarayanan et al.); either qualify the claim (accuracy gains are modest at scale, uncertainty gains are real) or cut it.
- In the unlabeled-data question, replace 'pretrain from scratch on the unlabeled corpus' as the top strategy with the 2026 answer: start from an existing foundation model and do continued/domain-adaptive pretraining on the unlabeled data, then fine-tune. Also soften 'validation loss plateaued = classic overfitting' (a plateau with train loss falling is a growing generalization gap; rising val loss is the overfit signal) and reconcile 'do NOT increase model size' with the chapter's own double-descent message (bigger + regularized can help; bigger with fixed tiny data may not).
- Add an interviewq to the Lottery Ticket section or fold it into the over-parameterization question — it is currently the only orphan section, and its 2026 interview value lives in its descendants (one-shot LLM pruning: SparseGPT/Wanda; why LTH-style retraining fails at scale without rewinding), none of which are mentioned.
- Add a scaling-law-study design question ('You have 1% of the target budget for experiments — design the runs that predict final loss') and a debugging archetype ('your extrapolation missed by 2x — list causes: LR schedule not scaled, data distribution shift across scales, fitting range too narrow, batch size off the critical-batch curve').
- Flag the compression-phase / information-bottleneck explanation (item 3 of Why Large Models Generalize) as contested rather than established — same Saxe et al. caveat needed as in Chapter 1.
- State exponent magnitudes' practical meaning once: with alpha_C ~ 0.05, 10x compute buys ~ 10^0.05 = ~12% loss reduction — interviewers love 'what does 2x compute actually buy you' and the chapter never converts exponents to intuition.


# sections/03_loss_functions.tex (Loss Functions: The Interview Playbook)

VERDICT: A well-scaffolded, genuinely useful playbook for classic supervised/metric/ranking losses with strong debugging questions — but it reads as a 2021-era MLE-generalist chapter. The losses a 2026 frontier-lab interviewer probes hardest (next-token CE mechanics at scale, loss masking/normalization conventions, DPO/Bradley-Terry, z-loss, SigLIP, MoE auxiliary losses) are entirely absent.

STAFF-LEVEL: Depth is right for L5 and solid L6 for a generalist MLE loop: the debugging questions with ranked hypotheses and the system-design questions (visual search, 100M-doc retrieval, click-data loss design) are genuinely L6. What's missing for a FRONTIER-LAB Staff loop is (a) the LLM loss surface entirely (pretraining CE mechanics, preference losses, auxiliary stabilization losses), and (b) scale numbers — the 2.56B-parameter ArcFace head is the only concrete memory estimate in 1581 lines; there is nothing on logits-tensor memory, similarity-matrix cost at batch 8192, or ANN index footprint. The L7 badges are earned in the multi-task question but nowhere else; several 'L6' design questions (recsys BPR) would read as L5 at an OpenAI/Anthropic loop where the analogous question is about SFT loss masking or DPO.

MISSING (critical/high):
- [critical] Next-token prediction cross-entropy as THE frontier-lab loss: per-token CE over sequences, perplexity = exp(mean NLL), bits-per-byte, typical loss values (~1.5-2.5 nats/token) and why a 0.01 delta matters, loss masking (prompt tokens in SFT, packing boundaries), and the mean-of-means vs global-token-mean gradient-accumulation bug with variable-length sequences (a real 2024-25 ecosystem bug in HF Trainer). :: This is the single most-used loss at OpenAI/Anthropic-tier labs and it gets zero dedicated treatment; the sequence-normalization bug is a canonical staff-level debugging question ('your SFT loss depends on batch composition — why?').
- [critical] Preference-optimization losses as loss functions: Bradley-Terry reward-model loss (-log sigma(r_w - r_l)), the DPO loss and its gradient interpretation (implicit reward, beta as KL strength), with pointers to IPO/KTO/SimPO and GRPO. Even if RLHF has its own chapter, a chapter titled 'Loss Functions' in 2026 must present the DPO equation. :: "Write down the DPO loss and explain what its gradient does" is now a standard frontier-lab screen; candidates prepped from this chapter would be blindsided.
- [high] Auxiliary stabilization losses: z-loss (PaLM's log^2 Z regularizer on the softmax normalizer) and logit soft-capping (Gemma 2), and why bf16 training at scale needs them. :: These are loss-level interventions for the loss-spike questions frontier labs actually ask; they belong next to the numerical-stability section that already exists.
- [high] SigLIP pairwise sigmoid contrastive loss (Zhai et al. 2023, SigLIP 2 in 2025) as the modern answer to 'InfoNCE needs huge batches': removes the softmax batch coupling, so gradient accumulation and device-local computation work. :: The chapter's own SimCLR-batch-size question begs for this answer; an interviewer asking 'how did CLIP successors fix the batch coupling?' expects SigLIP, and its absence dates the section.
- [high] MoE auxiliary losses: load-balancing loss (Switch/GShard form: alpha * N * sum f_i * P_i), router z-loss, and DeepSeek-V3's aux-loss-free bias-based balancing. :: Every frontier model in 2026 is MoE; 'what losses does an MoE add and what goes wrong without them' is a common L6 question and fits squarely in a loss-function taxonomy.
- [high] Cross-entropy systems cost at LLM scale: the B*T*V logits tensor (e.g., 8 x 4096 x 128K vocab in bf16 = ~8.6 GB before the backward) and fused/chunked CE kernels (Liger, Apple's Cut Cross-Entropy 2024). :: Staff interviews test whether you know the loss layer often dominates activation memory in pretraining; this is a natural 'estimate the memory' question the chapter never sets up.

CORRECTNESS:
- Asymmetric Loss mathresult (subsection 'Asymmetric Loss for Multi-Label'): the loss is written WITHOUT negative signs — L = (1-p)^{gamma+} log(p) for y=1 etc. As written both branches are negative and minimizing them drives p toward 0 for positives. The paper defines L+ = (1-p)^{gamma+} log(p) and the LOSS as -L. Add the minus signs.
- Label smoothing 'NOT for multi-label: Not well-defined (targets already soft)': wrong — multi-label targets are hard 0/1 per label; smoothing BCE targets (y -> y(1-eps)+eps/2) is well-defined and used. Fix the justification (the honest statement is that it's less standard, not undefined).
- InfoNCE temperature attributions ('tau = 0.07 (SimCLR), tau = 0.1 (CLIP)'): SimCLR's main results use tau = 0.1; 0.07 is MoCo's; CLIP's temperature is learned with init 0.07. As written both attributions are off.
- Cross-entropy gradient bullet 'All gradients are bounded in [-1,1], so CE never produces exploding gradients from the loss alone' — true only for the gradient w.r.t. logits of one token; worth a caveat that per-parameter gradients still explode via activations, or an interviewer will pounce on the overclaim.
- Multi-task question: Kendall et al. uncertainty weighting applies 1/(2 sigma^2) to the CE term; the paper's classification derivation gives 1/sigma^2. Minor, but it is presented as 'the' formula.

STALENESS:
- No LLM-era losses anywhere: no next-token CE mechanics, no DPO/Bradley-Terry/GRPO, no z-loss/soft-capping, no MoE auxiliary losses — the chapter's taxonomy diagram (Fig. 1) has no 'generation/preference' branch at all despite the tldr claiming coverage of 'generation'.
- Contrastive section stops at 2021 (MoCo, BYOL, Barlow Twins/VICReg mentions); SigLIP (2023) and SigLIP 2 (2025) — now the default image-text contrastive recipe — are absent.
- KD section is Hinton-2015 only; no LLM distillation (reverse KL, on-policy/GKD, synthetic-data distillation) which is how distillation is actually done in 2026.
- CTC section is legacy weight for a frontier-lab audience (modern ASR is attention encoder-decoder / RNN-T; Whisper is CE-trained); keep but shrink, or add one line positioning it as legacy.
- Metric-learning framing is CV-2019 flavored (ArcFace/CosFace/SphereFace get three formulas and a figure); fine as fundamentals but the 2026 interview weight has shifted to text-embedding training (the 100M-doc question covers this — the section intro should say so).

MUST-KNOW:
- Cross-entropy = NLL = KL to one-hot; gradient w.r.t. logits is p - y; LogSumExp/logits-form numerical stability (never softmax-then-log).
- Softmax CE vs sigmoid BCE: mutual exclusivity is the deciding factor; using softmax for multi-label is the classic bug.
- Next-token CE and perplexity mechanics, including loss masking and per-token vs per-sequence normalization (currently missing from the chapter but firmly in its territory).
- Focal loss: the (1-p_t)^gamma mechanism, when it applies (easy-example dominance, not all imbalance), gamma=2/alpha=0.25 defaults.
- Calibration story: why CE drives overconfidence, label smoothing vs post-hoc temperature scaling, when calibration matters (bidding, cascades, thresholds).
- InfoNCE: temperature's role, in-batch negatives, why batch size matters, why gradient accumulation fails, and the modern fixes (MoCo queue, GradCache, SigLIP sigmoid loss).
- Triplet loss mining (easy/semi-hard/hard) and why mining strategy matters more than the formula.
- DPO / Bradley-Terry preference loss shape and gradient interpretation (missing; a 2026 frontier-lab loop will end badly without it).
- Pointwise vs pairwise vs listwise ranking and the calibration-vs-ranking trade-off; position bias / IPW for click data.
- MSE optimizes conditional mean, MAE the median; Huber as the compromise; KD temperature and the T^2 factor.

IMPROVEMENTS:
- Focal loss section: explicitly address the counterintuitive alpha=0.25 (the RARE foreground class gets the LOWER weight because gamma already up-weights hard positives) — interviewers use this exact gotcha.
- Label smoothing 'When to use': it currently lists 'Knowledge distillation (makes teacher outputs more informative)' — reverse this per Müller et al. 2019 (LS-trained teachers produce tighter, LESS informative logit geometry and hurt distillation) or at least flag the controversy.
- 10M-product ArcFace question: the 2.56B-parameter head number is great — add the same concreteness elsewhere, e.g., 100M docs x 768d x fp32 = ~300 GB raw index in the semantic-search question, and the memory of the InfoNCE similarity matrix at batch 8192.
- InfoNCE temperature: add that CLIP's temperature is learned (init 1/0.07, clamped at 100) since the collapse-debugging question already assumes the reader knows this.
- Ranking section: add one paragraph on where these appear in 2026 loops — LTR questions now come almost exclusively from search/ads companies; for frontier labs, reframe listwise ideas toward reranker training and RLHF-style preference aggregation (Plackett-Luce shows up in both).
- Add an 'estimate the cost' interviewq archetype: 'What is the memory and FLOP cost of the CE loss layer for a 128K-vocab model at batch 1M tokens, and how do you reduce it?'
- Multi-task question: note Kendall et al.'s classification term uses 1/sigma^2 (not 1/(2 sigma^2)) — a pedantic interviewer may poke at the formula as written.


# sections/04_activation_normalization.tex (Activation Functions and Normalization)

VERDICT: Clean fundamentals and best-in-class BatchNorm train/eval debugging content, but the chapter spends roughly half its question budget on BatchNorm pathologies that frontier labs no longer live with, includes two near-duplicate question pairs, teaches the debunked internal-covariate-shift story, and leaves the 2026-critical material (pre/post-norm stability math, QK-norm, logit capping, large-scale loss spikes) as one-line follow-ups.

STAFF-LEVEL: The prose sections are L4/L5 — property bullets and formulas with the judgment concentrated in the interview boxes. The BatchNorm debugging boxes reach solid L6 for a generalist loop, but a frontier-lab L6/L7 loop centers on transformer training stability at scale, where this chapter is thinnest: pre/post-norm gets six bullets, QK-norm and soft-capping exist only as follow-up strings, and there is no loss-spike-at-scale scenario. Scale intuition is nearly absent — no activation-memory arithmetic, no op-cost framing (norms are memory-bound, fp32-computed), no logit-magnitude numbers; the only quantitative anchors are FP16 ranges and the 8/3-d factor. Net: a candidate who mastered only this chapter would handle an L5 screen well, survive an L6 generalist loop, and struggle in a frontier-lab L6/L7 conversation about training a large model stably.

MISSING (critical/high):
- [critical] QK-norm and attention-logit growth as a first-class subsection: uncontrolled query-key dot-product growth causes attention-entropy collapse and divergence at scale (ViT-22B anecdote); QK-norm (RMSNorm on Q and K) is now standard (ViT-22B, Chameleon, OLMo 2, Gemma 3). Include Gemma 2's logit soft-capping as the alternative. :: 'How do you keep a 30B+ model from diverging' is a signature frontier-lab L6/L7 question, and QK-norm is the expected first answer; here it exists only inside follow-up strings.
- [critical] A real pre-norm vs post-norm section: the Xiong et al. 2020 gradient analysis (post-norm gradient magnitude grows with depth, hence warmup dependence), pre-norm's residual-stream norm growth and the 'effectively shallower network' caveat, and 2024-26 hybrids — Gemma 2's pre+post double norm, OLMo 2's reordered norm, sandwich/peri-LN. :: Currently six bullet lines and a figure; this is the highest-value normalization topic in a 2026 loop and candidates need the WHY (gradient math) not just 'pre-norm is standard'.
- [high] Replace the internal-covariate-shift framing with the loss-landscape-smoothing explanation (Santurkar et al. 2018: BN works even when ICS is artificially increased; benefit is Lipschitz smoothing enabling larger LRs). :: Quoting ICS uncritically is itself a red flag at L6 — interviewers deliberately ask 'is the ICS story actually true?'; the chapter currently teaches the wrong answer in the body and buries the right one in an L7 bullet.
- [high] A large-scale loss-spike debugging interviewq: 70B model, bf16, spikes at 100B tokens — walk through activation/norm causes and mitigations (z-loss, QK-norm/clipping, embedding-norm growth, epsilon, data ordering, skip-batch strategies). :: This is THE 'have you actually trained something big' question at frontier labs; the existing NaN question is small-scale single-GPU flavored.

CORRECTNESS:
- 'Why Normalization?' section: internal covariate shift asserted as fact and as the mechanism — contradicted by Santurkar et al. 2018; an expert interviewer treats this as a canonical misconception (section sec:normalization, first paragraph).
- Pre-Norm advantages list: 'No learning rate warmup needed' — overstated; pre-norm reduces but does not eliminate warmup dependence, and all modern LLM recipes still warm up (subsection 'Pre-Norm vs. Post-Norm').
- RMSNorm mathresult omits epsilon entirely, inconsistent with the chapter's own NaN-debugging guidance about epsilon (subsection 'RMSNorm').
- Normalization Selection Guide row 'Modern LLM: RMSNorm — Faster, similar quality' plus Q4's 'every percentage of compute matters' framing implies material end-to-end savings; norm layers are a ~1% end-to-end cost on modern fused stacks. Q4's own red flag admits this, so the body contradicts the question box.
- Q7 'Problem 4: Gradient coupling ... conflicts with the per-token processing paradigm' — gradient coupling through batch statistics exists identically in CNNs, so it cannot be a Transformer-specific incompatibility as presented; an interviewer would poke at this.
- Q6 L7 bullet: 'dead neuron fraction and model quality follows a phase transition-like curve' — unsupported hand-waving presented as an expected staff-level claim; a candidate repeating it would get challenged for a citation.

STALENESS:
- 'Internal covariate shift' presented as the motivation for normalization ('Why Normalization?' section) — debunked as the primary mechanism since Santurkar et al. 2018; teaching it uncritically is a liability in a 2026 interview.
- Question weight is BatchNorm-heavy (4 of 8 questions) for an audience interviewing at labs where BatchNorm appears only as 'why not in transformers'; the marginal BatchNorm question crowds out pre/post-norm, QK-norm, and loss-spike material.
- Activation table recommends GELU for GPT-style decoders (GPT-2-era default); no GEGLU (Gemma), no squared ReLU, no discussion of what 2024-26 models actually chose.
- No 2024-26 normalization developments in the body: Gemma 2 double-norm and soft-capping, OLMo 2 norm reordering + QK-norm, peri-LN, DyT — only nGPT and QK-norm as name-drops inside grading bullets/follow-ups.
- Weight Normalization subsection ('used in some generative models') is legacy filler; either cut or replace with SpectralNorm which the selection table actually references.

MUST-KNOW:
- Why nonlinearity: stacked linear layers collapse to one linear map.
- Dying ReLU: zero gradient for x<=0 means no recovery; causes (large updates, bad init, high LR), diagnosis (fraction of always-zero activations), fixes (Leaky/GELU/SiLU).
- GELU vs ReLU vs SwiGLU: smooth gating, the multiplicative-gate expressivity argument, and the 3-matrix / 8/3-d parameter-budget trade-off.
- BatchNorm vs LayerNorm: which axis each normalizes, and the full derivation of why transformers use per-token normalization (padding, autoregressive inference, batch coupling).
- BatchNorm train/eval discrepancy: model.eval(), running statistics, SyncBatchNorm, and the batch-size-sensitivity diagnostic.
- RMSNorm: drops mean-centering and beta; why scale normalization alone suffices (Zhang & Sennrich); default for modern LLMs.
- Pre-norm vs post-norm: gradient path through the residual stream, warmup sensitivity, why post-norm diverges at depth — with the Xiong et al. gradient argument, not just the diagram.
- The NaN chain: overflow -> Inf -> NaN; attention 1/sqrt(d_k) scaling, normalization epsilon vs FP16, loss scaling, BF16 vs FP16 trade-off.
- QK-norm / attention-logit growth and why large-model training diverges without logit control (must be added to the body; fumbling this ends L6+ frontier-lab interviews).
- The honest answer on why normalization helps: loss-landscape smoothing / larger stable LRs, and that the ICS story is a known misconception.

IMPROVEMENTS:
- Merge Q1 ('Why don't Transformers use BatchNorm? Derive...') and Q7 ('Derive why BatchNorm doesn't work well for Transformers') — they are the same question with ~70% overlapping answers. Likewise Q5 (train-vs-inference behavior) and Q8 (train-val gap, suspect normalization) overlap heavily. Merging both pairs frees room for a pre/post-norm question and a loss-spike question, and fixes the imbalance where 4 of 8 questions are BatchNorm-centric.
- Fix the Activation Selection Guide table: 'GPT-style decoder: Recommended GELU, Alternative SwiGLU' is backwards for 2026 — SwiGLU (or GEGLU) is the default for any new decoder; GELU is the legacy choice. Also either add a SpectralNorm subsection or drop the 'GAN discriminator: SpectralNorm' row — the table recommends a technique the chapter never explains.
- Add ε to the RMSNorm mathresult formula (RMS(x) = sqrt(mean(x^2) + eps)); the NaN question later hinges on epsilon, and the formula as printed would divide by zero on a zero vector.
- Soften 'No learning rate warmup needed' under pre-norm advantages to 'greatly reduces warmup sensitivity' — every modern LLM run still uses warmup, and an interviewer will push back on the absolute claim.
- Quantify the RMSNorm speed claim honestly in the table ('10-15% faster' -> 'op-level speedup; end-to-end gain ~1%, the real appeal is simplicity/fewer params') — the Q4 red flag already says this, so the table contradicts the question.
- In Q1, tighten the padding argument: standard BN averages over B and T, so 'positions near the end see fewer real tokens' only applies to a per-position BN variant; as written it conflates two different masking schemes.
- Add concrete numbers somewhere: SwiGLU FFN params for Llama-2-7B (d=4096, hidden 11008 = 8/3*4096 rounded), activation memory per token per layer, and an example attention-logit magnitude blowup — the chapter currently contains almost no scale arithmetic.


# /home/user/deep-learning-essentials/sections/05_attention_transformers.tex (Attention Mechanisms and Transformers)

VERDICT: A strong 2023-2024 era chapter: the interviewq boxes with L5/L6/L7 ladders, KV-cache arithmetic, and FLOP derivations are genuinely staff-caliber and mostly numerically correct. But it is frozen at the LLaMA-2 moment — no MLA, no NSA/MoBA, no FlashAttention-3, no Llama-3/Gemma-2/Qwen-era design choices, no coding-implementation archetype — and a 2026 frontier-lab loop will probe exactly those gaps.

STAFF-LEVEL: The interviewq sections are genuinely L6-calibrated: graded answer ladders, correct back-of-envelope math, debugging and design archetypes, and 'have you served this' signals (PagedAttention, prefill/decode disaggregation, MFU). The expository first third (attention types, basic transformer anatomy, PE catalog) reads L4/L5-definitional, which is acceptable as scaffolding but over-weighted — Bahdanau/Luong alone gets more space than all post-2023 innovation combined, which gets zero. The decisive staff-level gap is currency: an L6/L7 candidate prepped solely on this chapter would whiff on MLA, NSA/MoBA, FA3, hybrid linear-attention models, and 1M-context production practice — precisely the topics 2026 frontier-lab interviewers use to distinguish 'studied the 2023 syllabus' from 'operates at the frontier'. Scale intuition is present and mostly correct but anchored to A100-era hardware; refresh all cost/latency anchors to H100/B200. Also missing entirely is the implement-it-live coding dimension, which at these labs is a gate, not a bonus.

MISSING (critical/high):
- [critical] Multi-head Latent Attention (MLA, DeepSeek-V2/V3/R1): low-rank KV compression to a latent c_KV (d_c=512 in V3 vs 2*n_h*d_h=32K for MHA), up-projection absorption at inference, and the decoupled-RoPE key trick (why RoPE can't be applied naively to a compressed latent) :: MLA is THE attention innovation of 2024-25 and the single most likely 'what's newer than GQA' question at a frontier lab in 2026. 'Compare GQA vs MLA' and 'why does MLA need a separate RoPE dimension' are now standard L6 questions. The chapter's KV-cache spectrum (MHA->GQA->MQA) is incomplete without it — MLA beats GQA on both cache size AND quality, breaking the chapter's stated quality/memory trade-off frame.
- [critical] A coding-implementation interview question: implement multi-head attention with causal masking in NumPy/PyTorch, then extend it with a KV cache decode loop :: This is the most common attention screen at OpenAI/Anthropic-tier labs — often the FIRST thing asked, before any conceptual question. The chapter has zero coding archetype. Fumbling the einsum shapes, mask broadcasting, or the softmax axis ends interviews. Should include the reference implementation and the classic follow-ups (make it GQA, add RoPE, cache K/V).
- [high] Trainable/native sparse attention: DeepSeek NSA (compression + selection + sliding window branches, hardware-aligned) and MoBA (Moonshot); DeepSeek-V3.2 sparse attention (DSA) as production deployment :: The sparse-attention section stops at Longformer/BigBird (2020). By 2026 the interview-relevant answer to 'how do production models do sub-quadratic attention' is NSA/MoBA-style trainable sparsity, not fixed patterns. An L7 candidate citing only BigBird signals staleness.
- [high] FlashAttention-3 (Hopper: warp specialization, TMA async, FP8, ~75% H100 utilization) and Flash-Decoding (split-KV parallelism for long-context decode) :: The FA question's L7 bullet claims FA 'doesn't help for decode' — Flash-Decoding is precisely the fix for long-context decode, and not knowing FA3 exists dates the candidate to 2023. One paragraph each closes the gap.
- [high] Post-LLaMA-2 architecture choices: QK-Norm (Qwen3, Gemma-3), logit soft-capping (Gemma-2) and its later removal, interleaved sliding-window/full attention layers (Gemma-2/3's 5:1 ratio, Mistral's rolling-buffer KV cache), RoPE base scaling to 500K-1M (Llama 3), vocab growth to 128K-256K, tied embeddings in small models :: The 'GPT-2 vs LLaMA' evolution question is the chapter's flagship currency question, but it ends in 2023. Its vocab claim (50K->32K, 'smaller is better') actually reversed — Llama 3 went to 128K, Gemma to 256K. An interviewer asking 'what changed after LLaMA' gets no support from this chapter. Add a third column (2024-25) to the comparison table.
- [high] Induction heads and basic attention-circuit interpretability (Olsson et al. 2022): QK vs OV circuits, induction heads as the mechanism of in-context learning :: The 'What Different Heads Learn' section is BERT-era probing lore. At Anthropic specifically, induction heads are canonical interview material and the modern answer to 'what do attention heads do'. A keyinsight box plus one follow-up question suffices.
- [high] Mamba-2 / SSD (state-space duality showing SSMs as a form of structured linear attention) and the 2025 linear-attention revival: GLA, DeltaNet/Gated DeltaNet, MiniMax-01's lightning attention at 456B scale, Qwen3-Next and Kimi Linear hybrids :: The chapter says Flash Attention 'largely made linear attention unnecessary' and calls the SSM ecosystem 'early stage' — both stale by 2026, when production hybrid linear-attention models ship at scale. The Transformer-vs-Mamba question needs its table row and hybrid examples refreshed (Nemotron-H, Granite 4 alongside Jamba).

CORRECTNESS:
- Pre-norm question (line ~1190): claims PaLM uses post-norm with careful initialization — PaLM is pre-norm (with parallel attention+FFN blocks). Better examples of post-norm-adjacent designs: BERT, DeepNet, or Gemma-2's sandwich (pre+post) RMSNorm. An expert interviewer will flag this immediately.
- FLOPs question L7 bullet (line ~1078): 'Chinchilla-optimal 7B... 5.88e21 FLOPs ~ 30K A100-hours' — at the chapter's own 40-60% MFU on a 312-TFLOPS A100 this is 8.7K-13K A100-hours (verified numerically); 30K implies ~17% MFU. Off by ~2.3x in an estimation question, which is exactly where precision is graded.
- Training-memory question (line ~890): activation estimate quoted as '~34 bytes/param x seq x batch' — dimensionally wrong (would give petabytes); the 34 is bytes per token per hidden-dim per layer in the s*b*h*(34+5as/h) formula. The final 16-30 GB range is right but the shown formula is garbled.
- KV-cache 128K question L7 bullet (line ~1124): 'H100 with 188 GB HBM3' — H100 is 80 GB (H100 NVL is 94 GB per GPU, 188 GB only as a dual-card pair); 141 GB is H200.
- Positional-encoding table (line ~531) rates sinusoidal extrapolation 'Good' while the text 60 lines earlier says it 'empirically degrades' — internal contradiction; the text is right.
- Attention-memory table (lines 549-563) understates real pre-FlashAttention memory by the head count h (and batch) — presents one head's matrix as 'the attention matrix'.
- FA question L7 bullet (line ~735): 'FA doesn't help for decode phase' — overstated; Flash-Decoding (split-KV) is the standard fix for long-context decode, and at long context decode is KV-cache-read bound, not only weight-load bound (also asserted too strongly in the attention/FFN ratio question's L7 bullet, line ~1471).
- FA-vs-linear question (line ~1547): '128K^2 ~ 16B operations per layer per head' counts matrix ENTRIES, not FLOPs (score FLOPs are 2*n^2*d_head ~ 4e12 per head); loose units in a quantitative answer.
- Scaling question L7 bullet (line ~1033): 'sqrt(d_k) scaling is equivalent to initializing W^Q, W^K with std ~1/sqrt(d_k)' — only equivalent at initialization; the logit scaling persists through training while init effects wash out. As stated, an interviewer would push back.
- Linear-attention 'In practice' note (line ~629) and Mamba question claim SSMs 'largely displaced' linear attention — by 2026 the opposite trend (hybrid linear-attention production models) makes this misleading as a 'strong answer'.

STALENESS:
- No MLA/DeepSeek anywhere — the KV-efficiency story ends at GQA (2023) despite MLA being standard interview material since mid-2024 and deployed in DeepSeek-V2/V3/R1.
- Sparse attention section cites only fixed-pattern 2020 methods (Longformer, BigBird); trainable sparse attention (DeepSeek NSA, MoBA, DeepSeek-V3.2 DSA, 2025) is absent.
- 'Flash Attention has largely made linear attention unnecessary' (line ~629) and the FA-vs-linear question's framing are stale: 2024-25 saw a linear-attention revival (GLA, DeltaNet, MiniMax-01 at 456B, Qwen3-Next/Kimi hybrid linear models).
- Mamba table row 'Ecosystem/tooling: Early stage' and hybrid examples limited to Jamba/Zamba — by 2026 production hybrids (Nemotron-H, Granite 4, Qwen3-Next) exist and Mamba-2/SSD (2024) is unmentioned.
- Flash Attention coverage stops at FA2; FA3 (2024, Hopper) and Flash-Decoding are missing.
- Architecture evolution stops at LLaMA-2: no QK-Norm, soft-capping, interleaved SWA/full layers, 500K-1M RoPE base, 128K-256K vocabularies; the vocab-shrank claim (50K->32K) inverted after 2023.
- All GPU numbers are A100-era; no H100/H200/B200 anchors for the estimation questions.
- ~90 lines on Bahdanau/Luong seq2seq attention (2014-15) is heavily over-weighted relative to its 2026 interview value — this material is essentially never probed beyond one lineage sentence.
- 128K treated as the extreme context frontier; 1M+ contexts shipped in production during 2024-25.

MUST-KNOW:
- Scaled dot-product attention formula and the sqrt(d_k) argument from first principles: Var(q.k)=d_k, softmax saturation, vanishing gradients — plus the modern follow-up (QK-Norm when scaling isn't enough).
- Complexity split: attention O(n^2 d) vs projections/FFN O(n d^2); derive when each dominates and what that implies for where to spend optimization effort.
- KV cache: the 2*L*n_kv*d_head*n*bytes formula cold, with real numbers (LLaMA-70B ~1.3 GB at 4K GQA, ~42 GB at 128K), and why long-context serving is memory-dominated.
- The full KV-efficiency spectrum MHA -> GQA -> MQA -> MLA with quality/memory trade-offs and why MLA breaks the old trade-off (must be added to the chapter).
- FlashAttention: exact (not approximate), IO-aware tiling + online softmax, O(n^2)->O(n) memory, memory-bound to compute-bound; when it does NOT help (short sequences, naive decode) and what fixes decode (Flash-Decoding).
- RoPE mechanics (rotation, R_{n-m} relative property, frequency spectrum) and the context-extension toolbox: Position Interpolation vs NTK-aware scaling vs YaRN vs continued pretraining, plus why naive extrapolation fails.
- Parameter and FLOP rules of thumb: params ~ 12Ld^2 + Vd, forward ~ 2P FLOPs/token, training ~ 6P, and converting to GPU-hours with realistic MFU on current hardware.
- Pre-norm vs post-norm gradient-flow argument and why pre-norm (RMSNorm) is the modern default.
- Training-memory decomposition for the '7B on one GPU?' question: 2+2+12 bytes/param (BF16 weights, grads, FP32 Adam states) plus activations, and the ZeRO/checkpointing levers.
- The GPT-2 -> LLaMA -> 2025-era architecture evolution with the WHY per change (stability / inference efficiency / quality-per-FLOP), extended through GQA, QK-Norm, interleaved SWA, and large vocabularies.
- Be able to WRITE multi-head causal attention with a KV cache in PyTorch/NumPy without hesitation — the most common screen and not currently in the chapter.

IMPROVEMENTS:
- Compress the Bahdanau/Luong section (lines 43-93) from ~2 pages to a half-page historical note. Three Luong score variants, their parameter counts, and global-vs-local details have near-zero 2026 interview value; keep only the additive-vs-multiplicative expressivity/speed argument and the lineage sentence. Reallocate the space to MLA.
- Reconcile the two crossover claims: line ~1455 says attention=FFN FLOPs at n=2d, while line ~1077 says attention FLOPs dominate at n>6d. Both are correct under different definitions (attention-block-total vs score-FLOPs-vs-all-matmul) but the chapter never says so — an interviewer will catch a candidate quoting both.
- The attention-memory table (lines 549-563) silently reports a SINGLE head's n x n matrix; real pre-FA training materializes n^2 x h per layer (32K would be ~64 GB/layer at h=32, not 2 GB). Add a 'per head, per layer, batch 1' caption and one sentence on the multiplier.
- Fix the garbled activation-memory formula in the 7B training-memory question (line ~890): '~34 bytes/param x seq x batch' has wrong units. Replace with the Korthikanti et al. per-layer formula: s*b*h*(34 + 5*a*s/h) bytes, and show it evaluates to ~30 GB for 7B/2K/b=1 without checkpointing, ~10 GB with Flash Attention.
- Update all hardware anchors: the chapter is A100-only (312 TFLOPS, 40/80 GB). Add H100 (~990 TFLOPS BF16, 80 GB HBM3), H200 (141 GB), B200 — cost-estimation answers quoted in A100-hours read as 2022 in a 2026 loop.
- Fix the softmax-saturation figure (lines 109-132): the legend entry 'Before saturation' is meaningless — the blue curve is just the 2-class softmax output. Relabel as 'softmax output' and 'gradient', and consider plotting attention-weight entropy vs logit scale instead, which is what the sqrt(d_k) argument actually concerns.
- Merge the duplicated positional-encoding comparisons: the bullet list at lines 435-444 and the table at lines 522-538 repeat each other and contradict on sinusoidal extrapolation ('empirically degrades' vs 'Good').
- Add one numerics-debugging interview question: 'loss spikes / attention logits overflow in BF16 at scale — diagnose' with QK-Norm, logit soft-capping, and max-logit monitoring as the answer ladder. This archetype (training-instability debugging tied to attention) is asked at every frontier lab and is absent.
- In the MQA section, correct/nuance 'Used in: PaLM, Falcon' — Falcon-40B/180B use grouped KV (effectively GQA); only Falcon-7B is true MQA.
- In the 1M-token design question, add that 1M-context production models exist (Gemini 1.5+) and the real recipe is context parallelism + interleaved sparse/full layers + RoPE base scaling — the current answer reads as if 1M context were open research.


# sections/06_embeddings_representations.tex — Embeddings and Representation Learning

VERDICT: A solid pedagogical skeleton with genuinely strong interview-question boxes, but the body is frozen around 2019 (Word2Vec/GloVe/ELMo/BERT get ~60% of the page count) and almost entirely misses the 2024–2026 embedding landscape (decoder-based embedders, Matryoshka as a first-class topic, MTEB currency, anisotropy). Several factual errors (GPT-2/BERT vocab sizes stated backwards, a misleading 'collapse is a global minimum of contrastive loss' claim) need fixing before a candidate memorizes them.

STAFF-LEVEL: Split personality: the body sections read L4/L5 (definitions and formulas of 2013–2019 methods, no scale numbers outside the tokenization note), while the interviewq boxes are genuinely L6 — Q2's catalog design and Q4's estimation are the right register, with real memory math and operational concerns (drift monitoring, tiered storage, A/B for embedding swaps). To be staff-grade throughout, the body needs the numbers currently quarantined in answers (corpus-embedding cost, embedder size/latency tiers, MTEB-vs-domain eval judgment) and a modern-model section; the ELMo/Word2Vec depth should shrink to interview-lineage footnotes. L7 signal is thin everywhere except Q2/Q5's L7 bullets — there is no content on organizational-scale concerns like embedding version governance across consumer teams (chapter 7 covers reindexing; ch6 could own the 'many downstream consumers of one embedding' problem).

MISSING (critical/high):
- [critical] Decoder-LLM-based embedding models (E5-Mistral, GritLM, NV-Embed, gte-Qwen, Qwen3-Embedding, LLM2Vec) and last-token pooling with instruction prompts :: By 2024–2026 the top of MTEB is dominated by 7B decoder embedders, not BERT-scale encoders. The 'Modern Sentence Embedders' table stops at E5/BGE/GTE (2023). A frontier-lab interviewer asking 'how would you build a SOTA embedder today' expects the candidate to know the encoder→decoder shift, why causal attention needs adaptation (LLM2Vec bidirectional conversion), and the cost trade-off of 7B vs 100M embedders at serving time.
- [high] Anisotropy of raw LM embeddings (Ethayarajh 2019; representation degeneration) and post-hoc fixes (whitening, contrastive fine-tuning) :: This is THE canonical explanation for the chapter's own claim that '[CLS] is a poor sentence embedding without fine-tuning' — the claim appears three times but the mechanism is never given. 'Why does mean-pooled BERT underperform GloVe on STS?' is a classic L5/L6 probe; the answer (narrow cone / high average cosine, frequency-correlated norms) also refutes the chapter's 'healthy avg cosine is 0.0–0.3' heuristic for raw LM features.
- [high] Matryoshka Representation Learning as a body section with the actual loss (weighted sum of InfoNCE at nested prefixes) and its adoption in OpenAI text-embedding-3 / Gemini embeddings :: MRL is mentioned four times inside answers but never defined. It is standard 2025-era practice (API embedders expose truncatable dims) and a common follow-up ('What is Matryoshka training?' is literally listed as a follow-up in Q8 with no supporting content).
- [high] MTEB in the body, plus its 2024–2025 problems (benchmark gaming/overfitting, MMTEB, BEIR) :: MTEB appears only inside Q8's answer. Staff-level eval judgment includes knowing that MTEB rank no longer predicts domain performance because training sets contaminate it — an interviewer probing eval practice expects 'MTEB as a shortlist filter, domain eval as the decision' plus awareness of the contamination critique.
- [high] SigLIP / SigLIP 2 (sigmoid loss) alongside CLIP; CLIP's learnable temperature and InfoNCE formulation :: The CLIP subsection is four bullet points with no loss function. SigLIP replaced CLIP as the default image-text embedder in most 2024+ stacks (and as the vision tower in VLMs); the sigmoid-vs-softmax batch-size argument is a real interview question. Currently the chapter can't support any question about how CLIP is actually trained.

CORRECTNESS:
- §Tokenization, 'Vocabulary size trade-off' (line ~415): 'Common sizes: 32K (GPT-2), 50K (BERT)' — both wrong and reversed. GPT-2 is 50,257 (byte-level BPE); BERT is 30,522 (WordPiece) — which the chapter itself states correctly in the BERT section (E_token ∈ R^{30,000×768}). The author appears to have confused LLaMA-2 (32K) with GPT-2.
- Q6 (BPE interaction), strong answer: repeats '128K in LLaMA 3 vs. 32K in GPT-2' — same error, in a spot a candidate would recite verbatim.
- Q5 (embedding collapse): 'Collapse is a degenerate global minimum of many loss functions. Consider a contrastive loss: if all embeddings are identical... the loss is −log(1/N)—a constant.' Misleading: for InfoNCE with negatives, the collapsed loss log N is NOT the global minimum — well-separated embeddings achieve lower loss, which is precisely why contrastive losses resist collapse. Collapse is a true global minimum only for positive-only objectives (BYOL/SimSiam without predictor+stop-grad). An expert interviewer would push back hard on this as stated; rewrite to distinguish the two cases.
- Q4 L7 bullet: 'embedding tables are often replicated across all GPUs' under model parallelism — in Megatron-style tensor parallelism embeddings are vocabulary-sharded (vocab-parallel embedding), not replicated. Replication happens in pure data parallelism, which is true of all weights and not a distinctive point.
- Q4: 'KV cache ~1–4 GB depending on batch size and sequence length' — understated for serving: at production batch sizes KV cache commonly reaches tens of GB and is the binding constraint; fine for batch=1 only. Qualify it.
- Q1/Q7: 'healthy range of average pairwise cosine is 0.0–0.3; >0.7 signals collapse' — stated without the anisotropy caveat; raw pre-trained LM features (no contrastive tuning) routinely average 0.5–0.9 cosine yet still rank usefully. As written, the heuristic misdiagnoses every vanilla-BERT feature space as collapsed.

STALENESS:
- 'Modern Sentence Embedders' table (E5, BGE, GTE, SimCSE, SBERT) is frozen at ~2023 — no decoder-based embedders, no API embedders (OpenAI text-embedding-3, Cohere Embed v3, Voyage), no Qwen3-Embedding/NV-Embed generation.
- Long-context handling cites only Longformer/BigBird; nothing on RoPE-based context scaling or ModernBERT, and no late-chunking for long-document embeddings.
- CLIP subsection predates SigLIP (2023) and SigLIP 2 (2025), which are now the default image-text embedders.
- Q6's L7 bullet cites 'MegaByte, byte-pair models' for tokenizer-free modeling — the 2026 reference is Byte Latent Transformer (Meta, Dec 2024), and 'byte-pair models' conflates BPE (a tokenizer) with tokenizer-free architectures.
- DINOv2 labeled 'SOTA transfer' — DINOv3 shipped in 2025; minor but easy to update.
- ~40% of the body (Word2Vec/GloVe/FastText/ELMo formulas) is pre-transformer material whose 2026 interview value is one or two conceptual questions at most — over-weighted relative to modern content that is absent.

MUST-KNOW:
- Why raw BERT [CLS]/mean-pooled embeddings are poor for similarity (anisotropy) and how contrastive fine-tuning (Sentence-BERT, SimCSE) fixes it — fumbling this ends embedding interviews.
- InfoNCE loss cold: formula, role of temperature τ (low τ up-weights hard negatives; typical 0.01–0.1), and in-batch negatives.
- Embedding memory arithmetic on demand: vocab×d×bytes for tables, N×d×bytes for corpora, FP32/FP16/INT8 = 4/2/1 bytes, and PQ compression ratios.
- Cosine = dot product for L2-normalized vectors (and L2 distance monotonic in cosine) — 'the most commonly tested fact,' as the book itself says.
- Embedding collapse: detect via average pairwise cosine + singular-value spectrum; prevent via negatives, normalization, VICReg/Barlow-Twins-style regularization; complete vs dimensional collapse distinction.
- Hard negatives as the single highest-leverage lever in embedding training, and why random/in-batch-only negatives produce weak or lexical embedders.
- Tokenization–embedding interaction: vocab-size trade-off (with correct numbers), rare-word fragmentation, and why non-English text fragments more.
- Modern embedder selection workflow: MTEB shortlist → domain eval set with Recall@K/NDCG → fine-tune with mined hard negatives if recall falls short; Matryoshka truncation for the dimension/cost knob.
- Static vs contextual trade-off with numbers (μs lookup vs ms forward pass; 20 MB FastText vs 250 MB DistilBERT).

IMPROVEMENTS:
- Rebalance: compress Word2Vec (skip-gram/CBOW/negative sampling), GloVe, and FastText into ~2 pages framed as 'lineage: negative sampling → InfoNCE' — the only part interviewers still probe — and cut the ELMo section (currently ~1 page with formula) to one paragraph of historical context. Reinvest the pages in a 'Modern Embedding Models (2024–2026)' section.
- Fix the vocabulary-size errors in the tokenization section (line ~415) and Q6, then add a corrected reference table: GPT-2 50,257 / BERT 30,522 / LLaMA-2 32K / LLaMA-3 128K / GPT-4o o200k ~200K, with the 'larger vocab ⇒ shorter sequences ⇒ cheaper inference' scaling argument (Tao et al. 2024 vocabulary scaling).
- Add an InfoNCE + temperature treatment to the body (it currently exists only inline inside Q1's answer): the formula, gradient intuition for why low τ up-weights hard negatives, and typical values (0.01–0.1) — three separate answers reference τ without this foundation.
- Add the anisotropy story before the pooling section so 'why [CLS] fails' has a mechanism; amend Q1/Q7's 'healthy cosine 0.0–0.3' heuristic with the caveat that raw (non-contrastively-tuned) LM features routinely sit at 0.5+ without being useless.
- In Q4, replace 'LLMs often tie input and output embedding weights' with the accurate nuance: tying is common in small models where the table is a large parameter fraction (GPT-2, Gemma), while most large models (LLaMA family, GPT-3+) untie; this is itself a good interview fact.
- Add the caveat to the king−man+woman example that the canonical result partly depends on excluding input words from candidates (Nissim et al. 2020) — an expert interviewer will push back on presenting it uncritically.
- Add one question in a missing archetype: 'Your embedding model wins offline (Recall@100 +4pts) but loses the online A/B — walk me through your investigation' (offline/online mismatch, eval-set staleness, ANN-recall interaction, position bias in labels).


# sections/07_similarity_metric_learning.tex — Similarity and Metric Learning Architectures

VERDICT: The stronger of the two chapters: correct architecture taxonomy (Siamese → triplet → two-tower → cross-encoder), excellent question archetypes with latency/cost estimation, and real operational content (Q11 on embedding-version migration is superb). But multiple GPU latency numbers are wrong by 3–10x (physically impossible in one case), metric learning stops before ArcFace, and the 2026 mainstays — late interaction/ColBERT, hybrid dense+sparse, LLM rerankers, logQ-corrected sampled softmax — exist only as name-drops in L7 bullets rather than teachable content.

STAFF-LEVEL: Mostly correctly calibrated for L5/L6: the architecture sections teach judgment (when to share weights, when each mining strategy fails, decision tree by corpus size/latency), and the estimation questions (Q4, Q10, Q12) plus Q11's migration design are authentic L6/L7 material — Q11 in particular is a question that separates people who have operated embedding systems from those who have read about them. Two gaps keep it below staff-grade: (1) the quantitative backbone is unreliable — three latency figures are wrong by 3–10x, and at L6+ the interviewer will check the arithmetic, so a candidate trained on these numbers gets burned precisely where the chapter tries to add staff signal; (2) the 2026 judgment layer is missing — hybrid-vs-pure-dense, ColBERT storage economics, LLM-reranker distillation, filtered-ANN pitfalls, and logQ correction are the topics that distinguish 'read the textbook' from 'shipped retrieval recently,' and all are absent or one-line. The Siamese/triplet sections could each lose a third of their length without losing interview value; the recovered space should fund the late-interaction and hybrid-retrieval sections.

MISSING (critical/high):
- [critical] Late interaction / ColBERT as a body section: MaxSim scoring formula, token-level index storage blowup (~an order of magnitude over single-vector), ColBERTv2 residual compression, PLAID serving :: ColBERT is the standard 2024–2026 answer to 'what sits between bi- and cross-encoders' and appears in this chapter only as follow-up name-drops ('What is ColBERT?' is asked twice as a follow-up with zero supporting content). Meanwhile Poly-encoder — rarely discussed in 2026 — gets a full section with figure and math. The relative weighting is inverted; an interviewer would expect MaxSim = Σ_i max_j q_i·d_j and the storage trade-off from any staff retrieval candidate.
- [high] Hybrid dense+sparse retrieval: BM25 + dense with Reciprocal Rank Fusion, and learned sparse (SPLADE)  :: Hybrid search is default production practice in 2026 (every vector DB ships it; it is the standard mitigation for exact-match/rare-entity failures of dense retrieval that Q1 diagnoses). The chapter fixes lexical-matching failures only by making the dense model better — the deploy-day answer 'run BM25 in parallel and fuse with RRF' is absent. SPLADE appears once in an L7 bullet.
- [high] Margin-softmax metric learning losses: ArcFace/CosFace (additive angular margin), and proxy-based losses (Proxy-NCA/Proxy-Anchor) vs pair-based :: The chapter's own flagship example — face verification — has been trained with ArcFace-style classification losses, not Siamese/triplet pairs, since ~2019. 'Why did face recognition move from triplet loss to ArcFace?' (no mining needed, O(N·C) proxies vs O(N³) triplets, explicit angular margin) is a standard metric-learning probe that the chapter cannot currently answer.
- [high] Sampled softmax with logQ correction for two-tower recommenders (Yi et al. 2019, YouTube) :: The negative-sampling table says 'popularity-based... corrects for exposure bias' but never gives the mechanism: subtract log Q(item) from the logit to unbias in-batch negatives toward the popularity distribution. This is THE canonical two-tower recsys interview item at Google/Meta-tier loops and directly deepens Q5's popularity-bias diagnosis and Q9's frequency-bias question.
- [high] LLM-based rerankers: cross-encoder T5-family (monoT5/RankT5), listwise LLM reranking (RankGPT/RankZephyr), API rerankers (Cohere Rerank), and distilling LLM judgments into small cross-encoders :: The reranking story is BERT-base/DistilBERT-only. In 2026 the quality frontier for reranking is LLM-based, and the standard production pattern is 'LLM-judge labels → distill into a serveable cross-encoder.' A frontier-lab interviewer asking 'how do you get reranker training data' expects this answer.

CORRECTNESS:
- §Cross-Encoder Computational Cost: 'For 1 million items... Two-tower: ~1 second (with ANN index); Cross-encoder: ~hours.' Both numbers are off: ANN retrieval over 1M vectors is single-digit milliseconds (the chapter's own Q10 says ~1ms for 100M), and 1M cross-encoder pairs at the chapter's own Q12 throughput (~8K pairs/s batched on an A100) is ~2 minutes, not hours (hours is a CPU number). Internally inconsistent with both Q10 and Q12.
- Q12 (QPS with reranker): 'batch=100, sequence length=256... ~10–15ms on an A100' for BERT-base is physically impossible: 100 seqs × ~56 GFLOPs/seq ≈ 5.6 TFLOPs; 5.6 TFLOPs in 12ms implies ~470 TFLOPS sustained — above the A100's 312 TFLOPS dense FP16 peak. Realistic: 35–60ms. The downstream QPS (83/s per GPU) and cost figures inherit a ~3x optimism.
- Q2 (multi-stage pipeline): reranking 1000 candidates with 12-layer BERT at seq len 256 'takes ~30–40ms on an A100' — off by ~10x. 1000 × 56 GFLOPs ≈ 56 TFLOPs ⇒ ≥180ms even at 100% FP16 utilization, realistically 300–500ms. Meeting a 40ms budget requires a distilled 2–6-layer model, seq len ~128, and/or multiple GPUs — say so explicitly.
- Q10 (HNSW QPS): the memory-latency derivation is incoherent — states '~10 ns per cache miss' then multiplies 256 × 100ns; treats each 3KB vector fetch as one cache miss when it is ~48 cache lines; arrives at 25μs, then asserts '0.5–2 ms per query' with no bridge. The conclusion (~1ms) is empirically right but the arithmetic a candidate would repeat does not produce it — fix by costing full 3KB vector fetches (256 candidates × 48 lines × ~100ns ≈ 1.2ms).
- Q4 (1B index estimate): HNSW neighbor IDs assumed 8 bytes; hnswlib/Faiss use 4-byte internal ids, so the 192 GB graph-overhead figure is ~2x high (order of magnitude still fine).
- Fig. efficiency-quality trade-off (§7.1.3): Siamese is plotted at higher quality than Two-Tower with no justification (both are bi-encoders; quality difference is not architectural), and the complexity labels O(n+m)/O(n+mk)/O(nm) use undefined n, m, k. Minor but confusing on the chapter's opening figure.

STALENESS:
- Reranking is BERT-base/RoBERTa/DistilBERT-only; no monoT5/RankT5, no listwise LLM reranking (RankGPT/RankZephyr), no API rerankers — the 2024–2026 reranking stack is absent.
- Poly-Encoder (2019) gets a full section while ColBERT/late interaction (the concept that actually survived to 2026) gets only follow-up mentions — weighting inverted relative to current interview value.
- ANN section reflects the ~2021 Faiss-era menu: no DiskANN in body, no GPU-native indexes (CAGRA, 2023), no binary-quantization-plus-rescore trend (2024–2025), no serverless/pgvector landscape judgment.
- ScaNN described as 'state-of-art' without mention of its SOAR upgrade (2023) or successors; acceptable but dated phrasing.
- No acknowledgment that embedding retrieval in 2026 is usually embedded in a RAG serving stack where the reranker may be the generator's own scoring or an LLM judge (a sentence pointing to the RAG chapter would suffice).
- Siamese/signature-verification framing is fine as pedagogy but reads 2015-era; needs an explicit 'in 2026 you would train this with ArcFace-style losses / a pretrained ViT' bridge.

MUST-KNOW:
- Bi-encoder vs cross-encoder trade-off cold, and the two-stage funnel (two-tower+ANN: 1B→1K at ~10ms; cross-encoder rerank: 1K→100 at tens of ms) with a latency budget per stage — the single most-asked retrieval design pattern.
- Two-tower decomposability: item embeddings precomputed offline + ANN index at query time — the one property that makes billion-scale retrieval possible, and its cost (no query-item interaction, single-vector bottleneck).
- Triplet loss with the easy/semi-hard/hard taxonomy, why easy triplets give zero gradient, and why batch-hard mining can collapse training.
- InfoNCE with in-batch negatives, temperature, and the false-negative poisoning failure of aggressive hard-negative mining — plus the mixed-negative standard practice.
- Storage and latency math at scale: 1B × 768-dim FP32 = ~3 TB raw; HNSW ≈ vectors + ~100–200 B/vector graph (sharded); IVF-PQ compresses to ~100 GB at 96 B/vector; HNSW = recall, PQ = memory.
- Recall@K as the stage-1 ceiling (nothing downstream recovers a missed document) vs NDCG/precision as stage-2 metrics — and the 'recall good, precision bad ⇒ ranking problem, add a reranker' diagnosis.
- The lexical-vs-semantic failure mode of two-tower models and the fix hierarchy: BM25 hard negatives → cross-encoder rerank → (hybrid BM25+dense fusion).
- Late interaction (ColBERT MaxSim) as the bi/cross middle ground and its storage cost — asked by name in 2026 loops.
- Embedding-space incompatibility across model versions: never mix old/new embeddings in one index; shadow indexing vs backward-compatible training vs adapters.
- Serving economics: roughly what a GPU reranker costs per QPS and when to degrade gracefully (skip reranking, shrink candidate set).

IMPROVEMENTS:
- Fix the three inconsistent/wrong latency claims (detailed in correctness) and make Q2/Q12 mutually consistent — an interviewer who does the FLOPs math will catch a candidate reciting 'rerank 1000 BERT-base pairs in 30–40ms on one A100.'
- Promote ColBERT from follow-up mentions to a full section between Poly-Encoder and Two-Stage; shrink Poly-Encoder to half its length and frame it as the historical precursor to late interaction.
- Add a 'Beyond pairs and triplets' subsection under metric learning covering ArcFace/CosFace with the modified-softmax formula cos(θ+m), and the pair-based vs proxy-based mining trade-off; update the face-verification application box accordingly.
- Extend the two-tower training section with the logQ-corrected sampled softmax formula (s(q,i) − log Q(i)) and wire it into Q5's popularity-bias fix list and Q9's causes.
- Add hybrid retrieval to the Two-Stage section: BM25 and dense retrieval in parallel → RRF fusion (score = Σ 1/(k+rank)), with the guidance that hybrid is the default 2026 production posture and the cheap fix for Q1's failure mode.
- In the ANN comparison table, add DiskANN (SSD, ~10B-scale, higher latency) and a GPU-index row (CAGRA), and demote LSH with an explicit 'legacy — rarely chosen today' note.
- Add a filtered-search warning box: pre-filter vs post-filter recall collapse at selective filters, and that production engines integrate filters into graph traversal.
- Add one 'defend this trade-off' question: 'Your team wants to replace the cross-encoder reranker with a 7B LLM reranker that improves NDCG@10 by 2 points at 8x cost — argue both sides' — currently no question forces arguing against a quality improvement.


# sections/08_self_supervised_learning.tex (Self-Supervised Learning)

VERDICT: A solid 2020-2022 vision-SSL chapter with genuinely good interview questions (collapse mechanisms, estimation, debugging), but it is frozen in the SimCLR/BYOL/MAE era and has one severe copy-paste defect (the MAE 'architecture' figure is actually a retrieval-architecture decision tree). For 2026 frontier-lab interviews it needs SigLIP, the DINOv2/v3 recipe, JEPA, and the 'data curation beats method' lesson, plus deduplication of two near-identical MLM-vs-CLM questions.

STAFF-LEVEL: Depth is right at L5/borderline-L6 for the vision-SSL core: the collapse and BYOL discussions genuinely reach L6, and the two estimation/debugging questions are the strongest staff content. It falls short of L6/L7 in three ways: (1) almost no scale numbers in the body (no memory footprints for batch-4096 contrastive training, no GPU-hours for a reference MAE/DINOv2 run, no CLIP-scale batch/compute figures) — everything quantitative lives inside two interview answers; (2) the 2023-2026 layer (SigLIP, DINOv2 internals, JEPA, data curation) that frontier labs actually probe is missing, so a candidate studying only this chapter will sound three years behind; (3) systems-flavored 'have you done this' signals (shuffling BN, LARS, all-gather negatives, dedup at 100M-image scale) are absent. The descriptive sections on SimCLR/MoCo/BYOL are L4/L5 recitation that could be compressed to fund the additions.

MISSING (critical/high):
- [critical] SigLIP / sigmoid contrastive loss (Zhai et al. 2023, SigLIP 2 in 2025) :: Every 2026 VLM vision tower is SigLIP-family, not CLIP. The sigmoid loss removes the global softmax over the batch, so it needs no all-gather of the full similarity matrix and works at smaller batch sizes — a canonical 'why did X replace Y' interview question. Currently SigLIP appears only in one follow-up line; it needs its own subsection with the loss formula next to InfoNCE.
- [critical] The DINOv2 recipe and iBOT (masked-image-modeling + self-distillation) :: The chapter recommends DINOv2 four times without ever explaining it. Staff candidates get asked what actually made it work: iBOT patch-level masked distillation + DINO image-level loss, curated LVD-142M data (retrieval-based curation), KoLeo regularizer, high-res adaptation phase, and distillation to smaller ViTs. DINOv3 (Aug 2025, gram-anchoring for dense features) should be name-checked as the 2026 SOTA backbone.
- [high] Autoregressive branch missing from the SSL taxonomy :: Next-token prediction is the single most successful SSL objective in history, yet the taxonomy figure (fig:ssl_taxonomy) has only contrastive/non-contrastive/masked branches. GPT appears in the comparison table but not the taxonomy. An interviewer will notice the framing treats LLM pretraining as an afterthought; add a fourth 'Autoregressive/Generative' branch (GPT, iGPT, AIM).
- [high] JEPA family: I-JEPA (2023), V-JEPA / V-JEPA 2 (2024-25) — predict in latent space, not pixel space :: The latent-prediction vs pixel-reconstruction debate (why MAE's pixel targets waste capacity on texture; why data2vec/JEPA predict representations) is a live 2026 discussion topic and the natural L7 follow-up to 'why does MAE have weak linear-probe performance'.
- [high] MoCo shuffling-BN and information leakage / shortcut solutions :: Classic 'have you actually trained one of these' signal: BatchNorm lets the network cheat by using batch statistics to identify positives, which is why MoCo shuffles samples across GPUs and SimCLR uses global BN. Also covers the general class of pretext-task shortcuts (JPEG artifacts, chromatic aberration). Great debugging-question material that is entirely absent.
- [high] Data curation for SSL at scale: dedup (SemDeDup), quality filtering, DataComp / MetaCLIP :: The DINOv2/DataComp-era lesson is that curated data beats method innovation at scale — the exact judgment-over-technique insight L6/L7 interviews probe. The debugging question mentions dedup in one line; it deserves a keyinsight box.

CORRECTNESS:
- CRITICAL — Sec. 'MAE (Masked Autoencoder)', subsubsection 'Architecture': the figure is a similarity-architecture decision tree (Cross-Encoder / Poly-Encoder / Two-Tower / rerank, labeled fig:arch_decision, caption 'Decision tree for similarity architecture selection') — wrong chapter's figure entirely; MAE has no architecture diagram at all.
- InfoNCE interviewq (Sec. Interview Questions): 'If all representations collapse to a constant... the loss becomes log(K+1) — the maximum possible value.' False: log(K+1) is the chance-level/uniform value, not the maximum; InfoNCE is unbounded above (negatives scoring higher than the positive push loss beyond log(K+1)). Say 'chance-level, far from the minimum' instead.
- InfoNCE MI bound stated as I(X;Y) >= log(K) - L with K = number of negatives (two locations); CPC's bound is log(N) with N total candidates, i.e., log(K+1) under this chapter's convention — off-by-one.
- BERT compute estimation answer: internally inconsistent '~40 epochs' then '~33 epochs' in the same paragraph; also 'original training used FP32' on TPUs is dubious (TPU training used bfloat16-era hardware) — the sanity-check discrepancy is better attributed to lower MFU and older chips.
- Repeated '6.7x more training signal => more data-efficient' claim (two interviewqs) is stated as settled fact; it is contested and interviewers will push (see improvements).
- masked-methods comparison table: data2vec 'Domain: Multi-modal' — misleading; it is modality-agnostic but single-modality-trained.
- BYOL answer asserts BatchNorm as a collapse-prevention mechanism without the known counter-result (BYOL works with GN+WS, no batch statistics).

STALENESS:
- No SigLIP/SigLIP 2 (2023/2025) anywhere except one follow-up line — CLIP softmax loss is presented as the current state of image-text contrastive learning.
- DINOv2 treated as a black-box recommendation; DINOv3 (Aug 2025) and V-JEPA 2 (2025) absent — the 2026 vision-backbone conversation is not represented.
- 'Multimodal -> data2vec, ImageBind' recommendation is 2022-era; also data2vec labeled 'Multi-modal' in the masked-methods table is misleading (it is a unified per-modality framework, trained separately on each modality).
- MoCo v3 (the ViT-era contrastive baseline, and the version anyone would use today) appears only as a passing mention in one strong answer; the MoCo section describes v1's queue, which MoCo v3 dropped.
- No mention of autoregressive image pretraining (iGPT, AIM 2024) in the 'why did AR win' discussion — the question is asked about text only, but the vision-side analogue is exactly what a vision-lab interviewer adds.
- Text-side recommendations ('Text understanding -> BERT, RoBERTa') predate ModernBERT (Dec 2024) and the LLM-embedding era.

MUST-KNOW:
- Representation collapse: the three types, and the collapse-prevention mechanism of each family (negatives/InfoNCE; predictor + stop-gradient/EMA; centering + sharpening; explicit variance-covariance terms).
- InfoNCE: write the loss, explain temperature as hard-negative weighting, state the MI lower-bound interpretation and its log-K saturation limitation.
- Why the projection head exists and why you use h (pre-projection) downstream.
- Augmentations define invariances — and therefore encode downstream-task assumptions (the color-jitter-vs-flower-species example).
- Stop-gradient/BYOL mechanics: what happens if you remove the predictor (instant collapse) and the alternating-optimization interpretation.
- MAE design: 75% masking, asymmetric encoder (visible patches only) — and why the linear-probe vs fine-tune gap exists for MAE.
- CLIP symmetric InfoNCE + zero-shot classification via prompt embeddings; and (to add) why SigLIP's sigmoid loss replaced it.
- MLM vs CLM trade-offs: signal density, [MASK] train-test mismatch, interface universality, and why decoder-only won at scale.
- The 6P FLOPs/token rule + MFU for back-of-envelope pretraining cost.
- Evaluation protocol: linear probe vs k-NN vs full fine-tune, and what each confounds.

IMPROVEMENTS:
- URGENT: Replace the figure under 'MAE (Masked Autoencoder) > Architecture' (currently fig:arch_decision — a cross-encoder/poly-encoder/two-tower decision tree pasted from a retrieval chapter) with an actual MAE diagram: visible patches -> large encoder -> insert mask tokens -> small decoder -> pixel reconstruction.
- Merge the two near-duplicate interview questions 'BERT vs GPT pre-training objectives' and 'Next-token prediction vs MLM — why did GPT win' — they cover ~80% identical ground (signal density, task alignment, interface universality) in the same chapter. Keep one, and move it (or cross-reference) to Chapter 9 where MLM/CLM is defined.
- Soften the twice-repeated claim that CLM's 100% token signal makes it '~6.7x more data-efficient': loss coverage is not sample efficiency. At equal scale MLM-family models were MORE sample-efficient for NLU (RoBERTa vs GPT-2), masking rate can profitably exceed 15% (Wettig et al., 'Should You Mask 15%?'), and UL2 shows mixture objectives compete. An expert interviewer will push back on the naive 6.7x framing; the strong answer should present it as 'more loss terms per FLOP' plus the interface/scaling arguments.
- In the BYOL collapse answer, update the BatchNorm mechanism claim with the 2020 follow-up result ('BYOL works even without batch statistics' — GroupNorm + weight standardization suffices); currently the strong answer states BN as a load-bearing mechanism, which the follow-up question then undermines.
- Rewrite Hypothesis 3 of the plateau-debugging answer ('10x fewer epochs but same wall-clock compute... you effectively trained 10x longer') — the logic is garbled as written. State it cleanly: at fixed step budget, 10x data means 10x fewer epochs; the model may now be under-trained per-example, so compare at matched compute and check whether loss curves have converged.
- Add a worked memory/compute comparison box: e.g., SimCLR batch 4096 ViT-B activations + similarity matrix vs MAE's 25%-token encoder cost — give actual GPU counts/hours for one reference run of each family. The chapter asserts 'MAE is 3-4x faster' without a single memory number anywhere.
- Update the method-selection table: replace 'Multimodal -> data2vec, ImageBind' with SigLIP(-2)/Perception-Encoder-era guidance, and add a row for 'VLM vision tower -> SigLIP / DINOv2' since that is the dominant 2026 use of vision SSL.
- Add a shortcut-learning debugging interviewq ('your contrastive loss drops to near zero in 2 epochs — what happened?'): covers BN leakage, augmentation leakage (borders, aspect ratio), duplicate images as false positives.
- Fix the InfoNCE off-by-one: with K negatives the CPC bound is I >= log(K+1) - L (K+1 candidates total); the chapter writes log(K) in both Sec. 'Understanding Why SSL Works' and the InfoNCE interviewq. Pick a convention and state it.


# sections/09_nlp_architectures.tex (NLP Architectures)

VERDICT: Well-organized and its estimation/DPO/RoPE questions are excellent, but the chapter's center of gravity is 2019-2023: BERT-variant minutiae get more space than the modern decoder stack, the model tables stop at GPT-4/LLaMA-2/Mixtral, and the alignment section ends at DPO with no GRPO, RLAIF/Constitutional AI, or reasoning-model training. For a 2026 frontier-lab loop this reads one generation stale in exactly the places interviewers probe hardest (attention variants + KV-cache math, MoE, test-time compute).

STAFF-LEVEL: The interview questions are mostly true L6 material — latency estimation, DPO derivation, decoder-only-dominance, and the two production design questions all test judgment and trade-offs, and the body text supplies real numbers in those spots. But the expository body is L4/L5 in its weighting: BERT-variant archaeology (RoBERTa hyperparameters, ALBERT factorization, DeBERTa equations) consumes prime pages while the things a 2026 L6/L7 loop actually drills — KV-cache arithmetic, GQA/MLA, MoE serving math, long-context extension, GRPO/reasoning post-training, parameter counting — are one-liners, follow-up bullets, or absent. Scale intuition is concentrated in exactly two interviewqs; the body's tables give parameter counts but never teach the reader to derive or use them (no memory footprints, no $/token, no KV-cache GB). Net: a candidate who studies this chapter will interview well on 2023 and get exposed on 2025-26 material; fixing the three critical gaps (MLA/KV-cache, MoE, GRPO/reasoning) plus the model-table refresh closes most of the distance.

MISSING (critical/high):
- [critical] Attention-variant progression MHA -> MQA -> GQA -> MLA (DeepSeek-V2/V3 multi-head latent attention) with the KV-cache size formula and a worked example :: This is arguably THE 2026 staff interview thread: KV cache bytes = 2 x layers x kv_heads x head_dim x bytes/param per token; why GQA cuts it 4-8x, why MLA's low-rank latent compresses it ~10x further and how it interacts with RoPE (decoupled RoPE keys). The chapter gives GQA/MQA one table cell each and never computes a KV cache size in the body (only a follow-up). Absent MLA = candidate looks pre-DeepSeek.
- [critical] Mixture-of-Experts as a first-class section: routing (top-k), load-balancing losses, DeepSeekMoE fine-grained + shared experts, aux-loss-free balancing (DeepSeek-V3), active vs total params, expert parallelism cost :: Frontier models in 2026 are MoE (GPT-4-class, Mixtral, DeepSeek-V3, Llama 4, Qwen3). The chapter's entire MoE coverage is one table row ('Mixtral 8x7B, MoE, 12B active'). 'Why does an 8x7B model have 47B params but 13B active, and what does that do to serving memory?' is a standard L6 question the chapter cannot answer.
- [critical] Reasoning models and RL for reasoning: GRPO (DeepSeek-R1), RLVR/verifiable rewards, process vs outcome reward models, test-time compute scaling, distillation of reasoning traces :: The alignment section stops at DPO (2023). Since o1 (Sept 2024) and R1 (Jan 2025), 'explain GRPO and why it drops the value network/critic relative to PPO' is a top-frequency frontier-lab question. PRMs appear only in one follow-up line. This is the largest currency gap in the chapter.
- [high] Constitutional AI / RLAIF :: The book targets Anthropic-tier labs and the RLHF section never mentions AI feedback, self-critique, or CAI. One subsection: critique-revision SFT stage + RLAIF preference stage, and why AI feedback scales where human preference labels do not.
- [high] Long-context methods section (body, not follow-ups): position interpolation, NTK-aware scaling, YaRN, RoPE base-frequency scaling (Llama 3.1's 500K base), attention sinks/StreamingLLM, needle-in-haystack and RULER evals, 'lost in the middle' :: Every 2026 model ships 128K+ context; interviewers ask how you extend a 4K-trained model to 128K and how you evaluate whether the context actually works. Currently the entire topic lives inside L7 bullets of the RoPE question.
- [high] Chat templates, special tokens, and the base->instruct distinction as a concrete artifact (ChatML/Llama template, why template mismatch silently destroys quality, system prompts) :: The number-one real-world fine-tuning bug in 2024-26 and a standard debugging probe ('your fine-tuned chat model performs worse than base — why?'). The tokenizer warningbox gestures at it; make it explicit content plus a debugging interviewq.
- [high] Speculative decoding (draft models, Medusa/EAGLE-style heads) at least at concept level, plus min-p sampling and constrained/structured decoding (JSON mode, grammar-constrained) :: Generation-strategies section covers 2019 sampling only. Even if serving has its own chapter, 'how do you get free decode speedup without changing the output distribution' (rejection-sampling guarantee) belongs next to greedy/top-p, and structured output is now a default production requirement.
- [high] Parameter-counting anatomy of a transformer: ~12*d^2 per layer breakdown (attention 4d^2, FFN 8d^2 or 3*d*d_ff for SwiGLU), embedding share, where LLaMA-7B's parameters live :: 'Roughly where are the 7B parameters?' is a classic screen for genuine familiarity; the chapter has model-size tables but never teaches the arithmetic that generates them.

CORRECTNESS:
- Sec. Instruction Tuning and RLHF, Stage 3: equation labeled L_PPO is not a PPO loss — it is the KL-regularized reward objective, written as a quantity that should be maximized while being presented as a loss; no clipped surrogate, no value function. An interviewer who asks 'walk me through PPO from your own notes' exposes this immediately.
- Sec. Tokenization, SentencePiece 'Used in: LLaMA, T5, mBERT' — mBERT uses WordPiece, not SentencePiece.
- BPE/tokenization interviewq: 'Byte-level BPE... Used by GPT-2, LLaMA' contradicts the same answer's later 'LLaMA (via BPE mode)' under SentencePiece; LLaMA-1/2 is SentencePiece BPE with byte fallback (byte-level BPE arrives with Llama 3/tiktoken-style vocabularies).
- Sec. ELECTRA: the loss equation's D(x_i) is undefined and the sign convention is inverted relative to the paper (paper: D outputs P(original), loss uses log D for original tokens); as printed it silently redefines D as P(replaced) — define D or flip the indicators.
- RoPE interviewq: 'ALiBi... RoPE has become more popular due to better empirical performance at very long contexts' — misleading; ALiBi extrapolates better untuned, and RoPE's long-context story depends on interpolation/base-scaling tricks. Also 'decaying long-range dependency' is an upper-bound property from the paper, not a robust empirical fact — present it as the paper's motivation.
- ICL interviewq Theory 3 states 'ICL works even when labels are random' as settled evidence; incomplete post-2023 (larger models are sensitive to label correctness and can learn flipped mappings).
- 'Mistral Innovations' presents sliding-window attention as a defining current feature; Mistral removed SWA in later versions — stale-as-stated.
- Sec. 'Why decoder-only dominates' strong answer: 'a decoder-only model with N parameters uses compute more efficiently than an encoder-decoder with N total' is presented as fact; T5/UL2-era controlled comparisons complicate this (encoder-decoder is compute-competitive at matched FLOPs on many tasks) — the robust claims are data format, interface, and infrastructure, which the answer also makes.

STALENESS:
- Alignment section ends at DPO (May 2023): no GRPO, no RLVR, no reasoning-model RL, no online-DPO/iterative variants, no RLAIF/Constitutional AI — the single most important gap for 2026 interviews.
- GPT table: GPT-4 at '8K/32K context, ~1.8T?' — three product generations behind; no reasoning models anywhere in the chapter.
- Open-source table tops out at LLaMA-2/Mixtral/Falcon (2023); no DeepSeek, Qwen, Llama-3/4, Gemma — and consequently no MLA and effectively no MoE.
- Instruction-tuning datasets listed are FLAN/Natural-Instructions/Alpaca/Dolly (early 2023); Tulu-3/UltraFeedback-era open post-training recipes are the current reference points.
- Generation strategies stop at top-p (2019); no min-p, no speculative decoding, no structured/constrained decoding, no reasoning-model sampling guidance.
- 'Vocabulary size: 30K-50K typical' and BERT-era special-token framing ([PAD]/[CLS]/[SEP]) reflect 2019 practice; modern chat special tokens and 100K-256K vocabs absent.
- Fallback-to-'GPT-4 via API' and 'zero-shot GPT-4' references throughout the design answers date the chapter; fine as examples but should say 'frontier LLM API'.
- ICL 'emerges at 10B+' claim is pre-2024; small curated-data models falsify the specific threshold.
- Encoder recommendations omit ModernBERT (Dec 2024), the obvious 2026 replacement for the chapter's repeated 'DeBERTa-base' advice.

MUST-KNOW:
- Prefill vs decode: compute-bound vs memory-bandwidth-bound, and the back-of-envelope (weights-bytes / bandwidth = per-token floor; 2P FLOPs/token forward) — fumbling this ends a 2026 infra-adjacent interview.
- KV cache: the size formula, a worked example, and the MQA/GQA/MLA progression as the fix (currently missing from the book at formula level).
- The modern decoder recipe and why: pre-norm RMSNorm, SwiGLU, RoPE, GQA, no biases — be able to justify each.
- RoPE mechanism (relative position via rotation) plus at least one context-extension method (PI/NTK/YaRN or base rescaling).
- RLHF pipeline (SFT -> RM (Bradley-Terry) -> KL-regularized RL) AND the DPO derivation — plus, in 2026, GRPO and why verifiable rewards changed post-training.
- Tokenization consequences: vocab-vs-sequence-length trade-off, byte fallback, why models fail character-counting/arithmetic, tokenizer/template mismatch as a debugging reflex.
- Sampling: temperature/top-k/top-p mechanics and per-use-case settings; why greedy/beam degenerate for open-ended text.
- Why decoder-only won (signal density, raw-data compatibility, interface universality, infra simplicity) and where encoders/encoder-decoders still win (cheap classification, embeddings, Whisper-style ASR, translation).
- MoE basics: active vs total parameters, top-k routing, load balancing, and the serving-memory implication.
- ICL as task retrieval/mesa-optimization with its practical sensitivities (format, ordering, recency), minus outdated hard thresholds.

IMPROVEMENTS:
- Modernize both model tables: GPT table should either stop pretending to track frontier models or extend through GPT-4o/o1/o3-era with honest 'undisclosed' fields (and drop the '~1.8T?' rumor as a table value); open-source table needs Llama 3.1/4, Qwen2.5/3, DeepSeek-V2/V3 (MLA + MoE), Gemma 2/3 — and a column for attention type (MHA/GQA/MLA) to set up the KV-cache discussion.
- Rebalance the chapter: compress RoBERTa/ALBERT/DeBERTa/ELECTRA detail (currently ~4 subsections with equations) into one comparison table + one ELECTRA insight box, and spend the pages on MoE, MLA/KV-cache math, and reasoning-model training. DeBERTa's disentangled-attention equations have near-zero 2026 interview value relative to their length.
- Fix the RLHF Stage-3 presentation: the equation labeled L_PPO is the KL-regularized reward objective (to be maximized), not the PPO loss; state J(theta) = E[r - beta*KL] as the objective, note it is maximized, and add one sentence that PPO's clipped surrogate + value model is the mechanism — this sets up the GRPO contrast (group-relative advantage, no critic) you should add.
- Update the multilingual document-understanding strong answer (L7 design question): the 2026 answer leads with an OCR-free multimodal LLM (Qwen2.5-VL/Gemini-class) reading page images, with the OCR+LayoutLMv3 pipeline as the cost-optimized/on-prem alternative; as written, a frontier-lab interviewer's first pushback is 'why are you building a 2021 pipeline instead of using a VLM?'
- Soften/annotate the ICL scale claim (stated twice: 'emerges around 10B+', 'nearly absent below ~1B'): 2024-26 small models (1-3B) trained on curated data show reliable ICL — the threshold was an artifact of the GPT-3-era data recipe; also nuance Theory 3's 'random labels don't matter' with Wei et al. 2023 (larger models DO override semantic priors and follow flipped labels), and mention the 'emergent abilities as metric artifact' critique (Schaeffer et al.).
- Tokenization section fixes: mBERT uses WordPiece not SentencePiece (remove from SentencePiece 'Used in' list); distinguish LLaMA-1/2's SentencePiece-BPE-with-byte-fallback from GPT-2/Llama-3-style byte-level BPE (the BPE interviewq currently lists LLaMA under both); update 'vocab 30K-50K typical' to 32K-256K with the scaling rationale.
- Add the missing debugging interviewq archetype: 'model outputs garbage/degraded after fine-tuning' with the real culprit list — chat template mismatch, tokenizer version drift, wrong EOS handling, learning-rate-induced forgetting — converting the existing warningbox into tested material.
- Add a cost-estimation interviewq: 'estimate $/day to serve 1M requests (500 in / 300 out tokens) on a 70B model' — forces tokens/s/GPU, batching, GPU-hour pricing; complements the existing latency question and is a very common L6 ask.
- In the ALiBi comparison (RoPE question), correct the framing: ALiBi actually extrapolates better zero-shot; RoPE won on quality at trained lengths plus the existence of cheap extension methods (PI/YaRN/base-rescaling) and ecosystem lock-in — not raw long-context superiority.
- Note that Mistral dropped sliding-window attention after v0.1 (the 'Mistral Innovations' list presents SWA as current practice); frame SWA as one point in the sparse-attention design space that modern models (Gemma 2's interleaved local-global) still use in hybrid form.


# sections/10_vision_architectures.tex (Vision Architectures)

VERDICT: A competent 2020-2023 vision survey with genuinely good interview-question boxes, but it is missing the topics a 2026 frontier-lab interviewer actually probes — self-supervised pretraining (MAE/DINOv2), CLIP/SigLIP contrastive pretraining, open-vocabulary detection, and the ViT-as-VLM-encoder role — and its flagship FLOPs-estimation answer mixes MACs and FLOPs conventions in a way an expert would immediately push back on.

STAFF-LEVEL: The interviewq boxes carry the chapter and mostly land at L6 — the debugging, edge-design, and visual-search answers have real numbers and real trade-offs, and the goodgreat ladders correctly separate L5 recall from L6/L7 judgment. The body sections, however, read as L4/L5 reference material (definitions and bullet lists), and the depth is pointed at the wrong decade: there is more body text on DenseNet and SSD than on everything self-supervised combined. The biggest Staff-level gap is judgment content about the 2023-2026 consolidation — why plain ViT + SSL + FlashAttention beat hierarchical designs, why detection is going open-vocabulary, and what the vision stack inside a frontier VLM looks like. Scale intuition is present for edge and search but missing for the case interviewers care most about: token counts and attention cost at native/high resolution (e.g., 1024px at P=14 is ~5.3K tokens) and the vision-token budget pressure in VLM serving.

MISSING (critical/high):
- [critical] Self-supervised vision pretraining: MAE (75% masking, asymmetric encoder-decoder, pixel targets), DINO/DINOv2, iBOT, and the contrastive-vs-masked-modeling distinction :: This is the single most probed vision topic at frontier labs in 2026 — every ViT discussion segues into 'how do you pretrain it without labels.' DINOv2 appears only inside strong-answer prose; MAE is never mentioned. 'Why does 75% masking work for images but 15% for text' is a stock question.
- [critical] CLIP / SigLIP as a section: dual-encoder architecture, InfoNCE loss with temperature, batch-size sensitivity, SigLIP's sigmoid loss fix, and why CLIP became the universal vision backbone :: CLIP is load-bearing in two of the chapter's own strong answers (visual search, transfer learning) but its mechanics are never taught. 'Write the CLIP loss' and 'why does SigLIP scale better' are asked constantly; SigLIP-family encoders are what 2026 VLMs actually ship.
- [high] Open-vocabulary detection and segmentation: Grounding DINO, OWL-ViT, GLIP, YOLO-World :: By 2026 'design a detector for classes you didn't train on' is a standard design question and the practical default for many products; the chapter's detection story is entirely closed-vocabulary.
- [critical] Detection metrics and mechanics: IoU, NMS (including pseudocode-level understanding), COCO mAP@0.5:0.95, anchor-free heads (FCOS/CenterNet) :: The chapter says DETR 'eliminates NMS' without ever defining NMS or mAP. 'Explain mAP' and 'implement NMS' are screening questions that end interviews when fumbled; 'anchor-free' appears in an answer but is never explained.
- [high] FPN as an actual subsection (top-down pathway, lateral connections, per-level assignment) :: FPN is referenced four times as the recommended solution (small objects, multi-scale) but its mechanism is never described anywhere in the chapter.
- [high] ViTDet and the 'plain ViT is enough' result; ViT-as-VLM-encoder practice: native/variable resolution (NaViT patch-n-pack, AnyRes tiling), ViT registers (Darcet et al. 2023), 2D-RoPE, vision token budgets :: The chapter's narrative implies hierarchical (Swin) is the modern path for dense tasks, but frontier labs standardized on plain ViT (SAM, DINOv2, ViTDet, every major VLM) because of SSL compatibility and FlashAttention. An L7 candidate should be able to argue this reversal; the interviewers themselves work on VLM encoders daily.
- [high] SAM architecture detail and SAM 2: MAE-pretrained ViT-H image encoder, prompt encoder, lightweight mask decoder, ambiguity-aware multi-mask output, amortized encoder design; SAM 2's streaming memory attention for video :: SAM gets three bullet points. 'Why did SAM put all compute in the encoder and almost none in the decoder' is a great systems-thinking probe, and SAM 2 (2024) is table stakes by 2026.
- [high] MACs vs FLOPs convention sidebar, and compute-bound vs memory-bound reasoning for vision backbones (why ViT-B can be faster than ResNet-50 on GPU despite 4x the 'FLOPs') :: The chapter's own follow-up asks 'why might ViT be faster than ResNet despite more FLOPs' but the body never answers it (arithmetic intensity, large GEMMs vs many small convs, depthwise convs being bandwidth-bound). This is core Staff-level hardware intuition.

CORRECTNESS:
- FLOPs estimation interviewq: internal convention inconsistency. The stated formula 2*k^2*Cin*Cout*H*W (factor-2, true FLOPs) is incompatible with the claimed 'accurate' total of 4.1 GFLOPs for ResNet-50 (that is the MAC count; true FLOPs ~8.2G). ViT arithmetic likewise: '4 x N x D^2 (QKV projection + output)' counts 4 matmuls at 1 FLOP per MAC while the narrative claims FLOPs; 17.6G is the MAC-convention figure. As the chapter's flagship estimation answer, this teaches candidates numbers that won't survive an expert's 'walk me through the factor of 2' follow-up.
- CNN evolution table: 'ResNet ... 78.6%' — unlabeled variant; ResNet-50 (the variant every other part of the chapter uses) is ~76.1% top-1 as published. Table also mixes ImageNet-1k-only and 22k-pretrained results (ConvNeXt 87.8%) without notation.
- CNN-vs-ViT strong answer: 'crossover point where ViT matches CNN is roughly 1M images' stated as fact — unsupported precision; original ViT evidence places it somewhere between 1.3M and 14M and DeiT-style recipes move it below 1.3M.
- Visual search answer: HNSW memory quoted as raw-vector-only (200 GB FP32) — omits graph adjacency overhead, a 5-10% to 20% addition that a search-infra interviewer would expect in the estimate.
- Detection selection table row 'Small objects -> Faster R-CNN + FPN' presented as current best practice — modern DETR variants with denoising (DINO) and high-res tiling approaches (SAHI) are the 2026 answers; the row is defensible history but weak as advice.

STALENESS:
- YOLO evolution stops at v8 (Jan 2023); missing YOLOv9/v10/v11 and the NMS-free v10 design, and missing RT-DETR — the current real-time SOTA family (Detection Paradigms and Selection sections).
- SAM section covers SAM 1 only; SAM 2 (2024, video + memory attention) is absent and expected knowledge by 2026.
- No open-vocabulary detection (Grounding DINO 2023, YOLO-World 2024) — a mainstream 2024-2026 topic absent entirely.
- The ViT-vs-CNN table's 'Needs large data (JFT-300M)' framing is 2020-era; DeiT/DeiT-III and modern recipes changed this in 2021-2022.
- No MAE (2021), DINOv2 (2023), or registers (2023) — the SSL wave that defines modern vision backbones is absent from the body; ConvNeXt-V2 (FCMAE) also unmentioned.
- Segmentation content is 2015-2018 (FCN, U-Net, DeepLab, Mask R-CNN) plus a 3-bullet SAM; Mask2Former (2022) and the mask-classification unification are missing.
- The 'Highest accuracy' recommendation (DETR variants, Cascade R-CNN) predates Co-DETR/DINO-with-foundation-backbone results that top COCO in the 2024-2026 era.
- Legacy weighting: DenseNet and SSD get body coverage while MAE/CLIP get none — inverted relative to 2026 interview value; DenseNet survives only as trivia ('why did it lose to ResNet'), which the follow-up already handles.

MUST-KNOW:
- ViT patch embedding math (N = HW/P^2) and how token count — and O(n^2) attention cost — scales with resolution; position-embedding interpolation when resolutions change
- Inductive bias vs data/pretraining trade-off, with the modern resolution: pretrained ViT (CLIP/DINOv2) beats from-scratch anything at small data
- CLIP contrastive pretraining (InfoNCE, temperature, batch-size dependence) and why it made ViT the universal backbone — even though the chapter doesn't teach it yet
- MAE self-supervised pretraining and why high mask ratios work for images
- DETR set prediction + Hungarian matching, why it kills NMS, and its failure modes (training cost, small objects) plus the Deformable/DINO fixes
- IoU, NMS, and COCO mAP mechanics — the screening-question layer
- FPN: why multi-scale features are necessary and how lateral connections work
- Effective vs theoretical receptive field and its sqrt-depth growth
- FLOPs vs latency: MACs convention, memory-bound vs compute-bound, why fewer FLOPs can be slower on target hardware
- SAM's promptable-segmentation design (heavy amortized encoder, light decoder) and SAM 2's video extension

IMPROVEMENTS:
- Fix the FLOPs question (section: 'Estimate the FLOPs for a ResNet-50...'): pick one convention and state it. The answer's own formula (2*k^2*Cin*Cout*H*W) yields ~8.2 GFLOPs for ResNet-50, but it declares 'Total: 4.1 GFLOPs (Actual: 4.1)' — 4.1 is the MAC count. Same for ViT: '4*N*D^2 for QKV + output' is 4 matmuls counted at 1 FLOP/MAC (true FLOPs = 8ND^2), and 17.6 G is the MAC-convention figure (true FLOPs ~35G). Add an explicit 'papers quote MACs, call them FLOPs' warning — this is itself a great interview talking point.
- Update the YOLO lineage ('v1 -> v2 -> v3 -> v4 -> v5 -> v8'): add YOLOv9/v10/v11, and specifically YOLOv10's NMS-free consistent-dual-assignment design (it directly connects to the DETR/NMS discussion) and RT-DETR, which beats YOLO on the real-time accuracy-latency frontier — this also makes the Detection Architecture Selection table stale (it recommends YOLOv5, a 2020 model, for real-time).
- In the CNN evolution table: label which ResNet variant the 78.6% refers to (ResNet-50 original is ~76.1%; 78.6 is closer to ResNet-152), and footnote that ConvNeXt's 87.8% uses ImageNet-22k pretraining — the table silently mixes 1k-only and extra-data numbers.
- Hedge the '1M images crossover' claim in the CNN-vs-ViT answer: the ViT paper's evidence brackets the crossover between ImageNet-1k (1.3M) and ImageNet-21k (14M) scales and it is recipe-dependent (DeiT shifts it); the current false precision invites an expert challenge. Also soften 'CNN scaling has diminishing returns' — it partially contradicts the same answer's L7 bullet about ConvNeXt matching ViT at scale.
- Add a short 'why frontier labs use plain ViT, not Swin' judgment paragraph after the Swin section: FlashAttention made global attention cheap at practical resolutions, MAE/CLIP pretraining requires plain tokens, windowing complicates scaling — then position Swin as historically important rather than the current recommendation.
- The visual-search answer should note HNSW graph-link memory overhead (~100-200 bytes/vector, i.e., another 10-20 GB at 100M scale) and that image-image cross-encoder reranking is unusual in practice (metadata/multi-signal rerankers are the norm) — as written, an interviewer running search infra would push back.
- Add a 'design the vision encoder for a multimodal LLM' interviewq (resolution/tiling strategy, token budget vs quality, SigLIP vs CLIP vs DINOv2 features, projector choice) — this is the highest-frequency 2026 vision design question at frontier labs and the chapter has nothing pointing at it.
- RoI Align appears twice ('not RoI Pool for better spatial alignment') without ever explaining the mechanism (bilinear sampling vs coordinate quantization) — one paragraph fixes a classic probe.


# sections/11_generative_models.tex (Generative Models)

VERDICT: Strong on 2013-2022 fundamentals — the ELBO, reparameterization, CFG, and flow-matching answers are among the best in the book — but the body is missing the 2023-2026 production stack (DiT, few-step distillation, evaluation, video, the AR-image-gen revival), the taxonomy figure misclassifies flow matching as implicit density, and the L7 Stable Diffusion cost-estimation answer contains component numbers that are off by up to three orders of magnitude.

STAFF-LEVEL: The interviewq layer is largely at the right altitude — the ELBO, CFG, WGAN, and flow-matching answers demand mechanism-level understanding and the debugging/design questions test practitioner instincts (conditional-vs-unconditional ablations, fidelity QA gates, LoRA fine-tuning plans). But the body sections are L4/L5 flashcard material, and the chapter's only true scale-intuition exercise (the SD cost estimate) is the one with broken numbers — which is worse than absent, because estimation questions are precisely where L6/L7 candidates get cross-examined. The larger Staff-level gap is that the chapter teaches how each model family works but not the 2026 production judgment layer: when to distill and what it costs in quality, how to evaluate beyond FID, U-Net-vs-DiT and pixel-vs-latent trade-offs, serving economics (cost per image, batching diffusion, VAE-decode share at low step counts), and how image generation is converging with the multimodal-LLM stack. Depth is misallocated: StyleGAN3's aliasing theory gets paragraphs an interviewer will never reach, while 'how would you evaluate this model' — a question every candidate will get — has no supporting section at all.

MISSING (critical/high):
- [critical] DiT (Diffusion Transformer): patchified latent tokens, adaLN-Zero conditioning, and why transformers replaced U-Nets (scaling laws, hardware efficiency, unified tooling) :: DiT is the backbone of SD3, Flux, Sora, and essentially every frontier image/video model since 2024; the chapter's architecture story is U-Net-only with DiT relegated to one L7 bullet. 'Why did diffusion move from U-Nets to transformers' is a stock 2026 question.
- [critical] Evaluation of generative models: FID definition (Frechet distance on Inception features) and its failure modes, CLIPScore, precision/recall for generative models, human preference evals (ELO/arena, GenEval, PartiPrompts) :: FID is invoked twice in strong answers but never defined anywhere. 'How would you evaluate your text-to-image model' is a near-guaranteed interview question and the chapter provides no answer; GAN section even lists 'evaluation difficulty' as a challenge without naming a single metric.
- [high] Score-based view and prediction parameterizations: epsilon-prediction as scaled score estimation, x0- and v-prediction, SDE/ODE duality, terminal-SNR/zero-SNR pitfalls, cosine noise schedule :: 'What does the denoiser actually estimate' is the canonical depth probe on diffusion; the word 'score' appears only in one L7 bullet. v-prediction and ZSNR are practical knowledge anyone who has trained a diffusion model has, i.e., exactly the 'have you done this' signal.
- [high] Few-step generation as a section: progressive distillation, consistency models/LCM, adversarial distillation (SDXL-Turbo/ADD), CFG distillation, and the modern sampler landscape (DPM-Solver++, Euler ancestral) as ODE solvers :: Production image generation in 2026 runs at 1-8 steps; the chapter treats 50-step DDIM as the baseline and mentions consistency models only in follow-ups. 'How do you get diffusion to 4 steps and what breaks' is a standard L6 systems question.
- [high] Video generation: 3D/causal VAEs, spatiotemporal patches, video DiT (Sora-class models), temporal consistency challenges, and cost scaling (seconds of video = orders of magnitude more tokens) :: Video/world models are a 2025-2026 frontier-lab hiring focus; a generative-models chapter with zero video content will leave candidates blank on 'extend your image pipeline to video' follow-ups the book itself asks (DETR chapter asks the video question; this one doesn't).
- [high] Autoregressive image generation and tokenizers: VQ-VAE/VQGAN mechanics (codebook, straight-through, perceptual+adversarial losses), FSQ/LFQ, VAR (next-scale prediction, NeurIPS 2024 best paper), and native multimodal image generation (GPT-4o-style AR image output) :: The taxonomy figure names PixelCNN and the comparison question names autoregressive models, but there is no body coverage; the 2024-2026 AR revival plus image tokenizers is exactly what frontier labs (building unified multimodal models) probe. VQ-VAE gets one table row despite being load-bearing for the whole AR branch.

CORRECTNESS:
- Taxonomy figure (sec. 'Major Approaches'): Flow Matching classified as implicit density alongside GANs — wrong; it is a continuous normalizing flow with tractable (if expensive) likelihood via change of variables.
- SD inference-cost interviewq: VAE encoder cost '~0.2 GFLOPs' and decoder '~0.5 GFLOPs' computed from parameter counts with no spatial multiplier — off by roughly three orders of magnitude (decoder ~1-2 TFLOPs at 512x512); U-Net per-pass '10-15 TFLOPs' overestimates by ~5-10x (real ~1-2 TFLOPs); '60% utilization' for batch-1 U-Net inference is far too high (memory-bound, ~10-20% MFU). The final latency is only right because the errors cancel.
- Same question: CLIP ViT-L/14 text encoding '~0.5 GFLOPs' — with ~85M-param text tower and 77 tokens the transformer rule gives ~10 GFLOPs; harmlessly negligible either way but numerically wrong by ~20x.
- WGAN interviewq: attributes mode collapse to JS 'giving equal loss for close and far distributions' — conflates the vanishing-gradient argument with mode-seeking; and overstates WGAN's fix ('mode coverage improves' presented unconditionally).
- CFG interviewq L7 bullet: score decomposition written as an equality with the guidance weight w inside — only valid at w=1; as printed it is a wrong identity.
- Comparison table: 'GAN latent space: Entangled' contradicts the chapter's own StyleGAN W-space section; 'Diffusion latent space: N/A' is misleading for latent diffusion models.
- Key-models table: DALL-E 2 innovation listed as 'CLIP guidance' — its mechanism is a diffusion prior over CLIP embeddings (unCLIP); CLIP guidance belongs to GLIDE.
- Latent diffusion section: 'Benefits: 8-16x less computation' — inconsistent with the f=8 compression arithmetic (64x fewer positions, 48x fewer dims) the chapter itself states later.

STALENESS:
- No DiT anywhere in the body — the diffusion architecture story ends at the 860M U-Net (2022) while every frontier model since 2024 is a transformer.
- Key-models table stops at DALL-E 2 / Imagen (2022); missing SDXL, SD3, Flux, and any 2024-2026 model — 'DALL-E 3' appears once in a decision list, nothing newer.
- Flow matching framed as an emerging L7 curiosity ('likely the default going forward') when by 2026 it is the shipped default — badge and placement are one hype-cycle behind.
- No few-step distillation section despite 1-8-step generation being the 2025-2026 production norm (LCM, Turbo/ADD, consistency distillation appear only in follow-ups).
- Zero video generation content in a 2026 generative-models chapter (Sora-class models, 3D VAEs, temporal consistency).
- No mention of the AR image-generation revival (VAR 2024, native multimodal image output in 2025 frontier models) — the comparison question's 'language -> autoregressive, images -> diffusion' framing is now outdated as stated.
- StyleGAN/GAN internals over-weighted (~40% of the chapter's body text) relative to their 2026 interview value; the chapter's own warningbox says 'default to diffusion,' undercutting the page allocation.
- 'When would you still choose a GAN in 2025?' follow-up hard-codes a year that reads stale in a 2026 book; also the modern answer (GAN losses live inside VAE/tokenizer training and adversarial distillation, not standalone generators) isn't in the chapter.

MUST-KNOW:
- ELBO decomposition (reconstruction + KL + posterior gap) and the reparameterization trick, including why REINFORCE is the high-variance alternative
- DDPM forward-process closed form, the epsilon-prediction objective, and the fact that the network estimates the (scaled) score
- Classifier-free guidance: conditioning dropout at train time, the extrapolation formula, the 2x inference cost, and the quality-diversity trade-off with scale
- Latent diffusion end-to-end: CLIP text encoder -> U-Net/DiT with cross-attention -> VAE decode, and why latent space (compute per position, O(n^2) attention)
- Flow matching / rectified flow: velocity-field regression on straight interpolation paths, why straighter paths mean fewer solver steps, and its adoption in SD3/Flux
- DiT: patchified latents + adaLN-Zero, and why transformers displaced U-Nets (scaling behavior, hardware, unification)
- The speed stack: DDIM/DPM-Solver as ODE solvers, then distillation (progressive, consistency/LCM, adversarial) to 1-4 steps and what quality degrades
- Why diffusion superseded GANs: stable regression loss and mode coverage vs min-max instability and collapse — with WGAN as the failed rescue attempt
- FID: what it measures, how it's computed, and why it can be gamed / disagrees with human preference
- VQ-VAE/VQGAN tokenization (codebook + straight-through) as the bridge to autoregressive image generation and unified multimodal models

IMPROVEMENTS:
- Fix the taxonomy figure (sec. Generative Model Taxonomy): Flow Matching is listed under 'Implicit Density' next to GANs. Flow matching defines a continuous normalizing flow with computable (change-of-variables) likelihood — it belongs on the explicit side; leaving this teaches candidates a classification an interviewer will immediately reject.
- Rewrite the SD cost-estimation answer (L7 estimation interviewq): (a) VAE encode/decode costed by parameter count ('~100M params so ~0.2 GFLOPs', '~0.5 GFLOPs') ignores that convolutions run at 512x512 — real cost is ~1-2 TFLOPs for the decoder, roughly 3 orders of magnitude off, and decode is a major fraction of runtime for 1-4-step distilled models; (b) applying 2*params*tokens with all 4096 latent positions to a hierarchical U-Net overestimates per-pass cost several-fold (real SD1.5 U-Net pass is ~1-2 TFLOPs, not 10-15); (c) the 60% utilization assumption is unrealistic for batch-1 U-Net inference, which is memory-bandwidth and kernel-launch bound (~10-20% MFU) — the answer lands near the right latency via compensating errors, which is exactly what an expert interviewer dismantles.
- Promote flow matching from an L7/low-frequency interviewq to a body section, and re-badge it L6/high-frequency: by 2026 it is the default formulation (SD3, Flux, most video models) and gets asked at the same rate DDPM did in 2023.
- Rebalance the GAN section: StyleGAN currently gets ~2 pages of internals (AdaIN math, style mixing layer ranges, StyleGAN3 aliasing) — cut to ~40% (keep mapping network/W-space disentanglement and weight demodulation as the interview-relevant ideas) and spend the recovered pages on DiT + evaluation. Keep WGAN, but reframe its payoff as the 'why diffusion won' discussion the follow-ups already gesture at.
- Fix the comparison table's internal contradiction: it labels GAN latent space 'Entangled' while the StyleGAN section's central claim is the disentangled W space; and 'Diffusion: N/A' for latent space is misleading given latent diffusion and DDIM inversion — reword to 'no semantic latent by default; inversion provides one.'
- In the WGAN answer, repair the mode-collapse causal claim ('JS gives equal loss for close-but-different and far-apart distributions, so no incentive to cover modes') — JS saturation explains vanishing gradients on disjoint supports; mode-seeking behavior is standardly attributed to the reverse-KL character of the non-saturating objective and to min-max dynamics, and WGAN-GP empirically still mode-collapses. As written it invites an expert correction.
- Fix the CFG answer's L7 bullet score identity: '∇log p(x|c) = ∇log p(x) + w∇log p(c|x)' is only Bayes' rule at w=1; with w it is the guided (tilted) score, not an equality — write it as the definition of the guided score.
- Correct the DALL-E 2 table row: 'CLIP guidance' is the wrong mechanism (that was GLIDE); DALL-E 2 (unCLIP) uses a diffusion prior over CLIP image embeddings plus a diffusion decoder.
- Latent diffusion section: '8-16x less computation' understates and confuses the claim — spatial positions drop 64x (f=8) and per-step attention cost far more; state the compression factors precisely since the SD walkthrough Q later gets them right (48x).
- Add two missing question archetypes: 'estimate the cost of training a text-to-image model from scratch' (data scale, A100/H100-hours — e.g., SD1.x's ~150K A100-hours — vs fine-tuning) and 'defend the trade-off' on U-Net vs DiT or pixel vs latent diffusion; also a fine-tuning-method selection question (full FT vs LoRA vs DreamBooth vs textual inversion) since the follow-ups keep pointing there.
- In the mode-collapse debugging answer, reconsider 'increase classifier-free guidance dropout rate for underrepresented classes' — raising conditioning dropout weakens the conditional pathway for exactly those classes; the defensible fix is the opposite (lower dropout / stronger conditioning signal or class-balanced sampling), so as written it may teach a wrong remedy.
- Product-photography answer: the 'CLIP similarity > 0.95' fidelity gate is arbitrary and fragile (CLIP cosine scales are not calibrated per-domain); replace with pixel/LPIPS comparison inside the preserved-product mask, which is what fidelity-critical pipelines actually do.


# 12_recommendation_systems.tex — Recommendation Systems

VERDICT: The strongest of the three chapters: the ads pipeline, calibration-to-auction cascade, embedding infrastructure, and debugging questions are genuinely staff-level. But it is frozen circa 2022-2023 — no generative recommenders, no exploration/off-policy machinery, one line on negative sampling — and it contains one badly wrong FLOP calculation and a misleading ESMM description.

STAFF-LEVEL: Depth is right for L6 in the ads/ranking sections: the debugging questions, the auction-economics reasoning, and the embedding-infra design question all test judgment and scale intuition with real numbers. Where it sags to L4/L5: the foundations (MF/NCF/taxonomy) are fine as warm-up but NCF gets uncritical space; sequential rec, evaluation, and cold start are definition-level with no OPE/bandit machinery; and the missing 2024-2026 generative-recommender arc means an L7 'where is this field going' conversation has no support. Scale numbers are generally present and mostly correct — the one badly wrong FLOP estimate stands out precisely because the rest is quantitative.

MISSING (critical/high):
- [critical] Two-Tower training mechanics: sampled softmax with logQ correction (Yi et al. 2019), temperature + L2-normalized embeddings, hard/mixed negative curricula, distillation from the ranker into the retriever :: This is the single most-probed retrieval-training detail in ranking loops ('how do you train the retrieval model, and what's wrong with in-batch negatives?'). The chapter gives it two bullet points with no formula. A candidate who cannot state the logQ correction fails the retrieval deep-dive.
- [high] Generative recommenders and semantic IDs: HSTU / Meta's 'Actions Speak Louder than Words' (2024), TIGER-style semantic-ID retrieval, scaling laws for sequential recommenders :: This is the biggest recsys development of 2024-2026 — Meta deployed trillion-parameter generative recommenders in production and it reframes the whole DLRM-era stack. A 2026 interviewer at a frontier lab will ask 'what replaces DLRM?' and the chapter has no answer. Also fixes the staleness of the SASRec/BERT4Rec-only sequential section.
- [high] Exploration and off-policy evaluation: contextual bandits (Thompson/UCB) as a real section, IPS/SNIPS/doubly-robust estimators, replay evaluation :: Counterfactual evaluation is name-dropped in three answers but never taught. L6/L7 ads interviews routinely ask 'how do you evaluate a new policy without launching it' — the candidate needs the IPS estimator and its variance problem, not the phrase 'counterfactual evaluation'.
- [high] Position-bias mitigation methods: position-as-feature with a shallow bias tower (YouTube Watch Next), PAL, IPW learning-to-rank :: Position bias gets a warningbox saying 'use inverse propensity weighting' with no mechanism. The shallow-tower trick is the standard production answer and a very common follow-up.

CORRECTNESS:
- 'Calculate the daily training data volume' question, Training compute bullet: claims ~10K FLOPs per impression for DLRM fwd+bwd (off by 3-4 orders of magnitude; even a 10M-param MLP is ~60 MFLOPs fwd+bwd) AND says 10^13 FLOPs takes '~33 seconds' on a 300-TFLOPS A100 — 10^13 / 3x10^14 = 0.03 s, an arithmetic error of 1000x. Two independent errors in one bullet.
- Section 'ESMM' (line ~571) and the ESMM interview question: 'the CVR prediction is obtained implicitly as the ratio CTCVR/CTR' — inverted relative to the paper; ESMM supervises the product of a pCTR tower and a pCVR tower exactly to avoid the division estimator, which is numerically unstable and can produce pCVR > 1. An expert interviewer will push back hard on this phrasing.
- Stage 4 Auction section: 'winner pays the minimum bid that would have won (second-price mechanism)' presented as the standard — misleading as a description of 2026 practice (first-price migration, quality-adjusted ad rank omitted).
- Two-Tower section claims 'Softmax cross-entropy or InfoNCE with in-batch negatives' with no mention of the popularity-bias problem in-batch sampling creates — without the logQ caveat the stated training recipe is subtly wrong for skewed catalogs.

STALENESS:
- Sequential recommendation stops at SASRec (2018) / BERT4Rec (2019); nothing on HSTU/generative recommenders (2024) or semantic-ID retrieval (TIGER, 2023) — the biggest currency gap in the chapter.
- Auction section presents second-price as current practice; display exchanges largely moved to first-price by 2019-2021 and auto-bidding dominates 2026 ads ML discussions.
- No mention of LLM-derived content embeddings for cold-start (standard practice by 2025); the cold-start table (popularity/onboarding/content) reads 2018.
- Feature-interaction lineage ends at DCN-v2 (2021)/AutoInt table row; fine as history, but the chapter should say explicitly that the 2024+ frontier moved to sequence/generative modeling rather than new cross layers.

MUST-KNOW:
- Multi-stage funnel with latency/candidate budgets at each stage, and WHY two-tower is the only option at retrieval scale (precomputation + ANN) while interaction models are ranking-only.
- Two-tower training: in-batch softmax negatives, the popularity bias they induce, and the logQ correction; hard-negative mining.
- Calibration: definition, Platt/isotonic/temperature, and the full miscalibration-to-revenue cascade through the auction (overcharge -> budget exhaustion -> fill-rate drop) — plus why AUC and calibration are orthogonal.
- ESMM and CVR sample-selection bias: why training CVR on clicks only is biased, the pCTCVR = pCTR x pCVR entire-space decomposition.
- Multi-task architectures: shared-bottom -> MMoE -> PLE, negative transfer, expert capture, and how to detect it (single-task baselines, gradient cosine similarity).
- Embedding-table scale math: why 99% of DLRM params are embeddings, the GB-level arithmetic (vocab x dim x bytes), sharding strategies, mixed dimensions, feature hashing.
- Offline-online gap taxonomy: selection bias, position bias, temporal shift, proxy-metric mismatch, feature leakage — with mitigations (temporal splits, IPW, A/B validation).
- DIN target-attention: why pooled user history loses the relevance signal and what target-conditioned attention fixes.
- Cold start as an exploration problem (bandits, content bootstrap, exploration slots with IPW correction), not a model trick.
- Position bias correction mechanisms (shallow position tower / IPW), since every 'train on clicks' proposal gets this follow-up.

IMPROVEMENTS:
- Fix the training-compute bullet in 'Calculate the daily training data volume' (see correctness): redo with ~50-100 MFLOPs per impression fwd+bwd for a realistic DLRM and correct the seconds arithmetic; the I/O-bound conclusion survives but the numbers as written would fail an interview.
- Rewrite the ESMM description: pCVR is the output of a dedicated tower trained implicitly through the product pCTCVR = pCTR x pCVR; the paper explicitly rejects computing it as the ratio CTCVR/CTR (unstable, can exceed 1). The current text teaches the rejected estimator.
- Add the Rendle et al. 2020 result to the NCF section (a well-tuned dot-product MF matches/beats NeuMF): it converts a dated L4 section into a staff-level 'know the controversy' nugget, and justifies why production retrieval still uses dot products.
- In the DLRM embedding-memory estimation question, add the production note that embedding tables use rowwise Adagrad (one scalar per row) precisely to avoid 3x Adam optimizer state — the 210 GB Adam figure is fine as an exercise but the rowwise-optimizer point is the L7 answer.
- Give the pre-ranking stage one concrete architecture (COLD or two-tower-with-SE-gating) and mention consistency-with-ranker training (distillation/ordering losses) — currently it is just 'a distilled smaller MLP'.
- Name the ANN trade-offs explicitly (HNSW: memory-heavy, high recall; IVF-PQ: compressed, lower recall; ScaNN anisotropic quantization) instead of listing three names — 'which index and why' is a standard follow-up the book itself asks (Q: Two-Tower trade-off follow-up) but never answers.
- Tighten the audience language: the tldr says 'FAANG'; the book targets frontier labs — reframe around ranking/ads roles at large platforms plus the generative-recommender shift.


# 13_graph_neural_networks.tex — Graph Neural Networks

VERDICT: A well-calibrated chapter for a low-frequency topic — honest about interview value, strong on trade-offs and the 'when NOT to use a GNN' judgment that Staff interviews actually test. Main defects: the GCN equation as written is mathematically wrong (double normalization), link-prediction training mechanics are absent despite link prediction being the top industrial GNN use case, and one interview answer directly contradicts Chapter 12's DIN premise.

STAFF-LEVEL: Depth is appropriately staff-flavored for a differentiator topic: the questions test system design, estimation, and judgment rather than definitions, and the L5/L6/L7 ladders are well-constructed. Too shallow for Staff: link-prediction practice (the actual industrial task), temporal graphs, and heterophily are missing or name-only, so a candidate drilled on this chapter can whiteboard architectures but stumbles on 'how do you split the data so you don't leak edges' — a question that ends interviews. Scale numbers are present and mostly sound (fan-out, memory), which is above par.

MISSING (critical/high):
- [high] Link-prediction training mechanics: edge scoring functions (dot product, MLP decoder, DistMult), negative edge sampling, evaluation (MRR, Hits@K), and leakage-safe splits (removing test edges from message passing, temporal splits) :: Link prediction is the dominant industrial GNN task (friend rec, item rec, KG completion) and the chapter's application tables point to it repeatedly, yet nowhere explains how to train or evaluate it. The leakage trap (test edges visible during message passing) is a classic 'have you actually done this' probe.

CORRECTNESS:
- Eq. (eq:gcn), Section 'GCN: Graph Convolutional Network': MEAN over degree-normalized neighbor terms double-normalizes; GCN's propagation is the symmetric-normalized SUM. The formula as printed is wrong.
- Sections 'GCN' and Q4/Q6: 'transductive... cannot handle unseen nodes without retraining' stated as an architectural absolute — over-strong; the limitation is the full-batch training formulation, and an expert will push back.
- Q6 ('production recommendation at scale'): 'a user's past interactions are all roughly equally informative' — contradicted by Chapter 12 (DIN) and by PinSage's importance-weighted aggregation; as written it is a red-flag statement the book elsewhere penalizes.
- Q5: minor internal inconsistency on whether 67 GB fits an 80 GB A100 ('does not fit... with margin' vs 'borderline'), and the separate 30 GB 'gradients' line is hand-wavy accounting.

STALENESS:
- Graph Transformer coverage stops at GPS (2022-2023); no Graphormer, no acknowledgment that by 2025-2026 transformers-with-structural-encodings largely won molecular benchmarks.
- No mention of GNN-LLM interaction (GraphRAG, LLM node featurization) — the main way graphs show up in 2026 frontier-lab conversations.
- Knowledge-graph section is fine but pre-dates the 2024+ reality that LLMs absorbed much KG-completion mindshare; one sentence of positioning would help.
- Given 2026 interview frequency at OpenAI/Anthropic-tier labs, the chapter's length is roughly right (the tldr's honesty is a strength) — no over-weighting flag beyond keeping it at current size.

MUST-KNOW:
- The message-passing template (MSG/AGG/UPDATE) and that L layers = L-hop receptive field.
- GCN vs GraphSAGE vs GAT: aggregation differences, transductive-vs-inductive implications for production, and the GATv1 static-attention flaw fixed by GATv2.
- Over-smoothing: the Laplacian-smoothing mechanism, the 2-3 layer rule, the cosine-similarity diagnostic, and how it differs from over-squashing and vanishing gradients.
- Neighbor-sampling fan-out arithmetic and why it (plus over-smoothing) caps depth; Cluster-GCN/GraphSAINT as alternatives.
- 1-WL bound: sum vs mean injectivity, GIN, when it matters (molecules, regular graphs) vs when features mask it.
- When NOT to use a GNN: hand-crafted graph features + XGBoost baseline, label propagation as a diagnostic, the >2% improvement bar.
- The production GNN-recsys serving pattern: train inductively, precompute item embeddings, ANN retrieval, incremental refresh (PinSage).
- Heterogeneous graphs: R-GCN with basis decomposition, and why edge-type semantics must not be conflated.

IMPROVEMENTS:
- Fix Eq. (eq:gcn): GCN is a SUM over neighbors of h_u / sqrt(d_v d_u) (then W, then sigma); wrapping the degree-normalized terms in MEAN(...) divides by |N(v)| a second time, i.e., double normalization. As written the formula is not GCN.
- Qualify the repeated absolute claim that GCN 'cannot handle unseen nodes without retraining': the weight matrices apply inductively to any graph; the transductive limitation comes from full-batch training and normalization over the fixed graph, not the parameterization. State the precise reason once (the GraphSAGE paper's actual argument) instead of asserting it as architectural.
- Reconcile Q6's claim that 'a user's past interactions are all roughly equally informative' with Chapter 12's DIN section, whose entire premise is that they are not — and with PinSage itself, which uses importance-based neighbor weighting. Replace with the honest argument: GAT's per-edge attention cost is not worth it at billions of edges when importance sampling approximates the benefit.
- In Q5, resolve the internal tension between 'does not fit on a single 80 GB A100 with margin' (67 GB) and the red flag's 'borderline even on an A100'; also flag that the 'gradients (30 GB)' line double-counts loosely — activation storage for backprop and parameter gradients are not the same object; make the accounting explicit.
- Add one worked pair of 1-WL-indistinguishable graphs (e.g., two triangles vs a 6-cycle) — the follow-up 'give an example' is asked in the book but never answered in the text.
- The fraud question's serving design (precompute hourly, lightweight MLP at transaction time) is good — add the numbers (embedding staleness window vs fraud velocity) to make the trade-off quantitative.


# 14_efficient_architectures.tex — Efficient Architectures

VERDICT: Solid 2023-2024 chapter with genuinely good MoE, quantization, and pruning trade-off content, but it is the most currency-exposed of the three: MoE practice moved to fine-grained experts and aux-loss-free balancing (DeepSeek-V3), Mamba-2 and MLA get one bullet or zero mention, and the CNN/NAS material is over-weighted for 2026 frontier-lab interviews. It also contains a materially wrong KV-cache number in a high-frequency question.

STAFF-LEVEL: The trade-off framing, red flags, and estimation questions are pitched correctly at L6 — this chapter tests judgment, not definitions, and its per-question L5/L6/L7 ladders are among the best of the three files. Two things pull it below Staff for a 2026 frontier lab: currency (its MoE/SSM/quantization frontier is 2023-2024, and L7 rows carry names — Mamba-2, on-policy distillation — that the body never teaches, so a reader cannot actually reach the L7 bar from this text) and the page allocation (mobile CNNs and NAS get depth a frontier-lab loop will never probe, while KV-cache efficiency, the most-probed efficiency math of the era, appears only inside one question — with the wrong number). Scale intuition is otherwise good: the chapter consistently attaches GB and FLOP figures to claims.

MISSING (critical/high):
- [critical] DeepSeek-V3-generation MoE: auxiliary-loss-free load balancing via per-expert bias adjustment, fine-grained expert segmentation (top-8 of 256, ~1/20-1/32 activation ratios), sigmoid routing, node-limited routing :: This is the 2025-2026 canonical answer to 'how do you balance experts without hurting quality' — the chapter teaches only the aux-loss era. Every frontier lab's MoE conversation now starts from DeepSeek-V3/Llama-4/Qwen3-class designs; 'typical N=8-64, k=1-2' is no longer typical.
- [high] Mamba-2 / State Space Duality as a real subsection: SSM-linear-attention equivalence, larger states, tensor-core-friendly formulation; plus the 2024-2025 hybrid landscape (Jamba-1.5, Zamba, Griffin/RecurrentGemma) and the linear-attention family (GLA, DeltaNet, RWKV-6/7) :: Mamba-2 exists only as one L7 bullet. The SSD result is the standard 2026 framing for the entire sub-quadratic space, and 'hybrids with ~1:7 attention:SSM ratios' is now empirical consensus interviewers expect candidates to know.
- [high] KV-cache-efficient attention as an efficiency topic: MQA/GQA sizing math and MLA (DeepSeek's latent KV compression), at least as a cross-referenced paragraph :: The chapter's own long-context question turns on KV-cache size, and MLA is one of the most-cited efficiency innovations of 2024-2026. Even if another chapter owns attention, an 'efficient architectures' chapter in 2026 that never says 'MLA' is incomplete.
- [high] Reasoning-model distillation: R1-style distillation of long chain-of-thought traces into small models, and on-policy/GKD-style distillation :: DeepSeek-R1-Distill is the canonical 2025-2026 distillation example and connects this chapter to the test-time-compute era; the pipeline question's L7 bullet mentions on-policy distillation but the body predates reasoning traces.

CORRECTNESS:
- Q3 (transformer vs Mamba at 100K tokens): 'For a 7B model with GQA (8 KV heads), 100K context in FP16 requires ~50GB of KV cache' — wrong. 2 x 32 layers x 8 kv_heads x 128 head_dim x 100K x 2B = ~13 GB; ~50 GB is the full-MHA (32 KV heads) figure. The stated config contradicts the stated number, in a high-frequency question where the interviewer will check the arithmetic.
- Pruning section: 'lottery ticket hypothesis (Frankle and Carlin, 2019)' — the author is Carbin. Citation error in a chapter that name-checks papers.
- Warningbox 'Quantization affects attention layers more than FFN layers... keep attention projections at higher precision' — over-generalized and presented as settled fact; outlier channels also live at FFN inputs/down-projections and many production recipes protect FFN down_proj and boundary layers. As a blanket rule it would draw expert pushback.
- Internal inconsistency: Mixtral active parameters given as ~13B (SSM/MoE sections), ~14B (keyinsight), and 15.5B (Q4's own accounting); real value is 12.9B of 46.7B — three conflicting numbers for the flagship example.
- 'Typical choices: N_experts = 8-64, k = 1-2' stated as current practice — stale to the point of being misleading in a 2026 interview (fine-grained 64-384 experts with top-6/8 dominate new MoEs).

STALENESS:
- MoE section is pre-DeepSeek-V3: no aux-loss-free balancing, no fine-grained/high-sparsity experts, expert counts and top-k presented as 8-64/top-1-2; the 2024 table rows (Mixtral, DeepSeek-MoE) are the newest content.
- Mamba-2 (2024) reduced to one L7 bullet; no Griffin/Zamba/linear-attention revival; the 'hybrid is likely the direction' sentence was correct in 2024 and is settled fact by 2026.
- Quantization table stops at 2023 methods; no rotation-based PTQ, no KV-cache quantization, no FP4/MXFP formats despite Blackwell being current hardware.
- No mention anywhere of MLA or of reasoning-model distillation — both named developments of the 2024-2026 window this book targets.
- Efficient-CNN + NAS material (~40% of the pre-questions body) is over-weighted relative to its 2026 interview value at frontier labs; NAS in particular survives mostly as history ('its main legacy is those architectures' — the chapter admits this, then spends pages on it).

MUST-KNOW:
- MoE fundamentals: parameter-vs-compute decoupling, memory equals TOTAL params, attention is shared (the classic estimation trap), and the ~2x-not-8x FLOP arithmetic for top-2-of-8.
- MoE failure modes: routing collapse mechanism, detection (expert utilization, router entropy), and the fix spectrum from aux loss through expert-choice to DeepSeek-V3's aux-loss-free bias balancing.
- KV-cache arithmetic cold: 2 x layers x kv_heads x head_dim x seq x bytes, and how GQA/MLA change it — this single derivation gates many efficiency interviews.
- Quantization decision framework: INT8/FP8 near-free defaults, INT4 for memory-bound fits, why reasoning/code degrade first (outlier features, compounding errors), and 'benchmark your task, not perplexity'.
- Structured vs unstructured pruning: why 90% unstructured sparsity gives ~0 GPU speedup, and 2:4 N:M as the hardware exception.
- Attention-vs-SSM trade-off: lossy fixed-size state vs exact retrieval, why hybrids won, and Mamba-2/SSD as the unifying frame.
- Modern LLM distillation: data-level generation (and reasoning-trace distillation) over logit matching, and why.
- Arithmetic intensity / roofline intuition: FLOPs are not latency; memory bandwidth governs decode, depthwise convs, and small-expert MoE alike.
- MoE-vs-dense decision under deployment constraints: memory-bound favors dense, throughput-bound serving favors MoE; fine-tuning fragility of routers.

IMPROVEMENTS:
- Fix the KV-cache number in Q3 (see correctness) and show the formula: 2 x layers x kv_heads x head_dim x seq_len x bytes — interviewers make candidates derive exactly this, so the worked number must be right.
- Unify Mixtral's active-parameter count: the chapter states ~13B, ~14B, and 15.5B in three places (actual: 12.9B active of 46.7B). Pick the real numbers and reuse them.
- Update the MoE 'typical choices' line and the architecture table through DeepSeek-V3 (aux-loss-free balancing, 256 fine-grained experts) and note Switch's capacity-factor/dropped-token machinery is largely superseded by dropless/balanced designs.
- Soften the warningbox claiming quantization hurts attention more than FFN: evidence is mixed — massive-activation outliers concentrate at specific channels including FFN down-projections, and common recipes protect down_proj and first/last layers. State it as 'sensitivity is layer- and channel-specific; measure per-layer error' rather than a blanket rule.
- Compress the CNN/NAS sections: keep depthwise-separable math and the arithmetic-intensity lesson (they generalize), but MobileNetV1-vs-V2-vs-V3 selection guidance and DARTS details are low-yield for 2026 frontier-lab loops; reinvest the pages in MoE/SSM/KV-efficiency.
- Correct the citation 'Frankle and Carlin, 2019' to Frankle and Carbin.
- In the MoE-vs-dense question, add the inference-economics angle explicitly: MoE decode is memory-bandwidth-bound, so at low batch the wins shrink; at high-batch serving with expert parallelism the picture inverts — currently only hinted at in an L7 bullet.
- Add one design-from-scratch question at the modern frontier: 'You have a fixed serving budget of X GB and Y ms/token — design the model (dense vs MoE, expert count/granularity, GQA/MLA, quantization)' — the chapter has the pieces but never composes them.


# sections/15_training_optimization.tex (Training Optimization)

VERDICT: The strongest kind of chapter for this book: the interview questions (3D parallelism, NaN debugging, throughput math, FP16/BF16/FP8) are genuinely representative of frontier-lab loops. But the expository body is 2022-era in places, the memory math is internally inconsistent (16 vs 18 bytes/param), the activation-memory formula is wrong by ~5-15x, and the optimizer story stops at AdamW when 2026 loops probe Muon, muP, and LLM-specific hyperparameter lore.

STAFF-LEVEL: The question set is genuinely L6-calibrated: quantitative memory/throughput math, topology-aware parallelism design, and multi-hypothesis debugging with targeted experiments are exactly what Staff loops test, and the L5/L6/L7 rubrics are well-differentiated. The body sections are a level below the questions: Regularization and much of the schedules/optimizer prose is L4/L5 definitional material, and the 'have you actually done this' details (beta2=0.95, no-WD-on-norms, z-loss, spike-skip restarts, MoE/context parallelism, muP) live only in follow-up prompts or are absent entirely -- meaning a candidate who studies only the body will still get exposed on the first depth probe. Scale intuition is mostly present but undermined by the 16-vs-18-bytes inconsistency and the broken activation formula; those must be fixed because interviewers redo this arithmetic live.

MISSING (critical/high):
- [critical] LLM pretraining hyperparameter lore: beta2=0.95 (not 0.999), weight decay 0.1, grad clip 1.0, and the standard practice of excluding LayerNorm/bias/embeddings from weight decay :: This is the single clearest 'have you actually trained an LLM' tell. The chapter gives beta2=0.999 and lambda=0.01 as LLM defaults, which contradicts GPT-3/LLaMA/Chinchilla practice; an interviewer will probe why beta2=0.95 improves spike resilience.
- [high] Muon and the post-AdamW optimizer landscape (Muon/MuonClip as used in Kimi K2/Moonlight, Shampoo/SOAP, Lion, 8-bit Adam) :: By 2026 'what comes after AdamW' is a standard frontier-lab question; Muon has been used for 1T-parameter-scale pretraining. Adafactor as the only alternative in the table reads as 2020.
- [high] muP / muTransfer (maximal update parametrization) for hyperparameter transfer across scale :: The standard answer to 'how do you pick the LR for a model too expensive to sweep' at every frontier lab; L6/L7 loops ask it directly. Currently absent even from L7 bullets.
- [high] Critical batch size and the gradient noise scale (McCandlish et al.), plus batch-size warmup/ramp used in GPT-3-style pretraining :: The chapter says linear scaling 'breaks at B>8K' with no principled framework; the gradient-noise-scale answer is what distinguishes L6 from L5 on any batch-size question.
- [high] Training-stability techniques at scale: z-loss (PaLM), QK-norm / logit growth control, skip-batch-and-restart practice for loss spikes, embedding-layer instabilities :: Loss spikes in real pretraining are handled with these; they appear only as a follow-up prompt ('how do PaLM/OPT handle spikes') with no answer content anywhere in the chapter.
- [high] Expert parallelism and MoE training (routing/load-balancing losses, aux-loss-free balancing a la DeepSeek-V3, all-to-all communication cost) :: Most 2026 frontier models are MoE; the distributed-training section covers only dense-model parallelism, with expert parallelism relegated to one L7 bullet.

CORRECTNESS:
- GPU Memory Estimation section: component table sums to 16 bytes/param but the stated total is '~18 per param' and the 7B example uses 18 (126 GB); Q2 uses 16 bytes/param for 70B (1120 GB). Internally inconsistent -- an interviewer doing the math live will catch it.
- Activation Memory subsection: 'Activation memory ~= 2*b*s*d*L*bytes' underestimates transformer activation memory by roughly 5-15x (omits QKV projections, 4d MLP intermediates, attention terms); the claim that the factor of 2 'accounts for forward and backward' is not how activation memory works.
- Regularization / Weight Decay: 'Typical values: lambda = 0.01 for AdamW (LLMs)' -- GPT-3, LLaMA, and essentially all modern LLM pretraining use 0.1.
- Adam subsection presents beta2=0.999 as the default without noting LLM pretraining universally uses beta2=0.95 for stability -- misleading by omission given the book's LLM framing.
- Q7 (distributed slowdown), Hypothesis 1: all-reduce volume stated as 2P bytes (14 GB) per GPU; ring all-reduce is ~2x gradient size (~28 GB), so the 560 ms estimate is ~2x low; and 'gradient_as_bucket_view=True' is misdescribed as the overlap fix (overlap is default; that flag saves memory).
- Q10: attributes the constant-LR+cooldown result to Chinchilla; Chinchilla used cosine and its schedule finding was about matching cosine horizon to training length.
- SGD Pitfalls: 'Without a schedule, momentum SGD often stalls at saddle points' -- unsupported; saddle stalling is not the characteristic failure of unscheduled momentum SGD (slow late-stage convergence is).

STALENESS:
- Optimizer coverage ends at AdamW/LAMB/Adafactor; no Muon, Shampoo/SOAP, Lion, or 8-bit optimizer states -- the 2024-2026 optimizer conversation is absent.
- All hardware math anchors on A100 (312 TFLOP/s); H100/B200 and FP8-era throughput should be the primary anchor in 2026.
- Regularization section (dropout placement, early stopping, MixUp/CutMix emphasis) is vision-era and overweighted relative to its 2026 interview value; LLM-era regularization practice is missing.
- LARS/LAMB get a full subsection but are near-dead in 2026 practice; compress to a paragraph and reallocate space to Muon/muP.
- Learning Rate Finder subsection is fast.ai-era with low interview value today; candidates citing it over muP/sweep-at-small-scale practice reads junior.
- No mention of MoE training anywhere in the distributed section despite MoE being the dominant frontier architecture by 2026.

MUST-KNOW:
- AdamW vs Adam+L2: why decoupled weight decay differs under adaptive scaling (the update equation, not just the slogan)
- Training memory math cold: params/grads/Adam states/master weights bytes-per-param, plus activation memory and what gradient checkpointing trades (O(sqrt(L)) memory for ~33% recompute)
- 6P FLOPs per token, MFU (~40-50%), peak BF16 FLOP/s for A100 and H100, and the ability to sanity-check tokens/sec against a published run
- BF16 vs FP16 (exponent vs mantissa, why loss scaling exists) and the FP8 E4M3/E5M2 story
- DDP vs ZeRO-1/2/3 (FSDP) vs TP vs PP: what each shards, its communication pattern, and which maps to NVLink vs inter-node fabric; composing 3D parallelism with concrete degrees
- Warmup + cosine and WSD schedules: why warmup exists, why cosine forces a fixed budget, why constant+cooldown enables continual pretraining
- The loss-spike / NaN debugging playbook: gradient norms, clipping at 1.0, data audit, precision issues -- in priority order
- Linear scaling rule, its breakdown, and the critical batch size concept
- LLM pretraining hyperparameters as a package: LR ~1e-4 to 3e-4 by size, beta2=0.95, wd=0.1 (excluding norms/embeddings), clip 1.0, warmup ~2K steps

IMPROVEMENTS:
- Reconcile the bytes-per-parameter accounting: the GPU Memory Estimation table lists components summing to 16 bytes/param (BF16 grads) but totals '~18'; Q2 then uses 16 bytes/param (1120 GB for 70B). State the convention explicitly (16 with BF16 grads, 18 if grads kept/accumulated in FP32) and use it consistently.
- Replace the activation-memory formula with the standard per-layer accounting (Korthikanti et al.: ~s*b*h*(34 + 5*a*s/h) bytes per layer in BF16 without recomputation, less with FlashAttention), and show one worked example; the current 2*b*s*d*L formula ignores QKV, MLP 4d intermediates, and norms.
- Modernize the Regularization section for the LLM era: state that modern LLM pretraining uses dropout=0 (single-epoch training, no overfitting) with weight decay 0.1 as the main explicit regularizer; keep dropout guidance for fine-tuning/small-data regimes. The current 'Transformer standard p=0.1' guidance is 2019-era.
- Q8 (13B LoRA vs full FT memory) duplicates chapter 16's territory almost verbatim; replace with a cross-reference and use the slot for an MFU-optimization or fault-tolerance question.
- In Q7 Hypothesis 1, fix the communication arithmetic: ring all-reduce moves ~2x the gradient size per GPU (~28 GB for a 7B BF16 model, ~1.1 s at 25 GB/s), and replace 'gradient_as_bucket_view=True' (a memory optimization) with the correct statement that DDP overlaps bucketed all-reduce with backward by default.
- In Q10, drop Chinchilla as a citation for constant+cooldown matching cosine (Chinchilla's finding was matching cosine length to training length); cite MiniCPM / Hagele et al. 2024 (WSD) instead.
- Add a short 'why warmup, mechanistically' upgrade: Adam's second-moment miscalibration early in training (RAdam analysis) plus curvature at init -- currently the warmup section is intuition-only.
- Note in the Throughput Estimation example that a 7B model does not fit for training on a single A100-80GB; frame the 3,000 tok/s as per-GPU throughput within a multi-GPU job to avoid teaching a wrong mental model.
- Add HSDP (hybrid sharding) and FSDP2/DTensor to the ZeRO/FSDP subsection -- the DDP-vs-ZeRO-3 binary is no longer how practitioners configure sharding.


# sections/16_transfer_learning.tex (Transfer Learning and PEFT)

VERDICT: Solid on the LoRA/QLoRA core -- the memory-budget and pipeline-design questions are exactly what gets asked -- but the PEFT landscape is frozen at mid-2023 (DoRA, rsLoRA, GaLore, multi-LoRA serving all missing or follow-up-only), the classic transfer-learning half is overweighted vision-era material, and the practical SFT layer (loss masking, packing, chat templates) that Staff interviewers use to detect real experience is absent.

STAFF-LEVEL: The PEFT half reaches L6 in the questions -- memory budgeting under a hardware constraint, pipeline design with eval gating, and the production-failure diagnosis are the right altitude, with usable L5/L6/L7 rubrics. The transfer-learning fundamentals half reads L4 (decision trees keyed on 'data size' and 'domain shift', vision-era feature-hierarchy framing) and should be compressed. The bigger Staff-level gap is that the chapter's depth stops where real practitioner scars begin: no loss masking/packing/template pitfalls, no multi-adapter serving economics, no time-or-dollar cost intuition (memory numbers are excellent, wall-clock numbers are absent), and the 2024+ method lineage exists only as rubric name-drops -- so a candidate can master this chapter and still fail the second-order probes that distinguish 'read about LoRA' from 'shipped a fine-tune'.

MISSING (critical/high):
- [critical] SFT mechanics: loss masking on prompt tokens, sequence packing, chat-template/EOS-token pitfalls, and extending vocab (training embeddings/lm_head alongside LoRA via modules_to_save) :: These are the 'have you actually fine-tuned an LLM' signals interviewers probe first ('why does your fine-tuned model echo the prompt?'); the chapter mentions formatting to a chat template but never the loss mask or packing.
- [high] Modern LoRA lineage: DoRA (weight-decomposed), rsLoRA (alpha/sqrt(r) scaling for high rank), LoRA+ (asymmetric LRs for A/B), PiSSA (SVD init), AdaLoRA, IA3 :: The chapter teaches only alpha/r scaling and Gaussian-A/zero-B init; by 2026 an interviewer asking 'how does scaling interact with rank' or 'what beats vanilla LoRA' expects rsLoRA/DoRA. All of these appear only as name-drops in L7 bullets or follow-ups with no content.
- [high] Multi-LoRA serving at scale: S-LoRA/Punica-style batched heterogeneous adapter inference, vLLM multi-LoRA, unmerged serving trade-offs :: The book itself asks 'how do you serve 100 different LoRA adapters efficiently?' as a follow-up but contains no content that answers it; this is a real Staff serving question in 2026.

CORRECTNESS:
- Fine-Tuning Pipeline Design, step 2: the model-size selection rule is incoherent as written (conflates model capacity with epochs-per-example); location: Section 'Fine-Tuning Pipeline Design', item 2.
- Q4 (10K-example pipeline), Stage 2: '70B would overfit' is a misleading justification -- larger models fine-tuned with PEFT are not notably more overfit-prone on 10K examples; the real trade-off is cost/serving.
- PEFT comparison table vs QLoRA paper: fixed '90-95% quality' for QLoRA (and '2-5% loss' in Q3) overstates a penalty the original paper explicitly claims to close; should be framed as configuration- and task-dependent.
- Q7/70B question: '~100M LoRA parameters' for rank-16 QKVO on a 70B GQA model overestimates by ~1.5x (GQA K/V projections are 8192x1024; the honest figure is ~65M) -- acceptable as an order estimate but the answer presents it as computed.
- PEFT Motivation: 'Gradients and optimizer states: additional 56+ GB' for a 7B FP32 model understates -- FP32 gradients are 28 GB and Adam states 56 GB, so 84 GB; the '120 GB total' elsewhere is right but the component line is off.

STALENESS:
- PEFT taxonomy and comparison tables are frozen at mid-2023 (LoRA/QLoRA/Adapters/Prefix/Prompt/BitFit); DoRA, rsLoRA, LoRA+, PiSSA, VeRA, GaLore -- all 2024+ and standard by 2026 -- are absent from the body.
- Serial adapters (Houlsby-style) and BitFit get body-level treatment despite near-zero 2026 usage, while multi-LoRA serving -- the actual 2026 production pattern -- has no content.
- ULMFiT gradual unfreezing presented as a named algorithm block is 2018-era with negligible current interview value.
- Prefix/prompt tuning coverage does not note that by 2026 these have largely lost to LoRA in practice and survive mainly as concepts feeding soft-prompt research; the 'Generation tasks' recommendation is dated.
- No mention that frontier-lab fine-tuning of 70B+ models routinely combines LoRA/full FT with FSDP/ZeRO multi-node -- the chapter's framing is single-GPU-centric.

MUST-KNOW:
- LoRA mechanics cold: W = W0 + (alpha/r)BA, why B initializes to zero, target-module choices, and merge-for-zero-latency inference
- The real source of PEFT memory savings: frozen base needs no gradients/optimizer states/master weights -- parameter-count reduction alone is negligible
- Full-FT vs LoRA vs QLoRA memory budgets for 7B/13B/70B, done live: bytes per component plus activations
- QLoRA internals: NF4, double quantization, paged optimizers, and the dequantize-compute path
- When LoRA fails: large distribution shift/new languages need high rank, all-linear targets, or full FT (low-rank hypothesis and its limits)
- Catastrophic forgetting: why frozen-base LoRA still forgets, detection via general benchmarks, mitigation via data mixing/rank/LR/early stopping
- The offline-good/production-bad diagnosis sequence: distribution shift, forgetting, leakage, metric mismatch
- LoRA hyperparameter defaults and why LoRA LR (1e-4 to 3e-4) is ~10x full-FT LR
- Negative transfer: definition, detection against a random-init baseline, and that it is distinct from catastrophic forgetting

IMPROVEMENTS:
- Rewrite the garbled model-selection rule in Fine-Tuning Pipeline Design step 2 ('for N training examples, you need at most a model where each example is seen ~3-5 times in a few epochs' -- as written this describes epochs, not model size); replace with a defensible heuristic tying model size to data volume, task complexity, and serving budget.
- Fix Q4 Stage 2's claim that 'a 70B model would overfit' on 10K examples: larger models are generally more sample-efficient with PEFT; the honest argument for 7B-13B is cost, iteration speed, and serving, not overfitting. An expert interviewer will push back on the stated reason.
- Reconcile the QLoRA quality claims: the comparison table says 90-95% and Q3 says '2-5% quality loss', but the QLoRA paper's headline result is parity with 16-bit LoRA when configured properly (all-layer adapters, NF4+DQ). Present it as task-dependent with parity achievable, not a fixed penalty.
- Add GQA dimensions to the LoRA parameter-math examples: Q1's strong answer assumes 4096x4096 K/V projections; LLaMA-2/3-style GQA makes k_proj/v_proj much smaller, and the L6 bullet already promises this -- the worked example should show it.
- Compress the vision-era first half (gradual unfreezing/ULMFiT algorithm block, discriminative LR, the two decision-tree figures) to a page and state plainly that in 2026 the dominant transfer pattern is foundation-model adaptation; keep LLRD as a one-paragraph note for encoder fine-tuning.
- The PEFT selection flowchart (memory-constrained? quality-critical?) is too coarse to be defensible at L6; replace with a table over the four axes the Q3 strong answer already uses (data quantity, compute, quality bar, serving constraints).
- In the merging discussion, add the quantization interaction explicitly in the body (you cannot merge adapters into a 4-bit base without dequantize-merge-requantize); it currently appears only inside the 70B interview answer.
- Add typical wall-clock anchors to the pipeline question (e.g., 10K examples x 3 epochs on one A100 at rank-16 LoRA is hours, not days) -- the chapter has memory numbers but zero time numbers.


# sections/20_reinforcement_learning.tex — Reinforcement Learning and RLHF

VERDICT: A well-executed 2023-era RLHF chapter: the PPO/DPO core, KL-penalty reasoning, and reward-hacking treatment are solid and interview-shaped. But it is missing the entire 2024–2026 reasoning-RL wave (GRPO, RLVR, DeepSeek-R1), which is now the single most likely RL topic in a frontier-lab loop, and it repeats the discredited 'o1 = PRM + MCTS search' story as fact.

STAFF-LEVEL: The chapter has the right skeleton for Staff (estimation, debugging, and design archetypes all present; Q9's reward-plateau/rising-KL question is genuinely L7). But its ceiling is 2023 RLHF: an L6/L7 loop at a frontier lab in 2026 will spend most of its RL time on GRPO/RLVR/reasoning-RL, training-infrastructure design (rollout/learner split, generation cost dominance), and verifiable-reward hacking — none of which exist here. Scale intuition is present for memory (Q5, with a gradient-sized hole) but absent for throughput, cost-in-dollars, and rollout economics. As written, a candidate who mastered only this chapter would present as a strong L5/borderline-L6 on RL and would be exposed the moment the interviewer says 'GRPO' or 'R1-Zero.'

MISSING (critical/high):
- [critical] GRPO (Group Relative Policy Optimization) and its descendants (DAPO, Dr. GRPO, RLOO, ReMax) :: GRPO is the workhorse of reasoning-model RL since DeepSeekMath/DeepSeek-R1: sample K responses per prompt, advantage = (r - group mean)/group std, no critic — cutting the 4-model setup to 3 and removing value-function training. In 2026 'explain GRPO and why it replaced PPO's critic' is asked more often than PPO details. Known pathologies are also probed: zero gradient when all K rewards are equal, std-normalization difficulty bias (Dr. GRPO fix), length bias, and DAPO's clip-higher/dynamic-sampling fixes. The chapter never mentions any of this.
- [critical] RLVR (RL with Verifiable Rewards) and the DeepSeek-R1 training recipe :: The dominant 2025–2026 paradigm: rule-based accuracy + format rewards on math/code, R1-Zero showing pure RL from a base model works (emergent long CoT, 'aha moments'), then the cold-start-SFT -> reasoning-RL -> rejection-sampling-SFT -> all-scenario-RL multi-stage pipeline, and distillation of reasoning to small models. Interviewers ask candidates to whiteboard exactly this. The chapter mentions 'verifiable reward' only as a bullet in when-to-choose-PPO.
- [high] RLHF/RL training infrastructure :: Staff systems questions: rollout (vLLM/SGLang inference engine) vs learner split, weight synchronization, colocated vs disaggregated actors, why generation dominates wall-clock cost, one-step-off-policy staleness and importance-ratio correction, frameworks (verl, OpenRLHF, TRL, NeMo-Aligner). 'Design the training system for RLHF at 70B' is a real L6/L7 question with zero support in the chapter.
- [high] Token-level mechanics: GAE, reward placement, per-token KL shaping :: GAE lambda appears once as an unexplained hyperparameter name (Q2 table). An interviewer probing PPO depth asks: where does the scalar reward attach (final token), how is the KL penalty actually implemented (per-token reward shaping r_t = -beta*KL_t + R at EOS, not a separate loss), how advantages are computed and whitened. This is the 'have you actually run this' signal.
- [high] Rejection sampling / Best-of-N / RAFT :: Llama 2 and Llama 3 post-training relied heavily on rejection-sampling fine-tuning; BoN is the standard inference-time alignment baseline (with the KL(BoN||ref) ~ log N - (N-1)/N result) and the simplest thing a team ships before RL. Its total absence makes the PPO-vs-DPO dichotomy look like the whole menu when it is not.
- [high] Reward hacking in verifiable/agentic settings :: The reward-hacking section is entirely RM-centric. 2025–2026 interviews probe hacking of verifiable rewards: unit-test special-casing, hardcoding expected outputs, sandbox exploitation in coding agents, and obfuscated reward hacking when CoT is monitored (OpenAI 2025). 'Design a reward for a coding agent that resists test-hacking' is a live question.

CORRECTNESS:
- Sec 20.1.2 + table 'Value-Based vs Policy-Based': claims value-based methods are unused in RLHF 'because vocabulary too large.' Wrong reason — max over vocab logits is a trivial argmax, and PPO's own critic is a learned value function. An expert interviewer will push back; real reasons are off-policy instability and long-horizon sparse-reward credit assignment.
- Sec 20.5.3 and Q10 strong answer: 'PRMs are foundational to o1/o3-style reasoning... generate multiple CoT paths, score each step, beam search or MCTS' — stated as fact; unsupported and contradicted by DeepSeek-R1's published ablations (PRM/MCTS found not fruitful). Teaching a candidate to say this in a 2026 frontier-lab interview is actively harmful.
- Q5 (RLHF memory estimation): gradient memory is omitted entirely (~14–28 GB per trained 7B model, two trained models), so the ~240 GB total and the '3x H100-80GB' feasibility claim are wrong — ironic given the answer's red-flag list penalizes forgetting optimizer states.
- DPO comparison table + Q2: 'Reward hacking risk: Lower' presented as fact — DPO's implicit reward is demonstrably exploited (length/style bias); the honest claim is 'different, not lower.'
- Internal inconsistency: sec 20.1.3 says PPO is more sample-efficient than REINFORCE via trajectory reuse, while table 20.1 lists policy-based methods as 'lower sample efficiency'; neither notes that RLHF practice usually runs a single PPO epoch.
- Q4: 'KL > 20 nats indicates significant drift' — unit-free (per token? per sequence?), and as a per-token figure would be absurd; needs qualification.
- Sec 20.3 KL discussion never notes the objective is REVERSE KL (mode-seeking) — not strictly an error, but the omission lets a candidate be trapped by the standard 'which direction of KL and why does it matter' follow-up.

STALENESS:
- No GRPO anywhere — the chapter's most recent methods are KTO/ORPO/SimPO (~2024); DeepSeekMath (2024) and DeepSeek-R1 (Jan 2025) redefined the field and are absent.
- o1/o3 characterized twice (sec 20.5.3, Q10) as PRM-scored inference-time search/MCTS — a 2023-era hypothesis contradicted by public evidence; the trained-with-outcome-RL account is the 2026 consensus.
- 'PPO is the standard' framing throughout (table 20.1, Q1) — for reasoning RL, critic-free GRPO-family methods are the 2026 default; PPO is the legacy RLHF optimizer.
- Alignment tax presented as inherent 'nicer but dumber' — dated; modern post-training typically raises MMLU/HumanEval/GSM8K.
- No mention of reasoning-model-specific failure modes (entropy collapse, length inflation, test-hacking) or CoT-monitorable reward hacking (2025 literature).
- '32K–128K tokens' vocabulary range (sec 20.1.2) — modern frontier vocabs reach ~256K (Gemma/Llama 4-era).
- Data-volume figures (SFT 10K–100K, RM 100K–500K) reflect InstructGPT/Llama-2-era scale, not 2026 synthetic-data-heavy practice.

MUST-KNOW:
- Whiteboard the three-stage RLHF pipeline and articulate why each stage exists (SFT = format, RM = judgment, RL = optimization under KL anchor).
- PPO clipped-ratio objective: write it, explain what clipping prevents, and explain why the KL penalty exists (imperfect RM proxy) with the beta-too-high/too-low failure modes.
- DPO loss and the implicit-reward derivation (closed-form optimal policy, partition-function cancellation), plus the online-vs-offline framing of PPO vs DPO.
- GRPO: group-relative advantage, why it drops the critic, and its failure modes (all-equal-reward zero gradient, length bias) — the chapter must add this; fumbling GRPO ends 2026 RL interviews.
- RLVR and the DeepSeek-R1 recipe: verifiable accuracy+format rewards, R1-Zero, cold-start multi-stage pipeline, distillation.
- Reward hacking: 3 concrete examples, detection via RM-score/human-eval divergence + KL monitoring, layered mitigations; Goodhart framing.
- The four-model memory footprint of PPO-RLHF and every lever to shrink it (LoRA, shared value head, quantized frozen models, GRPO removes critic, DPO removes RM).
- PRM vs ORM trade-offs stated correctly: dense credit assignment and test-time reranking value, WITHOUT the 'o1 = MCTS' myth.
- Bradley-Terry RM training end-to-end, including annotator noise (70–80% agreement) and the length/format biases the RM inherits.

IMPROVEMENTS:
- Rewrite 'Process vs Outcome Reward Models' (sec 20.5.3) and Q10: replace the 'PRMs are foundational to o1/o3, beam search/MCTS at inference' story with the current account — reasoning models are trained with outcome-based verifiable rewards; DeepSeek-R1 explicitly reported PRM and MCTS as unfruitful; PRMs survive mainly as test-time rerankers/verifiers (Best-of-N selection). Present the PRM+search story as a superseded 2023 hypothesis, not the mechanism.
- Fix the DPO comparison table row 'Reward hacking risk: Lower' and the Q2 strong answer: DPO has a different failure profile, not lower risk — its implicit reward is exploited via length/style just like an explicit RM, and offline data limits detection. Say 'no separate RM to saturate, but implicit-reward overoptimization (length bias) is well documented.'
- Q5 (memory estimation): add gradient memory (~14 GB FP16 or 28 GB FP32 per trained 7B model), which the answer omits entirely while its own red-flag list scolds candidates for forgetting optimizer states; recompute the total (~270–300 GB) and fix the '3x H100-80GB' claim. Also note the standard practice of a value head on the shared policy backbone, and that GRPO removes the critic entirely.
- Fix the value-based vs policy-based rationale (line ~32 and the table): 'vocabulary too large for value-based' is wrong — argmax over vocab logits is trivial (PPO's critic is itself a value function over the same states). The real reasons: off-policy TD instability at LLM scale, sparse terminal reward over 1000+ step horizons, and natural fit of policy gradients with autoregressive sampling. Mention ILQL as the counterexample an interviewer may raise.
- Update Q2/Q8 decision frameworks to include GRPO/RLVR as the third branch (verifiable tasks -> critic-free group-relative RL), and update 'Scaling to frontier' row — 'PPO used by OpenAI/Anthropic, DPO by Meta' is stale shorthand for 2026.
- Q4 (sycophancy): specify units for the 'KL > 20 nats' check (per-sequence vs per-token; 20 nats per token would be catastrophic, per sequence can be normal) and reconcile the internal tension between 'PPO reuses trajectories (sample efficient)' (sec 20.1.3) and the table's 'policy-based: lower sample efficiency' — add the practice note that RLHF typically runs 1 inner epoch.
- Modernize 'Alignment Tax' (sec 20.4.2): the 'nicer but dumber' framing is InstructGPT-era; 2024–2026 post-training generally improves most capability benchmarks (reasoning RL massively so). Reframe as: tax appears when preference data is misaligned with capability (over-cautious annotation), not as an inherent cost.
- Add a coding-archetype question: 'implement the DPO loss (or GRPO advantage computation) in PyTorch' — this is a common frontier-lab screen and the book has no code-level question in this chapter.
- Add concrete run-scale numbers somewhere: typical rollout batch (hundreds of prompts x K samples), KL targets (a few nats/sequence), generation:training cost ratio (~80/20), and a dollar estimate for an RLHF run to serve the estimation archetype.
- Update data-scale figures: '10K–100K SFT examples' (fig 20.1) and '100K–500K comparisons' are 2022–2023 numbers; frontier 2026 practice is millions of largely synthetic SFT examples and preference pairs — keep LIMA as the quality-over-quantity counterpoint but date it.


# sections/23_safety_alignment.tex — Safety, Alignment, and Interpretability

VERDICT: The production-safety engineering half (guardrail pipeline with latency budgets, over-refusal Pareto, hallucination system design) is genuinely Staff-caliber and the best question set in scope. But interpretability is frozen at mid-2024, the jailbreak taxonomy omits every canonical post-2023 attack (GCG, many-shot), and the alignment-research literacy a frontier lab actually probes in 2026 — alignment faking, CoT monitoring, agentic/indirect prompt injection — is entirely absent from a chapter with 'Alignment' in its title.

STAFF-LEVEL: Uneven in an interesting way. The systems-design questions (Q1, Q4, Q5) are the most Staff-appropriate material in either assigned chapter: real latency budgets, Pareto reasoning, root-cause bifurcation, organizational trade-offs. The expository sections beneath them are thinner than the answers they support. Interpretability sits at L5 (definitions plus one circuit) — passable for generalist MLE loops, but at an Anthropic/OpenAI-tier interview an L7 is expected to discuss SAE limitations, the refusal direction, and whether interp can serve as a production monitor. The two largest depth gaps: (1) no agentic-safety design despite agents being the 2026 deployment norm — 'secure an agent with browse+execute tools' is now a more likely design question than the chatbot filter; (2) no alignment-research fluency (alignment faking, CoT monitoring, scalable oversight), which frontier labs use precisely to separate L6 practitioners from L7 candidates with field-level judgment. Numeric scale intuition is good on latency but missing on classifier base rates and interp compute costs.

MISSING (critical/high):
- [critical] Agentic safety and indirect prompt injection :: In 2026 the dominant deployment surface is agents that read email/web/tool outputs and take actions. Indirect injection (instructions hidden in retrieved documents, web pages, tool results), the untrusted-data/private-data/exfiltration-channel 'lethal trifecta,' sandboxing, least-privilege tool permissions, and human-in-the-loop for irreversible actions are now core interview material ('design safety for an agent that can browse and execute code'). The chapter's prompt-injection coverage is one sentence about direct injection plus a bullet.
- [critical] Named modern jailbreak attacks: GCG adversarial suffixes (Zou et al. 2023), many-shot jailbreaking (Anthropic 2024), PAIR/TAP, Crescendo :: The attack taxonomy (role-play, encoding, multi-turn, few-shot) is 2022-era. GCG is THE canonical automated attack and transfers across models; many-shot jailbreaking is the long-context attack every frontier lab studied. Interviewers name these; a candidate who cannot is dated. Also missing: 'shallow safety alignment' (refusal lives in the first few output tokens), which explains why prefill/continuation attacks work.
- [high] Alignment-research literacy: alignment faking (Anthropic 2024), sleeper agents/model organisms, weak-to-strong generalization, scalable oversight/debate, deliberative alignment :: At OpenAI/Anthropic-tier labs, L6/L7 loops for safety-adjacent roles probe whether candidates track the actual alignment literature: 'what is alignment faking and why does it break RLHF assumptions,' 'can safety training be backdoor-persistent,' 'how do you supervise a model smarter than the supervisor.' The chapter contains none of this despite its title.
- [high] Chain-of-thought monitoring and faithfulness :: With reasoning models, monitoring the CoT is a first-class safety tool — and its limits (unfaithful CoT; optimization pressure against monitors producing obfuscated reward hacking, OpenAI 2025) are a hot interview topic bridging this chapter and the RL chapter. Absent entirely.
- [high] The guard-model ecosystem and streaming moderation: Llama Guard 3, ShieldGemma, Anthropic's Constitutional Classifiers (2025) :: The 2025–2026 production stack uses small guard LLMs (1B–8B) rather than DistilBERT-class encoders in many deployments, plus streaming-aware output filtering (holdback windows, incremental classification). Q1's blanket red flag that an LLM classifier 'adds 200ms+' will draw interviewer pushback because it contradicts current practice.
- [high] Base-rate math for safety classifiers :: Classic Staff estimation trap: harmful prompts at 0.1% prevalence with a 99%-accurate classifier yields ~10% precision — at 100M queries/day that is ~1M false blocks. This Bayes-rule reasoning is the difference between L5 and L6 answers on every filtering question in the chapter and appears nowhere.

CORRECTNESS:
- Sec 23.2.3 Detection Methods: 'Self-consistency (Wang et al., 2023)' misattributes the technique — Wang et al. 2023 is self-consistency for chain-of-thought reasoning; the described hallucination detector is SelfCheckGPT (Manakul et al., 2023).
- Q1 red flag ''use a large LLM as the safety classifier' — a 7B model adds 200ms+': overstated as a blanket claim; a short-output guard LLM on a modern GPU with prefix caching classifies in tens of ms, and this is standard production practice. An interviewer running a serving stack will push back.
- Q1 latency design implicitly assumes non-streaming responses (output filters run once on the complete response); presented as a complete production answer, it is misleading — streaming moderation is the actual hard part of the 50ms constraint.
- Q5: 'claims that appear in <40% of responses are likely hallucinated' — an arbitrary threshold stated as a rule; should be framed as a tunable operating point on a precision-recall curve.
- Sec 23.1 warningbox cites Jain & Wallace (2019) as settling 'attention is not explanation' — an expert may push back with Wiegreffe & Pinter's rebuttal ('Attention is not not Explanation'); one hedging clause ('the debate is nuanced; treat attention as diagnostic, not proof') makes the claim safe.
- Q3's induction-head mechanism ('finds the first occurrence of A, then copies what came after') slightly compresses the real attention pattern (the head attends to the token AFTER the first A via its previous-token-enriched key); acceptable simplification, but the L7 rung should state it precisely since this question invites mechanism-level probing.

STALENESS:
- Interpretability content stops at ~mid-2024 (SAEs extracting features); missing 2025 attribution graphs/circuit tracing, transcoders/crosscoders, and the SAE-skepticism debate.
- Jailbreak taxonomy is pre-GCG (July 2023): no adversarial suffixes, no many-shot jailbreaking (2024), no PAIR/TAP/Crescendo, no shallow-alignment framing.
- No alignment faking (Dec 2024) or sleeper agents (2024) in a chapter titled 'Safety, Alignment' — conspicuous at Anthropic-tier interviews.
- Guardrail stack presented as regex + DistilBERT-class encoders; the guard-LLM ecosystem (Llama Guard 3, ShieldGemma, Constitutional Classifiers 2025) that defines current practice is absent.
- Hallucination detection misses semantic entropy (Nature 2024) and the 2025 abstention-aware-evals movement; SelfCheckGPT never named.
- Zero agentic-safety content (indirect injection via tool outputs, computer use, sandboxing) although agents are the 2026 deployment norm.
- 'Bing Chat' naming (renamed Copilot in 2023).
- Safety benchmark table skews 2020–2023 (RealToxicityPrompts, ToxiGen); no 2024+ suites (StrongREJECT, XSTest, JailbreakBench, AgentHarm).

MUST-KNOW:
- Defense-in-depth guardrail architecture (input filter -> system prompt -> model alignment -> output filter -> monitoring) with per-layer latency numbers, and why system-prompt-only safety fails.
- Prompt injection vs jailbreak distinction — and the indirect-injection variant through retrieved documents/tool outputs, which is the live 2026 attack surface for agents.
- Named jailbreak mechanics: role-play, encoding, multi-turn, few-shot, plus GCG suffixes and many-shot jailbreaking; why safety training is 'shallow' and cannot be the only layer.
- Intrinsic vs extrinsic hallucination; the prevent/detect/mitigate stack; 'RAG grounds but does not solve' with the three RAG failure cases (retrieval failure, context conflict, reasoning errors).
- Claim-level verification pipeline: atomic claim extraction -> NLI entailment against context -> action policy, with latency estimates.
- The over-refusal/safety Pareto: diagnose model-level vs classifier-level refusal, measure BOTH directions on every change.
- Base-rate precision math for safety classifiers (low prevalence destroys precision even for accurate classifiers).
- Interp fundamentals that trap candidates: attention != explanation, probe presence != usage, superposition -> polysemantic neurons -> SAEs, plus one concrete circuit (induction heads) told correctly.
- Red teaming as a continuous program (threat model -> attack design -> execute -> severity triage -> patch -> retest -> monitor), not a pre-launch checkbox.
- Conversational fluency on 2024–2025 alignment results — alignment faking, sleeper agents, CoT-monitoring limits — expected at L6/L7 in frontier-lab loops.

IMPROVEMENTS:
- Fix the citation in Detection Methods (sec 23.2.3): 'Self-consistency (Wang et al., 2023)' — that paper is self-consistency for CoT reasoning accuracy, not hallucination detection. The sampling-based hallucination detector is SelfCheckGPT (Manakul et al., 2023); cite it, and add semantic entropy as its principled successor.
- Q1: add a streaming paragraph — production chatbots stream tokens, so output filtering must be incremental (chunked classification, N-token holdback window, kill-switch mid-stream), and the naive 'filter the complete response' design silently assumes non-streaming. This is a real interviewer follow-up the current answer cannot survive.
- Q1 red flags: soften ''use a large LLM as safety classifier' adds 200ms+' — small guard LLMs (Llama Guard 3 1B/8B, ShieldGemma) with prefix caching run in tens of ms and are standard 2025–2026 practice; the correct nuance is encoder-classifier vs guard-LLM trade-off (latency vs policy-conditioned flexibility), not a blanket dismissal.
- Q2: add GCG (automated suffix optimization, transferability) and many-shot jailbreaking as named categories; add the 'shallow alignment' insight (refusals concentrated in first response tokens, hence prefill/continuation attacks) — it upgrades the L7 rung from 'arms race dynamic' hand-waving to a mechanistic account of why jailbreaks work.
- Interp section: add the refusal-direction result (Arditi et al. 2024) as a second concrete worked example alongside induction heads — it is safety-relevant, one paragraph, and directly connects sec 23.1 to the steering-vectors question (Q8), which currently floats unanchored.
- Add a base-rate worked example to the guardrails section: prevalence 0.1%, TPR 95%, FPR 1% -> precision ~8.7%; then tie to Q4's threshold discussion. Cheap to add, high interview yield.
- Benchmarks table (sec 23.5.3): add StrongREJECT (jailbreak robustness), XSTest (over-refusal), AgentHarm (agentic harm); mark RealToxicityPrompts/ToxiGen as legacy-era so the candidate doesn't lead with 2020 benchmarks in 2026.
- Update 'Models like Perplexity and Bing Chat use this pattern' (sec 23.2.4) — Bing Chat has been Microsoft Copilot since late 2023; stale naming reads badly.
- Q7 (red teaming): add dangerous-capability uplift evals (bio/cyber) and external/third-party evaluation (AISIs, contracted red teams) to Step 1/3 — frontier-lab red-team programs are not only about jailbreaks and toxicity.
- Sec 23.1 SAE paragraph: add one sentence on cost/scale (SAEs are trained per layer on billions of activations; feature counts in the tens of millions for frontier models) and one on the 2025 fidelity debate, so the L7 answer isn't uncritically pro-SAE.
- Add a safety incident-response question archetype: 'a jailbreak goes viral against your production model — walk me through the first 24 hours' (mitigation via filters vs model patch, regression risk of over-refusal, comms, post-mortem into RLHF data). Tests operational judgment nothing else in the chapter covers.


# sections/21_multimodal_learning.tex (Multimodal Learning)

VERDICT: A well-organized, mostly accurate 2023-era VLM chapter with a good fusion taxonomy and one genuinely strong estimation question, but its model coverage is frozen at mid-2023 (CLIP/LLaVA/Flamingo/GPT-4V) and misses essentially everything a 2026 frontier-lab interviewer would probe about modern multimodal practice: native/dynamic resolution, video, unified understanding+generation models, omni/speech models, and VLM evaluation.

STAFF-LEVEL: Depth is appropriate for L5 and low-L6: the fusion taxonomy, token-budget math, and the CLIP compute estimation (Q5) are genuinely L6/L7 material, and the L5/L6/L7 rubrics inside questions are well calibrated. What keeps it below staff level is currency and production texture rather than conceptual depth: an L6/L7 candidate in 2026 must speak fluently about dynamic resolution, video token economics, unified generation models, and VLM eval suites, none of which the chapter equips. There is also almost no training-side scale intuition beyond Q5 (no data mixture numbers, no discussion of how much interleaved data native multimodal pretraining takes), which is where L7 conversations actually land.

MISSING (critical/high):
- [high] Native/dynamic resolution and 2D/multimodal RoPE (NaViT patch-packing, Qwen2-VL/Qwen2.5-VL M-RoPE, InternVL dynamic tiling as main text rather than one line) :: By 2026 every serious open VLM uses variable native resolution instead of fixed 336px + tiles; 'how do you handle a 4K screenshot' is a standard interview probe and the chapter's fixed-resolution framing gives a 2023 answer.
- [high] Video understanding as a first-class section: frame sampling policies, temporal pooling vs temporal attention, token budgets for 1hr video, long-video context math (e.g., 1 fps x 3600 frames x 100+ tokens/frame) :: Video-LM roles exploded in 2024-2025 (Gemini video, GPT-4o vision); currently video appears only in follow-up questions, but 'design video search / video QA' is now a top-3 multimodal system-design prompt.
- [high] Unified understanding+generation models: Chameleon (early-fusion discrete tokens), Transfusion (AR text + diffusion images in one transformer), Janus/Janus-Pro, GPT-4o native image generation :: The chapter frames 'natively multimodal' as proprietary-and-unknowable (GPT-4V/Gemini); since 2024 there are open recipes, and 'discrete VQ tokens vs continuous embeddings vs diffusion head' is a real L6/L7 architecture question.
- [high] Speech LLMs and audio generation: neural audio codecs (EnCodec/SoundStream, RVQ), audio-token LMs (VALL-E, AudioLM), GPT-4o-style end-to-end speech-to-speech and its latency motivation vs ASR->LLM->TTS cascades :: The audio section stops at Whisper (2022) + CLAP/ImageBind (2023) — understanding-only. The cascade-vs-native-speech tradeoff (latency, paralinguistics, barge-in) is a common 2026 design question and is entirely absent.
- [high] VLM evaluation: MMMU, MathVista, DocVQA, and hallucination benchmarks POPE/CHAIR :: Q6's own follow-up asks 'how do you evaluate visual hallucination rates?' but the chapter never names any benchmark or metric; an interviewer will expect POPE at minimum.

CORRECTNESS:
- Section 'Multimodal Alignment Across Modalities': ImageBind is attributed to 'Gong et al., 2023' — it is Girdhar et al., 2023 (Meta). An interviewer or reader will catch this.
- Q5 (CLIP compute estimation): 'With 1024 A100s (CLIP's actual scale)' — fabricated; CLIP was trained on V100s (ViT-L/14: ~256 V100s for ~12 days per the paper). Remove or correct the 'actual scale' claim.
- Q5 contains a LaTeX bug: '$\approx 590 days.' has an unclosed math environment (missing closing $), which will corrupt or fail the build.
- Q5 L7 bullet: '32K batch x ~2MB per sample ~= 64 GB' — a 224px RGB image is ~0.3-0.6 MB; the true binding constraint is activation memory per sample, not raw input bytes. The number as stated is hand-waved and the mechanism mislabeled.
- Q4 SigLIP loss formula uses sim/tau - b; the paper parameterizes as a learned multiplicative temperature t times sim plus bias b (b initialized ~-10). Since the L7 rubric explicitly discusses the bias term, the formula should match the paper's parameterization.
- VLM comparison table: LLaVA(-1) listed with 576 visual tokens and 'frozen LLaMA' — both wrong for v1 (256 tokens @ 224px, Vicuna backbone).

STALENESS:
- Entire model roster stops at 2023; no 2024-2025 systems (Qwen2-VL, Chameleon, Llama 3.2 Vision, Molmo, GPT-4o, Gemini 2.x) anywhere in main text.
- 'GPT-4V and Gemini: Natively Multimodal' section treats native multimodality as proprietary and unreproducible — false since Chameleon (2024) and Janus (2024-2025) published open recipes.
- Audio section is understanding-only and pre-dates the speech-to-speech era (GPT-4o, mid-2024); no audio generation, no codecs, no realtime/streaming considerations.
- Text-to-image section names DALL-E 3/Flux but omits rectified-flow/SD3-era framing and the shift of frontier image gen into unified multimodal models (GPT-4o image gen, 2025).
- No mention of video generation even as a pointer (Sora-class models, 2024), which candidates are increasingly asked to contrast with image diffusion.

MUST-KNOW:
- CLIP InfoNCE loss mechanics: N x N similarity matrix, symmetric loss, learned temperature (init 0.07), and why 32K batch size matters (in-batch negatives).
- Visual token budget arithmetic: 336px / patch-14 = 24x24 = 576 tokens, and the compression toolbox (Perceiver Resampler, token merging, tiling/dynamic resolution, pooling).
- Adapter-based (LLaVA/Flamingo/BLIP-2) vs natively multimodal tradeoff: cost, modularity, and the capability ceiling imposed by a frozen CLIP encoder (spatial reasoning, counting).
- Late fusion for retrieval vs cross-attention for reasoning, and the two-stage retrieve-then-rerank production pattern.
- SigLIP vs CLIP: sigmoid pairwise loss removes the global softmax all-gather; distributed-training implication.
- LLaVA's two-stage recipe (projector alignment on frozen models, then instruction tuning) as the canonical cheap-VLM template.
- Why VLMs hallucinate: LLM prior overriding weak/low-res visual signal; fixes at data (grounding), architecture (resolution), and inference (verification) levels.
- CLIP's known failure modes: spatial relations, counting, negation, compositionality — and that these motivate native multimodal pretraining.

IMPROVEMENTS:
- Update the VLM comparison table beyond 2023: add at least Qwen2-VL (dynamic resolution, M-RoPE), Chameleon (unified early fusion), and one 2025 model; the table currently ends at 'GPT-4V 2023 Unknown'.
- Fix the LLaVA row: original LLaVA used CLIP ViT-L/14 @ 224px = 256 visual tokens with Vicuna (not raw LLaMA); 576 tokens is LLaVA-1.5 @ 336px. As written, the table contradicts the 224px math taught in Q5.
- Q1's moderation latency numbers ('~20ms each on GPU') deserve a hardware anchor (which GPU, what batch size) — as-is they are unfalsifiable and an interviewer would push.
- Add one paragraph on why contrastive vision encoders (CLIP) are being replaced/augmented by SigLIP2/AIMv2-style encoders in 2025 VLM stacks, so the 'CLIP is the visual backbone' claim is dated correctly.
- In Q7, add the KV-cache interaction explicitly: 576 visual tokens per image also inflate the KV cache for every decode step in multi-turn chat — the follow-up asks this but the strong answer never covers it.


# sections/22_inference_optimization.tex (Inference Optimization and LLM Serving)

VERDICT: The strongest of the three chapters — real roofline math, correct KV-cache arithmetic, and an excellent debugging-question suite — but it has three critical 2026 gaps (MLA, FlashAttention/kernels, MoE serving) plus no treatment of reasoning-model workloads, and a couple of internally inconsistent communication-cost numbers an expert interviewer would catch.

STAFF-LEVEL: This chapter mostly hits staff level: it consistently reasons from hardware first principles with real numbers (roofline crossovers, KV sizing, Little's-law capacity planning), and its debugging questions test 'have you operated this' rather than definitions. The gap to a true 2026 L7 bar is topical, not stylistic: a staff candidate will be probed on MLA, MoE inference economics, FlashAttention's IO argument, prefix-cache-aware routing, and reasoning-model workload shifts, none of which the chapter covers. Secondary issue: a few communication-cost numbers are internally inconsistent, and at staff level interviewers specifically test whether your numbers compose — fix those before a candidate internalizes them.

MISSING (critical/high):
- [critical] Multi-head Latent Attention (MLA, DeepSeek-V2/V3): low-rank KV compression into a latent vector, ~an order-of-magnitude KV reduction beyond GQA, decoupled RoPE key :: MLA is the single biggest KV-cache innovation since GQA and powers the most-discussed open models of 2025 (DeepSeek-V3/R1); a 2026 interviewer asking 'how do you shrink KV cache' expects GQA -> MLA as the progression. Its total absence dates the chapter immediately.
- [critical] FlashAttention / Flash-Decoding and kernel-level optimization (fused kernels, tiling, IO-awareness, CUDA graphs for decode launch overhead) :: FlashAttention appears only as two words in the TGI bullet. It is the default attention implementation in every serving stack and 'why is FlashAttention faster — it does MORE FLOPs but less HBM IO' is one of the most-asked inference interview questions in existence.
- [high] MoE serving: expert parallelism, all-to-all routing cost, active-vs-total-parameter economics (why a 671B/37B-active model serves like a ~37B compute model but needs 671B of memory), expert load balancing at inference :: Mixtral/DeepSeek-V3-class MoE is the dominant frontier architecture in 2025-2026; the chapter relegates expert parallelism to a single follow-up line. Sizing an MoE deployment is now a standard L6/L7 design prompt.
- [high] Serving reasoning models / test-time compute: 10K-100K-token chains of thought shift workloads decode-heavy, blow up KV growth per request, change batch dynamics and cost-per-query; streaming/hiding thinking tokens; early termination :: o1/R1-style inference is the defining 2025-2026 serving workload change; every capacity-planning question in this chapter assumes 200-500-token outputs, which is exactly the assumption reasoning models break.
- [high] SGLang and RadixAttention (automatic prefix caching via radix tree), and a refresh of the framework list (TGI is fading; TensorRT-LLM/Dynamo evolution) :: SGLang is a top-2 open serving engine by 2025 and prefix caching gets one passing mention despite being a first-order cost lever (multi-turn chat, agents, shared system prompts).

CORRECTNESS:
- Trade-off Q2 (TP vs PP), 'Quantitative intuition': claims a 32 KB cross-node all-reduce over InfiniBand takes '~0.6 us per layer... ~50 us over 80 layers---still small.' This is a bandwidth-only calculation that ignores per-collective latency (~10-20 us each) and the 2 all-reduces/layer, so the real cost is ~2-4 ms per token — which is precisely why cross-node TP is avoided. As written, the math contradicts the section's own recommendation and an expert interviewer would flag it.
- Estimation Q1 Step 2: 'all-reduce overhead (~0.5 ms per layer with NVLink)' is inconsistent with the concluded 6-7 ms total latency (would be 40+ ms over 80 layers); off by ~1-2 orders of magnitude vs the <0.1 ms figure quoted in Trade-off Q2.
- Serving-design Q1: '400 concurrent requests per node' KV headroom math uses 1.3 GB/request at 4K FP16 KV, but the same answer specifies AWQ INT4 weights and later relies on ~21K tok/s node throughput at batch ~128 — the 400-concurrent and 40 tok/s/request figures do not simultaneously hold at that batch size; tighten so the numbers compose.
- Keyinsight 'Decode Is Memory-Bandwidth-Bound' uses a 70B FP16 model on a single A100 (140 GB weights vs 80 GB HBM) as its example without noting it would not fit; fine as an intuition pump but should say 'per aggregate GPU memory' to survive a pedantic interviewer.

STALENESS:
- No MLA anywhere — the KV-reduction story ends at GQA/MQA (2023 state of the art).
- Framework list (vLLM, TensorRT-LLM, TGI, llama.cpp) is the 2023-2024 lineup; SGLang absent, TGI overweighted for 2026.
- All hardware numbers are A100/H100; no H200/Blackwell, no FP4, and 'FP8 on H100+' as the frontier of low-precision serving is a 2024 statement.
- Speculative decoding coverage predates EAGLE-2/3 becoming the deployed default; Medusa is presented as the main self-drafting alternative.
- Workload assumptions (200-500 output tokens) predate reasoning models; nothing on long-CoT decode-dominated serving, which is the marquee 2026 workload.

MUST-KNOW:
- KV cache memory formula (2 x layers x kv_heads x d_head x seq x bytes) and the LLaMA-70B worked example, computable on a whiteboard.
- Prefill is compute-bound, decode is memory-bandwidth-bound; TTFT vs ITL as the corresponding user-facing metrics.
- The decode roofline argument: token latency ~= bytes-of-weights-read / HBM bandwidth; why quantization and batching are the two levers that attack it.
- PagedAttention (paged, non-contiguous KV with copy-on-write sharing) and continuous/iteration-level batching, plus chunked prefill for TTFT protection.
- GQA/MQA (and by 2026, MLA) as the KV-reduction hierarchy, with the compression ratios.
- Speculative decoding: draft-verify mechanics, exact-distribution guarantee, acceptance-rate/batch-size preconditions, and when it makes things slower.
- TP vs PP decision by interconnect: per-layer all-reduce needs NVLink; PP or DP across nodes; DP replicas beat PP when memory allows.
- Quantization arithmetic (params x bits / 8), that KV cache and weights are separate memory pools, and that quality loss is task-dependent (perplexity is a bad proxy).
- Tail-latency failure modes: prefill blocking, KV-pressure preemption, output-length variance.
- Disaggregated prefill/decode: why it exists (interference between bottleneck profiles) and the KV-transfer cost that gates it.

IMPROVEMENTS:
- Reconcile the all-reduce overhead numbers: Estimation Q1 says '~0.5 ms per layer with NVLink' yet concludes 6-7 ms total token latency (0.5 x 80 layers = 40 ms would dominate); Trade-off Q2 says '<0.1 ms'/'<1 ms'. Pick one consistent figure (~tens of microseconds/layer on NVLink at batch 1) and note there are TWO all-reduces per transformer layer (attention out-proj + MLP down-proj).
- In Estimation Q1, add one sentence that the 140-170 tok/s batch-1 figure is a roofline upper bound and production systems typically observe ~50-90 tok/s for 70B on 8xH100 — interviewers test whether candidates know real numbers vs theoretical.
- Add KV-cache reads to the decode bandwidth model in the main text (not just the L7 bullet): at 32K context x batch 64, KV reads rival weight reads and change the crossover math the chapter derives.
- Give prefix caching its own subsection with the hit-rate arithmetic (shared 2K system prompt across a batch saves 2K x per-token KV per request plus the prefill compute), since two questions rely on it.
- The serving-stack table should add an 'observability' layer (per-request tracing of queue/prefill/decode time) since debugging Q1's L7 rubric depends on it.


# sections/24_long_context_rag.tex (Long Context and RAG Systems)

VERDICT: The RAG half is practical, current-ish, and interview-ready (hybrid retrieval, parent-child chunking, RAGAS, strong debugging questions); the long-context half is frozen in 2023 (PI/NTK/YaRN with no native-long-context training story and no long-context evaluation), has zero interview questions of its own, and the flagship cost-comparison question contains a badly wrong prefill-latency claim.

STAFF-LEVEL: The RAG material sits comfortably at L6: it tests architecture judgment, has real numbers (index sizing, latency budgets, annualized cost), and its debugging questions are 'have you shipped this' quality. Two things hold it below the 2026 staff bar. First, the long-context half is L5 survey material — a table of 2023 methods with no questions, no eval methodology, no scale numbers (e.g., KV cost at 1M tokens, effective-vs-claimed context), and no awareness that frontier practice moved to native long-context training. Second, the chapter's frontier is 2023's RAG stack; a staff candidate in 2026 is expected to reason about GraphRAG vs vector RAG, contextual retrieval, agentic retrieval loops, and when cheap long context plus caching kills the RAG pipeline entirely — the chapter gestures at the last point but with broken latency numbers in its flagship estimation question.

MISSING (critical/high):
- [critical] Long-context evaluation: needle-in-a-haystack and its saturation, RULER, LongBench, and the claimed-vs-effective context length gap :: By 2026 'the model claims 1M context — how do you verify its EFFECTIVE context?' is a standard interview question; the chapter's eval section covers RAG only and never evaluates long context at all.
- [high] How long context is actually achieved in 2024-2026 models: continued pretraining with progressive length extension and long-data mixtures (Llama 3.1 128K recipe), LongRoPE, dual-chunk attention — i.e., extension as a training recipe, not an inference-time hack :: The chapter's method table is all 2023 inference-time tricks; frontier practice moved to native long-context training, and a staff candidate describing YaRN as the state of the art will sound dated.
- [high] GraphRAG (entity/community-summary graph construction, global vs local queries) :: Microsoft's GraphRAG (2024) made graph-based RAG a mainstream interview topic for 'summarize themes across the whole corpus' queries that vanilla top-k retrieval structurally cannot answer; absent entirely.
- [high] Contextual retrieval / late chunking (prepending LLM-generated chunk context before embedding — Anthropic 2024; embedding full documents then pooling per chunk) :: This is the highest-impact chunking advance since parent-child and directly upgrades the chapter's own chunking section; interviewers ask 'how do you fix chunks that lose their document context'.
- [high] Agentic RAG as a section (iterative retrieve-reason loops, retrieval as a tool call, self-RAG/CRAG-style reflection) and the 2026 'RAG vs agentic search over raw corpus' debate :: Agentic loops appear only inside two answers; by 2026 agentic retrieval is the default architecture for hard queries and 'when does an agent with grep/search beat a vector DB' is a live frontier-lab debate.

CORRECTNESS:
- Cost Q2 ('Estimate cost per query'): 'Latency: 128K prefill at ~1M tokens/sec (H100 optimized) ~= 128ms prefill' — off by more than an order of magnitude for any frontier-scale model (128K-token prefill compute alone for a 70B model on 8xH100 is seconds; real API TTFT at 128K is ~10-30s+). The downstream conclusion 'Latency: comparable' is therefore wrong: fast TTFT is one of RAG's main advantages, and the answer erases it.
- Same question: RAG 'LLM prefill (~3ms)' for 2,760 tokens is similarly optimistic (~100-300ms realistic); the per-query totals still favor RAG but the components would not survive an expert's scrutiny.
- Keyinsight 'The Retriever Is Usually the Bottleneck': the '~70% of the time' figure is an invented statistic stated as fact (and repeated in Q3). Reframe as a practitioner rule of thumb, not a measurement.
- 'Why two-stage retrieval?': describes bi-encoder retrieval as 'O(1) with ANN' — HNSW is O(log N); calling it O(1) is the kind of sloppiness a staff interviewer pokes at.
- ColBERT storage math (64KB for 512 tokens x 128 dims) silently assumes 1 byte/dim; at FP16 it is 128KB. State the precision assumption.
- 'Training on long data... Gemini 1.5 and GPT-4 Turbo... up to 1M-2M tokens' — GPT-4 Turbo was 128K; as written it attributes 1M-2M training lengths to both.

STALENESS:
- Context-extension table is entirely 2023 (PI, NTK, YaRN, Dynamic NTK); no LongRoPE, dual-chunk attention, or the now-standard continued-pretraining recipe; '4K or 8K trained' framing describes 2023 base models.
- No GraphRAG, contextual retrieval, late chunking, ColPali, Self-RAG/CRAG — i.e., none of the 2024-2025 RAG advances; the pipeline described is the 2023 canonical stack.
- API pricing anchored at $10/M input tokens 'as of 2024-2025' — materially high for 2026 frontier pricing, which changes the RAG-vs-long-context cost ratio the question is built around.
- 'Gemini 1.5 and GPT-4 Turbo... up to 1M-2M tokens' groups a 128K model with 1M-2M models; also cites 2023-2024 models as the long-context frontier.
- Embedding model list (text-embedding-3, BGE, E5, GTE) is fine but predates 2025 leaders (e.g., Gemini embedding, Qwen3-embedding, NV-Embed) — minor refresh.

MUST-KNOW:
- Why naive length extrapolation fails (OOD RoPE angles) and the PI vs NTK-aware vs YaRN distinction at intuition level (uniform vs frequency-dependent compression).
- The canonical RAG pipeline and why two-stage retrieval exists: fast imprecise bi-encoder recall, then precise expensive cross-encoder rerank.
- Hybrid retrieval: BM25 + dense with reciprocal rank fusion, and the exact-match failure mode of pure dense retrieval (IDs, error codes, citations).
- Chunking tradeoffs and the parent-child pattern (index small for precision, return parent for context).
- RAG evaluation decomposition: Recall@k for the retriever, RAGAS faithfulness/relevance/precision/recall end-to-end, and the retriever-first debugging protocol.
- Lost in the middle: U-shaped attention over context and its implication that reranker ordering matters for generation.
- The long-context vs RAG decision framework: corpus size, cost per query (input tokens dominate), freshness, attribution — and that prefix caching changes the multi-turn math.
- ColBERT late interaction (MaxSim) and its quality-vs-storage position between bi- and cross-encoders.
- Why 'LLM ignores the context' happens: parametric-knowledge override, and its fixes (grounding prompts, reordering, fine-tuning on context-faithful data).

IMPROVEMENTS:
- Add at least one interview question exercising the long-context half — e.g., 'Your 8K-trained model must serve 64K contexts next month; walk me through options, costs, and risks' (PI vs NTK vs YaRN vs continued pretraining vs RAG). The section exists but is never tested, and it IS asked.
- Add a long-context debugging question ('quality degrades beyond 32K — diagnose': position OOD vs lost-in-the-middle vs KV quantization error vs eval artifact).
- Fix the cost question's latency analysis (see correctness) and refresh pricing: state prices as illustrative variables, and add cached-input pricing (~0.1x) since the answer's own caching point depends on it.
- Quantify 'lost in the middle' with the actual finding (accuracy dropping up to ~20+ points when the answer is mid-context) rather than 'significantly more likely to be ignored'.
- In the ColBERT storage discussion, note ColBERTv2/PLAID residual compression (~10x smaller) so the 125x figure doesn't overstate the modern tradeoff.
- The enterprise question should include the vector-DB memory/replica cost estimate the L7 rubric requests (200 GB raw -> HNSW overhead -> RAM vs disk-backed tradeoff) in the strong answer body, not just the rubric.


# sections/17_production_systems.tex (Production ML Systems)

VERDICT: Strong classic-MLE production chapter with genuinely good debugging questions and the log-and-train pattern well taught, but it is written for the 2019-2023 feature-store/CTR world: there is essentially zero LLM-product operations content, and it contains a 5x sample-size arithmetic error that directly contradicts Chapter 18.

STAFF-LEVEL: Depth is right at L6 for classic ML systems: the interview questions carry real Staff signal (latency budgets, log-and-train, automated rollback, cost estimation with shown work). Where it reads L4/L5: the batch-vs-realtime and CI/CD comparison tables are definitional, and the body text lacks the numbers the questions have — logging volume, feature-store QPS-per-node, retrain cost. The bigger Staff-level failure is scope, not depth: a 2026 frontier-lab Staff loop will spend half its production round on LLM-product operations, which this chapter does not cover at all.

MISSING (critical/high):
- [critical] LLM production operations (LLMOps): eval suites as CI/CD deployment gates, prompt and model-version management, behavioral regression testing when swapping model versions, guardrail/safety-filter pipelines, semantic/prefix caching, per-token cost monitoring and routing :: At OpenAI/Anthropic-tier labs in 2026, 'production ML' predominantly means operating LLM products. The chapter's CI/CD section tests 'model beats baseline on held-out data' — for an LLM product the gate is an eval suite plus canary prompts plus safety regression, and a candidate who only knows PSI-on-tabular-features will fail the loop. This is the single largest gap in the chapter.
- [high] Graceful degradation and fallback architecture: fallback model tiers, feature-store-outage behavior (default values vs cached last-known-good vs degraded model), timeouts and hedged requests, load shedding :: 'What happens when the feature store goes down?' appears only as a follow-up, but it is a first-class Staff probe — designing the failure path is what L6 system design rounds actually grade.
- [high] Sample Ratio Mismatch (SRM): definition, chi-square check, common causes (bot filtering asymmetry, redirect losses, trigger-condition bugs) :: The strong answer name-drops 'monitor for sample ratio mismatch' but the chapter never defines it. Any experimentation-literate interviewer will follow up, and a candidate who memorized the phrase without the mechanism gets exposed.
- [high] Drift detection for embeddings/unstructured inputs: domain-classifier drift detection, MMD, PSI on embedding-space projections, monitoring LLM input distributions (topic mix, prompt-length drift) :: PSI/KS/KL are univariate-tabular methods; most 2026 systems have embedding or text inputs where they don't directly apply. 'How do you detect skew in embedding features?' is already a follow-up in the chapter without any supporting content.

CORRECTNESS:
- Section 'A/B Testing Basics', first bullet: claims ~1.6M samples per group for 2% CTR with 1% relative lift; correct value from the chapter's own formula is ~7.84M per group (5x error), and Chapter 18's estimation question computes 7.84M for the same scenario — direct internal contradiction
- Serving section: 'Continuous batching... can increase throughput by 2-4x compared to naive static batching' — conservative/miscalibrated baseline; vs truly naive static batching the gain is typically much larger (the 2-4x figure is vs already-optimized batching)
- Monitoring question strong answer: 'GPU utilization > 90%' as an alert threshold is misleading as written (high utilization is the goal of good batching; the pageable condition is saturation with rising queue latency)

STALENESS:
- No LLM-product production content at all (eval-gated deploys, prompt versioning, guardrails, cost-per-token monitoring, model-API dependency management) — the defining production-ML surface of 2026
- vLLM/paged attention (2023) is presented as the frontier of serving; missing prefix caching, chunked prefill, and any pointer to disaggregated prefill/decode which the book covers only in the inference chapter's L7 question
- Drift detection is univariate-tabular only (PSI/KS/KL); no embedding-based or LLM-input drift methods
- 'MapReduce-style' as the descriptor for batch inference reads 2015; say Spark/Ray batch jobs
- Tooling roster (MLflow, W&B, Great Expectations, TFDV) is fine but pre-2023; nothing on Ray Serve, KServe, or feature platforms (Tecton/Feast) that interviewers may name

MUST-KNOW:
- Training-serving skew: the five sources, and log-and-train as the gold-standard mitigation (with its storage cost)
- Point-in-time correctness / preventing label leakage in training-set construction
- A/B fundamentals: n ~ 16 p(1-p)/delta^2, power 80%, user-level randomization, 1-2 week minimum, and why peeking inflates false positives
- Sample Ratio Mismatch: what it is and that you check it before reading any result
- Data drift vs concept drift: definitions, detection (PSI vs delayed-label performance), and that concept drift is only detectable with label delay
- The deployment ladder: shadow -> canary -> gradual rollout, with automated rollback triggers
- The overnight-CTR-drop debugging order: system/deploy -> data pipeline -> external -> model, measurement artifacts first
- The five causes of offline-online divergence, especially position bias and feedback loops
- Latency budget decomposition for a two-stage retrieval+ranking system
- Continuous batching + paged KV cache: why LLM serving throughput depends on them

IMPROVEMENTS:
- Fix the sample-size number in Section 'A/B Testing Basics': for 2% CTR and 1% relative lift (delta=0.0002), n = 16(0.02)(0.98)/(0.0002)^2 = 7.84M per group, not '~1.6M'. Chapter 18's identical worked example correctly gets 7,840,000 — the two chapters currently contradict each other on the book's most quotable formula
- Promote the latency-budget breakdown out of the 10K-QPS strong answer into the body: a table decomposing a 200ms p99 budget (LB/network ~5ms, feature fetch ~10ms, retrieval ~10ms, rerank ~50ms, business logic ~10ms, headroom) is the single most reusable artifact for system-design rounds
- Update the continuous-batching numbers: '2-4x vs naive static batching' undersells it — vLLM-style continuous batching is 2-4x vs prior batched systems (Orca) and up to ~10-20x vs naive per-request serving; also add one line each on prefix/prompt caching and chunked prefill as 2024-25 production standard, with a cross-reference to the inference chapter
- In the monitoring question, reframe 'alert on GPU utilization > 90% sustained' — high sustained utilization is a capacity-planning signal, not an incident; alerting on it as written invites interviewer pushback
- Add an explicit 'measurement artifact first' step to the CTR-drop answer body (logging pipeline broken, metric definition change, dashboard timezone) — it is currently only in red flags and follow-ups
- Add a short subsection on retraining cadence economics: cost of daily full retrain vs hourly warm-start at ads scale, and what triggers justify off-schedule retrains


# sections/18_evaluation_metrics.tex (Evaluation Metrics)

VERDICT: The classical-metrics core (classification, ranking, calibration, significance, offline-online gap) is accurate and well-pitched, with only minor correctness nits — but LLM evaluation, the highest-value eval topic for a 2026 frontier-lab interview, is confined to one question box: no pass@k, no contamination, no eval-variance statistics, no reasoning-model evaluation.

STAFF-LEVEL: For classical metrics the depth is correctly Staff: the offline-online gap, calibration-harm examples, CUPED/sequential-testing name-drops, and the segmented-analysis insight are L6 signals, and the estimation question models real numeracy. Where it reads L4/L5: the multi-page n-gram-metrics coverage is definitional filler relative to its 2026 interview value. The Staff-level hole is LLM eval statistics and protocols — at the target labs the eval round is now mostly 'how do you know model B is better', asked about LLMs, and the chapter equips the candidate for that only via one (good) L7 question box.

MISSING (critical/high):
- [critical] Statistical rigor for LLM benchmark scores: standard error of an accuracy on an n-question eval (sqrt(p(1-p)/n)), paired-by-question model comparison, clustered standard errors when questions share sources (the approach in Anthropic's 2024 'statistical approach to model evals' work), and the resulting rule that a 2-point gap on a 1,000-question benchmark (~±2.8% CI) is often noise :: This is the bread-and-butter eval question at frontier labs in 2026 ('model B is 1.5 points better on MMLU — do you ship it?'), and the chapter's excellent significance machinery is never connected to LLM benchmarks. A Staff candidate at these labs who cannot put an error bar on an eval score fails the eval round.
- [high] pass@k: definition and the unbiased estimator (Chen et al. 2021, E[1 - C(n-c,k)/C(n,k)]), plus why naive k-sample estimation is biased :: pass@k is name-dropped once in the chatbot-eval answer but never defined; it is a standard formula question for any code-generation or agent role and one of the few 'write the formula' asks in modern eval interviews.
- [high] Benchmark contamination: detection (n-gram overlap against training data, canary strings, temporal holdouts like LiveBench-style rolling evals) and why it inflates reported scores :: Appears only as an L7 bullet ('contamination detection'); in 2026 it is a mainstream probe whenever a candidate cites a benchmark number.
- [high] Evaluating reasoning / test-time-compute models: maj@k / self-consistency, verifier-graded evaluation, budget-matched comparisons (accuracy at equal token/cost budgets), and why pass@1 at high sampling temperature is ill-defined without protocol details :: Reasoning models are the dominant 2025-26 model class; comparing a reasoning model to a standard model without cost-matching is exactly the kind of trap question frontier labs ask.

CORRECTNESS:
- ROC-AUC table row 'Range: 0.5 (random) to 1.0 (perfect)' — AUC ranges 0 to 1; below 0.5 means systematically inverted ranking, which is itself an interview probe ('what does AUC 0.3 tell you?'). State the full range
- BLEU guidance 'above 30 understandable, above 50 high quality' presented without the tokenization caveat — these thresholds are not comparable across tokenizers/languages; an expert interviewer would push back
- Calibration question: 'ECE of 0.15 means predicted probabilities are off by 15 percentage points on average' — ECE is a bin-weighted average of |acc-conf| gaps; the paraphrase is acceptable shorthand but can mislead (a model can have large offsetting within-bin errors); one qualifying clause would fix it
- Cross-chapter contradiction: this chapter's sample-size question correctly derives 7.84M per group for 2% CTR/1% relative lift while Chapter 17 states 1.6M for the identical setup — Chapter 17 is wrong, but the inconsistency damages trust in both

STALENESS:
- Generation-metrics section centers on BLEU/ROUGE/METEOR/ChrF/BERTScore (2002-2019); in 2026 interviews these are one sentence of context, while LLM-judge protocols, functional-correctness metrics, and eval statistics — the actual current practice — get a fraction of the space
- 'Elo ratings... used by Chatbot Arena' — Arena's methodology moved to Bradley-Terry with CIs and style control; citing raw Elo dates the book
- No mention of any post-2021 benchmark by name (MMLU-Pro, GPQA, SWE-bench, AIME) even as examples in the chatbot-eval question
- No reasoning-model or test-time-compute evaluation content anywhere

MUST-KNOW:
- Precision/recall/F1 from the confusion matrix, and the one-sentence rule for when each dominates
- ROC-AUC vs PR-AUC under class imbalance, with the FPR-denominator explanation of why ROC flatters
- Discrimination vs calibration; ECE; temperature scaling as the default fix that preserves AUC
- NDCG mechanics (gain, log discount, IDCG normalization) plus Recall@K for the retrieval stage
- Why accuracy fails (imbalance, error-type blindness) and when it is actually fine
- The five causes of offline-up/online-down, led by position bias
- Power analysis: n ~ 16 p(1-p)/delta^2, relative-vs-absolute lift disambiguation, peeking
- Multiple comparisons: Bonferroni vs Benjamini-Hochberg, and pre-registration
- LLM-as-judge biases (verbosity, position, self-enhancement) and their mitigations
- Putting an error bar on a benchmark score: sqrt(p(1-p)/n) and paired comparison

IMPROVEMENTS:
- Add a worked eval-noise example to the significance section: 1,000-question benchmark at 70% accuracy has SE ~1.45%, so 95% CI ~±2.8 points; then show the paired comparison shrinking the CI — this connects the chapter's existing machinery to the modern use case in half a page
- Update the human-eval subsection: Chatbot Arena has used Bradley-Terry MLE with confidence intervals (not raw Elo) since 2023-24, plus style/length controls; also mention length-controlled win rates (AlpacaEval 2.0) as the fix for verbosity bias
- Extend the metric-selection decision tree's generation branch: it currently terminates at BLEU/ROUGE vs Perplexity/Human — add an LLM-as-judge leaf, which the chapter itself says is 'increasingly common'
- Pin down or delete 'modern frontier LLMs achieve perplexity below 10 on standard benchmarks' — without naming corpus and tokenizer this is unfalsifiable and contradicts the chapter's own caveat that PPL is tokenizer-dependent
- Note sacreBLEU as the standardization fix when giving BLEU rules of thumb (the 30/50 thresholds are tokenization-dependent folklore)
- Add a row to the cheat-sheet table for 'LLM assistant / agent' -> pairwise win rate (human or judge) + task success rate + safety regression suite


# sections/19_decision_frameworks.tex (Decision Frameworks)

VERDICT: The interview questions are the best part (moderation cascade, latency decision tree, fine-tune-vs-custom, build-vs-buy all carry real Staff signal), but the reference tables are frozen in 2021-2022 BERT-era NLP, the Chinchilla key insight states the scaling relationship exactly backwards, and the compute-budget table is internally inconsistent — the highest concentration of correctness problems in the four files.

STAFF-LEVEL: Split personality: the interview-question boxes are solidly L6 with credible L7 tiers (phased decision gates, policy-model alignment, SLA renegotiation), but the reference tables and decision trees the chapter is named for are L4/L5 lookup material — definitional, dated, and in the Chinchilla case wrong. Scale intuition exists in the questions but not the frameworks: the chapter never gives the one number set that drives 2026 architecture decisions (cost per 1M tokens, API vs self-hosted crossover, reasoning-model cost multipliers). A Staff candidate who studies the questions will do well; one who memorizes the tables will sound five years out of date.

MISSING (critical/high):
- [critical] Prompt vs fine-tune vs train-from-scratch as the step-0 decision framework: when zero/few-shot with a frontier model suffices, when to add RAG, when to fine-tune (LoRA/full), when only a custom model works — as a first-class decision tree preceding the architecture tables :: This is THE architecture-selection question of 2026; the chapter's own fine-tune-vs-custom question answers it well, but the reference tables the reader will memorize still start from 'BERT vs T5 vs GPT', which is the wrong first question at a frontier lab.
- [critical] API model vs open-weights self-hosting decision, with the cost-crossover estimate: $/1M-token API pricing vs self-hosted throughput economics (e.g., 70B on 8xH100 at ~$16-25/hr serving X tok/s), plus data-governance, latency-tail, rate-limit, and capability-ceiling factors :: Every 2026 ML org makes this call and Staff interviews ask candidates to defend it with numbers ('at what volume does self-hosting win?'); the build-vs-buy question covers infra tooling but never the model itself.
- [high] Reasoning model (test-time compute) vs standard model routing: when o-series/R1-style deliberation is worth 10-50x token cost and added latency, cost-quality-latency triangle, and hybrid routing (cheap model + escalation) :: Reasoning models are a defining 2025-26 development and a natural fit for a 'when to use what' chapter; their complete absence is the chapter's clearest currency failure.
- [high] Inference-aware (post-Chinchilla) scaling: compute-optimal is not deployment-optimal — over-training small models far past 20 tokens/param (Llama-3 8B at ~15T tokens ~1,900 tok/param) because inference cost dominates at deployment :: Pairs with fixing the Chinchilla error; a Staff candidate who recites 20 tokens/param as the sizing rule without the inference-cost caveat is giving a 2022 answer and interviewers will push.
- [high] Modernized vision/NLP defaults in the selection tables: DeBERTa-v3/ModernBERT as encoder default, SAM/SAM-2 for segmentation, open-vocabulary detection (Grounding DINO/OWL-ViT), 'use a VLM zero-shot' as an option before training any vision model :: Interviewers probe 'would you train a detector or prompt a VLM/SAM pipeline?'; the tables' YOLO/Faster R-CNN/U-Net roster gives no way to answer.

CORRECTNESS:
- Chinchilla keyinsight (Sec. 'Model Size vs. Data Size'): 'N ~ 20 x D where N is parameters and D is tokens' — relationship inverted; should be D ~ 20N. This is a fumble-the-interview error if memorized as printed
- Compute Budget Guidelines table: FLOPs column inconsistent with its own params/tokens columns by ~5x under C=6ND (e.g., 40B params/800B tokens is ~1.9e23 FLOPs, listed under 1e24 and labeled 'Chinchilla', whose actual budget was ~5.8e23 for 70B/1.4T); also '4B = GPT-2 XL scale' (GPT-2 XL is 1.5B)
- Retrieval architecture: figure threshold '<1K items' for cross-encoder-only contradicts the table's '<10K items' two sections earlier
- Accuracy-latency Pareto figure: invented data points presented as 'typical' without a task or source — an interviewer-facing book should not teach candidates to quote these
- Latency techniques table: 'INT8 quantization 2-4x speedup, <1% loss' stated unconditionally — regime-dependent (see improvements); similarly 'Distillation 2-10x speedup, 1-5% loss' is fine as a range but the two tables in this chapter give different distillation numbers (2-10x vs 5-20x in Ch17's table) — harmonize

STALENESS:
- Architecture tables are 2021-22 vintage: BERT/RoBERTa/DeBERTa, T5/BART, GPT/LLaMA, YOLO/Faster R-CNN, SASRec/BERT4Rec — no reasoning models, no VLM/SAM options, no LLM-first decision path in the tables themselves
- Chinchilla presented as the final word on sizing; no inference-aware over-training (the actual 2023-26 practice), no mention that frontier sizing now weighs test-time compute
- Accuracy-latency Pareto figure built entirely from 2019-2020 encoder models
- 'GPT-4 scale (estimated) 400B/8T' — speculative 2023-era estimate presented in a reference table
- Loss table's ranking/implicit-feedback defaults (BPR, LambdaRank/ListNet) reflect pre-LLM recsys canon without noting current in-batch-softmax practice
- GPU/TPU answer cites A10G-era framing; fine, but no mention of the H100/H200/Trillium generation or FP8 serving that 2026 interviewers assume

MUST-KNOW:
- Chinchilla direction stated correctly — ~20 tokens per parameter — plus why 2026 practice over-trains past compute-optimal (inference cost dominance)
- GBM vs NN for tabular: the empirical GBM default and the specific conditions where NNs win
- Two-stage retrieval scale thresholds and per-stage metrics (Recall@K retrieval, NDCG rerank) with latency targets
- The latency-optimization ladder: profile -> quantization/graph opt -> distillation/pruning -> architectural change, with rough speedup/effort numbers
- The LLM-baseline-first strategy: prompt/fine-tune a foundation model in weeks, distill to a small model only when serving economics demand it
- Encoder vs decoder task matching (classification -> encoder; generation -> decoder) and when zero-shot LLM overrides the encoder default
- Start-simple escalation ladder (heuristic -> GBM/pretrained -> custom) and being able to defend stopping early
- Build vs buy vs open-source: the differentiation test, vendor lock-in, and TCO including engineering time
- Loss selection map: focal for imbalance, InfoNCE for in-batch negatives, Huber for outliers, Dice+CE for segmentation

IMPROVEMENTS:
- Fix the Chinchilla box (Section 'Model Size vs. Data Size'): it states N ~ 20 x D with N params and D tokens — inverted. Correct: D ~ 20 x N (20 tokens per parameter; Chinchilla 70B / 1.4T). As written it implies a 40B-param model should train on 2B tokens
- Recompute the compute-budget table with C = 6ND: e.g., 40M params x 0.8B tokens = 1.9e17 FLOPs, not 1e18; every row's FLOPs column is ~5x too high relative to its own params/tokens columns; also '4B params = GPT-2 XL scale' mislabels GPT-2 XL (1.5B)
- Reconcile the cross-encoder corpus thresholds: the retrieval figure says cross-encoder below 1K items, the architecture table says below 10K — pick one (the figure's <1K-to-low-thousands is closer to practice) and state the latency math that produces it
- Replace or clearly label the accuracy-latency Pareto figure: the plotted points (DistilBERT 88% -> Ensemble 99%) and the 100-50/x curve are fabricated 'typical' numbers on a dated model lineup; either use real numbers from a named benchmark or mark it schematic
- Qualify 'INT8 quantization: 2-4x speedup' by regime: true for compute-bound CNN/encoder inference via TensorRT; for memory-bandwidth-bound LLM decoding, weight-only INT8 yields closer to 1.5-2x — the distinction is itself interview material
- Rewrite the NLP decision tree with an LLM-first spine: (1) can a prompted frontier model hit quality/latency/cost targets? (2) if latency/cost fails, distill or fine-tune small; (3) encoder models for high-QPS classification; keep BERT+CRF only as a legacy note
- Update 'In 2024+, the line is blurring' in the BERT-vs-GPT answer to a 2026-current statement, and add the 'prompt an API model as the no-training baseline, then distill' path explicitly


# sections/appendix_question_index.tex (Question Index)

VERDICT: A genuinely useful navigation layer with a well-chosen cram list and a sensible type taxonomy, but the bookkeeping is broken — claimed counts contradict enumerated rows in at least five places, one question is listed twice, and several lists have awkward 'plus N more' overflow paragraphs — and the index confirms the book-wide currency gap: none of the 248 questions touches reasoning models, GRPO, or test-time compute.

STAFF-LEVEL: As a navigation artifact its level-calibration is right — L6-majority, estimation-heavy, debugging-rich — which is precisely a Staff loop's shape. Its weaknesses are executional (count drift, duplicates, overflow hacks) rather than conceptual, but those errors are visible to any careful reader and undermine trust in a book that elsewhere preaches 'reproducibility and validation gates'. The one substantive Staff-level gap it exposes: the cram list contains no debugging question and the index contains no 2025-26-era question, so a candidate who optimizes against this index alone will be current through roughly mid-2024.

MISSING (critical/high):
- [high] Debugging questions in the 2-hour cram list: currently all 20 are conceptual/trade-off/estimation — add at least 'CTR dropped 5% overnight' and 'training loss NaN'  :: Debugging rounds are among the most common Staff screens; a candidate who crams only this list walks in with zero rehearsed debugging drills despite the book having 38+ of them.
- [high] Chapter 18 (Evaluation) in the 'LLM / Foundation Models' company-focus row :: Frontier-lab loops are eval-heavy (eval design and eval statistics rounds are standard at OpenAI/Anthropic-tier); the LLM row currently routes candidates to seven chapters but omits evaluation entirely.
- [high] Any question anywhere in the 248 covering reasoning models / test-time compute / GRPO / o-series-vs-standard trade-offs (index-level observation; the content gap belongs to other chapters) :: The index makes the book-wide staleness auditable: 'PPO vs DPO vs RLHF' exists but nothing on GRPO or inference-time scaling, which are top-frequency 2026 asks; when those questions are added to chapters, the index and cram list must pick them up.

CORRECTNESS:
- Total question count contradicts itself: 248 (TLDR, summary table, frequency sum 118+99+31) vs 249 (by-topic table total row; also 89+127+33 actually-listed difficulty rows)
- L5 table contains the speculative-decoding question twice (rows 83 and 89), inflating the L5 count
- Debugging section prose 'Plus 3 more:' enumerates four questions; Estimation section claims 28 in both header and summary but effectively lists 31 with the appended L7 items; Conceptual (52 claimed / 49 listed) and First-Principles (23 claimed / 19 listed) tables are short of their claimed counts
- L7 subsection header '32 questions' vs 33 enumerated including the bolt-on table

STALENESS:
- '2024-2025 interviews' framing in a book being used in 2026
- The index reveals zero questions on reasoning models, test-time compute, GRPO, MLA/DeepSeek-style architecture changes, or agent evaluation across all 248 entries — the cram list's 'GPT-2 to LLaMA (2023)' is the most recent architecture-evolution question offered
- Company-focus table reflects a 2022-era employer landscape (no frontier-lab row, no applied-LLM/agents role archetype)

MUST-KNOW:
- The 2-hour cram list's core dozen: Flash Attention, MHA/GQA/MQA, 1/sqrt(d_k), GPT-2->LLaMA changes, RLHF pipeline, LoRA-family decision framework, 7B training-memory estimate, KV-cache-at-128K estimate, parallelism decision framework, PPO-vs-DPO, training-serving skew, 70B serving-stack design
- The estimation-question muscle: being able to do any of the 28+ back-of-envelope items (memory, FLOPs, QPS, cost) cold, since estimation is the least fakeable round
- At least two rehearsed debugging drills end-to-end (overnight CTR drop; NaN loss) with the triage order verbalized
- Knowing your loop's shape: which 3-4 chapters your target team's row prioritizes, per the company-focus table

IMPROVEMENTS:
- Regenerate every count programmatically from the interviewq boxes: TLDR/summary claim 248 while the by-topic table totals and prints 249; L7 section header says 32 but 33 are listed (32 rows plus the bolt-on 'question 33' table); Conceptual claims 52 but enumerates 49; First-Principles claims 23 but enumerates 19; Estimation claims 28 then appends 'plus 3 more L7 estimation questions' (=31); Debugging claims 38, lists 38, then says 'Plus 3 more' and names four
- De-duplicate the L5 table: rows 83 ('Explain speculative decoding. When use it and when not?') and 89 ('Speculative decoding: when use and when not?') are the same question listed twice
- Fold the overflow items ('Plus 3 more...' paragraphs and the standalone one-row L7 table for the RAG-reasoning question) into their parent longtables — the current bolt-ons look like generation artifacts and break the counts
- Update 'The most heavily tested topic area in 2024-2025 interviews' to 2026 framing, and refresh the Ads row's 'Meta, Google, Amazon, TikTok' to include the frontier labs the book targets
- Rebalance flagged by the index itself: Production Systems and Evaluation get 10 questions each vs 18 for Attention — for a Staff MLE loop, production+eval carry at least as much weight; commission 3-5 more questions each (logging-cost estimation, SRM debugging, eval-variance estimation, LLM regression-gate design)
- Add a 'Behavioral/experience' pointer or note that this book intentionally excludes the behavioral half of Staff loops, so cram-list users are not surprised


