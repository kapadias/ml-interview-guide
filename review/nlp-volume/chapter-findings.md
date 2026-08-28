# 01_classical_foundations.tex

VERDICT: Solid L5 breadth chapter with the right 'know when simple wins' framing, and the only place in either volume where TF-IDF/BM25 are actually derived — but it is overweight for 2026 breadth-insurance material, wholesale-duplicates Chapter 2's n-gram-LM and perplexity sections (down to colliding LaTeX labels), and misses the single best 2026 argument for its own existence: classical methods (fastText filters, KenLM perplexity, MinHash/n-gram dedup) running inside frontier-lab pretraining data pipelines.

STAFF: Reads L5 overall, with the BM25 treatment and the multilingual pipeline design question (Q7) reaching L6. What keeps it below staff: no scale numbers anywhere (index sizes, QPS, latency ladders for BM25 vs ANN, cost per query), no production mechanics (index-time vs query-time analysis appears only in an L7 ladder bullet), and the L7 ladder bullets name things (SPLADE, BM25F, learning-to-rank) the body never teaches — a candidate who studies only this chapter cannot deliver its own L7 answers. The judgment content (warning boxes on lowercasing, stopwords, 'it depends, measure it') is genuinely good and the right voice for staff interviews.

OVERLAP [partial] with 06_embeddings_representations, 07_similarity_metric_learning, 18_evaluation_metrics, 24_long_context_rag
STRONGER: NLP ch1 is the only place in either volume that derives TF-IDF/BM25 — DL 06/07/24 use BM25 heavily (hard-negative mining, hybrid retrieval stages, RRF) but always assume it, never teach it, so ch1 is the load-bearing prerequisite. For perplexity, DL 18's compact catalog entry is appropriately shallow; the real derivation belongs in NLP ch2, not ch1. The worse duplication is intra-NLP-volume: ch1 re-teaches ch2's n-gram/perplexity material, ch3's one-hot, and territory ch9 (BM25 usage, which correctly cross-refs ch1) and ch10 (perplexity) also cover.
RECOMMENDATION: Keep both volumes' current division for retrieval: ch1 stays canonical home for the TF-IDF/BM25 derivations; DL 07/24 keep their usage-level mentions (they already cross-assume correctly, no change needed). Fix the real problem: delete ch1's n-gram-LM + perplexity section (canonical home: NLP ch2 for derivation, NLP ch10 + DL 18 for the metric-catalog view), reduce one-hot to a ch3 cross-ref, and resolve the duplicate LaTeX labels. This recovers ~6–8 pages for the missing high-priority content.

MISSING (crit/high):
- [high] Inverted index mechanics behind BM25: postings lists, index-time vs query-time stemming, DAAT/TAAT evaluation, WAND/block-max WAND pruning :: The chapter derives BM25 and claims 'millions of documents in milliseconds' but never explains how; staff-level retrieval loops (and any search-infra role) probe exactly this, and ch9's two-sentence i
- [high] Classical NLP inside the 2026 LLM data pipeline: fastText/linear quality classifiers at trillion-token scale (DCLM, FineWeb-Edu), KenLM perplexity filtering (CCNet), MinHash/Jaccard dedup over n-gram shingles :: This is the modern, evidence-backed answer to the chapter's own thesis ('classical still matters'); frontier-lab interviewers ask 'how would you filter/dedup a 10T-token corpus' and the answer is this

CORRECTNESS:
- Line ~790: 'The Penn Treebank tagset defines 36 POS tags' — the standard citation (Marcus et al.; Jurafsky & Martin) is 45 tags (36 word classes + 9 punctuation/symbol). As written it will collide with interviewer flashcards; say '45 (36 word tags + punctuation)'.
- Lines 1171–1176: 'PP = exp(H) where H is the cross-entropy' is only true when H is in nats; ch2 defines H in bits with PP = 2^H. The base convention is inconsistent across chapters — state it explicitly.
- Eq. (bm25_idf): the presented BM25 IDF goes negative when df(q) > N/2; the text gives the formula with no caveat. This is a known gotcha (Lucene adds 1 inside the log precisely to avoid it) — the book should flag it before an interviewer does.
- Duplicate LaTeX labels sec:ngram_lm and sec:perplexity are defined in both ch1 and ch2 — multiply-defined labels; cross-references will silently resolve to the wrong chapter.
- KeyInsight lines 1247–1254 and Q10: 'classical pipelines can outperform fine-tuned BERT when training data < 100 examples' — contestable in 2026: SetFit-style contrastive fine-tuning and LLM few-shot generally dominate at that size. The latency/interpretability arguments hold; the small-data argument should be softened or evidence-qualified.

STALENESS:
- Dependency parse figure and text use pre-UD Stanford labels (prep/pobj); Universal Dependencies (case/obl) has been the community standard since ~2016 and is what a 2026 interviewer expects.
- 'BIOES improves NER F1 by 0.5–1.5% over BIO' is a CRF-era (Ratinov & Roth 2009) result; with modern transformer taggers the difference is usually noise — the unconditional claim needs qualification.
- POS 'current state-of-the-art ~97.5%' is numerically fine but should note POS tagging is a solved legacy benchmark, no longer an active leaderboard.
- The paradigm-shift narrative stops at 'RAG, Agents'; one sentence acknowledging the 2024+ reasoning-model/RLVR phase would keep the arc current (deep coverage belongs to ch7/8).
- The 'classical beats neural' examples (spam filters, Elasticsearch) are fine but pre-2020 flavored; the strongest 2026 examples are pretraining-pipeline filters (see missing topics).

MUST-KNOW:
- TF-IDF: derive tf, idf, sublinear variant; why locally-frequent-globally-rare is the discriminative signal
- BM25: full formula, roles of k1 (saturation) and b (length normalization), typical values (1.2, 0.75), asymptote at k1+1
- Why hybrid retrieval (BM25 + dense with RRF) beats either alone; vocabulary mismatch as the core TF-IDF/BM25 failure
- Stemming vs lemmatization: mechanism, outputs, when the choice matters, why subword tokenizers make both optional
- BIO/BIOES schemes and why independent per-token softmax produces invalid sequences (motivating CRF)
- Tokenization failure modes: contractions, CJK segmentation, agglutinative languages, and why this motivates subword methods
- When classical beats neural: latency (<1ms), interpretability, no-training-data regimes — with honest limits
- Bag-of-words limitations arc: word order → n-grams, frequency bias → TF-IDF, semantics → embeddings, context → transformers
- NLP task taxonomy (classification / sequence labeling / seq2seq / structured prediction) and what each implies for architecture and metric
- Preprocessing judgment: when lowercasing/stopword removal helps vs destroys signal ('it depends, measure it')

IMPROVEMENTS:
- Delete the entire 'N-Gram Language Models' section (lines 1082–1191, including the smoothing and perplexity subsections and interview Q9) and replace with a one-paragraph forward reference: it duplicates ch2 §§1–2 nearly one-to-one (chain rule, Markov, MLE, Laplace/add-k/interpolation/Kneser-Ney, perplexity) and even redefines ch2's LaTeX labels. Saves ~6 pages and eliminates drift risk.
- Fix the multiply-defined LaTeX labels: sec:ngram_lm (ch1:1084 vs ch2:31) and sec:perplexity (ch1:1163 vs ch2:247) — in the combined build \ref targets are ambiguous/wrong.
- Trim the one-hot section (sec:one_hot) to a cross-reference: ch3 re-teaches one-hot with the identical 'no similarity / sparsity' flaws.
- Add a 'classical NLP in the LLM stack' keyinsight box (fastText pretraining filters, KenLM, MinHash dedup, n-gram decontamination) — converts the chapter's defensive 'BM25 is still used at Google' claims into 2026-specific evidence a staff candidate can actually cite.
- Add a 3-line worked BM25 scoring example (two docs, one query, show saturation and length normalization numerically) — every formula-bearing section in ch2 has a worked example; the chapter's flagship formula has none.
- Cut the repeated interviewer-psychology filler ('LLM tourist', 'FAANG interviewers test X because...') — it appears in nearly every section; once per chapter suffices and the pages are better spent on the missing content above.
- Shrink parsing to 2–3 pages: keep constituency-vs-dependency contrast, complexity table, and 'why syntax still matters'; compress the CKY/Eisner/shift-reduce detail, which is rarely probed in 2026 (see ROI).
- Promote a second staff design question alongside the (strong) multilingual pipeline Q7 — e.g., 'design search over a 100M-product catalog: analyzer chain, stemming at index vs query time, CJK analyzers, BM25+dense hybrid with RRF' — this is where BM25 knowledge actually gets tested at L6+.

# 02_statistical_nlp.tex

VERDICT: The most mathematically rigorous chapter of the two — the HMM/Viterbi worked example, CRF partition-function treatment, and Kneser-Ney derivation are genuinely interview-grade and unique across both volumes — but it never teaches the label-bias problem its own follow-ups ask about twice, omits the backward algorithm its Baum-Welch section depends on, carries a wrong GPT-2 perplexity number, and over-invests in 2026-dead material (Baum-Welch detail, IBM Models 3–5) while missing the modern payoffs (constrained decoding, infini-gram, KenLM filtering) that would justify its page-weight to a frontier-lab candidate.

STAFF: The mathematics is genuine L5–L6: complete derivations, a correct worked Viterbi example (arithmetic verified), correct CRF gradient and LDA collapsed-Gibbs forms, and well-constructed answer ladders. The staff gap is judgment and currency, not rigor: no honest 2026 ask-frequency calibration (it claims classical structured prediction is asked far more than frontier loops actually do), no scale anchors (the Google 2T-token 5-gram story that motivates stupid backoff is absent even though the paper is cited), no modern-transfer map (which DP skills resurface in CTC, beam search, constrained decoding), and it cannot answer its own L6 follow-up on label bias. Fixing those four items would make this the strongest chapter in the volume.

OVERLAP [partial] with 18_evaluation_metrics, 09_nlp_architectures
STRONGER: For its core material (HMM, Viterbi, forward-backward, CRF, LDA, SMT) there is no overlap at all — the DL volume contains only a 'BERT + CRF' row in DL 09/19 tables with zero explanation, so ch2 is the unique home and load-bearing for the DL volume's own NER recommendation. The only real overlap is perplexity: DL 18 has a compact subsection (formula, intuition, tokenizer caveat) that is correct but derivation-free; ch2's treatment (cross-entropy derivation, worked magnitudes, five limitations) is clearly stronger.
RECOMMENDATION: Keep both for perplexity with roles made explicit: ch2 is the canonical derivation, DL 18 keeps its short metric-catalog entry (they are already consistent, so no merge needed). The duplication that actually needs fixing is inside the NLP volume — ch1's n-gram/perplexity section duplicates ch2 nearly verbatim (including LaTeX labels) and ch10 re-covers perplexity a third time; consolidate to ch2-derivation + ch10-metric-context and delete ch1's copy. No changes needed on the DL side.

MISSING (crit/high):
- [critical] The label bias problem and MEMMs as an actual body section (local vs global normalization, why CRFs fix it) :: It is the canonical L6 CRF probe; the chapter references it four times (two follow-ups, two ladders, one answer paragraph) but never explains it — the book cannot currently answer its own follow-up qu
- [high] The backward algorithm (β recursion) and forward-backward marginals :: The Baum-Welch E-step (lines 570–577) and the CRF gradient (line 809) both invoke β and forward-backward marginals that are never defined; as written the E-step is not computable from the text, and 'd
- [high] N-gram methods in the 2026 LLM stack: infini-gram/∞-gram suffix-array engines (Liu et al. 2024) over trillions of tokens, 13-gram-overlap benchmark decontamination, KenLM perplexity filtering in pretraining pipelines (CCNet) :: This is the currency bridge that makes n-gram depth defensible at a frontier lab in 2026 — data contamination and corpus filtering questions are asked directly, and they are answered with this chapter
- [high] Modern descendants of lattice/DP decoding: grammar/JSON-schema-constrained LLM decoding (FSA ∩ LM), beam search as approximate Viterbi, CTC as a forward-backward application :: Shows the interviewer that Viterbi/forward-backward fluency transfers to 2026 systems (structured output enforcement, speech, speculative decoding); currently the chapter's DP machinery has no forward

CORRECTNESS:
- Line 308: 'GPT-2 achieved perplexity ≈ 29 on Penn Treebank' — the GPT-2 paper reports 35.76 on PTB (zero-shot, 1.5B); 29 matches no standard GPT-2 benchmark (WikiText-103 is 17.48, WikiText-2 18.34). Fix the number or switch the dataset.
- Line 301: 'state-of-the-art n-gram models achieve perplexities around 50–100' for English — true only for models trained on very large corpora; on PTB itself KN 5-grams sit around 140. Juxtaposed with the (incorrect) GPT-2-on-PTB figure, it invites apples-to-oranges comparisons; specify training corpus and test set.
- §Smoothing 'Backoff' paragraph (lines 177–181) presents only stupid backoff, correctly noting it is unnormalized — but with Katz backoff absent, the text implies backoff is inherently unnormalized, which is wrong (Katz is properly normalized via Good-Turing discounting).
- Baum-Welch E-step (lines 570–577) uses the backward variable β without ever defining it — the algorithm is not executable from the text as written.
- Lines 358–361: HMMs 'for decades ... the dominant approach to sequence labeling tasks like part-of-speech tagging and named entity recognition' — NER only emerged with MUC-6 (1995) and discriminative models displaced HMMs for it within ~6 years; 'decades' fits POS/speech, not NER.
- Duplicate LaTeX labels shared with ch1 (sec:ngram_lm at ch2:31, sec:perplexity at ch2:247) — multiply-defined in the combined build; cross-references are ambiguous.

STALENESS:
- 'Perplexity ... has been the primary evaluation metric for language models from n-grams through GPT-4' — in 2026 perplexity/loss is primarily a scaling-law and data-quality metric; benchmark suites, preference evals, and LLM-as-judge drive model decisions. One sentence of reframing needed.
- The 500-sentence NER answer's 'use GPT-4 ... for silver labels' reads 2023; name current frontier models generically and add GLiNER-style zero-shot NER as the first baseline.
- BERTopic as the 'modern' endpoint of topic modeling is fine but should note the 2024+ practice of LLM-generated topic labels/summaries on top of clustering.
- No mention anywhere that HMM interview frequency has collapsed outside speech/bio-informatics roles; the chapter's framing ('interviewers expect you to know all three problems') was true in 2018, needs a 2026 qualifier.

MUST-KNOW:
- Chain rule decomposition and the Markov assumption; why V^{t-1} contexts force truncation
- The zero-probability problem and why smoothing is mandatory; Laplace's failure mode with large V
- Kneser-Ney: absolute discounting + continuation probability, with the 'Francisco' example
- Perplexity ↔ cross-entropy (PP = 2^H = exp of avg negative log-likelihood), branching-factor intuition, and its four limitations (tokenizer dependence, no coherence/factuality, domain sensitivity, not a task metric)
- The three HMM problems (likelihood/decoding/learning) and which algorithm solves each
- Viterbi: derivation, backpointers, O(TK²) time, log-space in practice; forward = same recursion with sum instead of max
- Generative vs discriminative: what HMMs assume that CRFs don't; feature flexibility as the practical win
- CRF partition function: why K^T is intractable, forward-algorithm computation, gradient = observed − expected feature counts
- Why (and when not) to put a CRF on BERT: input-side vs output-side dependencies, marginal gains at high data/scale
- Noisy channel decomposition (translation model × language model) and its reuse in speech and spelling correction

IMPROVEMENTS:
- Become the canonical n-gram/perplexity home: coordinate deletion of ch1's duplicate section and fix the colliding sec:ngram_lm / sec:perplexity labels; also reconcile with ch10's perplexity section (derivation here, metric-catalog there) so the volume has one derivation, not three.
- Add a short MEMM → label-bias → CRF subsection between §3 and §4 (one figure, one worked mass-conservation example) — closes the chapter's most glaring promised-but-missing item.
- Add the backward recursion (3–4 lines mirroring the forward box) and show γ/ξ computed from α·β so Baum-Welch and the CRF gradient are self-contained.
- Compress IBM Models 3–5 (fertility/distortion) and the Baum-Welch M-step detail to summary paragraphs; reinvest the ~3 pages in the label-bias section and modern bridges — SMT detail beyond the noisy channel and Model 1 has near-zero 2026 interview frequency.
- Recalibrate the frequency claims: 'Why add a CRF on BERT appears in a large fraction of NLP interviews' is overstated for 2026 — accurate framing is 'common in applied-NLP/industry loops, declining at frontier labs, where sequence-DP knowledge is instead probed via decoding, CTC, and constrained generation'.
- Add worked forward-algorithm numbers on the same 2-state 'the dog' example used for Viterbi — interviewers love asking for sum-vs-max on one lattice, and the contrast is the chapter's own red-flag item.
- In the 500-sentence NER design answer, lead with the 2026 baseline ordering: frontier-LLM/GLiNER zero-shot first to calibrate, then decide whether BERT-CRF fine-tuning is warranted; add expected F1 anchors so the answer has numbers.
- Add one honest-ROI box per section (as the SMT warning box already does well) stating how often each topic is asked in 2026 loops, so readers can allocate study time.

# 03_word_representations.tex — Word Representations and Embeddings

VERDICT: KEEP, high interview ROI, but needs a 2025-26 refresh of the 'modern embeddings' half and aggressive dedup against DL-06/07/24. The classical core (Word2Vec/GloVe/FastText/Levy-Goldberg) is the best treatment of this material in either volume and is genuinely still asked in 2026 screens. The modern half (Section 6) stops at early-2024 and misses the decoder-LLM embedding era that now dominates MTEB and interview follow-ups. Two of the ten interview questions are near-duplicates of DL-volume questions and should be merged or cross-referenced.

STAFF: Sections 1-5 are deliberately L4/L5 pedagogy done well — appropriate for foundations, and the L5/L6/L7 ladders are consistently well-calibrated (the L7 rungs cite real papers: Wiedemann 2019, Wang & Isola 2020, Ethayarajh 2019). Staff-level substance concentrates in Section 6 and Q8/Q10, which carry real numbers (300GB index math, <100ms latency budgets, HNSW vs IVF-PQ). What's missing for L6/L7 credibility in 2026: embedding-model migration economics (re-embedding + re-indexing 100M docs when the model updates — raised only as a follow-up question, never answered), cost framing (API embedding $/1M tokens vs self-hosted GPU), decoder-embedder trade-offs, and binary quantization. Interview ROI is high and justifies the page count: Word2Vec derivation remains a standard depth screen, and sentence-embedding/retrieval is a top-3 2026 NLP interview topic via RAG; shift weight from static-embedding hyperparameter minutiae toward the modern embedder training recipe.

OVERLAP [heavy] with 06_embeddings_representations.tex, 07_similarity_metric_learning.tex, 08_self_supervised_learning.tex, 24_long_context_rag.tex
STRONGER: Split. NLP-03 is decisively stronger on foundations: full derivations (GloVe ratio derivation, NEG objective with noise-distribution rationale, Levy-Goldberg), an evaluation section DL-06 lacks entirely, and 10 questions vs DL-06's compressed survey (DL-06 covers Word2Vec/GloVe/FastText in ~70 lines with bare equations). DL-06 is stronger on failure modes and production diagnostics (embedding collapse taxonomy with SVD diagnostics, BERT's three-component input embedding, tokenization-embedding interaction) and states the ELMo frozen-biLM fact correctly where NLP-03 fumbles it. DL-07 (123 hits on ANN/HNSW/InfoNCE/hard-negatives) and DL-24 (rerankers, ColBERT, hybrid BM25+dense, RAG pipelines) are stronger on everything downstream of the embedding itself.
RECOMMENDATION: Canonical home for word/sentence-embedding THEORY = NLP-03; keep DL-06 as the condensed cross-volume review but delete its duplicated static-vs-contextual question (DL-06 Q3 vs NLP-03 Q3 make nearly identical arguments) in favor of a cross-ref. Canonical home for retrieval SYSTEM DESIGN = DL-07/DL-24: merge NLP-03 Q10 (50M-doc pipeline) with DL-06 Q2/Q8 — three near-duplicate system-design answers across the set will drift out of sync (they already disagree on details like index sizes and reranker latencies); NLP-03 should keep exactly one production question (Q8) with pointers into DL-24 for reranking/hybrid retrieval. Keep both InfoNCE treatments only if NLP-03's cites DL-08's MI-bound derivation instead of restating it.

MISSING (crit/high):
- [high] Hierarchical softmax (Morin & Bengio 2005; Huffman-tree variant in Mikolov 2013) :: The chapter presents negative sampling as THE fix for the O(V) softmax, but 'what are the two approximations in Word2Vec, and when would you prefer hierarchical softmax (O(log V), exact normalized pro
- [critical] Decoder-LLM-based embedding models: E5-Mistral (Wang et al. 2024, synthetic-data training), GritLM, NV-Embed, LLM2Vec (bidirectional conversion), Qwen/gte-Qwen and Gemini/OpenAI text-embedding-3 class models; last-token pooling for causal LMs :: By 2026 the MTEB leaderboard is dominated by 7B-class decoder-based embedders, and 'why would you fine-tune a decoder LLM as an embedder, and what pooling do you use?' is a live interview question. Th
- [high] Anisotropy / representation degeneration as a real subsection (Ethayarajh 2019; Gao et al. representation degeneration; whitening/BERT-flow post-processing) :: The chapter itself says interviewers test anisotropy (it appears in Q4/Q7 L7 ladders and follow-ups at lines ~1256, 1437), yet the body never defines it or shows the cosine-in-a-narrow-cone picture. L

CORRECTNESS:
- Lines ~283-289: 'Mikolov et al. showed [NEG] is equivalent to maximizing a lower bound on the mutual information ... and that it approximates the full softmax gradient' — both claims wrong/misattributed (see improvements). This is the one outright error in an otherwise carefully checked chapter.
- Line ~675: ELMo layer weights described as 'learned during fine-tuning on the downstream task' — ELMo is feature-based; the biLM is frozen. Contradicts the (correct) DL-06 statement.
- Verified correct (no action): PMI formula (line ~100), subsampling formula 1-sqrt(t/f) (line ~313), NEG loss and 3/4-power noise distribution (lines ~270-299), k=5-20/2-5 recommendations, Levy-Goldberg shifted-PMI (line ~337), GloVe objective with x_max=100, alpha=3/4 (lines ~456-478), FastText n-gram decomposition of 'where' (line ~554 — enumerated and checked), C(10000,2)≈50M SBERT motivation (line ~770), 100M x 768 x 4B ≈ 300GB (Q8) and 50M ≈ 150GB (Q10) storage math.

STALENESS:
- Modern-embeddings narrative frozen at early 2024: no decoder-LLM embedders (E5-Mistral, NV-Embed, GritLM, Qwen3/gte-Qwen), no LLM2Vec, no synthetic-data embedding training — the dominant 2024-26 paradigm.
- MTEB described in its 2023 form (58 datasets); MMTEB (2025) and leaderboard-gaming caveats absent.
- Comparison table crowns E5/BGE/GTE 'State-of-art' — no longer true in 2026.
- No mention of OpenAI text-embedding-3 / Cohere / Voyage even as category representatives, though Q8 is about choosing a production embedding model.
- Vector-search storage discussion (Q8/Q10) predates the 2024-25 binary-quantization standard.

MUST-KNOW:
- Distributional hypothesis (Firth 1957) as the single answer to 'why do embeddings work'
- Skip-gram softmax objective and the negative-sampling reformulation, including the unigram^(3/4) noise distribution and why O(V) → O(k)
- Levy & Goldberg 2014: SGNS implicitly factorizes the shifted PMI matrix (PMI - log k) — the unification of count-based and predictive methods
- GloVe objective: weighted least squares on log co-occurrence, the ratio motivation (ice/steam), f(x) weighting
- FastText character n-grams: OOV handling via subword sum; how it differs from BPE tokenization
- Polysemy and static-vs-contextual: the 'bank/crane' argument and what ELMo layer-wise probing showed (lower=syntax, upper=semantics)
- SBERT bi-encoder vs cross-encoder cost asymmetry (O(n) embeddings + ANN vs O(n^2) forward passes)
- SimCSE: dropout-as-augmentation positives, InfoNCE with temperature, anisotropy and the uniformity-alignment framing
- Hard negatives: in-batch, BM25-mined, cross-encoder-mined — and why negative quality bounds embedding quality
- Intrinsic vs extrinsic evaluation, MTEB, and the discipline of evaluating on the target task (Recall@K/NDCG, not analogy scores)

IMPROVEMENTS:
- Fix the misattributed claim at lines ~283-289 ('Why This Works'): Mikolov et al. did NOT show NEG maximizes a mutual-information lower bound (that is InfoNCE, Oord et al. 2018), and NEG does NOT approximate the full-softmax gradient (NCE has that asymptotic property; NEG is a simplification whose actual fixed point is the shifted-PMI factorization the chapter itself cites via Levy-Goldberg). As written this would be a red flag if a candidate repeated it to a research interviewer.
- Section 5 (ELMo): correct the parenthetical at line ~675 — the s_l weights are learned by the downstream task model with the biLM FROZEN (feature-based transfer, not fine-tuning). DL-06 (lines ~135-150) states this correctly; sync the two.
- Refresh the 'Modern Embedding Models' table (line ~811) through 2025: add E5-Mistral/NV-Embed/GritLM class decoder embedders, OpenAI text-embedding-3 (which shipped Matryoshka-style truncation — a great production hook for the existing MRL section), Voyage/Cohere commercial models; and change the comparison table's 'E5/BGE/GTE = State-of-art' row (line ~970), which is stale in 2026.
- Update MTEB subsection: note MTEB's 2023 numbers (8 categories/58 datasets) have been superseded by MMTEB (2025, 500+ tasks, per-language leaderboards) and that leaderboard overfitting/contamination is now a known caveat — exactly the kind of measurement skepticism the section preaches.
- Deduplicate Q10 (50M-document similarity system, line ~1624) against DL-06 Q2 (100M product catalog) and DL-06 Q8 (semantic search) — three near-identical system-design answers across the two volumes will drift; keep ONE canonical retrieval-design question per volume and cross-ref (see overlap).
- Move Section 6.8's hard-negative/InfoNCE material into explicit dialogue with DL-07 and DL-08 (both derive InfoNCE and its MI bound); a one-line cross-ref avoids re-teaching the loss three times.
- Add SIF details to the Arora et al. baseline (the weight a/(a+p(w)) formula) — currently name-dropped without the formula, unlike everything else in the chapter.
- Minor: word2vec learning-rate row (table line ~374) — canonical defaults are 0.025 (skip-gram) / 0.05 (CBOW); Nomic 'Matryoshka training' is v1.5, not the original release.

# 04_neural_sequence_models.tex — Neural Sequence Models

VERDICT: KEEP but right-size: this is the canonical RNN/LSTM/seq2seq home for the two-volume set (the DL volume has no RNN chapter at all), and its vanishing-gradient and LSTM derivations are exactly what depth interviews still probe. However, it is a 1660-line chapter for material whose 2026 interview ROI is declining; cut ~15-20% (TextCNN, Luong-variant taxonomy) and reinvest one page in the modern-recurrence bridge (SSM/Mamba/RWKV/xLSTM) that currently lives only in L7 ladder bullets. One arithmetic error and one math-statement error need fixing.

STAFF: The mathematical spine (BPTT → vanishing bound → LSTM gradient flow) is genuine L5-L6 material and the chapter's red-flag lists are well-tuned to how interviewers actually grade this topic ('gates so it works' = fail, derive the diagonal Jacobian = pass). Where it reads L4/L5: judgment and scale content is thin in the body — no wall-clock numbers contrasting LSTM vs transformer training throughput (the actual reason parallelism won), no production anchors (GNMT), and the when-does-recurrence-still-win judgment (Q7) stops at 2021-era answers with the 2026 answer (SSMs) exiled to ladder bullets. Interview ROI is medium and declining: perhaps a quarter of 2026 NLP loops still probe LSTM internals at depth, but nearly all probe the 'why transformers' narrative, so the derivations earn their keep as the substrate of that story. Honest page-weight call: this is breadth-plus-one-derivation insurance — trim TextCNN and the Luong taxonomy (~15-20% of the chapter), keep the LSTM math intact, add the one-page SSM bridge.

OVERLAP [partial] with 05_attention_transformers.tex, 14_efficient_architectures.tex, 09_nlp_architectures.tex (light)
STRONGER: NLP-04 is the sole real home of RNN/LSTM/GRU/seq2seq/BPTT — the DL volume has no RNN chapter (grep confirms only scattered mentions), so this material is not duplicated and must stay. The duplication is confined to attention scoring: DL-05 lines 43-93 cover Bahdanau + all three Luong variants and is the STRONGER treatment there (prev-vs-current decoder state distinction, local attention as sliding-window precursor, parameter counts, when-additive-wins analysis); NLP-04's version is a strict subset and even gets the Bahdanau state nuance wrong. On the forward end, DL-14's Mamba/SSM section is the canonical continuation of this chapter's story (it explicitly frames Mamba's selectivity as LSTM-style gating) and NLP-04 currently fails to hand off to it. DL-09's encoder-vs-decoder discussion lightly overlaps the BiRNN→BERT/GPT foreshadowing — harmless, keep both.
RECOMMENDATION: Keep NLP-04 as canonical for everything recurrent and for the seq2seq→attention narrative. Merge the attention-scoring detail: cut NLP-04's Luong-variant subsections and comparison table to a paragraph + cross-ref into DL-05 (which every candidate reads for transformers anyway), fixing the Bahdanau-state contradiction and the 2014/2015 date inconsistency in the process. Add an explicit hand-off cross-ref to DL-14 for SSM/Mamba rather than duplicating it — one half-page bridge here, full treatment there. Also cross-ref NLP-05 (transformers) at the 'Attention Is All You Need' key insight so the volume doesn't re-teach scaled dot-product twice.

MISSING (crit/high):
- [high] Modern recurrence bridge: S4 → Mamba (selective SSM), RWKV, xLSTM (Beck et al. 2024), minLSTM/minGRU ('Were RNNs All We Needed?', 2024), linear attention as an RNN, hybrid stacks (Jamba) :: In 2026 the highest-value follow-up to 'why did transformers replace LSTMs' is 'so why is recurrence back?' — Mamba's selection mechanism is literally LSTM gating rediscovered in the SSM framework (DL

CORRECTNESS:
- Line ~443: parameter-count example 4x512x813 stated as 1,664,064; correct value is 1,665,024 (verified).
- Lines ~149-164 and Q1 (~1052): spectral radius / spectral norm conflation in the vanishing-gradient sufficient condition (see improvements) — the stated inequality chain does not support the rho(W_h) conclusion as written.
- Line ~839: Bahdanau score written on the current decoder state h_t^d; the original uses s_{t-1} — and DL-05 explicitly teaches the discrepancy this glosses over; also Bahdanau dated 2015 here vs 2014 in DL-05.
- Lines ~495-497: GRU update-gate convention inverted relative to Cho et al. 2014 / PyTorch (internally consistent, but needs a convention note).
- Line ~957: TextCNN filter widths given as {2,3,4,5}; Kim 2014 used {3,4,5} (minor).
- Verified correct (no action): RNN/BPTT equations, LSTM equations and the diag(f_j) gradient-flow argument (with 'higher-order terms' caveat properly noted), gradient-clipping formula, forget-gate-bias initialization (Jozefowicz 2015), variational dropout attribution (Gal & Ghahramani 2016), BiRNN causality argument, scheduled-sampling attribution (Bengio 2015), BERT-340M/GPT-3-175B figures.

STALENESS:
- Narrative ends in 2017; the 2020-2025 recurrence revival (Mamba, RWKV, xLSTM, Griffin, Jamba hybrids) appears only in two L7 ladder bullets and one follow-up, despite being the standard 2026 extension of every question this chapter trains.
- Q7's LSTM-for-streaming answer is where a 2026 interviewer expects Mamba/RWKV as the primary answer, with LSTM as the historical one — the ladder knows this but the strong answer doesn't.
- No acknowledgment that speech/on-device (the cited LSTM niches) also largely moved to transformers/conformers or SSMs by 2024-25; the niches list is directionally fine but reads c. 2021.

MUST-KNOW:
- Vanishing-gradient derivation: BPTT Jacobian product, tanh' in (0,1], exponential decay condition — with equations, not hand-waving
- LSTM equations cold, gate by gate, and the additive cell-state update as the structural fix (product of diag(f_j) ≈ I when forget gate saturates)
- Forget-gate bias initialization to 1-2 and why; dropout NOT on the recurrent path
- GRU vs LSTM: tied (1-z)/z gating, no separate cell state, 3/4 parameter ratio, comparable empirics
- LSTM parameter count 4·d_h·(d_h+d_x+1) as a live back-of-envelope skill
- BiRNN causality constraint and its direct mapping to BERT-vs-GPT (encoder vs decoder)
- Seq2seq information bottleneck → attention as weighted sum over encoder states → self-attention makes recurrence redundant (the three-step story, fluently)
- Teacher forcing and exposure bias, plus mitigations (scheduled sampling, sequence-level objectives) and the fact that LLM pretraining still uses teacher forcing
- Why transformers won: parallelism, O(1) path length, scaling behavior — and the honest counterpoint (O(n^2) cost, KV-cache growth) that motivates SSMs
- The 2026 coda: Mamba/RWKV/xLSTM as gated recurrence reborn — constant-state streaming vs attention's exact retrieval, and hybrid stacks (Jamba)

IMPROVEMENTS:
- Fix arithmetic in the parameter-count example (line ~443): 4 x 512 x 813 = 1,665,024, not 1,664,064. Embarrassing in a chapter that sells itself on rigor; verified by direct computation.
- Fix the vanishing-gradient bound statement (lines ~149-164, repeated in Q1's strong answer ~1052): the displayed inequality is in terms of ||W_h|| (spectral NORM / largest singular value), but the conclusion is stated via spectral RADIUS rho(W_h) < 1/gamma. Since rho(W) <= ||W|| with possible strict inequality, the radius condition does not imply the displayed bound decays. Pascanu et al. 2013 state the sufficient condition via the largest singular value — use that, or state the radius result asymptotically. An L7 interviewer using this chapter's own framing would poke exactly here.
- Align the Bahdanau presentation with DL-05: this chapter scores with the CURRENT decoder state h_t^d (line ~839); Bahdanau actually conditions on s_{t-1}, and DL-05 (line ~75) makes precisely this its 'key difference from Luong' teaching point. The two volumes currently contradict each other on a detail one of them treats as a nuance worth knowing. Also reconcile 'Bahdanau 2015' (here) vs 'Bahdanau 2014' (DL-05).
- Flag the GRU convention (lines ~495-497): h_t = (1-z)⊙h_{t-1} + z⊙h̃ reverses the roles of z relative to Cho et al. 2014 and PyTorch (where z multiplies the PREVIOUS state). It is internally consistent here, but a candidate cross-checking against the paper or torch.nn.GRU docs will think the book is wrong; add a one-line convention warning.
- Compress the Luong-variants subsection and comparison table (lines ~853-895): DL-05 covers the same three score functions with strictly more content (local vs global attention, parameter counts, the prev-vs-current state distinction). Keep the additive-vs-multiplicative intuition and the 'lineage to scaled dot-product' point; cross-ref DL-05 for the taxonomy.
- Compress TextCNN (lines ~940-1000) to half a page: 2026 loops essentially never probe TextCNN beyond 'CNNs as n-gram detectors + max-over-time pooling'; note Kim 2014 used filter widths {3,4,5} (the chapter says {2,3,4,5}) and framed the nonlinearity as tanh in the paper.
- Soften two overclaims: 'GRU trains about 25% faster' (parameters are 25% fewer; wall-clock speedup is smaller and workload-dependent) and Q3's 'LSTMs do not benefit proportionally from larger models and datasets' (Kaplan et al. 2020 showed LSTMs DO follow power laws — with worse constants and poor late-context utilization; that is the precise, stronger claim).
- Historical-arc table (line ~1012): either extend past 2017 (linear attention 2020, S4 2022, Mamba 2023, hybrids 2024) or add an explicit pointer to DL-14; also 'the +Self-attention / 2016' row should name its papers (Cheng et al. 2016, Parikh et al. 2016) or be folded into the transformer row.
- LSTM vs GRU table 'Typical use: Default for most tasks' (line ~534) reads 2018-vintage; rewrite as 'legacy default within the RNN family' to match the chapter's own transformer-era framing.

# 05_transformers.tex (Transformers and Modern Architectures)

VERDICT: Strong, mostly correct flagship chapter — the best pedagogical treatment of core attention math in either volume — but it is frozen in mid-2023: the GPT story ends at GPT-4, MLA is absent, and 60%+ of its content is duplicated (with drift) across DL chapters 05 and 09. Needs a 2026 currency pass and a de-duplication treaty more than it needs new depth.

STAFF: Solidly L5 with genuine L6 spikes: the why-questions (sqrt d_k derivation, 80/10/10 rationales, FFN-as-KV-memory with ROME/MEMIT, induction heads, why-decoder-only-won) are exactly what separates recitation from understanding, and the redflag/ladder scaffolding is well-calibrated. What keeps it from reading L6/L7 throughout: almost no numbers the candidate can compute with — no parameter/FLOPs accounting, no KV-cache formula, no cost-per-token arithmetic (all of which sit in DL 05/22, unreferenced); no MLA; and the modern-architecture narrative stops in 2023, so a candidate prepped only on this chapter will sound dated in a 2026 frontier-lab loop even while being rigorous. Fix currency + add the quantitative cross-refs and this becomes the strongest chapter in the two-volume set. Interview ROI within scope is maximal — this is the #1 tested territory and page weight is justified; the only trim candidates are the NSP minutiae (half a page is enough) and the second BERT fine-tuning list, freeing room for MLA and sinks.

OVERLAP [heavy] with 05_attention_transformers.tex (heavy: attention derivation, sqrt d_k, MHA, all five positional-encoding schemes with near-identical comparison table, pre/post-norm, RMSNorm, SwiGLU, MQA/GQA, Flash Attention, sliding window, complexity), 09_nlp_architectures.tex (heavy: three-paradigm comparison, BERT + MLM 80/10/10 + NSP, RoBERTa/ELECTRA/DeBERTa variants, GPT evolution, when-to-use guides), 14_efficient_architectures.tex (light: sparse/linear attention, SSM pointers), 22_inference_optimization.tex (light-moderate: KV cache, GQA-serving interaction)
STRONGER: NLP 05 is stronger on derivations and interview pedagogy: the variance proof is worked step-by-step vs. one sentence in DL 05 (line 107); the BERT/ELECTRA treatment is deeper and correct where DL 09's ELECTRA loss (line 233) has its indicator functions flipped relative to the paper (NLP 06 lines 146-155 has it right); DL 05 wrongly labels GPT-2 post-norm (line 756) where NLP 05 (line 950) is correct. DL 05 is clearly stronger on quantitative systems content: parameter count, FLOPs per token, KV-cache byte math with the LLaMA-7B worked example, and the excellent 7B training-memory estimation question (lines 879-914) — none of which NLP 05 has.
RECOMMENDATION: Split canonical ownership and cut the duplicates. Make NLP 05 the canonical home for attention/positional-encoding derivations and the BERT/GPT/T5 family deep-dives (its treatment is stronger and correct); DL 05 keeps the Bahdanau->scaled-dot-product lineage plus ALL quantitative analysis (params/FLOPs/KV-cache/training memory) as its canonical contribution, and compresses its positional-encoding and MQA/GQA/Flash subsections to a summary table + cross-ref. DL 09 should shrink its BERT/variants/ELECTRA subsections to a comparison table + pointer to NLP 05 and fix its ELECTRA loss sign bug. Non-negotiable regardless of split: sync the three found drift points (GPT-2 norm placement, ELECTRA loss, MHA-in-GPT-4 claim) — the candidate currently gets contradictory facts depending on which volume they open.

MISSING (crit/high):
- [critical] Multi-head Latent Attention (MLA, DeepSeek-V2/V3/R1, Kimi K2): low-rank KV compression, decoupled RoPE, why it beats GQA on the KV-cache/quality frontier :: By 2026 'compare MHA/MQA/GQA/MLA' is a standard frontier-lab probe. MLA appears only as an L7 name-drop (line 1565) and a table row in ch7; it is explained nowhere in either volume — the single bigges
- [high] KV-cache size formula and worked example (2 * layers * n_kv_heads * d_head * seq * bytes) :: The chapter's own follow-up (line 1571) asks 'What is the KV-cache size for a 70B model with 128K context?' but never teaches the formula. DL 05 (lines 819-830) and DL 22 have it; at minimum add the o
- [high] Attention sinks / StreamingLLM and why the first tokens absorb disproportionate attention; sliding-window + sink combination (used in gpt-oss, 2025) :: Directly follows from the sliding-window section (line 1061) and is a common 2025-26 probe on 'what breaks when you evict old KV entries'; connects softmax mechanics to serving behavior.
- [high] One-paragraph pointer to the reasoning-model era in the GPT evolution table (o1/o3, DeepSeek-R1, test-time compute) with cross-ref to ch7 :: The evolution table (line 926) ends at GPT-4 (2023). A candidate who narrates GPT history ending in 2023 sounds two years stale even though ch7 covers reasoning; the table needs one more row plus a po

CORRECTNESS:
- Line 103-105: 'every token would attend most strongly to itself (since ||x_i||^2 >= x_i^T x_j by Cauchy-Schwarz)' — Cauchy-Schwarz gives x_i.x_j <= ||x_i|| ||x_j||, which implies self-dominance only when norms are comparable. As stated the inequality is false in general (take ||x_j|| >> ||x_i||). Fix with 'for approximately equal-norm representations'.
- Line 626 (Decoder Block mathresult): cross-attention written as MultiHead(LN(h), LN(H_enc), LN(H_enc)) — standard pre-norm encoder-decoders (T5) normalize the decoder stream only; encoder output gets one final LN, not a fresh LN inside every decoder layer. A formula box invites exactness.
- Line 594: pre-norm 'eliminates the need for careful learning rate warmup' — overstated; every production LLM still uses warmup. Say 'greatly reduces warmup sensitivity'.
- Line 997: 'PaLM and Falcon use MQA' — Falcon-7B is MQA; Falcon-40B uses 8 KV-head grouped attention (GQA-style). Minor but this is a table-fact chapter.
- Line 530: 'Relative (Shaw) ... Used In: Transformer-XL, DeBERTa' — Transformer-XL uses Dai et al.'s distinct relative scheme and DeBERTa uses disentangled attention; neither uses Shaw's a_ij embeddings. Lineage is imprecise.
- Verified correct (no action): variance-of-dot-product derivation (lines 137-159); attention-matrix memory 32*4096^2*2B ~ 1GB (line 179) and 128K ~ 1.1TB (line 1296); RoPE formula and theta_i (line 488); sinusoidal PE (line 445); LoRA-free facts in the GQA table; 80/10/10 = 12% actually masked (line 851); GPT param counts (117M/1.5B/175B); receptive field l*w for sliding window (line 1067).

STALENESS:
- GPT family table (lines 926-943) ends at GPT-4/2023 — no GPT-4o, no o1/o3 reasoning era, no open-weights frontier (Llama 3, DeepSeek-V3/R1, Qwen). The strongest single staleness signal in the chapter.
- Efficiency summary table line 1086: 'Linear Attention — Research models' is no longer true; production hybrids shipped 2024-25 (Jamba, MiniMax-01, Qwen3-Next, IBM Granite 4.0). Ch7 covers SSMs, but this table row actively misinforms.
- Attention-variants table line 1017 'Used In: BERT, GPT-3, GPT-4' for MHA — GPT-3 used alternating dense/locally-banded sparse attention, and GPT-4's attention is undisclosed (the same chapter's GPT table calls it MoE, footnoted as unconfirmed). Speculative and internally inconsistent.
- BERT variants table (lines 881-904) contains nothing after ~2020; ModernBERT/EuroBERT era absent.
- Flash Attention section: FA2 mentioned in a strong answer, FA3 (2024, Hopper/FP8) only as an L7 bullet; the de-facto standard in 2026 serving is FA3 on H100-class hardware.
- Line 532 'ALiBi ... Used In: BLOOM, MPT' — accurate but these are 2022-23 models; note ALiBi lost to RoPE+scaling in practice.

MUST-KNOW:
- Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V, and the full variance derivation of why sqrt(d_k) (Var(q.k) = d_k -> softmax saturation -> vanishing gradients)
- Why separate Q/K/V projections; attention as soft dictionary lookup and its direct connection to the KV cache
- Multi-head mechanics: d_k = d/h, same total cost as single-head, role of W_O; evidence that heads specialize and many are prunable
- O(n^2 d) compute / O(n^2) memory, and what that implies for context length, training vs inference bottlenecks
- Positional encoding family: permutation equivariance, RoPE rotation math + relative-position property + base scaling/YaRN, ALiBi slopes, NoPE-with-causal-mask nuance
- Pre-norm vs post-norm formulas and the gradient-highway argument; RMSNorm; SwiGLU (8/3 d expansion)
- The three families (MLM/CLM/span corruption) and the four reasons decoder-only won (simplicity, data scaling, ICL, KV-cache)
- BERT masking 80/10/10 (12% actually [MASK]), why each fraction exists, RoBERTa's NSP removal, ELECTRA's RTD efficiency argument
- MHA vs MQA vs GQA vs MLA with KV-cache arithmetic for a named model (e.g., LLaMA-2 70B, g=8)
- Flash Attention: exact, IO-aware tiling + online softmax, O(n) memory, 2-4x wall-clock — systems optimization, not approximation

IMPROVEMENTS:
- De-duplicate against DL 05: positional encodings, MQA/GQA, Flash Attention, pre/post-norm, RMSNorm, SwiGLU, and the sqrt(d_k) argument all appear in both volumes with near-identical comparison tables. Decide canonical homes (see overlap) and replace one side with summaries + cross-refs; the drift is already real (DL 05 line 756 calls GPT-2 post-norm; this chapter line 950 correctly says pre-norm since GPT-2).
- Add quantitative systems content or cross-refs: parameter counting (12Ld^2 + Vd), FLOPs per token (6ND), and training-memory estimation live only in DL 05 (lines 841-914). The NLP chapter asks systems-flavored questions (Q9, Q10) without giving the candidate the arithmetic to answer the follow-ups.
- Promote MLA from an L7 bullet to a subsection in Section 9 with the low-rank KV projection math and a KV-cache comparison row in the table at line 1010.
- Tighten the causal-mask/NoPE warningbox (line 540) with a citation (Haviv et al. 2022; Kazemnejad et al. 2023) — it makes an empirical claim ('some research (NoPE)') with no reference in a book that otherwise cites everything.
- Q12 (BERT vs GPT-4 for ticket classification) is excellent production framing but its cost anchors ('500ms+ for GPT-4 API', '1000x cheaper') should be reframed as 'order-of-magnitude, 2023 prices' or updated — 2026 candidates will quote small-model APIs (Haiku/Flash/mini) that compress the gap.
- The 'Ensemble effect' justification for multi-head (line 274) is hand-wavy; replace with the Voita et al. 2019 / Michel et al. 'are sixteen heads really better than one' pruning evidence already referenced in the Q3 ladder, which is the actual staff-level answer.
- Sliding-window summary table 'Used In' column (line 1084): add Gemma 2/3 (interleaved local/global) and gpt-oss; Mistral alone is a 2023 snapshot.

# 06_pretraining_finetuning.tex (Pre-training, Fine-tuning, and Transfer Learning)

VERDICT: The best PEFT treatment in the two-volume set — LoRA/QLoRA math is deep, verified, and interview-shaped — and the data-pipeline section is uniquely valuable content the DL volume lacks. But several practice claims are wrong or dated (LoRA learning rate, Q+V-only default, QLoRA quality band), the chapter predates the reasoning-SFT/model-merging era, and its PEFT core is heavily duplicated with DL 16.

STAFF: The LoRA/QLoRA core is genuinely L6 material — intrinsic dimensionality, initialization rationale, verified memory arithmetic, and the DoRA/IA3 breadth are better than DL 16's parallel treatment. The data-pipeline section is the right instinct (staff interviews increasingly live here) but reads L5: it describes the 2023 pipeline shape without the 2024-26 judgment calls (model-based filtering beats heuristics, annealing phases, repeated-epoch economics, synthetic-data risk). The practice-level guidance is where it slips to L4/L5: wrong LoRA LR, dated Q+V default, no serving numbers for the multi-adapter question the chapter itself poses, and an SFT story that predates reasoning-trace distillation — the topic most likely to dominate a 2026 SFT conversation. Interview ROI: PEFT and SFT are top-five topics in current loops and deserve their page weight; prefix/prompt tuning and IA3 are now mostly comparison-question fodder and are already correctly sized at ~1 page total — do not expand them; spend the reclaimed duplicate-PEFT pages (post de-dup with DL 16) on reasoning-SFT, merging, and data-pipeline currency.

OVERLAP [heavy] with 16_transfer_learning.tex (heavy: LoRA core/init/scaling/target-modules/merging, QLoRA, adapters, prefix tuning, prompt tuning, PEFT comparison table, feature-extraction-vs-fine-tuning, gradual unfreezing, layer-wise LR decay), 09_nlp_architectures.tex (moderate: MLM 80/10/10 + rationale, NSP, RoBERTa, ELECTRA, span corruption/BART, SFT stage of RLHF pipeline), 08_self_supervised_learning.tex (light-moderate: BERT MLM as masked prediction, contrastive/InfoNCE framing, data-quality debugging), 02_learning_theory.tex (light: Chinchilla 20:1, overtraining/inference-optimal trade-off — treated in more depth there than here), 14_efficient_architectures.tex + 22_inference_optimization.tex (light: quantization methods adjacent to NF4; distillation)
STRONGER: NLP 06 is the stronger PEFT treatment on every axis that matters for interviews: intrinsic-dimensionality argument, alpha/r mechanics, DoRA/IA3 coverage, QLoRA memory math with verified numbers, and multi-adapter serving — DL 16's LoRA sections (lines 270-428) are a lighter sketch of the same material, though DL 16 uniquely contributes negative transfer, strategy-selection flowcharts, and pitfalls lists worth keeping. For pretraining objectives, NLP 06 is both deeper and more correct than DL 09 (whose ELECTRA loss has flipped indicators, line 233). The data-pipeline and SFT-data-quality sections have no real DL-volume counterpart. Conversely, DL 02 treats Chinchilla/overtraining better than this chapter's one paragraph (line 399).
RECOMMENDATION: Make NLP 06 the canonical home for (a) pretraining objectives, (b) the data pipeline, (c) the PEFT deep dive, and (d) SFT data/loss-masking. DL 16 should retain transfer-learning fundamentals, negative transfer, strategy selection, and vision-side transfer, compressing its LoRA/QLoRA/prefix/prompt subsections to a one-table summary + cross-ref (that alone reclaims ~300 lines of duplication). DL 09 keeps a one-page objectives recap feeding its architecture-selection guide, with the details pointed here — and must fix its ELECTRA loss to match this chapter's correct version. Keep Chinchilla canonical in DL 02 and replace this chapter's paragraph with a cross-ref plus the Llama-3 overtraining example. Harmonize the alpha=r vs alpha=2r guidance across volumes so the two books stop giving different defaults for the single most-asked hyperparameter.

MISSING (crit/high):
- [critical] LoRA learning-rate practice: LoRA needs ~10x the full-FT learning rate (1e-4 to 3e-4 vs 1e-5), and 'LoRA learns less and forgets less' (Biderman et al. 2024) as the evidence-based framing of the LoRA-vs-full-FT trade-off :: This is the single most common practical LoRA interview probe and the chapter not only omits it but gives the wrong LR (Q4 recommends 1e-5 to 2e-5 for a LoRA recipe). A candidate following this book w
- [critical] SFT on reasoning traces / distillation from reasoning models (DeepSeek-R1 distill, s1, LIMO 'less is more for reasoning', rejection-sampling fine-tuning) :: By 2026 the highest-frequency SFT question at frontier labs is about long-CoT SFT: what it teaches, why ~1K curated traces work (the LIMA result's modern sequel), and when SFT-distillation beats RL. T
- [high] Model merging: task arithmetic, SLERP, TIES/DARE, model soups; merging multiple LoRAs or checkpoints :: Standard 2025-26 interview topic ('you have a code adapter and a chat adapter — how do you combine them?'); completely absent from both volumes (verified).
- [high] Continued/domain-adaptive pretraining as a section (DAPT/TAPT, Gururangan et al. 2020; midtraining/annealing phases with high-quality data upweighting, Llama 3 / MiniCPM-style WSD schedules) :: Mentioned twice in question answers (Q4 step 7) but never taught; 'when do you continue pretraining vs fine-tune vs RAG' is a staple system-design probe and the annealing/midtraining stage is now stan
- [high] Synthetic data generation pipelines: Self-Instruct, Evol-Instruct, Magpie, distillation-with-filtering, and their contamination/model-collapse risks :: Q4 says 'use the base model to generate synthetic examples' in one line; 2026 loops ask for the actual pipeline and its failure modes. Most SFT data at every lab is now synthetic.
- [high] Modern data-pipeline currency: FineWeb/FineWeb-Edu, DCLM, Dolma, Nemotron-CC; model-based quality filtering (LLM-judged educational-value classifiers) alongside the heuristic/perplexity filters described; multi-epoch findings (Muennighoff et al.: ~4 epochs of repeated data are nearly free) :: Section 2 is the chapter's unique asset but cites only C4/RefinedWeb-era sources and heuristic filtering; the field's biggest 2024-25 result is that model-based filtering (DCLM, FineWeb-Edu) beats heu

CORRECTNESS:
- Line 982: broken LaTeX cross-reference ('Chapter~\ref{chap:pretraining_finetuning}+1' — self-reference plus a literal '+1'). Definite build-visible bug.
- Q4 strong answer (~line 1256): recommends LR 1e-5 to 2e-5 for a LoRA fine-tune — that is the full-FT range; standard LoRA practice (incl. the QLoRA paper) is 1e-4 to 3e-4. An interviewer would mark this answer down.
- Q2 strong answer (~line 1140): 'LoRA trains 2-3x faster due to fewer optimizer states' — overstated; the dominant saving is memory, wall-clock gain is typically ~1.2-1.5x since activations and the full backward pass remain.
- Line 842 + table line 959: 'QLoRA achieves 85-95% of full fine-tuning quality' contradicts the QLoRA paper's central claim of matching 16-bit full FT; stated as an unqualified fact it is wrong about the paper it cites.
- Line 1032 (LIMA keyinsight): 'competitive with GPT-4-era instruction-tuned models on many benchmarks' — overstates the paper's human-eval result (GPT-4 preferred in 57% of cases).
- Lines 721-734: 'Standard: apply to Q and V ... For most tasks, Q+V is sufficient' — faithful to the 2021 LoRA paper but contradicted by QLoRA's all-linear finding and 2024+ defaults; presented as current best practice, it is stale-to-wrong.
- Verified correct (no action): MLM/CLM/RTD/InfoNCE loss formulas (RTD indicator convention matches the ELECTRA paper — notably more correct than DL 09's flipped version); T5 span-corruption defaults (15%, mean span 3); LoRA parameter arithmetic 32*2*16*(4096+4096)=8.39M = 0.12% of 7B; QLoRA memory math (0.5 bytes/param -> 65B ~ 33GB; double-quant 0.5 -> 0.13 bits/param ~ 3GB saved; >780GB full-FT figure matches the paper); DoRA formula; prefix-tuning 2pdL count; EWC penalty form; Chinchilla ~20 tokens/param.

STALENESS:
- Data sources table (line 310): C4/RefinedWeb/BookCorpus era; missing FineWeb(-Edu), DCLM, Dolma, Nemotron-CC — the corpora every 2025-26 candidate should name. BookCorpus is also effectively deprecated/defunct.
- Quality filtering (line 373) describes only heuristic + perplexity filtering; model-based quality classification (FineWeb-Edu, DCLM fastText-on-LLM-labels) is the current standard and the known-better approach.
- Chinchilla discussion (line 399) is correct but should anchor overtraining with the canonical 2024+ example (Llama 3: 8B on 15T tokens, ~1875:1) as DL ch02 already does — currently the two volumes tell this story at different depths.
- Chat-format list (line 999): ChatML/Alpaca/LLaMA-[INST]/Vicuna is a 2023 snapshot; Llama-3 header format and Harmony (gpt-oss) are the 2025-26 reference points.
- Three-stage pipeline (line 1053) 'Pre-training -> SFT -> RLHF/DPO' needs a note that 2025-26 pipelines add midtraining/annealing and RLVR-style reasoning RL (cross-ref ch7/ch8); as written it describes the 2023 recipe.
- PEFT landscape stops at DoRA (2024); no rsLoRA/LoRA+/PiSSA even as table rows.

MUST-KNOW:
- The four pretraining objectives (MLM 15%, CLM 100%, span corruption, RTD) with the signal-per-token efficiency argument and the objective->capability mapping
- LoRA end-to-end: W = W0 + (alpha/r)BA, A Gaussian / B zero init and why, parameter arithmetic, merging for zero inference overhead, multi-adapter serving
- LoRA practice: all-linear targets as modern default, ~10x higher LR than full FT, rank/alpha selection, when LoRA fails (high-rank shifts, new languages/vocab), 'learns less, forgets less' trade-off
- QLoRA: NF4 (why normal-float beats uniform INT4), double quantization, paged optimizers, and the 65B-on-48GB memory walk-through
- Catastrophic forgetting: mechanism + the mitigation stack (low LR, gradual unfreezing, EWC, replay, early stopping, PEFT-as-structural-freezing)
- SFT mechanics: chat templates must match training, loss masked to assistant tokens only, and why (model the response distribution, not the user distribution)
- Data quality > quantity for SFT: LIMA and its reasoning-era sequels (s1/LIMO, R1-distill) — SFT sets behavior/format; ~1K curated examples can suffice
- Pretraining data pipeline: MinHash + exact dedup (Lee et al. 2022), heuristic/perplexity/model-based quality filtering, domain mixing effects (code -> reasoning), Chinchilla 20:1 vs inference-optimal overtraining
- Feature extraction vs full FT vs PEFT vs continued pretraining vs RAG as a single decision framework with the data-regime and freshness axes
- Prefix/prompt tuning and adapters: mechanisms, why only LoRA merges cleanly, prompt tuning's scale dependence (competitive only >10B)

IMPROVEMENTS:
- Fix the broken cross-reference at line 982: 'Chapter~\ref{chap:pretraining_finetuning}+1' renders as 'Chapter 6+1' — should reference the RLHF/alignment chapter's label.
- Correct the LoRA practice guidance: (a) which-modules advice at lines 721-734 presents Q+V-only as standard — QLoRA itself showed all-linear-layer adaptation is required to match full FT, and all-linear is the 2024+ default in the PEFT ecosystem; (b) add LoRA's higher LR; (c) soften 'LoRA trains 2-3x faster' (Q2, ~line 1140) to the honest ~1.2-1.5x wall-clock (backward still traverses the full network) + large memory savings.
- Reconcile the QLoRA quality claim: lines 842 and table line 959 say 85-95% of full FT, but the QLoRA paper's headline result is matching 16-bit full fine-tuning (with NF4 + all-linear adapters). State the paper's claim, then the practical caveat (gaps at low rank/limited target modules/harder domains).
- De-duplicate against DL 16: LoRA core idea/initialization/scaling/merging/target-modules, QLoRA, adapters, prefix and prompt tuning, and the PEFT comparison table all appear in both volumes. Also harmonize the alpha guidance (DL 16 line 355 says alpha=r; this chapter says alpha=2r) — not contradictory but reads as drift.
- Update Q4's recipe anchors: 'LLaMA-2 7B or 13B' base model and Alpaca/Vicuna chat formats are 2023 vocabulary; swap for Llama-3.x/Qwen-class models and modern templates, keeping the reasoning intact.
- Temper the LIMA claim (line 1032): LIMA was competitive with DaVinci-003/Bard in human evals but lost to GPT-4 in 57% of comparisons; 'competitive with GPT-4-era instruction-tuned models' overstates it. Pair LIMA with its 2025 reasoning-domain confirmations (s1, LIMO) for a stronger, current version of the same lesson.
- Reconsider warningbox item (5) at line 618 ('not freezing the embedding layer ... is a common mistake') — freezing embeddings is neither standard practice nor clearly beneficial; either justify or delete.
- Add a decision-framework diagram: prompt-engineering vs RAG vs PEFT vs full FT vs continued pretraining (data volume x latency x knowledge-freshness axes) — the chapter has all the ingredients scattered across Q2/Q4/Q8 but never assembles the single artifact interviewers ask candidates to draw.

# 07_llms_incontext_learning.tex (LLMs and In-Context Learning)

VERDICT: Strong L5-L6 chapter with genuinely good treatment of scaling economics, ICL mechanisms, and MoE/SSM basics, but it is frozen in mid-2024: the reasoning/test-time-compute section predates GRPO/RLVR and misattributes the mechanism of reasoning models to PRM-guided search; the agents/tool-use section is thin for 2026 loops; and roughly a third of the chapter (scaling laws, emergence, MoE, SSM) duplicates DL-volume chapters 02 and 14 nearly equation-for-equation. Fix currency in the reasoning section, expand agents, and dedupe aggressively — the underlying writing quality is high.

STAFF: Scaling laws, ICL, and MoE sections carry real L6 signal (over-training table with actual ratios, random-labels nuance, DeepSeek-V3 aux-loss-free balancing, four-model economics reasoning in Q1). Where it reads L4/L5: the tool-use/agents section (no numbers, no evals, no failure taxonomy), the test-time-compute section (no quantitative scaling results, no cost-per-query math, mechanism speculation), and the long-context solutions list (name-dropping without when-to-use judgment). Missing scale numbers throughout: no FLOPs worked example, no tokens/sec or $/1M-token figures anywhere, no latency numbers for CoT overhead. The L7 ladder bullets are consistently good, but the body doesn't always contain the material to reach them.

OVERLAP [heavy] with 02_learning_theory.tex (Scaling Laws + Emergent Abilities sections), 14_efficient_architectures.tex (MoE + SSM sections), 24_long_context_rag.tex (long-context/RoPE half), 22_inference_optimization.tex (KV cache), 05_attention_transformers.tex (GQA), 23_safety_alignment.tex (prompt injection/jailbreaks), 20_reinforcement_learning.tex (PRM/test-time compute), 09_nlp_architectures.tex (ICL + instruction-tuning subsections)
STRONGER: Split verdict. NLP ch7 is stronger on scaling economics (over-training table, warningbox), ICL depth, and prompting; DL ch14 is stronger on MoE mechanics (expert-choice routing, capacity factor, shared experts — all absent from ch7) and equal on SSM; DL ch02 duplicates the same Kaplan/Chinchilla constants and even the same GPT-3-vs-Chinchilla example plus an interview question making ch7's identical over-training point; DL ch22/24 are stronger on KV-cache/serving and RoPE extension; DL ch23 is much stronger on prompt injection (ch7's Q6 is a shallow duplicate of DL 23's territory).
RECOMMENDATION: Keep ch7 as the LLM-survey spine of the NLP volume but make it canonical ONLY for scaling-law economics, ICL, prompting, and test-time compute. Cut MoE/SSM to ~1.5-page summaries + pointers to DL ch14 (canonical home; port DeepSeek-V3 row there), slim DL ch02's scaling section to theory + pointer here, replace ch7's prompt-injection Q6 with a cross-ref to DL ch23 (which has three deeper questions on the same ground), and cross-ref DL ch22 for KV-cache/GQA instead of restating. The duplicated Chinchilla/MoE/Mamba equations across volumes are already drifting (Switch Transformer configs disagree between volumes) — exactly the failure mode dedup prevents.

MISSING (crit/high):
- [critical] GRPO / RLVR-trained reasoning models (DeepSeek-R1, R1-Zero, distilled reasoners) as the actual mechanism behind test-time compute — currently absent from BOTH volumes (grep confirms zero hits for GRPO/RLVR/DeepSeek-R1 anywhere) :: By 2026 'how are o1/R1-style models trained' is a top-3 LLM interview question; the chapter's PRM+tree-search story is the pre-R1 folk theory. The R1 report explicitly describes PRM and MCTS as failed
- [high] Agents beyond a half-page: MCP (Model Context Protocol), agent design patterns (planner/executor, sub-agents, memory), agent evaluation (SWE-bench Verified, tau-bench, GAIA), computer use, cost/latency budgeting for multi-step loops :: Agentic system design is the fastest-growing 2026 interview area at labs and product companies; the current tool-use section lists three generic challenges with no scale numbers, no eval story, and no
- [high] MLA (Multi-head Latent Attention) and modern KV-cache reduction (GQA/MQA recap pointer, KV quantization, attention sinks/StreamingLLM, H2O eviction) :: The chapter derives KV-cache size and names DeepSeek-V3's 'MLA' in a table without ever explaining it; MLA is the standard 2026 follow-up to any KV-cache question. GQA lives in DL ch05/22 (pointer nee
- [high] Prompt/context caching (Anthropic/OpenAI/Gemini cached prefixes) and its effect on few-shot vs fine-tune economics :: Caching changes the 'long prompts are expensive per query' argument used in the few-shot-vs-fine-tuning question (Q8) and is a standard production-design probe in 2026.
- [high] Test-time compute scaling results: compute-optimal test-time scaling (Snell et al. 2024), budget forcing (s1, 2025), overthinking/length control, sequential revision vs parallel sampling tradeoffs :: The section describes mechanisms qualitatively but has no quantitative results or decision framework for allocating inference compute — exactly the L6/L7 discriminator the chapter aspires to.

CORRECTNESS:
- Line ~1025 (MoE table): Switch Transformer row claims '1.6T total, 100B active, 128 experts'. The 1.6T configuration (Switch-C) used 2048 experts, and active parameters per token were far below 100B (T5-scale backbone). Both cells are wrong.
- Lines ~1476-1478 (Q7 strong answer) and Section 7.2: asserts o1-style models work via PRM step-scoring plus search. Unverified for OpenAI and contradicted by DeepSeek-R1's published ablations (PRM/MCTS reported as unsuccessful; GRPO with outcome-verifiable rewards is what worked). A candidate stating this as fact at a frontier lab would be corrected by the interviewer.
- Lines 252-262: 'Dai et al. (2023) showed a formal connection... equivalent to a single step of gradient descent' — overstated; the result holds under linear-attention approximations and its applicability to real ICL is contested. Q3's strong answer repeats 'formally showed'.
- Line 377: CoT effective only above '>60B parameters in the original findings' — Wei et al. 2022 reported emergence around ~100B (LaMDA 137B, PaLM 540B); 60B is not the paper's number.
- Line 772: broken self-referential cross-reference (chap:llms_icl cited as the RAG chapter).
- Minor: KV-cache formula (Eq. kv_cache) is correct and the 40GB/70B/128K example checks out (≈42GB with L=80, d_k=128, g=8, fp16) — but g is defined only as 'number of KV heads' without noting this assumes GQA; a pointer to GQA (DL ch05) would prevent misuse with h=64.

STALENESS:
- Context table stops at Gemini 1.5 1M (2024); no 2025-26 models or effective-context caveat beyond the RULER mention.
- o1/o3 are the only named reasoning models; no DeepSeek-R1, no distilled/open reasoners, no extended-thinking framing (chapter reads as written pre-Jan-2025).
- 'as of 2025' in the Mamba red flag betrays the vintage; Mamba-2/SSD only appears in follow-ups; hybrid examples stop at Jamba (2024).
- Instruction-tuning milestone table ends 2023 (Vicuna/Flan-PaLM); no synthetic-data-at-scale, no Tulu/open post-training recipes.
- MoE table ends at DeepSeek-V3 (Dec 2024); acceptable but no 2025 MoE wave (Llama 4, Qwen3-MoE) which interviewers now reference.
- '2--8 examples typical; beyond ~16 marginal' predates many-shot ICL results.
- Long-context solutions list omits attention sinks/StreamingLLM and KV-cache eviction/quantization, which superseded Landmark Attention in practice.

MUST-KNOW:
- Chinchilla compute-optimal scaling: L∝N^-0.076/D^-0.095, C≈6ND, D_opt≈20N — and why production models deliberately over-train (LLaMA-3 8B at 1875 tokens/param): train-once vs serve-forever economics.
- ICL mechanisms three ways: implicit Bayesian task inference, induction heads (Olsson 2022), task recognition — plus the Min et al. random-labels finding and what it implies demonstrations actually provide.
- CoT: why it works (decomposition, token-space working memory, serial-compute extension), when it hurts, unfaithful reasoning; self-consistency and zero-shot CoT.
- Test-time compute: ORM vs PRM, best-of-N with verifiers, and (must be added) RLVR/GRPO as how reasoning models are actually trained.
- MoE: top-K routing equation, capacity-vs-compute decoupling (Mixtral 47B/13B), expert collapse and the auxiliary balance loss αN·Σf_i·p_i, memory = total params, all-to-all serving cost.
- KV-cache size formula and the long-context failure mode (lost-in-the-middle U-curve, needle/RULER evaluation).
- Few-shot vs fine-tuning vs RAG decision framework (data volume, knowledge churn, latency/cost, grounding needs) and how they combine.
- SSM/Mamba in one breath: linear-time selective state, weaker precise recall/ICL, hybrids (Jamba) as the pragmatic answer.
- Prompt injection: direct vs indirect, defense-in-depth layers, why it is unsolved (capability-vs-security tension).
- Emergence debate: smooth perplexity scaling vs metric-induced discontinuities (Schaeffer 2023) — give the balanced answer.

IMPROVEMENTS:
- Rewrite Section 7 (Reasoning and Test-Time Compute) around three axes — (1) trained sequential reasoning via RL on verifiable rewards (R1/GRPO), (2) parallel sampling + verification (best-of-N, self-consistency), (3) search guided by PRMs — and stop presenting PRM+search as 'the mechanism' of o1/o3 (lines ~1476-1478 and Section 7.2 framing).
- Hedge the Dai et al. ICL-as-gradient-descent keyinsight (lines 252-262): the equivalence is derived for simplified/linear attention and was directly challenged (e.g., Shen et al. 2023); state it as a suggestive duality, not a formal result about real transformers.
- Cut Sections 9 (SSM) and 10 (MoE) to ~1.5 pages each with pointers to DL ch14, keeping only the NLP-specific angle (DeepSeek-V3 config, when-to-choose table); currently the router equation, balance loss, Mamba selectivity, and Jamba paragraph are duplicated nearly verbatim across volumes and will drift.
- Add a worked FLOPs/cost estimation example (e.g., 'estimate the cost to train a 70B on 15T tokens: 6ND = 6.3e24 FLOPs, at 40% MFU on H100s → GPU-hours → dollars') — Q1's follow-up asks candidates to do this and the body never demonstrates it.
- Fix the broken cross-reference at line 772: 'Chapter~\ref{chap:llms_icl} on RAG (Chapter 9)' points to this chapter itself; should be \ref{chap:rag}.
- Extend the context-window evolution table past Gemini 1.5 (2024) and add an 'advertised vs effective context' row citing RULER/NoLiMa-style findings; the table currently ends two years before the book's audience interviews.
- In the prompting decision diagram and advanced-prompting table, shrink the prompting-zoo (Reflexion, Generated Knowledge) and add a note that reasoning-native models absorb much of CoT's benefit (zero-shot CoT prompt engineering matters less when the model is RL-trained to think) — a nuance 2026 interviewers probe.
- Reconcile the lost-in-the-middle citation year with ch9 (2024 here, 2023 there).

# 08_rlhf_alignment.tex (RLHF and Alignment)

VERDICT: The best-written chapter of the three — the DPO derivation, Bradley-Terry treatment, and reward-hacking/Goodhart material are genuinely staff-grade — but it has one disqualifying currency hole: the entire GRPO/RLVR/reasoning-RL wave (Dec 2024 onward) is absent, and the chapter's own header says 'Current Alignment Recipe (2024--2025)'. It also near-duplicates DL ch20 in scope, structure, and six of eight interview questions. Add GRPO/RLVR as a first-class section, modernize the recipe, and dedupe against DL ch20; the mathematical core needs almost no work.

STAFF: The strongest staff signal in the volume: the DPO derivation with the Lagrangian sketch, the over-optimization curve, and the 'RM is the bottleneck' framing are exactly what L6/L7 loops probe, and the Q7 full-derivation question is a legitimate principal-level item. Weaknesses at staff altitude: almost no hyperparameter or scale numbers (no β magnitude, no KL budget, no preference-data scale beyond '50K-500K', no compute multiple of RLHF vs SFT — DL ch20 gives 3-5x), no debugging narrative (contrast DL ch20's 'reward plateaus while KL climbs' question), and the frozen-in-2024 frontier means an L7 candidate prepped only here would miss the RLVR reframing of Goodhart — the single most important conceptual update in alignment since DPO.

OVERLAP [heavy] with 20_reinforcement_learning.tex (near-total: pipeline, Bradley-Terry, PPO, KL, four-model table, DPO + variants table, reward hacking, CAI/RLAIF, PRM/ORM, plus 6/8 duplicated interview questions), 09_nlp_architectures.tex (Instruction Tuning and RLHF section: third copy of the same three-stage pipeline and RM loss), 23_safety_alignment.tex (safety guardrails, red-teaming, instruction hierarchy — adjacent, complementary)
STRONGER: NLP ch8 is the stronger alignment treatment: full DPO derivation (DL ch20 has only the keyinsight sketch), GAE and combined-reward mechanics, the over-optimization curve figure, KTO detail, RM evaluation question, and the alignment-recipe section. DL ch20's unique value is the RL-fundamentals ramp (MDP mapping to LLMs, REINFORCE variance, actor-critic), alignment tax, concrete β range (0.01-0.2) and compute multiples, and its debugging/estimation questions (RLHF memory estimation, reward-plateau-KL-climb diagnosis) — none of which ch8 has.
RECOMMENDATION: Make NLP ch8 the canonical alignment-pipeline home. DL ch20 should keep RL fundamentals + alignment tax + its estimation/debugging questions and cross-ref here for RM/DPO/CAI depth; shrink DL ch09's RLHF subsection to a pointer (it is a third, shallowest copy that will drift — its diagram already omits the KL anchor). Add GRPO/RLVR once, in ch8 (it is the alignment story), with a two-line pointer from DL ch20's PPO section. Merge the duplicated question banks: keep ch8's derivation-heavy set here, DL ch20's estimation/debugging set there, and delete the four questions that are near-verbatim twins. Import DL ch20's β numbers and 'RLAIF works when verification is easy' keyinsight into ch8 during the merge.

MISSING (crit/high):
- [critical] GRPO (group-relative advantage, critic-free PPO variant; DeepSeek-Math 2024, R1 2025) with the actual objective, plus DAPO/Dr. GRPO refinements :: GRPO is THE 2025-26 alignment interview question ('why does GRPO drop the value function? what does group normalization buy you? what biases does length-normalization introduce?'). A chapter teaching 
- [critical] RLVR — RL from verifiable rewards: rule-based rewards (answer checkers, unit tests, format rewards), R1-Zero emergence, cold-start SFT + reasoning-RL + rejection-sampling-SFT + preference-RL multi-stage recipes :: This is how reasoning capability is actually trained in 2025-26 and reframes the chapter's central Goodhart problem: verifiable rewards largely sidestep reward-model hacking, which is exactly the kind
- [high] Critic-free REINFORCE-family baselines for RLHF: RLOO, REINFORCE++ — and why PPO's clipping/critic may be unnecessary for bandit-like LLM episodes :: Standard 2025-26 follow-up to 'why PPO?'; interviewers use it to test whether the candidate understands what each PPO component actually buys in the single-step-reward LLM setting.
- [high] Reward model evaluation and modern RM forms: RewardBench, generative/LLM-as-judge reward models, self-rewarding LMs (Yuan et al. 2024) :: Q6 ('how do you evaluate an RM') is answered without the standard public benchmark or the generative-RM trend that dominates 2025 practice.
- [high] Online/iterative DPO and rejection-sampling pipelines as used in real recipes (Llama 2/3 rejection sampling, iterative rounds) — currently one parenthetical :: The offline-DPO limitation is stated but its standard industrial fix gets no mechanics; 'how would you close DPO's distribution gap' is a common L6 probe.

CORRECTNESS:
- KTO loss formula (lines ~715-722): oversimplified to a single σ(βr−z_ref) term with a weight w(y); the real objective treats desirable and undesirable examples with opposite-signed arguments and asymmetric weights — as stated, undesirable examples are handled incorrectly.
- Line ~894 (PRM section): 'the approach behind models like o1 and o3' — speculative attribution, contradicted by DeepSeek-R1's ablations for the one frontier reasoning model with a public training report; should be hedged to 'one approach to test-time search'.
- Table 'four models... approximately 280B parameters in GPU memory' for a 70B policy: only true if all four are 70B; commonly RM/critic differ in size — state the assumption.
- 'accuracy of 65--75% is typical' for RMs (Q6): fine as a rough number but should be tied to the dataset/agreement ceiling; on modern preference sets top RMs exceed this — low-severity.
- DPO derivation (Sections dpo/Q7): checked line by line — optimal-policy form, reward reparameterization, and Z(x) cancellation are all correct; no issues. Bradley-Terry, PPO clipped objective, and GAE formulas also verify.

STALENESS:
- Section header literally says 'The Current Alignment Recipe (2024--2025)'.
- No GRPO, RLVR, R1, rule-based rewards, or reasoning-RL anywhere — the chapter ends conceptually at Lightman et al. 2023 + o1 speculation.
- PRM-guided tree search presented as the mechanism 'behind models like o1 and o3' (line ~894) — pre-R1 folk theory now contradicted publicly.
- No RewardBench (2024), no generative reward models, no self-rewarding (2024), no Tulu-3-style open post-training recipes (late 2024).
- CAI section unaware of constitutional classifiers (2025) or the broader system-level-safety turn.
- DPO variants table is complete as of mid-2024 (IPO/KTO/ORPO/SimPO/cDPO/RSO) but nothing after.

MUST-KNOW:
- Three-stage pipeline (SFT → RM → PPO) with why each stage exists; why SFT alone is insufficient (comparison-vs-generation asymmetry, sycophancy, mode averaging).
- Bradley-Terry RM loss −log σ(r_w − r_l); RM = SFT backbone + scalar head; RM as the quality ceiling of the pipeline.
- RLHF objective E[r] − β·KL(π||π_ref); why the KL anchor exists, how β trades improvement vs hacking, how KL folds into per-token reward.
- PPO clipped surrogate + four-model memory footprint — and (to be added) GRPO's critic-free group-relative alternative and when verifiable rewards remove the RM entirely.
- Full DPO derivation: closed-form optimal policy → reward as log-ratio → partition-function cancellation in Bradley-Terry; DPO's implicit reward interpretation.
- DPO limitations (offline, no explicit reward, reference dependence) and variant selection: KTO for pointwise binary signals, SimPO/ORPO reference-free, iterative DPO for distribution gap.
- Reward hacking as Goodhart's law: verbosity/sycophancy/formatting manifestations, the Gao et al. proxy-vs-true over-optimization curve, defenses (KL, ensembles, iterative RM retraining, length penalties).
- ORM vs PRM: credit assignment, Lightman et al. results, PRM-guided search — and the honest 2026 caveat that outcome-verifiable RL (not PRM search) trained the current reasoning models.
- Constitutional AI: SL-CAI (critique-revise-SFT) vs RL-CAI (RLAIF), why principles beat a black-box RM for auditability, RLAIF reliability as a function of verifiability.
- Alignment tax and over-refusal (via DL ch20 or added here).

IMPROVEMENTS:
- Add a 'Beyond PPO: GRPO and Verifiable Rewards' section with the GRPO advantage formula (group-mean baseline, no critic → two models in memory, not four), a PPO-vs-GRPO-vs-DPO comparison table, and one new interview question; rewire the PRM section's o1/o3 claim to reflect R1's published evidence (PRM/MCTS as failed directions there).
- Retitle and rewrite 'The Current Alignment Recipe (2024--2025)' as a 2026 recipe: SFT → preference optimization (DPO/RLHF) → reasoning RL with verifiable rewards → safety training, noting which stages use which reward types.
- Fix or annotate the KTO loss (lines ~715-722): as written the single-formula version does not distinguish desirable vs undesirable examples correctly — actual KTO uses σ(β(r−z_ref)) for desirable and σ(β(z_ref−r)) for undesirable with asymmetric λ_D/λ_U weights; the current formula would mis-train undesirable examples.
- Give β magnitudes and KL-budget numbers in the PPO section (DL ch20 has 0.01-0.2 and adaptive-KL targeting; this chapter discusses β qualitatively only) — one of the few places DL ch20 outclasses this chapter.
- Qualify the '280B parameters in GPU memory' four-model claim: RM and critic need not be policy-sized (critic often initialized from RM; RM can be smaller), and note optimizer states/activations dominate beyond parameters — as written it invites a false-precision answer.
- Split RLAIF quality claims by task type: 'comparable to RLHF' holds best for harmlessness-style objectives (Bai et al., Lee et al.); for nuanced helpfulness human feedback still matters — the keyinsight in DL ch20 ('RLAIF works best when verification is easy') is the better framing; import it.
- Add one debugging-flavored question of the kind DL ch20 has ('reward plateaus while KL climbs — what is happening?') — this chapter's question set is more conceptual/mathematical and lighter on production diagnosis.
- Dedupe interview questions against DL ch20: six of eight here (pipeline walkthrough, reward hacking, DPO-vs-RLHF, CAI, RM quality, PRM-vs-ORM) have direct counterparts there; keep the deeper version in one volume and cross-ref.

# 09_rag.tex (Retrieval-Augmented Generation)

VERDICT: The most production-credible chapter of the three — real back-of-envelope math, a correct three-layer evaluation decomposition, and honest BM25/hybrid advocacy — and its retrieval-paradigms section (DPR/E5/BGE training details, SPLADE, ColBERT, distillation) is clearly the strongest RAG treatment across both volumes. Gaps are 2024-25 currency (GraphRAG, contextual retrieval, multimodal/ColPali, prompt-caching economics, retrieval security) and heavy duplication with DL ch24's RAG half and DL ch07's ANN internals, including ~8 near-twin interview questions. Consolidate here, modernize, and fix the Q10 GPU arithmetic.

STAFF: Consistently L5-L6 with genuine L6 peaks: the back-of-envelope index sizing, the three-layer failure decomposition, the 'chunk size is the most undertuned hyperparameter' warningbox, and the debugging question (Q9) are the kind of production judgment loops reward. What holds it below L7: no cost model in dollars (token economics never computed despite a scaling question), the one broken GPU calculation in the flagship scaling answer, security treated as a single clause, no failure-rate priors ('which layer usually fails'), and no ops lifecycle (reindexing, embedding migration, index consistency during updates gets only a follow-up mention). The L7 ladder bullets again promise material (multi-granularity indexing, adaptive re-ranking depth) the body never delivers.

OVERLAP [heavy] with 24_long_context_rag.tex (RAG half: pipeline diagram, chunking table, HyDE, query decomposition, cross-encoder/ColBERT, RAGAS, BM25-hybrid warning, long-context-vs-RAG table, and ~8 of 10 interview questions are near-twins), 07_similarity_metric_learning.tex (HNSW/IVF/PQ internals with deeper memory math than this chapter, plus a two-tower + ANN system design question covering similar scaling ground), 06_embeddings.tex (embedding models/contrastive training, adjacent)
STRONGER: NLP ch9 is the stronger RAG treatment overall: retrieval-paradigm depth (DPR/E5/BGE training details, hard-negative mining, SPLADE, distillation), vector-DB landscape, advanced patterns (Self-RAG, agentic RAG, context compression, citation), and richer production math. DL ch24 wins on exactly two fronts: the long-context-extension half (RoPE/PI/NTK/YaRN — unique, keep there) and a few individual artifacts (the ColBERT storage computation, the cost-per-query question, the 70%-retrieval-failure keyinsight, the reasoning-query diagnosis question). DL ch07 has the deeper ANN internals (HNSW graph-overhead math, IVF-PQ byte-level sizing).
RECOMMENDATION: Make NLP ch9 the canonical RAG home. Shrink DL ch24's RAG half to a ~3-page bridge (pipeline recap + long-context-vs-RAG decision + cost question) pointing here, keeping DL ch24 canonical for context extension; port its three best artifacts (cost question, failure-prior keyinsight, reasoning-diagnosis question) into ch9 before cutting. Slim ch9's ANN-algorithm subsection to decision-level tables + pointer to DL ch07 for internals (DL ch07's memory math is better; don't maintain two HNSW diagrams). Dedupe the interview-question twins across ch9/DL 24 — currently a candidate doing both volumes answers 'design enterprise RAG for 10M documents' and 'cross-encoder vs ColBERT vs bi-encoder' twice with subtly different numbers (e.g., ColBERT 20-40x vs 125x), which is worse than either alone.

MISSING (crit/high):
- [high] Contextual retrieval (Anthropic 2024: LLM-prepended chunk context + contextual BM25/embeddings) and late chunking as first-class techniques (late chunking currently only a follow-up mention) :: These are the two best-known 2024-25 answers to the chunk-context-loss problem the chapter spends a whole section on; interviewers explicitly name-check contextual retrieval and expect candidates to c
- [high] GraphRAG / knowledge-graph-augmented retrieval (Microsoft 2024): entity/community summarization for global 'summarize the whole corpus' queries :: Standard 2026 follow-up to 'when does vanilla RAG fail' — multi-hop and corpus-global questions; the chapter covers multi-hop iterative retrieval but not the graph-index alternative every enterprise s
- [high] Retrieval security: indirect prompt injection via poisoned documents, permission-aware retrieval/ACL filtering, multi-tenant index isolation :: 'Access permissions' appears once in a metadata-filtering bullet; in 2026 enterprise RAG interviews, document-level authz and injection-via-retrieved-content are core design requirements, and this vol

CORRECTNESS:
- Q10 (lines ~1602-1604): '200 GPU-seconds per second; deploy a pool of ~4-8 GPU instances' — arithmetic does not close without an unstated batching assumption; the memory math (50M chunks × 1024d × fp32 ≈ 200GB → ~50GB int8) and concurrency math (1K QPS × 0.5s = 500 slots) are correct, making the one bad line stand out.
- ColBERT index inflation '20-40x' (line ~307) contradicts DL ch24's 125x computation; both are defensible under different compression assumptions but the set contradicts itself.
- Line ~319: 'the cost is O(n) per document' — misleading notation for 'one forward pass per candidate, O(N) over candidates'.
- HNSW properties (O(log N) search, recall@10 >95% tuned), RRF k=60, InfoNCE with in-batch negatives, DPR/SPLADE/ColBERT citations and MaxSim formula: all spot-checked correct.
- Minor: 'Inverted index lookup is O(1) per term' ignores posting-list-length-dependent scoring cost; harmless simplification but an IR-literate interviewer may poke it.

STALENESS:
- No GraphRAG (2024), no contextual retrieval (2024), no ColPali/multimodal retrieval (2024), no late-chunking promotion beyond a follow-up line.
- Long-context references stop at 'Claude, Gemini 1.5'; RAG-vs-long-context table predates 1M-context commoditization and prompt caching.
- Vector-DB landscape table is 2024-vintage but acceptable (vendor churn argues for keeping it short); 'Matryoshka embeddings' mention is current.
- Agentic RAG examples (Perplexity, ChatGPT browsing) are fine but predate deep-research-style multi-step report agents that 2026 interviewers reference.
- Evaluation section: RAGAS-centric; no mention of 2025-era grounded-generation evals or citation-precision/recall metrics now standard in production scorecards.

MUST-KNOW:
- Two-phase architecture: offline (ingest → chunk → embed → ANN index) and online (encode → retrieve top-K → re-rank to top-k → assemble → generate → cite).
- BM25 vs dense retrieval failure-mode complementarity; hybrid retrieval with RRF as the production default; SPLADE as the learned-sparse middle ground.
- Bi-encoder vs cross-encoder vs ColBERT: the quality/latency/storage triangle and the retrieve-many-rerank-few pattern with real latency numbers.
- Chunking as a tuned hyperparameter: size/overlap tradeoffs, recursive and document-aware strategies, parent-child retrieval to decouple retrieval precision from generation context.
- ANN internals at decision level: HNSW vs IVF vs PQ tradeoffs (recall/latency/memory), metadata pre- vs post-filtering.
- Three-layer evaluation: retrieval (Recall@K, MRR, NDCG), generation (relevance/completeness), faithfulness (NLI, LLM-judge, citation verification); RAGAS; diagnose-the-layer-first debugging.
- Query transformation toolbox: rewriting, decomposition, HyDE with its failure modes; multi-hop and agentic retrieval; Self-RAG's when-to-retrieve decision.
- Lost-in-the-middle and context-assembly ordering; retrieve-fewer-better via aggressive re-ranking.
- RAG vs fine-tuning vs long context: knowledge-vs-behavior distinction, freshness, attribution, corpus size — and how they combine (fine-tune + RAG as default production pattern).
- Scale math on demand: index memory sizing, QPS per HNSW replica, re-rank GPU budgeting, generation concurrency, caching tiers.

IMPROVEMENTS:
- Fix Q10's re-ranking arithmetic (lines ~1602-1604): 200ms × 1000 QPS = 200 GPU-seconds of work per second, then the answer deploys '4-8 GPU instances' — a silent 25-50x gap. Either state the batching speedup assumption explicitly (batched cross-encoder scoring amortizes to ~5-10ms per candidate-set) or fix the fleet size; as written, an interviewer running the numbers fails the answer the chapter models as strong.
- Reconcile ColBERT storage overhead with DL ch24: this chapter says indices are '20-40x larger', DL ch24 computes 125x for the same setup; the truthful answer is compression-dependent (ColBERTv2 residual compression ≈ 6-10x, uncompressed 2 orders). Pick one framed number in one volume.
- Tighten the cross-encoder complexity notation (line ~319): 'cost is O(n) per document' conflates per-corpus O(N) forward passes with per-pair sequence-length cost; also harmonize 'impractical beyond ~1000 candidates' (Q8) with 'typically 20-100' (warningbox).
- Unify the lost-in-the-middle citation year within the chapter (Liu et al. 2023 at line ~1013/1338 vs 2024 elsewhere and in ch7).
- Add a short 'when NOT to use RAG' / failure-taxonomy subsection (corpus-global questions, reasoning-heavy queries, tiny static corpora) — DL ch24 has a 'fails for reasoning queries' diagnosis question this chapter's set lacks; port the idea, not the duplicate.
- Import DL ch24's one genuinely better artifact — the per-query cost-estimation question (RAG vs 128K long-context token economics) — into this chapter's question set, updated for cached-prompt pricing.
- Add RRF weighting/tuning nuance (when to weight dense vs sparse lists, alpha-blended hybrid scores as the pgvector/Weaviate-style alternative to rank fusion).
- State the 70%-retrieval-failure heuristic (DL ch24's keyinsight) or your own calibrated version in the three-layer evaluation section — the decomposition is here but no prior over which layer usually fails.

# 10_evaluation_metrics.tex — Evaluation, Metrics, and Decoding (NLP volume)

VERDICT: Strong chapter and the rightful canonical home for NLP/generation evaluation across the two volumes. The BLEU/ROUGE/perplexity derivations, metric-skepticism framing, LLM-as-judge bias catalog, and contamination treatment are genuinely interview-grade. But it is frozen in mid-2024: no reasoning-model/agentic evaluation, no length-controlled judging, no error bars on benchmark scores, and the benchmark table headlines saturated 2020-22 datasets. It also contains one real math error (the temperature worked example) and duplicates material owned by NLP ch12 and DL ch18/ch22. Fix the errors, refresh ~15% of the content, trim the duplication — this becomes an excellent chapter.

STAFF: Solid L5 throughout, genuine L6 in the pipeline-design and contamination questions (Q7/Q8 are the best in the chapter). What keeps it from L6/L7 as a whole: no operational scale numbers (judge-human agreement rates, eval-suite sizes/costs, sample sizes to resolve win-rate deltas), no error-bar discipline on benchmark scores, and zero coverage of the eval problems a 2026 staff candidate actually owns — reasoning-model variance, agent trajectory eval, judge drift across model versions. The L7 ladder rungs gesture at the right things (Goodhart, meta-evaluation) but the chapter body doesn't equip the candidate to say them with numbers.

OVERLAP [partial] with 18_evaluation_metrics.tex, 22_inference_optimization.tex, 09_nlp_architectures.tex, 24_long_context_rag.tex
STRONGER: Split by topic. Generation/NLP metrics: NLP ch10 is far stronger — DL ch18's §'NLP and Generation Metrics' (~90 lines, lines 276-361) is a compressed duplicate with no derivations, no faithfulness, no judge-bias depth. Classification metrics: DL ch18 is stronger (ROC/PR-AUC, calibration/ECE, significance testing, sample-size math) vs NLP ch10's bare P/R/F1. Speculative decoding: DL ch22 is much stronger (cost model, preconditions, debugging Q) and NLP ch12 also covers it with Medusa/EAGLE — NLP ch10's version is the weakest of three. Decoding strategies: NLP ch10 is stronger than DL ch09's 55-line 'Generation Strategies' sketch. The chatbot-eval design interview question appears in both volumes (NLP ch10 Q8, DL ch18 final Q) with near-identical three-tier answers.
RECOMMENDATION: Declare canonical homes and cut cross-duplication: (1) NLP ch10 = canonical for generation metrics, human eval, LLM-as-judge, benchmarks, decoding strategies — DL ch18 should shrink its NLP-metrics section to the comparison table + cross-ref; (2) DL ch18 = canonical for classification/ranking/regression metrics, calibration, and statistical significance — NLP ch10 trims its §4 to NER-specific content + pointer; (3) DL ch22 (with NLP ch12) = canonical for speculative decoding — NLP ch10 keeps a 5-line pointer; (4) merge the duplicated chatbot-eval interview question, keeping ch10's version. This saves the candidate roughly 8-10 redundant pages and removes the BLEU-scale inconsistency (0-1 vs 0-100) between volumes.

MISSING (crit/high):
- [critical] Evaluating reasoning models and agents: pass@k vs maj@k/cons@64, variance across sampled runs, token-budget-matched comparisons, eval temperature choices (greedy decoding degrades long-CoT models — DeepSeek-R1 explicitly recommends T≈0.6), verifier/RLVR-style checkable-answer evals, and agentic benchmarks (SWE-bench Verified, τ-bench, GAIA, BrowseComp, Terminal-Bench) :: This is the single most probable eval probe in a 2026 frontier-lab loop. The chapter's decoding advice ('deterministic tasks → T≤0.3') is actively inverted for reasoning models, and a candidate who ev
- [high] Statistical rigor for LLM eval deltas: paired bootstrap/clustered standard errors over prompts, CIs on benchmark scores (Anthropic's 'Adding Error Bars to Evals', 2024), how many prompts are needed to resolve a 2-point win-rate difference :: The chapter tells candidates to distrust metrics but never shows how to decide whether a 1.5-point MMLU delta is noise. DL ch18 has classical significance testing but nothing LLM-specific; interviewer
- [high] Length-controlled and style-controlled judging: AlpacaEval 2.0 LC win rate, LMSYS style control, Arena-Hard-Auto :: The chapter names verbosity bias as the #1 judge bias but omits the standard 2024-25 mitigations that every practitioner now uses; a follow-up ('how do you fix verbosity bias?') has a canonical answer
- [high] Benchmark landscape refresh: GPQA Diamond, MMLU-Pro, IFEval, LiveCodeBench, AIME/HMMT, Humanity's Last Exam, MMMU; and the 2025 Chatbot Arena critique ('The Leaderboard Illusion': private-variant testing, sampling asymmetries) :: The main table (line ~681) leads with HellaSwag/WinoGrande/TruthfulQA/BigBench/MT-Bench — all retired or saturated in frontier reporting. Reciting that table verbatim in 2026 reads as stale; the Arena

CORRECTNESS:
- Temperature example (lines ~819-822) is numerically wrong for all three temperatures given the stated logits, and the T=1 row sums to 0.90 — see improvements for correct values.
- ROUGE-L discussion (line ~298): 'β is typically set large (e.g., 1.2) to weight recall more heavily' — 1.2 is near-balanced, not large. The original ROUGE configuration effectively sets β→∞ (recall-only), and common implementations (rouge-score) use β=1. The parenthetical example undermines the claim.
- Repetition penalty (line ~866): 'divide the logit by θ>1' is only correct for positive logits; CTRL (Keskar et al. 2019) divides positive logits and multiplies negative logits by θ. As written, the formula makes already-unlikely repeated tokens more likely.
- Minor: cross-entropy is first defined in bits (log₂, line 40) while the perplexity mathresult switches to nats (ln) and then invokes 2^H 'in bits' — internally consistent but a candidate rehearsing from it can easily misquote; state both conventions once, cleanly.
- Minor: top-p set definition (line ~849) uses min over subsets without specifying minimality by cardinality/probability ordering; the prose fixes it but the displayed math is sloppy.
- Verified correct: BLEU clipped-precision/BP formulas, ROUGE-N recall formula, BERTScore P/R/F definitions, pass@k unbiased estimator 1 − C(n−c,k)/C(n,k), Cohen's kappa, GPT-2 WikiText-103 PPL ≈ 18 (actual 17.48), benchmark sizes (HumanEval 164, MBPP 974, TruthfulQA 817, WinoGrande 44K).

STALENESS:
- Benchmark table (line ~681) headlines MMLU/HellaSwag/WinoGrande/ARC/TruthfulQA/BigBench/MT-Bench — the 2022-23 canon. GPQA, MMLU-Pro, LiveBench appear only as one-line asides; SWE-bench, IFEval, AIME, HLE are absent entirely.
- MT-Bench presented as a current tool (line ~587); in practice superseded by Arena-Hard-Auto and AlpacaEval 2 LC by 2025.
- Chatbot Arena limitations (line ~577) list only population/prompt-distribution issues; the 2025 'Leaderboard Illusion' findings (private variant testing, unequal sampling) and LMSYS's style-control response are missing.
- LLM-as-judge section has no dedicated judge/reward models (Prometheus 2, Skywork), no G-Eval, no panel-of-judges — the mitigation list stops at 2023-era techniques.
- Decoding section omits min-p (standard in vLLM/llama.cpp since 2024) and treats speculative decoding speedup as 2-3x (EAGLE-2/3-era stacks report higher; fine as conservative, but worth a note).
- Faithfulness metrics list (FactCC 2020, SummaC 2022, QAFactEval) is pre-LLM-judge-era; in 2026 production the default is LLM/NLI hybrid claim verification pipelines with citation checking — the chapter's own summarization Q gets closer than the section body.

MUST-KNOW:
- Perplexity = exp(avg NLL); relation to cross-entropy in bits vs nats; tokenizer dependence and bits-per-byte for cross-model comparison
- BLEU derivation from scratch: clipped modified n-gram precision, geometric mean, brevity penalty — plus its 6 limitations and SacreBLEU reproducibility
- ROUGE vs BLEU orientation (recall vs precision) and why ROUGE is structurally blind to faithfulness/hallucination
- Learned metrics: BERTScore mechanics and COMET as the MT standard; why they beat surface overlap
- Entity-level strict F1 for NER (boundary + type) vs token-level; macro vs micro F1 under imbalance
- LLM-as-judge biases (verbosity, position, self-preference) and mitigations: position swapping, rubrics, length control, human calibration
- Benchmark contamination and saturation; private held-out sets, dynamic benchmarks (LiveBench), functional testing
- Decoding zoo: temperature/top-k/top-p mechanics, neural text degeneration, beam + length normalization, constrained/grammar decoding for structured output
- pass@k with the unbiased estimator for code; why execution-based eval is the only ground truth for code
- Three-tier eval pipeline design (automatic → human → production) with statistical significance on the deltas

IMPROVEMENTS:
- Fix the temperature worked example (lines ~819-822): for logits [2.0, 1.0, 0.5] the true probabilities are [0.63, 0.23, 0.14] at T=1, [0.84, 0.11, 0.04] at T=0.5, and [0.48, 0.29, 0.23] at T=2.0 — not the printed [0.50, 0.27, 0.13] / [0.71, 0.21, 0.08] / [0.39, 0.33, 0.28]. The printed T=1 values do not even sum to 1. An interviewer checking arithmetic here would flag the book.
- Shrink the speculative decoding subsection (lines ~895-918) to ~5 lines plus cross-refs: NLP ch12 already covers it in more depth including Medusa/EAGLE, and DL ch22 owns the cost model and failure modes. Three full treatments of the same algorithm across the set is the clearest duplication in either volume.
- Cut the generic precision/recall/F1 and macro/micro exposition (lines ~403-440) to a one-paragraph reminder with a pointer to DL ch18, keeping only the NLP-specific material (entity-level NER F1, SemEval partial matching, multi-label). DL ch18's classification section is stronger (adds ROC/PR-AUC, calibration) and is the canonical home.
- Deduplicate the chatbot-evaluation design question: ch10 Q8 (e-commerce support chatbot, line ~1505) and DL ch18's 'Design an evaluation framework for an LLM-based chatbot' (L7) are near-identical three-tier answers. Keep one canonical version (ch10's is more detailed) and turn the other into a pointer with a differentiating twist.
- Add a short 'evaluating RAG' pointer paragraph: this is the volume's evaluation chapter, but RAG eval (RAGAS, retrieval Recall@K/MRR/NDCG, faithfulness) lives only in ch09; a reader who starts here won't find it.
- Harmonize the BLEU scale with DL ch18: this chapter says 'scores above 0.40 are very good' (0-1 scale); DL ch18 says 'above 30 understandable, above 50 high quality' (0-100). Pick one convention and state the other.
- In the decoding key-insight (line ~940), add the reasoning-model caveat: greedy/T=0 causes repetition and degraded accuracy in long-CoT models — the 'right' default now depends on whether the model is a reasoner.
- Give the L6/L7 ladders operational numbers: human-human judge agreement (~80% on chat prefs), GPT-4-judge/human agreement (~85% on MT-Bench), typical eval-suite sizes and cost, prompts needed to detect a given win-rate delta. This is what separates 'has read about evals' from 'has run an eval program'.

# 11_practical_nlp_tasks.tex — Practical NLP Tasks and Applications (NLP volume)

VERDICT: A well-organized applied tour whose real asset is judgment — the baseline-first escalation ladder, constraint-driven decision tables, and error-compounding math are exactly what applied-NLP loops reward. But the chapter's center of gravity is 2019-2022: TOD pipelines and BERT span-extraction QA get TikZ figures while agents/function-calling get a bullet list; the LLM cost arithmetic is off by ~2 orders of magnitude at 2026 prices; and several 'LLM disadvantages' (inconsistent output structure) were solved by constrained decoding the sibling chapter itself teaches. Two LaTeX bugs (column-spec mismatches, a self-referential \ref) need fixing. Recommend a moderate revision: compress the classical 30%, modernize the LLM tier, keep the decision frameworks intact.

STAFF: The judgment layer (decision tables, 'always start simple', cascade architectures, error compounding, Q6/Q10) is genuine L6 material and better than most published prep. The task sections themselves read L4/L5: heavy mechanism description of legacy stacks, thin on scale numbers (no QPS/throughput/cost anchors anywhere except one stale price), and no 2026 agent-era framing — a staff candidate at a frontier lab will be pushed on 'why not just an LLM with tools + distillation?' for nearly every task here, and the chapter under-arms them for that exchange. The L7 ladder rungs are sensible but generic (feedback loops, org considerations) rather than carrying hard numbers or war stories.

OVERLAP [partial] with 19_decision_frameworks.tex, 24_long_context_rag.tex, 09_nlp_architectures.tex, 18_evaluation_metrics.tex, 17_production_systems.tex
STRONGER: NLP ch11 is stronger on everything task-specific: its constraint-driven decision tables and per-task classical/neural/LLM tiers have no real counterpart in DL ch19, whose NLP decision tree (lines 85-118) is a four-leaf sketch (BERT vs BERT+CRF vs T5 vs GPT) and whose latency-Pareto figure uses illustrative made-up numbers. For RAG system design, DL ch24 and NLP ch09 are both stronger and more detailed than ch11's open-domain-QA section and Q4 (chunk sizes, HyDE, ColBERT, RAGAS, long-context-vs-RAG tradeoff) — ch11's Q4 re-derives the same chunking/hybrid/rerank recipe a third time. DL ch09's 'when to use encoder/decoder/enc-dec' overlaps ch11's approach selection but from the architecture side; complementary rather than duplicative. DL ch17/ch18 overlap is light (deployment monitoring, chatbot eval) and acceptable.
RECOMMENDATION: Keep both chapters with sharpened lanes: NLP ch11 = canonical for task-level approach selection and per-task design patterns; DL ch19 = canonical for cross-modality architecture/loss/sizing decisions, and its NLP-specific decision tree should be reduced to a pointer at ch11 (it is currently both redundant and cruder). For RAG: canonical home is NLP ch09 (with DL ch24 owning long-context-vs-RAG); ch11 should keep the QA taxonomy and the Q4 interview question but compress Q4's ingestion/retrieval boilerplate to cross-refs, keeping only its unique content (permissions, freshness, no-answer pathway). Net saving: ~4-5 duplicated pages and one fewer place for chunk-size numbers to drift out of sync (ch11 says 256-512 tokens; verify against ch09/DL ch24 when consolidating).

MISSING (crit/high):
- [critical] Function calling / structured outputs as the modern extraction and dialogue interface: JSON-schema-constrained generation, tool schemas, agent loops with guardrails as the 2026 replacement for the NLU→DST→policy→NLG pipeline :: In 2026 'design a customer-support system' is asked as an agent-design question (tools, memory, escalation, injection defenses), not a TOD-pipeline question. The chapter's LLM-IE section still lists '
- [high] Long-context LLMs as the first answer for long-document summarization/QA: 128K-1M contexts, long-context degradation (lost-in-the-middle), when map-reduce still wins (cost, > context, parallelism) :: The long-doc section (line ~694) offers truncation/chunking/Longformer-LED as the toolkit. In 2026 the first-order answer to 'summarize a 100-page document' is 'it fits in a 200K window — the question
- [high] Zero-shot / open-type NER: GLiNER, plus honest treatment of LLM NER failure modes (boundary inconsistency, schema drift) and LLM-assisted annotation pipelines :: GLiNER-class models are the 2024-26 default for schema-flexible NER at BERT-like cost — it dissolves the chapter's 'rapidly changing label space → LLM only' table cell and is a strong L6 differentiato
- [high] Modern encoder story: DeBERTa-v3 as the accuracy leader, ModernBERT (Dec 2024) as the efficient default, SetFit/embedding+head classifiers for few-shot :: The chapter says 'BERT/RoBERTa is the modern default' — in 2026 naming DeBERTa-v3/ModernBERT is the cheap signal that the candidate is current; SetFit fills the 100-1K-examples cell of the decision ta
- [high] Updated LLM economics: per-million-token pricing tiers (small models at ~$0.05-0.60/1M input), batch APIs at ~50% discount, and how this changes the 'distill to BERT' calculus for offline extraction jobs :: The chapter's headline cost example ('LLM at $0.01 per 1K tokens... millions of dollars per day', line ~1735) is stale in both unit convention and magnitude; a staff candidate is expected to do this a

CORRECTNESS:
- Line ~131 and ~1072: tabular column-count mismatches (6-col spec with 5 columns; 5-col spec with 4) — build/rendering defect.
- Line ~610: \ref{chap:practical_tasks} self-reference where the RAG chapter (ch09) is meant.
- Line ~263: BIOES adjacent-entity example mislabeled/garbled ('[New York] [Times] as ORG-LOC vs. a single ORG').
- Line ~293-301: CRF key-insight overstates BERT+CRF gains and wrongly cites nested entity types as where CRF helps most — linear-chain CRFs cannot encode nesting.
- Line ~784: mBART described as enabling 'zero-shot translation between pairs not seen during fine-tuning' — mBART is fine-tuned per pair; the zero-shot claim belongs to multilingual NMT/mBART-50-style many-to-many training. Minor but citable.
- Verified correct: extractive-QA loss decomposition and SQuAD 2.0 null-span thresholding, distant supervision (Mintz 2009) and Lin et al. 2016 attention denoising, error-compounding arithmetic (0.9×0.85=76.5%; 0.9⁴≈65.6%), NB parameter count O(|V|×|C|), Viterbi O(T·K²), back-translation attribution (Sennrich 2016), Pegasus GSG description.

STALENESS:
- LLM pricing '$0.01 per 1K tokens' (line ~1735) — wrong unit convention and ~2 orders of magnitude high for the small-model tier doing this work in 2026.
- 'Inconsistent output structure' listed as an LLM extraction disadvantage (line ~1052) — solved by JSON-schema/grammar-constrained decoding, which is now the production default and is taught in the sibling chapter (ch10).
- Long-document toolkit stops at Longformer/LED 4K-16K (line ~706) — pre-dates commodity 128K-1M contexts; map-reduce survives as a cost/scale tactic, not the primary method.
- Task-oriented dialogue pipeline presented as the primary architecture with LLM systems as an afterthought — inverted relative to 2026 practice (LLM + function calling + guardrails first, pipeline as legacy vocabulary).
- Spider/WikiSQL as 'standard benchmarks' and TAPAS as the table-QA method — BIRD/execution-accuracy and LLM+SQL/code-interpreter approaches replaced both.
- 'LLMs underperform dedicated NMT' as a quality claim — contradicted by WMT23/24 results for high-resource pairs; only the cost/latency half of the argument survives.
- Coreference stops at Lee et al. 2017 span-ranking — fine as the anchor, but a one-line note that LLMs now handle coref implicitly in long contexts would prevent the section reading as a museum piece.
- Text-classification decision table: 'LLM: Overkill at >1K examples' needs nuance — frontier zero-shot now matches or beats fine-tuned BERT on many semantically hard tasks; the overkill argument is cost, not accuracy.

MUST-KNOW:
- Baseline-first escalation ladder (TF-IDF+LR → DistilBERT/BERT → LLM) with the constraint set (data, latency, cost, schema churn) that drives each step — and the ability to quantify each tier
- BIO/BIOES tagging, entity-level strict F1, why a CRF layer exists (invalid-sequence prevention, Viterbi) and when it no longer pays
- Extractive vs abstractive summarization tradeoff; intrinsic vs extrinsic hallucination; extract-then-polish as the high-stakes pattern
- RAG-based QA design: chunking, hybrid retrieval + rerank, citations, permission-aware retrieval, 'I don't know' pathway
- Distant supervision and its false-labeling problem; multi-instance learning as mitigation
- LLM-vs-fine-tuned decision calculus including distillation of LLM labels into a small serving model (the data flywheel)
- Pipeline error compounding math and confidence-gated human-in-the-loop mitigation
- TOD vocabulary (intent/slot, belief state, policy, NLG) and how an LLM+function-calling agent collapses it — plus what controllability is lost
- When an LLM is overkill: language ID, regex extraction, high-QPS spam, dedup — with cost/latency arithmetic at current prices

IMPROVEMENTS:
- Fix the LaTeX column-spec mismatches: table at line ~131 declares 6 columns (@{}lccccc@{}) but rows have 5 cells; table at line ~1072 declares 5 (@{}lcccc@{}) with 4-cell rows. Both render with a phantom trailing column.
- Fix the self-referential cross-reference at line ~610: '(Chapter~\ref{chap:practical_tasks} refers to Chapter~9 for RAG details)' points at this chapter's own label instead of the RAG chapter's, and the sentence reads as a placeholder.
- Rebalance page weight: cut TextCNN to two sentences (dead in 2026 loops), compress the TOD pipeline + TikZ figure by half (breadth-insurance vocabulary only), and shrink the BERT extractive-QA figure/derivation — reallocate to an 'LLM-era task stack' section (structured outputs, agents, distillation flywheel, batch inference).
- Repair the BIOES example (line ~263): '[New York] [Times] as ORG-LOC vs. a single ORG' is garbled — the standard motivation is separating adjacent same-type entities (B after E marks a boundary); as written it mislabels 'New York Times' and confuses the reader.
- Soften/correct the CRF key-insight (line ~293): the 0.5-1.5 F1 gain claim is at the optimistic end for BERT+CRF (published results often show ≤0.5 or no gain), and 'largest gains on nested types' is wrong — flat linear-chain CRFs cannot represent nested entities at all; nested NER needs span-based models, which the chapter correctly says 40 lines later.
- Update Q10's cost arithmetic to per-million-token pricing with 2026 tiers, and add the counter-case: when a batched small LLM beats training a bespoke classifier (low volume, changing schema, no labeling budget).
- Add one quantitative serving anchor for the model ladder: e.g., DistilBERT ~2-4ms/example batched on a T4, BERT-base ~10ms, small-LLM API ~300-800ms + $X/M tokens — the decision tables assert latency classes without ever grounding them.
- In Q4 (internal-docs RAG), trim the ingestion/retrieval details that triplicate NLP ch09 and DL ch24, and differentiate the answer with what those chapters lack: permission-aware retrieval and freshness/re-indexing, which are this question's actual point.

# 12_production_nlp.tex)

VERDICT: Highest-ROI chapter in the NLP volume and mostly well-executed (real formulas, worked memory math, good interviewq ladders), but roughly 70% of it re-derives material the DL volume owns (DL ch22 serving, ch14 quantization/distillation, ch09 tokenization summary, ch17 MLOps), with numeric drift between volumes and several outright wrong numbers — including in the flagship 10K-QPS strong answer. Currency stops at mid-2024: no MLA, no chunked prefill/disaggregated serving, no SGLang/FA-3/FP4, no reasoning-model serving. Fix the numbers, de-duplicate against DL 22/14/17, and modernize the serving section; keep tokenization + multilingual as this chapter's canonical core.

STAFF: Mostly reads L5-L6 in the right way — formulas plus concrete GB/ms numbers, and the interviewq ladders (esp. Q2 KV-cache, Q4 quantization) are genuinely staff-calibrated. Three things cap it below staff: (1) the flagship system-design answer's numbers are wrong by 3-10x, which is fatal in a chapter whose thesis is 'think in numbers'; (2) it asserts bandwidth-bound decode rather than deriving it — the roofline/crossover derivation that DL ch22 performs is exactly the L6→L7 discriminator; (3) sections 7-8 (multilingual mitigation bullets, MLOps) decay into L4-L5 checklists without quantities. The tokenization sections, by contrast, are the best treatment in either volume.

OVERLAP [heavy] with 22_inference_optimization (KV cache, PagedAttention, continuous batching, speculative decoding, quantization-for-serving, TP-vs-PP table, serving stack), 14_efficient_architectures (uniform-quantization formula, PTQ/QAT, GPTQ/AWQ/SmoothQuant/FP8, pruning, distillation loss + compression decision guide), 09_nlp_architectures (tokenization section: BPE/WordPiece/Unigram/SentencePiece), 17_production_systems (drift detection, A/B testing, monitoring, deployment patterns, continuous batching again), 15_training_optimization (mixed precision, gradient checkpointing)
STRONGER: DL ch22 is clearly stronger on serving mechanics — roofline math with the bandwidth/compute crossover (line 318, Estimation Qs), chunked prefill, disaggregated prefill/decode, GGUF/CPU path, and a capacity-planning answer whose arithmetic actually closes (1K QPS Q, line 398). DL ch14 is stronger and cleaner on quantization/pruning/distillation fundamentals (PTQ-vs-QAT table, N:M sparsity, attention-vs-FFN quantization sensitivity). NLP ch12 is stronger on tokenization (full algorithms, tokenizer tax, code tokenization, multilingual production) — DL ch09's tokenization is a thin summary — and uniquely contributes prefix-caching numbers, model routing/semantic caching, and LLM-flavored MLOps.
RECOMMENDATION: Split the canon: NLP ch12 becomes the canonical home for tokenization, multilingual production, and LLM-specific MLOps/cost optimization; DL ch22+14 remain canonical for serving mechanics and compression. Cut NLP ch12's sections 2-6 to a condensed 'what NLP/LLM workloads change' treatment (~40% of current length) with explicit cross-refs, and delete DL ch09's tokenization section down to a pointer. Before any cut, reconcile the contradicting numbers (30 vs 70 ms/token, 2-5x vs 2-8x batching, quantization formula conventions) — they are exactly the out-of-sync drift the two-volume format risks.

MISSING (crit/high):
- [critical] Multi-Head Latent Attention (MLA, DeepSeek-V2/V3) as the third KV-cache compression option beside MQA/GQA :: By 2026 'GQA vs MLA' is a standard serving question; the KV section (sec 2) presents GQA as the endpoint. MLA appears in the NLP volume only as a one-line L7 ladder mention in ch5 — a candidate studyi
- [high] Chunked prefill and prefill/decode disaggregation (DistServe, Mooncake, NVIDIA Dynamo pattern) :: The chapter's own 10K-QPS design (Q1) needs these to hit TTFT SLAs without stalling decode batches; DL ch22 already covers chunked prefill (warningbox line ~137) and disaggregated serving, so this vol
- [high] Roofline / arithmetic-intensity reasoning: the bandwidth-vs-compute crossover batch size :: Ch12 asserts 'memory-bandwidth-bound' but never shows the 140GB/2TB/s vs FLOPs arithmetic. DL ch22 (keyinsight line 318, Estimation Q at line 512) derives it. Staff loops explicitly ask candidates to 
- [high] Serving reasoning models: long decode traces from test-time compute change capacity planning (decode-heavy, KV-dominated, output-length variance) :: 2026 interviewers ask 'how does o1/R1-style inference change your serving stack?' — nothing in the chapter acknowledges that reasoning traces invert the prompt-heavy assumptions in its examples.

CORRECTNESS:
- Line ~1030-1033 (Q1 strong answer): 'A single H100 can serve a 7B INT4 model at ~500 QPS with 1K-token prompts... 10K QPS requires ~20 GPUs' — infeasible by the chapter's own physics; prefill alone needs ~7 PFLOPS sustained. The strongest answer in the chapter fails its own 'think in numbers' test.
- Lines 714-716: 'generating a single token takes ~30ms on an A100 (reading 140GB of weights at ~2TB/s memory bandwidth)' — 140/2 = 70ms; DL ch22 line 318 computes 70ms correctly. Direct cross-volume contradiction.
- Vocab table (lines 176-188): GPT-2 listed at ~32K (actual 50,257); BERT listed at ~50K (actual 30,522); Gemini listed at ~150K (SentencePiece vocab is 256K).
- Line 139: Unigram 'Used by: ... LLaMA (via SentencePiece)' — LLaMA's SentencePiece tokenizer is BPE, not Unigram; contradicts table line 202 and would be flagged by any tokenizer-literate interviewer.
- Lines 320-327 (keyinsight): 'for a 70B... KV-cache can exceed 40GB — larger than the model weights themselves' — 40GB < 140GB FP16 weights; only true vs INT4 weights or summed across a batch. As written, wrong.
- Line ~1050: per-request KV '~64MB' for 7B/1K-context/INT4-KV matches no config (MHA INT4 = 134MB; GQA-8 INT4 = 34MB).
- Minor, line 84-86: BPE OOV handling described as decomposing into 'longest matching subword units' — BPE applies merges in learned priority order; longest-match-first is WordPiece inference.
- Minor, lines 750-758: worked speculative-decoding example ('k=5, 80% acceptance ⇒ roughly 2.5x') is inconsistent with the formula printed immediately above it (which yields ~3-4x for any plausible cost ratio).

STALENESS:
- 'INT3/INT2 — Experimental, research only' (line 602): AQLM/QuIP#/QTIP made 2-3-bit deployable for 70B-class models by 2025; the blanket dismissal is out of date.
- Flash Attention 2 presented as the frontier (line 804); FA-3 (2024, Hopper-specific) is the production default in 2026.
- Serving frameworks list omits SGLang and the vLLM V1 re-architecture; hardware discussion stops at H100 — no Blackwell, no FP4.
- Running examples are LLaMA-2 7B / A100-era; 'Codex uses tokens for 1-24 spaces' is a 2021 detail; modern code tokenizers (digit splitting, pretokenizer regexes) unmentioned.
- Distillation section's 'distill from GPT-4/Claude into open models is standard practice' needs the 2025-era canonical example (R1-Distill) and a ToS caveat.
- Prefix caching described without RadixAttention or provider-side prompt-caching pricing, both standard by 2025.

MUST-KNOW:
- KV-cache memory formula 2·L·H_kv·S·d_h·b with one worked example (LLaMA-2 7B: 1.07GB at 2K FP16; ÷4 with GQA-8), and that KV — not weights — dominates long-context serving memory
- Prefill = compute-bound vs decode = memory-bandwidth-bound, and which metric each drives (TTFT vs ITL); be able to derive ~ms/token from weights-size ÷ HBM bandwidth
- Continuous batching (iteration-level scheduling) and PagedAttention (paged KV + copy-on-write prefix sharing) — what each fixes and the 2-4x throughput/95% utilization numbers
- KV-reduction ladder: MQA → GQA → MLA, plus KV quantization and sliding-window attention
- Quantization decision ladder: FP8/INT8 (SmoothQuant) near-free default → INT4 GPTQ (Hessian compensation) vs AWQ (activation-salient channels) → always re-evaluate on your task, and larger-model-quantized beats smaller-model-FP16
- Speculative decoding: draft-verify-accept/reject, the rejection-sampling proof that output distribution is exactly the target's, and the batch-size condition under which it stops helping
- BPE end-to-end (train merges by frequency, apply in priority order, byte-level ⇒ no UNK) + vocab-size vs sequence-length tradeoff + why LLMs fail character-counting
- Tokenizer tax: fertility differences (2-4x for Thai/Hindi), its cascading cost/context/latency effects, and mitigations (balanced tokenizer training, larger vocab, per-language monitoring)
- Distillation loss (temperature-scaled KL, T² factor) and the modern data-level variant (teacher-generated SFT data, on-policy distillation)
- Honest capacity math: QPS ↔ tokens/s ↔ concurrent requests (Little's law) ↔ GPU count, with TTFT/ITL/P99 SLA framing

IMPROVEMENTS:
- Rebuild interview Q1's capacity math (lines ~1015-1080): '~500 QPS per H100 for 7B INT4 with 1K-token prompts, so 10K QPS ≈ 20 GPUs' is 3-10x optimistic — show the token-budget arithmetic (500 QPS × 1K prompt tokens = 500K prefill tok/s × ~14 GFLOP/tok ≈ 7 PFLOPS, above H100 FP8 peak, before any decode). DL ch22's 1K-QPS/70B answer (line 398+) is the right template: model → tokens/s → concurrency → nodes.
- Reconcile with DL ch22 wherever both state numbers: 30ms/token vs 70ms/token for a 70B on A100; continuous batching '2-5x' (line 434) vs DL's '2-8x' (line 126); quantization formula conventions differ (NLP: q=round(x/s+z); DL ch14 line 320: x_q=round((x-z)/s)) — pick one convention for the set.
- Fix the vocabulary-size table (lines 176-188): GPT-2 belongs at ~50K (50,257), BERT at ~30K (30,522), Gemini/Gemma at 256K; add o200k. Also remove LLaMA from Unigram's 'Used by' (line 139) — LLaMA is BPE-via-SentencePiece, contradicting the chapter's own table at line 202.
- Add an MQA/GQA subsection or an explicit cross-reference to ch5 — the appendix attributes an MQA-vs-GQA question (12.6) to this chapter, but the body never introduces MQA.
- Promote speculative decoding's batch-size caveat (helps at batch 1-4, hurts near compute saturation) from the L7 ladder into the body — DL ch22 has the full 'when it helps/hurts' table (line 203); this is the #1 follow-up trap.
- Tighten the MLOps section (sec 8): it is generic checklist material duplicating DL ch17 (drift, A/B, canary, registry). Keep only the LLM-specific parts (TTFT/ITL/P99 targets, goodput, per-language monitoring, LLM-as-judge shadow evals, semantic caching, routing math) and cross-ref DL ch17 for feature stores/training-serving skew/CI-CD.
- Correct SentencePiece description (lines 143-148): it operates on raw text/Unicode with optional byte-fallback, not 'a raw byte stream'.
- In Q1, show the arithmetic behind the '~64MB per-request KV (7B, 1K context, INT4 KV)' claim — MHA INT4 gives ~134MB, GQA-8 gives ~34MB; 64MB matches no stated configuration.

# 13_safety_ethics.tex)

VERDICT: Solid, well-organized breadth chapter whose unique value is the fairness math (three definitions + impossibility + choose-per-context), toxicity-classifier failure modes, watermarking mechanics, and deployment governance. But its hallucination / prompt-injection / red-teaming half is a near-duplicate of DL ch23 (which treats guardrails and over-refusal at a higher engineering level), and it entirely misses the defining 2026 safety topic — agentic/tool-use security — along with named safety benchmarks and modern guard models. Keep the fairness/watermarking/governance half as canonical; trim and cross-ref the rest; add an agent-safety section.

STAFF: The fairness section is genuinely staff-grade — the impossibility result plus 'choosing a criterion is a product decision' framing is exactly the L6/L7 discriminator, and the resume-screening interviewq exercises it well. The content-moderation question (staged pipeline with per-stage latency) is the chapter's best systems artifact. But most of the safety-engineering half is number-light relative to the house style — no latency budgets, no FP/FN rates, no benchmark deltas — so it reads L5 where DL ch23's parallel material reads L6. The ethics question (Q5) is an L4-L5 listicle. The hard cap: without agentic threat modeling and reasoning-era safety, a candidate prepped only on this chapter will be visibly a year behind in a 2026 frontier-lab safety round.

OVERLAP [heavy] with 23_safety_alignment (hallucination types/causes/detection/mitigation, guardrails + classifier-vs-rule filtering, prompt injection + jailbreaks, red teaming — all near-duplicated), 17_production_systems (monitoring, staged rollout/canary), 18_evaluation_metrics (LLM-as-judge, human eval mechanics), 24_long_context_rag (RAG-as-grounding)
STRONGER: Split decision. DL ch23 is stronger on the engineering half: hallucination root-cause table adds sycophancy and knowledge-boundary (line 88-103), the guardrail design question carries an explicit 50ms parallel latency budget, it has a safety-benchmarks table, an L7 over-refusal question, and it uniquely covers mechanistic interpretability and steering vectors. NLP ch13 is stronger and unique on the sociotechnical half: fairness formulas + impossibility, bias-source pipeline and debiasing table, toxicity-classifier challenges (context, evasion, AAVE over-blocking), watermarking mechanics, and responsible-deployment governance (model cards, datasheets, incident response) — none of which exist in the DL volume.
RECOMMENDATION: Assign canonical homes: DL ch23 owns hallucination-detection systems, guardrail engineering, jailbreaks, and red-team programs; NLP ch13 owns bias/fairness, content moderation, watermarking/attribution, and responsible deployment. Trim NLP ch13's sections 2-4 to NLP-specific angles (summarization faithfulness via NLI, RAG-citation patterns, multilingual safety gaps) with cross-refs, freeing ~10 pages for the missing agent-safety section. Reconcile the two volumes' divergent hallucination-cause and detection tables before trimming — they already disagree on the cause list, which is the sync-drift the two-volume set must avoid.

MISSING (crit/high):
- [critical] Agentic safety: prompt injection against tool-using agents — data exfiltration via tool calls/markdown-image rendering, browsing/email attack surfaces, MCP-era tool supply chain, least-privilege + sandboxing + human confirmation for irreversible actions :: This is the #1 2026 frontier-lab safety interview topic. The chapter's indirect-injection coverage stops at 'RAG retrieves a poisoned page'; it never considers the model *acting* (sending an email, ex
- [high] Named safety evaluation benchmarks: TruthfulQA, HarmBench, BBQ, ToxiGen, XSTest/OR-Bench (over-refusal), plus dangerous-capability evals :: The chapter discusses red-teaming coverage metrics but names zero benchmarks; DL ch23 has a benchmarks table (line 265). A candidate asked 'how do you measure safety regressions between model versions
- [high] Over-refusal as a first-class, measurable topic (over-refusal benchmark construction, category-specific thresholds, safety-helpfulness Pareto tracking) :: Ch13's Q6 covers the tradeoff qualitatively; DL ch23's L7 question (line 454) shows the staff-level version with diagnosis (model-level vs classifier-level) and measurement. Frontier labs probe exactl
- [high] Modern guard models and moderation stack: Llama Guard 2/3/4, ShieldGemma, provider moderation endpoints, constitutional-classifier-style input/output screens :: The chapter cites only Llama Guard (2023); naming the current generation is a cheap, high-signal currency marker in guardrail design questions.

CORRECTNESS:
- Lines 702-714: impossibility result stated as the mutual incompatibility of demographic parity, equal opportunity, and calibration; the cited theorems (Kleinberg et al.; Chouldechova 2017) are about calibration vs balance/equalized-odds-style conditions — directionally right, formally imprecise. Also 'Kleinberg et al., 2016' is conventionally cited as 2017 (ITCS).
- Lines 336-341: DAN presented as the canonical direct-prompt-injection example; it is a jailbreak — internally inconsistent with the chapter's own jailbreaking subsection (line 461+).
- Line 1427-1429: 'green-list tokens should appear ~50% of the time' under the null — only for γ=0.5; γ is a design parameter and 0.25 is common in the original paper's experiments.
- Line 96-97: 'occupational statistics in web text do not match real-world gender distributions' — correct claim, but the following doctor/nurse analogy is attributed loosely to Bolukbasi et al. 2016 (line 104), whose headline example was programmer/homemaker; harmless but citable-nit in an interview-prep book.
- No errors found in the fairness metric formulas themselves (demographic parity, equal opportunity, calibration all stated correctly), the guardrail architecture, or the Kirchenbauer mechanism description.

STALENESS:
- Llama Guard cited as (Meta, 2023) with no successor models (Llama Guard 3/4, ShieldGemma) — guardrail tooling is two generations ahead.
- Jailbreak inventory is 2023-era (DAN, base64, roleplay) plus many-shot (2024); missing 2024-25 families: Crescendo-style multi-turn, low-resource-language, persona modulation.
- Watermarking section predates SynthID-Text's deployment (2024) — the only production-scale text watermark.
- EU AI Act framed as an emerging 'trend' rather than in-force law with dated obligations.
- 'Every production NLP team at a FAANG company' framing throughout; 2026 loops are frontier-lab flavored (model specs, system cards, agent deployments).
- No acknowledgment anywhere that reasoning models / agents changed the safety surface — the chapter reads as a 2024 chatbot-safety chapter.

MUST-KNOW:
- Intrinsic vs extrinsic hallucination, the four root causes (data noise, exposure bias, parametric priors overriding context, objective mismatch — fluency ≠ factuality), and why every mitigation is partial
- The layered hallucination defense: RAG-with-citations + NLI entailment checking + self-consistency (and its consensus-hallucination failure mode) + calibrated abstention + production sampling/monitoring
- Direct vs indirect prompt injection, why it exists (instructions and data share one channel — the SQL-injection analogy), and defense-in-depth: input classifier, instruction hierarchy (system>developer>user>tool), delimiter isolation, output filtering, canary tokens
- Injection vs jailbreak distinction, jailbreak taxonomy (roleplay, encoding, multi-turn escalation, many-shot, GCG adversarial suffixes)
- Input/output guardrail architecture with a latency budget, and why output filtering is non-negotiable (benign inputs can still yield harmful outputs)
- The three fairness definitions with formulas, the impossibility result under unequal base rates, and how to choose the criterion per application (parity for surfacing, equal opportunity when labels are trusted, calibration for exposed scores)
- Proxy variables — why 'remove the protected attribute' fails — and disaggregated + intersectional evaluation
- Red teaming as a structured program: risk taxonomy → manual (diverse testers) + automated (adversarial LLMs, GCG) → severity-gated remediation → continuous post-launch monitoring
- Watermarking: green/red-list logit biasing, z-test detection, and the three limits (paraphrase fragility, low-entropy text, quality-vs-δ tradeoff)
- The safety-helpfulness Pareto frontier and over-refusal: graduated responses over binary refusal, contextual thresholds, measuring both directions on every change
- (2026) The agentic threat model: untrusted content + tool access + exfiltration channel, and least-privilege/human-confirmation mitigations

IMPROVEMENTS:
- Tighten the impossibility-theorem statement (lines 702-714): Kleinberg et al. and Chouldechova prove calibration is incompatible with error-rate-balance conditions (equalized-odds-type); the text's 'demographic parity + equal opportunity + calibration' trio is a looser paraphrase (parity vs calibration under unequal base rates is trivially incompatible). State the actual theorem — this is exactly where a mathematically literate interviewer pushes.
- Move the DAN example out of the *direct injection* section (lines 336-341): DAN is a jailbreak, and the chapter itself later distinguishes injection from jailbreaking (line 461-463). Define the boundary crisply up front: injection = untrusted content co-opting the instruction channel; jailbreak = attacking the model's own policy.
- Import the numbers-first guardrail design from DL ch23 (Q1, 50ms parallel latency budget with per-component ms estimates) or cross-ref it — NLP ch13's guardrail section has architecture but no latency/cost quantities, breaking the house style.
- Add quantitative anchors to the bias/toxicity discussion: e.g., cite the documented AAVE false-positive rates (Sap et al. 2019) instead of 'more likely to flag' (line 630-633); add LLM-era bias evals (BBQ, discrim-eval) so the bias section isn't purely embedding-era (Bolukbasi 2016).
- In red teaming, name the modern automated attack families beyond GCG: PAIR/TAP (LLM-vs-LLM), Crescendo (multi-turn), and note many-shot is already covered — then connect attack families to which defense layer catches each.
- Replace 'regulatory trends (EU AI Act) increasingly require' (line 866) with the actual state: in force Aug 2024, GPAI obligations from Aug 2025 — concreteness here is an easy differentiator.
- Note in the watermarking detection paragraph that the green-list fraction γ is a tunable parameter (commonly 0.25-0.5); '~50% under the null' (line 1427) silently assumes γ=0.5.

# appendix_question_index.tex)

VERDICT: Useful navigation with honest frequency tagging and well-written L5/L6/L7 expectation prose, but as an index it is not trustworthy: it silently indexes only ~96 of the volume's 128 interviewq's while claiming comprehensiveness, contains at least three rows with no corresponding chapter question (12.6, 12.7, 13.7), paraphrase-drifts others, and lacks the summary-statistics/study-plan machinery the DL volume's appendix has — so the two volumes' appendices are structurally inconsistent for a set meant to be used together. Regenerate it mechanically from the interviewq blocks and integrate it with the DL index.

STAFF: The level-calibration prose is the most staff-authentic writing in the appendix — the L6 paragraph ('explains not just how but why: why decoder-only won, why DPO replaced RLHF in many settings, why Flash Attention is IO-bound') reads like a real interviewer's rubric. But as an artifact the appendix would fail a staff engineering review: hand-maintained, drifting from source, 25% incomplete, indexing phantom questions, and unintegrated with the sibling volume it ships alongside — precisely the maintain-two-copies failure mode the book set needs to engineer away. Low effort to fix (a build-time script over the interviewq environments), high payoff in trust.

OVERLAP [partial] with appendix_question_index (DL volume) — same artifact type, disjoint content, different structure
STRONGER: The DL volume's appendix is the stronger tool: verified summary statistics (248 questions by level/type/frequency), a 2-hour cram list, a 1-week day-by-day plan, a by-company-focus chapter mapping, and six question types including Estimation/Debugging/First-Principles. The NLP appendix's unique contributions are the ranked Top-20 with chapter/level columns and the per-level expectation essays — both worth porting to the DL side.
RECOMMENDATION: Keep both appendices (each must index its own volume) but unify the template: adopt DL's stats + study plans in the NLP appendix, port NLP's Top-20 ranking and L5/L6/L7 expectation prose to the DL appendix, and add one shared cross-volume study plan that names the canonical home for every duplicated topic (serving → DL 22, quantization/distillation → DL 14, tokenization/multilingual → NLP 12, fairness/watermarking → NLP 13, hallucination/guardrails/red-teaming → DL 23, RLHF → DL 20 + NLP 8) so a candidate prepares each topic exactly once from the stronger source. Generate both indices from source at build time to end the drift.

MISSING (crit/high):
- [critical] Complete question coverage: 32 of 128 chapter questions are unindexed (chapters contain 10/10/10/10/12/10/10/8/10/10/10/10/8 interviewq's; the index lists 96 rows) :: Dropped questions include freq-high items: ch12's 'reduce latency 5x' and production-quality-degradation debugging, ch12's model routing, ch13's ethics and safety-vs-helpfulness questions, ch1's perpl
- [high] Summary statistics and study plans (question counts by level/type/frequency; 2-hour cram; 1-week plan; by-team-focus mapping) :: The DL volume's appendix (lines 9-114) has all of these; the NLP appendix has only a Top-20. A candidate using both volumes gets inconsistent tooling and no NLP-side time-boxed plan.
- [high] Cross-volume integration: pointers to canonical DL-volume treatments (serving depth → DL ch22, RLHF math → DL ch20, quantization → DL ch14, safety systems → DL ch23) and a joint top-N spanning both books :: The volumes are used together and duplicate serving/quantization/hallucination/RLHF material; without a shared index a candidate studies the same topics twice from drifting sources — the exact waste t

CORRECTNESS:
- Row 12.6 (line 307): 'Compare Multi-Query Attention and Grouped-Query Attention' attributed to Chapter 12 — no such question exists in 12_production_nlp.tex (MQA appears there only inside one L6 ladder bullet); the real interviewq is in ch5.
- Row 12.7 (line 308): no continuous-batching/PagedAttention interviewq exists in ch12 — the topic is body-text only; the row indexes a nonexistent question.
- Row 13.7 (line 332): 'How do you build guardrails (input and output)...' — no corresponding interviewq exists in 13_safety_ethics.tex (guardrails are a body section only).
- Coverage claim at line 10 ('comprehensive index of interview questions drawn from every chapter') vs reality: 96 rows for 128 questions; ch12 indexes 6 real questions of 10, ch13 indexes 6 real of 8.
- Row 1.7 changes the question's substance ('Naive Bayes and SVMs' vs the chapter's 'Naive Bayes vs fine-tuned BERT'); several other rows are paraphrases rather than the asked question — risky when candidates drill from the index wording.
- The L5/L6/L7 by-level tables re-shorten question texts a second time, creating a third variant of some questions (e.g., 1.1 loses its 'When does TF-IDF fail?' clause) — another drift surface the mechanical regeneration would fix.

STALENESS:
- 'FAANG NLP interview loops' framing throughout; the 2026 market is frontier-lab framed, and the DL appendix already maps to team focus areas including Safety/Alignment — the NLP one should match.
- The by-type Mathematical list leans 2015-2019 (Viterbi, BLEU, Bahdanau-adjacent items) with nothing from the reasoning era; the only test-time-compute entry is 7.6.
- No index presence for GRPO/RLVR, agents, or modern eval suites because the chapters lack them — the index is current only up to the volume's mid-2024-to-early-2025 horizon.

MUST-KNOW:
- How to use the index: Top-20 first (the high-frequency spine: attention derivation, RLHF pipeline, RAG design, Transformers-vs-LSTM why, LoRA, BPE, KV-cache, hallucination, DPO-vs-RLHF, serving at scale, injection defense) — this list is genuinely well-chosen aside from two dated entries
- The L5/L6/L7 expectation paragraphs (lines 344-348, 409-412, 474-477) — accurate calibration of what each level must demonstrate, worth internalizing verbatim
- Frequency-tag triage: freq-low classical items (Kneser-Ney, noisy channel, CNN-for-text, LSTM-vs-GRU) get one-pass breadth coverage only; freq-high items get whiteboard-fluency practice
- Type-based drilling: Mathematical rows demand pen-and-paper derivation practice; System Design rows demand timed end-to-end delivery with numbers
- By-team prioritization (the closing keyinsight): ch9 for search/RAG teams, ch8 for alignment teams, ch12 for infra teams — extend mentally with the DL volume's canonical chapters for each

IMPROVEMENTS:
- Regenerate the index mechanically from the \begin{interviewq} blocks (question text, level badge, type badge, freq badge) with per-question \labels and hyperrefs, so it cannot drift; today's hand-copied rows already disagree with the source in wording, chapter attribution, and existence.
- Delete or re-home the phantom rows: 12.6 (MQA vs GQA — the actual interviewq lives in ch5, 05_transformers.tex line 1526), 12.7 (no standalone continuous-batching question exists in ch12), 13.7 (no guardrails interviewq exists in ch13). Either add the questions to the chapters or fix the rows.
- Fix conflations/drift in row text: 12.7's 'continuous batching (e.g., vLLM's PagedAttention)' conflates a scheduling technique with a memory-management technique; 1.7 'Compare Naive Bayes and SVMs' vs the actual ch1 question 'Naive Bayes over a fine-tuned BERT'; 12.4 drops INT8 from the chapter's GPTQ/AWQ/INT8 question.
- Adopt the DL appendix's structure (summary stats table, 2-hour cram, 1-week plan, by-team mapping) for consistency across the set, and add per-chapter question counts.
- Rebalance the Top-20 for 2026: LSTM gate-by-gate (#6) and BLEU derivation (#17) are breadth insurance at frontier labs, not top-20 material — demote in favor of RAG evaluation (9.4), hallucination-system design, and DPO-limitations follow-ups; keep the honest low-freq badges on Kneser-Ney (2.5) and noisy channel (2.7), which correctly signal that classical statistical NLP is now occasional-breadth only.
- Soften or substantiate 'these questions alone cover the core of 80% of FAANG NLP interview loops' (line 695-696) — an unverifiable precision claim in an otherwise honest artifact; also align the section title 'Chapter 12 — Production NLP Systems' with the chapter's actual title.
- Flag intra-volume duplicates in the index (perplexity appears in ch1, ch2, and ch10 rows 2.3/10.2) with 'see also' links so candidates don't treat them as distinct preparation items.

