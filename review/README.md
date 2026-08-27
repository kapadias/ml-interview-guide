# Staff-Level Review — Findings Log

Full findings behind the review of this book against 2026 Staff ML interview loops
at frontier labs (OpenAI/Anthropic-tier). The executive summary, upgrade plan, and
mockups live in the published review page; this directory preserves every detailed
finding with file/line references.

## Contents

| File | What it holds |
|---|---|
| `chapter-findings.md` | Per-chapter review of all 24 chapters + appendix: verdict, staff-level assessment, missing topics (prioritized), correctness concerns, staleness, must-know list, concrete improvements |
| `crosscutting-audits.md` | Three cross-cutting audits: whole-topic coverage gaps vs. frontier-lab loops; structure/pedagogy; 2026 currency sweep |
| `first-principles/interview-loop.md` | Round-by-round reconstruction of the 2026 Staff ML loop: formats, representative prompts, calibration signals, failure modes, weighting, and honest book-coverage per round |
| `first-principles/ideal-blueprint.md` | The ideal prep artifact designed from scratch (layer architecture, organizing spine, formats, minimum effective dose, role tracks) and how this book maps onto it |
| `first-principles/knowledge-kernel.md` | The 26 first principles from which most staff-level answers re-derive — statement, whiteboard derivation, attached numbers, questions unlocked, book coverage |
| `first-principles/numbers-fluency.md` | The complete quantitative-fluency layer: memorized constants, formulas, worked examples as spoken, traps, book coverage |
| `first-principles/round-simulations.md` | Seven simulated interview rounds run adversarially against the book, with exact failure points (verdict: 0 sufficient / 5 partial / 2 fail) |

## Headline correctness items (verified against source)

All re-checked by hand before publication — see `chapter-findings.md` for the full list:

- `sections/19_decision_frameworks.tex:208` — Chinchilla stated backwards (`N ≈ 20 × D`; should be `D ≈ 20N`)
- `sections/01_mathematical_foundations.tex:767,786` — "softmax does not increase rank / preserves column space" is false; the red-flag bullet penalizes the correct answer
- `sections/20_reinforcement_learning.tex:809` — "PRMs are foundational to o1/o3 … beam search or MCTS" taught as fact (superseded hypothesis)
- `sections/12_recommendation_systems.tex:571,1655` — ESMM's CVR described as the ratio CTCVR/CTR (the estimator the paper rejects)
- `sections/08_self_supervised_learning.tex:582` — the MAE "architecture" figure is a pasted retrieval decision tree (`fig:arch_decision`)
- `sections/17_production_systems.tex:250` — A/B sample size "~1.6M" (correct: 7.84M; contradicts ch. 18's own worked example)
- `main.tex` include order (…16, 20, 21–24, 17–19) — compiled chapter numbers contradict filenames and README

Review date: 2026-08-27.

## v2 additions (2026-08-27, second fan-out)

| File | What it holds |
|---|---|
| `nlp-volume/chapter-findings.md` | Per-chapter review of all 13 NLP-volume chapters + appendix, each with a verified overlap analysis against the DL volume (degree, stronger treatment, canonical-home recommendation) |
| `search-recsys/vector-retrieval-digest.md` | Full digest of Bruch, *Foundations of Vector Retrieval* (arXiv 2401.09350, 203 pp.): topic map, interview-relevant extractions with whiteboard versions and numbers, gap map vs existing ANN/vector-DB sections, 12 new interview questions |
| `search-recsys/applied-search-track.md` | The applied search-ranking chapter design (query understanding, BM25 internals, click models, LTR, hybrid, judgments, evaluation, RAG-era search), the ~68 h Search/RecSys track, and the ten search fumble tests |
| `program/conventional-ml-spec.md` | Ideal Conventional-ML volume spec (8 chapters with derivations, questions, red flags, page weights) + the diff checklist to run against the user's own document on arrival |
| `program/unified-program-architecture.md` | The four-volume + spine program architecture, dedup treaty (canonical home per overlap zone), round coverage map, and revised Phase A–E master plan |

Cross-volume drift confirmed and queued for Phase A: GPT-2 norm placement (DL-05 wrong), ELECTRA loss sign (DL-09 wrong), the o1=PRM+MCTS misattribution (three locations across both volumes), NLP-12 vs DL-22 serving-number drift, NLP-01/02 duplicate LaTeX labels, NLP appendix indexing ~96/128 questions with three phantom rows.
