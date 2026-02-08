# Deep Learning Essentials

A comprehensive deep learning interview reference for ML Engineering interviews. Written as a LaTeX book with 24 chapters, 248 interview questions, and practical frameworks — designed as an interview playbook, not a textbook.

## What's Inside

Every topic follows a consistent format: **What it is** → **Key formula** (only if you'd whiteboard it) → **When to use it** → **Common pitfalls**. Interview questions include model answers, red flags, what the interviewer is testing, and common follow-ups.

### Part I: Foundations
| # | Chapter | Questions |
|---|---------|-----------|
| 1 | Mathematical Foundations | 10 |
| 2 | Learning Theory | 10 |
| 3 | Loss Functions | 12 |
| 4 | Activation & Normalization | 8 |

### Part II: Core Architectures
| # | Chapter | Questions |
|---|---------|-----------|
| 5 | Attention & Transformers | 17 |
| 6 | Embeddings & Representations | 8 |
| 7 | Similarity & Metric Learning | 12 |
| 8 | Self-Supervised Learning | 10 |
| 9 | NLP Architectures | 10 |
| 10 | Vision Architectures | 8 |
| 11 | Generative Models | 10 |

### Part III: Specialized Domains
| # | Chapter | Questions |
|---|---------|-----------|
| 12 | Recommendation Systems | 15 |
| 13 | Graph Neural Networks | 8 |
| 14 | Efficient Architectures | 10 |

### Part IV: Learning & Optimization
| # | Chapter | Questions |
|---|---------|-----------|
| 15 | Training & Optimization | 12 |
| 16 | Transfer Learning | 10 |
| 20 | Reinforcement Learning & RLHF | 10 |

### Part V: Modern Frontiers
| # | Chapter | Questions |
|---|---------|-----------|
| 21 | Multimodal Learning | 10 |
| 22 | Inference Optimization | 12 |
| 23 | Safety & Alignment | 8 |
| 24 | Long Context & RAG | 10 |

### Part VI: Practice & Systems
| # | Chapter | Questions |
|---|---------|-----------|
| 17 | Production ML Systems | 10 |
| 18 | Evaluation & Metrics | 10 |
| 19 | Decision Frameworks | 8 |

### Appendix
- **Question Index** — all 248 questions organized by difficulty, type, frequency, and topic, with curated study plans

## At a Glance

- **248** interview questions with full answer scaffolds
- **24** chapters across 6 parts
- **~25,000** lines of LaTeX
- Difficulty-tagged questions (L5 / L6 / L7)
- Frequency ratings for interview likelihood
- Key insight boxes, warning boxes, and TL;DR summaries throughout

## Building the PDF

Requires a LaTeX distribution with `pdflatex` and common packages (`tcolorbox`, `tikz`, `pgfplots`, `listings`, `algorithm`, `hyperref`, etc.).

```bash
# Full build (recommended — resolves cross-references and TOC)
pdflatex main.tex && pdflatex main.tex

# Or with latexmk for automatic re-runs
latexmk -pdf main.tex
```

The double compile is needed to resolve the table of contents and internal cross-references.

## Project Structure

```
.
├── main.tex                  # Book root — preamble, macros, and part/chapter includes
└── sections/
    ├── 01_mathematical_foundations.tex
    ├── 02_learning_theory.tex
    ├── ...
    ├── 24_long_context_rag.tex
    └── appendix_question_index.tex
```

## Custom Environments

The book uses several custom `tcolorbox` environments:

| Environment | Purpose |
|-------------|---------|
| `interviewq` | Interview question with answer scaffold, red flags, and follow-ups |
| `keyinsight` | Critical concept or non-obvious takeaway |
| `warningbox` | Common mistakes and misconceptions |
| `mathresult` | Whiteboard-worthy equation or derivation |
| `tldr` | Chapter or section summary |

## License

This project is for personal use. All rights reserved.
