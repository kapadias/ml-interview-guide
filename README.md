# ML Interview Guide

A four-volume interview preparation program for Staff-level ML roles at frontier
labs and platform companies, covering ML breadth, ML depth, ML system design,
and ML coding rounds — in one repository.

## The volumes

| Vol | Directory | Scope | Status |
|----|-----------|-------|--------|
| I | `volumes/deep-learning` | Foundations, transformers, training & distributed systems, efficiency, RLHF/GRPO, inference & serving, safety, RAG | 24 chapters, 251 questions |
| II | `volumes/nlp` | Classical NLP through LLMs, alignment, RAG, production NLP | 13 chapters, 128 questions |
| III | `volumes/search-recommendation` | Search systems, vector retrieval, learning to rank, recommendations | scaffolded (Phase C) |
| IV | `volumes/conventional-ml` | Classical ML: trees/GBM, probabilistic foundations, applied craft, experimentation | scaffolded (Phase C; see `volumes/conventional-ml/docs/coverage-spec.md`) |

Cross-volume references use `[DL 5.3]`, `[NLP 9.2]`, `[SR 4]`, `[CML 2]`.

## Program layer (repo root)

- `style/essentials.sty` — shared LaTeX style (colors, boxes, badges, question macros)
- `kernel/` — Part 0: the 26-principle interview kernel + Numbers Card (Phase E)
- `plans/` — cross-volume study plans and role tracks (Phase E)
- `drills/` — runnable Python for coding rounds (Phases C/E)
- `ci/` — `facts.yaml` drift sentinels + `check_drift.py` (forbidden-claim grep,
  cross-volume label-collision check)
- `review/` — the complete staff-level review findings that drive the roadmap

## Build

Each volume builds standalone with [tectonic](https://tectonic-typesetting.github.io)
(or `latexmk -pdf`):

```bash
make all          # drift check + all four volumes
make deep-learning
make nlp
make check        # drift sentinels + label collisions only
```

## Provenance

Volumes I and II were imported (with full git history) from
`kapadias/deep-learning-essentials` and `kapadias/natural-language-processing-essentials`;
this repository is now the canonical home.
