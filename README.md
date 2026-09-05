# ML Interview Guide

A four-volume interview preparation program for Staff-level ML roles at frontier
labs and platform companies, covering ML breadth, ML depth, ML system design,
and ML coding rounds — in one repository.

**609 interview questions across ~2,100 pages**, every one carrying a level
badge (L5/L6/L7), a type, and how often it actually comes up.

## The volumes

| Vol | Directory | Scope | Status |
|----|-----------|-------|--------|
| I | `volumes/deep-learning` | Foundations, transformers, training and distributed systems, efficiency, RL and reasoning, inference and serving, pretraining at scale, GPU performance, agents, safety, long context, plus the interview-craft chapters (coding rounds, the staff loop) | 29 chapters, 301 questions |
| II | `volumes/nlp` | Classical and statistical NLP through LLMs, pretraining and adaptation, in-context learning, alignment, RAG, benchmarks, production NLP, safety | 13 chapters, 128 questions |
| III | `volumes/search-recommendation` | Query understanding, lexical and vector retrieval, neural retrieval and reranking, learning to rank, recommenders, ranking evaluation, production retrieval | 8 chapters, 92 questions |
| IV | `volumes/conventional-ml` | Linear models, trees and gradient boosting, unsupervised methods, probabilistic foundations, applied craft and the leakage taxonomy, experimentation and causal inference, time series, the classical coding canon | 8 chapters, 88 questions |

Every volume opens with **the Program Map**, which names the canonical home for
each of the 21 overlap zones. One topic, one deep treatment; everywhere else a
short summary and a pointer. Cross-volume references read `[DL 5]`, `[NLP 9]`,
`[SR 4]`, `[CML 2]`.

## Program layer (repo root)

- `kernel/` — Part 0: the 26-principle knowledge kernel, the Core Fifty tier, and
  the numbers card. Authored once, synced into every volume by `make sync-style`.
- `plans/` — study plans: a two-week breadth sprint, the LLM and search/recsys
  depth tracks, a coding week, and a night-before pass
- `drills/` — runnable, self-testing implementations for the coding round, plus
  `debug_round.py`, which plants one bug in a training script for you to diagnose
- `index/master_question_index.md` — all 609 questions, generated from the source
- `tools/gen_question_index.py` — regenerates every question index; `--check`
  fails when one is stale
- `ci/` — `facts.yaml` drift sentinels and `check_drift.py` (forbidden-claim grep
  plus a cross-volume label-collision check)
- `style/` — shared LaTeX style and the Program Map
- `review/` — the staff-level review findings that drove the roadmap

## Build

Each volume builds standalone with [tectonic](https://tectonic-typesetting.github.io):

```bash
make all          # verify + build all four volumes
make deep-learning
make verify       # drift sentinels + question-index staleness
make index        # regenerate the question indexes
make sync-style   # push the shared style, map and kernel into each volume
```

## Where to start

Read `plans/README.md`. It picks a plan for you based on how long you have and
which loop you are facing. In short: the kernel first, the chapters for your
track, the questions as the drill surface, and the numbers card last.

## Provenance

Volumes I and II were imported (with full git history) from
`kapadias/deep-learning-essentials` and
`kapadias/natural-language-processing-essentials`; this repository is now the
canonical home. Volumes III and IV were written here.
