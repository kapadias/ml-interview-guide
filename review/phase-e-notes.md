# Phase E inputs — deferred items collected during Phases A–D

Phase E is the program-wide performance layer: the Part 0 kernel, the Core-50
tier, the numbers card, generated question indexes, drill builds, and study
plans. These items were deliberately deferred into it.

## 1. Appendix question indexes must be regenerated, not patched

Both volumes ship `sections/appendix_question_index.tex` written by hand. They
have been drifting since Phase A (chapter renumbering) and are now materially
stale: they do not list any question added in Phases B–C, they still advertise
questions Phase D deleted, and they predate Volumes III and IV entirely.

The fix is generation, not editing: parse every `interviewq` block across all
four volumes (each carries a level badge, a `\badgetype{}`, and a
`\badgefreq{}`), and emit per-volume indexes plus a program-wide master index.
That also delivers the round-type tagging the plan calls for (breadth / depth /
coding / design), which lets the drill sets be produced mechanically.

Known-stale rows found during Phase D, in `volumes/deep-learning/sections/appendix_question_index.tex`:

- lines 178, 632 — "Static vs contextual embeddings" (question deleted, its
  production angle folded into the ch6 body)
- lines 182, 822 — "Build a semantic search system — how choose and evaluate
  embeddings" (consolidated into the single canonical design question in [SR 4])
- lines 331, 823 — "Design embedding system for 100M product catalog" (same
  consolidation)

Still accurate at the time of checking: ch6 rows for memory estimation (179,
703), BPE interaction (180, 564), debugging embedding quality (181, 762),
clustering diagnosis (330, 761), and collapse (332, 565).

Also note: DL ch7 merged two near-identical HNSW throughput questions into one,
and dropped three questions to pointers ([SR 3], [SR 4], [SR 8]); NLP ch3
dropped one; NLP ch6's PEFT question bank was revisited. Any hand-patch of the
index would go stale again on the next pass — generate it.

## 2. Volumes I and II still use inline preambles

`style/essentials.sty` is the shared preamble and is used by Volumes III and IV
(`\usepackage{essentials}`). Volumes I and II still carry ~330-line inline
preambles that are near-identical to it. Adopting the shared file for them was
deferred to avoid churning two volumes mid-content-work. When adopting, diff
the inline preamble against the style file first: they agreed on every macro
except `\headrulewidth` (a fancyhdr internal) at last check.

Note that `make sync-style` copies `essentials.sty` into every volume,
including the two that do not use it; those unused copies were deleted in Phase
D to avoid carrying a stale duplicate of the shared style. Re-running
`sync-style` recreates them harmlessly.

## 3. README question counts are stale

The repository README quotes question counts from before Phases B–D. Refresh it
from the generated index rather than by hand.

## 4. Content items noticed but out of scope when found

- Figure quality in DL ch3: the loss-taxonomy figure has several overlapping
  level-2 boxes and the selection decision tree has label collisions in its top
  branches. Both predate this work and want a dedicated figure-cleanup pass.
- DL ch9 retains a stale "in-context learning emerges around 10B+" line and a
  "GPT-4 via API" phrasing in a design answer; both were outside the assigned
  scope when spotted.
