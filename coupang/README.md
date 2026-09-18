# Coupang Search & Discovery — the four-round loop

A targeted cut of the four-volume corpus for one specific interview: **ML
depth**, **ML breadth**, a **leadership round that is really a technical project
deep-dive**, and **DS&A coding**. The requisition is **Staff Machine Learning
Engineer, Search & Discovery** (R0073340, Mountain View) — two headcount, four
onsite candidates, so the loop is a comparison rather than a bar check.

| Part | What it is | Count |
|---|---|---|
| `brief.md` | What the org is, what each round tests, the 15 leadership principles | — |
| `deepdive.md` | The set-piece ML-depth question, worked end to end | — |
| `breadth_round.md` | Parth's round: breadth + hiring manager + team fit in one slot — how to play it | — |
| `parth_round.md` | The same round worked end to end: sweep answers, the behavioural half, his follow-up tree | — |
| ML depth | The search funnel front to back: query understanding, retrieval, ranking, evaluation | 44 |
| ML breadth | Foundations, embeddings, tabular, evaluation, production craft | 34 |
| `project.md` | Narrative spine, the numbers to know cold, the interrogation bank, LP mapping | — |
| Coding | DS&A biased to search patterns, every solution tested | 20 |

## Building

    make coupang          # tests, then PDF and web
    make coupang-test     # just the coding solutions

## The set-piece question

The recruiter shared the exact question the ML-depth interviewer (Siwen) asked
another candidate, along with her scorecard. `deepdive.md` works it end to end:

> Design a retrieval and ranking system for an e-commerce mobile app search bar.
> Retrieve highly relevant products for ambiguous long-tail queries — e.g.
> "durable lightweight winter boots for toddlers" — and rank to optimise CVR.
> 10M products, 200 QPS, 100 ms, six months of logs.

Both halves get equal weight, because the scorecard note about the previous
candidate being thinner on retrieval was about that candidate, not a prediction
about this one. What it is still good for is the vocabulary she probes with on
the retrieval side — "contrastive learning, embedding models, or softmax" is her
list — which says where she puts the bar when she digs.

**Retrieval:** the contrastive objective, positives and their position bias,
in-batch sampling bias and the logQ correction, the zero-engagement items that
are never sampled as negatives at all, hard-negative refresh, false-negative
filtering, distillation, embedding geometry, and honest offline evaluation.

**Ranking:** why "optimise CVR" is a trap, pointwise vs pairwise vs listwise and
why the expected-value objective rules out a pure LambdaMART score, negative-
downsampling recalibration, propensity estimation three ways, ESMM, delayed
feedback, how L1 is actually trained, point-in-time correctness and
log-and-train, and exploration for cold listings.

**The seam:** how each stage is trained on the other's output, and why a
6-point recall win shows up as a flat A/B.

## Figures, decision blocks, callouts

Diagrams are hand-authored SVG in `tools/figures.py`, written once and rendered
twice: inlined into the page with CSS custom properties so they follow the
viewer's theme, and through cairosvg into PDFs for LaTeX. No TikZ, so no silent
figure breakage, and `render()` fails loudly on an unresolved palette token or a
dangling marker reference.

Two markdown constructs carry the parts that are hard to practise from prose:

    @decide Which contrastive objective?
    - chose: Sampled softmax / InfoNCE over the candidate set
    - over: Triplet loss with a margin, or BPR
    - why: the softmax normalises over many negatives at once ...
    - cost: large batches, and a temperature that needs tuning
    @end

    !say    what to actually say out loud
    !trap   the trap in the question
    !push   the follow-up she will make
    !num    the numbers to have in your head

Both render as styled blocks in the page and boxed environments in the PDF, so
the justification for every major choice — and the alternative it was chosen
over — is scannable rather than buried mid-paragraph.

## Where the content comes from

27 depth answers were written for this loop — the query-understanding,
multilingual, LTR, and multi-task ranking material the general deck does not
carry. The other 51 questions are reused from `drill/questions.json`, selected
by question text rather than by id, because the drill deck renumbers on every
merge and an id list silently re-points at different questions.

Company facts (the funnel stages, the 100+ ranking criteria, the multilingual
index, the 15 leadership principles, the published interview guidance) come
from Coupang's own job postings, engineering blog and careers site, and are
cited inline where they carry weight.

## The coding solutions are tested

`coupang/test_coding.py` exercises all 20 reference solutions — 63 assertions
including the edge cases each problem's write-up tells you to raise. `make
coupang` runs it before rendering anything, because shipping a prep document
with a subtly wrong solution is worse than shipping one with no code.

## Open

`project.md` is a scaffold. The narrative, the numbers and the
architecture-specific follow-ups get filled in once the project document lands.
