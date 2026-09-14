# Coupang Search & Discovery — the four-round loop

A targeted cut of the four-volume corpus for one specific interview: **ML
depth**, **ML breadth**, a **leadership round that is really a technical project
deep-dive**, and **DS&A coding**, at the Staff / L6 bar.

| Part | What it is | Count |
|---|---|---|
| `brief.md` | What the org is, what each round tests, the 15 leadership principles | — |
| `deepdive.md` | The set-piece ML-depth question, worked end to end, 11 diagrams | — |
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

Her scorecard on the previous candidate said he was stronger on ranking than on
"the retrieval side — like contrastive learning, embedding models, or softmax,"
and recommended an in-depth ML round to assess exactly that. So the retrieval-
training section is the long one: the contrastive objective, where positives and
negatives come from, in-batch sampling bias and the logQ correction, the
zero-engagement items that are never sampled as negatives at all, hard-negative
refresh, false-negative filtering, distillation, and how to evaluate a retriever
offline without fooling yourself.

Diagrams are pure ASCII on purpose. They render identically in the PDF and the
page, they cannot silently break the way TikZ does, and they are what you would
actually draw on the whiteboard.

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
