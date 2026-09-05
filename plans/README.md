# Study Plans

Five schedules over the four volumes. Pick one, follow it, ignore the others.

The program is 58 chapters and 543 questions across ~62,000 lines. Nobody reads
that. These plans decide what you read, in what order, and how you know you are
done with a day.

## Pick a plan

| Time you have | Loop you are facing | Plan |
|---|---|---|
| 2 weeks | Generalist ML loop: platform company, applied/product org, or a screen with a named "ML fundamentals" round | [`two-week-breadth.md`](two-week-breadth.md) |
| 4 weeks | Frontier lab (OpenAI/Anthropic-class): depth grill, LLM training or serving design, agents, safety | [`llm-depth-track.md`](llm-depth-track.md) |
| 4 weeks | Search, recommendation, ads, marketplace ranking, or a retrieval-heavy platform loop | [`search-recsys-track.md`](search-recsys-track.md) |
| 1 week | The ML coding round is your weak surface (or it is the next round on the calendar) | [`coding-week.md`](coding-week.md) |
| One evening | Tomorrow | [`night-before.md`](night-before.md) |

Two combinations that work: run `coding-week.md` inside week 3 of a depth track
(the depth reading is compatible with 3 hours of drilling a day, cover-to-cover
reading is not), and always end with `night-before.md` regardless of what came
before.

If you have four weeks and do not know which depth track: read
[DL 29] *The Staff Loop*, §"Anatomy of the Loop", then ask your recruiter which
of the five surfaces each of their rounds samples. Recruiters answer this
accurately and it costs you one email.

## How to use the program

**The kernel first.** Twenty-six principles from which most interview answers
can be re-derived on the spot — cross-entropy as compression, 6ND, the roofline,
attention as a soft KV lookup, Goodhart. Read them before the chapters so the
chapters land on a frame instead of a pile. One hour, once.

*Where it is:* `kernel/part0_kernel.tex` — 26 principles in six groups (A
objectives, B scale, C optimization, D the machine, E architecture and
adaptation, F proxies), each with its statement, where in the program it does
real work, and what an interviewer learns from your handling of it. It ends
with §"The Kernel on One Page," a single table of names and half-line
statements. That table is the night-before artifact.

**The Core Fifty** (`kernel/core50.tex`) is the tier between the kernel and the
543 questions: fifty items you cannot skip, each with the substance to state,
the pointer to the full treatment, and the specific wrong answer that costs the
signal. Its own cadence is five a day for ten days, spoken, marked red/amber/
green — only reds earn chapter time. Its closing checklist is the last-48-hours
pass.

*A build note:* the three `kernel/` files are LaTeX source shared across the
four volumes and are not yet wired into any volume's `main.tex`, so they do not
appear in the built PDFs. Read the `.tex` directly; it is plain enough. Their
long-form sources — with more worked examples than the cards carry — are
`review/first-principles/knowledge-kernel.md` and `numbers-fluency.md`.

**Chapters in between.** Every chapter in all four volumes opens with a `tldr`
box. Read it, work the chapter's questions aloud *before* reading their answers,
then read only the sections your misses point at. Cover-to-cover reading is for
the depth track's claimed specialty and nowhere else; the breadth sprint does
not have room for it and the breadth round does not reward it.

**Questions as the drill surface.** `index/master_question_index.md` lists all
543 questions with chapter, level (L5/L6/L7), type, and frequency. Frequency is
the filter that matters:

- `***` = asked constantly (243 questions)
- `**` = common (255)
- `*` = occasional (45)

Drill `***` first, always. Answer out loud, timed at two minutes, before you
open the chapter — writing is not speaking, and the round is spoken. Regenerate
the index after any content change with `make index`.

**The numbers card last.** GPU specs, the 6ND rule, the training memory ledger,
KV-cache bytes, serving economics, and four worked examples you must be able to
run end to end. It is a recall artifact, not a reading artifact: it goes in the
final week and the final evening, after the concepts have somewhere to attach.

*Where it is:* `kernel/numbers_card.tex` — seven sections (hardware; model
arithmetic; training at scale; inference and serving; retrieval, embeddings and
ranking; statistics and evaluation) closing with §7, "Orders of magnitude: the
night before," a fifteen-row table you should be able to produce cold. The
longer treatment, with the four canonical worked examples spelled out end to
end, is `review/first-principles/numbers-fluency.md` §8.

## What the program does not cover

Say this to yourself now rather than discovering it during a loop. Per
[DL 29] and `review/first-principles/interview-loop.md`, these surfaces decide
staff-level loops and no chapter here prepares them:

- **The project deep dive.** The loop's authenticity check; committees read its
  write-up first, and it produces vetoes constantly. Prepare it from your own
  history using [DL 29] §"The Project Deep Dive" as the structure.
- **Behavioral / staff-scope evidence.** The primary down-levelling instrument.
  [DL 29] §"Behavioral and Cross-Functional Rounds" gives the scenarios; the
  stories are yours.
- **The general SWE round.** Most frontier-lab loops contain at least one round
  with no ML in it — build a small stateful system, find the deadlock, profile
  the pipeline. Nothing in the four volumes touches it.

## Program state, as of these plans

- **Volume I — Deep Learning Essentials:** 29 chapters, 301 questions. Complete.
- **Volume II — NLP Essentials:** 13 chapters, 128 questions. Complete.
- **Volume III — Search & Recommendation Essentials:** 8 chapters, 92 questions.
  Complete.
- **Volume IV — Conventional ML Essentials:** 8 chapters, but **only [CML 2]
  Trees and Ensembles and [CML 6] Experimentation and Causal Inference are
  written** (22 questions between them). Chapters 1, 3, 4, 5, 7, and 8 are
  scaffolds — a title, a placeholder `tldr`, and section headings with
  `% TODO Phase C3`. No plan here sends you to them. Where a plan would have
  used them, it says so and names the substitute.

## Conventions

Cross-volume pointers are `[DL 5]`, `[NLP 9]`, `[SR 4]`, `[CML 2]` — volume and
chapter number, as defined in the Program Map at the front of every volume.
Chapter files live at `volumes/<volume>/sections/NN_*.tex`; the mapping is
`deep-learning` = DL, `nlp` = NLP, `search-recommendation` = SR,
`conventional-ml` = CML.

Question text is quoted from `index/master_question_index.md` with its LaTeX
escaping rendered (`times` → ×, `10^{22}` → 10^22). Wording is otherwise
verbatim, so every quoted question is greppable in the index.
