# Re-cut spec: turning textbook answers into spoken answers

The four volumes are a reference library — derivation-first, ~530 words per
answer, math in 82% of them. Nobody asks you to write equations in a breadth or
depth round; they ask you to *explain*. This deck is what you rehearse from.

Each question in your slice has `answer_source`, `red_flags_source` and
`followups_source` extracted from the book. Rewrite them into these fields, then
set `"status": "recut"`:

## `say` — required. The answer you would speak.

- **120–250 words.** Time it: a 2–3 minute answer is ~300 spoken words, and you
  want room to breathe. Long answers are the single biggest defect in the source.
- **Plain prose in 2–4 short paragraphs.** No bullet lists, no bold-label
  scaffolding like "**1. Class imbalance**:" — that is how you write, not how you
  talk. Write what a strong candidate actually says.
- **Lead with the answer, not the setup.** First sentence should already commit
  to a position. Interviewers reward candidates who answer first and elaborate
  second.
- **Strip the math.** Remove every equation unless the number or relationship
  *is* the answer (e.g. "attention is O(n²) in sequence length", "20 tokens per
  parameter"). Inline symbols are fine where a name would be clumsy. If the
  source's argument depends on a derivation, state the conclusion and why it
  holds in words.
- **Keep the concrete.** Named systems, real failure stories, the specific
  number — these are what make an answer sound like experience rather than
  recitation. Do not sand the specificity out while shortening.
- Do not invent facts. Everything must be supported by the source answer. If the
  source is wrong or stale, fix it and note that in `notes`.

## `numbers` — optional, 0–3 items.

Short strings: the figures worth quoting aloud for this question, each with its
unit and any assumption ("H100 BF16 dense: ~990 TFLOP/s", "Chinchilla: ~20
tokens per parameter"). Only include what a candidate would genuinely cite in
this answer. Most questions need none.

## `red_flags` — required, 2–3 items.

One line each, phrased as the wrong thing a candidate says or does. Compress the
source's list; keep the ones that actually cost the signal.

## `followups` — required, 2–3 items.

The questions an interviewer asks next. Short. Take from the source where good.

## `whiteboard` — optional, and rare.

Only for questions where an interviewer genuinely hands you a marker (deriving
attention, the bias–variance decomposition, a gradient). One or two lines of
LaTeX math in `$...$` or `$$...$$`, plus a sentence on what to say while writing
it. If in doubt, omit — fewer than one question in eight should have this.

## `topic` — required.

A short topic label for grouping and filtering, from this controlled list:
`Foundations`, `Architectures`, `Training`, `Evaluation`, `NLP & LLMs`,
`Alignment & Safety`, `Inference & Serving`, `Retrieval & Ranking`,
`Recommenders`, `Classical ML`, `Experimentation`, `Applied Craft`.

## `notes` — optional.

Anything the reviewer should know: a fact you corrected, a source answer that
was weak, a question you think should be cut.

Keep `id`, `volume`, `chapter`, `chapter_title`, `level`, `type`, `question`
unchanged. You may lightly reword `question` only to make it read like something
spoken aloud. Leave the `*_source` fields in place.
