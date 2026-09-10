# The drill deck

148 questions with the answer you would **say**, for ML breadth and depth
rounds. The four volumes in `volumes/` are a reference library — derivation
first, ~530 words per answer, math in 82% of them. Nobody asks you to write
equations in a breadth round; they ask you to explain. This is what you
rehearse from.

Each entry has:

| field | what it is |
|---|---|
| `say` | 120–250 words of prose. Roughly two minutes spoken. First sentence commits to an answer. |
| `numbers` | 0–3 figures worth quoting aloud, with units and assumptions. |
| `red_flags` | What loses the signal — phrased as the wrong thing a candidate says. |
| `followups` | Where the conversation goes next. |
| `whiteboard` | Rare (15 of 148). Only where an interviewer genuinely hands you a marker. |

## Building

    make drill        # PDF + web page
    make drill-pdf
    make drill-web

`drill/questions.base.json` is the input; `questions.json`, `drill.tex`,
`drill.pdf` and `web/index.html` all regenerate from it. Re-running the
pipeline is idempotent.

## The pipeline

    questions.base.json ──┐
    slices/slice7_repairs │
    drops.json            ├─ tools/merge_drill.py    ─→ questions.json
    drops_additions.json  │
    slices/slice6_additions ┘
                          │
    copyedits.json ───────┴─ tools/normalize_drill.py ─→ questions.json
                                                          │
                          tools/render_drill_pdf.py ──────┼─→ drill.tex → drill.pdf
                          tools/render_drill_web.py ──────┴─→ web/index.html

`merge_drill.py` validates the controlled topic list, the red-flag and
follow-up counts, and that no addition duplicates an existing question.
`normalize_drill.py` marks up bare identifiers (`d_k` → math, `min_child_weight`
→ code) and rewrites literal currency, because `$5 vs $120` is a syntactically
valid LaTeX math span that renders as italic nonsense without warning.
`render_drill_pdf.py` hard-fails on an unbalanced `$` or an unmapped non-ASCII
character rather than letting XeTeX drop a glyph silently.

## Two extraction bugs worth remembering

`tools/extract_drill.py` stripped LaTeX comments with `%.*?$`, which also ate
every escaped `\%`. Thirty-six of 150 answers were truncated mid-sentence — one
lost 374 words — and nothing failed. Fixed with a negative lookbehind; the 22
answers already re-cut from damaged text were re-cut again from the corrected
source.

The per-volume cap took questions in chapter order, so every deep-learning
chapter past 17 was cut before selection ever saw it. The deck had two
Inference & Serving questions and two Experimentation ones. Thirty were added
back from the dropped pool against thirty-three over-weighted drops.

## Coverage

| topic | n | | topic | n |
|---|---:|---|---|---:|
| Foundations | 8 | | Evaluation | 13 |
| Architectures | 17 | | Experimentation | 4 |
| Training | 18 | | Retrieval & Ranking | 16 |
| Inference & Serving | 9 | | Recommenders | 7 |
| NLP & LLMs | 25 | | Classical ML | 14 |
| Alignment & Safety | 9 | | Applied Craft | 8 |

L5 91 · L6 55 · L7 2. Median spoken answer 230 words.
