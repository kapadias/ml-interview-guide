#!/usr/bin/env python3
"""Render drill/questions.json to a LaTeX drill deck.

One question per entry: the spoken answer first, then the numbers worth quoting,
what loses the signal, and what gets asked next. Deliberately plain - this is a
document you reread under pressure, not a textbook.
"""
import json, os, re, sys
from collections import OrderedDict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOPIC_ORDER = ["Foundations", "Architectures", "Training", "Inference & Serving",
               "NLP & LLMs", "Alignment & Safety", "Evaluation", "Experimentation",
               "Retrieval & Ranking", "Recommenders", "Classical ML", "Applied Craft"]


# Written as \$ in the source JSON; must survive the math-span split below.
_CURRENCY = "\x00CUR\x00"

# XeTeX will happily set a glyph the font does not have as nothing at all, with
# no warning, so every non-ASCII character we actually use is mapped explicitly.
_UNI = {
    "\u2014": "---", "\u2013": "--", "\u00a0": "~", "\u00b7": r"$\cdot$",
    "\u00d7": r"$\times$", "\u2192": r"$\to$", "\u2248": r"$\approx$",
    "\u2264": r"$\leq$", "\u2265": r"$\geq$", "\u2212": "$-$",
    "\u221a": r"$\sqrt{\,}$", "\u00a7": r"\S{}", "\u00b1": r"$\pm$",
    "\u00b5": r"$\mu$", "\u00bd": r"$\tfrac12$", "\u2026": r"\ldots{}",
    "\u00b2": "$^2$", "\u00b3": "$^3$", "\u00b9": "$^1$",
    "\u2070": "$^0$", "\u2074": "$^4$", "\u2077": "$^7$", "\u2078": "$^8$",
    "\u2079": "$^9$", "\u1d4f": "$^k$", "\u2082": "$_2$", "\u208a": "$_+$",
    "\u03b1": r"$\alpha$", "\u03b2": r"$\beta$", "\u03bb": r"$\lambda$",
    "\u03c3": r"$\sigma$", "\u03c4": r"$\tau$", "\u03b4": r"$\delta$",
    "\u03c1": r"$\rho$", "\u03bc": r"$\mu$", "\u03b5": r"$\epsilon$",
    "\u03b3": r"$\gamma$", "\u03b8": r"$\theta$", "\u03c0": r"$\pi$",
    "\u2211": r"$\sum$", "\u221e": r"$\infty$", "\u2260": r"$\neq$",
}


def esc(t, where=""):
    """Markdown-ish -> LaTeX, preserving math spans untouched.

    Both delimiters matter: an earlier version split on `\\$[^$]*\\$` only, which
    matched the opening `$$` of a display block as an empty math span and then
    escaped the body's backslashes into \\textbackslash{} soup -- silently, with
    exit 0. Literal currency must be written \\$ in the JSON, because `$5 vs
    $120` is otherwise a perfectly valid (and wrong) math span.
    """
    t = t.replace("\\$", _CURRENCY)
    parts = re.split(r"(\$\$.*?\$\$|\$[^$]*\$)", t, flags=re.S)
    out = []
    for i, p in enumerate(parts):
        if i % 2:                                  # inside math
            out.append(p.replace(_CURRENCY, r"\$"))
            continue
        if "$" in p:
            raise SystemExit(
                f"unbalanced $ in {where or 'text'} (write literal currency as "
                f"\\\\$ in the JSON): {p[:120]!r}")
        p = p.replace("\\", "\\textbackslash{}")
        for ch, rep in [("&", "\\&"), ("%", "\\%"), ("#", "\\#"), ("_", "\\_"),
                        ("{", "\\{"), ("}", "\\}"), ("~", "\\textasciitilde{}"),
                        ("^", "\\textasciicircum{}")]:
            p = p.replace(ch, rep)
        p = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", p)
        p = re.sub(r"`(.+?)`", r"\\texttt{\1}", p)
        # Straight quotes must be split into open/close pairs; replacing both
        # with '' put a closing quote at the start of every quoted phrase.
        p = re.sub(r'(^|[\s(\[])"', r"\1``", p)
        p = p.replace('"', "''")
        for ch, rep in _UNI.items():
            p = p.replace(ch, rep)
        leftover = sorted({c for c in p if ord(c) > 127})
        if leftover:
            raise SystemExit(f"unmapped non-ASCII {leftover} in {where or 'text'}: {p[:120]!r}")
        out.append(p.replace(_CURRENCY, r"\$"))
    return "".join(out)


def render(qs):
    L = []
    A = L.append
    A(r"""\documentclass[10pt,twoside]{book}
\usepackage[margin=0.9in,inner=1.05in,top=0.85in,bottom=0.9in]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{xcolor}
\usepackage{enumitem}
\usepackage{titlesec}
\usepackage{fancyhdr}
\usepackage{microtype}
\usepackage{needspace}
\usepackage[hidelinks]{hyperref}
\definecolor{ink}{RGB}{25,32,44}
\definecolor{accent}{RGB}{0,74,132}
\definecolor{muted}{RGB}{110,120,135}
\definecolor{flag}{RGB}{150,45,40}
\setlist[itemize]{leftmargin=1.1em, itemsep=1pt, topsep=2pt, parsep=0pt}
\titleformat{\chapter}[display]{\sffamily\huge\bfseries\color{accent}}{}{0pt}{}
\titlespacing*{\chapter}{0pt}{-30pt}{18pt}
\pagestyle{fancy}\fancyhf{}
\fancyhead[LE,RO]{\sffamily\small\color{muted}\thepage}
\fancyhead[LO,RE]{\sffamily\small\color{muted}\leftmark}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\chaptermark}[1]{\markboth{#1}{}}
\newcommand{\qhead}[3]{%
  \par\vspace{14pt}\noindent
  {\sffamily\small\color{muted}#1 \textperiodcentered\ #2}\par\vspace{1pt}
  \noindent{\sffamily\large\bfseries\color{ink}\raggedright #3\par}\vspace{4pt}}
\newcommand{\lab}[1]{\par\vspace{5pt}\noindent{\sffamily\footnotesize\bfseries\color{accent}#1}\par\vspace{1pt}}
\newcommand{\flab}[1]{\par\vspace{5pt}\noindent{\sffamily\footnotesize\bfseries\color{flag}#1}\par\vspace{1pt}}
\color{ink}
\begin{document}
\frontmatter
\thispagestyle{empty}
\vspace*{2.2in}
{\sffamily\Huge\bfseries\color{accent} The Drill Deck\par}
\vspace{10pt}
{\sffamily\Large\color{ink} ML breadth and depth rounds, answered out loud\par}
\vspace{26pt}
{\normalsize\color{muted}%
Every question here is one you get asked. Every answer is what you say---%
two to three minutes of speech, not a page of derivation. The equations that
survived are the ones an interviewer actually hands you a marker for.\par}
\vspace{12pt}
{\normalsize\color{muted}Work the \textit{Say this} block until you can deliver it
without reading. The \textit{Numbers} are what you quote; the \textit{Loses the
signal} lines are what you must not do; the \textit{Then they ask} lines are where
the conversation goes next.\par}
\vfill
{\small\color{muted}Generated from the four-volume reference set. Regenerate with \texttt{make drill}.\par}
\cleardoublepage
\tableofcontents
\mainmatter""")

    groups = OrderedDict((t, []) for t in TOPIC_ORDER)
    for q in qs:
        groups.setdefault(q.get("topic", "Other"), []).append(q)

    for topic, items in groups.items():
        if not items:
            continue
        A(r"\chapter{%s}" % esc(topic))
        A(r"\addtocontents{toc}{\protect\thispagestyle{fancy}}")
        for q in items:
            A(r"\needspace{6\baselineskip}")
            w = "q%s" % q["id"]
            A(r"\qhead{%s}{%s}{%s}" % (esc(q["level"], w), esc(q["type"], w), esc(q["question"], w + " question")))
            A(r"\lab{Say this}")
            for para in [p for p in q["say"].split("\n") if p.strip()]:
                A(esc(para.strip(), w + " say") + r"\par\vspace{3pt}")
            if q.get("numbers"):
                A(r"\lab{Numbers}")
                A(r"\begin{itemize}")
                for n in q["numbers"]:
                    A(r"\item %s" % esc(n, w + " numbers"))
                A(r"\end{itemize}")
            if q.get("whiteboard"):
                A(r"\lab{If they hand you the marker}")
                wb = q["whiteboard"]
                wb = wb if isinstance(wb, str) else " ".join(wb)
                A(esc(wb, w + " whiteboard") + r"\par")
            A(r"\flab{Loses the signal}")
            A(r"\begin{itemize}")
            for rf in q["red_flags"]:
                A(r"\item %s" % esc(rf, w + " red_flags"))
            A(r"\end{itemize}")
            A(r"\lab{Then they ask}")
            A(r"\begin{itemize}")
            for fu in q["followups"]:
                A(r"\item %s" % esc(fu, w + " followups"))
            A(r"\end{itemize}")
            A(r"\vspace{2pt}\noindent{\color{muted}\rule{\linewidth}{0.3pt}}")
    A(r"\end{document}")
    return "\n".join(x for x in L if x != "")


def main():
    qs = json.load(open(os.path.join(ROOT, "drill", "questions.json")))["questions"]
    tex = render(qs)
    out = os.path.join(ROOT, "drill", "drill.tex")
    open(out, "w").write(tex)
    print(f"wrote {out} ({len(qs)} questions, {len(tex.splitlines())} lines)")


if __name__ == "__main__":
    main()
