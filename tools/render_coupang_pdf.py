#!/usr/bin/env python3
"""Render the Coupang loop document to LaTeX.

Four parts in interview order, plus the two prose sections. Shares the drill
deck's escaping rules (tools/render_drill_pdf.py) so the same silent-corruption
guards apply: display math survives, literal currency survives, and an unmapped
non-ASCII character stops the build instead of vanishing.
"""
import json, os, sys
from collections import OrderedDict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from render_drill_pdf import esc          # noqa: E402
from md2tex import convert                # noqa: E402

C = os.path.join(ROOT, "coupang")

PREAMBLE = r"""\documentclass[10pt,twoside]{book}
\usepackage[margin=0.9in,inner=1.05in,top=0.85in,bottom=0.9in]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{xcolor}
\usepackage{enumitem}
\usepackage{titlesec}
\usepackage{fancyhdr}
\usepackage{microtype}
\usepackage{needspace}
\usepackage{tabularx}
\usepackage{graphicx}
\graphicspath{{./}}
\newcolumntype{Y}{>{\raggedright\arraybackslash}X}
\usepackage{tcolorbox}
\usepackage{fvextra}
\usepackage[hidelinks]{hyperref}
\definecolor{ink}{RGB}{20,26,24}
\definecolor{accent}{RGB}{14,110,99}
\definecolor{muted}{RGB}{94,107,102}
\definecolor{flag}{RGB}{166,64,44}
\definecolor{soft}{RGB}{233,241,238}
\definecolor{rulec}{RGB}{201,212,207}
\setlist[itemize]{leftmargin=1.1em, itemsep=1pt, topsep=2pt, parsep=0pt}
\setlist[enumerate]{leftmargin=1.4em, itemsep=2pt, topsep=3pt, parsep=0pt}
\titleformat{\chapter}[display]{\sffamily\huge\bfseries\color{accent}}{}{0pt}{}
\titlespacing*{\chapter}{0pt}{-30pt}{16pt}
\pagestyle{fancy}\fancyhf{}
\fancyhead[LE,RO]{\sffamily\small\color{muted}\thepage}
\fancyhead[LO,RE]{\sffamily\small\color{muted}\leftmark}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\chaptermark}[1]{\markboth{#1}{}}
\newcommand{\cchapter}[1]{\chapter{#1}}
\newcommand{\csection}[1]{\par\vspace{13pt}\noindent{\sffamily\large\bfseries\color{ink}\raggedright #1\par}\vspace{4pt}}
\newcommand{\csubsection}[1]{\par\vspace{9pt}\noindent{\sffamily\normalsize\bfseries\color{accent}\raggedright #1\par}\vspace{2pt}}
\newenvironment{callout}{\begin{tcolorbox}[colback=soft,colframe=accent,boxrule=0.5pt,left=7pt,right=7pt,top=5pt,bottom=5pt]}{\end{tcolorbox}}
\definecolor{saybg}{RGB}{226,240,236}
\definecolor{trapbg}{RGB}{248,229,223}
\definecolor{pushbg}{RGB}{237,239,245}
\definecolor{numbg}{RGB}{243,241,228}
\definecolor{pushfg}{RGB}{58,74,110}
\definecolor{numfg}{RGB}{122,98,26}
% One macro for all four so the label, rule and spacing stay identical.
\newtcolorbox{co@box}[3]{colback=#1,colframe=#2,boxrule=0pt,leftrule=2.6pt,
  arc=2pt,left=8pt,right=8pt,top=5pt,bottom=5pt,
  before upper={{\sffamily\bfseries\footnotesize\color{#2}#3\par\vspace{2pt}}}}
\newenvironment{calloutsay}{\begin{co@box}{saybg}{accent}{SAY THIS}}{\end{co@box}}
\newenvironment{callouttrap}{\begin{co@box}{trapbg}{flag}{THE TRAP}}{\end{co@box}}
\newenvironment{calloutpush}{\begin{co@box}{pushbg}{pushfg}{THE PUSH}}{\end{co@box}}
\newenvironment{calloutnum}{\begin{co@box}{numbg}{numfg}{NUMBERS}}{\end{co@box}}
\newtcolorbox{mathboxtc}{colback=white,colframe=rulec,boxrule=0.6pt,arc=3pt,
  left=6pt,right=6pt,top=2pt,bottom=2pt}
\newenvironment{mathbox}{\begin{mathboxtc}}{\end{mathboxtc}}
% No tabularx here: it scans for its own \end at expansion time, so splitting
% one across \newenvironment's begin/end halves fails with \TX@get@body.
\newenvironment{decide}[1]{%
  \par\smallskip\noindent
  \begin{tcolorbox}[colback=white,colframe=accent,boxrule=1pt,arc=3pt,
    left=8pt,right=8pt,top=6pt,bottom=5pt,
    title={\sffamily\bfseries\small #1},coltitle=white,colbacktitle=accent,
    fonttitle=\sffamily]%
  \small%
}{\end{tcolorbox}\par\smallskip}
\newcommand{\drow}[2]{%
  \par\noindent
  \begin{minipage}[t]{58pt}\raggedright\strut
    {\sffamily\bfseries\scriptsize\color{accent}#1}\end{minipage}%
  \hspace{6pt}%
  \begin{minipage}[t]{\dimexpr\linewidth-64pt\relax}\strut #2\end{minipage}%
  \par\vspace{3.5pt}}
\newcommand{\qhead}[4]{%
  \par\vspace{15pt}\noindent
  {\sffamily\small\color{muted}#1 \textperiodcentered\ #2 \textperiodcentered\ #3}\par\vspace{1pt}
  \noindent{\sffamily\large\bfseries\color{ink}\raggedright #4\par}\vspace{4pt}}
\newcommand{\lab}[1]{\par\vspace{5pt}\noindent{\sffamily\footnotesize\bfseries\color{accent}#1}\par\vspace{1pt}}
\newcommand{\flab}[1]{\par\vspace{5pt}\noindent{\sffamily\footnotesize\bfseries\color{flag}#1}\par\vspace{1pt}}
\newcommand{\qsep}{\vspace{3pt}\noindent{\color{muted}\rule{\linewidth}{0.3pt}}}
\color{ink}
\begin{document}
\frontmatter
\thispagestyle{empty}
\vspace*{1.9in}
{\sffamily\Huge\bfseries\color{accent} Coupang\\[2pt] Search \& Discovery\par}
\vspace{12pt}
{\sffamily\Large\color{ink} The four-round loop, at the Staff bar\par}
\vspace{26pt}
{\normalsize\color{muted}%
ML depth, ML breadth, the project deep-dive, and DS\&A coding --- prepared
against this specific loop rather than against machine learning in general.
Every answer is what you say out loud, not what you would write down.\par}
\vspace{10pt}
{\small\color{muted}Company facts are drawn from Coupang's own job postings,
engineering blog and published interview guidance; they are cited where they
matter. Regenerate with \texttt{make coupang}.\par}
\cleardoublepage
\tableofcontents
\mainmatter
"""


def qa(qs, title, intro):
    L = [r"\chapter{%s}" % esc(title), convert(intro, title)]
    section = None
    for q in qs:
        if q["section"] != section:
            section = q["section"]
            L.append(r"\csection{%s}" % esc(section))
        w = "q%s" % q["id"]
        L.append(r"\needspace{6\baselineskip}")
        L.append(r"\qhead{%s}{%s}{%s}{%s}" % (
            esc(q["level"], w), esc(q["type"], w), esc(q["section"], w),
            esc(q["question"], w + " question")))
        L.append(r"\lab{Say this}")
        for para in [p for p in q["say"].split("\n") if p.strip()]:
            L.append(esc(para.strip(), w + " say") + r"\par\vspace{3pt}")
        if q.get("numbers"):
            L.append(r"\lab{Numbers}")
            L.append(r"\begin{itemize}")
            for n in q["numbers"]:
                L.append(r"\item %s" % esc(n, w + " numbers"))
            L.append(r"\end{itemize}")
        if q.get("whiteboard"):
            L.append(r"\lab{If they hand you the marker}")
            L.append(esc(q["whiteboard"], w + " whiteboard") + r"\par")
        if q.get("coupang"):
            L.append(r"\begin{callout}\textbf{At Coupang.} %s\end{callout}"
                     % esc(q["coupang"], w + " coupang"))
        L.append(r"\flab{Loses the signal}")
        L.append(r"\begin{itemize}")
        for rf in q["red_flags"]:
            L.append(r"\item %s" % esc(rf, w + " red_flags"))
        L.append(r"\end{itemize}")
        L.append(r"\lab{Then they ask}")
        L.append(r"\begin{itemize}")
        for fu in q["followups"]:
            L.append(r"\item %s" % esc(fu, w + " followups"))
        L.append(r"\end{itemize}")
        L.append(r"\qsep")
    return L


def coding(problems, intro):
    L = [r"\chapter{The coding round}", convert(intro, "coding")]
    section = None
    for p in problems:
        if p["section"] != section:
            section = p["section"]
            L.append(r"\csection{%s}" % esc(section))
        w = "p%s" % p["n"]
        L.append(r"\needspace{8\baselineskip}")
        L.append(r"\qhead{%s}{%s}{%s}{%s}" % (
            esc("Problem %d" % p["n"], w), esc(p["difficulty"], w),
            esc(p["section"], w), esc(p["title"], w)))
        L.append(esc(p["prompt"], w + " prompt") + r"\par\vspace{4pt}")
        L.append(r"\lab{What makes you recognise it}")
        L.append(esc(p["cue"], w + " cue") + r"\par")
        L.append(r"\lab{Approach}")
        L.append(esc(p["approach"], w + " approach") + r"\par")
        L.append(r"\lab{Complexity}")
        L.append(esc(p["complexity"], w + " complexity") + r"\par")
        L.append(r"\lab{Edge cases to raise before you code}")
        L.append(r"\begin{itemize}")
        for e in p["edges"]:
            L.append(r"\item %s" % esc(e, w + " edges"))
        L.append(r"\end{itemize}")
        L.append(r"\lab{Reference solution}")
        L.append(r"\begin{Verbatim}[fontsize=\small,xleftmargin=6pt,"
                 r"commandchars=\\\{\}]")
        for line in p["code"].split("\n"):
            L.append(line.replace("\\", "\\textbackslash{}")
                         .replace("{", "\\{").replace("}", "\\}"))
        L.append(r"\end{Verbatim}")
        L.append(r"\lab{Then they ask}")
        L.append(esc(p["followup"], w + " followup") + r"\par")
        L.append(r"\qsep")
    return L


DEPTH_INTRO = """
The round is organised the way the funnel runs, front to back: query
understanding, retrieval, ranking, evaluation. Interviewers rarely walk it in
order, but you should be able to, because "which stage are we in" is the
question behind most follow-ups.

Answers here run longer than in breadth --- this is the round where you are
allowed to take three minutes. What you are not allowed to do is stop at the
first level. Every answer below carries the follow-up that is actually coming.
"""

BREADTH_INTRO = """
The bank. Shorter answers. Two minutes, commit in the first sentence, stop. Breadth is
scored on coverage and on not bluffing, and the fastest way to fail it is to
keep talking past the point where you knew the answer.

The selection leans towards what a search and recommendations engineer is
actually asked --- embeddings, tabular models, calibration, experiment design,
and production failure --- rather than towards LLM pretraining internals.
"""

CODING_INTRO = """
Coupang's published guidance says 2–3 problems, and names what they grade:
*"Clean code, Bug free"*, *"Code validation, identification of edge cases"*,
and mastery of *"sorting, recursion, and dynamic programming."* The word
absent from that list is cleverness.

So the protocol matters as much as the algorithm. Restate the problem. Ask
about the edge cases *before* you write — the list under each problem below
is what to ask. State your complexity target out loud, then write. When you
think you are finished, walk one concrete example through your own code line by
line; that single habit catches most of what loses this round.

Their guidance also mentions Java and CS fundamentals. The solutions here are
Python because that is what an ML candidate will reach for, but if you are
asked to use Java, say so early rather than switching halfway.

Every solution below is exercised by `coupang/test_coding.py`.
"""


def main():
    deck = json.load(open(os.path.join(C, "deck.json")))
    L = [PREAMBLE]
    L.append(convert(open(os.path.join(C, "brief.md")).read(), "brief.md"))
    L.append(convert(open(os.path.join(C, "deepdive.md")).read(), "deepdive.md"))
    L += qa(deck["depth"], "The ML depth round", DEPTH_INTRO)
    L.append(convert(open(os.path.join(C, "breadth_round.md")).read(),
                     "breadth_round.md"))
    L.append(convert(open(os.path.join(C, "parth_round.md")).read(),
                     "parth_round.md"))
    L += qa(deck["breadth"], "The ML breadth round", BREADTH_INTRO)
    L.append(convert(open(os.path.join(C, "project.md")).read(), "project.md"))
    L += coding(deck["coding"], CODING_INTRO)
    L.append(r"\end{document}")
    tex = "\n".join(x for x in L if x != "")
    out = os.path.join(C, "coupang.tex")
    open(out, "w").write(tex)
    print("wrote %s (%d depth, %d breadth, %d coding, %d lines)"
          % (out, len(deck["depth"]), len(deck["breadth"]),
             len(deck["coding"]), len(tex.splitlines())))


if __name__ == "__main__":
    main()
