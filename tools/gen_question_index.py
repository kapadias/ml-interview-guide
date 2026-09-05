#!/usr/bin/env python3
"""Generate the per-volume question-index appendixes and the program master index.

Every interview question in the program lives in an `interviewq` environment that
carries a level badge, a type badge and a frequency badge. This script is the
single source of truth for the indexes: it parses those blocks and regenerates
the appendixes, so the indexes cannot drift from the content the way hand-written
ones did through Phases A-D.

Usage:  python3 tools/gen_question_index.py [--check]
        --check exits non-zero if the generated output differs from what is on
        disk, so CI can catch a stale index.
"""
import argparse, os, re, sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOLUMES = [
    ("deep-learning", "Volume I", "Deep Learning Essentials", "DL"),
    ("nlp", "Volume II", "NLP Essentials", "NLP"),
    ("search-recommendation", "Volume III", "Search & Recommendation Essentials", "SR"),
    ("conventional-ml", "Volume IV", "Conventional ML Essentials", "CML"),
]
LEVEL_NAME = {"Lfive": "L5", "Lsix": "L6", "Lseven": "L7"}
LEVEL_ORDER = ["L5", "L6", "L7"]
FREQ_NAME = {"freqhigh": "high", "freqmed": "medium", "freqlow": "low"}
FREQ_ORDER = ["high", "medium", "low"]

# Which interview round a question type drills. This is what makes per-round
# drill sets mechanical rather than hand-curated.
ROUND_OF_TYPE = {
    "Conceptual": "Breadth", "First Principles": "Breadth", "Applied": "Breadth",
    "Mathematical": "Depth", "Trade-off": "Depth", "Debugging": "Depth",
    "Estimation": "Depth", "Metric Derivation": "Depth", "Evaluation": "Depth",
    "Depth": "Depth",
    "Coding": "Coding",
    "System Design": "Design", "Architecture Design": "Design",
    "Program Design": "Design", "Protocol Design": "Design",
    "Behavioral": "Behavioral", "Communication": "Behavioral",
    "Judgment Call": "Behavioral",
}
ROUND_ORDER = ["Breadth", "Depth", "Coding", "Design", "Behavioral"]

QBLOCK = re.compile(r"\\begin\{interviewq\}(.*?)\\end\{interviewq\}", re.S)


def chapter_files(volume):
    """Section files in the order main.tex includes them, excluding the appendix."""
    main = open(os.path.join(ROOT, "volumes", volume, "main.tex")).read()
    out = []
    for inc in re.findall(r"\\include\{sections/([^}]+)\}", main):
        if "appendix" in inc:
            continue
        out.append(os.path.join(ROOT, "volumes", volume, "sections", inc + ".tex"))
    return out


def tidy(text):
    """Collapse a captured question into one clean line of LaTeX."""
    t = re.sub(r"\s+", " ", text).strip()
    return t.rstrip(".").strip()


def parse():
    """-> {volume_key: [chapter dicts]}; each chapter has its questions in order."""
    data = {}
    for vol, _, _, _ in VOLUMES:
        chapters = []
        for n, path in enumerate(chapter_files(vol), start=1):
            src = open(path).read()
            m = re.search(r"\\chapter\{(.+?)\}", src)
            title = tidy(m.group(1)) if m else os.path.basename(path)
            qs = []
            for block in QBLOCK.findall(src):
                qt = re.search(r"\\textbf\{Q:\s*(.*?)\}\s*\n", block, re.S)
                lvl = re.search(r"\\badge(Lfive|Lsix|Lseven)\b", block)
                typ = re.search(r"\\badgetype\{([^}]*)\}", block)
                frq = re.search(r"\\badgefreq\{\\(freqhigh|freqmed|freqlow)\}", block)
                if not (qt and lvl and typ and frq):
                    print(f"WARNING: malformed interviewq in {path}", file=sys.stderr)
                    continue
                ty = typ.group(1).strip()
                qs.append({
                    "text": tidy(qt.group(1)),
                    "level": LEVEL_NAME[lvl.group(1)],
                    "type": ty,
                    "freq": FREQ_NAME[frq.group(1)],
                    "round": ROUND_OF_TYPE.get(ty, "Depth"),
                })
            chapters.append({"num": n, "title": title, "questions": qs})
        data[vol] = chapters
    return data


def counts_table(rows, header):
    out = [r"\begin{tabular}{lr}", r"\toprule",
           rf"\textbf{{{header}}} & \textbf{{Count}} \\", r"\midrule"]
    for k, v in rows:
        out.append(rf"\quad {k} & {v} \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def render_volume(vol, roman, title, tag, chapters):
    qs = [q for c in chapters for q in c["questions"]]
    total = len(qs)
    by_lvl = Counter(q["level"] for q in qs)
    by_type = Counter(q["type"] for q in qs)
    by_freq = Counter(q["freq"] for q in qs)
    by_round = Counter(q["round"] for q in qs)

    L = []
    A = L.append
    A(r"% GENERATED FILE -- do not edit by hand.")
    A(r"% Regenerate with: make index   (tools/gen_question_index.py)")
    A(r"\chapter{Question Index}")
    A(r"\label{chap:question_index}")
    A("")
    A(r"\begin{tldr}")
    A(f"Every one of the {total} interview questions in {roman}, indexed by chapter, "
      r"by round, and by frequency. This appendix is generated from the questions "
      r"themselves, so it cannot drift from the chapters. Work the "
      r"\freqhigh{} list first: those are the questions that come up in almost every loop.")
    A(r"\end{tldr}")
    A("")
    A(r"\section{At a Glance}")
    A(r"\begin{table}[H]\centering")
    A(r"\caption{Question distribution}")
    rows = [(r"\textbf{Total}", rf"\textbf{{{total}}}")]
    rows += [(f"{l} questions", by_lvl.get(l, 0)) for l in LEVEL_ORDER if by_lvl.get(l)]
    rows += [(f"{r} round", by_round.get(r, 0)) for r in ROUND_ORDER if by_round.get(r)]
    rows += [(rf"\freqhigh{{}} asked constantly", by_freq.get("high", 0)),
             (rf"\freqmed{{}} common", by_freq.get("medium", 0)),
             (rf"\freqlow{{}} occasional", by_freq.get("low", 0))]
    A(counts_table(rows, "Category"))
    A(r"\end{table}")
    A("")

    # Highest-frequency questions first: the cram list.
    A(r"\section{Start Here: The \freqhigh{} Questions}")
    A("These come up in almost every loop at this level. If you have one evening, "
      "these are the evening.")
    A(r"\begin{enumerate}[leftmargin=*]")
    for c in chapters:
        for q in c["questions"]:
            if q["freq"] == "high":
                A(rf"\item \textbf{{Ch.~{c['num']}}} \quad {q['level']} \quad "
                  rf"\textit{{{q['type']}}} \quad {q['text']}")
    A(r"\end{enumerate}")
    A("")

    A(r"\section{By Chapter}")
    for c in chapters:
        if not c["questions"]:
            continue
        A(rf"\subsection*{{Chapter {c['num']}: {c['title']}}}")
        A(r"\begin{tabularx}{\textwidth}{@{}l l l X@{}}")
        A(r"\toprule")
        A(r"\textbf{Lvl} & \textbf{Freq} & \textbf{Type} & \textbf{Question} \\")
        A(r"\midrule")
        for q in c["questions"]:
            fr = {"high": r"\freqhigh{}", "medium": r"\freqmed{}", "low": r"\freqlow{}"}[q["freq"]]
            A(rf"{q['level']} & {fr} & {q['type']} & {q['text']} \\")
        A(r"\bottomrule")
        A(r"\end{tabularx}")
        A("")

    A(r"\section{By Round}")
    A("The rounds a loop is actually made of. Drill one column at a time.")
    for rnd in ROUND_ORDER:
        items = [(c, q) for c in chapters for q in c["questions"] if q["round"] == rnd]
        if not items:
            continue
        A(rf"\subsection*{{{rnd} ({len(items)} questions)}}")
        A(r"\begin{itemize}[leftmargin=*]")
        for c, q in items:
            A(rf"\item \textbf{{Ch.~{c['num']}}} \quad {q['level']} \quad {q['text']}")
        A(r"\end{itemize}")
        A("")
    return "\n".join(L) + "\n"


def render_master(data):
    L = []
    A = L.append
    total = sum(len(c["questions"]) for v in data.values() for c in v)
    A("# Master Question Index")
    A("")
    A("<!-- GENERATED FILE - do not edit by hand. Regenerate with `make index`. -->")
    A("")
    A(f"All **{total} questions** across the four volumes, so you can find a question "
      "without knowing which volume it lives in.")
    A("")
    A("| Volume | Questions | Breadth | Depth | Coding | Design | Behavioral |")
    A("|---|---|---|---|---|---|---|")
    for vol, roman, title, tag in VOLUMES:
        qs = [q for c in data[vol] for q in c["questions"]]
        r = Counter(q["round"] for q in qs)
        A(f"| {roman} — {title} | {len(qs)} | " +
          " | ".join(str(r.get(x, 0)) for x in ROUND_ORDER) + " |")
    A("")
    for vol, roman, title, tag in VOLUMES:
        A(f"## {roman} — {title}")
        A("")
        for c in data[vol]:
            if not c["questions"]:
                continue
            A(f"### [{tag} {c['num']}] {c['title']}")
            A("")
            for q in c["questions"]:
                star = {"high": "***", "medium": "**", "low": "*"}[q["freq"]]
                txt = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", q["text"])
                txt = txt.replace("\\", "").replace("$", "`")
                A(f"- `{q['level']}` `{q['type']}` {star} — {txt}")
            A("")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if any generated file is out of date")
    args = ap.parse_args()

    data = parse()
    outputs = {}
    for vol, roman, title, tag in VOLUMES:
        path = os.path.join(ROOT, "volumes", vol, "sections", "appendix_question_index.tex")
        outputs[path] = render_volume(vol, roman, title, tag, data[vol])
    outputs[os.path.join(ROOT, "index", "master_question_index.md")] = render_master(data)

    stale = []
    for path, content in outputs.items():
        existing = open(path).read() if os.path.exists(path) else None
        if existing != content:
            stale.append(os.path.relpath(path, ROOT))
            if not args.check:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                open(path, "w").write(content)

    total = sum(len(c["questions"]) for v in data.values() for c in v)
    if args.check:
        if stale:
            print("Question index is STALE; run `make index`:")
            for s in stale:
                print(f"  {s}")
            return 1
        print(f"Question index is up to date ({total} questions).")
        return 0
    print(f"Indexed {total} questions.")
    for s in stale:
        print(f"  wrote {s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
