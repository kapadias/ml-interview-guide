#!/usr/bin/env python3
"""A deliberately small markdown -> LaTeX converter.

Only handles the constructs used in coupang/*.md: ATX headings, paragraphs,
bullet lists, blockquotes, pipe tables, horizontal rules, and inline bold /
italic / code. It raises on anything it does not recognise rather than dropping
it silently, which is the failure mode that matters -- a converter that quietly
swallows a table is worse than one that stops.
"""
import re, sys

_UNI = {
    "\u2014": "---", "\u2013": "--", "\u00a0": "~", "\u00b7": r"$\cdot$",
    "\u00d7": r"$\times$", "\u2192": r"$\to$", "\u2248": r"$\approx$",
    "\u2264": r"$\leq$", "\u2265": r"$\geq$", "\u2212": "$-$",
    "\u00b1": r"$\pm$", "\u00b5": r"$\mu$", "\u00a7": r"\S{}",
    "\u2018": "`", "\u2019": "'", "\u201c": "``", "\u201d": "''",
    "\u00b2": "$^2$", "\u00b3": "$^3$", "\u2026": r"\ldots{}",
}


def inline(t, where=""):
    """Escape a span of body text, honouring **bold**, *italic* and `code`."""
    parts = re.split(r"(`[^`]*`)", t)
    out = []
    for i, p in enumerate(parts):
        if i % 2:                                  # inside backticks
            body = p[1:-1]
            for ch in "\\{}$&#^_%~":
                body = body.replace(ch, "\\" + ch if ch not in "\\^~"
                                    else {"\\": r"\textbackslash{}",
                                          "^": r"\textasciicircum{}",
                                          "~": r"\textasciitilde{}"}[ch])
            out.append(r"\texttt{%s}" % body)
            continue
        p = p.replace("\\", r"\textbackslash{}")
        for ch, rep in [("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_"),
                        ("{", r"\{"), ("}", r"\}"), ("$", r"\$"),
                        ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]:
            p = p.replace(ch, rep)
        p = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", p)
        p = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"\\textit{\1}", p)
        p = re.sub(r'(^|[\s(\[])"', r"\1``", p)
        p = p.replace('"', "''")
        for ch, rep in _UNI.items():
            p = p.replace(ch, rep)
        leftover = sorted({c for c in p if ord(c) > 127})
        if leftover:
            raise SystemExit("md2tex: unmapped non-ASCII %s in %s: %r"
                             % (leftover, where or "text", p[:90]))
        out.append(p)
    return "".join(out)


def convert(md, where=""):
    lines = md.split("\n")
    out, i = [], 0
    para = []

    def flush():
        if para:
            out.append(inline(" ".join(para), where) + r"\par")
            para.clear()

    while i < len(lines):
        ln = lines[i]
        s = ln.strip()

        if not s:
            flush(); i += 1; continue

        if s.startswith("```"):
            flush()
            lang = s[3:].strip()
            i += 1
            block = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i]); i += 1
            i += 1                                  # closing fence
            # Diagrams must be pure ASCII: XeTeX silently drops a glyph the
            # monospace font lacks, and box-drawing characters are exactly the
            # ones Latin Modern Mono does not have.
            bad = sorted({c for ln in block for c in ln if ord(c) > 126})
            if bad:
                raise SystemExit("md2tex: non-ASCII %s in a fenced block in %s "
                                 "- diagrams must be pure ASCII" % (bad, where))
            size = r"\footnotesize" if max((len(x) for x in block), default=0) > 74 else r"\small"
            out.append(r"\begin{Verbatim}[fontsize=%s,xleftmargin=4pt,samepage=false]" % size)
            out += block
            out.append(r"\end{Verbatim}")
            if lang:
                out.append(r"\nopagebreak{\footnotesize\color{muted}%s\par}\medskip"
                           % inline(lang, where))
            continue

        if s.startswith("#"):
            flush()
            lvl = len(s) - len(s.lstrip("#"))
            title = inline(s.lstrip("#").strip(), where)
            out.append({1: r"\cchapter{%s}", 2: r"\csection{%s}",
                        3: r"\csubsection{%s}"}.get(lvl, r"\csubsection{%s}") % title)
            i += 1; continue

        if re.fullmatch(r"-{3,}", s):
            flush()
            out.append(r"\medskip\hrule\medskip")
            i += 1; continue

        if s.startswith(">"):
            flush()
            block = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                block.append(lines[i].strip().lstrip(">").strip())
                i += 1
            out.append(r"\begin{callout}%s\end{callout}"
                       % inline(" ".join(x for x in block if x), where))
            continue

        if s.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            if len(rows) < 2 or not set(rows[1][0]) <= set("-: "):
                raise SystemExit("md2tex: table without a separator row in %s" % where)
            header, body = rows[0], rows[2:]
            n = len(header)
            # Ragged X columns throughout: justified narrow columns produce
            # pages of underfull-hbox warnings and visibly bad word spacing.
            # \noindent matters: a tabularx that starts a paragraph is pushed
            # right by \parindent and overflows the margin by exactly that much.
            out.append(r"\par\noindent{\small\setlength{\tabcolsep}{4pt}%")
            out.append(r"\begin{tabularx}{\linewidth}{%s}\hline" % ("Y" * n))
            out.append(" & ".join(r"\textbf{%s}" % inline(c, where) for c in header)
                       + r" \\ \hline")
            for r in body:
                r = (r + [""] * n)[:n]
                out.append(" & ".join(inline(c, where) for c in r) + r" \\")
            out.append(r"\hline\end{tabularx}}\par\medskip")
            continue

        if re.match(r"^[-*] ", s):
            flush()
            out.append(r"\begin{itemize}")
            while i < len(lines) and re.match(r"^\s*[-*] ", lines[i]):
                item = re.sub(r"^\s*[-*] ", "", lines[i])
                i += 1
                while i < len(lines) and lines[i].strip() and \
                        not re.match(r"^\s*[-*] |^\s*\d+\. |^#|^\||^>", lines[i]):
                    item += " " + lines[i].strip(); i += 1
                out.append(r"\item %s" % inline(item, where))
            out.append(r"\end{itemize}")
            continue

        if re.match(r"^\d+\. ", s):
            flush()
            out.append(r"\begin{enumerate}")
            while i < len(lines) and re.match(r"^\s*\d+\. ", lines[i]):
                item = re.sub(r"^\s*\d+\. ", "", lines[i])
                i += 1
                while i < len(lines) and lines[i].strip() and \
                        not re.match(r"^\s*[-*] |^\s*\d+\. |^#|^\||^>", lines[i]):
                    item += " " + lines[i].strip(); i += 1
                out.append(r"\item %s" % inline(item, where))
            out.append(r"\end{enumerate}")
            continue

        para.append(s)
        i += 1

    flush()
    return "\n".join(out)


if __name__ == "__main__":
    print(convert(open(sys.argv[1]).read(), sys.argv[1]))
