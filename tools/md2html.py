#!/usr/bin/env python3
"""Markdown -> HTML for the same narrow subset tools/md2tex.py handles.

Same principle: raise on anything unrecognised rather than dropping it.
"""
import html, re, sys


def inline(t):
    parts = re.split(r"(`[^`]*`)", t)
    out = []
    for i, p in enumerate(parts):
        if i % 2:
            out.append("<code>%s</code>" % html.escape(p[1:-1]))
            continue
        p = html.escape(p)
        p = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", p)
        p = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<em>\1</em>", p)
        out.append(p)
    return "".join(out)


def convert(md, where=""):
    lines = md.split("\n")
    out, i, para = [], 0, []

    def flush():
        if para:
            out.append("<p>%s</p>" % inline(" ".join(para)))
            para.clear()

    while i < len(lines):
        s = lines[i].strip()
        if not s:
            flush(); i += 1; continue

        if s.startswith("#"):
            flush()
            lvl = min(len(s) - len(s.lstrip("#")), 4)
            out.append("<h%d>%s</h%d>" % (lvl + 1, inline(s.lstrip("#").strip()), lvl + 1))
            i += 1; continue

        if re.fullmatch(r"-{3,}", s):
            flush(); out.append("<hr>"); i += 1; continue

        if s.startswith(">"):
            flush()
            block = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                block.append(lines[i].strip().lstrip(">").strip()); i += 1
            out.append("<blockquote>%s</blockquote>"
                       % inline(" ".join(x for x in block if x)))
            continue

        if s.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            if len(rows) < 2 or not set(rows[1][0]) <= set("-: "):
                raise SystemExit("md2html: table without separator row in %s" % where)
            n = len(rows[0])
            t = ['<div class="tw"><table><thead><tr>']
            t += ["<th>%s</th>" % inline(c) for c in rows[0]]
            t.append("</tr></thead><tbody>")
            for r in rows[2:]:
                r = (r + [""] * n)[:n]
                t.append("<tr>" + "".join("<td>%s</td>" % inline(c) for c in r) + "</tr>")
            t.append("</tbody></table></div>")
            out.append("".join(t))
            continue

        if re.match(r"^[-*] ", s) or re.match(r"^\d+\. ", s):
            flush()
            ordered = bool(re.match(r"^\d+\. ", s))
            pat = r"^\s*\d+\. " if ordered else r"^\s*[-*] "
            tag = "ol" if ordered else "ul"
            out.append("<%s>" % tag)
            while i < len(lines) and re.match(pat, lines[i]):
                item = re.sub(pat, "", lines[i]); i += 1
                while i < len(lines) and lines[i].strip() and \
                        not re.match(r"^\s*[-*] |^\s*\d+\. |^#|^\||^>", lines[i]):
                    item += " " + lines[i].strip(); i += 1
                out.append("<li>%s</li>" % inline(item))
            out.append("</%s>" % tag)
            continue

        para.append(s); i += 1

    flush()
    return "\n".join(out)


if __name__ == "__main__":
    print(convert(open(sys.argv[1]).read(), sys.argv[1]))
