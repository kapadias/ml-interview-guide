#!/usr/bin/env python3
"""Markdown -> HTML for the same narrow subset tools/md2tex.py handles.

Same principle: raise on anything unrecognised rather than dropping it.
"""
import html, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def FIGSVG(name):
    """Inline the figure with CSS custom properties so it follows the theme."""
    import figures
    return figures.render(name, figures.WEB)


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

        if s.startswith("@fig:"):
            flush()
            name, _, cap = s[5:].partition(" ")
            svg = FIGSVG(name.strip())
            out.append('<figure class="dg">%s%s</figure>'
                       % (svg, "<figcaption>%s</figcaption>" % inline(cap.strip())
                          if cap.strip() else ""))
            i += 1; continue

        if s.startswith("@decide"):
            flush()
            title = s[len("@decide"):].strip()
            rows, i = [], i + 1
            while i < len(lines) and not lines[i].strip().startswith("@end"):
                # Keys may be short phrases ("drop it if", "fallback"), not just words.
                m = re.match(r"^\s*-\s*([\w][\w ]{0,13})\s*:\s+(.*)$", lines[i])
                if m:
                    rows.append([m.group(1), m.group(2)])
                elif lines[i].strip() and rows:
                    rows[-1][1] += " " + lines[i].strip()
                i += 1
            i += 1
            body = "".join('<div class="drow"><span class="dk %s">%s</span>'
                           '<span class="dv">%s</span></div>'
                           % (k.lower().replace(" ", "-"),
                              html.escape(k.upper()), inline(v))
                           for k, v in rows)
            out.append('<div class="decide"><div class="dt">%s</div>%s</div>'
                       % (inline(title), body))
            continue

        m = re.match(r"^!(say|trap|push|num)\s+(.*)$", s)
        if m:
            flush()
            kind, body = m.group(1), m.group(2)
            i += 1
            while i < len(lines) and lines[i].strip() and \
                    not re.match(r"^!(say|trap|push|num)\s|^@|^#|^\||^>|^```|^\s*[-*] ", lines[i]):
                body += " " + lines[i].strip(); i += 1
            LBL = {"say": "Say this", "trap": "The trap",
                   "push": "The push", "num": "Numbers"}
            out.append('<div class="co %s"><span class="col">%s</span>%s</div>'
                       % (kind, LBL[kind], inline(body)))
            continue

        if s == "$$":
            flush()
            i += 1
            block = []
            while i < len(lines) and lines[i].strip() != "$$":
                block.append(lines[i]); i += 1
            i += 1
            out.append('<div class="mathbox">$$%s$$</div>'
                       % html.escape("\n".join(block)))
            continue

        if s.startswith("```"):
            flush()
            cap = s[3:].strip()
            i += 1
            block = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i]); i += 1
            i += 1
            wide = max((len(x) for x in block), default=0) > 74
            out.append('<pre class="fig%s"><code>%s</code></pre>'
                       % (" wide" if wide else "", html.escape("\n".join(block))))
            if cap:
                out.append('<p class="figcap">%s</p>' % inline(cap))
            continue

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
