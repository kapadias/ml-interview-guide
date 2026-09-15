#!/usr/bin/env python3
"""Hand-authored SVG figures for the Coupang deep-dive.

Each figure is written once and rendered twice: inline into the page with CSS
custom properties so it follows the viewer's theme, and through cairosvg into a
PDF with literal light-theme colours for LaTeX to include.

The rule from the diagramming brief: draw the mechanism the argument turns on,
label every arrow, one claim per figure. A box labelled "cache" is worth less
than the prose it sits next to.
"""
import os, re, sys

WEB = {  # resolve against the page's theme tokens
    "ink": "var(--ink)", "muted": "var(--muted)", "faint": "var(--faint)",
    "accent": "var(--accent)", "soft": "var(--accent-soft)",
    "flag": "var(--flag)", "flagsoft": "var(--flag-soft)",
    "surface": "var(--card)", "rule": "var(--rule)", "raise": "var(--raise)",
}
PRINT = {
    "ink": "#141A18", "muted": "#5E6B66", "faint": "#93A19B",
    "accent": "#0E6E63", "soft": "#D7E6E2",
    "flag": "#A6402C", "flagsoft": "#F5E4DF",
    "surface": "#FFFFFF", "rule": "#C9D4CF", "raise": "#F1F5F3",
}

FONT = "'Archivo',system-ui,-apple-system,'Segoe UI',sans-serif"
MONO = "'JetBrains Mono',ui-monospace,Menlo,monospace"


# ---------------------------------------------------------------- primitives
def e(t):
    """SVG is XML: an unescaped < in a label (s(q,d) = <f(q),g(d)>) is a parse
    error, and cairosvg reports it as a column number in a 4000-char line."""
    return (str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def box(x, y, w, h, label, sub=None, fill="surface", stroke="rule", tone="ink",
        rx=6, bold=True, size=13):
    """A labelled rectangle. `sub` is a second, smaller line beneath."""
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
           f'fill="{{{fill}}}" stroke="{{{stroke}}}" stroke-width="1.2"/>']
    cy = y + h / 2 + (0 if not sub else -7)
    weight = "600" if bold else "400"
    out.append(f'<text x="{x + w/2}" y="{cy + 4}" text-anchor="middle" '
               f'font-family="{FONT}" font-size="{size}" font-weight="{weight}" '
               f'fill="{{{tone}}}">{e(label)}</text>')
    if sub:
        out.append(f'<text x="{x + w/2}" y="{cy + 20}" text-anchor="middle" '
                   f'font-family="{MONO}" font-size="10.5" fill="{{muted}}">{sub}</text>')
    return "".join(out)


def arrow(x1, y1, x2, y2, label=None, tone="ink", dash=None, lx=None, ly=None,
          anchor="middle", marker="arrow"):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    out = [f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{{{tone}}}" '
           f'stroke-width="1.4"{d} marker-end="url(#{marker})"/>']
    if label:
        tx = lx if lx is not None else (x1 + x2) / 2
        ty = ly if ly is not None else (y1 + y2) / 2 - 5
        out.append(f'<text x="{tx}" y="{ty}" text-anchor="{anchor}" '
                   f'font-family="{MONO}" font-size="10" fill="{{muted}}">{e(label)}</text>')
    return "".join(out)


def text(x, y, s, size=11, tone="muted", anchor="start", family=FONT, weight="400"):
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{family}" '
            f'font-size="{size}" font-weight="{weight}" fill="{{{tone}}}">{e(s)}</text>')


def frame(w, h, body, label):
    # No inline style on the root: cairosvg renders a completely blank image
    # when it sees height:auto, silently. Page CSS sizes the web copy instead.
    return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{e(label)}" '
            f'xmlns="http://www.w3.org/2000/svg">'
            f'<defs>'
            f'<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
            f'markerHeight="6" orient="auto-start-reverse">'
            f'<path d="M0,1 L9,5 L0,9 z" fill="{{ink}}"/></marker>'
            f'<marker id="arrowA" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
            f'markerHeight="6" orient="auto-start-reverse">'
            f'<path d="M0,1 L9,5 L0,9 z" fill="{{accent}}"/></marker>'
            f'<marker id="arrowF" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
            f'markerHeight="6" orient="auto-start-reverse">'
            f'<path d="M0,1 L9,5 L0,9 z" fill="{{flag}}"/></marker>'
            f'</defs>{body}</svg>')


# ------------------------------------------------------------------ figures
def fig_funnel():
    """Claim: each stage sees ~10x fewer items and costs ~10x more per item."""
    b = []
    CX, MSX = 300, 556

    def ms(y, v, note=None):
        out = text(MSX, y, v, tone="accent", family=MONO, weight="600", size=12)
        if note:
            out += text(MSX, y + 15, note, tone="faint", size=9.5, family=MONO)
        return out

    b.append(box(140, 8, 320, 32, '"durable lightweight winter boots for toddlers"',
                 fill="raise", tone="ink", size=11, bold=False))
    b.append(arrow(CX, 40, CX, 62))
    b.append(box(150, 62, 300, 44, "Query understanding",
                 "spell / segment / attributes / intent", fill="soft", stroke="accent"))
    b.append(ms(90, "8 ms"))

    # fan-out: a distribution bar, so no arrow crosses a label
    b.append(f'<line x1="{CX}" y1="106" x2="{CX}" y2="138" stroke="{{ink}}" stroke-width="1.4"/>')
    b.append(text(CX + 10, 126, "structured query", size=10, family=MONO, tone="muted"))
    b.append(f'<line x1="130" y1="138" x2="470" y2="138" stroke="{{ink}}" stroke-width="1.4"/>')
    for x, name, sub in [(55, "Lexical", "BM25F"), (225, "Dense", "two-tower + HNSW"),
                         (395, "Behavioural", "q to item, from logs")]:
        cx = x + 75
        b.append(arrow(cx, 138, cx, 156))
        b.append(box(x, 156, 150, 46, name, sub,
                     fill="soft" if name == "Dense" else "surface",
                     stroke="accent" if name == "Dense" else "rule"))
        b.append(f'<line x1="{cx}" y1="202" x2="{cx}" y2="220" stroke="{{ink}}" stroke-width="1.4"/>')
    b.append(ms(180, "25 ms", "in parallel"))
    b.append(f'<line x1="130" y1="220" x2="470" y2="220" stroke="{{ink}}" stroke-width="1.4"/>')
    b.append(arrow(CX, 220, CX, 242))
    b.append(box(200, 242, 200, 32, "Fusion (RRF)", fill="surface"))

    b.append(f'<line x1="18" y1="292" x2="682" y2="292" stroke="{{flag}}" '
             f'stroke-width="1.1" stroke-dasharray="5 3"/>')
    b.append(text(18, 287, "RECALL IS SET ABOVE THIS LINE AND NEVER RECOVERED",
                  size=9.5, tone="flag", family=MONO, weight="600"))
    b.append(text(682, 287, "everything below only REORDERS", size=9.5, tone="flag",
                  family=MONO, weight="600", anchor="end"))

    b.append(arrow(CX, 274, CX, 310))
    b.append(text(CX + 10, 306, "~1000 candidates", size=10, family=MONO, tone="muted"))
    b.append(box(150, 310, 300, 38, "L1 ranker", "cheap; trained to imitate L2"))
    b.append(ms(333, "8 ms"))
    b.append(arrow(CX, 348, CX, 382))
    b.append(text(CX + 10, 378, "top ~150", size=10, family=MONO, tone="muted"))
    b.append(box(150, 382, 300, 44, "L2 ranker", "pCTR | pCVR | pRel, calibrated",
                 fill="soft", stroke="accent"))
    b.append(ms(409, "22 ms"))
    b.append(arrow(CX, 426, CX, 460))
    b.append(text(CX + 10, 456, "top ~50", size=10, family=MONO, tone="muted"))
    b.append(box(150, 460, 300, 38, "Slate policy", "dedup / diversity / blend"))
    b.append(ms(483, "5 ms"))
    b.append(arrow(CX, 498, CX, 524))
    b.append(text(CX, 540, "24 products on the phone", size=12, tone="ink",
                  anchor="middle", weight="600"))
    return frame(700, 556, "".join(b),
                 "Search funnel: query understanding, three parallel retrieval arms, "
                 "fusion, two ranking stages and a slate policy, with the latency "
                 "budget each stage receives")


def fig_latency():
    """Claim: the cross-encoder does not fit in the blocking path, so take it
    off the blocking path rather than pretending it is rare."""
    b = []
    X0, SCALE = 132, 4.0
    STAGES = [("QU", 8, "soft"), ("retrieval", 25, "accent"), ("fusion", 2, "rule"),
              ("L1", 8, "soft"), ("L2", 22, "accent"), ("policy", 5, "rule"),
              ("overhead", 18, "raise")]

    lx = X0
    for nm, msv, f in STAGES:
        w = max(msv * SCALE, 26)
        b.append(f'<rect x="{lx}" y="8" width="14" height="10" rx="2" '
                 f'fill="{{{f}}}" stroke="{{rule}}"/>')
        b.append(text(lx + 18, 17, f"{nm} {msv}", size=9.5, tone="muted", family=MONO))
        lx += 22 + len(nm) * 5.6 + 22
    b.append(text(8, 17, "STAGES", size=9.5, tone="ink", family=MONO, weight="600"))

    def bar(y, stages, label, note, tone="muted"):
        out = [text(8, y + 16, label, size=12, tone="ink", weight="600")]
        x = X0
        for nm, msv, f in stages:
            w = msv * SCALE
            out.append(f'<rect x="{x}" y="{y}" width="{w}" height="24" '
                       f'fill="{{{f}}}" stroke="{{surface}}" stroke-width="1"/>')
            if w > 36:
                out.append(f'<text x="{x + w/2}" y="{y + 16}" text-anchor="middle" '
                           f'font-family="{MONO}" font-size="9.5" fill="{{ink}}">{msv}</text>')
            x += w
        total = sum(m for _, m, _ in stages)
        out.append(text(x + 8, y + 16, f"{total} ms", size=11.5, tone=tone,
                        family=MONO, weight="600"))
        out.append(text(8, y + 33, note, size=9.5, tone="faint"))
        return "".join(out)

    XE = ("x-encoder", 15, "flagsoft")
    b.append(bar(38, STAGES, "A   all-CPU", "12 ms of headroom. Ship this first."))
    b.append(bar(92, STAGES + [XE], "B   + x-encoder",
                 "over budget, and p99 is now hostage to GPU batch queueing",
                 tone="flag"))
    b.append(bar(146, STAGES, "C   progressive",
                 "same blocking path; the x-encoder runs OFF it", tone="accent"))
    ax = X0 + 88 * SCALE + 62
    b.append(f'<rect x="{ax}" y="146" width="{15 * SCALE}" height="24" rx="3" '
             f'fill="{{flagsoft}}" stroke="{{accent}}" stroke-dasharray="4 3"/>')
    b.append(text(ax + 30, 163, "15 async", size=9.5, tone="accent",
                  family=MONO, anchor="middle"))

    x100 = X0 + 100 * SCALE
    b.append(f'<line x1="{x100}" y1="30" x2="{x100}" y2="182" stroke="{{flag}}" '
             f'stroke-width="1.4" stroke-dasharray="5 3"/>')
    b.append(text(x100 + 6, 28, "100 ms p99", size=10, tone="flag",
                  family=MONO, weight="600"))

    return frame(700, 190, "".join(b),
                 "Three latency budgets against a 100 ms p99 line: all-CPU fits, "
                 "adding a cross-encoder to the blocking path does not, and "
                 "progressive ranking moves it off the blocking path")


def fig_two_tower():
    """Claim: the towers are different sizes because one is offline."""
    b = []
    b.append(f'<rect x="8" y="8" width="330" height="212" rx="8" fill="{{raise}}" '
             f'stroke="{{rule}}" stroke-dasharray="5 4"/>')
    b.append(f'<rect x="354" y="8" width="338" height="212" rx="8" fill="{{soft}}" '
             f'stroke="{{accent}}" stroke-dasharray="5 4"/>')
    b.append(text(20, 28, "ONLINE  -  every request, ~5 ms", size=10.5, tone="muted",
                  family=MONO, weight="600"))
    b.append(text(366, 28, "OFFLINE  -  once a night, 10M items, no budget",
                  size=10.5, tone="accent", family=MONO, weight="600"))

    b.append(box(48, 44, 250, 40, '"winter boots for toddlers"', fill="surface",
                 size=11, bold=False))
    b.append(arrow(173, 84, 173, 104))
    b.append(box(78, 104, 190, 54, "Query tower", "4-6 layers, distilled"))
    b.append(text(173, 176, "f(q)", size=13, tone="ink", anchor="middle",
                  family=MONO, weight="600"))
    b.append(arrow(173, 158, 173, 166))

    b.append(box(390, 44, 266, 40, "title + brand + attrs + bullets",
                 fill="surface", size=11, bold=False))
    b.append(arrow(523, 84, 523, 104))
    b.append(box(404, 104, 238, 84, "Item tower", "12+ layers, image features",
                 fill="surface", stroke="accent"))
    b.append(text(523, 210, "g(d)", size=13, tone="accent", anchor="middle",
                  family=MONO, weight="600"))
    b.append(arrow(523, 188, 523, 198, tone="accent", marker="arrowA"))

    b.append(arrow(173, 186, 300, 244))
    b.append(arrow(523, 224, 400, 244, tone="accent", marker="arrowA"))
    b.append(box(230, 244, 240, 36, "s(q,d) = <f(q), g(d)>", fill="surface",
                 stroke="accent", size=12))
    return frame(700, 292, "".join(b),
                 "Two-tower retriever: a small online query tower and a much larger "
                 "offline item tower meeting at a dot product")


def fig_negatives():
    """Claim: in-batch negatives can never reach the zero-engagement tail."""
    b = []
    b.append(text(8, 18, "THE 10M-ITEM CORPUS", size=10.5, tone="muted",
                  family=MONO, weight="600"))
    b.append(f'<rect x="8" y="28" width="684" height="92" rx="6" fill="{{raise}}" '
             f'stroke="{{rule}}"/>')
    b.append(f'<rect x="14" y="34" width="150" height="80" rx="4" fill="{{soft}}" '
             f'stroke="{{accent}}"/>')
    b.append(text(89, 62, "items with", size=11, tone="accent", anchor="middle",
                  weight="600"))
    b.append(text(89, 77, "engagement", size=11, tone="accent", anchor="middle",
                  weight="600"))
    b.append(text(89, 96, "~ a few %", size=10, tone="muted", anchor="middle",
                  family=MONO))
    b.append(text(428, 66, "zero-engagement tail", size=12, tone="flag",
                  anchor="middle", weight="600"))
    b.append(text(428, 84, "never a positive, so NEVER an in-batch negative",
                  size=10.5, tone="flag", anchor="middle", family=MONO))
    b.append(text(428, 101, "nothing ever pushes these down", size=10,
                  anchor="middle", tone="muted"))

    y = 168
    for x, w, title, src, tone, mk in [
            (8, 216, "In-batch", "the other items in the batch", "accent", "arrowA"),
            (242, 216, "Mined hard", "BM25 / current ANN top-k", "ink", "arrow"),
            (476, 216, "Uniform", "drawn from the FULL index", "flag", "arrowF")]:
        b.append(box(x, y, w, 52, title, src, fill="surface",
                     stroke=tone, tone=tone))
    b.append(arrow(116, y, 89, 120, tone="accent", marker="arrowA"))
    b.append(arrow(350, y, 300, 120))
    b.append(arrow(584, y, 500, 120, tone="flag", marker="arrowF"))
    return frame(700, 232, "".join(b),
                 "Three negative sources over the corpus; only uniform sampling "
                 "reaches the zero-engagement tail")


def fig_esmm():
    """Claim: the CVR tower is supervised only through the product."""
    b = []
    b.append(box(190, 8, 320, 34, "IMPRESSION SPACE", "every impression is labelled",
                 fill="raise", size=11))
    b.append(arrow(280, 42, 200, 74))
    b.append(arrow(420, 42, 500, 74))
    b.append(box(90, 74, 220, 52, "pCTR tower", "loss: BCE on clicks",
                 fill="soft", stroke="accent"))
    b.append(box(390, 74, 220, 52, "pCTCVR tower", "loss: BCE on click AND convert",
                 fill="soft", stroke="accent"))
    b.append(arrow(200, 126, 320, 176, tone="accent", marker="arrowA"))
    b.append(arrow(500, 126, 380, 176, tone="accent", marker="arrowA"))
    b.append(text(350, 160, "pCTCVR = pCTR x pCVR", size=11, tone="muted",
                  anchor="middle", family=MONO))
    b.append(box(230, 176, 240, 52, "pCVR tower", "NO loss of its own",
                 fill="flagsoft", stroke="flag", tone="flag"))
    return frame(700, 240, "".join(b),
                 "ESMM: CTR and CTCVR towers supervised on all impressions, with the "
                 "CVR tower learned implicitly through their product")


def fig_seam():
    """Claim: each stage is trained on the other's output, so changing one
    invalidates the other."""
    b = []
    b.append(box(250, 10, 200, 44, "Retriever", "trained on clicks",
                 fill="soft", stroke="accent"))
    b.append(arrow(350, 54, 350, 96, "produces the candidate distribution",
                   lx=360, ly=79, anchor="start"))
    b.append(box(250, 96, 200, 44, "Ranker", "trained on THOSE candidates only",
                 fill="soft", stroke="accent"))
    b.append(arrow(350, 140, 350, 182, "decides what is shown", lx=360, ly=165,
                   anchor="start"))
    b.append(box(250, 182, 200, 44, "Click logs", "the only feedback you get"))
    b.append(f'<path d="M250,204 L120,204 L120,32 L250,32" fill="none" '
             f'stroke="{{flag}}" stroke-width="1.4" stroke-dasharray="5 3" '
             f'marker-end="url(#arrowF)"/>')
    b.append(f'<path d="M450,204 L580,204 L580,118 L450,118" fill="none" '
             f'stroke="{{flag}}" stroke-width="1.4" stroke-dasharray="5 3" '
             f'marker-end="url(#arrowF)"/>')
    b.append(text(112, 118, "trains", size=10, tone="flag", anchor="end", family=MONO))
    b.append(text(588, 160, "trains", size=10, tone="flag", family=MONO))
    return frame(700, 240, "".join(b),
                 "A closed loop: retriever feeds ranker feeds logs feeds both models")


def fig_surfaces():
    """Claim: four visible slots versus twenty changes the position-bias curve."""
    b = []
    b.append(text(8, 18, "MOBILE APP", size=11, tone="accent", family=MONO, weight="600"))
    b.append(f'<rect x="8" y="26" width="150" height="196" rx="10" fill="{{surface}}" '
             f'stroke="{{rule}}" stroke-width="1.4"/>')
    b.append(f'<rect x="18" y="36" width="130" height="16" rx="8" fill="{{raise}}"/>')
    n = 1
    for row in range(3):
        for col in range(2):
            x, y = 18 + col * 68, 60 + row * 52
            fill = "soft" if row < 2 else "raise"
            b.append(f'<rect x="{x}" y="{y}" width="60" height="44" rx="4" '
                     f'fill="{{{fill}}}" stroke="{{rule}}"/>')
            b.append(text(x + 30, y + 27, str(n), size=12, anchor="middle",
                          tone="accent" if row < 2 else "faint", weight="600"))
            n += 1
    b.append(f'<line x1="8" y1="162" x2="158" y2="162" stroke="{{flag}}" '
             f'stroke-width="1.2" stroke-dasharray="4 3"/>')
    b.append(text(162, 166, "fold", size=10, tone="flag", family=MONO))
    b.append(text(8, 242, "4 slots above the fold", size=11, tone="ink", weight="600"))
    b.append(text(8, 258, "steep bias curve, precision > recall", size=10.5, tone="muted"))

    b.append(text(250, 18, "WEB", size=11, tone="accent", family=MONO, weight="600"))
    b.append(f'<rect x="250" y="26" width="442" height="196" rx="8" fill="{{surface}}" '
             f'stroke="{{rule}}" stroke-width="1.4"/>')
    b.append(f'<rect x="262" y="36" width="300" height="16" rx="8" fill="{{raise}}"/>')
    n = 1
    for row in range(3):
        for col in range(5):
            x, y = 262 + col * 84, 60 + row * 52
            b.append(f'<rect x="{x}" y="{y}" width="76" height="44" rx="4" '
                     f'fill="{{soft}}" stroke="{{rule}}"/>')
            b.append(text(x + 38, y + 27, str(n), size=12, anchor="middle",
                          tone="accent", weight="600"))
            n += 1
    b.append(text(250, 242, "15-20 slots above the fold", size=11, tone="ink",
                  weight="600"))
    b.append(text(250, 258, "flatter curve, real comparison, recall > precision",
                  size=10.5, tone="muted"))
    return frame(700, 272, "".join(b),
                 "Mobile shows four slots above the fold, web fifteen to twenty, so "
                 "the position-bias curves differ")


def fig_skew():
    """Claim: logging the served features removes skew by construction."""
    b = []
    b.append(text(8, 18, "RECOMPUTE  (fragile)", size=11, tone="flag",
                  family=MONO, weight="600"))
    b.append(box(8, 28, 150, 40, "serving", "computes features", fill="surface"))
    b.append(box(8, 92, 150, 40, "training", "recomputes them", fill="surface"))
    b.append(arrow(83, 68, 83, 88, tone="flag", marker="arrowF"))
    b.append(text(168, 84, "different code, different", size=10.5, tone="flag")) 
    b.append(text(168, 99, "cutoff -> skew you find", size=10.5, tone="flag"))
    b.append(text(168, 114, "three months later", size=10.5, tone="flag"))

    b.append(text(372, 18, "LOG-AND-TRAIN  (robust)", size=11, tone="accent",
                  family=MONO, weight="600"))
    b.append(box(372, 28, 150, 40, "serving", "computes features",
                 fill="soft", stroke="accent"))
    b.append(arrow(522, 48, 556, 48, tone="accent", marker="arrowA"))
    b.append(box(556, 28, 136, 40, "log the vector", fill="surface", stroke="accent"))
    b.append(arrow(624, 68, 624, 88, tone="accent", marker="arrowA"))
    b.append(box(480, 92, 212, 40, "training reads exactly", "what serving computed",
                 fill="soft", stroke="accent"))
    return frame(700, 146, "".join(b),
                 "Recomputing features at training time creates skew; logging the "
                 "served feature vector removes it by construction")


def fig_losses():
    """Claim: the loss families differ in what a single gradient step can see."""
    b = []
    W, PW, GAP = 700, 165, 8
    xs = [8 + i * (PW + GAP) for i in range(4)]

    def panel(x, title, sub, tone):
        b.append(f'<rect x="{x}" y="8" width="{PW}" height="254" rx="7" '
                 f'fill="{{surface}}" stroke="{{{tone}}}" stroke-width="1.2"/>')
        b.append(f'<rect x="{x}" y="8" width="{PW}" height="30" rx="7" '
                 f'fill="{{{tone}}}"/>')
        b.append(f'<rect x="{x}" y="28" width="{PW}" height="10" fill="{{{tone}}}"/>')
        b.append(f'<text x="{x + PW/2}" y="28" text-anchor="middle" '
                 f'font-family="{FONT}" font-size="11.5" font-weight="700" '
                 f'fill="#FFFFFF">{e(title)}</text>')
        b.append(text(x + PW / 2, 54, sub, size=9.5, tone="muted", anchor="middle",
                      family=MONO))

    def node(cx, cy, label, tone="ink", r=15):
        b.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{{soft}}" '
                 f'stroke="{{{tone}}}" stroke-width="1.2"/>')
        b.append(text(cx, cy + 4, label, size=10, tone=tone, anchor="middle",
                      family=MONO, weight="600"))

    def foot(x, lines, tone="muted"):
        for k, ln in enumerate(lines):
            b.append(text(x + 9, 214 + k * 15, ln, size=9.8, tone=tone))

    # 1 -- regression
    x = xs[0]; cx = x + PW / 2
    panel(x, "REGRESSION", "MSE / Huber", "muted")
    node(cx, 88, "q"); node(cx, 150, "d")
    b.append(arrow(cx, 103, cx, 133, tone="muted"))
    b.append(text(cx, 182, "target = 1.0", size=10, tone="flag", anchor="middle",
                  family=MONO, weight="600"))
    foot(x, ["one pair, and an", "ABSOLUTE target that", "does not exist for clicks"])

    # 2 -- pairwise / triplet
    x = xs[1]; cx = x + PW / 2
    panel(x, "PAIRWISE", "triplet, BPR, RankNet", "muted")
    node(cx, 88, "q")
    node(cx - 38, 150, "d+", tone="accent"); node(cx + 38, 150, "d-", tone="flag")
    b.append(arrow(cx - 8, 103, cx - 33, 133, tone="accent", marker="arrowA"))
    b.append(arrow(cx + 8, 103, cx + 33, 133, tone="flag", marker="arrowF"))
    b.append(f'<line x1="{cx-23}" y1="176" x2="{cx+23}" y2="176" '
             f'stroke="{{ink}}" stroke-width="1"/>')
    b.append(text(cx, 192, "fixed margin m", size=10, tone="ink", anchor="middle",
                  family=MONO, weight="600"))
    foot(x, ["ONE negative at a time,", "and an absolute margin", "in a space you scale"])

    # 3 -- sampled softmax
    x = xs[2]; cx = x + PW / 2
    panel(x, "SAMPLED SOFTMAX", "InfoNCE / MNRL", "accent")
    node(cx, 88, "q")
    node(cx - 54, 150, "d+", tone="accent")
    for dx in (-16, 20, 56):
        node(cx + dx, 150, "d-", tone="muted", r=13)
    b.append(arrow(cx - 10, 103, cx - 48, 134, tone="accent", marker="arrowA"))
    for dx in (-16, 20, 56):
        b.append(arrow(cx + dx * 0.32, 103, cx + dx * 0.92, 134, tone="muted"))
    b.append(f'<rect x="{x+12}" y="170" width="{PW-24}" height="20" rx="4" '
             f'fill="{{soft}}" stroke="{{accent}}"/>')
    b.append(text(cx, 184, "normalised over ALL", size=9.5, tone="accent",
                  anchor="middle", family=MONO, weight="600"))
    foot(x, ["N negatives at once, and", "NO absolute target --", "only relative order"], tone="ink")

    # 4 -- listwise, metric-weighted
    x = xs[3]; cx = x + PW / 2
    panel(x, "LISTWISE", "LambdaRank / ListNet", "muted")
    for k in range(5):
        wgt = [1.0, 0.63, 0.5, 0.43, 0.39][k]
        bw = 78 * wgt
        b.append(f'<rect x="{x+22}" y="{74 + k*21}" width="{bw}" height="14" rx="2" '
                 f'fill="{{soft}}" stroke="{{accent}}" stroke-width="0.9"/>')
        b.append(text(x + 14, 85 + k * 21, str(k + 1), size=9, tone="muted",
                      family=MONO, anchor="middle"))
        b.append(text(x + 106, 85 + k * 21, "%.2f" % wgt, size=8.5, tone="accent",
                      family=MONO))
    b.append(text(cx, 192, "position discount", size=10, tone="ink", anchor="middle",
                  family=MONO, weight="600"))
    foot(x, ["the WHOLE list, weighted", "by what a swap does to", "the metric you report"])

    return frame(W, 272, "".join(b),
                 "What one gradient step sees under four loss families: a single "
                 "absolute target, one positive against one negative with a margin, "
                 "one positive normalised against many negatives, and a whole "
                 "position-discounted list")


FIGURES = {
    "funnel": fig_funnel, "latency": fig_latency, "two-tower": fig_two_tower,
    "negatives": fig_negatives, "esmm": fig_esmm, "seam": fig_seam,
    "surfaces": fig_surfaces, "skew": fig_skew, "losses": fig_losses,
}


def render(name, palette):
    svg = FIGURES[name]()
    # Several figures are inlined on one page, so marker ids must not collide.
    for mid in ("arrow", "arrowA", "arrowF"):
        svg = svg.replace('id="%s"' % mid, 'id="%s-%s"' % (mid, name))
        svg = svg.replace('url(#%s)' % mid, 'url(#%s-%s)' % (mid, name))
    for k, v in palette.items():
        svg = svg.replace("{%s}" % k, v)
    left = re.findall(r"\{([a-z]+)\}", svg)
    if left:
        sys.exit("figures: unresolved palette token(s) %s in %s" % (set(left), name))
    return svg


def main():
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "coupang", "figures")
    os.makedirs(out, exist_ok=True)
    import cairosvg
    for name in FIGURES:
        open(os.path.join(out, name + ".svg"), "w").write(render(name, PRINT))
        cairosvg.svg2pdf(bytestring=render(name, PRINT).encode(),
                         write_to=os.path.join(out, name + ".pdf"))
        cairosvg.svg2png(bytestring=render(name, PRINT).encode(),
                         write_to=os.path.join(out, name + ".png"), scale=1.6)
    print("rendered %d figures to %s" % (len(FIGURES), out))


if __name__ == "__main__":
    main()
