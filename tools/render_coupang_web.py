#!/usr/bin/env python3
"""Render the Coupang loop document to a self-contained page.

Same source as the PDF. The two prose parts are converted to HTML here rather
than shipped as markdown, so the page has no runtime markdown dependency.
"""
import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from md2html import convert                                  # noqa: E402
from render_coupang_pdf import CODING_INTRO                  # noqa: E402

C = os.path.join(ROOT, "coupang")
KEEP = ["id", "question", "say", "numbers", "red_flags", "followups", "whiteboard",
        "coupang", "section", "level", "type", "source"]


def slim(qs):
    return [{k: q[k] for k in KEEP if q.get(k)} for q in qs]


def main():
    deck = json.load(open(os.path.join(C, "deck.json")))
    data = {
        "brief": convert(open(os.path.join(C, "brief.md")).read(), "brief.md"),
        "project": convert(open(os.path.join(C, "project.md")).read(), "project.md"),
        "codingIntro": convert(CODING_INTRO, "coding intro"),
        "depth": slim(deck["depth"]),
        "breadth": slim(deck["breadth"]),
        "coding": deck["coding"],
    }
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    if "</script" in blob:
        sys.exit("content contains </script and would truncate the data block")
    tpl = open(os.path.join(ROOT, "tools", "coupang_page.html")).read()
    html = tpl.replace("/*__DATA__*/null", blob)
    out = os.path.join(C, "web", "index.html")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write(html)
    print("wrote %s (%d depth, %d breadth, %d coding, %.0f KB)"
          % (out, len(deck["depth"]), len(deck["breadth"]), len(deck["coding"]),
             len(html) / 1024))


if __name__ == "__main__":
    main()
