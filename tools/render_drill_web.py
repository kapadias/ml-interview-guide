#!/usr/bin/env python3
"""Render drill/questions.json to a self-contained drill page.

Same source as the PDF; different job. The PDF is what you read on a plane,
the page is what you rehearse against: it hides the answer until you have
tried to say it, times you against the length the answer was written to, and
remembers which ones you keep failing.

Only the re-cut fields ship. The `*_source` blobs (raw book text) stay in the
JSON and out of the page.
"""
import json, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEEP = ["id", "question", "say", "numbers", "red_flags", "followups",
        "whiteboard", "topic", "level", "type", "volume_name", "chapter_title"]
TOPIC_ORDER = ["Foundations", "Architectures", "Training", "Inference & Serving",
               "NLP & LLMs", "Alignment & Safety", "Evaluation", "Experimentation",
               "Retrieval & Ranking", "Recommenders", "Classical ML", "Applied Craft"]
ROUND_OF_TYPE = {
    "Conceptual": "Breadth", "First Principles": "Depth", "Mathematical": "Depth",
    "Trade-off": "Breadth", "Estimation": "Depth", "Debugging": "Depth",
    "Coding": "Coding", "System Design": "Design", "Architecture Design": "Design",
    "Behavioral": "Behavioral",
}


def slim(qs):
    out = []
    for q in qs:
        d = {k: q[k] for k in KEEP if q.get(k)}
        d["round"] = ROUND_OF_TYPE.get(q["type"], "Breadth")
        d["words"] = len(q["say"].split())
        out.append(d)
    return out


def main():
    qs = json.load(open(os.path.join(ROOT, "drill", "questions.json")))["questions"]
    data = json.dumps({"questions": slim(qs), "topics": TOPIC_ORDER},
                      ensure_ascii=False, separators=(",", ":"))
    tpl = open(os.path.join(ROOT, "tools", "drill_page.html")).read()
    html = tpl.replace("/*__DATA__*/null", data)
    out = os.path.join(ROOT, "drill", "web", "index.html")
    open(out, "w").write(html)
    print("wrote %s (%d questions, %.0f KB)" % (out, len(qs), len(html) / 1024))


if __name__ == "__main__":
    main()
