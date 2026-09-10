#!/usr/bin/env python3
"""Merge the re-cut drill slices into drill/questions.json.

questions.base.json (the 150 answers re-cut from the four volumes) is the
input and questions.json the output, so the whole pipeline re-runs from a clean
checkout. Order matters: repairs replace damaged entries, drops remove them,
then the rebalancing additions come in. Ids are reassigned last, in reading
order, so the deck's numbering matches the PDF and the web page.

Two bugs this merge exists to correct, both mine:

  * `tools/extract_drill.py` stripped LaTeX comments with `%.*?$`, which also
    ate every escaped `\\%`. Thirty-six answers were truncated mid-sentence;
    twenty-two of them had already been re-cut from the damaged text and are
    re-cut again here from the corrected source (slice7).
  * The per-volume cap took questions in chapter order, so every deep-learning
    chapter past 17 was cut before selection saw it -- the deck had two
    Inference & Serving questions and two Experimentation ones. slice6 adds
    thirty-one from the dropped pool; drops.json removes the same number of
    over-weighted ones.
"""
import json, os, sys
from collections import Counter, OrderedDict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "drill")

TOPIC_ORDER = ["Foundations", "Architectures", "Training", "Inference & Serving",
               "NLP & LLMs", "Alignment & Safety", "Evaluation", "Experimentation",
               "Retrieval & Ranking", "Recommenders", "Classical ML", "Applied Craft"]

REQUIRED = ["question", "say", "red_flags", "followups", "topic", "level", "type"]


def load(p):
    d = json.load(open(os.path.join(D, p)))
    return d["questions"] if isinstance(d, dict) else d


def main():
    base = {q["id"]: q for q in load("questions.base.json")}
    n0 = len(base)

    repairs = load("slices/slice7_repairs.json")
    for q in repairs:
        if q["id"] not in base:
            sys.exit("repair for unknown id %s" % q["id"])
        base[q["id"]] = q

    drops = json.load(open(os.path.join(D, "drops.json")))
    for k in drops:
        if int(k) not in base:
            sys.exit("drop for unknown id %s" % k)
        del base[int(k)]

    additions = load("slices/slice6_additions.json")
    if any(a.get("status") != "recut" for a in additions):
        sys.exit("slice6 additions are not all re-cut yet")
    # Near-duplicates the exact-text check below cannot see, caught by review.
    adrops = json.load(open(os.path.join(D, "drops_additions.json")))
    for k in adrops:
        n = len(additions)
        additions = [a for a in additions if a["question"].strip() != k.strip()]
        if len(additions) == n:
            sys.exit("addition drop matched nothing: %s" % k[:70])
    seen_q = {q["question"].strip().lower() for q in base.values()}
    for a in additions:
        if a["question"].strip().lower() in seen_q:
            sys.exit("addition duplicates an existing question: %s" % a["question"][:70])
        a["id"] = max(base) + 1
        base[a["id"]] = a

    qs = list(base.values())
    for q in qs:
        missing = [f for f in REQUIRED if not q.get(f)]
        if missing:
            sys.exit("id %s missing %s" % (q["id"], missing))
        if q["topic"] not in TOPIC_ORDER:
            sys.exit("id %s has topic %r outside the controlled list" % (q["id"], q["topic"]))
        if not 2 <= len(q["red_flags"]) <= 4:
            sys.exit("id %s has %d red flags" % (q["id"], len(q["red_flags"])))
        if not 2 <= len(q["followups"]) <= 4:
            sys.exit("id %s has %d followups" % (q["id"], len(q["followups"])))

    qs.sort(key=lambda q: (TOPIC_ORDER.index(q["topic"]), q["volume"], q["chapter"], q["id"]))
    for i, q in enumerate(qs, 1):
        q["id"] = i

    out = os.path.join(D, "questions.json")
    json.dump({"questions": qs}, open(out, "w"), indent=1, ensure_ascii=False)

    words = sorted(len(q["say"].split()) for q in qs)
    print("%d -> %d questions (%d repaired, %d dropped, %d added)"
          % (n0, len(qs), len(repairs), len(drops) + len(adrops), len(additions)))
    print("say words: median %d, p10 %d, p90 %d, max %d"
          % (words[len(words)//2], words[len(words)//10], words[len(words)*9//10], words[-1]))
    print("whiteboards: %d" % sum(1 for q in qs if q.get("whiteboard")))
    for t in TOPIC_ORDER:
        print("  %-22s %d" % (t, sum(1 for q in qs if q["topic"] == t)))
    print("levels:", dict(Counter(q["level"] for q in qs)))


if __name__ == "__main__":
    main()
