#!/usr/bin/env python3
"""Extract the high-frequency breadth/depth questions into drill/questions.json.

The four volumes are a reference library: derivation-first, ~530 words per answer,
math in 82% of them. That is the wrong shape for rehearsal. This pulls the
questions worth drilling into one structured file, which is then re-cut into
spoken-answer form and rendered to both a PDF and a web page.

Run once to seed drill/questions.json; after that the JSON is the source of truth
and this script is only for re-seeding.
"""
import json, os, re, sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QB = re.compile(r"\\begin\{interviewq\}(.*?)\\end\{interviewq\}", re.S)
KEEP_TYPES = {"Conceptual", "First Principles", "Trade-off", "Debugging",
              "Estimation", "Mathematical"}
VOL = {"deep-learning": ("DL", "Deep Learning"), "nlp": ("NLP", "NLP"),
       "search-recommendation": ("SR", "Search & Rec"),
       "conventional-ml": ("CML", "Classical ML")}
LEVEL = {"Lfive": "L5", "Lsix": "L6", "Lseven": "L7"}

# Per-volume caps, chosen so the deck reads as breadth across the whole field
# rather than 3/4 deep learning. Highest-frequency first within each volume.
CAPS = {"DL": 70, "NLP": 35, "SR": 20, "CML": 25}


def detex(t):
    """LaTeX -> markdown-ish text. Math is preserved as $...$ for later triage."""
    # Strip LaTeX comments but NOT escaped percent signs. Using r"%.*?$" here
    # silently truncated 20 of 150 answers at their first \% ("95\% accuracy"),
    # one of them losing two thirds of its text.
    t = re.sub(r"(?<!\\)%.*?$", "", t, flags=re.M)
    t = re.sub(r"\\(?:textbf|textit|emph)\{([^{}]*)\}", r"**\1**", t)
    t = re.sub(r"\\texttt\{([^{}]*)\}", r"`\1`", t)
    t = re.sub(r"\\ref\{[^}]*\}", "", t)
    t = re.sub(r"\\(?:label|index)\{[^}]*\}", "", t)
    t = re.sub(r"\\begin\{itemize\}|\\end\{itemize\}|\\begin\{enumerate\}|\\end\{enumerate\}", "", t)
    t = re.sub(r"\\item\s*", "\n- ", t)
    t = t.replace("---", "\u2014").replace("--", "\u2013")
    t = t.replace("``", '"').replace("''", '"').replace("\\&", "&").replace("\\%", "%")
    t = t.replace("\\$", "$").replace("\\_", "_").replace("\\#", "#")
    t = re.sub(r"\\\\", " ", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def chapters(volume):
    main = open(os.path.join(ROOT, "volumes", volume, "main.tex")).read()
    out = []
    for inc in re.findall(r"\\include\{sections/([^}]+)\}", main):
        if "appendix" in inc:
            continue
        out.append(os.path.join(ROOT, "volumes", volume, "sections", inc + ".tex"))
    return out


def main():
    found = defaultdict(list)
    for volume, (tag, vname) in VOL.items():
        for n, path in enumerate(chapters(volume), start=1):
            src = open(path).read()
            ct = re.search(r"\\chapter\{(.+?)\}", src)
            title = detex(ct.group(1)) if ct else os.path.basename(path)
            for block in QB.findall(src):
                q = re.search(r"\\textbf\{Q:\s*(.*?)\}\s*\n", block, re.S)
                lv = re.search(r"\\badge(Lfive|Lsix|Lseven)\b", block)
                ty = re.search(r"\\badgetype\{([^}]*)\}", block)
                fr = re.search(r"\\badgefreq\{\\(freq\w+)\}", block)
                if not (q and lv and ty and fr):
                    continue
                if ty.group(1) not in KEEP_TYPES or fr.group(1) != "freqhigh":
                    continue
                ans = block.split("\\stronganswer")[-1]
                red = ""
                if "\\redflags" in ans:
                    ans, rest = ans.split("\\redflags", 1)
                    red = rest.split("\\interviewtest")[0].split("\\goodgreat")[0].split("\\followups")[0]
                fu = ""
                m = re.search(r"\\followups\{(.*)\}\s*$", block, re.S)
                if m:
                    fu = m.group(1)
                found[tag].append({
                    "volume": tag, "volume_name": vname,
                    "chapter": n, "chapter_title": title,
                    "level": LEVEL[lv.group(1)], "type": ty.group(1),
                    "question": detex(q.group(1)),
                    "answer_source": detex(ans),
                    "red_flags_source": detex(red),
                    "followups_source": detex(fu),
                    "status": "needs_recut",
                })

    deck = []
    for tag, cap in CAPS.items():
        items = found[tag]
        # keep chapter order (roughly curriculum order), trim from the tail
        deck.extend(items[:cap])
        if len(items) > cap:
            print(f"  {tag}: {len(items)} available, kept {cap}", file=sys.stderr)
    for i, item in enumerate(deck, 1):
        item["id"] = i

    out = os.path.join(ROOT, "drill", "questions.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"questions": deck}, open(out, "w"), indent=2, ensure_ascii=False)
    words = [len(q["answer_source"].split()) for q in deck]
    words.sort()
    print(f"wrote {len(deck)} questions to drill/questions.json")
    print(f"  by volume: " + ", ".join(f"{t}={sum(1 for q in deck if q['volume']==t)}" for t in CAPS))
    print(f"  source answers: median {words[len(words)//2]} words, "
          f"{sum(1 for w in words if w > 260)} over 260 words")


if __name__ == "__main__":
    main()
