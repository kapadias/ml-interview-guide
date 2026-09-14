#!/usr/bin/env python3
"""Pair the tested reference solutions with their problem metadata.

The code lives in coupang/coding_a.py and coding_b.py so it can be imported and
exercised by coupang/test_coding.py -- shipping an interview-prep document with
untested code would be worse than shipping no code.
"""
import json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = os.path.join(ROOT, "coupang")
MARK = re.compile(r"^# --- (\d+)\. (.+?) -+$", re.M)


def blocks(path):
    src = open(path).read()
    marks = list(MARK.finditer(src))
    out = {}
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(src)
        out[int(m.group(1))] = src[m.end():end].strip("\n")
    return out


def main():
    code = {}
    for f in ["coding_a.py", "coding_b.py"]:
        code.update(blocks(os.path.join(C, f)))
    meta = json.load(open(os.path.join(C, "coding_meta.json")))["meta"]

    problems = []
    for m in meta:
        if m["n"] not in code:
            sys.exit("no code block for problem %d (%s)" % (m["n"], m["title"]))
        p = dict(m)
        p["code"] = code[m["n"]]
        problems.append(p)

    missing = set(code) - {m["n"] for m in meta}
    if missing:
        sys.exit("code blocks with no metadata: %s" % sorted(missing))

    json.dump({"problems": problems}, open(os.path.join(C, "coding.json"), "w"),
              indent=1, ensure_ascii=False)
    print("wrote coding.json: %d problems, %d lines of tested code"
          % (len(problems), sum(p["code"].count("\n") + 1 for p in problems)))


if __name__ == "__main__":
    main()
