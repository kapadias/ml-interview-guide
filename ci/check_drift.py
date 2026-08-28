#!/usr/bin/env python3
"""Drift-sentinel check: grep canonical-fact regexes across the program volumes.
Usage: python3 ci/check_drift.py [--root PARENT_DIR]
Volumes are sibling directories of the spine repo by default."""
import re, sys, glob, os, argparse

VOLUMES = {
    "dl": "deep-learning",
    "nlp": "nlp",
    "sr": "search-recommendation",
    "cml": "conventional-ml",
}

def load_facts(path):
    # minimal YAML-subset parser (no external deps): expects the facts.yaml layout above
    facts, cur = [], None
    for line in open(path):
        s = line.rstrip("\n")
        if re.match(r"\s*- id:", s):
            cur = {"id": s.split("id:",1)[1].strip(), "forbid": [], "scope": []}
            facts.append(cur)
        elif cur is not None and re.match(r"\s*fact:", s):
            cur["fact"] = s.split("fact:",1)[1].strip().strip('"')
        elif cur is not None and re.match(r"\s*- '", s):
            cur["forbid"].append(s.strip()[3:-1])
        elif cur is not None and re.match(r"\s*scope:", s):
            cur["scope"] = re.findall(r"\w+", s.split(":",1)[1])
    return facts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "volumes"))
    args = ap.parse_args()
    root = args.root
    facts = load_facts(os.path.join(os.path.dirname(os.path.abspath(__file__)), "facts.yaml"))
    failures = 0
    for f in facts:
        for vol in f["scope"]:
            vdir = os.path.join(root, VOLUMES.get(vol, vol))
            if not os.path.isdir(vdir):
                continue
            for tex in glob.glob(os.path.join(vdir, "**", "*.tex"), recursive=True):
                text = open(tex, errors="ignore").read()
                for pat in f["forbid"]:
                    for m in re.finditer(pat, text):
                        line = text[:m.start()].count("\n") + 1
                        print(f"DRIFT [{f['id']}] {tex}:{line}: matches forbidden /{pat}/")
                        print(f"    canonical: {f['fact']}")
                        failures += 1
    # cross-volume label collision check
    labels = {}
    for vol, d in VOLUMES.items():
        vdir = os.path.join(root, d)
        if not os.path.isdir(vdir): continue
        for tex in glob.glob(os.path.join(vdir, "**", "*.tex"), recursive=True):
            for i, line in enumerate(open(tex, errors="ignore"), 1):
                for lab in re.findall(r"\\label\{([^}]+)\}", line):
                    labels.setdefault((vol, lab), []).append(f"{tex}:{i}")
    for (vol, lab), locs in labels.items():
        if len(locs) > 1:
            print(f"LABEL COLLISION [{vol}] {lab}: " + "; ".join(locs)); failures += 1
    print(f"\n{failures} problem(s)." if failures else "\nAll drift sentinels and label checks passed.")
    sys.exit(1 if failures else 0)

if __name__ == "__main__":
    main()
