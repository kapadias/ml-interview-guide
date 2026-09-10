#!/usr/bin/env python3
"""Normalize drill/questions.json for rendering.

Two classes of defect the re-cut agents leave behind, both of which render
silently wrong rather than failing:

1. Bare identifiers. `d_k` and `min_child_weight` come out of the source
   answers unmarked, so LaTeX sets them as "d\\_k" in body text. Math symbols
   become inline math; code identifiers become \\texttt.
2. Literal currency. "$5 vs $120" is a syntactically valid math span, so it
   renders as italic "5vs" followed by loose text and nothing warns. Currency
   is rewritten to \\$ , which the renderer protects before splitting on math.
"""
import json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Bare identifier -> inline math. Applied only outside existing math spans.
MATH_IDENT = {
    "d_k": r"$d_k$", "d_head": r"$d_{head}$", "d_ff": r"$d_{ff}$",
    "d_c": r"$d_c$", "d_r": r"$d_r$", "W_O": r"$W_O$",
    "c_FP": r"$c_{FP}$", "c_FN": r"$c_{FN}$",
    "p_i": "$p_i$", "p_j": "$p_j$", "y_j": "$y_j$", "delta_ij": r"$\delta_{ij}$",
}

# Bare identifier -> \texttt, via backticks the renderer already understands.
CODE_IDENT = [
    "min_child_weight", "min_samples_leaf", "max_features", "n_init",
    "input_ids", "train_on_inputs", "merchant_id", "tool_use", "end_turn",
    "num_leaves", "colsample_bytree", "n_estimators", "learning_rate",
    "class_weight", "random_state", "scale_pos_weight", "max_position_embeddings",
]

FIELDS = ["question", "say", "whiteboard", "numbers", "red_flags", "followups"]


def fix_text(t, stats):
    parts = re.split(r"(\$\$.*?\$\$|\$[^$]*\$|`[^`]*`)", t, flags=re.S)
    for i, p in enumerate(parts):
        if i % 2:
            continue
        # Currency: a $ glued to a digit is money, not math. Do this first so
        # the identifier passes below cannot see it.
        p, n = re.subn(r"(?<!\\)\$(?=\d)", r"\\$", p)
        stats["currency"] += n
        for ident in CODE_IDENT:
            p, n = re.subn(r"(?<![\w`])" + re.escape(ident) + r"(?![\w`])", "`%s`" % ident, p)
            stats["code"] += n
        for ident, rep in MATH_IDENT.items():
            p, n = re.subn(r"(?<![\w$])" + re.escape(ident) + r"(?![\w$])", rep.replace("\\", "\\\\"), p)
            stats["math"] += n
        parts[i] = p
    return "".join(parts)


def copyedit(doc):
    """Hand rewrites that no rule can derive -- mostly derivations the re-cut
    left as unspeakable prose ("p_i times delta_ij minus p_j"). Kept here so a
    re-merge does not silently drop them; each must match exactly once."""
    path = os.path.join(ROOT, "drill", "copyedits.json")
    if not os.path.exists(path):
        return 0
    n = 0
    for needle, rep in json.load(open(path)):
        hits = 0
        for q in doc["questions"]:
            for f in FIELDS:
                v = q.get(f)
                if isinstance(v, str) and needle in v:
                    q[f] = v.replace(needle, rep); hits += 1
                elif isinstance(v, list):
                    for i, x in enumerate(v):
                        if needle in x:
                            v[i] = x.replace(needle, rep); hits += 1
        if hits != 1:
            sys.exit("copyedit matched %d times (want 1): %s" % (hits, needle[:70]))
        n += 1
    return n


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "drill", "questions.json")
    doc = json.load(open(path))
    stats = {"currency": 0, "code": 0, "math": 0}
    for q in doc["questions"]:
        for f in FIELDS:
            v = q.get(f)
            if isinstance(v, str):
                q[f] = fix_text(v, stats)
            elif isinstance(v, list):
                q[f] = [fix_text(x, stats) for x in v]
    edits = copyedit(doc)          # after the identifier pass: the needles quote its output
    json.dump(doc, open(path, "w"), indent=1, ensure_ascii=False)
    print("normalized %s: %d copy-edits, %d currency, %d code idents, %d math idents"
          % (os.path.basename(path), edits, stats["currency"], stats["code"], stats["math"]))


if __name__ == "__main__":
    main()
