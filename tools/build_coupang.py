#!/usr/bin/env python3
"""Assemble the Coupang Search & Discovery deck from two sources.

Questions already re-cut into spoken form in drill/questions.json are reused
verbatim and re-sectioned for this loop; everything the loop needs that the
general deck does not cover is written fresh in coupang/depth_new_*.json.

Round order is interview order: depth, breadth, then coding. The prose parts
(brief, project) are rendered alongside but live in markdown.
"""
import json, os, sys, glob
from collections import OrderedDict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = os.path.join(ROOT, "coupang")

# Reused drill-deck questions, keyed by a distinctive prefix of the question
# text rather than by id: ids are reassigned on every merge, so an id list
# silently re-points at different questions the next time the deck changes.
DEPTH_REUSE = OrderedDict([
    ("Lexical retrieval", ["Explain BM25. Why is it still used"]),
    ("Semantic retrieval", [
        "Bi-encoder vs. cross-encoder: why not cross-encode everything",
        "How does HNSW work",
        "Your two-tower model retrieves documents that share keywords",
        "Cosine similarity vs dot product vs L2"]),
    ("Hybrid retrieval", [
        "Lexical, dense, or hybrid",
        "You run BM25 and dense retrieval in parallel"]),
    ("Learning to rank", [
        "Pointwise vs",
        "Your click-trained ranker keeps favoring"]),
    ("Ranking and personalization", [
        "Two-Tower vs. interaction-based models",
        "How do you handle the cold start problem",
        "Why do recommendation models need calibration",
        "Your CTR model's offline AUC improved by 0.5%",
        "Estimate the memory required for DLRM"]),
    ("Evaluation", [
        "Derive NDCG from first principles",
        "Your new reranker improved offline NDCG by 4%",
        "Your new embedding model wins offline"]),
])
BREADTH_REUSE = OrderedDict([
    ("Foundations", [
        "Explain the bias-variance tradeoff",
        "Explain double descent",
        "What is KL divergence"]),
    ("Transformers", [
        "Explain self-attention step by step",
        "Why do we need multiple attention heads",
        "What is the computational complexity of self-attention"]),
    ("Embeddings and representations", [
        "How does subword tokenization (BPE)",
        "Explain the Word2Vec Skip-gram",
        "What are the differences between Word2Vec, GloVe and FastText",
        "Why are contextual embeddings better than static"]),
    ("Training", [
        "When do you use softmax cross-entropy vs",
        "Cross-entropy vs",
        "Adam vs SGD with momentum",
        "Training loss is NaN after 1000 steps"]),
    ("Classical ML and tabular", [
        "When would you deploy logistic regression instead",
        "Random forest vs. gradient boosting",
        "XGBoost, LightGBM or CatBoost",
        "Why does a gradient-boosted tree beat your MLP",
        "Training RMSE 0.31"]),
    ("LLMs in production", [
        "Compare few-shot prompting, fine-tuning, and RAG",
        "Your fine-tuned LLM generates fluent but factually wrong",
        "How do you detect and reduce hallucination"]),
    ("Evaluation and experimentation", [
        "When is random k-fold cross-validation invalid",
        "Precision is 0.95 but recall is 0.30",
        "Your model's AUC is 0.95 but calibration is poor",
        "We want to detect a 1% relative lift",
        "Your model's online metrics diverge from offline",
        "Shadow deployment vs A/B test vs multi-armed bandit"]),
    ("Production craft", [
        "You want to target-encode a 50K-cardinality",
        "Here is a pipeline.",
        "Fraud is 0.1% of transactions",
        "How would you debug poor embedding quality",
        "How do you handle training-serving skew",
        "Your model's CTR dropped 5% overnight"]),
])

# Section order for the depth round: the funnel, front to back.
DEPTH_SECTIONS = ["The funnel", "Query understanding", "Lexical retrieval",
                  "Semantic retrieval", "Hybrid retrieval", "Learning to rank",
                  "Ranking and personalization", "Evaluation"]
BREADTH_SECTIONS = list(BREADTH_REUSE)

KEEP = ["question", "say", "numbers", "red_flags", "followups", "whiteboard",
        "level", "type", "section", "coupang"]


def take(drill, keys, section):
    out = []
    for key in keys:
        hits = [q for q in drill if q["question"].startswith(key)]
        if len(hits) != 1:
            sys.exit("prefix %r matched %d questions (want exactly 1)"
                     % (key, len(hits)))
        q = dict(hits[0])
        q["section"] = section
        q["source"] = "drill deck"
        out.append({k: q[k] for k in KEEP + ["source"] if q.get(k)})
    return out


def main():
    drill = json.load(open(os.path.join(ROOT, "drill", "questions.json")))["questions"]

    new = []
    for f in sorted(glob.glob(os.path.join(C, "depth_new_*.json"))):
        new += json.load(open(f))["questions"]
    for q in new:
        q.setdefault("source", "written for this loop")

    depth = list(new)
    for section, ids in DEPTH_REUSE.items():
        depth += take(drill, ids, section)
    breadth = []
    for section, ids in BREADTH_REUSE.items():
        breadth += take(drill, ids, section)

    coding = json.load(open(os.path.join(C, "coding.json")))["problems"] \
        if os.path.exists(os.path.join(C, "coding.json")) else []

    for name, qs, order in [("depth", depth, DEPTH_SECTIONS),
                            ("breadth", breadth, BREADTH_SECTIONS)]:
        bad = {q["section"] for q in qs} - set(order)
        if bad:
            sys.exit("%s has sections outside the declared order: %s" % (name, bad))
        qs.sort(key=lambda q: order.index(q["section"]))
        for q in qs:
            q["round"] = name

    seen = {}
    for q in depth + breadth:
        k = " ".join(q["question"].lower().split())
        if k in seen:
            sys.exit("duplicate question across the deck: %s" % q["question"][:70])
        seen[k] = 1

    for i, q in enumerate(depth + breadth, 1):
        q["id"] = i

    doc = {"depth": depth, "breadth": breadth, "coding": coding}
    json.dump(doc, open(os.path.join(C, "deck.json"), "w"), indent=1, ensure_ascii=False)

    print("depth %d (%d written, %d reused), breadth %d reused, coding %d"
          % (len(depth), len(new), len(depth) - len(new), len(breadth), len(coding)))
    for name, qs, order in [("depth", depth, DEPTH_SECTIONS),
                            ("breadth", breadth, BREADTH_SECTIONS)]:
        print("  %s:" % name)
        for s in order:
            print("    %-32s %d" % (s, sum(1 for q in qs if q["section"] == s)))


if __name__ == "__main__":
    main()
