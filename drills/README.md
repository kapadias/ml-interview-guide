# Drills

Runnable, self-testing reference implementations for the problems in
Volume I, Chapter 28 (*ML Coding Rounds*). Every function here was executed
before it was printed in the book; each file asserts its own correctness
and prints a one-line result per problem.

Run them all:

```bash
python3 canon_a_attention_optim.py
python3 canon_b_inference_retrieval.py
python3 canon_c_search_classical.py
python3 canon_d_torch_debug.py     # needs torch; the others need only numpy
```

## Problem index

| File | Problems |
|---|---|
| `canon_a_attention_optim.py` | multi-head causal attention; AdamW step; LayerNorm forward + backward (checked against finite differences); sampling with temperature/top-k/top-p; BPE train + encode |
| `canon_b_inference_retrieval.py` | greedy decode with a KV cache (checked against full recomputation); BM25 over an inverted index; NDCG@k; two-tower in-batch softmax loss; HNSW-style greedy graph search (checked against brute force) |
| `canon_c_search_classical.py` | cleaned KV-cache decode; beam search with length normalization; logistic regression via SGD; k-means with k-means++; gradient boosting with stumps; conv2d via im2col |
| `canon_d_torch_debug.py` | minimal training loop with gradient clipping; planted-bug demonstrations (missing `zero_grad`, runaway learning rate, train/eval mode, causal-mask off-by-one producing NaN) |

## How to practice

Do not read these first. Set a 45-minute timer, implement the problem from
the prompt in Chapter 28 with no autocomplete and no assistant, write your
own tests, and only then diff against the file here. The gap between
understanding attention and writing it in twenty minutes is what the round
measures, and it closes only by doing it.

For the debug round, corrupt a working script yourself (or use the bug
switches in `canon_d_torch_debug.py`) and practice the diagnostic order from
Chapter 28: read the loss curve, overfit one batch, check the loop
mechanics, check shapes and alignment, check the data path, check the metric.
