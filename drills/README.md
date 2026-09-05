# Drills

Runnable, self-testing reference implementations for the problems in
Volume I, Chapter 28 (*ML Coding Rounds*) and Volume IV, Chapter 8 (*The From-Scratch Coding Canon*). Every function here was executed
before it was printed in the book; each file asserts its own correctness
and prints a one-line result per problem.

Run them all:

```bash
python3 canon_a_attention_optim.py
python3 canon_b_inference_retrieval.py
python3 canon_c_search_classical.py
python3 canon_d_torch_debug.py     # needs torch; the others need only numpy
python3 canon_e_classical.py
```

## Problem index

| File | Problems |
|---|---|
| `canon_a_attention_optim.py` | multi-head causal attention; AdamW step; LayerNorm forward + backward (checked against finite differences); sampling with temperature/top-k/top-p; BPE train + encode |
| `canon_b_inference_retrieval.py` | greedy decode with a KV cache (checked against full recomputation); BM25 over an inverted index; NDCG@k; two-tower in-batch softmax loss; HNSW-style greedy graph search (checked against brute force) |
| `canon_c_search_classical.py` | cleaned KV-cache decode; beam search with length normalization; logistic regression via SGD; k-means with k-means++; gradient boosting with stumps; conv2d via im2col |
| `canon_d_torch_debug.py` | minimal training loop with gradient clipping; planted-bug demonstrations (missing `zero_grad`, runaway learning rate, train/eval mode, causal-mask off-by-one producing NaN) |
| `canon_e_classical.py` | The classical from-scratch canon of Volume IV, Chapter 8: k-means with k-means++ and empty-cluster repair; logistic regression by minibatch SGD (finite-difference gradient check); decision-tree split finding and a small tree (checked against an exhaustive re-partitioning scan); boosting with second-order stumps; PCA by power iteration (checked against `np.linalg.svd`); brute-force kNN; kd-tree build and search (checked against brute force, with the high-dimensional degeneration measured); AUC (checked against pairwise counting); linear regression by normal equations and GD; naive Bayes; stratified folds; a CV harness with a measured feature-selection leak |
| `debug_round.py` | The debug-the-training-code round: generates a training script with one planted bug for you to diagnose (`--list`, `--bug N`, `--reveal FILE`, `--clean`) |

## How to practice

Do not read these first. Set a 45-minute timer, implement the problem from
the prompt in Chapter 28 with no autocomplete and no assistant, write your
own tests, and only then diff against the file here. The gap between
understanding attention and writing it in twenty minutes is what the round
measures, and it closes only by doing it.

For the debug round, use `debug_round.py`. It plants one bug from Chapter 28's
taxonomy in a working training script and does not tell you which:

```bash
python3 debug_round.py            # -> broken_train.py
python3 broken_train.py           # read the curve, classify the symptom
python3 debug_round.py --reveal broken_train.py
```

Work the diagnostic order before revealing: read the loss curve and classify the
symptom, overfit a single batch, check the loop mechanics, check shapes and label
alignment, check the data path, check the metric. Every symptom the script
documents was measured by running that variant, and `--list` also names the two
bugs from the taxonomy it deliberately does not plant, because on this data they
produce no visible symptom at all --- which is the lesson about leakage.
