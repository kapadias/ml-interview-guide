import numpy as np
from collections import Counter
rng = np.random.default_rng(0)
def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True); e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

# ---------- (h) greedy decode with a KV cache ----------
def decode_with_cache(prompt, Wq, Wk, Wv, Wemb, n_steps):
    """One head, one layer. Cache K,V so each new token costs O(T), not O(T^2)."""
    D = Wq.shape[0]
    K_cache, V_cache = [], []
    tokens = list(prompt)
    for t, tok in enumerate(tokens):            # prefill
        x = Wemb[tok]
        K_cache.append(x @ Wk); V_cache.append(x @ Wv)
    for _ in range(n_steps):                    # decode
        x = Wemb[tokens[-1]]
        q = x @ Wq
        k_new, v_new = x @ Wk, x @ Wv
        if len(tokens) > len(K_cache):          # append only the new position
            K_cache.append(k_new); V_cache.append(v_new)
        K = np.stack(K_cache); V = np.stack(V_cache)
        att = softmax(q @ K.T / np.sqrt(D))
        ctx = att @ V
        logits = Wemb @ ctx                     # tied output embedding
        nxt = int(np.argmax(logits))
        tokens.append(nxt)
        K_cache.append(Wemb[nxt] @ Wk); V_cache.append(Wemb[nxt] @ Wv)
    return tokens

V_, D_ = 11, 8
Wemb = rng.normal(size=(V_, D_)); Wq, Wk, Wv = (rng.normal(size=(D_, D_))*0.3 for _ in range(3))
out = decode_with_cache([1, 2, 3], Wq, Wk, Wv, Wemb, n_steps=4)
assert len(out) == 7 and all(0 <= t < V_ for t in out)
# equivalence check vs recompute-everything
def decode_no_cache(prompt, Wq, Wk, Wv, Wemb, n_steps):
    tokens = list(prompt)
    for _ in range(n_steps):
        X = Wemb[tokens]
        q = X[-1] @ Wq; K = X @ Wk; V = X @ Wv
        ctx = softmax(q @ K.T / np.sqrt(Wq.shape[0])) @ V
        tokens.append(int(np.argmax(Wemb @ ctx)))
    return tokens
assert decode_no_cache([1,2,3], Wq, Wk, Wv, Wemb, 4) == out, (out, decode_no_cache([1,2,3],Wq,Wk,Wv,Wemb,4))
print("(h) KV-cache decode ok; matches recompute:", out)

# ---------- (j) BM25 over a toy inverted index ----------
def build_index(docs):
    index, dl = {}, {}
    for did, text in docs.items():
        toks = text.lower().split(); dl[did] = len(toks)
        for term, tf in Counter(toks).items():
            index.setdefault(term, []).append((did, tf))
    return index, dl

def bm25(query, index, dl, k1=1.5, b=0.75):
    N = len(dl); avgdl = sum(dl.values()) / N
    scores = Counter()
    for term in query.lower().split():
        postings = index.get(term, [])
        if not postings: continue
        df = len(postings)
        idf = np.log(1 + (N - df + 0.5) / (df + 0.5))     # Lucene form, always > 0
        for did, tf in postings:
            denom = tf + k1 * (1 - b + b * dl[did] / avgdl)
            scores[did] += idf * tf * (k1 + 1) / denom
    return scores.most_common()

docs = {"d1": "the cat sat on the mat", "d2": "the dog sat", "d3": "cats and dogs and cats and more cats "*3}
index, dl = build_index(docs)
res = bm25("cat sat", index, dl)
assert res[0][0] == "d1", res
assert all(s > 0 for _, s in res)
print("(j) BM25 ok;", [(d, round(s, 3)) for d, s in res])

# ---------- (k) NDCG@k ----------
def dcg(rels, k):
    rels = np.asarray(rels[:k], dtype=float)
    disc = np.log2(np.arange(2, len(rels) + 2))       # positions 1..k -> log2(i+1)
    return float(((2 ** rels - 1) / disc).sum())

def ndcg(rels, k):
    ideal = sorted(rels, reverse=True)
    idcg = dcg(ideal, k)
    return dcg(rels, k) / idcg if idcg > 0 else 0.0

assert np.isclose(ndcg([3, 2, 3, 0, 1, 2], 6), 0.9488, atol=1e-4), ndcg([3,2,3,0,1,2], 6)
assert ndcg([0, 0, 0], 3) == 0.0
assert np.isclose(ndcg([3, 2, 1], 3), 1.0)
print("(k) NDCG ok; ndcg([3,2,3,0,1,2],6) =", round(ndcg([3,2,3,0,1,2],6), 4))

# ---------- (l) two-tower in-batch softmax loss ----------
def two_tower_loss(U, V, temperature=0.05, logQ=None):
    """U,V: (B,d) L2-normalized. Positives are the diagonal."""
    logits = U @ V.T / temperature
    if logQ is not None:
        logits = logits - logQ                        # sampled-softmax correction
    labels = np.arange(len(U))
    logZ = np.log(np.exp(logits - logits.max(1, keepdims=True)).sum(1)) + logits.max(1)
    return float(np.mean(logZ - logits[labels, labels]))

def l2norm(x): return x / np.linalg.norm(x, axis=-1, keepdims=True)
Bn, d = 4, 6
U = l2norm(rng.normal(size=(Bn, d))); V = l2norm(rng.normal(size=(Bn, d)))
loss_rand = two_tower_loss(U, V)
loss_perfect = two_tower_loss(U, U)                   # positives aligned
assert loss_perfect < loss_rand, (loss_perfect, loss_rand)
assert two_tower_loss(U, U) < np.log(Bn)
print("(l) two-tower ok; aligned =", round(loss_perfect, 4), " random =", round(loss_rand, 4))

# ---------- (m) HNSW-lite greedy search over a fixed layered graph ----------
import heapq
def greedy_search(q, entry, graph, vectors, ef=4, k=2):
    """Beam search on one layer: returns k nearest by L2."""
    def dist(i): return float(np.linalg.norm(vectors[i] - q))
    visited = {entry}
    cand = [(dist(entry), entry)]                     # min-heap frontier
    best = [(-dist(entry), entry)]                    # max-heap of results
    while cand:
        d, node = heapq.heappop(cand)
        if -best[0][0] < d and len(best) >= ef: break # frontier worse than results
        for nb in graph[node]:
            if nb in visited: continue
            visited.add(nb); dn = dist(nb)
            if len(best) < ef or dn < -best[0][0]:
                heapq.heappush(cand, (dn, nb)); heapq.heappush(best, (-dn, nb))
                if len(best) > ef: heapq.heappop(best)
    return [i for _, i in sorted((-nd, i) for nd, i in best)][:k]

pts = rng.normal(size=(20, 2))
graph = {i: list(np.argsort(np.linalg.norm(pts - pts[i], axis=1))[1:5]) for i in range(20)}
q = np.array([0.2, -0.1])
got = greedy_search(q, entry=0, graph=graph, vectors=pts, ef=6, k=2)
truth = list(np.argsort(np.linalg.norm(pts - q, axis=1))[:2])
assert got == truth, (got, truth)
print("(m) HNSW-lite ok; found", got, "== brute force", truth)
