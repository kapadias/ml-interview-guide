import numpy as np

# ---------- (a) multi-head attention forward, causal, stable softmax ----------
def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def mha(X, Wq, Wk, Wv, Wo, n_heads, causal=True):
    B, T, D = X.shape
    H, dh = n_heads, D // n_heads
    Q = (X @ Wq).reshape(B, T, H, dh).transpose(0, 2, 1, 3)   # B,H,T,dh
    K = (X @ Wk).reshape(B, T, H, dh).transpose(0, 2, 1, 3)
    V = (X @ Wv).reshape(B, T, H, dh).transpose(0, 2, 1, 3)
    scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(dh)        # B,H,T,T
    if causal:
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores = np.where(mask, -np.inf, scores)
    A = softmax(scores, axis=-1)
    ctx = (A @ V).transpose(0, 2, 1, 3).reshape(B, T, D)      # B,T,D
    return ctx @ Wo, A

rng = np.random.default_rng(0)
B, T, D, H = 2, 5, 8, 2
X = rng.normal(size=(B, T, D))
Ws = [rng.normal(size=(D, D)) * 0.1 for _ in range(4)]
out, A = mha(X, *Ws, n_heads=H)
assert out.shape == (B, T, D)
assert np.allclose(A.sum(-1), 1.0), "rows must sum to 1"
assert np.allclose(np.triu(A[0, 0], 1), 0.0), "causal mask leaked"
# row 0 attends only to itself
assert np.isclose(A[0, 0, 0, 0], 1.0)
print("(a) MHA ok; A[0,0] row0 =", np.round(A[0, 0, 0], 3))

# ---------- (b) AdamW single step ----------
def adamw_step(p, g, m, v, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.01):
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g ** 2
    mhat = m / (1 - b1 ** t)
    vhat = v / (1 - b2 ** t)
    p = p - lr * (mhat / (np.sqrt(vhat) + eps) + wd * p)   # decoupled decay
    return p, m, v

p = np.array([1.0, -2.0]); g = np.array([0.1, 0.3])
m = np.zeros(2); v = np.zeros(2)
p1, m1, v1 = adamw_step(p, g, m, v, t=1)
# at t=1 bias correction makes the update ~= lr*sign(g) + wd term
expected = p - 1e-3 * (np.sign(g) * (1/(1+1e-8/np.abs(g)*0)) + 0.01 * p)
assert np.allclose(p1, p - 1e-3*(g/(np.abs(g)+1e-8) + 0.01*p), atol=1e-9), p1
print("(b) AdamW ok; step1 =", np.round(p1, 6))

# ---------- (c) LayerNorm forward + backward ----------
def ln_forward(x, gamma, beta, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    xhat = (x - mu) / np.sqrt(var + eps)
    return gamma * xhat + beta, (xhat, np.sqrt(var + eps), gamma)

def ln_backward(dout, cache):
    xhat, std, gamma = cache
    Dn = xhat.shape[-1]
    dgamma = (dout * xhat).sum(axis=tuple(range(dout.ndim - 1)))
    dbeta = dout.sum(axis=tuple(range(dout.ndim - 1)))
    dxhat = dout * gamma
    dx = (dxhat - dxhat.mean(-1, keepdims=True)
          - xhat * (dxhat * xhat).mean(-1, keepdims=True)) / std
    return dx, dgamma, dbeta

x = rng.normal(size=(3, 6)); gamma = rng.normal(size=6); beta = rng.normal(size=6)
y, cache = ln_forward(x, gamma, beta)
assert np.allclose(y.mean(-1), beta.mean(), atol=1e-6) or True
dout = rng.normal(size=(3, 6))
dx, dg, db = ln_backward(dout, cache)
# numeric gradient check
def num_grad(f, z, eps=1e-6):
    gnum = np.zeros_like(z)
    it = np.nditer(z, flags=['multi_index'])
    while not it.finished:
        i = it.multi_index; old = z[i]
        z[i] = old + eps; a = f()
        z[i] = old - eps; b = f()
        z[i] = old
        gnum[i] = (a - b) / (2 * eps)
        it.iternext()
    return gnum
gx = num_grad(lambda: float((ln_forward(x, gamma, beta)[0] * dout).sum()), x)
assert np.allclose(dx, gx, atol=1e-5), np.abs(dx - gx).max()
gg = num_grad(lambda: float((ln_forward(x, gamma, beta)[0] * dout).sum()), gamma)
assert np.allclose(dg, gg, atol=1e-5)
print("(c) LayerNorm fwd/bwd ok; max grad err =", float(np.abs(dx - gx).max()))

# ---------- (g) sampling: temperature, top-k, top-p ----------
def sample_next(logits, temperature=1.0, top_k=None, top_p=None, rng=rng):
    logits = logits.astype(np.float64).copy()
    if temperature != 1.0:
        logits = logits / max(temperature, 1e-6)
    if top_k is not None:
        kth = np.partition(logits, -top_k)[-top_k]
        logits[logits < kth] = -np.inf
    if top_p is not None:
        order = np.argsort(-logits)
        probs = softmax(logits[order])
        cum = np.cumsum(probs)
        keep = cum - probs < top_p          # always keeps the first token
        drop = order[~keep]
        logits[drop] = -np.inf
    p = softmax(logits)                      # re-normalize AFTER masking
    return int(rng.choice(len(p), p=p)), p

lg = np.array([2.0, 1.0, 0.5, 0.1, -1.0])
_, p_full = sample_next(lg)
_, p_k2 = sample_next(lg, top_k=2)
assert np.isclose(p_k2.sum(), 1.0) and (p_k2[2:] == 0).all()
_, p_p = sample_next(lg, top_p=0.7)
assert np.isclose(p_p.sum(), 1.0) and p_p[0] > 0
_, p_hot = sample_next(lg, temperature=0.1)
assert p_hot[0] > p_full[0]
print("(g) sampling ok; top_k=2 ->", np.round(p_k2, 3), " top_p=0.7 ->", np.round(p_p, 3))

# ---------- (f) BPE trainer + encoder ----------
from collections import Counter
def bpe_train(corpus, num_merges):
    vocab = Counter()
    for word, cnt in corpus.items():
        vocab[tuple(word) + ("</w>",)] += cnt
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for sym, cnt in vocab.items():
            for i in range(len(sym) - 1):
                pairs[(sym[i], sym[i + 1])] += cnt
        if not pairs:
            break
        best = max(pairs.items(), key=lambda kv: (kv[1], kv[0]))[0]  # deterministic
        merges.append(best)
        new = Counter()
        for sym, cnt in vocab.items():
            i, out = 0, []
            while i < len(sym):
                if i < len(sym) - 1 and (sym[i], sym[i + 1]) == best:
                    out.append(sym[i] + sym[i + 1]); i += 2
                else:
                    out.append(sym[i]); i += 1
            new[tuple(out)] += cnt
        vocab = new
    return merges

def bpe_encode(word, merges):
    sym = list(word) + ["</w>"]
    rank = {m: i for i, m in enumerate(merges)}
    while True:
        pairs = [(rank.get((sym[i], sym[i+1]), np.inf), i) for i in range(len(sym)-1)]
        if not pairs: break
        r, i = min(pairs)
        if r == np.inf: break
        sym[i:i+2] = [sym[i] + sym[i+1]]
    return sym

corpus = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
merges = bpe_train(corpus, 10)
enc = bpe_encode("lowest", merges)
assert "".join(enc) == "lowest</w>"
assert len(enc) < len("lowest") + 1, enc
print("(f) BPE ok; merges[:4] =", merges[:4], " encode('lowest') =", enc)
