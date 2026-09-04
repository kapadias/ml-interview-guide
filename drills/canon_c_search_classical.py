import numpy as np
rng = np.random.default_rng(0)
def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True); e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

# ---------- cleaned KV-cache decode (book version) ----------
def greedy_decode(prompt, Wq, Wk, Wv, Wemb, n_new):
    D = Wq.shape[0]
    toks = list(prompt)
    K = [Wemb[t] @ Wk for t in toks]          # prefill: one pass over the prompt
    V = [Wemb[t] @ Wv for t in toks]
    for _ in range(n_new):
        q = Wemb[toks[-1]] @ Wq               # only the newest token needs a query
        att = softmax(q @ np.stack(K).T / np.sqrt(D))
        logits = Wemb @ (att @ np.stack(V))   # tied embeddings
        nxt = int(logits.argmax())
        toks.append(nxt)
        K.append(Wemb[nxt] @ Wk)              # extend cache by one row
        V.append(Wemb[nxt] @ Wv)
    return toks
Vv, Dd = 11, 8
Wemb = rng.normal(size=(Vv, Dd)); Wq, Wk, Wv = (rng.normal(size=(Dd, Dd))*0.3 for _ in range(3))
def slow(prompt, n):
    toks = list(prompt)
    for _ in range(n):
        X = Wemb[toks]
        ctx = softmax(X[-1] @ Wq @ (X @ Wk).T / np.sqrt(Dd)) @ (X @ Wv)
        toks.append(int((Wemb @ ctx).argmax()))
    return toks
assert greedy_decode([1,2,3], Wq, Wk, Wv, Wemb, 4) == slow([1,2,3], 4)
print("(h2) cleaned KV decode ok:", greedy_decode([1,2,3], Wq, Wk, Wv, Wemb, 4))

# ---------- (i) beam search ----------
def beam_search(step_fn, start, beam=3, max_len=5, eos=0, alpha=0.7):
    beams = [([start], 0.0)]                    # (tokens, sum logprob)
    done = []
    for _ in range(max_len):
        cand = []
        for toks, lp in beams:
            logp = np.log(step_fn(toks) + 1e-12)
            for tok in np.argsort(-logp)[:beam]:
                nt, nlp = toks + [int(tok)], lp + float(logp[tok])
                (done if tok == eos else cand).append((nt, nlp))
        if not cand: break
        beams = sorted(cand, key=lambda x: x[1], reverse=True)[:beam]
    done += beams
    # length-normalize so short sequences don't automatically win
    return max(done, key=lambda x: x[1] / (len(x[0]) ** alpha))

P = softmax(rng.normal(size=(6, 6)), axis=-1)
best = beam_search(lambda toks: P[toks[-1]], start=1, beam=3, max_len=4)
greedy = [1]
for _ in range(4):
    greedy.append(int(P[greedy[-1]].argmax()))
    if greedy[-1] == 0: break
def score(seq): return sum(np.log(P[seq[i]][seq[i+1]]) for i in range(len(seq)-1))
assert score(best[0]) >= score(greedy) - 1e-9, (best, greedy)
print("(i) beam ok; beam", best[0], round(score(best[0]),3), ">= greedy", greedy, round(score(greedy),3))

# ---------- (n) logistic regression via SGD ----------
def logreg_sgd(X, y, lr=0.1, epochs=200, l2=0.0, seed=0):
    rs = np.random.default_rng(seed)
    n, d = X.shape
    w = np.zeros(d); b = 0.0
    for _ in range(epochs):
        for i in rs.permutation(n):
            z = X[i] @ w + b
            p = 1 / (1 + np.exp(-z))            # sigma(z)
            g = p - y[i]                        # dL/dz for log loss  <- the whole trick
            w -= lr * (g * X[i] + l2 * w)
            b -= lr * g
    return w, b

Xp = rng.normal(size=(200, 2)); w_true = np.array([2.0, -3.0])
yp = (Xp @ w_true + 0.5 + rng.normal(scale=0.3, size=200) > 0).astype(float)
w, b = logreg_sgd(Xp, yp)
acc = (((Xp @ w + b) > 0) == yp).mean()
assert acc > 0.93, acc
assert np.corrcoef(w, w_true)[0,1] > 0.99
print("(n) logreg ok; acc =", round(float(acc),3), " w =", np.round(w,2))

# ---------- (o) k-means ----------
def kmeans(X, k, iters=50, seed=0):
    rs = np.random.default_rng(seed)
    C = [X[rs.integers(len(X))]]                             # k-means++ seeding
    for _ in range(k - 1):
        d2 = ((X[:, None, :] - np.array(C)[None]) ** 2).sum(-1).min(1)
        C.append(X[rs.choice(len(X), p=d2 / d2.sum())])       # far points more likely
    C = np.array(C)
    for _ in range(iters):
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)   # (n,k) vectorized
        a = d.argmin(1)
        newC = C.copy()
        for j in range(k):
            if (a == j).any():
                newC[j] = X[a == j].mean(0)
            else:                                # empty cluster: reseed on worst point
                newC[j] = X[d.min(1).argmax()]
        if np.allclose(newC, C): break
        C = newC
    inertia = float(((X - C[a]) ** 2).sum())
    return C, a, inertia

blobs = np.vstack([rng.normal(loc=(0,0), scale=.3, size=(50,2)),
                   rng.normal(loc=(5,5), scale=.3, size=(50,2)),
                   rng.normal(loc=(0,5), scale=.3, size=(50,2))])
C, a, inertia = kmeans(blobs, 3)
assert len(set(a)) == 3 and inertia < 60, (inertia, set(a))
# monotone decrease with k
_, _, in4 = kmeans(blobs, 4)
assert in4 <= inertia + 1e-6
print("(o) k-means ok; inertia k=3:", round(inertia,2), " k=4:", round(in4,2))

# ---------- (p) one round of gradient boosting with stumps ----------
def fit_stump(X, g):
    """Best axis-aligned split minimizing squared error on the residuals g."""
    best = (np.inf, 0, 0.0, 0.0, 0.0)
    for f in range(X.shape[1]):
        order = np.argsort(X[:, f]); xs, gs = X[order, f], g[order]
        for i in range(1, len(xs)):
            if xs[i] == xs[i-1]: continue
            L, R = gs[:i], gs[i:]
            sse = ((L - L.mean())**2).sum() + ((R - R.mean())**2).sum()
            if sse < best[0]:
                best = (sse, f, (xs[i] + xs[i-1]) / 2, L.mean(), R.mean())
    _, f, thr, vl, vr = best
    return f, thr, vl, vr

def gbm_fit(X, y, n_rounds=30, lr=0.1):
    F = np.full(len(y), y.mean())               # init with the base rate
    trees = []
    for _ in range(n_rounds):
        resid = y - F                           # neg gradient of squared loss
        f, thr, vl, vr = fit_stump(X, resid)
        F += lr * np.where(X[:, f] <= thr, vl, vr)
        trees.append((f, thr, vl, vr))
    return trees, y.mean()

def gbm_predict(X, trees, init, lr=0.1):
    F = np.full(len(X), init)
    for f, thr, vl, vr in trees:
        F += lr * np.where(X[:, f] <= thr, vl, vr)
    return F

Xg = rng.uniform(-3, 3, size=(120, 2))
yg = np.sin(Xg[:, 0]) + 0.5 * Xg[:, 1]
trees, init = gbm_fit(Xg, yg, n_rounds=60, lr=0.1)
pred = gbm_predict(Xg, trees, init)
mse_model = float(((yg - pred) ** 2).mean()); mse_base = float(((yg - yg.mean()) ** 2).mean())
assert mse_model < 0.25 * mse_base, (mse_model, mse_base)
print("(p) GBM stumps ok; MSE", round(mse_model,4), "vs baseline", round(mse_base,4))

# ---------- (e) conv2d via im2col ----------
def im2col(x, kh, kw, stride=1):
    H, W = x.shape
    oh, ow = (H - kh)//stride + 1, (W - kw)//stride + 1
    cols = np.empty((oh*ow, kh*kw))
    for i in range(oh):
        for j in range(ow):
            cols[i*ow + j] = x[i*stride:i*stride+kh, j*stride:j*stride+kw].ravel()
    return cols, oh, ow

def conv2d(x, kernel, stride=1):
    cols, oh, ow = im2col(x, *kernel.shape, stride)
    return (cols @ kernel.ravel()).reshape(oh, ow)

img = rng.normal(size=(6, 6)); ker = rng.normal(size=(3, 3))
out = conv2d(img, ker)
ref = np.array([[ (img[i:i+3, j:j+3] * ker).sum() for j in range(4)] for i in range(4)])
assert np.allclose(out, ref)
print("(e) conv2d/im2col ok; out shape", out.shape)
