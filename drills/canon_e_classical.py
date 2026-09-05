"""Classical from-scratch canon: the implementables of Volume IV, Chapter 8.

Every function here is the listing printed in the book, plus the test the answer
claims. References are exhaustive/brute-force or numpy's own linear algebra --
never a library version of the same algorithm, so the checks are real.

    python3 canon_e_classical.py      # numpy only
"""
import heapq
import time

import numpy as np

rng = np.random.default_rng(0)


def sigmoid(z):
    """Stable at both tails: never exponentiate a large positive number."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


# ---------- (a) k-means with k-means++ ----------
def sqdist(X, C):
    """||x-c||^2 = ||x||^2 + ||c||^2 - 2 x.c : one matmul, O(nkd)."""
    d2 = (X ** 2).sum(1)[:, None] + (C ** 2).sum(1)[None, :] - 2.0 * X @ C.T
    return np.maximum(d2, 0.0)          # cancellation can make these slightly negative


def kmeanspp(X, k, rs):
    C = [X[rs.integers(len(X))]]
    for _ in range(k - 1):
        d2 = sqdist(X, np.array(C)).min(1)              # distance to nearest chosen seed
        tot = d2.sum()
        p = d2 / tot if tot > 0 else np.full(len(X), 1.0 / len(X))
        C.append(X[rs.choice(len(X), p=p)])             # D^2 sampling
    return np.array(C, dtype=float)


def kmeans(X, k, n_init=5, max_iter=100, tol=1e-6, seed=0):
    rs = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):                             # restarts: Lloyd is local-only
        C = kmeanspp(X, k, rs)
        for _ in range(max_iter):
            d2 = sqdist(X, C)
            lab = d2.argmin(1)
            cnt = np.bincount(lab, minlength=k)
            far = np.argsort(-d2.min(1)) if (cnt == 0).any() else None
            newC, nxt = np.empty_like(C), 0
            for j in range(k):
                if cnt[j]:
                    newC[j] = X[lab == j].mean(0)
                else:                                   # empty cluster: reseed on the
                    newC[j] = X[far[nxt]]               # worst-served point (distinct
                    nxt += 1                            # point per empty cluster)
            shift = np.sqrt(((newC - C) ** 2).sum(1)).max()
            C = newC
            if shift <= tol:                            # tolerance, never ==
                break
        d2 = sqdist(X, C)
        lab = d2.argmin(1)                              # re-assign after the last move
        inertia = float(d2.min(1).sum())
        if best is None or inertia < best[2]:
            best = (C, lab, inertia)
    return best


def lloyd_only(X, C, max_iter=100, tol=1e-6):
    """Lloyd from a given init, no empty-cluster repair: the naive baseline."""
    for _ in range(max_iter):
        d2 = sqdist(X, C)
        lab = d2.argmin(1)
        newC = C.copy()
        for j in range(len(C)):
            if (lab == j).any():
                newC[j] = X[lab == j].mean(0)
        if np.sqrt(((newC - C) ** 2).sum(1)).max() <= tol:
            C = newC
            break
        C = newC
    return float(sqdist(X, C).min(1).sum())


blobs = np.vstack([rng.normal((0, 0), 0.3, (60, 2)),
                   rng.normal((5, 5), 0.3, (60, 2)),
                   rng.normal((0, 5), 0.3, (60, 2))])
truth = np.repeat([0, 1, 2], 60)
C, lab, inertia = kmeans(blobs, 3)
assert all(len(set(lab[truth == c])) == 1 for c in range(3)), "blobs not recovered"
assert len(set(lab.tolist())) == 3
# random init falls into bad local optima on exactly this data
rand_inertias = [lloyd_only(blobs, blobs[np.random.default_rng(s).choice(len(blobs), 3, False)].copy())
                 for s in range(200)]
bad = sum(v > inertia * 2 for v in rand_inertias)
# empty-cluster path: only 3 distinct locations, k=5 forces reseeds
Xdup = np.repeat(np.array([[0., 0.], [1., 0.], [0., 1.]]), 10, axis=0)
C5, _, in5 = kmeans(Xdup, 5, seed=1)
assert np.isfinite(C5).all() and in5 < 1e-12, (C5, in5)
# the expansion trick really does go negative on large coordinates
Xbig = blobs + 1e6
raw = (Xbig ** 2).sum(1)[:, None] + (Xbig ** 2).sum(1)[None, :] - 2 * Xbig @ Xbig.T
print(f"(a) k-means ok: inertia {inertia:.2f}, blobs recovered; random init lands "
      f"in a bad optimum {bad}/200 runs (worst {max(rand_inertias):.0f}); "
      f"k=5 on 3 distinct points -> inertia {in5:g}, no NaN; "
      f"unclipped expansion min {raw.min():.2e}")


# ---------- (b) logistic regression with minibatch SGD ----------
def logreg_obj(w, X, y, l2, mask):
    """mean(log(1+e^z) - y z) is the stable form of the BCE; grad is X^T(p-y)/n."""
    z = X @ w
    loss = float(np.mean(np.logaddexp(0.0, z) - y * z) + 0.5 * l2 * (mask * w * w).sum())
    grad = X.T @ (sigmoid(z) - y) / len(y) + l2 * mask * w
    return loss, grad


def logreg_sgd(X, y, lr=0.5, epochs=40, batch=32, l2=1e-3, seed=0):
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])                # bias as an appended column
    w = np.zeros(d + 1)
    mask = np.ones(d + 1)
    mask[-1] = 0.0                                      # never regularize the intercept
    rs, hist = np.random.default_rng(seed), []
    for _ in range(epochs):
        for idx in np.array_split(rs.permutation(n), max(1, n // batch)):   # reshuffle
            w -= lr * logreg_obj(w, Xb[idx], y[idx], l2, mask)[1]
        hist.append(logreg_obj(w, Xb, y, l2, mask)[0])
    return w, hist


n, d = 400, 3
Xl = rng.normal(size=(n, d))
w_true = np.array([2.0, -3.0, 1.0])
yl = (Xl @ w_true + 0.5 + rng.normal(scale=0.5, size=n) > 0).astype(float)
w, hist = logreg_sgd(Xl, yl)
Xb = np.hstack([Xl, np.ones((n, 1))])
mask = np.ones(d + 1)
mask[-1] = 0.0
acc = float((((Xb @ w) > 0) == yl).mean())
# finite-difference gradient check at a random point (the staff-level move)
w0 = rng.normal(size=d + 1) * 0.5
g = logreg_obj(w0, Xb, yl, 1e-3, mask)[1]
num, eps = np.zeros_like(g), 1e-6
for i in range(len(w0)):
    e = np.zeros_like(w0)
    e[i] = eps
    num[i] = (logreg_obj(w0 + e, Xb, yl, 1e-3, mask)[0]
              - logreg_obj(w0 - e, Xb, yl, 1e-3, mask)[0]) / (2 * eps)
rel = float(np.abs(num - g).max() / np.abs(g).max())
assert rel < 1e-6, rel
assert acc > 0.95 and hist[-1] < hist[0]
cos = float(w[:d] @ w_true / np.linalg.norm(w[:d]) / np.linalg.norm(w_true))
# perfectly separable data: ||w|| diverges without L2
Xs = np.array([[-2., 0.], [-1., 0.], [1., 0.], [2., 0.]])
ys = np.array([0., 0., 1., 1.])
norms = [float(np.linalg.norm(logreg_sgd(Xs, ys, lr=0.5, epochs=e, batch=4, l2=0.0)[0]))
         for e in (50, 500, 5000)]
reg_norm = float(np.linalg.norm(logreg_sgd(Xs, ys, lr=0.5, epochs=5000, batch=4, l2=1e-2)[0]))
assert norms[0] < norms[1] < norms[2] and reg_norm < norms[2]
print(f"(b) logreg ok: acc {acc:.3f}, loss {hist[0]:.4f} -> {hist[-1]:.4f}, "
      f"cos(w, truth) {cos:.4f}, finite-diff rel err {rel:.1e}; separable ||w|| "
      f"{norms[0]:.1f}/{norms[1]:.1f}/{norms[2]:.1f} at 50/500/5000 epochs "
      f"(L2 1e-2 holds it at {reg_norm:.1f})")


# ---------- (c) split finding: the sorted incremental sweep ----------
def best_split_gini(X, y, n_cls, min_leaf=1):
    """(gain, feature, threshold). Sort once per feature, then move one sample at a
    time from the right child to the left: O(d n (log n + K))."""
    n = len(y)
    tot = np.bincount(y, minlength=n_cls).astype(float)
    parent = 1.0 - ((tot / n) ** 2).sum()
    best = (0.0, -1, np.nan)
    for f in range(X.shape[1]):
        order = np.argsort(X[:, f], kind="mergesort")        # O(n log n), once
        xs, ys_ = X[order, f], y[order]
        left, right = np.zeros(n_cls), tot.copy()
        for i in range(n - 1):                               # i = last index on the left
            left[ys_[i]] += 1.0
            right[ys_[i]] -= 1.0                             # O(1) count update
            if xs[i] == xs[i + 1]:                           # split BETWEEN distinct values
                continue
            nl, nr = i + 1, n - i - 1
            if nl < min_leaf or nr < min_leaf:
                continue
            gl = 1.0 - ((left / nl) ** 2).sum()
            gr = 1.0 - ((right / nr) ** 2).sum()
            gain = parent - (nl * gl + nr * gr) / n          # weighted impurity decrease
            if gain > best[0] + 1e-12:
                best = (gain, f, 0.5 * (xs[i] + xs[i + 1]))  # midpoint threshold
    return best


def best_split_sse(X, g, min_leaf=1):
    """Regression twin: SSE = sum y^2 - (sum y)^2/n, so two accumulators per side."""
    n = len(g)
    S, Q = g.sum(), (g * g).sum()
    parent = Q - S * S / n
    best = (0.0, -1, np.nan, 0.0, 0.0)
    for f in range(X.shape[1]):
        order = np.argsort(X[:, f], kind="mergesort")
        xs, gs = X[order, f], g[order]
        sl = ql = 0.0
        for i in range(n - 1):
            sl += gs[i]
            ql += gs[i] * gs[i]
            if xs[i] == xs[i + 1]:
                continue
            nl, nr = i + 1, n - i - 1
            if nl < min_leaf or nr < min_leaf:
                continue
            sse = (ql - sl * sl / nl) + ((Q - ql) - (S - sl) ** 2 / nr)
            if parent - sse > best[0] + 1e-12:
                best = (parent - sse, f, 0.5 * (xs[i] + xs[i + 1]), sl / nl, (S - sl) / nr)
    return best


def grow(X, y, n_cls, depth=0, max_depth=6, min_leaf=1, min_gain=1e-7):
    counts = np.bincount(y, minlength=n_cls)
    leaf = ("leaf", int(counts.argmax()))                    # majority vote
    if depth >= max_depth or counts.max() == len(y) or len(y) < 2 * min_leaf:
        return leaf
    gain, f, thr = best_split_gini(X, y, n_cls, min_leaf)
    if f < 0 or gain < min_gain:            # no admissible split (identical rows): STOP
        return leaf
    m = X[:, f] <= thr
    return ("node", f, thr,
            grow(X[m], y[m], n_cls, depth + 1, max_depth, min_leaf),
            grow(X[~m], y[~m], n_cls, depth + 1, max_depth, min_leaf))


def tree_predict(tree, x):
    while tree[0] == "node":
        tree = tree[3] if x[tree[1]] <= tree[2] else tree[4]
    return tree[1]


def naive_split_gini(X, y, n_cls, min_leaf=1):
    """Reference: re-partition and recount at every candidate. O(n^2) per feature."""
    n = len(y)

    def gini(idx):
        if len(idx) == 0:
            return 0.0
        c = np.bincount(y[idx], minlength=n_cls) / len(idx)
        return 1.0 - (c ** 2).sum()

    parent, best = gini(np.arange(n)), (0.0, -1, np.nan)
    for f in range(X.shape[1]):
        vals = np.unique(X[:, f])
        for a, b in zip(vals[:-1], vals[1:]):
            t = 0.5 * (a + b)
            L = np.where(X[:, f] <= t)[0]
            R = np.where(X[:, f] > t)[0]
            if len(L) < min_leaf or len(R) < min_leaf:
                continue
            gain = parent - (len(L) * gini(L) + len(R) * gini(R)) / n
            if gain > best[0] + 1e-12:
                best = (gain, f, t)
    return best


def naive_split_sse(X, g, min_leaf=1):
    n, best = len(g), (0.0, -1, np.nan, 0.0, 0.0)
    parent = ((g - g.mean()) ** 2).sum()
    for f in range(X.shape[1]):
        vals = np.unique(X[:, f])
        for a, b in zip(vals[:-1], vals[1:]):
            t = 0.5 * (a + b)
            L, R = g[X[:, f] <= t], g[X[:, f] > t]
            if len(L) < min_leaf or len(R) < min_leaf:
                continue
            sse = ((L - L.mean()) ** 2).sum() + ((R - R.mean()) ** 2).sum()
            if parent - sse > best[0] + 1e-12:
                best = (parent - sse, f, t, L.mean(), R.mean())
    return best


for s in range(30):                    # duplicate values + a constant feature on purpose
    r = np.random.default_rng(s)
    nn, dd = int(r.integers(20, 120)), 4
    Xt = r.integers(0, 6, size=(nn, dd)).astype(float)
    Xt[:, 3] = 1.0
    yt = r.integers(0, 3, size=nn)
    a, b = best_split_gini(Xt, yt, 3, 2), naive_split_gini(Xt, yt, 3, 2)
    assert abs(a[0] - b[0]) < 1e-9 and a[1] == b[1] and abs(a[2] - b[2]) < 1e-9, (s, a, b)
    gt = r.normal(size=nn)
    a2, b2 = best_split_sse(Xt, gt, 2), naive_split_sse(Xt, gt, 2)
    assert abs(a2[0] - b2[0]) < 1e-8 and a2[1] == b2[1], (s, a2, b2)
times = {}
for nb in (500, 8000):
    Xb2 = rng.normal(size=(nb, 5))
    yb2 = (rng.random(nb) < 0.4).astype(int)
    t0 = time.perf_counter(); best_split_gini(Xb2, yb2, 2); t1 = time.perf_counter()
    naive_split_gini(Xb2, yb2, 2); t2 = time.perf_counter()
    times[nb] = (t2 - t1) / (t1 - t0)
Xc = rng.normal(size=(300, 2))
yc = ((Xc[:, 0] * Xc[:, 1]) > 0).astype(int)                 # XOR-ish: no good stump
acc_full = float(np.mean([tree_predict(grow(Xc, yc, 2, max_depth=8), x) for x in Xc] == yc))
acc_stump = float(np.mean([tree_predict(grow(Xc, yc, 2, max_depth=1), x) for x in Xc] == yc))
assert grow(np.zeros((20, 2)), np.array([0, 1] * 10), 2, max_depth=10)[0] == "leaf"
print(f"(c) split finding ok: matches the exhaustive scan on 30 random problems "
      f"(duplicates + constant feature); sweep is {times[500]:.1f}x faster at n=500 and "
      f"{times[8000]:.1f}x at n=8000; XOR tree acc {acc_full:.2f} vs stump {acc_stump:.2f}; "
      f"identical rows -> leaf, no infinite recursion")


# ---------- (d) boosting rounds with second-order (Newton) stumps ----------
def newton_stump(X, g, h, lam=1.0, min_leaf=5):
    """Split by the XGBoost gain; leaf weight w* = -G/(H+lambda)."""
    n = len(g)
    G, H = g.sum(), h.sum()
    root = G * G / (H + lam)
    best = (0.0, -1, np.nan, 0.0, 0.0)
    for f in range(X.shape[1]):
        order = np.argsort(X[:, f], kind="mergesort")
        xs, gs, hs = X[order, f], g[order], h[order]
        gl = hl = 0.0
        for i in range(n - 1):
            gl += gs[i]
            hl += hs[i]                                      # O(1) update, as in (c)
            if xs[i] == xs[i + 1] or i + 1 < min_leaf or n - i - 1 < min_leaf:
                continue
            gr, hr = G - gl, H - hl
            gain = 0.5 * (gl * gl / (hl + lam) + gr * gr / (hr + lam) - root)
            if gain > best[0]:
                best = (gain, f, 0.5 * (xs[i] + xs[i + 1]),
                        -gl / (hl + lam), -gr / (hr + lam))   # the leaf weights
    return best


def gbm_logistic(X, y, Xv, yv, rounds=600, lr=0.1, lam=1.0, patience=10):
    p0 = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
    F0 = float(np.log(p0 / (1 - p0)))            # log-odds prior = optimal constant model
    F, Fv, trees = np.full(len(y), F0), np.full(len(yv), F0), []
    tr_hist, va_hist, best = [], [], (np.inf, 0)
    for m in range(rounds):
        p = sigmoid(F)
        g, h = p - y, p * (1 - p)                # gradient and hessian IN LOGIT SPACE
        gain, f, thr, wl, wr = newton_stump(X, g, h, lam)
        if f < 0:
            break
        F += lr * np.where(X[:, f] <= thr, wl, wr)           # shrinkage
        Fv += lr * np.where(Xv[:, f] <= thr, wl, wr)
        trees.append((f, thr, wl, wr))
        tr_hist.append(float(np.mean(np.logaddexp(0.0, F) - y * F)))
        va_hist.append(float(np.mean(np.logaddexp(0.0, Fv) - yv * Fv)))
        if va_hist[-1] < best[0] - 1e-9:
            best = (va_hist[-1], m + 1)
        elif m + 1 - best[1] >= patience:                    # early stopping on validation
            break
    return trees, F0, tr_hist, va_hist, best[1]


def gbm_predict(Xq, trees, F0, lr=0.1):
    F = np.full(len(Xq), F0)
    for f, thr, wl, wr in trees:
        F += lr * np.where(Xq[:, f] <= thr, wl, wr)
    return F


Xg = rng.uniform(-3, 3, size=(600, 2))
yg = ((Xg[:, 0] ** 2 + Xg[:, 1] ** 2 < 4) ^ (Xg[:, 0] > 1.5)).astype(float)
Xgv = rng.uniform(-3, 3, size=(400, 2))
ygv = ((Xgv[:, 0] ** 2 + Xgv[:, 1] ** 2 < 4) ^ (Xgv[:, 0] > 1.5)).astype(float)
trees, F0, tr, va, kbest = gbm_logistic(Xg, yg, Xgv, ygv, lr=0.1)
assert all(tr[i + 1] <= tr[i] + 1e-12 for i in range(len(tr) - 1)), "train loss not monotone"
acc_b = float(((gbm_predict(Xgv, trees[:kbest], F0) > 0) == ygv).mean())
base = float(np.mean(np.logaddexp(0.0, F0) - ygv * F0))
assert va[kbest - 1] < 0.5 * base and acc_b > 0.85
# under squared loss (h=1) with lambda=0 the Newton leaf IS the mean negative residual
g2, h2 = np.array([1.0, -2.0, 0.5, 3.0]), np.ones(4)
_, _, thr2, wl2, wr2 = newton_stump(np.array([[0.], [1.], [2.], [3.]]), g2, h2, lam=0.0, min_leaf=1)
m2 = np.array([0., 1., 2., 3.]) <= thr2
assert abs(wl2 - (-g2[m2].mean())) < 1e-12 and abs(wr2 - (-g2[~m2].mean())) < 1e-12
_, _, _, va1, k1 = gbm_logistic(Xg, yg, Xgv, ygv, lr=1.0)
print(f"(d) boosting ok: train loss {tr[0]:.4f} -> {tr[-1]:.4f} monotone; val logloss "
      f"{va[kbest - 1]:.3f} vs {base:.3f} for the constant model, val acc {acc_b:.3f}, "
      f"best val round {kbest} of {len(tr)} run before early stopping; "
      f"squared-loss leaf == mean residual; "
      f"lr=1.0 needs {k1} rounds for val {min(va1):.3f} vs lr=0.1 needing {kbest}")


# ---------- (e) PCA via power iteration ----------
def pca_power(X, k, iters=100000, tol=1e-12, seed=0):
    """Top-k principal components without ever forming the d x d covariance."""
    rs = np.random.default_rng(seed)
    Xc = X - X.mean(0)                                  # centering is not optional
    n, d = Xc.shape
    comps, evals, its = [], [], []
    for _ in range(k):
        v = rs.normal(size=d)
        v /= np.linalg.norm(v)
        used = 0
        for t in range(iters):
            w = Xc.T @ (Xc @ v) / (n - 1)               # matrix-free: two matvecs, O(nd)
            for u in comps:                             # deflate: project out found PCs
                w -= (u @ w) * u
            nw = np.linalg.norm(w)
            if nw < 1e-300:
                break
            w /= nw                                     # renormalize EVERY step
            used = t + 1
            done = abs(abs(w @ v) - 1.0) < tol          # sign-insensitive convergence
            v = w
            if done:
                break
        lam = float(v @ (Xc.T @ (Xc @ v)) / (n - 1))    # Rayleigh quotient
        comps.append(v)
        evals.append(lam)
        its.append(used)
    return np.array(comps), np.array(evals), its


npca, dpca = 500, 6
A = rng.normal(size=(dpca, dpca))
Xp = rng.multivariate_normal(np.arange(dpca) * 10.0, A @ A.T, size=npca)
V, lam, its = pca_power(Xp, 4)
U_, S_, Vt = np.linalg.svd(Xp - Xp.mean(0), full_matrices=False)
ref_lam = S_ ** 2 / (npca - 1)
align = [abs(float(V[i] @ Vt[i])) for i in range(4)]
assert min(align) > 1 - 1e-9, align
rel_lam = float(np.abs(lam - ref_lam[:4]).max() / ref_lam[0])
assert rel_lam < 1e-9, rel_lam
# forgetting to center: PC1 collapses onto the mean direction
v_nc = rng.normal(size=dpca)
v_nc /= np.linalg.norm(v_nc)
for _ in range(500):
    v_nc = Xp.T @ (Xp @ v_nc)
    v_nc /= np.linalg.norm(v_nc)
mean_dir = Xp.mean(0) / np.linalg.norm(Xp.mean(0))


def exact_cov_data(C, n, seed=0):
    """Data whose EMPIRICAL covariance is exactly C, so the eigengap is exact."""
    r = np.random.default_rng(seed)
    Z = r.normal(size=(n, C.shape[0]))
    Z -= Z.mean(0)
    ev, U = np.linalg.eigh(Z.T @ Z / (n - 1))
    ev2, U2 = np.linalg.eigh(C)
    return Z @ (U @ np.diag(ev ** -0.5) @ U.T) @ (U2 @ np.diag(np.sqrt(ev2)) @ U2.T)


gap_iters = {}
for ratio in (0.5, 0.9, 0.99):
    Q = np.linalg.qr(rng.normal(size=(dpca, dpca)))[0]
    Cg = Q @ np.diag(np.array([1.0, ratio] + [0.05] * (dpca - 2))) @ Q.T
    gap_iters[ratio] = pca_power(exact_cov_data(Cg, 400, seed=1), 1, tol=1e-10, seed=3)[2][0]
assert gap_iters[0.5] < gap_iters[0.9] < gap_iters[0.99]
print(f"(e) PCA ok: |cos| with numpy SVD components {min(align):.12f}, eigenvalue rel err "
      f"{rel_lam:.1e} in {its} iterations; uncentered PC1 . mean-direction = "
      f"{abs(float(v_nc @ mean_dir)):.4f} but . true PC1 = {abs(float(v_nc @ Vt[0])):.4f}; "
      f"iterations at lambda2/lambda1 = 0.5/0.9/0.99: "
      f"{gap_iters[0.5]}/{gap_iters[0.9]}/{gap_iters[0.99]}")


# ---------- (f) brute-force kNN ----------
def knn(Xtr, ytr, Xq, k=5, classify=True):
    d2 = (Xq ** 2).sum(1)[:, None] + (Xtr ** 2).sum(1)[None, :] - 2 * Xq @ Xtr.T
    idx = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]   # O(n) select, not O(n log n) sort
    if classify:
        return np.array([np.bincount(ytr[r]).argmax() for r in idx])
    return ytr[idx].mean(1)


Xtr = rng.normal(size=(500, 4))
ytr = (Xtr[:, 0] + Xtr[:, 1] > 0).astype(int)
Xq = rng.normal(size=(50, 4))
ytest = (Xq[:, 0] + Xq[:, 1] > 0).astype(int)
D = ((Xq[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)     # explicit reference
ref_idx = np.argsort(D, axis=1)[:, :5]
got_idx = np.argpartition(D, 4, axis=1)[:, :5]
assert all(set(a.tolist()) == set(b.tolist()) for a, b in zip(got_idx, ref_idx))
assert (knn(Xtr, ytr, Xq, 5) == np.array([np.bincount(ytr[r]).argmax() for r in ref_idx])).all()
Xbig2 = rng.normal(size=(200000, 8))
qb = rng.normal(size=(20, 8))
d2b = (qb ** 2).sum(1)[:, None] + (Xbig2 ** 2).sum(1)[None, :] - 2 * qb @ Xbig2.T
t0 = time.perf_counter(); np.argpartition(d2b, 9, axis=1)[:, :10]; t1 = time.perf_counter()
np.argsort(d2b, axis=1)[:, :10]; t2 = time.perf_counter()
Xsc, Xqsc = Xtr.copy(), Xq.copy()
Xsc[:, 2] *= 1000.0
Xqsc[:, 2] *= 1000.0                                      # one feature in different units
mu, sd = Xsc.mean(0), Xsc.std(0)
a_raw = float((knn(Xtr, ytr, Xq, 5) == ytest).mean())
a_bad = float((knn(Xsc, ytr, Xqsc, 5) == ytest).mean())
a_std = float((knn((Xsc - mu) / sd, ytr, (Xqsc - mu) / sd, 5) == ytest).mean())
assert a_bad < a_raw - 0.2 and abs(a_std - a_raw) < 1e-9
print(f"(f) brute kNN ok: neighbor sets match a full sort; argpartition {(t2 - t1) / (t1 - t0):.1f}x "
      f"faster than argsort at n=200k; accuracy {a_raw:.2f} -> {a_bad:.2f} when one feature "
      f"is scaled by 1000, back to {a_std:.2f} after standardizing")


# ---------- (g) kd-tree build and k-NN search ----------
class Node:
    __slots__ = ("axis", "thr", "left", "right", "idx")


def kdbuild(X, idx, leaf_size=16):
    nd = Node()
    if len(idx) <= leaf_size:
        nd.axis, nd.idx = -1, idx
        return nd
    a = int((X[idx].max(0) - X[idx].min(0)).argmax())   # widest spread beats round-robin
    m = len(idx) // 2
    idx = idx[np.argpartition(X[idx, a], m)]            # median in O(n), not a full sort
    nd.axis, nd.thr, nd.idx = a, float(X[idx[m], a]), None
    nd.left = kdbuild(X, idx[:m], leaf_size)            # every coord here is <= thr
    nd.right = kdbuild(X, idx[m:], leaf_size)           # every coord here is >= thr
    return nd


def kdquery(nd, X, q, k, heap, stats):
    if nd.axis == -1:                                   # leaf: brute force the bucket
        for i in nd.idx:
            stats[0] += 1
            dist = float(((X[i] - q) ** 2).sum())
            if len(heap) < k:
                heapq.heappush(heap, (-dist, int(i)))   # max-heap of the k best
            elif dist < -heap[0][0]:
                heapq.heapreplace(heap, (-dist, int(i)))
        return
    diff = q[nd.axis] - nd.thr
    near, far = (nd.left, nd.right) if diff <= 0 else (nd.right, nd.left)
    kdquery(near, X, q, k, heap, stats)                 # descend the near side first
    if len(heap) < k or diff * diff < -heap[0][0]:      # squared vs squared, consistently
        kdquery(far, X, q, k, heap, stats)              # sphere crosses the plane: visit


def kdknn(root, X, q, k):
    heap, stats = [], [0]
    kdquery(root, X, q, k, heap, stats)
    return sorted((-d, i) for d, i in heap), stats[0]


frac = {}
for dk in (2, 5, 10, 20, 50):
    P = rng.normal(size=(4000, dk))
    root = kdbuild(P, np.arange(len(P)))
    seen = []
    for _ in range(30):
        q = rng.normal(size=dk)
        got, nvis = kdknn(root, P, q, 5)
        assert np.allclose([gg[0] for gg in got], np.sort(((P - q) ** 2).sum(1))[:5])
        seen.append(nvis)
    frac[dk] = 100 * float(np.mean(seen)) / len(P)
assert frac[2] < 5 and frac[20] > 95
print("(g) kd-tree ok: exact match with brute force in every dimension; "
      "distances computed " + ", ".join(f"d={k}: {v:.0f}%" for k, v in frac.items())
      + " -- pruning stops firing and it degenerates to brute force")


# ---------- (h) AUC from scratch ----------
def auc(y, s):
    """Mann-Whitney U in rank form: O(n log n); ties get half credit."""
    y, s = np.asarray(y, dtype=float), np.asarray(s, dtype=float)
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    i = 0
    while i < len(s):                        # average the ranks inside each tied group
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = 0.5 * (i + 1 + j + 1)
        i = j + 1
    npos, nneg = float(y.sum()), float((1 - y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")                  # undefined when only one class is present
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def auc_pairs(y, s):
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(np.mean([[(p > q) + 0.5 * (p == q) for q in neg] for p in pos]))


for t in range(200):
    r = np.random.default_rng(t)
    nn = int(r.integers(4, 60))
    yv = (r.random(nn) < 0.4).astype(float)
    sv = r.integers(0, 5, size=nn).astype(float) if t % 2 else r.normal(size=nn)
    a_fast, a_slow = auc(yv, sv), auc_pairs(yv, sv)
    assert (np.isnan(a_fast) and np.isnan(a_slow)) or abs(a_fast - a_slow) < 1e-12, t
assert auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
assert auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0
assert auc([0, 0, 1, 1], [1.0, 1.0, 1.0, 1.0]) == 0.5
assert np.isnan(auc([1, 1, 1], [1.0, 2.0, 3.0]))
print("(h) AUC ok: matches pairwise counting on 200 random cases including heavy ties; "
      "perfect 1.0, reversed 0.0, all-tied 0.5, single-class nan")


# ---------- (i) linear regression: normal equations and gradient descent ----------
def linreg_normal(X, y, l2=0.0):
    Xb_ = np.hstack([X, np.ones((len(X), 1))])
    P = np.eye(Xb_.shape[1])
    P[-1, -1] = 0.0                                        # don't penalize the intercept
    return np.linalg.solve(Xb_.T @ Xb_ + l2 * P, Xb_.T @ y)   # solve, never inv()


def linreg_gd(X, y, steps=2000, l2=0.0):
    Xb_ = np.hstack([X, np.ones((len(X), 1))])
    nn = len(y)
    w_ = np.zeros(Xb_.shape[1])
    L = float(np.linalg.eigvalsh(Xb_.T @ Xb_ / nn).max()) + l2   # 1/L is a safe step size
    for _ in range(steps):
        w_ -= (1.0 / L) * (Xb_.T @ (Xb_ @ w_ - y) / nn + l2 * w_)
    return w_


Xr = rng.normal(size=(200, 4))
beta = np.array([1.5, -2.0, 0.0, 3.0])
yr = Xr @ beta + 0.7 + rng.normal(scale=0.3, size=200)
w_ne = linreg_normal(Xr, yr)
w_ls = np.linalg.lstsq(np.hstack([Xr, np.ones((200, 1))]), yr, rcond=None)[0]
w_gd = linreg_gd(Xr, yr)
assert np.abs(w_ne - w_ls).max() < 1e-10 and np.abs(w_gd - w_ls).max() < 1e-8
errs = {}
for kappa in (1e4, 1e8):
    dk = 5
    Uk = np.linalg.qr(rng.normal(size=(200, dk)))[0]
    Vk = np.linalg.qr(rng.normal(size=(dk, dk)))[0]
    Xk = Uk @ np.diag(np.logspace(0, -np.log10(kappa), dk)) @ Vk.T
    bk = rng.normal(size=dk)
    yk = Xk @ bk
    errs[kappa] = (float(np.abs(np.linalg.solve(Xk.T @ Xk, Xk.T @ yk) - bk).max()),
                   float(np.abs(np.linalg.lstsq(Xk, yk, rcond=None)[0] - bk).max()))
assert errs[1e8][0] > 1e4 * errs[1e8][1]
print(f"(i) linear regression ok: normal equations agree with lstsq to "
      f"{np.abs(w_ne - w_ls).max():.1e} and GD to {np.abs(w_gd - w_ls).max():.1e}; at "
      f"cond(X)=1e8 the normal equations err {errs[1e8][0]:.1e} vs QR {errs[1e8][1]:.1e} "
      f"(cond 1e4: {errs[1e4][0]:.1e} vs {errs[1e4][1]:.1e}) -- X^T X squares the condition number")


# ---------- (j) multinomial naive Bayes in log space ----------
def nb_fit(counts, y, n_cls, alpha=1.0):
    """counts: (n, V) term counts. Everything in logs; alpha is Laplace smoothing."""
    logprior = np.log(np.bincount(y, minlength=n_cls) + alpha) - np.log(len(y) + alpha * n_cls)
    cw = np.vstack([counts[y == c].sum(0) + alpha for c in range(n_cls)])    # (K, V)
    return logprior, np.log(cw) - np.log(cw.sum(1, keepdims=True))


def nb_predict(counts, logprior, loglik):
    return (counts @ loglik.T + logprior).argmax(1)      # add logs; never multiply probs


Vv, Kk = 30, 3
theta = rng.dirichlet(np.ones(Vv) * 0.5, size=Kk)
ynb = rng.integers(0, Kk, size=300)
counts = np.vstack([rng.multinomial(40, theta[c]) for c in ynb]).astype(float)
lp, ll = nb_fit(counts, ynb, Kk)
acc_nb = float((nb_predict(counts, lp, ll) == ynb).mean())
zeros = int((np.vstack([counts[ynb == c].sum(0) for c in range(Kk)]) == 0).sum())
lp2, ll2 = nb_fit(np.array([[2., 0.], [0., 2.]]), np.array([0, 1]), 2, alpha=1.0)
assert np.allclose(np.exp(ll2), [[0.75, 0.25], [0.25, 0.75]])    # hand-checkable
assert acc_nb > 0.9 and zeros > 0
print(f"(j) naive Bayes ok: train acc {acc_nb:.3f}; hand-checked smoothed table "
      f"[[0.75, 0.25], [0.25, 0.75]]; {zeros} count cells are zero, i.e. log(0) without smoothing")


# ---------- (k) stratified k-fold from scratch ----------
def stratified_folds(y, k=5, seed=0):
    """Deal each class round-robin so every fold sees the population class mix."""
    rs = np.random.default_rng(seed)
    fold = np.empty(len(y), dtype=int)
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rs.shuffle(idx)
        fold[idx] = np.arange(len(idx)) % k
    return [(np.where(fold != f)[0], np.where(fold == f)[0]) for f in range(k)]


yimb = np.array([0] * 180 + [1] * 20)
folds = stratified_folds(yimb, 5, seed=1)
assert sorted(np.concatenate([te for _, te in folds]).tolist()) == list(range(len(yimb)))
assert all(not (set(tr.tolist()) & set(te.tolist())) for tr, te in folds)
rates = [round(float(yimb[te].mean()), 3) for _, te in folds]
plain = [round(float(yimb[p].mean()), 3)
         for p in np.array_split(np.random.default_rng(0).permutation(len(yimb)), 5)]
assert max(rates) - min(rates) < 1e-9 and max(plain) - min(plain) > 0.05
print(f"(k) stratified folds ok: positive rate {rates} in every fold (population "
      f"{yimb.mean():.2f}); unstratified splits of the same data give {plain}")

# ---------- (l) CV harness, and what feature selection outside the fold is worth ----------
def cross_val_score(fit_predict, X, y, k=5, seed=0):
    """fit_predict(Xtr, ytr, Xte) -> yhat. EVERYTHING fitted -- scalers, imputers,
    encoders, feature selection, tuning -- must happen INSIDE this callable."""
    out = []
    for tr, te in stratified_folds(y, k, seed):
        yhat = fit_predict(X[tr], y[tr], X[te])
        out.append(float((yhat == y[te]).mean()))
    return float(np.mean(out)), float(np.std(out))   # report the spread, not just mean


def corr_scores(X, y):
    return np.abs(((X - X.mean(0)) * (y - y.mean())[:, None]).mean(0)
                  / (X.std(0) * y.std() + 1e-12))


def select_and_fit(Xtr, ytr, Xte, m=10):
    """Selection is INSIDE, so it only ever sees the training fold."""
    keep = np.argsort(-corr_scores(Xtr, ytr))[:m]
    mu0, mu1 = Xtr[ytr == 0][:, keep].mean(0), Xtr[ytr == 1][:, keep].mean(0)
    d0 = ((Xte[:, keep] - mu0) ** 2).sum(1)
    d1 = ((Xte[:, keep] - mu1) ** 2).sum(1)
    return (d1 < d0).astype(float)


def centroid_fit(Xtr, ytr, Xte):
    mu0, mu1 = Xtr[ytr == 0].mean(0), Xtr[ytr == 1].mean(0)
    return (((Xte - mu1) ** 2).sum(1) < ((Xte - mu0) ** 2).sum(1)).astype(float)


honest, leaked = [], []
for s in range(10):                     # pure noise: the true accuracy is exactly 0.5
    r = np.random.default_rng(s)
    Xn = r.normal(size=(100, 2000))
    yn = (r.random(100) < 0.5).astype(float)
    honest.append(cross_val_score(select_and_fit, Xn, yn, k=5, seed=s)[0])
    pre = np.argsort(-corr_scores(Xn, yn))[:10]     # THE LEAK: selection sees all rows
    leaked.append(cross_val_score(centroid_fit, Xn[:, pre], yn, k=5, seed=s)[0])
assert abs(np.mean(honest) - 0.5) < 0.06 and np.mean(leaked) > 0.75
print(f"(l) CV harness ok: on 2000 pure-noise features the honest score is "
      f"{np.mean(honest):.3f} (range {min(honest):.2f}-{max(honest):.2f}, truth 0.5) but "
      f"selecting features before the split reports {np.mean(leaked):.3f} "
      f"(range {min(leaked):.2f}-{max(leaked):.2f})")

print("\nall classical-canon drills passed.")
