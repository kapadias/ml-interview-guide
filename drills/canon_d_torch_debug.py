import torch, torch.nn as nn
torch.manual_seed(0)

# ---------- (d) minimal training loop with grad clipping ----------
def train(model, loader, epochs=2, lr=1e-3, clip=1.0, device="cpu"):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    lossf = nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train()                                  # 1. train mode
        total, n = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)            # 2. clear grads FIRST
            logits = model(xb)
            loss = lossf(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)   # 3. after backward, before step
            opt.step()
            total += loss.item() * len(xb); n += len(xb)
        yield ep, total / n

X = torch.randn(256, 10); y = (X[:, 0] + X[:, 1] > 0).long()
ds = torch.utils.data.TensorDataset(X, y)
loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)
model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
losses = [l for _, l in train(model, loader, epochs=6, lr=1e-2)]
assert losses[-1] < losses[0], losses
print("(d) training loop ok; loss", round(losses[0],4), "->", round(losses[-1],4))

# ---------- planted-bug demo: the four classics, verified to actually break ----------
def run(bug=None, epochs=6):
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Dropout(0.5), nn.Linear(32, 2))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2 if bug != "lr" else 5.0)
    lf = nn.CrossEntropyLoss()
    out = []
    for _ in range(epochs):
        m.train(); tot, n = 0.0, 0
        for xb, yb in loader:
            if bug != "zero_grad":
                opt.zero_grad(set_to_none=True)
            loss = lf(m(xb), yb)
            loss.backward(); opt.step()
            tot += loss.item()*len(xb); n += len(xb)
        out.append(tot/n)
    return out

clean = run()
assert run("zero_grad")[-1] > clean[-1], "accumulating grads should hurt"
assert run("lr")[-1] > clean[-1], "lr=5.0 should diverge/stall"
# train/eval mode: dropout active at eval inflates loss and adds variance
m = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Dropout(0.5), nn.Linear(32, 2))
m.train(); a = m(X).std().item()
m.eval();  b = m(X).std().item()
assert abs(a - b) > 1e-3, (a, b)
print("(bug demo) ok; clean", round(clean[-1],4),
      "| no zero_grad", round(run("zero_grad")[-1],4),
      "| lr=5.0", round(run("lr")[-1],4))

# ---------- causal-mask off-by-one, verified ----------
import numpy as np
T = 4
right = np.triu(np.ones((T, T), dtype=bool), k=1)   # correct: strictly above diagonal
wrong = np.triu(np.ones((T, T), dtype=bool), k=0)   # bug: also masks self
assert right[0].sum() == T - 1 and not right[0, 0]
assert wrong[0, 0], "k=0 masks the diagonal -> row 0 has nothing to attend to (NaN)"
s = np.where(wrong, -np.inf, np.zeros((T, T)))
with np.errstate(invalid="ignore"):
    row0 = np.exp(s[0] - np.max(s[0])); row0 = row0 / row0.sum()
assert np.isnan(row0).all(), "all -inf row -> 0/0 -> NaN, the classic symptom"
print("(mask) ok; k=0 gives an all -inf first row -> NaN loss")
