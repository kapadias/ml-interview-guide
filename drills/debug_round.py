#!/usr/bin/env python3
"""Debug-round practice: generate a training script with one planted bug.

Volume I, Chapter 28 describes the debug-the-training-code round and its bug
taxonomy. Reading a taxonomy is not practice. This hands you a script that
trains badly and does not tell you why.

    python3 debug_round.py                 # random bug -> broken_train.py
    python3 debug_round.py --bug 3         # a specific bug
    python3 debug_round.py --list          # symptoms only, no answers
    python3 debug_round.py --reveal FILE   # check your diagnosis afterwards

Work Chapter 28's diagnostic order before revealing: read the loss curve and
classify the symptom, try to overfit a single batch, check the loop mechanics,
check shapes and label alignment, check the data path, check the metric. Ten
minutes.

Every symptom quoted below was measured by running the variant, not assumed.
A correct run of the reference ends near: train loss 0.19, train acc 0.98,
val acc 0.80.
"""
import argparse, base64, random, re, sys

REFERENCE = '''"""Tabular classifier. A correct run reaches ~0.80 val accuracy."""
import torch, torch.nn as nn

torch.manual_seed(0)
N, D = 700, 40
X = torch.randn(N, D)
signal = X[:, 0] * X[:, 1] + 0.8 * X[:, 2] - 0.6 * X[:, 3]
y = (signal + 0.35 * torch.randn(N) > 0).long()

# hold out the last 200 rows, then sort the training set by label
Xtr, ytr, Xva, yva = X[:500], y[:500], X[500:], y[500:]
order = torch.argsort(ytr)
Xtr, ytr = Xtr[order], ytr[order]

mu, sd = Xtr.mean(0), Xtr.std(0)
Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd

train_dl = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(Xtr, ytr), batch_size=64, shuffle=True)

model = nn.Sequential(nn.Linear(D, 256), nn.ReLU(), nn.Dropout(0.5),
                      nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.5),
                      nn.Linear(256, 2))
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
lossf = nn.CrossEntropyLoss()

for epoch in range(25):
    model.train()
    total, n = 0.0, 0
    for xb, yb in train_dl:
        opt.zero_grad(set_to_none=True)
        loss = lossf(model(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        total += loss.item() * len(xb); n += len(xb)
    model.eval()
    with torch.no_grad():
        tr_acc = (model(Xtr).argmax(1) == ytr).float().mean().item()
        acc = (model(Xva).argmax(1) == yva).float().mean().item()
    if epoch % 8 == 0 or epoch == 24:
        print(f"epoch {epoch:2d}  train loss {total/n:.4f}  "
              f"train acc {tr_acc:.3f}  val acc {acc:.3f}")
'''

# (needle, replacement, name, measured symptom, explanation)
BUGS = [
    ("        opt.zero_grad(set_to_none=True)\n", "",
     "missing zero_grad",
     "Underfits: train acc stalls near 0.89 instead of 0.98, val acc 0.745 vs 0.800.",
     "PyTorch accumulates gradients, so each step uses the running sum of every\n"
     "previous batch. The direction is stale and the effective step size drifts.\n"
     "Note how undramatic it is -- no crash, no NaN, just a model that is quietly\n"
     "worse. That is why this bug survives in real codebases for months.\n\n"
     "Worth knowing: with a tight clip_grad_norm (1.0 rather than this script's\n"
     "10.0) clipping renormalizes the inflated gradient and hides this bug almost\n"
     "completely. Gradient clipping masks exactly the bugs that corrupt gradient\n"
     "magnitude."),

    ("    model.train()\n", "    model.eval()\n",
     "eval mode during training",
     "Train loss collapses to ~0.003, far below a correct run's 0.19, while val "
     "accuracy drops to ~0.785.",
     "model.eval() disables dropout and switches BatchNorm to running statistics.\n"
     "Training in eval mode silently removes your regularization: the model\n"
     "memorizes the training set, so train loss looks spectacular while validation\n"
     "gets worse. A train loss far below what the task's noise floor allows is the\n"
     "tell -- with 35% label noise, near-zero train loss is not possible honestly."),

    ("lr=1e-3", "lr=5.0",
     "learning rate far too high",
     "Loss explodes to ~2.6 million in the first epoch, then sits at chance "
     "(val 0.500) and never recovers.",
     "Adam's update is scale-invariant, so a bad learning rate is catastrophic\n"
     "regardless of the loss magnitude. Divergence inside one epoch is the\n"
     "signature. The habit that catches it: check that the first step moves\n"
     "parameters by roughly the learning rate."),

    ("lossf = nn.CrossEntropyLoss()", "lossf = nn.CrossEntropyLoss(reduction='sum')",
     "wrong loss reduction",
     "Reported loss is ~12.5 instead of ~0.19 -- about 64x, the batch size -- yet "
     "accuracy still ends up fine.",
     "reduction='sum' scales both the loss and its gradient by the batch size, so\n"
     "the printed numbers are meaningless and the effective learning rate is 64x\n"
     "what you configured. Here clipping absorbs most of the gradient damage, which\n"
     "is why accuracy survives -- so the only visible symptom is a loss whose\n"
     "magnitude makes no sense. Trust that instinct: cross-entropy on two balanced\n"
     "classes starts near ln(2) = 0.693, and a starting loss far from it means the\n"
     "reduction, the label encoding, or the class count is wrong.\n\n"
     "The same bug in gradient accumulation, where nothing clips, silently\n"
     "reweights every micro-batch by its token count."),

    ("y = (signal + 0.35 * torch.randn(N) > 0).long()",
     "y = (signal + 0.35 * torch.randn(N) > 0).long().roll(1)",
     "labels misaligned by one",
     "Val accuracy pinned near chance (~0.53) even though train accuracy climbs to 1.0.",
     "The label vector is shifted relative to its rows, so there is no learnable\n"
     "relationship -- the model memorizes noise, which is why train accuracy still\n"
     "rises while validation never moves. High train accuracy with chance validation\n"
     "on a task you believe is learnable means the pairing is broken, not the model.\n"
     "Always spot-check that row i's label belongs to row i."),

    ("        loss.backward()",
     "        loss.detach().requires_grad_(True).backward()",
     "gradient path detached",
     "Nothing learns: loss flat at ~0.707 from the first epoch, train accuracy ~0.48.",
     "backward() on a detached tensor produces no gradient for any parameter, so the\n"
     "model never leaves its initialization. Flat-from-step-one is the cleanest\n"
     "symptom in the taxonomy -- the signal is not reaching the parameters at all.\n"
     "Overfitting a single batch fails instantly, which localizes it in seconds."),
]

# Bugs this harness deliberately does not plant, because on this data they produce
# no measurable symptom -- which is the lesson.
SILENT = """Two bugs from Chapter 28's taxonomy are NOT in this set, because on this
data they produce no measurable symptom at all:

  * Fitting the scaler on train+val before splitting. Measured: val accuracy
    0.800 either way, identical to three decimal places.
  * Leaving the loader unshuffled over label-sorted data. Measured: 0.795 versus
    0.800 -- inside run-to-run noise.

That is not a flaw in the exercise, it is the point. Leakage and correlated
batches do not announce themselves in a training curve; they cost you a fraction
of a point offline and then a launch in production. You cannot debug them by
staring at loss -- you catch them by auditing the data path, which is why
Chapter 28 puts "check the data path" in the diagnostic order even when the curve
looks healthy, and why [CML 6] leads with Twyman's law.
"""


def build(idx):
    needle, repl, *_ = BUGS[idx]
    if needle not in REFERENCE:
        sys.exit(f"internal error: bug {idx + 1} no longer applies to the reference")
    return REFERENCE.replace(needle, repl, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bug", type=int, help=f"which bug to plant (1-{len(BUGS)})")
    ap.add_argument("--out", default="broken_train.py", help="output path")
    ap.add_argument("--list", action="store_true", help="symptoms only, no answers")
    ap.add_argument("--reveal", metavar="FILE", help="reveal the bug planted in FILE")
    ap.add_argument("--clean", action="store_true", help="write the correct reference instead")
    args = ap.parse_args()

    if args.list:
        print("Measured symptoms (a correct run: train loss 0.19, train acc 0.98, "
              "val acc 0.80):\n")
        for i, (_, _, _, symptom, _) in enumerate(BUGS, 1):
            print(f"  {i}. {symptom}")
        print("\n" + SILENT)
        return

    if args.reveal:
        src = open(args.reveal).read()
        m = re.search(r"# ANSWER:([A-Za-z0-9+/=]+)", src)
        if not m:
            sys.exit(f"{args.reveal} has no answer marker -- was it made by this script?")
        idx = int(base64.b64decode(m.group(1)).decode())
        _, _, name, symptom, why = BUGS[idx]
        print(f"Bug {idx + 1}: {name}\n\nMeasured symptom: {symptom}\n\n{why}")
        return

    if args.clean:
        open(args.out, "w").write(REFERENCE)
        print(f"Wrote the correct reference to {args.out}.")
        return

    idx = (args.bug - 1) if args.bug else random.randrange(len(BUGS))
    if not 0 <= idx < len(BUGS):
        sys.exit(f"--bug must be between 1 and {len(BUGS)}")

    marker = base64.b64encode(str(idx).encode()).decode()
    open(args.out, "w").write(build(idx) + f"\n# ANSWER:{marker}\n")
    print(f"Wrote {args.out} with one planted bug.\n"
          f"Run it, classify the symptom, then diagnose before revealing:\n"
          f"    python3 {args.out}\n"
          f"    python3 {sys.argv[0]} --reveal {args.out}\n"
          f"Compare against the correct run with --clean if you need a baseline.")


if __name__ == "__main__":
    main()
