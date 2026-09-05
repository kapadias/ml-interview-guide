# Coding Week — Seven Days

For the ML coding round. It is a gate: at frontier labs a weak ML-coding round
is close to an auto-reject regardless of seniority, because it is read as
"cannot actually do the work," and it carries roughly 25–30% of the technical
decision with the least grade inflation of any round.

Everything here is built on [DL 28] *ML Coding Rounds* — sixteen problems with
reference implementations, traps, and follow-up ladders — and the runnable
`drills/` at the repository root. Every listing in that chapter was executed
before it was printed, and every one has a self-testing version in `drills/`.

**Budget: 2.5–3.5 hours a day, 7 days.** [DL 28] itself prescribes a two-week
ramp; this is the compressed version, and the compression is real. It gets you
one clean pass over every problem plus two days of debug practice, not the three
to five repetitions that make attention motor memory. If you have two weeks,
use the chapter's own cadence in §"A Practice Protocol" instead.

## The rules, every day

From [DL 28] §"A Practice Protocol":

- **Timer on. Plain editor. No autocomplete, no assistant, no reference.** Open
  the reference implementation only after the timer stops.
- **Write your own tests.** The problem is not done when it runs; it is done
  when you have checked it — attention rows summing to one, a causal mask
  leaving position *t* with exactly *t*+1 non-zeros, a gradient matching finite
  differences, loss on a balanced two-class problem starting near ln 2 ≈ 0.693.
- **Narrate.** Out loud, alone, to the wall. The round is graded partly on
  communication, and saying "now I need the head dimension" catches your own
  errors. Silent perfect work scores below narrated imperfect work.
- **Shapes as comments before code.** `Q: (B, H, T, dh)`, `scores: (B, H, T, T)`,
  `out: (B, T, D)`. This is the single highest-return habit in the round.
- **When the timer expires, finish the sentence, not the function.** "I'd
  complete the merge-heads reshape here, then test that rows sum to one" turns
  an unfinished solution into a demonstrated plan.

## The self-grading rubric

Grade against the rubric the interviewer uses, not against "did it eventually
work." [DL 28]'s six checks:

1. **Does it run.**
2. **Are the shapes right** — first time, not after debugging.
3. **Is the softmax numerically stable** — max subtracted before `exp`.
4. **Is the mask applied *before* the softmax**, not after.
5. **Did you test anything without being told to.**
6. **Can you state the complexity and the memory footprint.**

Score all six after every problem, in a log. The pattern is stable across
problems and tells you what to drill — someone who always misses check 5 has a
habit problem, not a knowledge problem, and it is fixable in a day.

Add a seventh line to your log: **minutes to first running version.** The round
rewards steady visible progress; the checkpoint from [DL 29] is a working v1 by
minute 20 of a 60-minute round, and cutting scope aloud if there is none by
minute 25.

## Running the reference implementations

```bash
cd drills
python3 canon_a_attention_optim.py     # numpy only
python3 canon_b_inference_retrieval.py # numpy only
python3 canon_c_search_classical.py    # numpy only
python3 canon_d_torch_debug.py         # needs torch
python3 debug_round.py --list          # the bug taxonomy, symptoms only
```

Each file asserts its own correctness and prints one line per problem. Run them
once now to confirm your environment works — then do not open them again until
a timer has expired. `drills/README.md` documents the first four; the
`debug_round.py` generator is newer than that table.

---

## Day 1 — Attention and the optimizer

**Problem 1 — 45 min.** [DL 28] L5 `***` — "Implement multi-head self-attention
with a causal mask in numpy."
Reference: `drills/canon_a_attention_optim.py`, section (a).
The graded traps: masking after the softmax instead of before; forgetting the
1/sqrt(d_k) scale; a non-stable softmax; losing the head dimension in the
reshape/transpose.

**Problem 2 — 25 min.** [DL 28] L5 `***` — "Implement one AdamW update step in
numpy."
Reference: same file, section (b).
The trap: applying weight decay to the gradient (that is Adam with L2) instead
of decoupled from it (that is AdamW). Know why the distinction exists.

**Then 30 min:** re-implement problem 1 from scratch a second time, timed. The
second run is where the fluency comes from.

**Done when** attention runs in under 25 minutes with a test you wrote, and you
can state its complexity and memory footprint unprompted.

## Day 2 — Backward passes and the training loop

**Problem 3 — 45 min.** [DL 28] L6 `**` — "Implement LayerNorm forward and
backward in numpy."
Reference: `drills/canon_a_attention_optim.py`, section (c) — checked against
finite differences, which is exactly the test you should write yourself.
If you cannot recall the three-term backward, do what the chapter says: write
the forward, name the intermediates, and derive it live while narrating.

**Problem 4 — 35 min.** [DL 28] L5 `***` — "Write a minimal training loop with
gradient clipping in PyTorch."
Reference: `drills/canon_d_torch_debug.py`, `train()`.
Ordering matters and is graded: `zero_grad` → forward → loss → `backward` →
clip → `step`, and `model.train()`/`model.eval()` at the right boundaries.

**Then 30 min:** verify your LayerNorm backward with your own finite-difference
check. If your max gradient error is not around 1e-9, find out why before
moving on.

## Day 3 — Tokenization and sampling

**Problem 5 — 45 min.** [DL 28] L5 `***` — "Implement BPE—train the merges on a
corpus, then encode a word."
Reference: `drills/canon_a_attention_optim.py`, section (f).
Two halves, and candidates routinely finish only one. Budget 25 minutes for
training the merges and 20 for applying them in order.

**Problem 6 — 25 min.** [DL 28] L5 `***` — "Implement sampling from logits with
temperature, top-k, and top-p."
Reference: same file, section (g).
Traps: applying temperature after the softmax; forgetting to renormalize after
truncation; an off-by-one on the nucleus cutoff (the token that crosses *p* is
included).

**Then 40 min:** the follow-up ladder from [DL 28] on both problems — for BPE,
"optimize the merge-finding step when the corpus is large"; for sampling, batch
it.

## Day 4 — Inference

**Problem 7 — 45 min.** [DL 28] L6 `***` — "Implement greedy decoding with a KV
cache."
Reference: `drills/canon_b_inference_retrieval.py`, section (h), which checks
the cached decode against full recomputation — write that equivalence test
yourself before looking.
This problem is the one most likely to appear in a frontier-lab loop, because it
tests whether you understand what the cache actually holds.

**Problem 8 — 35 min.** [DL 28] L6 `**` — "Implement beam search with length
normalization."
Reference: `drills/canon_c_search_classical.py`, section (i).
Traps: comparing raw log-probabilities across different lengths; mishandling
finished beams; forgetting that the length penalty has an exponent you must be
able to justify.

**Then 30 min:** state, out loud, the KV cache's memory cost per token for a
model of your choosing, and connect it to the formula in [DL 19]. The follow-up
after "implement it" is almost always "now tell me what it costs."

## Day 5 — Your track's four problems

Pick one lane. Do not do both; the round will not ask you for both.

### Search / recommendation lane
- **30 min** — [DL 28] L5 `**` — "Build a toy inverted index and score queries with BM25." (`canon_b`, section j)
- **25 min** — [DL 28] L5 `***` — "Implement NDCG@k." (`canon_b`, section k)
- **30 min** — [DL 28] L6 `**` — "Implement the in-batch softmax loss for a two-tower retrieval model." (`canon_b`, section l)
- **35 min** — [DL 28] L6 `**` — "Implement greedy graph search over a fixed HNSW-style neighbor graph." (`canon_b`, section m — checked against brute force)

The NDCG trap is the ideal DCG: it uses the *available* grades sorted
descending, not a hypothetical perfect list. The two-tower trap is forgetting
that the in-batch negatives make this a softmax over the batch, and that
temperature is a real hyperparameter with a defensible value.

### Classical / generalist lane
- **30 min** — [DL 28] L5 `***` — "Implement logistic regression with SGD from scratch, deriving the gradient." (`canon_c`, section n)
- **30 min** — [DL 28] L5 `***` — "Implement k-means with k-means++ initialization." (`canon_c`, section o)
- **40 min** — [DL 28] L6 `**` — "Implement gradient boosting with decision stumps." (`canon_c`, section p)

Derive the logistic gradient out loud before coding it — the interviewer asked
for the derivation in the prompt, and skipping it costs the point even if the
code is right. For k-means, the k-means++ seeding *is* the question; uniform
initialization is the wrong answer.

### If your loop touches vision
- **35 min** — [DL 28] L6 `*` — "Implement 2-D convolution as a matrix multiply (im2col)." (`canon_c`, section e)

**Then 30 min:** whichever lane you did not do, read the reference
implementations for. You will not be asked to write them, but "I've implemented
NDCG" and "I know what NDCG is" sound different.

## Day 6 — The debug round

The format that cannot be prepared for by memorizing implementations. [DL 28]
§"The Debug-the-Training-Code Round" gives the eight-bug taxonomy and the fixed
diagnostic order; `drills/debug_round.py` generates a script with one of the
eight planted and does not tell you which.

**First, 15 min: memorize the diagnostic order.** Not the bug list — the order,
because it is what is graded:

1. **Read the loss curve and classify the symptom.** Flat at ln(n_classes) from
   step one → no signal reaches the parameters. Diverging to inf/nan → learning
   rate, an all-(−∞) softmax row, or log(0). Descending then plateauing early →
   schedule or capacity. Validation *better* than train → dropout still on at
   eval, or leakage.
2. **Overfit a single batch.** Four examples to near-zero loss. Thirty seconds,
   and it halves the search space: if it fails, the bug is in the optimization
   path, not the data.
3. **Check the loop mechanics** — the seven lines around
   `zero_grad`/`backward`/`step`, mode switching, where clipping sits.
4. **Check shapes and alignment** — inputs, labels, logits; the input/label
   offset; the mask's first row.
5. **Check the data path** — shuffling, the split, transforms fit before
   splitting, class balance in a batch.
6. **Check the metric** last. Validation numbers that are too good are usually
   leakage or a metric computed on the training set.

**Then six rounds, 10 minutes each, ~75 min:**

```bash
python3 debug_round.py            # random bug -> broken_train.py
python3 broken_train.py           # read the symptom
# diagnose out loud, in the order above, before touching anything
python3 debug_round.py --reveal broken_train.py
```

Say the symptom-to-cause mapping aloud each time: "loss is flat at 0.693, which
is ln 2, so the model is outputting a constant — that points at the gradient
path rather than the data, so I check `zero_grad` and `requires_grad` before I
look at the loader." The grader scores the diagnostic order, not the reading
speed.

**Then 45 min: corrupt your own code.** Take your day-1 attention and day-2
training loop, plant a bug in each, leave them overnight, and diagnose them
tomorrow. Two bugs worth planting because they are famous and instructive:
`np.triu(..., k=0)` instead of `k=1` for the causal mask — which masks the
diagonal, gives row 0 an all-(−∞) softmax, and produces NaN on the very first
step — and a missing `zero_grad`, which does *not* crash and merely costs you
accuracy, which is exactly why it survives in real codebases for months.

`drills/canon_d_torch_debug.py` demonstrates both quantitatively; read it after
you have practiced, not before.

## Day 7 — Mixed draws under pressure

**90 min — five random problems, 15 minutes each.** Draw five of the sixteen
from a hat. Fifteen minutes is deliberately not enough to finish most of them:
the exercise is to get to a running partial solution with correct shapes and a
stated plan for the rest, which is what a real round's clock actually produces.

**45 min — one full mock with another person.** One problem, 45 minutes, them
watching. If you have nobody, record yourself and watch it back; you will hear
the silences.

**30 min — the two bugs you planted on day 6.**

**15 min — read your rubric log.** Six checks × ~16 problems. Whichever check
you missed most is the single thing to fix before the round. If it is check 5
(testing unprompted), fix it by writing the test *first* on your next three
problems.

---

# The gate

You are ready when, on a problem drawn at random:

- There is a running version with correct shapes inside 20 minutes.
- You wrote at least one real test without deciding to.
- The softmax is stable and the mask precedes it, without you thinking about it.
- You can state complexity and memory when asked, and you volunteer them when
  not.
- You narrated the whole time.

# What this week does not cover

**The general SWE / systems round.** Build an in-memory KV store with
transactions, find the deadlock, profile the pipeline 20× faster, write a
bounded work queue with retries and graceful shutdown. It appears in most
frontier-lab loops, it is weighted comparably to the ML coding round, and
nothing in these four volumes touches it. Notebook-grade code — no functions, no
error paths, global state — is the classic rejection reason for strong modellers
at staff level. Budget separate time: three or four small stateful systems built
end to end with tests, plus a refresh on Python concurrency and profiling.

**AI-assistant rules.** Some rounds now allow an assistant and grade how you
direct and verify it; some forbid it. The policy differs per company *and per
round*. Ask out loud at the start of every coding round. Guessing wrong in
either direction is expensive.
