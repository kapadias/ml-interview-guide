# The questions Parth asks

This is the round that decides. `breadth_round.md` is the map of it — how to
pace it, where he pokes, what he is deciding. This document is the worked
version: the sweep answers written out at the length you should actually say
them, the behavioural half answered end to end, and his follow-up tree.

The requisition is known — **Staff Machine Learning Engineer, Search &
Discovery**, req R0073340, Mountain View — so there is nothing to establish
about level and nothing to ask about it. There are two seats and four onsite
candidates, which means the round is a comparison, not a bar check.

> Read the technical half once and the behavioural half three times. The
> technical material here is compressed from `deepdive.md`, which you are
> preparing anyway for Siwen's round. The behavioural half has no other home in
> this corpus except `project.md`, and `project.md` is scored by a different
> interviewer against one project. This round is scored by the hiring manager
> against *you*.

## What is actually being tested

Three things in one slot, and they are weighted in the order you would not
guess.

| | What he is sampling | What a failure looks like |
|---|---|---|
| **Breadth** | Whether you go blind outside your specialism | Fluent on retrieval, hand-waving on experiment design |
| **Hiring manager** | Whether he can picture you owning something on Monday | A technically perfect round with no answer to "what would you work on" |
| **Team fit** | Whether you want *this* job or *a* job | Enthusiasm with no specifics attached to it |

The breadth sweep is the part candidates prepare and the part that separates
least — all four onsite candidates will clear it. The hiring-manager half is
where two of them get chosen.

!trap The failure mode of this round is not being wrong. It is being forgettable. Four candidates will all give competent answers about two-tower retrieval; the one who gets picked is the one who said which problem on this team they would take first and why the other one can wait.

## The shape of the hour

He samples deliberately, so expect a lot of topic changes and do not read them
as rejections. A change of subject thirty seconds after you finish is him
getting what he needed, not him losing interest.

| Minutes | What | How to play it |
|---|---|---|
| 0–5 | The opening, and it is a *second* conversation | Advance the first one, do not replay it |
| 5–30 | The breadth sweep, five to eight topics | Two minutes each. Commit, one reason, trade-off, stop |
| 30–40 | The hiring-manager turn | Scope, first ninety days, prioritisation, conflict |
| 40–50 | Behavioural, against the leadership principles | SBI, one sentence of situation, a number at the end |
| 50–60 | Your questions | The highest-signal five minutes you control |

The seam between minute 30 and minute 31 is the thing to listen for. When the
question stops being "how does X work" and becomes "how would you decide X
here," the round has changed character and your answers should get shorter,
more first-person, and more specific to their stack.

# Part one: the breadth sweep

Seven areas. For each one: the position to commit to in the first sentence, the
one reason that matters, the trade-off, and the depth to *offer* rather than
take. Everything below is written at roughly the length to say it.

## Ranking

!say Ranking here is not a relevance problem. Over half of Coupang's sales come through search and recommendations, and the ranker is where relevance gets traded against what your own postings call product attractiveness — ratings, price, brand, delivery speed. If I designed a pure relevance ranker I would have designed someone else's search engine.

Then the mechanics, briefly: pointwise for calibrated probabilities, pairwise
or listwise when you only need the order, and the reason the choice is forced
here is that multi-objective scoring multiplies probabilities by things with
units — price, margin, delivery time — and a LambdaMART score is not a
probability.

@decide Pointwise or listwise for a multi-objective ranker?
- chose: Pointwise calibrated heads, combined at scoring time
- over: A single listwise model trained directly on the blended objective
- why: the weights between relevance and attractiveness are a product decision that changes quarterly, and calibrated heads let you change them without retraining. A listwise model bakes last quarter's trade-off into the weights.
- cost: you give up the direct optimisation of the ranking metric, and you now own two calibration problems instead of one
- offer depth: the recalibration after negative downsampling, if he wants it
@end

!push "So how do you know the weights are right?" — You do not, offline. You pick them on the efficient frontier of a two-metric plot — relevance proxy against GMV per session — and you settle the point on that frontier with an experiment, not an argument.

## Retrieval

Compressed from `deepdive.md`; the full version is the depth round's material.
Four sentences is the right size here.

Two-tower dense retrieval alongside the lexical index, fused rather than
replaced, because BM25 still wins on exact model numbers and part codes.
Trained contrastively with sampled softmax over in-batch negatives, which
introduces a popularity bias that the **logQ correction** removes by
subtracting the log sampling probability from each logit. Hard negatives mined
from the lexical tail, refreshed periodically, with a similarity ceiling to
filter false negatives — an unlabelled positive used as a negative is worse
than no negative at all. HNSW for the index; int8 quantisation because 10M
items at 256 dimensions is 2.6 GB and fits in RAM on one machine.

!say The unglamorous part is the one I would raise first: a large fraction of the catalogue has never been shown, so it is never sampled as a negative and never appears in a positive pair. The model has no opinion about it at all, and that is invisible in offline recall because the eval set came from the incumbent's own impressions.

## Computer vision and multimodal

The recruiter named this, and the job description asks for multi-modal learning
explicitly. You do not need research depth. You need a position.

!say Product images are the highest-value untapped signal in a third-party marketplace, because the attributes missing from bad seller titles — "lightweight", "winter", the actual colour — are visible in the photograph. A CLIP-style dual encoder over image and title is the obvious way in.

@decide Where does the image signal actually go?
- chose: Image embeddings as ranking features and as a catalogue-enrichment source first
- over: A multimodal retrieval index serving image and text vectors jointly
- why: enrichment gets you attribute coverage on the tail, which is where the recall problem actually is, and it is measurable in weeks. A joint index is a bigger commitment with a worse failure mode — if the shared space is poor you degrade head queries to help the tail.
- cost: you do not get true image-query search, which matters if customers search by photograph
- watch for: embeddings going stale when sellers swap photographs, which they do constantly
@end

Two further things worth having ready: OCR on listing images is not a side quest
in a Korean marketplace, because sellers put specifications and sizing into the
image rather than the text field; and the cost model is entirely dominated by
re-embedding, so you need a change-detection hash before you need a bigger GPU.

## ML infrastructure

The postings name Spark, Airflow, Kubeflow and MLflow. He will ask how much you
know. The answer that lands is not a tool tour — it is naming the failure these
tools exist to prevent.

!say The thing I care about in a training pipeline is point-in-time correctness. Almost every unexplained offline-to-online gap I have chased came down to a feature that was computed after the event it was supposed to predict, or a serving value that did not match the training value. So I would rather log features at serving time and train on the log than recompute them in Spark and hope the two agree.

Then the map, quickly: Spark for the aggregations, Airflow for orchestration and
backfills, Kubeflow for the training graph, MLflow for the registry — and the
point that matters, which is that the model artefact and the feature definitions
have to be versioned *together*, because rolling back a model onto a changed
feature is how a rollback makes an incident worse.

!push "How would you catch training–serving skew before it ships?" — Log a sample of serving feature vectors, replay the same keys through the training path, and alert on distribution divergence per feature rather than on aggregate model output. Aggregate output hides a single broken feature almost perfectly.

## LLMs

The job description asks for innovation with them, which means he wants to know
whether you are credulous.

@decide Where does an LLM belong in a 200 ms search path?
- chose: Offline and near-line, not in the request path
- over: A generative reranker or query rewriter called per query
- why: a request-path call spends your entire latency budget and adds a dependency that fails differently from everything else in the funnel. Offline, the same model is worth far more: attribute extraction to repair seller text, synonym and query-rewrite candidate generation, and relevance judgements as a teacher to distil into a small cross-encoder.
- cost: you do not get true zero-shot handling of a query nobody has ever typed
- exception: head and torso queries, where the rewrite is precomputed and cached, so the online cost is a lookup
@end

!num Cost is the argument that ends this discussion. At 200 QPS, one LLM call per query is roughly 17M calls a day. Even at a tenth of a cent, that is five figures a day for a rewrite you could have cached for the 20% of query traffic that is not tail.

## Experimentation

Any senior ML role tests this, and it is the area where fluency is most
diagnostic of having actually shipped.

Session-level randomisation, not request-level, because a customer who sees two
different rankers in one session gives you a contaminated observation. Size
from the baseline variance of the metric you will actually read, and know the
number before you start. Interleaving when you need ranking sensitivity — it is
one to two orders of magnitude more sensitive than an A/B for ordering changes,
and it cannot measure the things you actually care about, so it is a triage
tool, not a decision tool.

!say A flat result and an underpowered result look identical on a dashboard and mean opposite things. Before I call a result neutral I want the minimum detectable effect the experiment actually had. Most "neutral" ranking experiments I have seen were powered to detect an effect three times larger than any ranking change has ever produced.

!push "What would you have done if a guardrail moved?" — Answer with the rule you set *before* looking, not with a judgement made after. The decision rule written down in advance is the whole signal in this question.

## Classical ML

XGBoost and LightGBM are named in the postings, and the instinct to answer
"everything is a transformer now" fails here.

GBDTs still win on heterogeneous tabular features with a few million rows, and
they win decisively on iteration speed, which is the thing that actually
determines how many ideas you test in a quarter. Neural wins when you need
shared representations — embeddings for items and queries, multi-task heads,
transfer from one surface to another — which in a ranking stack is exactly the
L2, so the honest answer is usually GBDT for L1 and neural for L2.

The two traps to name unprompted: target encoding leaks unless the encoding is
computed out of fold, and negative downsampling shifts your probabilities, so a
downsampled model needs the log-odds correction before anything multiplies its
output by a price.

!num Downsampling negatives by a factor w multiplies the odds by w. To recover a calibrated probability: p = p' / (p' + (1 - p')/w). If you skip it, every expected-value calculation downstream is wrong by a constant factor and the ordering only survives if the weights are uniform.

## The sweep answer that is not about a topic

Somewhere in here he will hit something you have not built. That moment is
worth more than the six answers before it, because it is the only one where he
learns how you think rather than what you have read.

!say I haven't built that. Here is what I would expect to be true and how I would check it in a day — and then name the actual check.

Three rules. Never invent a number, because arithmetic is the one thing he can
verify on the spot. Never stop at "I don't know," because he still has fifty
minutes to fill and a dead end costs him. And never apologise for more than one
clause. Reasoning to a defensible answer from first principles in front of him
is a *better* signal than having memorised it, and both of you know that.

# Part two: the hiring-manager and behavioural half

This is the half that picks two candidates out of four, and it is the half
almost nobody rehearses out loud. Coupang's published guidance asks for **SBI —
Situation, Behaviour, Impact** — and tells candidates to *"incorporate data into
examples"* because *"data drives everything that we do."* A story without a
number does not score. Read that literally: it is a scoring criterion, not
advice.

## SBI at conversational length

The template sounds like a template when the proportions are wrong. Most
candidates spend four sentences on situation and one on impact. Invert it.

**One sentence of situation. Four of behaviour, in the first person. One of
impact, with a number and a unit.**

Weak: *"Our search system had some quality issues, so the team worked on
improving relevance and we saw good results."*

Strong: *"Tail queries were returning nothing useful — 15% of tail sessions
ended without a click. I pulled a hundred of them by hand and found two thirds
were failing at retrieval rather than ranking, which pointed at the lexical
index rather than the model everyone was blaming. I built a dense arm and fused
it with RRF rather than a weighted sum, because the score scales were not
comparable across queries. The hard part was negatives — in-batch alone gave a
model that matched keywords without matching intent, so I mined hard negatives
from the BM25 tail. Tail zero-result rate went from 15% to 4%, and tail-session
CTR rose 8% over three weeks on 5% of traffic with head CTR flat inside the
guardrail."*

That second answer is ninety seconds and it scores *Dive Deep* (a hundred
queries by hand), *Wow the Customer* (framed on what a customer experienced),
*Demand Excellence* (did not accept the first model) and *Deliver Results with
Grit* (shipped, measured, guardrailed) without naming a principle once. Naming
the principle out loud is the tell of a coached candidate; evidencing it is the
point.

!trap Use "I" where it is true and "the team" where it is true, and do not blur them. Forty minutes of "we" makes your scope unassessable, which at Staff is fatal in a different way from being wrong — he cannot argue with it, he just cannot pick you.

## The five he will actually ask

### "What would you want to work on here?"

Not a pleasantry. It is a scope question and it is the single highest-variance
answer in the round.

@decide How do you answer the scope question?
- chose: One named problem on their stack, with the reason it is first
- over: A technology you are excited about, or a menu of three options
- why: naming a problem proves you have modelled their system rather than read their job posting. Naming a technology proves the opposite, and a menu reads as not having decided.
- cost: you might name a problem the team already solved, which is survivable and recoverable
- recover it by: asking what they have already tried, and meaning it
@end

!say Tail-query recall, and specifically the catalogue side of it. In a third-party marketplace the retrieval failure on a query like "durable lightweight winter boots for toddlers" is usually a text problem before it is a model problem — the attributes are not in the title. That is where I would start, because it is measurable in weeks and it makes every downstream model better. Ranking objective work is more interesting and I would want it second, not first.

"Tail-query recall" is an answer. "LLMs" is not. Notice the second half of that
answer does as much work as the first: naming what you would do *second*, and
why, is what makes it a judgement rather than a preference.

### "What do your first ninety days look like?"

Read the code and the logs. Find the measurement gap — ask what the
offline-to-online correlation looks like and expect a fuzzy answer, because it
usually is fuzzy. Ship one small thing end to end to prove the loop works: that
you can get a change from an idea through training, review, experiment and
rollout, and that you know who has to say yes at each step. Only then propose
the bigger thing, with the credibility of having shipped once.

!trap A six-month redesign described as a ninety-day plan reads as risky, not ambitious. The candidates who lose this question are almost always the ones who tried hardest to sound impressive on it.

### "How do you decide what not to do?"

This is *Ruthless Prioritization* in technical clothing. Their own wording is
worth knowing: *"To focus on what we must win, we give up what we want to do."*
The word is *give up*, not *defer*.

So the answer needs a thing you killed, not a thing you postponed. Have one
where you killed something **you** wanted, with the reason, and with what you
learned about the cost of having started it at all. The strongest version of
this story includes the sunk cost honestly: two engineer-months already spent
and stopped anyway.

!push "Who disagreed with that call?" — This is where the answer either becomes real or collapses. Have the name of the position, the strongest version of their argument, and how the decision actually got made.

### "How do you work with a PM who wants something you think is wrong?"

*Disagree and Commit*, and they want both halves. Candidates deliver the
disagreement and forget the commit, or deliver the commit and sound like they
never had a position.

The shape that works: state the disagreement as a prediction with a number
attached and a date. *"I think this will move CVR by less than half a percent
and cost six weeks, and here is the estimate."* Then commit, execute properly,
and go back and check the prediction. The last step is the whole answer — a
disagreement you never scored is an opinion; one you scored, whichever way it
went, is a measurement culture.

!say And I was wrong about one of them. We shipped it, it moved the metric more than I predicted, and the reason was a segment I had not looked at. That changed how I estimate now — I look at the segment breakdown before I give a number.

### "What would you need from me?"

Answer it specifically or it reads as not having thought about the job. Good
answers are concrete and slightly uncomfortable: clarity on who owns the ranking
objective weighting, access to the logs in week one rather than week six, one
named partner on the catalogue side, and an agreed decision rule for the first
experiment before it runs.

## The leadership principles as he will actually raise them

He will not say "tell me about a time you demonstrated Ruthless
Prioritization." He will ask a normal question and score the answer against the
principle. This table is the translation.

| Principle | The question it arrives as | What a Staff answer contains |
|---|---|---|
| **Wow the Customer** | "Why was that the right metric?" | The customer outcome the metric was a proxy for, and where the proxy breaks |
| **Ruthless Prioritization** | "How do you decide what not to do?" | Something you killed that you wanted, with the sunk cost stated |
| **Company-wide Perspective** | "Who consumed your model's output?" | A downstream team, and what you did to protect them from your change |
| **Dive Deep** | "How did you find that bug?" | The specific method, the time it took, and the thing you looked at that nobody else did |
| **Think Systematically** | "How do you stop it happening again?" | The guardrail or test you added, not just the fix |
| **Simplify** | "What did you remove?" | A model, a pipeline, or a feature you deleted, and what it freed |
| **Disagree and Commit** | "Tell me about a disagreement" | The prediction you made, and the fact that you went back and scored it |
| **Hire and Develop the Best** | "Who did you make better?" | A named person, and what they can do now that they could not before |
| **Deliver Results with Grit** | "What happened to it?" | Still running, still measured, with the number today not the launch number |
| **Influence without Authority** | "How did you get that dependency?" | Data you put in front of someone who did not report to you |
| **Aim High and Find a Way** | "What was the goal?" | A target that looked unreasonable when set, and the working-backwards step |
| **Learn Voraciously** | "What were you wrong about?" | A real technical belief you changed, with what changed it |
| **Demand Excellence** | "What did you refuse to ship?" | A bar you held when holding it was expensive |
| **Move with Urgency** | "How fast?" | Something shipped in two weeks that others would have specified for two months |
| **Hate Waste** | "What did it cost?" | A cost halved with quality held flat — *save*, not cut |

You are not trying to hit fifteen. Three stories that between them cover six or
seven, with *Dive Deep*, *Wow the Customer* and *Deliver Results with Grit*
non-negotiable, is a full hand. The principle in that list most often missing
from a Staff candidate's kit is **Hire and Develop the Best**, because it is the
one that cannot be evidenced from your own work.

## The story portfolio

Fill this in once and it becomes what you revise the night before. Four slots.
The constraint is that each one needs a number you can defend and a version
that fits in ninety seconds.

| Slot | What it has to contain | Principles it covers | Your story |
|---|---|---|---|
| **The shipped win** | Business metric, interval, duration, guardrail | Wow the Customer, Deliver Results with Grit | |
| **The debugging story** | A specific method and how long it took | Dive Deep, Think Systematically | |
| **The bad call** | A decision you made that cost something | Learn Voraciously, Disagree and Commit | |
| **The person** | Someone measurably better because of you | Hire and Develop the Best, Demand Excellence | |

The bad-call slot is the one candidates substitute away from — they offer a bug
instead of a decision, or a constraint imposed by someone else. He will notice.
A bug you fixed is *Dive Deep* evidence and you already have that slot filled.
What this slot needs is a call that was yours, that a reasonable engineer would
have made differently, and that cost something you can name.

!trap Do not reuse a story Parth already heard in the screen. He screened you himself, he will recognise it, and a repeated story converts a conversation into a pitch. Know which stories you have already spent.

# The second-conversation opening

He has already spoken with you, so the opening cannot be the same opening. This
is worth rehearsing verbatim, because it is thirty seconds that sets the
register for the whole hour.

Do not re-tell what he has heard. Reference the first conversation and *advance*
it:

!say When we spoke you mentioned the attractiveness side of the ranking objective. I went back and looked at how we handled the equivalent trade-off, and the thing I got wrong was that we set the weighting once and never revisited it — the right answer was a frontier we re-picked each quarter, and we did not find that out until delivery promises changed underneath us.

That does three things simultaneously: it proves you listened, it proves you
thought about the job between conversations, and it opens on a self-correction,
which is the most reliable way to sound senior that exists. Have one prepared.
It does not have to be large. It has to be real, and it has to have a
conclusion.

# His follow-up tree

Where he goes when he decides to push one level rather than change topic. These
are the second questions, and having an answer to the second question is most of
what distinguishes the four candidates.

## On the technical sweep

- "You said fuse rather than replace — how do you fuse?" → RRF over ranks, not a
  weighted sum of scores, because dense cosine and BM25 are not on comparable
  scales across queries. Weighted sums need per-query normalisation you cannot
  estimate at serving time.
- "What breaks first when you scale the catalogue ten times?" → Not the index.
  The negative distribution, and the fraction of items with no engagement at
  all, which is where retrieval quality actually degrades.
- "How would you know an LLM rewrite helped?" → On the subset it fires on, not
  overall. A rewrite that fires on 4% of queries cannot move a site-wide metric
  detectably, so the experiment has to be triggered-traffic only or it will read
  flat for the wrong reason.
- "Where would you put a cross-encoder?" → Over the top 50 after L2, 10–15 ms on
  a T4, and only once the offline-to-online correlation is good enough to trust
  a small win.

## On the hiring-manager answers

- "Why not the ranking objective first?" → Because catalogue text is upstream of
  every model in the funnel, so fixing it compounds, and because it is
  measurable in weeks rather than a quarter. Also because the objective
  weighting is a decision I would not want to make in month one without knowing
  who owns it today.
- "What if the measurement gap turns out to be fine?" → Then the ninety days
  gets shorter and I take the bigger problem sooner. That is a good outcome and
  I would say so.
- "What would make you leave in a year?" → Answer honestly. The honest answer —
  no line of sight from the work to a customer outcome, or an inability to
  measure — is also the answer that says you are the kind of engineer they want.
- "How do you feel about the delivery-speed side of ranking?" → It is the part
  of the problem that is actually theirs. Anyone can improve relevance; the
  interesting work is the trade-off against fulfilment, and it is the reason
  this job is different from the same job at a pure marketplace.

# Ten things that would lose this round

1. Going three levels deep unprompted, because that is Siwen's round and this is
   not.
2. Answering the scope question with a technology instead of a problem.
3. A behavioural story with no number in it.
4. Forty minutes of "we".
5. Re-telling a story he already heard in the screen.
6. A ninety-day plan that is really a six-month redesign.
7. Inventing a number he can check.
8. A disagreement story with no commit, or a commit story with no disagreement.
9. Having nothing for *Hire and Develop the Best*.
10. Asking about levelling. It is settled, it is a Staff req, and asking reads as
    the only thing you were really interested in.
