# The project deep-dive round

Nominally a leadership round. In practice: you pick one project, talk for ten
minutes, and then spend thirty-five minutes being pushed on it. It is scored
against the 15 leadership principles, but the evidence is technical — the
interviewer is checking whether the detail holds up four levels down. Coupang's
own guidance names **SBI (Situation, Behaviour, Impact)** and tells candidates
to "incorporate data into examples" because "data drives everything that we do."

> **Send me your project document and I will fill this in properly** — the
> narrative, the numbers, the specific follow-ups your architecture invites, and
> where each leadership principle is evidenced. Everything below is the
> structure it will be poured into, and is worth reading first so you know what
> I will be asking you for.

## Choosing which project

Pick for **depth of decision-making**, not for glamour. The best project for
this round has: a decision you made that a reasonable engineer would have made
differently, a number attached to the outcome, something that went wrong, and at
least one other team involved. A flawless project is a bad interview project —
there is nothing to dive into.

Rule of thumb: if you cannot name three things you would do differently, pick a
different project.

A project that touches ranking, retrieval, recommendations, or large-scale
feature pipelines will land harder here than one that does not, because the
interviewer can go deeper with you. But a deeply-understood project outside the
domain beats a shallowly-understood one inside it.

## The narrative spine — ten beats, ten minutes

Most candidates deliver beats 1, 5 and 9 and wonder why the round felt thin.
The ones that score deliver 3, 4, 7 and 10.

1. **The business problem, in one sentence a PM would recognise.** Not "we
   improved NDCG." Something like "tail queries returned nothing useful and 15%
   of sessions with a tail query ended without a click."
2. **Why it was worth doing.** The size of the prize, estimated before you
   started. If you cannot say how you sized it, say that — but know that this is
   where *Wow the Customer* and *Ruthless Prioritization* get scored.
3. **What you inherited.** The existing system, honestly. Its actual failure
   mode, not a strawman.
4. **The decision point.** The two or three real options, and why you chose the
   one you chose. **This is the centre of the round.** Have the rejected option
   ready in as much detail as the chosen one.
5. **What you built.** Architecture at the level of a whiteboard box diagram,
   then one level down on the part that was hard.
6. **What broke.** Something always breaks. The specific bug, how you found it,
   how long it took. *Dive Deep* lives here.
7. **How you knew it worked.** The experiment design, the metric, the guardrail
   metrics, the duration, the decision rule you set *before* you looked.
8. **The result, with a number and a unit.** And the confidence interval or the
   sample size, because a Staff candidate who quotes a point estimate without
   either invites an easy follow-up.
9. **What it cost.** Serving cost, latency, engineer-months, complexity added.
   *Hate Waste* is a principle here — "we save costs rather than simply cut
   them."
10. **What you would do differently, and what you learned.** Non-defensively.
    *Learn Voraciously* says "Ego is the enemy," and they mean it as a scoring
    criterion.

## The numbers you must know cold

Being unable to produce these is the single most common way this round goes
badly. Write them on one page and memorise it. If a number is an estimate, know
that it is, and say so.

| | What to have ready |
|---|---|
| **Scale** | Rows of training data. Number of users / items / queries. QPS at peak. Index size. |
| **Model** | Parameter count or tree count and depth. Feature count. Training time and on what hardware. |
| **Latency** | p50 and p99, and the budget you were given. Where the time went, by stage. |
| **Cost** | Serving cost per million requests, or GPU-hours per training run. Before and after. |
| **Offline result** | The metric, the delta, and the size of the eval set. |
| **Online result** | The primary metric, the delta, the confidence interval, the experiment duration, the traffic allocation. |
| **Guardrails** | What you watched that was *not* allowed to regress, and whether it did. |
| **Team** | How many engineers, over how long, and which parts were yours specifically. |

The last row matters more than people expect. At L6 they are calibrating your
scope. Be precise about what you personally did versus what the team did —
overclaiming is the fastest way to lose the round, and under-claiming loses it
more slowly.

## The interrogation bank

These are the follow-ups. Rehearse against them out loud; most have no single
right answer, and the failure mode is hesitation rather than being wrong.

### Problem framing
- Why was this the right problem to work on? What did you *not* do because of it?
- Who decided this mattered — you or someone else? What would you have picked instead?
- How did you size the opportunity before committing? What was the estimate, and how close was it?
- What would have happened if you had done nothing for six months?
- Who was the customer, and how do you know this helped them rather than just moving your metric?

### Data
- Where did the training labels come from? What is the bias in that source?
- How did you construct negatives? What happens if you sample them differently?
- What was the train/validation split, and why is it not leaking?
- How stale was the freshest feature at serving time? How did you find out?
- What fraction of your data was junk, and how did you detect it?
- Did the label distribution shift over the period you trained on?

### The modelling choice
- Why this architecture and not the obvious simpler thing? What would the simpler thing have scored?
- What was your baseline, and is it a strong baseline or a convenient one?
- What did you try that did not work? Why did it not work?
- How did you pick the hyperparameters? How sensitive was the result to them?
- If you had ten times the data, would you have made the same choice? A tenth?
- What is the failure mode of this model that you have not fixed?

### Evaluation
- What was the offline metric, and why that one? What does it fail to capture?
- Did offline and online agree? If they did, why do you trust that they will keep agreeing?
- How long did the experiment run, and why that long? Who chose the sample size?
- What were the guardrail metrics? Did any move? What would you have done if one had?
- Was there a novelty effect? How did you rule it out?
- Could this have been a false positive? What is your multiple-comparisons story?

### Production
- What broke in production that you did not see in offline evaluation?
- How did you roll it out? What was the rollback plan and did you ever use it?
- What monitoring did you add? What alert would have caught the bug earlier?
- What is the on-call burden of what you built?
- What happens to this system if you leave tomorrow?
- What did it cost to serve, and what did you do about that?

### Impact and hindsight
- Is it still running? Has the effect held up?
- What is the second-order effect — did another team's metric move because of you?
- What would you do differently with what you know now?
- What did you get wrong?
- If you had to cut this project in half, which half?

### Collaboration and conflict
- Who disagreed with you, and what was their argument? Steelman it.
- What did you do when you could not get a dependency prioritised?
- Tell me about a time you were overruled on this project.
- Who did you make better, and how do you know?
- How did you explain this to someone non-technical, and what did they take away?

## Mapping your story to the principles

Fill this in once and it becomes the thing you revise the night before. You are
not trying to hit all 15 — you are trying to have two or three stories that
between them give clean evidence for six or seven, with *Dive Deep*,
*Wow the Customer* and *Deliver Results with Grit* non-negotiable.

| Principle | Your evidence | Number attached |
|---|---|---|
| Wow the Customer | | |
| Ruthless Prioritization | | |
| Company-wide Perspective | | |
| Dive Deep | | |
| Think Systematically | | |
| Simplify | | |
| Disagree and Commit | | |
| Hire and Develop the Best | | |
| Deliver Results with Grit | | |
| Influence without Authority | | |
| Aim High and Find a Way | | |
| Learn Voraciously | | |
| Demand Excellence | | |
| Move with Urgency | | |
| Hate Waste | | |

## SBI, without sounding like a template

Coupang asks for Situation / Behaviour / Impact. Used mechanically it sounds
rehearsed. The fix is proportion: **one sentence of situation, four of
behaviour, one of impact with a number.** Most candidates invert this and spend
four sentences on context.

Weak: *"Our search system had some quality issues, so the team worked on
improving relevance, and we saw good results."*

Strong: *"Tail queries were returning nothing useful — 15% of tail sessions
ended without a click. I pulled a hundred of them and found two thirds were
failing at retrieval rather than ranking, which pointed at the lexical index
rather than the model everyone was blaming. I built a dense retrieval arm and
fused it with RRF rather than a weighted sum, because the score scales were not
comparable across queries. The hard part was negatives: in-batch negatives alone
gave us a model that matched keywords without matching intent, so I mined hard
negatives from the BM25 tail. Zero-result rate on tail queries went from 15% to
4%, and tail-session CTR rose 8% over a three-week A/B on 5% of traffic, with
head-query CTR flat inside the guardrail."*

The second one is scoring *Dive Deep* (pulled a hundred queries), *Wow the
Customer* (framed on customer outcome), *Demand Excellence* (did not accept the
first model), and *Deliver Results with Grit* (shipped and measured) without
ever naming a principle.

## Anti-patterns

- **The metric with no customer.** "We improved NDCG by 4 points" and nothing about what a user experienced.
- **The point estimate.** A lift with no interval, no duration, no traffic share.
- **The flawless project.** Nothing went wrong, nothing was cut, no one disagreed. Reads as either shallow involvement or a rehearsed story.
- **"We".** Forty minutes of "we" with no "I" makes your scope unassessable. Use "the team" where it is true and "I" where it is true.
- **Defending the rejected alternative badly.** If you cannot argue the option you did not take, the interviewer concludes you never seriously considered it.
- **Running out of depth.** Every level you can go down is a level of credibility. The round is designed to find your floor; it should be low.
