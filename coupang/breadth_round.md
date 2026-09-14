# Parth's round: breadth, hiring manager, and team fit at once

This is not three rounds. It is one slot carrying three evaluations, run by the
person who decides. He has **two headcount and four candidates at onsite**, one
requisition at L6-1 and one at L6-2, and the recruiter puts you at the L6-2 end.
So this round decides not only whether you get an offer but which one.

It is also your **second** conversation with him — he screened every candidate
himself before the loop. That changes the opening more than people expect, and
there is a section on it below.

## How it differs from Siwen's round

@decide How do you pace Parth's round versus Siwen's?
- chose: Short answers, commit fast, stop, let him steer
- over: The depth-round instinct of going three levels down
- why: he samples across topics deliberately. The recruiter's words were that Siwen "will focus on one thing and just go dive deep on it," while Parth "tries to poke around different topics" because he wants "360 knowledge on certain ML topics." Going deep unprompted burns his agenda and reads as not listening.
- cost: you will leave depth on the table in places, which feels bad and is correct
- recover it by: offering the depth rather than taking it -- "there's a subtlety in how the negatives are mined, want me to go into it?"
@end

!say Two minutes, then stop. In a breadth round the failure is not being wrong, it is still talking forty seconds after you had already answered.

The rhythm to aim for is: commit in the first sentence, give the one reason that
matters, name the trade-off, stop. If he wants more he will ask, and the asking
is the signal he is looking for anyway.

## Where he will poke

The recruiter named four areas explicitly, and the job description implies the
rest. Have a two-minute answer ready for every cell.

| Area | Why it is on his list | What to be ready for |
|---|---|---|
| **Ranking** | His org's core, and the recruiter named it first | LTR families, position bias, calibration, multi-objective, offline/online gap |
| **Retrieval** | The other half of Search & Discovery | Two-tower training, negatives, ANN, hybrid fusion — the depth-round material, compressed |
| **Computer vision** | Recruiter: "maybe you worked on some kind of computer vision" | Product image embeddings, multimodal retrieval, CLIP-style training, OCR on listings |
| **ML infrastructure** | Recruiter: "how much you know about ML infra" | Spark, Airflow, Kubeflow, MLflow — all named in the JD — feature stores, training pipelines, serving, versioning |
| **LLMs** | The JD asks for innovation with them | Where they belong offline vs online, distillation, cost per query, evaluation |
| **Experimentation** | Any senior ML role | Sizing, interleaving, guardrails, novelty, when a flat result is underpowered |
| **Classical ML** | The JD names XGBoost and LightGBM | GBDT vs neural on tabular, target encoding, leakage, calibration |

Multimodal is the one most candidates have not prepared and the JD asks for
explicitly. You do not need research depth — you need a position: product images
are a strong signal for exactly the attributes that are missing from bad seller
text ("lightweight", "winter"), a CLIP-style dual encoder is the obvious way in,
and the cost is that image embeddings go stale when sellers swap photos. That
is enough to hold a five-minute conversation, which is all this round wants.

## The "I don't know" protocol

A breadth round is designed to find your edge. It will. The only question is
what you do in the two sentences after you hit it.

!say I haven't built that. Here's what I'd expect to be true and how I'd check it — and then say the thing you would actually check.

Three rules. Never bluff a number, because the one thing he can verify on the
spot is arithmetic. Never say "I don't know" and stop, because that is a dead
end and he still has to fill the time. And never over-apologise — one clause,
then reason from what you do know. Getting to a defensible answer from first
principles in front of him is a *better* signal than having memorised it.

## The hiring-manager half

Somewhere in the middle he stops sampling and starts deciding. The questions
shift from "what do you know" to "what would you own." Have these ready.

- **"What would you want to work on here?"** Not a pleasantry — it is a scope
  question. Answer with a specific problem from their stack, not a technology.
  "Tail-query recall" is an answer. "LLMs" is not.
- **"What does your first ninety days look like?"** Say: read the code and the
  logs, find the measurement gap, ship something small that proves the loop
  works end to end, and only then propose the bigger thing. Candidates who
  describe a six-month redesign in their first ninety days read as risky.
- **"How do you decide what not to do?"** This is *Ruthless Prioritization*
  arriving in technical clothing. Have a real example where you killed
  something.
- **"How do you work with a PM who wants something you think is wrong?"**
  *Disagree and Commit*. They want both halves: the disagreement, and the
  committing.
- **"What would you need from me?"** Answer it honestly and specifically.
  Vagueness here reads as not having thought about the job.

## Level calibration: L6-1 versus L6-2

He is deciding which requisition you fit. The difference is not how much you
know; it is the radius of what you talk about owning.

| | Reads as L6-1 | Reads as L6-2 |
|---|---|---|
| **Scope** | "I owned the ranking model" | "I owned ranking quality, which meant fixing the labelling pipeline two teams upstream" |
| **Problem choice** | Given a problem, solved it well | Chose the problem, and can say what you gave up |
| **Failure** | Describes a bug you fixed | Describes a bad call you made and what it cost |
| **Others** | Worked well with the team | Named a person who is better because of you |
| **Numbers** | Model metrics | Business metrics, and the cost side of the ledger |

None of this means inflating. It means that when you have a choice between the
narrow true answer and the wider true answer, take the wider one.

## Team fit runs in both directions

The recruiter's read is that you and Parth are "genuinely a very, very good
match," and that he "has been here a long time." Use that: a long-tenured
manager has seen several strategy cycles and will respond to questions about
durability rather than hype.

What he is listening for: whether you want *this* job or *a* job. Specificity is
the whole tell. Mentioning that recommendations and search together are more
than half of Coupang's sales, or that the ranking objective mixes relevance with
delivery speed, does more than any statement of enthusiasm.

**Questions worth asking him**, in rough priority:

- Which requisition is this, and what does the first year look like differently
  at L6-1 versus L6-2? *Ask it directly. He has two, you may as well know.*
- What is the split between search and recommendations on this team, and where
  is the bigger gap right now?
- How much of the ranking objective is relevance versus attractiveness, and who
  owns that weighting?
- What does the offline-to-online correlation look like today? *A manager who
  can answer this precisely has a healthy measurement culture; one who cannot is
  telling you where your first ninety days go.*
- Who else would I work with most closely, and what do they need that they are
  not getting?
- What is the thing about this team you would change if you could?

Avoid asking about work-life balance, promotion timelines, or whether the team
uses a particular framework. None of them are bad questions; all of them are
better asked of the recruiter.

## The second-conversation problem

He has already spoken with you. So the opening cannot be the same opening.

Do not re-tell the stories he has already heard — he will recognise them, and it
reads as a rehearsed pitch rather than a conversation. Instead, reference the
first conversation and advance it: *"When we spoke you mentioned X. I went and
looked at how we handled that, and the thing I got wrong was…"* That does three
things at once — it shows you listened, it shows you thought about the job
between conversations, and it opens on a self-correction, which is the single
most reliable way to sound senior.

Have one thing prepared that you have learned or reconsidered since you last
spoke. It does not have to be large. It has to be real.

!trap The worst outcome in this round is a good technical performance that leaves him unable to picture you on his team. Coverage gets you past the bar; specificity about what you would own is what picks you out of four candidates for two seats.
