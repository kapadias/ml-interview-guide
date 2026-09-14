# The loop

Four rounds: **ML depth**, **ML breadth**, a **leadership round that is really a
technical project deep-dive**, and **DS&A coding**. Staff / L6.

## What the org actually is

Coupang runs Search & Discovery as a revenue centre, not a platform team. Their
own job postings say it plainly: *"Over half of the Coupang sales are directly
attributed to search and recommendation."* That single sentence should shape how
you talk in all four rounds — every ranking decision you describe is a decision
about half the company's revenue, and you should frame trade-offs in those terms
rather than in NDCG points alone.

The published architecture is a conventional three-stage funnel, and knowing
their vocabulary for it costs you nothing:

1. **Query understanding.** Normalisation, segmentation, intent, spell
   correction, synonyms. The org explicitly hires for "query/intent
   understanding, query recommendation and search widgets quality."
2. **Retrieval.** A few thousand products selected out of tens of millions.
3. **Ranking.** The few thousand ordered by a model over **100+ criteria** that
   mix *search relevance* with what they call *product attractiveness* —
   ratings, review counts, price, brand, and delivery speed.

That second half of the ranking objective is the thing to internalise. At
Coupang, ranking is explicitly not a pure relevance problem. A relevant product
that cannot be Rocket-delivered tomorrow is worth less to the customer than a
slightly less relevant one that can, and the ranker is where that judgement gets
made. If you walk in and design a pure relevance system, you have designed
someone else's search engine.

Two more facts worth having: the catalogue is millions of products against
petabytes of behavioural data, and the index is **multilingual — Korean,
English, and Mandarin**. Korean is agglutinative and written without spaces
between all morphemes, so tokenisation is a real modelling decision rather than
a library call. If you are asked about internationalisation, that is the
specific thing to reach for.

The job description names the tools: Spark, Airflow, Kubeflow, MLflow for
pipelines; PyTorch/TensorFlow, XGBoost and LightGBM for models; AWS, GCP,
Vertex AI, BigQuery, SageMaker for infrastructure. It also names the three
directions they want pushed — **LLMs, embedding-based retrieval, and
multi-modal learning**. Have an opinion on each, and specifically have an
opinion on where an LLM does *not* belong in a 200 ms search path.

## What each round is testing at L6

**ML depth.** Not "do you know retrieval" — every candidate knows retrieval.
The L6 bar is whether you can hold a whole stage of the funnel in your head and
defend a choice against the alternative you rejected. Expect to be pushed one
level past your first answer every time. The tell that you are at the bar:
you volunteer the failure mode of your own proposal before they ask.

**ML breadth.** Coverage without hand-waving. They are checking that you are
not a one-system specialist who would be useless when the problem turns out to
be a data leak, a calibration bug, or an experiment design flaw. Answers here
should be shorter than in depth — two minutes, commit to a position, stop.
Rambling in breadth reads as uncertainty.

**Leadership / project deep-dive.** This is the round people underestimate. It
is scored against the 15 leadership principles below, but the *evidence* is
technical: they will go three or four levels into one project and see whether
the detail holds up. Coupang's own interview guidance says to prepare
"operation principles, reasons behind choosing a strategy as well as the
trade-offs," and to use **SBI — Situation, Behaviour, Impact**. Their guidance
also says to "incorporate data into examples" because "data drives everything
that we do." A story without numbers will not score.

**Coding.** Their published guidance is 2–3 problems, and what they say they
grade is revealing: *"Clean code, Bug free"*, *"Code validation, identification
of edge cases"*, and algorithm mastery in *"sorting, recursion, and dynamic
programming."* Note what is missing — cleverness. Optimal-but-broken loses to
correct-and-tested. State your complexity, then walk one example through your
own code out loud before you say you are done.

## The 15 leadership principles

These are Coupang's, verbatim, and the deep-dive round is graded against them.
The right-hand column is where each one most naturally shows up — you are not
expected to hit all 15, you are expected to have two or three stories that hit
six or seven of them between them.

| Principle | Coupang's words | Where it shows up |
|---|---|---|
| **Wow the Customer** | "We exist to transform customers' lives for the better. The customer is the beginning and the end in every decision we make." | Why the metric you optimised was the right proxy for the customer |
| **Ruthless Prioritization** | "To focus on what we must win, we give up what we want to do. Laser focus requires courage and confidence." | What you cut from the project, and what it cost you |
| **Company-wide Perspective** | "Leaders think like owners… We understand and consider upstream and downstream implications." | Who consumed your model's output, and what you did to protect them |
| **Dive Deep** | "Operational excellence requires hands-on leadership with a passion for detail. We dig down to the smallest details." | The debugging story. This is the one the depth round is secretly testing |
| **Think Systematically** | "We build scalable processes with prompt feedback mechanisms. We take measures not only to fix defects, but also to prevent them." | The guardrail you added after the incident, not just the fix |
| **Simplify** | "Complexity is the enemy of scale, speed, and customer experience." | The model or pipeline you deleted |
| **Disagree and Commit** | "Constructive confrontation is essential to good decisions. Leaders challenge openly when they disagree." | The time you lost the argument and executed anyway — they want both halves |
| **Hire and Develop the Best** | "Leaders raise the performance level of the team with every hire and promotion." | Mentoring, and a specific person who got better |
| **Deliver Results with Grit** | "Leaders deliver impact in timely fashion and with high quality. We do not confuse long hours or effort with results." | Shipped, measured, still running |
| **Influence without Authority** | "Leaders lead by communicating their ideas clearly. We drive alignment through data and insights." | How you got another team to change their schema for you |
| **Aim High and Find a Way** | "We aim for jaw-dropping results and work backwards." | The goal that looked unreasonable when you set it |
| **Learn Voraciously** | "We are hungry for the best ideas and seek them from all sources… Ego is the enemy." | What you were wrong about |
| **Demand Excellence** | "Leaders have a passion for excellence and do not tolerate mediocrity." | The bar you refused to lower |
| **Move with Urgency** | "Urgency is a sense of crisis. We treat inaction as a threat to survival. We learn by doing." | The thing you shipped in two weeks instead of specifying for two months |
| **Hate Waste** | "Leaders find ways to do more with less. We save costs rather than simply cut them." | The serving-cost or GPU-hour number. Note the distinction they draw: *save*, not *cut* |

Principle 15's phrasing is worth a beat. "We save costs rather than simply cut
them" means an efficiency story where you removed capability scores badly; one
where you held quality flat and halved the cost scores well.

## Budgeting the time inside a round

A 45-minute technical round is roughly three questions. Candidates lose it by
spending 20 minutes on the first. Aim for two to three minutes of answer, then
stop and let them steer. If you genuinely need longer — a system design —
say the shape first ("three stages, and the interesting one is the second"),
then expand. Announcing the structure buys you the time; launching into detail
without it does not.

For the deep-dive round, assume 10 minutes of narrative and 35 minutes of
interrogation. Most people prepare the 10 and not the 35.
