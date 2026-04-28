# TruePath AI - Hackathon Problem Statement

## 1. The Problem

> Sprints close on time. Tickets ship. Velocity charts look healthy.
> And the AWS bill keeps going up.

Engineering projects begin with a clear intent - *"Reduce AWS cloud cost
by 25%"*, *"Cut release lead time in half"*, *"Eliminate manual
deployments"* - and then the **active backlog quietly drifts away from
it**. Capacity work, refactoring, urgent firefighting, and adjacent
feature requests slip in sprint by sprint. The team is busy, tickets
close on time, dashboards look fine. **And yet, three months later, the
original goal has not been hit.** Leadership only finds out at the
quarterly review - when it is too expensive to course-correct.

This is **silent project drift**, and it is the single most common
reason cloud cost programs, transformation initiatives, and platform
investments under-deliver against the very objectives they were funded
for.

## 2. Why This Matters

- A typical AWS cost-optimization initiative is a **6-12 month,
  multi-million-dollar effort**. If half the sprint goes into adding
  capacity instead of reducing it, the savings target is missed even
  though everyone is "working on cloud."
- Engineering work that *increases* the cloud bill (scaling, expansion,
  new rollouts) often **looks identical** to engineering work that
  *decreases* it (rightsizing, decommissioning, Savings Plans). Both
  touch AWS, both ship into production, both feel productive.
- Existing tooling tracks **velocity, burn-down, and ticket completion**
  - it has no concept of project intent. It cannot tell you that 4 of
  your 8 active tickets are actively pushing the AWS bill *up*.

## 3. The Gap We're Filling

Modern observability stops at *infrastructure*. Modern project
management stops at *throughput*. There is no system that
systematically asks, every sprint:

> **Is the active work actually moving us toward the goal we said
> we would hit?**

Manual reviews are subjective, infrequent, and political. Pure rule-
based dashboards miss tickets that use unfamiliar phrasing, and pure
black-box ML "project health" tools deliver verdicts that engineering
leaders cannot read off the page - and rightly do not trust to make
million-dollar prioritization calls.

**The right answer is a hybrid: AI/ML for the language understanding,
explicit rules for the policy decisions, and a transparent audit trail
for everything in between.**

## 4. Our Solution - TruePath AI

**TruePath AI is an AI-powered, fully-explainable intent-vs-execution
drift detector for engineering backlogs.**

Given (1) the project's stated intent in plain English and (2) the
current sprint backlog of Jira-style tickets, TruePath AI produces an
**auditable verdict** on whether the team's active work will achieve
the stated goal - or silently drift away from it.

The system is a **hybrid AI + rules pipeline**. ML/NLP models do the
heavy lifting where natural language is involved (understanding what
each ticket is *really* about, even when the engineer phrased it in
their own words); deterministic rules and thresholds do the policy
work (deciding when "in between" becomes "off track"). Every model
output is paired with a human-readable evidence trail, so leaders see
both the AI verdict *and* why it landed there.

The system has three transparent stages:

1. **Intent Analysis (NLP)** - Uses NLP (sentence segmentation, goal-
   verb detection, keyword extraction, and embedding-based similarity)
   to parse the project description into structured *intent themes*,
   plus an explicit blocklist (`scaling`, `expansion`, `maintenance`)
   so that *how* engineering happens cannot be mistaken for *what* the
   project is trying to achieve.
2. **Ticket Analysis (ML Classification)** - Classifies each backlog
   ticket against 12 themes (including four FinOps sub-themes:
   `cost_optimization`, `efficiency`, `rightsizing`, `cleanup`) using
   a hybrid model: a lexicon-grounded baseline plus semantic
   similarity (sentence embeddings) so tickets land on the right theme
   even when the engineer phrased it in their own words.
3. **Drift Detection + Explanation (AI Reasoning)** - Compares intent
   themes against ticket themes using two interpretable signals
   (**proportion** + **direction**), then uses an AI explanation layer
   to produce an IT / cloud-language report that states **why** costs
   are not reducing despite an active backlog, and offers **neutral,
   decision-supportive** recommendations.

## 5. Demo Scenario - AWS Cloud Cost Optimization

The bundled demo models a scenario every cloud team will recognize:

| | |
|---|---|
| **Project Intent**     | *"Reduce AWS cloud cost by 25 percent over the next two quarters."* |
| **Sprint Backlog**     | 8 Jira-style tickets - **4 cost-cutting** (rightsize EC2, Savings Plans, decommission idle RDS, delete unattached EBS) and **4 capacity-adding** (scale EKS cluster, add Aurora read replicas, add payments capacity, scale Kafka brokers). |
| **Verdict**            | **OFF TRACK** - cloud costs will not reduce on current sprint backlog. |
| **Evidence**           | *"4/8 tickets (50%) directly target cloud spend. 4/8 tickets (50%) are infrastructure scaling - work that ADDS capacity to the AWS bill instead of reducing it. That is why cloud costs are NOT reducing despite an active sprint backlog: as much or more engineering effort is going into adding capacity as into cutting spend."* |
| **Drift Direction**    | `infrastructure scaling` (clear, single dominant off-intent theme) |
| **Recommendation**     | Decision-support options: prioritize rightsizing & cleanup; deprioritize `infrastructure scaling` unless additional capacity is genuinely required; or, if growth is now the actual goal, explicitly update the project intent and re-baseline the cost target. |

The full report is generated in **under a second**, end-to-end, with
the ML/NLP layer running locally on CPU.

## 6. AI / ML Components

TruePath AI uses ML and AI deliberately - in the places where
*language understanding* matters - and pairs every model output with a
deterministic, auditable rule. Concretely:

- **NLP intent parsing.** Sentence segmentation, goal-verb tagging,
  and keyword extraction extract the project's primary goal,
  supporting goals, and constraints from a free-form paragraph.
- **Sentence-embedding ticket classifier.** Each ticket is encoded
  with a pretrained transformer (e.g. `sentence-transformers`,
  MiniLM-class), then compared against per-theme prototype vectors
  derived from the FinOps + DevOps lexicon. This catches tickets that
  *mean* "rightsizing" or "decommissioning" without using those exact
  words.
- **Hybrid scoring.** Lexicon hits act as an interpretable prior;
  embedding similarity acts as a generalization layer. The two are
  combined into a single per-ticket theme score so the system is
  both **AI-powered** and **debuggable**.
- **AI-generated explanation layer.** The drift verdict, contributor
  list, and recommendations are rendered through a templated AI
  reasoning layer that grounds every sentence in a specific number
  from the rule engine - so the report reads like a senior FinOps
  reviewer wrote it, but every claim can be traced back to a ticket.
- **Auditable, not opaque.** Every model output (theme score, drift
  direction, contributor ranking) ships with the evidence that
  produced it. Engineering leaders see *both* the AI verdict *and*
  why it landed there.

## 7. Tech Stack

| Layer                       | Tech                                                                                   | Role in TruePath AI                                                                               |
|-----------------------------|----------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| **Language & runtime**      | Python 3.9+ (CPU-only, no GPU required)                                                | Hosts the entire pipeline; one `python -m` entry point                                            |
| **Standard library**        | `re`, `json`, `argparse`, `pathlib`, `textwrap`, `typing`                              | Sentence splitting, regex rules, CLI, project file loading, pretty-printing                       |
| **NLP / Intent parsing**    | Rule-based NLP (regex + curated goal-verb / constraint vocabularies); optional spaCy for richer sentence segmentation & POS tagging | Stage 1 - `intent_analyzer.parse_intent`, `extract_intent_themes`                                 |
| **ML ticket classifier**    | `sentence-transformers` (MiniLM-class, e.g. `all-MiniLM-L6-v2`) on PyTorch + Hugging Face Transformers | Stage 2 - encodes each ticket and matches it against per-theme prototype vectors                  |
| **Vector math & similarity**| NumPy, scikit-learn (cosine similarity, optional TF-IDF baseline)                      | Hybrid scoring: embedding similarity blended with the lexicon prior                               |
| **Rule engine**             | Pure-Python thresholds (`aligned_min`, `drift_min`, `direction_threshold`)             | Stage 3 - turns ML scores into the Aligned / Partial / Significant verdict                        |
| **AI explanation layer**    | Templated reasoning grounded in rule-engine numbers (deterministic, reproducible)      | Stage 3 - cloud-language report, contributor reasons, decision-supportive recommendations         |
| **Data format**             | JSON for projects + ticket backlog; plain Python dicts for pipeline outputs            | Lightweight, Jira-friendly input; debuggable intermediate state                                   |
| **CLI / demo**              | `argparse`, ASCII-only formatting (PowerShell- and bash-friendly)                      | `python -m truepath_ai.main [--data path]`                                                        |
| **Testing**                 | pytest                                                                                 | Drift scenarios + cloud-language explainer regression tests                                       |
| **Version control**         | Git + GitHub                                                                           | Reproducible history; PR-able audit trail for every rule and threshold                            |
| **Packaging**               | Plain Python package layout, no build step                                             | Clone & run - the demo works out of the box                                                       |

The stack is **intentionally lightweight**: the rule-based core runs
on the standard library alone, and the ML layer is one small
sentence-embedding model that scores an entire sprint in under a
second on CPU. There is no training pipeline, no model registry, and
no per-customer fine-tuning required to deploy this system inside an
engineering organization.

## 8. What Makes TruePath AI Different

- **AI where language matters, rules where policy matters.** ML
  models classify natural-language tickets; explicit thresholds turn
  those classifications into a verdict. Best of both worlds.
- **Auditable AI.** Every verdict can be traced to a model score, a
  matched keyword, or a threshold in plain Python. No hidden state.
- **Reproducible.** Same input -> same output, every time. Safe to
  put in front of an executive committee.
- **Domain-aware vocabulary.** The system speaks cloud and FinOps
  natively - *"infrastructure scaling," "cost optimization
  initiatives," "tickets that ADD spend," "decommissioning idle
  resources."* - and the embedding layer generalizes that vocabulary
  to new phrasings.
- **Decision-supportive, not prescriptive.** TruePath AI surfaces
  evidence and options. Humans still make the call - but they make it
  with a clear, evidence-based picture of where the engineering effort
  is actually going.
- **Low adoption cost.** Works on the same Jira summary text
  engineering already writes. The ML layer is small, runs on CPU, and
  needs no per-customer training.

## 9. Impact

If TruePath AI runs at the start of every sprint planning session, an
engineering organization gets:

- **Early warning** when the next sprint is going to bend cloud spend
  the wrong way - *before* the executive review, not after.
- **Quantified evidence** for hard prioritization conversations
  ("4 of our 8 tickets ADD spend - here's the list").
- A **shared neutral vocabulary** between engineering, FinOps, and
  leadership: *intent themes, drift direction, in-intent / OFF-intent
  tickets, ADDS spend.*
- **Lightweight ML footprint** - small embedding model, CPU-friendly,
  no per-customer retraining, no governance overhead.

## 10. The 60-Second Demo Pitch

> "Our project intent is to reduce AWS cloud cost by 25%. Our sprint
> has 8 tickets. Half of them rightsize and decommission - that's the
> work that actually reduces the AWS bill. The other half scale up
> EKS, add Aurora replicas, expand Kafka - that work *adds* to the AWS
> bill.
>
> TruePath AI uses NLP and sentence embeddings to read each ticket the
> way a senior FinOps engineer would, then flags this in plain English:
> *'cloud costs will not reduce on current sprint backlog - as much or
> more engineering effort is going into adding capacity as into cutting
> spend.'*
>
> AI does the language understanding; explicit rules turn that into a
> verdict. Every keyword, model score, threshold, and recommendation is
> readable off the page. The team gets the verdict, the evidence, and
> the decision-support options - and they make the call. That's
> TruePath AI."
