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

Manual reviews are subjective, infrequent, and political. Black-box
ML "project health" tools deliver verdicts that engineering leaders
cannot read off the page - and rightly do not trust to make
million-dollar prioritization calls.

## 4. Our Solution - TruePath AI

**TruePath AI is a rule-based, fully-explainable intent-vs-execution
drift detector for engineering backlogs.**

Given (1) the project's stated intent in plain English and (2) the
current sprint backlog of Jira-style tickets, TruePath AI produces a
**deterministic, auditable verdict** on whether the team's active work
will achieve the stated goal - or silently drift away from it.

The system has three transparent stages:

1. **Intent Analysis** - Parses the project description into structured
   *intent themes*, with an explicit blocklist (`scaling`, `expansion`,
   `maintenance`) so that *how* engineering happens cannot be mistaken
   for *what* the project is trying to achieve.
2. **Ticket Analysis** - Classifies each backlog ticket against a
   small, auditable lexicon of 12 themes (including four FinOps
   sub-themes: `cost_optimization`, `efficiency`, `rightsizing`,
   `cleanup`).
3. **Drift Detection + Explanation** - Compares intent themes against
   ticket themes using two interpretable rules (**proportion** +
   **direction**), then produces an IT / cloud-language report that
   states **why** costs are not reducing despite an active backlog, and
   offers **neutral, decision-supportive** recommendations.

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

The full report is generated in **under a second**, with no API calls,
no model weights, and no training data.

## 6. What Makes TruePath AI Different

- **Zero ML. Zero black box.** Every verdict can be traced to a
  keyword, a threshold, or a rule in plain Python. Judges, engineering
  leaders, and CFOs can read the rules off the page.
- **Deterministic and reproducible.** Same input -> same output, every
  time. Safe to put in front of an executive committee.
- **Domain-aware vocabulary.** The system speaks cloud and FinOps
  natively - *"infrastructure scaling," "cost optimization
  initiatives," "tickets that ADD spend," "decommissioning idle
  resources."*
- **Decision-supportive, not prescriptive.** TruePath AI surfaces
  evidence and options. Humans still make the call - but they make it
  with a clear, evidence-based picture of where the engineering effort
  is actually going.
- **Zero adoption cost.** Works on the same Jira summary text
  engineering already writes. No new tool to deploy, no new field to
  populate.

## 7. Impact

If TruePath AI runs at the start of every sprint planning session, an
engineering organization gets:

- **Early warning** when the next sprint is going to bend cloud spend
  the wrong way - *before* the executive review, not after.
- **Quantified evidence** for hard prioritization conversations
  ("4 of our 8 tickets ADD spend - here's the list").
- A **shared neutral vocabulary** between engineering, FinOps, and
  leadership: *intent themes, drift direction, in-intent / OFF-intent
  tickets, ADDS spend.*
- **No additional tooling overhead, no model retraining, no
  governance overhead.**

## 8. The 60-Second Demo Pitch

> "Our project intent is to reduce AWS cloud cost by 25%. Our sprint
> has 8 tickets. Half of them rightsize and decommission - that's the
> work that actually reduces the AWS bill. The other half scale up
> EKS, add Aurora replicas, expand Kafka - that work *adds* to the AWS
> bill.
>
> TruePath AI flags this in plain English: *'cloud costs will not
> reduce on current sprint backlog - as much or more engineering
> effort is going into adding capacity as into cutting spend.'*
>
> No machine learning. No black box. Every keyword, threshold, and
> rule is readable off the page. The team gets the verdict, the
> evidence, and the decision-support options - and they make the call.
> That's TruePath AI."
