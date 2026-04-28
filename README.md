# TruePath AI

**Sprints close on time. Tickets ship. Velocity charts look healthy.
And the AWS bill keeps going up.** That is *silent project drift* -
the single most common reason cloud cost programs and platform
investments under-deliver against the goals they were funded for.
Engineering work that *increases* the AWS bill (scaling, expansion,
new rollouts) often looks identical to engineering work that
*decreases* it (rightsizing, decommissioning, Savings Plans). Existing
tooling tracks velocity, burn-down, and ticket completion - it has no
concept of project intent, and no way to flag that 4 of your 8 active
tickets are pushing the bill *up*.

**TruePath AI is a rule-based, fully-explainable drift detector.**
Given a project's stated intent in plain English (*"Reduce AWS cloud
cost by 25%"*) and the current sprint backlog of Jira-style tickets,
it produces a **deterministic, auditable verdict** on whether the
active work will hit the goal - or quietly drift away from it. **Zero
ML. Zero black box.** Every keyword, threshold, and rule is readable
off the page. The bundled demo flips the verdict to **OFF TRACK** when
half the sprint goes into capacity-adding work, and explains *why* in
cloud language: *"cloud costs are NOT reducing despite an active
sprint backlog: as much or more engineering effort is going into
adding capacity as into cutting spend."* See
[`PROBLEM_STATEMENT.md`](PROBLEM_STATEMENT.md) for the full pitch.

## Project layout

```
hackathon/
├── data/
│   └── sample_project.json     # demo project + tickets (drift baked in)
├── truepath_ai/
│   ├── __init__.py             # the public API
│   ├── intent_analyzer.py      # parse_intent(text)        -> structured intent
│   ├── task_analyzer.py        # classify_tasks(tickets)   -> labels + themes
│   ├── drift_detector.py       # detect_intent_drift(...)  -> verdict
│   ├── explainer.py            # explain_drift(...)        -> text report
│   └── main.py                 # the demo entry point
├── requirements.txt
└── README.md
```

Five small modules, one for each stage of the pipeline. The compute
modules return plain dicts; the explainer turns those dicts into a
demo-ready text block.

## Run the demo

```bash
python -m truepath_ai.main
```

You'll see three labelled stages:

```
[1/3] Intent Analysis        - primary goal, supporting goals, constraints
[2/3] Ticket Analysis        - per-ticket label, dominant themes, matches
[3/3] Drift Detection        - verdict, theme distribution, top contributors
```

Use a different project file with `--data`:

```bash
python -m truepath_ai.main --data path/to/your_project.json
```

## How it works (judge cheat-sheet)

1. **Intent analysis** - `parse_intent` splits the description into
   sentences, picks the highest-scoring "goal" sentence as the primary,
   and tags the rest as goals or constraints (e.g. "40 percent" or "by
   Friday" makes a sentence a constraint).

2. **Ticket analysis** - `classify_tasks` matches every ticket against
   a small `THEME_LEXICON` (cost_optimization, efficiency, rightsizing,
   cleanup, scaling, observability, maintenance, expansion, security,
   ux, data, experimentation). The lexicon is tuned for cloud / IT work
   - it includes terms like *rightsize, savings, spot, terraform, iam,
   cloudwatch* - so Jira tickets land on sensible themes. The team's
   **dominant themes** are the ones with the most keyword hits across
   the whole sprint backlog. Each ticket is then labelled `Aligned`,
   `Neutral`, or `Potentially misaligned`.

3. **Drift detection** - `detect_intent_drift` compares the **intent
   themes** (themes the project description hits) with the **ticket
   themes**. Two simple signals decide the verdict:

   - **proportion** - what fraction of tickets have a top theme that
     the intent also has?
   - **direction**  - of the tickets that don't fit, do they all pile
     up on a single off-intent theme?

   Verdict buckets:

   | Aligned share        | Direction share          | Verdict             |
   |----------------------|--------------------------|---------------------|
   | `>= 0.70`            | -                        | Aligned             |
   | `0.40 - 0.70`        | `< 0.35`                 | Partial Drift       |
   | `0.40 - 0.70`        | `>= 0.35` (escalation)   | Significant Drift   |
   | `< 0.40`             | -                        | Significant Drift   |

   All four thresholds are keyword arguments you can tune.

4. **Explanation** - `explain_drift` turns the structured verdict into a
   text report: headline, reason, theme distribution table, top
   contributors with one-line reasons, and a recommendation.

## Public API

```python
from truepath_ai import (
    parse_intent,           # text    -> structured intent dict
    classify_tasks,         # tickets -> per-ticket labels + dominant themes
    find_themes,            # text    -> dominant themes
    detect_intent_drift,    # intent + tickets -> drift verdict (dict)
    explain_drift,          # intent + tickets -> demo-ready text report
    THEME_LEXICON,          # rule-book: theme -> keyword list
)
```

## Requirements

Python 3.9+. The whole project runs on the standard library - no
install step needed.
