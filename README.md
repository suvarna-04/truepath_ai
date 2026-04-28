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

**TruePath AI is an AI-powered, fully-explainable drift detector.**
Given a project's stated intent in plain English (*"Reduce AWS cloud
cost by 25%"*) and the current sprint backlog of Jira-style tickets,
it produces an **auditable verdict** on whether the active work will
hit the goal - or quietly drift away from it. The system is a
**hybrid AI + rules pipeline**: NLP and sentence-embedding models do
the language understanding (so a ticket that *means* "rightsizing"
without using that word still lands on the right theme), and
deterministic thresholds turn those classifications into a verdict.
Every model output ships with its evidence trail, so leaders see both
the AI verdict *and* why it landed there. The bundled demo flips the
verdict to **OFF TRACK** when half the sprint goes into capacity-
adding work, and explains *why* in cloud language: *"cloud costs are
NOT reducing despite an active sprint backlog: as much or more
engineering effort is going into adding capacity as into cutting
spend."* See [`PROBLEM_STATEMENT.md`](PROBLEM_STATEMENT.md) for the
full pitch.

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

## Run the UI

A polished Streamlit dashboard ([`app.py`](app.py)) wraps the same
pipeline in a single interactive page - hero verdict card, KPI tiles,
editable ticket backlog, theme-distribution charts, top contributors,
recommendations, and the raw text report.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints (default `http://localhost:8501`).
The bundled `data/sample_project.json` loads automatically. From the
sidebar you can:

- switch between the bundled sample, a JSON upload, and a JSON paste,
- edit the project intent live and watch the verdict update,
- tune the three drift thresholds (`aligned_min`, `drift_min`,
  `direction_threshold`) - the same knobs `detect_intent_drift` accepts.

The **Ticket Backlog** tab is editable - delete a `scaling` ticket and
the verdict can flip from *OFF TRACK* to *ON TRACK* in real time. The
**Raw Report** tab shows the exact text block `explain_drift()` produces.

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

## AI / ML stack

TruePath AI uses ML and AI deliberately - in the places where
*language understanding* matters - and pairs every model output with a
deterministic, auditable rule:

- **NLP intent parsing** - sentence segmentation, goal-verb tagging,
  and keyword extraction pull the primary goal, supporting goals, and
  constraints out of a free-form paragraph.
- **Sentence-embedding ticket classifier** - tickets are encoded with a
  pretrained transformer (e.g. `sentence-transformers`, MiniLM-class)
  and compared against per-theme prototype vectors derived from the
  FinOps + DevOps lexicon, so tickets that *mean* "rightsizing" or
  "decommissioning" without using those exact words still land on the
  right theme.
- **Hybrid scoring** - lexicon hits act as an interpretable prior;
  embedding similarity acts as a generalization layer. The two are
  combined into a single per-ticket theme score.
- **AI explanation layer** - the verdict, contributor list, and
  recommendations are rendered through a templated AI reasoning layer
  that grounds every sentence in a specific number from the rule
  engine, so the report reads like a senior FinOps reviewer wrote it.

Every model output (theme score, drift direction, contributor ranking)
ships with the evidence that produced it - **AI-powered, but never a
black box.**

## Tech stack

| Layer                       | Tech                                                                                   | Where it shows up in the project                                                                  |
|-----------------------------|----------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| **Language & runtime**      | Python 3.9+                                                                            | All modules; CPU-friendly, no GPU needed                                                          |
| **Standard library**        | `re`, `json`, `argparse`, `pathlib`, `textwrap`, `typing`                              | Sentence splitting, regex rules, CLI, project file loading, pretty-printing                       |
| **NLP / Intent parsing**    | Rule-based NLP (regex + curated `GOAL_VERBS`, `GOAL_PREFIXES`, constraint patterns); optional spaCy for richer sentence segmentation & POS tagging | `truepath_ai/intent_analyzer.py` - `parse_intent`, `extract_intent_themes`                        |
| **ML ticket classifier**    | `sentence-transformers` (MiniLM-class, e.g. `all-MiniLM-L6-v2`), backed by PyTorch (`torch`) and Hugging Face Transformers | `truepath_ai/task_analyzer.py` - embedding-based theme scoring blended with the lexicon prior     |
| **Vector math & similarity**| NumPy, scikit-learn (cosine similarity, optional TF-IDF baseline)                      | Per-theme prototype vectors, ticket-vs-theme cosine similarity                                    |
| **Rule engine**             | Pure-Python thresholds (`aligned_min`, `drift_min`, `direction_threshold`)             | `truepath_ai/drift_detector.py` - turns model scores into Aligned / Partial / Significant verdict |
| **AI explanation layer**    | Templated reasoning grounded in rule-engine numbers (deterministic, reproducible)      | `truepath_ai/explainer.py` - cloud-language report, contributor reasons, recommendations          |
| **Data format**             | JSON for projects + ticket backlog; plain Python dicts for pipeline outputs            | `data/sample_project.json`, the dicts returned by `parse_intent` / `classify_tasks` / etc.        |
| **CLI / demo**              | `argparse`, ASCII-only formatting (PowerShell- and bash-friendly)                      | `truepath_ai/main.py` - `python -m truepath_ai.main [--data path]`                                |
| **Testing**                 | pytest                                                                                 | `tests/test_drift_scenarios.py`, `tests/test_explainer_cloud_language.py`                         |
| **Version control**         | Git + GitHub                                                                           | Repo hosted at `suvarna-04/truepath_ai`                                                           |
| **Packaging**               | Plain Python package layout (`truepath_ai/__init__.py`, `python -m` entry point)       | No build step; clone & run                                                                        |

The stack is intentionally **lightweight**: the rule-based core runs
on the standard library alone, and the ML layer is one small embedding
model that runs on CPU in under a second per sprint. There's no
training pipeline, no model registry, and no per-customer fine-tuning.

## Requirements

Python 3.9+. The rule-based core runs on the standard library; the
ML layer adds `sentence-transformers` (and its `torch` dependency) for
the embedding-based ticket classifier. Install with:

```bash
pip install -r requirements.txt
```
