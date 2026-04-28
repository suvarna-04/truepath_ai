"""TruePath AI - Hackathon demo entry point.

Run end-to-end:

    python -m truepath_ai.main                          # uses data/sample_project.json
    python -m truepath_ai.main --data my_project.json   # load a different file

Walks through three labelled stages so judges can follow the demo:

    [1/3] Intent Analysis
    [2/3] Ticket Analysis
    [3/3] Drift Detection
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import List, Union

try:
    # Preferred: run as a package, e.g. `python -m truepath_ai.main`
    from .explainer import explain_drift
    from .intent_analyzer import extract_intent_themes, parse_intent
    from .semantic_classifier import get_default_classifier
    from .task_analyzer import classify_tasks
except ImportError:
    # Fallback: run directly as `python main.py` from inside the
    # truepath_ai/ folder. Add the parent dir to sys.path so the package
    # can still be resolved by absolute import.
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
    from truepath_ai.explainer import explain_drift
    from truepath_ai.intent_analyzer import (
        extract_intent_themes,
        parse_intent,
    )
    from truepath_ai.semantic_classifier import get_default_classifier
    from truepath_ai.task_analyzer import classify_tasks


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "sample_project.json"
)

# Used only as a fallback if the data file is missing - so the demo
# always has something interesting to show.
DEMO_PROJECT: dict = {
    "title": "AWS Cloud Cost Optimization",
    "intent": (
        "Reduce AWS cloud cost by 25 percent over the next two quarters "
        "by rightsizing over-provisioned EC2 and RDS resources, adopting "
        "Savings Plans and Spot instances, and decommissioning idle "
        "services and unused storage."
    ),
    "tasks": [
        # 4 cost-cutting tickets - in-intent
        {"id": "CLOUD-101", "description": "Rightsize over-provisioned production EC2 instances flagged by AWS Compute Optimizer to reduce monthly compute spend."},
        {"id": "CLOUD-102", "description": "Apply Savings Plans across steady-state EC2 and Fargate usage to reduce on-demand spend."},
        {"id": "CLOUD-103", "description": "Decommission idle RDS databases that have had zero connections for 30+ days to eliminate unused spend."},
        {"id": "CLOUD-104", "description": "Delete unattached EBS volumes and unused Elastic IPs across dev accounts to clean up unused resources."},
        # 4 capacity-adding tickets - off-intent (drift direction)
        {"id": "PLAT-201", "description": "Scale up production EKS cluster from 20 to 40 nodes to add capacity for the Black Friday traffic launch."},
        {"id": "PLAT-202", "description": "Add 3 read replicas to the orders Aurora cluster to scale read throughput for the new analytics service."},
        {"id": "PLAT-203", "description": "Increase node group capacity for the payments microservice to add headroom for peak traffic."},
        {"id": "PLAT-204", "description": "Scale up Kafka broker shards to handle the new event-driven analytics pipeline."},
    ],
}


# ---------------------------------------------------------------------------
# Pretty-print helpers (ASCII only - works in any terminal incl. PowerShell)
# ---------------------------------------------------------------------------

WIDTH = 72
BAR = "=" * WIDTH
SEP = "-" * WIDTH


def _section(step: str, title: str) -> None:
    """Print a clearly-separated section header."""
    print()
    print(f"{step}  {title}")
    print(SEP)


def _wrap(text: str, indent: str = "  ") -> str:
    return textwrap.fill(
        text,
        width=WIDTH - 2,
        initial_indent=indent,
        subsequent_indent=indent,
    )


def _print_banner(project_title: str) -> None:
    print(BAR)
    print("TruePath AI".center(WIDTH))
    print("Detect when tickets drift away from project intent".center(WIDTH))
    print(BAR)
    print(f"Project: {project_title}")
    _print_ai_status()


def _print_ai_status() -> None:
    """Tell the user whether the embedding-based AI layer is active.

    Loading the model is opt-in implicit (`classify_tasks` does it on
    first use), so this preview surfaces the status up-front for the
    demo. If the ML stack is not installed, we say so politely and
    explain that the system has dropped to its rule-based fallback.

    The `sentence-transformers` import-error warning emitted by the
    classifier is intentionally suppressed here - we're about to print
    the exact same information in a friendlier format.
    """
    import warnings as _warnings

    classifier = get_default_classifier()
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", RuntimeWarning)
        is_online = classifier.is_available()

    if is_online:
        print(
            f"AI layer    : ONLINE  (sentence-embedding theme "
            f"classifier, model={classifier.model_name})"
        )
    else:
        err = classifier.load_error() or "ML stack unavailable"
        print(
            "AI layer    : offline  (lexicon-only fallback - "
            "install requirements.txt to enable the embedding layer)"
        )
        print(f"               reason: {err}")


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def _print_intent_section(intent_text: str) -> None:
    _section("[1/3]", "Intent Analysis")
    parsed = parse_intent(intent_text)

    print("\nPrimary goal:")
    if parsed["primary_goal"]:
        print(_wrap(parsed["primary_goal"]))
    else:
        print("  (none detected)")

    print("\nSupporting goals & constraints:")
    if parsed["supporting"]:
        for item in parsed["supporting"]:
            print(f"  [{item['kind']:<10}] {item['text']}")
    else:
        print("  (none)")

    print()
    keywords = ", ".join(parsed["keywords"][:12]) or "(none detected)"
    print(f"Intent keywords: {keywords}")


def _print_task_section(
    tasks: List[Union[str, dict]],
    intent_themes: List[str],
) -> None:
    """Print the per-ticket section.

    Each ticket is tagged against the *project intent* (not just the
    backlog's dominant cluster) so the [2/3] view tells the same story
    as the [3/3] drift verdict at a glance:

        [in intent]  - ticket's top theme is one of the intent themes
        [OFF intent] - ticket's top theme is set but NOT an intent theme
        [no theme]   - no theme keywords matched at all
    """
    _section("[2/3]", "Ticket Analysis")
    classification = classify_tasks(tasks)
    intent_set = set(intent_themes)

    in_intent = off_intent = no_theme = 0
    for t in classification["tasks"]:
        theme = t["top_theme"]
        if theme is None:
            no_theme += 1
        elif theme in intent_set:
            in_intent += 1
        else:
            off_intent += 1

    print(f"\nSprint Backlog: {len(tasks)} ticket(s)")
    print(
        f"  in intent:  {in_intent}    "
        f"OFF intent: {off_intent}    "
        f"no theme:   {no_theme}"
    )
    print()

    tag_width = len("[OFF intent]")
    for t in classification["tasks"]:
        theme = t["top_theme"]
        if theme is None:
            tag = "[no theme]"
            theme_label = "(no theme)"
        elif theme in intent_set:
            tag = "[in intent]"
            theme_label = theme
        else:
            tag = "[OFF intent]"
            theme_label = theme

        hits = t["matched_keywords"]
        if hits:
            evidence = f"{len(hits)} hits: {', '.join(hits)}"
        else:
            evidence = "0 hits"

        print(
            f"  {t['id']}  {tag:<{tag_width}}  "
            f"{theme_label}  ({evidence})"
        )
        print(f'      "{t["description"]}"')

        # When the AI layer rescued this ticket from the lexicon's
        # blind spot, surface the model's evidence so reviewers can
        # see WHY the embedding model voted the way it did.
        if t.get("classified_by") == "semantic":
            score = t.get("semantic_top_score")
            score_text = f"{score:.2f}" if score is not None else "n/a"
            print(
                f"      [AI] embedding model assigned theme "
                f"'{t['top_theme']}' (cosine={score_text})"
            )


def _print_drift_section(
    intent_text: str,
    tasks: List[Union[str, dict]],
    title: str,
) -> None:
    _section("[3/3]", "Drift Detection")
    print()
    print(explain_drift(intent_text, tasks, project_title=title))


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

def _coerce_project(data: dict) -> dict:
    """Normalize either project shape into a single one used by the runner.

    Supported shapes:

      Legacy: {"title", "description", "goals": [...],
               "tasks": [{"id", "description"}, ...]}
      New:    {"title", "intent": "...",
               "tasks": [str | {"id", "description"}, ...]}
    """
    if "intent" in data:
        intent = data["intent"]
    else:
        description = (data.get("description") or "").strip()
        goals = data.get("goals") or []
        # Re-attach a period to each goal so they parse as separate sentences.
        sentences = (
            [description]
            + [g.strip().rstrip(".") + "." for g in goals if g.strip()]
        )
        intent = " ".join(s for s in sentences if s).strip()

    raw_tasks = data.get("tasks", []) or []
    tasks: List[Union[str, dict]] = []
    for t in raw_tasks:
        if isinstance(t, str):
            tasks.append(t)
        elif isinstance(t, dict):
            if t.get("id"):
                tasks.append({
                    "id": str(t["id"]),
                    "description": t.get("description", ""),
                })
            else:
                tasks.append(t.get("description", ""))

    return {
        "title": data.get("title", "Untitled Project"),
        "intent": intent,
        "tasks": tasks,
    }


def _load_project(path: Path) -> dict:
    """Load the project from disk; fall back to the built-in demo if missing."""
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return _coerce_project(json.load(f))
    return DEMO_PROJECT


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(project: dict) -> None:
    """Run all three stages and print the report."""
    title = project.get("title", "Untitled Project")
    intent = project.get("intent", "")
    tasks = project.get("tasks", [])

    intent_themes = extract_intent_themes(intent)

    _print_banner(title)
    _print_intent_section(intent)
    _print_task_section(tasks, intent_themes)
    _print_drift_section(intent, tasks, title)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the TruePath AI hackathon demo end-to-end."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to a project JSON file (default: data/sample_project.json).",
    )
    args = parser.parse_args()
    run(_load_project(args.data))


if __name__ == "__main__":
    main()
