"""Intent drift - decide whether the backlog still matches the project.

Plain-English version (for judges):

    Compare the THEMES of the project intent against the THEMES of the
    actual tasks. Two simple, transparent signals - never ML:

      * proportion - what fraction of tasks have a top theme that the
        intent also has?
      * direction  - of the tasks that don't fit, do they all pile up on
        a single off-intent theme? A clear pull in one direction is
        worse than the same number of misses scattered around.

    Decision rules (each threshold is exposed as a knob):

      aligned_share >= aligned_min          -> "Aligned"
      aligned_share <  drift_min            -> "Significant Drift"
      otherwise (in between):
        if a single off-intent theme captures >= direction_threshold
          of all tasks                      -> "Significant Drift"  (escalate)
        else                                -> "Partial Drift"
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

from .intent_analyzer import extract_intent_themes
from .task_analyzer import classify_tasks


def detect_intent_drift(
    intent_text: str,
    tasks: List[Union[str, dict]],
    *,
    top_intent_themes: int = 5,
    aligned_min: float = 0.70,
    drift_min: float = 0.40,
    direction_threshold: float = 0.35,
) -> Dict[str, object]:
    """Compare intent themes vs task themes and return a drift verdict.

    Args:
        intent_text:        free-form description of the project intent.
        tasks:              list of task strings or {id, description} dicts.
        top_intent_themes:  how many themes from the intent count as in-scope.
        aligned_min:        proportion required to call the backlog "Aligned".
        drift_min:          below this proportion the backlog is always
                            "Significant Drift".
        direction_threshold: share of tasks needed on a single off-intent
                            theme to escalate "Partial" to "Significant".

    Returns:
        A dict shaped like::

            {
                "drift_level": "Aligned" | "Partial Drift" | "Significant Drift",
                "aligned_share": 0.625,
                "intent_themes": ["optimization", "data"],
                "task_theme_distribution": [
                    {"name": "optimization", "tasks": 3,
                     "share": 0.375, "in_intent": True},
                    ...
                ],
                "drift_direction": "ux" | None,
                "counts": {"aligned": 5, "off_intent": 2,
                           "unthemed": 1, "total": 8},
                "reason": "...human-readable explanation matching the verdict...",
            }
    """
    intent_themes = extract_intent_themes(intent_text, top_k=top_intent_themes)
    classification = classify_tasks(tasks)
    per_task = classification["tasks"]
    total = len(per_task)

    aligned_count = 0
    off_count = 0
    unthemed_count = 0
    task_theme_counts: Dict[str, int] = {}
    off_theme_counts: Dict[str, int] = {}

    for t in per_task:
        top = t["top_theme"]
        if top is None:
            unthemed_count += 1
            continue
        task_theme_counts[top] = task_theme_counts.get(top, 0) + 1
        if top in intent_themes:
            aligned_count += 1
        else:
            off_count += 1
            off_theme_counts[top] = off_theme_counts.get(top, 0) + 1

    aligned_share = (aligned_count / total) if total else 0.0

    drift_direction: Optional[str] = (
        max(off_theme_counts, key=off_theme_counts.get)
        if off_theme_counts
        else None
    )
    direction_share = (
        off_theme_counts[drift_direction] / total
        if drift_direction and total
        else 0.0
    )

    # ---- Apply the decision rules from the module docstring -------------
    if total == 0:
        level = "Aligned"
        reason = "No tickets to evaluate."
    elif aligned_share >= aligned_min:
        level = "Aligned"
        reason = (
            f"{aligned_count}/{total} tickets ({aligned_share:.0%}) match "
            f"the intent themes {intent_themes}. Above the aligned "
            f"threshold of {aligned_min:.0%}."
        )
    elif aligned_share < drift_min:
        level = "Significant Drift"
        if drift_direction:
            reason = (
                f"Only {aligned_share:.0%} of tickets match intent themes "
                f"{intent_themes}; the sprint backlog has shifted toward "
                f"'{drift_direction}' ({direction_share:.0%} of tickets)."
            )
        else:
            reason = (
                f"Only {aligned_share:.0%} of tickets match intent themes "
                f"{intent_themes}; most tickets have no clear theme."
            )
    elif drift_direction and direction_share >= direction_threshold:
        # Borderline proportion + a clear directional pull -> escalate.
        level = "Significant Drift"
        reason = (
            f"{aligned_share:.0%} of tickets match intent themes "
            f"{intent_themes}, but {direction_share:.0%} concentrate "
            f"around '{drift_direction}', a clear off-intent direction."
        )
    else:
        level = "Partial Drift"
        if drift_direction:
            reason = (
                f"{aligned_share:.0%} of tickets match intent themes "
                f"{intent_themes}; some drift toward '{drift_direction}' "
                f"({direction_share:.0%}), but no single off-intent theme "
                f"dominates."
            )
        else:
            reason = (
                f"{aligned_share:.0%} of tickets match intent themes "
                f"{intent_themes}, between drift thresholds "
                f"({drift_min:.0%}-{aligned_min:.0%})."
            )

    distribution = sorted(
        [
            {
                "name": name,
                "tasks": count,
                "share": round(count / total, 3) if total else 0.0,
                "in_intent": name in intent_themes,
            }
            for name, count in task_theme_counts.items()
        ],
        key=lambda x: (-x["tasks"], x["name"]),
    )

    return {
        "drift_level": level,
        "aligned_share": round(aligned_share, 3),
        "intent_themes": intent_themes,
        "task_theme_distribution": distribution,
        "drift_direction": drift_direction,
        "counts": {
            "aligned": aligned_count,
            "off_intent": off_count,
            "unthemed": unthemed_count,
            "total": total,
        },
        "reason": reason,
    }
