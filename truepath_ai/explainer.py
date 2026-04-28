"""Demo-friendly drift report, framed in IT / cloud-cost language.

Plain-English version (for judges):

    Turn the structured output of `detect_intent_drift` into a single
    human-readable text block - the kind of output you'd put on a slide
    or paste into Slack. When the project intent is a cost-optimization
    initiative, the report is rephrased so reviewers can immediately see
    *why cloud spend is not coming down* even though engineers are busy.

This module only formats and narrates. Every fact in the report is
produced by the rule-based pipeline, so the report can be defended
end-to-end without invoking any ML model.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

from .drift_detector import detect_intent_drift
from .task_analyzer import classify_tasks


WIDTH = 72
BAR = "=" * WIDTH
SEP = "-" * WIDTH


# ---------------------------------------------------------------------------
# IT / cloud phrasing
# ---------------------------------------------------------------------------
# Every internal theme name is also rendered as a friendly phrase so the
# narrative reads like an IT/cloud status update instead of a lexicon
# dump. Edit this table to bias the prose toward your team's vocabulary.
THEME_PHRASES: Dict[str, str] = {
    # FinOps sub-themes
    "cost_optimization": "cost optimization initiatives",
    "efficiency":        "efficiency & performance tuning",
    "rightsizing":       "rightsizing & commitment pricing",
    "cleanup":           "decommissioning idle resources",
    # Cloud / DevOps general themes
    "scaling":           "infrastructure scaling",
    "observability":     "monitoring & dashboards",
    "maintenance":       "platform maintenance & tech-debt",
    "expansion":         "new rollouts & regional expansion",
    "security":          "security & compliance",
    "ux":                "UX / interface work",
    "data":              "data & analytics",
    "experimentation":   "spikes & prototypes",
}

# When any of these themes appears in the intent, treat the project as
# a cloud-cost-optimization initiative and frame the whole report in
# cloud-spend language.
COST_INTENT_THEMES = frozenset({
    "cost_optimization", "efficiency", "rightsizing", "cleanup",
})

# Off-intent themes that ACTIVELY INCREASE the AWS bill - capacity
# moves and new rollouts both raise spend. We use this set to build
# the "why aren't costs coming down?" explanation.
SPEND_INCREASING_THEMES = frozenset({"scaling", "expansion"})


def _phrase(theme: Optional[str]) -> str:
    """Render a theme name as an IT/cloud-friendly phrase."""
    if theme is None:
        return "(no theme)"
    return THEME_PHRASES.get(theme, theme)


def _is_cost_project(intent_themes: List[str]) -> bool:
    """Return True iff the project intent is a cost-optimization initiative.

    The signal we trust is *all four FinOps sub-themes* appearing in the
    intent themes. This happens when ``intent_analyzer.extract_intent_themes``
    detects a cost-reduction phrasing and auto-fills the canonical FinOps
    set. It rules out incidental matches (e.g. a UX intent that happens
    to contain the word "improve" and so lights up only the
    ``efficiency`` theme on its own).
    """
    return COST_INTENT_THEMES.issubset(intent_themes)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _verdict_headline(drift_level: str, cost_mode: bool) -> str:
    """Map the drift level to a clear demo-friendly headline."""
    if drift_level == "Aligned":
        if cost_mode:
            return "Verdict: ON TRACK  (cloud cost reduction is being addressed)"
        return "Verdict: NO DRIFT  (Aligned)"
    if drift_level == "Partial Drift":
        if cost_mode:
            return "Verdict: AT RISK  (cloud spend reduction partially off-course)"
        return "Verdict: DRIFT DETECTED  (Partial)"
    # Significant Drift
    if cost_mode:
        return "Verdict: OFF TRACK  (cloud costs will not reduce on current sprint backlog)"
    return "Verdict: DRIFT DETECTED  (Significant)"


def _pick_contributors(
    classified_tasks: List[dict],
    drift_direction: Optional[str],
    intent_themes: List[str],
    limit: int,
) -> List[dict]:
    """Pick the top tasks pulling the backlog away from the project intent.

    Order of preference - the most drift-causing first:

      1. Tasks whose top theme equals the drift direction.
         (more matched keywords = more emphatically off-intent)
      2. Tasks with no theme at all (unclear purpose).
      3. Other off-intent tasks (different theme, still off-intent).

    This rule is purely deterministic - judges can read this function
    and see exactly why a ticket ended up on the offenders list.
    """
    by_hits = lambda t: -len(t["matched_keywords"])  # noqa: E731

    direction_tasks = sorted(
        [t for t in classified_tasks if t["top_theme"] == drift_direction
         and drift_direction is not None],
        key=by_hits,
    )
    unthemed_tasks = [t for t in classified_tasks if t["top_theme"] is None]
    other_off_tasks = sorted(
        [t for t in classified_tasks
         if t["top_theme"] is not None
         and t["top_theme"] != drift_direction
         and t["top_theme"] not in intent_themes],
        key=by_hits,
    )

    return (direction_tasks + unthemed_tasks + other_off_tasks)[:limit]


def _contributor_reason(
    task: dict,
    drift_direction: Optional[str],
    intent_themes: List[str],
    cost_mode: bool,
) -> str:
    """One-line, plain-English reason this ticket is on the offenders list."""
    top = task["top_theme"]
    if top is None:
        if cost_mode:
            return (
                "No cloud-cost or DevOps theme matched - it's unclear how "
                "this ticket contributes to reducing cloud spend."
            )
        return (
            "Doesn't match any project theme - it's unclear how this "
            "ticket fits the goal."
        )
    if cost_mode and top in SPEND_INCREASING_THEMES:
        return (
            f"Top theme '{_phrase(top)}' adds capacity to the AWS bill "
            f"instead of reducing it."
        )
    if drift_direction and top == drift_direction:
        return (
            f"Pulls toward '{_phrase(top)}', which is not part of the "
            f"project intent."
        )
    return (
        f"Top theme '{_phrase(top)}' is outside the project intent."
    )


def _why_cloud_costs_section(
    theme_view: dict,
    classification: dict,
    cost_mode: bool,
) -> List[str]:
    """Build the 'Why' narrative.

    In cost mode this section explicitly states *why cloud costs are
    (or are not) reducing despite active engineering work*, by
    contrasting the share of cost-cutting tickets against the share of
    capacity-adding tickets.
    """
    lines: List[str] = ["Why", SEP]

    if not cost_mode:
        # Plain mode: just relay the rule engine's reason verbatim.
        lines.append(theme_view["reason"])
        return lines

    counts = theme_view["counts"]
    aligned = counts["aligned"]
    total = counts["total"]
    aligned_share = theme_view["aligned_share"]
    direction = theme_view["drift_direction"]
    level = theme_view["drift_level"]
    distribution = theme_view["task_theme_distribution"]

    spend_inc_count = sum(
        row["tasks"] for row in distribution
        if row["name"] in SPEND_INCREASING_THEMES
    )
    spend_inc_share = spend_inc_count / total if total else 0.0
    spend_inc_phrases = ", ".join(
        _phrase(row["name"])
        for row in distribution
        if row["name"] in SPEND_INCREASING_THEMES
    )

    # 1) How much of the backlog actually attacks cloud spend?
    lines.append(
        f"{aligned}/{total} tickets ({aligned_share:.0%}) directly "
        f"target cloud spend - rightsizing, decommissioning idle "
        f"resources, savings plans, or efficiency tuning."
    )

    # 2) How much of the backlog INCREASES cloud spend?
    if spend_inc_count > 0:
        lines.append(
            f"{spend_inc_count}/{total} tickets ({spend_inc_share:.0%}) "
            f"are {spend_inc_phrases} - work that ADDS capacity to the "
            f"AWS bill instead of reducing it."
        )

    # 3) The verdict-specific punchline.
    if level == "Aligned":
        lines.append(
            "Cloud cost reduction is on track: cost optimization "
            "initiatives dominate the sprint backlog."
        )
    elif level == "Partial Drift":
        if direction in SPEND_INCREASING_THEMES:
            lines.append(
                f"Cloud cost reduction is at risk: cost optimization "
                f"initiatives are running in parallel with "
                f"'{_phrase(direction)}', which pushes spend the other "
                f"way."
            )
        else:
            lines.append(
                "Cloud cost reduction is at risk: a meaningful slice of "
                "the sprint backlog is on work that does not reduce "
                "spend."
            )
    else:  # Significant Drift
        if spend_inc_count >= max(aligned, 1) and spend_inc_count > 0:
            lines.append(
                "That is why cloud costs are NOT reducing despite an "
                "active sprint backlog: as much or more engineering "
                "effort is going into adding capacity as into cutting "
                "spend."
            )
        elif direction in SPEND_INCREASING_THEMES:
            lines.append(
                f"That is why cloud costs are NOT reducing despite an "
                f"active sprint backlog: cost optimization initiatives "
                f"are being out-prioritized by '{_phrase(direction)}'."
            )
        elif direction:
            lines.append(
                f"That is why cloud costs are NOT reducing despite an "
                f"active sprint backlog: the bulk of the work is on "
                f"'{_phrase(direction)}', not on cost optimization "
                f"initiatives."
            )
        else:
            lines.append(
                "That is why cloud costs are NOT reducing despite an "
                "active sprint backlog: most tickets are on neutral or "
                "unrelated work, not cost optimization initiatives."
            )

    return lines


def _recommendation_bullets(
    theme_view: dict,
    contributors: List[dict],
    cost_mode: bool,
) -> List[str]:
    """Return a list of decision-supportive bullets for the report.

    The bullets describe *options* the team can choose between - they
    are deliberately neutral ("Prioritize ... unless ...", "If growth
    is now the goal, update the intent"), not prescriptive ("you must
    cancel ticket X"). Every bullet is derived deterministically from
    the verdict, the dominant off-intent direction, and the contributor
    list, so a reviewer can audit each suggestion.
    """
    level = theme_view["drift_level"]
    direction = theme_view["drift_direction"]
    direction_phrase = _phrase(direction) if direction else None
    ids = ", ".join(t["id"] for t in contributors) if contributors else ""

    bullets: List[str] = []

    # ---- Aligned -----------------------------------------------------
    if level == "Aligned":
        if cost_mode:
            bullets.append(
                "Continue prioritizing rightsizing, decommissioning idle "
                "resources, and savings-plan adoption - the work that is "
                "already moving cloud spend down."
            )
            bullets.append(
                "As new tickets land, review whether any infrastructure "
                "scaling or rollout work is justified by demand and "
                "approved budget."
            )
            bullets.append(
                "If growth becomes a goal in a future planning cycle, "
                "explicitly update the project intent so subsequent drift "
                "checks reflect the new objective."
            )
        else:
            bullets.append(
                "Continue prioritizing the on-intent work that is "
                "already shaping the sprint backlog."
            )
            bullets.append(
                "Re-run this check when the sprint backlog changes "
                "shape, in case new tickets quietly drift the team off "
                "intent."
            )
        return bullets

    # ---- Drift detected (Partial or Significant) ---------------------
    if cost_mode:
        # 1) Prioritize cost-cutting work (the corrective lever).
        bullets.append(
            "Prioritize rightsizing and decommissioning idle / unused "
            "resources - these are the tickets that directly reduce "
            "cloud spend."
        )

        # 2) Deprioritize the dominant off-intent direction, with a hedge.
        if direction in SPEND_INCREASING_THEMES:
            bullets.append(
                f"Deprioritize '{direction_phrase}' tickets unless the "
                f"additional capacity is genuinely required by demand "
                f"and the additional AWS spend is approved."
            )
        elif direction:
            bullets.append(
                f"Deprioritize '{direction_phrase}' tickets unless they "
                f"directly support a cost optimization initiative."
            )

        # 3) Explicitly offer the "update the intent" exit.
        if direction in SPEND_INCREASING_THEMES:
            bullets.append(
                f"If growth is now the actual goal, explicitly update "
                f"the project intent to include '{direction_phrase}' "
                f"and re-baseline the cost target accordingly."
            )
        elif direction:
            bullets.append(
                f"If the project's true goal has shifted toward "
                f"'{direction_phrase}', explicitly update the project "
                f"intent so future drift checks measure against the "
                f"new objective."
            )
        else:
            bullets.append(
                "If the project's goal has shifted, explicitly update "
                "the project intent so future drift checks measure "
                "against the new objective."
            )

        # 4) Pointer to the specific tickets worth revisiting.
        if ids:
            bullets.append(f"Tickets worth a second look: {ids}.")
        return bullets

    # ---- Plain (non-cost) drift --------------------------------------
    if direction:
        bullets.append(
            "Continue prioritizing the on-intent work that is already "
            "moving the project forward."
        )
        bullets.append(
            f"Deprioritize '{direction_phrase}' tickets unless they "
            f"directly support the current project intent."
        )
        bullets.append(
            f"If '{direction_phrase}' is now the actual goal, "
            f"explicitly update the project intent to include it."
        )
    else:
        bullets.append(
            "Continue prioritizing the on-intent work that is already "
            "moving the project forward."
        )
        bullets.append(
            "Review the borderline tickets and decide, per ticket, "
            "whether to re-scope onto intent or to broaden the project "
            "intent description."
        )

    if ids:
        bullets.append(f"Tickets worth a second look: {ids}.")
    return bullets


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def explain_drift(
    intent_text: str,
    tasks: List[Union[str, dict]],
    *,
    project_title: str = "Project",
    top_contributors: int = 3,
) -> str:
    """Build a clear, demo-ready drift report as a single text block.

    The report answers, in order:

      1. **Was drift detected?**       - one-line verdict at the top.
      2. **Why?**                       - in cost-optimization projects
                                          this section spells out why
                                          cloud spend is (or is not)
                                          reducing despite active work.
      3. **Which tickets contributed?** - top-N tickets ranked by the
                                          deterministic rule in
                                          ``_pick_contributors``.
      4. **What should we do?**         - short recommendation matched
                                          to the verdict.

    Every value printed here comes from the rule-based pipeline, so the
    report is fully auditable.
    """
    if not tasks:
        return (
            f"{BAR}\n"
            f"TruePath AI - Intent Drift Report\n"
            f"Project: {project_title}\n"
            f"{BAR}\n"
            f"No tickets supplied; nothing to evaluate."
        )

    theme_view = detect_intent_drift(intent_text, tasks)
    classification = classify_tasks(tasks)
    cost_mode = _is_cost_project(theme_view["intent_themes"])
    contributors = _pick_contributors(
        classification["tasks"],
        theme_view["drift_direction"],
        theme_view["intent_themes"],
        limit=top_contributors,
    )

    counts = theme_view["counts"]
    aligned_share_pct = f"{theme_view['aligned_share']:.0%}"

    intent_phrases = (
        ", ".join(_phrase(t) for t in theme_view["intent_themes"])
        or "(none detected)"
    )
    direction_phrase = _phrase(theme_view["drift_direction"]) \
        if theme_view["drift_direction"] else "(none)"

    lines: List[str] = [
        BAR,
        "TruePath AI - Intent Drift Report",
        f"Project: {project_title}",
        BAR,
        _verdict_headline(theme_view["drift_level"], cost_mode),
        f"Cost-aligned tickets: {counts['aligned']}/{counts['total']} "
        f"({aligned_share_pct})"
        if cost_mode else
        f"Aligned tickets: {counts['aligned']}/{counts['total']} "
        f"({aligned_share_pct})",
        "",
    ]

    lines.extend(_why_cloud_costs_section(theme_view, classification, cost_mode))

    lines.extend([
        "",
        f"Intent themes  : {intent_phrases}",
        f"Drift direction: {direction_phrase}",
        "",
        "Sprint Backlog theme distribution",
        SEP,
    ])

    if theme_view["task_theme_distribution"]:
        for row in theme_view["task_theme_distribution"]:
            tag = "in intent" if row["in_intent"] else "OFF intent"
            phrase = _phrase(row["name"])
            lines.append(
                f"  {phrase:<34} {row['tasks']:>3}  "
                f"({row['share']:.0%})  {tag}"
            )
    else:
        lines.append("  (no themes matched any ticket)")

    lines.append("")
    lines.append(
        "Top contributors to drift" if not cost_mode else
        "Tickets driving the sprint backlog away from cloud cost reduction"
    )
    lines.append(SEP)
    if not contributors:
        lines.append(
            "  None - every ticket lands on a cost optimization initiative."
            if cost_mode else
            "  None - every ticket lands within the project's intent themes."
        )
    else:
        for t in contributors:
            top = t["top_theme"]
            phrase = _phrase(top)
            if top is None:
                tag = "no theme"
            elif top in theme_view["intent_themes"]:
                tag = "in intent"
            elif cost_mode and top in SPEND_INCREASING_THEMES:
                tag = "ADDS spend"
            else:
                tag = "OFF intent"
            lines.append(f"  {t['id']}  theme={phrase}  [{tag}]")
            lines.append(f'      "{t["description"]}"')
            lines.append(
                "      reason: "
                + _contributor_reason(
                    t,
                    theme_view["drift_direction"],
                    theme_view["intent_themes"],
                    cost_mode,
                )
            )
            lines.append("")

    lines.append("Recommendation")
    lines.append(SEP)
    lines.append(
        "Decision-support options (the team picks what fits the "
        "situation; this report is not prescriptive):"
    )
    for bullet in _recommendation_bullets(theme_view, contributors, cost_mode):
        lines.append(f"  - {bullet}")
    lines.append(BAR)

    return "\n".join(lines)
