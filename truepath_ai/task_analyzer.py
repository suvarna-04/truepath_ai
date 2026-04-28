"""Task analysis - figure out what each task is about and whether the
backlog as a whole has a clear theme.

Plain-English version (for judges):

    Given a list of tasks, this module:
      1. matches each task against a small THEME_LEXICON. The lexicon
         is split into twelve themes:
            - four FinOps sub-themes that describe cost-saving work:
              cost_optimization, efficiency, rightsizing, cleanup.
            - eight general cloud / DevOps themes:
              scaling, observability, maintenance, expansion,
              security, ux, data, experimentation.
      2. (when the ML layer is available) for tickets whose lexicon
         match is empty, falls back to a sentence-embedding classifier
         that compares the ticket against per-theme prototype vectors
         built from the same lexicon. This catches tickets that *mean*
         "rightsizing" or "decommissioning" without using those exact
         words.
      3. ranks the team's "dominant themes" by total keyword hits,
      4. labels each task as "Aligned", "Neutral", or
         "Potentially misaligned" based on how strongly it lands on
         a dominant theme.

The lexicon is the auditable prior; the embedding model is a
generalization layer that only fires when the lexicon would otherwise
have produced no answer at all. Every per-task dict therefore carries
both the matched keywords and (when the ML layer is on) the semantic
score that produced the verdict, so the chain of evidence stays fully
inspectable end-to-end.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# Theme lexicon - the rule book.
# Edit the lists to bias the system toward your project's vocabulary.
# ---------------------------------------------------------------------------
THEME_LEXICON: Dict[str, List[str]] = {
    # ---- FinOps sub-themes -------------------------------------------------
    # These four themes are deliberately narrow so a cost-optimization
    # project intent lands on the *right* sub-concept (cost vs efficiency
    # vs capacity-shape vs waste removal) instead of one giant bucket.
    "cost_optimization": [
        # The "WHY": cost / spend / budget vocabulary.
        "cost", "costs", "spend", "spending", "savings", "saving",
        "budget", "budgets", "billing", "bill", "expense", "expenses",
        "expensive", "expenditure", "finops",
        "reduce", "reduces", "reducing", "reduction",
        "lower", "lowered", "lowering",
        "cut", "cutting", "trim", "trimmed", "shrink", "shrinking",
        "save", "saved", "saves",
    ],
    "efficiency": [
        # "DOING MORE WITH LESS": utilisation, performance, tuning.
        "optimize", "optimization", "optimal",
        "efficient", "efficiency",
        "performance", "latency", "throughput",
        "utilization", "utilisation",
        "faster", "speed", "speedup", "tune", "tuning", "cache", "caches",
        "streamline", "improve", "improvement",
    ],
    "rightsizing": [
        # Capacity / instance-shape / commitment-pricing decisions.
        # Note: "autoscale*" lives here (not in `scaling`) because
        # autoscaling is fundamentally about matching capacity to demand,
        # i.e. right-sizing, rather than expanding capacity outright.
        "rightsize", "rightsizing", "downsize", "downsizing",
        "oversized", "undersized",
        "over-provisioned", "under-provisioned",
        "overprovisioned", "underprovisioned",
        "autoscale", "autoscaling", "autoscaler", "autoscalers",
        "spot", "reserved", "graviton",
    ],
    "cleanup": [
        # Decommissioning idle and unused resources.
        "decommission", "decommissioned", "decommissioning",
        "idle", "unused", "unattached",
        "orphan", "orphaned", "stale", "dormant",
        "retired", "retire",
        "delete", "deletion", "remove", "removal", "purge",
        "lifecycle", "tiering", "glacier",
        "clean", "cleanup",
    ],
    # ---- Cloud / DevOps general themes ------------------------------------
    # Useful for classifying tasks. `scaling`, `expansion`, and
    # `maintenance` are explicitly NOT promoted as intent themes for a
    # cost-optimization project (see INTENT_THEME_BLOCKLIST in
    # intent_analyzer.py) - they describe HOW engineering work happens,
    # not WHAT a cost-cutting project is trying to achieve.
    "scaling": [
        # Capacity expansion - cluster scaling, replica counts, sharding.
        # Distinct from `rightsizing` (matching size to load) and from
        # `expansion` (rolling out new things).
        "scale", "scaling", "scaled", "scalable", "scaleup", "scale-up",
        "capacity", "headroom",
        "horizontally", "vertically",
        "replica", "replicas", "replication",
        "shard", "shards", "sharding",
        "grow", "growth",
    ],
    "observability": [
        # Monitoring, dashboards, alerts, logs, traces.
        "monitor", "monitoring", "monitored", "observability",
        "log", "logs", "logging", "telemetry",
        "metric", "metrics",
        "alert", "alerts", "alarm", "alarms",
        "dashboard", "dashboards",
        "visualize", "visualizes", "visualise", "visualises",
        "visualization", "visualisation",
        "cloudwatch", "datadog", "prometheus", "grafana", "splunk",
        "trace", "tracing", "traces", "apm",
        "instrument", "instrumentation",
    ],
    "maintenance": [
        "fix", "bug", "bugfix",
        "refactor", "refactoring", "refactored",
        "maintain", "maintenance",
        "update", "updates", "upgrade", "upgrades", "upgrading",
        "patch", "patches", "patching",
        "deprecate", "deprecated", "rewrite", "tech-debt", "stability",
        "terraform", "module", "modules", "legacy", "rotate",
    ],
    "expansion": [
        # Rolling out new things / new regions / new integrations.
        "expand", "expansion",
        "add", "new", "additional", "extend", "extension", "broaden",
        "integrate", "integration", "connector",
        "rollout", "roll-out", "launch",
        "deploy", "deployment", "provision", "provisioning",
        "migrate", "migration", "multi-region", "region",
    ],
    "security": [
        "secure", "security", "encrypt", "encryption", "auth",
        "authentication", "authorization", "vulnerability", "compliance",
        "hipaa", "gdpr", "audit", "permission",
        "iam", "policy", "policies", "kms", "secrets", "soc",
        "credentials",
    ],
    "ux": [
        # Note: `dashboard` lives in `observability`. UX here is
        # specifically about user-facing interface work.
        "interface", "design", "layout", "visual", "color",
        "style", "theme", "usability", "screen", "page", "component",
        "homepage", "leaderboard", "gamified", "badge", "badges",
    ],
    "data": [
        # ML / analytics on data - observability moved out into its own theme.
        "analyze", "analysis", "analytics", "report",
        "data", "insight", "model", "score", "predict", "prediction",
        "classify", "classification", "rank", "ranking",
    ],
    "experimentation": [
        "experiment", "prototype", "explore", "poc", "research", "trial",
        "spike", "investigate", "study",
    ],
}


# ---------------------------------------------------------------------------
# Theme matching helpers
# ---------------------------------------------------------------------------

def _theme_hits(text: str) -> Tuple[Dict[str, List[str]], int]:
    """Return per-theme matched keywords and the total hit count for a text."""
    tokens = set(re.findall(r"[a-zA-Z][a-zA-Z\-]+", text.lower()))
    matches: Dict[str, List[str]] = {}
    total = 0
    for theme, lex in THEME_LEXICON.items():
        hits = [kw for kw in lex if kw in tokens]
        if hits:
            matches[theme] = hits
            total += len(hits)
    return matches, total


def find_themes(text: str, top_k: int = 3) -> List[str]:
    """Return the most prominent themes for a chunk of text.

    Themes are ranked by how many lexicon keywords they hit. Ties are
    broken alphabetically so the output is stable across runs.
    """
    matches, _ = _theme_hits(text)
    ranked = sorted(matches.items(), key=lambda x: (-len(x[1]), x[0]))
    return [theme for theme, _ in ranked[:top_k]]


def _normalize(items: List[Union[str, dict]]) -> List[dict]:
    """Accept plain strings or {id, description} dicts; return dicts.

    Auto-generates IDs (T1, T2, ...) when none are provided so the
    pretty-printed report always has stable handles for each task.
    """
    out: List[dict] = []
    for i, it in enumerate(items):
        if isinstance(it, str):
            out.append({"id": f"T{i + 1}", "description": it})
        elif isinstance(it, dict):
            out.append({
                "id": str(it.get("id") or f"T{i + 1}"),
                "description": it.get("description", ""),
            })
        else:
            raise TypeError(f"Unsupported task type: {type(it).__name__}")
    return out


# ---------------------------------------------------------------------------
# Public classifier
# ---------------------------------------------------------------------------

def _resolve_semantic_classifier(
    use_semantic: bool,
    classifier: Optional[Any],
) -> Optional[Any]:
    """Return a usable semantic classifier, or None.

    Kept as a tiny helper so the heavy import only fires when ML is
    actually requested - the lexicon-only path stays free of any
    `sentence-transformers` / `torch` cost.
    """
    if not use_semantic:
        return None
    if classifier is not None:
        return classifier if getattr(classifier, "is_available", lambda: False)() else None
    try:
        from .semantic_classifier import get_default_classifier
    except Exception:
        return None
    candidate = get_default_classifier()
    return candidate if candidate.is_available() else None


def classify_tasks(
    tasks: List[Union[str, dict]],
    *,
    top_themes: int = 3,
    aligned_threshold: int = 2,
    use_semantic: bool = True,
    semantic_threshold: float = 0.30,
    semantic_classifier: Optional[Any] = None,
) -> Dict[str, object]:
    """Identify dominant themes in a backlog and label each task.

    Plain-English version of the rules:

      1. Each task is scanned for theme keywords from THEME_LEXICON
         (the lexicon prior).
      2. If `use_semantic` is True and the embedding model is
         available, every ticket is also scored by a sentence-
         transformer against per-theme prototype vectors. The semantic
         score is recorded on the ticket as evidence and is used as a
         FALLBACK only - it picks a theme when the lexicon produced
         no match at all. This keeps the rule-based contract intact
         (a ticket that lights up the lexicon always wins by lexicon
         hits) while still letting the ML layer rescue tickets phrased
         in their own words.
      3. Hits are summed across the whole backlog. The top `top_themes`
         themes by hit count are the *dominant themes*.
      4. Each task is labelled:
           "Aligned"               - top theme is dominant AND the task
                                     hit at least `aligned_threshold`
                                     keywords for that theme.
           "Neutral"               - hit at least one theme keyword, but
                                     the top theme is non-dominant or
                                     below the threshold.
           "Potentially misaligned" - no theme keywords matched at all
                                      AND the embedding fallback was
                                      below `semantic_threshold` (or
                                      offline).

    Returns a dict shaped like::

        {
            "themes": [{"name": ..., "hits": ..., "tasks": [...]}, ...],
            "dominant_themes": ["optimization", "expansion", ...],
            "semantic_enabled": True | False,
            "tasks": [
                {
                    "id": "T1",
                    "description": "...",
                    "top_theme": "optimization" | None,
                    "matched_keywords": ["faster", "performance"],
                    "all_matches": {"optimization": [...], ...},
                    "semantic_scores": {"optimization": 0.41, ...} | {},
                    "semantic_top_theme": "optimization" | None,
                    "semantic_top_score": 0.41 | None,
                    "classified_by": "lexicon" | "semantic" | "none",
                    "label": "Aligned" | "Neutral" | "Potentially misaligned",
                    "reason": "...one-line, judge-friendly explanation...",
                },
                ...
            ],
        }

    The extra `semantic_*` and `classified_by` fields are always
    present so downstream callers can tell which signal produced the
    verdict and audit the model's contribution per ticket.
    """
    normalized = _normalize(tasks)
    classifier = _resolve_semantic_classifier(use_semantic, semantic_classifier)
    semantic_enabled = classifier is not None

    aggregate: Dict[str, int] = {t: 0 for t in THEME_LEXICON}
    aggregate_tasks: Dict[str, List[str]] = {t: [] for t in THEME_LEXICON}
    per_task: List[dict] = []

    for item in normalized:
        matches, _ = _theme_hits(item["description"])

        if matches:
            top_theme = max(matches, key=lambda t: len(matches[t]))
            top_hits = matches[top_theme]
            classified_by = "lexicon"
        else:
            top_theme = None
            top_hits = []
            classified_by = "none"

        # Always record semantic evidence when the ML layer is on, so a
        # judge can audit the model's view *and* see that the lexicon
        # outranked it on a tie. Falling back only when the lexicon is
        # silent keeps the existing deterministic contract intact.
        if classifier is not None:
            scores = classifier.score_themes(item["description"])
            if scores:
                sem_top_theme, sem_top_score = max(scores.items(), key=lambda kv: kv[1])
            else:
                sem_top_theme, sem_top_score = None, None

            if (
                top_theme is None
                and sem_top_theme is not None
                and sem_top_score is not None
                and sem_top_score >= semantic_threshold
            ):
                top_theme = sem_top_theme
                classified_by = "semantic"
        else:
            scores = {}
            sem_top_theme = None
            sem_top_score = None

        for theme, hits in matches.items():
            aggregate[theme] += len(hits)
            aggregate_tasks[theme].append(item["id"])

        # When the semantic fallback supplies the top theme, count it
        # toward the per-theme aggregate as a single "soft" hit so the
        # backlog-wide dominant-theme tally still reflects what the
        # ticket is actually about.
        if classified_by == "semantic" and top_theme is not None:
            aggregate[top_theme] += 1
            aggregate_tasks[top_theme].append(item["id"])

        per_task.append({
            "id": item["id"],
            "description": item["description"],
            "top_theme": top_theme,
            "matched_keywords": top_hits,
            "all_matches": matches,
            "semantic_scores": scores,
            "semantic_top_theme": sem_top_theme,
            "semantic_top_score": (
                round(sem_top_score, 4) if sem_top_score is not None else None
            ),
            "classified_by": classified_by,
        })

    # Rank dominant themes - only themes hit by at least one task qualify.
    ranked = sorted(
        [(t, c) for t, c in aggregate.items() if c > 0],
        key=lambda x: (-x[1], x[0]),
    )
    dominant = [name for name, _ in ranked[:top_themes]]

    # Label each task with a transparent reason string.
    for task in per_task:
        top = task["top_theme"]
        hits = len(task["matched_keywords"])
        source = task["classified_by"]
        if top is None:
            task["label"] = "Potentially misaligned"
            task["reason"] = (
                "No theme keywords matched and the semantic model "
                "did not pass the confidence threshold, so this ticket "
                "doesn't obviously fit any theme of the current sprint "
                "backlog."
                if semantic_enabled else
                "No theme keywords matched, so this ticket doesn't "
                "obviously fit any theme of the current sprint backlog."
            )
        elif source == "semantic":
            task["label"] = (
                "Aligned" if top in dominant else "Neutral"
            )
            score = task["semantic_top_score"]
            task["reason"] = (
                f"Semantic model recovered top theme '{top}' "
                f"(cosine={score:.2f}) for a ticket the lexicon could "
                f"not match. Counted as a soft hit toward the backlog "
                f"theme tally."
            )
        elif top in dominant and hits >= aligned_threshold:
            task["label"] = "Aligned"
            task["reason"] = (
                f"Top theme '{top}' is dominant in this sprint backlog "
                f"({hits} keyword hits, threshold={aligned_threshold})."
            )
        elif top in dominant:
            task["label"] = "Neutral"
            task["reason"] = (
                f"Top theme '{top}' is dominant but only {hits} keyword "
                f"hit(s), below the threshold of {aligned_threshold}."
            )
        else:
            task["label"] = "Neutral"
            task["reason"] = (
                f"Top theme '{top}' is present but not dominant in this "
                f"sprint backlog (dominant themes: {dominant})."
            )

    themes_summary = [
        {"name": name, "hits": count, "tasks": aggregate_tasks[name]}
        for name, count in ranked
    ]

    return {
        "themes": themes_summary,
        "dominant_themes": dominant,
        "semantic_enabled": semantic_enabled,
        "tasks": per_task,
    }
