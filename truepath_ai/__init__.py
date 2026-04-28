"""TruePath AI - Detect when ongoing tasks drift away from project intent.

Public API at a glance::

    from truepath_ai import (
        parse_intent,             # text  -> structured intent dict
        extract_intent_themes,    # text  -> intent themes (with FinOps rules)
        INTENT_THEME_BLOCKLIST,   # themes never promoted as intent themes
        classify_tasks,           # tasks -> per-task labels + dominant themes
        find_themes,              # text  -> dominant themes (raw lexicon)
        detect_intent_drift,      # intent + tasks -> drift verdict
        explain_drift,            # intent + tasks -> demo-ready text report
        THEME_LEXICON,            # rule-book: theme name -> trigger keywords
    )

Every function above is rule-based and explainable. No machine learning,
no model weights - just keyword lists, regex, and proportion thresholds
that you can read and tune in a few minutes.
"""

__version__ = "0.3.0"

from .intent_analyzer import (
    INTENT_THEME_BLOCKLIST,
    extract_intent_themes,
    parse_intent,
)
from .task_analyzer import THEME_LEXICON, classify_tasks, find_themes
from .drift_detector import detect_intent_drift
from .explainer import explain_drift

__all__ = [
    "parse_intent",
    "extract_intent_themes",
    "INTENT_THEME_BLOCKLIST",
    "classify_tasks",
    "find_themes",
    "THEME_LEXICON",
    "detect_intent_drift",
    "explain_drift",
]
