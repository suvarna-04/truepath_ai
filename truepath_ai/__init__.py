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
        SemanticThemeClassifier,  # AI/ML layer: sentence-embedding theme classifier
        get_default_classifier,   # process-wide singleton (lazy ML load)
    )

The pipeline is a hybrid AI + rules system. The lexicon, regex patterns,
and proportion thresholds are the deterministic backbone you can read
off the page; the `SemanticThemeClassifier` is a sentence-transformer
embedding layer that recovers tickets phrased in their own words. Every
verdict ships with the evidence (matched keywords, semantic scores,
ticket-level reasons) that produced it - AI-powered, never a black box.
"""

__version__ = "0.4.0"

from .intent_analyzer import (
    INTENT_THEME_BLOCKLIST,
    extract_intent_themes,
    parse_intent,
)
from .task_analyzer import THEME_LEXICON, classify_tasks, find_themes
from .drift_detector import detect_intent_drift
from .explainer import explain_drift
from .semantic_classifier import (
    SemanticThemeClassifier,
    THEME_PROTOTYPE_DESCRIPTIONS,
    get_default_classifier,
)

__all__ = [
    "parse_intent",
    "extract_intent_themes",
    "INTENT_THEME_BLOCKLIST",
    "classify_tasks",
    "find_themes",
    "THEME_LEXICON",
    "detect_intent_drift",
    "explain_drift",
    "SemanticThemeClassifier",
    "THEME_PROTOTYPE_DESCRIPTIONS",
    "get_default_classifier",
]
