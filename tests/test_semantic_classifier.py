"""Tests for the AI / ML layer (sentence-embedding theme classifier).

Two layers of guarantees are pinned down here:

  1. *Offline contract* - the rule-based pipeline keeps working even
     when `sentence-transformers` / `torch` are not installed. The
     classifier reports `is_available()` as False, and
     `classify_tasks(use_semantic=True)` simply skips the ML branch
     and returns lexicon-only results. This is the contract that lets
     judges run the demo on a barebones Python install.

  2. *Online contract* - when the embedding model IS available, the
     classifier:
        * scores tickets against every theme,
        * recovers a sensible top theme for paraphrased tickets that
          the lexicon would otherwise miss, and
        * exposes the per-ticket semantic evidence on every result.
     These tests are skipped automatically when the ML stack is not
     installed, so the suite stays green on minimal environments.

Run from the repo root with::

    python -m unittest discover tests
    # or
    pytest tests
"""

from __future__ import annotations

import importlib
import os
import sys
import unittest
from unittest import mock

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from truepath_ai import (  # noqa: E402
    SemanticThemeClassifier,
    THEME_LEXICON,
    THEME_PROTOTYPE_DESCRIPTIONS,
    classify_tasks,
)


def _ml_stack_available() -> bool:
    """True iff `sentence-transformers` (and friends) load cleanly."""
    try:
        importlib.import_module("sentence_transformers")
    except Exception:
        return False
    try:
        importlib.import_module("torch")
    except Exception:
        return False
    return True


# ---------------------------------------------------------------------------
# Module-level / contract tests - run on every environment.
# ---------------------------------------------------------------------------
class SemanticClassifierContract(unittest.TestCase):
    """Behaviour the classifier must honour regardless of whether the
    ML stack is installed.
    """

    def test_every_theme_has_a_prototype_description(self):
        # If a theme exists in the lexicon, the embedding layer needs a
        # prototype description for it - otherwise it would silently
        # default to the bare snake_case theme name when building the
        # prototype matrix.
        for theme in THEME_LEXICON:
            self.assertIn(
                theme,
                THEME_PROTOTYPE_DESCRIPTIONS,
                f"Theme {theme!r} is missing from "
                f"THEME_PROTOTYPE_DESCRIPTIONS.",
            )

    def test_classifier_reports_unavailable_when_import_fails(self):
        """Simulate the 'no torch installed' case and verify graceful
        degradation: `is_available()` is False, scoring returns empty
        dicts, and `top_theme` returns None.
        """
        clf = SemanticThemeClassifier()

        # Force the import inside `_ensure_model_loaded` to fail.
        with mock.patch.dict(sys.modules, {"sentence_transformers": None}):
            with self.assertWarns(RuntimeWarning):
                self.assertFalse(clf.is_available())

        self.assertEqual(clf.score_themes("rightsize the cluster"), {})
        self.assertIsNone(clf.top_theme("rightsize the cluster"))
        self.assertIsNotNone(clf.load_error())

    def test_classify_tasks_falls_back_to_lexicon_when_ml_offline(self):
        """`classify_tasks(use_semantic=True)` with an offline model
        must produce the same lexicon-only verdicts as
        `use_semantic=False`."""
        tasks = [
            "Rightsize over-provisioned EC2 instances to reduce monthly compute spend.",
            "Scale the production EKS cluster from 20 to 40 nodes.",
        ]

        offline_clf = SemanticThemeClassifier()
        with mock.patch.dict(sys.modules, {"sentence_transformers": None}):
            # Trigger the load attempt so is_available() returns False.
            offline_clf.is_available()

        result_with = classify_tasks(
            tasks,
            use_semantic=True,
            semantic_classifier=offline_clf,
        )
        result_without = classify_tasks(tasks, use_semantic=False)

        self.assertFalse(result_with["semantic_enabled"])
        self.assertEqual(
            [t["top_theme"] for t in result_with["tasks"]],
            [t["top_theme"] for t in result_without["tasks"]],
        )
        # And the per-task evidence still exposes the new fields, even
        # when the model was offline - they're just empty / None.
        for t in result_with["tasks"]:
            self.assertIn("semantic_scores", t)
            self.assertIn("semantic_top_theme", t)
            self.assertIn("classified_by", t)
            self.assertEqual(t["semantic_scores"], {})
            self.assertIsNone(t["semantic_top_theme"])
            self.assertIn(t["classified_by"], {"lexicon", "none"})


# ---------------------------------------------------------------------------
# Online tests - only run when the ML stack is actually installed.
# ---------------------------------------------------------------------------
@unittest.skipUnless(
    _ml_stack_available(),
    "sentence-transformers / torch not installed; skipping ML-online tests.",
)
class SemanticClassifierOnline(unittest.TestCase):
    """Behaviour we expect when the embedding model is loadable.

    The model load can be slow (the first call downloads weights), so
    we share one classifier across the test methods of this class.
    """

    @classmethod
    def setUpClass(cls):
        cls.clf = SemanticThemeClassifier(eager=True)
        if not cls.clf.is_available():
            raise unittest.SkipTest(
                f"Embedding model failed to initialise: "
                f"{cls.clf.load_error()}"
            )

    def test_score_themes_returns_one_score_per_theme(self):
        scores = self.clf.score_themes(
            "Rightsize over-provisioned EC2 instances to reduce spend."
        )
        self.assertEqual(set(scores.keys()), set(THEME_LEXICON.keys()))
        for theme, score in scores.items():
            self.assertIsInstance(score, float)
            # Cosine similarity is bounded; with normalized vectors
            # we expect roughly [-1, 1] in practice (the prototypes
            # are made of positive sentences so most scores skew
            # positive, but we don't depend on a tighter bound).
            self.assertGreaterEqual(score, -1.0)
            self.assertLessEqual(score, 1.0)

    def test_top_theme_picks_a_finops_theme_for_paraphrased_cost_ticket(self):
        """A ticket that MEANS rightsizing/cleanup but uses none of the
        lexicon keywords must still land in the FinOps cluster.
        """
        text = (
            "Shrink the over-allocated Postgres footprint by moving it "
            "onto a smaller, cheaper database instance."
        )
        top = self.clf.top_theme(text)
        self.assertIn(
            top,
            {"rightsizing", "cleanup", "cost_optimization", "efficiency"},
            f"Expected a FinOps theme for paraphrased cost ticket; got {top!r}.",
        )

    def test_classify_tasks_recovers_unmatched_ticket_via_semantic(self):
        """`classify_tasks` should rescue a ticket the lexicon misses."""
        tasks = [
            # Ticket whose words sidestep the lexicon entirely but
            # semantically matches the FinOps cluster.
            "Shrink the over-allocated Postgres footprint by moving it "
            "onto a smaller, cheaper database instance.",
        ]

        result = classify_tasks(tasks, use_semantic=True)
        self.assertTrue(result["semantic_enabled"])
        self.assertEqual(len(result["tasks"]), 1)

        ticket = result["tasks"][0]
        self.assertEqual(ticket["matched_keywords"], [])
        self.assertIsNotNone(ticket["top_theme"])
        self.assertEqual(ticket["classified_by"], "semantic")
        self.assertIn(
            ticket["top_theme"],
            {"rightsizing", "cleanup", "cost_optimization", "efficiency"},
            f"Expected a FinOps theme; got {ticket['top_theme']!r}.",
        )
        # Semantic evidence must be present in the per-task dict.
        self.assertGreater(len(ticket["semantic_scores"]), 0)
        self.assertIsNotNone(ticket["semantic_top_score"])
        self.assertGreaterEqual(ticket["semantic_top_score"], 0.30)

    def test_lexicon_match_outranks_semantic_when_both_signal(self):
        """Determinism contract: when the lexicon already matches a
        ticket, the lexicon top theme wins. The semantic score is
        recorded as evidence but does not override the rule.
        """
        tasks = [
            "Rightsize over-provisioned EC2 instances to reduce monthly compute spend.",
        ]
        result = classify_tasks(tasks, use_semantic=True)
        ticket = result["tasks"][0]
        self.assertEqual(ticket["classified_by"], "lexicon")
        self.assertGreater(len(ticket["matched_keywords"]), 0)
        # Even so, the semantic scores must be filled in - that's the
        # AI-evidence contract.
        self.assertGreater(len(ticket["semantic_scores"]), 0)


if __name__ == "__main__":
    unittest.main()
