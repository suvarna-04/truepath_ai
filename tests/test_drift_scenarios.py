"""Drift-detection validation suite.

These tests pin down the exact, deterministic behaviour we want from
``detect_intent_drift`` for an AWS Cloud Cost Optimization project.

The two scenarios the product spec calls out are:

  1. The majority of tasks involve scaling or expansion.
  2. Few or no tasks directly reduce cost.

Each test maps onto the rule it exercises in ``drift_detector.py`` so a
future change to thresholds or lexicon either keeps these green or fails
loudly with an explainable reason.

Run from the repo root with::

    python -m unittest discover tests
"""

from __future__ import annotations

import os
import sys
import unittest

# Make `truepath_ai` importable when the tests are invoked from the
# repo root (CI / `python -m unittest discover tests`) without forcing
# users to install the package.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from truepath_ai import (  # noqa: E402  (sys.path tweak above is intentional)
    INTENT_THEME_BLOCKLIST,
    detect_intent_drift,
)


COST_OPTIMIZATION_INTENT = (
    "We aim to reduce AWS cloud cost by 25 percent over the next two "
    "quarters by optimizing compute, storage, and database spend across "
    "all production accounts. Rightsize over-provisioned EC2, RDS, and "
    "EBS resources. Adopt Savings Plans and Spot instances for "
    "steady-state and batch workloads. Improve cost visibility through "
    "CloudWatch dashboards and AWS Budgets alerts. Decommission idle "
    "services that have had no activity for 30 days."
)


# ---------------------------------------------------------------------------
# Significant-drift scenarios - the two cases called out in the spec.
# ---------------------------------------------------------------------------
class SignificantDriftScenarios(unittest.TestCase):
    """The backlog is dominated by scaling/expansion work and cost
    work is absent or sparse. We expect ``Significant Drift`` every time.
    """

    def test_all_tasks_are_scaling_or_expansion(self):
        """Zero cost-aligned tasks -> proportion rule fires immediately."""
        tasks = [
            "Scale the production EKS cluster from 20 to 40 nodes.",
            "Add 3 read replicas to the orders Aurora cluster.",
            "Roll out the orders microservice to the APAC region.",
            "Deploy the new feature flag service across all environments.",
            "Provision additional Kafka brokers to handle peak traffic.",
            "Increase node group capacity for the staging EKS cluster.",
        ]

        verdict = detect_intent_drift(COST_OPTIMIZATION_INTENT, tasks)

        self.assertEqual(verdict["drift_level"], "Significant Drift")
        self.assertEqual(verdict["counts"]["aligned"], 0)
        self.assertEqual(verdict["aligned_share"], 0.0)
        # The off-intent direction must be one of the blocklisted growth
        # themes - that's the whole point of the scenario.
        self.assertIn(
            verdict["drift_direction"],
            {"scaling", "expansion"},
            f"Expected scaling/expansion drift, got {verdict['drift_direction']!r}",
        )

    def test_few_cost_tasks_against_growth_majority(self):
        """A couple of cost tickets, six growth tickets -> proportion rule."""
        tasks = [
            # Two cost-aligned ("few") tickets:
            "Rightsize over-provisioned EC2 instances to reduce monthly compute spend.",
            "Decommission idle RDS databases that have had zero connections for 30 days.",
            # Six growth tickets - the majority:
            "Scale the production EKS cluster from 20 to 40 nodes.",
            "Add 3 read replicas to the orders Aurora cluster.",
            "Roll out the orders microservice to the APAC region.",
            "Deploy the new feature flag service across all environments.",
            "Provision additional Kafka brokers to handle peak traffic.",
            "Increase node group capacity for the staging EKS cluster.",
        ]

        verdict = detect_intent_drift(COST_OPTIMIZATION_INTENT, tasks)

        self.assertEqual(verdict["drift_level"], "Significant Drift")
        # 2/8 = 0.25 - safely below the 0.40 drift threshold.
        self.assertLess(verdict["aligned_share"], 0.40)
        self.assertEqual(verdict["counts"]["aligned"], 2)
        self.assertEqual(verdict["counts"]["off_intent"], 6)

    def test_borderline_majority_escalates_via_direction(self):
        """50/50 split is borderline by proportion, but every off-intent
        task pulls in the same direction (`scaling`), so the
        direction_threshold (35%) escalates the verdict to Significant.
        """
        tasks = [
            # 5 cost-aligned tickets:
            "Rightsize over-provisioned EC2 instances to reduce monthly compute spend.",
            "Apply Savings Plans across steady-state EC2 usage to reduce on-demand spend.",
            "Decommission idle RDS databases to eliminate unused spend.",
            "Delete unattached EBS volumes to clean up unused resources.",
            "Migrate batch ETL workloads from on-demand to Spot instances to cut compute cost.",
            # 5 off-intent tickets, all scaling:
            "Scale the production EKS cluster from 20 to 40 nodes.",
            "Add 3 read replicas to the orders Aurora cluster.",
            "Increase node group capacity for the staging EKS cluster.",
            "Add capacity headroom to the payments microservice replicas.",
            "Scale up the recommendations service shards to handle peak traffic.",
        ]

        verdict = detect_intent_drift(COST_OPTIMIZATION_INTENT, tasks)

        self.assertEqual(verdict["drift_level"], "Significant Drift")
        self.assertEqual(verdict["aligned_share"], 0.5)
        self.assertEqual(verdict["drift_direction"], "scaling")


# ---------------------------------------------------------------------------
# Negative scenarios - making sure we do NOT cry wolf.
# ---------------------------------------------------------------------------
class NonSignificantScenarios(unittest.TestCase):
    """Sanity checks: when the backlog actually matches intent, or the
    off-intent work is scattered, we must not flag Significant Drift.
    """

    def test_aligned_when_backlog_is_cost_focused(self):
        tasks = [
            "Rightsize over-provisioned EC2 instances to reduce monthly compute spend.",
            "Apply Savings Plans across steady-state EC2 usage to reduce spend.",
            "Decommission idle RDS databases to eliminate unused spend.",
            "Delete unattached EBS volumes to clean up unused resources.",
            "Migrate ETL workloads from on-demand to Spot instances to cut cost.",
            "Enable S3 lifecycle policies to move cold objects to Glacier and reduce cost.",
            "Tune Lambda memory configurations to reduce billed duration and lower invocation cost.",
            "Set up a CloudWatch dashboard for monthly AWS spend alerts.",
            "Add AWS Budgets alerts to flag forecast cost overruns.",
        ]

        verdict = detect_intent_drift(COST_OPTIMIZATION_INTENT, tasks)

        self.assertEqual(verdict["drift_level"], "Aligned")
        self.assertGreaterEqual(verdict["aligned_share"], 0.70)

    def test_partial_drift_when_off_intent_is_scattered(self):
        """50% aligned + four DIFFERENT off-intent themes (one task each)
        -> middle band, no direction concentrates >= 35%, so Partial.
        """
        tasks = [
            # 4 cost-aligned tickets:
            "Rightsize over-provisioned EC2 instances to reduce monthly compute spend.",
            "Apply Savings Plans for steady-state EC2 to reduce spend.",
            "Decommission idle RDS databases to eliminate unused spend.",
            "Delete unattached EBS volumes to clean up unused resources.",
            # 4 off-intent tickets, each on a different theme:
            "Refactor the legacy notifications microservice to migrate off the deprecated SQS queue.",
            "Rotate IAM access keys used by the deployment pipeline and store them in AWS Secrets Manager.",
            "Redesign the internal admin console homepage with a refreshed color palette.",
            "Spike a prototype VR onboarding experience for new platform engineering hires.",
        ]

        verdict = detect_intent_drift(COST_OPTIMIZATION_INTENT, tasks)

        self.assertEqual(verdict["drift_level"], "Partial Drift")
        self.assertEqual(verdict["aligned_share"], 0.5)
        # No single off-intent theme should hit the direction threshold.
        for entry in verdict["task_theme_distribution"]:
            if not entry["in_intent"]:
                self.assertLess(entry["share"], 0.35)


# ---------------------------------------------------------------------------
# Contracts the scenarios above depend on - if any of these break the
# tests above are no longer meaningful.
# ---------------------------------------------------------------------------
class IntentThemeBlocklistContract(unittest.TestCase):
    """Scaling, expansion, and maintenance must never be treated as
    intent themes for a cost-optimization project. Without this
    contract the borderline scenario above would silently classify
    growth tickets as 'aligned'.
    """

    def test_blocklist_excludes_growth_and_maintenance(self):
        self.assertIn("scaling", INTENT_THEME_BLOCKLIST)
        self.assertIn("expansion", INTENT_THEME_BLOCKLIST)
        self.assertIn("maintenance", INTENT_THEME_BLOCKLIST)


class DriftVerdictExplainability(unittest.TestCase):
    """Every drift verdict must carry a human-readable reason that
    references the numbers behind the decision - that is the whole
    'explainable, deterministic' contract.
    """

    def test_significant_drift_reason_mentions_share_and_direction(self):
        tasks = [
            "Scale the production EKS cluster from 20 to 40 nodes.",
            "Add 3 read replicas to the orders Aurora cluster.",
            "Increase node group capacity for the staging EKS cluster.",
            "Add capacity headroom to the payments microservice replicas.",
            "Scale up the recommendations service shards to handle peak traffic.",
        ]

        verdict = detect_intent_drift(COST_OPTIMIZATION_INTENT, tasks)

        self.assertEqual(verdict["drift_level"], "Significant Drift")
        self.assertIsInstance(verdict["reason"], str)
        # The reason must show the percentage AND name the off-intent
        # direction, so a reviewer can audit the call without rerunning.
        self.assertIn("%", verdict["reason"])
        self.assertIn(verdict["drift_direction"], verdict["reason"])


if __name__ == "__main__":
    unittest.main()
