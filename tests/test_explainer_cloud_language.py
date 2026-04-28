"""Validate that the explainer frames its output in IT / cloud language.

These tests pin down the contract that, for cost-optimization projects,
the report:

    1. Uses cloud-cost phrasing such as "cloud spend",
       "infrastructure scaling", and "cost optimization initiatives"
       instead of raw lexicon names like "scaling" or "cost_optimization".
    2. When the verdict is Significant Drift, EXPLICITLY states why
       cloud costs are not reducing despite an active backlog.
    3. Still works, in plain mode, for non-cost-optimization projects
       (no false advertising of "cloud spend" when the project isn't
       a cost project).

Run from the repo root with::

    python -m unittest discover tests
"""

from __future__ import annotations

import os
import sys
import unittest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from truepath_ai import explain_drift  # noqa: E402


COST_INTENT = (
    "We aim to reduce AWS cloud cost by 25 percent over the next two "
    "quarters by optimizing compute, storage, and database spend. "
    "Rightsize over-provisioned EC2 and RDS resources. Adopt Savings "
    "Plans and Spot instances. Decommission idle services."
)


# ---------------------------------------------------------------------------
# Cost-mode reports must speak fluent cloud / FinOps.
# ---------------------------------------------------------------------------
class CloudPhrasingForCostProjects(unittest.TestCase):
    """Whenever the project intent matches a FinOps theme, the report
    must read like an IT/cloud cost briefing - not like a lexicon dump.
    """

    def _significant_drift_report(self) -> str:
        tasks = [
            # Two cost tickets - "few".
            "Rightsize over-provisioned EC2 instances to reduce monthly compute spend.",
            "Decommission idle RDS databases that have had zero connections for 30 days.",
            # Six growth tickets - the majority.
            "Scale the production EKS cluster from 20 to 40 nodes.",
            "Add 3 read replicas to the orders Aurora cluster.",
            "Roll out the orders microservice to the APAC region.",
            "Deploy the new feature flag service across all environments.",
            "Provision additional Kafka brokers to handle peak traffic.",
            "Increase node group capacity for the staging EKS cluster.",
        ]
        return explain_drift(
            COST_INTENT, tasks, project_title="AWS Cloud Cost Optimization"
        )

    def test_report_uses_cloud_cost_vocabulary(self):
        report = self._significant_drift_report()

        # Core phrases from the spec.
        self.assertIn("cloud spend", report.lower())
        self.assertIn("cost optimization initiatives", report.lower())
        self.assertIn("infrastructure scaling", report.lower())

        # Raw lexicon names must NOT bleed into the user-facing prose.
        # ("cost_optimization" is the internal theme name; the report
        # should always render it as the friendly phrase.)
        self.assertNotIn("cost_optimization", report)

    def test_significant_drift_states_why_costs_are_not_reducing(self):
        report = self._significant_drift_report()

        # The verdict must communicate that cost reduction is failing.
        self.assertIn("OFF TRACK", report)

        # The report must explicitly explain WHY costs are not coming down.
        # The exact wording is a contract test on the explainer copy.
        self.assertIn(
            "cloud costs are NOT reducing despite an active sprint backlog",
            report,
        )

    def test_recommendation_pushes_back_on_capacity_work(self):
        report = self._significant_drift_report()

        # The report (overall) must name the corrective lever (FinOps work)
        # and the cause-and-effect on the AWS bill.
        self.assertIn("cost optimization initiatives", report.lower())
        self.assertIn("AWS bill", report)

    def test_recommendation_lists_neutral_decision_options(self):
        """The Recommendation section must read as decision-support:
        multiple bulleted options, neutral wording, and the three
        signature exits (prioritize FinOps, deprioritize scaling,
        update intent if growth is the goal).
        """
        report = self._significant_drift_report()

        # Pull just the Recommendation section so we don't accidentally
        # match phrases that live in the "Why" section.
        rec_section = report.split("Recommendation\n")[-1]
        rec_lower = rec_section.lower()

        # 1) The section is bulleted with at least three options.
        bullets = [
            line for line in rec_section.splitlines()
            if line.lstrip().startswith("- ")
        ]
        self.assertGreaterEqual(
            len(bullets), 3,
            f"Expected >=3 bulleted options in the Recommendation "
            f"section. Got {len(bullets)}:\n{rec_section}",
        )

        # 2) The three signature options requested by the spec.
        # (a) Prioritize rightsizing + cleanup work.
        self.assertIn("prioritize", rec_lower)
        self.assertIn("rightsizing", rec_lower)
        self.assertTrue(
            "decommission" in rec_lower or "cleanup" in rec_lower,
            f"Expected a cleanup/decommission option:\n{rec_section}",
        )
        # (b) Deprioritize scaling unless required.
        self.assertIn("deprioritize", rec_lower)
        self.assertIn("infrastructure scaling", rec_lower)
        self.assertIn("unless", rec_lower)
        # (c) Explicitly update project intent if growth is the goal.
        self.assertIn("project intent", rec_lower)
        self.assertIn("goal", rec_lower)

        # 3) Tone: neutral, decision-support - not prescriptive.
        for forbidden in ("you must", "you should", "must immediately",
                          "must cancel", "you have to"):
            self.assertNotIn(
                forbidden, rec_lower,
                f"Recommendation should be neutral; found {forbidden!r}",
            )

    def test_partial_drift_flags_risk_in_cloud_terms(self):
        # 3 cost tickets + 3 scattered off-intent tickets ->
        # aligned_share = 0.50 (middle band) and no single off-intent
        # theme reaches the 35% direction threshold, so "Partial Drift".
        tasks = [
            # 3 cost tickets:
            "Rightsize over-provisioned EC2 instances to reduce monthly compute spend.",
            "Apply Savings Plans across steady-state EC2 usage to reduce on-demand spend.",
            "Decommission idle RDS databases to eliminate unused spend.",
            # 3 off-intent tickets, scattered themes:
            "Scale the production EKS cluster from 20 to 40 nodes.",
            "Roll out the orders microservice to the APAC region.",
            "Refactor the legacy notifications microservice to migrate off the deprecated SQS queue.",
        ]
        report = explain_drift(
            COST_INTENT, tasks, project_title="AWS Cloud Cost Optimization"
        )
        # Verdict and prose must talk about RISK, not OFF TRACK.
        self.assertIn("AT RISK", report)
        self.assertIn("Cloud cost reduction is at risk", report)

    def test_aligned_report_uses_cloud_language_too(self):
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
        report = explain_drift(
            COST_INTENT, tasks, project_title="AWS Cloud Cost Optimization"
        )

        self.assertIn("ON TRACK", report)
        self.assertIn("cost optimization initiatives", report.lower())

    def test_table_uses_friendly_phrases_not_internal_keys(self):
        """The 'Backlog theme distribution' table must render every
        theme as a friendly phrase, not as the snake_case lexicon key.
        """
        report = self._significant_drift_report()

        # The distribution and contributor lines should reference
        # "infrastructure scaling", not the bare theme name.
        self.assertIn("infrastructure scaling", report)


# ---------------------------------------------------------------------------
# Plain-mode reports must NOT pretend to be cloud-cost reports.
# ---------------------------------------------------------------------------
class PlainModeStaysGeneric(unittest.TestCase):
    """When the project intent has nothing to do with FinOps, the
    explainer should fall back to its generic phrasing - we don't want
    a UX project's report to randomly mention 'cloud spend'.
    """

    def test_non_cost_project_does_not_mention_cloud_spend(self):
        intent = (
            "Redesign the customer onboarding flow to improve activation. "
            "Simplify the signup screen and add a guided product tour."
        )
        tasks = [
            "Redesign the homepage with a refreshed color palette.",
            "Add a guided onboarding tour with tooltips on each screen.",
            "Run a usability study on the new signup layout.",
        ]
        report = explain_drift(intent, tasks, project_title="Onboarding redesign")

        self.assertNotIn("cloud spend", report.lower())
        self.assertNotIn("AWS bill", report)
        self.assertNotIn("OFF TRACK", report)


if __name__ == "__main__":
    unittest.main()
