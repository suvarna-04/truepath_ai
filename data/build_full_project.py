"""Build a single TruePath AI project JSON from the three sprint workbooks.

Reads ``data/sprint_1.csv.xlsx``, ``sprint_2.csv.xlsx``, and
``sprint_3.csv.xlsx`` (all share the same 21-column schema), normalises
every row, and writes ``data/full_project.json`` with:

  * project metadata (id, title, intent, goals)
  * a ``sprints`` array describing each sprint window
  * a ``tasks`` array of all 90 tickets, with **every** column from the
    spreadsheets preserved as a structured field

The shape stays compatible with ``truepath_ai.main._coerce_project``:
``id`` and ``description`` are the keys the classifier reads. The
``description`` we emit is ``"{title}. {description}"`` so titles also
contribute to lexicon hits; the original ticket text is preserved under
``description_raw``. Every other column from the workbooks is attached
as metadata for the Streamlit dashboard and any downstream tooling.

Re-run after editing any of the workbooks::

    python data/build_full_project.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
WORKBOOKS = [
    DATA_DIR / "sprint_1.csv.xlsx",
    DATA_DIR / "sprint_2.csv.xlsx",
    DATA_DIR / "sprint_3.csv.xlsx",
]
OUTPUT = DATA_DIR / "full_project.json"


# ---------------------------------------------------------------------------
# Project intent + goals - written here so the JSON can be loaded as-is.
# ---------------------------------------------------------------------------
PROJECT_TITLE = "AWS Cloud Cost Optimization 2026"
PROJECT_INTENT = (
    "Reduce AWS cloud cost by 25 percent across our production workloads "
    "over the next two quarters by rightsizing over-provisioned EC2, RDS, "
    "and EBS resources, adopting Savings Plans and Reserved Instances for "
    "steady-state workloads, decommissioning idle services and unused "
    "storage, and tiering cold data to Glacier and Intelligent-Tiering."
)
PROJECT_GOALS = [
    "Rightsize over-provisioned EC2, RDS, EBS, and Lambda resources flagged by AWS Compute Optimizer and Trusted Advisor",
    "Adopt Savings Plans and 3-year Reserved Instances for steady-state EC2, EKS, and Aurora workloads",
    "Decommission idle clusters, retired services, and unattached resources older than 30 days",
    "Apply S3 lifecycle and Intelligent-Tiering policies to move cold data to Glacier",
    "Track monthly run-rate against the 25 percent reduction target via CloudWatch dashboards and AWS Budgets alerts",
]


# ---------------------------------------------------------------------------
# Calendar mapping - re-project the three sprints onto April / May / June
# 2026 so the dashboard can present a clean month-by-month timeline.
# Each entry overrides the sprint name + window. The ticket-level
# `created_at` is kept from the original spreadsheet but the sprint
# context dates are normalized.
# ---------------------------------------------------------------------------
SPRINT_TO_MONTH: Dict[int, Dict[str, str]] = {
    1: {
        "month": 4,
        "month_label": "April 2026",
        "sprint_name": "April 2026 - Cost Optimization Kickoff",
        "sprint_start": "2026-04-06",
        "sprint_end":   "2026-04-17",
    },
    2: {
        "month": 5,
        "month_label": "May 2026",
        "sprint_name": "May 2026 - Mixed Capacity & Cost",
        "sprint_start": "2026-05-04",
        "sprint_end":   "2026-05-15",
    },
    3: {
        "month": 6,
        "month_label": "June 2026",
        "sprint_name": "June 2026 - Promo Capacity & New Region",
        "sprint_start": "2026-06-01",
        "sprint_end":   "2026-06-12",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_iso_date(value: Any) -> Any:
    """Convert pandas / numpy / datetime values into ISO-8601 date strings."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "isoformat"):
        try:
            return value.date().isoformat()
        except AttributeError:
            return value.isoformat()
    return str(value)


def _split_pipe(value: Any) -> List[str]:
    """Pipe-delimited string -> trimmed list. Empty / NaN -> []."""
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.split("|") if part.strip()]


def _clean_scalar(value: Any) -> Any:
    """Convert NaN/NaT to None, leave everything else untouched."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if pd.isna(value):
        return None
    return value


def _row_to_ticket(row: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize one spreadsheet row into a TruePath AI ticket dict."""
    title = str(row.get("title", "")).strip()
    description = str(row.get("description", "")).strip()

    # Combined text the rule engine sees - includes the title so a
    # ticket like "Rightsize ..." still hits the rightsizing lexicon
    # even if the description happens not to repeat the verb.
    if title and description:
        classifier_text = f"{title}. {description}"
    elif title:
        classifier_text = title
    else:
        classifier_text = description

    sprint_number = (
        int(row["sprint_number"]) if pd.notna(row.get("sprint_number")) else None
    )
    month_meta = SPRINT_TO_MONTH.get(sprint_number, {})
    sprint_name = month_meta.get("sprint_name") or _clean_scalar(row.get("sprint_name"))
    sprint_start = month_meta.get("sprint_start") or _to_iso_date(row.get("sprint_start"))
    sprint_end = month_meta.get("sprint_end") or _to_iso_date(row.get("sprint_end"))

    return {
        "id": str(row["ticket_id"]).strip(),
        "title": title,
        "description": classifier_text,
        "description_raw": description,
        "type": _clean_scalar(row.get("type")),
        "status": _clean_scalar(row.get("status")),
        "priority": _clean_scalar(row.get("priority")),
        "story_points": (
            int(row["story_points"])
            if pd.notna(row.get("story_points"))
            else None
        ),
        "assignee": _clean_scalar(row.get("assignee")),
        "reporter": _clean_scalar(row.get("reporter")),
        "labels": _split_pipe(row.get("labels")),
        "components": _split_pipe(row.get("components")),
        "aws_services": _split_pipe(row.get("aws_services")),
        "epic_link": _clean_scalar(row.get("epic_link")),
        "ground_truth_theme": _clean_scalar(row.get("ground_truth_theme")),
        "estimated_monthly_impact_usd": (
            int(row["estimated_monthly_impact_usd"])
            if pd.notna(row.get("estimated_monthly_impact_usd"))
            else None
        ),
        "created_at": _to_iso_date(row.get("created_at")),
        "sprint_number": sprint_number,
        "sprint_name": sprint_name,
        "sprint_start": sprint_start,
        "sprint_end": sprint_end,
        "month": month_meta.get("month"),
        "month_label": month_meta.get("month_label"),
    }


def _sprint_summary(df: pd.DataFrame, sprint_number: int) -> Dict[str, Any]:
    """Per-sprint summary block: window, ticket count, projected impact."""
    rows = df[df["sprint_number"] == sprint_number]
    impacts = rows["estimated_monthly_impact_usd"].dropna()
    month_meta = SPRINT_TO_MONTH.get(sprint_number, {})
    return {
        "sprint_number": sprint_number,
        "month": month_meta.get("month"),
        "month_label": month_meta.get("month_label"),
        "sprint_name": (
            month_meta.get("sprint_name")
            or str(rows["sprint_name"].iloc[0])
        ),
        "sprint_start": (
            month_meta.get("sprint_start")
            or _to_iso_date(rows["sprint_start"].iloc[0])
        ),
        "sprint_end": (
            month_meta.get("sprint_end")
            or _to_iso_date(rows["sprint_end"].iloc[0])
        ),
        "ticket_count": int(len(rows)),
        "projected_monthly_impact_usd": int(impacts.sum()) if not impacts.empty else 0,
        "savings_tickets": int((impacts < 0).sum()),
        "spend_increasing_tickets": int((impacts > 0).sum()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    frames: List[pd.DataFrame] = []
    for wb in WORKBOOKS:
        if not wb.exists():
            raise FileNotFoundError(wb)
        # Each workbook's first sheet holds the data; ignore any extras.
        xls = pd.ExcelFile(wb)
        sheet = xls.sheet_names[0]
        frames.append(pd.read_excel(wb, sheet_name=sheet))

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["sprint_number", "ticket_id"]).reset_index(drop=True)

    project_ids = sorted(df["project_id"].dropna().unique().tolist())
    project_id = project_ids[0] if project_ids else "UNKNOWN"

    sprints = [
        _sprint_summary(df, int(n))
        for n in sorted(df["sprint_number"].unique())
    ]
    tasks = [_row_to_ticket(row) for row in df.to_dict(orient="records")]

    total_impact = sum(
        t["estimated_monthly_impact_usd"]
        for t in tasks
        if t["estimated_monthly_impact_usd"] is not None
    )

    project = {
        "project_id": project_id,
        "title": PROJECT_TITLE,
        "intent": PROJECT_INTENT,
        "goals": PROJECT_GOALS,
        "summary": {
            "total_tickets": len(tasks),
            "total_sprints": len(sprints),
            "projected_monthly_impact_usd": total_impact,
            "savings_tickets": sum(
                1 for t in tasks
                if (t["estimated_monthly_impact_usd"] or 0) < 0
            ),
            "spend_increasing_tickets": sum(
                1 for t in tasks
                if (t["estimated_monthly_impact_usd"] or 0) > 0
            ),
            "ground_truth_theme_distribution": {
                (str(theme) if pd.notna(theme) else None): int(count)
                for theme, count in (
                    df["ground_truth_theme"]
                    .value_counts(dropna=False)
                    .items()
                )
            },
        },
        "sprints": sprints,
        "tasks": tasks,
    }

    OUTPUT.write_text(
        json.dumps(project, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size:,} bytes)")
    print(f"  project_id : {project_id}")
    print(f"  title      : {PROJECT_TITLE}")
    print(f"  sprints    : {len(sprints)}")
    print(f"  tickets    : {len(tasks)}")
    print(f"  net impact : ${total_impact:+,d} / month")
    print()
    for sp in sprints:
        print(
            f"    Sprint {sp['sprint_number']}: {sp['ticket_count']} tickets, "
            f"impact ${sp['projected_monthly_impact_usd']:+,d}/mo "
            f"({sp['savings_tickets']} save, "
            f"{sp['spend_increasing_tickets']} add)"
        )


if __name__ == "__main__":
    main()
