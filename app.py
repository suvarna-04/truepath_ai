"""TruePath AI - Streamlit dashboard.

A polished, interactive UI for the rule-based intent-vs-execution drift
detector. Reuses the public API exported by ``truepath_ai`` directly -
no glue code, no API layer, no ML.

Run it with::

    streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import plotly.graph_objects as go
import streamlit as st

from truepath_ai import (
    classify_tasks,
    detect_intent_drift,
    explain_drift,
    extract_intent_themes,
    parse_intent,
)
from truepath_ai.explainer import (
    COST_INTENT_THEMES,
    SPEND_INCREASING_THEMES,
    THEME_PHRASES,
    _is_cost_project,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_PATH = Path(__file__).resolve().parent / "data" / "sample_project.json"

ACCENT_ALIGNED = "#2dd4bf"   # teal
ACCENT_PARTIAL = "#f59e0b"   # amber
ACCENT_DRIFT = "#ef4444"     # red
ACCENT_NEUTRAL = "#64748b"   # slate
SURFACE_BG = "#111a2e"
APP_BG = "#0b1220"
MUTED_TEXT = "#94a3b8"

VERDICT_STYLE: Dict[str, Dict[str, str]] = {
    "Aligned": {
        "color": ACCENT_ALIGNED,
        "headline_plain": "ALIGNED",
        "headline_cost": "ON TRACK",
        "subline_plain": "Sprint backlog matches the project intent.",
        "subline_cost": "Cloud cost reduction is being addressed.",
    },
    "Partial Drift": {
        "color": ACCENT_PARTIAL,
        "headline_plain": "PARTIAL DRIFT",
        "headline_cost": "AT RISK",
        "subline_plain": "Some tickets are pulling away from intent.",
        "subline_cost": "Cloud spend reduction is partially off-course.",
    },
    "Significant Drift": {
        "color": ACCENT_DRIFT,
        "headline_plain": "SIGNIFICANT DRIFT",
        "headline_cost": "OFF TRACK",
        "subline_plain": "Most of the work is no longer aligned with the project intent.",
        "subline_cost": "Cloud costs will not reduce on the current sprint backlog.",
    },
}


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="TruePath AI - Intent Drift Detector",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Custom CSS - the "pleasant" polish layer.
# Streamlit's default styling is fine; this just sharpens the typography,
# rounds the cards, and tightens spacing so the dashboard feels designed.
# ---------------------------------------------------------------------------

st.markdown(
    f"""
    <style>
    /* Hide the deploy / hamburger chrome - looks cleaner for a demo. */
    #MainMenu {{ visibility: hidden; }}
    footer    {{ visibility: hidden; }}
    header    {{ visibility: hidden; }}

    .block-container {{
        padding-top: 1.4rem;
        padding-bottom: 3rem;
        max-width: 1280px;
    }}

    /* App-wide typography tweaks. */
    html, body, [class*="css"] {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                     "Helvetica Neue", Arial, sans-serif;
    }}

    /* Header strip. */
    .tp-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.4rem 0 1.0rem 0;
        border-bottom: 1px solid rgba(148, 163, 184, 0.15);
        margin-bottom: 1.4rem;
    }}
    .tp-header .tp-brand {{
        display: flex; align-items: center; gap: 0.7rem;
    }}
    .tp-header .tp-logo {{
        width: 38px; height: 38px; border-radius: 10px;
        background: linear-gradient(135deg, {ACCENT_ALIGNED}, #6366f1);
        display: flex; align-items: center; justify-content: center;
        font-size: 20px;
    }}
    .tp-header h1 {{
        font-size: 1.35rem; margin: 0; letter-spacing: -0.01em;
        color: #e6edf7;
    }}
    .tp-header .tp-tag {{
        color: {MUTED_TEXT}; font-size: 0.85rem; margin-top: 2px;
    }}
    .tp-header .tp-badges {{ display: flex; gap: 6px; }}
    .tp-pill {{
        font-size: 0.72rem; padding: 4px 10px; border-radius: 999px;
        border: 1px solid rgba(148, 163, 184, 0.30);
        color: {MUTED_TEXT}; background: rgba(255,255,255,0.02);
        letter-spacing: 0.04em; text-transform: uppercase;
    }}

    /* Generic card. */
    .tp-card {{
        background: {SURFACE_BG};
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 14px;
        padding: 1.1rem 1.25rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.20);
    }}

    /* Hero verdict card - the demo centerpiece. */
    .tp-hero {{
        position: relative;
        background:
            radial-gradient(900px 280px at 0% 0%,
              rgba(45,212,191,0.08), transparent 60%),
            {SURFACE_BG};
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 18px;
        padding: 1.4rem 1.6rem;
    }}
    .tp-hero-pill {{
        display: inline-block;
        padding: 0.42rem 1.0rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.95rem;
        letter-spacing: 0.06em;
    }}
    .tp-hero-headline {{
        font-size: 1.85rem;
        font-weight: 700;
        margin: 0.55rem 0 0.25rem 0;
        letter-spacing: -0.01em;
        color: #f1f5f9;
    }}
    .tp-hero-subline {{
        color: {MUTED_TEXT};
        font-size: 0.98rem;
        margin: 0 0 0.4rem 0;
        max-width: 760px;
    }}
    .tp-hero-reason {{
        background: rgba(255,255,255,0.03);
        border-left: 3px solid rgba(148,163,184,0.25);
        padding: 0.7rem 0.9rem;
        border-radius: 8px;
        color: #cbd5e1;
        font-size: 0.92rem;
        margin-top: 0.5rem;
    }}

    /* KPI tile. */
    .tp-kpi {{
        background: {SURFACE_BG};
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 14px;
        padding: 0.95rem 1.1rem;
        height: 100%;
    }}
    .tp-kpi-label {{
        color: {MUTED_TEXT};
        font-size: 0.72rem;
        letter-spacing: 0.10em;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }}
    .tp-kpi-value {{
        font-size: 1.65rem;
        font-weight: 700;
        color: #f1f5f9;
        line-height: 1.1;
    }}
    .tp-kpi-sub {{
        color: {MUTED_TEXT};
        font-size: 0.78rem;
        margin-top: 0.2rem;
    }}

    /* Theme chips. */
    .tp-chip {{
        display: inline-block;
        padding: 4px 10px;
        margin: 3px 5px 3px 0;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 500;
        letter-spacing: 0.02em;
    }}
    .tp-chip-intent {{
        background: rgba(45,212,191,0.10);
        border: 1px solid {ACCENT_ALIGNED};
        color: {ACCENT_ALIGNED};
    }}
    .tp-chip-off {{
        background: rgba(239,68,68,0.10);
        border: 1px solid {ACCENT_DRIFT};
        color: {ACCENT_DRIFT};
    }}
    .tp-chip-neutral {{
        background: rgba(148,163,184,0.10);
        border: 1px solid rgba(148,163,184,0.5);
        color: #cbd5e1;
    }}
    .tp-chip-keyword {{
        background: rgba(99,102,241,0.10);
        border: 1px solid rgba(99,102,241,0.5);
        color: #a5b4fc;
    }}

    /* Section heading inside cards. */
    .tp-section-h {{
        font-size: 0.78rem;
        color: {MUTED_TEXT};
        letter-spacing: 0.10em;
        text-transform: uppercase;
        margin: 0 0 0.55rem 0;
    }}

    /* Contributor row in the drift tab. */
    .tp-contrib {{
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(148, 163, 184, 0.10);
        border-radius: 12px;
        padding: 0.85rem 1.0rem;
        margin-bottom: 0.55rem;
    }}
    .tp-contrib .tp-contrib-id {{
        font-family: "SF Mono", Menlo, Consolas, monospace;
        font-weight: 600;
        color: #f1f5f9;
        margin-right: 0.5rem;
    }}
    .tp-contrib .tp-contrib-desc {{
        color: #cbd5e1;
        font-size: 0.92rem;
        margin: 0.35rem 0 0.35rem 0;
        font-style: italic;
    }}
    .tp-contrib .tp-contrib-reason {{
        color: {MUTED_TEXT};
        font-size: 0.85rem;
    }}

    /* Make the raw-report code block readable. */
    .tp-report pre {{
        background: #0a1020 !important;
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 10px;
        padding: 1rem !important;
        font-size: 0.82rem !important;
        line-height: 1.45 !important;
        max-height: 640px;
        overflow: auto;
    }}

    /* Sidebar polish. */
    section[data-testid="stSidebar"] {{
        border-right: 1px solid rgba(148, 163, 184, 0.10);
    }}
    section[data-testid="stSidebar"] .block-container {{
        padding-top: 1rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _phrase(theme: Optional[str]) -> str:
    if theme is None:
        return "(no theme)"
    return THEME_PHRASES.get(theme, theme.replace("_", " "))


@st.cache_data(show_spinner=False)
def _load_sample() -> dict:
    """Load the bundled sample project, normalised into intent + tasks."""
    if not SAMPLE_PATH.exists():
        return {
            "title": "AWS Cloud Cost Optimization",
            "intent": (
                "Reduce AWS cloud cost by 25 percent over the next two "
                "quarters by rightsizing over-provisioned EC2 and RDS "
                "resources, adopting Savings Plans and Spot instances, "
                "and decommissioning idle services and unused storage."
            ),
            "tasks": [],
        }
    with SAMPLE_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "intent" in data:
        intent = data["intent"]
    else:
        description = (data.get("description") or "").strip()
        goals = data.get("goals") or []
        sentences = [description] + [
            g.strip().rstrip(".") + "." for g in goals if g.strip()
        ]
        intent = " ".join(s for s in sentences if s).strip()

    raw_tasks = data.get("tasks", []) or []
    tasks: List[Dict[str, str]] = []
    for t in raw_tasks:
        if isinstance(t, str):
            tasks.append({"id": "", "description": t})
        elif isinstance(t, dict):
            tasks.append(
                {
                    "id": str(t.get("id", "") or ""),
                    "description": t.get("description", "") or "",
                }
            )

    return {
        "title": data.get("title", "Untitled Project"),
        "intent": intent,
        "tasks": tasks,
    }


def _bootstrap_state() -> None:
    """Initialise st.session_state from the bundled sample on first run."""
    if "bootstrapped" in st.session_state:
        return
    sample = _load_sample()
    st.session_state["title"] = sample["title"]
    st.session_state["intent"] = sample["intent"]
    st.session_state["tasks"] = sample["tasks"]
    st.session_state["aligned_min"] = 0.70
    st.session_state["drift_min"] = 0.40
    st.session_state["direction_threshold"] = 0.35
    st.session_state["bootstrapped"] = True


def _verdict_color(level: str) -> str:
    return VERDICT_STYLE.get(level, VERDICT_STYLE["Significant Drift"])["color"]


def _normalise_tasks_for_pipeline(tasks: List[Dict[str, str]]) -> List[dict]:
    """Convert the editable-table rows into the shape the pipeline expects.

    Empty rows are dropped, IDs auto-fall-back to TASK-N so the explainer
    still has something to print.
    """
    cleaned: List[dict] = []
    for idx, t in enumerate(tasks, start=1):
        desc = (t.get("description") or "").strip()
        if not desc:
            continue
        tid = (t.get("id") or "").strip() or f"TASK-{idx:03d}"
        cleaned.append({"id": tid, "description": desc})
    return cleaned


# ---------------------------------------------------------------------------
# Sidebar - data source + intent + thresholds
# ---------------------------------------------------------------------------

def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:0.6rem;
                        margin-bottom:0.6rem;">
              <div style="width:32px; height:32px; border-radius:9px;
                          background:linear-gradient(135deg,{ACCENT_ALIGNED},#6366f1);
                          display:flex; align-items:center; justify-content:center;
                          font-size:18px;">🧭</div>
              <div>
                <div style="font-weight:700; font-size:1rem;">TruePath AI</div>
                <div style="color:{MUTED_TEXT}; font-size:0.74rem;">
                  Intent-vs-Execution Drift</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        st.markdown("##### Data source")
        source = st.radio(
            "Data source",
            ("Bundled sample", "Upload JSON", "Paste JSON"),
            label_visibility="collapsed",
            key="source",
        )

        if source == "Upload JSON":
            uploaded = st.file_uploader(
                "Upload a project JSON",
                type=["json"],
                label_visibility="collapsed",
            )
            if uploaded is not None and st.button(
                "Load uploaded file", use_container_width=True
            ):
                try:
                    data = json.loads(uploaded.read().decode("utf-8"))
                    project = _coerce_uploaded(data)
                    st.session_state["title"] = project["title"]
                    st.session_state["intent"] = project["intent"]
                    st.session_state["tasks"] = project["tasks"]
                    st.success("Loaded.")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Could not parse JSON: {exc}")

        elif source == "Paste JSON":
            pasted = st.text_area(
                "Paste a project JSON",
                value="",
                height=180,
                label_visibility="collapsed",
                placeholder='{"title": "...", "intent": "...", "tasks": [...]}',
            )
            if pasted.strip() and st.button(
                "Load pasted JSON", use_container_width=True
            ):
                try:
                    data = json.loads(pasted)
                    project = _coerce_uploaded(data)
                    st.session_state["title"] = project["title"]
                    st.session_state["intent"] = project["intent"]
                    st.session_state["tasks"] = project["tasks"]
                    st.success("Loaded.")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Could not parse JSON: {exc}")

        st.markdown("##### Project")
        st.text_input(
            "Project title",
            key="title",
            label_visibility="collapsed",
            placeholder="Project title",
        )
        st.text_area(
            "Project intent",
            key="intent",
            height=170,
            label_visibility="collapsed",
            placeholder="What is this project trying to achieve?",
        )

        with st.expander("Drift thresholds", expanded=False):
            st.caption(
                "Every threshold is a knob - the same knobs that "
                "`detect_intent_drift` accepts as keyword arguments."
            )
            st.slider(
                "Aligned threshold (aligned_min)",
                min_value=0.50,
                max_value=0.95,
                step=0.05,
                key="aligned_min",
                help="At/above this aligned share, the verdict is 'Aligned'.",
            )
            st.slider(
                "Drift threshold (drift_min)",
                min_value=0.10,
                max_value=0.60,
                step=0.05,
                key="drift_min",
                help="Below this aligned share, the verdict is always "
                "'Significant Drift'.",
            )
            st.slider(
                "Direction escalation (direction_threshold)",
                min_value=0.10,
                max_value=0.60,
                step=0.05,
                key="direction_threshold",
                help="Share of tickets on a single off-intent theme that "
                "escalates Partial -> Significant Drift.",
            )

        st.divider()
        if st.button(
            "Reset to demo",
            use_container_width=True,
            help="Reload the bundled sample_project.json.",
        ):
            sample = _load_sample()
            st.session_state["title"] = sample["title"]
            st.session_state["intent"] = sample["intent"]
            st.session_state["tasks"] = sample["tasks"]
            st.session_state["source"] = "Bundled sample"
            st.rerun()


def _coerce_uploaded(data: dict) -> dict:
    """Normalize an uploaded JSON into the {title, intent, tasks[]} shape."""
    if "intent" in data:
        intent = data["intent"]
    else:
        description = (data.get("description") or "").strip()
        goals = data.get("goals") or []
        sentences = [description] + [
            g.strip().rstrip(".") + "." for g in goals if g.strip()
        ]
        intent = " ".join(s for s in sentences if s).strip()

    raw_tasks = data.get("tasks", []) or []
    tasks: List[Dict[str, str]] = []
    for t in raw_tasks:
        if isinstance(t, str):
            tasks.append({"id": "", "description": t})
        elif isinstance(t, dict):
            tasks.append(
                {
                    "id": str(t.get("id", "") or ""),
                    "description": t.get("description", "") or "",
                }
            )

    return {
        "title": data.get("title", "Untitled Project"),
        "intent": intent,
        "tasks": tasks,
    }


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

def _render_header(project_title: str) -> None:
    st.markdown(
        f"""
        <div class="tp-header">
          <div class="tp-brand">
            <div class="tp-logo">🧭</div>
            <div>
              <h1>TruePath AI</h1>
              <div class="tp-tag">
                Detecting silent project drift &middot;
                <span style="color:#cbd5e1;">{project_title or "Untitled Project"}</span>
              </div>
            </div>
          </div>
          <div class="tp-badges">
            <span class="tp-pill">Rule-based</span>
            <span class="tp-pill">Zero ML</span>
            <span class="tp-pill">Auditable</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Hero verdict card
# ---------------------------------------------------------------------------

def _aligned_share_gauge(aligned_share: float, color: str) -> go.Figure:
    pct = max(0.0, min(1.0, aligned_share)) * 100
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix": "%", "font": {"size": 36, "color": "#f1f5f9"}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "rgba(148,163,184,0.4)",
                    "tickfont": {"color": MUTED_TEXT, "size": 10},
                },
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": "rgba(255,255,255,0.04)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "rgba(239,68,68,0.18)"},
                    {"range": [40, 70], "color": "rgba(245,158,11,0.18)"},
                    {"range": [70, 100], "color": "rgba(45,212,191,0.20)"},
                ],
                "threshold": {
                    "line": {"color": "#f1f5f9", "width": 2},
                    "thickness": 0.75,
                    "value": pct,
                },
            },
            domain={"x": [0, 1], "y": [0, 1]},
        )
    )
    fig.update_layout(
        height=220,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e6edf7"),
    )
    return fig


def _render_hero(verdict: dict, cost_mode: bool) -> None:
    level = verdict["drift_level"]
    style = VERDICT_STYLE.get(level, VERDICT_STYLE["Significant Drift"])
    color = style["color"]
    headline = style["headline_cost"] if cost_mode else style["headline_plain"]
    subline = style["subline_cost"] if cost_mode else style["subline_plain"]

    counts = verdict["counts"]
    aligned = counts["aligned"]
    total = counts["total"]
    aligned_pct = (
        f"{verdict['aligned_share']:.0%}" if total else "0%"
    )

    col_left, col_right = st.columns([3, 2], gap="large")
    with col_left:
        st.markdown(
            f"""
            <div class="tp-hero">
              <div class="tp-hero-pill"
                   style="background:{color}1A; color:{color};
                          border:1px solid {color};">
                {level.upper()}
              </div>
              <div class="tp-hero-headline">{headline}</div>
              <div class="tp-hero-subline">{subline}</div>
              <div class="tp-hero-reason">{verdict["reason"]}</div>
              <div style="color:{MUTED_TEXT}; font-size:0.85rem; margin-top:0.6rem;">
                Aligned tickets: <span style="color:#f1f5f9; font-weight:600;">
                {aligned}/{total}</span> &middot;
                Aligned share:
                <span style="color:{color}; font-weight:700;">{aligned_pct}</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_right:
        st.markdown(
            f"""
            <div class="tp-card" style="padding:0.7rem 0.8rem 0.3rem 0.8rem;">
              <div class="tp-section-h">Aligned share</div>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            _aligned_share_gauge(verdict["aligned_share"], color),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------

def _render_kpis(verdict: dict, cost_mode: bool) -> None:
    counts = verdict["counts"]
    direction = verdict["drift_direction"]
    direction_phrase = _phrase(direction) if direction else "None"
    direction_color = (
        ACCENT_DRIFT if direction
        and (cost_mode and direction in SPEND_INCREASING_THEMES
             or not cost_mode)
        else (ACCENT_PARTIAL if direction else ACCENT_NEUTRAL)
    )

    aligned_pct = (
        f"{counts['aligned'] / counts['total']:.0%}"
        if counts["total"] else "0%"
    )
    off_pct = (
        f"{counts['off_intent'] / counts['total']:.0%}"
        if counts["total"] else "0%"
    )

    cols = st.columns(4, gap="medium")
    with cols[0]:
        st.markdown(
            f"""
            <div class="tp-kpi">
              <div class="tp-kpi-label">Total tickets</div>
              <div class="tp-kpi-value">{counts["total"]}</div>
              <div class="tp-kpi-sub">in current sprint backlog</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            f"""
            <div class="tp-kpi">
              <div class="tp-kpi-label">In intent</div>
              <div class="tp-kpi-value" style="color:{ACCENT_ALIGNED};">
                {counts["aligned"]}
              </div>
              <div class="tp-kpi-sub">{aligned_pct} of backlog</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cols[2]:
        st.markdown(
            f"""
            <div class="tp-kpi">
              <div class="tp-kpi-label">Off intent</div>
              <div class="tp-kpi-value" style="color:{ACCENT_DRIFT};">
                {counts["off_intent"]}
              </div>
              <div class="tp-kpi-sub">{off_pct} of backlog</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cols[3]:
        st.markdown(
            f"""
            <div class="tp-kpi">
              <div class="tp-kpi-label">Drift direction</div>
              <div class="tp-kpi-value" style="color:{direction_color};
                                                font-size:1.15rem;
                                                line-height:1.2;
                                                margin-top:0.4rem;">
                {direction_phrase}
              </div>
              <div class="tp-kpi-sub">dominant off-intent theme</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Tab 1 - Intent Analysis
# ---------------------------------------------------------------------------

def _render_intent_tab(
    intent_text: str,
    intent_themes: List[str],
) -> None:
    parsed = parse_intent(intent_text)

    st.markdown('<div class="tp-section-h">Primary goal</div>', unsafe_allow_html=True)
    if parsed["primary_goal"]:
        st.markdown(
            f"""
            <div class="tp-card" style="border-left:3px solid {ACCENT_ALIGNED};">
              <div style="font-size:1.0rem; color:#f1f5f9; line-height:1.5;">
                {parsed["primary_goal"]}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("No primary goal detected. Try a more directive intent statement.")

    st.write("")
    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.markdown(
            '<div class="tp-section-h">Supporting goals & constraints</div>',
            unsafe_allow_html=True,
        )
        if parsed["supporting"]:
            rows_html: List[str] = []
            for item in parsed["supporting"]:
                kind = item["kind"]
                color = ACCENT_ALIGNED if kind == "goal" else ACCENT_PARTIAL
                rows_html.append(
                    f"""
                    <div style="display:flex; gap:0.6rem; align-items:flex-start;
                                margin-bottom:0.5rem;">
                      <span class="tp-chip" style="background:{color}1A;
                                                   color:{color};
                                                   border:1px solid {color};
                                                   min-width:80px;
                                                   text-align:center;">
                        {kind}
                      </span>
                      <span style="color:#cbd5e1; line-height:1.4;">
                        {item["text"]}
                      </span>
                    </div>
                    """
                )
            st.markdown(
                f"""<div class="tp-card">{"".join(rows_html)}</div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="tp-card" style="color:#94a3b8;">'
                "No supporting goals or constraints detected.</div>",
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown(
            '<div class="tp-section-h">Intent themes</div>',
            unsafe_allow_html=True,
        )
        if intent_themes:
            chips = "".join(
                f'<span class="tp-chip tp-chip-intent">{_phrase(t)}</span>'
                for t in intent_themes
            )
        else:
            chips = (
                '<span style="color:#94a3b8;">'
                "No intent themes matched the lexicon.</span>"
            )
        st.markdown(
            f"""<div class="tp-card">{chips}</div>""",
            unsafe_allow_html=True,
        )

        st.write("")
        st.markdown(
            '<div class="tp-section-h">Top intent keywords</div>',
            unsafe_allow_html=True,
        )
        keywords = parsed["keywords"][:18]
        if keywords:
            kchips = "".join(
                f'<span class="tp-chip tp-chip-keyword">{k}</span>'
                for k in keywords
            )
        else:
            kchips = (
                '<span style="color:#94a3b8;">No keywords extracted.</span>'
            )
        st.markdown(
            f"""<div class="tp-card">{kchips}</div>""",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Tab 2 - Ticket backlog (editable)
# ---------------------------------------------------------------------------

def _render_tickets_tab(
    classification: dict,
    intent_themes: List[str],
) -> None:
    st.caption(
        "Edit any cell to live-update the verdict. "
        "Add rows with the **+** button at the bottom of the table; "
        "delete rows with the checkbox column."
    )

    ticket_rows = st.session_state.get("tasks", [])
    classified = {t["id"]: t for t in classification["tasks"]}
    intent_set = set(intent_themes)

    table_rows: List[Dict[str, object]] = []
    for row in ticket_rows:
        tid = (row.get("id") or "").strip()
        desc = row.get("description") or ""
        if not desc.strip():
            table_rows.append({
                "ID": tid,
                "Description": desc,
                "Status": "—",
                "Theme": "—",
                "Hits": "",
            })
            continue

        c = classified.get(tid) if tid else None
        if c is None:
            for cand in classification["tasks"]:
                if cand["description"] == desc:
                    c = cand
                    break

        if c is None:
            status = "—"
            theme_label = "—"
            hits = ""
        else:
            top = c["top_theme"]
            if top is None:
                status = "no theme"
                theme_label = "—"
            elif top in intent_set:
                status = "in intent"
                theme_label = _phrase(top)
            else:
                if top in SPEND_INCREASING_THEMES:
                    status = "ADDS spend"
                else:
                    status = "OFF intent"
                theme_label = _phrase(top)
            hits = ", ".join(c["matched_keywords"][:6]) or "—"

        table_rows.append({
            "ID": tid,
            "Description": desc,
            "Status": status,
            "Theme": theme_label,
            "Hits": hits,
        })

    edited = st.data_editor(
        table_rows,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "ID": st.column_config.TextColumn(
                "Ticket ID",
                width="small",
                help="e.g. CLOUD-101 (left blank: auto-numbered as TASK-NNN).",
            ),
            "Description": st.column_config.TextColumn(
                "Description",
                width="large",
                help="One-line summary - same text Jira already has.",
            ),
            "Status": st.column_config.TextColumn(
                "Status",
                width="small",
                help="in intent / OFF intent / ADDS spend / no theme. "
                "Read-only - derived from the description.",
                disabled=True,
            ),
            "Theme": st.column_config.TextColumn(
                "Top theme",
                width="medium",
                disabled=True,
            ),
            "Hits": st.column_config.TextColumn(
                "Matched keywords",
                width="medium",
                disabled=True,
            ),
        },
        key="ticket_editor",
    )

    new_tasks: List[Dict[str, str]] = []
    for r in edited:
        new_tasks.append({
            "id": str(r.get("ID", "") or "").strip(),
            "description": str(r.get("Description", "") or "").strip(),
        })

    if new_tasks != st.session_state.get("tasks"):
        st.session_state["tasks"] = new_tasks
        st.rerun()


# ---------------------------------------------------------------------------
# Tab 3 - Drift Detection charts + contributors + recommendations
# ---------------------------------------------------------------------------

def _theme_distribution_bar(verdict: dict) -> Optional[go.Figure]:
    rows = verdict["task_theme_distribution"]
    if not rows:
        return None

    rows = sorted(rows, key=lambda r: r["tasks"])  # so largest sits on top
    names = [_phrase(r["name"]) for r in rows]
    values = [r["tasks"] for r in rows]
    colors = [
        ACCENT_ALIGNED if r["in_intent"]
        else (ACCENT_DRIFT if r["name"] in SPEND_INCREASING_THEMES
              else ACCENT_PARTIAL)
        for r in rows
    ]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker=dict(color=colors,
                        line=dict(color="rgba(255,255,255,0.05)", width=1)),
            text=[
                f"{v}  ({r['share']:.0%})"
                for v, r in zip(values, rows)
            ],
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Tickets: %{x}<br>"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        height=max(220, 42 * len(rows) + 60),
        margin=dict(l=10, r=60, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(148,163,184,0.10)",
            zeroline=False,
            tickfont=dict(color=MUTED_TEXT),
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            tickfont=dict(color="#e6edf7"),
        ),
        showlegend=False,
        font=dict(color="#e6edf7"),
    )
    return fig


def _intent_donut(verdict: dict) -> Optional[go.Figure]:
    counts = verdict["counts"]
    total = counts["total"]
    if total == 0:
        return None

    labels = ["In intent", "Off intent", "No theme"]
    values = [counts["aligned"], counts["off_intent"], counts["unthemed"]]
    colors = [ACCENT_ALIGNED, ACCENT_DRIFT, ACCENT_NEUTRAL]

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.62,
            marker=dict(colors=colors,
                        line=dict(color=APP_BG, width=3)),
            textinfo="label+percent",
            textfont=dict(color="#f1f5f9", size=12),
            hovertemplate="<b>%{label}</b><br>%{value} tickets<extra></extra>",
            sort=False,
        )
    )
    fig.add_annotation(
        text=f"<b>{total}</b><br><span style='color:{MUTED_TEXT};font-size:12px;'>tickets</span>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(color="#f1f5f9", size=22),
    )
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        font=dict(color="#e6edf7"),
    )
    return fig


def _render_drift_tab(
    verdict: dict,
    classification: dict,
    cost_mode: bool,
) -> None:
    col_dist, col_pie = st.columns([3, 2], gap="large")

    with col_dist:
        st.markdown(
            '<div class="tp-section-h">Theme distribution</div>',
            unsafe_allow_html=True,
        )
        bar = _theme_distribution_bar(verdict)
        if bar is None:
            st.markdown(
                '<div class="tp-card" style="color:#94a3b8;">'
                "No themes matched any ticket.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.plotly_chart(
                bar,
                use_container_width=True,
                config={"displayModeBar": False},
            )
            st.markdown(
                f"""
                <div style="display:flex; gap:0.7rem; flex-wrap:wrap;
                            margin-top:-0.5rem; color:{MUTED_TEXT};
                            font-size:0.78rem;">
                  <span><span style="display:inline-block; width:10px; height:10px;
                    background:{ACCENT_ALIGNED}; border-radius:2px;
                    margin-right:5px;"></span>In intent</span>
                  <span><span style="display:inline-block; width:10px; height:10px;
                    background:{ACCENT_DRIFT}; border-radius:2px;
                    margin-right:5px;"></span>Off intent (ADDS spend)</span>
                  <span><span style="display:inline-block; width:10px; height:10px;
                    background:{ACCENT_PARTIAL}; border-radius:2px;
                    margin-right:5px;"></span>Off intent (other)</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col_pie:
        st.markdown(
            '<div class="tp-section-h">Backlog composition</div>',
            unsafe_allow_html=True,
        )
        donut = _intent_donut(verdict)
        if donut is None:
            st.info("No tickets to analyse yet.")
        else:
            st.plotly_chart(
                donut,
                use_container_width=True,
                config={"displayModeBar": False},
            )

    st.write("")
    st.markdown(
        '<div class="tp-section-h">Top contributors to drift</div>',
        unsafe_allow_html=True,
    )

    intent_set = set(verdict["intent_themes"])
    direction = verdict["drift_direction"]

    by_hits = lambda t: -len(t["matched_keywords"])  # noqa: E731
    direction_tasks = sorted(
        [t for t in classification["tasks"]
         if direction is not None and t["top_theme"] == direction],
        key=by_hits,
    )
    unthemed_tasks = [
        t for t in classification["tasks"] if t["top_theme"] is None
    ]
    other_off_tasks = sorted(
        [t for t in classification["tasks"]
         if t["top_theme"] is not None
         and t["top_theme"] != direction
         and t["top_theme"] not in intent_set],
        key=by_hits,
    )
    contributors = (direction_tasks + unthemed_tasks + other_off_tasks)[:5]

    if not contributors:
        st.markdown(
            f"""<div class="tp-card" style="color:{ACCENT_ALIGNED};">
            Every ticket lands within the project's intent themes.</div>""",
            unsafe_allow_html=True,
        )
    else:
        for t in contributors:
            top = t["top_theme"]
            if top is None:
                tag, tag_color = "no theme", ACCENT_NEUTRAL
                reason = (
                    "No theme matched - it's unclear how this ticket "
                    "contributes to the goal."
                )
            elif cost_mode and top in SPEND_INCREASING_THEMES:
                tag, tag_color = "ADDS spend", ACCENT_DRIFT
                reason = (
                    f"Top theme '{_phrase(top)}' adds capacity to the "
                    f"AWS bill instead of reducing it."
                )
            elif direction and top == direction:
                tag, tag_color = "OFF intent", ACCENT_DRIFT
                reason = (
                    f"Pulls toward '{_phrase(top)}', which is not part "
                    f"of the project intent."
                )
            else:
                tag, tag_color = "OFF intent", ACCENT_PARTIAL
                reason = (
                    f"Top theme '{_phrase(top)}' is outside the project intent."
                )

            hits_html = "".join(
                f'<span class="tp-chip tp-chip-keyword">{k}</span>'
                for k in t["matched_keywords"][:6]
            ) or '<span style="color:#94a3b8;">no keyword hits</span>'

            st.markdown(
                f"""
                <div class="tp-contrib">
                  <div style="display:flex; justify-content:space-between;
                              align-items:center;">
                    <div>
                      <span class="tp-contrib-id">{t["id"]}</span>
                      <span class="tp-chip"
                            style="background:{tag_color}1A;
                                   border:1px solid {tag_color};
                                   color:{tag_color};">{tag}</span>
                    </div>
                    <span style="color:{MUTED_TEXT}; font-size:0.78rem;">
                      theme: <span style="color:#cbd5e1;">{_phrase(top)}</span>
                    </span>
                  </div>
                  <div class="tp-contrib-desc">"{t["description"]}"</div>
                  <div class="tp-contrib-reason">{reason}</div>
                  <div style="margin-top:0.45rem;">{hits_html}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")
    st.markdown(
        '<div class="tp-section-h">Recommendation</div>',
        unsafe_allow_html=True,
    )
    bullets = _build_recommendations(verdict, contributors, cost_mode)
    bullet_html = "".join(
        f"""
        <li style="margin-bottom:0.45rem; color:#cbd5e1; line-height:1.5;">
          {b}
        </li>
        """
        for b in bullets
    )
    st.markdown(
        f"""
        <div class="tp-card"
             style="border-left:3px solid {_verdict_color(verdict["drift_level"])};">
          <div style="color:{MUTED_TEXT}; font-size:0.85rem;
                      margin-bottom:0.4rem;">
            Decision-support options - the team picks what fits the situation.
            This report is not prescriptive.
          </div>
          <ul style="padding-left:1.1rem; margin:0;">{bullet_html}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _build_recommendations(
    verdict: dict,
    contributors: List[dict],
    cost_mode: bool,
) -> List[str]:
    """Mirror ``explainer._recommendation_bullets`` for the rendered card.

    We deliberately re-derive these in the UI (instead of regex-parsing the
    text report) so the bullets render cleanly as HTML and stay aligned
    with the verdict color.
    """
    level = verdict["drift_level"]
    direction = verdict["drift_direction"]
    direction_phrase = _phrase(direction) if direction else None
    ids = ", ".join(t["id"] for t in contributors) if contributors else ""

    bullets: List[str] = []
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
                "Continue prioritizing the on-intent work that is already "
                "shaping the sprint backlog."
            )
            bullets.append(
                "Re-run this check when the sprint backlog changes shape, "
                "in case new tickets quietly drift the team off intent."
            )
        return bullets

    if cost_mode:
        bullets.append(
            "Prioritize rightsizing and decommissioning idle / unused "
            "resources - these are the tickets that directly reduce "
            "cloud spend."
        )
        if direction in SPEND_INCREASING_THEMES:
            bullets.append(
                f"Deprioritize <b>'{direction_phrase}'</b> tickets unless "
                f"the additional capacity is genuinely required by demand "
                f"and the additional AWS spend is approved."
            )
        elif direction:
            bullets.append(
                f"Deprioritize <b>'{direction_phrase}'</b> tickets unless "
                f"they directly support a cost optimization initiative."
            )
        if direction in SPEND_INCREASING_THEMES:
            bullets.append(
                f"If growth is now the actual goal, explicitly update the "
                f"project intent to include <b>'{direction_phrase}'</b> "
                f"and re-baseline the cost target accordingly."
            )
        elif direction:
            bullets.append(
                f"If the project's true goal has shifted toward "
                f"<b>'{direction_phrase}'</b>, explicitly update the "
                f"project intent so future drift checks measure against "
                f"the new objective."
            )
        else:
            bullets.append(
                "If the project's goal has shifted, explicitly update the "
                "project intent so future drift checks measure against "
                "the new objective."
            )
        if ids:
            bullets.append(f"Tickets worth a second look: <code>{ids}</code>.")
        return bullets

    if direction:
        bullets.append(
            "Continue prioritizing the on-intent work that is already "
            "moving the project forward."
        )
        bullets.append(
            f"Deprioritize <b>'{direction_phrase}'</b> tickets unless "
            f"they directly support the current project intent."
        )
        bullets.append(
            f"If <b>'{direction_phrase}'</b> is now the actual goal, "
            f"explicitly update the project intent to include it."
        )
    else:
        bullets.append(
            "Continue prioritizing the on-intent work that is already "
            "moving the project forward."
        )
        bullets.append(
            "Review the borderline tickets and decide, per ticket, whether "
            "to re-scope onto intent or to broaden the project intent "
            "description."
        )
    if ids:
        bullets.append(f"Tickets worth a second look: <code>{ids}</code>.")
    return bullets


# ---------------------------------------------------------------------------
# Tab 4 - Raw report
# ---------------------------------------------------------------------------

def _render_report_tab(intent_text: str, tasks: List[dict], title: str) -> None:
    if not tasks:
        st.info("Add tickets to generate a report.")
        return

    report = explain_drift(intent_text, tasks, project_title=title)

    st.caption(
        "This is the exact text that `explain_drift()` produces - the same "
        "block you would paste into Slack or a slide. Use the copy button "
        "in the top-right corner of the code block."
    )
    st.markdown('<div class="tp-report">', unsafe_allow_html=True)
    st.code(report, language="text")
    st.markdown("</div>", unsafe_allow_html=True)

    st.download_button(
        "Download report (.txt)",
        data=report,
        file_name=f"truepath_report_{title.lower().replace(' ', '_')}.txt",
        mime="text/plain",
        use_container_width=False,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _bootstrap_state()
    _render_sidebar()

    title = st.session_state.get("title", "Untitled Project")
    intent_text = st.session_state.get("intent", "")
    tasks_state: List[Dict[str, str]] = st.session_state.get("tasks", [])
    pipeline_tasks = _normalise_tasks_for_pipeline(tasks_state)

    _render_header(title)

    if not intent_text.strip():
        st.warning(
            "Enter a project intent in the sidebar to start the analysis."
        )
        return

    intent_themes = extract_intent_themes(intent_text)
    cost_mode = _is_cost_project(intent_themes)

    if not pipeline_tasks:
        st.markdown(
            f"""
            <div class="tp-hero">
              <div class="tp-hero-pill"
                   style="background:{ACCENT_NEUTRAL}1A; color:{ACCENT_NEUTRAL};
                          border:1px solid {ACCENT_NEUTRAL};">
                NO TICKETS
              </div>
              <div class="tp-hero-headline">Add tickets to evaluate drift.</div>
              <div class="tp-hero-subline">
                Use the <b>Ticket Backlog</b> tab below to add Jira-style
                tickets, or load a project JSON from the sidebar.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        verdict = {
            "drift_level": "Aligned",
            "aligned_share": 0.0,
            "intent_themes": intent_themes,
            "task_theme_distribution": [],
            "drift_direction": None,
            "counts": {"aligned": 0, "off_intent": 0, "unthemed": 0,
                       "total": 0},
            "reason": "No tickets to evaluate.",
        }
        classification = {"tasks": [], "themes": [], "dominant_themes": []}
    else:
        verdict = detect_intent_drift(
            intent_text,
            pipeline_tasks,
            aligned_min=st.session_state["aligned_min"],
            drift_min=st.session_state["drift_min"],
            direction_threshold=st.session_state["direction_threshold"],
        )
        classification = classify_tasks(pipeline_tasks)
        _render_hero(verdict, cost_mode)

    st.write("")
    _render_kpis(verdict, cost_mode)
    st.write("")

    tab_intent, tab_tickets, tab_drift, tab_report = st.tabs(
        [
            "Intent Analysis",
            f"Ticket Backlog ({len(pipeline_tasks)})",
            "Drift Detection",
            "Raw Report",
        ]
    )

    with tab_intent:
        _render_intent_tab(intent_text, intent_themes)
    with tab_tickets:
        _render_tickets_tab(classification, intent_themes)
    with tab_drift:
        _render_drift_tab(verdict, classification, cost_mode)
    with tab_report:
        _render_report_tab(intent_text, pipeline_tasks, title)


if __name__ == "__main__":
    main()
