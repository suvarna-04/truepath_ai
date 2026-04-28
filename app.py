"""TruePath AI - Interactive Streamlit Dashboard.

A polished, responsive UI for the rule-based intent-vs-execution drift
detector. Plugs directly into the ``truepath_ai`` public API.

    streamlit run app.py
"""

from __future__ import annotations

import html as html_mod
import json
from pathlib import Path
from typing import Dict, List, Optional

import plotly.graph_objects as go
import streamlit as st

from truepath_ai import (
    THEME_LEXICON,
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
# Constants & palette
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent / "data"
# Project to load on first launch. Falls back to the next available
# `data/*.json` if this one is missing.
DEFAULT_PROJECT_NAME = "full_project.json"
# Backwards-compat alias - some helpers still reference SAMPLE_PATH.
SAMPLE_PATH = DATA_DIR / DEFAULT_PROJECT_NAME

C_TEAL      = "#2dd4bf"
C_TEAL_DIM  = "#0d9488"
C_AMBER     = "#f59e0b"
C_AMBER_DIM = "#d97706"
C_RED       = "#ef4444"
C_RED_DIM   = "#dc2626"
C_INDIGO    = "#818cf8"
C_SLATE     = "#64748b"
C_SURFACE   = "#111a2e"
C_BG        = "#0b1220"
C_TEXT      = "#e6edf7"
C_MUTED     = "#94a3b8"
C_DIM       = "#64748b"
C_CARD_BORDER = "rgba(148,163,184,0.12)"

VERDICT_MAP = {
    "Aligned": {
        "color": C_TEAL, "dim": C_TEAL_DIM,
        "plain": "ALIGNED", "cost": "ON TRACK",
        "sub_plain": "Sprint backlog matches the project intent.",
        "sub_cost": "Cloud cost reduction is being addressed.",
        "glow": f"rgba(45,212,191,0.12)",
    },
    "Partial Drift": {
        "color": C_AMBER, "dim": C_AMBER_DIM,
        "plain": "PARTIAL DRIFT", "cost": "AT RISK",
        "sub_plain": "Some tickets are pulling away from intent.",
        "sub_cost": "Cloud spend reduction is partially off-course.",
        "glow": f"rgba(245,158,11,0.12)",
    },
    "Significant Drift": {
        "color": C_RED, "dim": C_RED_DIM,
        "plain": "SIGNIFICANT DRIFT", "cost": "OFF TRACK",
        "sub_plain": "Most work has drifted from the project intent.",
        "sub_cost": "Cloud costs will not reduce on current sprint backlog.",
        "glow": f"rgba(239,68,68,0.12)",
    },
}


def _esc(text: str) -> str:
    return html_mod.escape(str(text))


def _phrase(theme: Optional[str]) -> str:
    if theme is None:
        return "(no theme)"
    return THEME_PHRASES.get(theme, theme.replace("_", " "))


def _verdict_style(level: str) -> dict:
    return VERDICT_MAP.get(level, VERDICT_MAP["Significant Drift"])


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TruePath AI",
    page_icon="https://em-content.zobj.net/source/twitter/376/compass_1f9ed.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
st.markdown(f"""
<style>
/* ---- Reset & base ---- */
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{
    padding: 1.2rem 1.6rem 3rem 1.6rem;
    max-width: 1360px;
}}
html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI",
                 Roboto, "Helvetica Neue", Arial, sans-serif;
}}
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ---- Scrollbar ---- */
::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: rgba(148,163,184,0.25); border-radius: 3px; }}

/* ---- Header ---- */
.tp-hdr {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.3rem 0 1.0rem 0;
    border-bottom: 1px solid {C_CARD_BORDER};
    margin-bottom: 1.2rem;
    flex-wrap: wrap; gap: 0.6rem;
}}
.tp-hdr-brand {{ display: flex; align-items: center; gap: 0.65rem; }}
.tp-hdr-logo {{
    width: 40px; height: 40px; border-radius: 12px;
    background: linear-gradient(135deg, {C_TEAL}, {C_INDIGO});
    display: flex; align-items: center; justify-content: center;
    font-size: 22px; flex-shrink: 0;
    box-shadow: 0 0 16px rgba(45,212,191,0.25);
}}
.tp-hdr h1 {{ font-size: 1.3rem; margin: 0; color: {C_TEXT}; font-weight: 700; }}
.tp-hdr-sub {{ color: {C_MUTED}; font-size: 0.82rem; }}
.tp-hdr-pills {{ display: flex; gap: 6px; flex-wrap: wrap; }}
.tp-pill {{
    font-size: 0.68rem; padding: 3px 10px; border-radius: 999px;
    border: 1px solid rgba(148,163,184,0.25);
    color: {C_MUTED}; background: rgba(255,255,255,0.015);
    letter-spacing: 0.06em; text-transform: uppercase; white-space: nowrap;
}}

/* ---- Card system ---- */
.tp-card {{
    background: {C_SURFACE};
    border: 1px solid {C_CARD_BORDER};
    border-radius: 16px;
    padding: 1.1rem 1.2rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.18);
    transition: box-shadow 0.2s ease, border-color 0.2s ease;
}}
.tp-card:hover {{
    border-color: rgba(148,163,184,0.22);
    box-shadow: 0 8px 32px rgba(0,0,0,0.25);
}}

/* ---- Hero ---- */
.tp-hero {{
    border-radius: 20px; padding: 1.5rem 1.8rem;
    border: 1px solid {C_CARD_BORDER};
    position: relative; overflow: hidden;
    transition: all 0.3s ease;
}}
.tp-hero-pill {{
    display: inline-block; padding: 0.4rem 1.05rem;
    border-radius: 999px; font-weight: 700;
    font-size: 0.88rem; letter-spacing: 0.07em;
    transition: transform 0.15s ease;
}}
.tp-hero-pill:hover {{ transform: scale(1.04); }}
.tp-hero-title {{ font-size: 1.7rem; font-weight: 800; margin: 0.5rem 0 0.2rem 0; color: #f1f5f9; }}
.tp-hero-sub {{ color: {C_MUTED}; font-size: 0.95rem; margin-bottom: 0.5rem; max-width: 720px; }}
.tp-hero-reason {{
    background: rgba(255,255,255,0.025);
    border-left: 3px solid rgba(148,163,184,0.2);
    padding: 0.65rem 0.85rem; border-radius: 8px;
    color: #cbd5e1; font-size: 0.90rem; line-height: 1.55;
    margin-top: 0.5rem;
}}
.tp-hero-stat {{ color: {C_MUTED}; font-size: 0.82rem; margin-top: 0.55rem; }}
.tp-hero-stat b {{ color: #f1f5f9; }}

/* ---- KPI ---- */
.tp-kpi {{
    background: {C_SURFACE}; border: 1px solid {C_CARD_BORDER};
    border-radius: 14px; padding: 0.85rem 1.0rem;
    transition: transform 0.15s ease, box-shadow 0.2s ease;
}}
.tp-kpi:hover {{ transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,0,0,0.22); }}
.tp-kpi-lbl {{ color: {C_MUTED}; font-size: 0.68rem; letter-spacing: 0.10em; text-transform: uppercase; margin-bottom: 0.2rem; }}
.tp-kpi-val {{ font-size: 1.55rem; font-weight: 800; line-height: 1.15; color: #f1f5f9; }}
.tp-kpi-note {{ color: {C_DIM}; font-size: 0.74rem; margin-top: 0.15rem; }}

/* ---- Chips ---- */
.tp-chip {{
    display: inline-block; padding: 3px 10px; margin: 3px 4px 3px 0;
    border-radius: 999px; font-size: 0.75rem; font-weight: 500;
    letter-spacing: 0.02em; transition: transform 0.1s ease;
}}
.tp-chip:hover {{ transform: scale(1.06); }}
.tp-chip-intent {{ background: rgba(45,212,191,0.10); border: 1px solid {C_TEAL}; color: {C_TEAL}; }}
.tp-chip-off    {{ background: rgba(239,68,68,0.10); border: 1px solid {C_RED}; color: {C_RED}; }}
.tp-chip-amber  {{ background: rgba(245,158,11,0.10); border: 1px solid {C_AMBER}; color: {C_AMBER}; }}
.tp-chip-kw     {{ background: rgba(129,140,248,0.10); border: 1px solid rgba(129,140,248,0.5); color: {C_INDIGO}; }}
.tp-chip-neutral {{ background: rgba(148,163,184,0.08); border: 1px solid rgba(148,163,184,0.4); color: #cbd5e1; }}
.tp-chip-sem    {{ background: rgba(168,85,247,0.10); border: 1px solid rgba(168,85,247,0.5); color: #c084fc; }}

/* ---- Section titles ---- */
.tp-sh {{
    font-size: 0.72rem; color: {C_MUTED}; letter-spacing: 0.10em;
    text-transform: uppercase; margin: 0 0 0.5rem 0; font-weight: 600;
}}

/* ---- Contributor card ---- */
.tp-contrib {{
    background: rgba(255,255,255,0.015);
    border: 1px solid rgba(148,163,184,0.10);
    border-radius: 14px; padding: 0.9rem 1.0rem;
    margin-bottom: 0.6rem;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}}
.tp-contrib:hover {{
    border-color: rgba(148,163,184,0.2);
    box-shadow: 0 4px 16px rgba(0,0,0,0.15);
}}
.tp-contrib-top {{ display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.4rem; }}
.tp-contrib-id {{ font-family: 'JetBrains Mono', 'SF Mono', Menlo, Consolas, monospace; font-weight: 600; color: #f1f5f9; }}
.tp-contrib-desc {{ color: #cbd5e1; font-size: 0.90rem; margin: 0.3rem 0; font-style: italic; line-height: 1.45; }}
.tp-contrib-reason {{ color: {C_MUTED}; font-size: 0.82rem; line-height: 1.4; }}
.tp-contrib-evidence {{ margin-top: 0.4rem; }}

/* ---- Ticket status badges ---- */
.tp-status {{
    display: inline-block; padding: 2px 8px; border-radius: 6px;
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.03em;
}}
.tp-status-in   {{ background: rgba(45,212,191,0.12); color: {C_TEAL}; border: 1px solid rgba(45,212,191,0.3); }}
.tp-status-off  {{ background: rgba(239,68,68,0.12); color: {C_RED}; border: 1px solid rgba(239,68,68,0.3); }}
.tp-status-adds {{ background: rgba(239,68,68,0.15); color: {C_RED}; border: 1px solid rgba(239,68,68,0.35); }}
.tp-status-none {{ background: rgba(148,163,184,0.10); color: {C_SLATE}; border: 1px solid rgba(148,163,184,0.3); }}
.tp-status-sem  {{ background: rgba(168,85,247,0.12); color: #c084fc; border: 1px solid rgba(168,85,247,0.3); }}

/* ---- Report block ---- */
.tp-report pre {{
    background: #080e1a !important;
    border: 1px solid rgba(148,163,184,0.12);
    border-radius: 12px; padding: 1rem !important;
    font-size: 0.80rem !important; line-height: 1.5 !important;
    max-height: 600px; overflow: auto;
}}

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {{
    border-right: 1px solid rgba(148,163,184,0.08);
}}

/* ---- Responsive ---- */
@media (max-width: 768px) {{
    .block-container {{ padding: 0.8rem 0.8rem 2rem 0.8rem; }}
    .tp-hdr {{ flex-direction: column; align-items: flex-start; }}
    .tp-hero {{ padding: 1.0rem 1.1rem; }}
    .tp-hero-title {{ font-size: 1.3rem; }}
    .tp-kpi-val {{ font-size: 1.3rem; }}
    .tp-contrib-top {{ flex-direction: column; align-items: flex-start; }}
}}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _bundled_projects() -> List[Path]:
    """Discover bundled project JSONs in `data/`.

    Sorted so that ``full_project.json`` and ``sample_project.json`` come
    first (in that order); any other JSON dropped into `data/` is listed
    afterwards alphabetically.
    """
    if not DATA_DIR.exists():
        return []
    priority = {"full_project.json": 0, "sample_project.json": 1}
    return sorted(
        DATA_DIR.glob("*.json"),
        key=lambda p: (priority.get(p.name, 99), p.name),
    )


def _default_project_path() -> Optional[Path]:
    """Pick the project to load on first launch."""
    options = _bundled_projects()
    if not options:
        return None
    preferred = next(
        (p for p in options if p.name == DEFAULT_PROJECT_NAME), None
    )
    return preferred or options[0]


@st.cache_data(show_spinner=False)
def _load_bundled(path_str: str) -> dict:
    """Load and normalize a bundled project JSON. Cached by path string."""
    p = Path(path_str)
    if not p.exists():
        return {"title": "Untitled Project", "intent": "", "tasks": []}
    with p.open("r", encoding="utf-8") as f:
        return _coerce(json.load(f))


def _load_sample() -> dict:
    """Backwards-compat shim - returns the default bundled project."""
    default = _default_project_path()
    if default is None:
        return {"title": "AWS Cloud Cost Optimization", "intent": "", "tasks": []}
    return _load_bundled(str(default))


def _coerce(data: dict) -> dict:
    if "intent" in data:
        intent = data["intent"]
    else:
        desc = (data.get("description") or "").strip()
        goals = data.get("goals") or []
        parts = [desc] + [g.strip().rstrip(".") + "." for g in goals if g.strip()]
        intent = " ".join(s for s in parts if s).strip()
    raw = data.get("tasks", []) or []
    tasks = []
    for t in raw:
        if isinstance(t, str):
            tasks.append({"id": "", "description": t})
        elif isinstance(t, dict):
            tasks.append({
                "id": str(t.get("id", "") or ""),
                "description": t.get("description", "") or "",
                # Preserve month / sprint context so the dashboard can
                # filter by month and label tickets without re-loading.
                "month": t.get("month"),
                "month_label": t.get("month_label"),
                "sprint_number": t.get("sprint_number"),
                "sprint_name": t.get("sprint_name"),
            })
    return {"title": data.get("title", "Untitled Project"), "intent": intent, "tasks": tasks}


# ---------------------------------------------------------------------------
# Month-filter helpers
# ---------------------------------------------------------------------------
ALL_MONTHS_LABEL = "All months"


def _available_months(tasks: List[Dict]) -> List[str]:
    """Return distinct month labels in chronological order.

    Order is by ``month`` (1-12) when present, falling back to label
    sort order. If no ticket has month metadata, returns an empty list
    so the picker can be hidden.
    """
    seen: Dict[str, int] = {}
    for t in tasks:
        label = t.get("month_label")
        if not label:
            continue
        order = t.get("month")
        try:
            order_int = int(order) if order is not None else 99
        except (TypeError, ValueError):
            order_int = 99
        # Keep the smallest month order seen for each label.
        if label not in seen or order_int < seen[label]:
            seen[label] = order_int
    return [label for label, _ in sorted(seen.items(), key=lambda kv: (kv[1], kv[0]))]


def _filter_tasks_by_month(tasks: List[Dict], month_label: str) -> List[Dict]:
    """Return only tickets whose `month_label` matches.

    ``ALL_MONTHS_LABEL`` is a passthrough.
    """
    if not month_label or month_label == ALL_MONTHS_LABEL:
        return tasks
    return [t for t in tasks if t.get("month_label") == month_label]


def _bootstrap():
    if "ready" in st.session_state:
        return
    default = _default_project_path()
    s = _load_bundled(str(default)) if default else {
        "title": "AWS Cloud Cost Optimization", "intent": "", "tasks": [],
    }
    st.session_state.update(
        title=s["title"], intent=s["intent"], tasks=s["tasks"],
        project_path=str(default) if default else "",
        aligned_min=0.70, drift_min=0.40, direction_threshold=0.35,
        ready=True,
    )


def _clean_tasks(tasks: List[Dict[str, str]]) -> List[dict]:
    out = []
    for i, t in enumerate(tasks, 1):
        d = (t.get("description") or "").strip()
        if not d:
            continue
        out.append({"id": (t.get("id") or "").strip() or f"TASK-{i:03d}", "description": d})
    return out


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
def _header(title: str):
    st.markdown(f"""
    <div class="tp-hdr">
      <div class="tp-hdr-brand">
        <div class="tp-hdr-logo">&#x1F9ED;</div>
        <div>
          <h1>TruePath AI</h1>
          <div class="tp-hdr-sub">Detecting silent project drift &middot;
            <span style="color:#cbd5e1;">{_esc(title)}</span></div>
        </div>
      </div>
      <div class="tp-hdr-pills">
        <span class="tp-pill">AI-Powered</span>
        <span class="tp-pill">Hybrid AI + Rules</span>
        <span class="tp-pill">Fully Auditable</span>
        <span class="tp-pill">Deterministic</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def _sidebar():
    with st.sidebar:
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:0.55rem; margin-bottom:0.5rem;">
          <div style="width:34px; height:34px; border-radius:10px;
                      background:linear-gradient(135deg,{C_TEAL},{C_INDIGO});
                      display:flex; align-items:center; justify-content:center;
                      font-size:18px; box-shadow:0 0 12px rgba(45,212,191,0.2);">&#x1F9ED;</div>
          <div>
            <div style="font-weight:700; font-size:0.95rem; color:{C_TEXT};">TruePath AI</div>
            <div style="color:{C_MUTED}; font-size:0.70rem;">Intent Drift Detector v0.4</div>
          </div>
        </div>""", unsafe_allow_html=True)
        st.divider()

        st.markdown(f'<div class="tp-sh">Data Source</div>', unsafe_allow_html=True)
        src = st.radio("src", ("Bundled project", "Upload JSON", "Paste JSON"),
                       label_visibility="collapsed", key="source")
        if src == "Bundled project":
            options = _bundled_projects()
            if not options:
                st.caption("No bundled project files found in `data/`.")
            else:
                # Pretty label for each option includes a quick ticket count
                # so reviewers can pick the demo size without opening files.
                def _label(p: Path) -> str:
                    try:
                        with p.open("r", encoding="utf-8") as f:
                            n = len(json.load(f).get("tasks", []) or [])
                        return f"{p.stem}  ({n} tickets)"
                    except Exception:
                        return p.stem

                current = st.session_state.get("project_path") or str(options[0])
                try:
                    default_idx = [str(p) for p in options].index(current)
                except ValueError:
                    default_idx = 0
                choice = st.selectbox(
                    "project_pick",
                    options,
                    index=default_idx,
                    format_func=_label,
                    label_visibility="collapsed",
                )
                if st.button("Load project", use_container_width=True):
                    p = _load_bundled(str(choice))
                    st.session_state.update(
                        title=p["title"], intent=p["intent"],
                        tasks=p["tasks"], project_path=str(choice),
                    )
                    st.toast(f"Loaded {choice.name}", icon="✅")
                    st.rerun()
        elif src == "Upload JSON":
            up = st.file_uploader("upload", type=["json"], label_visibility="collapsed")
            if up and st.button("Load file", use_container_width=True):
                try:
                    p = _coerce(json.loads(up.read().decode()))
                    st.session_state.update(title=p["title"], intent=p["intent"], tasks=p["tasks"])
                    st.toast("Project loaded.", icon="✅")
                except Exception as e:
                    st.error(f"Invalid JSON: {e}")
        elif src == "Paste JSON":
            txt = st.text_area("paste", value="", height=160, label_visibility="collapsed",
                               placeholder='{"title":"...","intent":"...","tasks":[...]}')
            if txt.strip() and st.button("Load JSON", use_container_width=True):
                try:
                    p = _coerce(json.loads(txt))
                    st.session_state.update(title=p["title"], intent=p["intent"], tasks=p["tasks"])
                    st.toast("Project loaded.", icon="✅")
                except Exception as e:
                    st.error(f"Invalid JSON: {e}")

        st.markdown(f'<div class="tp-sh" style="margin-top:1rem;">Project</div>', unsafe_allow_html=True)
        st.text_input("Title", key="title", label_visibility="collapsed", placeholder="Project title")
        st.text_area("Intent", key="intent", height=140, label_visibility="collapsed",
                     placeholder="Describe the project's intent...")

        with st.expander("Drift Thresholds", expanded=False):
            st.caption("Tune the three knobs that `detect_intent_drift()` uses.")
            st.slider("Aligned threshold", 0.50, 0.95, step=0.05, key="aligned_min",
                      help="At or above this share → Aligned.")
            st.slider("Drift threshold", 0.10, 0.60, step=0.05, key="drift_min",
                      help="Below this share → always Significant Drift.")
            st.slider("Direction escalation", 0.10, 0.60, step=0.05, key="direction_threshold",
                      help="Off-intent concentration that escalates Partial → Significant.")

        st.divider()
        if st.button("Reset to demo", use_container_width=True, type="secondary"):
            default = _default_project_path()
            s = _load_bundled(str(default)) if default else _load_sample()
            st.session_state.update(
                title=s["title"], intent=s["intent"], tasks=s["tasks"],
                project_path=str(default) if default else "",
                source="Bundled project",
            )
            st.rerun()


# ---------------------------------------------------------------------------
# Hero verdict
# ---------------------------------------------------------------------------
def _gauge(share: float, color: str) -> go.Figure:
    pct = max(0, min(100, round(share * 100)))
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=pct,
        number=dict(suffix="%", font=dict(size=38, color="#f1f5f9", family="Inter")),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor="rgba(148,163,184,0.3)",
                      tickfont=dict(color=C_MUTED, size=10)),
            bar=dict(color=color, thickness=0.26),
            bgcolor="rgba(255,255,255,0.03)", borderwidth=0,
            steps=[
                dict(range=[0, 40], color="rgba(239,68,68,0.14)"),
                dict(range=[40, 70], color="rgba(245,158,11,0.14)"),
                dict(range=[70, 100], color="rgba(45,212,191,0.16)"),
            ],
            threshold=dict(line=dict(color="#f1f5f9", width=2), thickness=0.75, value=pct),
        ),
        domain=dict(x=[0, 1], y=[0, 1]),
    ))
    fig.update_layout(
        height=210, margin=dict(l=10, r=10, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=C_TEXT),
    )
    return fig


def _hero(verdict: dict, cost: bool):
    lv = verdict["drift_level"]
    vs = _verdict_style(lv)
    c = vs["color"]
    hl = vs["cost"] if cost else vs["plain"]
    sl = vs["sub_cost"] if cost else vs["sub_plain"]
    cnt = verdict["counts"]
    pct = f'{verdict["aligned_share"]:.0%}' if cnt["total"] else "0%"

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown(f"""
        <div class="tp-hero" style="background: radial-gradient(800px 280px at 0% 0%, {vs['glow']}, transparent 60%), {C_SURFACE};">
          <div class="tp-hero-pill" style="background:{c}18; color:{c}; border:1px solid {c};">{lv.upper()}</div>
          <div class="tp-hero-title">{_esc(hl)}</div>
          <div class="tp-hero-sub">{_esc(sl)}</div>
          <div class="tp-hero-reason">{_esc(verdict["reason"])}</div>
          <div class="tp-hero-stat">
            Aligned tickets: <b>{cnt["aligned"]}/{cnt["total"]}</b> &middot;
            Aligned share: <b style="color:{c};">{pct}</b> &middot;
            Drift direction: <b style="color:{C_RED if verdict['drift_direction'] else C_DIM};">
              {_esc(_phrase(verdict['drift_direction']) if verdict['drift_direction'] else 'None')}</b>
          </div>
        </div>""", unsafe_allow_html=True)
    with right:
        st.markdown(f"""<div class="tp-card" style="padding:0.6rem 0.8rem 0.2rem 0.8rem;">
          <div class="tp-sh">Aligned Share</div>""", unsafe_allow_html=True)
        st.plotly_chart(_gauge(verdict["aligned_share"], c),
                        use_container_width=True, config=dict(displayModeBar=False))
        st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# KPI tiles
# ---------------------------------------------------------------------------
def _kpis(verdict: dict, cost: bool):
    cnt = verdict["counts"]
    total = cnt["total"] or 1
    cols = st.columns(5, gap="small")
    tiles = [
        ("Total Tickets", str(cnt["total"]), "sprint backlog", C_TEXT),
        ("In Intent", str(cnt["aligned"]), f'{cnt["aligned"]/total:.0%} of backlog', C_TEAL),
        ("Off Intent", str(cnt["off_intent"]), f'{cnt["off_intent"]/total:.0%} of backlog', C_RED),
        ("Unthemed", str(cnt["unthemed"]), f'{cnt["unthemed"]/total:.0%} unclassified', C_SLATE),
        ("Drift Direction",
         _phrase(verdict["drift_direction"]) if verdict["drift_direction"] else "None",
         "dominant off-intent theme",
         C_RED if verdict["drift_direction"] and (
             verdict["drift_direction"] in SPEND_INCREASING_THEMES if cost else True
         ) else C_DIM),
    ]
    for col, (lbl, val, note, clr) in zip(cols, tiles):
        with col:
            fs = "1.55rem" if len(val) < 6 else "1.05rem"
            st.markdown(f"""
            <div class="tp-kpi">
              <div class="tp-kpi-lbl">{lbl}</div>
              <div class="tp-kpi-val" style="color:{clr}; font-size:{fs};">{_esc(val)}</div>
              <div class="tp-kpi-note">{_esc(note)}</div>
            </div>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tab 1 - Intent Analysis
# ---------------------------------------------------------------------------
def _tab_intent(intent_text: str, intent_themes: List[str]):
    parsed = parse_intent(intent_text)

    st.markdown('<div class="tp-sh">Primary Goal</div>', unsafe_allow_html=True)
    if parsed["primary_goal"]:
        st.markdown(f"""
        <div class="tp-card" style="border-left:3px solid {C_TEAL};">
          <div style="font-size:1.0rem; color:#f1f5f9; line-height:1.55;">{_esc(parsed["primary_goal"])}</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.info("No primary goal detected. Try a more directive intent statement.")

    st.write("")
    c1, c2 = st.columns(2, gap="medium")

    with c1:
        st.markdown('<div class="tp-sh">Supporting Goals & Constraints</div>', unsafe_allow_html=True)
        if parsed["supporting"]:
            rows = []
            for item in parsed["supporting"]:
                k = item["kind"]
                clr = C_TEAL if k == "goal" else C_AMBER
                rows.append(f"""
                <div style="display:flex; gap:0.55rem; align-items:flex-start; margin-bottom:0.45rem;">
                  <span class="tp-chip" style="background:{clr}15; color:{clr}; border:1px solid {clr};
                        min-width:78px; text-align:center; flex-shrink:0;">{k}</span>
                  <span style="color:#cbd5e1; line-height:1.45; font-size:0.92rem;">{_esc(item["text"])}</span>
                </div>""")
            st.markdown(f'<div class="tp-card">{"".join(rows)}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="tp-card" style="color:{C_MUTED};">No supporting goals or constraints detected.</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="tp-sh">Intent Themes</div>', unsafe_allow_html=True)
        if intent_themes:
            chips = "".join(f'<span class="tp-chip tp-chip-intent">{_esc(_phrase(t))}</span>' for t in intent_themes)
        else:
            chips = f'<span style="color:{C_MUTED};">No themes matched.</span>'
        st.markdown(f'<div class="tp-card">{chips}</div>', unsafe_allow_html=True)

        st.write("")
        st.markdown('<div class="tp-sh">Top Intent Keywords</div>', unsafe_allow_html=True)
        kws = parsed["keywords"][:20]
        if kws:
            kchips = "".join(f'<span class="tp-chip tp-chip-kw">{_esc(k)}</span>' for k in kws)
        else:
            kchips = f'<span style="color:{C_MUTED};">No keywords extracted.</span>'
        st.markdown(f'<div class="tp-card">{kchips}</div>', unsafe_allow_html=True)

    if parsed.get("raw_sentences"):
        st.write("")
        with st.expander("Parsed sentences (debug view)", expanded=False):
            for i, s in enumerate(parsed["raw_sentences"], 1):
                st.markdown(f'`{i}.` {_esc(s)}')


# ---------------------------------------------------------------------------
# Tab 2 - Tickets
# ---------------------------------------------------------------------------
def _tab_tickets(
    classification: dict,
    intent_themes: List[str],
    filtered_tasks: List[Dict],
    active_month: str,
):
    intent_set = set(intent_themes)
    sem_on = classification.get("semantic_enabled", False)

    if sem_on:
        st.markdown(f"""<div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.6rem;">
          <span class="tp-chip tp-chip-sem">Semantic model active</span>
          <span style="color:{C_MUTED}; font-size:0.80rem;">
            Tickets with no lexicon match are classified by sentence embeddings.
          </span>
        </div>""", unsafe_allow_html=True)

    if active_month != ALL_MONTHS_LABEL:
        st.caption(
            f"Showing **{active_month}** tickets only ({len(filtered_tasks)} "
            f"rows). Edits stay scoped to this month; tickets in the other "
            f"months are preserved untouched. Switch the timeline filter to "
            f"`All months` to see and edit the full backlog."
        )
    else:
        st.caption(
            "Edit ID or Description to live-update the verdict. Add rows "
            "with **+**; delete with the checkbox."
        )

    # Only the active month's tickets feed the table; the classifier
    # output is keyed by id/description so we map back to it from there.
    ticket_rows = filtered_tasks
    classified = {}
    for t in classification["tasks"]:
        classified[t["id"]] = t
        classified[t["description"]] = t

    table_data = []
    for row in ticket_rows:
        tid = (row.get("id") or "").strip()
        desc = (row.get("description") or "").strip()
        c = classified.get(tid) or classified.get(desc)

        if not desc or c is None:
            table_data.append({"ID": tid, "Description": desc, "Status": "---",
                               "Theme": "---", "Source": "---", "Keywords": ""})
            continue

        top = c["top_theme"]
        src = c.get("classified_by", "lexicon")
        if top is None:
            status, theme_lbl = "no theme", "---"
        elif top in intent_set:
            status = "in intent"
            theme_lbl = _phrase(top)
        elif top in SPEND_INCREASING_THEMES:
            status = "ADDS spend"
            theme_lbl = _phrase(top)
        else:
            status = "OFF intent"
            theme_lbl = _phrase(top)

        kws = ", ".join(c["matched_keywords"][:6]) or "---"
        src_label = {"lexicon": "lexicon", "semantic": "semantic", "none": "---"}.get(src, src)

        table_data.append({"ID": tid, "Description": desc, "Status": status,
                           "Theme": theme_lbl, "Source": src_label, "Keywords": kws})

    edited = st.data_editor(
        table_data, num_rows="dynamic", use_container_width=True, hide_index=True,
        column_config={
            "ID": st.column_config.TextColumn("Ticket ID", width="small"),
            "Description": st.column_config.TextColumn("Description", width="large"),
            "Status": st.column_config.TextColumn("Status", width="small", disabled=True),
            "Theme": st.column_config.TextColumn("Top Theme", width="medium", disabled=True),
            "Source": st.column_config.TextColumn("Classified By", width="small", disabled=True),
            "Keywords": st.column_config.TextColumn("Matched Keywords", width="medium", disabled=True),
        },
        key="ticket_editor",
    )

    # Build an id-keyed lookup for the previously-displayed tickets so
    # edits in the data_editor do NOT wipe their month / sprint
    # metadata. Without this, every rerun would rebuild rows with only
    # id+description and the timeline filter (which keys off
    # month_label) would silently disappear after the first
    # interaction.
    prev_by_id = {
        str(t.get("id") or ""): t
        for t in filtered_tasks
        if t.get("id")
    }

    edited_filtered: List[Dict] = []
    for r in edited:
        tid = str(r.get("ID", "") or "").strip()
        desc = str(r.get("Description", "") or "").strip()
        prev = prev_by_id.get(tid, {})
        # When the user is filtered to a specific month, default any
        # newly-added row to that month so it stays visible after the
        # add (otherwise it would have month_label=None and instantly
        # disappear from the filtered view).
        default_month_label = (
            active_month if active_month != ALL_MONTHS_LABEL else None
        )
        edited_filtered.append({
            "id": tid,
            "description": desc,
            "month": prev.get("month"),
            "month_label": prev.get("month_label") or default_month_label,
            "sprint_number": prev.get("sprint_number"),
            "sprint_name": prev.get("sprint_name"),
        })

    # Merge the edited slice back into the FULL session task list:
    #   - "All months"  -> straight replace (the table covered everything).
    #   - per-month     -> keep all other-month tickets as-is, swap in
    #                      the edited rows for the active month.
    if active_month == ALL_MONTHS_LABEL:
        new_tasks = edited_filtered
    else:
        existing = st.session_state.get("tasks") or []
        new_tasks = [
            t for t in existing
            if t.get("month_label") != active_month
        ]
        new_tasks.extend(edited_filtered)

    if new_tasks != st.session_state.get("tasks"):
        st.session_state["tasks"] = new_tasks
        st.rerun()

    st.write("")
    st.markdown('<div class="tp-sh">Ticket Detail Cards</div>', unsafe_allow_html=True)
    for t in classification["tasks"]:
        top = t["top_theme"]
        src = t.get("classified_by", "lexicon")
        if top is None:
            badge_cls, badge_txt = "tp-status-none", "no theme"
        elif top in intent_set:
            badge_cls, badge_txt = "tp-status-in", "in intent"
        elif top in SPEND_INCREASING_THEMES:
            badge_cls, badge_txt = "tp-status-adds", "ADDS spend"
        else:
            badge_cls, badge_txt = "tp-status-off", "OFF intent"

        if src == "semantic":
            badge_cls, badge_txt = "tp-status-sem", f"semantic ({badge_txt})"

        kw_chips = "".join(f'<span class="tp-chip tp-chip-kw">{_esc(k)}</span>' for k in t["matched_keywords"][:8])
        sem_score = t.get("semantic_top_score")
        sem_html = ""
        if sem_score is not None:
            sem_html = f' <span class="tp-chip tp-chip-sem">cosine={sem_score:.3f}</span>'

        reason = _esc(t.get("reason", ""))
        st.markdown(f"""
        <div class="tp-contrib">
          <div class="tp-contrib-top">
            <div><span class="tp-contrib-id">{_esc(t["id"])}</span>
              <span class="{badge_cls} tp-status">{badge_txt}</span></div>
            <span style="color:{C_MUTED}; font-size:0.75rem;">
              theme: <span style="color:#cbd5e1;">{_esc(_phrase(top))}</span>
              &middot; via <span style="color:#cbd5e1;">{_esc(src)}</span></span>
          </div>
          <div class="tp-contrib-desc">"{_esc(t["description"])}"</div>
          <div class="tp-contrib-reason">{reason}</div>
          <div class="tp-contrib-evidence">{kw_chips}{sem_html}</div>
        </div>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tab 3 - Drift Detection
# ---------------------------------------------------------------------------
def _bar_chart(verdict: dict) -> Optional[go.Figure]:
    rows = verdict["task_theme_distribution"]
    if not rows:
        return None
    rows = sorted(rows, key=lambda r: r["tasks"])
    names = [_phrase(r["name"]) for r in rows]
    values = [r["tasks"] for r in rows]
    colors = [C_TEAL if r["in_intent"] else (C_RED if r["name"] in SPEND_INCREASING_THEMES else C_AMBER)
              for r in rows]
    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.04)", width=1),
                    cornerradius=4),
        text=[f"{v}  ({r['share']:.0%})" for v, r in zip(values, rows)],
        textposition="outside", cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>%{x} tickets<extra></extra>",
    ))
    fig.update_layout(
        height=max(200, 44 * len(rows) + 50),
        margin=dict(l=10, r=65, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.08)", zeroline=False,
                   tickfont=dict(color=C_MUTED)),
        yaxis=dict(showgrid=False, zeroline=False, tickfont=dict(color=C_TEXT, size=12)),
        showlegend=False, font=dict(color=C_TEXT, family="Inter"),
    )
    return fig


def _donut(verdict: dict) -> Optional[go.Figure]:
    cnt = verdict["counts"]
    if cnt["total"] == 0:
        return None
    labels = ["In intent", "Off intent", "No theme"]
    vals = [cnt["aligned"], cnt["off_intent"], cnt["unthemed"]]
    fig = go.Figure(go.Pie(
        labels=labels, values=vals, hole=0.64, sort=False,
        marker=dict(colors=[C_TEAL, C_RED, C_SLATE], line=dict(color=C_BG, width=3)),
        textinfo="label+percent", textfont=dict(color="#f1f5f9", size=11),
        hovertemplate="<b>%{label}</b><br>%{value} tickets<extra></extra>",
    ))
    fig.add_annotation(
        text=f"<b>{cnt['total']}</b><br><span style='color:{C_MUTED};font-size:11px'>tickets</span>",
        x=0.5, y=0.5, showarrow=False, font=dict(color="#f1f5f9", size=22),
    )
    fig.update_layout(
        height=290, margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False, font=dict(color=C_TEXT, family="Inter"),
    )
    return fig


def _tab_drift(verdict: dict, classification: dict, cost: bool):
    cd, cp = st.columns([3, 2], gap="large")
    with cd:
        st.markdown('<div class="tp-sh">Theme Distribution</div>', unsafe_allow_html=True)
        bar = _bar_chart(verdict)
        if bar:
            st.plotly_chart(bar, use_container_width=True, config=dict(displayModeBar=False))
            st.markdown(f"""
            <div style="display:flex; gap:0.7rem; flex-wrap:wrap; color:{C_MUTED}; font-size:0.75rem; margin-top:-0.3rem;">
              <span><span style="display:inline-block;width:10px;height:10px;background:{C_TEAL};border-radius:2px;margin-right:4px;"></span>In intent</span>
              <span><span style="display:inline-block;width:10px;height:10px;background:{C_RED};border-radius:2px;margin-right:4px;"></span>Off intent (ADDS spend)</span>
              <span><span style="display:inline-block;width:10px;height:10px;background:{C_AMBER};border-radius:2px;margin-right:4px;"></span>Off intent (other)</span>
            </div>""", unsafe_allow_html=True)
        else:
            st.info("No themes matched any ticket.")
    with cp:
        st.markdown('<div class="tp-sh">Backlog Composition</div>', unsafe_allow_html=True)
        donut = _donut(verdict)
        if donut:
            st.plotly_chart(donut, use_container_width=True, config=dict(displayModeBar=False))
        else:
            st.info("No tickets to chart.")

    st.write("")
    st.markdown('<div class="tp-sh">Top Contributors to Drift</div>', unsafe_allow_html=True)

    intent_set = set(verdict["intent_themes"])
    direction = verdict["drift_direction"]
    by_hits = lambda t: -len(t["matched_keywords"])
    d_tasks = sorted([t for t in classification["tasks"] if direction and t["top_theme"] == direction], key=by_hits)
    u_tasks = [t for t in classification["tasks"] if t["top_theme"] is None]
    o_tasks = sorted([t for t in classification["tasks"]
                      if t["top_theme"] and t["top_theme"] != direction and t["top_theme"] not in intent_set], key=by_hits)
    contributors = (d_tasks + u_tasks + o_tasks)[:5]

    if not contributors:
        st.markdown(f'<div class="tp-card" style="color:{C_TEAL};">Every ticket is within the project\'s intent themes.</div>', unsafe_allow_html=True)
    else:
        for t in contributors:
            top = t["top_theme"]
            src = t.get("classified_by", "lexicon")
            if top is None:
                tag, tc = "no theme", C_SLATE
                reason = "No theme matched -- unclear how this ticket contributes."
            elif cost and top in SPEND_INCREASING_THEMES:
                tag, tc = "ADDS spend", C_RED
                reason = f"'{_phrase(top)}' adds capacity to the AWS bill instead of reducing it."
            elif direction and top == direction:
                tag, tc = "OFF intent", C_RED
                reason = f"Pulls toward '{_phrase(top)}', not part of the project intent."
            else:
                tag, tc = "OFF intent", C_AMBER
                reason = f"Top theme '{_phrase(top)}' is outside the project intent."

            if src == "semantic":
                score = t.get("semantic_top_score")
                reason += f" (classified by semantic model, cosine={score:.3f})" if score else " (semantic fallback)"

            kw_chips = "".join(f'<span class="tp-chip tp-chip-kw">{_esc(k)}</span>' for k in t["matched_keywords"][:6])
            sem_chip = ""
            if t.get("semantic_top_score") is not None:
                sem_chip = f'<span class="tp-chip tp-chip-sem">sem={t["semantic_top_score"]:.3f}</span>'

            st.markdown(f"""
            <div class="tp-contrib">
              <div class="tp-contrib-top">
                <div><span class="tp-contrib-id">{_esc(t["id"])}</span>
                  <span class="tp-chip" style="background:{tc}18; border:1px solid {tc}; color:{tc};">{tag}</span></div>
                <span style="color:{C_MUTED}; font-size:0.75rem;">theme: <span style="color:#cbd5e1;">{_esc(_phrase(top))}</span></span>
              </div>
              <div class="tp-contrib-desc">"{_esc(t["description"])}"</div>
              <div class="tp-contrib-reason">{_esc(reason)}</div>
              <div class="tp-contrib-evidence">{kw_chips}{sem_chip}</div>
            </div>""", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="tp-sh">Recommendations</div>', unsafe_allow_html=True)
    bullets = _recommendations(verdict, contributors, cost)
    bhtml = "".join(f'<li style="margin-bottom:0.4rem; color:#cbd5e1; line-height:1.5;">{b}</li>' for b in bullets)
    vc = _verdict_style(verdict["drift_level"])["color"]
    st.markdown(f"""
    <div class="tp-card" style="border-left:3px solid {vc};">
      <div style="color:{C_MUTED}; font-size:0.82rem; margin-bottom:0.35rem;">
        Decision-support options -- the team picks what fits. This report is not prescriptive.</div>
      <ul style="padding-left:1.1rem; margin:0;">{bhtml}</ul>
    </div>""", unsafe_allow_html=True)


def _recommendations(verdict: dict, contribs: list, cost: bool) -> List[str]:
    lv = verdict["drift_level"]
    d = verdict["drift_direction"]
    dp = _phrase(d) if d else None
    ids = ", ".join(t["id"] for t in contribs)
    b: List[str] = []
    if lv == "Aligned":
        if cost:
            b.append("Continue prioritizing rightsizing, cleanup, and savings-plan adoption.")
            b.append("Review new scaling/rollout tickets against demand justification.")
        else:
            b.append("Continue prioritizing on-intent work.")
            b.append("Re-run this check when the backlog changes shape.")
        return b
    if cost:
        b.append("Prioritize rightsizing and decommissioning idle resources.")
        if d in SPEND_INCREASING_THEMES:
            b.append(f"Deprioritize <b>'{_esc(dp)}'</b> unless capacity is genuinely required and spend approved.")
            b.append(f"If growth is now the goal, update the project intent to include <b>'{_esc(dp)}'</b>.")
        elif d:
            b.append(f"Deprioritize <b>'{_esc(dp)}'</b> unless it directly supports cost optimization.")
            b.append(f"If the goal has shifted toward <b>'{_esc(dp)}'</b>, update the intent.")
        else:
            b.append("If the project goal has shifted, update the intent explicitly.")
    else:
        b.append("Continue prioritizing on-intent work.")
        if d:
            b.append(f"Deprioritize <b>'{_esc(dp)}'</b> unless it supports the current intent.")
            b.append(f"If <b>'{_esc(dp)}'</b> is now the goal, update the project intent.")
        else:
            b.append("Review borderline tickets: re-scope onto intent or broaden the intent.")
    if ids:
        b.append(f"Tickets worth a second look: <code>{_esc(ids)}</code>.")
    return b


# ---------------------------------------------------------------------------
# Tab 4 - Pipeline Transparency
# ---------------------------------------------------------------------------
def _tab_pipeline(classification: dict, verdict: dict, cost: bool):
    st.markdown(f"""
    <div class="tp-card" style="margin-bottom:1rem;">
      <div class="tp-sh">How TruePath AI Works</div>
      <div style="color:#cbd5e1; font-size:0.90rem; line-height:1.6;">
        <b>1. Intent Analysis</b> -- parses the project description into a primary goal,
        supporting goals, constraints, and intent themes using rule-based NLP
        (sentence segmentation, goal-verb scoring, keyword extraction).<br>
        <b>2. Ticket Classification</b> -- each ticket is matched against a 12-theme
        lexicon ({len(THEME_LEXICON)} themes, {sum(len(v) for v in THEME_LEXICON.values())} keywords).
        When the semantic model is available, tickets with zero lexicon hits are rescued
        by sentence-embedding cosine similarity.<br>
        <b>3. Drift Detection</b> -- compares intent themes vs ticket themes using two
        transparent rules: <em>proportion</em> (aligned share) and <em>direction</em>
        (off-intent concentration). Three thresholds produce the verdict.<br>
        <b>4. Explanation</b> -- turns the structured verdict into a cloud-language report
        with evidence, contributors, and decision-support recommendations.
      </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown('<div class="tp-sh">Theme Lexicon Browser</div>', unsafe_allow_html=True)
        selected = st.selectbox("Theme", list(THEME_LEXICON.keys()), format_func=_phrase,
                                label_visibility="collapsed")
        if selected:
            kws = THEME_LEXICON[selected]
            chips = "".join(f'<span class="tp-chip tp-chip-kw">{_esc(k)}</span>' for k in kws)
            finops = selected in {"cost_optimization", "efficiency", "rightsizing", "cleanup"}
            tag = '<span class="tp-chip tp-chip-intent">FinOps</span>' if finops else '<span class="tp-chip tp-chip-neutral">DevOps</span>'
            st.markdown(f"""<div class="tp-card">
              <div style="margin-bottom:0.4rem;">{tag}
                <span style="color:{C_MUTED}; font-size:0.82rem; margin-left:0.5rem;">{len(kws)} keywords</span></div>
              {chips}</div>""", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="tp-sh">Classification Pipeline Status</div>', unsafe_allow_html=True)
        sem = classification.get("semantic_enabled", False)
        sem_color = C_TEAL if sem else C_SLATE
        sem_label = "Active" if sem else "Offline (lexicon-only)"
        dominant = classification.get("dominant_themes", [])
        dom_chips = "".join(f'<span class="tp-chip tp-chip-intent">{_esc(_phrase(d))}</span>' for d in dominant)

        st.markdown(f"""<div class="tp-card">
          <div style="margin-bottom:0.5rem;">
            <span style="color:{C_MUTED}; font-size:0.82rem;">Semantic model:</span>
            <span class="tp-chip" style="background:{sem_color}15; border:1px solid {sem_color}; color:{sem_color};">{sem_label}</span>
          </div>
          <div style="margin-bottom:0.5rem;">
            <span style="color:{C_MUTED}; font-size:0.82rem;">Dominant themes:</span><br>{dom_chips or '<span style="color:'+C_MUTED+';">none</span>'}
          </div>
          <div>
            <span style="color:{C_MUTED}; font-size:0.82rem;">Cost-optimization mode:</span>
            <span style="color:{'#f1f5f9' if cost else C_MUTED}; font-weight:600;">{'Yes' if cost else 'No'}</span>
          </div>
        </div>""", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="tp-sh">Decision Rules</div>', unsafe_allow_html=True)
    am = st.session_state.get("aligned_min", 0.70)
    dm = st.session_state.get("drift_min", 0.40)
    dt = st.session_state.get("direction_threshold", 0.35)
    st.markdown(f"""
    <div class="tp-card">
      <table style="width:100%; color:#cbd5e1; font-size:0.88rem; border-collapse:collapse;">
        <tr style="border-bottom:1px solid rgba(148,163,184,0.12);">
          <td style="padding:0.5rem 0.6rem; color:{C_MUTED};">Aligned share</td>
          <td style="padding:0.5rem 0.6rem; color:{C_MUTED};">Direction share</td>
          <td style="padding:0.5rem 0.6rem; color:{C_MUTED};">Verdict</td></tr>
        <tr style="border-bottom:1px solid rgba(148,163,184,0.08);">
          <td style="padding:0.5rem 0.6rem;">&ge; {am:.0%}</td>
          <td style="padding:0.5rem 0.6rem;">--</td>
          <td style="padding:0.5rem 0.6rem;"><span style="color:{C_TEAL}; font-weight:600;">Aligned</span></td></tr>
        <tr style="border-bottom:1px solid rgba(148,163,184,0.08);">
          <td style="padding:0.5rem 0.6rem;">{dm:.0%} -- {am:.0%}</td>
          <td style="padding:0.5rem 0.6rem;">&lt; {dt:.0%}</td>
          <td style="padding:0.5rem 0.6rem;"><span style="color:{C_AMBER}; font-weight:600;">Partial Drift</span></td></tr>
        <tr style="border-bottom:1px solid rgba(148,163,184,0.08);">
          <td style="padding:0.5rem 0.6rem;">{dm:.0%} -- {am:.0%}</td>
          <td style="padding:0.5rem 0.6rem;">&ge; {dt:.0%} (escalation)</td>
          <td style="padding:0.5rem 0.6rem;"><span style="color:{C_RED}; font-weight:600;">Significant Drift</span></td></tr>
        <tr>
          <td style="padding:0.5rem 0.6rem;">&lt; {dm:.0%}</td>
          <td style="padding:0.5rem 0.6rem;">--</td>
          <td style="padding:0.5rem 0.6rem;"><span style="color:{C_RED}; font-weight:600;">Significant Drift</span></td></tr>
      </table>
    </div>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tab 5 - Raw Report
# ---------------------------------------------------------------------------
def _tab_report(intent_text: str, tasks: List[dict], title: str):
    if not tasks:
        st.info("Add tickets to generate a report.")
        return
    report = explain_drift(intent_text, tasks, project_title=title)
    st.caption("The exact output of `explain_drift()` -- paste into Slack or a slide deck.")
    st.markdown('<div class="tp-report">', unsafe_allow_html=True)
    st.code(report, language="text")
    st.markdown("</div>", unsafe_allow_html=True)
    st.download_button("Download report (.txt)", data=report,
                       file_name=f"truepath_{title.lower().replace(' ','_')}.txt",
                       mime="text/plain")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    _bootstrap()
    _sidebar()

    title = st.session_state.get("title", "Untitled Project")
    intent_text = st.session_state.get("intent", "")
    all_raw_tasks = st.session_state.get("tasks", [])

    _header(title)

    # ---- Month picker -----------------------------------------------------
    # Only shown when the loaded data has month metadata. Reviewers can
    # flip between "All months" / April / May / June and watch the
    # verdict shift as the project drifts further from intent.
    months = _available_months(all_raw_tasks)
    if months:
        options = [ALL_MONTHS_LABEL] + months
        # Show counts inline so the picker doubles as a quick summary.
        counts = {ALL_MONTHS_LABEL: len(all_raw_tasks)}
        for m in months:
            counts[m] = sum(1 for t in all_raw_tasks if t.get("month_label") == m)
        current = st.session_state.get("month_filter", ALL_MONTHS_LABEL)
        if current not in options:
            current = ALL_MONTHS_LABEL
        st.markdown(
            '<div class="tp-sh" style="margin-bottom:0.35rem;">'
            'Timeline filter</div>',
            unsafe_allow_html=True,
        )
        picked = st.radio(
            "month_filter_radio",
            options,
            index=options.index(current),
            horizontal=True,
            label_visibility="collapsed",
            format_func=lambda m: f"{m}  ({counts.get(m, 0)})",
            key="month_filter",
        )
    else:
        picked = ALL_MONTHS_LABEL

    raw_tasks = _filter_tasks_by_month(all_raw_tasks, picked)
    pipeline_tasks = _clean_tasks(raw_tasks)

    if picked != ALL_MONTHS_LABEL:
        st.caption(
            f"Showing **{picked}** only - {len(pipeline_tasks)} of "
            f"{len(all_raw_tasks)} tickets. Verdict, KPIs, theme chart, "
            f"and contributors below all reflect this month's slice."
        )

    if not intent_text.strip():
        st.warning("Enter a project intent in the sidebar to begin analysis.")
        return

    intent_themes = extract_intent_themes(intent_text)
    cost = _is_cost_project(intent_themes)

    if not pipeline_tasks:
        st.markdown(f"""
        <div class="tp-hero" style="background:{C_SURFACE};">
          <div class="tp-hero-pill" style="background:{C_SLATE}18; color:{C_SLATE}; border:1px solid {C_SLATE};">NO TICKETS</div>
          <div class="tp-hero-title">Add tickets to evaluate drift.</div>
          <div class="tp-hero-sub">Use the <b>Ticket Backlog</b> tab or load a JSON from the sidebar.</div>
        </div>""", unsafe_allow_html=True)
        verdict = dict(drift_level="Aligned", aligned_share=0.0, intent_themes=intent_themes,
                       task_theme_distribution=[], drift_direction=None,
                       counts=dict(aligned=0, off_intent=0, unthemed=0, total=0),
                       reason="No tickets to evaluate.")
        classification = dict(tasks=[], themes=[], dominant_themes=[], semantic_enabled=False)
    else:
        verdict = detect_intent_drift(intent_text, pipeline_tasks,
                                      aligned_min=st.session_state["aligned_min"],
                                      drift_min=st.session_state["drift_min"],
                                      direction_threshold=st.session_state["direction_threshold"])
        classification = classify_tasks(pipeline_tasks)
        _hero(verdict, cost)

    st.write("")
    _kpis(verdict, cost)
    st.write("")

    tabs = st.tabs(["Intent Analysis",
                     f"Ticket Backlog ({len(pipeline_tasks)})",
                     "Drift Detection",
                     "Pipeline & Rules",
                     "Raw Report"])

    with tabs[0]:
        _tab_intent(intent_text, intent_themes)
    with tabs[1]:
        _tab_tickets(classification, intent_themes, raw_tasks, picked)
    with tabs[2]:
        _tab_drift(verdict, classification, cost)
    with tabs[3]:
        _tab_pipeline(classification, verdict, cost)
    with tabs[4]:
        _tab_report(intent_text, pipeline_tasks, title)


if __name__ == "__main__":
    main()
