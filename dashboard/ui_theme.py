"""
AI-Based Intelligent Trading System — Enterprise Terminal UI Theme & Design System
Zinc/Slate Flat Enterprise Palette with Functional Emerald Accent.
Strict typography (Inter/System Sans 11-14px), tabular numerals, 1px crisp borders.
"""

from typing import List, Dict, Optional, Any
import plotly.graph_objects as go
import streamlit as st

# ==============================================================================
# 1. STRICT DESIGN TOKENS
# ==============================================================================
COLOR_BG = "#09090B"            # Application background
COLOR_SURFACE_PRI = "#111113"   # Primary surface / cards
COLOR_SURFACE_SEC = "#18181B"   # Secondary surface / controls
COLOR_BORDER = "#27272A"        # Crisp 1px borders
COLOR_TEXT_PRI = "#FAFAFA"      # Primary high-contrast text
COLOR_TEXT_SEC = "#A1A1AA"      # Secondary neutral text
COLOR_TEXT_MUTED = "#71717A"    # Subtitles, labels, metadata
COLOR_EMERALD = "#10B981"       # Positive / Bullish / Live
COLOR_AMBER = "#F59E0B"         # Warning / Neutral / Hold
COLOR_RED = "#EF4444"           # Negative / Bearish / Sell
COLOR_CYAN = "#06B6D4"          # Analytical / Informational highlight

# ==============================================================================
# 2. GLOBAL TERMINAL CSS INJECTION
# ==============================================================================
TERMINAL_CSS = f"""
<style>
    /* Global Base Reset */
    html, body, [data-testid="stAppViewContainer"] {{
        background-color: {COLOR_BG} !important;
        color: {COLOR_TEXT_PRI} !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
        font-size: 13px !important;
        letter-spacing: -0.01em !important;
    }}

    /* Viewport filling workspace container */
    .main .block-container {{
        padding: 0.6rem 1.2rem 1.0rem 1.2rem !important;
        max-width: 100% !important;
    }}

    /* Tabular Numerals for Financial Data */
    .tabular-num, td, th, .stMetric, [data-testid="stMetricValue"] {{
        font-variant-numeric: tabular-nums !important;
        font-feature-settings: "tnum" 1 !important;
    }}

    /* Sidebar Refinement */
    [data-testid="stSidebar"] {{
        background-color: {COLOR_SURFACE_PRI} !important;
        border-right: 1px solid {COLOR_BORDER} !important;
        padding-top: 0.5rem !important;
    }}
    [data-testid="stSidebar"] .block-container {{
        padding-top: 1.0rem !important;
        padding-bottom: 1.0rem !important;
        padding-left: 0.9rem !important;
        padding-right: 0.9rem !important;
    }}
    [data-testid="stSidebar"] hr {{
        border-color: {COLOR_BORDER} !important;
        margin: 0.6rem 0 !important;
    }}

    /* Streamlit Widget Styling Overrides */
    div[data-testid="stRadio"] > label {{
        font-size: 11px !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        color: {COLOR_TEXT_MUTED} !important;
        letter-spacing: 0.05em !important;
    }}
    div[data-testid="stRadio"] div[role="radiogroup"] > label {{
        padding: 0.25rem 0.4rem !important;
        font-size: 12px !important;
        color: {COLOR_TEXT_SEC} !important;
    }}
    div[data-testid="stRadio"] div[role="radiogroup"] > label:hover {{
        color: {COLOR_TEXT_PRI} !important;
    }}

    /* Selectboxes and Inputs */
    div[data-baseweb="select"] > div {{
        background-color: {COLOR_SURFACE_SEC} !important;
        border: 1px solid {COLOR_BORDER} !important;
        border-radius: 2px !important;
        color: {COLOR_TEXT_PRI} !important;
        font-size: 12px !important;
        min-height: 32px !important;
    }}
    div[data-baseweb="popover"] ul {{
        background-color: {COLOR_SURFACE_SEC} !important;
        border: 1px solid {COLOR_BORDER} !important;
    }}

    /* Buttons */
    div.stButton > button {{
        background-color: {COLOR_SURFACE_SEC} !important;
        color: {COLOR_TEXT_PRI} !important;
        border: 1px solid {COLOR_BORDER} !important;
        border-radius: 2px !important;
        font-size: 11px !important;
        font-weight: 600 !important;
        letter-spacing: 0.03em !important;
        text-transform: uppercase !important;
        padding: 0.35rem 0.75rem !important;
        min-height: 28px !important;
        transition: none !important;
    }}
    div.stButton > button:hover {{
        background-color: {COLOR_BORDER} !important;
        border-color: {COLOR_TEXT_MUTED} !important;
        color: #FFFFFF !important;
    }}
    div.stButton > button:active {{
        background-color: {COLOR_SURFACE_PRI} !important;
    }}

    /* Tabs Styling */
    div[data-testid="stTabs"] button[role="tab"] {{
        font-size: 12px !important;
        font-weight: 600 !important;
        color: {COLOR_TEXT_MUTED} !important;
        padding: 0.4rem 0.8rem !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        background: transparent !important;
    }}
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{
        color: {COLOR_TEXT_PRI} !important;
        border-bottom: 2px solid {COLOR_EMERALD} !important;
    }}

    /* Expanders */
    div[data-testid="stExpander"] {{
        background-color: {COLOR_SURFACE_PRI} !important;
        border: 1px solid {COLOR_BORDER} !important;
        border-radius: 2px !important;
        margin-bottom: 0.5rem !important;
    }}
    div[data-testid="stExpander"] summary {{
        font-size: 12px !important;
        font-weight: 600 !important;
        color: {COLOR_TEXT_SEC} !important;
        padding: 0.4rem 0.6rem !important;
    }}
    div[data-testid="stExpander"] summary:hover {{
        color: {COLOR_TEXT_PRI} !important;
    }}

    /* Tables & DataFrames */
    [data-testid="stDataFrame"] {{
        border: 1px solid {COLOR_BORDER} !important;
        border-radius: 2px !important;
        background-color: {COLOR_SURFACE_PRI} !important;
    }}

    /* Custom Terminal Header / Status Bar */
    .top-status-bar {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        background-color: {COLOR_SURFACE_PRI};
        border-bottom: 1px solid {COLOR_BORDER};
        padding: 0.4rem 0.75rem;
        margin: -0.6rem -1.2rem 0.8rem -1.2rem;
        font-size: 11px;
    }}
    .top-status-left {{
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }}
    .top-system-title {{
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 0.06em;
        color: {COLOR_TEXT_PRI};
        text-transform: uppercase;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }}
    .top-system-title::before {{
        content: "";
        display: inline-block;
        width: 6px;
        height: 6px;
        background-color: {COLOR_EMERALD};
        border-radius: 1px;
    }}
    .top-context-label {{
        color: {COLOR_TEXT_MUTED};
        font-weight: 500;
    }}
    .top-context-val {{
        color: {COLOR_TEXT_PRI};
        font-weight: 600;
        font-family: monospace;
    }}
    .top-status-right {{
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }}

    /* Status Badges */
    .term-badge {{
        display: inline-flex;
        align-items: center;
        font-size: 10px;
        font-weight: 700;
        font-family: monospace;
        letter-spacing: 0.05em;
        padding: 0.15rem 0.45rem;
        border-radius: 2px;
        text-transform: uppercase;
        border: 1px solid {COLOR_BORDER};
        background: {COLOR_SURFACE_SEC};
        color: {COLOR_TEXT_SEC};
    }}
    .term-badge.badge-live {{
        border-color: {COLOR_EMERALD};
        color: {COLOR_EMERALD};
        background: rgba(16, 185, 129, 0.08);
    }}
    .term-badge.badge-warn {{
        border-color: {COLOR_AMBER};
        color: {COLOR_AMBER};
        background: rgba(245, 158, 11, 0.08);
    }}
    .term-badge.badge-err {{
        border-color: {COLOR_RED};
        color: {COLOR_RED};
        background: rgba(239, 68, 68, 0.08);
    }}
    .term-badge.badge-cyan {{
        border-color: {COLOR_CYAN};
        color: {COLOR_CYAN};
        background: rgba(6, 182, 212, 0.08);
    }}

    /* Section Bar */
    .term-section {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: {COLOR_TEXT_SEC};
        border-bottom: 1px solid {COLOR_BORDER};
        padding: 0.35rem 0;
        margin-top: 0.75rem;
        margin-bottom: 0.5rem;
    }}
    .term-section-meta {{
        font-size: 10px;
        color: {COLOR_TEXT_MUTED};
        font-family: monospace;
        font-weight: 400;
    }}

    /* Dense KPI Metric Strip */
    .kpi-strip {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
        gap: 1px;
        background-color: {COLOR_BORDER};
        border: 1px solid {COLOR_BORDER};
        border-radius: 2px;
        margin-bottom: 0.65rem;
    }}
    .kpi-cell {{
        background-color: {COLOR_SURFACE_PRI};
        padding: 0.45rem 0.65rem;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }}
    .kpi-cell-label {{
        font-size: 10px;
        font-weight: 600;
        color: {COLOR_TEXT_MUTED};
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.15rem;
    }}
    .kpi-cell-val {{
        font-size: 14px;
        font-weight: 700;
        color: {COLOR_TEXT_PRI};
        line-height: 1.15;
        font-variant-numeric: tabular-nums;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
    }}
    .kpi-cell-sub {{
        font-size: 10px;
        color: {COLOR_TEXT_MUTED};
        margin-top: 0.15rem;
        font-variant-numeric: tabular-nums;
    }}

    /* Signal Cards */
    .panel-box {{
        background-color: {COLOR_SURFACE_PRI};
        border: 1px solid {COLOR_BORDER};
        border-radius: 2px;
        padding: 0.6rem 0.75rem;
        margin-bottom: 0.45rem;
    }}
    .signal-row {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 11px;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }}

    /* Alert / Disclosure Box */
    .term-alert {{
        background-color: {COLOR_SURFACE_PRI};
        border: 1px solid {COLOR_BORDER};
        border-left: 3px solid {COLOR_AMBER};
        padding: 0.45rem 0.75rem;
        font-size: 11px;
        color: {COLOR_TEXT_SEC};
        line-height: 1.4;
        margin-bottom: 0.6rem;
    }}
    .term-alert.alert-info {{
        border-left-color: {COLOR_CYAN};
    }}
    .term-alert.alert-emerald {{
        border-left-color: {COLOR_EMERALD};
    }}
    .term-alert.alert-red {{
        border-left-color: {COLOR_RED};
    }}

    /* Semantic Text Utilities */
    .pos {{ color: {COLOR_EMERALD} !important; }}
    .neg {{ color: {COLOR_RED} !important; }}
    .warn {{ color: {COLOR_AMBER} !important; }}
    .info {{ color: {COLOR_CYAN} !important; }}
    .muted {{ color: {COLOR_TEXT_MUTED} !important; }}
</style>
"""


# ==============================================================================
# 3. PLOTLY ENTERPRISE TERMINAL THEME HELPER
# ==============================================================================
def apply_terminal_layout(
    fig: go.Figure,
    height: Optional[int] = None,
    title: Optional[str] = None,
    show_legend: bool = True
) -> go.Figure:
    """
    Apply uniform, hyper-dense Zinc/Slate styling to any Plotly figure.
    Removes gradients, sets subtle 1px gridlines, compact axis labels,
    and minimal margins to maximize vertical workspace.
    """
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLOR_SURFACE_PRI,
        plot_bgcolor=COLOR_SURFACE_PRI,
        font=dict(
            family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            size=10,
            color=COLOR_TEXT_SEC
        ),
        margin=dict(l=35, r=15, t=28 if title else 14, b=20),
        showlegend=show_legend,
        legend=dict(
            bgcolor=COLOR_SURFACE_SEC,
            bordercolor=COLOR_BORDER,
            borderwidth=1,
            font=dict(size=9, color=COLOR_TEXT_SEC),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0
        ),
        xaxis=dict(
            gridcolor=COLOR_BORDER,
            zerolinecolor=COLOR_BORDER,
            showline=True,
            linecolor=COLOR_BORDER,
            tickfont=dict(size=9, color=COLOR_TEXT_MUTED)
        ),
        yaxis=dict(
            gridcolor=COLOR_BORDER,
            zerolinecolor=COLOR_BORDER,
            showline=True,
            linecolor=COLOR_BORDER,
            tickfont=dict(size=9, color=COLOR_TEXT_MUTED)
        )
    )
    if height:
        fig.update_layout(height=height)
    if title:
        fig.update_layout(
            title=dict(
                text=title.upper(),
                font=dict(size=11, color=COLOR_TEXT_PRI, family="monospace"),
                x=0.01,
                y=0.98,
                xanchor="left",
                yanchor="top"
            )
        )
    return fig


# ==============================================================================
import textwrap

# ==============================================================================
# 4. REUSABLE UI RENDERERS
# ==============================================================================
def render_html(html_str: str) -> None:
    """Renders HTML in Streamlit ensuring no line has leading whitespace that could trigger Markdown code blocks."""
    cleaned = "\n".join(line.strip() for line in html_str.splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


def render_top_status_bar(
    page_title: str,
    ticker: Optional[str] = None,
    date_str: Optional[str] = None,
    mode: str = "OFFLINE",
    stream_badge: Optional[str] = None
) -> None:
    """Renders the compact single-line top status bar."""
    ticker_html = f"<span style='color:{COLOR_BORDER};'>|</span> <span class='top-context-label'>ASSET:</span> <span class='top-context-val'>{ticker}</span>" if ticker else ""
    date_html = f"<span style='color:{COLOR_BORDER};'>|</span> <span class='top-context-label'>SESSION:</span> <span class='top-context-val'>{date_str}</span>" if date_str else ""
    
    badge_cls = "badge-live" if mode == "LIVE" else ("badge-warn" if mode == "DEMO" else "term-badge")
    mode_badge = f"<span class='term-badge {badge_cls}'>FEED: {mode}</span>"
    stream_badge_html = f"<span class='term-badge {badge_cls}'>{stream_badge}</span>" if stream_badge else ""

    render_html(f"""
    <div class="top-status-bar">
        <div class="top-status-left">
            <span class="top-system-title">AI TRADING SYSTEM</span>
            <span style="color:{COLOR_BORDER};">|</span>
            <span style="color:{COLOR_TEXT_PRI}; font-weight:600;">{page_title.upper()}</span>
            {ticker_html}
            {date_html}
        </div>
        <div class="top-status-right">
            {stream_badge_html}
            {mode_badge}
        </div>
    </div>
    """)


def render_kpi_strip(items: List[Dict[str, Any]]) -> None:
    """
    Renders a hyper-dense, 1px bordered metric strip.
    items: list of dicts with keys: 'label', 'val', 'sub' (optional), 'cls' (optional)
    """
    cells = []
    for item in items:
        lbl = str(item.get("label", "")).upper()
        val = str(item.get("val", ""))
        sub = str(item.get("sub", ""))
        cls_name = item.get("cls", "")
        
        sub_html = f"<div class='kpi-cell-sub'>{sub}</div>" if sub else ""
        cells.append(f'<div class="kpi-cell"><div class="kpi-cell-label">{lbl}</div><div class="kpi-cell-val {cls_name}">{val}</div>{sub_html}</div>')
        
    render_html(f'<div class="kpi-strip">{"".join(cells)}</div>')


def render_section_header(title: str, meta: Optional[str] = None) -> None:
    """Renders a clean, compact section header with optional right-aligned metadata."""
    meta_html = f"<span class='term-section-meta'>{meta}</span>" if meta else ""
    render_html(f'<div class="term-section"><span>{title.upper()}</span>{meta_html}</div>')


def render_alert_box(text: str, level: str = "warning", title: Optional[str] = None) -> None:
    """Renders a restrained disclosure/alert box."""
    lvl_cls = "alert-emerald" if level == "success" else ("alert-info" if level == "info" else ("alert-red" if level == "error" else ""))
    title_html = f"<div style='font-weight:700; color:{COLOR_TEXT_PRI}; margin-bottom:2px;'>{title.upper()}</div>" if title else ""
    render_html(f'<div class="term-alert {lvl_cls}">{title_html}{text}</div>')

