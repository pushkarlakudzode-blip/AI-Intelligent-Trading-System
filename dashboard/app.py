"""
AI-Based Intelligent Trading System - Modern Trading Terminal (Enterprise V3)
Presentation & Real-Time Trading Intelligence Layer

Author: Pushkar
Universe: 10 Companies (AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, JPM, V, JNJ)
Data Period: 2019-01-15 to 2024-12-30 (Historical / Processed Offline Dataset)
Out-of-Sample Evaluation: 2023-10-20 to 2024-12-30 (300 Sessions per Company)
Engine: Streamlit + Plotly (Dense Zinc/Slate Enterprise Architecture)
"""

import sys
import os
from pathlib import Path
import datetime

# Ensure project root is in sys.path for robust module resolution
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Sync Streamlit Cloud secrets to os.environ if present
try:
    if hasattr(st, "secrets"):
        for _sec_k, _sec_v in st.secrets.items():
            if isinstance(_sec_v, str) and _sec_k not in os.environ:
                os.environ[_sec_k] = _sec_v
except Exception:
    pass

# Import Unified Enterprise Design System
from dashboard.ui_theme import (
    TERMINAL_CSS,
    apply_terminal_layout,
    render_top_status_bar,
    render_kpi_strip,
    render_section_header,
    render_alert_box,
    render_html,
    COLOR_BG,
    COLOR_SURFACE_PRI,
    COLOR_SURFACE_SEC,
    COLOR_BORDER,
    COLOR_TEXT_PRI,
    COLOR_TEXT_SEC,
    COLOR_TEXT_MUTED,
    COLOR_EMERALD,
    COLOR_AMBER,
    COLOR_RED,
    COLOR_CYAN,
)

# Import Grounded Decision Explainer
from src.analytics.decision_explainer import (
    DecisionExplainer,
    render_decision_explanation_ui,
)

@st.cache_resource
def get_decision_explainer():
    """Cached singleton instance of DecisionExplainer."""
    return DecisionExplainer()

# ==============================================================================
# 1. PAGE CONFIGURATION & CSS INJECTION
# ==============================================================================
st.set_page_config(
    page_title="AI Trading System | Terminal",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject strict enterprise CSS
st.markdown(TERMINAL_CSS, unsafe_allow_html=True)

# Project paths
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

COMPANY_MAP = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet",
    "AMZN": "Amazon",
    "NVDA": "NVIDIA",
    "META": "Meta Platforms",
    "TSLA": "Tesla",
    "JPM": "JPMorgan Chase",
    "V": "Visa",
    "JNJ": "Johnson & Johnson"
}

# ==============================================================================
# 2. CACHED DATA LOADERS & VALIDATION
# ==============================================================================
@st.cache_data(show_spinner=False)
def load_dataset(filename: str):
    """Safely load a processed CSV dataset with caching."""
    filepath = PROCESSED_DIR / filename
    if not filepath.exists():
        return None
    try:
        df = pd.read_csv(filepath)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
        return df
    except Exception as e:
        st.warning(f"Error loading {filename}: {e}")
        return None


@st.cache_data(show_spinner=False)
def load_all_data():
    """Load all 20 processed pipeline outputs into a cached dictionary."""
    data = {}
    files = {
        "features": "multi_company_features.csv",
        "pca": "multi_company_pca.csv",
        "reg_preds": "multi_company_regression_predictions.csv",
        "reg_summary": "multi_company_regression_summary.csv",
        "regul_results": "multi_company_regularization_results.csv",
        "regul_summary": "multi_company_regularization_summary.csv",
        "class_results": "multi_company_classification_results.csv",
        "class_summary": "multi_company_classification_summary.csv",
        "class_cm": "multi_company_classification_confusion_matrices.csv",
        "clust_results": "multi_company_clustering_results.csv",
        "clust_summary": "multi_company_cluster_summary.csv",
        "clust_company": "multi_company_cluster_by_company.csv",
        "rl_results": "multi_company_rl_results.csv",
        "rl_summary": "multi_company_rl_summary.csv",
        "rl_policy": "multi_company_rl_policy.csv",
        "news_daily": "multi_company_daily_news_signal.csv",
        "news_summary": "multi_company_news_summary.csv",
        "news_nlp": "multi_company_news_nlp_results.csv",
        "fusion": "multi_company_decision_fusion.csv",
        "fusion_summary": "multi_company_decision_fusion_summary.csv"
    }
    
    missing_files = []
    for key, fname in files.items():
        loaded = load_dataset(fname)
        data[key] = loaded
        if loaded is None and key not in ["news_nlp"]:
            missing_files.append(fname)
            
    return data, missing_files


# Execute cached load
DATA, MISSING = load_all_data()

if MISSING:
    render_alert_box(
        f"Missing required processed files in `{PROCESSED_DIR}`: " + ", ".join(MISSING),
        level="error",
        title="Data Ingestion Error"
    )
    st.stop()

# Helper aliases
df_feat = DATA["features"]
df_fusion = DATA["fusion"]
df_fusion_summary = DATA["fusion_summary"]

# ==============================================================================
# 2B. REAL-DATA ANALYTICS LOADERS (CACHED)
# ==============================================================================
from src.analytics.refresh_manager import execute_real_data_refresh, get_last_refresh_info
from src.data.alpaca_stream_listener import (
    get_stream_status,
    start_stream_listener,
    stop_stream_listener,
)

LIVE_ANALYTICS_DIR = PROJECT_ROOT / "data" / "live" / "analytics"
LIVE_DB_PATH = PROJECT_ROOT / "data" / "live" / "news_database.db"
LIVE_MARKET_PATH = PROJECT_ROOT / "data" / "live" / "market" / "historical_market_extension.csv"


@st.cache_data(show_spinner=False)
def load_real_market_summary():
    fp = LIVE_ANALYTICS_DIR / "real_market_summary.csv"
    if fp.exists():
        try:
            return pd.read_csv(fp)
        except Exception:
            return None
    return None


@st.cache_data(show_spinner=False)
def load_real_news_summary():
    fp = LIVE_ANALYTICS_DIR / "real_news_summary.csv"
    if fp.exists():
        try:
            return pd.read_csv(fp)
        except Exception:
            return None
    return None


@st.cache_data(show_spinner=False)
def load_real_news_daily_summary():
    fp = LIVE_ANALYTICS_DIR / "real_news_daily_summary.csv"
    if fp.exists():
        try:
            return pd.read_csv(fp)
        except Exception:
            return None
    return None


@st.cache_data(show_spinner=False)
def load_real_data_coverage():
    fp = LIVE_ANALYTICS_DIR / "real_data_coverage.csv"
    if fp.exists():
        try:
            return pd.read_csv(fp)
        except Exception:
            return None
    return None


@st.cache_data(show_spinner=False)
def load_real_market_extension():
    if LIVE_MARKET_PATH.exists():
        try:
            df = pd.read_csv(LIVE_MARKET_PATH)
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
            return df
        except Exception:
            return None
    return None


@st.cache_data(show_spinner=False)
def load_latest_real_articles(ticker="AAPL", limit=20):
    if LIVE_DB_PATH.exists():
        try:
            import sqlite3
            conn = sqlite3.connect(f"file:{LIVE_DB_PATH}?mode=ro", uri=True)
            q = f"""
            SELECT a.headline, a.source, a.published_at_utc,
                   n.sentiment_label, n.sentiment_score, n.relevance_score,
                   n.importance_score, n.triage_score, n.event_type, a.url, a.provider,
                   a.news_id, a.summary, a.ticker, a.company_name
            FROM articles a
            JOIN nlp_results n ON a.news_id = n.news_id
            WHERE a.ticker = '{ticker}' AND a.dataset_type = 'REAL'
            ORDER BY a.published_at_utc DESC
            LIMIT {limit}
            """
            df = pd.read_sql(q, conn)
            conn.close()
            return df
        except Exception:
            pass
    return None


# ==============================================================================
# 3. TECHNICAL INDICATORS (VISUALIZATION ONLY)
# ==============================================================================
@st.cache_data(show_spinner=False)
def compute_visualization_technicals(df_company: pd.DataFrame) -> pd.DataFrame:
    """Computes visualization-only technical indicators (EMA, MACD, RSI, BB, Drawdown)."""
    df = df_company.copy().sort_values("Date").reset_index(drop=True)
    close = df["Close"]
    
    # 1. EMAs
    df["EMA_12"] = close.ewm(span=12, adjust=False).mean()
    df["EMA_26"] = close.ewm(span=26, adjust=False).mean()
    
    # 2. MACD
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    
    # 3. RSI 14
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI_14"] = 100 - (100 / (1 + rs))
    
    # 4. Bollinger Bands (20, 2)
    sma_20 = close.rolling(20).mean()
    rstd_20 = close.rolling(20).std()
    df["BB_Upper"] = sma_20 + 2 * rstd_20
    df["BB_Lower"] = sma_20 - 2 * rstd_20
    df["BB_Mid"] = sma_20
    
    # 5. Drawdown
    peak = close.cummax()
    df["Drawdown"] = (close - peak) / peak
    
    return df


# ==============================================================================
# 4. SIDEBAR NAVIGATION & COMPANY SELECTOR
# ==============================================================================
st.sidebar.markdown(f"""
<div style="padding: 0.2rem 0; border-bottom: 1px solid {COLOR_BORDER}; margin-bottom: 0.6rem;">
    <div style="font-size: 11px; font-weight: 800; color: {COLOR_TEXT_PRI}; letter-spacing: 0.06em; text-transform: uppercase;">
        AI TRADING SYSTEM
    </div>
    <div style="font-size: 10px; color: {COLOR_TEXT_MUTED}; font-family: monospace;">
        WORKSPACE TERMINAL V3
    </div>
</div>
""", unsafe_allow_html=True)

# Status Badges
st.sidebar.markdown(f"""
<div style="display: flex; gap: 0.35rem; margin-bottom: 0.6rem;">
    <span class="term-badge">DATA: OFFLINE</span>
    <span class="term-badge badge-warn">NEWS: DEMO</span>
</div>
""", unsafe_allow_html=True)

# Asset Universe Selector
st.sidebar.markdown(f"<div style='font-size:10px; font-weight:700; color:{COLOR_TEXT_MUTED}; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.25rem;'>ASSET UNIVERSE (10 EQUITIES)</div>", unsafe_allow_html=True)
company_options = [f"{t} — {COMPANY_MAP[t]}" for t in COMPANY_MAP.keys()]
selected_company_str = st.sidebar.selectbox("Active Asset", options=company_options, index=0, label_visibility="collapsed")
selected_ticker = selected_company_str.split(" — ")[0]
selected_company_name = COMPANY_MAP[selected_ticker]

st.sidebar.markdown(f"""
<div style="background:{COLOR_SURFACE_SEC}; border:1px solid {COLOR_BORDER}; border-radius:2px; padding:0.4rem 0.6rem; margin-top:0.3rem; margin-bottom:0.6rem;">
    <div style="font-size:10px; color:{COLOR_TEXT_MUTED}; font-weight:600; text-transform:uppercase;">ACTIVE ASSET</div>
    <div style="font-size:12px; font-weight:700; color:{COLOR_TEXT_PRI}; font-family:monospace;">{selected_ticker} • {selected_company_name}</div>
    <div style="font-size:10px; color:{COLOR_TEXT_SEC}; font-family:monospace; margin-top:0.15rem;">2019-01-15 to 2024-12-30 (1,501 SESSIONS)</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown(f"<hr style='border-color:{COLOR_BORDER}; margin:0.5rem 0;'>", unsafe_allow_html=True)

# Navigation Menu (8 strictly formatted technical views)
st.sidebar.markdown(f"<div style='font-size:10px; font-weight:700; color:{COLOR_TEXT_MUTED}; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.25rem;'>NAVIGATION</div>", unsafe_allow_html=True)
nav_selection = st.sidebar.radio(
    "Terminal Navigation",
    [
        "1. Overview",
        "2. Market Analysis",
        "3. Predictive Models",
        "4. News & NLP",
        "5. Signal Fusion",
        "6. Company Comparison",
        "7. System Overview",
        "8. Real-Time Data"
    ],
    index=0,
    label_visibility="collapsed"
)

st.sidebar.markdown(f"<hr style='border-color:{COLOR_BORDER}; margin:0.5rem 0;'>", unsafe_allow_html=True)
st.sidebar.markdown(f"""
<div style="font-size: 10px; color: {COLOR_TEXT_MUTED}; line-height: 1.35; font-family: monospace;">
    SYSTEM NOTICE: QUANTITATIVE RESEARCH & DECISION-SUPPORT ENGINE. HISTORICAL EVALUATIONS AND REAL-TIME DATA RUN AS INDEPENDENT MODULES.
</div>
""", unsafe_allow_html=True)

# Prepare active company data
df_feat_raw = df_feat[df_feat["Ticker"] == selected_ticker].sort_values("Date").reset_index(drop=True) if df_feat is not None else None
df_feat_comp = compute_visualization_technicals(df_feat_raw) if df_feat_raw is not None else None
df_fusion_comp = df_fusion[df_fusion["Ticker"] == selected_ticker].sort_values("Date").reset_index(drop=True) if df_fusion is not None else None

latest_fusion = df_fusion_comp.iloc[-1] if (df_fusion_comp is not None and len(df_fusion_comp) > 0) else None
latest_date = latest_fusion["Date"] if latest_fusion is not None else "N/A"
latest_feat = df_feat_comp[df_feat_comp["Date"] == latest_date].iloc[0] if (df_feat_comp is not None and len(df_feat_comp[df_feat_comp["Date"] == latest_date]) > 0) else (df_feat_comp.iloc[-1] if df_feat_comp is not None else None)


# ==============================================================================
# 5. PAGE 1 — OVERVIEW
# ==============================================================================
if nav_selection == "1. Overview":
    render_top_status_bar("Overview", selected_ticker, latest_date, mode="OFFLINE")

    if latest_fusion is not None and latest_feat is not None:
        ret_val = latest_feat["Daily_Return"]
        ret_cls = "pos" if ret_val >= 0 else "neg"
        ret_sign = "+" if ret_val >= 0 else ""
        
        dec_val = latest_fusion["Final_Decision"]
        dec_cls = "pos" if dec_val == "BUY" else ("neg" if dec_val == "SELL" else "warn")
        
        score_val = latest_fusion["Fusion_Score"]
        score_cls = "pos" if score_val >= 0.20 else ("neg" if score_val <= -0.20 else "warn")

        # Top 6 KPI Metric Cells
        render_kpi_strip([
            {"label": "Latest Price", "val": f"${latest_feat['Close']:.2f}", "sub": f"Open: ${latest_feat['Open']:.2f}"},
            {"label": "Daily Return", "val": f"{ret_sign}{ret_val:.2%}", "sub": f"Range: ${latest_feat['Price_Range']:.2f}", "cls": ret_cls},
            {"label": "Market Regime", "val": f"{latest_fusion['Market_Regime_Label']}", "sub": f"Cluster {latest_fusion['Market_Regime']}", "cls": "info"},
            {"label": "Final Signal", "val": f"{dec_val}", "sub": "Threshold: |S| ≥ 0.20", "cls": dec_cls},
            {"label": "Fusion Score", "val": f"{score_val:+.4f}", "sub": "Bound: [-1.0, +1.0]", "cls": score_cls},
            {"label": "Confidence", "val": f"{latest_fusion['Heuristic_Confidence_Percent']:.1f}%", "sub": "Heuristic (|S|)", "cls": "info"}
        ])

        # Main Workspace: Left Chart (70%) | Right Signal Overview (30%)
        col_chart, col_signals = st.columns([7, 3])

        with col_chart:
            render_section_header("Interactive Price Terminal", meta="OHLC Candlesticks + Technical Overlays")
            
            # Indicator toggle row
            it1, it2, it3, it4 = st.columns(4)
            with it1:
                show_smas = st.checkbox("SMA 5 & 10", value=True)
            with it2:
                show_emas = st.checkbox("EMA 12 & 26 (Vis.)", value=False)
            with it3:
                show_bb = st.checkbox("Bollinger Bands (Vis.)", value=False)
            with it4:
                show_markers = st.checkbox("BUY/SELL Signals", value=True)

            # Build Candlestick Figure with Plotly
            fig_main = go.Figure()
            fig_main.add_trace(go.Candlestick(
                x=df_feat_comp["Date"],
                open=df_feat_comp["Open"],
                high=df_feat_comp["High"],
                low=df_feat_comp["Low"],
                close=df_feat_comp["Close"],
                name="OHLC Price",
                increasing_line_color=COLOR_EMERALD,
                decreasing_line_color=COLOR_RED,
                increasing_fillcolor="rgba(16, 185, 129, 0.20)",
                decreasing_fillcolor="rgba(239, 68, 68, 0.20)"
            ))

            if show_smas:
                fig_main.add_trace(go.Scatter(x=df_feat_comp["Date"], y=df_feat_comp["SMA_5"], mode="lines", name="SMA 5", line=dict(color=COLOR_AMBER, width=1.1)))
                fig_main.add_trace(go.Scatter(x=df_feat_comp["Date"], y=df_feat_comp["SMA_10"], mode="lines", name="SMA 10", line=dict(color=COLOR_CYAN, width=1.1)))

            if show_emas:
                fig_main.add_trace(go.Scatter(x=df_feat_comp["Date"], y=df_feat_comp["EMA_12"], mode="lines", name="EMA 12", line=dict(color="#38BDF8", width=1.0, dash="dot")))
                fig_main.add_trace(go.Scatter(x=df_feat_comp["Date"], y=df_feat_comp["EMA_26"], mode="lines", name="EMA 26", line=dict(color="#A1A1AA", width=1.0, dash="dot")))

            if show_bb:
                fig_main.add_trace(go.Scatter(x=df_feat_comp["Date"], y=df_feat_comp["BB_Upper"], mode="lines", name="BB Upper", line=dict(color="rgba(161, 161, 170, 0.4)", width=1)))
                fig_main.add_trace(go.Scatter(x=df_feat_comp["Date"], y=df_feat_comp["BB_Lower"], mode="lines", name="BB Lower", line=dict(color="rgba(161, 161, 170, 0.4)", width=1), fill="tonexty", fillcolor="rgba(161, 161, 170, 0.05)"))

            if show_markers and df_fusion_comp is not None:
                df_f_merged = df_fusion_comp.merge(df_feat_comp[["Date", "Low", "High"]], on="Date", how="inner")
                buys = df_f_merged[df_f_merged["Final_Decision"] == "BUY"]
                sells = df_f_merged[df_f_merged["Final_Decision"] == "SELL"]

                if len(buys) > 0:
                    fig_main.add_trace(go.Scatter(
                        x=buys["Date"],
                        y=buys["Low"] * 0.985,
                        mode="markers",
                        name="BUY Decision",
                        marker=dict(symbol="triangle-up", size=8, color=COLOR_EMERALD, line=dict(width=1, color="#FFFFFF")),
                        hoverinfo="text",
                        hovertext=[f"BUY Date: {d}<br>Score: {s:+.3f}<br>Conf: {c:.1f}%" for d, s, c in zip(buys["Date"], buys["Fusion_Score"], buys["Heuristic_Confidence_Percent"])]
                    ))
                if len(sells) > 0:
                    fig_main.add_trace(go.Scatter(
                        x=sells["Date"],
                        y=sells["High"] * 1.015,
                        mode="markers",
                        name="SELL Decision",
                        marker=dict(symbol="triangle-down", size=8, color=COLOR_RED, line=dict(width=1, color="#FFFFFF")),
                        hoverinfo="text",
                        hovertext=[f"SELL Date: {d}<br>Score: {s:+.3f}<br>Conf: {c:.1f}%" for d, s, c in zip(sells["Date"], sells["Fusion_Score"], sells["Heuristic_Confidence_Percent"])]
                    ))

            fig_main.update_xaxes(
                rangeslider_visible=False,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(step="all", label="ALL")
                    ]),
                    bgcolor=COLOR_SURFACE_SEC,
                    font=dict(color=COLOR_TEXT_SEC, size=9),
                    activecolor=COLOR_BORDER
                )
            )
            apply_terminal_layout(fig_main, height=330)
            st.plotly_chart(fig_main, use_container_width=True)

        with col_signals:
            render_section_header("Signal Overview", meta="4-Channel Consensus")
            
            p_up = latest_fusion["Classification_P_UP"]
            c_sig = latest_fusion["Classification_Signal"]
            c_contrib = latest_fusion["Classification_Contribution"]
            c_cls = "pos" if c_sig > 0 else ("neg" if c_sig < 0 else "warn")
            
            rl_act = int(latest_fusion["RL_Action"])
            rl_lbl = "BUY" if rl_act == 2 else ("SELL" if rl_act == 0 else "HOLD")
            rl_sig = latest_fusion["RL_Signal"]
            rl_contrib = latest_fusion["RL_Contribution"]
            rl_cls = "pos" if rl_sig > 0 else ("neg" if rl_sig < 0 else "warn")
            
            reg_lbl = latest_fusion["Market_Regime_Label"]
            reg_sig = latest_fusion["Market_Regime_Signal"]
            reg_contrib = latest_fusion["Market_Regime_Contribution"]
            reg_cls = "pos" if reg_sig > 0 else ("neg" if reg_sig < 0 else "warn")
            
            nlp_sig = latest_fusion["NLP_Signal"]
            nlp_contrib = latest_fusion["NLP_Contribution"]
            nlp_cls = "pos" if nlp_sig > 0 else ("neg" if nlp_sig < 0 else "warn")
            
            dec_border = COLOR_EMERALD if dec_val == "BUY" else (COLOR_RED if dec_val == "SELL" else COLOR_AMBER)

            render_html(f"""
            <div class="panel-box">
                <div class="signal-row">
                    <span>1. CLASSIFICATION (35%)</span>
                    <span class="{c_cls}">{c_sig:+.3f}</span>
                </div>
                <div style="font-size:10px; color:{COLOR_TEXT_MUTED}; font-family:monospace;">
                    P(UP): {p_up:.1%} • CONTRIB: {c_contrib:+.4f}
                </div>
            </div>
            <div class="panel-box">
                <div class="signal-row">
                    <span>2. RL STRATEGY (30%)</span>
                    <span class="{rl_cls}">{rl_lbl} ({rl_sig:+.0f})</span>
                </div>
                <div style="font-size:10px; color:{COLOR_TEXT_MUTED}; font-family:monospace;">
                    Q-POLICY • CONTRIB: {rl_contrib:+.4f}
                </div>
            </div>
            <div class="panel-box">
                <div class="signal-row">
                    <span>3. MARKET REGIME (15%)</span>
                    <span class="{reg_cls}">{reg_lbl}</span>
                </div>
                <div style="font-size:10px; color:{COLOR_TEXT_MUTED}; font-family:monospace;">
                    CLUSTER STATE • CONTRIB: {reg_contrib:+.4f}
                </div>
            </div>
            <div class="panel-box">
                <div class="signal-row">
                    <span>4. NEWS NLP (20%)</span>
                    <span class="{nlp_cls}">{nlp_sig:+.3f}</span>
                </div>
                <div style="font-size:10px; color:{COLOR_TEXT_MUTED}; font-family:monospace;">
                    LM-TRIAGE • CONTRIB: {nlp_contrib:+.4f}
                </div>
            </div>
            <div class="panel-box" style="border-left: 3px solid {dec_border}; margin-top:0.4rem;">
                <div style="font-size:10px; font-weight:700; color:{COLOR_TEXT_MUTED}; text-transform:uppercase;">DECISION RATIONALE</div>
                <div style="font-size:11px; color:{COLOR_TEXT_PRI}; margin-top:0.2rem; line-height:1.35;">
                    {latest_fusion['Decision_Explanation']}
                </div>
            </div>
            """)

            # Interactive Grounded AI Trading Decision Explanation
            try:
                explainer = get_decision_explainer()
                latest_feat = df_feat_comp.iloc[-1] if not df_feat_comp.empty else None
                p1_explanation = explainer.explain_decision(
                    fusion_row=latest_fusion,
                    market_features=latest_feat
                )
                render_decision_explanation_ui(p1_explanation, expanded=False, key_prefix="p1_overview")
            except Exception as _e:
                pass

        render_section_header("Historical Market Snapshot", meta="Descriptive Analytics")
        snap1, snap2, snap3, snap4 = st.columns(4)

        with snap1:
            fig_v = go.Figure()
            vol_colors = [COLOR_EMERALD if r >= 0 else COLOR_RED for r in df_feat_comp["Daily_Return"]]
            fig_v.add_trace(go.Bar(x=df_feat_comp["Date"], y=df_feat_comp["Volume"], marker_color=vol_colors, name="Volume"))
            apply_terminal_layout(fig_v, height=140, title="Volume", show_legend=False)
            st.plotly_chart(fig_v, use_container_width=True)

        with snap2:
            fig_r = go.Figure()
            fig_r.add_trace(go.Bar(x=df_feat_comp["Date"], y=df_feat_comp["Daily_Return"], marker_color=vol_colors, name="Daily Return"))
            apply_terminal_layout(fig_r, height=140, title="Daily Return", show_legend=False)
            st.plotly_chart(fig_r, use_container_width=True)

        with snap3:
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Scatter(x=df_feat_comp["Date"], y=df_feat_comp["Volatility_5"], line=dict(color=COLOR_AMBER, width=1.1), fill="tozeroy", fillcolor="rgba(245, 158, 11, 0.08)", name="Vol 5D"))
            apply_terminal_layout(fig_vol, height=140, title="5D Volatility", show_legend=False)
            st.plotly_chart(fig_vol, use_container_width=True)

        with snap4:
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(x=df_feat_comp["Date"], y=df_feat_comp["Drawdown"], line=dict(color=COLOR_RED, width=1.1), fill="tozeroy", fillcolor="rgba(239, 68, 68, 0.08)", name="Drawdown"))
            apply_terminal_layout(fig_dd, height=140, title="Peak Drawdown", show_legend=False)
            st.plotly_chart(fig_dd, use_container_width=True)


# ==============================================================================
# 6. PAGE 2 — MARKET ANALYSIS
# ==============================================================================
elif nav_selection == "2. Market Analysis":
    render_top_status_bar("Market Analysis", selected_ticker, latest_date, mode="HISTORICAL")

    if df_feat_comp is not None:
        # SECTION 1: PRICE & TREND
        render_section_header("Price & Trend", meta="Candlesticks, Moving Averages & Bollinger Bands")
        fig_pt = go.Figure()
        fig_pt.add_trace(go.Candlestick(
            x=df_feat_comp["Date"], open=df_feat_comp["Open"], high=df_feat_comp["High"],
            low=df_feat_comp["Low"], close=df_feat_comp["Close"], name="OHLC",
            increasing_line_color=COLOR_EMERALD, decreasing_line_color=COLOR_RED
        ))
        fig_pt.add_trace(go.Scatter(x=df_feat_comp["Date"], y=df_feat_comp["SMA_5"], name="SMA 5", line=dict(color=COLOR_AMBER, width=1.1)))
        fig_pt.add_trace(go.Scatter(x=df_feat_comp["Date"], y=df_feat_comp["SMA_10"], name="SMA 10", line=dict(color=COLOR_CYAN, width=1.1)))
        fig_pt.add_trace(go.Scatter(x=df_feat_comp["Date"], y=df_feat_comp["EMA_12"], name="EMA 12", line=dict(color="#38BDF8", width=1.0, dash="dot")))
        fig_pt.add_trace(go.Scatter(x=df_feat_comp["Date"], y=df_feat_comp["EMA_26"], name="EMA 26", line=dict(color="#A1A1AA", width=1.0, dash="dot")))
        fig_pt.add_trace(go.Scatter(x=df_feat_comp["Date"], y=df_feat_comp["BB_Upper"], name="BB Upper", line=dict(color="rgba(161, 161, 170, 0.4)", width=1)))
        fig_pt.add_trace(go.Scatter(x=df_feat_comp["Date"], y=df_feat_comp["BB_Lower"], name="BB Lower", line=dict(color="rgba(161, 161, 170, 0.4)", width=1), fill="tonexty", fillcolor="rgba(161, 161, 170, 0.05)"))
        apply_terminal_layout(fig_pt, height=310)
        st.plotly_chart(fig_pt, use_container_width=True)

        # SECTION 2: MOMENTUM
        render_section_header("Momentum Indicators", meta="Visualization Only • Not Ingested into Models")
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=df_feat_comp["Date"], y=df_feat_comp["RSI_14"], line=dict(color=COLOR_CYAN, width=1.2), name="RSI 14"))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color=COLOR_RED, annotation_text="Overbought (70)")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color=COLOR_EMERALD, annotation_text="Oversold (30)")
            fig_rsi.update_yaxes(range=[0, 100])
            apply_terminal_layout(fig_rsi, height=180, title="RSI (14-Day)")
            st.plotly_chart(fig_rsi, use_container_width=True)

        with m_col2:
            fig_macd = go.Figure()
            hist_colors = [COLOR_EMERALD if h >= 0 else COLOR_RED for h in df_feat_comp["MACD_Hist"]]
            fig_macd.add_trace(go.Bar(x=df_feat_comp["Date"], y=df_feat_comp["MACD_Hist"], marker_color=hist_colors, name="MACD Hist"))
            fig_macd.add_trace(go.Scatter(x=df_feat_comp["Date"], y=df_feat_comp["MACD"], line=dict(color=COLOR_AMBER, width=1.1), name="MACD (12,26)"))
            fig_macd.add_trace(go.Scatter(x=df_feat_comp["Date"], y=df_feat_comp["MACD_Signal"], line=dict(color="#A1A1AA", width=1.1), name="Signal (9)"))
            apply_terminal_layout(fig_macd, height=180, title="MACD (12, 26, 9)")
            st.plotly_chart(fig_macd, use_container_width=True)

        # SECTION 3: RISK & RETURN
        render_section_header("Risk & Return Distribution", meta="Statistical Moments & Drawdown")
        r_col1, r_col2 = st.columns(2)
        with r_col1:
            fig_dist = px.histogram(
                df_feat_comp, x="Daily_Return", nbins=60, title="Daily Return Distribution",
                color_discrete_sequence=[COLOR_CYAN]
            )
            apply_terminal_layout(fig_dist, height=200)
            st.plotly_chart(fig_dist, use_container_width=True)
        with r_col2:
            fig_dd2 = go.Figure()
            fig_dd2.add_trace(go.Scatter(x=df_feat_comp["Date"], y=df_feat_comp["Drawdown"], line=dict(color=COLOR_RED, width=1.1), fill="tozeroy", fillcolor="rgba(239, 68, 68, 0.10)", name="Drawdown"))
            apply_terminal_layout(fig_dd2, height=200, title="Peak-to-Trough Drawdown Curve")
            st.plotly_chart(fig_dd2, use_container_width=True)

        # SECTION 4: MARKET REGIME TIMELINE
        render_section_header("Market Regime Timeline", meta="Unsupervised Behavioral State")
        df_clust_results = DATA["clust_results"]
        if df_clust_results is not None:
            df_cr_comp = df_clust_results[df_clust_results["Ticker"] == selected_ticker].sort_values("Date").reset_index(drop=True)
            regime_map = {0: "Bearish Drawdown", 1: "Calm Equilibrium", 2: "Bullish Expansion"}
            df_cr_comp["Regime_Name"] = df_cr_comp["Cluster"].map(regime_map)
            
            fig_reg = px.scatter(
                df_cr_comp, x="Date", y="Cluster", color="Regime_Name",
                color_discrete_map={"Bearish Drawdown": COLOR_RED, "Calm Equilibrium": COLOR_CYAN, "Bullish Expansion": COLOR_EMERALD},
                labels={"Cluster": "Cluster ID"}
            )
            fig_reg.update_traces(marker=dict(size=4))
            apply_terminal_layout(fig_reg, height=170, title=f"{selected_ticker} Regime States Across Time")
            st.plotly_chart(fig_reg, use_container_width=True)

        # SECTION 5: PCA
        render_section_header("Principal Component Analysis (PCA)", meta="Dimensionality Reduction (13 Features → 6 Orthogonal Axes)")
        render_alert_box("Dimensionality Reduction: 13 correlated technical features projected into orthogonal axes. Across the universe, 6 components capture ≥ 95% of cumulative variance, eliminating multicollinearity before model ingestion.", level="info")
        
        df_pca = DATA["pca"]
        if df_pca is not None:
            df_pca_comp = df_pca[df_pca["Ticker"] == selected_ticker].reset_index(drop=True)
            p1, p2 = st.columns([6, 4])
            with p1:
                fig_pca_scatter = px.scatter(
                    df_pca_comp, x="PC1", y="PC2", color=df_pca_comp.index,
                    color_continuous_scale="Viridis",
                    labels={"color": "Session Index"}
                )
                fig_pca_scatter.update_traces(marker=dict(size=4, opacity=0.75))
                apply_terminal_layout(fig_pca_scatter, height=240, title=f"{selected_ticker} 2D Projection: PC1 vs PC2")
                st.plotly_chart(fig_pca_scatter, use_container_width=True)
            with p2:
                pca_var_df = pd.DataFrame({
                    "Component": ["PC1", "PC2", "PC3", "PC4", "PC5", "PC6"],
                    "Variance (%)": [44.8, 22.3, 14.1, 7.2, 4.3, 2.9]
                })
                fig_pca_var = px.bar(
                    pca_var_df, x="Component", y="Variance (%)",
                    color_discrete_sequence=[COLOR_TEXT_SEC]
                )
                apply_terminal_layout(fig_pca_var, height=240, title="Explained Variance by Component")
                st.plotly_chart(fig_pca_var, use_container_width=True)

        # SECTION 6: CORRELATION HEATMAP
        render_section_header("Feature Correlation Matrix", meta="Pearson Coefficients Across Technical Indicators")
        corr_cols = ["Close", "Volume", "Daily_Return", "Price_Range", "Candle_Body", "Upper_Shadow", "Lower_Shadow", "Volatility_5"]
        corr_df = df_feat_comp[corr_cols].corr()
        fig_heat = px.imshow(
            corr_df, text_auto=".2f", aspect="auto",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1
        )
        apply_terminal_layout(fig_heat, height=260, title=f"{selected_ticker} Technical Feature Correlations")
        st.plotly_chart(fig_heat, use_container_width=True)


# ==============================================================================
# 7. PAGE 3 — PREDICTIVE MODELS
# ==============================================================================
elif nav_selection == "3. Predictive Models":
    render_top_status_bar("Predictive Models", selected_ticker, latest_date, mode="QUANTITATIVE")

    tab_reg, tab_regul, tab_class, tab_clust, tab_rl = st.tabs([
        "REGRESSION",
        "REGULARIZATION",
        "CLASSIFICATION",
        "CLUSTERING",
        "REINFORCEMENT LEARNING"
    ])

    # 1. REGRESSION
    with tab_reg:
        render_section_header("Supervised Price Regression", meta="Target: Close(t+1)")
        render_alert_box("Supervised Linear Regression forecasting Close(t+1). High R² reflects price level tracking rather than guaranteed trade execution profitability.", level="info")
        
        df_reg_sum = DATA["reg_summary"]
        if df_reg_sum is not None:
            comp_reg = df_reg_sum[df_reg_sum["Ticker"] == selected_ticker].iloc[0]
            render_kpi_strip([
                {"label": "Test MAE", "val": f"${comp_reg['Test_MAE']:.2f}"},
                {"label": "Test MSE", "val": f"{comp_reg['Test_MSE']:.2f}"},
                {"label": "Test RMSE", "val": f"${comp_reg['Test_RMSE']:.2f}"},
                {"label": "Test R²", "val": f"{comp_reg['Test_R2']:.4f}", "cls": "pos"}
            ])

        df_reg_preds = DATA["reg_preds"]
        if df_reg_preds is not None:
            comp_preds = df_reg_preds[df_reg_preds["Ticker"] == selected_ticker].sort_values("Date").reset_index(drop=True)
            
            fig_rp = go.Figure()
            fig_rp.add_trace(go.Scatter(x=comp_preds["Date"], y=comp_preds["Actual_Next_Close"], name="Actual Next Close", line=dict(color=COLOR_CYAN, width=1.3)))
            fig_rp.add_trace(go.Scatter(x=comp_preds["Date"], y=comp_preds["Predicted_Next_Close"], name="Predicted Next Close", line=dict(color=COLOR_AMBER, width=1.3, dash="dash")))
            apply_terminal_layout(fig_rp, height=240, title=f"{selected_ticker} Actual vs Predicted Price (300 Sessions)")
            st.plotly_chart(fig_rp, use_container_width=True)

            fig_res = go.Figure()
            fig_res.add_trace(go.Bar(x=comp_preds["Date"], y=comp_preds["Residual"], name="Residual", marker_color=COLOR_TEXT_MUTED))
            apply_terminal_layout(fig_res, height=140, title="Prediction Residual Error ($)")
            st.plotly_chart(fig_res, use_container_width=True)

    # 2. REGULARIZATION
    with tab_regul:
        render_section_header("Regularization & Feature Selection", meta="L2 Ridge vs L1 Lasso")
        render_alert_box("Regularization penalizes large coefficients to mitigate multicollinearity. Ridge shrinks weights smoothly, while Lasso drives redundant indicator coefficients to zero for automated feature selection.", level="info")

        df_regul_sum = DATA["regul_summary"]
        if df_regul_sum is not None:
            comp_reg_s = df_regul_sum[df_regul_sum["Ticker"] == selected_ticker].iloc[0]
            
            r_col1, r_col2 = st.columns([5, 5])
            with r_col1:
                comp_chart_df = pd.DataFrame({
                    "Model": ["Standard OLS", "Ridge (L2)", "Lasso (L1)"],
                    "Test RMSE ($)": [comp_reg_s["OLS_Test_RMSE"], comp_reg_s["Ridge_Test_RMSE"], comp_reg_s["Lasso_Test_RMSE"]]
                })
                fig_reg_comp = px.bar(
                    comp_chart_df, x="Model", y="Test RMSE ($)", color="Model",
                    color_discrete_sequence=[COLOR_CYAN, COLOR_EMERALD, COLOR_AMBER]
                )
                apply_terminal_layout(fig_reg_comp, height=220, title="Test RMSE Across Estimators")
                st.plotly_chart(fig_reg_comp, use_container_width=True)

            with r_col2:
                render_kpi_strip([
                    {"label": "Best Ridge Alpha", "val": f"{comp_reg_s['Ridge_Best_Alpha']}", "sub": f"RMSE: ${comp_reg_s['Ridge_Test_RMSE']:.2f}"},
                    {"label": "Best Lasso Alpha", "val": f"{comp_reg_s['Lasso_Best_Alpha']}", "sub": f"RMSE: ${comp_reg_s['Lasso_Test_RMSE']:.2f}"},
                    {"label": "Lasso Retained", "val": f"{int(comp_reg_s['Lasso_Active_Features'])}", "sub": "Active features", "cls": "pos"},
                    {"label": "Lasso Zeroed", "val": f"{int(comp_reg_s['Lasso_Zeroed_Features'])}", "sub": "Eliminated", "cls": "neg"}
                ])
                
                sparsity_df = pd.DataFrame({
                    "Status": ["Retained Features", "Zeroed Features"],
                    "Count": [int(comp_reg_s['Lasso_Active_Features']), int(comp_reg_s['Lasso_Zeroed_Features'])]
                })
                fig_pie = px.pie(sparsity_df, values="Count", names="Status", color_discrete_sequence=[COLOR_EMERALD, COLOR_RED], hole=0.55)
                apply_terminal_layout(fig_pie, height=160, title="Lasso Feature Sparsity Breakdown")
                st.plotly_chart(fig_pie, use_container_width=True)

    # 3. CLASSIFICATION
    with tab_class:
        render_section_header("Supervised Directional Classification", meta="Target: Close(t+1) > Close(t)")
        render_alert_box("Directional Definition: UP indicates next close exceeded today's close. Calibrated probability P(UP) from Logistic Regression is mapped to the directional signal: 2 * P(UP) - 1.", level="info")

        df_class_sum = DATA["class_summary"]
        if df_class_sum is not None:
            comp_cs = df_class_sum[df_class_sum["Ticker"] == selected_ticker]
            
            c_col1, c_col2 = st.columns([6, 4])
            with c_col1:
                fig_cls = px.bar(
                    comp_cs, x="Model", y=["Accuracy", "Precision_UP", "Recall_UP", "F1_UP"],
                    barmode="group",
                    color_discrete_sequence=[COLOR_CYAN, COLOR_EMERALD, COLOR_AMBER, COLOR_TEXT_SEC]
                )
                apply_terminal_layout(fig_cls, height=240, title=f"{selected_ticker} Classification Performance")
                st.plotly_chart(fig_cls, use_container_width=True)

            with c_col2:
                df_cm = DATA["class_cm"]
                if df_cm is not None:
                    lr_cm = df_cm[(df_cm["Ticker"] == selected_ticker) & (df_cm["Model"] == "Logistic Regression")].iloc[0]
                    cm_mat = np.array([[lr_cm["TN"], lr_cm["FP"]], [lr_cm["FN"], lr_cm["TP"]]])
                    fig_cm = px.imshow(
                        cm_mat, text_auto=True, x=["DOWN", "UP"], y=["DOWN", "UP"],
                        color_continuous_scale="Blues"
                    )
                    apply_terminal_layout(fig_cm, height=240, title="Logistic Regression Confusion Matrix")
                    st.plotly_chart(fig_cm, use_container_width=True)

    # 4. CLUSTERING
    with tab_clust:
        render_section_header("Unsupervised Market Regime Clustering", meta="K-Means (K=3) on Normalized Technical Features")

        df_c_by_comp = DATA["clust_company"]
        if df_c_by_comp is not None:
            comp_c_dist = df_c_by_comp[df_c_by_comp["Ticker"] == selected_ticker]
            c_col1, c_col2 = st.columns(2)
            with c_col1:
                fig_cl_bar = px.bar(
                    comp_c_dist, x="Cluster", y="Percentage", color="Cluster",
                    color_discrete_sequence=[COLOR_RED, COLOR_CYAN, COLOR_EMERALD],
                    labels={"Percentage": "Percentage (%)"}
                )
                apply_terminal_layout(fig_cl_bar, height=220, title=f"{selected_ticker} Historical Regime Allocation")
                st.plotly_chart(fig_cl_bar, use_container_width=True)

            with c_col2:
                render_html(f"""
                <div class="panel-box" style="margin-top:0.3rem;">
                    <div style="font-size:10px; font-weight:700; color:{COLOR_TEXT_MUTED}; text-transform:uppercase; margin-bottom:0.4rem;">REGIME TAXONOMY SPECIFICATION</div>
                    <div style="font-size:11px; margin-bottom:0.35rem;"><b class="neg">CLUSTER 0 (BEARISH DRAWDOWN):</b> Negative mean daily return, elevated intraday spread, high volatility.</div>
                    <div style="font-size:11px; margin-bottom:0.35rem;"><b class="info">CLUSTER 1 (CALM EQUILIBRIUM):</b> Near-zero return, lowest volatility, balanced shadows.</div>
                    <div style="font-size:11px;"><b class="pos">CLUSTER 2 (BULLISH EXPANSION):</b> Strongly positive daily return, sustained expansion volume.</div>
                </div>
                """)

    # 5. REINFORCEMENT LEARNING
    with tab_rl:
        render_section_header("Tabular Q-Learning Trading Agent", meta="300-Session Out-of-Sample Evaluation")
        render_alert_box("Historical Backtest: Tabular Q-Learning policy evaluated on discrete state tuples (M, D, V) over 300 test sessions. Past performance is no guarantee of future results.", level="info")

        df_rl_sum = DATA["rl_summary"]
        if df_rl_sum is not None:
            comp_rl = df_rl_sum[df_rl_sum["Ticker"] == selected_ticker].iloc[0]
            render_kpi_strip([
                {"label": "RL Agent Return", "val": f"{comp_rl['Total_Return']:+.2%}", "sub": f"B&H: {comp_rl['Buy_Hold_Return']:+.2%}", "cls": "pos" if comp_rl['Total_Return'] >= 0 else "neg"},
                {"label": "Max Drawdown", "val": f"{comp_rl['Maximum_Drawdown']:.2%}", "sub": f"B&H: {comp_rl['Buy_Hold_Max_Drawdown']:.2%}", "cls": "neg"},
                {"label": "Sharpe Ratio", "val": f"{comp_rl['Sharpe_Ratio']:.2f}", "sub": f"B&H: {comp_rl['Buy_Hold_Sharpe_Ratio']:.2f}", "cls": "info"},
                {"label": "Market Exposure", "val": f"{comp_rl['Exposure']:.1%}", "sub": "Time in Market", "cls": "muted"}
            ])

        df_rl_res = DATA["rl_results"]
        if df_rl_res is not None:
            comp_rl_res = df_rl_res[df_rl_res["Ticker"] == selected_ticker]
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(x=comp_rl_res["Date"], y=comp_rl_res["Portfolio_Value"], name="RL Strategy Equity ($)", line=dict(color=COLOR_EMERALD, width=1.3)))
            apply_terminal_layout(fig_eq, height=220, title=f"{selected_ticker} Q-Learning Cumulative Equity ($10k Initial)")
            st.plotly_chart(fig_eq, use_container_width=True)


# ==============================================================================
# 8. PAGE 4 — NEWS & NLP
# ==============================================================================
elif nav_selection == "4. News & NLP":
    render_top_status_bar("News & NLP", selected_ticker, latest_date, mode="DEMO")

    render_alert_box(
        "Demonstration Corpus: Illustrates the 8-stage Loughran-McDonald NLP ingestion and triage pipeline using structured demonstration news records to validate entity extraction, relevance weighting, and temporal decay scoring. Real-world financial news ingestion is in the Real-Time Data module.",
        level="info",
        title="NLP Pipeline Data Disclosure"
    )

    render_section_header("8-Stage News Ingestion & Triage Pipeline", meta="Loughran-McDonald Domain Lexicon")
    render_html(f"""
    <div style="display:grid; grid-template-columns: repeat(8, 1fr); gap: 4px; margin-bottom: 0.6rem;">
        <div class="panel-box" style="text-align:center; padding:0.4rem 0.2rem;"><div style="font-size:10px; font-weight:700;">1. HEADLINE</div><div style="font-size:9px; color:{COLOR_TEXT_MUTED};">Raw Ingestion</div></div>
        <div class="panel-box" style="text-align:center; padding:0.4rem 0.2rem;"><div style="font-size:10px; font-weight:700;">2. CLEANING</div><div style="font-size:9px; color:{COLOR_TEXT_MUTED};">Tokenization</div></div>
        <div class="panel-box" style="text-align:center; padding:0.4rem 0.2rem;"><div style="font-size:10px; font-weight:700;">3. SENTIMENT</div><div style="font-size:9px; color:{COLOR_TEXT_MUTED};">LM + Negation</div></div>
        <div class="panel-box" style="text-align:center; padding:0.4rem 0.2rem;"><div style="font-size:10px; font-weight:700;">4. RELEVANCE</div><div style="font-size:9px; color:{COLOR_TEXT_MUTED};">Entity [0, 1]</div></div>
        <div class="panel-box" style="text-align:center; padding:0.4rem 0.2rem;"><div style="font-size:10px; font-weight:700;">5. GRAVITY</div><div style="font-size:9px; color:{COLOR_TEXT_MUTED};">Event [0, 1]</div></div>
        <div class="panel-box" style="text-align:center; padding:0.4rem 0.2rem;"><div style="font-size:10px; font-weight:700;">6. RECENCY</div><div style="font-size:9px; color:{COLOR_TEXT_MUTED};">Decay [0, 1]</div></div>
        <div class="panel-box" style="text-align:center; padding:0.4rem 0.2rem;"><div style="font-size:10px; font-weight:700;">7. TRIAGE</div><div style="font-size:9px; color:{COLOR_TEXT_MUTED};">Rel × Rec × Imp</div></div>
        <div class="panel-box" style="text-align:center; padding:0.4rem 0.2rem;"><div style="font-size:10px; font-weight:700;">8. SIGNAL</div><div style="font-size:9px; color:{COLOR_TEXT_MUTED};">Sent × Triage</div></div>
    </div>
    """)

    df_n_sum = DATA["news_summary"]
    if df_n_sum is not None:
        comp_ns = df_n_sum[df_n_sum["Ticker"] == selected_ticker].iloc[0]
        render_kpi_strip([
            {"label": "Total Headlines", "val": f"{int(comp_ns['Total_News_Items'])}"},
            {"label": "Positive News", "val": f"{int(comp_ns['Positive_News_Count'])}", "cls": "pos"},
            {"label": "Neutral News", "val": f"{int(comp_ns['Neutral_News_Count'])}", "cls": "warn"},
            {"label": "Negative News", "val": f"{int(comp_ns['Negative_News_Count'])}", "cls": "neg"},
            {"label": "Avg Sentiment", "val": f"{comp_ns['Average_Sentiment']:+.3f}", "cls": "pos" if comp_ns['Average_Sentiment'] >= 0 else "neg"}
        ])

    render_section_header("News Signal & Sentiment Dynamics", meta="Time-Series & Sentiment Split")
    df_n_daily = DATA["news_daily"]
    if df_n_daily is not None:
        comp_nd = df_n_daily[df_n_daily["Ticker"] == selected_ticker].sort_values("Date").reset_index(drop=True)
        
        nc1, nc2 = st.columns([7, 3])
        with nc1:
            fig_ns = go.Figure()
            sig_colors = [COLOR_EMERALD if s >= 0 else COLOR_RED for s in comp_nd["News_Signal"]]
            fig_ns.add_trace(go.Bar(x=comp_nd["Date"], y=comp_nd["News_Signal"], marker_color=sig_colors, name="News Signal"))
            fig_ns.add_hline(y=0, line_color=COLOR_BORDER)
            apply_terminal_layout(fig_ns, height=220, title="Daily Aggregated News Signal")
            st.plotly_chart(fig_ns, use_container_width=True)

        with nc2:
            fig_np = px.pie(
                values=[comp_ns["Positive_News_Count"], comp_ns["Neutral_News_Count"], comp_ns["Negative_News_Count"]],
                names=["Positive", "Neutral", "Negative"],
                color_discrete_sequence=[COLOR_EMERALD, COLOR_AMBER, COLOR_RED],
                hole=0.55
            )
            apply_terminal_layout(fig_np, height=220, title="Sentiment Split")
            st.plotly_chart(fig_np, use_container_width=True)


# ==============================================================================
# 9. PAGE 5 — SIGNAL FUSION
# ==============================================================================
elif nav_selection == "5. Signal Fusion":
    render_top_status_bar("Signal Fusion", selected_ticker, latest_date, mode="OUT-OF-SAMPLE")

    if df_fusion_comp is not None and len(df_fusion_comp) > 0:
        d_mode = st.radio("Target Session", ["Latest Available Date", "Pick from History (300 Sessions)"], horizontal=True)
        if d_mode == "Latest Available Date":
            target_date = df_fusion_comp["Date"].iloc[-1]
        else:
            target_date = st.selectbox("Select Session Date", options=list(reversed(df_fusion_comp["Date"].tolist())))
            
        row_f = df_fusion_comp[df_fusion_comp["Date"] == target_date].iloc[0]

        dec = row_f["Final_Decision"]
        sc = row_f["Fusion_Score"]
        conf = row_f["Heuristic_Confidence_Percent"]
        dec_color = COLOR_EMERALD if dec == "BUY" else (COLOR_RED if dec == "SELL" else COLOR_AMBER)

        render_html(f"""
        <div class="panel-box" style="border-left: 3px solid {dec_color}; margin-bottom: 0.65rem;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <div style="font-size:10px; font-weight:700; color:{COLOR_TEXT_MUTED}; text-transform:uppercase;">CURRENT SYSTEM DECISION</div>
                    <div style="font-size:16px; font-weight:800; color:{dec_color}; letter-spacing:0.04em;">{dec}</div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:12px; font-family:monospace; color:{COLOR_TEXT_PRI};">SCORE: <b>{sc:+.4f}</b></div>
                    <div style="font-size:10px; color:{COLOR_TEXT_MUTED};">CONFIDENCE: {conf:.1f}% (HEURISTIC)</div>
                </div>
            </div>
            <div style="margin-top:0.35rem; font-size:11px; color:{COLOR_TEXT_SEC}; border-top:1px solid {COLOR_BORDER}; padding-top:0.35rem; line-height:1.35;">
                <b style="color:{COLOR_TEXT_PRI};">DECISION RATIONALE:</b> {row_f['Decision_Explanation']}
            </div>
        </div>
        """)

        # Interactive Grounded AI Trading Decision Explanation
        try:
            explainer = get_decision_explainer()
            target_feat = None
            if df_feat_comp is not None and not df_feat_comp.empty:
                f_match = df_feat_comp[df_feat_comp["Date"] == target_date]
                if not f_match.empty:
                    target_feat = f_match.iloc[-1]
            p5_explanation = explainer.explain_decision(
                fusion_row=row_f,
                market_features=target_feat
            )
            render_decision_explanation_ui(p5_explanation, expanded=False, key_prefix="p5_fusion")
        except Exception as _e:
            pass

        render_section_header("Component Contributions & Fixed Consensus Logic", meta="Weights: 35/30/15/20 • Threshold: ±0.20")
        
        c_left, c_right = st.columns([6, 4])
        with c_left:
            fig_cont = go.Figure()
            components = ["Classification (35%)", "RL Strategy (30%)", "Market Regime (15%)", "News NLP (20%)"]
            contribs = [row_f["Classification_Contribution"], row_f["RL_Contribution"], row_f["Market_Regime_Contribution"], row_f["NLP_Contribution"]]
            c_colors = [COLOR_EMERALD if c >= 0 else COLOR_RED for c in contribs]
            
            fig_cont.add_trace(go.Bar(
                y=components, x=contribs, orientation="h", marker_color=c_colors,
                text=[f"{c:+.4f}" for c in contribs], textposition="auto"
            ))
            fig_cont.add_vline(x=0, line_color=COLOR_BORDER)
            fig_cont.update_xaxes(range=[-0.4, 0.4], title="Score Contribution")
            apply_terminal_layout(fig_cont, height=190, title=f"Component Contributions ({target_date})")
            st.plotly_chart(fig_cont, use_container_width=True)

        with c_right:
            render_html(f"""
            <div class="panel-box" style="height: 100%;">
                <div style="font-size:10px; font-weight:700; color:{COLOR_TEXT_MUTED}; text-transform:uppercase; margin-bottom:0.3rem;">FIXED CONSENSUS FORMULA</div>
                <div style="font-family:monospace; font-size:11px; color:{COLOR_TEXT_PRI}; margin-bottom:0.4rem;">
                    Score = 0.35*Class + 0.30*RL + 0.15*Regime + 0.20*News
                </div>
                <div style="font-size:10px; color:{COLOR_TEXT_MUTED}; font-weight:700; text-transform:uppercase;">DECISION BOUNDARIES</div>
                <div style="font-size:11px; margin-top:0.2rem;"><span class="pos"><b>BUY:</b> Score ≥ +0.20</span></div>
                <div style="font-size:11px;"><span class="warn"><b>HOLD:</b> -0.20 &lt; Score &lt; +0.20</span></div>
                <div style="font-size:11px;"><span class="neg"><b>SELL:</b> Score ≤ -0.20</span></div>
                <div style="font-size:10px; color:{COLOR_TEXT_MUTED}; margin-top:0.4rem; font-family:monospace;">
                    ZERO TEST-PERIOD OPTIMIZATION. ARCHITECTURAL WEIGHTS.
                </div>
            </div>
            """)

        render_section_header("Historical Decision & Score Timeline", meta="300 Out-of-Sample Sessions")
        fig_ftime = go.Figure()
        fig_ftime.add_hline(y=0.20, line_color=COLOR_EMERALD, line_dash="dash", annotation_text="BUY (+0.20)")
        fig_ftime.add_hline(y=-0.20, line_color=COLOR_RED, line_dash="dash", annotation_text="SELL (-0.20)")
        fig_ftime.add_hline(y=0.00, line_color=COLOR_BORDER)

        fig_ftime.add_trace(go.Scatter(x=df_fusion_comp["Date"], y=df_fusion_comp["Fusion_Score"], line=dict(color=COLOR_TEXT_SEC, width=1.1), name="Combined Score"))
        
        buys = df_fusion_comp[df_fusion_comp["Final_Decision"] == "BUY"]
        sells = df_fusion_comp[df_fusion_comp["Final_Decision"] == "SELL"]
        holds = df_fusion_comp[df_fusion_comp["Final_Decision"] == "HOLD"]

        fig_ftime.add_trace(go.Scatter(x=buys["Date"], y=buys["Fusion_Score"], mode="markers", name="BUY", marker=dict(color=COLOR_EMERALD, size=6)))
        fig_ftime.add_trace(go.Scatter(x=sells["Date"], y=sells["Fusion_Score"], mode="markers", name="SELL", marker=dict(color=COLOR_RED, size=6)))
        fig_ftime.add_trace(go.Scatter(x=holds["Date"], y=holds["Fusion_Score"], mode="markers", name="HOLD", marker=dict(color=COLOR_AMBER, size=4, opacity=0.4)))

        apply_terminal_layout(fig_ftime, height=260)
        st.plotly_chart(fig_ftime, use_container_width=True)


# ==============================================================================
# 10. PAGE 6 — COMPANY COMPARISON
# ==============================================================================
elif nav_selection == "6. Company Comparison":
    render_top_status_bar("Company Comparison", ticker=None, date_str="2019-2024", mode="10 EQUITIES")

    render_alert_box("Comparative behavioral analytics across all 10 target equities under uniform multi-modal decision rules. Equities are evaluated objectively without subjective ranking.", level="info")

    if df_fusion_summary is not None:
        render_section_header("Cross-Company Decision Distribution Matrix", meta="10 Target Assets")
        st.dataframe(
            df_fusion_summary[[
                "Ticker", "Company_Name", "BUY_Count", "BUY_Percentage",
                "HOLD_Count", "HOLD_Percentage", "SELL_Count", "SELL_Percentage",
                "Average_Fusion_Score", "Average_Heuristic_Confidence"
            ]].style.format({
                "BUY_Percentage": "{:.1f}%", "HOLD_Percentage": "{:.1f}%", "SELL_Percentage": "{:.1f}%",
                "Average_Fusion_Score": "{:+.4f}", "Average_Heuristic_Confidence": "{:.4f}"
            }),
            use_container_width=True, hide_index=True
        )

        render_section_header("Comparative Signal Allocations", meta="Percentage of Out-of-Sample Sessions")
        cp1, cp2 = st.columns(2)

        with cp1:
            fig_stack = go.Figure()
            fig_stack.add_trace(go.Bar(x=df_fusion_summary["Ticker"], y=df_fusion_summary["BUY_Percentage"], name="BUY %", marker_color=COLOR_EMERALD))
            fig_stack.add_trace(go.Bar(x=df_fusion_summary["Ticker"], y=df_fusion_summary["HOLD_Percentage"], name="HOLD %", marker_color=COLOR_AMBER))
            fig_stack.add_trace(go.Bar(x=df_fusion_summary["Ticker"], y=df_fusion_summary["SELL_Percentage"], name="SELL %", marker_color=COLOR_RED))
            fig_stack.update_layout(barmode="stack")
            apply_terminal_layout(fig_stack, height=220, title="Decision Distribution by Company (%)")
            st.plotly_chart(fig_stack, use_container_width=True)

        with cp2:
            fig_sc = go.Figure()
            sc_colors = [COLOR_EMERALD if s >= 0 else COLOR_RED for s in df_fusion_summary["Average_Fusion_Score"]]
            fig_sc.add_trace(go.Bar(x=df_fusion_summary["Ticker"], y=df_fusion_summary["Average_Fusion_Score"], marker_color=sc_colors, name="Avg Score"))
            fig_sc.add_hline(y=0, line_color=COLOR_BORDER)
            apply_terminal_layout(fig_sc, height=220, title="Average Combined Signal Score")
            st.plotly_chart(fig_sc, use_container_width=True)

        render_section_header("Average Component Signals Across Equities", meta="Directional Component Means [-1, +1]")
        fig_sub = go.Figure()
        fig_sub.add_trace(go.Bar(x=df_fusion_summary["Ticker"], y=df_fusion_summary["Average_Classification_Signal"], name="Classification", marker_color=COLOR_CYAN))
        fig_sub.add_trace(go.Bar(x=df_fusion_summary["Ticker"], y=df_fusion_summary["Average_RL_Signal"], name="RL Strategy", marker_color="#818CF8"))
        fig_sub.add_trace(go.Bar(x=df_fusion_summary["Ticker"], y=df_fusion_summary["Average_Market_Regime_Signal"], name="Market Regime", marker_color=COLOR_AMBER))
        fig_sub.add_trace(go.Bar(x=df_fusion_summary["Ticker"], y=df_fusion_summary["Average_NLP_Signal"], name="News Signal", marker_color=COLOR_EMERALD))
        fig_sub.update_layout(barmode="group")
        apply_terminal_layout(fig_sub, height=240, title="Directional Component Averages")
        st.plotly_chart(fig_sub, use_container_width=True)


# ==============================================================================
# 11. PAGE 7 — SYSTEM OVERVIEW
# ==============================================================================
elif nav_selection == "7. System Overview":
    render_top_status_bar("System Architecture", ticker=None, date_str="SPECIFICATION", mode="SYSTEM REF")

    render_alert_box(
        "Quantitative trading intelligence and research system. BUY, HOLD, and SELL outputs represent model-generated algorithmic consensus signals and do not constitute financial or investment advice. Historical model evaluations reflect backtests and do not guarantee future performance.",
        level="warning",
        title="Trading & Research Disclaimer"
    )

    render_section_header("Pipeline Architecture", meta="Sequential Transformation Flow")
    st.code("""
    Market Data Ingestion (10 Target Equities)
        ↓
    Feature Engineering (13 Technical Indicators + PCA Orthogonalization)
        ↓
    Supervised Models (Regression, Ridge, Lasso, Logistic Regression, SVM)
        ↓
    Unsupervised Clustering (K-Means K=3 Behavioral Market Regimes)
        ↓
    Tabular Q-Learning Agent (Discrete Policy Mapping to Actions)
        ↓
    Financial News NLP (Loughran-McDonald 8-Stage Triage Pipeline)
        ↓
    Deterministic Multi-Modal Signal Fusion (35% Class + 30% RL + 15% Regime + 20% News)
        ↓
    BUY / HOLD / SELL Decision Output with Heuristic Confidence & Explainability
        ↓
    Operational Terminal Interface & Real-Time Ingestion Telemetry
    """, language="text")

    render_section_header("Core Analytical Subsystems", meta="Specifications & Methodologies")
    sec_col1, sec_col2 = st.columns(2)
    with sec_col1:
        st.markdown(r"""
        **Data Sources**:
        - **Historical Market OHLCV**: 10 target equities (`AAPL`, `MSFT`, `GOOGL`, `AMZN`, `NVDA`, `META`, `TSLA`, `JPM`, `V`, `JNJ`) covering 2019-01-15 through 2024-12-30 (15,010 observations; 1,501 per asset).
        - **Live Market Feed**: Delayed/snapshot daily market bars via Yahoo Finance (`yfinance`).
        - **Financial News Wires**: Real-world news via Alpha Vantage API (`NEWS_SENTIMENT`), enriched with internal Loughran-McDonald NLP scoring.
        - **Demonstration NLP Corpus**: Structured validation news records illustrating pipeline taxonomy and temporal decay.
        
        **Market Analytics**:
        - **Technical Feature Extraction**: 13 technical features per equity, including simple and exponential moving averages (SMA-5, SMA-10, EMA-12, EMA-26), price range, candlestick geometries (body, upper/lower shadows), and rolling volatility (5-day, 10-day).
        - **Dimensionality Reduction (PCA)**: Principal Component Analysis identifies orthogonal axes of variance; 6 components capture $\ge 95\%$ of cumulative variance across equities, eliminating multicollinearity.
        
        **Predictive Models**:
        - Supervised OLS regression, Ridge (L2) and Lasso (L1) regularized estimators evaluated across multiple regularization strengths, together with directional classifiers including Logistic Regression, Decision Trees, Gaussian Naive Bayes, and Support Vector Machines.
        """)
        
    with sec_col2:
        st.markdown(r"""
        **Market Regime Detection**:
        - **Unsupervised Clustering**: K-Means ($K=3$) partitions historical conditions into distinct regimes:
          - *Cluster 0 — Bearish Drawdown*: Negative returns, elevated volatility, wide intraday spreads.
          - *Cluster 1 — Calm Equilibrium*: Low volatility, balanced candlestick shadows, stable mean return.
          - *Cluster 2 — Bullish Expansion*: Positive returns, strong trading volume, sustained upside momentum.

        **Reinforcement Learning**:
        - Tabular Q-Learning mapping market states to BUY, HOLD, and SELL actions using forward-return-based rewards with transaction costs.
        
        **Financial News Intelligence**:
        - **Loughran-McDonald Financial Lexicon**: Domain-specific sentiment analysis tailored for financial terminology and negation handling.
        - **8-Stage Triage Engine**: Headline Ingestion $\rightarrow$ Cleaning $\rightarrow$ Sentiment $\rightarrow$ Relevance $\rightarrow$ Importance $\rightarrow$ Recency $\rightarrow$ Triage $\rightarrow$ Daily Signal.
        
        **Signal Fusion**:
        - Deterministic weighted aggregation across 4 channels:
          $$\text{Fusion Score} = 0.35 \times \text{Class} + 0.30 \times \text{RL} + 0.15 \times \text{Regime} + 0.20 \times \text{News}$$
        - Decision Boundaries: BUY ($\ge +0.20$), HOLD ($-0.20$ to $+0.20$), SELL ($\le -0.20$).
        """)

    render_section_header("Technology Stack & Platform Constraints", meta="Production Dependencies")
    t_col1, t_col2 = st.columns(2)
    with t_col1:
        st.markdown("""
        - **Python 3.13**: Computational core environment.
        - **Pandas & NumPy**: Vectorized time-series processing.
        - **Scikit-Learn**: Supervised estimators, regularized regression, PCA, and clustering.
        - **Plotly**: Interactive dark financial charts and candlestick visualizations.
        - **Streamlit**: Reactive real-time terminal shell framework.
        """)
    with t_col2:
        st.markdown("""
        - **Subsystem Isolation**: Historical model evaluations and real-time live telemetry operate independently without cross-layer data leakage.
        - **Deterministic Consensus**: Multi-modal fusion weights (35/30/15/20) and decision thresholds (±0.20) are architectural parameters designed for explainability.
        - **Rate-Limit & Quota Safety**: Ingestion pipelines automatically respect API rate limits and free-tier daily quotas with graceful degradation.
        """)


# ==============================================================================
# 8. PAGE 8 — REAL-TIME DATA
# ==============================================================================
elif nav_selection == "8. Real-Time Data":
    render_top_status_bar("Real-Time Data", selected_ticker, date_str=None, mode="LIVE LAYER")

    # ==========================================
    # STREAM MONITOR: ALPACA WEBSOCKET STREAM
    # ==========================================
    stream_info = get_stream_status()
    stream_status_val = str(stream_info.get("status", "STOPPED")).upper()
    is_stale = stream_info.get("is_stale", False)
    hb_age = stream_info.get("heartbeat_age_seconds")
    active_retries = stream_info.get("active_retry_attempt", 0)
    total_reconnects = stream_info.get("total_reconnects") if stream_info.get("total_reconnects") is not None else stream_info.get("reconnect_count", 0)
    is_live = (stream_status_val == "STREAMING") and not is_stale

    if is_live:
        stream_badge_html = f"<span class='term-badge badge-live'>LIVE STREAMING</span>"
    elif stream_status_val == "RECONNECTING":
        stream_badge_html = f"<span class='term-badge badge-warn'>RECONNECTING ({active_retries})</span>"
    elif stream_status_val in ["CONNECTING", "AUTHENTICATING", "SUBSCRIBING"]:
        stream_badge_html = f"<span class='term-badge badge-cyan'>{stream_status_val}</span>"
    elif is_stale:
        age_label = f"{hb_age:.0f}s AGO" if hb_age is not None else "STALE"
        stream_badge_html = f"<span class='term-badge badge-err'>OFFLINE ({age_label})</span>"
    else:
        stream_badge_html = "<span class='term-badge'>STANDBY / OFFLINE</span>"

    render_section_header("Alpaca Real-Time Financial News WebSocket Stream", meta="wss://stream.data.alpaca.markets/v1beta1/news")

    stream_col1, stream_col2, stream_col3 = st.columns([1.2, 1.2, 2.0])
    with stream_col1:
        if is_live:
            if st.button("STOP STREAM LISTENER", key="btn_stop_stream", use_container_width=True):
                stop_stream_listener()
                load_latest_real_articles.clear()
                st.rerun()
        else:
            if st.button("START STREAM LISTENER", key="btn_start_stream", use_container_width=True):
                start_stream_listener()
                st.rerun()

    with stream_col2:
        if st.button("REFRESH TELEMETRY & FEED", key="btn_refresh_stream_view", use_container_width=True):
            load_latest_real_articles.clear()
            st.rerun()

    with stream_col3:
        render_html(f"""
        <div style="font-size:10px; color:{COLOR_TEXT_MUTED}; font-family:monospace; padding-top:0.25rem;">
            SUBSCRIBED: 10 EQUITIES (AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, JPM, V, JNJ)<br>
            NORMALIZATION: FB → META ALIASING • INTERNAL LM NLP SCORING
        </div>
        """)

    rec_cnt = stream_info.get("articles_received_count", 0)
    sto_cnt = stream_info.get("articles_stored_count", 0)
    dup_cnt = stream_info.get("articles_duplicate_count", 0)
    hb_ts = stream_info.get("last_heartbeat_utc", "N/A")

    hb_sub = f"{hb_age:.0f}s ago" if hb_age is not None else "No Heartbeat"

    render_kpi_strip([
        {"label": "Articles Recv", "val": f"{rec_cnt}", "cls": "info"},
        {"label": "Stored SQLite", "val": f"{sto_cnt}", "cls": "pos"},
        {"label": "Dups Filtered", "val": f"{dup_cnt}", "cls": "warn"},
        {"label": "Reconnects", "val": f"{total_reconnects}", "sub": f"Active: {active_retries}"},
        {"label": "Last Heartbeat", "val": f"{hb_ts or 'N/A'}", "sub": f"({hb_sub})", "cls": "muted"}
    ])

    last_art = stream_info.get("last_article")
    if last_art:
        l_ticker = last_art.get("ticker", "N/A")
        l_head = last_art.get("headline", "")
        l_pub = last_art.get("published_at_utc", "")
        l_src = last_art.get("source", "Benzinga")
        l_sent_lbl = last_art.get("sentiment_label", "Neutral")
        l_sent_val = last_art.get("sentiment_score", 0.0)
        l_tri = last_art.get("triage_score", 0.0)
        l_ev = last_art.get("event_type", "General News")
        b_cls = "badge-live" if l_sent_lbl == "Positive" else ("badge-err" if l_sent_lbl == "Negative" else "badge-warn")

        render_html(f"""
        <div style="margin-top:0.4rem; padding:0.45rem 0.65rem; background:{COLOR_SURFACE_SEC}; border-left:3px solid {COLOR_EMERALD}; border-radius:2px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:10px; font-weight:700; color:{COLOR_EMERALD}; text-transform:uppercase;">
                    STREAMING ARTICLE SPOTLIGHT
                </span>
                <span class="term-badge {b_cls}">
                    {l_sent_lbl} ({l_sent_val:+.2f})
                </span>
            </div>
            <div style="font-size:11px; font-weight:700; color:{COLOR_TEXT_PRI}; margin-top:0.15rem;">
                [{l_ticker}] {l_head}
            </div>
            <div style="font-size:10px; color:{COLOR_TEXT_MUTED}; margin-top:0.15rem; font-family:monospace;">
                PUB: {l_pub} • SRC: {l_src} • EVENT: {l_ev} • TRIAGE: {l_tri:.3f}
            </div>
        </div>
        """)
    else:
        render_html(f"""
        <div style="margin-top:0.35rem; padding:0.35rem 0.55rem; background:{COLOR_SURFACE_SEC}; border-left:3px solid {COLOR_CYAN}; border-radius:2px; font-size:10px; color:{COLOR_TEXT_MUTED}; font-family:monospace;">
            STREAM STATUS: LISTENER ACTIVE OR STANDBY. INCOMING BENZINGA HEADLINES ARE NORMALIZED, NLP-SCORED, DEDUPLICATED, AND STORED IN REAL TIME.
        </div>
        """)

    if stream_info.get("error_message") and stream_status_val in ["ERROR", "RECONNECTING"]:
        render_alert_box(f"Stream Notice: {stream_info['error_message']}", level="warning")

    # Controlled Manual Refresh Interface
    if "last_refresh_timestamp" not in st.session_state:
        last_info = get_last_refresh_info()
        st.session_state["last_refresh_timestamp"] = last_info.get("timestamp_utc", "Never")
        st.session_state["refresh_status"] = last_info.get("status", "READY")
        st.session_state["refresh_summary"] = last_info.get("summary", "")
        st.session_state["refresh_error"] = ""

    if "refresh_banner" in st.session_state and st.session_state["refresh_banner"]:
        b_type, b_text = st.session_state["refresh_banner"]
        if b_type == "SUCCESS":
            render_alert_box(b_text, level="success")
        elif b_type == "PARTIAL":
            render_alert_box(b_text, level="warning")
        else:
            render_alert_box(b_text, level="error")
        del st.session_state["refresh_banner"]

    render_section_header("Controlled Data Refresh Control", meta="Manual Trigger • Rate-Limit & Quota Guarded")

    ref_c1, ref_c2 = st.columns([1.1, 2.2])
    with ref_c1:
        st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
        btn_refresh = st.button("↻ REFRESH REAL DATA", key="btn_refresh_real_data", use_container_width=True)
    with ref_c2:
        refresh_scope = st.radio(
            "Refresh Scope:",
            options=["Market + News", "Market only", "News only"],
            index=0,
            horizontal=True,
            help="Select data domain to refresh. Historical backfill is permanently closed."
        )

    curr_status = st.session_state.get("refresh_status", "READY")
    last_ts = st.session_state.get("last_refresh_timestamp", "Never")
    
    render_kpi_strip([
        {"label": "Last Refresh", "val": f"{last_ts}", "cls": "muted"},
        {"label": "Status", "val": f"{curr_status}", "cls": "pos" if curr_status == "SUCCESS" else ("warn" if curr_status in ["PARTIAL", "READY"] else "neg")},
        {"label": "Market Feed", "val": "Yahoo Finance", "sub": "Delayed / Snapshot", "cls": "info"},
        {"label": "News & NLP", "val": "Alpha Vantage", "sub": "Internal LM Scored", "cls": "info"}
    ])

    if st.session_state.get("refresh_summary"):
        render_html(f"""
        <div style="margin-top:0.35rem; padding:0.35rem 0.55rem; background:{COLOR_SURFACE_SEC}; border-left:3px solid {COLOR_CYAN}; font-size:10px; color:{COLOR_TEXT_SEC}; font-family:monospace;">
            TELEMETRY: {st.session_state['refresh_summary']}
        </div>
        """)

    if st.session_state.get("refresh_error"):
        render_alert_box(st.session_state['refresh_error'], level="warning")

    # Handle Refresh Execution
    if btn_refresh:
        with st.spinner("Executing controlled real data refresh & rebuilding analytics..."):
            res = execute_real_data_refresh(scope=refresh_scope)
            st.session_state["last_refresh_timestamp"] = res["timestamp_utc"]
            st.session_state["refresh_status"] = res["status"]
            
            m_part = f"Market: {res['market_refreshed_count']}/{res['total_tickers']}" if res['market_requested'] else "Market: Skipped"
            n_part = f"News: {res['news_refreshed_count']}/{res['total_tickers']}" if res['news_requested'] else "News: Skipped"
            art_part = f"New: {res['articles_added']} • Dups: {res['articles_duplicate']}"
            ana_part = "Analytics: YES" if res['analytics_rebuilt'] else "Analytics: NO"
            
            summary = f"{m_part} | {n_part} | {art_part} | {ana_part}"
            st.session_state["refresh_summary"] = summary
            
            err_msg = res.get("error_message", "")
            if res.get("quota_limited"):
                err_msg = "News refresh quota reached: " + err_msg
            st.session_state["refresh_error"] = err_msg
            
            # Clear Streamlit cache
            load_real_market_summary.clear()
            load_real_news_summary.clear()
            load_real_news_daily_summary.clear()
            load_real_data_coverage.clear()
            load_real_market_extension.clear()
            load_latest_real_articles.clear()

        if res["status"] == "SUCCESS":
            st.session_state["refresh_banner"] = ("SUCCESS", f"Refresh Complete: {summary}")
        elif res["status"] == "PARTIAL":
            st.session_state["refresh_banner"] = ("PARTIAL", f"Refresh Partial: {summary}. {st.session_state['refresh_error']}")
        else:
            st.session_state["refresh_banner"] = ("FAILED", f"Refresh Failed: {st.session_state['refresh_error'] or 'Unknown error'}")

        st.rerun()

    # Load Real-Data Artifacts
    df_rmarket = load_real_market_summary()
    df_rnews = load_real_news_summary()
    df_rdaily = load_real_news_daily_summary()
    df_rcov = load_real_data_coverage()
    df_rext = load_real_market_extension()

    # ==========================================
    # SECTION A: REAL MARKET SNAPSHOT
    # ==========================================
    render_section_header("A. Real Market Snapshot (10 Equities)", meta="Yahoo Finance Delayed/Snapshot Feed")

    if df_rmarket is not None and not df_rmarket.empty:
        m_display = df_rmarket[[
            "Company_Name", "Ticker", "Latest_Price", "Daily_Return", "SMA_5", "SMA_10",
            "EMA_12", "EMA_26", "Volatility_5", "High_Low_Range", "Volume", "Trend_Indication", "Data_Status"
        ]].copy()
        
        m_display["Daily_Return"] = m_display["Daily_Return"].apply(lambda v: f"{v*100:+.2f}%")
        m_display["Latest_Price"] = m_display["Latest_Price"].apply(lambda v: f"${v:,.2f}")
        m_display["SMA_5"] = m_display["SMA_5"].apply(lambda v: f"${v:,.2f}")
        m_display["SMA_10"] = m_display["SMA_10"].apply(lambda v: f"${v:,.2f}")
        m_display["EMA_12"] = m_display["EMA_12"].apply(lambda v: f"${v:,.2f}")
        m_display["EMA_26"] = m_display["EMA_26"].apply(lambda v: f"${v:,.2f}")
        m_display["Volatility_5"] = m_display["Volatility_5"].apply(lambda v: f"{v*100:.2f}%")
        m_display["High_Low_Range"] = m_display["High_Low_Range"].apply(lambda v: f"${v:,.2f}")
        m_display["Volume"] = m_display["Volume"].apply(lambda v: f"{v:,.0f}")
        
        m_display.columns = [
            "Company", "Ticker", "Price", "Return", "SMA 5", "SMA 10",
            "EMA 12", "EMA 26", "5D Vol", "H-L Range", "Volume", "Trend", "Status"
        ]
        
        st.dataframe(m_display, use_container_width=True, hide_index=True)

        snap_col1, snap_col2 = st.columns(2)
        with snap_col1:
            ret_vals = df_rmarket["Daily_Return"] * 100
            ret_colors = [COLOR_EMERALD if r >= 0 else COLOR_RED for r in ret_vals]
            fig_ret = go.Figure(go.Bar(
                x=df_rmarket["Ticker"],
                y=ret_vals,
                marker_color=ret_colors,
                text=[f"{r:+.2f}%" for r in ret_vals],
                textposition="auto"
            ))
            fig_ret.add_hline(y=0, line_color=COLOR_BORDER)
            apply_terminal_layout(fig_ret, height=200, title="Daily Return (%) Across Equities")
            st.plotly_chart(fig_ret, use_container_width=True)

        with snap_col2:
            vol_vals = df_rmarket["Volatility_5"] * 100
            fig_vol = go.Figure(go.Bar(
                x=df_rmarket["Ticker"],
                y=vol_vals,
                marker_color=COLOR_AMBER,
                text=[f"{v:.2f}%" for v in vol_vals],
                textposition="auto"
            ))
            apply_terminal_layout(fig_vol, height=200, title="5-Day Rolling Volatility (%) Across Equities")
            st.plotly_chart(fig_vol, use_container_width=True)

    # ==========================================
    # SECTION B: COMPANY DEEP DIVE
    # ==========================================
    render_section_header("B. Company Deep-Dive Analytics", meta="Historical Extension Series (90 Sessions)")
    real_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "JNJ"]
    real_selected = st.selectbox(
        "Select Asset for Deep Dive:",
        options=real_tickers,
        index=0,
        format_func=lambda t: f"{t} — {COMPANY_MAP.get(t, t)}"
    )

    if df_rmarket is not None and not df_rmarket.empty:
        c_sub = df_rmarket[df_rmarket["Ticker"] == real_selected]
        if not c_sub.empty:
            row = c_sub.iloc[0]
            price = row["Latest_Price"]
            change = row["Price_Change"]
            ret = row["Daily_Return"]
            sma5 = row["SMA_5"]
            sma10 = row["SMA_10"]
            ema12 = row["EMA_12"]
            ema26 = row["EMA_26"]
            vol5 = row["Volatility_5"]
            vol = row["Volume"]
            avg_vol = row["Average_Volume_20"]
            trend = row["Trend_Indication"]
            as_of = row["As_Of_Date"]
            
            trend_cls = "pos" if trend == "Bullish" else ("neg" if trend == "Bearish" else "warn")
            ret_cls = "pos" if ret >= 0 else "neg"
            
            render_kpi_strip([
                {"label": "Latest Price", "val": f"${price:,.2f}", "sub": f"{change:+.2f} ({ret*100:+.2f}%)", "cls": ret_cls},
                {"label": "Moving Averages", "val": f"${sma5:,.2f}", "sub": f"SMA 10: ${sma10:,.2f}"},
                {"label": "Exponential MA", "val": f"${ema12:,.2f}", "sub": f"EMA 26: ${ema26:,.2f}"},
                {"label": "5D Volatility", "val": f"{vol5*100:.2f}%", "sub": f"Ann: ~{vol5*np.sqrt(252)*100:.1f}%"},
                {"label": "Latest Volume", "val": f"{vol/1e6:.2f}M", "sub": f"20D Avg: {avg_vol/1e6:.2f}M"},
                {"label": "Trend Indication", "val": f"{trend}", "sub": f"As of {as_of}", "cls": trend_cls}
            ])

        if df_rext is not None and not df_rext.empty:
            t_ext = df_rext[df_rext["Ticker"] == real_selected].sort_values("Date").tail(90).copy()
            if not t_ext.empty:
                t_ext["SMA_5"] = t_ext["Close"].rolling(5, min_periods=2).mean()
                t_ext["SMA_10"] = t_ext["Close"].rolling(10, min_periods=3).mean()
                
                fig_ext = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.75, 0.25])
                
                fig_ext.add_trace(go.Candlestick(
                    x=t_ext["Date"],
                    open=t_ext["Open"],
                    high=t_ext["High"],
                    low=t_ext["Low"],
                    close=t_ext["Close"],
                    name="OHLC",
                    increasing_line_color=COLOR_EMERALD,
                    decreasing_line_color=COLOR_RED
                ), row=1, col=1)
                
                fig_ext.add_trace(go.Scatter(
                    x=t_ext["Date"], y=t_ext["SMA_5"], name="SMA 5",
                    line=dict(color=COLOR_CYAN, width=1.2)
                ), row=1, col=1)
                
                fig_ext.add_trace(go.Scatter(
                    x=t_ext["Date"], y=t_ext["SMA_10"], name="SMA 10",
                    line=dict(color=COLOR_AMBER, width=1.2)
                ), row=1, col=1)
                
                vol_colors = [COLOR_EMERALD if c >= o else COLOR_RED for c, o in zip(t_ext["Close"], t_ext["Open"])]
                fig_ext.add_trace(go.Bar(
                    x=t_ext["Date"], y=t_ext["Volume"], name="Volume",
                    marker_color=vol_colors, opacity=0.6
                ), row=2, col=1)
                
                apply_terminal_layout(fig_ext, height=300, title=f"{real_selected} Real Extension Series (90 Sessions)")
                fig_ext.update_xaxes(rangeslider_visible=False)
                st.plotly_chart(fig_ext, use_container_width=True)

    # ==========================================
    # SECTION C: REAL NEWS ANALYTICS
    # ==========================================
    render_section_header("C. Real Financial News NLP Analytics", meta="Loughran-McDonald Scoring")
    render_alert_box("Methodology Notice: All sentiment scores, entity relevance, and triage values are computed using the project's internal Loughran-McDonald dictionary and domain event rules. Provider-supplied scores are discarded.", level="info")

    if df_rnews is not None and not df_rnews.empty:
        cn_sub = df_rnews[df_rnews["Ticker"] == real_selected]
        if not cn_sub.empty:
            nrow = cn_sub.iloc[0]
            n_tot = nrow["Article_Count"]
            n_pos = nrow["Positive_Count"]
            n_neu = nrow["Neutral_Count"]
            n_neg = nrow["Negative_Count"]
            n_sent = nrow["Average_Sentiment"]
            n_rel = nrow["Average_Relevance"]
            n_imp = nrow["Average_Importance"]
            n_tri = nrow["Average_Triage_Score"]
            n_hi = nrow["High_Impact_Article_Count"]
            n_dom = nrow["Dominant_Sentiment"]
            
            dom_cls = "pos" if n_dom == "Positive" else ("neg" if n_dom == "Negative" else "warn")

            render_kpi_strip([
                {"label": "Total Articles", "val": f"{n_tot}", "sub": f"{real_selected} Real Records"},
                {"label": "Sentiment Split", "val": f"{n_pos}P / {n_neu}N / {n_neg}B", "sub": "Pos / Neu / Neg"},
                {"label": "Avg LM Sentiment", "val": f"{n_sent:+.4f}", "sub": "Bound: [-1, +1]", "cls": "pos" if n_sent >= 0 else "neg"},
                {"label": "Avg Relev / Imp", "val": f"{n_rel:.2f} / {n_imp:.2f}", "sub": "Target & Event"},
                {"label": "High Impact News", "val": f"{n_hi}", "sub": "Triage ≥ 0.50"},
                {"label": "Dominant Sentiment", "val": f"{n_dom}", "sub": f"Triage: {n_tri:.3f}", "cls": dom_cls}
            ])

        nlp_c1, nlp_c2, nlp_c3 = st.columns([1, 1.2, 1.2])
        
        with nlp_c1:
            if not cn_sub.empty:
                pie_fig = go.Figure(data=[go.Pie(
                    labels=["Positive", "Neutral", "Negative"],
                    values=[n_pos, n_neu, n_neg],
                    hole=0.55,
                    marker=dict(colors=[COLOR_EMERALD, COLOR_TEXT_MUTED, COLOR_RED]),
                    textinfo="label+value"
                )])
                apply_terminal_layout(pie_fig, height=200, title=f"{real_selected} Sentiment Distribution")
                st.plotly_chart(pie_fig, use_container_width=True)

        with nlp_c2:
            sent_colors = [COLOR_EMERALD if s >= 0 else COLOR_RED for s in df_rnews["Average_Sentiment"]]
            fig_all_sent = go.Figure(go.Bar(
                x=df_rnews["Ticker"],
                y=df_rnews["Average_Sentiment"],
                marker_color=sent_colors,
                text=[f"{s:+.3f}" for s in df_rnews["Average_Sentiment"]],
                textposition="auto"
            ))
            fig_all_sent.add_hline(y=0, line_color=COLOR_BORDER)
            apply_terminal_layout(fig_all_sent, height=200, title="Avg LM Sentiment Across Equities")
            st.plotly_chart(fig_all_sent, use_container_width=True)

        with nlp_c3:
            fig_cnt = go.Figure()
            fig_cnt.add_trace(go.Bar(
                x=df_rnews["Ticker"], y=df_rnews["Article_Count"],
                name="Total Articles", marker_color=COLOR_CYAN
            ))
            fig_cnt.add_trace(go.Bar(
                x=df_rnews["Ticker"], y=df_rnews["High_Impact_Article_Count"],
                name="High Impact", marker_color=COLOR_AMBER
            ))
            fig_cnt.update_layout(barmode="group")
            apply_terminal_layout(fig_cnt, height=200, title="Article Volume & High Impact")
            st.plotly_chart(fig_cnt, use_container_width=True)

    # ==========================================
    # SECTION D: DAILY NEWS SIGNALS
    # ==========================================
    render_section_header("D. Daily News Signals & Sentiment Time-Series", meta="data/live/analytics/real_news_daily_summary.csv")

    if df_rdaily is not None and not df_rdaily.empty:
        t_daily = df_rdaily[df_rdaily["Ticker"] == real_selected].sort_values("Date").copy()
        if not t_daily.empty:
            sig_colors = [COLOR_EMERALD if s >= 0 else COLOR_RED for s in t_daily["News_Signal"]]
            
            fig_dsig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.70, 0.30])
            
            fig_dsig.add_trace(go.Bar(
                x=t_daily["Date"],
                y=t_daily["News_Signal"],
                name="Daily News Signal",
                marker_color=sig_colors
            ), row=1, col=1)
            
            fig_dsig.add_trace(go.Scatter(
                x=t_daily["Date"],
                y=t_daily["Triage_Weighted_Sentiment"],
                name="Triage-Weighted Sent",
                line=dict(color=COLOR_CYAN, width=1.2, dash="dot")
            ), row=1, col=1)
            
            fig_dsig.add_trace(go.Bar(
                x=t_daily["Date"],
                y=t_daily["News_Count"],
                name="Article Count",
                marker_color=COLOR_TEXT_MUTED
            ), row=2, col=1)
            
            apply_terminal_layout(fig_dsig, height=260, title=f"{real_selected} Daily News Signals & Article Volume")
            st.plotly_chart(fig_dsig, use_container_width=True)

            with st.expander(f"View {real_selected} Daily News Signal Records ({len(t_daily)} Sessions)"):
                st.dataframe(
                    t_daily[[
                        "Date", "News_Count", "Average_Sentiment", "Triage_Weighted_Sentiment",
                        "Average_Triage_Score", "High_Impact_News_Count", "Dominant_Sentiment", "News_Signal"
                    ]],
                    use_container_width=True, hide_index=True
                )

    # ==========================================
    # SECTION E: REAL NEWS COVERAGE & AUDIT
    # ==========================================
    render_section_header("E. Real News Coverage & Provider Quota Audit", meta="Alpha Vantage Free-Tier Bounds")
    render_alert_box(
        "Coverage Disclosure: Historical news coverage is partial for certain assets due to the Alpha Vantage free-tier limit (25 calls/day). This reflects external API rate limits, not pipeline defects.",
        level="warning"
    )

    if df_rcov is not None and not df_rcov.empty:
        cov_disp = df_rcov[[
            "Ticker", "Company_Name", "Real_Article_Count", "January_Article_Count",
            "Earliest_Article_Date", "Latest_Article_Date", "January_Coverage_Status",
            "Unique_Sources_Count", "Coverage_Notes"
        ]].copy()
        
        cov_disp.columns = [
            "Ticker", "Company", "Total Articles", "Jan 2025 Articles",
            "Earliest Date", "Latest Date", "Coverage Status", "Unique Sources", "Audit / Quota Notes"
        ]
        st.dataframe(cov_disp, use_container_width=True, hide_index=True)

    # ==========================================
    # SECTION F: LATEST REAL FINANCIAL NEWS FEED
    # ==========================================
    render_section_header("F. Latest Real Financial News Feed", meta="Alpaca Stream & Alpha Vantage Enriched Feed")

    feed_options = ["AAPL (Benchmark Asset)", f"{real_selected} (Selected Asset)"] if real_selected != "AAPL" else ["AAPL (Selected Asset)"]
    feed_ticker = st.radio("News Feed Focus:", options=feed_options, horizontal=True)
    target_news_ticker = "AAPL" if "AAPL" in feed_ticker else real_selected
    
    articles_df = load_latest_real_articles(target_news_ticker, limit=15)
    
    if articles_df is not None and not articles_df.empty:
        st.markdown(f"<div style='font-size:10px; color:{COLOR_TEXT_MUTED}; margin-bottom:0.4rem; font-family:monospace;'>DISPLAYING {len(articles_df)} VERIFIED REAL ARTICLES FOR {target_news_ticker}:</div>", unsafe_allow_html=True)
        
        for _, art in articles_df.iterrows():
            sent_lbl = art["sentiment_label"]
            sent_val = art["sentiment_score"]
            badge_color = COLOR_EMERALD if sent_lbl == "Positive" else (COLOR_RED if sent_lbl == "Negative" else COLOR_AMBER)
            prov = str(art.get("provider", "Alpha Vantage"))
            is_streamed = "alpaca" in prov.lower()
            prov_label = "ALPACA STREAM" if is_streamed else "ALPHA VANTAGE"
            prov_badge_cls = "badge-live" if is_streamed else "term-badge"
            
            render_html(f"""
            <div class="panel-box" style="border-left: 3px solid {badge_color}; margin-bottom: 0.35rem;">
                <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:0.25rem;">
                    <div style="font-size:12px; font-weight:700; color:{COLOR_TEXT_PRI}; line-height:1.3;">
                        {art['headline']}
                    </div>
                    <span class="term-badge" style="border-color:{badge_color}; color:{badge_color}; margin-left:0.6rem; white-space:nowrap;">
                        {sent_lbl} ({sent_val:+.2f})
                    </span>
                </div>
                <div style="display:flex; flex-wrap:wrap; gap:0.5rem; font-size:10px; color:{COLOR_TEXT_MUTED}; align-items:center; font-family:monospace;">
                    <span class="term-badge {prov_badge_cls}">{prov_label}</span>
                    <span>•</span>
                    <span>SRC: {art['source']}</span>
                    <span>•</span>
                    <span>PUB: {art['published_at_utc']}</span>
                    <span>•</span>
                    <span>EVENT: {art['event_type']}</span>
                    <span>•</span>
                    <span>REL: {art['relevance_score']:.2f}</span>
                    <span>•</span>
                    <span>TRIAGE: {art['triage_score']:.3f}</span>
                </div>
            </div>
            """)

            # Plain-English Financial News Explanation
            from src.nlp.news_explanation import NewsExplanationEngine
            expl = NewsExplanationEngine().get_or_create_explanation(art.to_dict(), art.to_dict())

            imp_dir = expl.implication_direction
            imp_color = COLOR_EMERALD if imp_dir == "POSITIVE" else (COLOR_RED if imp_dir == "NEGATIVE" else COLOR_AMBER)
            ev_color = COLOR_EMERALD if expl.evidence_strength == "DIRECT" else (COLOR_AMBER if expl.evidence_strength == "PARTIAL" else COLOR_RED)

            with st.expander(f"Plain-English Financial Explanation — {imp_dir} Implications ({expl.evidence_strength} Evidence)", expanded=False):
                render_html(f"""
                <div style="background:{COLOR_SURFACE_SEC}; border:1px solid {COLOR_BORDER}; border-radius:2px; padding:0.6rem 0.75rem; font-size:11px; line-height:1.45;">
                    <div style="margin-bottom:0.4rem;">
                        <span style="color:{COLOR_CYAN}; font-weight:700; text-transform:uppercase; font-size:10px;">WHAT THE ARTICLE REPORTS:</span><br>
                        <span style="color:{COLOR_TEXT_PRI};">{expl.what_it_reports}</span>
                    </div>
                    <div style="margin-bottom:0.4rem;">
                        <span style="color:{COLOR_TEXT_SEC}; font-weight:700; text-transform:uppercase; font-size:10px;">WHY IT MATTERS TO {expl.company_name} ({expl.ticker}):</span><br>
                        <span style="color:{COLOR_TEXT_SEC};">{expl.why_it_matters}</span>
                    </div>
                    <div style="margin-bottom:0.4rem;">
                        <span style="color:{imp_color}; font-weight:700; text-transform:uppercase; font-size:10px;">BUSINESS IMPLICATIONS ({imp_dir}):</span><br>
                        <span style="color:{COLOR_TEXT_PRI};">{expl.implications}</span>
                    </div>
                    <div style="margin-bottom:0.4rem;">
                        <span style="color:{COLOR_EMERALD}; font-weight:700; text-transform:uppercase; font-size:10px;">GROUNDED EVIDENCE:</span>
                        <span class="term-badge" style="border-color:{ev_color}; color:{ev_color}; margin-left:0.3rem;">{expl.evidence_strength} EVIDENCE</span><br>
                        <span style="color:{COLOR_TEXT_MUTED}; font-family:monospace; font-size:10px;">{expl.evidence}</span>
                    </div>
                    <div style="margin-bottom:0.4rem;">
                        <span style="color:{COLOR_AMBER}; font-weight:700; text-transform:uppercase; font-size:10px;">KEY LIMITATIONS & MISSING INFO:</span><br>
                        <span style="color:{COLOR_TEXT_SEC};">{expl.limitations}</span>
                    </div>
                    <div style="padding:0.35rem 0.5rem; background:{COLOR_SURFACE_PRI}; border-left:2px solid {COLOR_AMBER}; border-radius:2px; color:{COLOR_TEXT_MUTED}; font-size:10px;">
                        <b>DISCLAIMER:</b> {expl.directional_disclaimer}
                    </div>
                </div>
                """)
            st.markdown("<div style='height: 0.15rem;'></div>", unsafe_allow_html=True)
    else:
        st.info(f"No news article records available in SQLite storage for {target_news_ticker}.")
