"""
AI-Based Intelligent Trading System - Modern Trading Terminal (Dashboard V2)
Presentation & Real-Time Trading Intelligence Layer

Author: Pushkar
Universe: 10 Companies (AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, JPM, V, JNJ)
Data Period: 2019-01-15 to 2024-12-30 (Historical / Processed Offline Dataset)
Out-of-Sample Evaluation: 2023-10-20 to 2024-12-30 (300 Sessions per Company)
Engine: Streamlit + Plotly (Dark Trading Terminal Interface)
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


# ==========================================
# 1. PAGE CONFIGURATION & DARK TERMINAL CSS
# ==========================================
st.set_page_config(
    page_title="AI Trading System | Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Density Dark Terminal Theme CSS
st.markdown("""
<style>
    /* Dark Terminal Theme Foundation */
    .stApp {
        background-color: #0b0e14 !important;
        color: #e2e8f0 !important;
    }
    .main .block-container {
        padding-top: 1.0rem !important;
        padding-bottom: 2.0rem !important;
        padding-left: 1.8rem !important;
        padding-right: 1.8rem !important;
        max-width: 98% !important;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0d1117 !important;
        border-right: 1px solid #1e2638 !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: #1e2638 !important;
        margin: 0.8rem 0 !important;
    }
    
    /* Terminal Header */
    .terminal-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-bottom: 1px solid #1e2638;
        padding-bottom: 0.6rem;
        margin-bottom: 1.0rem;
    }
    .terminal-title {
        font-size: 1.75rem;
        font-weight: 800;
        color: #f8fafc;
        letter-spacing: -0.5px;
        margin: 0;
    }
    .terminal-subtitle {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 0.15rem;
    }
    
    /* Status Badges */
    .badge-container {
        display: flex;
        gap: 0.4rem;
        align-items: center;
    }
    .status-badge {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        border-radius: 4px;
        font-size: 0.70rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        font-family: monospace;
    }
    .badge-offline {
        background: rgba(56, 189, 248, 0.12);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }
    .badge-demo {
        background: rgba(245, 158, 11, 0.12);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    .badge-ticker {
        background: rgba(129, 140, 248, 0.15);
        color: #818cf8;
        border: 1px solid rgba(129, 140, 248, 0.35);
        font-size: 0.82rem;
        padding: 0.25rem 0.65rem;
    }

    /* KPI Cards */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        gap: 0.6rem;
        margin-bottom: 1.0rem;
    }
    .kpi-card {
        background: #11151f;
        border: 1px solid #1e2638;
        border-radius: 6px;
        padding: 0.75rem 0.85rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.25);
    }
    .kpi-label {
        font-size: 0.70rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.25rem;
    }
    .kpi-value {
        font-size: 1.35rem;
        font-weight: 700;
        color: #f8fafc;
        line-height: 1.2;
    }
    .kpi-sub {
        font-size: 0.72rem;
        color: #64748b;
        margin-top: 0.2rem;
    }

    /* Color semantics */
    .color-bullish { color: #00c076 !important; }
    .color-bearish { color: #ff4d4d !important; }
    .color-neutral { color: #f59e0b !important; }
    .color-info    { color: #38bdf8 !important; }

    /* Decision Spotlight Banner */
    .decision-banner-buy {
        background: linear-gradient(135deg, rgba(0, 192, 118, 0.15), rgba(16, 185, 129, 0.05));
        border: 1px solid rgba(0, 192, 118, 0.45);
        border-left: 5px solid #00c076;
        border-radius: 6px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.8rem;
    }
    .decision-banner-sell {
        background: linear-gradient(135deg, rgba(255, 77, 77, 0.15), rgba(239, 68, 68, 0.05));
        border: 1px solid rgba(255, 77, 77, 0.45);
        border-left: 5px solid #ff4d4d;
        border-radius: 6px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.8rem;
    }
    .decision-banner-hold {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15), rgba(217, 119, 6, 0.05));
        border: 1px solid rgba(245, 158, 11, 0.45);
        border-left: 5px solid #f59e0b;
        border-radius: 6px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.8rem;
    }

    /* Signal Component Box */
    .signal-box {
        background: #121722;
        border: 1px solid #1e2638;
        border-radius: 6px;
        padding: 0.7rem 0.85rem;
        margin-bottom: 0.45rem;
    }
    .signal-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.78rem;
        font-weight: 700;
        color: #cbd5e1;
        margin-bottom: 0.2rem;
    }
    .signal-desc {
        font-size: 0.72rem;
        color: #94a3b8;
    }

    /* Terminal Alert / Limitation Box */
    .terminal-alert {
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.35);
        border-left: 4px solid #f59e0b;
        padding: 0.75rem 1.0rem;
        border-radius: 6px;
        color: #fde68a;
        font-size: 0.85rem;
        line-height: 1.45;
        margin-bottom: 1.0rem;
    }
    
    /* Section Divider */
    .terminal-section {
        font-size: 1.05rem;
        font-weight: 700;
        color: #f1f5f9;
        letter-spacing: -0.2px;
        margin-top: 1.2rem;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
        border-bottom: 1px solid #1e2638;
        padding-bottom: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
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

# ==========================================
# 2. CACHED DATA LOADERS & VALIDATION
# ==========================================
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
    st.error(f"⚠️ Warning: Missing required processed files in `{PROCESSED_DIR}`:")
    for mf in MISSING:
        st.write(f"- `{mf}`")
    st.info("Ensure all analytical data pipelines and model artifacts have completed generation.")
    st.stop()

# Helper aliases
df_feat = DATA["features"]
df_fusion = DATA["fusion"]
df_fusion_summary = DATA["fusion_summary"]

# ==========================================
# 2B. REAL-DATA ANALYTICS LOADERS (CACHED)
# ==========================================
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

# ==========================================
# 3. TECHNICAL INDICATORS (VISUALIZATION ONLY)
# ==========================================
@st.cache_data(show_spinner=False)
def compute_visualization_technicals(df_company: pd.DataFrame) -> pd.DataFrame:
    """
    Computes visualization-only technical indicators (EMA, MACD, RSI, BB, Drawdown)
    directly from existing market data.
    IMPORTANT: These indicators DO NOT affect any ML models, RL policy, NLP, or fusion decisions.
    """
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

# Plotly Terminal Layout Helper
def apply_terminal_layout(fig, height=None, title=None, show_legend=True):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#10141d",
        plot_bgcolor="#10141d",
        font=dict(color="#cbd5e1", family="Inter, system-ui, sans-serif", size=11),
        margin=dict(l=45, r=30, t=35 if title else 15, b=25),
        showlegend=show_legend,
        legend=dict(
            bgcolor="rgba(16, 20, 29, 0.75)",
            bordercolor="#262f40",
            borderwidth=1,
            font=dict(size=10),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(
            gridcolor="#1a2233",
            zerolinecolor="#2d3748",
            showline=True,
            linecolor="#2d3748"
        ),
        yaxis=dict(
            gridcolor="#1a2233",
            zerolinecolor="#2d3748",
            showline=True,
            linecolor="#2d3748"
        )
    )
    if height:
        fig.update_layout(height=height)
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=12, color="#f1f5f9")))
    return fig


# ==========================================
# 4. SIDEBAR NAVIGATION & COMPANY SELECTOR
# ==========================================
st.sidebar.markdown("""
<div style="padding: 0.2rem 0; border-bottom: 1px solid #1e2638; margin-bottom: 0.8rem;">
    <div style="font-size: 1.15rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.3px;">
        ⚡ AI TRADING SYSTEM
    </div>
    <div style="font-size: 0.75rem; color: #94a3b8; font-weight: 500;">
        Historical Analytics Terminal
    </div>
</div>
""", unsafe_allow_html=True)

# Status Badges
st.sidebar.markdown("""
<div style="display: flex; gap: 0.4rem; margin-bottom: 0.8rem;">
    <span class="status-badge badge-offline">DATA: OFFLINE</span>
    <span class="status-badge badge-demo">NEWS: DEMO</span>
</div>
""", unsafe_allow_html=True)

# Prominent Company Selector
st.sidebar.markdown("<div style='font-size:0.75rem; font-weight:700; color:#64748b; margin-bottom:0.2rem;'>ASSET UNIVERSE (10 EQUITIES)</div>", unsafe_allow_html=True)
company_options = [f"{t} — {COMPANY_MAP[t]}" for t in COMPANY_MAP.keys()]
selected_company_str = st.sidebar.selectbox("Active Asset", options=company_options, index=0, label_visibility="collapsed")
selected_ticker = selected_company_str.split(" — ")[0]
selected_company_name = COMPANY_MAP[selected_ticker]

st.sidebar.markdown(f"""
<div style="background:#131824; border:1px solid #1e2638; border-radius:5px; padding:0.5rem 0.7rem; margin-top:0.4rem; margin-bottom:0.8rem;">
    <div style="font-size:0.72rem; color:#64748b;">SELECTED ASSET</div>
    <div style="font-size:0.95rem; font-weight:700; color:#f1f5f9;">{selected_ticker} • {selected_company_name}</div>
    <div style="font-size:0.68rem; color:#94a3b8; margin-top:0.2rem;">Range: 2019-01-15 to 2024-12-30 (1,501 Sessions)</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

# Navigation Menu
st.sidebar.markdown("<div style='font-size:0.75rem; font-weight:700; color:#64748b; margin-bottom:0.3rem;'>TERMINAL NAVIGATION</div>", unsafe_allow_html=True)
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

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size: 0.70rem; color: #64748b; line-height: 1.4;">
    <b>Notice</b>: Quantitative trading intelligence terminal. Historical model evaluations and real-time data analytics operate as separate modular subsystems.
</div>
""", unsafe_allow_html=True)

# Prepare active company data
df_feat_raw = df_feat[df_feat["Ticker"] == selected_ticker].sort_values("Date").reset_index(drop=True) if df_feat is not None else None
df_feat_comp = compute_visualization_technicals(df_feat_raw) if df_feat_raw is not None else None
df_fusion_comp = df_fusion[df_fusion["Ticker"] == selected_ticker].sort_values("Date").reset_index(drop=True) if df_fusion is not None else None

latest_fusion = df_fusion_comp.iloc[-1] if (df_fusion_comp is not None and len(df_fusion_comp) > 0) else None
latest_date = latest_fusion["Date"] if latest_fusion is not None else "N/A"
latest_feat = df_feat_comp[df_feat_comp["Date"] == latest_date].iloc[0] if (df_feat_comp is not None and len(df_feat_comp[df_feat_comp["Date"] == latest_date]) > 0) else (df_feat_comp.iloc[-1] if df_feat_comp is not None else None)


# ==========================================
# 5. PAGE 1 — OVERVIEW
# ==========================================
if nav_selection == "1. Overview":
    # Terminal Header
    st.markdown(f"""
    <div class="terminal-header">
        <div>
            <div class="terminal-title">{selected_company_name} ({selected_ticker})</div>
            <div class="terminal-subtitle">Historical Market Intelligence • Out-of-Sample Decision Consensus</div>
        </div>
        <div class="badge-container">
            <span class="status-badge badge-ticker">{selected_ticker}</span>
            <span class="status-badge badge-offline">SESSION: {latest_date}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if latest_fusion is not None and latest_feat is not None:
        # Top 6 Semantic KPI Cards
        ret_val = latest_feat["Daily_Return"]
        ret_class = "color-bullish" if ret_val >= 0 else "color-bearish"
        ret_sign = "+" if ret_val >= 0 else ""
        
        dec_val = latest_fusion["Final_Decision"]
        dec_class = "color-bullish" if dec_val == "BUY" else ("color-bearish" if dec_val == "SELL" else "color-neutral")
        
        score_val = latest_fusion["Fusion_Score"]
        score_class = "color-bullish" if score_val >= 0.20 else ("color-bearish" if score_val <= -0.20 else "color-neutral")
        
        st.markdown(f"""
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Latest Price</div>
                <div class="kpi-value">${latest_feat['Close']:.2f}</div>
                <div class="kpi-sub">Open: ${latest_feat['Open']:.2f}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Daily Change</div>
                <div class="kpi-value {ret_class}">{ret_sign}{ret_val:.2%}</div>
                <div class="kpi-sub">Range: ${latest_feat['Price_Range']:.2f}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Market Condition</div>
                <div class="kpi-value color-info" style="font-size:1.05rem; padding-top:0.2rem;">{latest_fusion['Market_Regime_Label']}</div>
                <div class="kpi-sub">Cluster {latest_fusion['Market_Regime']}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Final Decision</div>
                <div class="kpi-value {dec_class}">{dec_val}</div>
                <div class="kpi-sub">Threshold: |Score| ≥ 0.20</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Combined Signal Score</div>
                <div class="kpi-value {score_class}">{score_val:+.4f}</div>
                <div class="kpi-sub">Range: [-1.00, +1.00]</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Signal Confidence</div>
                <div class="kpi-value color-info">{latest_fusion['Heuristic_Confidence_Percent']:.1f}%</div>
                <div class="kpi-sub">Heuristic (|Score|)</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Main Workspace: Left Chart (70%) | Right Signal Overview (30%)
        col_chart, col_signals = st.columns([7, 3])

        with col_chart:
            st.markdown("<div style='font-size:0.88rem; font-weight:700; color:#f1f5f9; margin-bottom:0.3rem;'>📊 INTERACTIVE PRICE TERMINAL (CANDLESTICK + OVERLAYS)</div>", unsafe_allow_html=True)
            
            # Indicator toggle row
            it1, it2, it3, it4 = st.columns(4)
            with it1:
                show_smas = st.checkbox("SMA 5 & 10", value=True)
            with it2:
                show_emas = st.checkbox("EMA 12 & 26 (Vis. only)", value=False)
            with it3:
                show_bb = st.checkbox("Bollinger Bands (Vis. only)", value=False)
            with it4:
                show_markers = st.checkbox("BUY/SELL Signals", value=True)

            # Build Candlestick Figure with Plotly
            fig_main = go.Figure()

            # Base Candlestick
            fig_main.add_trace(go.Candlestick(
                x=df_feat_comp["Date"],
                open=df_feat_comp["Open"],
                high=df_feat_comp["High"],
                low=df_feat_comp["Low"],
                close=df_feat_comp["Close"],
                name="OHLC Price",
                increasing_line_color="#00c076",
                decreasing_line_color="#ff4d4d",
                increasing_fillcolor="rgba(0, 192, 118, 0.4)",
                decreasing_fillcolor="rgba(255, 77, 77, 0.4)"
            ))

            # SMAs
            if show_smas:
                fig_main.add_trace(go.Scatter(x=df_feat_comp["Date"], y=df_feat_comp["SMA_5"], mode="lines", name="SMA 5", line=dict(color="#f59e0b", width=1.3)))
                fig_main.add_trace(go.Scatter(x=df_feat_comp["Date"], y=df_feat_comp["SMA_10"], mode="lines", name="SMA 10", line=dict(color="#38bdf8", width=1.3)))

            # EMAs
            if show_emas:
                fig_main.add_trace(go.Scatter(x=df_feat_comp["Date"], y=df_feat_comp["EMA_12"], mode="lines", name="EMA 12", line=dict(color="#22d3ee", width=1.1, dash="dot")))
                fig_main.add_trace(go.Scatter(x=df_feat_comp["Date"], y=df_feat_comp["EMA_26"], mode="lines", name="EMA 26", line=dict(color="#a855f7", width=1.1, dash="dot")))

            # Bollinger Bands
            if show_bb:
                fig_main.add_trace(go.Scatter(x=df_feat_comp["Date"], y=df_feat_comp["BB_Upper"], mode="lines", name="BB Upper", line=dict(color="rgba(148, 163, 184, 0.5)", width=1)))
                fig_main.add_trace(go.Scatter(x=df_feat_comp["Date"], y=df_feat_comp["BB_Lower"], mode="lines", name="BB Lower", line=dict(color="rgba(148, 163, 184, 0.5)", width=1), fill="tonexty", fillcolor="rgba(148, 163, 184, 0.07)"))

            # BUY / SELL Markers from Decision Fusion output
            if show_markers and df_fusion_comp is not None:
                # Merge Close price for marker positioning
                df_f_merged = df_fusion_comp.merge(df_feat_comp[["Date", "Low", "High"]], on="Date", how="inner")
                buys = df_f_merged[df_f_merged["Final_Decision"] == "BUY"]
                sells = df_f_merged[df_f_merged["Final_Decision"] == "SELL"]

                if len(buys) > 0:
                    fig_main.add_trace(go.Scatter(
                        x=buys["Date"],
                        y=buys["Low"] * 0.985,
                        mode="markers",
                        name="BUY Decision",
                        marker=dict(symbol="triangle-up", size=9, color="#00c076", line=dict(width=1, color="#ffffff")),
                        hoverinfo="text",
                        hovertext=[f"BUY Date: {d}<br>Score: {s:+.3f}<br>Conf: {c:.1f}%" for d, s, c in zip(buys["Date"], buys["Fusion_Score"], buys["Heuristic_Confidence_Percent"])]
                    ))
                if len(sells) > 0:
                    fig_main.add_trace(go.Scatter(
                        x=sells["Date"],
                        y=sells["High"] * 1.015,
                        mode="markers",
                        name="SELL Decision",
                        marker=dict(symbol="triangle-down", size=9, color="#ff4d4d", line=dict(width=1, color="#ffffff")),
                        hoverinfo="text",
                        hovertext=[f"SELL Date: {d}<br>Score: {s:+.3f}<br>Conf: {c:.1f}%" for d, s, c in zip(sells["Date"], sells["Fusion_Score"], sells["Heuristic_Confidence_Percent"])]
                    ))

            fig_main.update_xaxes(
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(step="all", label="ALL")
                    ]),
                    bgcolor="#161c28",
                    font=dict(color="#cbd5e1", size=10),
                    activecolor="#2563eb"
                )
            )
            apply_terminal_layout(fig_main, height=450)
            st.plotly_chart(fig_main, use_container_width=True)

        with col_signals:
            st.markdown("<div style='font-size:0.88rem; font-weight:700; color:#f1f5f9; margin-bottom:0.3rem;'>⚡ SIGNAL OVERVIEW</div>", unsafe_allow_html=True)
            
            # 1. Classification
            p_up = latest_fusion["Classification_P_UP"]
            c_sig = latest_fusion["Classification_Signal"]
            c_contrib = latest_fusion["Classification_Contribution"]
            c_color = "color-bullish" if c_sig > 0 else ("color-bearish" if c_sig < 0 else "color-neutral")
            st.markdown(f"""
            <div class="signal-box">
                <div class="signal-header">
                    <span>1. CLASSIFICATION (35%)</span>
                    <span class="{c_color}">{c_sig:+.3f}</span>
                </div>
                <div class="signal-desc">
                    Est. Chance of Next-Day Rise: <b>{p_up:.1%}</b><br>
                    Contribution: <b>{c_contrib:+.4f}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # 2. RL Strategy
            rl_act = int(latest_fusion["RL_Action"])
            rl_lbl = "BUY" if rl_act == 2 else ("SELL" if rl_act == 0 else "HOLD")
            rl_sig = latest_fusion["RL_Signal"]
            rl_contrib = latest_fusion["RL_Contribution"]
            rl_color = "color-bullish" if rl_sig > 0 else ("color-bearish" if rl_sig < 0 else "color-neutral")
            st.markdown(f"""
            <div class="signal-box">
                <div class="signal-header">
                    <span>2. RL STRATEGY (30%)</span>
                    <span class="{rl_color}">{rl_lbl} ({rl_sig:+.0f})</span>
                </div>
                <div class="signal-desc">
                    Q-Learning Frozen Policy Action<br>
                    Contribution: <b>{rl_contrib:+.4f}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # 3. Market Condition
            reg_lbl = latest_fusion["Market_Regime_Label"]
            reg_sig = latest_fusion["Market_Regime_Signal"]
            reg_contrib = latest_fusion["Market_Regime_Contribution"]
            reg_color = "color-bullish" if reg_sig > 0 else ("color-bearish" if reg_sig < 0 else "color-neutral")
            st.markdown(f"""
            <div class="signal-box">
                <div class="signal-header">
                    <span>3. MARKET CONDITION (15%)</span>
                    <span class="{reg_color}">{reg_lbl}</span>
                </div>
                <div class="signal-desc">
                    Normalized Behavioral Cluster<br>
                    Contribution: <b>{reg_contrib:+.4f}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # 4. News Signal
            nlp_sig = latest_fusion["NLP_Signal"]
            nlp_contrib = latest_fusion["NLP_Contribution"]
            nlp_color = "color-bullish" if nlp_sig > 0 else ("color-bearish" if nlp_sig < 0 else "color-neutral")
            st.markdown(f"""
            <div class="signal-box">
                <div class="signal-header">
                    <span>4. NEWS SIGNAL (20%)</span>
                    <span class="{nlp_color}">{nlp_sig:+.3f}</span>
                </div>
                <div class="signal-desc">
                    Demonstration Triage Sentiment<br>
                    Contribution: <b>{nlp_contrib:+.4f}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Why this decision banner
            banner_class = "decision-banner-buy" if dec_val == "BUY" else ("decision-banner-sell" if dec_val == "SELL" else "decision-banner-hold")
            st.markdown(f"""
            <div class="{banner_class}">
                <div style="font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.5px;">WHY THIS DECISION?</div>
                <div style="font-size:0.83rem; margin-top:0.25rem; line-height:1.4;">
                    {latest_fusion['Decision_Explanation']}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Market Snapshot Section: 4 Compact Charts (Volume, Returns, Volatility, Drawdown)
        st.markdown("<div class='terminal-section'>📉 HISTORICAL MARKET SNAPSHOT (DESCRIPTIVE ANALYTICS)</div>", unsafe_allow_html=True)
        snap1, snap2, snap3, snap4 = st.columns(4)

        with snap1:
            fig_v = go.Figure()
            vol_colors = ["#00c076" if r >= 0 else "#ff4d4d" for r in df_feat_comp["Daily_Return"]]
            fig_v.add_trace(go.Bar(x=df_feat_comp["Date"], y=df_feat_comp["Volume"], marker_color=vol_colors, name="Volume"))
            apply_terminal_layout(fig_v, height=190, title="Trading Volume", show_legend=False)
            st.plotly_chart(fig_v, use_container_width=True)

        with snap2:
            fig_r = go.Figure()
            fig_r.add_trace(go.Bar(x=df_feat_comp["Date"], y=df_feat_comp["Daily_Return"], marker_color=vol_colors, name="Daily Return"))
            apply_terminal_layout(fig_r, height=190, title="Daily Return Series", show_legend=False)
            st.plotly_chart(fig_r, use_container_width=True)

        with snap3:
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Scatter(x=df_feat_comp["Date"], y=df_feat_comp["Volatility_5"], line=dict(color="#f59e0b", width=1.2), fill="tozeroy", fillcolor="rgba(245, 158, 11, 0.1)", name="Vol 5D"))
            apply_terminal_layout(fig_vol, height=190, title="5-Day Rolling Volatility", show_legend=False)
            st.plotly_chart(fig_vol, use_container_width=True)

        with snap4:
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(x=df_feat_comp["Date"], y=df_feat_comp["Drawdown"], line=dict(color="#ff4d4d", width=1.2), fill="tozeroy", fillcolor="rgba(255, 77, 77, 0.12)", name="Drawdown"))
            apply_terminal_layout(fig_dd, height=190, title="Peak-to-Trough Drawdown", show_legend=False)
            st.plotly_chart(fig_dd, use_container_width=True)


# ==========================================
# 6. PAGE 2 — MARKET ANALYSIS
# ==========================================
elif nav_selection == "2. Market Analysis":
    st.markdown(f"""
    <div class="terminal-header">
        <div>
            <div class="terminal-title">Market & Quantitative Technical Analytics</div>
            <div class="terminal-subtitle">{selected_company_name} ({selected_ticker}) • Technical Indicators, Risk Metrics, and PCA</div>
        </div>
        <div class="badge-container">
            <span class="status-badge badge-ticker">{selected_ticker}</span>
            <span class="status-badge badge-offline">HISTORICAL</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if df_feat_comp is not None:
        # SECTION 1: PRICE & TREND
        st.markdown("<div class='terminal-section'>1. PRICE & TREND (CANDLESTICKS, MOVING AVERAGES & BOLLINGER BANDS)</div>", unsafe_allow_html=True)
        fig_pt = go.Figure()
        fig_pt.add_trace(go.Candlestick(
            x=df_feat_comp["Date"], open=df_feat_comp["Open"], high=df_feat_comp["High"],
            low=df_feat_comp["Low"], close=df_feat_comp["Close"], name="OHLC",
            increasing_line_color="#00c076", decreasing_line_color="#ff4d4d"
        ))
        fig_pt.add_trace(go.Scatter(x=df_feat_comp["Date"], y=df_feat_comp["SMA_5"], name="SMA 5", line=dict(color="#f59e0b", width=1.2)))
        fig_pt.add_trace(go.Scatter(x=df_feat_comp["Date"], y=df_feat_comp["SMA_10"], name="SMA 10", line=dict(color="#38bdf8", width=1.2)))
        fig_pt.add_trace(go.Scatter(x=df_feat_comp["Date"], y=df_feat_comp["EMA_12"], name="EMA 12", line=dict(color="#22d3ee", width=1, dash="dot")))
        fig_pt.add_trace(go.Scatter(x=df_feat_comp["Date"], y=df_feat_comp["EMA_26"], name="EMA 26", line=dict(color="#a855f7", width=1, dash="dot")))
        fig_pt.add_trace(go.Scatter(x=df_feat_comp["Date"], y=df_feat_comp["BB_Upper"], name="BB Upper", line=dict(color="rgba(148, 163, 184, 0.4)", width=1)))
        fig_pt.add_trace(go.Scatter(x=df_feat_comp["Date"], y=df_feat_comp["BB_Lower"], name="BB Lower", line=dict(color="rgba(148, 163, 184, 0.4)", width=1), fill="tonexty", fillcolor="rgba(148, 163, 184, 0.06)"))
        apply_terminal_layout(fig_pt, height=380)
        st.plotly_chart(fig_pt, use_container_width=True)

        # SECTION 2: MOMENTUM (RSI & MACD)
        st.markdown("<div class='terminal-section'>2. MOMENTUM INDICATORS (VISUALIZATION ONLY)</div>", unsafe_allow_html=True)
        st.caption("⚠️ Note: These momentum indicators are computed purely for dashboard visual analysis and DO NOT influence model predictions or fusion scores.")
        
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=df_feat_comp["Date"], y=df_feat_comp["RSI_14"], line=dict(color="#38bdf8", width=1.3), name="RSI 14"))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="#ff4d4d", annotation_text="Overbought (70)")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="#00c076", annotation_text="Oversold (30)")
            fig_rsi.update_yaxes(range=[0, 100])
            apply_terminal_layout(fig_rsi, height=220, title="RSI (14-Day)")
            st.plotly_chart(fig_rsi, use_container_width=True)

        with m_col2:
            fig_macd = go.Figure()
            hist_colors = ["#00c076" if h >= 0 else "#ff4d4d" for h in df_feat_comp["MACD_Hist"]]
            fig_macd.add_trace(go.Bar(x=df_feat_comp["Date"], y=df_feat_comp["MACD_Hist"], marker_color=hist_colors, name="MACD Hist"))
            fig_macd.add_trace(go.Scatter(x=df_feat_comp["Date"], y=df_feat_comp["MACD"], line=dict(color="#f59e0b", width=1.2), name="MACD (12,26)"))
            fig_macd.add_trace(go.Scatter(x=df_feat_comp["Date"], y=df_feat_comp["MACD_Signal"], line=dict(color="#a855f7", width=1.2), name="Signal (9)"))
            apply_terminal_layout(fig_macd, height=220, title="MACD (12, 26, 9)")
            st.plotly_chart(fig_macd, use_container_width=True)

        # SECTION 3: RISK & VOLATILITY
        st.markdown("<div class='terminal-section'>3. RISK & RETURN DISTRIBUTION</div>", unsafe_allow_html=True)
        r_col1, r_col2 = st.columns(2)
        with r_col1:
            fig_dist = px.histogram(
                df_feat_comp, x="Daily_Return", nbins=60, title="Daily Return Distribution & Fat Tails",
                color_discrete_sequence=["#38bdf8"]
            )
            apply_terminal_layout(fig_dist, height=230)
            st.plotly_chart(fig_dist, use_container_width=True)
        with r_col2:
            fig_dd2 = go.Figure()
            fig_dd2.add_trace(go.Scatter(x=df_feat_comp["Date"], y=df_feat_comp["Drawdown"], line=dict(color="#ff4d4d", width=1.2), fill="tozeroy", fillcolor="rgba(255, 77, 77, 0.15)", name="Drawdown"))
            apply_terminal_layout(fig_dd2, height=230, title="Peak-to-Trough Drawdown Curve")
            st.plotly_chart(fig_dd2, use_container_width=True)

        # SECTION 4: MARKET REGIME TIMELINE
        st.markdown("<div class='terminal-section'>4. MARKET CONDITION / REGIME TIMELINE</div>", unsafe_allow_html=True)
        df_clust_results = DATA["clust_results"]
        if df_clust_results is not None:
            df_cr_comp = df_clust_results[df_clust_results["Ticker"] == selected_ticker].sort_values("Date").reset_index(drop=True)
            regime_map = {0: "Bearish Drawdown", 1: "Calm Equilibrium", 2: "Bullish Expansion"}
            df_cr_comp["Regime_Name"] = df_cr_comp["Cluster"].map(regime_map)
            
            fig_reg = px.scatter(
                df_cr_comp, x="Date", y="Cluster", color="Regime_Name",
                color_discrete_map={"Bearish Drawdown": "#ff4d4d", "Calm Equilibrium": "#38bdf8", "Bullish Expansion": "#00c076"},
                title=f"{selected_ticker} — Unsupervised Market Condition State Across Time",
                labels={"Cluster": "Cluster ID"}
            )
            fig_reg.update_traces(marker=dict(size=4))
            apply_terminal_layout(fig_reg, height=200)
            st.plotly_chart(fig_reg, use_container_width=True)

        # SECTION 5: PCA
        st.markdown("<div class='terminal-section'>5. PRINCIPAL COMPONENT ANALYSIS (PCA)</div>", unsafe_allow_html=True)
        st.info("💡 **Dimensionality Reduction**: The 13 correlated technical features are projected into orthogonal axes. Across the universe, 6 components capture **≥ 95%** of cumulative variance, eliminating multicollinearity before model ingestion.")
        
        df_pca = DATA["pca"]
        if df_pca is not None:
            df_pca_comp = df_pca[df_pca["Ticker"] == selected_ticker].reset_index(drop=True)
            p1, p2 = st.columns([6, 4])
            with p1:
                fig_pca_scatter = px.scatter(
                    df_pca_comp, x="PC1", y="PC2", color=df_pca_comp.index,
                    color_continuous_scale="Viridis",
                    title=f"{selected_ticker} — 2D Projection: PC1 (Trend) vs PC2 (Volatility)",
                    labels={"color": "Session Index"}
                )
                fig_pca_scatter.update_traces(marker=dict(size=5, opacity=0.7))
                apply_terminal_layout(fig_pca_scatter, height=270)
                st.plotly_chart(fig_pca_scatter, use_container_width=True)
            with p2:
                # Variance breakdown mock-up based on notebook 1 results
                pca_var_df = pd.DataFrame({
                    "Component": ["PC1", "PC2", "PC3", "PC4", "PC5", "PC6"],
                    "Variance (%)": [44.8, 22.3, 14.1, 7.2, 4.3, 2.9],
                    "Cumulative (%)": [44.8, 67.1, 81.2, 88.4, 92.7, 95.6]
                })
                fig_pca_var = px.bar(
                    pca_var_df, x="Component", y="Variance (%)",
                    title="Explained Variance by Component (≥ 95% at PC6)",
                    color_discrete_sequence=["#818cf8"]
                )
                apply_terminal_layout(fig_pca_var, height=270)
                st.plotly_chart(fig_pca_var, use_container_width=True)

        # SECTION 6: CORRELATION HEATMAP
        st.markdown("<div class='terminal-section'>6. FEATURE CORRELATION MATRIX</div>", unsafe_allow_html=True)
        corr_cols = ["Close", "Volume", "Daily_Return", "Price_Range", "Candle_Body", "Upper_Shadow", "Lower_Shadow", "Volatility_5"]
        corr_df = df_feat_comp[corr_cols].corr()
        fig_heat = px.imshow(
            corr_df, text_auto=".2f", aspect="auto",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title=f"{selected_ticker} Technical Feature Pearson Correlations"
        )
        apply_terminal_layout(fig_heat, height=320)
        st.plotly_chart(fig_heat, use_container_width=True)


# ==========================================
# 7. PAGE 3 — PREDICTIVE MODELS
# ==========================================
elif nav_selection == "3. Predictive Models":
    st.markdown(f"""
    <div class="terminal-header">
        <div>
            <div class="terminal-title">Predictive Models & Machine Learning Signals</div>
            <div class="terminal-subtitle">{selected_company_name} ({selected_ticker}) • Multi-Model Predictive Algorithms & Quantitative Signals</div>
        </div>
        <div class="badge-container">
            <span class="status-badge badge-ticker">{selected_ticker}</span>
            <span class="status-badge badge-offline">QUANTITATIVE MODELS</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab_reg, tab_regul, tab_class, tab_clust, tab_rl = st.tabs([
        "📈 Regression",
        "🎯 Regularization",
        "⚖️ Classification",
        "🔍 Clustering (Regimes)",
        "🤖 Reinforcement Learning"
    ])

    # 1. REGRESSION
    with tab_reg:
        st.markdown("<div class='terminal-section'>HOW CLOSE WAS THE PREDICTED NEXT-DAY PRICE?</div>", unsafe_allow_html=True)
        st.caption("Supervised Linear Regression forecasting $y_{t+1} = \\text{Close}_{t+1}$. Note: High $R^2$ reflects price tracking, not guaranteed trading execution profitability.")
        
        df_reg_sum = DATA["reg_summary"]
        if df_reg_sum is not None:
            comp_reg = df_reg_sum[df_reg_sum["Ticker"] == selected_ticker].iloc[0]
            rc1, rc2, rc3, rc4 = st.columns(4)
            with rc1:
                st.metric("Test MAE", f"${comp_reg['Test_MAE']:.2f}")
            with rc2:
                st.metric("Test MSE", f"{comp_reg['Test_MSE']:.2f}")
            with rc3:
                st.metric("Test RMSE", f"${comp_reg['Test_RMSE']:.2f}")
            with rc4:
                st.metric("Test R²", f"{comp_reg['Test_R2']:.4f}")

        df_reg_preds = DATA["reg_preds"]
        if df_reg_preds is not None:
            comp_preds = df_reg_preds[df_reg_preds["Ticker"] == selected_ticker].sort_values("Date").reset_index(drop=True)
            
            # Actual vs Predicted Plotly Chart
            fig_rp = go.Figure()
            fig_rp.add_trace(go.Scatter(x=comp_preds["Date"], y=comp_preds["Actual_Next_Close"], name="Actual Next Close", line=dict(color="#38bdf8", width=1.5)))
            fig_rp.add_trace(go.Scatter(x=comp_preds["Date"], y=comp_preds["Predicted_Next_Close"], name="Predicted Next Close", line=dict(color="#f59e0b", width=1.5, dash="dash")))
            apply_terminal_layout(fig_rp, height=300, title=f"{selected_ticker} — Out-of-Sample Actual vs Predicted Price (300 Sessions)")
            st.plotly_chart(fig_rp, use_container_width=True)

            # Residuals
            fig_res = go.Figure()
            fig_res.add_trace(go.Bar(x=comp_preds["Date"], y=comp_preds["Residual"], name="Residual (Actual - Pred)", marker_color="#818cf8"))
            apply_terminal_layout(fig_res, height=180, title="Prediction Residual Error ($)")
            st.plotly_chart(fig_res, use_container_width=True)

    # 2. REGULARIZATION
    with tab_regul:
        st.markdown("<div class='terminal-section'>REGULARIZATION: SHRINKAGE & FEATURE SELECTION</div>", unsafe_allow_html=True)
        st.info("💡 **Why Regularization?**: Regularization penalizes overly large model coefficients to mitigate multicollinearity. Ridge ($L_2$) shrinks weights smoothly, while Lasso ($L_1$) drives redundant indicator coefficients to zero for automated feature selection.")

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
                    title="Test RMSE Comparison Across Estimators",
                    color_discrete_sequence=["#38bdf8", "#00c076", "#f59e0b"]
                )
                apply_terminal_layout(fig_reg_comp, height=260)
                st.plotly_chart(fig_reg_comp, use_container_width=True)

            with r_col2:
                st.markdown("##### Hyperparameters & Feature Sparsity")
                st.markdown(f"""
                - **Optimal Ridge $\\alpha$**: `{comp_reg_s['Ridge_Best_Alpha']}` (Test RMSE: `${comp_reg_s['Ridge_Test_RMSE']:.2f}`)
                - **Optimal Lasso $\\alpha$**: `{comp_reg_s['Lasso_Best_Alpha']}` (Test RMSE: `${comp_reg_s['Lasso_Test_RMSE']:.2f}`)
                - **Lasso Active Features Retained**: `{int(comp_reg_s['Lasso_Active_Features'])}`
                - **Lasso Redundant Features Zeroed**: `{int(comp_reg_s['Lasso_Zeroed_Features'])}`
                """)
                
                sparsity_df = pd.DataFrame({
                    "Status": ["Active Retained Features", "Zeroed Eliminated Features"],
                    "Count": [int(comp_reg_s['Lasso_Active_Features']), int(comp_reg_s['Lasso_Zeroed_Features'])]
                })
                fig_pie = px.pie(sparsity_df, values="Count", names="Status", color_discrete_sequence=["#00c076", "#ff4d4d"], hole=0.5)
                apply_terminal_layout(fig_pie, height=190, title="Lasso Feature Sparsity Breakdown")
                st.plotly_chart(fig_pie, use_container_width=True)

    # 3. CLASSIFICATION
    with tab_class:
        st.markdown("<div class='terminal-section'>SUPERVISED DIRECTIONAL CLASSIFICATION (UP vs DOWN)</div>", unsafe_allow_html=True)
        st.info("💡 **Directional Definition**: 'UP' means next closing price was higher than today's close ($y_{t+1} > y_t$). The calibrated probability $P_{\\text{UP}}$ from **Logistic Regression** is mapped to the directional signal $2 \\times P_{\\text{UP}} - 1$.")

        df_class_sum = DATA["class_summary"]
        if df_class_sum is not None:
            comp_cs = df_class_sum[df_class_sum["Ticker"] == selected_ticker]
            
            c_col1, c_col2 = st.columns([6, 4])
            with c_col1:
                fig_cls = px.bar(
                    comp_cs, x="Model", y=["Accuracy", "Precision_UP", "Recall_UP", "F1_UP"],
                    barmode="group", title=f"{selected_ticker} — Classification Model Performance Metrics",
                    color_discrete_sequence=["#38bdf8", "#00c076", "#f59e0b", "#818cf8"]
                )
                apply_terminal_layout(fig_cls, height=280)
                st.plotly_chart(fig_cls, use_container_width=True)

            with c_col2:
                df_cm = DATA["class_cm"]
                if df_cm is not None:
                    lr_cm = df_cm[(df_cm["Ticker"] == selected_ticker) & (df_cm["Model"] == "Logistic Regression")].iloc[0]
                    cm_mat = np.array([[lr_cm["TN"], lr_cm["FP"]], [lr_cm["FN"], lr_cm["TP"]]])
                    fig_cm = px.imshow(
                        cm_mat, text_auto=True, x=["DOWN", "UP"], y=["DOWN", "UP"],
                        color_continuous_scale="Blues", title="Logistic Regression Confusion Matrix"
                    )
                    apply_terminal_layout(fig_cm, height=250)
                    st.plotly_chart(fig_cm, use_container_width=True)

    # 4. CLUSTERING
    with tab_clust:
        st.markdown("<div class='terminal-section'>UNSUPERVISED MARKET CONDITION CLUSTERING</div>", unsafe_allow_html=True)
        st.caption("K-Means ($K=3$) descriptive market conditions segmented on scale-normalized behavioral technical features.")

        df_c_by_comp = DATA["clust_company"]
        if df_c_by_comp is not None:
            comp_c_dist = df_c_by_comp[df_c_by_comp["Ticker"] == selected_ticker]
            c_col1, c_col2 = st.columns(2)
            with c_col1:
                fig_cl_bar = px.bar(
                    comp_c_dist, x="Cluster", y="Percentage", color="Cluster",
                    color_discrete_sequence=["#ff4d4d", "#38bdf8", "#00c076"],
                    title=f"{selected_ticker} — Historical Market Condition Allocation",
                    labels={"Percentage": "Percentage of Sessions (%)"}
                )
                apply_terminal_layout(fig_cl_bar, height=270)
                st.plotly_chart(fig_cl_bar, use_container_width=True)

            with c_col2:
                st.markdown("##### Regime Descriptive Taxonomy")
                st.markdown("""
                - **Cluster 0 — Bearish Drawdown**: Negative mean daily return, elevated intraday spread, high volatility.
                - **Cluster 1 — Calm Equilibrium**: Near-zero return, lowest volatility, balanced shadows.
                - **Cluster 2 — Bullish Expansion**: Strongly positive daily return, sustained expansion volume.
                """)

    # 5. REINFORCEMENT LEARNING
    with tab_rl:
        st.markdown("<div class='terminal-section'>TABULAR Q-LEARNING TRADING AGENT (HISTORICAL BACKTEST)</div>", unsafe_allow_html=True)
        st.caption("⚠️ Historical Backtest: Q-Learning policy evaluated on discrete state tuples $(M, D, V)$ over the 300 test sessions. Past performance is no guarantee of future results.")

        df_rl_sum = DATA["rl_summary"]
        if df_rl_sum is not None:
            comp_rl = df_rl_sum[df_rl_sum["Ticker"] == selected_ticker].iloc[0]
            rl1, rl2, rl3, rl4 = st.columns(4)
            with rl1:
                st.metric("RL Agent Return", f"{comp_rl['Total_Return']:+.2%}", delta=f"B&H: {comp_rl['Buy_Hold_Return']:+.2%}")
            with rl2:
                st.metric("Max Drawdown", f"{comp_rl['Maximum_Drawdown']:.2%}", delta=f"B&H: {comp_rl['Buy_Hold_Max_Drawdown']:.2%}", delta_color="inverse")
            with rl3:
                st.metric("Sharpe Ratio", f"{comp_rl['Sharpe_Ratio']:.2f}", delta=f"B&H: {comp_rl['Buy_Hold_Sharpe_Ratio']:.2f}")
            with rl4:
                st.metric("Market Exposure", f"{comp_rl['Exposure']:.1%}")

        df_rl_res = DATA["rl_results"]
        if df_rl_res is not None:
            comp_rl_res = df_rl_res[df_rl_res["Ticker"] == selected_ticker]
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(x=comp_rl_res["Date"], y=comp_rl_res["Portfolio_Value"], name="RL Strategy Equity ($)", line=dict(color="#00c076", width=1.5)))
            apply_terminal_layout(fig_eq, height=270, title=f"{selected_ticker} — Q-Learning Agent Cumulative Equity Growth ($10,000 Init)")
            st.plotly_chart(fig_eq, use_container_width=True)


# ==========================================
# 8. PAGE 4 — NEWS & NLP
# ==========================================
elif nav_selection == "4. News & NLP":
    # Terminal Header
    st.markdown(f"""
    <div class="terminal-header">
        <div>
            <div class="terminal-title">Financial News NLP & Sentiment Triage</div>
            <div class="terminal-subtitle">{selected_company_name} ({selected_ticker}) • Demonstration NLP Ingestion Pipeline</div>
        </div>
        <div class="badge-container">
            <span class="status-badge badge-ticker">{selected_ticker}</span>
            <span class="status-badge badge-demo">DEMONSTRATION CORPUS</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # NLP Pipeline Data Disclosure
    st.markdown("""
    <div class="terminal-alert">
        ℹ️ <b>NLP PIPELINE DATA DISCLOSURE</b><br>
        This section illustrates the 8-stage Loughran-McDonald NLP ingestion and triage pipeline using structured demonstration news records to validate entity extraction, relevance weighting, and temporal decay scoring. Real-world financial news ingestion and live analytics are available in the <b>Real-Time Data</b> module.
    </div>
    """, unsafe_allow_html=True)

    # How News Is Analyzed: Visual 8-Step Pipeline
    st.markdown("<div class='terminal-section'>HOW NEWS IS ANALYZED: 8-STAGE TRIAGE PIPELINE</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="display:flex; justify-content:space-between; gap:0.3rem; margin-bottom:1.0rem; flex-wrap:wrap;">
        <div class="signal-box" style="flex:1; min-width:110px; text-align:center;"><b>1. Headline</b><br><span style="font-size:0.68rem; color:#94a3b8;">Raw Text Ingestion</span></div>
        <div class="signal-box" style="flex:1; min-width:110px; text-align:center;"><b>2. Cleaning</b><br><span style="font-size:0.68rem; color:#94a3b8;">Tokenize & Clean</span></div>
        <div class="signal-box" style="flex:1; min-width:110px; text-align:center;"><b>3. Sentiment</b><br><span style="font-size:0.68rem; color:#94a3b8;">LM-Lexicon + Negation</span></div>
        <div class="signal-box" style="flex:1; min-width:110px; text-align:center;"><b>4. Relevance</b><br><span style="font-size:0.68rem; color:#94a3b8;">Entity Match [0, 1]</span></div>
        <div class="signal-box" style="flex:1; min-width:110px; text-align:center;"><b>5. Importance</b><br><span style="font-size:0.68rem; color:#94a3b8;">Event Gravity [0, 1]</span></div>
        <div class="signal-box" style="flex:1; min-width:110px; text-align:center;"><b>6. Recency</b><br><span style="font-size:0.68rem; color:#94a3b8;">Decay Factor [0, 1]</span></div>
        <div class="signal-box" style="flex:1; min-width:110px; text-align:center;"><b>7. Triage Score</b><br><span style="font-size:0.68rem; color:#94a3b8;">Relev × Rec × Imp</span></div>
        <div class="signal-box" style="flex:1; min-width:110px; text-align:center;"><b>8. Daily Signal</b><br><span style="font-size:0.68rem; color:#94a3b8;">Avg Sent × Avg Triage</span></div>
    </div>
    """, unsafe_allow_html=True)

    # Formulas
    f_col1, f_col2 = st.columns(2)
    with f_col1:
        st.latex(r"\text{News Impact Score (Triage)} = \text{Relevance} \times \text{Recency} \times \text{Importance}")
    with f_col2:
        st.latex(r"\text{Daily News Signal} = \text{Average Sentiment} \times \text{Average News Impact Score}")

    # News Summary Metrics
    df_n_sum = DATA["news_summary"]
    if df_n_sum is not None:
        comp_ns = df_n_sum[df_n_sum["Ticker"] == selected_ticker].iloc[0]
        n1, n2, n3, n4, n5 = st.columns(5)
        with n1:
            st.metric("Total Headlines", int(comp_ns["Total_News_Items"]))
        with n2:
            st.metric("Positive News", int(comp_ns["Positive_News_Count"]))
        with n3:
            st.metric("Neutral News", int(comp_ns["Neutral_News_Count"]))
        with n4:
            st.metric("Negative News", int(comp_ns["Negative_News_Count"]))
        with n5:
            st.metric("Avg Sentiment", f"{comp_ns['Average_Sentiment']:+.3f}")

    # Charts: Daily News Signal Timeline & Sentiment Pie
    st.markdown("<div class='terminal-section'>NEWS SIGNAL & SENTIMENT DYNAMICS</div>", unsafe_allow_html=True)
    df_n_daily = DATA["news_daily"]
    if df_n_daily is not None:
        comp_nd = df_n_daily[df_n_daily["Ticker"] == selected_ticker].sort_values("Date").reset_index(drop=True)
        
        nc1, nc2 = st.columns([7, 3])
        with nc1:
            fig_ns = go.Figure()
            sig_colors = ["#00c076" if s >= 0 else "#ff4d4d" for s in comp_nd["News_Signal"]]
            fig_ns.add_trace(go.Bar(x=comp_nd["Date"], y=comp_nd["News_Signal"], marker_color=sig_colors, name="Daily News Signal"))
            apply_terminal_layout(fig_ns, height=240, title=f"{selected_ticker} — Daily News Signal [-1.0, +1.0] (Neutral on Zero-News Days)")
            st.plotly_chart(fig_ns, use_container_width=True)

        with nc2:
            sent_dist = pd.DataFrame({
                "Sentiment": ["Positive", "Neutral", "Negative"],
                "Count": [int(comp_ns["Positive_News_Count"]), int(comp_ns["Neutral_News_Count"]), int(comp_ns["Negative_News_Count"])]
            })
            fig_spie = px.pie(sent_dist, values="Count", names="Sentiment", color="Sentiment",
                              color_discrete_map={"Positive": "#00c076", "Neutral": "#f59e0b", "Negative": "#ff4d4d"},
                              title="Sentiment Distribution", hole=0.45)
            apply_terminal_layout(fig_spie, height=240)
            st.plotly_chart(fig_spie, use_container_width=True)

    # Headline-Level Records Table
    df_n_nlp = DATA["news_nlp"]
    if df_n_nlp is not None:
        st.markdown("<div class='terminal-section'>HEADLINE-LEVEL NLP RECORDS (DEMONSTRATION CORPUS)</div>", unsafe_allow_html=True)
        comp_articles = df_n_nlp[df_n_nlp["Ticker"] == selected_ticker].sort_values("Date", ascending=False).reset_index(drop=True)
        st.dataframe(
            comp_articles[["Date", "Headline", "Sentiment_Label", "Sentiment_Score", "Relevance_Score", "Importance_Score", "Triage_Score"]].style.format({
                "Sentiment_Score": "{:+.3f}", "Relevance_Score": "{:.2f}", "Importance_Score": "{:.2f}", "Triage_Score": "{:.3f}"
            }),
            use_container_width=True
        )


# ==========================================
# 9. PAGE 5 — SIGNAL FUSION
# ==========================================
elif nav_selection == "5. Signal Fusion":
    st.markdown(f"""
    <div class="terminal-header">
        <div>
            <div class="terminal-title">Multi-Modal Signal Fusion Terminal</div>
            <div class="terminal-subtitle">{selected_company_name} ({selected_ticker}) • 4-Channel Weighted Consensus Engine</div>
        </div>
        <div class="badge-container">
            <span class="status-badge badge-ticker">{selected_ticker}</span>
            <span class="status-badge badge-offline">OUT-OF-SAMPLE TEST</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if df_fusion_comp is not None and len(df_fusion_comp) > 0:
        # Session Target Selector
        d_mode = st.radio("Target Session", ["Latest Available Date", "Pick from History (300 OOS Days)"], horizontal=True)
        if d_mode == "Latest Available Date":
            target_date = df_fusion_comp["Date"].iloc[-1]
        else:
            target_date = st.selectbox("Select Trading Session Date", options=list(reversed(df_fusion_comp["Date"].tolist())))
            
        row_f = df_fusion_comp[df_fusion_comp["Date"] == target_date].iloc[0]

        # Top Decision Spotlight
        dec = row_f["Final_Decision"]
        sc = row_f["Fusion_Score"]
        conf = row_f["Heuristic_Confidence_Percent"]
        b_class = "decision-banner-buy" if dec == "BUY" else ("decision-banner-sell" if dec == "SELL" else "decision-banner-hold")
        
        st.markdown(f"""
        <div class="{b_class}">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <div style="font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.5px;">CURRENT SYSTEM SIGNAL</div>
                    <div style="font-size:1.8rem; font-weight:800; letter-spacing:1px; margin-top:0.1rem;">{dec}</div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:0.85rem; color:#cbd5e1;">Combined Signal Score: <b style="font-size:1.2rem;">{sc:+.4f}</b></div>
                    <div style="font-size:0.85rem; color:#94a3b8;">Signal Confidence: <b>{conf:.1f}%</b> (Heuristic)</div>
                </div>
            </div>
            <div style="margin-top:0.5rem; font-size:0.85rem; border-top:1px solid rgba(255,255,255,0.1); padding-top:0.4rem;">
                <b>Why this decision?</b> {row_f['Decision_Explanation']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Horizontal Contribution Visualization & Formula
        st.markdown("<div class='terminal-section'>COMPONENT CONTRIBUTIONS & FIXED CONSENSUS LOGIC</div>", unsafe_allow_html=True)
        
        c_left, c_right = st.columns([6, 4])
        with c_left:
            # Horizontal Bar Chart
            fig_cont = go.Figure()
            components = ["Classification (35%)", "RL Strategy (30%)", "Market Condition (15%)", "News NLP (20%)"]
            contribs = [row_f["Classification_Contribution"], row_f["RL_Contribution"], row_f["Market_Regime_Contribution"], row_f["NLP_Contribution"]]
            c_colors = ["#00c076" if c >= 0 else "#ff4d4d" for c in contribs]
            
            fig_cont.add_trace(go.Bar(
                y=components, x=contribs, orientation="h", marker_color=c_colors,
                text=[f"{c:+.4f}" for c in contribs], textposition="auto"
            ))
            fig_cont.add_vline(x=0, line_color="#475569", line_dash="dash")
            fig_cont.update_xaxes(range=[-0.4, 0.4], title="Score Contribution")
            apply_terminal_layout(fig_cont, height=230, title=f"Component Contributions for {target_date}")
            st.plotly_chart(fig_cont, use_container_width=True)

        with c_right:
            st.markdown("##### Fixed Multi-Channel Formula")
            st.latex(r"\text{Score} = 0.35 \times \text{Class} + 0.30 \times \text{RL} + 0.15 \times \text{Regime} + 0.20 \times \text{NLP}")
            st.markdown("""
            **Fixed Thresholds**:
            - **BUY**: Score ≥ +0.20
            - **HOLD**: between -0.20 and +0.20
            - **SELL**: Score ≤ -0.20
            """)
            st.caption("Zero test-period optimization. Weights and thresholds are manually fixed architectural choices.")

        # Historical Timeline
        st.markdown("<div class='terminal-section'>HISTORICAL FUSION SCORE & DECISION TIMELINE (300 SESSIONS)</div>", unsafe_allow_html=True)
        fig_ftime = go.Figure()
        
        # Upper threshold filled band
        fig_ftime.add_hline(y=0.20, line_color="#00c076", line_dash="dash", annotation_text="BUY Threshold (+0.20)")
        fig_ftime.add_hline(y=-0.20, line_color="#ff4d4d", line_dash="dash", annotation_text="SELL Threshold (-0.20)")
        fig_ftime.add_hline(y=0.00, line_color="#475569", line_dash="dot")

        fig_ftime.add_trace(go.Scatter(x=df_fusion_comp["Date"], y=df_fusion_comp["Fusion_Score"], line=dict(color="#cbd5e1", width=1.3), name="Combined Score"))
        
        buys = df_fusion_comp[df_fusion_comp["Final_Decision"] == "BUY"]
        sells = df_fusion_comp[df_fusion_comp["Final_Decision"] == "SELL"]
        holds = df_fusion_comp[df_fusion_comp["Final_Decision"] == "HOLD"]

        fig_ftime.add_trace(go.Scatter(x=buys["Date"], y=buys["Fusion_Score"], mode="markers", name="BUY", marker=dict(color="#00c076", size=7)))
        fig_ftime.add_trace(go.Scatter(x=sells["Date"], y=sells["Fusion_Score"], mode="markers", name="SELL", marker=dict(color="#ff4d4d", size=7)))
        fig_ftime.add_trace(go.Scatter(x=holds["Date"], y=holds["Fusion_Score"], mode="markers", name="HOLD", marker=dict(color="#f59e0b", size=5, opacity=0.4)))

        apply_terminal_layout(fig_ftime, height=330)
        st.plotly_chart(fig_ftime, use_container_width=True)


# ==========================================
# 10. PAGE 6 — COMPANY COMPARISON
# ==========================================
elif nav_selection == "6. Company Comparison":
    st.markdown("""
    <div class="terminal-header">
        <div>
            <div class="terminal-title">Multi-Company Cross-Asset Comparison Matrix</div>
            <div class="terminal-subtitle">Comparative Behavioral Analytics Across All 10 Target Equities (No Ranking)</div>
        </div>
        <div class="badge-container">
            <span class="status-badge badge-offline">10 EQUITIES</span>
            <span class="status-badge badge-demo">HISTORICAL BENCHMARK</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.info("💡 **Analytical Methodology Note**: This screen illustrates cross-company behavioral heterogeneity under uniform multi-modal decision rules. Equities are evaluated objectively without subjective ranking.")

    if df_fusion_summary is not None:
        # Cross-Company Summary Matrix
        st.markdown("<div class='terminal-section'>1. CROSS-COMPANY DECISION DISTRIBUTION TABLE</div>", unsafe_allow_html=True)
        st.dataframe(
            df_fusion_summary[[
                "Ticker", "Company_Name", "BUY_Count", "BUY_Percentage",
                "HOLD_Count", "HOLD_Percentage", "SELL_Count", "SELL_Percentage",
                "Average_Fusion_Score", "Average_Heuristic_Confidence"
            ]].style.format({
                "BUY_Percentage": "{:.1f}%", "HOLD_Percentage": "{:.1f}%", "SELL_Percentage": "{:.1f}%",
                "Average_Fusion_Score": "{:+.4f}", "Average_Heuristic_Confidence": "{:.4f}"
            }),
            use_container_width=True
        )

        st.markdown("<div class='terminal-section'>2. COMPARATIVE VISUALIZATIONS</div>", unsafe_allow_html=True)
        cp1, cp2 = st.columns(2)

        with cp1:
            # Stacked Bar: Decision Distribution
            fig_stack = go.Figure()
            fig_stack.add_trace(go.Bar(x=df_fusion_summary["Ticker"], y=df_fusion_summary["BUY_Percentage"], name="BUY %", marker_color="#00c076"))
            fig_stack.add_trace(go.Bar(x=df_fusion_summary["Ticker"], y=df_fusion_summary["HOLD_Percentage"], name="HOLD %", marker_color="#f59e0b"))
            fig_stack.add_trace(go.Bar(x=df_fusion_summary["Ticker"], y=df_fusion_summary["SELL_Percentage"], name="SELL %", marker_color="#ff4d4d"))
            fig_stack.update_layout(barmode="stack")
            apply_terminal_layout(fig_stack, height=280, title="Decision Distribution by Company (% of Sessions)")
            st.plotly_chart(fig_stack, use_container_width=True)

        with cp2:
            # Average Combined Signal Score
            fig_sc = go.Figure()
            sc_colors = ["#00c076" if s >= 0 else "#ff4d4d" for s in df_fusion_summary["Average_Fusion_Score"]]
            fig_sc.add_trace(go.Bar(x=df_fusion_summary["Ticker"], y=df_fusion_summary["Average_Fusion_Score"], marker_color=sc_colors, name="Avg Score"))
            fig_sc.add_hline(y=0, line_color="#475569")
            apply_terminal_layout(fig_sc, height=280, title="Average Combined Signal Score Across Equities")
            st.plotly_chart(fig_sc, use_container_width=True)

        # Component Signal Breakdown Across Equities
        st.markdown("<div class='terminal-section'>3. AVERAGE COMPONENT SIGNALS BY COMPANY</div>", unsafe_allow_html=True)
        fig_sub = go.Figure()
        fig_sub.add_trace(go.Bar(x=df_fusion_summary["Ticker"], y=df_fusion_summary["Average_Classification_Signal"], name="Classification", marker_color="#38bdf8"))
        fig_sub.add_trace(go.Bar(x=df_fusion_summary["Ticker"], y=df_fusion_summary["Average_RL_Signal"], name="RL Strategy", marker_color="#818cf8"))
        fig_sub.add_trace(go.Bar(x=df_fusion_summary["Ticker"], y=df_fusion_summary["Average_Market_Regime_Signal"], name="Market Condition", marker_color="#f59e0b"))
        fig_sub.add_trace(go.Bar(x=df_fusion_summary["Ticker"], y=df_fusion_summary["Average_NLP_Signal"], name="News Signal", marker_color="#10b981"))
        fig_sub.update_layout(barmode="group")
        apply_terminal_layout(fig_sub, height=300, title="Directional Component Signal Averages [-1, +1]")
        st.plotly_chart(fig_sub, use_container_width=True)


# ==========================================
# 11. PAGE 7 — PROJECT INFORMATION
# ==========================================
# 11. PAGE 7 — SYSTEM OVERVIEW
# ==========================================
elif nav_selection == "7. System Overview":
    st.markdown("""
    <div class="terminal-header">
        <div>
            <div class="terminal-title">System Architecture & Technical Specifications</div>
            <div class="terminal-subtitle">AI-Based Intelligent Trading System • Platform Architecture & Analytical Methodology Reference</div>
        </div>
        <div class="badge-container">
            <span class="status-badge badge-offline">PLATFORM ARCHITECTURE</span>
            <span class="status-badge badge-demo">QUANTITATIVE FRAMEWORK</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # System Disclaimer Box
    st.markdown("""
    <div class="terminal-alert">
        ⚠️ <b>SYSTEM TRADING & RESEARCH DISCLAIMER</b><br>
        This platform is a quantitative trading intelligence and market research system. BUY, HOLD, and SELL outputs represent model-generated algorithmic consensus signals and do <b>not constitute individualized financial or investment advice</b>.<br><br>
        • Historical model evaluations and backtests reflect algorithm behavior over historical test periods and do not guarantee future performance.<br>
        • Demonstration NLP records illustrate pipeline taxonomy, sentiment scoring, and temporal decay scoring.<br>
        • Signal fusion weights (35/30/15/20) and decision thresholds (±0.20) represent deterministic architectural parameters without lookahead optimization.<br>
        • Signal confidence reflects multi-modal consensus magnitude rather than calibrated statistical probabilities.
    </div>
    """, unsafe_allow_html=True)

    # 1. System Architecture & Pipeline Flowchart
    st.markdown("<div class='terminal-section'>1. SYSTEM ARCHITECTURE & PIPELINE FLOWCHART</div>", unsafe_allow_html=True)
    st.code("""
    Market Data
        ↓
    Data Preparation & Feature Engineering
        ↓
    Predictive Analytics
        ↓
    Market Regime Detection
        ↓
    Reinforcement Learning
        ↓
    Financial News Intelligence
        ↓
    Signal Fusion
        ↓
    BUY / HOLD / SELL
        ↓
    Analytics & Visualization
    """, language="text")

    # 2. Analytical Subsystems Overview
    st.markdown("<div class='terminal-section'>2. CORE ANALYTICAL SUBSYSTEMS</div>", unsafe_allow_html=True)
    
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
        
        **Market Regime Detection**:
        - **Unsupervised Clustering**: K-Means ($K=3$) and DBSCAN algorithms partition historical market conditions into distinct behavioral regimes:
          - *Cluster 0 — Bearish Drawdown*: Negative returns, elevated volatility, wide intraday spreads.
          - *Cluster 1 — Calm Equilibrium*: Low volatility, balanced candlestick shadows, stable mean return.
          - *Cluster 2 — Bullish Expansion*: Positive returns, strong trading volume, sustained upside momentum.
        """)
        
    with sec_col2:
        st.markdown(r"""
        **Reinforcement Learning**:
        - Tabular Q-Learning that maps market states to BUY, HOLD, and SELL actions using forward-return-based rewards with transaction costs.
        
        **Financial News Intelligence**:
        - **Loughran-McDonald Financial Lexicon**: Domain-specific sentiment analysis tailored for financial terminology and negation handling.
        - **8-Stage Triage Engine**: Computes Headline Ingestion $\rightarrow$ Cleaning $\rightarrow$ Sentiment $\rightarrow$ Entity Relevance $\rightarrow$ Event Importance $\rightarrow$ Recency Decay $\rightarrow$ Triage Score $\rightarrow$ Daily News Signal.
        
        **Signal Fusion**:
        - **Multi-Modal Consensus**: Deterministic weighted aggregation across 4 independent analytical channels:
          $$\text{Fusion Score} = 0.35 \times \text{Direction} + 0.30 \times \text{RL Agent} + 0.15 \times \text{Regime} + 0.20 \times \text{News}$$
        - **Decision Boundaries**: BUY ($\ge +0.20$), HOLD ($-0.20$ to $+0.20$), SELL ($\le -0.20$).
        
        **Explainability**:
        - **Transparent Decision Rationale**: Every generated signal is decomposed into component contributions, heuristic confidence percentages, and natural-language narrative rationales.
        
        **Real-Time Data Layer**:
        - **Operational Live Telemetry**: Independent subsystem providing on-demand market snapshots, live Alpha Vantage news ingestion, Loughran-McDonald NLP enrichment, automatic deduplication, and full analytics pipeline regeneration with rate-limit and quota protection.
        """)

    # 3. Technology Stack & Platform Capabilities
    st.markdown("<div class='terminal-section'>3. TECHNOLOGY STACK & PLATFORM CAPABILITIES</div>", unsafe_allow_html=True)
    t_col1, t_col2 = st.columns(2)
    with t_col1:
        st.markdown("""
        **Technology Stack**:
        - **Python 3.13**: Core computational environment.
        - **Pandas & NumPy**: High-performance quantitative vector and time-series processing.
        - **Scikit-Learn**: Supervised regression, classification, regularized models, PCA, and clustering.
        - **Plotly**: High-density interactive dark financial charts and candlestick visualizations.
        - **Streamlit**: Reactive real-time analytics terminal framework.
        - **SQLite**: Local structured live storage for ingested news articles, NLP metadata, and audit logs.
        """)
    with t_col2:
        st.markdown("""
        **Methodological & Design Specifications**:
        - **Subsystem Isolation**: Historical model evaluations and real-time live telemetry operate independently without cross-layer data leakage.
        - **Deterministic Consensus**: Multi-modal fusion weights (35/30/15/20) and decision thresholds (±0.20) are architectural parameters designed for explainability.
        - **Rate-Limit & Quota Safety**: Ingestion pipelines automatically respect API rate limits and free-tier daily quotas with graceful degradation.
        - **No Execution Friction**: Backtests evaluate predictive signal accuracy and do not simulate slippage or exchange fees.
        """)

# ==========================================
# 8. PAGE 8 — REAL-TIME DATA
# ==========================================
elif nav_selection == "8. Real-Time Data":
    # Terminal Header
    st.markdown("""
    <div class="terminal-header">
        <div>
            <div class="terminal-title">Real-Time Data & Analytics Terminal</div>
            <div class="terminal-subtitle">Independent Live & Real-World Market & Financial News Analytics Subsystem</div>
        </div>
        <div class="badge-container">
            <span class="status-badge" style="background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.4);">LIVE LAYER</span>
            <span class="status-badge badge-offline">DELAYED / SNAPSHOT</span>
            <span class="status-badge badge-ticker">10 EQUITIES</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

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
        stream_badge_html = '<span style="background:rgba(0,192,118,0.20); color:#00c076; border:1px solid #00c076; font-size:0.75rem; font-weight:800; padding:0.25rem 0.65rem; border-radius:4px; font-family:monospace; letter-spacing:0.5px;">● LIVE STREAMING</span>'
    elif stream_status_val == "RECONNECTING":
        stream_badge_html = f'<span style="background:rgba(245,158,11,0.20); color:#f59e0b; border:1px solid #f59e0b; font-size:0.75rem; font-weight:800; padding:0.25rem 0.65rem; border-radius:4px; font-family:monospace; letter-spacing:0.5px;">● RECONNECTING (Attempt {active_retries})</span>'
    elif stream_status_val in ["CONNECTING", "AUTHENTICATING", "SUBSCRIBING"]:
        stream_badge_html = f'<span style="background:rgba(56,189,248,0.20); color:#38bdf8; border:1px solid #38bdf8; font-size:0.75rem; font-weight:800; padding:0.25rem 0.65rem; border-radius:4px; font-family:monospace; letter-spacing:0.5px;">● {stream_status_val}</span>'
    elif is_stale:
        age_label = f"{hb_age:.0f}s AGO" if hb_age is not None else "STALE"
        stream_badge_html = f'<span style="background:rgba(239,68,68,0.20); color:#ef4444; border:1px solid #ef4444; font-size:0.75rem; font-weight:800; padding:0.25rem 0.65rem; border-radius:4px; font-family:monospace; letter-spacing:0.5px;">● OFFLINE (HEARTBEAT {age_label})</span>'
    else:
        stream_badge_html = '<span style="background:rgba(148,163,184,0.15); color:#94a3b8; border:1px solid #64748b; font-size:0.75rem; font-weight:800; padding:0.25rem 0.65rem; border-radius:4px; font-family:monospace; letter-spacing:0.5px;">● STANDBY / OFFLINE</span>'

    st.markdown(f"""
    <div style="background:#131824; border:1px solid #1e2638; border-radius:6px; padding:0.9rem 1.1rem; margin-bottom:1.0rem;">
        <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #1e2638; padding-bottom:0.5rem; margin-bottom:0.8rem;">
            <div>
                <span style="font-size:0.95rem; font-weight:800; color:#f8fafc; letter-spacing:0.4px;">
                    📡 ALPACA REAL-TIME FINANCIAL NEWS WEBSOCKET STREAM
                </span>
                <span style="font-size:0.72rem; color:#64748b; font-family:monospace; margin-left:0.6rem;">
                    <code>wss://stream.data.alpaca.markets/v1beta1/news</code>
                </span>
            </div>
            <div>
                {stream_badge_html}
            </div>
        </div>
    """, unsafe_allow_html=True)

    stream_col1, stream_col2, stream_col3 = st.columns([1.2, 1.2, 2.0])
    with stream_col1:
        if is_live:
            if st.button("⏹ STOP STREAM LISTENER", key="btn_stop_stream", use_container_width=True):
                stop_stream_listener()
                load_latest_real_articles.clear()
                st.rerun()
        else:
            if st.button("▶ START STREAM LISTENER", key="btn_start_stream", use_container_width=True):
                start_stream_listener()
                st.rerun()

    with stream_col2:
        if st.button("↻ REFRESH TELEMETRY & FEED", key="btn_refresh_stream_view", use_container_width=True):
            load_latest_real_articles.clear()
            st.rerun()

    with stream_col3:
        st.markdown("""
        <div style="font-size:0.72rem; color:#94a3b8; padding-top:0.35rem; font-family:monospace;">
            <b>Subscribed Universe</b>: 10 Equities (AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, JPM, V, JNJ)<br>
            <b>Auto-Normalization</b>: FB → META Aliasing • Loughran-McDonald NLP Pipeline
        </div>
        """, unsafe_allow_html=True)

    rec_cnt = stream_info.get("articles_received_count", 0)
    sto_cnt = stream_info.get("articles_stored_count", 0)
    dup_cnt = stream_info.get("articles_duplicate_count", 0)
    hb_ts = stream_info.get("last_heartbeat_utc", "N/A")

    if active_retries > 0:
        rec_status_html = f"<span style='color:#f59e0b; font-weight:700;'>Attempt {active_retries}</span> <span style='color:#64748b; font-size:0.70rem;'>(Total: {total_reconnects})</span>"
    else:
        rec_status_html = f"<span style='color:#00c076; font-weight:700;'>0 Active</span> <span style='color:#64748b; font-size:0.70rem;'>(Total: {total_reconnects})</span>"

    if hb_age is not None:
        hb_sub = f"{hb_age:.0f}s ago"
    else:
        hb_sub = "No Heartbeat"

    st.markdown(f"""
    <div style="display:grid; grid-template-columns: repeat(5, 1fr); gap:0.6rem; background:#0e131d; border:1px solid #1a2233; border-radius:4px; padding:0.6rem 0.8rem; margin-top:0.6rem; font-size:0.78rem;">
        <div>
            <span style="color:#64748b; font-size:0.68rem; font-weight:700; text-transform:uppercase;">Tracked Articles Recv:</span><br>
            <span style="color:#38bdf8; font-family:monospace; font-weight:700; font-size:0.95rem;">{rec_cnt}</span>
        </div>
        <div>
            <span style="color:#64748b; font-size:0.68rem; font-weight:700; text-transform:uppercase;">Stored in SQLite:</span><br>
            <span style="color:#00c076; font-family:monospace; font-weight:700; font-size:0.95rem;">{sto_cnt}</span>
        </div>
        <div>
            <span style="color:#64748b; font-size:0.68rem; font-weight:700; text-transform:uppercase;">Duplicates Filtered:</span><br>
            <span style="color:#f59e0b; font-family:monospace; font-weight:700; font-size:0.95rem;">{dup_cnt}</span>
        </div>
        <div>
            <span style="color:#64748b; font-size:0.68rem; font-weight:700; text-transform:uppercase;">Reconnection State:</span><br>
            <span style="font-family:monospace; font-size:0.85rem;">{rec_status_html}</span>
        </div>
        <div>
            <span style="color:#64748b; font-size:0.68rem; font-weight:700; text-transform:uppercase;">Last Stream Heartbeat:</span><br>
            <span style="color:#f1f5f9; font-family:monospace; font-weight:600; font-size:0.75rem;">{hb_ts or 'N/A'}</span><br>
            <span style="color:#64748b; font-size:0.68rem; font-family:monospace;">({hb_sub})</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

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
        b_col = "#00c076" if l_sent_lbl == "Positive" else ("#ff4d4d" if l_sent_lbl == "Negative" else "#f59e0b")

        st.markdown(f"""
        <div style="margin-top:0.6rem; padding:0.6rem 0.85rem; background:rgba(16, 185, 129, 0.08); border-left:3px solid #10b981; border-radius:3px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:0.70rem; font-weight:700; color:#34d399; text-transform:uppercase; letter-spacing:0.5px;">
                    ⚡ LAST RECEIVED STREAMING ARTICLE SPOTLIGHT
                </span>
                <span style="color:{b_col}; font-weight:700; font-size:0.72rem; border:1px solid {b_col}; border-radius:3px; padding:0.1rem 0.4rem;">
                    {l_sent_lbl} ({l_sent_val:+.2f})
                </span>
            </div>
            <div style="font-size:0.86rem; font-weight:700; color:#f1f5f9; margin-top:0.25rem;">
                [{l_ticker}] {l_head}
            </div>
            <div style="display:flex; gap:0.8rem; font-size:0.70rem; color:#94a3b8; margin-top:0.25rem; font-family:monospace;">
                <span>Published: {l_pub}</span>
                <span>•</span>
                <span>Source: {l_src}</span>
                <span>•</span>
                <span>Event: {l_ev}</span>
                <span>•</span>
                <span>Triage: {l_tri:.3f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="margin-top:0.5rem; padding:0.45rem 0.75rem; background:rgba(56, 189, 248, 0.05); border-left:3px solid #38bdf8; border-radius:3px; font-size:0.74rem; color:#94a3b8;">
            ℹ️ <b>Stream Status</b>: WebSocket listener active or on standby. Incoming headlines for the 10 subscribed equities are automatically normalized, Loughran-McDonald NLP scored, deduplicated, and displayed in real time.
        </div>
        """, unsafe_allow_html=True)

    if stream_info.get("error_message") and stream_status_val in ["ERROR", "RECONNECTING"]:
        st.markdown(f"""
        <div style="margin-top:0.4rem; padding:0.45rem 0.75rem; background:rgba(245, 158, 11, 0.08); border-left:3px solid #f59e0b; border-radius:3px; font-size:0.74rem; color:#fde68a;">
            ⚠️ Stream Notice: {stream_info['error_message']}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Controlled Manual Refresh Interface (Sections 1, 2, 3, 6, 7, 9, 11)
    if "last_refresh_timestamp" not in st.session_state:
        last_info = get_last_refresh_info()
        st.session_state["last_refresh_timestamp"] = last_info.get("timestamp_utc", "Never")
        st.session_state["refresh_status"] = last_info.get("status", "READY")
        st.session_state["refresh_summary"] = last_info.get("summary", "")
        st.session_state["refresh_error"] = ""

    if "refresh_banner" in st.session_state and st.session_state["refresh_banner"]:
        b_type, b_text = st.session_state["refresh_banner"]
        if b_type == "SUCCESS":
            st.success(b_text)
        elif b_type == "PARTIAL":
            st.warning(b_text)
        else:
            st.error(b_text)
        del st.session_state["refresh_banner"]

    # Refresh Control Card
    st.markdown("""
    <div style="background:#131824; border:1px solid #1e2638; border-radius:6px; padding:0.9rem 1.1rem; margin-bottom:1.0rem;">
        <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #1e2638; padding-bottom:0.5rem; margin-bottom:0.8rem;">
            <div style="font-size:0.92rem; font-weight:800; color:#f8fafc; letter-spacing:0.3px;">
                🔄 CONTROLLED DATA REFRESH CONTROL
            </div>
            <div style="font-size:0.72rem; color:#94a3b8; font-family:monospace;">
                MANUAL TRIGGER ONLY • SAFE ON-DEMAND EXECUTION
            </div>
        </div>
    """, unsafe_allow_html=True)

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

    # Status & Metadata Box
    curr_status = st.session_state.get("refresh_status", "READY")
    status_color = "#00c076" if curr_status == "SUCCESS" else ("#f59e0b" if curr_status in ["PARTIAL", "READY"] else "#ff4d4d")
    last_ts = st.session_state.get("last_refresh_timestamp", "Never")
    
    st.markdown(f"""
    <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:0.6rem; background:#0e131d; border:1px solid #1a2233; border-radius:4px; padding:0.6rem 0.8rem; margin-top:0.6rem; font-size:0.78rem;">
        <div>
            <span style="color:#64748b; font-size:0.68rem; font-weight:700; text-transform:uppercase;">Last Successful Refresh:</span><br>
            <span style="color:#f1f5f9; font-family:monospace; font-weight:600;">{last_ts}</span>
        </div>
        <div>
            <span style="color:#64748b; font-size:0.68rem; font-weight:700; text-transform:uppercase;">Refresh Status:</span><br>
            <span style="color:{status_color}; font-weight:700; text-transform:uppercase;">● {curr_status}</span>
        </div>
        <div>
            <span style="color:#64748b; font-size:0.68rem; font-weight:700; text-transform:uppercase;">Market Source:</span><br>
            <span style="color:#e2e8f0;">Yahoo Finance (yfinance) — Delayed / Snapshot</span>
        </div>
        <div>
            <span style="color:#64748b; font-size:0.68rem; font-weight:700; text-transform:uppercase;">News & NLP:</span><br>
            <span style="color:#e2e8f0;">Alpha Vantage News • Internal Loughran-McDonald</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.get("refresh_summary"):
        st.markdown(f"""
        <div style="margin-top:0.5rem; padding:0.45rem 0.75rem; background:rgba(56, 189, 248, 0.08); border-left:3px solid #38bdf8; border-radius:3px; font-size:0.75rem; color:#bae6fd;">
            <b>Telemetry</b>: {st.session_state['refresh_summary']}
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.get("refresh_error"):
        st.markdown(f"""
        <div style="margin-top:0.4rem; padding:0.45rem 0.75rem; background:rgba(245, 158, 11, 0.08); border-left:3px solid #f59e0b; border-radius:3px; font-size:0.75rem; color:#fde68a;">
            ⚠️ {st.session_state['refresh_error']}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Handle Refresh Execution
    if btn_refresh:
        with st.spinner("Executing controlled real data refresh & rebuilding analytics..."):
            res = execute_real_data_refresh(scope=refresh_scope)
            st.session_state["last_refresh_timestamp"] = res["timestamp_utc"]
            st.session_state["refresh_status"] = res["status"]
            
            m_part = f"Market: {res['market_refreshed_count']}/{res['total_tickers']}" if res['market_requested'] else "Market: Skipped"
            n_part = f"News: {res['news_refreshed_count']}/{res['total_tickers']}" if res['news_requested'] else "News: Skipped"
            art_part = f"New: {res['articles_added']} • Dups: {res['articles_duplicate']}"
            ana_part = "Analytics rebuilt: YES" if res['analytics_rebuilt'] else "Analytics rebuilt: NO"
            
            summary = f"{m_part} | {n_part} | {art_part} | {ana_part}"
            st.session_state["refresh_summary"] = summary
            
            err_msg = res.get("error_message", "")
            if res.get("quota_limited"):
                err_msg = "News refresh unavailable: Alpha Vantage API quota/rate limit reached. " + err_msg
            st.session_state["refresh_error"] = err_msg
            
            # Clear Streamlit cache to load fresh files
            load_real_market_summary.clear()
            load_real_news_summary.clear()
            load_real_news_daily_summary.clear()
            load_real_data_coverage.clear()
            load_real_market_extension.clear()
            load_latest_real_articles.clear()

        if res["status"] == "SUCCESS":
            if res.get("news_requested") and res.get("articles_added", 0) == 0:
                st.session_state["refresh_banner"] = ("SUCCESS", f"✅ Refresh Complete (No new unique articles): {summary}")
            else:
                st.session_state["refresh_banner"] = ("SUCCESS", f"✅ Refresh Complete: {summary}")
        elif res["status"] == "PARTIAL":
            st.session_state["refresh_banner"] = ("PARTIAL", f"⚠️ Refresh Partial: {summary}. {st.session_state['refresh_error']}")
        else:
            st.session_state["refresh_banner"] = ("FAILED", f"❌ Refresh Failed: {st.session_state['refresh_error'] or 'Unknown error'}")

        st.rerun()

    # Section G: Data Status Banner
    st.markdown("""
    <div style="background:#101522; border:1px solid #1e2638; border-radius:6px; padding:0.9rem 1.1rem; margin-bottom:1.0rem;">
        <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #1e2638; padding-bottom:0.5rem; margin-bottom:0.7rem;">
            <span style="font-weight:800; font-size:0.95rem; color:#f1f5f9; letter-spacing:0.5px;">⚡ REAL DATA STATUS DISCLOSURE</span>
            <span style="font-size:0.75rem; color:#38bdf8; font-family:monospace;">INDEPENDENT LIVE ANALYTICS SUBSYSTEM</span>
        </div>
        <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:0.8rem; font-size:0.82rem;">
            <div>
                <div style="color:#64748b; font-size:0.70rem; text-transform:uppercase; font-weight:700;">Market Feed</div>
                <div style="color:#f8fafc; font-weight:600; margin-top:2px;">Yahoo Finance (yfinance)</div>
                <div style="color:#f59e0b; font-size:0.72rem; margin-top:2px;">● Status: Delayed / Snapshot</div>
            </div>
            <div>
                <div style="color:#64748b; font-size:0.70rem; text-transform:uppercase; font-weight:700;">Financial News</div>
                <div style="color:#f8fafc; font-weight:600; margin-top:2px;">Alpha Vantage News</div>
                <div style="color:#38bdf8; font-size:0.72rem; margin-top:2px;">● Status: Historical + Latest Retrieved</div>
            </div>
            <div>
                <div style="color:#64748b; font-size:0.70rem; text-transform:uppercase; font-weight:700;">NLP Scoring</div>
                <div style="color:#f8fafc; font-weight:600; margin-top:2px;">Internal Loughran-McDonald</div>
                <div style="color:#10b981; font-size:0.72rem; margin-top:2px;">● Domain-specific Dictionary</div>
            </div>
            <div>
                <div style="color:#64748b; font-size:0.70rem; text-transform:uppercase; font-weight:700;">Historical News Window</div>
                <div style="color:#f8fafc; font-weight:600; margin-top:2px;">January 2025</div>
                <div style="color:#f59e0b; font-size:0.72rem; margin-top:2px;">● Partial / Capped for 4 Equities</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Section H: Real Data Subsystem Isolation Notice
    st.markdown("""
    <div class="terminal-alert" style="border-left-color: #38bdf8; background: rgba(56, 189, 248, 0.08); color: #bae6fd;">
        ℹ️ <b>REAL-DATA SUBSYSTEM ISOLATION DISCLOSURE</b><br>
        This section uses independently collected real-world market and financial-news data. It operates as an independent telemetry layer separate from the historical model evaluation dataset.<br>
        All technical indicators, moving averages, and news sentiment scores are purely analytical — they do <b>NOT</b> feed into the historical model signals or Decision Fusion engine, and do <b>NOT</b> constitute guarantees of trading performance.
    </div>
    """, unsafe_allow_html=True)

    # Load Real-Data Analytics Artifacts (with error handling)
    df_rmarket = load_real_market_summary()
    df_rnews = load_real_news_summary()
    df_rdaily = load_real_news_daily_summary()
    df_rcov = load_real_data_coverage()
    df_rext = load_real_market_extension()

    if df_rmarket is None or df_rmarket.empty:
        st.warning("⚠️ Real market analytics summary is currently unavailable (`data/live/analytics/real_market_summary.csv` not found).")
    if df_rnews is None or df_rnews.empty:
        st.warning("⚠️ Real news analytics summary is currently unavailable (`data/live/analytics/real_news_summary.csv` not found).")
    if df_rcov is None or df_rcov.empty:
        st.warning("⚠️ Real data coverage audit is currently unavailable (`data/live/analytics/real_data_coverage.csv` not found).")

    # ==========================================
    # SECTION A: REAL MARKET SNAPSHOT
    # ==========================================
    st.markdown("<div class='terminal-section'>A. REAL MARKET SNAPSHOT (10 EQUITIES)</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.75rem; color:#94a3b8; margin-bottom:0.6rem;">
        <b>Source</b>: Yahoo Finance / yfinance &nbsp;•&nbsp; <b>Status</b>: Delayed / Snapshot &nbsp;•&nbsp; <i>Note: Analytical technical estimators from daily snapshot feeds; not exchange-grade tick-by-tick real-time data.</i>
    </div>
    """, unsafe_allow_html=True)

    if df_rmarket is not None and not df_rmarket.empty:
        # Table display
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
            "Company", "Ticker", "Latest Price", "Daily Return", "SMA 5", "SMA 10",
            "EMA 12", "EMA 26", "5D Volatility", "H-L Range", "Volume", "Trend", "Status"
        ]
        
        st.dataframe(m_display, use_container_width=True, hide_index=True)

        # Plotly Visuals for Snapshot
        snap_col1, snap_col2 = st.columns(2)
        with snap_col1:
            ret_vals = df_rmarket["Daily_Return"] * 100
            ret_colors = ["#00c076" if r >= 0 else "#ff4d4d" for r in ret_vals]
            fig_ret = go.Figure(go.Bar(
                x=df_rmarket["Ticker"],
                y=ret_vals,
                marker_color=ret_colors,
                text=[f"{r:+.2f}%" for r in ret_vals],
                textposition="auto"
            ))
            fig_ret.add_hline(y=0, line_color="#475569", line_dash="dash", line_width=1)
            fig_ret = apply_terminal_layout(fig_ret, height=280, title="Daily Price Return (%) by Company (Delayed Snapshot)")
            st.plotly_chart(fig_ret, use_container_width=True)

        with snap_col2:
            vol_vals = df_rmarket["Volatility_5"] * 100
            fig_vol = go.Figure(go.Bar(
                x=df_rmarket["Ticker"],
                y=vol_vals,
                marker_color="#818cf8",
                text=[f"{v:.2f}%" for v in vol_vals],
                textposition="auto"
            ))
            fig_vol = apply_terminal_layout(fig_vol, height=280, title="5-Day Rolling Volatility (%) by Company")
            st.plotly_chart(fig_vol, use_container_width=True)

    # ==========================================
    # SECTION B: COMPANY SELECTOR & DEEP DIVE
    # ==========================================
    st.markdown("<div class='terminal-section'>B. COMPANY DEEP-DIVE ANALYTICS</div>", unsafe_allow_html=True)
    real_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "JNJ"]
    real_selected = st.selectbox(
        "Select Company for Real-Data Deep Dive:",
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
            
            trend_class = "color-bullish" if trend == "Bullish" else ("color-bearish" if trend == "Bearish" else "color-neutral")
            ret_class = "color-bullish" if ret >= 0 else "color-bearish"
            
            # KPI Cards Grid
            st.markdown(f"""
            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-label">LATEST PRICE (SNAPSHOT)</div>
                    <div class="kpi-value">${price:,.2f}</div>
                    <div class="kpi-sub {ret_class}">{change:+.2f} ({ret*100:+.2f}%)</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">MOVING AVERAGES</div>
                    <div class="kpi-value">${sma5:,.2f}</div>
                    <div class="kpi-sub">SMA 5 vs SMA 10: ${sma10:,.2f}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">EXPONENTIAL MA</div>
                    <div class="kpi-value">${ema12:,.2f}</div>
                    <div class="kpi-sub">EMA 12 vs EMA 26: ${ema26:,.2f}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">5-DAY VOLATILITY</div>
                    <div class="kpi-value">{vol5*100:.2f}%</div>
                    <div class="kpi-sub">Annualized: ~{vol5*np.sqrt(252)*100:.1f}%</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">LATEST VOLUME</div>
                    <div class="kpi-value">{vol/1e6:.2f}M</div>
                    <div class="kpi-sub">20D Avg: {avg_vol/1e6:.2f}M</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">TECHNICAL TREND</div>
                    <div class="kpi-value {trend_class}">{trend}</div>
                    <div class="kpi-sub">As of {as_of}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Plotly Price & MA Chart for Selected Company (from Extension Data)
        if df_rext is not None and not df_rext.empty:
            t_ext = df_rext[df_rext["Ticker"] == real_selected].sort_values("Date").tail(90).copy()
            if not t_ext.empty:
                t_ext["SMA_5"] = t_ext["Close"].rolling(5, min_periods=2).mean()
                t_ext["SMA_10"] = t_ext["Close"].rolling(10, min_periods=3).mean()
                
                fig_ext = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.75, 0.25])
                
                fig_ext.add_trace(go.Candlestick(
                    x=t_ext["Date"],
                    open=t_ext["Open"],
                    high=t_ext["High"],
                    low=t_ext["Low"],
                    close=t_ext["Close"],
                    name="Price (OHLC)",
                    increasing_line_color="#00c076",
                    decreasing_line_color="#ff4d4d"
                ), row=1, col=1)
                
                fig_ext.add_trace(go.Scatter(
                    x=t_ext["Date"], y=t_ext["SMA_5"], name="SMA 5",
                    line=dict(color="#38bdf8", width=1.5)
                ), row=1, col=1)
                
                fig_ext.add_trace(go.Scatter(
                    x=t_ext["Date"], y=t_ext["SMA_10"], name="SMA 10",
                    line=dict(color="#f59e0b", width=1.5)
                ), row=1, col=1)
                
                vol_colors = ["#00c076" if c >= o else "#ff4d4d" for c, o in zip(t_ext["Close"], t_ext["Open"])]
                fig_ext.add_trace(go.Bar(
                    x=t_ext["Date"], y=t_ext["Volume"], name="Volume",
                    marker_color=vol_colors, opacity=0.6
                ), row=2, col=1)
                
                fig_ext = apply_terminal_layout(
                    fig_ext, height=420,
                    title=f"{real_selected} — Real Historical Extension Series (Recent 90 Sessions with SMA 5 / 10)"
                )
                fig_ext.update_xaxes(rangeslider_visible=False)
                st.plotly_chart(fig_ext, use_container_width=True)

    # ==========================================
    # SECTION C: REAL NEWS ANALYTICS
    # ==========================================
    st.markdown("<div class='terminal-section'>C. REAL FINANCIAL NEWS NLP ANALYTICS</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background: rgba(129, 140, 248, 0.08); border-left: 4px solid #818cf8; padding: 0.5rem 0.8rem; border-radius: 4px; font-size: 0.80rem; color: #c7d2fe; margin-bottom: 0.8rem;">
        <b>METHODOLOGY NOTICE</b>: All sentiment scores, entity relevance, and triage values are computed using the project's <b>INTERNAL Loughran-McDonald (LM) dictionary</b> and domain event rules. Alpha Vantage's provider sentiment is discarded.
    </div>
    """, unsafe_allow_html=True)

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
            
            dom_class = "color-bullish" if n_dom == "Positive" else ("color-bearish" if n_dom == "Negative" else "color-neutral")
            
            # NLP KPI Grid for active company
            st.markdown(f"""
            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-label">TOTAL ARTICLES</div>
                    <div class="kpi-value">{n_tot}</div>
                    <div class="kpi-sub">{real_selected} Real Records</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">SENTIMENT SPLIT</div>
                    <div class="kpi-value">{n_pos}P / {n_neu}N / {n_neg}B</div>
                    <div class="kpi-sub">Pos / Neu / Neg</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">AVG LM SENTIMENT</div>
                    <div class="kpi-value">{n_sent:+.4f}</div>
                    <div class="kpi-sub">Bound: [-1.0, +1.0]</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">AVG RELEVANCE / IMP</div>
                    <div class="kpi-value">{n_rel:.2f} / {n_imp:.2f}</div>
                    <div class="kpi-sub">Target Relevance & Event Imp</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">HIGH IMPACT NEWS</div>
                    <div class="kpi-value">{n_hi}</div>
                    <div class="kpi-sub">Triage ≥ 0.50 or Imp ≥ 0.70</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">DOMINANT SENTIMENT</div>
                    <div class="kpi-value {dom_class}">{n_dom}</div>
                    <div class="kpi-sub">Avg Triage: {n_tri:.4f}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # NLP Charts Grid
        nlp_c1, nlp_c2, nlp_c3 = st.columns([1, 1.2, 1.2])
        
        with nlp_c1:
            # Donut chart for selected company
            if not cn_sub.empty:
                pie_fig = go.Figure(data=[go.Pie(
                    labels=["Positive", "Neutral", "Negative"],
                    values=[n_pos, n_neu, n_neg],
                    hole=0.55,
                    marker=dict(colors=["#00c076", "#94a3b8", "#ff4d4d"]),
                    textinfo="label+value"
                )])
                pie_fig = apply_terminal_layout(pie_fig, height=270, title=f"{real_selected} — Sentiment Distribution")
                st.plotly_chart(pie_fig, use_container_width=True)

        with nlp_c2:
            # Average Sentiment Comparison across 10 Companies
            sent_colors = ["#00c076" if s >= 0 else "#ff4d4d" for s in df_rnews["Average_Sentiment"]]
            fig_all_sent = go.Figure(go.Bar(
                x=df_rnews["Ticker"],
                y=df_rnews["Average_Sentiment"],
                marker_color=sent_colors,
                text=[f"{s:+.3f}" for s in df_rnews["Average_Sentiment"]],
                textposition="auto"
            ))
            fig_all_sent.add_hline(y=0, line_color="#475569", line_dash="dash")
            fig_all_sent = apply_terminal_layout(fig_all_sent, height=270, title="Average LM Sentiment by Company")
            st.plotly_chart(fig_all_sent, use_container_width=True)

        with nlp_c3:
            # Article Counts across Companies
            fig_cnt = go.Figure()
            fig_cnt.add_trace(go.Bar(
                x=df_rnews["Ticker"], y=df_rnews["Article_Count"],
                name="Total Articles", marker_color="#38bdf8"
            ))
            fig_cnt.add_trace(go.Bar(
                x=df_rnews["Ticker"], y=df_rnews["High_Impact_Article_Count"],
                name="High Impact", marker_color="#f59e0b"
            ))
            fig_cnt = apply_terminal_layout(fig_cnt, height=270, title="News Volume & High-Impact Counts")
            fig_cnt.update_layout(barmode="group")
            st.plotly_chart(fig_cnt, use_container_width=True)

    # ==========================================
    # SECTION D: DAILY NEWS SIGNALS
    # ==========================================
    st.markdown("<div class='terminal-section'>D. DAILY NEWS SIGNALS & SENTIMENT TIME-SERIES</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.75rem; color:#94a3b8; margin-bottom:0.6rem;">
        <b>Data Source</b>: <code>data/live/analytics/real_news_daily_summary.csv</code> &nbsp;•&nbsp; <i>Notice: Analytical news signals represent triage-weighted daily sentiment estimators and do <b>NOT</b> constitute guaranteed trading signals.</i>
    </div>
    """, unsafe_allow_html=True)

    if df_rdaily is not None and not df_rdaily.empty:
        t_daily = df_rdaily[df_rdaily["Ticker"] == real_selected].sort_values("Date").copy()
        if not t_daily.empty:
            sig_colors = ["#00c076" if s >= 0 else "#ff4d4d" for s in t_daily["News_Signal"]]
            
            fig_dsig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10, row_heights=[0.70, 0.30])
            
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
                line=dict(color="#38bdf8", width=1.5, dash="dot")
            ), row=1, col=1)
            
            fig_dsig.add_trace(go.Bar(
                x=t_daily["Date"],
                y=t_daily["News_Count"],
                name="Daily News Count",
                marker_color="#818cf8"
            ), row=2, col=1)
            
            fig_dsig = apply_terminal_layout(
                fig_dsig, height=360,
                title=f"{real_selected} — Daily News Signal & Article Volume Time-Series"
            )
            st.plotly_chart(fig_dsig, use_container_width=True)

            # Expandable details table
            with st.expander(f"View {real_selected} Daily News Signal Records ({len(t_daily)} Sessions)"):
                st.dataframe(
                    t_daily[[
                        "Date", "News_Count", "Average_Sentiment", "Triage_Weighted_Sentiment",
                        "Average_Triage_Score", "High_Impact_News_Count", "Dominant_Sentiment", "News_Signal"
                    ]],
                    use_container_width=True, hide_index=True
                )
        else:
            st.info(f"No daily news signal records found for {real_selected}.")

    # ==========================================
    # SECTION E: REAL NEWS COVERAGE & AUDIT
    # ==========================================
    st.markdown("<div class='terminal-section'>E. REAL NEWS COVERAGE & PROVIDER QUOTA AUDIT</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="terminal-alert" style="border-left-color: #f59e0b; background: rgba(245, 158, 11, 0.08); color: #fde68a;">
        ⚠️ <b>DATA COVERAGE DISCLOSURE</b><br>
        Historical coverage is incomplete for some companies because of the Alpha Vantage free-tier request limit (25 calls/day).
        This is an <b>external API quota limitation</b>, NOT a software defect or pipeline failure. Partial coverage is acknowledged openly and not represented as exhaustive.
    </div>
    """, unsafe_allow_html=True)

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
    # SECTION F: LATEST REAL AAPL NEWS
    # ==========================================
    st.markdown("<div class='terminal-section'>F. LATEST REAL FINANCIAL NEWS FEED (AAPL FOCUS)</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.75rem; color:#94a3b8; margin-bottom:0.8rem;">
        <b>Source</b>: Real historical news (Alpha Vantage) and continuous live WebSocket stream (Alpaca) &nbsp;•&nbsp; <i>Enriched with internal Loughran-McDonald NLP scoring, event taxonomy, and deduplication.</i>
    </div>
    """, unsafe_allow_html=True)

    feed_options = ["AAPL (Benchmark Asset)", f"{real_selected} (Selected Asset)"] if real_selected != "AAPL" else ["AAPL (Selected Asset)"]
    feed_ticker = st.radio(
        "Select News Feed Focus:",
        options=feed_options,
        horizontal=True
    )
    target_news_ticker = "AAPL" if "AAPL" in feed_ticker else real_selected
    
    articles_df = load_latest_real_articles(target_news_ticker, limit=15)
    
    if articles_df is not None and not articles_df.empty:
        st.markdown(f"<div style='font-size:0.80rem; color:#64748b; margin-bottom:0.5rem;'>Displaying {len(articles_df)} latest verified real articles for <b>{target_news_ticker}</b>:</div>", unsafe_allow_html=True)
        
        for _, art in articles_df.iterrows():
            sent_lbl = art["sentiment_label"]
            sent_val = art["sentiment_score"]
            badge_color = "#00c076" if sent_lbl == "Positive" else ("#ff4d4d" if sent_lbl == "Negative" else "#f59e0b")
            prov = str(art.get("provider", "Alpha Vantage"))
            is_streamed = "alpaca" in prov.lower()
            prov_label = "ALPACA STREAM" if is_streamed else "ALPHA VANTAGE"
            prov_color = "#38bdf8" if is_streamed else "#94a3b8"
            
            st.markdown(f"""
            <div style="background:#11151f; border:1px solid #1e2638; border-left:4px solid {badge_color}; border-radius:5px; padding:0.75rem 1.0rem; margin-bottom:0.3rem;">
                <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:0.3rem;">
                    <div style="font-size:0.92rem; font-weight:700; color:#f1f5f9; line-height:1.35;">
                        {art['headline']}
                    </div>
                    <span style="background:rgba(255,255,255,0.05); color:{badge_color}; border:1px solid {badge_color}; font-size:0.68rem; font-weight:700; padding:0.15rem 0.45rem; border-radius:3px; text-transform:uppercase; margin-left:0.8rem; white-space:nowrap;">
                        {sent_lbl} ({sent_val:+.2f})
                    </span>
                </div>
                <div style="display:flex; flex-wrap:wrap; gap:0.6rem; font-size:0.72rem; color:#94a3b8; align-items:center;">
                    <span><b>Provider:</b> <span style="color:{prov_color}; font-weight:600;">{prov_label}</span></span>
                    <span>•</span>
                    <span><b>Source:</b> {art['source']}</span>
                    <span>•</span>
                    <span><b>Published:</b> {art['published_at_utc']}</span>
                    <span>•</span>
                    <span><b>Event:</b> <span style="color:#e2e8f0;">{art['event_type']}</span></span>
                    <span>•</span>
                    <span><b>Relevance:</b> {art['relevance_score']:.2f}</span>
                    <span>•</span>
                    <span><b>Importance:</b> {art['importance_score']:.2f}</span>
                    <span>•</span>
                    <span><b>Triage:</b> {art['triage_score']:.3f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Phase 3: Intelligent Financial News Explanation
            from src.nlp.news_explanation import NewsExplanationEngine
            expl = NewsExplanationEngine().get_or_create_explanation(art.to_dict(), art.to_dict())

            imp_dir = expl.implication_direction
            imp_color = "#00c076" if imp_dir == "POSITIVE" else ("#ff4d4d" if imp_dir == "NEGATIVE" else ("#a855f7" if imp_dir == "MIXED" else "#f59e0b"))
            ev_color = "#00c076" if expl.evidence_strength == "DIRECT" else ("#f59e0b" if expl.evidence_strength == "PARTIAL" else "#ef4444")

            with st.expander(f"💡 Plain-English Financial Explanation — {imp_dir} Implications ({expl.evidence_strength} Evidence)", expanded=False):
                st.markdown(f"""
                <div style="background:#0b0f17; border:1px solid #1e2638; border-radius:4px; padding:0.75rem 0.9rem; font-size:0.78rem; line-height:1.5;">
                    <div style="margin-bottom:0.55rem;">
                        <span style="color:#38bdf8; font-weight:700; text-transform:uppercase; font-size:0.70rem;">📌 What The Article Reports:</span><br>
                        <span style="color:#f1f5f9;">{expl.what_it_reports}</span>
                    </div>
                    <div style="margin-bottom:0.55rem;">
                        <span style="color:#818cf8; font-weight:700; text-transform:uppercase; font-size:0.70rem;">🏢 Why It May Matter to {expl.company_name} ({expl.ticker}):</span><br>
                        <span style="color:#cbd5e1;">{expl.why_it_matters}</span>
                    </div>
                    <div style="margin-bottom:0.55rem;">
                        <span style="color:{imp_color}; font-weight:700; text-transform:uppercase; font-size:0.70rem;">⚖️ Potential Business Implications ({imp_dir}):</span><br>
                        <span style="color:#e2e8f0;">{expl.implications}</span>
                    </div>
                    <div style="margin-bottom:0.55rem;">
                        <span style="color:#34d399; font-weight:700; text-transform:uppercase; font-size:0.70rem;">🔍 Grounded Evidence:</span>
                        <span style="background:rgba(255,255,255,0.05); color:{ev_color}; border:1px solid {ev_color}; font-size:0.65rem; font-weight:700; padding:0.1rem 0.35rem; border-radius:3px; margin-left:0.4rem;">{expl.evidence_strength} EVIDENCE</span><br>
                        <span style="color:#94a3b8; font-family:monospace; font-size:0.72rem;">{expl.evidence}</span>
                    </div>
                    <div style="margin-bottom:0.55rem;">
                        <span style="color:#f59e0b; font-weight:700; text-transform:uppercase; font-size:0.70rem;">⚠️ Key Limitations & Missing Information:</span><br>
                        <span style="color:#fde68a; font-size:0.74rem;">{expl.limitations}</span>
                    </div>
                    <div style="padding:0.45rem 0.65rem; background:rgba(245, 158, 11, 0.08); border-left:3px solid #f59e0b; border-radius:3px; color:#fde68a; font-size:0.70rem; margin-top:0.4rem;">
                        <b>DISCLAIMER:</b> {expl.directional_disclaimer}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("<div style='height: 0.35rem;'></div>", unsafe_allow_html=True)
    else:
        st.info(f"No individual news article records currently available in SQLite storage for {target_news_ticker}. See summary analytics above.")

