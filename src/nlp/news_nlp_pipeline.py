"""
Financial News NLP Pipeline for Real-World Data Ingestion.

This module provides reusable text processing, sentiment scoring, entity relevance,
event importance, recency decay, triage calculation, and daily aggregation for financial news.
Reuses the validated Loughran-McDonald-inspired methodology established in Assignment 7.

Output records are strictly marked with Dataset_Type = "REAL".
"""

import re
import datetime
from datetime import timezone
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

# =====================================================================
# 1. FINANCIAL SENTIMENT LEXICON & NEGATION TERMS (Assignment 7 Reused)
# =====================================================================
FINANCIAL_POS_TERMS = set([
    "profit", "profits", "profitable", "growth", "grow", "grows", "growing", "gain", "gains", "surge", "surges",
    "surging", "record", "records", "beat", "beats", "beating", "outperform", "outperforms", "rally", "rallies",
    "bullish", "upgrade", "upgrades", "upgraded", "dividend", "dividends", "expansion", "expands", "strong",
    "stronger", "strongest", "revenue", "revenues", "success", "successful", "upside", "milestone", "innovation",
    "breakthrough", "exceed", "exceeds", "exceeded", "exceeding", "soar", "soars", "soaring", "momentum", "leader",
    "partnership", "partnerships", "approved", "approval", "climbs", "climbing", "rebound", "rebounds", "advances",
    "optimism", "recovery", "boost", "boosts", "boosted", "top", "robust", "highs", "all-time"
])

FINANCIAL_NEG_TERMS = set([
    "loss", "losses", "plunge", "plunges", "plunged", "decline", "declines", "declining", "drop", "drops",
    "dropping", "fall", "falls", "falling", "slump", "slumps", "slumping", "miss", "misses", "missed",
    "downgrade", "downgrades", "downgraded", "lawsuit", "lawsuits", "penalty", "penalties", "probe", "probes",
    "investigation", "investigations", "regulatory", "regulation", "warning", "warnings", "warns", "cut",
    "cuts", "cutting", "weak", "weaker", "weakest", "weakening", "deficit", "antitrust", "crash", "crashes",
    "tumble", "tumbles", "breach", "breaches", "risk", "risks", "sue", "sues", "sued", "complaint", "complaints",
    "hurt", "hurts", "slowdown", "delay", "delays", "underweight", "headwinds", "dips", "pause", "halt",
    "contracted", "voids", "voided", "bottlenecks", "constraints", "restructuring"
])

NEGATIONS = set([
    "not", "no", "never", "fails", "failed", "failure", "hardly", "without", "neither", "barely"
])

# =====================================================================
# 2. COMPANY ENTITY DICTIONARY (All 10 Companies)
# =====================================================================
COMPANY_ENTITIES: Dict[str, List[str]] = {
    "AAPL": ["apple", "aapl", "iphone", "ipad", "mac", "macbook", "tim cook", "app store", "vision pro", "ios", "siri"],
    "MSFT": ["microsoft", "msft", "azure", "copilot", "windows", "satya nadella", "office 365", "xbox", "activision", "teams"],
    "GOOGL": ["google", "alphabet", "googl", "goog", "gemini", "android", "youtube", "sundar pichai", "google cloud", "deepmind", "waymo"],
    "AMZN": ["amazon", "amzn", "aws", "andy jassy", "prime", "e-commerce", "cloud", "alexa", "kuiper"],
    "NVDA": ["nvidia", "nvda", "jensen huang", "blackwell", "hopper", "h100", "b200", "geforce", "gpu", "cuda", "semiconductor"],
    "META": ["meta", "meta platforms", "facebook", "mark zuckerberg", "instagram", "whatsapp", "llama", "quest", "reality labs", "metaverse"],
    "TSLA": ["tesla", "tsla", "elon musk", "cybertruck", "model 3", "model y", "gigafactory", "supercharger", "fsd", "robotaxi"],
    "JPM": ["jpmorgan", "jpmorgan chase", "jpm", "jamie dimon", "chase", "commercial bank", "investment bank", "net interest income"],
    "V": ["visa", "v", "payments network", "credit cards", "debit", "cross-border", "payment volume", "interchange"],
    "JNJ": ["johnson & johnson", "jnj", "talc", "pharmaceutical", "medtech", "darzalex", "stelara", "fda approval", "medical devices"]
}

EVENT_IMPORTANCE_MAP: Dict[str, float] = {
    "Earnings / Financials": 0.90,
    "Regulatory / Legal": 0.85,
    "Product / Innovation": 0.75,
    "Analyst Action": 0.65,
    "Corporate Governance": 0.50,
    "General / Macro": 0.20
}

EVENT_KEYWORDS: Dict[str, List[str]] = {
    "Earnings / Financials": ["earnings", "revenue", "profit", "q1", "q2", "q3", "q4", "fiscal", "eps", "dividend", "guidance", "sales", "quarterly", "cash flow"],
    "Regulatory / Legal": ["sec", "doj", "lawsuit", "antitrust", "investigation", "probe", "fine", "court", "judge", "settlement", "patent", "appeal", "breach"],
    "Product / Innovation": ["launch", "chip", "ai", "model", "release", "feature", "unveil", "platform", "cloud", "copilot", "device", "supercomputer", "blackwell", "gpu"],
    "Analyst Action": ["upgrade", "downgrade", "price target", "analyst", "rating", "outperform", "overweight", "underweight", "buy rating", "reiterate"],
    "Corporate Governance": ["ceo", "executive", "board", "shareholder", "restructuring", "layoff", "appoint", "resign", "stake", "buyback", "management"]
}


# =====================================================================
# 3. TEXT PREPROCESSING & FINANCIAL SENTIMENT
# =====================================================================
def clean_financial_text(text: str) -> str:
    """Clean and normalize financial headline/summary text."""
    if not isinstance(text, str):
        return ""
    # Strip HTML tags
    clean = re.sub(r"<[^>]+>", " ", text)
    # Strip URLs
    clean = re.sub(r"http\S+|www\.\S+", " ", clean)
    # Strip special punctuation but preserve alphanumeric
    clean = re.sub(r"[^\w\s\$\%\-\.]", " ", clean)
    # Collapse multiple whitespaces
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def compute_financial_sentiment(text: str) -> Tuple[str, float]:
    """
    Compute financial polarity score and discrete label with 3-token negation window.
    Returns: (Sentiment_Label: str, Sentiment_Score: float in [-1.0, +1.0])
    """
    clean = clean_financial_text(text)
    tokens = re.findall(r"\b[a-zA-Z]{2,}\b", clean.lower())
    
    pos_count = 0
    neg_count = 0
    
    for i, token in enumerate(tokens):
        # 3-word lookback window for negation detection
        window = tokens[max(0, i - 3):i]
        is_negated = any(n in window for n in NEGATIONS)
        
        if token in FINANCIAL_POS_TERMS:
            if is_negated:
                neg_count += 1
            else:
                pos_count += 1
        elif token in FINANCIAL_NEG_TERMS:
            if is_negated:
                pos_count += 1
            else:
                neg_count += 1
                
    total_sentiment_terms = pos_count + neg_count
    if total_sentiment_terms == 0:
        score = 0.0
    else:
        score = (pos_count - neg_count) / total_sentiment_terms
        
    if score > 0.05:
        label = "Positive"
    elif score < -0.05:
        label = "Negative"
    else:
        label = "Neutral"
        
    return label, round(float(score), 4)


# =====================================================================
# 4. ENTITY RELEVANCE, EVENT IMPORTANCE & RECENCY
# =====================================================================
def compute_company_relevance(ticker: str, headline: str, summary: str = "") -> float:
    """
    Compute continuous relevance score in [0.10, 1.00] based on entity proximity.
    - 1.00: Core ticker or flagship entity in headline
    - 0.70: Core entity in summary/body text
    - 0.30: Secondary mention or sector news
    """
    target = ticker.upper()
    entities = COMPANY_ENTITIES.get(target, [target.lower()])
    
    headline_lower = headline.lower()
    text_lower = f"{headline} {summary}".lower()
    
    # 1. Direct core entity match in headline
    if target.lower() in headline_lower or any(e in headline_lower for e in entities):
        return 1.00
    # 2. Entity match in body text
    elif target.lower() in text_lower or any(e in text_lower for e in entities):
        return 0.70
    else:
        return 0.30


def compute_event_importance(headline: str, summary: str = "") -> Tuple[str, float]:
    """
    Determine the financial event category and map to intrinsic importance score.
    Returns: (Event_Type: str, Importance_Score: float in [0.20, 0.90])
    """
    full_text = f"{headline} {summary}".lower()
    
    for category, keywords in EVENT_KEYWORDS.items():
        if any(kw in full_text for kw in keywords):
            return category, EVENT_IMPORTANCE_MAP[category]
            
    return "General / Macro", 0.20


def compute_recency_score(published_at_utc: Union[str, datetime.datetime], eval_time_utc: Optional[datetime.datetime] = None) -> float:
    """
    Compute temporal decay score in [0.0, 1.0] with strict anti-leakage guard.
    - Future news relative to eval_time -> 0.00 (FORBIDDEN LOOKAHEAD)
    - Same day (Age 0) -> 1.00
    - 1 Day prior -> 0.70
    - 2 Days prior -> 0.40
    - > 2 Days prior -> max(0.10, exp(-0.5 * age_days))
    """
    if eval_time_utc is None:
        eval_time_utc = datetime.datetime.now(timezone.utc)
    elif eval_time_utc.tzinfo is None:
        eval_time_utc = eval_time_utc.replace(tzinfo=timezone.utc)
        
    if isinstance(published_at_utc, str):
        pub_dt = pd.to_datetime(published_at_utc)
        if pub_dt.tzinfo is None:
            pub_dt = pub_dt.tz_localize("UTC")
        else:
            pub_dt = pub_dt.tz_convert("UTC")
    elif isinstance(published_at_utc, datetime.datetime):
        pub_dt = published_at_utc if published_at_utc.tzinfo is not None else published_at_utc.replace(tzinfo=timezone.utc)
    else:
        return 1.00

    # Calculate difference in days
    time_diff = eval_time_utc - pub_dt
    age_days = time_diff.total_seconds() / 86400.0
    
    if age_days < -0.01:  # Future article beyond 15-minute clock drift
        return 0.00
    elif age_days <= 1.0:
        return 1.00
    elif age_days <= 2.0:
        return 0.70
    elif age_days <= 3.0:
        return 0.40
    else:
        return round(float(max(0.10, np.exp(-0.5 * (age_days - 2.0)))), 4)


def compute_triage_score(relevance: float, recency: float, importance: float) -> float:
    """
    Calculate combined news triage score:
    Triage_Score = Relevance_Score * Recency_Score * Importance_Score in [0.0, 1.0]
    """
    return round(float(relevance * recency * importance), 4)


# =====================================================================
# 5. FULL PIPELINE EXECUTION FOR BATCH OF ARTICLES
# =====================================================================
def process_articles_nlp(articles_df: pd.DataFrame, eval_time_utc: Optional[datetime.datetime] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Processes a DataFrame of raw normalized news articles through the complete NLP pipeline.
    
    Args:
        articles_df: DataFrame containing at least:
                     ['News_ID', 'Ticker', 'Company_Name', 'Published_At_UTC', 'Headline', 'Summary']
        eval_time_utc: Reference evaluation timestamp for recency calculation.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            1. article_nlp_df: Enriched article-level DataFrame with sentiment, relevance, importance, triage.
            2. daily_signals_df: Aggregated daily signals per (Ticker, Date).
    """
    if articles_df.empty:
        return pd.DataFrame(), pd.DataFrame()
        
    df = articles_df.copy()
    
    # 1. Clean combined text
    df["Clean_Headline"] = df["Headline"].apply(clean_financial_text)
    df["Clean_Summary"] = df["Summary"].fillna("").apply(clean_financial_text)
    df["Full_Text"] = df["Clean_Headline"] + " " + df["Clean_Summary"]
    
    # 2. Financial Sentiment
    sent_results = [compute_financial_sentiment(t) for t in df["Full_Text"]]
    df["Sentiment_Label"] = [r[0] for r in sent_results]
    df["Sentiment_Score"] = [r[1] for r in sent_results]
    
    # 3. Entity Relevance
    df["Relevance_Score"] = [
        compute_company_relevance(t, h, s)
        for t, h, s in zip(df["Ticker"], df["Clean_Headline"], df["Clean_Summary"])
    ]
    
    # 4. Event Importance
    evt_results = [
        compute_event_importance(h, s)
        for h, s in zip(df["Clean_Headline"], df["Clean_Summary"])
    ]
    df["Event_Type"] = [e[0] for e in evt_results]
    df["Importance_Score"] = [e[1] for e in evt_results]
    
    # 5. Recency Score
    df["Recency_Score"] = [
        compute_recency_score(dt, eval_time_utc)
        for dt in df["Published_At_UTC"]
    ]
    
    # 6. Triage Score
    df["Triage_Score"] = (
        df["Relevance_Score"] * df["Recency_Score"] * df["Importance_Score"]
    ).round(4)
    
    # Set explicit Real dataset label
    df["Dataset_Type"] = "REAL"
    df["Processed_At_UTC"] = datetime.datetime.now(timezone.utc).isoformat()
    
    # Extract Date string for daily grouping (YYYY-MM-DD from UTC publication time)
    df["Date"] = pd.to_datetime(df["Published_At_UTC"]).dt.strftime("%Y-%m-%d")
    
    # 7. Aggregate into Daily Signals
    daily_records = []
    for (ticker, dt), group in df.groupby(["Ticker", "Date"]):
        n_items = len(group)
        company_name = group["Company_Name"].iloc[0] if "Company_Name" in group.columns else ticker
        avg_sent = round(float(group["Sentiment_Score"].mean()), 4)
        avg_triage = round(float(group["Triage_Score"].mean()), 4)
        high_impact_cnt = int(((group["Importance_Score"] >= 0.75) & (group["Relevance_Score"] >= 0.70)).sum())
        
        triage_sum = group["Triage_Score"].sum()
        if triage_sum > 0:
            weighted_sent = round(float((group["Sentiment_Score"] * group["Triage_Score"]).sum() / triage_sum), 4)
        else:
            weighted_sent = avg_sent
            
        if weighted_sent > 0.05:
            dom_sent = "Positive"
        elif weighted_sent < -0.05:
            dom_sent = "Negative"
        else:
            dom_sent = "Neutral"
            
        news_signal = round(float(np.clip(avg_sent * avg_triage, -1.0, 1.0)), 4)
        
        daily_records.append({
            "Ticker": ticker,
            "Company_Name": company_name,
            "Date": dt,
            "News_Count": n_items,
            "Average_Sentiment": avg_sent,
            "Triage_Weighted_Sentiment": weighted_sent,
            "Average_Triage_Score": avg_triage,
            "High_Impact_News_Count": high_impact_cnt,
            "Dominant_Sentiment": dom_sent,
            "News_Signal": news_signal,
            "Dataset_Type": "REAL"
        })
        
    daily_df = pd.DataFrame(daily_records)
    if not daily_df.empty:
        daily_df = daily_df.sort_values(by=["Ticker", "Date"]).reset_index(drop=True)
        
    return df, daily_df
