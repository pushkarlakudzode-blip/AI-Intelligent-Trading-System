"""
Data Storage & Quality Layer for Real-Time and Historical Data Ingestion.

Manages persistent storage in SQLite (data/live/news_database.db) and CSV,
deterministic deduplication via SHA-256 News_ID, validation reporting,
and ingestion telemetry logging.
"""

import hashlib
import json
import logging
import os
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd

logger = logging.getLogger(__name__)

# Base directory for live data
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_LIVE_DATA_DIR = PROJECT_ROOT / "data" / "live"

LIVE_DATA_DIR = DEFAULT_LIVE_DATA_DIR
LIVE_NEWS_DIR = LIVE_DATA_DIR / "news"
LIVE_NLP_DIR = LIVE_DATA_DIR / "nlp"
LIVE_MARKET_DIR = LIVE_DATA_DIR / "market"

DB_PATH = LIVE_DATA_DIR / "news_database.db"
CSV_ARTICLES_PATH = LIVE_NEWS_DIR / "news_articles.csv"
CSV_NLP_PATH = LIVE_NLP_DIR / "news_nlp_results.csv"
CSV_DAILY_SIGNAL_PATH = LIVE_NLP_DIR / "daily_news_signal.csv"
CSV_VALIDATION_PATH = LIVE_NEWS_DIR / "news_validation_report.csv"

SUPPORTED_TICKERS = {"AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "JNJ"}


def configure_storage(base_dir: Optional[Union[str, Path]] = None):
    """
    Configure storage paths.
    
    If base_dir is None, resets to default production live data directory (data/live/).
    If base_dir is provided (e.g. temporary directory during unit testing),
    redirects all SQLite DB and CSV operations to the isolated path.
    """
    global LIVE_DATA_DIR, LIVE_NEWS_DIR, LIVE_NLP_DIR, LIVE_MARKET_DIR
    global DB_PATH, CSV_ARTICLES_PATH, CSV_NLP_PATH, CSV_DAILY_SIGNAL_PATH, CSV_VALIDATION_PATH

    if base_dir is None:
        LIVE_DATA_DIR = DEFAULT_LIVE_DATA_DIR
    else:
        LIVE_DATA_DIR = Path(base_dir)

    LIVE_NEWS_DIR = LIVE_DATA_DIR / "news"
    LIVE_NLP_DIR = LIVE_DATA_DIR / "nlp"
    LIVE_MARKET_DIR = LIVE_DATA_DIR / "market"

    DB_PATH = LIVE_DATA_DIR / "news_database.db"
    CSV_ARTICLES_PATH = LIVE_NEWS_DIR / "news_articles.csv"
    CSV_NLP_PATH = LIVE_NLP_DIR / "news_nlp_results.csv"
    CSV_DAILY_SIGNAL_PATH = LIVE_NLP_DIR / "daily_news_signal.csv"
    CSV_VALIDATION_PATH = LIVE_NEWS_DIR / "news_validation_report.csv"


def generate_news_id(ticker: str, published_at_utc: str, headline: str, url: str) -> str:
    """
    Generate a deterministic SHA-256 hash identifier for an article.
    Ensures identical articles from multiple query fetches are never duplicated.
    """
    normalized_string = f"{ticker.strip().upper()}|{published_at_utc.strip()}|{headline.strip().lower()}|{url.strip().lower()}"
    return hashlib.sha256(normalized_string.encode("utf-8")).hexdigest()[:32]


def init_storage():
    """Ensure all required live directories exist and SQLite database schema is initialized."""
    LIVE_NEWS_DIR.mkdir(parents=True, exist_ok=True)
    LIVE_NLP_DIR.mkdir(parents=True, exist_ok=True)
    LIVE_MARKET_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. Articles table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS articles (
        news_id TEXT PRIMARY KEY,
        ticker TEXT NOT NULL,
        company_name TEXT,
        published_at_utc TEXT NOT NULL,
        headline TEXT NOT NULL,
        summary TEXT,
        source TEXT,
        url TEXT,
        provider TEXT DEFAULT 'Alpha Vantage',
        retrieved_at_utc TEXT NOT NULL,
        provider_sentiment REAL,
        provider_relevance REAL,
        dataset_type TEXT DEFAULT 'REAL'
    )
    """)

    # 2. NLP results table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS nlp_results (
        news_id TEXT PRIMARY KEY,
        ticker TEXT NOT NULL,
        published_at_utc TEXT NOT NULL,
        sentiment_label TEXT,
        sentiment_score REAL,
        relevance_score REAL,
        event_type TEXT,
        importance_score REAL,
        recency_score REAL,
        triage_score REAL,
        processed_at_utc TEXT NOT NULL,
        dataset_type TEXT DEFAULT 'REAL',
        FOREIGN KEY (news_id) REFERENCES articles(news_id)
    )
    """)

    # 3. Validation audit log
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS validation_logs (
        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp_utc TEXT NOT NULL,
        ticker TEXT,
        issue_type TEXT NOT NULL,
        details TEXT,
        action_taken TEXT
    )
    """)

    # 4. News explanations table (Phase 3)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS news_explanations (
        news_id TEXT PRIMARY KEY,
        ticker TEXT NOT NULL,
        company_name TEXT,
        what_it_reports TEXT,
        why_it_matters TEXT,
        implications TEXT,
        implication_direction TEXT,
        evidence TEXT,
        evidence_strength TEXT,
        limitations TEXT,
        directional_disclaimer TEXT,
        provider_used TEXT,
        created_at_utc TEXT,
        FOREIGN KEY (news_id) REFERENCES articles(news_id)
    )
    """)

    # Create indexes for fast lookup
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_articles_ticker ON articles(ticker)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_articles_pubdate ON articles(published_at_utc)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_nlp_ticker ON nlp_results(ticker)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_explanations_ticker ON news_explanations(ticker)")

    conn.commit()
    conn.close()


def get_existing_news_ids() -> Set[str]:
    """Retrieve the set of all existing News_IDs from SQLite database to prevent duplicate ingestion."""
    init_storage()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT news_id FROM articles")
    rows = cursor.fetchall()
    conn.close()
    return {r[0] for r in rows}


def validate_article(article: Dict) -> Tuple[bool, Optional[str]]:
    """
    Validate article quality, structure, and lookahead constraints.
    
    Checks:
    - Missing or empty headline
    - Missing or malformed timestamp
    - Future publication timestamp (strict lookahead guard)
    - Unsupported ticker
    
    Returns: (is_valid: bool, issue_reason: Optional[str])
    """
    # 1. Headline validation
    headline = str(article.get("Headline", "")).strip()
    if not headline or len(headline) < 5:
        return False, "EMPTY_OR_TRUNCATED_HEADLINE"

    # 2. Ticker validation
    ticker = str(article.get("Ticker", "")).strip().upper()
    if ticker not in SUPPORTED_TICKERS:
        return False, f"UNSUPPORTED_TICKER: {ticker}"

    # 3. Timestamp validation
    pub_at = article.get("Published_At_UTC")
    if not pub_at:
        return False, "MISSING_TIMESTAMP"

    try:
        pub_dt = pd.to_datetime(pub_at)
        if pub_dt.tzinfo is None:
            pub_dt = pub_dt.tz_localize("UTC")
        else:
            pub_dt = pub_dt.tz_convert("UTC")
            
        now_utc = datetime.now(timezone.utc)
        # Lookahead guard: Allow maximum 15-minute future clock drift
        if pub_dt > now_utc + timedelta(minutes=15):
            return False, f"FUTURE_PUBLICATION_TIMESTAMP: {pub_dt.isoformat()} > {now_utc.isoformat()}"
    except Exception as e:
        return False, f"INVALID_TIMESTAMP_FORMAT: {e}"

    return True, None


def log_validation_issue(ticker: str, issue_type: str, details: str, action_taken: str = "REJECTED"):
    """Record validation audit failure into database and CSV log."""
    init_storage()
    now_utc = datetime.now(timezone.utc).isoformat()
    
    # 1. Database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO validation_logs (timestamp_utc, ticker, issue_type, details, action_taken)
    VALUES (?, ?, ?, ?, ?)
    """, (now_utc, ticker, issue_type, details, action_taken))
    conn.commit()
    conn.close()

    # 2. CSV report
    row_df = pd.DataFrame([{
        "Timestamp_UTC": now_utc,
        "Ticker": ticker,
        "Issue_Type": issue_type,
        "Details": details,
        "Action_Taken": action_taken
    }])
    if CSV_VALIDATION_PATH.exists():
        row_df.to_csv(CSV_VALIDATION_PATH, mode="a", header=False, index=False)
    else:
        row_df.to_csv(CSV_VALIDATION_PATH, mode="w", header=True, index=False)


def store_articles(articles: List[Dict]) -> Tuple[int, int, int]:
    """
    Validate, deduplicate, and persist normalized articles to SQLite and CSV.
    
    Args:
        articles: List of standardized article dictionaries.
        
    Returns:
        Tuple[int, int, int]: (added_count, duplicate_count, rejected_count)
    """
    init_storage()
    existing_ids = get_existing_news_ids()
    
    to_insert = []
    added_count = 0
    duplicate_count = 0
    rejected_count = 0

    for art in articles:
        # Quality validation
        is_valid, issue = validate_article(art)
        if not is_valid:
            rejected_count += 1
            log_validation_issue(art.get("Ticker", "UNKNOWN"), issue, str(art.get("Headline", ""))[:80], "REJECTED")
            continue

        news_id = art["News_ID"]
        if news_id in existing_ids:
            duplicate_count += 1
            continue

        existing_ids.add(news_id)
        to_insert.append(art)
        added_count += 1

    if not to_insert:
        return 0, duplicate_count, rejected_count

    # 1. Insert into SQLite
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    for art in to_insert:
        cursor.execute("""
        INSERT OR IGNORE INTO articles (
            news_id, ticker, company_name, published_at_utc, headline, summary,
            source, url, provider, retrieved_at_utc, provider_sentiment, provider_relevance, dataset_type
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            art["News_ID"],
            art["Ticker"],
            art.get("Company_Name", art["Ticker"]),
            art["Published_At_UTC"],
            art["Headline"],
            art.get("Summary", ""),
            art.get("Source", ""),
            art.get("URL", ""),
            art.get("Provider", "Alpha Vantage"),
            art.get("Retrieved_At_UTC", datetime.now(timezone.utc).isoformat()),
            art.get("Provider_Sentiment", 0.0),
            art.get("Provider_Relevance", 0.0),
            "REAL"
        ))
    conn.commit()
    conn.close()

    # 2. Sync to CSV from SQLite to ensure perfect schema consistency & sorting
    conn = sqlite3.connect(DB_PATH)
    all_articles_df = pd.read_sql("""
        SELECT news_id AS News_ID, ticker AS Ticker, company_name AS Company_Name,
               published_at_utc AS Published_At_UTC, headline AS Headline, summary AS Summary,
               source AS Source, url AS URL, provider AS Provider,
               retrieved_at_utc AS Retrieved_At_UTC, provider_sentiment AS Provider_Sentiment,
               provider_relevance AS Provider_Relevance, dataset_type AS Dataset_Type
        FROM articles
        ORDER BY published_at_utc DESC
    """, conn)
    conn.close()
    all_articles_df.to_csv(CSV_ARTICLES_PATH, index=False)

    return added_count, duplicate_count, rejected_count


def store_nlp_results(nlp_df: pd.DataFrame, daily_signals_df: pd.DataFrame):
    """
    Store NLP enrichment records into SQLite and CSV files.
    """
    if nlp_df.empty:
        return

    init_storage()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    for _, row in nlp_df.iterrows():
        cursor.execute("""
        INSERT OR REPLACE INTO nlp_results (
            news_id, ticker, published_at_utc, sentiment_label, sentiment_score,
            relevance_score, event_type, importance_score, recency_score, triage_score,
            processed_at_utc, dataset_type
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            row["News_ID"],
            row["Ticker"],
            row["Published_At_UTC"],
            row["Sentiment_Label"],
            row["Sentiment_Score"],
            row["Relevance_Score"],
            row["Event_Type"],
            row["Importance_Score"],
            row["Recency_Score"],
            row["Triage_Score"],
            row.get("Processed_At_UTC", datetime.now(timezone.utc).isoformat()),
            "REAL"
        ))
    conn.commit()

    # Sync article-level NLP CSV from SQLite
    all_nlp_df = pd.read_sql("""
        SELECT news_id AS News_ID, ticker AS Ticker, published_at_utc AS Published_At_UTC,
               sentiment_label AS Sentiment_Label, sentiment_score AS Sentiment_Score,
               relevance_score AS Relevance_Score, event_type AS Event_Type,
               importance_score AS Importance_Score, recency_score AS Recency_Score,
               triage_score AS Triage_Score, processed_at_utc AS Processed_At_UTC,
               dataset_type AS Dataset_Type
        FROM nlp_results
        ORDER BY published_at_utc DESC
    """, conn)
    conn.close()
    all_nlp_df.to_csv(CSV_NLP_PATH, index=False)

    # Sync daily aggregated signals CSV
    if not daily_signals_df.empty:
        if CSV_DAILY_SIGNAL_PATH.exists():
            try:
                existing_daily = pd.read_csv(CSV_DAILY_SIGNAL_PATH)
                combined = pd.concat([existing_daily, daily_signals_df]).drop_duplicates(subset=["Ticker", "Date"], keep="last")
            except Exception:
                combined = daily_signals_df
            combined.sort_values(by=["Ticker", "Date"]).to_csv(CSV_DAILY_SIGNAL_PATH, index=False)
        else:
            daily_signals_df.sort_values(by=["Ticker", "Date"]).to_csv(CSV_DAILY_SIGNAL_PATH, index=False)


def get_stored_articles(ticker: Optional[str] = None) -> pd.DataFrame:
    """Retrieve stored articles from SQLite database."""
    init_storage()
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT * FROM articles"
    params = []
    if ticker:
        query += " WHERE ticker = ?"
        params.append(ticker.upper())
    query += " ORDER BY published_at_utc DESC"
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def get_stored_nlp_results(ticker: Optional[str] = None) -> pd.DataFrame:
    """Retrieve stored NLP results from SQLite database."""
    init_storage()
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT * FROM nlp_results"
    params = []
    if ticker:
        query += " WHERE ticker = ?"
        params.append(ticker.upper())
    query += " ORDER BY published_at_utc DESC"
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def store_explanation(explanation_dict: Dict[str, Any]) -> None:
    """Store or update a news explanation record in SQLite."""
    init_storage()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    INSERT OR REPLACE INTO news_explanations (
        news_id, ticker, company_name, what_it_reports, why_it_matters,
        implications, implication_direction, evidence, evidence_strength,
        limitations, directional_disclaimer, provider_used, created_at_utc
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        explanation_dict["news_id"],
        explanation_dict["ticker"],
        explanation_dict.get("company_name", ""),
        explanation_dict.get("what_it_reports", ""),
        explanation_dict.get("why_it_matters", ""),
        explanation_dict.get("implications", ""),
        explanation_dict.get("implication_direction", "UNCERTAIN"),
        explanation_dict.get("evidence", ""),
        explanation_dict.get("evidence_strength", "INSUFFICIENT"),
        explanation_dict.get("limitations", ""),
        explanation_dict.get("directional_disclaimer", ""),
        explanation_dict.get("provider_used", "GroundedAnalyticalEngine"),
        explanation_dict.get("created_at_utc", "")
    ))
    conn.commit()
    conn.close()


def get_explanation(news_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve explanation for a specific news_id from SQLite."""
    init_storage()
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    cur = conn.cursor()
    cur.execute("""
    SELECT news_id, ticker, company_name, what_it_reports, why_it_matters,
           implications, implication_direction, evidence, evidence_strength,
           limitations, directional_disclaimer, provider_used, created_at_utc
    FROM news_explanations WHERE news_id = ?
    """, (news_id,))
    row = cur.fetchone()
    conn.close()
    if row:
        return {
            "news_id": row[0],
            "ticker": row[1],
            "company_name": row[2],
            "what_it_reports": row[3],
            "why_it_matters": row[4],
            "implications": row[5],
            "implication_direction": row[6],
            "evidence": row[7],
            "evidence_strength": row[8],
            "limitations": row[9],
            "directional_disclaimer": row[10],
            "provider_used": row[11],
            "created_at_utc": row[12]
        }
    return None


def get_stored_explanations(ticker: Optional[str] = None) -> pd.DataFrame:
    """Retrieve stored explanations from SQLite database."""
    init_storage()
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    query = "SELECT * FROM news_explanations"
    params = []
    if ticker:
        query += " WHERE ticker = ?"
        params.append(ticker.upper())
    query += " ORDER BY created_at_utc DESC"
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df

