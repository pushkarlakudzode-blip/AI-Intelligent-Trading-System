"""
Phase 5C: Live News Data Quality, Coverage & Readiness Auditor.
==============================================================
Provides comprehensive, read-only auditing of:
1. Ingestion volumes, deduplication, and acceptance/rejection rates.
2. Ticker coverage across all 10 supported portfolio equities.
3. Timestamp interpretation across UTC and US Market Time (Pre, Regular, Post, Weekend).
4. Ingestion latency (publication vs retrieval time).
5. Feed outage vs genuine zero-news period distinction.
6. NLP and explanation pipeline coverage.
7. Data quality defects (missing text, invalid timestamps, clock skews).
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone, timedelta
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pytz

logger = logging.getLogger("phase5c_auditor")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "live" / "news_database.db"
DEFAULT_STATUS_PATH = PROJECT_ROOT / "data" / "live" / "alpaca_stream_status.json"
PHASE5C_DIR = PROJECT_ROOT / "data" / "evaluation" / "phase5c"

SUPPORTED_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "JNJ"
]

COMPANY_NAMES = {
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


@dataclass
class NewsDataQualityReport:
    total_articles: int
    total_nlp_results: int
    total_explanations: int
    total_validation_logs: int
    articles_by_provider: Dict[str, int]
    articles_by_ticker: Dict[str, int]
    session_window_counts: Dict[str, int]
    duplicate_count: int
    missing_headline_count: int
    missing_summary_count: int
    missing_nlp_count: int
    live_stream_latency_stats: Dict[str, float]
    feed_health_status: Dict[str, Any]
    zero_news_vs_outage_summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Phase5CNewsAuditor:
    """
    Read-only quality and coverage auditor for real-time and historical financial news data.
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        status_path: Optional[Path] = None
    ):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.status_path = Path(status_path) if status_path else DEFAULT_STATUS_PATH
        self.tickers = sorted(SUPPORTED_TICKERS)
        self.eastern_tz = pytz.timezone("US/Eastern")

    def load_database_tables(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load SQLite database tables in read-only mode."""
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")

        import sqlite3
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        try:
            df_art = pd.read_sql_query("SELECT * FROM articles", conn)
            df_nlp = pd.read_sql_query("SELECT * FROM nlp_results", conn)
            df_exp = pd.read_sql_query("SELECT * FROM news_explanations", conn)
            df_val = pd.read_sql_query("SELECT * FROM validation_logs", conn)
        finally:
            conn.close()

        return df_art, df_nlp, df_exp, df_val

    def load_stream_status(self) -> Dict[str, Any]:
        """Load Alpaca stream status telemetry JSON."""
        if not self.status_path.exists():
            return {"status": "OFFLINE", "connected": False, "error": "Status file missing"}
        try:
            with open(self.status_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            return {"status": "ERROR", "error": str(e)}

    def classify_session_window(self, dt_utc: pd.Timestamp) -> str:
        """
        Classify UTC timestamp into US Market Session window:
        - Weekend: Saturday or Sunday in US/Eastern
        - Pre-Market: 04:00 - 09:30 US/Eastern (Mon-Fri)
        - Regular Hours: 09:30 - 16:00 US/Eastern (Mon-Fri)
        - After-Hours: 16:00 - 20:00 US/Eastern (Mon-Fri)
        - Overnight: 20:00 - 04:00 US/Eastern (Mon-Fri)
        """
        if pd.isna(dt_utc):
            return "Unknown"
        if dt_utc.tzinfo is None:
            dt_utc = dt_utc.tz_localize("UTC")
        dt_est = dt_utc.tz_convert(self.eastern_tz)

        # Check weekend
        if dt_est.dayofweek >= 5:
            return "Weekend"

        t = dt_est.time()
        t_pre_start = pd.to_datetime("04:00:00").time()
        t_mkt_open = pd.to_datetime("09:30:00").time()
        t_mkt_close = pd.to_datetime("16:00:00").time()
        t_post_end = pd.to_datetime("20:00:00").time()

        if t_pre_start <= t < t_mkt_open:
            return "Pre-Market (04:00-09:30)"
        elif t_mkt_open <= t < t_mkt_close:
            return "Regular Hours (09:30-16:00)"
        elif t_mkt_close <= t < t_post_end:
            return "After-Hours (16:00-20:00)"
        else:
            return "Overnight (20:00-04:00)"

    def audit_timestamps_and_latency(self, df_art: pd.DataFrame) -> Dict[str, Any]:
        """
        Audit publication vs ingestion latency and timestamp integrity.
        """
        df = df_art.copy()
        df["pub_dt"] = pd.to_datetime(df["published_at_utc"], format="ISO8601", errors="coerce")
        df["ret_dt"] = pd.to_datetime(df["retrieved_at_utc"], format="ISO8601", errors="coerce")

        invalid_pub = df["pub_dt"].isna().sum()
        invalid_ret = df["ret_dt"].isna().sum()

        df["latency_sec"] = (df["ret_dt"] - df["pub_dt"]).dt.total_seconds()

        # Provider breakdown
        provider_stats = {}
        for prov in df["provider"].unique():
            sub = df[df["provider"] == prov]
            lat = sub["latency_sec"].dropna()
            provider_stats[prov] = {
                "article_count": len(sub),
                "mean_latency_sec": round(float(lat.mean()), 2) if len(lat) > 0 else 0.0,
                "median_latency_sec": round(float(lat.median()), 2) if len(lat) > 0 else 0.0,
                "p95_latency_sec": round(float(np.percentile(lat, 95)), 2) if len(lat) > 0 else 0.0,
                "min_latency_sec": round(float(lat.min()), 2) if len(lat) > 0 else 0.0,
                "max_latency_sec": round(float(lat.max()), 2) if len(lat) > 0 else 0.0,
                "negative_latency_count": int((lat < 0).sum())
            }

        alpaca_stats = provider_stats.get("Alpaca", {
            "article_count": 0, "mean_latency_sec": 0.0, "median_latency_sec": 0.0,
            "p95_latency_sec": 0.0, "min_latency_sec": 0.0, "max_latency_sec": 0.0, "negative_latency_count": 0
        })

        return {
            "invalid_published_timestamps": int(invalid_pub),
            "invalid_retrieved_timestamps": int(invalid_ret),
            "provider_latency_stats": provider_stats,
            "alpaca_live_latency_stats": alpaca_stats
        }

    def audit_ticker_coverage(self, df_art: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze article coverage by ticker and provider.
        """
        ct = pd.crosstab(df_art["ticker"], df_art["provider"])
        # Ensure all supported tickers are represented
        for t in self.tickers:
            if t not in ct.index:
                ct.loc[t] = 0
        ct = ct.loc[self.tickers].copy()
        # Add Total column and Total row
        ct["Total"] = ct.sum(axis=1)
        tot_row = ct.sum(axis=0)
        tot_row.name = "Total"
        ct = pd.concat([ct, pd.DataFrame(tot_row).T])
        return ct

    def audit_daily_coverage(self, df_art: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate news article volume by calendar date and ticker.
        """
        df = df_art.copy()
        df["pub_dt"] = pd.to_datetime(df["published_at_utc"], format="ISO8601", errors="coerce")
        df["date"] = df["pub_dt"].dt.strftime("%Y-%m-%d")

        daily = df.groupby(["date", "ticker"]).size().unstack(fill_value=0)
        for t in self.tickers:
            if t not in daily.columns:
                daily[t] = 0
        ordered_cols = [t for t in self.tickers]
        daily = daily[ordered_cols]
        daily["Total_Day_Articles"] = daily.sum(axis=1)
        daily["Active_Tickers_With_News"] = (daily[ordered_cols] > 0).sum(axis=1)
        return daily.reset_index()

    def distinguish_outage_vs_zero_news(
        self,
        stream_status: Dict[str, Any],
        df_art: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Formulate and test rigorous criteria to distinguish genuine quiet market periods
        from feed disconnections or API outages.
        """
        is_connected = stream_status.get("connected", False)
        status_str = stream_status.get("status", "UNKNOWN")
        last_heartbeat = stream_status.get("last_heartbeat_utc")
        reconnect_count = stream_status.get("reconnect_count", 0)

        heartbeat_fresh = False
        heartbeat_age_sec = None
        if last_heartbeat:
            try:
                hb_dt = pd.to_datetime(last_heartbeat).tz_convert(timezone.utc)
                now_utc = datetime.now(timezone.utc)
                heartbeat_age_sec = (now_utc - hb_dt).total_seconds()
                heartbeat_fresh = bool(heartbeat_age_sec <= 60.0)
            except Exception:
                pass

        criteria = {
            "rule_1_feed_operational": "WebSocket TCP connection established, authenticated, and subscribed",
            "rule_2_heartbeat_liveness": "Heartbeat or ping/pong frame received within trailing 60 seconds",
            "rule_3_absence_of_error_codes": "Zero unhandled close codes (e.g., 4001 auth failed, 4004 rate limited)",
            "rule_4_zero_news_definition": (
                "Feed is verified healthy (Rules 1-3 active), but no article frames for tracked symbols "
                "arrive because provider published no matching headlines during interval"
            ),
            "rule_5_feed_outage_definition": (
                "Connection broken, unauthenticated, heartbeat age > 60s, or listener thread terminated"
            )
        }

        current_feed_diagnosis = "STREAMING_HEALTHY" if (is_connected and heartbeat_fresh) else "DEGRADED_OR_OFFLINE"

        alpaca_articles = df_art[df_art["provider"] == "Alpaca"]
        last_streamed_article = None
        if not alpaca_articles.empty:
            latest = alpaca_articles.sort_values("published_at_utc", ascending=False).iloc[0]
            last_streamed_article = {
                "ticker": latest["ticker"],
                "headline": latest["headline"],
                "published_at_utc": latest["published_at_utc"],
                "retrieved_at_utc": latest["retrieved_at_utc"]
            }

        return {
            "criteria_definition": criteria,
            "current_feed_diagnosis": current_feed_diagnosis,
            "is_connected": is_connected,
            "status_label": status_str,
            "last_heartbeat_utc": last_heartbeat,
            "heartbeat_age_seconds": round(heartbeat_age_sec, 1) if heartbeat_age_sec is not None else None,
            "heartbeat_fresh_le_60s": heartbeat_fresh,
            "reconnect_count": reconnect_count,
            "last_streamed_article": last_streamed_article
        }

    def run_full_audit(self, output_dir: Optional[Path] = None) -> NewsDataQualityReport:
        """
        Execute full Phase 5C audit, compile findings, and export CSV/JSON artifacts.
        """
        out_dir = output_dir or PHASE5C_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        df_art, df_nlp, df_exp, df_val = self.load_database_tables()
        stream_status = self.load_stream_status()

        # 1. Deduplication & Completeness
        dup_cnt = int(df_art["news_id"].duplicated().sum())
        missing_h = int((df_art["headline"].fillna("").str.strip() == "").sum())
        missing_s = int((df_art["summary"].fillna("").str.strip() == "").sum())

        # 2. NLP and Explanation Match
        art_ids = set(df_art["news_id"])
        nlp_ids = set(df_nlp["news_id"])
        exp_ids = set(df_exp["news_id"])
        missing_nlp = len(art_ids - nlp_ids)

        # 3. Timestamp & Latency Audit
        time_audit = self.audit_timestamps_and_latency(df_art)

        # 4. Trading Session Windows
        df_art["pub_dt"] = pd.to_datetime(df_art["published_at_utc"], format="ISO8601", errors="coerce")
        df_art["session_window"] = df_art["pub_dt"].apply(self.classify_session_window)
        session_counts = df_art["session_window"].value_counts().to_dict()

        # 5. Ticker Coverage
        df_cov = self.audit_ticker_coverage(df_art)
        df_cov.to_csv(out_dir / "ticker_coverage_summary.csv")

        # 6. Daily Coverage
        df_daily = self.audit_daily_coverage(df_art)
        df_daily.to_csv(out_dir / "daily_coverage_summary.csv", index=False)

        # 7. Session Breakdown by Provider
        df_sess_prov = pd.crosstab(df_art["session_window"], df_art["provider"], margins=True)
        df_sess_prov.to_csv(out_dir / "session_window_distribution.csv")

        # 8. Latency Table
        df_lat = pd.DataFrame(time_audit["provider_latency_stats"]).T
        df_lat.to_csv(out_dir / "latency_distribution.csv")

        # 9. Feed Outage vs Zero News
        outage_analysis = self.distinguish_outage_vs_zero_news(stream_status, df_art)

        # 10. Compile Report Object
        prov_counts = df_art["provider"].value_counts().to_dict()
        ticker_counts = df_art["ticker"].value_counts().to_dict()

        report = NewsDataQualityReport(
            total_articles=len(df_art),
            total_nlp_results=len(df_nlp),
            total_explanations=len(df_exp),
            total_validation_logs=len(df_val),
            articles_by_provider=prov_counts,
            articles_by_ticker=ticker_counts,
            session_window_counts=session_counts,
            duplicate_count=dup_cnt,
            missing_headline_count=missing_h,
            missing_summary_count=missing_s,
            missing_nlp_count=missing_nlp,
            live_stream_latency_stats=time_audit["alpaca_live_latency_stats"],
            feed_health_status=stream_status,
            zero_news_vs_outage_summary=outage_analysis
        )

        with open(out_dir / "news_ingestion_audit_summary.json", "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)

        logger.info(f"Phase 5C audit complete. Artifacts written to {out_dir}")
        return report
