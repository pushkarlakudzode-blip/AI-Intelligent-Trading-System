"""
Alpaca Real-Time Financial News Stream Listener.

Provides a persistent background WebSocket listener that connects to the
Alpaca Market Data News stream (Benzinga-powered):
    wss://stream.data.alpaca.markets/v1beta1/news

Capabilities:
1. Subscribes to all 10 supported project tickers:
   AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, JPM, V, JNJ
   (with historical FB -> META normalization).
2. Runs in a dedicated background thread or standalone process,
   completely independent of Streamlit UI reruns.
3. Automatically normalizes incoming messages to the standard schema,
   validates against quality/lookahead constraints, deduplicates using
   deterministic News_IDs, and stores articles in SQLite storage.
4. Executes the internal Loughran-McDonald NLP pipeline for newly added articles
   and persists sentiment, triage, and daily signal metrics.
5. Implements resilient auto-reconnection, ping/pong keepalives, exponential
   backoff, and thread-safe telemetry exports to data/live/alpaca_stream_status.json.
6. Protects credentials strictly; never exposes secrets in logs or status files.
"""

import html
import json
import logging
import os
import re
import sys
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import pandas as pd
from dotenv import load_dotenv

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger("alpaca_stream")

ALPACA_NEWS_WS_URL = "wss://stream.data.alpaca.markets/v1beta1/news"
DEFAULT_STATUS_PATH = PROJECT_ROOT / "data" / "live" / "alpaca_stream_status.json"

SUPPORTED_TICKERS: List[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "JNJ"
]

TICKER_ALIASES: Dict[str, str] = {
    "FB": "META"
}

COMPANY_NAMES: Dict[str, str] = {
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


def get_alpaca_credentials() -> Tuple[Optional[str], Optional[str]]:
    """Retrieve Alpaca credentials without printing or exposing values."""
    key_id = os.getenv("ALPACA_API_KEY_ID")
    secret_key = os.getenv("ALPACA_API_SECRET_KEY")

    clean_key = key_id.strip() if key_id else None
    clean_secret = secret_key.strip() if secret_key else None

    if clean_key and (clean_key.startswith("<") or clean_key == "your_key_id_here"):
        clean_key = None
    if clean_secret and (clean_secret.startswith("<") or clean_secret == "your_secret_key_here"):
        clean_secret = None

    return clean_key, clean_secret


def sanitize_text(text: str, key_id: Optional[str] = None, secret_key: Optional[str] = None) -> str:
    """Sanitize text to guarantee credentials never leak into logs or telemetry."""
    res = str(text)
    if key_id and len(key_id) > 4:
        res = res.replace(key_id, "[REDACTED_KEY]")
    if secret_key and len(secret_key) > 4:
        res = res.replace(secret_key, "[REDACTED_SECRET]")
    return res


def clean_html(raw_html: str) -> str:
    """Strip HTML markup and unescape characters to produce clean plain text."""
    if not raw_html or not isinstance(raw_html, str):
        return ""
    clean = re.sub(r"<[^>]+>", " ", raw_html)
    clean = html.unescape(clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def parse_stream_timestamp(raw_time: Any) -> Optional[str]:
    """Parse RFC-3339 / ISO timestamp string to standardized UTC: YYYY-MM-DDTHH:MM:SSZ."""
    if raw_time is None:
        return None
    try:
        if isinstance(raw_time, str):
            dt = pd.to_datetime(raw_time.strip())
            if dt.tzinfo is None:
                dt = dt.tz_localize("UTC")
            else:
                dt = dt.tz_convert("UTC")
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        elif isinstance(raw_time, (int, float)):
            dt = datetime.fromtimestamp(float(raw_time), tz=timezone.utc)
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception as exc:
        logger.debug("Failed to parse stream timestamp '%s': %s", raw_time, exc)
    return None


def normalize_stream_article(item: Dict[str, Any], target_ticker: str) -> Optional[Dict[str, Any]]:
    """
    Normalize raw Alpaca stream JSON dictionary for a specific ticker into project schema.
    """
    from src.data.data_storage import generate_news_id

    if not isinstance(item, dict):
        return None

    headline = (item.get("headline") or "").strip()
    if not headline:
        return None

    pub_raw = item.get("created_at") or item.get("updated_at")
    pub_utc = parse_stream_timestamp(pub_raw)
    if not pub_utc:
        return None

    # Check future-dated lookahead protection (allow max 15-minute clock drift)
    try:
        pub_dt = datetime.strptime(pub_utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        now_utc = datetime.now(timezone.utc)
        if pub_dt > now_utc + timedelta(minutes=15):
            logger.warning("Rejected future-dated streamed article: %s (%s)", headline, pub_utc)
            return None
    except Exception:
        pass

    summary = (item.get("summary") or "").strip()
    if not summary and item.get("content"):
        cleaned_content = clean_html(item["content"])
        if cleaned_content:
            summary = cleaned_content[:250].rsplit(" ", 1)[0] + "..." if len(cleaned_content) > 250 else cleaned_content

    url = (item.get("url") or "").strip()
    source = (item.get("source") or "Benzinga").strip()
    ticker_upper = target_ticker.upper()

    news_id = generate_news_id(ticker_upper, pub_utc, headline, url)

    return {
        "News_ID": news_id,
        "Ticker": ticker_upper,
        "Company_Name": COMPANY_NAMES.get(ticker_upper, ticker_upper),
        "Published_At_UTC": pub_utc,
        "Headline": headline,
        "Summary": summary,
        "Source": source,
        "URL": url,
        "Provider": "Alpaca",
        "Retrieved_At_UTC": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "Provider_Sentiment": 0.0,
        "Provider_Relevance": 0.50,
        "Dataset_Type": "REAL"
    }


class AlpacaStreamListener:
    """
    Persistent background thread listener for the Alpaca Real-Time News WebSocket stream.
    """

    def __init__(
        self,
        tickers: Optional[List[str]] = None,
        ws_url: str = ALPACA_NEWS_WS_URL,
        status_path: Optional[Path] = None,
        db_path: Optional[Path] = None,
        reconnect_delay_base: float = 2.0,
        reconnect_delay_max: float = 30.0,
        max_reconnects: Optional[int] = None,
        custom_store_fn: Optional[Callable[[List[Dict]], Tuple[int, int, int]]] = None,
        custom_nlp_fn: Optional[Callable[[pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame]]] = None,
        custom_nlp_store_fn: Optional[Callable[[pd.DataFrame, pd.DataFrame], None]] = None
    ):
        self.tickers = [t.strip().upper() for t in (tickers or SUPPORTED_TICKERS)]
        self.ws_url = ws_url
        self.status_path = Path(status_path) if status_path else DEFAULT_STATUS_PATH
        self.db_path = Path(db_path) if db_path else None
        self.reconnect_delay_base = reconnect_delay_base
        self.reconnect_delay_max = reconnect_delay_max
        self.max_reconnects = max_reconnects

        self.custom_store_fn = custom_store_fn
        self.custom_nlp_fn = custom_nlp_fn
        self.custom_nlp_store_fn = custom_nlp_store_fn

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._active_ws = None
        self._last_status_write_time = 0.0
        self._status_write_min_interval = 4.0

        self._status_data: Dict[str, Any] = {
            "status": "STOPPED",
            "connected": False,
            "authenticated": False,
            "subscribed": False,
            "endpoint": self.ws_url,
            "process_id": os.getpid(),
            "connected_at_utc": None,
            "last_heartbeat_utc": None,
            "subscribed_tickers": self.tickers,
            "articles_received_count": 0,
            "articles_stored_count": 0,
            "articles_duplicate_count": 0,
            "non_tracked_articles_count": 0,
            "raw_messages_received_count": 0,
            "active_retry_attempt": 0,
            "total_reconnects": 0,
            "reconnect_count": 0,
            "last_article": None,
            "error_code": None,
            "error_message": None,
            "close_code": None,
            "close_reason": None
        }

        # Ensure directory for status output exists
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        self._save_status_file(force=True)

    def start(self) -> None:
        """Start the background stream listener thread if not running."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                logger.info("AlpacaStreamListener is already running.")
                return

            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run_loop,
                daemon=True,
                name="AlpacaNewsStreamListener"
            )
            self._thread.start()
            logger.info("AlpacaStreamListener background thread started.")

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the background thread to stop and close connection cleanly."""
        logger.info("Stopping AlpacaStreamListener...")
        self._stop_event.set()

        # Close active WebSocket connection directly to unblock recv()
        with self._lock:
            if self._active_ws is not None:
                try:
                    self._active_ws.close()
                except Exception:
                    pass

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

        with self._lock:
            self._update_status(
                status="STOPPED",
                connected=False,
                authenticated=False,
                subscribed=False
            )
        logger.info("AlpacaStreamListener stopped.")

    def is_alive(self) -> bool:
        """Return True if background listener thread is currently alive."""
        return self._thread is not None and self._thread.is_alive()

    def get_status(self) -> Dict[str, Any]:
        """Return a copy of the current in-memory status dictionary."""
        with self._lock:
            return dict(self._status_data)

    def _update_status(self, force_save: bool = True, **kwargs: Any) -> None:
        """Update in-memory status and persist to status JSON file."""
        for k, v in kwargs.items():
            self._status_data[k] = v
        self._save_status_file(force=force_save)

    def _save_status_file(self, force: bool = True) -> None:
        """Persist status JSON file atomically with optional write throttling."""
        now = time.time()
        if not force and (now - self._last_status_write_time) < self._status_write_min_interval:
            return
        self._last_status_write_time = now
        try:
            tmp_path = self.status_path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._status_data, f, indent=2)
            tmp_path.replace(self.status_path)
        except Exception as e:
            logger.debug("Failed to write stream status file: %s", e)

    def _run_loop(self) -> None:
        """Continuous connection, ingestion, and reconnection loop."""
        reconnect_attempts = 0

        while not self._stop_event.is_set():
            key_id, secret_key = get_alpaca_credentials()
            if not key_id or not secret_key:
                err_msg = "ALPACA_API_KEY_ID or ALPACA_API_SECRET_KEY is not configured in .env."
                logger.error(err_msg)
                with self._lock:
                    self._update_status(status="ERROR", error_message=err_msg, force_save=True)
                break

            now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            with self._lock:
                self._update_status(
                    status="CONNECTING" if reconnect_attempts == 0 else "RECONNECTING",
                    error_message=None,
                    error_code=None,
                    active_retry_attempt=reconnect_attempts,
                    reconnect_count=reconnect_attempts,
                    force_save=True
                )

            try:
                from websockets.sync.client import connect as ws_connect
            except ImportError as e:
                err_msg = f"websockets dependency error: {e}"
                logger.error(err_msg)
                with self._lock:
                    self._update_status(status="ERROR", error_message=err_msg, force_save=True)
                break

            ws = None
            try:
                logger.info("Connecting to %s ...", self.ws_url)
                ws = ws_connect(
                    self.ws_url,
                    open_timeout=10,
                    close_timeout=5,
                    ping_interval=20,
                    ping_timeout=20
                )
                with self._lock:
                    self._active_ws = ws

                # 1. Await server greeting
                greeting_raw = ws.recv(timeout=5.0)
                greeting_data = json.loads(greeting_raw) if isinstance(greeting_raw, str) else json.loads(greeting_raw.decode("utf-8"))
                if not (isinstance(greeting_data, list) and len(greeting_data) > 0 and greeting_data[0].get("T") == "success" and greeting_data[0].get("msg") == "connected"):
                    raise ConnectionError(f"Unexpected server greeting: {greeting_raw}")

                now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                with self._lock:
                    self._status_data["raw_messages_received_count"] += 1
                    self._update_status(
                        status="AUTHENTICATING",
                        connected=True,
                        connected_at_utc=now_str,
                        last_heartbeat_utc=now_str,
                        force_save=True
                    )

                # 2. Authenticate
                auth_payload = json.dumps({"action": "auth", "key": key_id, "secret": secret_key})
                ws.send(auth_payload)

                auth_resp_raw = ws.recv(timeout=5.0)
                auth_data = json.loads(auth_resp_raw) if isinstance(auth_resp_raw, str) else json.loads(auth_resp_raw.decode("utf-8"))
                auth_items = auth_data if isinstance(auth_data, list) else [auth_data]

                authenticated = False
                for item in auth_items:
                    if item.get("T") == "success" and item.get("msg") == "authenticated":
                        authenticated = True
                        break
                    elif item.get("T") == "error":
                        code = item.get("code")
                        msg = sanitize_text(item.get("msg", "Auth error"), key_id, secret_key)
                        if code == 406:
                            # 406 is "connection limit exceeded" - another connection is active or still closing.
                            logger.warning("Alpaca connection limit exceeded (code 406). Backing off to allow previous connection to close.")
                            with self._lock:
                                self._update_status(
                                    status="RECONNECTING",
                                    error_code=406,
                                    error_message="Alpaca connection limit exceeded (code 406). Backing off to allow previous connection to close.",
                                    force_save=True
                                )
                            raise ConnectionResetError("Connection limit exceeded (code 406)")
                        else:
                            with self._lock:
                                self._update_status(
                                    status="ERROR",
                                    error_code=code,
                                    error_message=f"Alpaca auth error {code}: {msg}",
                                    force_save=True
                                )
                            raise PermissionError(f"Alpaca auth rejected: code={code}, msg={msg}")

                if not authenticated:
                    raise PermissionError(f"Unconfirmed authentication: {auth_resp_raw}")

                now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                with self._lock:
                    self._status_data["raw_messages_received_count"] += 1
                    self._update_status(authenticated=True, status="SUBSCRIBING", last_heartbeat_utc=now_str, force_save=True)

                # 3. Subscribe to 10 supported tickers
                sub_payload = json.dumps({"action": "subscribe", "news": self.tickers})
                ws.send(sub_payload)

                sub_resp_raw = ws.recv(timeout=5.0)
                sub_data = json.loads(sub_resp_raw) if isinstance(sub_resp_raw, str) else json.loads(sub_resp_raw.decode("utf-8"))
                sub_items = sub_data if isinstance(sub_data, list) else [sub_data]

                subscribed = False
                for item in sub_items:
                    if item.get("T") == "subscription":
                        subscribed = True
                        break
                    elif item.get("T") == "error":
                        code = item.get("code")
                        msg = sanitize_text(item.get("msg", "Subscription error"), key_id, secret_key)
                        with self._lock:
                            self._update_status(
                                status="ERROR",
                                error_code=code,
                                error_message=f"Alpaca subscribe error {code}: {msg}",
                                force_save=True
                            )
                        raise ValueError(f"Subscription rejected: code={code}, msg={msg}")

                if not subscribed:
                    raise ValueError(f"Subscription unconfirmed: {sub_resp_raw}")

                # Successful connection & subscription -> reset reconnect backoff & active attempts
                reconnect_attempts = 0
                now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                with self._lock:
                    self._status_data["raw_messages_received_count"] += 1
                    self._update_status(
                        status="STREAMING",
                        connected=True,
                        authenticated=True,
                        subscribed=True,
                        active_retry_attempt=0,
                        reconnect_count=0,
                        error_code=None,
                        error_message=None,
                        close_code=None,
                        close_reason=None,
                        last_heartbeat_utc=now_str,
                        force_save=True
                    )
                logger.info("AlpacaStreamListener actively streaming news for %d tickers.", len(self.tickers))

                # 4. Ingestion loop
                while not self._stop_event.is_set():
                    try:
                        frame_raw = ws.recv(timeout=1.0)
                        now_utc_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                        with self._lock:
                            self._status_data["raw_messages_received_count"] += 1
                            self._update_status(last_heartbeat_utc=now_utc_str, force_save=False)

                        frame_data = json.loads(frame_raw) if isinstance(frame_raw, str) else json.loads(frame_raw.decode("utf-8"))
                        items = frame_data if isinstance(frame_data, list) else [frame_data]

                        for item in items:
                            if item.get("T") == "n":
                                self._process_incoming_news_item(item)
                            elif item.get("T") == "error":
                                err_text = sanitize_text(str(item), key_id, secret_key)
                                logger.warning("Stream error notification: %s", err_text)
                    except TimeoutError:
                        # Normal socket timeout on quiet channels
                        now_utc_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                        with self._lock:
                            self._update_status(last_heartbeat_utc=now_utc_str, force_save=False)
                        continue

            except (PermissionError, KeyboardInterrupt):
                # Fatal auth or termination -> do not loop endlessly
                break
            except Exception as e_sock:
                close_code = getattr(e_sock, "code", None)
                close_reason = getattr(e_sock, "reason", None)
                if hasattr(e_sock, "rcvd") and e_sock.rcvd:
                    close_code = close_code or getattr(e_sock.rcvd, "code", None)
                    close_reason = close_reason or getattr(e_sock.rcvd, "reason", None)

                sanitized_err = sanitize_text(str(e_sock), key_id, secret_key)
                logger.warning("Stream connection disrupted (code=%s): %s", close_code, sanitized_err)
                with self._lock:
                    self._status_data["total_reconnects"] += 1
                    self._update_status(
                        status="RECONNECTING",
                        connected=False,
                        authenticated=False,
                        subscribed=False,
                        active_retry_attempt=reconnect_attempts + 1,
                        reconnect_count=reconnect_attempts + 1,
                        error_code=close_code,
                        error_message=sanitized_err,
                        close_code=close_code,
                        close_reason=close_reason,
                        force_save=True
                    )
            finally:
                with self._lock:
                    self._active_ws = None
                if ws is not None:
                    try:
                        ws.close()
                    except Exception:
                        pass

            if self._stop_event.is_set():
                break

            reconnect_attempts += 1
            if self.max_reconnects is not None and reconnect_attempts >= self.max_reconnects:
                logger.info("Reached maximum reconnect attempts (%d). Exiting loop.", self.max_reconnects)
                break

            delay = min(self.reconnect_delay_base * (1.5 ** (reconnect_attempts - 1)), self.reconnect_delay_max)
            logger.info("Reconnecting in %.1f seconds (Attempt %d)...", delay, reconnect_attempts)
            time.sleep(delay)

        with self._lock:
            final_status = "ERROR" if self._status_data.get("status") == "ERROR" else "STOPPED"
            self._update_status(
                status=final_status,
                connected=False,
                authenticated=False,
                subscribed=False,
                active_retry_attempt=0,
                force_save=True
            )

    def _process_incoming_news_item(self, item: Dict[str, Any]) -> None:
        """
        Normalize, validate, deduplicate, persist, and run NLP on an incoming article frame.
        STRICT SYMBOL ASSOCIATION RULE:
        An article is only associated with a tracked equity if that equity's symbol
        (or an official alias like FB -> META) is explicitly present in the item's
        'symbols' array from Alpaca. Unrelated articles are never relabeled.
        """
        raw_symbols = item.get("symbols", [])
        if not raw_symbols:
            with self._lock:
                self._status_data["non_tracked_articles_count"] += 1
            return

        # Identify which supported tickers this article targets (mapping aliases like FB -> META)
        target_tickers: Set[str] = set()
        for sym in raw_symbols:
            sym_clean = str(sym).strip().upper()
            sym_mapped = TICKER_ALIASES.get(sym_clean, sym_clean)
            if sym_mapped in self.tickers:
                target_tickers.add(sym_mapped)

        if not target_tickers:
            # Article does not mention any of our 10 tracked equities
            with self._lock:
                self._status_data["non_tracked_articles_count"] += 1
            return

        with self._lock:
            self._status_data["articles_received_count"] += 1

        from src.data.data_storage import validate_article, store_articles, store_nlp_results
        from src.nlp.news_nlp_pipeline import process_articles_nlp

        store_fn = self.custom_store_fn or store_articles
        nlp_fn = self.custom_nlp_fn or process_articles_nlp
        nlp_store_fn = self.custom_nlp_store_fn or store_nlp_results

        for ticker in sorted(list(target_tickers)):
            norm = normalize_stream_article(item, target_ticker=ticker)
            if not norm:
                continue

            # Validate quality, structure, and lookahead constraints
            is_valid, issue = validate_article(norm)
            if not is_valid:
                logger.warning("Stream article validation failed for %s: %s", ticker, issue)
                continue

            # Persist to storage with deduplication
            added, dups, rejs = store_fn([norm])

            with self._lock:
                if dups > 0:
                    self._status_data["articles_duplicate_count"] += dups

                if added > 0:
                    self._status_data["articles_stored_count"] += added

                    # Execute internal Loughran-McDonald NLP pipeline on new article
                    nlp_df = pd.DataFrame()
                    daily_df = pd.DataFrame()
                    try:
                        new_df = pd.DataFrame([norm])
                        nlp_df, daily_df = nlp_fn(new_df)
                        nlp_store_fn(nlp_df, daily_df)
                    except Exception as e_nlp:
                        logger.error("Error executing NLP pipeline on streamed article: %s", e_nlp)

                    # Extract NLP enrichment fields for last-article telemetry
                    sent_lbl = "Neutral"
                    sent_score = 0.0
                    triage_score = 0.0
                    event_type = "General News"
                    if not nlp_df.empty:
                        row_0 = nlp_df.iloc[0]
                        sent_lbl = str(row_0.get("Sentiment_Label", "Neutral"))
                        sent_score = float(row_0.get("Sentiment_Score", 0.0))
                        triage_score = float(row_0.get("Triage_Score", 0.0))
                        event_type = str(row_0.get("Event_Type", "General News"))

                    self._status_data["last_article"] = {
                        "news_id": norm["News_ID"],
                        "ticker": ticker,
                        "headline": norm["Headline"][:120],
                        "published_at_utc": norm["Published_At_UTC"],
                        "received_at_utc": norm["Retrieved_At_UTC"],
                        "source": norm["Source"],
                        "sentiment_label": sent_lbl,
                        "sentiment_score": round(sent_score, 4),
                        "triage_score": round(triage_score, 4),
                        "event_type": event_type
                    }

                    logger.info("STREAM INGESTED: [%s] %s (Published: %s, Sent: %s, Event: %s)",
                                ticker, norm["Headline"][:70], norm["Published_At_UTC"], sent_lbl, event_type)

            self._save_status_file()


# Module-level singleton manager for application-wide access
_GLOBAL_STREAM_LISTENER: Optional[AlpacaStreamListener] = None
_GLOBAL_LOCK = threading.Lock()


def get_stream_listener(
    tickers: Optional[List[str]] = None,
    status_path: Optional[Path] = None,
    db_path: Optional[Path] = None
) -> AlpacaStreamListener:
    """Get or create the global AlpacaStreamListener singleton instance."""
    global _GLOBAL_STREAM_LISTENER
    with _GLOBAL_LOCK:
        if _GLOBAL_STREAM_LISTENER is None:
            _GLOBAL_STREAM_LISTENER = AlpacaStreamListener(
                tickers=tickers,
                status_path=status_path,
                db_path=db_path
            )
        return _GLOBAL_STREAM_LISTENER


def start_stream_listener() -> AlpacaStreamListener:
    """Ensure the global stream listener singleton is active and running."""
    listener = get_stream_listener()
    if not listener.is_alive():
        listener.start()
    return listener


def stop_stream_listener(timeout: float = 5.0, status_path: Optional[Path] = None) -> None:
    """Stop the global stream listener singleton if active."""
    global _GLOBAL_STREAM_LISTENER
    with _GLOBAL_LOCK:
        if _GLOBAL_STREAM_LISTENER is not None:
            _GLOBAL_STREAM_LISTENER.stop(timeout=timeout)
        else:
            target_path = Path(status_path) if status_path else DEFAULT_STATUS_PATH
            if target_path.exists():
                try:
                    with open(target_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    data["status"] = "STOPPED"
                    data["connected"] = False
                    data["authenticated"] = False
                    data["subscribed"] = False
                    data["active_retry_attempt"] = 0
                    data["reconnect_count"] = 0
                    with open(target_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)
                except Exception:
                    pass


def get_stream_status(
    status_path: Optional[Path] = None,
    stale_threshold_seconds: float = 30.0
) -> Dict[str, Any]:
    """
    Read the latest stream telemetry from active in-memory singleton or status JSON file.
    Designed for fast, non-blocking calls inside Streamlit dashboard reruns.
    Enforces stale heartbeat detection: If the status claims STREAMING but the recorded
    heartbeat is older than stale_threshold_seconds, the status is automatically degraded
    to OFFLINE with a clear diagnostic explanation.
    """
    global _GLOBAL_STREAM_LISTENER
    raw_status: Optional[Dict[str, Any]] = None

    if _GLOBAL_STREAM_LISTENER is not None and _GLOBAL_STREAM_LISTENER.is_alive():
        raw_status = _GLOBAL_STREAM_LISTENER.get_status()
    else:
        target_path = Path(status_path) if status_path else DEFAULT_STATUS_PATH
        if target_path.exists():
            try:
                with open(target_path, "r", encoding="utf-8") as f:
                    raw_status = json.load(f)
            except Exception:
                raw_status = None

    if raw_status is None:
        return {
            "status": "STOPPED",
            "connected": False,
            "authenticated": False,
            "subscribed": False,
            "endpoint": ALPACA_NEWS_WS_URL,
            "process_id": None,
            "connected_at_utc": None,
            "last_heartbeat_utc": None,
            "subscribed_tickers": SUPPORTED_TICKERS,
            "articles_received_count": 0,
            "articles_stored_count": 0,
            "articles_duplicate_count": 0,
            "non_tracked_articles_count": 0,
            "raw_messages_received_count": 0,
            "active_retry_attempt": 0,
            "total_reconnects": 0,
            "reconnect_count": 0,
            "last_article": None,
            "error_code": None,
            "error_message": None,
            "close_code": None,
            "close_reason": None,
            "is_stale": False,
            "heartbeat_age_seconds": None
        }

    status = dict(raw_status)
    if "total_reconnects" not in status:
        status["total_reconnects"] = status.get("reconnect_count", 0)
    if "active_retry_attempt" not in status:
        status["active_retry_attempt"] = 0

    # Calculate Heartbeat Age & Stale Heartbeat Detection
    last_hb = status.get("last_heartbeat_utc")
    hb_age: Optional[float] = None
    is_stale = False

    if last_hb:
        try:
            hb_clean = str(last_hb).replace(" UTC", "").strip()
            hb_dt = datetime.strptime(hb_clean, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            now_dt = datetime.now(timezone.utc)
            hb_age = max(0.0, (now_dt - hb_dt).total_seconds())
            if hb_age > stale_threshold_seconds:
                is_stale = True
        except Exception:
            pass

    status["heartbeat_age_seconds"] = round(hb_age, 1) if hb_age is not None else None
    status["is_stale"] = is_stale

    if is_stale and status.get("status") in ["STREAMING", "CONNECTING", "AUTHENTICATING", "SUBSCRIBING"]:
        status["status"] = "OFFLINE"
        status["connected"] = False
        status["authenticated"] = False
        status["subscribed"] = False
        status["error_message"] = f"Heartbeat is stale ({hb_age:.0f}s old > {stale_threshold_seconds}s limit). Listener process may be stopped."

    return status
