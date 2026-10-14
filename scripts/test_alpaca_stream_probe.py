"""
Alpaca Real-Time News WebSocket Handshake Probe.

Connects to the documented Alpaca news WebSocket endpoint:
    wss://stream.data.alpaca.markets/v1beta1/news
Performs the documented authentication handshake, subscribes to AAPL news,
and listens for up to 15 seconds to evaluate real-time streaming availability.

STRICT ISOLATION & SAFETY RULES:
- Read-only probe: Zero writes to disk or databases.
- Never prints or logs API keys, secret keys, or authentication payloads.
- Does not modify existing historical data, research, models, or dashboard.
- Zero Alpha Vantage requests.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("alpaca_probe")

ALPACA_NEWS_WS_URL = "wss://stream.data.alpaca.markets/v1beta1/news"
TARGET_TICKER = "AAPL"
LISTEN_DURATION_SECONDS = 15.0


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


def sanitize_message(msg: str, key_id: Optional[str] = None, secret_key: Optional[str] = None) -> str:
    """Sanitize message to ensure credentials never appear in output."""
    res = str(msg)
    if key_id and len(key_id) > 4:
        res = res.replace(key_id, "[REDACTED_KEY_ID]")
    if secret_key and len(secret_key) > 4:
        res = res.replace(secret_key, "[REDACTED_SECRET_KEY]")
    return res


def run_probe() -> Dict[str, Any]:
    """Execute the isolated Alpaca news WebSocket probe."""
    result: Dict[str, Any] = {
        "endpoint": ALPACA_NEWS_WS_URL,
        "credentials_configured": False,
        "connected": False,
        "authenticated": False,
        "subscribed": False,
        "error_code": None,
        "error_message": None,
        "messages_received_count": 0,
        "headlines_received_count": 0,
        "headlines_sample": [],
        "listen_duration_seconds": 0.0,
        "clean_disconnect": False
    }

    key_id, secret_key = get_alpaca_credentials()
    if not key_id or not secret_key:
        result["error_message"] = "ALPACA_API_KEY_ID or ALPACA_API_SECRET_KEY is not configured in .env."
        logger.error(result["error_message"])
        return result

    result["credentials_configured"] = True
    logger.info("Alpaca credentials detected (Key prefix: %s..., Key len: %d, Secret len: %d)",
                key_id[:2], len(key_id), len(secret_key))

    # Import websockets sync client
    try:
        from websockets.sync.client import connect as ws_connect
    except ImportError as e:
        result["error_message"] = f"websockets library not installed or incompatible: {e}"
        logger.error(result["error_message"])
        return result

    logger.info("Connecting to %s ...", ALPACA_NEWS_WS_URL)
    ws = None
    try:
        ws = ws_connect(
            ALPACA_NEWS_WS_URL,
            open_timeout=10,
            close_timeout=5
        )
        logger.info("TCP/TLS connection established. Awaiting server greeting...")

        # 1. Receive Greeting
        try:
            greeting_raw = ws.recv(timeout=5.0)
            logger.info("Server greeting received: %s", sanitize_message(str(greeting_raw), key_id, secret_key))
            greeting_data = json.loads(greeting_raw) if isinstance(greeting_raw, str) else json.loads(greeting_raw.decode("utf-8"))
            if isinstance(greeting_data, list) and len(greeting_data) > 0:
                first_msg = greeting_data[0]
                if first_msg.get("T") == "success" and first_msg.get("msg") == "connected":
                    result["connected"] = True
                    logger.info("Connected to Alpaca news stream successfully.")
                else:
                    logger.warning("Unexpected greeting content: %s", first_msg)
            elif isinstance(greeting_data, dict):
                if greeting_data.get("T") == "success" and greeting_data.get("msg") == "connected":
                    result["connected"] = True
        except TimeoutError:
            result["error_message"] = "Timed out waiting for initial server greeting."
            logger.error(result["error_message"])
            return result

        # 2. Authenticate
        logger.info("Sending authentication payload...")
        auth_payload = json.dumps({
            "action": "auth",
            "key": key_id,
            "secret": secret_key
        })
        ws.send(auth_payload)

        try:
            auth_resp_raw = ws.recv(timeout=5.0)
            auth_resp_str = sanitize_message(str(auth_resp_raw), key_id, secret_key)
            logger.info("Auth response received: %s", auth_resp_str)
            auth_data = json.loads(auth_resp_raw) if isinstance(auth_resp_raw, str) else json.loads(auth_resp_raw.decode("utf-8"))

            auth_items = auth_data if isinstance(auth_data, list) else [auth_data]
            for item in auth_items:
                msg_type = item.get("T")
                msg_text = item.get("msg")
                if msg_type == "success" and msg_text == "authenticated":
                    result["authenticated"] = True
                    logger.info("AUTHENTICATION SUCCEEDED: Access granted to Alpaca news stream.")
                elif msg_type == "error":
                    result["error_code"] = item.get("code")
                    result["error_message"] = sanitize_message(item.get("msg", "Unknown auth error"), key_id, secret_key)
                    logger.error("AUTHENTICATION REJECTED: code=%s, msg=%s", result["error_code"], result["error_message"])
        except TimeoutError:
            result["error_message"] = "Timed out waiting for authentication response."
            logger.error(result["error_message"])
            return result

        # 3. Subscribe (if authenticated)
        if result["authenticated"]:
            logger.info("Subscribing to '%s' news...", TARGET_TICKER)
            sub_payload = json.dumps({
                "action": "subscribe",
                "news": [TARGET_TICKER]
            })
            ws.send(sub_payload)

            try:
                sub_resp_raw = ws.recv(timeout=5.0)
                sub_resp_str = sanitize_message(str(sub_resp_raw), key_id, secret_key)
                logger.info("Subscription response: %s", sub_resp_str)
                sub_data = json.loads(sub_resp_raw) if isinstance(sub_resp_raw, str) else json.loads(sub_resp_raw.decode("utf-8"))
                sub_items = sub_data if isinstance(sub_data, list) else [sub_data]
                for item in sub_items:
                    if item.get("T") == "subscription":
                        result["subscribed"] = True
                        logger.info("SUBSCRIPTION SUCCEEDED: Active news channels: %s", item.get("news"))
                    elif item.get("T") == "error":
                        result["error_code"] = item.get("code")
                        result["error_message"] = sanitize_message(item.get("msg", "Subscription error"), key_id, secret_key)
                        logger.error("SUBSCRIPTION ERROR: code=%s, msg=%s", result["error_code"], result["error_message"])
            except TimeoutError:
                result["error_message"] = "Timed out waiting for subscription response."
                logger.error(result["error_message"])

        # 4. Listen for up to LISTEN_DURATION_SECONDS
        if result["subscribed"]:
            logger.info("Listening for incoming news messages for up to %.1f seconds...", LISTEN_DURATION_SECONDS)
            start_time = time.time()
            deadline = start_time + LISTEN_DURATION_SECONDS

            while time.time() < deadline:
                remaining = max(0.1, deadline - time.time())
                try:
                    msg_raw = ws.recv(timeout=min(2.0, remaining))
                    result["messages_received_count"] += 1
                    msg_str = sanitize_message(str(msg_raw), key_id, secret_key)
                    msg_data = json.loads(msg_raw) if isinstance(msg_raw, str) else json.loads(msg_raw.decode("utf-8"))
                    items = msg_data if isinstance(msg_data, list) else [msg_data]

                    for it in items:
                        if it.get("T") == "n":
                            result["headlines_received_count"] += 1
                            headline = it.get("headline", "")
                            pub_time = it.get("created_at", "")
                            symbols = it.get("symbols", [])
                            logger.info("REAL HEADLINE RECEIVED: [%s] %s (%s)",
                                        ",".join(symbols), headline[:80], pub_time)
                            if len(result["headlines_sample"]) < 3:
                                result["headlines_sample"].append({
                                    "headline": headline[:100],
                                    "created_at": pub_time,
                                    "symbols": symbols
                                })
                        else:
                            logger.info("Incoming non-news message: %s", str(it)[:100])
                except TimeoutError:
                    # Normal timeout while waiting for news items
                    continue
                except Exception as e_recv:
                    logger.warning("Error receiving stream frame: %s", sanitize_message(str(e_recv), key_id, secret_key))
                    break

            result["listen_duration_seconds"] = round(time.time() - start_time, 2)
            logger.info("Listening window concluded. Duration: %.2fs, Messages: %d, Headlines: %d",
                        result["listen_duration_seconds"],
                        result["messages_received_count"],
                        result["headlines_received_count"])

    except Exception as e_conn:
        result["error_message"] = sanitize_message(str(e_conn), key_id, secret_key)
        logger.error("Connection exception: %s", result["error_message"])
    finally:
        if ws is not None:
            try:
                ws.close()
                result["clean_disconnect"] = True
                logger.info("WebSocket connection closed cleanly.")
            except Exception as e_close:
                logger.warning("Exception during close: %s", e_close)

    return result


if __name__ == "__main__":
    logger.info("=== ALPACA REAL-TIME NEWS WEBSOCKET PROBE START ===")
    res = run_probe()
    logger.info("=== ALPACA REAL-TIME NEWS WEBSOCKET PROBE RESULT ===")
    print("\nPROBE SUMMARY JSON:")
    print(json.dumps(res, indent=2))
