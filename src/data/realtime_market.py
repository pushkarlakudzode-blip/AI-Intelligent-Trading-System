"""
Real-Time & Intraday Market Data Ingestion Layer.

Fetches live snapshot, intraday, and historical extension market data using yfinance
across all 10 project equities. Implements timezone-aware UTC timestamps,
provider latency disclosure, graceful error degradation, and historical data extension.

Data is explicitly marked:
Data_Source = "yfinance"
Data_Status = "Delayed / Snapshot"
"""

import logging
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Generator, List, Optional, Union

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LIVE_MARKET_DIR = PROJECT_ROOT / "data" / "live" / "market"
DEFAULT_EXTENSION_PATH = LIVE_MARKET_DIR / "historical_market_extension.csv"

SUPPORTED_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "JNJ"]

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


def _validate_ticker(ticker: str) -> str:
    """Validate that the ticker is part of the 10-company target universe."""
    clean_ticker = ticker.strip().upper()
    if clean_ticker not in SUPPORTED_TICKERS:
        raise ValueError(f"Ticker '{clean_ticker}' is not in the supported universe: {SUPPORTED_TICKERS}")
    return clean_ticker


def get_current_quote(ticker: str) -> Dict[str, Union[str, float, int]]:
    """
    Fetch the most recent available price quote snapshot for a ticker.
    
    Returns a standardized dictionary containing:
    ['Ticker', 'Company_Name', 'Timestamp', 'Price', 'Open', 'High', 'Low', 'Close', 'Volume', 'Data_Source', 'Data_Status']
    """
    clean_ticker = _validate_ticker(ticker)
    now_utc = datetime.now(timezone.utc).isoformat()
    
    default_error_response = {
        "Ticker": clean_ticker,
        "Company_Name": COMPANY_NAMES.get(clean_ticker, clean_ticker),
        "Timestamp": now_utc,
        "Price": 0.0,
        "Open": 0.0,
        "High": 0.0,
        "Low": 0.0,
        "Close": 0.0,
        "Volume": 0,
        "Data_Source": "yfinance",
        "Data_Status": "Unavailable / Error"
    }

    try:
        yf_ticker = yf.Ticker(clean_ticker)
        fast_info = getattr(yf_ticker, "fast_info", None)
        
        if fast_info is not None:
            last_price = float(getattr(fast_info, "last_price", 0.0) or 0.0)
            open_price = float(getattr(fast_info, "open", last_price) or last_price)
            high_price = float(getattr(fast_info, "day_high", last_price) or last_price)
            low_price = float(getattr(fast_info, "day_low", last_price) or last_price)
            prev_close = float(getattr(fast_info, "previous_close", last_price) or last_price)
            volume = int(getattr(fast_info, "last_volume", 0) or 0)
            
            if last_price > 0:
                return {
                    "Ticker": clean_ticker,
                    "Company_Name": COMPANY_NAMES.get(clean_ticker, clean_ticker),
                    "Timestamp": now_utc,
                    "Price": round(last_price, 4),
                    "Open": round(open_price, 4),
                    "High": round(high_price, 4),
                    "Low": round(low_price, 4),
                    "Close": round(last_price, 4),
                    "Previous_Close": round(prev_close, 4),
                    "Volume": volume,
                    "Data_Source": "yfinance",
                    "Data_Status": "Delayed / Snapshot"
                }

        # Fallback to history 1d
        hist = yf_ticker.history(period="1d", interval="1m")
        if not hist.empty:
            last_row = hist.iloc[-1]
            last_ts = hist.index[-1]
            if last_ts.tzinfo is None:
                last_ts = last_ts.tz_localize("UTC")
            else:
                last_ts = last_ts.tz_convert("UTC")
                
            return {
                "Ticker": clean_ticker,
                "Company_Name": COMPANY_NAMES.get(clean_ticker, clean_ticker),
                "Timestamp": last_ts.isoformat(),
                "Price": round(float(last_row["Close"]), 4),
                "Open": round(float(last_row["Open"]), 4),
                "High": round(float(last_row["High"]), 4),
                "Low": round(float(last_row["Low"]), 4),
                "Close": round(float(last_row["Close"]), 4),
                "Volume": int(last_row["Volume"]),
                "Data_Source": "yfinance",
                "Data_Status": "Delayed / Snapshot"
            }
            
        logger.warning(f"Empty quote returned for {clean_ticker}")
        default_error_response["Data_Status"] = "Market Closed / No Data"
        return default_error_response

    except Exception as e:
        logger.error(f"Error retrieving quote for {clean_ticker}: {e}")
        default_error_response["Data_Status"] = f"Error: {type(e).__name__}"
        return default_error_response


def get_latest_market_data(ticker: str) -> pd.DataFrame:
    """
    Fetch the latest daily OHLCV bar for a given ticker.
    """
    clean_ticker = _validate_ticker(ticker)
    try:
        yf_ticker = yf.Ticker(clean_ticker)
        df = yf_ticker.history(period="5d", interval="1d")
        if df.empty:
            return pd.DataFrame()
            
        # Clean index and columns
        df = df.reset_index()
        df["Ticker"] = clean_ticker
        df["Company_Name"] = COMPANY_NAMES.get(clean_ticker, clean_ticker)
        
        # Ensure UTC datetime
        date_col = "Date" if "Date" in df.columns else ("Datetime" if "Datetime" in df.columns else df.columns[0])
        df["Timestamp"] = pd.to_datetime(df[date_col], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        df["Date"] = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")
        
        df["Data_Source"] = "yfinance"
        df["Data_Status"] = "Delayed / Snapshot"
        
        cols = ["Date", "Timestamp", "Ticker", "Company_Name", "Open", "High", "Low", "Close", "Volume", "Data_Source", "Data_Status"]
        return df[[c for c in cols if c in df.columns]]
    except Exception as e:
        logger.error(f"Failed to fetch latest market data for {clean_ticker}: {e}")
        return pd.DataFrame()


def get_intraday_market_data(ticker: str, interval: str = "1m", period: str = "1d") -> pd.DataFrame:
    """
    Fetch intraday market series for a given ticker and resolution (e.g. interval='1m', '5m', '15m').
    """
    clean_ticker = _validate_ticker(ticker)
    try:
        yf_ticker = yf.Ticker(clean_ticker)
        df = yf_ticker.history(period=period, interval=interval)
        if df.empty:
            return pd.DataFrame()
            
        df = df.reset_index()
        df["Ticker"] = clean_ticker
        df["Company_Name"] = COMPANY_NAMES.get(clean_ticker, clean_ticker)
        
        date_col = "Datetime" if "Datetime" in df.columns else ("Date" if "Date" in df.columns else df.columns[0])
        df["Timestamp"] = pd.to_datetime(df[date_col], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        df["Data_Source"] = "yfinance"
        df["Data_Status"] = "Delayed Intraday"
        
        cols = ["Timestamp", "Ticker", "Company_Name", "Open", "High", "Low", "Close", "Volume", "Data_Source", "Data_Status"]
        return df[[c for c in cols if c in df.columns]]
    except Exception as e:
        logger.error(f"Failed to fetch intraday data for {clean_ticker}: {e}")
        return pd.DataFrame()


def stream_market_data(
    ticker: str,
    interval_seconds: float = 2.0,
    max_ticks: int = 5
) -> Generator[Dict[str, Union[str, float, int]], None, None]:
    """
    Streaming generator emulating live tick delivery via yfinance polling.
    Yields successive quotes up to max_ticks.
    """
    clean_ticker = _validate_ticker(ticker)
    for tick_idx in range(max_ticks):
        quote = get_current_quote(clean_ticker)
        quote["Tick_Index"] = tick_idx + 1
        yield quote
        if tick_idx < max_ticks - 1:
            time.sleep(interval_seconds)


def fetch_historical_market_extension(
    tickers: Optional[List[str]] = None,
    start_date: str = "2024-12-31",
    end_date: Optional[str] = None,
    save_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Fetch historical daily market extension from the end of the academic dataset forward.
    
    IMPORTANT:
    - Does NOT modify data/raw/multi_company_stock_data.csv.
    - Saves separately to data/live/market/historical_market_extension.csv.
    - Validates OHLC integrity (High >= max(Open, Close), Low <= min(Open, Close), Volume >= 0).
    - Deduplicates on (Ticker, Date) and sorts chronologically.
    """
    if tickers is None:
        tickers = SUPPORTED_TICKERS
    else:
        tickers = [_validate_ticker(t) for t in tickers]
        
    if end_date is None:
        end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
    LIVE_MARKET_DIR.mkdir(parents=True, exist_ok=True)
    target_path = save_path or DEFAULT_EXTENSION_PATH
    
    all_dfs = []
    
    for ticker in tickers:
        try:
            yf_ticker = yf.Ticker(ticker)
            df = yf_ticker.history(start=start_date, end=end_date, interval="1d", auto_adjust=False)
            if df.empty:
                logger.info(f"No new historical extension rows for {ticker} between {start_date} and {end_date}")
                continue
                
            df = df.reset_index()
            date_col = "Date" if "Date" in df.columns else df.columns[0]
            df["Date"] = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")
            df["Ticker"] = ticker
            
            # Select and order required columns
            req_cols = ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]
            df = df[[c for c in req_cols if c in df.columns]].copy()
            
            # Data quality validation
            # 1. Non-negative prices
            valid_mask = (df["Open"] > 0) & (df["High"] > 0) & (df["Low"] > 0) & (df["Close"] > 0) & (df["Volume"] >= 0)
            df = df[valid_mask].copy()
            
            # 2. High/Low boundary checks
            df["High"] = df[["Open", "High", "Close"]].max(axis=1)
            df["Low"] = df[["Open", "Low", "Close"]].min(axis=1)
            
            all_dfs.append(df)
            
        except Exception as e:
            logger.error(f"Error fetching historical extension for {ticker}: {e}")
            
    if not all_dfs:
        logger.info("No extension data fetched.")
        return pd.DataFrame()
        
    extension_df = pd.concat(all_dfs, ignore_index=True)
    extension_df["Data_Source"] = "yfinance"
    extension_df["Data_Status"] = "Historical Extension"
    
    # Deduplicate and sort
    extension_df = extension_df.drop_duplicates(subset=["Ticker", "Date"]).sort_values(by=["Ticker", "Date"]).reset_index(drop=True)
    
    # Save to dedicated live location
    extension_df.to_csv(target_path, index=False)
    logger.info(f"Saved {len(extension_df)} historical extension records to {target_path}")
    
    return extension_df


def refresh_market_data(
    tickers: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None
) -> Dict[str, Union[bool, int, Dict]]:
    """
    Refresh the latest daily snapshot bars for supported tickers,
    merge with existing extension storage, and preserve data integrity.
    
    Returns:
        Dict: Telemetry summary:
        ['success', 'refreshed_count', 'total_requested', 'details']
    """
    target_tickers = tickers or SUPPORTED_TICKERS
    target_path = Path(save_path) if save_path else DEFAULT_EXTENSION_PATH
    
    LIVE_MARKET_DIR.mkdir(parents=True, exist_ok=True)
    
    new_dfs = []
    details = {}
    
    for ticker in target_tickers:
        clean_ticker = _validate_ticker(ticker)
        try:
            yf_ticker = yf.Ticker(clean_ticker)
            df = yf_ticker.history(period="5d", interval="1d", auto_adjust=False)
            if df.empty:
                details[clean_ticker] = "No data returned"
                continue
                
            df = df.reset_index()
            date_col = "Date" if "Date" in df.columns else df.columns[0]
            df["Date"] = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")
            df["Ticker"] = clean_ticker
            
            req_cols = ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]
            df = df[[c for c in req_cols if c in df.columns]].copy()
            
            # Quality checks
            valid_mask = (df["Open"] > 0) & (df["High"] > 0) & (df["Low"] > 0) & (df["Close"] > 0) & (df["Volume"] >= 0)
            df = df[valid_mask].copy()
            df["High"] = df[["Open", "High", "Close"]].max(axis=1)
            df["Low"] = df[["Open", "Low", "Close"]].min(axis=1)
            
            if not df.empty:
                new_dfs.append(df)
                details[clean_ticker] = "SUCCESS"
            else:
                details[clean_ticker] = "Filtered out (invalid values)"
        except Exception as e:
            logger.error(f"Error refreshing market data for {clean_ticker}: {e}")
            details[clean_ticker] = f"Error: {e}"
            
    refreshed_cnt = sum(1 for status in details.values() if status == "SUCCESS")
    
    if new_dfs:
        combined_dfs = []
        if target_path.exists():
            try:
                existing_df = pd.read_csv(target_path)
                combined_dfs.append(existing_df)
            except Exception as e:
                logger.warning(f"Could not read existing market extension: {e}")
                
        combined_dfs.extend(new_dfs)
        merged_df = pd.concat(combined_dfs, ignore_index=True)
        merged_df["Data_Source"] = "yfinance"
        merged_df["Data_Status"] = "Historical Extension"
        merged_df = merged_df.drop_duplicates(subset=["Ticker", "Date"]).sort_values(by=["Ticker", "Date"]).reset_index(drop=True)
        merged_df.to_csv(target_path, index=False)
        logger.info(f"Updated market extension at {target_path} with {len(merged_df)} total records.")
        
    return {
        "success": refreshed_cnt == len(target_tickers),
        "refreshed_count": refreshed_cnt,
        "total_requested": len(target_tickers),
        "details": details
    }

