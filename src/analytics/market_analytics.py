"""
Real Market Analytics Module.

Calculates technical indicators and analytical summary metrics for real market data
across the 10 target companies using historical extensions and real-time/delayed snapshots.

Indicators calculated:
- Latest price, price change, and daily return
- Moving averages: SMA 5, SMA 10, EMA 12, EMA 26
- Volatility: 5-day rolling standard deviation of returns
- High/low trading range
- Volume and 20-day average volume
- Trend indication: Bullish / Bearish / Neutral

Note: These are analytical metrics only, explicitly labeled as 'Delayed / Snapshot',
and do NOT constitute financial or trading recommendations.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MARKET_EXTENSION_PATH = PROJECT_ROOT / "data" / "live" / "market" / "historical_market_extension.csv"

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


def load_real_market_data(file_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Load real daily market data from the live extension storage."""
    target_path = Path(file_path) if file_path else DEFAULT_MARKET_EXTENSION_PATH
    if not target_path.exists():
        logger.warning(f"Market extension file not found at {target_path}")
        return pd.DataFrame()
        
    df = pd.read_csv(target_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by=["Ticker", "Date"]).reset_index(drop=True)
    return df


def compute_ticker_market_analytics(ticker_df: pd.DataFrame) -> Optional[Dict[str, Union[str, float, int]]]:
    """
    Compute market analytics for a single ticker's time series.
    Requires at least 2 rows of historical price data.
    """
    if ticker_df.empty or len(ticker_df) < 2:
        return None
        
    df = ticker_df.sort_values(by="Date").copy()
    ticker = df["Ticker"].iloc[-1]
    company_name = COMPANY_NAMES.get(ticker, ticker)
    
    # Price returns
    df["Daily_Return"] = df["Close"].pct_change()
    
    # Moving averages
    df["SMA_5"] = df["Close"].rolling(window=5, min_periods=2).mean()
    df["SMA_10"] = df["Close"].rolling(window=10, min_periods=3).mean()
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()
    
    # 5-day rolling volatility
    df["Volatility_5"] = df["Daily_Return"].rolling(window=5, min_periods=2).std()
    
    # Volume rolling average
    df["Avg_Volume_20"] = df["Volume"].rolling(window=20, min_periods=3).mean()
    
    # High-Low Range
    df["High_Low_Range"] = df["High"] - df["Low"]
    
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    latest_price = float(latest["Close"])
    prev_price = float(prev["Close"])
    price_change = float(latest_price - prev_price)
    daily_ret = float(latest["Daily_Return"]) if pd.notna(latest["Daily_Return"]) else 0.0
    
    sma_5 = float(latest["SMA_5"]) if pd.notna(latest["SMA_5"]) else latest_price
    sma_10 = float(latest["SMA_10"]) if pd.notna(latest["SMA_10"]) else latest_price
    ema_12 = float(latest["EMA_12"]) if pd.notna(latest["EMA_12"]) else latest_price
    ema_26 = float(latest["EMA_26"]) if pd.notna(latest["EMA_26"]) else latest_price
    vol_5 = float(latest["Volatility_5"]) if pd.notna(latest["Volatility_5"]) else 0.0
    hl_range = float(latest["High_Low_Range"]) if pd.notna(latest["High_Low_Range"]) else 0.0
    avg_vol_20 = int(latest["Avg_Volume_20"]) if pd.notna(latest["Avg_Volume_20"]) else int(latest["Volume"])
    
    # Basic technical trend indication (analytical classification, NOT a trading recommendation)
    if latest_price > sma_10 and sma_5 > sma_10:
        trend = "Bullish"
    elif latest_price < sma_10 and sma_5 < sma_10:
        trend = "Bearish"
    else:
        trend = "Neutral"
        
    as_of_date = latest["Date"].strftime("%Y-%m-%d") if hasattr(latest["Date"], "strftime") else str(latest["Date"])[:10]
    now_utc = datetime.now(timezone.utc).isoformat()
    
    return {
        "Ticker": ticker,
        "Company_Name": company_name,
        "As_Of_Date": as_of_date,
        "Latest_Price": round(latest_price, 2),
        "Price_Change": round(price_change, 2),
        "Daily_Return": round(daily_ret, 4),
        "SMA_5": round(sma_5, 2),
        "SMA_10": round(sma_10, 2),
        "EMA_12": round(ema_12, 2),
        "EMA_26": round(ema_26, 2),
        "Volatility_5": round(vol_5, 4),
        "High_Low_Range": round(hl_range, 2),
        "Volume": int(latest["Volume"]),
        "Average_Volume_20": avg_vol_20,
        "Trend_Indication": trend,
        "Data_Source": "yfinance",
        "Data_Status": "Delayed / Snapshot",
        "Generated_At_UTC": now_utc
    }


def compute_real_market_summary(market_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Compute real market analytics summary across all supported tickers.
    
    Args:
        market_df: Optional DataFrame of daily market data. If None, loads from live storage.
        
    Returns:
        DataFrame with one summary row per company.
    """
    if market_df is None or market_df.empty:
        market_df = load_real_market_data()
        
    if market_df.empty:
        logger.warning("No market data available to compute summary.")
        return pd.DataFrame()
        
    summary_rows = []
    for ticker in SUPPORTED_TICKERS:
        t_sub = market_df[market_df["Ticker"] == ticker]
        metrics = compute_ticker_market_analytics(t_sub)
        if metrics:
            summary_rows.append(metrics)
            
    res_df = pd.DataFrame(summary_rows)
    return res_df
