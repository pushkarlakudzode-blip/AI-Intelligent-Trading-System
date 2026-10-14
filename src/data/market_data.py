"""
Market Data Module for AI-Based Intelligent Trading System.

This module provides reusable functions to fetch, validate, and manage
historical market data using yfinance. Designed to support multiple assets/companies
and serve as the foundational data layer for downstream feature engineering,
machine learning models, and trading agents.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import yfinance as yf

# Configure logger for the market data module
logger = logging.getLogger(__name__)

# Standard required columns for market data
REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]


def validate_market_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate, clean, and standardize market data into a consistent format.

    Performs the following validation and cleaning steps:
    1. Checks for non-empty pandas DataFrame.
    2. Flattens MultiIndex columns if present.
    3. Moves 'Date' from index to column if necessary.
    4. Normalizes column names (case-insensitive mapping to Open, High, Low, Close, Volume).
    5. Confirms all required columns exist.
    6. Converts 'Date' to timezone-naive datetime64 and sorts chronologically.
    7. Removes duplicate dates keeping the first occurrence.
    8. Ensures numeric data types and reports/handles missing values via ffill and bfill.

    Args:
        df: Raw market data pandas DataFrame.

    Returns:
        pd.DataFrame: Cleaned and validated DataFrame with columns:
                      ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

    Raises:
        TypeError: If input is not a pandas DataFrame.
        ValueError: If DataFrame is empty or required columns are missing.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    data = df.copy()

    # Flatten MultiIndex columns if returned by yfinance
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # If 'Date' is in the index, reset index to make it a column
    has_date_col = any(str(c).lower() in ["date", "datetime", "timestamp"] for c in data.columns)
    if not has_date_col:
        if isinstance(data.index, pd.DatetimeIndex) or str(data.index.name).lower() in [
            "date",
            "datetime",
        ]:
            data = data.reset_index()
            if "index" in data.columns and "Date" not in data.columns:
                data = data.rename(columns={"index": "Date"})

    # Standardize column casing and names
    rename_mapping = {}
    for col in data.columns:
        col_str = str(col).strip().lower()
        if col_str in ["date", "datetime", "timestamp"]:
            rename_mapping[col] = "Date"
        elif col_str == "open":
            rename_mapping[col] = "Open"
        elif col_str == "high":
            rename_mapping[col] = "High"
        elif col_str == "low":
            rename_mapping[col] = "Low"
        elif col_str == "close":
            rename_mapping[col] = "Close"
        elif col_str == "volume":
            rename_mapping[col] = "Volume"

    data = data.rename(columns=rename_mapping)

    # Verify all required columns are present
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if missing_cols:
        raise ValueError(f"DataFrame is missing required market data columns: {missing_cols}")

    # Convert Date to datetime and ensure timezone-naive timestamps
    data["Date"] = pd.to_datetime(data["Date"])
    if data["Date"].dt.tz is not None:
        data["Date"] = data["Date"].dt.tz_localize(None)

    # Sort chronologically
    data = data.sort_values(by="Date", ascending=True)

    # Handle duplicate dates
    duplicate_count = data.duplicated(subset=["Date"]).sum()
    if duplicate_count > 0:
        logger.warning(
            f"Detected {duplicate_count} duplicate Date entries. Removing duplicates (keeping first)."
        )
        data = data.drop_duplicates(subset=["Date"], keep="first")

    # Cast numeric columns and handle missing values
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # Check for missing values
    missing_counts = data[REQUIRED_COLUMNS].isnull().sum()
    total_missing = missing_counts.sum()
    if total_missing > 0:
        missing_dict = {
            col: int(cnt) for col, cnt in missing_counts.items() if cnt > 0
        }
        logger.warning(
            f"Missing values detected: {missing_dict}. Applying forward-fill then backward-fill."
        )
        # In financial time series, ffill preserves the last known price, and bfill handles leading NaNs
        data[numeric_cols] = data[numeric_cols].ffill().bfill()

        # Drop any remaining unfillable rows (if all prices were NaN)
        remaining_missing = data[REQUIRED_COLUMNS].isnull().any(axis=1).sum()
        if remaining_missing > 0:
            logger.warning(
                f"Dropping {remaining_missing} rows where values could not be filled."
            )
            data = data.dropna(subset=REQUIRED_COLUMNS)

    # Format Volume cleanly as integer if values are whole numbers
    if (data["Volume"] % 1 == 0).all():
        data["Volume"] = data["Volume"].astype("int64")

    # Reorder columns to the standard format and reset range index
    data = data[REQUIRED_COLUMNS].reset_index(drop=True)
    return data


def fetch_market_data(
    ticker: str,
    start_date: Union[str, datetime, pd.Timestamp],
    end_date: Union[str, datetime, pd.Timestamp],
    auto_adjust: bool = False,
) -> pd.DataFrame:
    """
    Fetch historical market data for a given ticker symbol using yfinance.

    Accepts any valid stock ticker (e.g. AAPL, MSFT, GOOGL, NVDA, TSLA, RELIANCE.NS)
    and returns a clean, validated pandas DataFrame with standard OHLCV columns.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT', 'GOOGL').
        start_date: Start date for historical data (e.g., '2020-01-01' or datetime).
        end_date: End date for historical data (e.g., '2024-01-01' or datetime).
        auto_adjust: Whether to adjust OHLC for splits and dividends (default: False).

    Returns:
        pd.DataFrame: Cleaned DataFrame with columns:
                      ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

    Raises:
        ValueError: If ticker is invalid, date range is improper, or no data is found.
        ConnectionError: If network/API error prevents data retrieval.
    """
    if not ticker or not isinstance(ticker, str) or not ticker.strip():
        raise ValueError("A valid non-empty ticker symbol string must be provided.")

    clean_ticker = ticker.strip().upper()

    # Parse and validate dates
    try:
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
    except Exception as exc:
        raise ValueError(
            f"Invalid date format. Please provide valid dates (e.g. 'YYYY-MM-DD'). Error: {exc}"
        ) from exc

    if start_dt >= end_dt:
        raise ValueError(
            f"start_date ({start_date}) must be earlier than end_date ({end_date})."
        )

    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")

    logger.info(
        f"Fetching market data for ticker '{clean_ticker}' from {start_str} to {end_str}..."
    )

    try:
        # Fetch using yf.Ticker.history
        ticker_obj = yf.Ticker(clean_ticker)
        raw_df = ticker_obj.history(start=start_str, end=end_str, auto_adjust=auto_adjust)

        # Fallback to yf.download if history returned empty
        if raw_df is None or raw_df.empty:
            raw_df = yf.download(
                clean_ticker,
                start=start_str,
                end=end_str,
                auto_adjust=auto_adjust,
                progress=False,
            )

    except Exception as exc:
        raise ConnectionError(
            f"Failed to fetch market data for ticker '{clean_ticker}' from Yahoo Finance: {exc}"
        ) from exc

    if raw_df is None or raw_df.empty:
        raise ValueError(
            f"No market data returned for ticker '{clean_ticker}' between {start_str} and {end_str}. "
            "Please check the ticker symbol and verify the date range includes trading days."
        )

    # Validate, clean, and standardize the fetched data
    clean_df = validate_market_data(raw_df)
    return clean_df


def fetch_multiple_market_data(
    tickers: List[str],
    start_date: Union[str, datetime, pd.Timestamp],
    end_date: Union[str, datetime, pd.Timestamp],
    auto_adjust: bool = False,
    ignore_errors: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch and validate historical market data for multiple tickers.

    Args:
        tickers: List of stock ticker symbols (e.g., ['AAPL', 'MSFT', 'GOOGL']).
        start_date: Start date for historical data.
        end_date: End date for historical data.
        auto_adjust: Whether to adjust OHLC for splits and dividends.
        ignore_errors: If True, skips tickers that fail and continues with others;
                       if False, raises exception on first failure.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping ticker symbol to clean DataFrame.

    Raises:
        ValueError: If tickers list is empty.
    """
    if not tickers or not isinstance(tickers, (list, tuple)):
        raise ValueError("tickers must be a non-empty list or tuple of ticker symbols.")

    results: Dict[str, pd.DataFrame] = {}

    for sym in tickers:
        sym_clean = sym.strip().upper()
        try:
            results[sym_clean] = fetch_market_data(
                ticker=sym_clean,
                start_date=start_date,
                end_date=end_date,
                auto_adjust=auto_adjust,
            )
        except Exception as exc:
            if ignore_errors:
                logger.warning(f"Could not fetch data for ticker '{sym_clean}': {exc}")
            else:
                raise

    return results


def save_market_data(
    df: pd.DataFrame,
    filepath: Union[str, Path],
    index: bool = False,
) -> None:
    """
    Save validated market data to a CSV file.

    Args:
        df: Validated market data DataFrame.
        filepath: Destination file path (e.g., 'data/raw/AAPL.csv').
        index: Whether to write row index (default: False).
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    logger.info(f"Market data successfully saved to {path.resolve()}")


def load_market_data(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load and re-validate market data from a CSV file.

    Args:
        filepath: Path to the CSV file to load.

    Returns:
        pd.DataFrame: Cleaned and validated DataFrame.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")

    df = pd.read_csv(path)
    return validate_market_data(df)
