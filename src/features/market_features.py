"""
Market Features Module for AI-Based Intelligent Trading System.

This module provides common, reusable feature engineering functions designed to
operate on clean market data (OHLCV) produced by src/data/market_data.py.
It calculates price action features, candlestick geometries, moving averages,
rolling volatility, and target classification labels without future-data leakage.
"""

from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd


def calculate_daily_return(df: pd.DataFrame, price_col: str = "Close") -> pd.Series:
    """
    Calculate daily percentage return based on Close price.

    Formula:
        Daily_Return_t = (Close_t - Close_{t-1}) / Close_{t-1}

    Args:
        df: DataFrame containing the price column.
        price_col: Column name to calculate return on (default: 'Close').

    Returns:
        pd.Series: Daily percentage return series.
    """
    if price_col not in df.columns:
        raise ValueError(f"Required price column '{price_col}' not found in DataFrame.")
    return df[price_col].pct_change()


def calculate_price_range(
    df: pd.DataFrame, high_col: str = "High", low_col: str = "Low"
) -> pd.Series:
    """
    Calculate the intraday price range (High - Low).

    Args:
        df: DataFrame containing High and Low columns.
        high_col: Column name for intraday high (default: 'High').
        low_col: Column name for intraday low (default: 'Low').

    Returns:
        pd.Series: Intraday price range series.
    """
    if high_col not in df.columns or low_col not in df.columns:
        raise ValueError(f"Columns '{high_col}' and '{low_col}' must exist in DataFrame.")
    return df[high_col] - df[low_col]


def calculate_candle_body(
    df: pd.DataFrame, close_col: str = "Close", open_col: str = "Open"
) -> pd.Series:
    """
    Calculate candlestick real body length (Close - Open).

    A positive value indicates a bullish (green) candle, while a negative
    value indicates a bearish (red) candle.

    Args:
        df: DataFrame containing Close and Open columns.
        close_col: Column name for close price (default: 'Close').
        open_col: Column name for open price (default: 'Open').

    Returns:
        pd.Series: Candlestick real body values.
    """
    if close_col not in df.columns or open_col not in df.columns:
        raise ValueError(f"Columns '{close_col}' and '{open_col}' must exist in DataFrame.")
    return df[close_col] - df[open_col]


def calculate_upper_shadow(
    df: pd.DataFrame,
    high_col: str = "High",
    open_col: str = "Open",
    close_col: str = "Close",
) -> pd.Series:
    """
    Calculate candlestick upper shadow (wick) length: High - max(Open, Close).

    Args:
        df: DataFrame containing High, Open, and Close columns.
        high_col: Column name for high price (default: 'High').
        open_col: Column name for open price (default: 'Open').
        close_col: Column name for close price (default: 'Close').

    Returns:
        pd.Series: Candlestick upper shadow lengths.
    """
    for col in [high_col, open_col, close_col]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame.")
    return df[high_col] - df[[open_col, close_col]].max(axis=1)


def calculate_lower_shadow(
    df: pd.DataFrame,
    low_col: str = "Low",
    open_col: str = "Open",
    close_col: str = "Close",
) -> pd.Series:
    """
    Calculate candlestick lower shadow (tail) length: min(Open, Close) - Low.

    Args:
        df: DataFrame containing Low, Open, and Close columns.
        low_col: Column name for low price (default: 'Low').
        open_col: Column name for open price (default: 'Open').
        close_col: Column name for close price (default: 'Close').

    Returns:
        pd.Series: Candlestick lower shadow lengths.
    """
    for col in [low_col, open_col, close_col]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame.")
    return df[[open_col, close_col]].min(axis=1) - df[low_col]


def calculate_sma(
    df: pd.DataFrame,
    window: int,
    price_col: str = "Close",
) -> pd.Series:
    """
    Calculate Simple Moving Average (SMA) over a configurable rolling window.

    Args:
        df: DataFrame containing the price column.
        window: Integer window size (e.g., 5, 10, 20).
        price_col: Price column to compute the moving average on (default: 'Close').

    Returns:
        pd.Series: Simple moving average series.
    """
    if price_col not in df.columns:
        raise ValueError(f"Required price column '{price_col}' not found in DataFrame.")
    if not isinstance(window, int) or window <= 0:
        raise ValueError("Window must be a positive integer.")
    return df[price_col].rolling(window=window).mean()


def calculate_volatility(
    df: pd.DataFrame,
    window: int = 5,
    return_col: Optional[str] = None,
    price_col: str = "Close",
) -> pd.Series:
    """
    Calculate rolling historical volatility based on Daily Return.

    Args:
        df: DataFrame containing either return or price columns.
        window: Rolling window size (default: 5).
        return_col: Name of precomputed return column if available.
                    If None or missing, computed directly from `price_col`.
        price_col: Price column used if return_col is not supplied (default: 'Close').

    Returns:
        pd.Series: Rolling standard deviation of daily returns.
    """
    if not isinstance(window, int) or window <= 0:
        raise ValueError("Window must be a positive integer.")

    if return_col is not None and return_col in df.columns:
        returns = df[return_col]
    else:
        returns = calculate_daily_return(df, price_col=price_col)

    return returns.rolling(window=window).std()


def _compute_single_asset_features(
    df: pd.DataFrame,
    sma_windows: Sequence[int] = (5, 10),
    volatility_window: int = 5,
    drop_na: bool = False,
) -> pd.DataFrame:
    required_cols = ["Open", "High", "Low", "Close"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for feature engineering: {missing}")

    data = df.copy()

    # 1. Daily return
    data["Daily_Return"] = calculate_daily_return(data, price_col="Close")

    # 2. Price range
    data["Price_Range"] = calculate_price_range(data, high_col="High", low_col="Low")

    # 3. Candle body
    data["Candle_Body"] = calculate_candle_body(data, close_col="Close", open_col="Open")

    # 4. Upper shadow
    data["Upper_Shadow"] = calculate_upper_shadow(
        data, high_col="High", open_col="Open", close_col="Close"
    )

    # 5. Lower shadow
    data["Lower_Shadow"] = calculate_lower_shadow(
        data, low_col="Low", open_col="Open", close_col="Close"
    )

    # 6. Simple Moving Averages (SMA-5, SMA-10, etc.)
    for w in sma_windows:
        data[f"SMA_{w}"] = calculate_sma(data, window=w, price_col="Close")

    # 7. Rolling Volatility
    data[f"Volatility_{volatility_window}"] = calculate_volatility(
        data, window=volatility_window, return_col="Daily_Return"
    )

    if drop_na:
        data = data.dropna().reset_index(drop=True)

    return data


def create_market_features(
    df: pd.DataFrame,
    sma_windows: Sequence[int] = (5, 10),
    volatility_window: int = 5,
    drop_na: bool = False,
) -> pd.DataFrame:
    """
    Apply common feature engineering pipeline to a market DataFrame.

    Generates price action, candlestick, moving average, and volatility features
    while preserving original OHLCV data. If 'Ticker' column is present, feature
    calculations are performed independently within each ticker group to guarantee
    zero cross-company leakage.

    Engineered columns added:
    - Daily_Return: Daily percentage price return.
    - Price_Range: High - Low intraday range.
    - Candle_Body: Close - Open real body.
    - Upper_Shadow: High - max(Open, Close).
    - Lower_Shadow: min(Open, Close) - Low.
    - SMA_{w}: Simple Moving Average for each window in `sma_windows` (default 5, 10).
    - Volatility_{w}: Rolling volatility based on Daily Return (default 5).

    Args:
        df: Clean market data DataFrame containing Date, Open, High, Low, Close, Volume.
        sma_windows: Sequence of integer window sizes for SMAs (default: (5, 10)).
        volatility_window: Integer window size for rolling volatility (default: 5).
        drop_na: If True, drops initial rows with NaN caused by rolling windows.
                 If False, preserves all rows (default: False).

    Returns:
        pd.DataFrame: New DataFrame containing original columns plus engineered features.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if "Ticker" in df.columns:
        feature_dfs = []
        for _, grp in df.groupby("Ticker", sort=False):
            sub = grp.copy()
            if "Date" in sub.columns:
                sub = sub.sort_values("Date")
            feat_df = _compute_single_asset_features(
                sub,
                sma_windows=sma_windows,
                volatility_window=volatility_window,
                drop_na=drop_na,
            )
            feature_dfs.append(feat_df)
        res = pd.concat(feature_dfs, ignore_index=True)
        if "Date" in res.columns:
            res = res.sort_values(by=["Ticker", "Date"]).reset_index(drop=True)
        return res

    return _compute_single_asset_features(
        df,
        sma_windows=sma_windows,
        volatility_window=volatility_window,
        drop_na=drop_na,
    )


def _compute_single_asset_target(
    df: pd.DataFrame,
    close_col: str = "Close",
    target_col: str = "Target_Next_Day",
    drop_last_row: bool = False,
) -> pd.DataFrame:
    if close_col not in df.columns:
        raise ValueError(f"Required close column '{close_col}' not found in DataFrame.")

    data = df.copy()

    # Shift Close price backwards by 1 to get next day's Close
    next_day_close = data[close_col].shift(-1)

    # 1 if next day Close > today Close, 0 if <=, NaN for the final row
    data[target_col] = np.where(
        next_day_close.isna(),
        np.nan,
        (next_day_close > data[close_col]).astype(float),
    )

    if drop_last_row:
        data = data.iloc[:-1].copy().reset_index(drop=True)
        # If last row is dropped, target has no NaNs and can be cast to int
        data[target_col] = data[target_col].astype(int)

    return data


def create_next_day_target(
    df: pd.DataFrame,
    close_col: str = "Close",
    target_col: str = "Target_Next_Day",
    drop_last_row: bool = False,
) -> pd.DataFrame:
    """
    Create a forward-looking directional classification target:
        1 = next day's Close > today's Close (Bullish)
        0 = next day's Close <= today's Close (Bearish or Flat)

    Strictly guarantees that no future-data leakage enters the feature columns.
    If 'Ticker' column is present, target generation is grouped by Ticker so the last
    row of company A never uses the first row of company B as its next-day close.

    Args:
        df: DataFrame containing Close price.
        close_col: Name of close price column (default: 'Close').
        target_col: Name of target column to create (default: 'Target_Next_Day').
        drop_last_row: If True, drops the final row where target is undefined.
                       If False, retains the row with NaN in target (default: False).

    Returns:
        pd.DataFrame: New DataFrame with the directional target column added.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if "Ticker" in df.columns:
        target_dfs = []
        for _, grp in df.groupby("Ticker", sort=False):
            sub = grp.copy()
            if "Date" in sub.columns:
                sub = sub.sort_values("Date")
            tgt_df = _compute_single_asset_target(
                sub,
                close_col=close_col,
                target_col=target_col,
                drop_last_row=drop_last_row,
            )
            target_dfs.append(tgt_df)
        res = pd.concat(target_dfs, ignore_index=True)
        if "Date" in res.columns:
            res = res.sort_values(by=["Ticker", "Date"]).reset_index(drop=True)
        return res

    return _compute_single_asset_target(
        df,
        close_col=close_col,
        target_col=target_col,
        drop_last_row=drop_last_row,
    )

