# Multi-Company Stock Market Master Dataset Verification Report
**Project**: AI-Based Intelligent Trading System  
**Dataset Path**: `data/raw/multi_company_stock_data.csv`  
**Summary Path**: `data/processed/multi_company_data_summary.csv`  
**Generated At**: 2026-10-03 16:00:28  

---

## 1. Executive Summary
This report documents the verification and quality audit of the newly constructed 10-company normalized master historical stock dataset. The dataset serves as the multi-company foundational raw data layer for the **AI-Based Intelligent Trading System**, expanding the system beyond single-asset (AAPL) analysis while preserving all existing single-company analytical artifacts intact.

- **Target Companies**: 10 companies across technology, consumer cyclical, financials, healthcare, and payments.
- **Data Source**: Yahoo Finance via `yfinance` daily historical feeds (`auto_adjust=False`).
- **Date Range**: `2019-01-02` through `2024-12-31`.
- **Total Master Rows**: `15,100` rows.
- **Data Format**: Normalized long-format (`Date, Ticker, Open, High, Low, Close, Volume`).
- **Data Quality**: **Zero missing values** across all OHLCV columns for all 10 companies; **zero duplicate Ticker-Date combinations**.

---

## 2. Company & Ticker Directory
All 10 requested companies were successfully acquired without errors or missing sessions:

| Ticker | Company Name | Sector / Industry | Status | Rows Acquired |
| :---: | :--- | :--- | :---: | :---: |
| **AAPL** | Apple | Technology / Consumer Electronics | ✅ Success | 1,510 |
| **MSFT** | Microsoft | Technology / Software & Cloud | ✅ Success | 1,510 |
| **GOOGL** | Alphabet | Technology / Internet Services | ✅ Success | 1,510 |
| **AMZN** | Amazon | Consumer Cyclical / E-Commerce & Cloud | ✅ Success | 1,510 |
| **NVDA** | NVIDIA | Technology / Semiconductors & AI Hardware | ✅ Success | 1,510 |
| **META** | Meta Platforms | Communication Services / Social Media | ✅ Success | 1,510 |
| **TSLA** | Tesla | Consumer Cyclical / Electric Vehicles & Clean Energy | ✅ Success | 1,510 |
| **JPM** | JPMorgan Chase | Financial Services / Diversified Banking | ✅ Success | 1,510 |
| **V** | Visa | Financial Services / Transaction & Payment Services | ✅ Success | 1,510 |
| **JNJ** | Johnson & Johnson | Healthcare / Pharmaceuticals & Medical Devices | ✅ Success | 1,510 |

---

## 3. Dataset-Level Descriptive Statistics

### Company Row Counts and Coverage Windows
Every company has identical trading session counts (1,510 sessions), corresponding exactly to the active trading calendar of the New York Stock Exchange and NASDAQ from 2019-01-02 to 2024-12-31:

| Ticker | Company | Row Count | First Date | Last Date | Missing Values (O,H,L,C,V) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **AAPL** | Apple | 1,510 | 2019-01-02 | 2024-12-31 | 0 / 0 / 0 / 0 / 0 |
| **MSFT** | Microsoft | 1,510 | 2019-01-02 | 2024-12-31 | 0 / 0 / 0 / 0 / 0 |
| **GOOGL** | Alphabet | 1,510 | 2019-01-02 | 2024-12-31 | 0 / 0 / 0 / 0 / 0 |
| **AMZN** | Amazon | 1,510 | 2019-01-02 | 2024-12-31 | 0 / 0 / 0 / 0 / 0 |
| **NVDA** | NVIDIA | 1,510 | 2019-01-02 | 2024-12-31 | 0 / 0 / 0 / 0 / 0 |
| **META** | Meta Platforms | 1,510 | 2019-01-02 | 2024-12-31 | 0 / 0 / 0 / 0 / 0 |
| **TSLA** | Tesla | 1,510 | 2019-01-02 | 2024-12-31 | 0 / 0 / 0 / 0 / 0 |
| **JPM** | JPMorgan Chase | 1,510 | 2019-01-02 | 2024-12-31 | 0 / 0 / 0 / 0 / 0 |
| **V** | Visa | 1,510 | 2019-01-02 | 2024-12-31 | 0 / 0 / 0 / 0 / 0 |
| **JNJ** | Johnson & Johnson | 1,510 | 2019-01-02 | 2024-12-31 | 0 / 0 / 0 / 0 / 0 |

---

## 4. Data Cleaning & Integrity Audit

1. **Date Parsing & Timezone Handling**:
   - Dates were parsed to `datetime64[ns]` and verified to be timezone-naive (`tz_localize(None)`).
   - In the exported CSV, dates are standardized as `YYYY-MM-DD` strings.
2. **Sorting & Ordering**:
   - The master dataframe is strictly sorted by `Ticker` (alphabetical) and `Date` (chronological).
3. **Deduplication**:
   - Checked `(Ticker, Date)` composite uniqueness. Exactly **0 duplicates** were detected.
4. **Data Types**:
   - `Date`: `object` (`YYYY-MM-DD` string in CSV, `datetime64[ns]` in memory)
   - `Ticker`: `object` (string)
   - `Open`: `float64` (numeric)
   - `High`: `float64` (numeric)
   - `Low`: `float64` (numeric)
   - `Close`: `float64` (numeric)
   - `Volume`: `int64` (numeric whole shares)
5. **No Blind Imputation**:
   - No forward-filling (`ffill`) or backward-filling (`bfill`) across non-trading days was applied.
   - All historical records reflect authentic exchange trading sessions.
   - No synthetic price points or fabricated dates were introduced.
6. **Market Outliers Preserved**:
   - Historical volatility events (e.g., COVID-19 crash in March 2020, 2024 tech earnings volatility) were legitimately retained without artificial clipping.

---

## 5. Explicit Validation Checklist

| Check # | Requirement | Expected | Observed | Result |
| :---: | :--- | :--- | :--- | :---: |
| **1** | Number of unique tickers | Exactly 10 | 10 | ✅ PASS |
| **2** | Exact ticker set matching | AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, JPM, V, JNJ | AAPL, AMZN, GOOGL, JNJ, JPM, META, MSFT, NVDA, TSLA, V | ✅ PASS |
| **3** | Duplicate Ticker-Date check | 0 duplicate rows | 0 duplicates | ✅ PASS |
| **4** | Date range compliance | Within 2019-01-01 to 2024-12-31 | 2019-01-02 to 2024-12-31 | ✅ PASS |
| **5** | Required OHLCV columns | Date, Ticker, Open, High, Low, Close, Volume | Date, Ticker, Open, High, Low, Close, Volume | ✅ PASS |
| **6** | Missing values per company | 0 for all companies | 0 missing across all 10 tickers | ✅ PASS |
| **7** | Total master row count | 15,100 rows | 15,100 rows | ✅ PASS |
| **8** | Rows per company | 1,510 rows per company | 1,510 rows for all 10 companies | ✅ PASS |

---

## 6. Output Artifact Verification
The following files were created and verified on disk:
1. `data/raw/multi_company_stock_data.csv` (`1,455,588` bytes, `15,100` rows)
2. `data/processed/multi_company_data_summary.csv` (`638` bytes, `10` rows)
3. `data/processed/multi_company_data_report.md` (this report)

---
*Report certified: All 10 companies successfully downloaded, cleaned, and verified.*
