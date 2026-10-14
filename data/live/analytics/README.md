# Real-Data Analytics Layer Documentation

## 1. Overview & Purpose

The **Real-Data Analytics Layer** provides an isolated, read-only analytics subsystem within the **AI-Based Intelligent Trading System**. It computes reproducible technical market indicators, company-level sentiment and impact summaries, daily news signals, and systematic coverage audits across the 10 target equities:
- **AAPL** (Apple)
- **MSFT** (Microsoft)
- **GOOGL** (Alphabet)
- **AMZN** (Amazon)
- **NVDA** (NVIDIA)
- **META** (Meta Platforms)
- **TSLA** (Tesla)
- **JPM** (JPMorgan Chase)
- **V** (Visa)
- **JNJ** (Johnson & Johnson)

This layer is strictly segregated from the offline academic modeling pipeline (Assignments 1–8) and decision fusion components.

---

## 2. Architecture & File Inventory

The analytics layer produces four machine-readable summary tables stored in `data/live/analytics/`:

| Output File | Record Count | Description | Primary Columns |
| :--- | :--- | :--- | :--- |
| `real_market_summary.csv` | 10 rows (1 per ticker) | Latest market metrics, moving averages, rolling volatility, volume, and analytical trend indications. | `Ticker`, `Latest_Price`, `Daily_Return`, `SMA_5`, `SMA_10`, `EMA_12`, `EMA_26`, `Volatility_5`, `Volume`, `Trend_Indication` |
| `real_news_summary.csv` | 10 rows (1 per ticker) | Company-level sentiment distribution, Loughran-McDonald mean scores, relevance, importance, triage score, and high-impact counts. | `Ticker`, `Article_Count`, `Positive_Count`, `Neutral_Count`, `Negative_Count`, `Average_Sentiment`, `Average_Relevance`, `Average_Importance`, `Average_Triage_Score`, `High_Impact_Article_Count`, `Dominant_Sentiment` |
| `real_news_daily_summary.csv` | 193 rows (date-ticker) | Daily session-level aggregated news signals and triage-weighted sentiment. | `Ticker`, `Date`, `News_Count`, `Average_Sentiment`, `Triage_Weighted_Sentiment`, `Average_Triage_Score`, `News_Signal` |
| `real_data_coverage.csv` | 10 rows (1 per ticker) | Historical coverage audit identifying January 2025 completeness vs partial/capped status and unique sources. | `Ticker`, `Real_Article_Count`, `January_Article_Count`, `Earliest_Article_Date`, `Latest_Article_Date`, `January_Coverage_Status`, `Unique_Sources_Count`, `Coverage_Notes` |

---

## 3. Strict Architectural Isolation

1. **Academic Assignments 1–8 Decoupled**:
   - The academic notebooks (`notebooks/01_data_eda.ipynb` through `08_decision_fusion.ipynb`) and academic CSV outputs in `data/raw/` and `data/processed/` are completely isolated and untouched.
   - Machine learning models (regression, classification, clustering, reinforcement learning) and decision fusion calculations operate independently on baseline historical datasets.
2. **Decision Fusion Boundary**:
   - The analytics outputs in this folder are **NOT** connected to the Decision Fusion engine (`src/decision/decision_fusion.py`) or the RL trading policy.
   - They serve purely as exploratory, diagnostic, and monitoring analytics for the live/real-data pipeline.
3. **No Database Modification**:
   - Analytics routines connect to `data/live/news_database.db` via read-only SQLite URI (`file:...mode=ro`) or parse pre-existing processed live CSV files.
   - Zero modifications, updates, or fixture writes occur during analytics computation.

---

## 4. Data Provenance & Methodological Independence

### Internal NLP Scoring vs. External Sentiment
- All sentiment, relevance, importance, and triage scores are derived **exclusively** from our internal `src/nlp/news_nlp_pipeline.py`.
- We utilize the financial domain-specific **Loughran-McDonald (LM) dictionary** along with regex-based event extraction (earnings, product launches, acquisitions, litigation, leadership changes).
- **Alpha Vantage's proprietary provider sentiment scores are intentionally discarded** to maintain complete algorithmic ownership and scientific reproducibility.

### Real News Authenticity (Zero Synthetic Data)
- **100% of news records represent authentic, real-world articles** fetched via the Alpha Vantage `NEWS_SENTIMENT` endpoint.
- Zero mock, synthetic, or fabricated news records exist in the production database or analytics outputs.

---

## 5. Coverage Limitations & Alpha Vantage Free-Tier Boundaries

Historical financial news backfill was conducted for the target window of **January 2025** (`2025-01-01` to `2025-01-31`).

### Coverage Classification Policy:
1. **Complete January Coverage** (`AAPL`, `MSFT`, `META`, `TSLA`, `V`, `JNJ`):
   - All bounded time sub-windows were retrieved without reaching provider payload limits or were verified as exhaustive.
   - `AAPL` additionally holds 50 real-time live articles from October 2026.
2. **Partial / Capped January Coverage** (`GOOGL`, `AMZN`, `NVDA`, `JPM`):
   - Alpha Vantage Free API keys impose a standard limit of **25 requests per day** and **5 calls per minute**.
   - These four high-volume equities encountered the 25-request/day provider quota limit during subsequent sub-window slicing.
   - The records captured represent authentic January 2025 articles (50–59 per ticker), but coverage across the entire month is **partial/capped**, not exhaustive.
   - **Crucially: This is an external API quota limitation, NOT a software defect or pipeline failure.**

---

## 6. Disclaimers & Operational Assumptions

> [!WARNING]
> **Market Data Latency**: All stock price metrics and technical indicators are sourced via `yfinance` and are explicitly tagged as **`"Delayed / Snapshot"`** or **`"Historical Extension"`**. Data does not represent exchange-grade, tick-by-tick real-time market feeds.

> [!IMPORTANT]
> **No Trading Performance Guarantee**: All trend indications (e.g., `"Bullish"`, `"Bearish"`, `"Neutral"`), moving averages, rolling volatility, and news triage scores are **heuristic analytical metrics** designed for educational, research, and diagnostic monitoring. They do **NOT** constitute financial advice, investment recommendations, or guarantees of algorithmic trading performance.
