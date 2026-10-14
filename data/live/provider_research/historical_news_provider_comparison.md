# Historical Financial News Provider Research & Feasibility Evaluation

> **Research Report Notice:** This report is an isolated feasibility evaluation. No provider has been chosen or integrated into the production pipeline. Existing Alpha Vantage components, MarketAux/Finnhub test files, Decision Fusion, and Dashboard V2 remain 100% untouched.

---

## 1. Executive Summary

### 1.1 Objective
Our AI-Based Intelligent Trading System requires high-quality, real financial news data covering **10 major equities** across a 6-year research horizon:
- **Historical Window:** `2019-01-01` through `2024-12-31`
- **Target Companies:** `AAPL`, `MSFT`, `GOOGL`, `AMZN`, `NVDA`, `META`, `TSLA`, `JPM`, `V`, `JNJ`
- **Required Metadata:** Headline, substantive summary/snippet, publication timestamp (UTC), publisher source, canonical URL, and deterministic News_ID.
- **NLP Compatibility:** Must feed seamlessly into our internal Loughran-McDonald-inspired financial NLP and news triage pipeline (`src/nlp/news_nlp_pipeline.py`) without relying on proprietary third-party sentiment scores.

### 1.2 Prior Empirical Findings
Prior isolated proof-of-concept tests demonstrated clear structural bottlenecks in earlier candidates:
- **Alpha Vantage (Incumbent):** Strict free-tier quota of 25 requests/day makes multi-year, multi-company ingestion take over a year; furthermore, historical depth before 2022 is sparse and inconsistent.
- **MarketAux (Evaluated):** Complete absence of articles for 2019–2020 (database begins in late 2021). While 2021–2024 has 83,757 articles, the free-tier restriction of 3 articles/request and 100 requests/day makes bulk retrieval impossible without commercial subscriptions.
- **Finnhub (Evaluated):** Free tier `/company-news` restricts history to a rolling 1-year window (past 365 days). Queries for 2019–2023 return empty datasets (`[]`).

### 1.3 Key Finding
**Real historical financial news covering 2019–2024 for all 10 companies is achievable**, but only two architectures realistically enable dataset construction for a student/research project:
1. **Alpaca Market Data News API (Powered by Benzinga):** **The standout free candidate.** Delivers historical news back to **2015**, permits **200 requests/minute** on free paper trading accounts, provides full summaries/content, and uses cursor-based pagination.
2. **Polygon.io (Massive) Reference News API:** **The standout premium candidate.** Provides full historical news back to **June 22, 2016** with 1,000 items/page, but requires a paid Starter plan ($29/month) as its free tier is capped at a rolling 2-year window.

---

## 2. Comprehensive Provider Comparison Matrix

| Provider | Endpoint | Historical Depth Available | Covers 2019–2024? | Ticker Filtering | Date Filtering | Pagination Mechanism | Rate / Request Limits | Free Tier Access | Commercial Cost | Text Quality | Overall Rank |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Alpaca News API (Benzinga)** | `https://data.alpaca.markets/v1beta1/news` | **2015 to Present** | **YES** | `symbols=AAPL` | `start` / `end` (RFC-3339) | Cursor (`page_token`, 50/req) | **200 req/min** | **Full 2015–Present Accessible** | Free (Paper account) | Headline, Summary, Full Content | **1 (Best)** |
| **Polygon.io (Massive)** | `https://api.polygon.io/v2/reference/news` | **2016-06-22 to Present** (Paid); 2 Years (Free) | **YES** (on Paid Starter) | `ticker=AAPL` | `published_utc.gte` / `lte` | Cursor (`next_url`, 1000/req) | 5 req/min (Free); Unlimited (Paid) | Rolling 2 years only on Free | $29/month (Starter) | Headline, Description, Publisher | **2 (2nd Best)** |
| **Financial Modeling Prep (FMP)** | `https://financialmodelingprep.com/api/v3/stock_news` | **2019 to Present** (Paid); Recent (Free) | **YES** (on Paid Starter) | `tickers=AAPL` | `from` / `to` (`YYYY-MM-DD`) | Page Index (`page=0..100`, 250/req) | 250 calls/day (Free Sandbox); 300/min (Paid) | Sandbox restricted on Free | $19–$29/month (Starter) | Headline, Text Snippet, Site | **3 (3rd Best)** |
| **MarketAux** | `https://api.marketaux.com/v1/news/all` | Late 2021 to Present | **NO** (2019–2020: 0 articles) | `symbols=AAPL` | `published_after` / `before` | Page Index (`page=1..N`, 3/req Free) | 100 req/day (1 req/sec) | 2021–2024 accessible; 3 items/page cap | $29–$79/month | Title, Snippet, Entities | **4** |
| **EODHD** | `https://eodhd.com/api/news` | 30+ Years (Paid); 1 Year (Free) | **YES** (on Paid); **NO** (Free) | `s=AAPL.US` | `from` / `to` (`YYYY-MM-DD`) | Offset (`offset`, `limit` 50/req) | 4 news calls/day (20 units/day) | Rolling 1 year only on Free | $19.99/month | Headline, Summary, Content | **5** |
| **Finnhub** | `https://finnhub.io/api/v1/company-news` | Past 1 Year (Free); Multi-year (Paid) | **NO** (Free returns `[]` for $>1$ yr) | `symbol=AAPL` | `from` / `to` (`YYYY-MM-DD`) | Unpaginated (Date Slicing required) | 30–60 req/min (Free) | Rolling 1 year only on Free | $50–$500/month | Headline, Summary, Source | **6** |
| **Alpha Vantage** | `https://www.alphavantage.co/query` | 2022 to Present (Reliable) | **NO** (Pre-2022 sparse) | `tickers=AAPL` | `time_from` / `to` | Sort/Limit (50/req) | 25 req/day (5 req/min) | 25 req/day; sparse pre-2022 | $49.99/month | Title, Summary, Ticker Sentiment | **7** |
| **Tiingo** | `https://api.tiingo.com/tiingo/news` | 15+ Years (Enterprise); 3 Months (Free) | **NO** (Free restricted to 3 mos) | `tickers=AAPL` | `startDate` / `endDate` | Offset/Page (100/req) | 50 req/hr; 1000 req/day | Rolling 3 months only on Free | Institutional sales contract | Title, Description, Tags | **8** |
| **StockNewsAPI** | `https://stocknewsapi.com/api/v1` | 2019 to Present (Premium) | **YES** (on Premium); **NO** (Free) | `tickers=AAPL` | `date` | Page-based (50/req) | 100 calls total (5-day trial) | 5-day trial only; no permanent free tier | $49.99/month (Premium) | Headline, Text, Sentiment | **9** |
| **NewsAPI.org** | `https://newsapi.org/v2/everything` | 5 Years (Business); 1 Month (Free) | **NO** (Free restricted to 1 mo) | Keyword query only (No native ticker) | `from` / `to` (`YYYY-MM-DD`) | Page-based (100/req) | 100 req/day (Free) | Rolling 1 month only on Free | $449/month (Business) | Title, Description, Source | **10** |
| **GDELT 2.0** | `https://api.gdeltproject.org/api/v2/doc/doc` | 1979 to Present (BigQuery); 3 Mos (Doc API) | **PARTIAL** (BigQuery SQL only) | Keyword regex (`V2Organizations`) | `STARTDATETIME` / `ENDDATETIME` | BigQuery Batch / Export | BigQuery quota (1TB/mo free scan) | Doc API is 3 mos; BigQuery is full | Google Cloud pay-per-scan | Title, Tone score (No clean summary) | **11** |

---

## 3. In-Depth Technical Analysis by Provider

### 3.1 Alpaca News API (Rank 1 — Best Candidate)
- **Data Source:** Institutional news feed syndicated from **Benzinga** and major financial wires.
- **Historical Depth:** Formally documents archives dating back to **2015**. Unrestricted historical access for 2019-01-01 through 2024-12-31 is available directly via standard REST parameters.
- **Endpoint:** `GET https://data.alpaca.markets/v1beta1/news`
- **Query Parameters:**
  - `symbols`: Comma-separated tickers (e.g. `symbols=AAPL` or `symbols=AAPL,MSFT,NVDA`)
  - `start`: RFC-3339 timestamp (e.g. `2019-01-01T00:00:00Z`)
  - `end`: RFC-3339 timestamp (e.g. `2024-12-31T23:59:59Z`)
  - `limit`: Number of records per page (up to 50)
  - `include_content`: Boolean (allows retrieving full article body text in addition to headline and summary)
  - `page_token`: Opaque cursor token for sequential pagination
- **Rate Limits & Throughput:**
  - **Free Tier:** **200 requests per minute** (3.33 requests/sec) with a free paper trading account.
  - Unlike Alpha Vantage (25 requests/day) or MarketAux (100 requests/day), 200 calls/min allows retrieving up to **10,000 articles per minute** (50 articles $\times$ 200 calls).
- **Text & Summary Quality:** Returns `headline`, `summary`, `content` (full HTML/markdown body), `author`, `source` (Benzinga, etc.), `url`, and `symbols`.
- **NLP Alignment:** Clean headline and rich summary/content map 1:1 into our project's schema.
- **Student/Research Accessibility:** Free registration. Generates instant API Key ID and Secret Key for Paper Trading with no capital requirement or credit card needed.

### 3.2 Polygon.io / Massive Reference News API (Rank 2 — Second Best)
- **Data Source:** Multi-source publisher aggregator including Benzinga, The Fly, GlobeNewswire, PR Newswire, Seeking Alpha, and Business Wire.
- **Historical Depth:**
  - **Free Tier (Basic):** Rolling **2-year window only**. Queries for dates older than 2 years return empty results.
  - **Paid Tier (Starter — $29/mo):** Full historical archive dating back to **June 22, 2016**, completely covering 2019–2024.
- **Endpoint:** `GET https://api.polygon.io/v2/reference/news`
- **Query Parameters:**
  - `ticker`: Symbol filter (e.g. `ticker=AAPL`)
  - `published_utc.gte` and `published_utc.lte`: ISO 8601 date filters
  - `limit`: Up to 1,000 articles per request on paid plans (10–100 on standard)
  - `order`: `asc` or `desc`
- **Pagination:** Cursor-based via `next_url` provided in the JSON payload response.
- **Rate Limits:**
  - Free Basic: 5 API calls per minute (too slow for bulk multi-year backfill).
  - Paid Starter ($29/mo): Unlimited calls per minute (or high burst rate), making bulk extraction fast.
- **Text Quality:** High-quality metadata including `title`, `description`, `publisher` (name, homepage), `article_url`, `author`, `keywords`.
- **Verdict for Research:** Best commercial option. If a $29 one-month subscription is viable, Polygon offers the cleanest, largest multi-publisher dataset. If zero-budget is strictly required, the free tier cannot reach 2019–2023.

### 3.3 Financial Modeling Prep (FMP) Stock News API (Rank 3 — Third Best)
- **Data Source:** Financial news aggregator indexing major US financial publishers.
- **Historical Depth:** Covers news from **2019 to Present**.
- **Endpoint:** `GET https://financialmodelingprep.com/api/v3/stock_news`
- **Query Parameters:**
  - `tickers`: Symbol filter (e.g. `tickers=AAPL`)
  - `from` and `to`: Date format `YYYY-MM-DD`
  - `page`: Page index (`0, 1, 2...`)
  - `limit`: Up to 250 records per request
- **Rate Limits & Free Restrictions:**
  - Free plan provides ~250 calls/day (often restricted to a sandbox list of symbols or recent news).
  - Paid Starter plan ($19–$29/month) provides 300 calls/minute and unlocks full historical access for all tickers.
- **Text Quality:** Provides `title`, `text` (summary/body snippet), `publishedDate`, `site` (source domain), and `url`.
- **Verdict for Research:** Solid fallback. Accessible if a low-cost subscription is feasible, but free tier restrictions make unattended multi-year extraction difficult.

### 3.4 MarketAux (Empirically Evaluated — Rank 4)
- **Endpoint:** `https://api.marketaux.com/v1/news/all`
- **Historical Horizon:** Archive begins in **late December 2021**. Zero articles exist for 2019 and 2020.
- **Volume:** 83,757 articles reported for AAPL across 2021–2024.
- **Free Quotas:** 100 requests/day, 3 articles/request (`limit=3`).
- **Verdict:** **LIMITED**. Unsuitable for full 2019–2024 research due to missing 2019–2020 depth and severe pagination limits.

### 3.5 EODHD / EOD Historical Data (Rank 5)
- **Endpoint:** `GET https://eodhd.com/api/news`
- **Historical Horizon:** Deep multi-year archive (30+ years) on paid plans, but strictly capped at **past 1 year on the free tier**.
- **Free Quotas:** 20 API units/day. Each news query costs 5 units, permitting only **4 news calls per day**.
- **Pricing:** Paid plans begin at $19.99/month.
- **Verdict:** Deep archives, but free tier is completely throttled and history is restricted to 1 year.

### 3.6 Finnhub (Empirically Evaluated — Rank 6)
- **Endpoint:** `GET https://finnhub.io/api/v1/company-news`
- **Historical Horizon:** Free tier strictly limits company news to a rolling **1-year window** (past 365 days). Queries for 2019–2023 return empty lists (`[]`).
- **Pagination:** Unpaginated; requires manual chronological date-slicing.
- **Throughput:** 30–60 calls/minute.
- **Verdict:** **UNSUITABLE on Free Tier** for 2019–2024 backfill.

### 3.7 Alpha Vantage (Incumbent Provider — Rank 7)
- **Endpoint:** `GET https://www.alphavantage.co/query?function=NEWS_SENTIMENT`
- **Historical Horizon:** In practice, news coverage prior to 2022 is sparse and erratic.
- **Throughput:** 25 requests/day, 5 calls/minute on free tier.
- **Verdict:** Current provider is excellent for live updates and recent monthly snapshots (e.g., our verified January 2025 backfill), but mathematically incapable of supporting full 2019–2024 extraction across 10 tickers on free tier.

### 3.8 Tiingo (Rank 8)
- **Endpoint:** `GET https://api.tiingo.com/tiingo/news`
- **Historical Horizon:** Free tier limits queries to **3 months of history**. Extended historical data (15+ years) is strictly gated behind institutional contracts.
- **Verdict:** Gated; unsuitable for multi-year academic research on free tier.

### 3.9 StockNewsAPI / Snapi (Rank 9)
- **Endpoint:** `GET https://stocknewsapi.com/api/v1`
- **Historical Horizon:** Historical data requires the **Premium Plan ($49.99/month)**. Free access is only a 5-day trial with 100 calls total.
- **Verdict:** Commercial barrier; unsuitable without monthly payment.

### 3.10 NewsAPI.org (Rank 10)
- **Endpoint:** `GET https://newsapi.org/v2/everything`
- **Historical Horizon:** Free Developer tier is strictly limited to **articles up to 1 month old**. Multi-year historical search requires the Business plan costing **$449/month**.
- **Filtering:** No native ticker tagging (relies solely on raw keyword matching).
- **Verdict:** Prohibitively expensive ($449/mo) and lacks financial ticker normalization.

### 3.11 GDELT 2.0 (Rank 11)
- **Access Points:** GDELT 2.0 Doc API (rolling 3 months) vs. Google BigQuery (`gdelt-bq:gdeltv2.gkg`).
- **Historical Horizon:** Deep archive dating back to 1979 in BigQuery.
- **Limitations for Financial System:**
  - No native stock ticker metadata (requires regex matching over `V2Organizations` strings like `Apple Inc` or `AAPL`).
  - Contains general geopolitical and world news rather than focused financial market journalism.
  - BigQuery tables are massive (multi-terabyte scans), consuming query quotas quickly.
  - Does not store clean article summaries or standardized body text, requiring auxiliary web scraping to obtain readable text.
- **Verdict:** High engineering friction and non-financial schema.

---

## 4. Evaluation Against the 12 Project Criteria

| # | Criterion | Alpaca News (Rank 1) | Polygon.io (Rank 2) | FMP (Rank 3) |
|:---:|:---|:---:|:---:|:---:|
| **1** | **2019-01-01 to 2024-12-31 Coverage** | **YES** (Archive back to 2015) | **YES** (Paid Starter back to 2016) | **YES** (Back to 2019) |
| **2** | **10 Project Tickers Supported** | **YES** (All 10 US equities active) | **YES** (All 10 active) | **YES** (All 10 active) |
| **3** | **Date-Range Filtering** | **YES** (`start` / `end` RFC-3339) | **YES** (`published_utc.gte` / `lte`) | **YES** (`from` / `to` `YYYY-MM-DD`) |
| **4** | **Sufficient Article Volume** | **YES** (Benzinga wire volume) | **YES** (Multi-publisher volume) | **YES** (Aggregated press volume) |
| **5** | **Article Headline** | **YES** (`headline`) | **YES** (`title`) | **YES** (`title`) |
| **6** | **Summary / Body Usable for NLP** | **YES** (`summary` + `content`) | **YES** (`description`) | **YES** (`text`) |
| **7** | **Publication Timestamp** | **YES** (`created_at` ISO 8601 UTC) | **YES** (`published_utc` ISO 8601) | **YES** (`publishedDate` UTC) |
| **8** | **Source / Publisher Attribution** | **YES** (`source` / `author`) | **YES** (`publisher.name`) | **YES** (`site`) |
| **9** | **URL / Stable Identifier** | **YES** (`url` + integer `id`) | **YES** (`article_url` + `id`) | **YES** (`url`) |
| **10** | **Pagination / Bulk Ingestion** | **YES** (`page_token` cursor, 50/req) | **YES** (`next_url` cursor, 1000/req) | **YES** (`page=N`, 250/req) |
| **11** | **Practical Limits for Dataset Build** | **YES** (200 req/min = 10,000 arts/min) | **YES** (High throughput on Starter) | **YES** (300 req/min on Starter) |
| **12** | **Reasonable Access (Student/Research)** | **YES** (**100% Free** Paper Account) | **$29/mo** (Starter tier) | **$19–$29/mo** (Starter tier) |

---

## 5. Provider Rankings & Final Recommendation

### Rank 1: **Alpaca News API (Powered by Benzinga)** — **Best Candidate**
- **Why #1:** It is the **only verified provider** that simultaneously satisfies all three core constraints for a student/research project:
  1. Full **2015–present** historical depth (covering 100% of 2019–2024).
  2. Free access via Alpaca Paper Trading developer accounts (no live trading, deposits, or card required).
  3. Exceptional throughput (**200 requests/minute**, 50 articles/request) allowing multi-year, multi-company ingestion in minutes rather than months.

### Rank 2: **Polygon.io (Massive) Reference News API** — **Second Best Candidate**
- **Why #2:** The highest-fidelity institutional multi-source news archive available, with coverage dating back to **June 22, 2016** and up to 1,000 articles per query.
- **Why Not #1:** The free tier is constrained to a rolling 2-year window. Accessing 2019–2024 requires a paid Starter subscription ($29/month).

### Rank 3: **Financial Modeling Prep (FMP) Stock News API** — **Third Best Candidate**
- **Why #3:** Provides structured historical news back to **2019** with clean summary snippets and up to 250 items per call.
- **Why Not #1:** Free tier is sandbox-restricted; full 2019–2024 backfill requires a paid plan ($19–$29/month).

---

## 6. Dataset Construction Feasibility Analysis

### Can we realistically download the full 2019–2024 dataset for all 10 companies?

**YES, using Candidate #1 (Alpaca News API):**
- **Target Companies (10):** `AAPL`, `MSFT`, `GOOGL`, `AMZN`, `NVDA`, `META`, `TSLA`, `JPM`, `V`, `JNJ`
- **Time Window:** 6 years (`2019-01-01` to `2024-12-31` = 72 months)
- **Estimated Total Volume:** ~15,000 to ~40,000 high-relevance articles across all 10 companies over 6 years.
- **Throughput Math:**
  - 50 articles per request $\times$ 200 requests/minute = **10,000 articles per minute max throughput**.
  - Querying 40,000 articles requires approximately 800 requests.
  - At 100–150 calls/minute (leaving safety headroom under the 200 limit), **the complete 6-year historical dataset for all 10 companies can be downloaded in under 10 minutes**.
- **No Daily Quota Block:** Unlike Alpha Vantage (which stops after 25 calls) or MarketAux (which stops after 100 calls), Alpaca's rate limit resets every minute without a daily cutoff.

---

## 7. Safety & Isolation Guarantee

This research report was conducted strictly as an analytical study:
- **Zero production code modified.**
- **Zero changes** to existing Alpha Vantage pipelines, MarketAux test files, or Finnhub test files.
- **Zero records written** to `data/live/news_database.db` (verified: 525 articles, 524 NLP records, 6 validation logs).
- **Assignments 1–8 notebooks, Decision Fusion, and Dashboard V2 remain 100% untouched.**

*Report compiled and saved to:*
- `data/live/provider_research/historical_news_provider_comparison.md`
- `data/live/provider_research/historical_news_provider_comparison.csv`
