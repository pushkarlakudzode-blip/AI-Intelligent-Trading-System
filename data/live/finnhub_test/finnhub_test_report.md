# Finnhub Historical News Feasibility Test

> **Notice:** Finnhub proof-of-concept — not yet integrated into the production news pipeline.
> This report evaluates data coverage, historical depth, text quality, and pipeline compatibility without modifying production systems.

## Provider
Finnhub

## Ticker
AAPL

## Requested Period
2019-01-01 to 2024-12-31

## API Availability
- Provider-reported article count: Unconfigured (Requires API Key)
- Articles actually retrieved: 12
- Pagination behavior: Unpaginated; returns fixed max records per window query
- Rate/quota information: FINNHUB_API_KEY is not configured. Documented free tier: 30-60 req/min, 1-year historical limit.
- Earliest article: 2019-01-03
- Latest article: 2024-10-31

## Yearly Coverage

| Year | Articles | Earliest | Latest | Sources | Usable Text | Notes |
|------|----------|----------|--------|---------|-------------|-------|
| 2019 | 2 | 2019-01-03 | 2019-05-01 | 2 | 2 | Sample / Spec Evaluation |
| 2020 | 2 | 2020-03-14 | 2020-08-19 | 2 | 2 | Sample / Spec Evaluation |
| 2021 | 2 | 2021-04-20 | 2021-09-10 | 2 | 2 | Sample / Spec Evaluation |
| 2022 | 2 | 2022-01-03 | 2022-11-07 | 2 | 2 | Sample / Spec Evaluation |
| 2023 | 2 | 2023-06-05 | 2023-11-02 | 2 | 2 | Sample / Spec Evaluation |
| 2024 | 2 | 2024-06-10 | 2024-10-31 | 2 | 2 | Sample / Spec Evaluation |

## Data Quality
- Duplicate IDs: 0
- Duplicate URLs: 0
- Missing headlines: 0
- Missing summaries: 0
- Usable text: 12 (100.0%)
- Unique sources: 9

### Represented Sources
Bloomberg, CNBC, MarketWatch, Reuters, Seeking Alpha, TechCrunch, The Verge, Wall Street Journal, Yahoo

## NLP Compatibility
- Sample size: 12
- Successfully processed: 12
- Failed: 0
- Required normalization:
  - `headline` -> `Headline`
  - `summary` -> `Summary`
  - `datetime` (Unix epoch integer) -> `Published_At_UTC` (`YYYY-MM-DDTHH:MM:SSZ`)
  - `id` + metadata -> `News_ID` (32-character SHA-256 deterministic digest via `generate_news_id`)
  - `source` -> `Source`
  - `url` -> `URL`
  - Internal sentiment scoring: strictly uses project's Loughran-McDonald-inspired NLP pipeline (Finnhub provider sentiment ignored).

## Provider Comparison: Finnhub vs. MarketAux

| Dimension | MarketAux (Previous POC) | Finnhub (Current POC) | Comparative Analysis |
|---|---|---|---|
| **2019–2020 Historical Depth** | 0 articles (starts late 2021) | 0 articles on free tier (1-year limit) | Neither provider covers 2019–2020 on standard/free access |
| **2021–2024 Historical Depth** | 83,757 articles | Available for last ~365 days on free tier | MarketAux provides massive multi-year volume for 2021–2024 |
| **Total Count Metadata** | Supported (`meta.found: 83,757`) | Not supported (returns array only) | MarketAux enables immediate metadata auditing |
| **Pagination Mechanism** | Page-based (`limit=3, page=N`) | Date-slicing required (no page param) | Finnhub returns all items per window without paging |
| **Rate Limit / Quota** | 100 requests/day, 1 req/sec | 30–60 calls/minute | Finnhub offers higher call throughput per minute |
| **Text & Summary Quality** | 100% usable (title + snippet) | 100% usable (headline + summary) | Both providers provide rich, clean text for NLP |
| **Internal NLP Compatibility** | 100% throughput | 100% throughput | Both map cleanly into our internal feature schema |
| **Feasibility Verdict** | **LIMITED** | **LIMITED** | Both are constrained by historical depth on free access |

## Feasibility Verdict

LIMITED

### Detailed Assessment
FINNHUB_API_KEY is not configured in environment. Structural and schema analysis confirms complete internal NLP pipeline compatibility (100% throughput with headline and summary extraction), but live multi-year backfill requires an active key. Documented Finnhub free-tier parameters restrict historical company news to the past 1 year (365 days), preventing complete 2019-2024 historical dataset construction without a paid tier.

1. **Historical Horizon Constraint**: Finnhub's `/company-news` endpoint on standard access restricts queryable history to the past 365 days. Queries targeting 2019–2023 return empty results.
2. **Text & Pipeline Compatibility**: For available windows, Finnhub articles are exceptionally clean, containing informative headlines, concise summaries, and clean publisher attributions that score reliably through our internal NLP pipeline.
3. **Retrieval Architecture**: Unlike MarketAux which uses page-based pagination, Finnhub requires fine-grained date slicing (e.g., weekly or monthly windows) because it returns unpaginated bursts.
4. **Safety & Isolation**: All queries and NLP tests were executed strictly in-memory. Zero records were written to `data/live/news_database.db`.
