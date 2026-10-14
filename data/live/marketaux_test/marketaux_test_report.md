# MarketAux Historical News Feasibility Test

> **Notice:** MarketAux proof-of-concept — not yet integrated into the production news pipeline.
> This report evaluates data coverage, structural quality, and pipeline compatibility without modifying production systems.

## Provider
MarketAux

## Ticker
AAPL

## Requested Period
2019-01-01 to 2024-12-31

## API Availability
- Provider-reported article count: 83757
- Articles actually retrieved: 12
- Pagination limit: 3 articles/page
- Rate/quota information: Free Tier: 100 req/day, 1 req/sec, max 3 items/page
- Earliest article: 2021-12-30
- Latest article: 2024-12-30

## Yearly Coverage

| Year | Articles | Earliest | Latest | Sources | Usable Text |
|------|----------|----------|--------|---------|-------------|
| 2019 | 0 | N/A | N/A | 0 | 0 |
| 2020 | 0 | N/A | N/A | 0 | 0 |
| 2021 | 20717 | 2021-12-30 | 2021-12-30 | 3 | 3 |
| 2022 | 24207 | 2022-12-30 | 2022-12-30 | 3 | 3 |
| 2023 | 20445 | 2023-12-30 | 2023-12-30 | 3 | 3 |
| 2024 | 18273 | 2024-12-30 | 2024-12-30 | 2 | 3 |

## Data Quality
- Duplicate IDs: 0
- Duplicate URLs: 0
- Missing titles: 0
- Missing descriptions: 0
- Usable text: 12 (100.0%)
- Unique sources: 9

### Represented Sources
benzinga.com, forbes.com, gurufocus.com, marketwatch.com, mottcapitalmanagement.com, news.google.com, raskmedia.com.au, seekingalpha.com, thestockmarketwatch.com

## NLP Compatibility
- Sample size: 12
- Successfully processed: 12
- Failed: 0
- Required normalization:
  - `title` -> `Headline`
  - `description` / `snippet` -> `Summary`
  - `published_at` (ISO 8601 string) -> `Published_At_UTC` (UTC format: `YYYY-MM-DDTHH:MM:SSZ`)
  - `uuid` + metadata hash -> `News_ID` (32-character SHA-256 deterministic digest)
  - `entities[].sentiment_score` -> `Provider_Sentiment`
  - `entities[].match_score` -> `Provider_Relevance`

## Feasibility Verdict

LIMITED

### Detailed Assessment
MarketAux reports 83757 articles for AAPL across 2021-2024, but years 2019, 2020 have 0 articles available (historical coverage begins in late 2021). Additionally, standard tier pagination limits (3 articles per request, 100 requests per day) mean bulk retrieval of all 83,757 articles would require over 27,000 API requests. Hence, coverage is LIMITED.

1. **Schema & Quality Alignment**: MarketAux provides high-quality metadata including entity mapping, sentiment scores, and full URLs. Its text structure aligns smoothly with our NLP feature extraction requirements.
2. **Quota & Rate Boundaries**: The standard free tier caps response page sizes to 3 articles (`limit=3`) and daily queries to 100 requests. An exhaustive backfill of AAPL across 2019–2024 requires either paid tier access or strict window sampling.
3. **Isolation Guarantee**: All operations performed in this test were strictly read-only and in-memory. Zero records were inserted into `data/live/news_database.db`.
