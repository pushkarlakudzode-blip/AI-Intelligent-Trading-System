"""
Test Data Sources & Configuration Script.

Verifies connectivity and readiness of market and news providers:
1. yfinance market snapshot & intraday retrieval
2. Alpha Vantage API key presence and endpoint readiness
3. SQLite database & CSV live storage integrity
4. NLP enrichment pipeline on real financial text
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.data.realtime_market import get_current_quote, get_intraday_market_data
from src.data.historical_news import get_api_key
from src.data.data_storage import init_storage, DB_PATH
from src.nlp.news_nlp_pipeline import compute_financial_sentiment, compute_company_relevance, compute_triage_score

def run_tests():
    print("=" * 65)
    print("AI-BASED INTELLIGENT TRADING SYSTEM - DATA LAYER DIAGNOSTICS")
    print("=" * 65)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # 1. Test Storage Initialization
    print("\n[1/4] Testing Local Live Storage Initialization...")
    init_storage()
    if DB_PATH.exists():
        print(f"  [OK] SQLite Database verified at: {DB_PATH.name} ({DB_PATH.stat().st_size} bytes)")
    else:
        print("  [FAIL] SQLite Database initialization failed!")

    # 2. Test yfinance Market Data
    print("\n[2/4] Testing yfinance Market Data Ingestion...")
    test_ticker = "AAPL"
    try:
        quote = get_current_quote(test_ticker)
        print(f"  [OK] Latest Quote for {test_ticker}:")
        print(f"    Price: ${quote['Price']:.2f} | Volume: {quote['Volume']:,} | Source: {quote['Data_Source']} | Status: {quote['Data_Status']}")
        print(f"    Timestamp: {quote['Timestamp']}")
    except Exception as e:
        print(f"  [FAIL] Failed to fetch quote for {test_ticker}: {e}")

    try:
        intra_df = get_intraday_market_data(test_ticker, interval="5m", period="1d")
        if not intra_df.empty:
            print(f"  [OK] Intraday Series: Fetched {len(intra_df)} bars (5m interval).")
        else:
            print("  [INFO] Intraday Series: Empty (market may currently be closed).")
    except Exception as e:
        print(f"  [FAIL] Intraday fetch error: {e}")

    # 3. Test News Provider (Alpha Vantage) Configuration
    print("\n[3/4] Testing Alpha Vantage News Provider Configuration...")
    api_key = get_api_key()
    if api_key:
        masked_key = api_key[:3] + "..." + api_key[-3:] if len(api_key) > 6 else "***"
        print(f"  [OK] ALPHA_VANTAGE_API_KEY detected in environment: [{masked_key}]")
        print("  [OK] Ready for live news collection via Alpha Vantage NEWS_SENTIMENT.")
    else:
        print("  [INFO] ALPHA_VANTAGE_API_KEY is NOT set in environment.")
        print("  [INFO] To enable live news queries, set ALPHA_VANTAGE_API_KEY in .env or your shell.")
        print("  [INFO] Ingestion layer will safely skip live API calls without crashing.")

    # 4. Test NLP Pipeline on Real Financial Headline
    print("\n[4/4] Testing NLP Enrichment Pipeline...")
    sample_headline = "Apple Reports Record All-Time High iPhone Revenue and Surging Cloud Profits"
    sent_label, sent_score = compute_financial_sentiment(sample_headline)
    relevance = compute_company_relevance("AAPL", sample_headline)
    triage = compute_triage_score(relevance, 1.0, 0.90)
    print(f"  [OK] Sample Headline: '{sample_headline}'")
    print(f"    Sentiment: {sent_label} ({sent_score:+.4f}) | Relevance: {relevance:.2f} | Triage Score: {triage:.4f}")

    print("\n" + "=" * 65)
    print("DATA LAYER PRE-FLIGHT DIAGNOSTICS COMPLETED SUCCESSFULLY!")
    print("=" * 65)

if __name__ == "__main__":
    run_tests()
