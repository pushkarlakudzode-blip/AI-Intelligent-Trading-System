"""
Preflight Audit Script for Phase 4: Rigorous Out-of-Sample Evaluation
Inspects:
1. Historical market datasets (raw & features)
2. Classification models and outputs
3. RL policy and outputs
4. Market regime / clustering outputs
5. News NLP and daily news signals
6. Decision fusion module & weights
7. Dates, split boundaries, lookahead risks, and data availability
"""

import json
import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
LIVE_DIR = PROJECT_ROOT / "data" / "live"

print("=" * 80)
print("AUDIT 1: RAW MARKET DATA & PROCESSED FEATURES")
print("=" * 80)

stock_raw_file = RAW_DIR / "multi_company_stock_data.csv"
if stock_raw_file.exists():
    df_raw = pd.read_csv(stock_raw_file)
    print(f"Raw stock data exists: {stock_raw_file}")
    print(f"Shape: {df_raw.shape}, Tickers: {df_raw['Ticker'].unique().tolist()}")
    print(f"Date range: {df_raw['Date'].min()} to {df_raw['Date'].max()}")
    print(f"Columns: {df_raw.columns.tolist()[:10]}")
else:
    print(f"Raw stock data NOT found at {stock_raw_file}")

feat_file = PROCESSED_DIR / "multi_company_features.csv"
if feat_file.exists():
    df_feat = pd.read_csv(feat_file)
    print(f"\nFeatures data exists: {feat_file}")
    print(f"Shape: {df_feat.shape}, Tickers: {df_feat['Ticker'].unique().tolist()}")
    print(f"Date range: {df_feat['Date'].min()} to {df_feat['Date'].max()}")
    print(f"Columns: {df_feat.columns.tolist()[:15]}")
    # Check rows per ticker
    print("Rows per ticker in features:")
    print(df_feat.groupby("Ticker")["Date"].agg(["min", "max", "count"]))

print("\n" + "=" * 80)
print("AUDIT 2: UPSTREAM SUB-MODELS (NOTEBOOKS & PROCESSED CSVs)")
print("=" * 80)

# Check classification
class_file = PROCESSED_DIR / "multi_company_classification_results.csv"
if class_file.exists():
    df_class = pd.read_csv(class_file)
    print(f"Classification results: {class_file}")
    print(f"Shape: {df_class.shape}, Columns: {df_class.columns.tolist()}")
    print(f"Date range: {df_class['Date'].min()} to {df_class['Date'].max()}")

# Check RL
rl_file = PROCESSED_DIR / "multi_company_rl_results.csv"
if rl_file.exists():
    df_rl = pd.read_csv(rl_file)
    print(f"\nRL results: {rl_file}")
    print(f"Shape: {df_rl.shape}, Columns: {df_rl.columns.tolist()}")
    print(f"Date range: {df_rl['Date'].min()} to {df_rl['Date'].max()}")

# Check RL policy / Q-tables
rl_policy_file = PROCESSED_DIR / "multi_company_rl_policy.csv"
if rl_policy_file.exists():
    df_policy = pd.read_csv(rl_policy_file)
    print(f"RL policy shape: {df_policy.shape}, Columns: {df_policy.columns.tolist()}")

# Check clustering / regime
clust_file = PROCESSED_DIR / "multi_company_clustering_results.csv"
if clust_file.exists():
    df_clust = pd.read_csv(clust_file)
    print(f"\nClustering results: {clust_file}")
    print(f"Shape: {df_clust.shape}, Columns: {df_clust.columns.tolist()}")
    print(f"Date range: {df_clust['Date'].min()} to {df_clust['Date'].max()}")

# Check daily news signal
news_sig_file = PROCESSED_DIR / "multi_company_daily_news_signal.csv"
if news_sig_file.exists():
    df_news_sig = pd.read_csv(news_sig_file)
    print(f"\nDaily news signals: {news_sig_file}")
    print(f"Shape: {df_news_sig.shape}, Columns: {df_news_sig.columns.tolist()}")
    print(f"Date range: {df_news_sig['Date'].min()} to {df_news_sig['Date'].max()}")

# Check news NLP results
news_nlp_file = PROCESSED_DIR / "multi_company_news_nlp_results.csv"
if news_nlp_file.exists():
    df_news_nlp = pd.read_csv(news_nlp_file)
    print(f"\nNews NLP articles: {news_nlp_file}")
    print(f"Shape: {df_news_nlp.shape}, Columns: {df_news_nlp.columns.tolist()}")
    if "published_date" in df_news_nlp.columns:
        print(f"Date range: {df_news_nlp['published_date'].min()} to {df_news_nlp['published_date'].max()}")
    elif "Date" in df_news_nlp.columns:
        print(f"Date range: {df_news_nlp['Date'].min()} to {df_news_nlp['Date'].max()}")

print("\n" + "=" * 80)
print("AUDIT 3: DECISION FUSION DATASET")
print("=" * 80)
fusion_file = PROCESSED_DIR / "multi_company_decision_fusion.csv"
if fusion_file.exists():
    df_fusion = pd.read_csv(fusion_file)
    print(f"Decision fusion file: {fusion_file}")
    print(f"Shape: {df_fusion.shape}")
    print(f"Columns: {df_fusion.columns.tolist()}")
    print(f"Date range: {df_fusion['Date'].min()} to {df_fusion['Date'].max()}")
    print("Decision counts overall:")
    print(df_fusion["Final_Decision"].value_counts())
    print("\nDecision counts per ticker:")
    print(df_fusion.groupby(["Ticker", "Final_Decision"]).size().unstack(fill_value=0))

print("\n" + "=" * 80)
print("AUDIT 4: NOTEBOOK 08 SPLIT & TRAINING METADATA")
print("=" * 80)
nb_path = PROJECT_ROOT / "notebooks" / "08_decision_fusion.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for idx in [5, 6, 7, 8, 9, 10, 11, 12, 17, 18, 20, 21, 22, 25, 26]:
    if idx < len(nb["cells"]):
        c = nb["cells"][idx]
        print(f"--- Cell {idx} ({c['cell_type']}) ---")
        src = "".join(c["source"])
        # print first 15 lines
        lines = src.split("\n")
        print("\n".join(lines[:20]))
        if len(lines) > 20:
            print(f"... ({len(lines)-20} more lines)")
