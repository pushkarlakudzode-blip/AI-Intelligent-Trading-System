# Phase 4: Rigorous Out-of-Sample Evaluation Report
## AI-Based Intelligent Trading System

**Evaluation Date**: October 2026  
**Test Window**: October 20, 2023 – December 30, 2024 (300 Out-of-Sample Trading Sessions)  
**Universe**: 10 Equities (`AAPL`, `AMZN`, `GOOGL`, `JNJ`, `JPM`, `META`, `MSFT`, `NVDA`, `TSLA`, `V`)  
**Evaluation Harness**: [`src/evaluation/out_of_sample_evaluator.py`](file:///c:/Users/pushk/OneDrive/Desktop/Pushkar/AI-Intelligent-Trading-System/src/evaluation/out_of_sample_evaluator.py)  
**Artifact Directory**: [`data/evaluation/phase4/`](file:///c:/Users/pushk/OneDrive/Desktop/Pushkar/AI-Intelligent-Trading-System/data/evaluation/phase4/)

---

## 1. Executive Summary

Phase 4 establishes an institutional-grade, zero-leakage out-of-sample evaluation of the AI-Based Intelligent Trading System across all 10 tracked equities.

### Key Empirical Findings:
1. **Severe Underperformance Relative to Buy-and-Hold**:
   - The **Equal-Weight Buy & Hold Benchmark** achieved **+68.01% cumulative return** (+54.63% annualized, Sharpe 2.40, Max Drawdown -12.43%).
   - The **Existing Decision Fusion Strategy (Full 4-Channel with News)** achieved **+26.37% cumulative return** (+21.73% annualized, Sharpe 1.60, Max Drawdown -7.98%).
   - Active strategy alpha was **-4.76% annualized** ($\beta = 0.55$) due to extreme capital idling: Decision Fusion spent **54.8% of the time in cash** (mean exposure: 45.2%), sitting out massive rallies in `NVDA` (+224.2% B&H vs +83.1% Fusion), `META` (+89.5% B&H vs -2.6% Fusion), and `AAPL` (+44.7% B&H vs +2.0% Fusion).
   - In `V` (Visa), the system had **zero BUY signals across all 300 days**, resulting in 0.0% return vs +35.3% Buy-and-Hold.

2. **News Sentiment Demonstrates ZERO Incremental Trading Value**:
   - **Ablated Decision Fusion (Market-Only, No News)** achieved **+28.70% cumulative return** (Sharpe 1.80, Max Drawdown -8.69%).
   - The news component actually **reduced cumulative return by -2.33%** and reduced the Sharpe ratio from 1.80 to 1.60.
   - Paired $t$-test between daily returns of Full Fusion vs. No-News Fusion: $t = -0.5783$, $p = 0.5635$ (**not statistically significant**).
   - Wilcoxon signed-rank test: $W = 3494.5$, $p = 0.8413$ (**not statistically significant**).
   - **Root Cause**: Breaking news was present on only **4.0% of test sessions** (12 days per stock across 300 sessions). The remaining 96.0% were zero-padded neutrals. The few news catalysts that did fire diluted prevailing bullish technical momentum on key breakout days.

3. **High Turnover Creates Substantial Friction Drag**:
   - The multi-modal strategy generated **2,900% total portfolio turnover** (29 full portfolio turnovers over 300 trading days).
   - At 10 bps transaction friction (commission + slippage), cumulative return drops from **+30.09% (frictionless)** to **+26.37%**. At 50 bps, return collapses to **+12.53%** (Sharpe 0.70).

4. **Execution Timing Microstructure Risk**:
   - When executed Market-on-Close (Close-to-Close), return is **+26.37%** (Sharpe 1.60).
   - When executed Market-on-Open (holding intraday Open-to-Close), return drops to **+0.77%** (Sharpe 0.12), proving that virtually all equity alpha in this period was overnight gap risk rather than intraday drift.

---

## 2. Recommendation on Paper Trading Readiness

### Verdict: **NOT READY FOR LIVE CAPITAL OR UNRESTRICTED PAPER TRADING (CONDITIONAL SANDBOX ONLY)**

> [!WARNING]
> While the code infrastructure is mathematically sound and adheres to zero-leakage standards, deploying the current decision-fusion policy to live paper trading would misallocate capital.

### Specific Architectural Deficiencies Precluding Live Deployment:
1. **Severe Bull-Market Opportunity Cost**:
   - The fixed $\pm 0.20$ threshold and equal-weight consensus logic produced 57.6% HOLD decisions, completely missing secular momentum in mega-cap equities.
2. **Component Disconnect & Asymmetric Bias**:
   - The Logistic Regression classification model suffered from negative directional bias across all equities (mean signal contribution: -0.01 to -0.08), constantly depressing the fused score below the BUY threshold.
   - For `V` (Visa), the tabular RL agent generated persistent SELL actions, locking the fusion score near -0.20 and producing 0 BUY signals in 14 months.
3. **News Pipeline Sparsity**:
   - The current historical news feature set (120 demo articles) cannot support production alpha. Live news ingestion via the newly implemented Alpaca WebSocket listener must accumulate at least 90–180 days of continuous multi-article daily histories before news signals can be credibly included in live trading decisions.
4. **Lack of Dynamic Rebalancing & Volatility-Targeted Position Sizing**:
   - The existing architecture uses discrete state jumps (0% $\to$ 10% $\to$ 0%) with binary thresholds rather than continuous Kelly or risk-parity sizing, driving excessive turnover (29x) and whipsaw friction.

---

## 3. Comprehensive Performance Tables

### Table 1: Multi-Asset Portfolio Performance (10 Equities, 10 bps Friction)
All strategies simulated on \$100,000 initial capital across 300 chronological out-of-sample sessions (`2023-10-20` to `2024-12-30`).

| Strategy | Cum. Return | Ann. Return | Ann. Vol | Sharpe ($R_f=3\%$) | Sortino | Max DD | Mean Exp. | Turnover | Beta ($\beta$) | Alpha ($\alpha$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Buy & Hold Benchmark** | **+68.01%** | **+54.63%** | 17.55% | **2.40** | 2.37 | -12.43% | 100.0% | 100% | 1.00 | 0.00% |
| **RL Policy Only** | +46.19% | +37.57% | 12.03% | 2.46 | 2.62 | -8.55% | 50.1% | 2680% | 0.63 | +3.96% |
| **Decision Fusion (No News)** | +28.70% | +23.61% | 10.43% | 1.80 | 1.83 | -8.69% | 44.8% | 3470% | 0.53 | -2.20% |
| **Decision Fusion (Full with News)**| +26.37% | +21.73% | 10.75% | 1.60 | 1.64 | -7.98% | 45.2% | 2900% | 0.55 | -4.76% |
| **Classification Only** | +13.59% | +11.30% | 5.95% | 1.32 | 1.40 | -4.02% | 18.2% | 350% | 0.23 | +0.51% |

### Uncertainty Estimates (Stationary Block Bootstrap, 1,000 Iterations, 95% CI)
- **Buy & Hold**: Sharpe `[0.93, 4.38]` | Cum. Return `[+23.30%, +150.57%]`
- **Decision Fusion (Full)**: Sharpe `[0.11, 3.67]` | Cum. Return `[+4.34%, +63.63%]`
- **Decision Fusion (No News)**: Sharpe `[0.17, 3.78]` | Cum. Return `[+5.13%, +62.68%]`
- **RL Only**: Sharpe `[0.93, 4.46]` | Cum. Return `[+16.99%, +95.84%]`
- **Classification Only**: Sharpe `[-0.23, 3.05]` | Cum. Return `[+1.70%, +28.29%]`

---

### Table 2: Statistical News Ablation Significance (Full vs. No News)
Evaluating the null hypothesis $H_0: \mu_{\text{Full}} = \mu_{\text{NoNews}}$ on daily portfolio returns.

| Test Metric | Value | Interpretation |
| :--- | :---: | :--- |
| **Mean Daily Return Difference** | -0.000060 (-1.50% annualized) | Full with news slightly lagged No-News |
| **Paired $t$-statistic** | -0.5783 | Difference is within random variance |
| **Paired $t$-test $p$-value** | **0.5635** | **Fail to reject $H_0$** (No significant alpha) |
| **Wilcoxon Signed-Rank Statistic**| 3,494.5 | Non-parametric test confirms $t$-test |
| **Wilcoxon $p$-value** | **0.8413** | **Fail to reject $H_0$** |
| **Conclusion** | **NOT SIGNIFICANT** | News NLP does not add measurable value |

---

### Table 3: Transaction Cost & Slippage Drag Sensitivity
Simulating the Full Decision Fusion portfolio under varying friction tiers.

| Friction Tier | Cum. Return | Ann. Return | Ann. Vol | Sharpe | Max Drawdown | Total Turnover |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **0 bps (Frictionless)** | +30.09% | +24.73% | 10.74% | 1.83 | -7.63% | 2900% |
| **5 bps (Low Institutional)** | +28.22% | +23.22% | 10.74% | 1.72 | -7.80% | 2900% |
| **10 bps (Baseline Model)** | **+26.37%** | **+21.73%** | **10.75%** | **1.60** | **-7.98%** | **2900%** |
| **20 bps (Retail Spread/Impact)**| +22.76% | +18.80% | 10.75% | 1.38 | -8.33% | 2900% |
| **50 bps (Adverse Execution)** | +12.53% | +10.43% | 10.78% | 0.70 | -9.37% | 2900% |

---

### Table 4: Per-Stock Breakdown (Decision Fusion Full vs. Buy & Hold)
Simulated as individual standalone accounts (\$100,000 each, 10 bps friction, Long-Only).

| Ticker | Company Name | Fusion CumRet | Fusion Sharpe | Fusion MaxDD | Fusion Exposure | B&H CumRet | B&H Sharpe | B&H MaxDD | Active Diff |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **AAPL** | Apple | +2.02% | -0.08 | -6.93% | 15.7% | +44.71% | 1.40 | -16.71% | -42.69% |
| **AMZN** | Amazon | +56.26% | 1.48 | -11.95% | 85.3% | +75.10% | 1.70 | -19.49% | -18.84% |
| **GOOGL**| Alphabet | -1.55% | -0.11 | -13.08% | 31.0% | +39.46% | 1.02 | -22.24% | -41.01% |
| **JNJ** | Johnson & Johnson | -7.70% | -1.17 | -17.50% | 27.3% | -5.57% | -0.44 | -14.36% | -2.12% |
| **JPM** | JPMorgan Chase | +17.27% | 0.95 | -12.12% | 27.7% | +67.52% | 1.92 | -10.13% | -50.24% |
| **META** | Meta Platforms | -2.60% | -0.34 | -15.61% | 24.7% | +89.51% | 1.62 | -18.43% | -92.12% |
| **MSFT** | Microsoft | +24.79% | 0.88 | -15.49% | 98.3% | +28.90% | 1.02 | -15.49% | -4.12% |
| **NVDA** | NVIDIA | +83.14% | 1.61 | -30.28% | 55.3% | +224.16%| 2.18 | -27.05% | -141.02%|
| **TSLA** | Tesla | +104.13%| 1.28 | -49.34% | 86.3% | +90.31% | 1.14 | -45.67% | +13.82% |
| **V** | Visa | +0.00% | 0.00 | 0.00% | 0.0% | +35.28% | 1.45 | -12.61% | -35.28% |

---

## 4. Key Limitations & Methodological Disclosures

1. **Extreme News Data Sparsity**:
   The historical news dataset consists of only 120 articles across 300 test days (12 articles per ticker). On 96% of days, `NLP_Signal = 0.0`. Evaluating the causal alpha of news requires a dense, continuous news dataset spanning years.
2. **Survivorship Bias in Asset Universe**:
   The 10 tracked equities represent the mega-cap survivors of the 2019–2024 US equity market. The strong Buy-and-Hold benchmark performance (+68.0%) reflects this selection bias.
3. **Bull Market Regime Specificity**:
   The out-of-sample test window (Oct 2023 – Dec 2024) was characterized by a massive AI-driven equity expansion. A strategy with a strong cash-preservation bias (55% idle cash) inherently underperformed a buy-and-hold index during this unique period.
4. **Execution Microstructure Assumptions**:
   Market-on-Close fills assume execution at the official 4:00 PM closing price without market impact. In practice, closing auction volume and order imbalances could introduce additional slippage.

---

## 5. Roadmap to Paper-Trading Readiness

To prepare this system for live paper trading, the following engineering enhancements must be executed:

1. **Accumulate Continuous Live News**:
   Run the newly built Alpaca WebSocket streamer in background daemon mode to accumulate real-time multi-headline coverage for at least 90 continuous trading sessions.
2. **Dynamic Volatility-Scaled Sizing**:
   Replace binary discrete thresholds ($\pm 0.20$) with continuous sigmoid or Kelly-criterion sizing to eliminate whipsaw turnover and allow fractional exposure.
3. **Walk-Forward Model Retraining**:
   Implement rolling quarterly retraining for the classification and RL models to prevent stale policy drift.
4. **Explicit Short-Borrow and Overnight Gap Modeling**:
   Incorporate intraday execution logic and borrow cost limits before enabling short positions.
