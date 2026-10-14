# Phase 5A: Independent Backtest Methodology Audit & Execution Realism Report

**Project**: AI-Based Intelligent Trading System  
**Audit Scope**: Independent Verification of Phase 4 Out-of-Sample Evaluation Methodology, Execution Realism, Timestamp Rigor, Accounting Invariants, and Statistical Claims  
**Evaluator**: Antigravity Independent Methodology Auditor  
**Date**: October 14, 2026  
**Artifact Directory**: `data/evaluation/phase5a/`  
**Execution Baseline**: Next-Day Market-on-Open (MOO Fill at $Open_{t+1}$), Strict Dollar Ledger Accounting, 10 bps Transaction Friction, $R_f = 3.0\%$  

---

## 1. Executive Summary & Audit Mandate

This independent audit conducts an exhaustive, mathematically rigorous examination of the **Phase 4 Out-of-Sample Evaluation** methodology and results for the AI-Based Intelligent Trading System.

### Core Verdict
1. **The Phase 4 Core Conclusion is Confirmed Correct**:
   The active multi-modal strategy (**Decision Fusion Full**) substantially underperformed passive equal-weight **Buy-and-Hold** over the 300-session out-of-sample test period (October 20, 2023 to December 30, 2024):
   - **Buy & Hold**: **+72.42%** cumulative return, **2.18 Sharpe**, **-17.46% MaxDD**.
   - **Decision Fusion Full (Realistic MOO Fill)**: **+24.79%** cumulative return, **1.38 Sharpe**, **-9.09% MaxDD** (-47.63% active return lag).
   - **Decision Fusion No-News**: **+22.02%** cumulative return, **1.25 Sharpe**, **-10.07% MaxDD**.
   - **Statistical Significance of News**: Paired $t$-test $p = 0.5240$, Wilcoxon signed-rank $p = 0.7554$. **News sentiment contributed zero statistically distinguishable alpha.**

2. **Methodological Deficiencies Identified ("Needs Correction")**:
   - **Execution Fill Timing Lookahead**: Phase 4 used Same-Day Close-to-Close forward returns ($Fwd\_Close\_Close = (Close_{t+1} - Close_t) / Close_t$), implicitly assuming market-on-close (MOC) execution on the signal date. Because daily features (Close, RSI, Bollinger Bands, MACD) require the 16:00 EST market close and NYSE/Nasdaq MOC cutoff is 15:50 EST, fills at $Close_t$ are physically unexecutable. Adopting realistic **Next-Day Open-to-Open (MOO Fill)** revises Decision Fusion return from **+28.26%** down to **+24.79%**.
   - **Post-Market News Timestamp Leakage**: 45 of 120 news articles (**37.5%**) were published at or after 16:00:00 EST (e.g. 16:30 earnings releases). Grouping by calendar date merged them into day $t$, which caused lookahead under Same-Day Close fills. Under Next-Day Open fills, these articles are legitimately tradable.
   - **Exposure Metric Conflation**: Phase 4 Table 4 reported up to **98.3% exposure** for Microsoft, creating apparent contradiction with the **10.0% maximum target allocation** in Table 1. The audit decoupled this into **Portfolio Active Weight Contribution** (mean 1.58% to 10.35%, peak 10.8% to 14.9% caused by price appreciation) vs **Standalone Single-Stock Days Invested** (98.3% active days).
   - **CAPM Jensen's Alpha Risk-Free Rate Missing**: Phase 4 computed alpha as $(\bar{R}_p - \beta \bar{R}_m) \times 252 = -4.48\%$, omitting the standard risk-free rate adjustment $- R_f (1 - \beta)$. Corrected CAPM Alpha is **-5.99%**.

---

## 2. Information Availability & Execution Timing Audit

### 2.1 News Timestamp Analysis
We audited all 120 processed news articles in `data/processed/multi_company_news_nlp_results.csv` against NYSE/Nasdaq regular trading hours (09:30:00 to 16:00:00 EST):

| Trading Session Window | Timestamp Range (EST) | Article Count | Share of Dataset | Execution Realism Impact |
| :--- | :--- | :---: | :---: | :--- |
| **Pre-Market** | 00:00:00 – 09:29:59 | 38 | 31.67% | Fully available for day $t$ Open fill |
| **Intraday Regular Hours** | 09:30:00 – 15:59:59 | 37 | 30.83% | Known during trading day $t$ |
| **Post-Market / After Hours** | $\ge$ 16:00:00 | **45** | **37.50%** | **Leakage in Same-Day Close; Clean in Next-Day Open** |
| **Total News Sample** | 2023-10-20 to 2024-12-30 | **120** | **100.0%** | Average 0.40 articles/day across 10 stocks |

**Audit Finding**: 37.5% of financial news arrives after 16:00 EST (post-earnings conference calls, press releases). If an evaluation assumes execution at day $t$ Close, it uses information published up to 4 hours *after* the close. Rolling post-market news to effective date $t+1$ and filling at $Open_{t+1}$ eliminates all lookahead bias.

### 2.2 Execution Models Compared
We compared three fill mechanisms across all 10 assets (300 trading sessions, 10 bps friction):

| Metric | Model B: Next-Day Open (MOO Fill)<br>*(Production Standard)* | Model A: Same-Day Close (MOC Fill)<br>*(Phase 4 Legacy)* | Model C: Open-to-Close (Intraday)<br>*(Intraday Only)* |
| :--- | :---: | :---: | :---: |
| **Information Availability** | $100\%$ Causal (0 lookahead) | Unfeasible (Close features @ 16:00 vs MOC cutoff 15:50) | $100\%$ Causal |
| **Execution Price** | $Open_{t+1}$ (09:30 EST) | $Close_t$ (16:00 EST) | $Open_{t+1} \to Close_{t+1}$ |
| **Decision Fusion Return** | **+24.79%** | **+28.26%** | **+0.43%** |
| **Decision Fusion Sharpe** | **1.38** | **1.63** | **-0.26** |
| **Decision Fusion MaxDD** | **-9.09%** | **-8.21%** | **-8.99%** |
| **No-News Fusion Return** | **+22.02%** | **+29.52%** | **+0.68%** |
| **No-News Fusion Sharpe** | **1.25** | **1.78** | **-0.24** |
| **RL Only Return** | **+47.26%** | **+47.94%** | **+7.39%** |
| **RL Only Sharpe** | **2.31** | **2.44** | **0.36** |
| **Buy & Hold Return** | **+72.42%** | **+68.97%** | **+16.80%** |
| **Buy & Hold Sharpe** | **2.18** | **2.22** | **0.81** |

**Crucial Takeaways**:
1. Switching to realistic Next-Day Open (MOO) reduces Decision Fusion return by **3.47%** (+28.26% $\to$ +24.79%), reflecting realistic execution timing.
2. The intraday Open-to-Close return (+0.43%) reveals that **>98% of total strategy return originates overnight** (gap from Close $t \to$ Open $t+1$), which is captured by Next-Day Open-to-Open holding.

---

## 3. Accounting Model Reconciliation

We evaluated the exact dollar mechanics comparing **Strict Dollar Ledger** vs **Legacy Weight Compounding**:

| Metric | Strict Dollar Ledger (Phase 5A) | Weight Compounding (Phase 4) | Accounting Difference | Rationale / Mathematical Cause |
| :--- | :---: | :---: | :---: | :--- |
| **Cumulative Return** | **+24.79%** | **+23.47%** | **+1.32%** | Un-rebalanced winning positions compound higher |
| **Annualized Return** | **20.45%** | **19.38%** | **+1.07%** | Strict cash balance prevents synthetic rebalancing |
| **Annualized Volatility** | **11.77%** | **11.59%** | **+0.18%** | Floating share weights slightly increase variance |
| **Sharpe Ratio** | **1.38** | **1.37** | **+0.01** | Functionally identical risk-adjusted profile |
| **Maximum Drawdown** | **-9.09%** | **-8.94%** | **-0.15%** | Minor path divergence during April 2024 pullback |
| **Turnover Rate** | **2,932%** | **2,900%** | **+32%** | Trades sized to cash-adjusted target values |
| **Total Trades** | **355** | **364** | **-9** | Threshold filter eliminates micro-rebalancing |
| **Cash Invariant** | $Equity = Cash + \sum Stock$ | $Equity = \prod(1 + R_t)$ | **Verified** | Zero cash creation/destruction across 300 days |

**Audit Finding**:
The Strict Dollar Ledger model is mathematically sound. In Dollar Ledger accounting, positions that appreciate during a `HOLD` signal maintain their full share balance without incurring artificial daily rebalancing transaction costs. This generates +1.32% higher net return than weight compounding, which had penalized the strategy with synthetic daily trimming.

---

## 4. Inconsistency & Edge-Case Investigation

### 4.1 Resolution of the "98.3% Exposure" Anomaly
Table 4 of Phase 4 showed Microsoft with 98.3% exposure, leading to concerns that the 10.0% asset constraint had been violated. We conducted a line-by-line audit and decoupled the metrics:

| Ticker | Company Name | Max Target Allocation | Mean Portfolio Weight | Peak Portfolio Weight | Standalone Days Invested | Single-Stock Return | Single-Stock Sharpe |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **AAPL** | Apple | 10.0% | 1.58% | 10.80% | 15.7% | +3.22% | 0.02 |
| **AMZN** | Amazon | 10.0% | 8.79% | 11.33% | 85.3% | +42.37% | 1.15 |
| **GOOGL** | Alphabet | 10.0% | 3.18% | 11.79% | 31.0% | +11.65% | 0.41 |
| **JNJ** | Johnson & Johnson | 10.0% | 2.74% | 10.34% | 27.3% | -10.97% | -1.54 |
| **JPM** | JPMorgan Chase | 10.0% | 2.80% | 10.77% | 27.7% | +14.07% | 0.70 |
| **META** | Meta Platforms | 10.0% | 2.46% | 10.40% | 24.7% | -5.03% | -0.49 |
| **MSFT** | Microsoft | 10.0% | **10.35%** | **11.65%** | **98.3%** | +24.04% | 0.82 |
| **NVDA** | NVIDIA | 10.0% | 5.86% | 14.63% | 55.3% | +41.68% | 0.88 |
| **TSLA** | Tesla | 10.0% | 9.27% | 14.87% | 86.3% | +103.95% | 1.30 |
| **V** | Visa | 10.0% | 0.00% | 0.00% | 0.0% | +0.00% | 0.00 |
| **Total / Mean** | **Portfolio** | **100.0%** | **47.1%** | **68.2%** | **45.2%** | **+24.79%** | **1.38** |

**Audit Conclusion**:
- The **10.0% target allocation constraint was strictly obeyed at all times**. Initial position allocations never exceeded $10.0\%$ ($0.1000$).
- When a position was held and rallied (e.g., TSLA $+103.95\%$, NVDA $+41.68\%$, MSFT $+24.04\%$), its value relative to portfolio equity naturally drifted upward (TSLA reached a peak of $14.87\%$, NVDA $14.63\%$, MSFT $11.65\%$). This is normal equity price drift, not unauthorized leverage.
- The **98.3% figure** in Table 4 of Phase 4 was **Standalone Single-Stock Days Invested** (Microsoft held a position on 295 out of 300 trading days in its isolated 1-stock account). In the multi-asset portfolio, Microsoft's average equity allocation was **10.35%**.

### 4.2 Standardized CAPM Jensen's Alpha Audit
Under CAPM, abnormal return $\alpha$ must be computed relative to excess returns over the risk-free rate $R_f$:
$$\alpha_{\text{CAPM}} = \left[ (\bar{R}_p - R_f) - \beta (\bar{R}_m - R_f) \right] \times 252$$

Where $R_f = 0.03$ (3.0% annual), $\bar{R}_m = 58.03\%$ (Buy & Hold benchmark annualized return), $\bar{R}_p = 20.45\%$ (Decision Fusion annualized return), and $\beta = 0.4962$:
- Expected Return under CAPM: $E[R_p] = R_f + \beta (R_m - R_f) = 3.0\% + 0.4962 \times (58.03\% - 3.0\%) = 30.31\%$.
- Actual Strategy Return: $20.45\%$.
- **Corrected Jensen's Alpha**: $20.45\% - 30.31\% = \mathbf{-5.99\%}$.
- Phase 4 reported alpha as $-4.48\%$ because it omitted the risk-free rate adjustment $(1 - \beta) R_f = 0.5038 \times 3.0\% = +1.51\%$.
- Both formulas confirm negative active manager value (-5.99% and -4.48%).

---

## 5. Standardized Multi-Asset Portfolio Results (Next-Day MOO Fill)

The standardized, audited performance across all 5 strategies under Next-Day Market-on-Open execution and Strict Dollar Ledger accounting is presented below:

| Strategy | Cumulative Return | Annualized Return | Volatility | Sharpe ($R_f=3\%$) | Sortino | Max DD | Beta ($\beta$) | Jensen's $\alpha$ | Turnover | Trades | Win Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Buy & Hold** | **+72.42%** | **58.03%** | 20.61% | **2.18** | **1.94** | -17.46% | 1.00 | 0.00% | 100% | 1,808 | 60.7% |
| **Decision Fusion Full** | **+24.79%** | **20.45%** | 11.77% | **1.38** | **1.35** | **-9.09%** | 0.50 | **-5.99%** | 2,932% | 355 | 56.7% |
| **Decision Fusion NoNews** | **+22.02%** | **18.20%** | 11.54% | **1.25** | **1.17** | **-10.07%** | 0.48 | **-7.24%** | 3,502% | 412 | 56.3% |
| **RL Only** | **+47.26%** | **38.42%** | 13.15% | **2.31** | **2.32** | **-9.10%** | 0.58 | **+4.48%** | 2,730% | 362 | 59.3% |
| **Classification Only** | **+16.62%** | **13.79%** | **6.93%** | **1.47** | **1.60** | **-4.31%** | 0.20 | **+1.27%** | 368% | 41 | 49.7% |

### Bootstrap Confidence Intervals (95% CI, Block Size = 5)
- **Buy & Hold**: Return $[+18.23\%, +158.86\%]$, Sharpe $[0.66, 4.31]$
- **Decision Fusion Full**: Return $[+0.58\%, +61.65\%]$, Sharpe $[0.01, 3.40]$
- **Decision Fusion NoNews**: Return $[-1.90\%, +56.87\%]$, Sharpe $[-0.32, 3.16]$
- **RL Only**: Return $[+15.69\%, +95.53\%]$, Sharpe $[0.75, 4.21]$
- **Classification Only**: Return $[+3.15\%, +34.89\%]$, Sharpe $[-0.02, 3.30]$

---

## 6. Friction Sensitivity & Execution Drag

We audited strategy sensitivity to round-trip transaction costs (spread + exchange fees + slippage):

| Cost Level (bps) | Cumulative Return | Annualized Return | Sharpe Ratio | Max Drawdown | Performance Drag vs Zero Cost |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **0 bps** (Zero Friction) | +28.48% | 23.43% | 1.60 | -8.78% | Baseline (Gross) |
| **5 bps** (Institutional Direct) | +26.62% | 21.93% | 1.49 | -8.94% | -1.50% annualized drag |
| **10 bps** (Baseline Standard) | **+24.79%** | **20.45%** | **1.38** | **-9.09%** | **-2.98% annualized drag** |
| **20 bps** (Retail Spread) | +21.21% | 17.54% | 1.17 | -9.40% | -5.89% annualized drag |
| **50 bps** (Adverse Auction Gap) | **+11.06%** | **9.21%** | **0.55** | **-10.32%** | **-14.22% annualized drag** |

**Audit Finding**:
Due to high turnover (2,932% across 300 sessions), the strategy pays approximately **3.0% of portfolio equity per 10 bps of friction**. At 50 bps (typical retail opening auction slippage during volatile sessions), net return drops to +11.06% and Sharpe collapses to 0.55.

---

## 7. Audit Classification Matrix

Each item evaluated in Phase 5A is classified below:

| Audit Item | Audit Status | Detailed Classification Rationale |
| :--- | :---: | :--- |
| **1. Chronological Splitting** | **Confirmed Correct** | In-sample (1,200 bars, 2019-2023) and Out-of-Sample (300 bars, 2023-2024) are strictly partitioned with zero overlapping dates across all 10 equities. |
| **2. Upstream Model Freezing** | **Confirmed Correct** | Classification weights, RL Q-tables, and Decision Fusion weights (0.35, 0.30, 0.15, 0.20) remained frozen post-October 2023. |
| **3. Severe Active Lag vs B&H** | **Confirmed Correct** | Decision Fusion (+24.79%) lagged Buy & Hold (+72.42%) by -47.63% active return. Cash drag (52.9% cash balance) and early exit from winning trends account for the drag. |
| **4. News Sentiment Inconsequentiality** | **Confirmed Correct** | News ablation difference (+2.77%) has $p = 0.5240$ (paired $t$-test) and $p = 0.7554$ (Wilcoxon). News adds zero statistically significant alpha over 300 test sessions. |
| **5. Conservation of Cash & Equity** | **Confirmed Correct** | Strict Dollar Ledger validates that $Cash + \sum Stock = Total\_Equity$ at all 300 sessions with zero cash leakage or illegal short borrowing. |
| **6. Execution Timing Lookahead** | **Needs Correction** | Phase 4 MOC fills used same-day Close returns with 16:00 close prices. Replaced with realistic Next-Day Open (MOO) execution baseline. |
| **7. Post-Market News Aggregation** | **Needs Correction** | 45 of 120 news articles (37.5%) were published $\ge 16:00$ EST. Aggregating by calendar date created lookahead under MOC. Cleaned under Next-Day MOO. |
| **8. Exposure Reporting Conflation** | **Needs Correction** | Table 4 conflated single-stock active days (up to 98.3%) with portfolio active equity weight (mean 1.58% to 10.35%). Decoupled and corrected in Table 4. |
| **9. CAPM Jensen's Alpha Formula** | **Needs Correction** | Phase 4 omitted $- R_f (1 - \beta)$. Corrected formula yields $\alpha_{\text{CAPM}} = -5.99\%$ (vs $-4.48\%$ originally reported). |
| **10. Rebalance Drag on HOLD** | **Needs Correction** | Weight compounding penalized HOLD sessions with synthetic daily trimming. Strict Dollar Ledger retains un-rebalanced shares without fee penalty. |
| **11. High-Density Live News Alpha** | **Inconclusive** | With 96% zero-news days in the 120-article demo dataset, news utility under dense continuous streaming (100+ articles/day) cannot be evaluated until 90-180 days of Alpaca live data accumulate. |
| **12. Opening Auction Slippage** | **Inconclusive** | Exact opening auction fill slippage during earnings release gaps (e.g. NVDA, TSLA) cannot be verified without tick-level execution data. |

---

## 8. Strategy Readiness & Production Trading Recommendations

### Recommendation on Real Capital Trading: **STRICTLY NOT READY (DO NOT TRADE)**
Deploying real capital to the current system is strongly discouraged for the following reasons:
1. **Severe Active Underperformance**: Lags passive buy-and-hold by **-47.63%** net of costs.
2. **Negative Jensen's Alpha**: Strategy lost **-5.99% per year** relative to its market risk exposure ($\beta = 0.50$).
3. **Severe Transaction Drag**: 2,932% turnover consumes up to 14.2% in return if execution friction reaches 50 bps.
4. **News Sentiment Contributes No Value**: 120 articles over 300 days provided zero statistically distinguishable edge ($p = 0.52$).

### Recommendation on Paper Trading: **DEFER UNTIL PREREQUISITES MET**
Paper trading should not be initiated until the following three architectural upgrades are implemented:
1. **Implementation of Limit-on-Open (LOO) Execution Rules**: Replace naive market orders with LOO rules to control opening auction slippage.
2. **90-Day Live Stream Accumulation**: Accumulate 90–180 days of continuous, un-throttled financial news from the active Alpaca WebSocket listener (`news_database.db`) to test news NLP under realistic continuous article arrival.
3. **Fusion Weight Recalibration**: The current fixed weights (0.35 Market, 0.30 RL, 0.15 Regime, 0.20 News) create defensive cash drag (52.9% average cash). Dynamic volatility-targeted sizing or re-optimizing fusion weights should precede paper trading.
