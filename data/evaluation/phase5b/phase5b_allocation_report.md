# Phase 5B: Portfolio Allocation, Cooldown & Turnover Diagnosis Report

**Project**: AI-Based Intelligent Trading System  
**Audit Scope**: Diagnosis of Cash Drag, Premature Exits, and High Turnover; Experimental Evaluation of Anti-Whipsaw Cooldown Rules, Volatility-Scaled Sizing, and Dynamic Cash Management  
**Baseline**: Phase 5A Audit Baseline (Next-Day MOO Fill, Strict Dollar Ledger, 10 bps friction, $R_f = 3.0\%$)  
**Date**: October 14, 2026  
**Artifact Directory**: `data/evaluation/phase5b/`  

---

## 1. Executive Summary & Diagnosis Mandate

In Phase 5A, the independent backtest audit confirmed that the active multi-modal trading strategy (**Decision Fusion Full**) achieved **+24.79%** cumulative return under realistic Next-Day Market-on-Open execution, lagging the passive **Buy-and-Hold** benchmark (**+72.42%**) by **-47.63%** active return. The audit highlighted three critical operational symptoms:
1. **Excessive Cash Drag**: The portfolio maintained an average of **52.9% in idle cash**.
2. **Extreme Turnover**: Total portfolio turnover reached **2,932%** (to 3,356%), incurring 350+ trades in 300 sessions.
3. **Premature Exits & Whipsawing**: Positions were frequently liquidated after only 2–3 days of holding, missing substantial compounding during major uptrends.

### Core Objectives of Phase 5B:
- **Audit the exact codebase mechanics** causing cash drag, rapid liquidations, and turnover.
- **Formulate mathematically grounded remedies**: anti-whipsaw cooldown periods, hysteresis exit thresholds, volatility-scaled risk parity sizing, and dynamic cash fungibility.
- **Conduct strictly separated out-of-sample experiments**: calibrate parameters exclusively on a **Validation Period** (first 150 trading days: 2023-10-20 to 2024-05-24) and evaluate generalization on an **Untouched Final Test Period** (last 150 trading days: 2024-05-28 to 2024-12-30) and the full 300-session history.
- **Preserve production systems**: zero mutation of live database (`news_database.db`), trained models, or Phase 1–6 research artifacts.

---

## 2. Root-Cause Diagnosis: Why the System Stays in Cash and Whipsaws

Our line-by-line code audit and measured empirical diagnosis reveal three primary structural drivers:

### 2.1 Driver 1: The Non-Fungible 10% Sizing Cap (Structural Cash Trap)
- **Code Reference**: [`src/evaluation/out_of_sample_evaluator.py#L494-L497`](file:///c:/Users/pushk/OneDrive/Desktop/Pushkar/AI-Intelligent-Trading-System/src/evaluation/out_of_sample_evaluator.py#L494-L497)
  ```python
  if dec == "BUY":
      target_val = max_asset_weight * total_eq_start  # max_asset_weight = 0.10
      delta = target_val - curr_val
      if delta > 0:
          buy_amt = min(delta, cash)
  ```
- **The Mechanism**: Each of the 10 portfolio equities is rigidly constrained to a maximum of $10.0\%$ ($0.10$) of portfolio equity. Cash freed by sold positions is **siloed and non-fungible**—it cannot be redeployed to other assets that are currently signaling strong BUY or trending upward.
- **Measured Telemetry**:
  - Across 300 trading sessions, the portfolio held an average of only **4.52 active stocks out of 10** (median: 5, min: 1, max: 7).
  - On **zero days (0%)** were 8, 9, or 10 stocks held simultaneously.
  - **Visa (`V`)**: Received **0 BUY signals** across the entire 300-day evaluation period. Consequently, Visa's 10% allocation capacity sat in **100% idle cash for all 300 sessions**.
  - **Alphabet (`GOOGL`)**: Received only 10 BUY signals in 300 days (invested only 31.0% of days).
  - **Apple (`AAPL`)**: Received 21 BUY signals vs 105 SELL signals (invested only 15.7% of days).
  - **Meta (`META`)**: Invested only 24.7% of days.
  - **Direct Mathematical Consequence**: $4.52 \text{ active stocks} \times 10\% = 45.2\%$ equity exposure, forcing an average of **54.8% structural idle cash** (empirically observed at **52.9%** after price drift).

### 2.2 Driver 2: High Sensitivity to 1-Day RL Noise & Symmetric Thresholds (Premature Exits)
- **Code Reference**: [`src/evaluation/out_of_sample_evaluator.py#L272-L280`](file:///c:/Users/pushk/OneDrive/Desktop/Pushkar/AI-Intelligent-Trading-System/src/evaluation/out_of_sample_evaluator.py#L272-L280)
  ```python
  Fusion_Score = (0.35 * Class_Signal + 0.30 * RL_Signal + 0.15 * Regime_Signal + 0.20 * NLP_Signal)
  Decision = "BUY" if Fusion_Score >= 0.20 else ("SELL" if Fusion_Score <= -0.20 else "HOLD")
  ```
- **The Mechanism**:
  1. On 96% of days, News NLP is 0.0, and Market Regime is Calm (signal 0.0).
  2. The Classification signal is typically mild ($0.0$ to $0.10$, contributing $0.018$).
  3. Consequently, the RL signal (weight 0.30) dominates: when the RL agent switches from action 1 (HOLD, signal 0.0) to action 0 (SELL, signal -1.0) on a 1-day retracement, the score plummets:
     $$\text{Fusion Score} \approx 0.30 \times (-1.0) + 0.02 = -0.28 \le -0.20 \implies \mathbf{SELL}$$
  4. The system executes a **100% full liquidation** immediately upon a single down-tick.
- **Measured Telemetry**:
  - Out of 147 total round-trip trades across the portfolio, **77 trades (52.4%) were whipsaws lasting $\le 3$ days**!
  - **NVIDIA (`NVDA`)**: Experienced 33 round-trips with an **average holding duration of 5.03 days** and a **median holding duration of only 2.0 days**! 23 out of 33 round-trips (**69.7%**) were whipsaws lasting $\le 3$ days. In 2024's historic secular bull market (+150%+ gain for NVDA), the system panicked on 2-day minor pullbacks, liquidated completely, and sat in cash while NVDA gapped up at the open.
  - **Johnson & Johnson (`JNJ`)**: 32 round-trips, **median duration 2.0 days**, 22 whipsaws (**68.8%**).
  - **Amazon (`AMZN`)**: 25 round-trips, median duration 4.0 days, 10 whipsaws (**40.0%**).
  - **Meta (`META`)**: 14 round-trips, median duration 3.0 days, 8 whipsaws (**57.1%**).

### 2.3 Driver 3: Binary Full-Liquidation Sizing (Extreme Turnover)
- **The Mechanism**: Sizing is binary all-or-nothing ($0\% \leftrightarrow 10\%$). Every exit liquidates the entire position, paying turnover fees on 100% of the asset's equity. Every re-entry repurchases 10% of portfolio equity from cash, paying turnover fees again.
- **Measured Telemetry**:
  - Total trade events: **350 trades** in 300 sessions.
  - Portfolio turnover: **3,356%** (~33.6x portfolio equity).
  - At 50 bps execution friction (opening auction retail spread + slippage), net return collapses from **+28.48% (gross)** down to **+11.06% (net)**, destroying **17.42%** of portfolio equity.

---

## 3. Experimental Design & Methodology

To address these drivers without lookahead bias, we formulated and tested four experimental interventions:

1. **Experiment A (Current Baseline)**: Static 10% cap per stock, raw decision fusion signals (BUY $\ge +0.20$, SELL $\le -0.20$), Strict Dollar Ledger cash accounting.
2. **Experiment B1–B4 (Anti-Whipsaw Rules)**:
   - **Exp B1 (Cooldown $K=3$)**: Minimum holding period of 3 trading days before voluntary SELL execution is permitted.
   - **Exp B2 (Cooldown $K=5$)**: Minimum holding period of 5 trading days.
   - **Exp B3 (Hysteresis Exit Threshold)**: Asymmetric exit threshold—maintain BUY threshold at $\ge +0.20$, but require Fusion Score $\le -0.30$ (rather than $-0.20$) to trigger liquidation.
   - **Exp B4 (Combined Anti-Whipsaw)**: 3-day cooldown + mild hysteresis (SELL $\le -0.25$).
3. **Experiment C1 & C2 (Volatility-Scaled Sizing)**:
   - **Exp C1 (Inverse-Volatility Sizing, 10% Cap)**: Position size inversely proportional to 20-day historical realized volatility $\sigma_{20d}$, capped at 10.0% per asset:
     $$w_i \propto \frac{1}{\sigma_{i, 20d}}$$
   - **Exp C2 (Inverse-Volatility Sizing + Dynamic Fungible Cash, 20% Cap)**: Solves the cash trap by allowing active assets to absorb idle cash up to an expanded **20.0% cap per asset**, targeting 90% total portfolio exposure when active assets $< 10$.
4. **Experiment B+C (Integrated Optimal Strategy)**: Combines 3-day cooldown ($K=3$), inverse-volatility risk budgeting, and dynamic cash fungibility (20% cap).
5. **Experiment D (Passive Equal-Weight Benchmark)**: Equal-weight Buy & Hold (10% allocation per asset on day 0, 0% active turnover).

### Strict Temporal Separation (Task 6 Compliance)
To prevent lookahead bias in threshold selection, we partitioned the 300 test days chronologically:
- **Validation Split**: 150 trading days (October 20, 2023 to May 24, 2024). Used exclusively to evaluate parameter sensitivity ($K \in \{2, 3, 5, 7\}$, Hysteresis $\in \{-0.25, -0.30, -0.35\}$).
- **Untouched Final Test Split**: 150 trading days (May 28, 2024 to December 30, 2024). Evaluated with frozen parameters.
- **Full Period**: 300 trading days (October 20, 2023 to December 30, 2024).

---

## 4. Empirical Results & Findings

### 4.1 Root-Cause Diagnosis Breakdown (Active Stocks & Whipsaws)

```
Active Stocks Distribution Across 300 Trading Sessions:
  1 stock held:    1 day   ( 0.3%)
  2 stocks held:  10 days  ( 3.3%)  ##
  3 stocks held:  49 days  (16.3%)  ########
  4 stocks held:  82 days  (27.3%)  ##############
  5 stocks held:  99 days  (33.0%)  #################
  6 stocks held:  49 days  (16.3%)  ########
  7 stocks held:  10 days  ( 3.3%)  ##
  8-10 stocks:     0 days  ( 0.0%)
```

**Per-Stock Holding Duration & Whipsaw Telemetry**:

| Ticker | Company Name | BUY Days | SELL Days | HOLD Days | Round-Trips | Avg Duration | Median Duration | Whipsaws ($\le 3\text{d}$) | Whipsaw Ratio |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **AAPL** | Apple | 21 | 105 | 174 | 9 | 5.2 days | 3.0 days | 5 | 55.6% |
| **AMZN** | Amazon | 142 | 33 | 125 | 25 | 10.2 days | 4.0 days | 10 | 40.0% |
| **GOOGL** | Alphabet | 10 | 23 | 267 | 7 | 13.3 days | 6.0 days | 2 | 28.6% |
| **JNJ** | Johnson & Johnson | 65 | 153 | 82 | 32 | 2.6 days | **2.0 days** | **22** | **68.8%** |
| **JPM** | JPMorgan Chase | 39 | 97 | 164 | 16 | 5.2 days | 5.0 days | 7 | 43.8% |
| **META** | Meta Platforms | 18 | 39 | 243 | 14 | 5.3 days | 3.0 days | 8 | 57.1% |
| **MSFT** | Microsoft | 117 | 1 | 182 | 2 | 147.5 days | 147.5 days | 0 | 0.0% |
| **NVDA** | NVIDIA | 99 | 72 | 129 | 33 | 5.0 days | **2.0 days** | **23** | **69.7%** |
| **TSLA** | Tesla | 81 | 13 | 206 | 9 | 28.8 days | 25.0 days | 0 | 0.0% |
| **V** | Visa | 0 | 143 | 157 | 0 | 0.0 days | 0.0 days | 0 | 0.0% |
| **Total** | **Portfolio** | **592** | **679** | **1,729** | **147** | **8.1 days** | **3.0 days** | **77** | **52.4%** |

---

### 4.2 Out-of-Sample Generalization (Validation vs Untouched Test vs Full)

To verify that improvements were not the result of in-sample overfitting, we tracked performance across both independent 150-day windows:

| Experiment Configuration | Validation Return (150d) | Validation Sharpe | Validation Turnover | Test Return (150d) | Test Sharpe | Test Turnover | Full 300d Return | Full 300d Sharpe | Full 300d Turnover | Full Cash |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp A: Baseline Fusion** | +12.90% | 1.56 | 1365% | +10.52% | 1.21 | 1791% | **+24.79%** | **1.38** | 3356% | 52.9% |
| **Exp B1: Cooldown $K=3$** | +14.52% | 1.69 | 1136% | +12.40% | 1.44 | 1494% | **+28.13%** | **1.54** | 2843% | 50.0% |
| **Exp B2: Cooldown $K=5$** | +15.75% | 1.75 | 1036% | +10.61% | 1.14 | 1238% | **+27.50%** | **1.42** | 2489% | 46.2% |
| **Exp B3: Hysteresis (Sell $\le -0.30$)** | +18.23% | 1.96 | 1005% | +15.32% | 1.75 | 1378% | **+36.41%** | **1.86** | 2604% | 48.0% |
| **Exp B4: Combined ($K=3, \text{Sell}\le-0.25$)** | +18.36% | 2.03 | 1066% | +12.32% | 1.42 | 1394% | **+32.99%** | **1.73** | 2696% | 49.2% |
| **Exp C1: Vol-Scaled (10% Cap)** | +10.07% | 1.68 | 953% | +3.78% | 0.47 | 1340% | **+14.21%** | **1.09** | 2411% | 64.1% |
| **Exp C2: Vol-Scaled + Dynamic Cash (20%)**| +21.24% | 1.65 | 1983% | +12.81% | 1.04 | 3129% | **+36.76%** | **1.35** | 5702% | 21.2% |
| **Exp B+C: Cooldown $K=3$ + Dynamic (20%)** | +22.92% | 1.78 | 1639% | +16.05% | 1.33 | 2566% | **+42.35%** | **1.55** | 4798% | 18.1% |
| **Exp D: Passive Buy & Hold** | **+37.57%** | **2.81** | **100%** | **+29.26%** | **2.03** | **100%** | **+72.43%** | **2.18** | **100%** | **0.0%** |

**Out-of-Sample Validation Findings**:
1. **Cooldown $K=3$ Generalizes Robustly**: On both the validation split (+14.52% vs +12.90%) and the untouched test split (+12.40% vs +10.52%), Cooldown $K=3$ consistently outperformed baseline while cutting turnover by ~500 percentage points.
2. **Hysteresis Exit ($\le -0.30$) Outperforms Across All Splits**: On validation (+18.23% vs +12.90%) and untouched test (+15.32% vs +10.52%), hysteresis exit achieved the highest risk-adjusted Sharpe (1.86 full period) and reduced turnover by 752 percentage points.
3. **Volatility Scaling with Rigid 10% Cap (Exp C1) Degrades Return**: Because lower-volatility assets (JNJ, V) underperformed or had 0 buys while high-volatility assets (TSLA, NVDA) were scaled down, cash drag expanded to **64.1%**, dropping test return to just +3.78%.
4. **Dynamic Cash Allocation (Exp C2 & B+C) Resolves Cash Drag**: Slashes idle cash from 52.9% down to 18.1%, lifting full-period return from +24.79% to **+42.35%**.

---

### 4.3 Standardized Full 300-Day Experiment Comparison (10 bps Friction)

| Experiment | Cumulative Return | Annualized Return | Volatility | Sharpe ($R_f=3\%$) | Sortino | Max DD | Mean Cash | Turnover | Trades | Whipsaws | Beta ($\beta$) | Jensen's $\alpha$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp A: Baseline Fusion** | **+24.79%** | 20.45% | 11.8% | 1.38 | 1.35 | **-9.09%** | 52.9% | 3,356% | 350 | 77 | 0.50 | -5.99% |
| **Exp B1: Cooldown $K=3$** | **+28.13%** | 23.15% | 12.1% | 1.54 | 1.50 | **-9.12%** | 50.0% | 2,843% | 310 | 33 | 0.51 | -4.51% |
| **Exp B2: Cooldown $K=5$** | **+27.50%** | 22.64% | 12.9% | 1.42 | 1.36 | -10.35% | 46.2% | 2,489% | 286 | 0 | 0.55 | -6.52% |
| **Exp B3: Hysteresis Exit** | **+36.41%** | **29.80%** | 12.8% | **1.86** | **1.81** | **-8.44%** | 48.0% | **2,604%** | **284** | 32 | 0.55 | **-0.91%** |
| **Exp B4: Combined ($K=3, -0.25$)** | **+32.99%** | 27.06% | 12.5% | 1.73 | 1.72 | -8.90% | 49.2% | 2,696% | 295 | 26 | 0.54 | -2.34% |
| **Exp C1: Vol-Scaled (10% Cap)** | **+14.21%** | 11.81% | **7.7%** | 1.09 | 1.00 | **-6.41%** | 64.1% | 2,411% | 364 | 77 | 0.33 | -6.21% |
| **Exp C2: Vol-Scaled Dynamic (20%)** | **+36.76%** | 30.08% | 18.5% | 1.35 | 1.28 | -12.82% | 21.2% | 5,702% | 366 | 78 | 0.76 | -9.24% |
| **Exp B+C: Optimized (20% Cap)** | **+42.35%** | **34.53%** | 18.3% | 1.55 | 1.47 | -12.18% | **18.1%** | 4,798% | 334 | 30 | 0.76 | -5.65% |
| **Exp D: Buy & Hold Benchmark** | **+72.43%** | **58.03%** | 20.6% | **2.18** | **1.94** | -17.46% | 0.0% | **100%** | **10** | 0 | 1.00 | +0.00% |

---

### 4.4 Transaction Friction Sensitivity (0 to 50 bps)

We examined whether the experimental improvements survive realistic execution frictions (slippage, bid-ask spreads, and commissions):

| Experiment Configuration | 0 bps (Gross) Return | 0 bps Sharpe | 10 bps (Baseline) Return | 10 bps Sharpe | 20 bps (Retail) Return | 20 bps Sharpe | 50 bps (Adverse) Return | 50 bps Sharpe |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp A: Baseline Fusion** | +28.48% | 1.60 | +24.79% | 1.38 | +21.21% | 1.17 | +11.06% | 0.55 |
| **Exp B1: Cooldown $K=3$** | +31.29% | 1.71 | +28.13% | 1.54 | +25.04% | 1.37 | +16.20% | 0.85 |
| **Exp B3: Hysteresis Exit ($\le -0.30$)** | **+39.35%** | **2.01** | **+36.41%** | **1.86** | **+33.53%** | **1.72** | **+25.24%** | **1.29** |
| **Exp C1: Vol-Scaled (10% Cap)** | +16.73% | 1.33 | +14.21% | 1.09 | +11.75% | 0.85 | +4.67% | 0.15 |
| **Exp C2: Vol-Scaled Dynamic (20%)** | +43.10% | 1.56 | +36.76% | 1.35 | +30.70% | 1.14 | +14.16% | 0.53 |
| **Exp B+C: Optimized (20% Cap)** | **+47.78%** | 1.72 | **+42.35%** | 1.55 | **+37.11%** | 1.37 | **+22.62%** | 0.86 |
| **Exp D: Passive Buy & Hold** | **+72.56%** | **2.18** | **+72.43%** | **2.18** | **+72.29%** | **2.18** | **+71.90%** | **2.16** |

**Key Friction Sensitivity Takeaways**:
1. **Hysteresis Exit is the Most Friction-Resilient Active Strategy**: At 50 bps, while baseline return collapses to +11.06% (Sharpe 0.55), Experiment B3 preserves **+25.24% return** and an annualized **1.29 Sharpe**. By eliminating 45 false-positive liquidations, it saves ~14% in friction penalty.
2. **Buy & Hold Remains Dominant Under High Friction**: Because Buy & Hold incurs only 10 initial trades (100% turnover), its return is virtually unaffected by friction (+72.56% at 0 bps vs +71.90% at 50 bps).

---

## 5. Evidence-Based Assessment: Justified vs Inconclusive

### A. Strongly Justified by Empirical Evidence
1. **Asymmetric Hysteresis Exit Threshold (Sell $\le -0.30$)**:
   - *Justification*: Prevents 1-day noise whipsaws from triggering full liquidations. Improves full-period cumulative return from **+24.79% to +36.41%** (+11.62% gain), increases Sharpe from **1.38 to 1.86**, cuts MaxDD from **-9.09% to -8.44%**, and reduces turnover from **3,356% to 2,604%** (-752%). Crucially, it demonstrated consistent out-of-sample generalization (+18.23% on validation, +15.32% on untouched test).
2. **Minimum Holding Cooldown Period ($K=3$ days)**:
   - *Justification*: Halves whipsaws $\le 3$ days from 77 to 33, reducing turnover by 513 percentage points and lifting cumulative return from +24.79% to +28.13% across all splits.
3. **Dynamic Cash Fungibility (Relaxing 10% Cap to 20% When Active Stocks $< 10$)**:
   - *Justification*: Slashes dead cash from 52.9% to 18.1%, directly solving the cash trap and lifting cumulative return to **+42.35%**.

### B. Deficiencies Identified / Not Justified
1. **Symmetric Linear Thresholding (BUY $\ge +0.20$, SELL $\le -0.20$)**:
   - *Flaw*: Unstable across daily noise. High RL weight causes instantaneous flip from HOLD to full liquidation on a 1-day dip.
2. **Rigid 10% Static Sizing Cap Without Cash Fungibility**:
   - *Flaw*: Structurally locks >50% of portfolio equity in zero-yield cash whenever 5 or more assets have no BUY signal.
3. **Inverse-Volatility Sizing Under a Static 10% Cap (Exp C1)**:
   - *Flaw*: Sizing high-volatility winners down without allowing low-volatility assets to scale above 10% causes cash drag to surge to 64.1%, cutting return to +14.21%.

### C. Inconclusive Findings
1. **Opening Cross Auction Fill Slippage on Post-Earnings Gaps**:
   - *Status*: While next-day MOO fills eliminate lookahead, whether opening auction fills during high-beta gap days (e.g. NVDA, TSLA) incur 10 bps or 30+ bps of market impact cannot be verified without tick-level execution data.
2. **Optimal Dynamic Cap Size Under Prolonged Bear Markets**:
   - *Status*: Sizing up to 20% per asset yielded +42.35% during 2023–2024 (a strong bull regime), but increased MaxDD from -9.09% to -12.18%. Its downside protection in a sustained 2008- or 2022-style bear market remains untested.

---

## 6. Production Trading & System Readiness Recommendations

### Recommendation on Live Capital Trading: **STRICTLY NOT APPROVED (DO NOT TRADE)**
Deploying real money remains strictly unapproved for the following empirical reasons:
1. **Persistent Underperformance vs Passive Benchmark**: Even the best optimized active configuration (Exp B+C at +42.35%) still lags passive Buy & Hold (+72.43%) by **-30.08%** net return.
2. **Negative Risk-Adjusted Alpha**: Relative to market risk exposure ($\beta = 0.50$ to $0.76$), active management generated negative annual Jensen's Alpha (-0.91% to -5.99%).
3. **High Turnover Dependency**: 2,600% to 4,800% turnover makes active returns vulnerable to broker commissions and execution delays.

### Actionable Next Steps for System Engineering (Pre-Paper Trading):
1. **Adopt Hysteresis Exit & Cooldown in Strategy Config**: Update decision fusion logic in non-production configurations to require $\text{Fusion Score} \le -0.30$ for SELL liquidations and enforce a 3-day minimum holding period.
2. **Implement Limit-on-Open (LOO) Protection**: Add price collar rules to prevent fills on gap-openings exceeding $\pm 2.5\%$.
3. **Accumulate Live Streaming Telemetry**: Allow the active background Alpaca WebSocket news listener (`news_database.db`, currently holding 655+ articles) to accumulate 90–180 days of dense continuous financial news before re-evaluating the news signal component.
