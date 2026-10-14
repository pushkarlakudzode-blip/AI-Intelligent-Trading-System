# AI-Based Intelligent Trading System

An integrated end-to-end algorithmic trading and decision-support system leveraging machine learning, quantitative finance, natural language processing, and reinforcement learning.

---

## 📌 Project Overview

The **AI-Based Intelligent Trading System** brings together diverse analytical and machine learning paradigms into a single unified architecture:
- **Data Ingestion & Engineering**: Multi-asset historical price data acquisition and automated cleaning.
- **Feature Engineering**: Technical indicators, momentum metrics, volatility estimators, and macroeconomic factors.
- **Predictive Modeling**: Supervised regression, regularization techniques, and directional trend classification.
- **Market Regime Detection**: Unsupervised clustering identifying distinct market volatility and trend regimes.
- **Sentiment Analysis**: NLP pipeline processing financial news and market sentiment signals.
- **Execution & Policy Learning**: Reinforcement learning trading agent for automated decision-making and portfolio allocation.
- **Integrated Decision Engine**: Combining predictive signals, risk management guardrails, and sentiment scoring into trade recommendations.
- **Interactive Dashboard**: Streamlit-based monitoring and visualization interface.

---

## 📁 Project Structure

```text
AI-Intelligent-Trading-System/
├── data/
│   ├── raw/                  # Unprocessed raw market feeds and news datasets
│   └── processed/            # Cleaned, aligned, and feature-engineered datasets
├── notebooks/                # Assignment experiments & research prototyping
│   ├── 01_data_eda.ipynb                 # Exploratory data analysis & baseline stats
│   ├── 02_regression.ipynb               # Linear & multi-variable regression models
│   ├── 03_regularization.ipynb           # Ridge, Lasso, and ElasticNet regularization
│   ├── 04_classification.ipynb           # Directional / trend classification models
│   ├── 05_clustering.ipynb               # Regime detection & unsupervised clustering
│   ├── 06_reinforcement_learning.ipynb   # RL agent policy training and evaluation
│   └── 07_nlp_news.ipynb                 # Financial news sentiment analysis (NLP)
├── src/                      # Modular production-grade codebase
│   ├── data/                 # Ingestion, downloading, and dataset loading utilities
│   ├── features/             # Feature extractors, technical indicators & transformations
│   ├── models/               # Model training, inference, evaluation, and registry
│   ├── nlp/                  # News scrapers, tokenizers, and sentiment scorers
│   └── decision/             # Signal fusion, portfolio allocation & risk management
├── dashboard/                # Interactive Streamlit dashboard application
├── tests/                    # Unit tests and validation suite
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── .gitignore                # Git ignore rules
```

---

## 🛠️ Setup & Installation

1. **Clone or navigate to the repository**:
   ```bash
   cd AI-Intelligent-Trading-System
   ```

2. **Activate the virtual environment**:
   - Windows (PowerShell):
     ```powershell
     .venv\Scripts\Activate.ps1
     ```
   - Linux / macOS:
     ```bash
     source .venv/bin/activate
     ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## ⚙️ Development Guidelines

- **Notebooks (`notebooks/`)**: Reserved for experiments, visualization, hypothesis testing, and assignment demonstrations.
- **Source Code (`src/`)**: Clean, reusable, tested production logic for pipelines, feature calculations, model definitions, and trade decision rules.
