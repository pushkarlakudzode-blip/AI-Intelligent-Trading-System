"""
Intelligent Financial News Explanation Engine.

Phase 3 Extension: Plain-English explanation generation for tracked company news.

Provides grounded, non-hallucinatory explanations analyzing:
1. What the article reports (factual synthesis).
2. Why it matters to the specific company (business model relevance).
3. Potential positive, negative, mixed, or uncertain implications.
4. Grounded textual evidence (quotes, tokens, event categories).
5. Explicit limitations and missing information (quantitative omissions, unverified claims).
6. Clear separation between news sentiment and trading decisions (sentiment != price prediction).

Architecture:
- BaseNewsExplanationProvider: Abstract provider interface.
- GroundedAnalyticalExplanationProvider: Deterministic, rule-grounded, zero-hallucination local engine.
- LLMNewsExplanationProvider: Extensible provider abstraction for LLMs with graceful offline fallback.
- NewsExplanationEngine: High-level facade with SQLite caching and persistence.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
import logging
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.nlp.news_nlp_pipeline import (
    clean_financial_text,
    compute_financial_sentiment,
    compute_event_importance,
    COMPANY_ENTITIES,
    FINANCIAL_POS_TERMS,
    FINANCIAL_NEG_TERMS
)

logger = logging.getLogger("news_explanation")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "live" / "news_database.db"

# Tracked portfolio companies
SUPPORTED_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "JNJ"
]

COMPANY_NAMES = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet",
    "AMZN": "Amazon",
    "NVDA": "NVIDIA",
    "META": "Meta Platforms",
    "TSLA": "Tesla",
    "JPM": "JPMorgan Chase",
    "V": "Visa",
    "JNJ": "Johnson & Johnson"
}

# Core business descriptions for grounded relevance synthesis
COMPANY_BUSINESS_PROFILES: Dict[str, Dict[str, str]] = {
    "AAPL": {
        "core_business": "consumer technology hardware (iPhone, Mac, iPad, Wearables) and high-margin digital services (App Store, iCloud, Apple Pay)",
        "revenue_drivers": "hardware replacement cycles, services monetization, and ecosystem customer retention",
        "primary_risks": "global consumer discretionary demand, supply chain concentration in Asia, and app store regulatory antitrust inquiries"
    },
    "MSFT": {
        "core_business": "enterprise cloud computing (Azure), enterprise productivity software (M365, Teams), and enterprise AI solutions (Copilot)",
        "revenue_drivers": "commercial cloud adoption, enterprise seat expansions, and software recurring subscriptions",
        "primary_risks": "IT budget contractions, cloud migration deceleration, and enterprise software competitive pressures"
    },
    "GOOGL": {
        "core_business": "digital advertising (Google Search, YouTube), enterprise cloud infrastructure (Google Cloud), and AI research",
        "revenue_drivers": "ad auction pricing, query volume, cloud contract expansions, and subscription services",
        "primary_risks": "ad budget cyclicality, regulatory antitrust scrutiny in search/ad-tech, and AI search transition costs"
    },
    "AMZN": {
        "core_business": "e-commerce retail marketplace, cloud infrastructure (AWS), logistics fulfillment, and advertising services",
        "revenue_drivers": "AWS cloud workload demand, retail consumer spending, third-party seller services, and digital ads",
        "primary_risks": "retail fulfillment labor/shipping costs, enterprise cloud spending slowdowns, and regulatory retail scrutiny"
    },
    "NVDA": {
        "core_business": "accelerated computing architecture, enterprise data center AI GPUs (Blackwell, Hopper), and the CUDA software stack",
        "revenue_drivers": "hyperscaler AI capital expenditure, data center accelerator shipments, and enterprise compute demand",
        "primary_risks": "customer concentration among top cloud providers, export restriction controls, and semiconductor cyclicality"
    },
    "META": {
        "core_business": "digital advertising across family of apps (Facebook, Instagram, WhatsApp) and artificial intelligence infrastructure",
        "revenue_drivers": "daily active user engagement, ad impression volume, ad pricing efficiency, and AI recommendation delivery",
        "primary_risks": "digital ad market softening, user privacy/platform policy changes, and Reality Labs capital expenditure drag"
    },
    "TSLA": {
        "core_business": "electric vehicle manufacturing (Model 3/Y, Cybertruck), autonomous software (FSD), and utility energy storage (Megapack)",
        "revenue_drivers": "automotive vehicle deliveries, average selling prices, automotive gross margins, and energy storage deployments",
        "primary_risks": "EV price competition, margin compression, regulatory scrutiny of driver assistance systems, and factory execution"
    },
    "JPM": {
        "core_business": "diversified global banking services: consumer banking, commercial lending, investment banking advisory, and asset management",
        "revenue_drivers": "net interest income (NII), credit card fees, capital markets underwriting, and advisory advisory fees",
        "primary_risks": "interest rate fluctuations, credit defaults/loan loss provisions, regulatory capital requirements, and macro slowdowns"
    },
    "V": {
        "core_business": "global retail payments technology network facilitating electronic funds transfers between consumers, merchants, and banks",
        "revenue_drivers": "payment volume, transaction processing counts, cross-border remittance fees, and value-added payment services",
        "primary_risks": "cross-border travel/spending slowdowns, merchant litigation over interchange fees, and regulatory routing mandates"
    },
    "JNJ": {
        "core_business": "healthcare innovation spanning innovative medicines (oncology, immunology, neuroscience) and MedTech surgical devices",
        "revenue_drivers": "prescription pharmaceutical sales volume, medical device hospital adoption, and global patent exclusivity",
        "primary_risks": "patent expiration (loss of exclusivity), generic/biosimilar erosion, clinical trial setbacks, and product liability litigation"
    }
}

STANDARD_DIRECTIONAL_DISCLAIMER = (
    "DISCLAIMER: Financial news sentiment and qualitative explanations reflect reported media developments "
    "and do NOT predict stock price direction. Market prices may already discount reported information, and "
    "market sentiment frequently diverges from fundamental price action. A negative headline does not constitute "
    "a SELL signal, nor does a positive headline constitute a BUY signal."
)


@dataclass
class NewsExplanation:
    """Standardized plain-English explanation record for a financial news article."""
    news_id: str
    ticker: str
    company_name: str
    what_it_reports: str
    why_it_matters: str
    implications: str
    implication_direction: str  # "POSITIVE", "NEGATIVE", "MIXED", "UNCERTAIN"
    evidence: str
    evidence_strength: str      # "DIRECT", "PARTIAL", "INSUFFICIENT"
    limitations: str
    directional_disclaimer: str
    provider_used: str
    created_at_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BaseNewsExplanationProvider(ABC):
    """Abstract base class for financial news explanation generators."""

    @abstractmethod
    def generate_explanation(
        self,
        article: Dict[str, Any],
        nlp_result: Optional[Dict[str, Any]] = None
    ) -> NewsExplanation:
        """Generate a grounded plain-English explanation for a single article."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider dependencies/credentials are ready."""
        pass


class GroundedAnalyticalExplanationProvider(BaseNewsExplanationProvider):
    """
    Deterministic, rule-grounded, zero-hallucination explanation provider.

    Strict Constraints:
    - Never invents facts, financial figures, relationships, or causal claims.
    - Strictly extracts terms, numbers, entities, and event scopes present in the article.
    - Explicitly flags insufficient company-specific evidence when present.
    - Separates news sentiment from stock price forecasts or trading recommendations.
    """

    def is_available(self) -> bool:
        return True

    def generate_explanation(
        self,
        article: Dict[str, Any],
        nlp_result: Optional[Dict[str, Any]] = None
    ) -> NewsExplanation:
        news_id = str(article.get("News_ID") or article.get("news_id") or "")
        ticker = str(article.get("Ticker") or article.get("ticker") or "").strip().upper()
        headline = str(article.get("Headline") or article.get("headline") or "").strip()
        summary = str(article.get("Summary") or article.get("summary") or "").strip()
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        # 1. Validate Tracked Ticker
        if ticker not in SUPPORTED_TICKERS:
            return NewsExplanation(
                news_id=news_id,
                ticker=ticker,
                company_name=COMPANY_NAMES.get(ticker, ticker),
                what_it_reports=f"Article references an untracked ticker symbol: {ticker}.",
                why_it_matters=f"This portfolio system tracks only 10 specific equities ({', '.join(SUPPORTED_TICKERS)}).",
                implications="No company-specific portfolio implications can be derived for untracked equities.",
                implication_direction="UNCERTAIN",
                evidence="Unsupported ticker token.",
                evidence_strength="INSUFFICIENT",
                limitations="Untracked asset outside system monitoring scope.",
                directional_disclaimer=STANDARD_DIRECTIONAL_DISCLAIMER,
                provider_used="GroundedAnalyticalEngine",
                created_at_utc=now_utc
            )

        company_name = COMPANY_NAMES.get(ticker, ticker)
        profile = COMPANY_BUSINESS_PROFILES.get(ticker, {
            "core_business": "commercial operations",
            "revenue_drivers": "product sales and services",
            "primary_risks": "market and operational uncertainty"
        })

        clean_h = clean_financial_text(headline)
        clean_s = clean_financial_text(summary)
        full_text = f"{clean_h} {clean_s}".strip()

        # 2. Check for Insufficient / Empty Text
        if not clean_h or len(clean_h) < 5 or (len(full_text) < 15 and not clean_s):
            return NewsExplanation(
                news_id=news_id,
                ticker=ticker,
                company_name=company_name,
                what_it_reports="The article record contains an empty, missing, or severely truncated text snippet.",
                why_it_matters=f"Without substantive reporting details, no reliable assessment can be made for {company_name}.",
                implications="Insufficient information to determine whether this item carries business implications.",
                implication_direction="UNCERTAIN",
                evidence="No substantive textual content available.",
                evidence_strength="INSUFFICIENT",
                limitations="Severe data deficit: Headline or summary lacks descriptive sentences or factual assertions.",
                directional_disclaimer=STANDARD_DIRECTIONAL_DISCLAIMER,
                provider_used="GroundedAnalyticalEngine",
                created_at_utc=now_utc
            )

        # 3. Analyze Entity Match & Evidence Strength
        entities = COMPANY_ENTITIES.get(ticker, [ticker.lower()])
        headline_lower = clean_h.lower()
        summary_lower = clean_s.lower()
        text_lower = full_text.lower()

        direct_headline_match = (ticker.lower() in headline_lower) or any(e in headline_lower for e in entities)
        summary_match = (ticker.lower() in summary_lower) or any(e in summary_lower for e in entities)

        # Extract explicit financial figures, percentages, and currencies if present in text
        figures_found = re.findall(r"\$\d+(?:\.\d+)?(?:\s*(?:billion|million|B|M|k|T))?|\d+(?:\.\d+)?%", full_text, re.IGNORECASE)
        quarters_found = re.findall(r"\b(?:Q[1-4]|FY\d{2,4}|fiscal\s*(?:year|\d{2,4})?)\b", full_text, re.IGNORECASE)

        # 4. Extract Event Category & Sentiment
        if nlp_result:
            event_type = nlp_result.get("event_type") or nlp_result.get("Event_Type") or "General News"
            sent_label = nlp_result.get("sentiment_label") or nlp_result.get("Sentiment_Label") or "Neutral"
            sent_score = float(nlp_result.get("sentiment_score", nlp_result.get("Sentiment_Score", 0.0)))
        else:
            evt_info = compute_event_importance(clean_h, clean_s)
            event_type = evt_info[0]
            sent_info = compute_financial_sentiment(full_text)
            sent_label = sent_info[0]
            sent_score = sent_info[1]

        # Extract matched positive and negative terms from text
        tokens = re.findall(r"\b[a-zA-Z]{2,}\b", text_lower)
        pos_terms_matched = sorted(list({t for t in tokens if t in FINANCIAL_POS_TERMS}))
        neg_terms_matched = sorted(list({t for t in tokens if t in FINANCIAL_NEG_TERMS}))

        # 5. Determine Evidence Strength
        if direct_headline_match:
            evidence_strength = "DIRECT"
        elif summary_match:
            evidence_strength = "PARTIAL"
        else:
            evidence_strength = "INSUFFICIENT"

        # 6. Synthesize "What the article reports"
        if evidence_strength == "INSUFFICIENT":
            what_it_reports = (
                f"The report discusses broader market developments: \"{headline}\". "
                f"However, neither {company_name} nor its primary branded assets are explicitly cited in the core reporting."
            )
        else:
            figures_str = f" citing key figures ({', '.join(figures_found[:3])})" if figures_found else ""
            what_it_reports = (
                f"The article reports on {event_type.lower()}: \"{headline}\"{figures_str}."
            )
            if clean_s and len(clean_s) > 20:
                first_sentence = clean_s.split(".")[0].strip()
                if first_sentence and len(first_sentence) < 160:
                    what_it_reports += f" Specifically, it highlights that {first_sentence}."

        # 7. Synthesize "Why it may matter to the specific company"
        if evidence_strength == "INSUFFICIENT":
            why_it_matters = (
                f"INSUFFICIENT COMPANY-SPECIFIC EVIDENCE: While {company_name} ({ticker}) is part of this monitored portfolio, "
                f"the text does not disclose firm-specific operational metrics, product developments, or financial guidance. "
                f"Relevance is limited to general sector contagion or macro sentiment."
            )
        elif event_type == "Earnings / Financials":
            why_it_matters = (
                f"Operating in {profile['core_business']}, earnings, revenue, and capital distribution developments "
                f"directly influence {company_name}'s valuation fundamentals, where {profile['revenue_drivers']} "
                f"determine cash flow resilience."
            )
        elif event_type == "Product / Innovation":
            why_it_matters = (
                f"Product developments and technological updates impact {company_name}'s competitive moat in "
                f"{profile['core_business']}, where adoption velocity drives multi-quarter growth."
            )
        elif event_type == "Regulatory / Legal":
            why_it_matters = (
                f"Operating in {profile['core_business']}, legal actions and regulatory scrutiny intersect with "
                f"{profile['primary_risks']}, potentially impacting compliance costs, operational flexibility, or market structure."
            )
        elif event_type == "Analyst Action":
            why_it_matters = (
                f"Brokerage ratings and price target revisions reflect institutional sentiment regarding {company_name}'s "
                f"operating trajectory, though they represent third-party opinions rather than corporate disclosures."
            )
        elif event_type == "Corporate Governance":
            why_it_matters = (
                f"Leadership decisions, capital allocation, and governance matters affect the strategic direction "
                f"and operational execution of {company_name}."
            )
        else:
            why_it_matters = (
                f"Macroeconomic trends and sector-wide sentiment can indirectly influence {company_name}'s "
                f"valuation multiples and broader equity market demand."
            )

        # 8. Synthesize "Implications" and "Implication Direction"
        if evidence_strength == "INSUFFICIENT":
            implications = (
                f"Direct business implications for {company_name} are uncertain because the article lacks company-level data. "
                f"Any price movement would likely stem from broad market correlation rather than company-specific catalysts."
            )
            implication_direction = "UNCERTAIN"
        elif len(pos_terms_matched) > 0 and len(neg_terms_matched) > 0:
            pos_joined = ", ".join(pos_terms_matched[:3])
            neg_joined = ", ".join(neg_terms_matched[:3])
            implications = (
                f"MIXED IMPLICATIONS: The reporting contains both constructive elements (e.g. {pos_joined}) "
                f"and counterbalancing headwinds or cautionary markers (e.g. {neg_joined}). "
                f"Net operational impact depends on which factor dominates in subsequent quarterly disclosures."
            )
            implication_direction = "MIXED"
        elif sent_score > 0.05 or (len(pos_terms_matched) > len(neg_terms_matched) and sent_score >= 0.0):
            pos_joined = ", ".join(pos_terms_matched[:4]) if pos_terms_matched else "constructive tone"
            implications = (
                f"POTENTIAL POSITIVE IMPLICATIONS: Supported by favorable terminology ({pos_joined}), the reported development "
                f"points to potential operational momentum or expansion. "
                f"However, actual realization remains subject to operational execution and competitive response."
            )
            implication_direction = "POSITIVE"
        elif sent_score < -0.05 or (len(neg_terms_matched) > len(pos_terms_matched) and sent_score <= 0.0):
            neg_joined = ", ".join(neg_terms_matched[:4]) if neg_terms_matched else "cautious tone"
            implications = (
                f"POTENTIAL NEGATIVE IMPLICATIONS: Supported by risk indicators ({neg_joined}), this development highlights "
                f"potential operational friction, cost pressure, or regulatory headwind. "
                f"However, existing corporate reserves and enterprise market share may mitigate long-term damage."
            )
            implication_direction = "NEGATIVE"
        else:
            implications = (
                f"UNCERTAIN / NEUTRAL IMPLICATIONS: The reporting conveys factual or procedural developments without "
                f"definitive positive or negative momentum. Material business impact will depend on subsequent formal filings."
            )
            implication_direction = "UNCERTAIN"

        # 9. Synthesize "Evidence Supporting the Explanation"
        evidence_items = []
        evidence_items.append(f"Headline text: \"{headline}\"")
        if direct_headline_match:
            evidence_items.append(f"Core entity match: Identified {company_name} ({ticker}) in headline.")
        elif summary_match:
            evidence_items.append(f"Secondary entity match: Identified {company_name} in body text.")
        else:
            evidence_items.append(f"No direct entity match found for {company_name} in text.")

        if figures_found:
            evidence_items.append(f"Reported financial metrics: {', '.join(figures_found[:4])}")
        if quarters_found:
            evidence_items.append(f"Reported periods: {', '.join(quarters_found[:3])}")
        if pos_terms_matched:
            evidence_items.append(f"Positive lexicon terms: {', '.join(pos_terms_matched[:4])}")
        if neg_terms_matched:
            evidence_items.append(f"Negative lexicon terms: {', '.join(neg_terms_matched[:4])}")

        evidence_items.append(f"Loughran-McDonald sentiment classification: {sent_label} (Score: {sent_score:+.4f})")
        evidence_items.append(f"Assigned event taxonomy: {event_type}")

        evidence = " | ".join(evidence_items)

        # 10. Synthesize "Important Limitations & Missing Information"
        limitations_items = []
        if not clean_s or len(clean_s) < 30:
            limitations_items.append("Text is restricted to a headline snippet without full article body.")
        if not figures_found:
            limitations_items.append("No audited financial metrics, dollar revenue impacts, or margin figures are stated.")
        if evidence_strength != "DIRECT":
            limitations_items.append(f"Reporting does not focus exclusively on {company_name}; conclusions are partially inferred from peer or sector context.")
        limitations_items.append("Timing and duration of business impact are unquantified.")
        limitations_items.append("Third-party news reporting has not been cross-verified against official SEC 10-K/10-Q regulatory filings.")

        limitations = " • ".join(limitations_items)

        return NewsExplanation(
            news_id=news_id,
            ticker=ticker,
            company_name=company_name,
            what_it_reports=what_it_reports,
            why_it_matters=why_it_matters,
            implications=implications,
            implication_direction=implication_direction,
            evidence=evidence,
            evidence_strength=evidence_strength,
            limitations=limitations,
            directional_disclaimer=STANDARD_DIRECTIONAL_DISCLAIMER,
            provider_used="GroundedAnalyticalEngine",
            created_at_utc=now_utc
        )


class LLMNewsExplanationProvider(BaseNewsExplanationProvider):
    """
    Extensible LLM provider abstraction.

    Secure Design:
    - Never hardcodes or logs API keys.
    - Does NOT transmit private project files or data without explicit user configuration.
    - If no key is configured or remote call fails, gracefully falls back to GroundedAnalyticalExplanationProvider.
    """

    def __init__(self, fallback_provider: Optional[BaseNewsExplanationProvider] = None):
        self.fallback_provider = fallback_provider or GroundedAnalyticalExplanationProvider()
        self.api_key = os.getenv("LLM_EXPLANATION_API_KEY") or os.getenv("GEMINI_API_KEY")

    def is_available(self) -> bool:
        return bool(self.api_key and len(self.api_key.strip()) > 10)

    def generate_explanation(
        self,
        article: Dict[str, Any],
        nlp_result: Optional[Dict[str, Any]] = None
    ) -> NewsExplanation:
        if not self.is_available():
            logger.debug("LLM provider not configured. Falling back to GroundedAnalyticalEngine.")
            return self.fallback_provider.generate_explanation(article, nlp_result)

        try:
            # Here an external LLM SDK could be invoked with strict schema prompts.
            # In absence of authorized external network call or valid model endpoint,
            # execute safe grounded fallback to ensure deterministic reliability.
            raise NotImplementedError("LLM endpoint configured but external network access deferred to offline fallback.")
        except Exception as e:
            logger.info("LLM provider unavailable or deferred (%s). Executing safe grounded fallback.", e)
            return self.fallback_provider.generate_explanation(article, nlp_result)


class NewsExplanationEngine:
    """
    Primary Facade for News Explanation.

    Coordinates providers, SQLite caching, and batch persistence.
    """

    def __init__(
        self,
        primary_provider: Optional[BaseNewsExplanationProvider] = None,
        db_path: Optional[Path] = None
    ):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.provider = primary_provider or GroundedAnalyticalExplanationProvider()

    def get_or_create_explanation(
        self,
        article: Dict[str, Any],
        nlp_result: Optional[Dict[str, Any]] = None,
        force_refresh: bool = False
    ) -> NewsExplanation:
        """
        Retrieve existing explanation from SQLite storage, or generate and store it on-demand.
        """
        news_id = str(article.get("News_ID") or article.get("news_id") or "").strip()
        ticker = str(article.get("Ticker") or article.get("ticker") or "").strip().upper()

        if not force_refresh and news_id and self.db_path.exists():
            cached = self._load_explanation_from_db(news_id)
            if cached:
                return cached

        # Generate fresh explanation using active provider
        explanation = self.provider.generate_explanation(article, nlp_result)

        # Persist to database if news_id exists
        if news_id and self.db_path.exists():
            self._save_explanation_to_db(explanation)

        return explanation

    def explain_articles(
        self,
        articles_df: pd.DataFrame,
        nlp_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate explanations for a batch of articles in a DataFrame.
        """
        if articles_df.empty:
            return pd.DataFrame()

        nlp_map: Dict[str, Dict[str, Any]] = {}
        if nlp_df is not None and not nlp_df.empty:
            for _, r in nlp_df.iterrows():
                nid = str(r.get("News_ID") or r.get("news_id") or "")
                if nid:
                    nlp_map[nid] = r.to_dict()

        records = []
        for _, art in articles_df.iterrows():
            art_dict = art.to_dict()
            nid = str(art_dict.get("News_ID") or art_dict.get("news_id") or "")
            nlp_dict = nlp_map.get(nid)
            expl = self.get_or_create_explanation(art_dict, nlp_dict)
            records.append(expl.to_dict())

        return pd.DataFrame(records)

    def _load_explanation_from_db(self, news_id: str) -> Optional[NewsExplanation]:
        """Read explanation from SQLite news_explanations table."""
        try:
            conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
            cur = conn.cursor()
            cur.execute("""
                SELECT news_id, ticker, company_name, what_it_reports, why_it_matters,
                       implications, implication_direction, evidence, evidence_strength,
                       limitations, directional_disclaimer, provider_used, created_at_utc
                FROM news_explanations
                WHERE news_id = ?
            """, (news_id,))
            row = cur.fetchone()
            conn.close()

            if row:
                return NewsExplanation(
                    news_id=row[0],
                    ticker=row[1],
                    company_name=row[2],
                    what_it_reports=row[3],
                    why_it_matters=row[4],
                    implications=row[5],
                    implication_direction=row[6],
                    evidence=row[7],
                    evidence_strength=row[8],
                    limitations=row[9],
                    directional_disclaimer=row[10],
                    provider_used=row[11],
                    created_at_utc=row[12]
                )
        except Exception as e:
            logger.debug("Could not read explanation from DB: %s", e)
        return None

    def _save_explanation_to_db(self, exp: NewsExplanation) -> None:
        """Insert or replace explanation in SQLite news_explanations table."""
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute("""
            CREATE TABLE IF NOT EXISTS news_explanations (
                news_id TEXT PRIMARY KEY,
                ticker TEXT NOT NULL,
                company_name TEXT,
                what_it_reports TEXT,
                why_it_matters TEXT,
                implications TEXT,
                implication_direction TEXT,
                evidence TEXT,
                evidence_strength TEXT,
                limitations TEXT,
                directional_disclaimer TEXT,
                provider_used TEXT,
                created_at_utc TEXT,
                FOREIGN KEY (news_id) REFERENCES articles(news_id)
            )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_explanations_ticker ON news_explanations(ticker)")

            cur.execute("""
            INSERT OR REPLACE INTO news_explanations (
                news_id, ticker, company_name, what_it_reports, why_it_matters,
                implications, implication_direction, evidence, evidence_strength,
                limitations, directional_disclaimer, provider_used, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                exp.news_id, exp.ticker, exp.company_name, exp.what_it_reports, exp.why_it_matters,
                exp.implications, exp.implication_direction, exp.evidence, exp.evidence_strength,
                exp.limitations, exp.directional_disclaimer, exp.provider_used, exp.created_at_utc
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug("Failed to store explanation in DB: %s", e)
