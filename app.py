from __future__ import annotations

import html
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_credit_analyst.edgar import fetch_latest_10k, fetch_recent_10ks  # noqa: E402
from ai_credit_analyst.extraction import (  # noqa: E402
    DEFAULT_OPENAI_MODEL,
    FIELD_LABELS,
    RATIO_LABELS,
    extract_financials,
    format_financial_value,
    format_ratio_value,
)
from ai_credit_analyst.exceptions import (  # noqa: E402
    ConfigurationError,
    EdgarRateLimitError,
    EdgarResponseError,
    ExtractionParseError,
    FilingNotFoundError,
    FinancialExtractionError,
    MemoGenerationError,
    MemoParseError,
    OpenAIConfigurationError,
    OpenAIExtractionError,
    PDFGenerationError,
    TickerNotFoundError,
)
from ai_credit_analyst.memo import (  # noqa: E402
    build_credit_memo_pdf,
    build_peer_comparison_pdf,
    generate_credit_memo,
)
from ai_credit_analyst.quantitative import (  # noqa: E402
    PRIOR_YEAR_EXTRACTION_MODEL,
    build_quantitative_model_result,
)

DEFAULT_PREVIEW_CHARS = 12000
MEMO_PREPARATION_COOLDOWN_SECONDS = 5
MAX_COMPARISON_TICKERS = 3
QUANTITATIVE_INTER_MODEL_DELAY_SECONDS = 3
PRIMARY_METRIC_FIELDS = (
    "revenue",
    "net_income",
    "total_assets",
    "total_liabilities",
    "total_debt",
    "cash_and_cash_equivalents",
    "ebitda",
    "operating_cash_flow",
)
PRIMARY_RATIO_FIELDS = (
    "debt_to_equity",
    "interest_coverage",
    "current_ratio",
    "net_debt_to_ebitda",
)


@st.cache_data(ttl=3600, show_spinner=False)
def load_latest_10k(
    ticker: str,
    requester_name: str,
    requester_email: str,
) -> tuple[Any, str]:
    return fetch_latest_10k(
        ticker=ticker,
        requester_name=requester_name,
        requester_email=requester_email,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def load_recent_10ks(
    ticker: str,
    requester_name: str,
    requester_email: str,
    *,
    limit: int = 2,
) -> tuple[tuple[Any, str], ...]:
    return fetch_recent_10ks(
        ticker=ticker,
        requester_name=requester_name,
        requester_email=requester_email,
        limit=limit,
    )


def initialize_session_state() -> None:
    st.session_state.setdefault("filing_metadata", None)
    st.session_state.setdefault("filing_text", None)
    st.session_state.setdefault("financial_result", None)
    st.session_state.setdefault("memo_result", None)
    st.session_state.setdefault("quantitative_result", None)
    st.session_state.setdefault("prior_filing_metadata", None)
    st.session_state.setdefault("prior_financial_result", None)
    st.session_state.setdefault("last_ticker", "")
    st.session_state.setdefault("analysis_cache", {})
    st.session_state.setdefault("active_mode", "")
    st.session_state.setdefault("active_tickers", ())
    st.session_state.setdefault("active_results", {})
    st.session_state.setdefault("active_failures", {})


def apply_report_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --app-bg: #0e1117;
            --panel-bg: #121925;
            --panel-bg-elevated: #161f2d;
            --panel-border: #243245;
            --panel-border-soft: #1b2635;
            --text-primary: #edf2f7;
            --text-secondary: #97a4b5;
            --accent: #4a90d9;
            --accent-soft: rgba(74, 144, 217, 0.14);
            --success: #3ca978;
            --success-soft: rgba(60, 169, 120, 0.14);
            --warning: #d39a45;
            --warning-soft: rgba(211, 154, 69, 0.16);
            --danger: #d65f5f;
            --danger-soft: rgba(214, 95, 95, 0.16);
            --stable: #7e8898;
            --stable-soft: rgba(126, 136, 152, 0.18);
        }
        .stApp {
            background:
                radial-gradient(circle at top right, rgba(74, 144, 217, 0.08), transparent 28%),
                radial-gradient(circle at top left, rgba(60, 169, 120, 0.05), transparent 24%),
                linear-gradient(180deg, #0e1117 0%, #0b1018 100%);
            color: var(--text-primary);
        }
        .block-container {
            max-width: 1460px;
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        h1, h2, h3, h4, h5, h6, p, span, div, label {
            color: var(--text-primary);
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b1018 0%, #0d1420 100%);
            border-right: 1px solid var(--panel-border-soft);
        }
        [data-testid="stSidebar"] .block-container {
            padding-top: 1.25rem;
        }
        .workspace-shell,
        .report-shell,
        .section-shell,
        .callout-shell,
        .footer-shell {
            background: linear-gradient(180deg, rgba(22, 31, 45, 0.96) 0%, rgba(15, 22, 33, 0.96) 100%);
            border: 1px solid var(--panel-border);
            border-radius: 18px;
            box-shadow: 0 16px 40px rgba(5, 10, 18, 0.32);
        }
        .workspace-shell {
            padding: 1.3rem 1.35rem 1.15rem 1.35rem;
            margin-bottom: 1.1rem;
        }
        .report-shell {
            padding: 1.5rem 1.6rem;
            margin-bottom: 1rem;
        }
        .section-shell,
        .callout-shell {
            padding: 1.15rem 1.2rem;
            margin-bottom: 1rem;
        }
        .footer-shell {
            margin-top: 1.5rem;
            padding: 0.9rem 1.1rem;
            text-align: center;
            color: var(--text-secondary);
            font-size: 0.84rem;
        }
        .workspace-kicker,
        .section-kicker {
            color: var(--accent);
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.16em;
            text-transform: uppercase;
        }
        .workspace-title {
            font-size: 2.15rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            margin: 0.35rem 0 0.35rem 0;
        }
        .workspace-copy,
        .section-copy,
        .muted-copy {
            color: var(--text-secondary);
            font-size: 0.96rem;
            line-height: 1.55;
        }
        .report-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 1.5rem;
        }
        .report-company {
            font-size: 2.55rem;
            font-weight: 700;
            line-height: 1.05;
            letter-spacing: -0.03em;
            margin: 0.3rem 0 0.45rem 0;
        }
        .report-ticker {
            color: var(--text-secondary);
            font-size: 1.25rem;
            font-weight: 600;
            margin-left: 0.45rem;
        }
        .report-meta {
            color: var(--text-secondary);
            font-size: 0.94rem;
            margin-top: 0.25rem;
        }
        .header-badges {
            display: flex;
            gap: 0.55rem;
            flex-wrap: wrap;
            justify-content: flex-end;
        }
        .badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 0.48rem 0.85rem;
            border-radius: 999px;
            border: 1px solid transparent;
            font-size: 0.9rem;
            font-weight: 700;
            letter-spacing: 0.01em;
            white-space: nowrap;
        }
        .badge-rating-strong {
            background: var(--success-soft);
            border-color: rgba(60, 169, 120, 0.38);
            color: #7de0ad;
        }
        .badge-rating-investment {
            background: var(--accent-soft);
            border-color: rgba(74, 144, 217, 0.4);
            color: #8ec1f3;
        }
        .badge-rating-speculative {
            background: var(--warning-soft);
            border-color: rgba(211, 154, 69, 0.42);
            color: #f3be73;
        }
        .badge-rating-distressed {
            background: var(--danger-soft);
            border-color: rgba(214, 95, 95, 0.42);
            color: #ff9999;
        }
        .badge-outlook-positive {
            background: var(--success-soft);
            border-color: rgba(60, 169, 120, 0.32);
            color: #7de0ad;
        }
        .badge-outlook-stable {
            background: var(--stable-soft);
            border-color: rgba(126, 136, 152, 0.34);
            color: #c3cad3;
        }
        .badge-outlook-negative {
            background: var(--danger-soft);
            border-color: rgba(214, 95, 95, 0.36);
            color: #ff9999;
        }
        .section-title {
            font-size: 1.15rem;
            font-weight: 700;
            margin: 0.3rem 0 0.8rem 0;
        }
        .justification-box {
            border-left: 4px solid var(--accent);
            background: rgba(74, 144, 217, 0.08);
            border-radius: 14px;
            padding: 1rem 1rem 1rem 1.1rem;
        }
        .outlook-box {
            border-left: 4px solid var(--stable);
            background: rgba(126, 136, 152, 0.1);
            border-radius: 14px;
            padding: 1rem 1rem 1rem 1.1rem;
        }
        .outlook-box.positive {
            border-left-color: var(--success);
            background: rgba(60, 169, 120, 0.1);
        }
        .outlook-box.negative {
            border-left-color: var(--danger);
            background: rgba(214, 95, 95, 0.1);
        }
        .analysis-status {
            background: linear-gradient(180deg, rgba(19, 26, 37, 0.96) 0%, rgba(14, 20, 30, 0.96) 100%);
            border: 1px solid var(--panel-border);
            border-radius: 14px;
            padding: 0.95rem 1rem;
            margin: 0.55rem 0 0.9rem 0;
        }
        .analysis-status-title {
            color: var(--text-primary);
            font-size: 0.97rem;
            font-weight: 700;
            margin-bottom: 0.15rem;
        }
        .analysis-status-detail {
            color: var(--text-secondary);
            font-size: 0.88rem;
        }
        div[data-testid="stProgressBar"] > div > div {
            background-color: rgba(74, 144, 217, 0.22);
        }
        div[data-testid="stProgressBar"] > div > div > div {
            background: linear-gradient(90deg, #4a90d9 0%, #6caef0 100%);
        }
        div[data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(19, 26, 37, 0.96) 0%, rgba(15, 22, 33, 0.96) 100%);
            border: 1px solid var(--panel-border);
            border-radius: 16px;
            padding: 0.95rem 1rem;
            min-height: 134px;
        }
        div[data-testid="stMetricLabel"] {
            color: var(--text-secondary);
        }
        div[data-testid="stMetricValue"] {
            color: var(--text-primary);
            font-size: 1.85rem;
            font-weight: 700;
            line-height: 1.05;
        }
        div[data-testid="stMetricDelta"] {
            color: var(--text-secondary);
        }
        .ratio-card {
            border-radius: 16px;
            padding: 1rem 1rem 0.95rem 1rem;
            border: 1px solid var(--panel-border);
            min-height: 146px;
        }
        .ratio-card.good {
            background: linear-gradient(180deg, rgba(16, 32, 25, 0.96) 0%, rgba(11, 24, 19, 0.96) 100%);
            border-color: rgba(60, 169, 120, 0.35);
        }
        .ratio-card.warn {
            background: linear-gradient(180deg, rgba(37, 28, 17, 0.96) 0%, rgba(25, 19, 12, 0.96) 100%);
            border-color: rgba(211, 154, 69, 0.36);
        }
        .ratio-card.bad {
            background: linear-gradient(180deg, rgba(37, 20, 20, 0.96) 0%, rgba(27, 14, 14, 0.96) 100%);
            border-color: rgba(214, 95, 95, 0.36);
        }
        .ratio-card.neutral {
            background: linear-gradient(180deg, rgba(19, 26, 37, 0.96) 0%, rgba(15, 22, 33, 0.96) 100%);
        }
        .ratio-label {
            color: var(--text-secondary);
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }
        .ratio-value {
            color: var(--text-primary);
            font-size: 2.05rem;
            font-weight: 700;
            line-height: 1.05;
            margin: 0.4rem 0 0.45rem 0;
        }
        .ratio-status {
            display: inline-flex;
            padding: 0.28rem 0.6rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.02em;
        }
        .ratio-card.good .ratio-status {
            background: rgba(60, 169, 120, 0.16);
            color: #7de0ad;
        }
        .ratio-card.warn .ratio-status {
            background: rgba(211, 154, 69, 0.18);
            color: #f3be73;
        }
        .ratio-card.bad .ratio-status {
            background: rgba(214, 95, 95, 0.18);
            color: #ff9a9a;
        }
        .ratio-card.neutral .ratio-status {
            background: rgba(126, 136, 152, 0.18);
            color: #c3cad3;
        }
        .ratio-note {
            color: var(--text-secondary);
            font-size: 0.82rem;
            margin-top: 0.55rem;
            line-height: 1.45;
        }
        .download-shell {
            margin-top: 20px;
            margin-bottom: 1rem;
        }
        .download-note {
            color: var(--text-secondary);
            font-size: 0.88rem;
            line-height: 1.5;
        }
        .comparison-card {
            background: linear-gradient(180deg, rgba(19, 26, 37, 0.96) 0%, rgba(15, 22, 33, 0.96) 100%);
            border: 1px solid var(--panel-border);
            border-radius: 18px;
            padding: 1.1rem 1.1rem 1rem 1.1rem;
            min-height: 218px;
            box-shadow: 0 14px 36px rgba(5, 10, 18, 0.26);
        }
        .comparison-card.error {
            border-color: rgba(214, 95, 95, 0.36);
            background: linear-gradient(180deg, rgba(37, 20, 20, 0.96) 0%, rgba(27, 14, 14, 0.96) 100%);
        }
        .comparison-company {
            font-size: 1.18rem;
            font-weight: 700;
            line-height: 1.2;
            margin-bottom: 0.18rem;
        }
        .comparison-ticker {
            color: var(--text-secondary);
            font-size: 0.85rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.85rem;
        }
        .rating-circle {
            width: 90px;
            height: 90px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.85rem;
            font-weight: 800;
            color: white;
            margin-bottom: 0.85rem;
        }
        .rating-circle-strong {
            background: linear-gradient(180deg, #3ca978 0%, #2d825d 100%);
        }
        .rating-circle-investment {
            background: linear-gradient(180deg, #4a90d9 0%, #3b75b2 100%);
        }
        .rating-circle-speculative {
            background: linear-gradient(180deg, #d39a45 0%, #ac7831 100%);
        }
        .rating-circle-distressed {
            background: linear-gradient(180deg, #d65f5f 0%, #a94747 100%);
        }
        .rating-circle-failed {
            background: linear-gradient(180deg, #d65f5f 0%, #a94747 100%);
            font-size: 2rem;
        }
        .comparison-card-copy {
            color: var(--text-secondary);
            font-size: 0.88rem;
            line-height: 1.5;
            margin-top: 0.8rem;
        }
        .comparison-table-shell {
            background: linear-gradient(180deg, rgba(19, 26, 37, 0.96) 0%, rgba(15, 22, 33, 0.96) 100%);
            border: 1px solid var(--panel-border);
            border-radius: 18px;
            padding: 1.1rem 1.15rem;
            margin-bottom: 1rem;
        }
        .comparison-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }
        .comparison-table th,
        .comparison-table td {
            border: 1px solid rgba(36, 50, 69, 0.9);
            padding: 0.7rem 0.75rem;
            text-align: center;
        }
        .comparison-table th {
            background: rgba(18, 25, 37, 0.98);
            color: var(--text-primary);
            font-weight: 700;
        }
        .comparison-table td:first-child,
        .comparison-table th:first-child {
            text-align: left;
        }
        .comp-good {
            background: rgba(60, 169, 120, 0.14);
            color: #89e3b4;
        }
        .comp-warn {
            background: rgba(211, 154, 69, 0.16);
            color: #f4c57f;
        }
        .comp-bad {
            background: rgba(214, 95, 95, 0.16);
            color: #ffaaaa;
        }
        .comp-neutral {
            background: rgba(126, 136, 152, 0.14);
            color: #c7d0db;
        }
        .comp-best {
            font-weight: 800;
            box-shadow: inset 0 0 0 1px rgba(74, 144, 217, 0.42);
        }
        .risk-matrix-card {
            background: linear-gradient(180deg, rgba(19, 26, 37, 0.96) 0%, rgba(15, 22, 33, 0.96) 100%);
            border: 1px solid var(--panel-border);
            border-radius: 18px;
            padding: 1rem 1.05rem;
            min-height: 220px;
        }
        .risk-matrix-card.error {
            border-color: rgba(214, 95, 95, 0.36);
        }
        .risk-matrix-list {
            margin: 0.75rem 0 0 0;
            padding-left: 1.1rem;
            color: var(--text-primary);
        }
        .risk-matrix-list li {
            margin-bottom: 0.45rem;
            line-height: 1.42;
        }
        .comparison-caption {
            color: var(--text-secondary);
            font-size: 0.86rem;
            margin-top: 0.55rem;
        }
        .quant-card {
            border-radius: 16px;
            padding: 1rem 1rem 0.95rem 1rem;
            border: 1px solid var(--panel-border);
            min-height: 245px;
        }
        .quant-card.good {
            background: linear-gradient(180deg, rgba(16, 32, 25, 0.96) 0%, rgba(11, 24, 19, 0.96) 100%);
            border-color: rgba(60, 169, 120, 0.35);
        }
        .quant-card.warn {
            background: linear-gradient(180deg, rgba(37, 28, 17, 0.96) 0%, rgba(25, 19, 12, 0.96) 100%);
            border-color: rgba(211, 154, 69, 0.36);
        }
        .quant-card.bad {
            background: linear-gradient(180deg, rgba(37, 20, 20, 0.96) 0%, rgba(27, 14, 14, 0.96) 100%);
            border-color: rgba(214, 95, 95, 0.36);
        }
        .quant-card.neutral {
            background: linear-gradient(180deg, rgba(19, 26, 37, 0.96) 0%, rgba(15, 22, 33, 0.96) 100%);
        }
        .quant-label {
            color: var(--text-secondary);
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }
        .quant-value {
            color: var(--text-primary);
            font-size: 2.15rem;
            font-weight: 700;
            line-height: 1.05;
            margin: 0.35rem 0 0.5rem 0;
        }
        .quant-copy {
            color: var(--text-secondary);
            font-size: 0.84rem;
            line-height: 1.5;
        }
        .fscore-grid {
            display: grid;
            grid-template-columns: repeat(9, minmax(0, 1fr));
            gap: 0.35rem;
            margin: 0.75rem 0 0.65rem 0;
        }
        .fscore-box {
            border-radius: 10px;
            min-height: 44px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.82rem;
            font-weight: 800;
            border: 1px solid transparent;
        }
        .fscore-box.pass {
            background: rgba(60, 169, 120, 0.18);
            border-color: rgba(60, 169, 120, 0.34);
            color: #7de0ad;
        }
        .fscore-box.fail {
            background: rgba(126, 136, 152, 0.16);
            border-color: rgba(126, 136, 152, 0.28);
            color: #c3cad3;
        }
        .fscore-box.unknown {
            background: rgba(74, 144, 217, 0.18);
            border-color: rgba(74, 144, 217, 0.34);
            color: #a9cefb;
        }
        .fscore-labels {
            display: grid;
            grid-template-columns: repeat(9, minmax(0, 1fr));
            gap: 0.35rem;
        }
        .fscore-label {
            color: var(--text-secondary);
            font-size: 0.68rem;
            line-height: 1.35;
        }
        .trend-arrow-up {
            color: #7de0ad;
            font-weight: 800;
        }
        .trend-arrow-down {
            color: #ff9a9a;
            font-weight: 800;
        }
        .trend-arrow-flat {
            color: #c3cad3;
            font-weight: 800;
        }
        div[data-baseweb="input"] > div,
        div[data-baseweb="select"] > div,
        div[data-baseweb="textarea"] > div {
            background-color: rgba(19, 26, 37, 0.98);
            border-color: var(--panel-border);
        }
        .stTextInput input,
        .stTextArea textarea,
        .stNumberInput input {
            color: var(--text-primary);
        }
        .stButton > button,
        .stDownloadButton > button {
            border-radius: 12px;
            border: 1px solid rgba(74, 144, 217, 0.34);
            background: linear-gradient(180deg, rgba(74, 144, 217, 0.96) 0%, rgba(60, 121, 190, 0.96) 100%);
            color: white;
            font-weight: 700;
            min-height: 2.9rem;
        }
        .stDownloadButton > button {
            width: 100%;
        }
        .stButton > button:hover,
        .stDownloadButton > button:hover {
            border-color: rgba(108, 174, 240, 0.7);
            background: linear-gradient(180deg, rgba(89, 158, 226, 0.98) 0%, rgba(72, 135, 204, 0.98) 100%);
        }
        div[data-testid="stExpander"] details {
            background: linear-gradient(180deg, rgba(19, 26, 37, 0.96) 0%, rgba(15, 22, 33, 0.96) 100%);
            border: 1px solid var(--panel-border);
            border-radius: 14px;
        }
        div[data-testid="stExpander"] summary {
            color: var(--text-primary);
            font-weight: 600;
        }
        .stAlert {
            background-color: rgba(19, 26, 37, 0.95);
            border: 1px solid var(--panel-border);
            color: var(--text-primary);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _escape(value: str) -> str:
    return html.escape(value)


def parse_ticker_input(raw_value: str) -> list[str]:
    tickers: list[str] = []
    seen: set[str] = set()
    for part in raw_value.split(","):
        normalized = part.strip().upper()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        tickers.append(normalized)

    if not tickers:
        raise ValueError("Enter at least one ticker symbol.")
    if len(tickers) > MAX_COMPARISON_TICKERS:
        raise ValueError("Enter up to 3 ticker symbols separated by commas.")
    return tickers


def build_analysis_cache_key(
    *,
    ticker: str,
    extraction_model: str,
    memo_model: str,
) -> str:
    return f"{ticker.upper()}::{extraction_model}::{memo_model}"


def format_analysis_error(exc: Exception) -> str:
    if isinstance(exc, ConfigurationError):
        return str(exc)
    if isinstance(exc, TickerNotFoundError):
        return str(exc)
    if isinstance(exc, FilingNotFoundError):
        return str(exc)
    if isinstance(exc, EdgarRateLimitError):
        return str(exc)
    if isinstance(exc, EdgarResponseError):
        return str(exc)
    if isinstance(exc, OpenAIConfigurationError):
        return str(exc)
    if isinstance(exc, ExtractionParseError):
        return f"Could not parse the financial extraction output: {exc}"
    if isinstance(exc, MemoParseError):
        return f"Could not parse the credit memo output: {exc}"
    if isinstance(exc, OpenAIExtractionError):
        return str(exc)
    if isinstance(exc, MemoGenerationError):
        return str(exc)
    if isinstance(exc, FinancialExtractionError):
        return str(exc)
    return f"Unexpected analysis error: {exc}"


def clear_single_company_state() -> None:
    st.session_state["filing_metadata"] = None
    st.session_state["filing_text"] = None
    st.session_state["financial_result"] = None
    st.session_state["memo_result"] = None
    st.session_state["quantitative_result"] = None
    st.session_state["prior_filing_metadata"] = None
    st.session_state["prior_financial_result"] = None
    st.session_state["last_ticker"] = ""


def store_single_company_state(ticker: str, result_bundle: dict[str, Any]) -> None:
    st.session_state["active_mode"] = "single"
    st.session_state["active_tickers"] = (ticker,)
    st.session_state["active_results"] = {ticker: result_bundle}
    st.session_state["active_failures"] = {}
    st.session_state["filing_metadata"] = result_bundle["filing_metadata"]
    st.session_state["filing_text"] = result_bundle["filing_text"]
    st.session_state["financial_result"] = result_bundle["financial_result"]
    st.session_state["memo_result"] = result_bundle["memo_result"]
    st.session_state["quantitative_result"] = result_bundle.get("quantitative_result")
    st.session_state["prior_filing_metadata"] = result_bundle.get("prior_filing_metadata")
    st.session_state["prior_financial_result"] = result_bundle.get("prior_financial_result")
    st.session_state["last_ticker"] = ticker


def store_comparison_state(
    *,
    requested_tickers: list[str],
    results: dict[str, dict[str, Any]],
    failures: dict[str, str],
) -> None:
    st.session_state["active_mode"] = "comparison"
    st.session_state["active_tickers"] = tuple(requested_tickers)
    st.session_state["active_results"] = results
    st.session_state["active_failures"] = failures
    clear_single_company_state()


def get_rating_circle_class(rating: str) -> str:
    base_rating = rating.strip().upper().rstrip("+-")
    if base_rating in {"AAA", "AA", "A"}:
        return "rating-circle-strong"
    if base_rating == "BBB":
        return "rating-circle-investment"
    if base_rating in {"BB", "B"}:
        return "rating-circle-speculative"
    return "rating-circle-distressed"


def get_rating_badge_class(rating: str) -> str:
    base_rating = rating.strip().upper().rstrip("+-")
    if base_rating in {"AAA", "AA", "A"}:
        return "badge-rating-strong"
    if base_rating == "BBB":
        return "badge-rating-investment"
    if base_rating in {"BB", "B"}:
        return "badge-rating-speculative"
    return "badge-rating-distressed"


def get_outlook_badge_class(direction: str) -> str:
    normalized_direction = direction.strip().lower()
    if normalized_direction == "positive":
        return "badge-outlook-positive"
    if normalized_direction == "negative":
        return "badge-outlook-negative"
    return "badge-outlook-stable"


def get_outlook_box_class(direction: str) -> str:
    normalized_direction = direction.strip().lower()
    if normalized_direction == "positive":
        return "positive"
    if normalized_direction == "negative":
        return "negative"
    return ""


def get_ratio_status(ratio_name: str, value: float | None) -> tuple[str, str]:
    if value is None:
        return "neutral", "Unavailable"

    if ratio_name == "debt_to_equity":
        if value < 1.5:
            return "good", "Healthy"
        if value <= 3.0:
            return "warn", "Borderline"
        return "bad", "Concerning"

    if ratio_name == "interest_coverage":
        if value > 5:
            return "good", "Healthy"
        if value >= 2:
            return "warn", "Borderline"
        return "bad", "Concerning"

    if ratio_name == "current_ratio":
        if value > 1.5:
            return "good", "Healthy"
        if value >= 1.0:
            return "warn", "Borderline"
        return "bad", "Concerning"

    if ratio_name == "net_debt_to_ebitda":
        if value < 2.0:
            return "good", "Healthy"
        if value <= 4.0:
            return "warn", "Borderline"
        return "bad", "Concerning"

    return "neutral", "Computed"


def get_altman_status(score: float | None) -> tuple[str, str]:
    if score is None:
        return "neutral", "Unavailable"
    if score > 2.99:
        return "good", "Safe Zone"
    if score >= 1.81:
        return "warn", "Gray Zone"
    return "bad", "Distress Zone"


def get_piotroski_status(piotroski_result: Any) -> tuple[str, str]:
    score = piotroski_result.score
    evaluated_count = getattr(piotroski_result, "evaluated_count", 0)
    if score is None or evaluated_count == 0:
        return "neutral", "Unavailable"
    pass_rate = score / evaluated_count
    if pass_rate >= 0.8:
        return "good", "Strong"
    if pass_rate >= 0.5:
        return "warn", "Moderate"
    return "bad", "Weak"


def get_cash_flow_adequacy_status(value: float | None) -> tuple[str, str]:
    if value is None:
        return "neutral", "Unavailable"
    if value > 1.5:
        return "good", "Strong"
    if value >= 1.0:
        return "warn", "Adequate"
    return "bad", "Weak"


def format_quant_score(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2f}"


def format_percent_change(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.1%}"


def get_trend_arrow(improving: bool | None) -> tuple[str, str]:
    if improving is None:
        return "→", "trend-arrow-flat"
    if improving:
        return "↑", "trend-arrow-up"
    return "↓", "trend-arrow-down"


def ratio_prefers_higher(ratio_name: str) -> bool:
    return ratio_name in {"interest_coverage", "current_ratio"}


def get_best_ratio_ticker(
    *,
    ratio_name: str,
    requested_tickers: tuple[str, ...],
    results: dict[str, dict[str, Any]],
) -> str | None:
    available_values: list[tuple[str, float]] = []
    for ticker in requested_tickers:
        result_bundle = results.get(ticker)
        if not result_bundle:
            continue
        ratio_value = result_bundle["financial_result"].ratios[ratio_name].value
        if ratio_value is None:
            continue
        available_values.append((ticker, ratio_value))

    if not available_values:
        return None
    if ratio_prefers_higher(ratio_name):
        return max(available_values, key=lambda item: item[1])[0]
    return min(available_values, key=lambda item: item[1])[0]


def render_failure_alerts(failures: dict[str, str]) -> None:
    for ticker, message in failures.items():
        st.warning(f"{ticker}: {message}")


def render_workspace_header() -> None:
    st.markdown(
        """
        <div class="workspace-shell">
            <div class="workspace-kicker">Institutional Credit Workflow</div>
            <div class="workspace-title">AI Credit Analyst Agent</div>
            <div class="workspace-copy">
                Pull the latest SEC 10-K, extract normalized financials, compute core credit ratios,
                and generate a committee-ready memo in one pass.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state() -> None:
    st.markdown(
        """
        <div class="section-shell">
            <div class="section-kicker">How It Works</div>
            <div class="section-title">Run a full public-company credit review from a single ticker.</div>
            <div class="section-copy">
                The app fetches the latest 10-K from SEC EDGAR, extracts core financial statement values,
                computes credit ratios, and produces a structured rating memo with filing-cited evidence.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_header(memo_result: Any) -> None:
    st.markdown(
        f"""
        <div class="report-shell">
            <div class="report-header">
                <div>
                    <div class="section-kicker">Credit Risk Memo</div>
                    <div class="report-company">
                        {_escape(memo_result.company_name)}
                        <span class="report-ticker">{_escape(memo_result.ticker)}</span>
                    </div>
                    <div class="report-meta">Most recent 10-K filed on {_escape(memo_result.filing_date)}</div>
                </div>
                <div class="header-badges">
                    <span class="badge {get_rating_badge_class(memo_result.credit_risk_rating.rating)}">{_escape(memo_result.credit_risk_rating.rating)}</span>
                    <span class="badge {get_outlook_badge_class(memo_result.outlook.direction)}">{_escape(memo_result.outlook.direction)} Outlook</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overview_and_rating(memo_result: Any) -> None:
    overview_col, rating_col = st.columns([1.15, 1.0])
    with overview_col:
        st.markdown(
            f"""
            <div class="section-shell">
                <div class="section-kicker">Company Overview</div>
                <div class="section-copy">{_escape(memo_result.company_overview)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with rating_col:
        st.markdown(
            f"""
            <div class="callout-shell">
                <div class="section-kicker">Credit Risk Rating</div>
                <div class="section-title">{_escape(memo_result.credit_risk_rating.rating)}</div>
                <div class="justification-box">{_escape(memo_result.credit_risk_rating.justification)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_financial_summary(financial_result: Any) -> None:
    st.markdown(
        """
        <div class="section-shell">
            <div class="section-kicker">Financial Summary</div>
            <div class="section-copy">Normalized 10-K statement values and derived credit ratios.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(financial_result.normalization_note)

    for start_index in range(0, len(PRIMARY_METRIC_FIELDS), 4):
        metric_columns = st.columns(4)
        for field_name, column in zip(PRIMARY_METRIC_FIELDS[start_index : start_index + 4], metric_columns):
            metric = financial_result.metrics[field_name]
            with column:
                st.metric(
                    label=FIELD_LABELS[field_name],
                    value=format_financial_value(metric.value),
                )

    st.markdown(
        """
        <div style="margin-top: 30px; margin-bottom: 0.7rem;">
            <div class="section-kicker">Credit Ratios</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    ratio_columns = st.columns(4)
    for ratio_name, column in zip(PRIMARY_RATIO_FIELDS, ratio_columns):
        ratio = financial_result.ratios[ratio_name]
        ratio_class, ratio_status = get_ratio_status(ratio_name, ratio.value)
        ratio_note = ratio.note or "Computed from normalized extracted values."
        with column:
            st.markdown(
                f"""
                <div class="ratio-card {ratio_class}">
                    <div class="ratio-label">{_escape(RATIO_LABELS[ratio_name])}</div>
                    <div class="ratio-value">{_escape(format_ratio_value(ratio.value, ratio_name))}</div>
                    <div class="ratio-status">{_escape(ratio_status)}</div>
                    <div class="ratio-note">{_escape(ratio_note)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_quantitative_model_cards(quantitative_result: Any) -> None:
    altman_class, altman_zone = get_altman_status(quantitative_result.altman_z_score.score)
    piotroski_class, piotroski_label = get_piotroski_status(quantitative_result.piotroski_f_score)
    adequacy_class, adequacy_label = get_cash_flow_adequacy_status(quantitative_result.cash_flow_adequacy.ratio)

    piotroski_boxes = ""
    piotroski_labels = ""
    for criterion in quantitative_result.piotroski_f_score.criteria:
        if criterion.passed is None:
            box_class = "unknown"
            box_value = "?"
            label_text = f"{_escape(criterion.label)}<br>Insufficient data"
        elif criterion.passed:
            box_class = "pass"
            box_value = "1"
            label_text = _escape(criterion.label)
        else:
            box_class = "fail"
            box_value = "0"
            label_text = _escape(criterion.label)
        piotroski_boxes += (
            f'<div class="fscore-box {box_class}">{box_value}</div>'
        )
        piotroski_labels += f'<div class="fscore-label">{label_text}</div>'

    score_display = "N/A"
    if quantitative_result.piotroski_f_score.score is not None and quantitative_result.piotroski_f_score.evaluated_count > 0:
        score_display = (
            f"{quantitative_result.piotroski_f_score.score}/"
            f"{quantitative_result.piotroski_f_score.evaluated_count}"
        )

    columns = st.columns(3, gap="medium")
    with columns[0]:
        st.markdown(
            f"""
            <div class="quant-card {altman_class}">
                <div class="quant-label">Altman Z-Score</div>
                <div class="quant-value">{_escape(format_quant_score(quantitative_result.altman_z_score.score))}</div>
                <div class="ratio-status">{_escape(altman_zone)}</div>
                <div class="quant-copy" style="margin-top: 0.7rem;">
                    The Altman Z-Score predicts the probability of bankruptcy within a two-year window with approximately 80-90% historical accuracy.
                </div>
                <div class="quant-copy" style="margin-top: 0.7rem;">{_escape(quantitative_result.altman_z_score.note or "Computed from extracted statement values and market capitalization inputs.")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with columns[1]:
        st.markdown(
            f"""
            <div class="quant-card {piotroski_class}">
                <div class="quant-label">Piotroski F-Score</div>
                <div class="quant-value">{_escape(score_display)}</div>
                <div class="ratio-status">{_escape(piotroski_label)}</div>
                <div class="fscore-grid">{piotroski_boxes or '<div class="fscore-box fail">N/A</div>'}</div>
                <div class="fscore-labels">{piotroski_labels or '<div class="fscore-label">Prior year filing required.</div>'}</div>
                <div class="quant-copy" style="margin-top: 0.7rem;">{_escape(quantitative_result.piotroski_f_score.note or "Nine-point financial health score using profitability, leverage, liquidity, and operating efficiency tests.")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with columns[2]:
        st.markdown(
            f"""
            <div class="quant-card {adequacy_class}">
                <div class="quant-label">Cash Flow Adequacy</div>
                <div class="quant-value">{_escape(format_ratio_value(quantitative_result.cash_flow_adequacy.ratio, "net_debt_to_ebitda"))}</div>
                <div class="ratio-status">{_escape(adequacy_label)}</div>
                <div class="quant-copy" style="margin-top: 0.7rem;">{_escape(quantitative_result.cash_flow_adequacy.assessment)}</div>
                <div class="quant-copy" style="margin-top: 0.7rem;">{_escape(quantitative_result.cash_flow_adequacy.note or "Operating cash flow divided by near-term debt obligations, interest expense, and capital expenditures.")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_altman_breakdown(quantitative_result: Any) -> None:
    rows = [
        {
            "Component": component.label,
            "Raw Value": "N/A" if component.raw_value is None else f"{component.raw_value:.2f}",
            "Weighted Contribution": "N/A" if component.weighted_contribution is None else f"{component.weighted_contribution:.2f}",
            "Measure": component.description,
        }
        for component in quantitative_result.altman_z_score.components
    ]
    st.markdown(
        """
        <div class="comparison-table-shell">
            <div class="section-kicker">Altman Breakdown</div>
            <div class="section-copy">Each component shows the underlying ratio, its weighted contribution, and the balance-sheet or earnings dimension it captures.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(rows, use_container_width=True, hide_index=True)


def render_trend_analysis(quantitative_result: Any) -> None:
    st.markdown(
        """
        <div class="comparison-table-shell">
            <div class="section-kicker">Trend Analysis</div>
            <div class="section-copy">Year-over-year changes across the key operating, leverage, liquidity, and cash-flow indicators.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if quantitative_result.trend_analysis_note and not quantitative_result.trend_analysis:
        st.info(quantitative_result.trend_analysis_note)
        return

    table_rows: list[dict[str, str]] = []
    has_notes = any(row.note for row in quantitative_result.trend_analysis)
    for row in quantitative_result.trend_analysis:
        arrow, _ = get_trend_arrow(row.improving)
        if not row.prior_data_reliable:
            prior_value = "N/A"
            absolute_change = "N/A"
        elif row.is_ratio:
            prior_value = format_ratio_value(row.prior_value, "net_debt_to_ebitda")
            absolute_change = format_ratio_value(row.absolute_change, "net_debt_to_ebitda")
        else:
            prior_value = format_financial_value(row.prior_value)
            absolute_change = format_financial_value(row.absolute_change)

        if row.is_ratio:
            current_value = format_ratio_value(row.current_value, "net_debt_to_ebitda")
        else:
            current_value = format_financial_value(row.current_value)
        direction = "N/A" if not row.prior_data_reliable else arrow

        row_data = {
            "Metric": row.label,
            "Current": current_value,
            "Prior": prior_value,
            "Change": absolute_change,
            "% Change": format_percent_change(row.percent_change),
            "Direction": direction,
        }
        if has_notes:
            row_data["Note"] = row.note or ""
        table_rows.append(row_data)

    trend_dataframe = pd.DataFrame(table_rows)
    st.dataframe(trend_dataframe, use_container_width=True, hide_index=True)

    ratio_history = quantitative_result.ratio_history
    if any(prior is not None for _, prior in ratio_history.values()):
        line_chart_data = pd.DataFrame(
            {
                RATIO_LABELS[ratio_name]: [prior_value, current_value]
                for ratio_name, (current_value, prior_value) in ratio_history.items()
            },
            index=["Prior Year", "Current Year"],
        )
        st.line_chart(line_chart_data, use_container_width=True, height=260)


def render_debt_maturity_profile(quantitative_result: Any) -> None:
    st.markdown(
        """
        <div class="comparison-table-shell">
            <div class="section-kicker">Debt Maturity Profile</div>
            <div class="section-copy">Scheduled debt maturities by year with maturity-wall monitoring and weighted-average tenor.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    debt_profile = quantitative_result.debt_maturity_profile
    if debt_profile.note and not debt_profile.tranches:
        st.info(debt_profile.note)
        return

    year_totals: dict[int, float] = {}
    for tranche in debt_profile.tranches:
        year_totals[tranche.maturity_year] = year_totals.get(tranche.maturity_year, 0.0) + tranche.amount_millions

    if year_totals:
        maturity_chart = pd.DataFrame(
            {"Debt Maturing (USD millions)": list(year_totals.values())},
            index=[str(year) for year in year_totals.keys()],
        )
        st.bar_chart(maturity_chart, use_container_width=True, height=280)

    if debt_profile.weighted_average_maturity is not None:
        st.caption(f"Weighted Average Maturity: {debt_profile.weighted_average_maturity:.1f} years")
    if debt_profile.note:
        st.caption(debt_profile.note)

    tranche_rows = [
        {
            "Amount": format_financial_value(tranche.amount_millions * 1_000_000),
            "Maturity Year": tranche.maturity_year,
            "Interest Rate": tranche.interest_rate or "N/A",
            "Description": tranche.description,
        }
        for tranche in debt_profile.tranches
    ]
    st.dataframe(tranche_rows, use_container_width=True, hide_index=True)

    for warning in debt_profile.warnings:
        st.warning(
            f"Maturity Wall Warning: {warning.percentage_of_total_debt:.0%} of total debt matures in {warning.maturity_year}."
        )


def render_quantitative_models(quantitative_result: Any) -> None:
    st.markdown(
        """
        <div class="section-shell">
            <div class="section-kicker">Quantitative Models</div>
            <div class="section-copy">Institutional-grade scoring models and forward risk indicators.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_quantitative_model_cards(quantitative_result)
    render_altman_breakdown(quantitative_result)
    render_trend_analysis(quantitative_result)
    render_debt_maturity_profile(quantitative_result)


def render_financial_evidence(financial_result: Any) -> None:
    with st.expander("Financial Line Item Evidence"):
        evidence_rows: list[dict[str, str]] = []
        for field_name in FIELD_LABELS:
            metric = financial_result.metrics[field_name]
            evidence_rows.append(
                {
                    "Metric": FIELD_LABELS[field_name],
                    "Value": format_financial_value(metric.value),
                    "Evidence": metric.source_quote or "No direct source quote returned.",
                }
            )
        for ratio_name in RATIO_LABELS:
            ratio = financial_result.ratios[ratio_name]
            evidence_rows.append(
                {
                    "Metric": RATIO_LABELS[ratio_name],
                    "Value": format_ratio_value(ratio.value, ratio_name),
                    "Evidence": ratio.note or "Computed from normalized extracted values.",
                }
            )
        st.dataframe(evidence_rows, use_container_width=True, hide_index=True)


def render_points_section(
    *,
    title: str,
    points: tuple[Any, ...],
) -> None:
    st.markdown(
        f"""
        <div style="margin-bottom: 0.6rem;">
            <div class="section-kicker">{_escape(title)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    for point in points:
        with st.expander(point.title, expanded=False):
            st.markdown(point.explanation)
            st.caption(f"Evidence: {point.evidence}")


def render_outlook(memo_result: Any) -> None:
    st.markdown(
        f"""
        <div class="callout-shell">
            <div class="section-kicker">Outlook</div>
            <div class="outlook-box {get_outlook_box_class(memo_result.outlook.direction)}">
                <strong>{_escape(memo_result.outlook.direction)}</strong><br>
                {_escape(memo_result.outlook.justification)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_source_expanders(
    *,
    filing_metadata: Any,
    filing_text: str,
    financial_result: Any,
    memo_result: Any,
    preview_chars: int,
) -> None:
    with st.expander("Source Filing Preview"):
        st.markdown(f"**Raw filing URL:** {filing_metadata.filing_text_url}")
        st.text_area(
            "Raw filing text preview",
            filing_text[:preview_chars],
            height=320,
            disabled=True,
            label_visibility="collapsed",
        )

    with st.expander("Extraction Diagnostics"):
        st.caption(
            f"Chunk strategy: `{financial_result.chunk_strategy}` | "
            f"Chunks sent: {financial_result.chunk_count} | "
            f"Prepared text length: {financial_result.prepared_text_length:,} characters | "
            f"Primary 10-K isolated: {financial_result.used_primary_document}"
        )
        st.caption(financial_result.normalization_note)
        chunk_rows = [
            {
                "Label": chunk.label,
                "Strategy": chunk.strategy,
                "Start": chunk.start_char,
                "End": chunk.end_char,
                "Characters": len(chunk.text),
            }
            for chunk in financial_result.chunks
        ]
        st.dataframe(chunk_rows, use_container_width=True, hide_index=True)

    with st.expander("Memo Context Excerpts"):
        context_rows = [
            {
                "Section": section.title,
                "Excerpt": section.text[:800] + ("..." if len(section.text) > 800 else ""),
            }
            for section in memo_result.context_sections
        ]
        st.dataframe(context_rows, use_container_width=True, hide_index=True)


def render_download_actions(
    *,
    filing_metadata: Any,
    filing_text: str,
    financial_result: Any,
    memo_result: Any,
    quantitative_result: Any,
) -> None:
    st.markdown('<div class="download-shell"></div>', unsafe_allow_html=True)
    download_columns = st.columns(2)
    with download_columns[0]:
        try:
            pdf_bytes = build_credit_memo_pdf(
                memo_result,
                filing_url=filing_metadata.filing_text_url,
                extraction_methodology=(
                    "Raw 10-K text retrieved from SEC EDGAR. "
                    f"Relevant sections isolated with {financial_result.chunk_strategy} chunking. "
                    f"{financial_result.normalization_note} "
                    "Structured financial extraction was used to populate the memo, and credit ratios were computed from normalized values."
                ),
                quantitative_result=quantitative_result,
            )
        except PDFGenerationError as exc:
            st.warning(str(exc))
        else:
            st.download_button(
                label="Download as PDF",
                data=pdf_bytes,
                file_name=f"{memo_result.ticker}_credit_memo.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

    with download_columns[1]:
        st.download_button(
            label="Download raw filing",
            data=filing_text,
            file_name=f"{filing_metadata.ticker}_{filing_metadata.accession_number}_10k.txt",
            mime="text/plain",
            use_container_width=True,
        )


def render_report(
    *,
    filing_metadata: Any,
    filing_text: str,
    financial_result: Any,
    memo_result: Any,
    quantitative_result: Any,
    preview_chars: int,
) -> None:
    render_header(memo_result)
    render_download_actions(
        filing_metadata=filing_metadata,
        filing_text=filing_text,
        financial_result=financial_result,
        memo_result=memo_result,
        quantitative_result=quantitative_result,
    )
    render_overview_and_rating(memo_result)
    render_financial_summary(financial_result)
    if quantitative_result is not None:
        render_quantitative_models(quantitative_result)
    render_financial_evidence(financial_result)

    risk_col, strength_col = st.columns(2, gap="small")
    with risk_col:
        render_points_section(
            title="Key Risk Factors",
            points=memo_result.key_risk_factors,
        )
    with strength_col:
        render_points_section(
            title="Key Strengths",
            points=memo_result.key_strengths,
        )

    render_outlook(memo_result)
    render_source_expanders(
        filing_metadata=filing_metadata,
        filing_text=filing_text,
        financial_result=financial_result,
        memo_result=memo_result,
        preview_chars=preview_chars,
    )


def render_comparison_header(
    *,
    requested_tickers: tuple[str, ...],
    results: dict[str, dict[str, Any]],
    failures: dict[str, str],
) -> None:
    completed_count = len(results)
    failed_count = len(failures)
    st.markdown(
        f"""
        <div class="report-shell">
            <div class="report-header">
                <div>
                    <div class="section-kicker">Peer Credit Comparison</div>
                    <div class="report-company">{" vs ".join(_escape(ticker) for ticker in requested_tickers)}</div>
                    <div class="report-meta">
                        Completed analyses: {completed_count} | Failed analyses: {failed_count}
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_comparison_rating_cards(
    *,
    requested_tickers: tuple[str, ...],
    results: dict[str, dict[str, Any]],
    failures: dict[str, str],
) -> None:
    st.markdown(
        """
        <div class="section-shell">
            <div class="section-kicker">Rating Cards</div>
            <div class="section-copy">Instant view of rating and outlook dispersion across the selected peer set.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    columns = st.columns(len(requested_tickers), gap="medium")
    for ticker, column in zip(requested_tickers, columns):
        result_bundle = results.get(ticker)
        with column:
            if result_bundle:
                memo_result = result_bundle["memo_result"]
                st.markdown(
                    f"""
                    <div class="comparison-card">
                        <div class="comparison-company">{_escape(memo_result.company_name)}</div>
                        <div class="comparison-ticker">{_escape(memo_result.ticker)}</div>
                        <div class="rating-circle {get_rating_circle_class(memo_result.credit_risk_rating.rating)}">
                            {_escape(memo_result.credit_risk_rating.rating)}
                        </div>
                        <span class="badge {get_outlook_badge_class(memo_result.outlook.direction)}">
                            {_escape(memo_result.outlook.direction)} Outlook
                        </span>
                        <div class="comparison-card-copy">Filed {_escape(memo_result.filing_date)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="comparison-card error">
                        <div class="comparison-company">{_escape(ticker)}</div>
                        <div class="comparison-ticker">Analysis Unavailable</div>
                        <div class="rating-circle rating-circle-failed">X</div>
                        <div class="comparison-card-copy">{_escape(failures.get(ticker, "Analysis failed."))}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_comparison_ratio_table(
    *,
    requested_tickers: tuple[str, ...],
    results: dict[str, dict[str, Any]],
    failures: dict[str, str],
) -> None:
    header_cells = "".join(f"<th>{_escape(ticker)}</th>" for ticker in requested_tickers)
    row_markup: list[str] = []

    for ratio_name in PRIMARY_RATIO_FIELDS:
        best_ticker = get_best_ratio_ticker(
            ratio_name=ratio_name,
            requested_tickers=requested_tickers,
            results=results,
        )
        value_cells: list[str] = []
        for ticker in requested_tickers:
            result_bundle = results.get(ticker)
            if result_bundle:
                ratio = result_bundle["financial_result"].ratios[ratio_name]
                ratio_class, _ = get_ratio_status(ratio_name, ratio.value)
                cell_classes = [f"comp-{ratio_class}"]
                if ticker == best_ticker:
                    cell_classes.append("comp-best")
                display_value = format_ratio_value(ratio.value, ratio_name)
            else:
                cell_classes = ["comp-neutral"]
                display_value = "Failed"
            value_cells.append(
                f"<td class=\"{' '.join(cell_classes)}\">{_escape(display_value)}</td>"
            )

        row_markup.append(
            f"""
            <tr>
                <td>{_escape(RATIO_LABELS[ratio_name])}</td>
                {''.join(value_cells)}
            </tr>
            """
        )

    st.markdown(
        f"""
        <div class="comparison-table-shell">
            <div class="section-kicker">Ratio Comparison</div>
            <div class="section-copy">Cells are color-coded using the same ratio health thresholds as the single-company memo.</div>
            <table class="comparison-table">
                <thead>
                    <tr>
                        <th>Ratio</th>
                        {header_cells}
                    </tr>
                </thead>
                <tbody>
                    {''.join(row_markup)}
                </tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if failures:
        st.caption(
            "Failed issuers remain in the table so the peer set stays visible even when one company could not be processed."
        )


def render_comparison_bar_chart(
    *,
    requested_tickers: tuple[str, ...],
    results: dict[str, dict[str, Any]],
) -> None:
    successful_tickers = [ticker for ticker in requested_tickers if ticker in results]
    if not successful_tickers:
        return

    chart_data = pd.DataFrame(
        {
            ticker: [
                results[ticker]["financial_result"].metrics["revenue"].value or 0,
                results[ticker]["financial_result"].metrics["net_income"].value or 0,
                results[ticker]["financial_result"].metrics["total_debt"].value or 0,
                results[ticker]["financial_result"].metrics["cash_and_cash_equivalents"].value or 0,
            ]
            for ticker in successful_tickers
        },
        index=["Revenue", "Net Income", "Total Debt", "Cash"],
    )

    st.markdown(
        """
        <div class="section-shell">
            <div class="section-kicker">Scale Comparison</div>
            <div class="section-copy">Grouped bars compare normalized financial scale across the peer set.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.bar_chart(chart_data, use_container_width=True, height=360)
    st.caption(
        "Metrics shown: Revenue, Net Income, Total Debt, and Cash. Values are normalized to raw dollars before charting."
    )


def render_risk_summary_matrix(
    *,
    requested_tickers: tuple[str, ...],
    results: dict[str, dict[str, Any]],
    failures: dict[str, str],
) -> None:
    st.markdown(
        """
        <div class="section-shell">
            <div class="section-kicker">Risk Summary Matrix</div>
            <div class="section-copy">Top three risk titles by issuer for fast relative-value review.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    columns = st.columns(len(requested_tickers), gap="small")
    for ticker, column in zip(requested_tickers, columns):
        result_bundle = results.get(ticker)
        with column:
            if result_bundle:
                memo_result = result_bundle["memo_result"]
                risk_items = "".join(
                    f"<li>{_escape(point.title)}</li>" for point in memo_result.key_risk_factors[:3]
                )
                st.markdown(
                    f"""
                    <div class="risk-matrix-card">
                        <div class="comparison-company">{_escape(memo_result.company_name)}</div>
                        <div class="comparison-ticker">{_escape(memo_result.ticker)}</div>
                        <ol class="risk-matrix-list">
                            {risk_items}
                        </ol>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="risk-matrix-card error">
                        <div class="comparison-company">{_escape(ticker)}</div>
                        <div class="comparison-ticker">Unavailable</div>
                        <div class="comparison-card-copy">{_escape(failures.get(ticker, "Analysis failed."))}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_comparison_download_actions(
    *,
    requested_tickers: tuple[str, ...],
    results: dict[str, dict[str, Any]],
    failures: dict[str, str],
) -> None:
    if not results:
        return

    st.markdown('<div class="download-shell"></div>', unsafe_allow_html=True)
    ordered_successes = [results[ticker] for ticker in requested_tickers if ticker in results]
    try:
        pdf_bytes = build_peer_comparison_pdf(
            tuple(result_bundle["memo_result"] for result_bundle in ordered_successes),
            filing_urls={
                result_bundle["memo_result"].ticker: result_bundle["filing_metadata"].filing_text_url
                for result_bundle in ordered_successes
            },
            failed_tickers=failures,
        )
    except PDFGenerationError as exc:
        st.warning(str(exc))
        return

    st.download_button(
        label="Download comparison PDF",
        data=pdf_bytes,
        file_name=f"{'_'.join(requested_tickers)}_credit_comparison.pdf",
        mime="application/pdf",
        use_container_width=True,
    )


def render_comparison_dashboard(
    *,
    requested_tickers: tuple[str, ...],
    results: dict[str, dict[str, Any]],
    failures: dict[str, str],
) -> None:
    render_comparison_header(
        requested_tickers=requested_tickers,
        results=results,
        failures=failures,
    )
    if failures:
        render_failure_alerts(failures)
    render_comparison_download_actions(
        requested_tickers=requested_tickers,
        results=results,
        failures=failures,
    )
    render_comparison_rating_cards(
        requested_tickers=requested_tickers,
        results=results,
        failures=failures,
    )
    render_comparison_ratio_table(
        requested_tickers=requested_tickers,
        results=results,
        failures=failures,
    )
    render_comparison_bar_chart(
        requested_tickers=requested_tickers,
        results=results,
    )
    render_risk_summary_matrix(
        requested_tickers=requested_tickers,
        results=results,
        failures=failures,
    )


def render_footer() -> None:
    st.markdown(
        """
        <div class="footer-shell">
            Built by Chase Nielsen | Data from SEC EDGAR
        </div>
        """,
        unsafe_allow_html=True,
    )


def update_analysis_status(
    *,
    status_placeholder: Any,
    progress_bar: Any,
    percent_complete: int,
    title: str,
    detail: str,
) -> None:
    progress_bar.progress(percent_complete, text=title)
    status_placeholder.markdown(
        f"""
        <div class="analysis-status">
            <div class="analysis-status-title">{_escape(title)}</div>
            <div class="analysis-status-detail">{_escape(detail)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def run_company_analysis(
    *,
    ticker: str,
    requester_name: str,
    requester_email: str,
    openai_api_key: str,
    extraction_model: str,
    memo_model: str,
    status_placeholder: Any,
    progress_bar: Any,
    progress_start: int,
    progress_end: int,
    overall_title: str | None = None,
) -> dict[str, Any]:
    progress_span = max(progress_end - progress_start, 1)

    def scaled_progress(percent_complete: int) -> int:
        return progress_start + int((percent_complete / 100) * progress_span)

    def report_stage(relative_percent: int, phase_title: str, phase_detail: str) -> None:
        update_analysis_status(
            status_placeholder=status_placeholder,
            progress_bar=progress_bar,
            percent_complete=min(scaled_progress(relative_percent), progress_end),
            title=overall_title or phase_title,
            detail=phase_detail if overall_title is None else f"{phase_title} {phase_detail}",
        )

    report_stage(
        8,
        "Fetching 10-K from SEC EDGAR...",
        "Resolving the latest annual filing and downloading the raw submission text.",
    )
    filing_metadata, filing_text = load_latest_10k(
        ticker=ticker,
        requester_name=requester_name,
        requester_email=requester_email,
    )

    report_stage(
        22,
        "Extracting financial data...",
        f"Running statement-focused chunk extraction with {extraction_model}.",
    )

    def handle_chunk_progress(current_index: int, total_chunks: int, _: Any) -> None:
        extraction_progress = 22
        if total_chunks > 0:
            extraction_progress += int((current_index / total_chunks) * 33)
        report_stage(
            min(extraction_progress, 55),
            "Extracting financial data...",
            f"Extracting chunk {current_index}/{total_chunks} with {extraction_model}.",
        )

    financial_result = extract_financials(
        raw_filing_text=filing_text,
        company_name=filing_metadata.company_name,
        ticker=filing_metadata.ticker,
        filing_date=filing_metadata.filing_date,
        api_key=openai_api_key,
        model=extraction_model,
        chunk_progress_callback=handle_chunk_progress,
    )

    report_stage(
        70,
        "Computing credit ratios...",
        "Recomputing ratios from normalized filing values before memo assembly.",
    )

    report_stage(
        76,
        "Preparing memo generation...",
        "Applying the required cooldown before the final memo call.",
    )
    for second in range(MEMO_PREPARATION_COOLDOWN_SECONDS):
        time.sleep(1)
        report_stage(
            76 + int(((second + 1) / MEMO_PREPARATION_COOLDOWN_SECONDS) * 8),
            "Preparing memo generation...",
            f"Cooldown in progress: {MEMO_PREPARATION_COOLDOWN_SECONDS - second - 1} seconds remaining.",
        )

    report_stage(
        88,
        "Generating credit memo...",
        f"Drafting the final committee-style memo with {memo_model}.",
    )
    memo_result = generate_credit_memo(
        raw_filing_text=filing_text,
        company_name=filing_metadata.company_name,
        ticker=filing_metadata.ticker,
        filing_date=filing_metadata.filing_date,
        financial_result=financial_result,
        api_key=openai_api_key,
        model=memo_model,
    )
    return {
        "filing_metadata": filing_metadata,
        "filing_text": filing_text,
        "financial_result": financial_result,
        "memo_result": memo_result,
    }


def augment_single_company_with_quantitative_models(
    *,
    result_bundle: dict[str, Any],
    requester_name: str,
    requester_email: str,
    openai_api_key: str,
    extraction_model: str,
    status_placeholder: Any,
    progress_bar: Any,
    progress_start: int,
    progress_end: int,
) -> dict[str, Any]:
    progress_span = max(progress_end - progress_start, 1)

    def scaled_progress(percent_complete: int) -> int:
        return progress_start + int((percent_complete / 100) * progress_span)

    def report_stage(percent_complete: int, title: str, detail: str) -> None:
        update_analysis_status(
            status_placeholder=status_placeholder,
            progress_bar=progress_bar,
            percent_complete=min(scaled_progress(percent_complete), progress_end),
            title=title,
            detail=detail,
        )

    current_metadata = result_bundle["filing_metadata"]
    current_financial_result = result_bundle["financial_result"]
    prior_filing_metadata = None
    prior_financial_result = None

    report_stage(
        12,
        "Fetching prior-year 10-K...",
        "Retrieving the second-most-recent annual filing for multi-year model inputs.",
    )
    try:
        recent_filings = load_recent_10ks(
            current_metadata.ticker,
            requester_name=requester_name,
            requester_email=requester_email,
            limit=2,
        )
    except Exception:
        recent_filings = ((current_metadata, result_bundle["filing_text"]),)

    if len(recent_filings) > 1:
        prior_filing_metadata, prior_filing_text = recent_filings[1]
        report_stage(
            28,
            "Extracting prior-year financial data...",
            f"Running the prior-year extraction with {PRIOR_YEAR_EXTRACTION_MODEL}.",
        )
        time.sleep(QUANTITATIVE_INTER_MODEL_DELAY_SECONDS)

        def handle_prior_chunk_progress(current_index: int, total_chunks: int, _: Any) -> None:
            extraction_progress = 28
            if total_chunks > 0:
                extraction_progress += int((current_index / total_chunks) * 32)
            report_stage(
                min(extraction_progress, 60),
                "Extracting prior-year financial data...",
                f"Extracting prior-year chunk {current_index}/{total_chunks} with {PRIOR_YEAR_EXTRACTION_MODEL}.",
            )

        try:
            prior_financial_result = extract_financials(
                raw_filing_text=prior_filing_text,
                company_name=prior_filing_metadata.company_name,
                ticker=prior_filing_metadata.ticker,
                filing_date=prior_filing_metadata.filing_date,
                api_key=openai_api_key,
                model=PRIOR_YEAR_EXTRACTION_MODEL,
                chunk_progress_callback=handle_prior_chunk_progress,
            )
        except Exception:
            prior_filing_metadata = None
            prior_financial_result = None

    report_stage(
        72,
        "Running quantitative models...",
        "Computing Altman Z-Score, Piotroski F-Score, trend analysis, cash flow adequacy, and debt maturity profile.",
    )
    time.sleep(QUANTITATIVE_INTER_MODEL_DELAY_SECONDS)
    quantitative_result = build_quantitative_model_result(
        company_name=current_metadata.company_name,
        ticker=current_metadata.ticker,
        filing_date=current_metadata.filing_date,
        raw_filing_text=result_bundle["filing_text"],
        current_result=current_financial_result,
        prior_result=prior_financial_result,
        prior_filing_date=None if prior_filing_metadata is None else prior_filing_metadata.filing_date,
        api_key=openai_api_key,
        debt_model=extraction_model,
    )
    result_bundle["quantitative_result"] = quantitative_result
    result_bundle["prior_filing_metadata"] = prior_filing_metadata
    result_bundle["prior_financial_result"] = prior_financial_result

    report_stage(
        100,
        "Analysis complete",
        "Single-company memo and quantitative model stack are ready.",
    )
    return result_bundle


def run_single_analysis(
    *,
    ticker: str,
    requester_name: str,
    requester_email: str,
    openai_api_key: str,
    extraction_model: str,
    memo_model: str,
) -> None:
    status_placeholder = st.empty()
    progress_bar = st.progress(0, text="Fetching 10-K from SEC EDGAR...")
    cache = dict(st.session_state.get("analysis_cache", {}))
    cache_key = build_analysis_cache_key(
        ticker=ticker,
        extraction_model=extraction_model,
        memo_model=memo_model,
    )

    try:
        if cache_key in cache:
            update_analysis_status(
                status_placeholder=status_placeholder,
                progress_bar=progress_bar,
                percent_complete=20,
                title="Using cached analysis...",
                detail=f"Reusing the completed analysis for {ticker}.",
            )
            result_bundle = dict(cache[cache_key])
        else:
            result_bundle = run_company_analysis(
                ticker=ticker,
                requester_name=requester_name,
                requester_email=requester_email,
                openai_api_key=openai_api_key,
                extraction_model=extraction_model,
                memo_model=memo_model,
                status_placeholder=status_placeholder,
                progress_bar=progress_bar,
                progress_start=0,
                progress_end=76,
            )

        if result_bundle.get("quantitative_result") is None:
            result_bundle = augment_single_company_with_quantitative_models(
                result_bundle=result_bundle,
                requester_name=requester_name,
                requester_email=requester_email,
                openai_api_key=openai_api_key,
                extraction_model=extraction_model,
                status_placeholder=status_placeholder,
                progress_bar=progress_bar,
                progress_start=76,
                progress_end=100,
            )

        cache[cache_key] = result_bundle
        st.session_state["analysis_cache"] = cache
    except Exception:
        update_analysis_status(
            status_placeholder=status_placeholder,
            progress_bar=progress_bar,
            percent_complete=100,
            title="Analysis failed",
            detail="Review the error below, wait if rate limits were triggered, and retry.",
        )
        progress_bar.empty()
        status_placeholder.empty()
        raise

    store_single_company_state(ticker.upper(), result_bundle)
    progress_bar.empty()
    status_placeholder.empty()


def run_comparison_analysis(
    *,
    tickers: list[str],
    requester_name: str,
    requester_email: str,
    openai_api_key: str,
    extraction_model: str,
    memo_model: str,
) -> None:
    status_placeholder = st.empty()
    progress_bar = st.progress(0, text=f"Analyzing {tickers[0]} (1/{len(tickers)})...")
    cache = dict(st.session_state.get("analysis_cache", {}))
    results: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    company_progress_budget = 92

    try:
        for index, ticker in enumerate(tickers, start=1):
            overall_title = f"Analyzing {ticker} ({index}/{len(tickers)})..."
            progress_start = int(((index - 1) / len(tickers)) * company_progress_budget)
            progress_end = int((index / len(tickers)) * company_progress_budget)
            cache_key = build_analysis_cache_key(
                ticker=ticker,
                extraction_model=extraction_model,
                memo_model=memo_model,
            )

            if cache_key in cache:
                update_analysis_status(
                    status_placeholder=status_placeholder,
                    progress_bar=progress_bar,
                    percent_complete=progress_end,
                    title=overall_title,
                    detail="Using cached analysis from this session.",
                )
                results[ticker] = cache[cache_key]
                continue

            try:
                result_bundle = run_company_analysis(
                    ticker=ticker,
                    requester_name=requester_name,
                    requester_email=requester_email,
                    openai_api_key=openai_api_key,
                    extraction_model=extraction_model,
                    memo_model=memo_model,
                    status_placeholder=status_placeholder,
                    progress_bar=progress_bar,
                    progress_start=progress_start,
                    progress_end=progress_end,
                    overall_title=overall_title,
                )
            except Exception as exc:
                failures[ticker] = format_analysis_error(exc)
                if isinstance(exc, (ConfigurationError, OpenAIConfigurationError)):
                    for remaining_ticker in tickers[index:]:
                        failures.setdefault(remaining_ticker, failures[ticker])
                    break
                continue

            cache[cache_key] = result_bundle
            results[ticker] = result_bundle

        update_analysis_status(
            status_placeholder=status_placeholder,
            progress_bar=progress_bar,
            percent_complete=100,
            title="Building comparison...",
            detail="Compiling the comparison dashboard and peer export package.",
        )
        st.session_state["analysis_cache"] = cache
        store_comparison_state(
            requested_tickers=tickers,
            results=results,
            failures=failures,
        )
    finally:
        progress_bar.empty()
        status_placeholder.empty()


def main() -> None:
    st.set_page_config(page_title="AI Credit Analyst Agent", layout="wide")
    initialize_session_state()
    apply_report_styles()
    render_workspace_header()

    with st.sidebar:
        st.markdown("### How to use")
        st.markdown("Enter a ticker. Click Analyze. Review your credit memo.")

        with st.expander("Settings", expanded=False):
            requester_name = st.text_input(
                "SEC requester / organization",
                value=os.getenv("SEC_REQUESTER_NAME", ""),
                help="Used to build the SEC-required User-Agent header.",
            ).strip()
            requester_email = st.text_input(
                "SEC contact email",
                value=os.getenv("SEC_REQUESTER_EMAIL", ""),
                help="SEC fair access guidelines require a contact email in the User-Agent.",
            ).strip()
            openai_api_key = st.text_input(
                "OpenAI API key",
                value=os.getenv("OPENAI_API_KEY", ""),
                type="password",
            ).strip()
            preview_chars = int(
                st.number_input(
                    "Source preview characters",
                    min_value=1000,
                    max_value=50000,
                    value=DEFAULT_PREVIEW_CHARS,
                    step=1000,
                )
            )
            extraction_model_options = ["gpt-4o", "gpt-4o-mini"]
            extraction_model_default = os.getenv("OPENAI_EXTRACTION_MODEL", DEFAULT_OPENAI_MODEL)
            if extraction_model_default not in extraction_model_options:
                extraction_model_default = DEFAULT_OPENAI_MODEL
            extraction_model = st.selectbox(
                "Extraction model",
                options=extraction_model_options,
                index=extraction_model_options.index(extraction_model_default),
                help="Use gpt-4o-mini if extraction is hitting rate limits.",
            )
            memo_model = st.text_input(
                "Memo generation model",
                value=os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
                help="Default is gpt-4o for memo generation.",
            ).strip() or DEFAULT_OPENAI_MODEL

    input_col, button_col = st.columns([4.6, 1.2])
    with input_col:
        ticker_input = st.text_input(
            "Ticker",
            value="NFLX",
            max_chars=32,
            help="Enter one ticker or up to three tickers separated by commas.",
        ).strip()
    with button_col:
        st.markdown("<div style='height: 1.85rem;'></div>", unsafe_allow_html=True)
        analyze_clicked = st.button("Analyze", type="primary", use_container_width=True)

    analysis_failed = False
    if analyze_clicked:
        try:
            tickers = parse_ticker_input(ticker_input)
            if len(tickers) == 1:
                run_single_analysis(
                    ticker=tickers[0],
                    requester_name=requester_name,
                    requester_email=requester_email,
                    openai_api_key=openai_api_key,
                    extraction_model=extraction_model,
                    memo_model=memo_model,
                )
            else:
                run_comparison_analysis(
                    tickers=tickers,
                    requester_name=requester_name,
                    requester_email=requester_email,
                    openai_api_key=openai_api_key,
                    extraction_model=extraction_model,
                    memo_model=memo_model,
                )
        except ValueError as exc:
            st.error(str(exc))
            analysis_failed = True
        except Exception as exc:
            st.error(format_analysis_error(exc))
            analysis_failed = True

    filing_metadata = st.session_state.get("filing_metadata")
    filing_text = st.session_state.get("filing_text")
    financial_result = st.session_state.get("financial_result")
    memo_result = st.session_state.get("memo_result")
    quantitative_result = st.session_state.get("quantitative_result")
    active_mode = st.session_state.get("active_mode")
    active_tickers = st.session_state.get("active_tickers", ())
    active_results = st.session_state.get("active_results", {})
    active_failures = st.session_state.get("active_failures", {})

    if analysis_failed:
        render_footer()
        return

    if active_mode == "comparison" and active_tickers:
        if not active_results and not active_failures:
            render_empty_state()
            render_footer()
            return
        render_comparison_dashboard(
            requested_tickers=active_tickers,
            results=active_results,
            failures=active_failures,
        )
        render_footer()
        return

    if not all([filing_metadata, filing_text, financial_result, memo_result]):
        render_empty_state()
        render_footer()
        return

    render_report(
        filing_metadata=filing_metadata,
        filing_text=filing_text,
        financial_result=financial_result,
        memo_result=memo_result,
        quantitative_result=quantitative_result,
        preview_chars=preview_chars,
    )
    render_footer()


if __name__ == "__main__":
    main()
