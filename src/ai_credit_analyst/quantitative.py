from __future__ import annotations

import json
import re
from datetime import date
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    BadRequestError,
)

from ai_credit_analyst.exceptions import (
    ExtractionParseError,
    OpenAIConfigurationError,
    OpenAIExtractionError,
)
from ai_credit_analyst.extraction import (
    DEFAULT_OPENAI_MODEL,
    FIELD_LABELS,
    FIELD_ORDER,
    PRIMARY_FIELD_ORDER,
    RATIO_LABELS,
    build_filing_chunks,
    build_openai_client,
    call_openai_responses_with_retries,
    compute_credit_ratios,
)
from ai_credit_analyst.models import (
    AltmanComponent,
    AltmanZScoreResult,
    CashFlowAdequacyResult,
    DebtMaturityProfileResult,
    DebtMaturityTranche,
    FilingChunk,
    FinancialExtractionResult,
    MaturityWallWarning,
    PiotroskiCriterion,
    PiotroskiFScoreResult,
    QuantitativeModelResult,
    TrendAnalysisRow,
)

PRIOR_YEAR_EXTRACTION_MODEL = "gpt-4o-mini"
DEBT_MATURITY_MODEL = DEFAULT_OPENAI_MODEL
DEBT_MATURITY_MAX_OUTPUT_TOKENS = 2500
DEBT_MATURITY_CHUNK_WINDOW_CHARS = 16000

QUANT_TREND_FIELDS: list[tuple[str, bool, bool]] = [
    ("revenue", False, True),
    ("net_income", False, True),
    ("total_debt", False, False),
    ("cash_and_cash_equivalents", False, True),
    ("operating_cash_flow", False, True),
    ("debt_to_equity", True, False),
    ("interest_coverage", True, True),
    ("current_ratio", True, True),
    ("net_debt_to_ebitda", True, False),
]

RATIO_HISTORY_FIELDS = [
    "debt_to_equity",
    "interest_coverage",
    "current_ratio",
    "net_debt_to_ebitda",
]

DEBT_MATURITY_SYSTEM_PROMPT = """You are a credit analyst. Extract every debt tranche or maturity mentioned in this filing.

Work only from the supplied debt excerpt.
Return a JSON object with one key:
- tranches: array of objects with amount_millions, maturity_year, interest_rate, and description

Rules:
1. amount_millions must be a numeric amount in USD millions.
2. If the filing states debt amounts in billions or thousands, convert to millions.
3. maturity_year must be a four-digit year.
4. description should identify the tranche, instrument, or debt bucket.
5. If no specific maturity schedule is found, return {"tranches": []}.
6. ONLY extract debt tranches that are explicitly listed in a maturity schedule table or a numbered list of notes or bonds with specific amounts and due dates.
7. Do NOT infer or fabricate tranches from general descriptions of credit facilities, revolving lines, delayed draw facilities, bridge loans, or liquidity backstops.
8. Do not invent maturities or interpolate missing tranches.

Return JSON only."""

DEBT_MATURITY_USER_PROMPT_TEMPLATE = """Company: {company_name}
Ticker: {ticker}
Filing date: {filing_date}
Reported total debt (USD millions): {total_debt_millions}

Debt excerpt:
<<<EXCERPT
{excerpt}
EXCERPT>>>"""

DEBT_MATURITY_JSON_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "name": "debt_maturity_schedule",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "tranches": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "amount_millions": {"type": "number"},
                        "maturity_year": {"type": "integer"},
                        "interest_rate": {
                            "anyOf": [
                                {"type": "string"},
                                {"type": "null"},
                            ]
                        },
                        "description": {"type": "string"},
                    },
                    "required": [
                        "amount_millions",
                        "maturity_year",
                        "interest_rate",
                        "description",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["tranches"],
        "additionalProperties": False,
    },
}

TREND_DATA_QUALITY_NOTE = "⚠ Data quality - verify prior year extraction"
TREND_PERCENT_CHANGE_ALERT_THRESHOLD = 5.0
PRIOR_YEAR_VALUE_RATIO_THRESHOLD = 0.001
PRIOR_YEAR_RESCALING_FACTOR = 1_000


def _safe_divide(numerator: float | int | None, denominator: float | int | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def _coalesce_numeric(*values: float | int | None) -> float | None:
    for value in values:
        if value is not None:
            return float(value)
    return None


def _normalize_market_cap_value(raw_value: Any) -> float | None:
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        return None
    if raw_value <= 0:
        return None
    return float(raw_value)


def fetch_market_cap(ticker: str) -> tuple[float | None, str | None]:
    try:
        import yfinance as yf  # type: ignore
    except ImportError:
        return None, "yfinance is not installed. Book value will be used as a proxy for market capitalization."

    try:
        security = yf.Ticker(ticker)
        fast_info = getattr(security, "fast_info", None)
        if fast_info is not None:
            for key in ("market_cap", "marketCap"):
                try:
                    market_cap = _normalize_market_cap_value(fast_info.get(key))
                except Exception:
                    market_cap = None
                if market_cap is not None:
                    return market_cap, None

        info = getattr(security, "info", None)
        if isinstance(info, dict):
            for key in ("marketCap", "market_cap"):
                market_cap = _normalize_market_cap_value(info.get(key))
                if market_cap is not None:
                    return market_cap, None
    except Exception as exc:
        return None, f"yfinance market cap lookup failed: {exc}. Book value will be used as a proxy for market capitalization."

    return None, "yfinance did not return market capitalization. Book value will be used as a proxy for market capitalization."


def resolve_retained_earnings(
    financial_result: FinancialExtractionResult,
) -> tuple[float | None, str | None]:
    metrics = financial_result.metrics
    retained_earnings = metrics["retained_earnings"].value
    if retained_earnings is not None:
        return float(retained_earnings), None

    total_equity = metrics["total_shareholders_equity"].value
    common_stock = metrics["common_stock"].value
    additional_paid_in_capital = metrics["additional_paid_in_capital"].value
    if total_equity is not None and common_stock is not None and additional_paid_in_capital is not None:
        derived_value = float(total_equity) - float(common_stock) - float(additional_paid_in_capital)
        return (
            derived_value,
            "Retained earnings were not explicitly extracted; derived as Total Shareholders' Equity minus Common Stock minus Additional Paid-In Capital.",
        )

    if total_equity is not None:
        return (
            float(total_equity) * 0.7,
            "Retained earnings were not explicitly available; 70% of total shareholders' equity was used as a rough proxy.",
        )

    return None, "Retained earnings could not be resolved from the filing."


def compute_altman_z_score(
    *,
    financial_result: FinancialExtractionResult,
    ticker: str,
) -> AltmanZScoreResult:
    metrics = financial_result.metrics
    working_capital = None
    if metrics["total_current_assets"].value is not None and metrics["total_current_liabilities"].value is not None:
        working_capital = float(metrics["total_current_assets"].value) - float(metrics["total_current_liabilities"].value)

    retained_earnings, retained_earnings_note = resolve_retained_earnings(financial_result)

    market_cap, market_cap_note = fetch_market_cap(ticker)
    if market_cap is None:
        equity_proxy = metrics["total_shareholders_equity"].value
        market_cap = None if equity_proxy is None else float(equity_proxy)
        fallback_note = "Book value used as proxy for market capitalization."
        market_cap_note = fallback_note if market_cap_note is None else f"{market_cap_note} {fallback_note}"

    total_assets = _coalesce_numeric(metrics["total_assets"].value)
    total_liabilities = _coalesce_numeric(metrics["total_liabilities"].value)
    ebit = _coalesce_numeric(metrics["ebit"].value)
    revenue = _coalesce_numeric(metrics["revenue"].value)

    components = (
        AltmanComponent(
            label="1.2 x Working Capital / Total Assets",
            raw_value=_safe_divide(working_capital, total_assets),
            weighted_contribution=None,
            description="Liquidity relative to the asset base.",
        ),
        AltmanComponent(
            label="1.4 x Retained Earnings / Total Assets",
            raw_value=_safe_divide(retained_earnings, total_assets),
            weighted_contribution=None,
            description="Cumulative profitability and balance-sheet seasoning.",
        ),
        AltmanComponent(
            label="3.3 x EBIT / Total Assets",
            raw_value=_safe_divide(ebit, total_assets),
            weighted_contribution=None,
            description="Operating earnings power relative to assets.",
        ),
        AltmanComponent(
            label="0.6 x Market Cap / Total Liabilities",
            raw_value=_safe_divide(market_cap, total_liabilities),
            weighted_contribution=None,
            description="Equity cushion against total obligations.",
        ),
        AltmanComponent(
            label="1.0 x Revenue / Total Assets",
            raw_value=_safe_divide(revenue, total_assets),
            weighted_contribution=None,
            description="Asset turnover and commercial productivity.",
        ),
    )
    weights = (1.2, 1.4, 3.3, 0.6, 1.0)
    weighted_components: list[AltmanComponent] = []
    contributions: list[float] = []
    for component, weight in zip(components, weights):
        weighted_value = None if component.raw_value is None else component.raw_value * weight
        weighted_components.append(
            AltmanComponent(
                label=component.label,
                raw_value=component.raw_value,
                weighted_contribution=weighted_value,
                description=component.description,
            )
        )
        if weighted_value is not None:
            contributions.append(weighted_value)

    score = sum(contributions) if len(contributions) == len(weighted_components) else None
    if score is None:
        zone = "Unavailable"
    elif score > 2.99:
        zone = "Safe Zone"
    elif score >= 1.81:
        zone = "Gray Zone"
    else:
        zone = "Distress Zone"

    note_parts = [note for note in (market_cap_note, retained_earnings_note) if note]
    if score is None:
        note_parts.append("Altman Z-Score could not be fully computed because one or more required inputs are missing.")

    return AltmanZScoreResult(
        score=score,
        zone=zone,
        note=" ".join(note_parts) if note_parts else None,
        components=tuple(weighted_components),
    )


def _compute_margin(
    metrics: dict[str, Any],
    *,
    numerator_field: str,
) -> float | None:
    return _safe_divide(metrics[numerator_field].value, metrics["revenue"].value)


def compute_piotroski_f_score(
    *,
    current_result: FinancialExtractionResult,
    prior_result: FinancialExtractionResult | None,
) -> PiotroskiFScoreResult:
    if prior_result is None:
        return PiotroskiFScoreResult(
            score=None,
            interpretation="Unavailable",
            note="Prior year filing not available - multi-year models require 2+ years of data.",
            criteria=tuple(),
        )

    current_metrics = current_result.metrics
    prior_metrics = prior_result.metrics
    current_ratios = current_result.ratios
    prior_ratios = prior_result.ratios

    current_roa = _safe_divide(current_metrics["net_income"].value, current_metrics["total_assets"].value)
    prior_roa = _safe_divide(prior_metrics["net_income"].value, prior_metrics["total_assets"].value)

    current_margin = _compute_margin(current_metrics, numerator_field="gross_profit")
    prior_margin = _compute_margin(prior_metrics, numerator_field="gross_profit")
    margin_note = "Gross margin used."
    if current_margin is None or prior_margin is None:
        current_margin = _compute_margin(current_metrics, numerator_field="ebit")
        prior_margin = _compute_margin(prior_metrics, numerator_field="ebit")
        margin_note = "Gross margin unavailable; operating margin used."

    current_asset_turnover = _safe_divide(current_metrics["revenue"].value, current_metrics["total_assets"].value)
    prior_asset_turnover = _safe_divide(prior_metrics["revenue"].value, prior_metrics["total_assets"].value)

    def build_binary_criterion(
        *,
        label: str,
        passed: bool | None,
        pass_note: str,
        fail_note: str,
        missing_note: str,
    ) -> PiotroskiCriterion:
        if passed is None:
            return PiotroskiCriterion(label=label, passed=None, note=missing_note)
        return PiotroskiCriterion(label=label, passed=passed, note=pass_note if passed else fail_note)

    criteria = (
        PiotroskiCriterion(
            label="Positive Net Income",
            passed=bool((current_metrics["net_income"].value or 0) > 0),
            note="Net income is positive for the current year." if (current_metrics["net_income"].value or 0) > 0 else "Net income is not positive for the current year.",
        ),
        PiotroskiCriterion(
            label="Positive Operating Cash Flow",
            passed=bool((current_metrics["operating_cash_flow"].value or 0) > 0),
            note="Operating cash flow is positive for the current year." if (current_metrics["operating_cash_flow"].value or 0) > 0 else "Operating cash flow is not positive for the current year.",
        ),
        build_binary_criterion(
            label="ROA Improved",
            passed=None if current_roa is None or prior_roa is None else current_roa > prior_roa,
            pass_note="Return on assets improved year over year.",
            fail_note="Return on assets did not improve year over year.",
            missing_note="Insufficient data to evaluate return on assets year over year.",
        ),
        PiotroskiCriterion(
            label="Cash Flow Exceeds Net Income",
            passed=(
                current_metrics["operating_cash_flow"].value is not None
                and current_metrics["net_income"].value is not None
                and float(current_metrics["operating_cash_flow"].value) > float(current_metrics["net_income"].value)
            ),
            note="Operating cash flow exceeds net income." if (
                current_metrics["operating_cash_flow"].value is not None
                and current_metrics["net_income"].value is not None
                and float(current_metrics["operating_cash_flow"].value) > float(current_metrics["net_income"].value)
            ) else "Operating cash flow does not exceed net income.",
        ),
        build_binary_criterion(
            label="Lower Debt-to-Equity",
            passed=(
                None
                if current_ratios["debt_to_equity"].value is None or prior_ratios["debt_to_equity"].value is None
                else float(current_ratios["debt_to_equity"].value) < float(prior_ratios["debt_to_equity"].value)
            ),
            pass_note="Debt-to-equity improved year over year.",
            fail_note="Debt-to-equity did not improve year over year.",
            missing_note="Insufficient data to evaluate debt-to-equity year over year.",
        ),
        build_binary_criterion(
            label="Higher Current Ratio",
            passed=(
                None
                if current_ratios["current_ratio"].value is None or prior_ratios["current_ratio"].value is None
                else float(current_ratios["current_ratio"].value) > float(prior_ratios["current_ratio"].value)
            ),
            pass_note="Current ratio improved year over year.",
            fail_note="Current ratio did not improve year over year.",
            missing_note="Insufficient data to evaluate current ratio year over year.",
        ),
        PiotroskiCriterion(
            label="No Share Dilution",
            passed=True,
            note="Assumed no dilution because shares outstanding were not extracted.",
        ),
        build_binary_criterion(
            label="Higher Margin",
            passed=None if current_margin is None or prior_margin is None else current_margin > prior_margin,
            pass_note=f"{margin_note} Margin improved year over year.",
            fail_note=f"{margin_note} Margin did not improve year over year.",
            missing_note=f"{margin_note} Insufficient data to evaluate margin year over year.",
        ),
        build_binary_criterion(
            label="Higher Asset Turnover",
            passed=None if current_asset_turnover is None or prior_asset_turnover is None else current_asset_turnover > prior_asset_turnover,
            pass_note="Asset turnover improved year over year.",
            fail_note="Asset turnover did not improve year over year.",
            missing_note="Insufficient data to evaluate asset turnover year over year.",
        ),
    )

    evaluated_count = sum(1 for criterion in criteria if criterion.passed is not None)
    score = sum(1 for criterion in criteria if criterion.passed is True)
    missing_count = len(criteria) - evaluated_count

    if evaluated_count == 0:
        interpretation = "Unavailable"
    elif score / evaluated_count >= 0.8:
        interpretation = "Strong"
    elif score / evaluated_count >= 0.5:
        interpretation = "Moderate"
    else:
        interpretation = "Weak"

    return PiotroskiFScoreResult(
        score=score,
        interpretation=interpretation,
        note=(
            f"{score}/{evaluated_count} evaluated criteria passed ({missing_count} criteria lacked sufficient data)."
            if missing_count > 0
            else f"{score}/{len(criteria)} criteria passed."
        ),
        criteria=criteria,
        evaluated_count=evaluated_count,
        missing_count=missing_count,
    )


def compute_cash_flow_adequacy(
    financial_result: FinancialExtractionResult,
) -> CashFlowAdequacyResult:
    metrics = financial_result.metrics
    near_term_debt = _coalesce_numeric(metrics["current_portion_of_long_term_debt"].value)
    note: str | None = None
    if near_term_debt is None and metrics["total_debt"].value is not None:
        near_term_debt = float(metrics["total_debt"].value) * 0.1
        note = "Current portion of long-term debt was unavailable; 10% of total debt was used as a proxy for near-term debt obligations."

    denominator_inputs = (
        near_term_debt,
        _coalesce_numeric(metrics["interest_expense"].value),
        _coalesce_numeric(metrics["capital_expenditures"].value),
    )
    if any(value is None for value in denominator_inputs):
        return CashFlowAdequacyResult(
            ratio=None,
            assessment="Unavailable",
            note="Cash flow adequacy could not be computed because near-term debt, interest expense, or capital expenditures are missing.",
            near_term_debt_used=near_term_debt,
        )

    denominator = float(denominator_inputs[0]) + float(denominator_inputs[1]) + float(denominator_inputs[2])
    ratio = _safe_divide(financial_result.metrics["operating_cash_flow"].value, denominator)
    if ratio is None:
        return CashFlowAdequacyResult(
            ratio=None,
            assessment="Unavailable",
            note="Operating cash flow is missing or denominator is zero.",
            near_term_debt_used=near_term_debt,
        )

    if ratio > 1.5:
        assessment = "Strong - self-funding with margin"
    elif ratio >= 1.0:
        assessment = "Adequate - obligations covered"
    else:
        assessment = "Weak - external financing likely required"

    return CashFlowAdequacyResult(
        ratio=ratio,
        assessment=assessment,
        note=note,
        near_term_debt_used=near_term_debt,
    )


def _build_trend_row(
    *,
    label: str,
    current_value: float | None,
    prior_value: float | None,
    prefers_higher: bool,
    is_ratio: bool,
    note: str | None = None,
) -> TrendAnalysisRow:
    absolute_change = None
    percent_change = None

    improving: bool | None = None
    prior_data_reliable = prior_value not in (None, 0) and note is None
    if current_value is not None and prior_value not in (None, 0) and note is None:
        absolute_change = current_value - prior_value
        percent_change = absolute_change / abs(prior_value)
        improving = current_value > prior_value if prefers_higher else current_value < prior_value
        if is_ratio and abs(percent_change) > TREND_PERCENT_CHANGE_ALERT_THRESHOLD:
            prior_value = None
            percent_change = None
            improving = None
            note = TREND_DATA_QUALITY_NOTE
            prior_data_reliable = False

    return TrendAnalysisRow(
        label=label,
        current_value=current_value,
        prior_value=prior_value,
        absolute_change=absolute_change,
        percent_change=percent_change,
        improving=improving,
        prefers_higher=prefers_higher,
        is_ratio=is_ratio,
        note=note,
        prior_data_reliable=prior_data_reliable,
    )


def reconcile_prior_year_financial_result(
    *,
    current_result: FinancialExtractionResult,
    prior_result: FinancialExtractionResult | None,
) -> FinancialExtractionResult | None:
    if prior_result is None:
        return None

    adjusted_metrics: dict[str, Any] = {}
    scaled_fields: list[str] = []
    suppressed_fields: list[str] = []

    for field_name in FIELD_ORDER:
        prior_metric = prior_result.metrics[field_name]
        current_metric = current_result.metrics[field_name]
        adjusted_value = None if prior_metric.value is None else float(prior_metric.value)

        if adjusted_value is not None and current_metric.value not in (None, 0):
            current_abs = abs(float(current_metric.value))
            if adjusted_value == 0 and current_abs > 0:
                adjusted_value = None
                suppressed_fields.append(FIELD_LABELS[field_name])
            elif abs(adjusted_value) < current_abs * PRIOR_YEAR_VALUE_RATIO_THRESHOLD:
                candidate_value = adjusted_value * PRIOR_YEAR_RESCALING_FACTOR
                if abs(candidate_value) >= current_abs * PRIOR_YEAR_VALUE_RATIO_THRESHOLD:
                    adjusted_value = candidate_value
                    scaled_fields.append(FIELD_LABELS[field_name])
                else:
                    adjusted_value = None
                    suppressed_fields.append(FIELD_LABELS[field_name])

        adjusted_metrics[field_name] = prior_metric.__class__(
            value=adjusted_value,
            source_quote=prior_metric.source_quote,
            source_chunk_label=prior_metric.source_chunk_label,
        )

    adjusted_ratios = compute_credit_ratios(adjusted_metrics)
    for ratio_name in RATIO_HISTORY_FIELDS:
        current_ratio = current_result.ratios[ratio_name].value
        prior_ratio = adjusted_ratios[ratio_name].value
        if current_ratio not in (None, 0) and prior_ratio == 0:
            adjusted_ratios[ratio_name] = adjusted_ratios[ratio_name].__class__(
                value=None,
                note=TREND_DATA_QUALITY_NOTE,
            )
            continue
        if current_ratio is None or prior_ratio in (None, 0):
            continue
        percent_change = abs((float(current_ratio) - float(prior_ratio)) / abs(float(prior_ratio)))
        if percent_change > TREND_PERCENT_CHANGE_ALERT_THRESHOLD:
            adjusted_ratios[ratio_name] = adjusted_ratios[ratio_name].__class__(
                value=None,
                note=TREND_DATA_QUALITY_NOTE,
            )

    normalization_fragments: list[str] = [prior_result.normalization_note.rstrip(".")]
    if scaled_fields:
        normalization_fragments.append(
            "Prior-year scale reconciliation multiplied likely 1,000x unit mismatches for: "
            + ", ".join(sorted(set(scaled_fields)))
        )
    if suppressed_fields:
        normalization_fragments.append(
            "Unreliable prior-year values were suppressed for: "
            + ", ".join(sorted(set(suppressed_fields)))
        )

    return FinancialExtractionResult(
        metrics=adjusted_metrics,
        ratios=adjusted_ratios,
        chunk_strategy=prior_result.chunk_strategy,
        chunk_count=prior_result.chunk_count,
        prepared_text_length=prior_result.prepared_text_length,
        used_primary_document=prior_result.used_primary_document,
        chunks=prior_result.chunks,
        detected_statement_unit_label=prior_result.detected_statement_unit_label,
        detected_statement_unit_multiplier=prior_result.detected_statement_unit_multiplier,
        normalization_note=". ".join(fragment for fragment in normalization_fragments if fragment).strip() + ".",
    )


def build_trend_analysis(
    *,
    current_result: FinancialExtractionResult,
    prior_result: FinancialExtractionResult | None,
) -> tuple[tuple[TrendAnalysisRow, ...], str | None, dict[str, tuple[float | None, float | None]]]:
    if prior_result is None:
        return (
            tuple(),
            "Prior year filing not available - multi-year models require 2+ years of data.",
            {ratio_name: (current_result.ratios[ratio_name].value, None) for ratio_name in RATIO_HISTORY_FIELDS},
        )

    rows: list[TrendAnalysisRow] = []
    for field_name, is_ratio, prefers_higher in QUANT_TREND_FIELDS:
        if is_ratio:
            current_value = current_result.ratios[field_name].value
            prior_value = prior_result.ratios[field_name].value
            label = RATIO_LABELS[field_name]
            note = prior_result.ratios[field_name].note if prior_value is None else None
        else:
            current_value = _coalesce_numeric(current_result.metrics[field_name].value)
            prior_value = _coalesce_numeric(prior_result.metrics[field_name].value)
            label = FIELD_LABELS[field_name]
            note = None
        rows.append(
            _build_trend_row(
                label=label,
                current_value=current_value,
                prior_value=prior_value,
                prefers_higher=prefers_higher,
                is_ratio=is_ratio,
                note=note,
            )
        )

    row_lookup = {row.label: row for row in rows}
    ratio_history = {}
    for ratio_name in RATIO_HISTORY_FIELDS:
        row = row_lookup[RATIO_LABELS[ratio_name]]
        ratio_history[ratio_name] = (
            current_result.ratios[ratio_name].value,
            None if not row.prior_data_reliable else prior_result.ratios[ratio_name].value,
        )
    return tuple(rows), None, ratio_history


def _find_debt_maturity_chunks(
    raw_filing_text: str,
    current_result: FinancialExtractionResult,
) -> tuple[FilingChunk, ...]:
    existing_chunks = [
        chunk
        for chunk in current_result.chunks
        if chunk.label in {"debt_note", "notes_to_financials"}
    ]
    if existing_chunks:
        return tuple(existing_chunks)

    chunks, _, _, _ = build_filing_chunks(raw_filing_text)
    candidate_chunks = [
        chunk
        for chunk in chunks
        if chunk.label in {"debt_note", "notes_to_financials"}
    ]
    if candidate_chunks:
        return tuple(candidate_chunks)

    upper_text = raw_filing_text.upper()
    for pattern in (
        r"LONG[\s-]TERM\s+DEBT",
        r"DEBT\s+MATURIT",
        r"FUTURE\s+MATURITIES",
        r"SENIOR\s+NOTES",
        r"BORROWINGS",
    ):
        match = re.search(pattern, upper_text)
        if not match:
            continue
        start_char = max(0, match.start() - 800)
        end_char = min(len(raw_filing_text), start_char + DEBT_MATURITY_CHUNK_WINDOW_CHARS)
        return (
            FilingChunk(
                label="debt_maturity_search",
                strategy="quantitative_models_search",
                start_char=start_char,
                end_char=end_char,
                text=raw_filing_text[start_char:end_char].strip(),
            ),
        )

    return tuple()


def extract_debt_maturity_schedule(
    *,
    raw_filing_text: str,
    company_name: str,
    ticker: str,
    filing_date: str,
    financial_result: FinancialExtractionResult,
    api_key: str | None = None,
    model: str = DEFAULT_OPENAI_MODEL,
) -> DebtMaturityProfileResult:
    maturity_chunks = _find_debt_maturity_chunks(raw_filing_text, financial_result)
    total_debt_millions = _safe_divide(financial_result.metrics["total_debt"].value, 1_000_000)
    if not maturity_chunks:
        return DebtMaturityProfileResult(
            tranches=tuple(),
            weighted_average_maturity=None,
            warnings=tuple(),
            note="No specific maturity schedule identified in filing.",
        )

    excerpt = "\n\n".join(chunk.text for chunk in maturity_chunks[:2]).strip()
    if not excerpt:
        return DebtMaturityProfileResult(
            tranches=tuple(),
            weighted_average_maturity=None,
            warnings=tuple(),
            note="No specific maturity schedule identified in filing.",
        )

    client = build_openai_client(api_key=api_key)
    try:
        response = call_openai_responses_with_retries(
            lambda: client.responses.create(
                model=model,
                instructions=DEBT_MATURITY_SYSTEM_PROMPT,
                input=DEBT_MATURITY_USER_PROMPT_TEMPLATE.format(
                    company_name=company_name,
                    ticker=ticker,
                    filing_date=filing_date,
                    total_debt_millions="N/A" if total_debt_millions is None else f"{total_debt_millions:.1f}",
                    excerpt=excerpt,
                ),
                temperature=0,
                max_output_tokens=DEBT_MATURITY_MAX_OUTPUT_TOKENS,
                store=False,
                text={"format": DEBT_MATURITY_JSON_SCHEMA},
            ),
            rate_limit_exception_cls=OpenAIExtractionError,
        )
    except AuthenticationError as exc:
        raise OpenAIConfigurationError(
            "OpenAI rejected the API key during debt maturity extraction. Check OPENAI_API_KEY and retry."
        ) from exc
    except BadRequestError as exc:
        raise OpenAIExtractionError(
            f"OpenAI rejected the debt maturity extraction request: {exc}"
        ) from exc
    except APIConnectionError as exc:
        raise OpenAIExtractionError(
            f"OpenAI connection failed during debt maturity extraction: {exc}"
        ) from exc
    except APIStatusError as exc:
        raise OpenAIExtractionError(
            f"OpenAI returned HTTP {exc.status_code} during debt maturity extraction."
        ) from exc
    except Exception as exc:
        raise OpenAIExtractionError(
            f"Unexpected OpenAI error during debt maturity extraction: {exc}"
        ) from exc

    output_text = getattr(response, "output_text", None)
    if not output_text:
        raise ExtractionParseError("OpenAI returned no structured output for debt maturity extraction.")

    try:
        payload = json.loads(output_text)
    except json.JSONDecodeError as exc:
        raise ExtractionParseError("OpenAI returned invalid JSON for debt maturity extraction.") from exc

    if not isinstance(payload, dict):
        raise ExtractionParseError("Debt maturity extraction output was not a JSON object.")

    raw_tranches = payload.get("tranches")
    if not isinstance(raw_tranches, list):
        raise ExtractionParseError("Debt maturity extraction output did not include a tranches array.")

    tranches: list[DebtMaturityTranche] = []
    for item in raw_tranches:
        if not isinstance(item, dict):
            raise ExtractionParseError("Debt maturity extraction output contained a non-object tranche.")
        amount_millions = item.get("amount_millions")
        maturity_year = item.get("maturity_year")
        interest_rate = item.get("interest_rate")
        description = item.get("description")
        if isinstance(amount_millions, bool) or not isinstance(amount_millions, (int, float)):
            raise ExtractionParseError("Debt maturity tranche amount_millions must be numeric.")
        if isinstance(maturity_year, bool) or not isinstance(maturity_year, int):
            raise ExtractionParseError("Debt maturity tranche maturity_year must be an integer.")
        if interest_rate is not None and not isinstance(interest_rate, str):
            raise ExtractionParseError("Debt maturity tranche interest_rate must be a string or null.")
        if not isinstance(description, str) or not description.strip():
            raise ExtractionParseError("Debt maturity tranche description must be a non-empty string.")
        tranches.append(
            DebtMaturityTranche(
                amount_millions=float(amount_millions),
                maturity_year=maturity_year,
                interest_rate=interest_rate.strip() if isinstance(interest_rate, str) and interest_rate.strip() else None,
                description=description.strip(),
            )
        )

    if not tranches:
        return DebtMaturityProfileResult(
            tranches=tuple(),
            weighted_average_maturity=None,
            warnings=tuple(),
            note="No specific maturity schedule identified in filing.",
        )

    validation_note_parts: list[str] = []
    if total_debt_millions and total_debt_millions > 0:
        filtered_tranches = [
            tranche for tranche in tranches if tranche.amount_millions <= total_debt_millions
        ]
        excluded_count = len(tranches) - len(filtered_tranches)
        if excluded_count > 0:
            validation_note_parts.append(
                f"Excluded {excluded_count} tranche(s) that exceeded reported total debt."
            )
        tranches = filtered_tranches
        if not tranches:
            return DebtMaturityProfileResult(
                tranches=tuple(),
                weighted_average_maturity=None,
                warnings=tuple(),
                note="Maturity schedule could not be reliably extracted from this filing.",
            )
        if sum(tranche.amount_millions for tranche in tranches) > total_debt_millions * 1.5:
            return DebtMaturityProfileResult(
                tranches=tuple(),
                weighted_average_maturity=None,
                warnings=tuple(),
                note="Maturity schedule could not be reliably extracted from this filing.",
            )

    filing_year = date.fromisoformat(filing_date).year
    total_maturity_amount = sum(tranche.amount_millions for tranche in tranches)
    weighted_average_maturity = None
    historical_only = total_maturity_amount > 0 and all(tranche.maturity_year < filing_year for tranche in tranches)
    if total_maturity_amount > 0:
        if historical_only:
            validation_note_parts.append("Unable to compute - all extracted maturities are historical.")
        else:
            weighted_average_maturity = sum(
                max(tranche.maturity_year - filing_year, 0) * tranche.amount_millions
                for tranche in tranches
            ) / total_maturity_amount

    year_totals: dict[int, float] = {}
    for tranche in tranches:
        year_totals[tranche.maturity_year] = year_totals.get(tranche.maturity_year, 0.0) + tranche.amount_millions

    total_debt_millions = total_debt_millions or total_maturity_amount
    warnings: list[MaturityWallWarning] = []
    if total_debt_millions and total_debt_millions > 0 and not historical_only:
        for maturity_year, year_amount in sorted(year_totals.items()):
            percentage = year_amount / total_debt_millions
            if percentage > 0.20:
                warnings.append(
                    MaturityWallWarning(
                        maturity_year=maturity_year,
                        percentage_of_total_debt=percentage,
                    )
                )

    return DebtMaturityProfileResult(
        tranches=tuple(sorted(tranches, key=lambda tranche: (tranche.maturity_year, tranche.description))),
        weighted_average_maturity=weighted_average_maturity,
        warnings=tuple(warnings),
        note=" ".join(validation_note_parts) if validation_note_parts else None,
    )


def build_quantitative_model_result(
    *,
    company_name: str,
    ticker: str,
    filing_date: str,
    raw_filing_text: str,
    current_result: FinancialExtractionResult,
    prior_result: FinancialExtractionResult | None = None,
    prior_filing_date: str | None = None,
    api_key: str | None = None,
    debt_model: str = DEFAULT_OPENAI_MODEL,
) -> QuantitativeModelResult:
    prior_result = reconcile_prior_year_financial_result(
        current_result=current_result,
        prior_result=prior_result,
    )
    altman_result = compute_altman_z_score(
        financial_result=current_result,
        ticker=ticker,
    )
    piotroski_result = compute_piotroski_f_score(
        current_result=current_result,
        prior_result=prior_result,
    )
    cash_flow_adequacy = compute_cash_flow_adequacy(current_result)
    trend_rows, trend_note, ratio_history = build_trend_analysis(
        current_result=current_result,
        prior_result=prior_result,
    )

    try:
        debt_profile = extract_debt_maturity_schedule(
            raw_filing_text=raw_filing_text,
            company_name=company_name,
            ticker=ticker,
            filing_date=filing_date,
            financial_result=current_result,
            api_key=api_key,
            model=debt_model,
        )
    except Exception as exc:
        debt_profile = DebtMaturityProfileResult(
            tranches=tuple(),
            weighted_average_maturity=None,
            warnings=tuple(),
            note=f"Debt maturity extraction failed: {exc}",
        )

    return QuantitativeModelResult(
        altman_z_score=altman_result,
        piotroski_f_score=piotroski_result,
        cash_flow_adequacy=cash_flow_adequacy,
        trend_analysis=trend_rows,
        trend_analysis_note=trend_note,
        ratio_history=ratio_history,
        debt_maturity_profile=debt_profile,
        prior_year_available=prior_result is not None,
        prior_year_filing_date=prior_filing_date,
    )
