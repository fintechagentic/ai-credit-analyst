from __future__ import annotations

import html
import json
import os
import re
import time
from typing import Any, Callable

from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)

from ai_credit_analyst.exceptions import (
    ExtractionParseError,
    OpenAIConfigurationError,
    OpenAIExtractionError,
)
from ai_credit_analyst.models import (
    FilingChunk,
    FinancialExtractionResult,
    MetricValue,
    RatioValue,
)

OPENAI_RESPONSES_API_URL = "https://api.openai.com/v1/responses"
DEFAULT_OPENAI_MODEL = "gpt-4o"
DEFAULT_CHUNK_WINDOW_CHARS = 22000
SECTION_CONTEXT_PREFIX_CHARS = 1000
FALLBACK_CHUNK_SIZE_CHARS = 16000
FALLBACK_CHUNK_OVERLAP_CHARS = 2500
MIN_DIGIT_COUNT_NEAR_ANCHOR = 10
MAX_OUTPUT_TOKENS = 2500
OPENAI_RATE_LIMIT_BASE_DELAY_SECONDS = 10
OPENAI_RATE_LIMIT_MAX_RETRIES = 3
EXTRACTION_INTER_CHUNK_DELAY_SECONDS = 3

PRIMARY_FIELD_ORDER = [
    "revenue",
    "net_income",
    "total_assets",
    "total_current_assets",
    "total_current_liabilities",
    "total_liabilities",
    "total_shareholders_equity",
    "total_debt",
    "cash_and_cash_equivalents",
    "interest_expense",
    "ebit",
    "ebitda",
    "operating_cash_flow",
    "capital_expenditures",
]

SUPPLEMENTAL_FIELD_ORDER = [
    "retained_earnings",
    "common_stock",
    "additional_paid_in_capital",
    "gross_profit",
    "current_portion_of_long_term_debt",
]

FIELD_ORDER = PRIMARY_FIELD_ORDER + SUPPLEMENTAL_FIELD_ORDER

RATIO_ORDER = [
    "debt_to_equity",
    "interest_coverage",
    "current_ratio",
    "net_debt_to_ebitda",
    "free_cash_flow_yield",
]

FIELD_LABELS = {
    "revenue": "Revenue",
    "net_income": "Net Income",
    "total_assets": "Total Assets",
    "total_current_assets": "Total Current Assets",
    "total_current_liabilities": "Total Current Liabilities",
    "total_liabilities": "Total Liabilities",
    "total_shareholders_equity": "Total Shareholders' Equity",
    "total_debt": "Total Debt",
    "cash_and_cash_equivalents": "Cash and Cash Equivalents",
    "interest_expense": "Interest Expense",
    "ebit": "EBIT",
    "ebitda": "EBITDA",
    "operating_cash_flow": "Operating Cash Flow",
    "capital_expenditures": "Capital Expenditures",
    "retained_earnings": "Retained Earnings",
    "common_stock": "Common Stock",
    "additional_paid_in_capital": "Additional Paid-In Capital",
    "gross_profit": "Gross Profit",
    "current_portion_of_long_term_debt": "Current Portion of Long-Term Debt",
}

RATIO_LABELS = {
    "debt_to_equity": "Debt-to-Equity",
    "interest_coverage": "Interest Coverage",
    "current_ratio": "Current Ratio",
    "net_debt_to_ebitda": "Net Debt / EBITDA",
    "free_cash_flow_yield": "Free Cash Flow Yield",
}

ANCHOR_PATTERNS: list[tuple[str, tuple[str, ...], int]] = [
    (
        "balance_sheet",
        (
            r"CONSOLIDATED\s+BALANCE\s+SHEETS",
            r"CONSOLIDATED\s+STATEMENTS\s+OF\s+FINANCIAL\s+POSITION",
        ),
        20000,
    ),
    (
        "income_statement",
        (
            r"CONSOLIDATED\s+STATEMENTS\s+OF\s+OPERATIONS",
            r"CONSOLIDATED\s+STATEMENTS\s+OF\s+INCOME",
            r"CONSOLIDATED\s+STATEMENTS\s+OF\s+EARNINGS",
        ),
        20000,
    ),
    (
        "cash_flow_statement",
        (r"CONSOLIDATED\s+STATEMENTS\s+OF\s+CASH\s+FLOWS",),
        20000,
    ),
    (
        "debt_note",
        (
            r"LONG[\s-]TERM\s+DEBT",
            r"DEBT\s+OBLIGATIONS",
            r"BORROWINGS",
        ),
        DEFAULT_CHUNK_WINDOW_CHARS,
    ),
    (
        "property_equipment_note",
        (
            r"PROPERTY,\s+PLANT\s+AND\s+EQUIPMENT",
            r"PROPERTY\s+AND\s+EQUIPMENT",
        ),
        DEFAULT_CHUNK_WINDOW_CHARS,
    ),
    (
        "depreciation_note",
        (
            r"DEPRECIATION\s+(?:AND|&)\s+AMORTIZATION",
            r"AMORTIZATION\s+EXPENSE",
        ),
        DEFAULT_CHUNK_WINDOW_CHARS,
    ),
]

FALLBACK_NOTE_PATTERNS: list[tuple[str, tuple[str, ...], int]] = [
    (
        "notes_to_financials",
        (r"NOTES\s+TO\s+CONSOLIDATED\s+FINANCIAL\s+STATEMENTS",),
        DEFAULT_CHUNK_WINDOW_CHARS,
    ),
]

STATEMENT_UNIT_PATTERNS: list[tuple[str, int, tuple[str, ...]]] = [
    (
        "billions",
        1_000_000_000,
        (
            r"(?:amounts|dollars|tabular amounts)\s+in\s+billions",
            r"in\s+billions(?:,|\b)",
        ),
    ),
    (
        "millions",
        1_000_000,
        (
            r"(?:amounts|dollars|tabular amounts)\s+in\s+millions",
            r"in\s+millions(?:,|\b)",
        ),
    ),
    (
        "thousands",
        1_000,
        (
            r"(?:amounts|dollars|tabular amounts)\s+in\s+thousands",
            r"in\s+thousands(?:,|\b)",
        ),
    ),
]

SYSTEM_PROMPT = """You are a senior credit analyst extracting audited annual financial statement data from a single company's Form 10-K.

Work only from the provided filing excerpt. Use the most recent fiscal year shown in the excerpt.

Extraction rules:
1. Prefer consolidated annual financial statements and note disclosures over narrative text.
2. Return absolute raw USD numbers. If the filing says values are in millions or thousands, convert them into full dollars.
3. Do not use quarterly figures, segment figures, prior-year comparative columns, non-GAAP measures, or outside knowledge.
4. If a value is not explicitly stated in the excerpt, return null.
5. Never hallucinate or estimate.
6. Preserve the sign shown in the filing for balance sheet and cash flow metrics.
7. Return interest expense as a positive expense magnitude, even if the filing shows it in parentheses or with a minus sign.
8. Return capital expenditures as a positive cash outflow magnitude.
9. For total_debt, only return a value if the excerpt explicitly states total debt or provides enough debt components in the same excerpt to sum short-term and long-term borrowings. Otherwise return null.
10. For current_portion_of_long_term_debt, return the current portion, current maturities, or short-term borrowings due within one year as a positive number.
11. For retained_earnings, use the retained earnings or accumulated deficit line from shareholders' equity. Preserve the sign shown in the filing.
12. For common_stock and additional_paid_in_capital, use the balance sheet or equity footnote line items when explicit.
13. For gross_profit, use gross profit or gross margin dollars from the latest fiscal year. If not explicit, return null.
14. For ebit, use operating income or income from operations.
15. For ebitda, use explicit EBITDA if present. If EBITDA is not explicit, compute it only when both operating income and depreciation and amortization are explicit in the same excerpt. Otherwise return null.
16. source_quote must be the exact supporting line from the excerpt. If a value is derived from two explicit lines, join the two exact lines with " || ".

Output only valid JSON that matches the schema exactly."""

USER_PROMPT_TEMPLATE = """Company: {company_name}
Ticker: {ticker}
Filing date: {filing_date}
Chunk label: {chunk_label}

Target fields and definitions:
- revenue: total net revenue, total revenues, or net sales for the latest fiscal year
- net_income: net income attributable to the company for the latest fiscal year
- total_assets: total assets
- total_current_assets: total current assets
- total_current_liabilities: total current liabilities
- total_liabilities: total liabilities
- total_shareholders_equity: total stockholders' or shareholders' equity
- total_debt: short-term borrowings plus current portion of long-term debt plus long-term debt
- cash_and_cash_equivalents: cash and cash equivalents, excluding restricted cash unless the filing combines them in one line
- interest_expense: interest expense for the latest fiscal year, returned as a positive number
- ebit: operating income or income from operations
- ebitda: explicit EBITDA, or operating income plus depreciation and amortization only when both are explicit in this excerpt
- operating_cash_flow: net cash provided by operating activities
- capital_expenditures: purchases or acquisitions of property and equipment, returned as a positive number
- retained_earnings: retained earnings or accumulated deficit from the equity section
- common_stock: common stock or capital stock balance when explicit
- additional_paid_in_capital: additional paid-in capital or APIC when explicit
- gross_profit: gross profit for the latest fiscal year
- current_portion_of_long_term_debt: current maturities of long-term debt or short-term borrowings due within one year

Return only JSON.

Filing excerpt:
<<<EXCERPT
{excerpt}
EXCERPT>>>"""


def _nullable_number_schema(description: str) -> dict[str, Any]:
    return {
        "description": description,
        "anyOf": [
            {"type": "number"},
            {"type": "null"},
        ],
    }


def _nullable_string_schema(description: str) -> dict[str, Any]:
    return {
        "description": description,
        "anyOf": [
            {"type": "string"},
            {"type": "null"},
        ],
    }


def _metric_object_schema(label: str) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "value": _nullable_number_schema(
                f"Absolute raw USD number for {label}, or null if not present."
            ),
            "source_quote": _nullable_string_schema(
                f"Exact supporting filing line for {label}, or null if unavailable."
            ),
        },
        "required": ["value", "source_quote"],
        "additionalProperties": False,
    }


EXTRACTION_JSON_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "name": "financial_extraction",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            field_name: _metric_object_schema(FIELD_LABELS[field_name])
            for field_name in FIELD_ORDER
        },
        "required": FIELD_ORDER,
        "additionalProperties": False,
    },
}


def get_openai_api_key(api_key: str | None = None) -> str:
    resolved_key = (api_key or os.getenv("OPENAI_API_KEY", "")).strip()
    if not resolved_key:
        raise OpenAIConfigurationError(
            "OpenAI API key is required for financial extraction. "
            "Enter it in the app sidebar or set OPENAI_API_KEY in the environment."
        )
    return resolved_key


def build_openai_client(api_key: str | None = None) -> OpenAI:
    return OpenAI(api_key=get_openai_api_key(api_key))


def call_openai_responses_with_retries(
    request_callable: Callable[[], Any],
    *,
    rate_limit_exception_cls: type[Exception],
) -> Any:
    for attempt in range(OPENAI_RATE_LIMIT_MAX_RETRIES + 1):
        try:
            return request_callable()
        except RateLimitError as exc:
            if attempt >= OPENAI_RATE_LIMIT_MAX_RETRIES:
                raise rate_limit_exception_cls(
                    "OpenAI rate limit reached after 3 retries. Wait 60 seconds and try again."
                ) from exc
            time.sleep(OPENAI_RATE_LIMIT_BASE_DELAY_SECONDS * (2**attempt))
        except APIStatusError as exc:
            if exc.status_code != 429:
                raise
            if attempt >= OPENAI_RATE_LIMIT_MAX_RETRIES:
                raise rate_limit_exception_cls(
                    "OpenAI rate limit reached after 3 retries. Wait 60 seconds and try again."
                ) from exc
            time.sleep(OPENAI_RATE_LIMIT_BASE_DELAY_SECONDS * (2**attempt))


def format_financial_value(value: int | float | None) -> str:
    if value is None:
        return "N/A"

    absolute_value = abs(float(value))
    if absolute_value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.1f}B"
    if absolute_value >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    if absolute_value >= 1_000:
        return f"${value / 1_000:.1f}K"
    return f"${value:.1f}"


def format_ratio_value(value: float | None, ratio_name: str) -> str:
    if value is None:
        return "N/A"
    if ratio_name == "free_cash_flow_yield":
        return f"{value:.2%}"
    return f"{value:.2f}x"


def _detect_unit_from_text(text: str) -> tuple[str, int]:
    search_text = text[:3500]
    for unit_label, multiplier, patterns in STATEMENT_UNIT_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, search_text, re.IGNORECASE):
                return unit_label, multiplier
    return "dollars", 1


def _build_chunk_unit_multiplier_map(
    chunks: list[FilingChunk],
    prepared_text: str,
) -> tuple[dict[str, int], str, int]:
    chunk_multiplier_map: dict[str, int] = {}
    detected_labels: list[str] = []

    for chunk in chunks:
        unit_label, multiplier = _detect_unit_from_text(chunk.text)
        if multiplier == 1:
            continue
        chunk_multiplier_map[chunk.label] = multiplier
        detected_labels.append(unit_label)

    if chunk_multiplier_map:
        primary_unit_label = max(set(detected_labels), key=detected_labels.count)
        primary_unit_multiplier = {
            label: multiplier
            for label, multiplier, _ in STATEMENT_UNIT_PATTERNS
        }[primary_unit_label]
        return chunk_multiplier_map, primary_unit_label, primary_unit_multiplier

    fallback_unit_label, fallback_multiplier = _detect_unit_from_text(prepared_text)
    return chunk_multiplier_map, fallback_unit_label, fallback_multiplier


def _extract_plain_numeric_candidates(quote: str) -> list[float]:
    candidates: list[float] = []
    for match in re.finditer(r"\(?\$?([0-9][0-9,]*(?:\.[0-9]+)?)\)?", quote):
        raw_text = match.group(0)
        numeric_text = match.group(1).replace(",", "")
        try:
            value = float(numeric_text)
        except ValueError:
            continue
        if raw_text.startswith("(") and raw_text.endswith(")"):
            value *= -1
        candidates.append(value)
    return candidates


def _extract_explicit_scaled_candidates(quote: str) -> list[float]:
    candidates: list[float] = []
    scale_map = {
        "billion": 1_000_000_000,
        "billions": 1_000_000_000,
        "bn": 1_000_000_000,
        "b": 1_000_000_000,
        "million": 1_000_000,
        "millions": 1_000_000,
        "mm": 1_000_000,
        "m": 1_000_000,
        "thousand": 1_000,
        "thousands": 1_000,
        "k": 1_000,
    }
    pattern = re.compile(
        r"\(?\$?\s*([0-9]+(?:\.[0-9]+)?)\s*(billion|billions|million|millions|thousand|thousands|bn|mm|[bmk])\b",
        re.IGNORECASE,
    )
    for match in pattern.finditer(quote):
        numeric_value = float(match.group(1))
        scale_token = match.group(2).lower()
        scaled_value = numeric_value * scale_map[scale_token]
        if match.group(0).strip().startswith("("):
            scaled_value *= -1
        candidates.append(scaled_value)
    return candidates


def _approximately_equal(left: float, right: float) -> bool:
    left_abs = abs(left)
    right_abs = abs(right)
    baseline = max(left_abs, right_abs, 1.0)
    return abs(left_abs - right_abs) <= baseline * 0.02


def _determine_metric_multiplier(
    metric: MetricValue,
    *,
    detected_multiplier: int,
) -> int:
    if metric.value is None or detected_multiplier == 1:
        return 1

    source_quote = metric.source_quote or ""
    current_value = float(metric.value)

    explicit_scaled_candidates = _extract_explicit_scaled_candidates(source_quote)
    for candidate in explicit_scaled_candidates:
        if _approximately_equal(current_value, candidate):
            return 1
        if _approximately_equal(current_value * detected_multiplier, candidate):
            return detected_multiplier

    plain_candidates = _extract_plain_numeric_candidates(source_quote)
    for candidate in plain_candidates:
        if _approximately_equal(current_value, candidate * detected_multiplier):
            return 1
        if _approximately_equal(current_value, candidate):
            return detected_multiplier

    if abs(current_value) < 1_000_000_000:
        return detected_multiplier

    return 1


def normalize_extracted_metrics(
    metrics: dict[str, MetricValue],
    chunks: list[FilingChunk],
    prepared_text: str,
) -> tuple[dict[str, MetricValue], str, int, str]:
    chunk_multiplier_map, detected_unit_label, detected_unit_multiplier = (
        _build_chunk_unit_multiplier_map(chunks, prepared_text)
    )
    normalized_metrics: dict[str, MetricValue] = {}
    normalized_field_labels: list[str] = []

    for field_name in FIELD_ORDER:
        metric = metrics[field_name]
        chunk_multiplier = chunk_multiplier_map.get(
            metric.source_chunk_label or "",
            detected_unit_multiplier,
        )
        metric_multiplier = _determine_metric_multiplier(
            metric,
            detected_multiplier=chunk_multiplier,
        )
        normalized_value = None if metric.value is None else metric.value * metric_multiplier
        normalized_metrics[field_name] = MetricValue(
            value=normalized_value,
            source_quote=metric.source_quote,
            source_chunk_label=metric.source_chunk_label,
        )
        if metric_multiplier > 1:
            normalized_field_labels.append(FIELD_LABELS[field_name])

    if detected_unit_multiplier == 1 or not normalized_field_labels:
        normalization_note = "Values already in dollars; no normalization applied."
    else:
        normalization_note = (
            f"Values normalized from {detected_unit_label} to dollars before ratio computation."
        )

    return (
        normalized_metrics,
        detected_unit_label,
        detected_unit_multiplier,
        normalization_note,
    )


def extract_primary_10k_document(raw_filing_text: str) -> tuple[str, bool]:
    document_pattern = re.compile(
        r"<DOCUMENT>(?P<body>.*?)</DOCUMENT>",
        re.IGNORECASE | re.DOTALL,
    )
    type_pattern = re.compile(r"<TYPE>\s*([^\n<]+)", re.IGNORECASE)

    documents = list(document_pattern.finditer(raw_filing_text))
    if not documents:
        return raw_filing_text, False

    exact_match: str | None = None
    fallback_match: str | None = None

    for document_match in documents:
        body = document_match.group("body")
        type_match = type_pattern.search(body)
        if not type_match:
            continue

        document_type = type_match.group(1).strip().upper()
        if document_type == "10-K":
            exact_match = body
            break
        if document_type.startswith("10-K") and fallback_match is None:
            fallback_match = body

    chosen_document = exact_match or fallback_match
    if chosen_document is None:
        return raw_filing_text, False

    return chosen_document, True


def clean_filing_text(raw_text: str) -> str:
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", text)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?is)<br\s*/?>", "\n", text)
    text = re.sub(
        r"(?is)</(p|div|tr|li|table|h1|h2|h3|h4|h5|h6|center|font)>",
        "\n",
        text,
    )
    text = re.sub(r"(?is)</(td|th)>", " | ", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\| *", " | ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def prepare_filing_text_for_extraction(raw_filing_text: str) -> tuple[str, bool]:
    primary_document_text, used_primary_document = extract_primary_10k_document(
        raw_filing_text
    )
    cleaned_text = clean_filing_text(primary_document_text)
    return cleaned_text, used_primary_document


def _window_has_numeric_density(text: str) -> bool:
    digit_count = len(re.findall(r"\d", text))
    if digit_count >= MIN_DIGIT_COUNT_NEAR_ANCHOR:
        return True

    for line in text.splitlines():
        stripped_line = line.strip()
        if sum(char.isdigit() for char in stripped_line) < 3:
            continue
        if "|" in stripped_line or "$" in stripped_line or "(" in stripped_line:
            return True

    return False


def _build_targeted_chunks(prepared_text: str) -> list[FilingChunk]:
    upper_text = prepared_text.upper()
    chunks: list[FilingChunk] = []

    for label, patterns, window_chars in [*ANCHOR_PATTERNS, *FALLBACK_NOTE_PATTERNS]:
        selected_match: re.Match[str] | None = None

        for pattern in patterns:
            regex = re.compile(pattern, re.IGNORECASE)
            for match in regex.finditer(upper_text):
                window_end = min(len(prepared_text), match.start() + window_chars)
                candidate_window = prepared_text[match.start():window_end]
                if not _window_has_numeric_density(candidate_window):
                    continue
                selected_match = match
                break
            if selected_match is not None:
                break

        if selected_match is None:
            continue

        start_char = max(0, selected_match.start() - SECTION_CONTEXT_PREFIX_CHARS)
        end_char = min(len(prepared_text), start_char + window_chars)

        chunk_text = prepared_text[start_char:end_char].strip()
        if not chunk_text:
            continue

        chunks.append(
            FilingChunk(
                label=label,
                strategy="financial_statement_anchors",
                start_char=start_char,
                end_char=end_char,
                text=chunk_text,
            )
        )

    return chunks


def _build_fallback_chunks(prepared_text: str) -> list[FilingChunk]:
    chunks: list[FilingChunk] = []
    start_char = 0
    text_length = len(prepared_text)

    while start_char < text_length:
        end_char = min(text_length, start_char + FALLBACK_CHUNK_SIZE_CHARS)
        chunk_text = prepared_text[start_char:end_char].strip()
        if chunk_text:
            chunks.append(
                FilingChunk(
                    label=f"fallback_window_{len(chunks) + 1}",
                    strategy="sliding_window_fallback",
                    start_char=start_char,
                    end_char=end_char,
                    text=chunk_text,
                )
            )

        if end_char >= text_length:
            break

        start_char = end_char - FALLBACK_CHUNK_OVERLAP_CHARS

    return chunks


def build_filing_chunks(raw_filing_text: str) -> tuple[list[FilingChunk], str, int, bool]:
    prepared_text, used_primary_document = prepare_filing_text_for_extraction(
        raw_filing_text
    )

    targeted_chunks = _build_targeted_chunks(prepared_text)
    if targeted_chunks:
        return (
            targeted_chunks,
            "financial_statement_anchors",
            len(prepared_text),
            used_primary_document,
        )

    fallback_chunks = _build_fallback_chunks(prepared_text)
    return (
        fallback_chunks,
        "sliding_window_fallback",
        len(prepared_text),
        used_primary_document,
    )


def _coerce_metric_value(
    field_name: str,
    metric_payload: dict[str, Any],
    chunk_label: str,
) -> MetricValue:
    raw_value = metric_payload.get("value")
    raw_quote = metric_payload.get("source_quote")

    if raw_value is None:
        return MetricValue(value=None, source_quote=None, source_chunk_label=None)

    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        raise ExtractionParseError(
            f"Model returned a non-numeric value for {field_name}: {raw_value!r}"
        )

    if raw_quote is not None and not isinstance(raw_quote, str):
        raise ExtractionParseError(
            f"Model returned a non-string source_quote for {field_name}: {raw_quote!r}"
        )

    source_quote = raw_quote.strip() if isinstance(raw_quote, str) else None
    return MetricValue(
        value=raw_value,
        source_quote=source_quote or None,
        source_chunk_label=chunk_label,
    )


def _extract_chunk_metrics(
    client: OpenAI,
    chunk: FilingChunk,
    *,
    company_name: str,
    ticker: str,
    filing_date: str,
    model: str,
) -> dict[str, MetricValue]:
    user_prompt = USER_PROMPT_TEMPLATE.format(
        company_name=company_name,
        ticker=ticker,
        filing_date=filing_date,
        chunk_label=chunk.label,
        excerpt=chunk.text,
    )

    try:
        response = call_openai_responses_with_retries(
            lambda: client.responses.create(
                model=model,
                instructions=SYSTEM_PROMPT,
                input=user_prompt,
                temperature=0,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                store=False,
                text={"format": EXTRACTION_JSON_SCHEMA},
            ),
            rate_limit_exception_cls=OpenAIExtractionError,
        )
    except AuthenticationError as exc:
        raise OpenAIConfigurationError(
            "OpenAI rejected the API key. Check OPENAI_API_KEY and retry."
        ) from exc
    except BadRequestError as exc:
        raise OpenAIExtractionError(
            f"OpenAI rejected the extraction request: {exc}"
        ) from exc
    except APIConnectionError as exc:
        raise OpenAIExtractionError(
            f"OpenAI connection failed during extraction: {exc}"
        ) from exc
    except APIStatusError as exc:
        raise OpenAIExtractionError(
            f"OpenAI returned HTTP {exc.status_code} during extraction."
        ) from exc
    except Exception as exc:
        raise OpenAIExtractionError(
            f"Unexpected OpenAI error during extraction: {exc}"
        ) from exc

    output_text = getattr(response, "output_text", None)
    if not output_text:
        raise ExtractionParseError(
            "OpenAI returned no structured output for the extraction request."
        )

    try:
        payload = json.loads(output_text)
    except json.JSONDecodeError as exc:
        raise ExtractionParseError(
            "OpenAI returned invalid JSON for the extraction request."
        ) from exc

    if not isinstance(payload, dict):
        raise ExtractionParseError("OpenAI extraction output was not a JSON object.")

    metrics: dict[str, MetricValue] = {}
    for field_name in FIELD_ORDER:
        metric_payload = payload.get(field_name)
        if not isinstance(metric_payload, dict):
            raise ExtractionParseError(
                f"OpenAI extraction output is missing object data for {field_name}."
            )
        metrics[field_name] = _coerce_metric_value(
            field_name=field_name,
            metric_payload=metric_payload,
            chunk_label=chunk.label,
        )

    return metrics


def _merge_chunk_metric_results(
    chunk_metric_results: list[dict[str, MetricValue]],
) -> dict[str, MetricValue]:
    merged: dict[str, MetricValue] = {}

    for field_name in FIELD_ORDER:
        selected_metric = MetricValue(value=None, source_quote=None, source_chunk_label=None)
        for chunk_metrics in chunk_metric_results:
            metric_value = chunk_metrics[field_name]
            if metric_value.value is None:
                continue
            selected_metric = metric_value
            break
        merged[field_name] = selected_metric

    return merged


def _safe_divide(
    numerator: float | int | None,
    denominator: float | int | None,
) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def compute_credit_ratios(
    metrics: dict[str, MetricValue],
    market_cap: float | int | None = None,
) -> dict[str, RatioValue]:
    total_liabilities = metrics["total_liabilities"].value
    total_equity = metrics["total_shareholders_equity"].value
    ebit = metrics["ebit"].value
    interest_expense = metrics["interest_expense"].value
    current_assets = metrics["total_current_assets"].value
    current_liabilities = metrics["total_current_liabilities"].value
    total_debt = metrics["total_debt"].value
    cash = metrics["cash_and_cash_equivalents"].value
    ebitda = metrics["ebitda"].value
    operating_cash_flow = metrics["operating_cash_flow"].value
    capital_expenditures = metrics["capital_expenditures"].value

    ratios: dict[str, RatioValue] = {}

    debt_to_equity = _safe_divide(total_liabilities, total_equity)
    ratios["debt_to_equity"] = RatioValue(
        value=debt_to_equity,
        note=None
        if debt_to_equity is not None
        else "Cannot compute because total liabilities or shareholders' equity is missing or zero.",
    )

    interest_coverage = _safe_divide(ebit, interest_expense)
    ratios["interest_coverage"] = RatioValue(
        value=interest_coverage,
        note=None
        if interest_coverage is not None
        else "Cannot compute because EBIT or interest expense is missing or zero.",
    )

    current_ratio = _safe_divide(current_assets, current_liabilities)
    ratios["current_ratio"] = RatioValue(
        value=current_ratio,
        note=None
        if current_ratio is not None
        else "Cannot compute because current assets or current liabilities is missing or zero.",
    )

    if total_debt is None or cash is None or ebitda in (None, 0):
        ratios["net_debt_to_ebitda"] = RatioValue(
            value=None,
            note="Cannot compute because total debt, cash, or EBITDA is missing or zero.",
        )
    else:
        ratios["net_debt_to_ebitda"] = RatioValue(
            value=(float(total_debt) - float(cash)) / float(ebitda),
            note=None,
        )

    if market_cap in (None, 0):
        ratios["free_cash_flow_yield"] = RatioValue(
            value=None,
            note="Skipped because market cap data is not available yet.",
        )
    elif operating_cash_flow is None or capital_expenditures is None:
        ratios["free_cash_flow_yield"] = RatioValue(
            value=None,
            note="Cannot compute because operating cash flow or capital expenditures is missing.",
        )
    else:
        free_cash_flow = float(operating_cash_flow) - float(capital_expenditures)
        ratios["free_cash_flow_yield"] = RatioValue(
            value=free_cash_flow / float(market_cap),
            note=None,
        )

    return ratios


def extract_financials(
    *,
    raw_filing_text: str,
    company_name: str,
    ticker: str,
    filing_date: str,
    api_key: str | None = None,
    model: str = DEFAULT_OPENAI_MODEL,
    chunk_progress_callback: Callable[[int, int, FilingChunk], None] | None = None,
    inter_chunk_delay_seconds: int = EXTRACTION_INTER_CHUNK_DELAY_SECONDS,
) -> FinancialExtractionResult:
    chunks, chunk_strategy, prepared_text_length, used_primary_document = build_filing_chunks(
        raw_filing_text
    )
    prepared_text, _ = prepare_filing_text_for_extraction(raw_filing_text)
    if not chunks:
        raise OpenAIExtractionError("No filing chunks were available for extraction.")

    client = build_openai_client(api_key=api_key)

    chunk_metric_results: list[dict[str, MetricValue]] = []
    total_chunks = len(chunks)
    for chunk_index, chunk in enumerate(chunks, start=1):
        if chunk_progress_callback is not None:
            chunk_progress_callback(chunk_index, total_chunks, chunk)
        chunk_metric_results.append(
            _extract_chunk_metrics(
                client,
                chunk,
                company_name=company_name,
                ticker=ticker,
                filing_date=filing_date,
                model=model,
            )
        )
        if chunk_index < total_chunks and inter_chunk_delay_seconds > 0:
            time.sleep(inter_chunk_delay_seconds)

    merged_metrics = _merge_chunk_metric_results(chunk_metric_results)
    normalized_metrics, detected_unit_label, detected_unit_multiplier, normalization_note = (
        normalize_extracted_metrics(
            merged_metrics,
            chunks,
            prepared_text,
        )
    )
    ratios = compute_credit_ratios(normalized_metrics)

    return FinancialExtractionResult(
        metrics=normalized_metrics,
        ratios=ratios,
        chunk_strategy=chunk_strategy,
        chunk_count=len(chunks),
        prepared_text_length=prepared_text_length,
        used_primary_document=used_primary_document,
        chunks=tuple(chunks),
        detected_statement_unit_label=detected_unit_label,
        detected_statement_unit_multiplier=detected_unit_multiplier,
        normalization_note=normalization_note,
    )
