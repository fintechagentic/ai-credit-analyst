from __future__ import annotations

import html
import io
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
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.graphics.charts.barcharts import HorizontalBarChart, VerticalBarChart
from reportlab.graphics.shapes import Circle, Drawing, Line, Rect, String
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from ai_credit_analyst.exceptions import (
    MemoGenerationError,
    MemoParseError,
    OpenAIConfigurationError,
    PDFGenerationError,
)
from ai_credit_analyst.extraction import (
    DEFAULT_OPENAI_MODEL,
    FIELD_LABELS,
    PRIMARY_FIELD_ORDER,
    RATIO_LABELS,
    RATIO_ORDER,
    build_openai_client,
    call_openai_responses_with_retries,
    format_financial_value,
    format_ratio_value,
    prepare_filing_text_for_extraction,
)
from ai_credit_analyst.models import (
    AltmanZScoreResult,
    CreditMemoResult,
    CreditOutlook,
    CreditRiskRating,
    DebtMaturityProfileResult,
    FinancialExtractionResult,
    FinancialSummaryRow,
    MemoContextSection,
    MemoPoint,
    QuantitativeModelResult,
    TrendAnalysisRow,
)

OPENAI_RESPONSES_API_URL = "https://api.openai.com/v1/responses"
MEMO_MAX_OUTPUT_TOKENS = 4000
ALLOWED_RATINGS = [
    "AAA",
    "AA+",
    "AA",
    "AA-",
    "A+",
    "A",
    "A-",
    "BBB+",
    "BBB",
    "BBB-",
    "BB+",
    "BB",
    "BB-",
    "B+",
    "B",
    "B-",
    "CCC+",
    "CCC",
    "CCC-",
]
ALLOWED_OUTLOOKS = ["Positive", "Stable", "Negative"]

PDF_NAVY = colors.HexColor("#1B2A4A")
PDF_NAVY_LIGHT = colors.HexColor("#304769")
PDF_STEEL = colors.HexColor("#506D91")
PDF_TEXT = colors.HexColor("#333333")
PDF_MUTED = colors.HexColor("#5D6470")
PDF_LIGHT_GRAY = colors.HexColor("#F3F5F8")
PDF_TABLE_ALT = colors.HexColor("#F5F7FA")
PDF_BORDER = colors.HexColor("#C9D1DD")
PDF_GREEN = colors.HexColor("#2E8B57")
PDF_BLUE = colors.HexColor("#4A90D9")
PDF_ORANGE = colors.HexColor("#D38B2C")
PDF_RED = colors.HexColor("#C44747")
PDF_YELLOW = colors.HexColor("#D8A73D")
PDF_WHITE = colors.white
PDF_SUMMARY_BG = colors.HexColor("#E9F1FB")
PDF_CONFIDENTIAL = colors.HexColor("#B64040")

PDF_PAGE_MARGIN = 1.0 * inch
PDF_CONTENT_WIDTH = LETTER[0] - (2 * PDF_PAGE_MARGIN)
PDF_HEADER_ACCENT_HEIGHT = 0.12 * inch

PDF_RATIO_THRESHOLDS: dict[str, tuple[str, float, float, str]] = {
    "Debt-to-Equity": ("low", 1.5, 3.0, "Healthy: < 1.5x"),
    "Interest Coverage": ("high", 2.0, 5.0, "Healthy: > 5.0x"),
    "Current Ratio": ("high", 1.0, 1.5, "Healthy: > 1.5x"),
    "Net Debt / EBITDA": ("low", 2.0, 4.0, "Healthy: < 2.0x"),
}

PDF_RATIO_BENCHMARKS: dict[str, float] = {
    "Debt-to-Equity": 1.0,
    "Interest Coverage": 5.0,
    "Current Ratio": 1.5,
    "Net Debt / EBITDA": 2.0,
}

PDF_RATING_PERCENTILES = {
    "AAA": 99,
    "AA": 95,
    "A": 85,
    "BBB": 60,
    "BB": 35,
    "B": 15,
    "CCC": 5,
}

NARRATIVE_CONTEXT_PATTERNS: list[tuple[str, str, tuple[str, ...], int]] = [
    (
        "business_overview",
        "Business Overview",
        (
            r"ITEM\s+1\.?\s+BUSINESS",
            r"BUSINESS\s+OVERVIEW",
        ),
        5000,
    ),
    (
        "risk_factors",
        "Risk Factors",
        (
            r"ITEM\s+1A\.?\s+RISK\s+FACTORS",
            r"RISK\s+FACTORS",
        ),
        7000,
    ),
    (
        "md_and_a",
        "Management Discussion and Analysis",
        (
            r"ITEM\s+7\.?\s+MANAGEMENT\S*\s+DISCUSSION",
            r"MANAGEMENT\S*\s+DISCUSSION\s+AND\s+ANALYSIS",
        ),
        7000,
    ),
    (
        "liquidity",
        "Liquidity and Capital Resources",
        (r"LIQUIDITY\s+AND\s+CAPITAL\s+RESOURCES",),
        6000,
    ),
]

MEMO_SYSTEM_PROMPT = """You are a senior corporate credit analyst at a major bank preparing an internal credit committee brief on a public U.S. issuer.

Your job is to produce a conservative, defensible credit memo using only the supplied 10-K evidence and normalized financial data.

Judgment rules:
1. Credit quality must be earned. Do not assign a generous rating because the issuer is large, well known, or in technology.
2. Rating calibration:
   - AAA: extraordinary resilience, minimal leverage, exceptional stability, negligible downside risk.
   - AA range: very strong balance sheet and cash generation with limited business risk.
   - A range: strong franchise, good liquidity, manageable leverage, but still exposed to meaningful industry or execution risk.
   - BBB range: adequate investment grade; leverage, concentration, cyclicality, or financial policy materially constrain the profile.
   - BB range: speculative profile with clear pressure on resilience or financial flexibility.
   - B range: highly speculative with thin credit protection.
   - CCC range: vulnerable and dependent on favorable conditions.
3. For large-cap technology, media, and platform businesses, most defensible outcomes usually sit in the A to BBB+ range unless leverage is unusually low and cash flow durability is exceptionally strong, or unless financial policy and business risk are clearly more aggressive.
4. Risks must be issuer-specific. Do not list generic risks such as competition, regulation, or macro conditions unless the supplied evidence makes them concrete.
5. Strengths must also be issuer-specific and tied to the supplied numbers or filing language.
6. Cite specific evidence in every risk and strength. Evidence should quote or tightly paraphrase the supplied excerpts.
7. Use formal third-person committee language. State conclusions directly. Do not hedge with phrases like "it appears," "may suggest," or "could indicate."
8. Do not invent facts, peers, market data, or management commentary outside the supplied context.
9. Return exactly five risk factors and exactly three strengths.
10. Copy the provided financial summary table exactly into the output. Do not rename labels or change values.
11. All monetary values supplied in the financial summary are already normalized to raw dollars. Do not infer or apply any additional unit conversion.

Output only valid JSON matching the schema exactly."""

MEMO_USER_PROMPT_TEMPLATE = """Company: {company_name}
Ticker: {ticker}
Filing date: {filing_date}

Financial summary table to copy exactly. Values are already normalized to raw dollars and pre-formatted for display:
{financial_summary_json}

Extracted metric evidence. All monetary values are already normalized to raw dollars:
{metric_evidence_json}

Key filing excerpts:
{context_sections_json}

Required output sections:
- company_overview: 2-3 sentences on business model, sector, and scale
- financial_summary_table: copy the provided table exactly
- credit_risk_rating: one rating from the allowed scale plus a one-paragraph justification
- key_risk_factors: exactly 5, each with title, 2-3 sentence explanation, and specific evidence
- key_strengths: exactly 3, each with title, 2-3 sentence explanation, and specific evidence
- outlook: Positive, Stable, or Negative with a 2-3 sentence justification

Return JSON only."""

def build_financial_summary_rows(
    financial_result: FinancialExtractionResult,
) -> tuple[FinancialSummaryRow, ...]:
    rows: list[FinancialSummaryRow] = []

    for field_name in PRIMARY_FIELD_ORDER:
        metric = financial_result.metrics[field_name]
        rows.append(
            FinancialSummaryRow(
                label=FIELD_LABELS[field_name],
                value=format_financial_value(metric.value),
                evidence=metric.source_quote or "Not found in filing excerpt.",
            )
        )

    for ratio_name in RATIO_ORDER:
        ratio = financial_result.ratios[ratio_name]
        evidence = ratio.note or "Computed from extracted filing values."
        rows.append(
            FinancialSummaryRow(
                label=RATIO_LABELS[ratio_name],
                value=format_ratio_value(ratio.value, ratio_name),
                evidence=evidence,
            )
        )

    return tuple(rows)


def _select_preferred_match(matches: list[re.Match[str]]) -> re.Match[str] | None:
    if not matches:
        return None

    for match in matches:
        if match.start() >= 3000:
            return match
    if len(matches) > 1:
        return matches[1]
    return matches[0]


def _extract_narrative_context_sections(prepared_text: str) -> list[MemoContextSection]:
    upper_text = prepared_text.upper()
    sections: list[MemoContextSection] = []

    for label, title, patterns, window_chars in NARRATIVE_CONTEXT_PATTERNS:
        matches: list[re.Match[str]] = []
        for pattern in patterns:
            matches.extend(re.finditer(pattern, upper_text, re.IGNORECASE))

        selected_match = _select_preferred_match(matches)
        if selected_match is None:
            continue

        start_char = max(0, selected_match.start() - 400)
        end_char = min(len(prepared_text), start_char + window_chars)
        text = prepared_text[start_char:end_char].strip()
        if not text:
            continue

        sections.append(MemoContextSection(label=label, title=title, text=text))

    if not any(section.label == "business_overview" for section in sections):
        fallback_text = prepared_text[:4500].strip()
        if fallback_text:
            sections.insert(
                0,
                MemoContextSection(
                    label="business_overview",
                    title="Business Overview",
                    text=fallback_text,
                ),
            )

    return sections


def build_memo_context_sections(
    raw_filing_text: str,
    financial_result: FinancialExtractionResult,
) -> tuple[MemoContextSection, ...]:
    prepared_text, _ = prepare_filing_text_for_extraction(raw_filing_text)
    sections = _extract_narrative_context_sections(prepared_text)

    for chunk in financial_result.chunks:
        if chunk.label not in {"debt_note", "balance_sheet", "cash_flow_statement"}:
            continue

        title = {
            "debt_note": "Debt Note Excerpt",
            "balance_sheet": "Balance Sheet Excerpt",
            "cash_flow_statement": "Cash Flow Statement Excerpt",
        }[chunk.label]

        sections.append(
            MemoContextSection(
                label=chunk.label,
                title=title,
                text=chunk.text[:4000].strip(),
            )
        )

    deduped_sections: list[MemoContextSection] = []
    seen_labels: set[str] = set()
    for section in sections:
        if section.label in seen_labels:
            continue
        seen_labels.add(section.label)
        deduped_sections.append(section)

    return tuple(deduped_sections)


def _build_metric_evidence_payload(
    financial_result: FinancialExtractionResult,
) -> dict[str, dict[str, str | None]]:
    payload: dict[str, dict[str, str | None]] = {}
    for field_name in PRIMARY_FIELD_ORDER:
        metric = financial_result.metrics[field_name]
        payload[field_name] = {
            "label": FIELD_LABELS[field_name],
            "value": format_financial_value(metric.value),
            "source_quote": metric.source_quote,
        }
    for ratio_name in RATIO_ORDER:
        ratio = financial_result.ratios[ratio_name]
        payload[ratio_name] = {
            "label": RATIO_LABELS[ratio_name],
            "value": format_ratio_value(ratio.value, ratio_name),
            "source_quote": ratio.note or "Computed from extracted filing values.",
        }
    return payload


def _memo_point_schema(section_name: str) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": f"Short {section_name} title.",
            },
            "explanation": {
                "type": "string",
                "description": f"2-3 sentence {section_name} explanation.",
            },
            "evidence": {
                "type": "string",
                "description": "Specific filing evidence supporting the point.",
            },
        },
        "required": ["title", "explanation", "evidence"],
        "additionalProperties": False,
    }


def _build_memo_json_schema(
    financial_summary: tuple[FinancialSummaryRow, ...],
) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "name": "credit_memo",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "company_overview": {"type": "string"},
                "financial_summary_table": {
                    "type": "array",
                    "minItems": len(financial_summary),
                    "maxItems": len(financial_summary),
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string"},
                            "value": {"type": "string"},
                            "evidence": {"type": "string"},
                        },
                        "required": ["label", "value", "evidence"],
                        "additionalProperties": False,
                    },
                },
                "credit_risk_rating": {
                    "type": "object",
                    "properties": {
                        "rating": {
                            "type": "string",
                            "enum": ALLOWED_RATINGS,
                        },
                        "justification": {"type": "string"},
                    },
                    "required": ["rating", "justification"],
                    "additionalProperties": False,
                },
                "key_risk_factors": {
                    "type": "array",
                    "minItems": 5,
                    "maxItems": 5,
                    "items": _memo_point_schema("risk factor"),
                },
                "key_strengths": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 3,
                    "items": _memo_point_schema("credit strength"),
                },
                "outlook": {
                    "type": "object",
                    "properties": {
                        "direction": {
                            "type": "string",
                            "enum": ALLOWED_OUTLOOKS,
                        },
                        "justification": {"type": "string"},
                    },
                    "required": ["direction", "justification"],
                    "additionalProperties": False,
                },
            },
            "required": [
                "company_overview",
                "financial_summary_table",
                "credit_risk_rating",
                "key_risk_factors",
                "key_strengths",
                "outlook",
            ],
            "additionalProperties": False,
        },
    }


def _coerce_memo_point(raw_point: dict[str, Any], section_name: str) -> MemoPoint:
    title = raw_point.get("title")
    explanation = raw_point.get("explanation")
    evidence = raw_point.get("evidence")

    if not all(isinstance(item, str) and item.strip() for item in (title, explanation, evidence)):
        raise MemoParseError(f"Memo output contained an invalid {section_name} entry.")

    return MemoPoint(
        title=title.strip(),
        explanation=explanation.strip(),
        evidence=evidence.strip(),
    )


def _parse_credit_memo_response(
    output_text: str,
    *,
    company_name: str,
    ticker: str,
    filing_date: str,
    financial_summary: tuple[FinancialSummaryRow, ...],
    context_sections: tuple[MemoContextSection, ...],
) -> CreditMemoResult:
    try:
        payload = json.loads(output_text)
    except json.JSONDecodeError as exc:
        raise MemoParseError("OpenAI returned invalid JSON for memo generation.") from exc

    if not isinstance(payload, dict):
        raise MemoParseError("OpenAI memo output was not a JSON object.")

    company_overview = payload.get("company_overview")
    if not isinstance(company_overview, str) or not company_overview.strip():
        raise MemoParseError("Memo output is missing a valid company overview.")

    rating_payload = payload.get("credit_risk_rating")
    if not isinstance(rating_payload, dict):
        raise MemoParseError("Memo output is missing credit_risk_rating.")

    rating = rating_payload.get("rating")
    justification = rating_payload.get("justification")
    if rating not in ALLOWED_RATINGS or not isinstance(justification, str) or not justification.strip():
        raise MemoParseError("Memo output contained an invalid credit rating section.")

    outlook_payload = payload.get("outlook")
    if not isinstance(outlook_payload, dict):
        raise MemoParseError("Memo output is missing outlook.")

    direction = outlook_payload.get("direction")
    outlook_justification = outlook_payload.get("justification")
    if (
        direction not in ALLOWED_OUTLOOKS
        or not isinstance(outlook_justification, str)
        or not outlook_justification.strip()
    ):
        raise MemoParseError("Memo output contained an invalid outlook section.")

    raw_risk_factors = payload.get("key_risk_factors")
    raw_strengths = payload.get("key_strengths")
    if not isinstance(raw_risk_factors, list) or len(raw_risk_factors) != 5:
        raise MemoParseError("Memo output must contain exactly five risk factors.")
    if not isinstance(raw_strengths, list) or len(raw_strengths) != 3:
        raise MemoParseError("Memo output must contain exactly three strengths.")

    risk_factors = tuple(
        _coerce_memo_point(raw_point, "risk factor") for raw_point in raw_risk_factors
    )
    strengths = tuple(
        _coerce_memo_point(raw_point, "strength") for raw_point in raw_strengths
    )

    raw_summary = payload.get("financial_summary_table")
    if not isinstance(raw_summary, list) or len(raw_summary) != len(financial_summary):
        raise MemoParseError("Memo output did not return the expected financial summary table.")

    return CreditMemoResult(
        company_name=company_name,
        ticker=ticker,
        filing_date=filing_date,
        company_overview=company_overview.strip(),
        financial_summary=financial_summary,
        credit_risk_rating=CreditRiskRating(
            rating=rating,
            justification=justification.strip(),
        ),
        key_risk_factors=risk_factors,
        key_strengths=strengths,
        outlook=CreditOutlook(
            direction=direction,
            justification=outlook_justification.strip(),
        ),
        context_sections=context_sections,
    )


def generate_credit_memo(
    *,
    raw_filing_text: str,
    company_name: str,
    ticker: str,
    filing_date: str,
    financial_result: FinancialExtractionResult,
    api_key: str | None = None,
    model: str = DEFAULT_OPENAI_MODEL,
) -> CreditMemoResult:
    financial_summary = build_financial_summary_rows(financial_result)
    context_sections = build_memo_context_sections(raw_filing_text, financial_result)

    user_prompt = MEMO_USER_PROMPT_TEMPLATE.format(
        company_name=company_name,
        ticker=ticker,
        filing_date=filing_date,
        financial_summary_json=json.dumps(
            [row.__dict__ for row in financial_summary],
            indent=2,
        ),
        metric_evidence_json=json.dumps(
            _build_metric_evidence_payload(financial_result),
            indent=2,
        ),
        context_sections_json=json.dumps(
            [section.__dict__ for section in context_sections],
            indent=2,
        ),
    )

    client = build_openai_client(api_key=api_key)

    try:
        response = call_openai_responses_with_retries(
            lambda: client.responses.create(
                model=model,
                instructions=MEMO_SYSTEM_PROMPT,
                input=user_prompt,
                temperature=0.2,
                max_output_tokens=MEMO_MAX_OUTPUT_TOKENS,
                store=False,
                text={"format": _build_memo_json_schema(financial_summary)},
            ),
            rate_limit_exception_cls=MemoGenerationError,
        )
    except AuthenticationError as exc:
        raise OpenAIConfigurationError(
            "OpenAI rejected the API key during memo generation. Check OPENAI_API_KEY and retry."
        ) from exc
    except BadRequestError as exc:
        raise MemoGenerationError(
            f"OpenAI rejected the memo generation request: {exc}"
        ) from exc
    except APIConnectionError as exc:
        raise MemoGenerationError(
            f"OpenAI connection failed during memo generation: {exc}"
        ) from exc
    except APIStatusError as exc:
        raise MemoGenerationError(
            f"OpenAI returned HTTP {exc.status_code} during memo generation."
        ) from exc
    except Exception as exc:
        raise MemoGenerationError(
            f"Unexpected OpenAI error during memo generation: {exc}"
        ) from exc

    output_text = getattr(response, "output_text", None)
    if not output_text:
        raise MemoParseError("OpenAI returned no structured memo output.")

    return _parse_credit_memo_response(
        output_text,
        company_name=company_name,
        ticker=ticker,
        filing_date=filing_date,
        financial_summary=financial_summary,
        context_sections=context_sections,
    )


def _escape_pdf_text(text: str | None) -> str:
    if not text:
        return "N/A"
    return html.escape(text).replace("\n", "<br/>")


def _wrap_url_for_paragraph(url: str) -> str:
    return html.escape(url)


def _parse_display_value(display_value: str | None) -> float | None:
    if display_value is None:
        return None

    cleaned = display_value.strip()
    if not cleaned or cleaned.upper() == "N/A":
        return None

    sign = -1 if cleaned.startswith("-") else 1
    cleaned = cleaned.lstrip("+-$")

    if cleaned.endswith("%"):
        return sign * (float(cleaned[:-1].replace(",", "")) / 100)
    if cleaned.endswith("x"):
        return sign * float(cleaned[:-1].replace(",", ""))

    suffix_multipliers = {
        "B": 1_000_000_000,
        "M": 1_000_000,
        "K": 1_000,
    }
    suffix = cleaned[-1].upper()
    if suffix in suffix_multipliers:
        return sign * float(cleaned[:-1].replace(",", "")) * suffix_multipliers[suffix]

    return sign * float(cleaned.replace(",", ""))


def _summary_lookup(memo: CreditMemoResult) -> dict[str, str]:
    return {row.label: row.value for row in memo.financial_summary}


def _base_rating_key(rating: str) -> str:
    base_rating = rating.strip().upper().rstrip("+-")
    if base_rating.startswith("AAA"):
        return "AAA"
    if base_rating.startswith("AA"):
        return "AA"
    if base_rating.startswith("A"):
        return "A"
    if base_rating.startswith("BBB"):
        return "BBB"
    if base_rating.startswith("BB"):
        return "BB"
    if base_rating.startswith("B"):
        return "B"
    return "CCC"


def _rating_percentile(rating: str) -> int:
    return PDF_RATING_PERCENTILES[_base_rating_key(rating)]


def _first_sentence(text: str) -> str:
    sentence_match = re.search(r"(.+?[.!?])(\s|$)", text.strip())
    if sentence_match:
        return sentence_match.group(1).strip()
    return text.strip()


def _get_rating_color(rating: str) -> colors.Color:
    base_rating = rating.strip().upper().rstrip("+-")
    if base_rating in {"AAA", "AA", "A"}:
        return PDF_GREEN
    if base_rating == "BBB":
        return PDF_BLUE
    if base_rating in {"BB", "B"}:
        return PDF_ORANGE
    return PDF_RED


def _get_outlook_color(direction: str) -> colors.Color:
    normalized = direction.strip().lower()
    if normalized == "positive":
        return PDF_GREEN
    if normalized == "negative":
        return PDF_RED
    return PDF_MUTED


def _get_ratio_status(label: str, value: float | None) -> tuple[str, colors.Color, str]:
    direction, lower_threshold, upper_threshold, healthy_text = PDF_RATIO_THRESHOLDS[label]
    if value is None:
        return "Unavailable", PDF_MUTED, healthy_text

    if direction == "low":
        if value < lower_threshold:
            return "Healthy", PDF_GREEN, healthy_text
        if value <= upper_threshold:
            return "Borderline", PDF_YELLOW, healthy_text
        return "Concerning", PDF_RED, healthy_text

    if value > upper_threshold:
        return "Healthy", PDF_GREEN, healthy_text
    if value >= lower_threshold:
        return "Borderline", PDF_YELLOW, healthy_text
    return "Concerning", PDF_RED, healthy_text


def _choose_financial_scale(values: list[float]) -> tuple[float, str]:
    if not values:
        return 1.0, "USD"

    maximum = max(abs(value) for value in values)
    if maximum >= 1_000_000_000:
        return 1_000_000_000, "USD billions"
    if maximum >= 1_000_000:
        return 1_000_000, "USD millions"
    if maximum >= 1_000:
        return 1_000, "USD thousands"
    return 1.0, "USD"


def _format_axis_tick(value: float) -> str:
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _build_pdf_styles() -> dict[str, ParagraphStyle]:
    sample_styles = getSampleStyleSheet()
    return {
        "cover_title": ParagraphStyle(
            "cover_title",
            parent=sample_styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=25,
            leading=29,
            textColor=PDF_NAVY,
            alignment=TA_LEFT,
            spaceAfter=12,
        ),
        "cover_company": ParagraphStyle(
            "cover_company",
            parent=sample_styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=23,
            leading=27,
            textColor=PDF_NAVY,
            alignment=TA_LEFT,
            spaceAfter=8,
        ),
        "rating_display": ParagraphStyle(
            "rating_display",
            parent=sample_styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=26,
            textColor=PDF_NAVY,
            alignment=TA_LEFT,
            spaceAfter=8,
        ),
        "section_header_text": ParagraphStyle(
            "section_header_text",
            parent=sample_styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=19,
            textColor=PDF_NAVY,
            alignment=TA_LEFT,
            spaceAfter=2,
        ),
        "section_subtitle": ParagraphStyle(
            "section_subtitle",
            parent=sample_styles["BodyText"],
            fontName="Helvetica",
            fontSize=8.2,
            leading=10,
            textColor=PDF_MUTED,
            alignment=TA_LEFT,
        ),
        "body": ParagraphStyle(
            "body",
            parent=sample_styles["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=13.8,
            textColor=PDF_TEXT,
            alignment=TA_LEFT,
            spaceAfter=7,
        ),
        "body_small": ParagraphStyle(
            "body_small",
            parent=sample_styles["BodyText"],
            fontName="Helvetica",
            fontSize=8.6,
            leading=10.8,
            textColor=PDF_TEXT,
            alignment=TA_LEFT,
            spaceAfter=5,
        ),
        "body_muted": ParagraphStyle(
            "body_muted",
            parent=sample_styles["BodyText"],
            fontName="Helvetica",
            fontSize=8.8,
            leading=10.8,
            textColor=PDF_MUTED,
            alignment=TA_LEFT,
            spaceAfter=4,
        ),
        "body_italic": ParagraphStyle(
            "body_italic",
            parent=sample_styles["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=8.4,
            leading=10.6,
            textColor=colors.HexColor("#6D7886"),
            alignment=TA_LEFT,
            leftIndent=14,
            borderPadding=3,
            spaceBefore=2,
            spaceAfter=9,
        ),
        "point_title": ParagraphStyle(
            "point_title",
            parent=sample_styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=12,
            textColor=PDF_NAVY,
            alignment=TA_LEFT,
            spaceAfter=4,
        ),
        "table_header": ParagraphStyle(
            "table_header",
            parent=sample_styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=8.6,
            leading=10,
            textColor=PDF_WHITE,
            alignment=TA_LEFT,
        ),
        "table_group": ParagraphStyle(
            "table_group",
            parent=sample_styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=8.2,
            leading=10,
            textColor=PDF_NAVY,
            alignment=TA_LEFT,
        ),
        "table_cell": ParagraphStyle(
            "table_cell",
            parent=sample_styles["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=PDF_TEXT,
            alignment=TA_LEFT,
        ),
        "table_cell_value": ParagraphStyle(
            "table_cell_value",
            parent=sample_styles["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=PDF_TEXT,
            alignment=TA_RIGHT,
        ),
        "table_cell_center": ParagraphStyle(
            "table_cell_center",
            parent=sample_styles["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=PDF_TEXT,
            alignment=TA_CENTER,
        ),
        "caption": ParagraphStyle(
            "caption",
            parent=sample_styles["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=9.5,
            textColor=PDF_MUTED,
            alignment=TA_LEFT,
            spaceAfter=5,
        ),
        "caption_right": ParagraphStyle(
            "caption_right",
            parent=sample_styles["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=9.5,
            textColor=PDF_MUTED,
            alignment=TA_RIGHT,
            spaceAfter=0,
        ),
        "scorecard_ratio": ParagraphStyle(
            "scorecard_ratio",
            parent=sample_styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=10.5,
            leading=12,
            textColor=PDF_NAVY,
            alignment=TA_LEFT,
        ),
        "scorecard_value": ParagraphStyle(
            "scorecard_value",
            parent=sample_styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=20,
            textColor=PDF_TEXT,
            alignment=TA_LEFT,
        ),
        "scorecard_status": ParagraphStyle(
            "scorecard_status",
            parent=sample_styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=8.4,
            leading=10,
            textColor=PDF_TEXT,
            alignment=TA_LEFT,
        ),
        "summary_title": ParagraphStyle(
            "summary_title",
            parent=sample_styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=10.2,
            leading=12,
            textColor=PDF_NAVY,
            alignment=TA_LEFT,
            spaceAfter=4,
        ),
    }


def _build_section_header(
    title: str,
    styles: dict[str, ParagraphStyle],
    *,
    subtitle: str | None = None,
) -> Table:
    content: list[Any] = [Paragraph(title.upper(), styles["section_header_text"])]
    if subtitle:
        content.append(Paragraph(_escape_pdf_text(subtitle), styles["section_subtitle"]))

    header = Table(
        [["", content]],
        colWidths=[0.1 * inch, PDF_CONTENT_WIDTH - (0.1 * inch)],
    )
    header.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, 0), PDF_STEEL),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LINEBELOW", (0, 0), (-1, -1), 0.35, PDF_BORDER),
            ]
        )
    )
    return header


def _build_section_divider() -> Drawing:
    drawing = Drawing(PDF_CONTENT_WIDTH, 10)
    drawing.add(Line(0, 5, PDF_CONTENT_WIDTH, 5, strokeColor=PDF_BORDER, strokeWidth=0.4))
    return drawing


def _build_cover_rating_drawing(memo: CreditMemoResult) -> Drawing:
    rating_color = _get_rating_color(memo.credit_risk_rating.rating)
    outlook_color = _get_outlook_color(memo.outlook.direction)
    badge_width = max(90, 26 + (len(memo.outlook.direction) * 7.0))
    rating_font_size = 72 if len(memo.credit_risk_rating.rating) <= 2 else 58

    drawing = Drawing(200, 240)
    drawing.add(String(20, 222, "ISSUER RATING", fontName="Helvetica-Bold", fontSize=10, fillColor=PDF_NAVY))
    drawing.add(Circle(92, 142, 68, fillColor=rating_color, strokeColor=rating_color))
    drawing.add(
        String(
            92,
            120,
            memo.credit_risk_rating.rating,
            fontName="Helvetica-Bold",
            fontSize=rating_font_size,
            fillColor=PDF_WHITE,
            textAnchor="middle",
        )
    )
    drawing.add(
        Rect(
            47,
            38,
            badge_width,
            28,
            rx=10,
            ry=10,
            fillColor=outlook_color,
            strokeColor=outlook_color,
        )
    )
    drawing.add(
        String(
            47 + (badge_width / 2),
            48,
            f"{memo.outlook.direction} Outlook",
            fontName="Helvetica-Bold",
            fontSize=10,
            fillColor=PDF_WHITE,
            textAnchor="middle",
        )
    )
    drawing.add(
        String(
            20,
            10,
            f"Filed {memo.filing_date}",
            fontName="Helvetica",
            fontSize=9,
            fillColor=PDF_MUTED,
        )
    )
    return drawing


def _build_cover_page_story(memo: CreditMemoResult, styles: dict[str, ParagraphStyle]) -> list[Any]:
    company_block = [
        Paragraph("CREDIT RISK MEMO", styles["cover_title"]),
        Paragraph(f"{_escape_pdf_text(memo.company_name)} ({_escape_pdf_text(memo.ticker)})", styles["cover_company"]),
        Paragraph(
            "Institutional-format credit review generated from the issuer's most recent Form 10-K filing.",
            styles["body"],
        ),
        Paragraph(f"Filing Date: {_escape_pdf_text(memo.filing_date)}", styles["body_muted"]),
    ]

    cover_layout = Table(
        [[company_block, _build_cover_rating_drawing(memo)]],
        colWidths=[4.2 * inch, 2.3 * inch],
    )
    cover_layout.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )

    return [
        Spacer(1, 1.2 * inch),
        cover_layout,
    ]


def _build_metric_chart(memo: CreditMemoResult) -> Drawing:
    summary = _summary_lookup(memo)
    metric_specs = [
        ("Revenue", "Revenue"),
        ("Net Income", "Net Income"),
        ("Total Debt", "Total Debt"),
        ("Cash", "Cash and Cash Equivalents"),
        ("Operating Cash Flow", "Operating Cash Flow"),
    ]

    raw_values = [
        _parse_display_value(summary.get(summary_label))
        for _, summary_label in metric_specs
    ]
    usable_values = [value for value in raw_values if value is not None]
    divisor, scale_label = _choose_financial_scale(usable_values or [1.0])
    scaled_values = [0.0 if value is None else value / divisor for value in raw_values]

    chart = HorizontalBarChart()
    chart.x = 100
    chart.y = 24
    chart.width = 330
    chart.height = 140
    chart.data = [scaled_values]
    chart.categoryAxis.categoryNames = [label for label, _ in metric_specs]
    chart.categoryAxis.reverseDirection = True
    chart.categoryAxis.labels.fontName = "Helvetica"
    chart.categoryAxis.labels.fontSize = 8
    chart.categoryAxis.labels.fillColor = PDF_TEXT
    chart.valueAxis.valueMin = 0
    chart.valueAxis.valueMax = max(max(scaled_values, default=1.0) * 1.15, 1.0)
    chart.valueAxis.valueStep = max(chart.valueAxis.valueMax / 4, 0.25)
    chart.valueAxis.labels.fontName = "Helvetica"
    chart.valueAxis.labels.fontSize = 7
    chart.valueAxis.labels.fillColor = PDF_MUTED
    chart.valueAxis.labelTextFormat = _format_axis_tick
    chart.valueAxis.strokeColor = PDF_BORDER
    chart.valueAxis.gridStrokeColor = colors.HexColor("#E3E8EF")
    chart.valueAxis.gridStrokeWidth = 0.35
    chart.categoryAxis.strokeColor = PDF_BORDER
    chart.bars[0].fillColor = PDF_NAVY
    chart.bars[0].strokeColor = PDF_NAVY
    chart.barLabelFormat = lambda value: format_financial_value(value * divisor)
    chart.barLabels.fontName = "Helvetica"
    chart.barLabels.fontSize = 7
    chart.barLabels.fillColor = PDF_TEXT
    chart.barLabels.nudge = 8

    drawing = Drawing(470, 230)
    drawing.add(Rect(0, 0, 470, 230, fillColor=PDF_WHITE, strokeColor=PDF_BORDER, strokeWidth=0.5))
    drawing.add(chart)
    drawing.add(
        String(
            12,
            212,
            "CHART 1. KEY FINANCIAL METRICS",
            fontName="Helvetica-Bold",
            fontSize=10,
            fillColor=PDF_NAVY,
        )
    )
    drawing.add(
        String(
            12,
            198,
            f"Revenue, earnings, debt, liquidity, and operating cash flow shown in {scale_label.lower()}.",
            fontName="Helvetica",
            fontSize=7.8,
            fillColor=PDF_MUTED,
        )
    )
    return drawing


def _build_financial_summary_table(
    memo: CreditMemoResult,
    styles: dict[str, ParagraphStyle],
) -> Table:
    section_map = {
        "Income Statement": {
            "Revenue",
            "Net Income",
            "Interest Expense",
            "EBIT",
            "EBITDA",
        },
        "Balance Sheet": {
            "Total Assets",
            "Total Current Assets",
            "Total Current Liabilities",
            "Total Liabilities",
            "Total Shareholders' Equity",
            "Total Debt",
            "Cash and Cash Equivalents",
        },
        "Cash Flow": {
            "Operating Cash Flow",
            "Capital Expenditures",
        },
        "Credit Ratios": {
            "Debt-to-Equity",
            "Interest Coverage",
            "Current Ratio",
            "Net Debt / EBITDA",
            "Free Cash Flow Yield",
        },
    }

    table_data: list[list[Any]] = [
        [
            Paragraph("Metric", styles["table_header"]),
            Paragraph("Value", styles["table_header"]),
            Paragraph("Evidence", styles["table_header"]),
        ]
    ]
    table_styles: list[tuple[Any, ...]] = [
        ("BACKGROUND", (0, 0), (-1, 0), PDF_NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), PDF_WHITE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("BOX", (0, 0), (-1, -1), 0.35, PDF_BORDER),
        ("LINEBELOW", (0, 0), (-1, 0), 0.35, PDF_BORDER),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
    ]
    current_row_index = 1
    data_row_index = 0

    for section_title, labels in section_map.items():
        matching_rows = [row for row in memo.financial_summary if row.label in labels]
        if not matching_rows:
            continue

        table_data.append(
            [
                Paragraph(_escape_pdf_text(section_title), styles["table_group"]),
                "",
                "",
            ]
        )
        table_styles.extend(
            [
                ("SPAN", (0, current_row_index), (-1, current_row_index)),
                ("BACKGROUND", (0, current_row_index), (-1, current_row_index), colors.HexColor("#EAF0F7")),
                ("LINEABOVE", (0, current_row_index), (-1, current_row_index), 0.35, PDF_BORDER),
            ]
        )
        current_row_index += 1

        for row in matching_rows:
            background_color = PDF_WHITE if data_row_index % 2 == 0 else PDF_TABLE_ALT
            table_data.append(
                [
                    Paragraph(_escape_pdf_text(row.label), styles["table_cell"]),
                    Paragraph(_escape_pdf_text(row.value), styles["table_cell_value"]),
                    Paragraph(_escape_pdf_text(row.evidence), styles["table_cell"]),
                ]
            )
            table_styles.extend(
                [
                    ("BACKGROUND", (0, current_row_index), (-1, current_row_index), background_color),
                    ("LINEBELOW", (0, current_row_index), (-1, current_row_index), 0.25, PDF_BORDER),
                ]
            )
            current_row_index += 1
            data_row_index += 1

    summary_table = Table(
        table_data,
        colWidths=[1.65 * inch, 0.95 * inch, 3.9 * inch],
        repeatRows=1,
    )
    summary_table.setStyle(TableStyle(table_styles))
    return summary_table


def _build_status_dot(color: colors.Color) -> Drawing:
    drawing = Drawing(12, 12)
    drawing.add(Circle(6, 6, 4, fillColor=color, strokeColor=color))
    return drawing


def _build_ratio_scorecard_table(styles: dict[str, ParagraphStyle], memo: CreditMemoResult) -> Table:
    summary = _summary_lookup(memo)
    panels: list[Table] = []

    for label in PDF_RATIO_THRESHOLDS:
        value = _parse_display_value(summary.get(label))
        status_text, status_color, healthy_text = _get_ratio_status(label, value)
        status_row = Table(
            [
                [
                    _build_status_dot(status_color),
                    Paragraph(status_text, styles["scorecard_status"]),
                ]
            ]
        )
        status_row.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ]
            )
        )

        panel = Table(
            [
                [Paragraph(_escape_pdf_text(label), styles["scorecard_ratio"])],
                [Paragraph(_escape_pdf_text(summary.get(label, "N/A")), styles["scorecard_value"])],
                [status_row],
                [Paragraph(_escape_pdf_text(healthy_text), styles["caption"])],
            ],
            colWidths=[3.1 * inch],
        )
        panel.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), PDF_WHITE),
                    ("BOX", (0, 0), (-1, -1), 0.4, PDF_BORDER),
                    ("LEFTPADDING", (0, 0), (-1, -1), 10),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ]
            )
        )
        panels.append(panel)

    table = Table(
        [
            [panels[0], panels[1]],
            [panels[2], panels[3]],
        ],
        colWidths=[3.2 * inch, 3.2 * inch],
        rowHeights=[1.38 * inch, 1.38 * inch],
    )
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    return table


def _build_points_flowables(
    title: str,
    points: tuple[MemoPoint, ...],
    styles: dict[str, ParagraphStyle],
) -> list[Any]:
    flowables: list[Any] = [
        _build_section_header(
            title,
            styles,
            subtitle="Issuer-specific factors supported by filing evidence.",
        ),
        Spacer(1, 0.12 * inch),
    ]
    for index, point in enumerate(points, start=1):
        flowables.append(
            Paragraph(f"{index}. {_escape_pdf_text(point.title)}", styles["point_title"])
        )
        flowables.append(Paragraph(_escape_pdf_text(point.explanation), styles["body_small"]))
        flowables.append(Paragraph(_escape_pdf_text(point.evidence), styles["body_italic"]))
        flowables.append(Spacer(1, 0.06 * inch))
    return flowables


def _build_executive_summary_box(memo: CreditMemoResult, styles: dict[str, ParagraphStyle]) -> Table:
    primary_strength = memo.key_strengths[0]
    primary_risk = memo.key_risk_factors[0]
    rows = [
        [Paragraph("Executive Summary", styles["summary_title"])],
        [
            Paragraph(
                f"The issuer is assessed at <b>{_escape_pdf_text(memo.credit_risk_rating.rating)}</b> with a "
                f"<b>{_escape_pdf_text(memo.outlook.direction)}</b> outlook.",
                styles["body_small"],
            )
        ],
        [
            Paragraph(
                f"Primary strength: <b>{_escape_pdf_text(primary_strength.title)}</b>. "
                f"{_escape_pdf_text(_first_sentence(primary_strength.explanation))}",
                styles["body_small"],
            )
        ],
        [
            Paragraph(
                f"Primary risk: <b>{_escape_pdf_text(primary_risk.title)}</b>. "
                f"{_escape_pdf_text(_first_sentence(primary_risk.explanation))}",
                styles["body_small"],
            )
        ],
    ]
    box = Table(rows, colWidths=[PDF_CONTENT_WIDTH])
    box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PDF_SUMMARY_BG),
                ("BOX", (0, 0), (-1, -1), 0.45, colors.HexColor("#B9CAE3")),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return box


def _build_peer_context_line(memo: CreditMemoResult, styles: dict[str, ParagraphStyle]) -> Paragraph:
    percentile = _rating_percentile(memo.credit_risk_rating.rating)
    base_rating = _base_rating_key(memo.credit_risk_rating.rating)
    issuer_reference = memo.company_name.split()[0].title()
    return Paragraph(
        (
            "For reference, the median credit rating for S&amp;P 500 companies is BBB+. "
            f"{html.escape(issuer_reference)}'s {base_rating} rating places it in the "
            f"{percentile}th percentile of investment-grade issuers."
        ),
        styles["body_muted"],
    )


def _build_ratio_benchmark_chart(memo: CreditMemoResult) -> Drawing:
    summary = _summary_lookup(memo)
    ratio_labels = list(PDF_RATIO_BENCHMARKS.keys())
    short_labels = ["D/E", "Coverage", "Current", "ND/EBITDA"]
    benchmark_percentages: list[float] = []
    for label in ratio_labels:
        value = _parse_display_value(summary.get(label))
        benchmark = PDF_RATIO_BENCHMARKS[label]
        benchmark_percentages.append(0.0 if value is None else (value / benchmark) * 100)
    maximum = max(benchmark_percentages + [100.0]) * 1.12

    chart = VerticalBarChart()
    chart.x = 42
    chart.y = 38
    chart.width = 385
    chart.height = 138
    chart.data = [benchmark_percentages]
    chart.categoryAxis.categoryNames = short_labels
    chart.categoryAxis.labels.fontName = "Helvetica"
    chart.categoryAxis.labels.fontSize = 8
    chart.categoryAxis.labels.fillColor = PDF_TEXT
    chart.valueAxis.valueMin = 0
    chart.valueAxis.valueMax = max(maximum, 120.0)
    chart.valueAxis.valueStep = max(chart.valueAxis.valueMax / 4, 25.0)
    chart.valueAxis.labels.fontName = "Helvetica"
    chart.valueAxis.labels.fontSize = 7
    chart.valueAxis.labels.fillColor = PDF_MUTED
    chart.valueAxis.labelTextFormat = lambda value: f"{value:.0f}%"
    chart.valueAxis.gridStrokeColor = colors.HexColor("#E3E8EF")
    chart.valueAxis.gridStrokeWidth = 0.35
    chart.valueAxis.strokeColor = PDF_BORDER
    chart.categoryAxis.strokeColor = PDF_BORDER
    chart.bars[0].fillColor = PDF_NAVY
    chart.bars[0].strokeColor = PDF_NAVY
    chart.barLabelFormat = lambda value: f"{value:.0f}%"
    chart.barLabels.fontName = "Helvetica"
    chart.barLabels.fontSize = 7
    chart.barLabels.fillColor = PDF_TEXT
    chart.barLabels.nudge = 6
    chart.groupSpacing = 18
    chart.barSpacing = 6

    drawing = Drawing(470, 230)
    drawing.add(Rect(0, 0, 470, 230, fillColor=PDF_WHITE, strokeColor=PDF_BORDER, strokeWidth=0.5))
    drawing.add(chart)
    benchmark_y = chart.y + (chart.height * (100 / chart.valueAxis.valueMax))
    drawing.add(Line(chart.x, benchmark_y, chart.x + chart.width, benchmark_y, strokeColor=PDF_STEEL, strokeWidth=1))
    drawing.add(
        String(
            chart.x + chart.width + 6,
            benchmark_y - 2,
            "Benchmark",
            fontName="Helvetica",
            fontSize=7.5,
            fillColor=PDF_MUTED,
        )
    )
    drawing.add(
        String(
            12,
            212,
            "CHART 3. CREDIT RATIO PERFORMANCE VS BENCHMARK",
            fontName="Helvetica-Bold",
            fontSize=10,
            fillColor=PDF_NAVY,
        )
    )
    drawing.add(
        String(
            12,
            198,
            "Each bar shows company ratio as a percentage of the general investment-grade benchmark.",
            fontName="Helvetica",
            fontSize=7.8,
            fillColor=PDF_MUTED,
        )
    )
    drawing.add(
        String(
            12,
            12,
            "D/E = Debt-to-Equity | ND/EBITDA = Net Debt / EBITDA",
            fontName="Helvetica",
            fontSize=7.5,
            fillColor=PDF_MUTED,
        )
    )
    return drawing


def _get_quantitative_status_color(status: str) -> colors.Color:
    normalized = status.strip().lower()
    if "safe" in normalized or "strong" in normalized:
        return PDF_GREEN
    if "gray" in normalized or "moderate" in normalized or "adequate" in normalized:
        return PDF_YELLOW
    if "distress" in normalized or "weak" in normalized:
        return PDF_RED
    return PDF_MUTED


def _color_to_hex(color: colors.Color) -> str:
    return "#{:02X}{:02X}{:02X}".format(
        int(color.red * 255),
        int(color.green * 255),
        int(color.blue * 255),
    )


def _build_altman_breakdown_table(
    quantitative_result: QuantitativeModelResult,
    styles: dict[str, ParagraphStyle],
) -> Table:
    table_data: list[list[Any]] = [
        [
            Paragraph("Component", styles["table_header"]),
            Paragraph("Raw Value", styles["table_header"]),
            Paragraph("Weighted", styles["table_header"]),
            Paragraph("Interpretation", styles["table_header"]),
        ]
    ]
    table_styles: list[tuple[Any, ...]] = [
        ("BACKGROUND", (0, 0), (-1, 0), PDF_NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), PDF_WHITE),
        ("BOX", (0, 0), (-1, -1), 0.35, PDF_BORDER),
        ("LINEBELOW", (0, 0), (-1, 0), 0.35, PDF_BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (1, 1), (2, -1), "RIGHT"),
    ]

    for row_index, component in enumerate(quantitative_result.altman_z_score.components, start=1):
        background_color = PDF_WHITE if row_index % 2 == 1 else PDF_TABLE_ALT
        table_data.append(
            [
                Paragraph(_escape_pdf_text(component.label), styles["table_cell"]),
                Paragraph("N/A" if component.raw_value is None else f"{component.raw_value:.2f}", styles["table_cell_value"]),
                Paragraph("N/A" if component.weighted_contribution is None else f"{component.weighted_contribution:.2f}", styles["table_cell_value"]),
                Paragraph(_escape_pdf_text(component.description), styles["table_cell"]),
            ]
        )
        table_styles.extend(
            [
                ("BACKGROUND", (0, row_index), (-1, row_index), background_color),
                ("LINEBELOW", (0, row_index), (-1, row_index), 0.25, PDF_BORDER),
            ]
        )

    table = Table(
        table_data,
        colWidths=[2.2 * inch, 0.75 * inch, 0.85 * inch, 2.6 * inch],
        repeatRows=1,
    )
    table.setStyle(TableStyle(table_styles))
    return table


def _build_piotroski_visual(
    quantitative_result: QuantitativeModelResult,
    styles: dict[str, ParagraphStyle],
) -> Table:
    score = quantitative_result.piotroski_f_score.score
    interpretation = quantitative_result.piotroski_f_score.interpretation
    score_color = _get_quantitative_status_color(interpretation)
    score_text = "N/A"
    if score is not None and quantitative_result.piotroski_f_score.evaluated_count > 0:
        score_text = f"{score}/{quantitative_result.piotroski_f_score.evaluated_count}"
    header = Table(
        [
            [
                Paragraph("Piotroski F-Score", styles["summary_title"]),
                Paragraph(score_text, styles["scorecard_value"]),
                Paragraph(_escape_pdf_text(interpretation), styles["scorecard_status"]),
            ]
        ],
        colWidths=[1.8 * inch, 0.8 * inch, 1.0 * inch],
    )
    header.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PDF_WHITE),
                ("BOX", (0, 0), (-1, -1), 0.35, PDF_BORDER),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("ALIGN", (1, 0), (1, 0), "RIGHT"),
                ("TEXTCOLOR", (2, 0), (2, 0), score_color),
            ]
        )
    )

    cells: list[Any] = []
    if quantitative_result.piotroski_f_score.criteria:
        for criterion in quantitative_result.piotroski_f_score.criteria:
            if criterion.passed is None:
                fill_color = colors.HexColor("#2B4C7E")
                box_value = "?"
                label_text = f"{_escape_pdf_text(criterion.label)}<br/>Insufficient data"
            elif criterion.passed:
                fill_color = PDF_GREEN
                box_value = "1"
                label_text = _escape_pdf_text(criterion.label)
            else:
                fill_color = colors.HexColor("#6E7886")
                box_value = "0"
                label_text = _escape_pdf_text(criterion.label)
            cell = Table(
                [
                    [Paragraph(box_value, styles["table_cell_center"])],
                    [Paragraph(label_text, styles["caption"])],
                ],
                colWidths=[0.68 * inch],
            )
            cell.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (0, 0), fill_color),
                        ("TEXTCOLOR", (0, 0), (0, 0), PDF_WHITE),
                        ("BOX", (0, 0), (-1, -1), 0.3, PDF_BORDER),
                        ("LEFTPADDING", (0, 0), (-1, -1), 4),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                        ("TOPPADDING", (0, 0), (-1, -1), 4),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ]
                )
            )
            cells.append(cell)
    else:
        cells.append(Paragraph(_escape_pdf_text(quantitative_result.piotroski_f_score.note or "Prior year data unavailable."), styles["body_small"]))

    grid = Table([cells], colWidths=[0.72 * inch] * max(len(cells), 1))
    grid.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    note_row = Paragraph(
        _escape_pdf_text(
            quantitative_result.piotroski_f_score.note
            or "Nine-point financial health score using profitability, leverage, liquidity, and operating efficiency tests."
        ),
        styles["body_small"],
    )
    return Table([[header], [Spacer(1, 0.08 * inch)], [grid], [Spacer(1, 0.08 * inch)], [note_row]], colWidths=[PDF_CONTENT_WIDTH])


def _build_cash_flow_adequacy_panel(
    quantitative_result: QuantitativeModelResult,
    styles: dict[str, ParagraphStyle],
) -> Table:
    adequacy = quantitative_result.cash_flow_adequacy
    value_text = "N/A" if adequacy.ratio is None else f"{adequacy.ratio:.2f}x"
    assessment_color = _get_quantitative_status_color(adequacy.assessment)
    panel = Table(
        [
            [Paragraph("Cash Flow Adequacy", styles["summary_title"])],
            [Paragraph(_escape_pdf_text(value_text), styles["scorecard_value"])],
            [Paragraph(_escape_pdf_text(adequacy.assessment), styles["scorecard_status"])],
            [Paragraph(_escape_pdf_text(adequacy.note or "Operating cash flow divided by near-term obligations, interest expense, and capital expenditures."), styles["body_small"])],
        ],
        colWidths=[PDF_CONTENT_WIDTH],
    )
    panel.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PDF_WHITE),
                ("BOX", (0, 0), (-1, -1), 0.4, PDF_BORDER),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("TEXTCOLOR", (0, 2), (0, 2), assessment_color),
            ]
        )
    )
    return panel


def _format_trend_table_value(row: TrendAnalysisRow, attribute_name: str) -> str:
    value = getattr(row, attribute_name)
    if attribute_name in {"prior_value", "absolute_change"} and not row.prior_data_reliable:
        return "N/A"
    if row.is_ratio:
        return "N/A" if value is None else f"{value:.2f}x"
    return format_financial_value(value)


def _build_trend_analysis_table(
    quantitative_result: QuantitativeModelResult,
    styles: dict[str, ParagraphStyle],
) -> Table:
    rows = quantitative_result.trend_analysis
    if not rows:
        return Table([[Paragraph(_escape_pdf_text(quantitative_result.trend_analysis_note or "Trend analysis unavailable."), styles["body_small"])]], colWidths=[PDF_CONTENT_WIDTH])

    table_data: list[list[Any]] = [
        [
            Paragraph("Metric", styles["table_header"]),
            Paragraph("Current", styles["table_header"]),
            Paragraph("Prior", styles["table_header"]),
            Paragraph("Change", styles["table_header"]),
            Paragraph("% Change", styles["table_header"]),
            Paragraph("Direction", styles["table_header"]),
        ]
    ]
    table_styles: list[tuple[Any, ...]] = [
        ("BACKGROUND", (0, 0), (-1, 0), PDF_NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), PDF_WHITE),
        ("BOX", (0, 0), (-1, -1), 0.35, PDF_BORDER),
        ("LINEBELOW", (0, 0), (-1, 0), 0.35, PDF_BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 1), (5, -1), "CENTER"),
    ]
    for row_index, row in enumerate(rows, start=1):
        background = PDF_WHITE if row_index % 2 == 1 else PDF_TABLE_ALT
        arrow = "N/A" if not row.prior_data_reliable else "→"
        arrow_color = PDF_MUTED
        if row.prior_data_reliable and row.improving is True:
            arrow = "↑"
            arrow_color = PDF_GREEN
        elif row.prior_data_reliable and row.improving is False:
            arrow = "↓"
            arrow_color = PDF_RED

        metric_text = _escape_pdf_text(row.label)
        if row.note:
            metric_text = (
                f"{metric_text}<br/><font size=\"7\" color=\"{_color_to_hex(PDF_MUTED)}\">"
                f"{_escape_pdf_text(row.note)}</font>"
            )

        table_data.append(
            [
                Paragraph(metric_text, styles["table_cell"]),
                Paragraph(_escape_pdf_text(_format_trend_table_value(row, "current_value")), styles["table_cell_value"]),
                Paragraph(_escape_pdf_text(_format_trend_table_value(row, "prior_value")), styles["table_cell_value"]),
                Paragraph(_escape_pdf_text(_format_trend_table_value(row, "absolute_change")), styles["table_cell_value"]),
                Paragraph("N/A" if row.percent_change is None else f"{row.percent_change:.1%}", styles["table_cell_center"]),
                Paragraph(f'<font color="{_color_to_hex(arrow_color)}">{arrow}</font>', styles["table_cell_center"]),
            ]
        )
        table_styles.extend(
            [
                ("BACKGROUND", (0, row_index), (-1, row_index), background),
                ("LINEBELOW", (0, row_index), (-1, row_index), 0.25, PDF_BORDER),
            ]
        )
    table = Table(
        table_data,
        colWidths=[1.8 * inch, 0.75 * inch, 0.75 * inch, 0.75 * inch, 0.75 * inch, 0.55 * inch],
        repeatRows=1,
    )
    table.setStyle(TableStyle(table_styles))
    return table


def _build_debt_maturity_chart(
    profile: DebtMaturityProfileResult,
) -> Drawing:
    year_totals: dict[int, float] = {}
    for tranche in profile.tranches:
        year_totals[tranche.maturity_year] = year_totals.get(tranche.maturity_year, 0.0) + tranche.amount_millions

    chart = VerticalBarChart()
    chart.x = 42
    chart.y = 34
    chart.width = 385
    chart.height = 138
    chart.data = [list(year_totals.values()) or [0.0]]
    chart.categoryAxis.categoryNames = [str(year) for year in year_totals] or ["N/A"]
    chart.categoryAxis.labels.fontName = "Helvetica"
    chart.categoryAxis.labels.fontSize = 8
    chart.categoryAxis.labels.fillColor = PDF_TEXT
    chart.valueAxis.valueMin = 0
    chart.valueAxis.valueMax = max([1.0] + list(year_totals.values())) * 1.2
    chart.valueAxis.valueStep = max(chart.valueAxis.valueMax / 4, 25.0)
    chart.valueAxis.labels.fontName = "Helvetica"
    chart.valueAxis.labels.fontSize = 7
    chart.valueAxis.labels.fillColor = PDF_MUTED
    chart.valueAxis.labelTextFormat = _format_axis_tick
    chart.valueAxis.gridStrokeColor = colors.HexColor("#E3E8EF")
    chart.valueAxis.gridStrokeWidth = 0.35
    chart.valueAxis.strokeColor = PDF_BORDER
    chart.categoryAxis.strokeColor = PDF_BORDER
    chart.bars[0].fillColor = PDF_NAVY
    chart.bars[0].strokeColor = PDF_NAVY
    chart.barLabelFormat = lambda value: format_financial_value(value * 1_000_000)
    chart.barLabels.fontName = "Helvetica"
    chart.barLabels.fontSize = 7
    chart.barLabels.fillColor = PDF_TEXT
    chart.barLabels.nudge = 6

    drawing = Drawing(470, 220)
    drawing.add(Rect(0, 0, 470, 220, fillColor=PDF_WHITE, strokeColor=PDF_BORDER, strokeWidth=0.5))
    drawing.add(chart)
    drawing.add(
        String(
            12,
            202,
            "DEBT MATURITY PROFILE",
            fontName="Helvetica-Bold",
            fontSize=10,
            fillColor=PDF_NAVY,
        )
    )
    drawing.add(
        String(
            12,
            188,
            "Debt maturing by calendar year, shown in normalized dollars.",
            fontName="Helvetica",
            fontSize=7.8,
            fillColor=PDF_MUTED,
        )
    )
    return drawing


def _build_debt_maturity_table(
    profile: DebtMaturityProfileResult,
    styles: dict[str, ParagraphStyle],
) -> Table:
    if not profile.tranches:
        return Table([[Paragraph(_escape_pdf_text(profile.note or "No specific maturity schedule identified in filing."), styles["body_small"])]], colWidths=[PDF_CONTENT_WIDTH])

    table_data: list[list[Any]] = [
        [
            Paragraph("Amount", styles["table_header"]),
            Paragraph("Year", styles["table_header"]),
            Paragraph("Rate", styles["table_header"]),
            Paragraph("Description", styles["table_header"]),
        ]
    ]
    table_styles: list[tuple[Any, ...]] = [
        ("BACKGROUND", (0, 0), (-1, 0), PDF_NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), PDF_WHITE),
        ("BOX", (0, 0), (-1, -1), 0.35, PDF_BORDER),
        ("LINEBELOW", (0, 0), (-1, 0), 0.35, PDF_BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (0, 1), (1, -1), "RIGHT"),
    ]
    for row_index, tranche in enumerate(profile.tranches, start=1):
        background = PDF_WHITE if row_index % 2 == 1 else PDF_TABLE_ALT
        table_data.append(
            [
                Paragraph(format_financial_value(tranche.amount_millions * 1_000_000), styles["table_cell_value"]),
                Paragraph(str(tranche.maturity_year), styles["table_cell_value"]),
                Paragraph(_escape_pdf_text(tranche.interest_rate or "N/A"), styles["table_cell_center"]),
                Paragraph(_escape_pdf_text(tranche.description), styles["table_cell"]),
            ]
        )
        table_styles.extend(
            [
                ("BACKGROUND", (0, row_index), (-1, row_index), background),
                ("LINEBELOW", (0, row_index), (-1, row_index), 0.25, PDF_BORDER),
            ]
        )

    table = Table(
        table_data,
        colWidths=[0.9 * inch, 0.65 * inch, 0.8 * inch, 4.05 * inch],
        repeatRows=1,
    )
    table.setStyle(TableStyle(table_styles))
    return table


def _draw_page_chrome(
    pdf_canvas: Canvas,
    *,
    page_number: int,
    total_pages: int,
    memo: CreditMemoResult,
    generated_on: str,
) -> None:
    page_width, page_height = LETTER
    pdf_canvas.saveState()

    pdf_canvas.setFillColor(PDF_STEEL)
    pdf_canvas.rect(0, page_height - PDF_HEADER_ACCENT_HEIGHT, page_width, PDF_HEADER_ACCENT_HEIGHT, stroke=0, fill=1)

    if page_number == 1:
        pdf_canvas.setFillColor(PDF_NAVY)
        pdf_canvas.rect(0, page_height - 0.9 * inch, page_width, 0.78 * inch, stroke=0, fill=1)
        pdf_canvas.setFillColor(PDF_WHITE)
        pdf_canvas.setFont("Helvetica-Bold", 11)
        pdf_canvas.drawString(PDF_PAGE_MARGIN, page_height - 0.5 * inch, memo.company_name)
        pdf_canvas.setFillColor(colors.HexColor("#D7DEE7"))
        pdf_canvas.setFont("Helvetica-Bold", 8)
        pdf_canvas.drawRightString(page_width - PDF_PAGE_MARGIN, page_height - 0.5 * inch, "AI CREDIT ANALYST AGENT")

        pdf_canvas.setFillColor(PDF_MUTED)
        pdf_canvas.setFont("Helvetica", 8)
        pdf_canvas.drawCentredString(
            page_width / 2,
            0.76 * inch,
            f"AI Credit Analyst Agent | Generated {generated_on} | Data sourced from SEC EDGAR",
        )
        pdf_canvas.setFillColor(PDF_CONFIDENTIAL)
        pdf_canvas.setFont("Helvetica-Bold", 7.5)
        pdf_canvas.drawCentredString(
            page_width / 2,
            0.56 * inch,
            "CONFIDENTIAL — FOR EDUCATIONAL PURPOSES ONLY",
        )
        pdf_canvas.setFillColor(PDF_MUTED)
        pdf_canvas.setFont("Helvetica", 6.8)
        pdf_canvas.drawCentredString(
            page_width / 2,
            0.38 * inch,
            "This report was generated by an AI system and is intended for educational purposes only.",
        )
        pdf_canvas.drawCentredString(
            page_width / 2,
            0.25 * inch,
            "It does not constitute financial advice.",
        )
    else:
        pdf_canvas.setStrokeColor(PDF_BORDER)
        pdf_canvas.setLineWidth(0.4)
        pdf_canvas.line(PDF_PAGE_MARGIN, page_height - 0.72 * inch, page_width - PDF_PAGE_MARGIN, page_height - 0.72 * inch)
        pdf_canvas.setFillColor(PDF_NAVY)
        pdf_canvas.setFont("Helvetica-Bold", 10)
        pdf_canvas.drawString(PDF_PAGE_MARGIN, page_height - 0.56 * inch, f"{memo.company_name} ({memo.ticker})")
        pdf_canvas.setFillColor(colors.HexColor("#AAB3BF"))
        pdf_canvas.setFont("Helvetica-Bold", 8)
        pdf_canvas.drawRightString(page_width - PDF_PAGE_MARGIN, page_height - 0.56 * inch, "AI CREDIT ANALYST AGENT")
        pdf_canvas.setFillColor(PDF_CONFIDENTIAL)
        pdf_canvas.setFont("Helvetica-Bold", 7.5)
        pdf_canvas.drawString(PDF_PAGE_MARGIN, 0.42 * inch, "CONFIDENTIAL — FOR EDUCATIONAL PURPOSES ONLY")

    pdf_canvas.setFillColor(PDF_MUTED)
    pdf_canvas.setFont("Helvetica", 8)
    pdf_canvas.drawRightString(
        page_width - PDF_PAGE_MARGIN,
        0.42 * inch,
        f"Page {page_number} of {total_pages}",
    )
    pdf_canvas.restoreState()


class _NumberedCanvas(Canvas):
    def __init__(self, *args: Any, memo: CreditMemoResult, generated_on: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._saved_page_states: list[dict[str, Any]] = []
        self._memo = memo
        self._generated_on = generated_on

    def showPage(self) -> None:
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self) -> None:
        total_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            _draw_page_chrome(
                self,
                page_number=self._pageNumber,
                total_pages=total_pages,
                memo=self._memo,
                generated_on=self._generated_on,
            )
            Canvas.showPage(self)
        Canvas.save(self)


def build_credit_memo_pdf(
    memo: CreditMemoResult,
    *,
    filing_url: str | None = None,
    extraction_methodology: str | None = None,
    quantitative_result: QuantitativeModelResult | None = None,
) -> bytes:
    try:
        buffer = io.BytesIO()
        document = SimpleDocTemplate(
            buffer,
            pagesize=LETTER,
            rightMargin=1.0 * inch,
            leftMargin=1.0 * inch,
            topMargin=1.0 * inch,
            bottomMargin=1.0 * inch,
            title=f"{memo.company_name} Credit Risk Memo",
            author="AI Credit Analyst Agent",
            pageCompression=0,
        )

        styles = _build_pdf_styles()
        generated_on = date.today().strftime("%B %d, %Y")
        summary = _summary_lookup(memo)
        methodology_text = extraction_methodology or (
            "Raw 10-K text retrieved from SEC EDGAR, relevant financial statement sections isolated, "
            "values normalized to dollars, and standard credit ratios computed deterministically from the extracted results."
        )

        story: list[Any] = []
        story.extend(_build_cover_page_story(memo, styles))
        story.append(PageBreak())

        story.append(
            _build_section_header(
                "Financial Overview",
                styles,
                subtitle="Executive takeaway and statement-level financial view.",
            )
        )
        story.append(Spacer(1, 0.18 * inch))
        story.append(_build_executive_summary_box(memo, styles))
        story.append(Spacer(1, 0.18 * inch))
        story.append(Paragraph(_escape_pdf_text(memo.company_overview), styles["body"]))
        story.append(Spacer(1, 0.08 * inch))
        story.append(_build_section_divider())
        story.append(Spacer(1, 0.14 * inch))
        story.append(_build_metric_chart(memo))
        story.append(Spacer(1, 0.2 * inch))
        story.append(_build_financial_summary_table(memo, styles))
        story.append(PageBreak())

        story.append(
            _build_section_header(
                "Credit Analysis",
                styles,
                subtitle="Rating rationale and ratio scorecard against internal thresholds.",
            )
        )
        story.append(Spacer(1, 0.18 * inch))
        story.append(
            Paragraph(
                f"{_escape_pdf_text(memo.credit_risk_rating.rating)}",
                styles["rating_display"],
            )
        )
        story.append(Paragraph(_escape_pdf_text(memo.credit_risk_rating.justification), styles["body"]))
        story.append(_build_peer_context_line(memo, styles))
        story.append(Spacer(1, 0.1 * inch))
        story.append(_build_section_divider())
        story.append(Spacer(1, 0.16 * inch))
        story.append(_build_ratio_scorecard_table(styles, memo))
        story.append(Spacer(1, 0.16 * inch))
        story.append(
            Paragraph(
                "Scorecard thresholds mirror the application-level credit ratio calibration used in the memo workflow.",
                styles["caption"],
            )
        )

        if quantitative_result is not None:
            story.append(PageBreak())
            story.append(
                _build_section_header(
                    "Quantitative Risk Models",
                    styles,
                    subtitle="Altman Z-Score, Piotroski F-Score, and cash flow adequacy diagnostics.",
                )
            )
            story.append(Spacer(1, 0.16 * inch))
            altman_color = _get_quantitative_status_color(quantitative_result.altman_z_score.zone)
            story.append(
                Paragraph(
                    (
                        f"<b>Altman Z-Score:</b> {_escape_pdf_text('N/A' if quantitative_result.altman_z_score.score is None else f'{quantitative_result.altman_z_score.score:.2f}')}"
                        f" | <font color=\"{_color_to_hex(altman_color)}\">{_escape_pdf_text(quantitative_result.altman_z_score.zone)}</font>"
                    ),
                    styles["body"],
                )
            )
            if quantitative_result.altman_z_score.note:
                story.append(Paragraph(_escape_pdf_text(quantitative_result.altman_z_score.note), styles["body_small"]))
            story.append(_build_altman_breakdown_table(quantitative_result, styles))
            story.append(Spacer(1, 0.16 * inch))
            story.append(_build_piotroski_visual(quantitative_result, styles))
            story.append(Spacer(1, 0.16 * inch))
            story.append(_build_cash_flow_adequacy_panel(quantitative_result, styles))
            story.append(PageBreak())
            story.append(
                _build_section_header(
                    "Year-over-Year Trend Analysis",
                    styles,
                    subtitle="Key metric and ratio changes between the current and prior annual filings.",
                )
            )
            story.append(Spacer(1, 0.16 * inch))
            story.append(_build_trend_analysis_table(quantitative_result, styles))
            story.append(PageBreak())
            story.append(
                _build_section_header(
                    "Debt Maturity Profile",
                    styles,
                    subtitle="Debt maturities by year, weighted-average tenor, and maturity-wall flags.",
                )
            )
            story.append(Spacer(1, 0.16 * inch))
            if quantitative_result.debt_maturity_profile.tranches:
                story.append(_build_debt_maturity_chart(quantitative_result.debt_maturity_profile))
                story.append(Spacer(1, 0.16 * inch))
                if quantitative_result.debt_maturity_profile.weighted_average_maturity is not None:
                    story.append(
                        Paragraph(
                            f"<b>Weighted Average Maturity:</b> {quantitative_result.debt_maturity_profile.weighted_average_maturity:.1f} years",
                            styles["body_small"],
                        )
                    )
                if quantitative_result.debt_maturity_profile.note:
                    story.append(
                        Paragraph(
                            _escape_pdf_text(quantitative_result.debt_maturity_profile.note),
                            styles["body_small"],
                        )
                    )
                for warning in quantitative_result.debt_maturity_profile.warnings:
                    story.append(
                        Paragraph(
                            f"<b>Maturity Wall Warning:</b> {warning.percentage_of_total_debt:.0%} of total debt matures in {warning.maturity_year}.",
                            styles["body_small"],
                        )
                    )
            else:
                story.append(
                    Paragraph(
                        _escape_pdf_text(
                            quantitative_result.debt_maturity_profile.note
                            or "No specific maturity schedule identified in filing."
                        ),
                        styles["body_small"],
                    )
                )
            story.append(Spacer(1, 0.12 * inch))
            story.append(_build_debt_maturity_table(quantitative_result.debt_maturity_profile, styles))

        story.append(PageBreak())

        story.extend(_build_points_flowables("KEY RISK FACTORS", memo.key_risk_factors, styles))
        story.append(Spacer(1, 0.12 * inch))
        story.append(_build_section_divider())
        story.append(Spacer(1, 0.18 * inch))
        story.extend(_build_points_flowables("KEY STRENGTHS", memo.key_strengths, styles))
        story.append(PageBreak())

        story.append(
            _build_section_header(
                "Outlook & Appendix",
                styles,
                subtitle="Forward view, benchmark context, and report provenance.",
            )
        )
        story.append(Spacer(1, 0.18 * inch))
        story.append(
            Paragraph(
                f"<b>{_escape_pdf_text(memo.outlook.direction)}</b>: {_escape_pdf_text(memo.outlook.justification)}",
                styles["body"],
            )
        )
        story.append(Spacer(1, 0.08 * inch))
        story.append(_build_section_divider())
        story.append(Spacer(1, 0.14 * inch))
        story.append(_build_ratio_benchmark_chart(memo))
        story.append(Spacer(1, 0.2 * inch))
        story.append(
            _build_section_header(
                "Data Sources",
                styles,
                subtitle="Primary document link and extraction methodology.",
            )
        )
        story.append(Spacer(1, 0.16 * inch))
        story.append(
            Paragraph(
                f"<b>SEC filing URL:</b> {_wrap_url_for_paragraph(filing_url or 'Not provided')}",
                styles["body_small"],
            )
        )
        story.append(
            Paragraph(
                f"<b>Extraction methodology:</b> {_escape_pdf_text(methodology_text)}",
                styles["body_small"],
            )
        )
        story.append(
            Paragraph(
                f"<b>Appendix note:</b> Financial chart labels and tables are derived from the structured summary table included in the memo. "
                f"Unavailable values are shown as N/A. Actual ratio values used for benchmark comparison: "
                f"Debt-to-Equity {_escape_pdf_text(summary.get('Debt-to-Equity', 'N/A'))}, "
                f"Interest Coverage {_escape_pdf_text(summary.get('Interest Coverage', 'N/A'))}, "
                f"Current Ratio {_escape_pdf_text(summary.get('Current Ratio', 'N/A'))}, "
                f"Net Debt / EBITDA {_escape_pdf_text(summary.get('Net Debt / EBITDA', 'N/A'))}.",
                styles["body_small"],
            )
        )

        document.build(
            story,
            canvasmaker=lambda *args, **kwargs: _NumberedCanvas(
                *args,
                memo=memo,
                generated_on=generated_on,
                **kwargs,
            ),
        )
        return buffer.getvalue()
    except Exception as exc:
        raise PDFGenerationError(f"Failed to build credit memo PDF: {exc}") from exc


def _build_comparison_cover_story(
    memos: tuple[CreditMemoResult, ...],
    styles: dict[str, ParagraphStyle],
) -> list[Any]:
    company_names = ", ".join(f"{memo.company_name} ({memo.ticker})" for memo in memos)
    rating_summary = " | ".join(
        f"{memo.ticker}: {memo.credit_risk_rating.rating} / {memo.outlook.direction}"
        for memo in memos
    )
    return [
        Spacer(1, 1.15 * inch),
        Paragraph("PEER CREDIT COMPARISON", styles["cover_title"]),
        Paragraph(_escape_pdf_text(company_names), styles["cover_company"]),
        Paragraph(
            "Relative credit comparison generated from each issuer's most recent Form 10-K filing.",
            styles["body"],
        ),
        Paragraph(_escape_pdf_text(rating_summary), styles["body_muted"]),
    ]


def _build_comparison_rating_card(memo: CreditMemoResult, styles: dict[str, ParagraphStyle]) -> Table:
    rating_color = _get_rating_color(memo.credit_risk_rating.rating)
    outlook_color = _get_outlook_color(memo.outlook.direction)
    circle_font_size = 26 if len(memo.credit_risk_rating.rating) <= 3 else 22

    drawing = Drawing(172, 112)
    drawing.add(Circle(38, 56, 28, fillColor=rating_color, strokeColor=rating_color))
    drawing.add(
        String(
            38,
            47,
            memo.credit_risk_rating.rating,
            fontName="Helvetica-Bold",
            fontSize=circle_font_size,
            fillColor=PDF_WHITE,
            textAnchor="middle",
        )
    )
    badge_width = max(66, 16 + (len(memo.outlook.direction) * 6.5))
    drawing.add(
        Rect(
            82,
            44,
            badge_width,
            20,
            rx=8,
            ry=8,
            fillColor=outlook_color,
            strokeColor=outlook_color,
        )
    )
    drawing.add(
        String(
            82 + (badge_width / 2),
            50,
            memo.outlook.direction,
            fontName="Helvetica-Bold",
            fontSize=8.5,
            fillColor=PDF_WHITE,
            textAnchor="middle",
        )
    )

    card = Table(
        [
            [Paragraph(_escape_pdf_text(memo.company_name), styles["scorecard_ratio"])],
            [Paragraph(_escape_pdf_text(memo.ticker), styles["caption"])],
            [drawing],
            [Paragraph(_escape_pdf_text(memo.credit_risk_rating.justification), styles["body_small"])],
        ],
        colWidths=[2.06 * inch],
    )
    card.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PDF_WHITE),
                ("BOX", (0, 0), (-1, -1), 0.45, PDF_BORDER),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
            ]
        )
    )
    return card


def _build_failed_comparison_card(
    ticker: str,
    message: str,
    styles: dict[str, ParagraphStyle],
) -> Table:
    drawing = Drawing(172, 112)
    drawing.add(Circle(38, 56, 28, fillColor=PDF_RED, strokeColor=PDF_RED))
    drawing.add(
        String(
            38,
            47,
            "X",
            fontName="Helvetica-Bold",
            fontSize=28,
            fillColor=PDF_WHITE,
            textAnchor="middle",
        )
    )
    card = Table(
        [
            [Paragraph(_escape_pdf_text(ticker), styles["scorecard_ratio"])],
            [Paragraph("Unavailable", styles["caption"])],
            [drawing],
            [Paragraph(_escape_pdf_text(message), styles["body_small"])],
        ],
        colWidths=[2.06 * inch],
    )
    card.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PDF_WHITE),
                ("BOX", (0, 0), (-1, -1), 0.45, PDF_BORDER),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
            ]
        )
    )
    return card


def _build_comparison_rating_cards_table(
    memos: tuple[CreditMemoResult, ...],
    styles: dict[str, ParagraphStyle],
    failed_tickers: dict[str, str] | None = None,
) -> Table:
    cards: list[Any] = [_build_comparison_rating_card(memo, styles) for memo in memos]
    for ticker, message in (failed_tickers or {}).items():
        cards.append(_build_failed_comparison_card(ticker, message, styles))

    if not cards:
        cards.append(_build_failed_comparison_card("N/A", "No successful analyses available.", styles))

    while len(cards) < 3:
        cards.append(Table([[""]], colWidths=[2.06 * inch]))

    table = Table(
        [cards[:3]],
        colWidths=[2.1 * inch, 2.1 * inch, 2.1 * inch],
    )
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    return table


def _comparison_summary_value(memo: CreditMemoResult, label: str) -> str:
    return _summary_lookup(memo).get(label, "N/A")


def _build_comparison_metric_chart(memos: tuple[CreditMemoResult, ...]) -> Drawing:
    metric_labels = ["Revenue", "Net Income", "Total Debt", "Cash and Cash Equivalents"]
    category_names = ["Revenue", "Net Income", "Total Debt", "Cash"]
    palette = [PDF_NAVY, PDF_STEEL, PDF_BLUE]

    raw_values: list[list[float]] = []
    for memo in memos:
        memo_values = [_parse_display_value(_comparison_summary_value(memo, label)) or 0.0 for label in metric_labels]
        raw_values.append(memo_values)

    usable_values = [value for memo_values in raw_values for value in memo_values if value]
    divisor, scale_label = _choose_financial_scale(usable_values or [1.0])
    scaled_values = [[value / divisor for value in memo_values] for memo_values in raw_values]

    chart = VerticalBarChart()
    chart.x = 42
    chart.y = 38
    chart.width = 385
    chart.height = 140
    chart.data = scaled_values
    chart.categoryAxis.categoryNames = category_names
    chart.categoryAxis.labels.fontName = "Helvetica"
    chart.categoryAxis.labels.fontSize = 8
    chart.categoryAxis.labels.fillColor = PDF_TEXT
    chart.valueAxis.valueMin = 0
    chart.valueAxis.valueMax = max([1.0] + [value for series in scaled_values for value in series]) * 1.2
    chart.valueAxis.valueStep = max(chart.valueAxis.valueMax / 4, 0.25)
    chart.valueAxis.labels.fontName = "Helvetica"
    chart.valueAxis.labels.fontSize = 7
    chart.valueAxis.labels.fillColor = PDF_MUTED
    chart.valueAxis.labelTextFormat = _format_axis_tick
    chart.valueAxis.gridStrokeColor = colors.HexColor("#E3E8EF")
    chart.valueAxis.gridStrokeWidth = 0.35
    chart.valueAxis.strokeColor = PDF_BORDER
    chart.categoryAxis.strokeColor = PDF_BORDER
    chart.barSpacing = 4
    chart.groupSpacing = 12

    for index, memo in enumerate(memos):
        chart.bars[index].fillColor = palette[index % len(palette)]
        chart.bars[index].strokeColor = palette[index % len(palette)]
        chart.bars[index].name = memo.ticker

    chart.barLabelFormat = lambda value: format_financial_value(value * divisor)
    chart.barLabels.fontName = "Helvetica"
    chart.barLabels.fontSize = 6.6
    chart.barLabels.fillColor = PDF_TEXT
    chart.barLabels.nudge = 5

    drawing = Drawing(470, 240)
    drawing.add(Rect(0, 0, 470, 240, fillColor=PDF_WHITE, strokeColor=PDF_BORDER, strokeWidth=0.5))
    drawing.add(chart)
    drawing.add(
        String(
            12,
            220,
            "FINANCIAL SCALE COMPARISON",
            fontName="Helvetica-Bold",
            fontSize=10,
            fillColor=PDF_NAVY,
        )
    )
    drawing.add(
        String(
            12,
            206,
            f"Revenue, earnings, debt, and liquidity shown in {scale_label.lower()}.",
            fontName="Helvetica",
            fontSize=7.8,
            fillColor=PDF_MUTED,
        )
    )
    legend_x = 42
    for index, memo in enumerate(memos):
        color = palette[index % len(palette)]
        drawing.add(Rect(legend_x, 14, 10, 10, fillColor=color, strokeColor=color))
        drawing.add(
            String(
                legend_x + 14,
                15,
                memo.ticker,
                fontName="Helvetica-Bold",
                fontSize=7.5,
                fillColor=PDF_TEXT,
            )
        )
        legend_x += 74
    return drawing


def _build_comparison_financial_table(
    memos: tuple[CreditMemoResult, ...],
    styles: dict[str, ParagraphStyle],
) -> Table:
    metric_rows = [
        "Revenue",
        "Net Income",
        "Total Debt",
        "Cash and Cash Equivalents",
        "Operating Cash Flow",
        "EBITDA",
    ]
    table_data: list[list[Any]] = [[Paragraph("Metric", styles["table_header"])]]
    table_data[0].extend(
        Paragraph(_escape_pdf_text(f"{memo.company_name} ({memo.ticker})"), styles["table_header"])
        for memo in memos
    )

    table_styles: list[tuple[Any, ...]] = [
        ("BACKGROUND", (0, 0), (-1, 0), PDF_NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), PDF_WHITE),
        ("BOX", (0, 0), (-1, -1), 0.35, PDF_BORDER),
        ("LINEBELOW", (0, 0), (-1, 0), 0.35, PDF_BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
    ]

    for row_index, label in enumerate(metric_rows, start=1):
        background = PDF_WHITE if row_index % 2 == 1 else PDF_TABLE_ALT
        row = [Paragraph(_escape_pdf_text(label), styles["table_cell"])]
        row.extend(
            Paragraph(_escape_pdf_text(_comparison_summary_value(memo, label)), styles["table_cell_value"])
            for memo in memos
        )
        table_data.append(row)
        table_styles.extend(
            [
                ("BACKGROUND", (0, row_index), (-1, row_index), background),
                ("LINEBELOW", (0, row_index), (-1, row_index), 0.25, PDF_BORDER),
            ]
        )

    company_width = (PDF_CONTENT_WIDTH - 1.6 * inch) / max(len(memos), 1)
    financial_table = Table(
        table_data,
        colWidths=[1.6 * inch] + [company_width] * len(memos),
        repeatRows=1,
    )
    financial_table.setStyle(TableStyle(table_styles))
    return financial_table


def _build_comparison_ratio_table(
    memos: tuple[CreditMemoResult, ...],
    styles: dict[str, ParagraphStyle],
) -> Table:
    ratio_rows = [
        ("Debt-to-Equity", "Lower is better"),
        ("Interest Coverage", "Higher is better"),
        ("Current Ratio", "Higher is better"),
        ("Net Debt / EBITDA", "Lower is better"),
    ]
    table_data: list[list[Any]] = [
        [
            Paragraph("Ratio", styles["table_header"]),
            *[
                Paragraph(_escape_pdf_text(f"{memo.company_name} ({memo.ticker})"), styles["table_header"])
                for memo in memos
            ],
        ]
    ]
    table_styles: list[tuple[Any, ...]] = [
        ("BACKGROUND", (0, 0), (-1, 0), PDF_NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), PDF_WHITE),
        ("BOX", (0, 0), (-1, -1), 0.35, PDF_BORDER),
        ("LINEBELOW", (0, 0), (-1, 0), 0.35, PDF_BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
    ]

    for row_index, (label, _) in enumerate(ratio_rows, start=1):
        row = [Paragraph(_escape_pdf_text(label), styles["table_cell"])]
        for memo_index, memo in enumerate(memos, start=1):
            display_value = _comparison_summary_value(memo, label)
            value = _parse_display_value(display_value)
            status_text, status_color, healthy_text = _get_ratio_status(label, value)
            cell = Table(
                [
                    [Paragraph(_escape_pdf_text(display_value), styles["scorecard_value"])],
                    [Paragraph(_escape_pdf_text(status_text), styles["caption"])],
                    [Paragraph(_escape_pdf_text(healthy_text), styles["caption"])],
                ],
                colWidths=[1.45 * inch],
            )
            cell.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), colors.Color(status_color.red, status_color.green, status_color.blue, alpha=0.12)),
                        ("BOX", (0, 0), (-1, -1), 0.3, status_color),
                        ("LEFTPADDING", (0, 0), (-1, -1), 6),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                        ("TOPPADDING", (0, 0), (-1, -1), 5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ]
                )
            )
            row.append(cell)
        table_data.append(row)
        table_styles.append(("LINEBELOW", (0, row_index), (-1, row_index), 0.25, PDF_BORDER))

    company_width = (PDF_CONTENT_WIDTH - 1.6 * inch) / max(len(memos), 1)
    ratio_table = Table(
        table_data,
        colWidths=[1.6 * inch] + [company_width] * len(memos),
        repeatRows=1,
    )
    ratio_table.setStyle(TableStyle(table_styles))
    return ratio_table


def _build_comparison_summary_matrix(
    memos: tuple[CreditMemoResult, ...],
    styles: dict[str, ParagraphStyle],
    failed_tickers: dict[str, str] | None = None,
) -> Table:
    columns: list[list[Any]] = []
    for memo in memos:
        content: list[Any] = [
            Paragraph(_escape_pdf_text(f"{memo.company_name} ({memo.ticker})"), styles["scorecard_ratio"]),
            Paragraph("Top Risks", styles["summary_title"]),
        ]
        for index, point in enumerate(memo.key_risk_factors[:3], start=1):
            content.append(Paragraph(f"{index}. {_escape_pdf_text(point.title)}", styles["body_small"]))
        content.append(Spacer(1, 0.05 * inch))
        content.append(Paragraph("Top Strengths", styles["summary_title"]))
        for index, point in enumerate(memo.key_strengths[:2], start=1):
            content.append(Paragraph(f"{index}. {_escape_pdf_text(point.title)}", styles["body_small"]))
        card = Table([[item] for item in content], colWidths=[2.06 * inch])
        card.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), PDF_WHITE),
                    ("BOX", (0, 0), (-1, -1), 0.4, PDF_BORDER),
                    ("LEFTPADDING", (0, 0), (-1, -1), 9),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        columns.append(card)

    for ticker, message in (failed_tickers or {}).items():
        card = Table(
            [
                [Paragraph(_escape_pdf_text(ticker), styles["scorecard_ratio"])],
                [Paragraph("Analysis unavailable for this issuer.", styles["body_small"])],
                [Paragraph(_escape_pdf_text(message), styles["body_italic"])],
            ],
            colWidths=[2.06 * inch],
        )
        card.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), PDF_WHITE),
                    ("BOX", (0, 0), (-1, -1), 0.4, PDF_BORDER),
                    ("LEFTPADDING", (0, 0), (-1, -1), 9),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        columns.append(card)

    while len(columns) < 3:
        columns.append(Table([[""]], colWidths=[2.06 * inch]))

    matrix = Table([columns[:3]], colWidths=[2.1 * inch, 2.1 * inch, 2.1 * inch])
    matrix.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    return matrix


def _build_comparison_data_sources(
    memos: tuple[CreditMemoResult, ...],
    styles: dict[str, ParagraphStyle],
    filing_urls: dict[str, str] | None = None,
    failed_tickers: dict[str, str] | None = None,
) -> list[Any]:
    flowables: list[Any] = [
        _build_section_header(
            "Data Sources",
            styles,
            subtitle="Primary SEC filing links and comparison methodology.",
        ),
        Spacer(1, 0.14 * inch),
    ]
    for memo in memos:
        flowables.append(
            Paragraph(
                f"<b>{_escape_pdf_text(memo.ticker)}:</b> {_wrap_url_for_paragraph((filing_urls or {}).get(memo.ticker, 'Not provided'))}",
                styles["body_small"],
            )
        )
    if failed_tickers:
        flowables.append(Spacer(1, 0.06 * inch))
        flowables.append(Paragraph("Unavailable issuers", styles["summary_title"]))
        for ticker, message in failed_tickers.items():
            flowables.append(
                Paragraph(
                    f"<b>{_escape_pdf_text(ticker)}:</b> {_escape_pdf_text(message)}",
                    styles["body_small"],
                )
            )
    flowables.append(
        Paragraph(
            (
                "Methodology: each issuer was processed sequentially using the same SEC retrieval, "
                "statement-focused extraction, normalization, ratio computation, and memo generation workflow. "
                "Comparison tables use the normalized financial summary values already embedded in each memo."
            ),
            styles["body_small"],
        )
    )
    return flowables


def _draw_comparison_page_chrome(
    pdf_canvas: Canvas,
    *,
    page_number: int,
    total_pages: int,
    title_text: str,
    generated_on: str,
) -> None:
    page_width, page_height = LETTER
    pdf_canvas.saveState()
    pdf_canvas.setFillColor(PDF_STEEL)
    pdf_canvas.rect(0, page_height - PDF_HEADER_ACCENT_HEIGHT, page_width, PDF_HEADER_ACCENT_HEIGHT, stroke=0, fill=1)

    if page_number == 1:
        pdf_canvas.setFillColor(PDF_NAVY)
        pdf_canvas.rect(0, page_height - 0.9 * inch, page_width, 0.78 * inch, stroke=0, fill=1)
        pdf_canvas.setFillColor(PDF_WHITE)
        pdf_canvas.setFont("Helvetica-Bold", 11)
        pdf_canvas.drawString(PDF_PAGE_MARGIN, page_height - 0.5 * inch, "Peer Credit Comparison")
        pdf_canvas.setFillColor(colors.HexColor("#D7DEE7"))
        pdf_canvas.setFont("Helvetica-Bold", 8)
        pdf_canvas.drawRightString(page_width - PDF_PAGE_MARGIN, page_height - 0.5 * inch, "AI CREDIT ANALYST AGENT")
        pdf_canvas.setFillColor(PDF_MUTED)
        pdf_canvas.setFont("Helvetica", 8)
        pdf_canvas.drawCentredString(
            page_width / 2,
            0.76 * inch,
            f"AI Credit Analyst Agent | Generated {generated_on} | Data sourced from SEC EDGAR",
        )
    else:
        pdf_canvas.setStrokeColor(PDF_BORDER)
        pdf_canvas.setLineWidth(0.4)
        pdf_canvas.line(PDF_PAGE_MARGIN, page_height - 0.72 * inch, page_width - PDF_PAGE_MARGIN, page_height - 0.72 * inch)
        pdf_canvas.setFillColor(PDF_NAVY)
        pdf_canvas.setFont("Helvetica-Bold", 10)
        pdf_canvas.drawString(PDF_PAGE_MARGIN, page_height - 0.56 * inch, title_text)
        pdf_canvas.setFillColor(colors.HexColor("#AAB3BF"))
        pdf_canvas.setFont("Helvetica-Bold", 8)
        pdf_canvas.drawRightString(page_width - PDF_PAGE_MARGIN, page_height - 0.56 * inch, "AI CREDIT ANALYST AGENT")

    pdf_canvas.setFillColor(PDF_CONFIDENTIAL)
    pdf_canvas.setFont("Helvetica-Bold", 7.5)
    pdf_canvas.drawString(PDF_PAGE_MARGIN, 0.42 * inch, "CONFIDENTIAL — FOR EDUCATIONAL PURPOSES ONLY")
    pdf_canvas.setFillColor(PDF_MUTED)
    pdf_canvas.setFont("Helvetica", 8)
    pdf_canvas.drawRightString(page_width - PDF_PAGE_MARGIN, 0.42 * inch, f"Page {page_number} of {total_pages}")
    pdf_canvas.restoreState()


class _ComparisonNumberedCanvas(Canvas):
    def __init__(self, *args: Any, title_text: str, generated_on: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._saved_page_states: list[dict[str, Any]] = []
        self._title_text = title_text
        self._generated_on = generated_on

    def showPage(self) -> None:
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self) -> None:
        total_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            _draw_comparison_page_chrome(
                self,
                page_number=self._pageNumber,
                total_pages=total_pages,
                title_text=self._title_text,
                generated_on=self._generated_on,
            )
            Canvas.showPage(self)
        Canvas.save(self)


def build_peer_comparison_pdf(
    memos: tuple[CreditMemoResult, ...],
    *,
    filing_urls: dict[str, str] | None = None,
    failed_tickers: dict[str, str] | None = None,
) -> bytes:
    if not memos:
        raise PDFGenerationError("Failed to build comparison PDF: no completed company analyses were provided.")

    try:
        buffer = io.BytesIO()
        document = SimpleDocTemplate(
            buffer,
            pagesize=LETTER,
            rightMargin=1.0 * inch,
            leftMargin=1.0 * inch,
            topMargin=1.0 * inch,
            bottomMargin=1.0 * inch,
            title="Peer Credit Comparison",
            author="AI Credit Analyst Agent",
            pageCompression=0,
        )
        styles = _build_pdf_styles()
        generated_on = date.today().strftime("%B %d, %Y")
        title_text = " | ".join(memo.ticker for memo in memos)

        story: list[Any] = []
        story.extend(_build_comparison_cover_story(memos, styles))
        story.append(PageBreak())

        story.append(
            _build_section_header(
                "Rating Comparison",
                styles,
                subtitle="Committee-style relative rating view across the selected peer set.",
            )
        )
        story.append(Spacer(1, 0.18 * inch))
        story.append(_build_comparison_rating_cards_table(memos, styles, failed_tickers))
        story.append(PageBreak())

        story.append(
            _build_section_header(
                "Financial Comparison",
                styles,
                subtitle="Scale, earnings, debt, and liquidity viewed on a common normalized basis.",
            )
        )
        story.append(Spacer(1, 0.16 * inch))
        story.append(_build_comparison_metric_chart(memos))
        story.append(Spacer(1, 0.18 * inch))
        story.append(_build_comparison_financial_table(memos, styles))
        story.append(PageBreak())

        story.append(
            _build_section_header(
                "Ratio Comparison",
                styles,
                subtitle="Core credit ratios calibrated against the same healthy-range thresholds used in the application.",
            )
        )
        story.append(Spacer(1, 0.16 * inch))
        story.append(_build_comparison_ratio_table(memos, styles))
        story.append(Spacer(1, 0.14 * inch))
        story.append(
            Paragraph(
                "Healthy ranges: Debt-to-Equity < 1.5x, Interest Coverage > 5.0x, Current Ratio > 1.5x, Net Debt / EBITDA < 2.0x.",
                styles["caption"],
            )
        )
        story.append(PageBreak())

        story.append(
            _build_section_header(
                "Relative Risks & Strengths",
                styles,
                subtitle="Top cited risk and strength themes presented issuer by issuer.",
            )
        )
        story.append(Spacer(1, 0.16 * inch))
        story.append(_build_comparison_summary_matrix(memos, styles, failed_tickers))
        story.append(Spacer(1, 0.18 * inch))
        story.extend(_build_comparison_data_sources(memos, styles, filing_urls, failed_tickers))

        document.build(
            story,
            canvasmaker=lambda *args, **kwargs: _ComparisonNumberedCanvas(
                *args,
                title_text=title_text,
                generated_on=generated_on,
                **kwargs,
            ),
        )
        return buffer.getvalue()
    except PDFGenerationError:
        raise
    except Exception as exc:
        raise PDFGenerationError(f"Failed to build comparison PDF: {exc}") from exc
