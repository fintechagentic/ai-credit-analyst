from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest
from openai import RateLimitError

from ai_credit_analyst.extraction import (
    _extract_chunk_metrics,
    FIELD_ORDER,
    build_filing_chunks,
    compute_credit_ratios,
    extract_financials,
    format_financial_value,
    format_ratio_value,
    normalize_extracted_metrics,
    prepare_filing_text_for_extraction,
)
from ai_credit_analyst.exceptions import OpenAIExtractionError
from ai_credit_analyst.models import FilingChunk, MetricValue


RAW_FILING_TEXT = """
<SEC-DOCUMENT>
<DOCUMENT>
<TYPE>10-K
<TEXT>
<html><body>
<h1>CONSOLIDATED BALANCE SHEETS</h1>
<table>
<tr><td>Cash and cash equivalents</td><td>5,000</td></tr>
<tr><td>Total current assets</td><td>15,000</td></tr>
<tr><td>Total assets</td><td>50,000</td></tr>
<tr><td>Total current liabilities</td><td>8,000</td></tr>
<tr><td>Total liabilities</td><td>20,000</td></tr>
<tr><td>Total shareholders' equity</td><td>30,000</td></tr>
</table>
<h1>CONSOLIDATED STATEMENTS OF OPERATIONS</h1>
<table>
<tr><td>Revenue</td><td>30,000</td></tr>
<tr><td>Net income</td><td>4,000</td></tr>
<tr><td>Operating income</td><td>5,000</td></tr>
<tr><td>Interest expense</td><td>(200)</td></tr>
</table>
<h1>CONSOLIDATED STATEMENTS OF CASH FLOWS</h1>
<table>
<tr><td>Net cash provided by operating activities</td><td>6,000</td></tr>
<tr><td>Purchases of property and equipment</td><td>(1,500)</td></tr>
</table>
<h1>LONG-TERM DEBT</h1>
<table>
<tr><td>Current portion of long-term debt</td><td>500</td></tr>
<tr><td>Long-term debt, net of current portion</td><td>4,500</td></tr>
</table>
<h1>DEPRECIATION AND AMORTIZATION</h1>
<table>
<tr><td>Depreciation and amortization</td><td>700</td></tr>
</table>
</body></html>
</TEXT>
</DOCUMENT>
<DOCUMENT>
<TYPE>EX-13
<TEXT>Exhibit text that should be ignored.</TEXT>
</DOCUMENT>
"""


def build_metric(
    value: int | float | None,
    quote: str | None,
    chunk_label: str,
) -> MetricValue:
    return MetricValue(value=value, source_quote=quote, source_chunk_label=chunk_label)


def build_rate_limit_error() -> RateLimitError:
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx.Response(429, request=request)
    return RateLimitError("rate limited", response=response, body=None)


def with_supplemental_defaults(
    metrics: dict[str, MetricValue],
    *,
    chunk_label: str = "supplemental",
) -> dict[str, MetricValue]:
    defaults = {
        "retained_earnings": build_metric(None, None, chunk_label),
        "common_stock": build_metric(None, None, chunk_label),
        "additional_paid_in_capital": build_metric(None, None, chunk_label),
        "gross_profit": build_metric(None, None, chunk_label),
        "current_portion_of_long_term_debt": build_metric(None, None, chunk_label),
    }
    combined = dict(defaults)
    combined.update(metrics)
    return combined


def test_prepare_filing_text_for_extraction_uses_primary_10k_document() -> None:
    prepared_text, used_primary_document = prepare_filing_text_for_extraction(
        RAW_FILING_TEXT
    )

    assert used_primary_document is True
    assert "CONSOLIDATED BALANCE SHEETS" in prepared_text
    assert "Exhibit text that should be ignored." not in prepared_text


def test_build_filing_chunks_prefers_statement_anchors() -> None:
    chunks, strategy, prepared_length, used_primary_document = build_filing_chunks(
        RAW_FILING_TEXT
    )

    labels = [chunk.label for chunk in chunks]
    assert strategy == "financial_statement_anchors"
    assert used_primary_document is True
    assert prepared_length > 0
    assert "balance_sheet" in labels
    assert "income_statement" in labels
    assert "cash_flow_statement" in labels
    assert "debt_note" in labels


def test_build_filing_chunks_falls_back_when_no_anchors_exist() -> None:
    raw_text = "This filing has no useful anchor headings but it does have numbers 12345 67890."

    chunks, strategy, _, used_primary_document = build_filing_chunks(raw_text)

    assert strategy == "sliding_window_fallback"
    assert used_primary_document is False
    assert len(chunks) == 1
    assert chunks[0].label == "fallback_window_1"


def test_compute_credit_ratios_handles_nulls_and_zero_denominators() -> None:
    metrics = with_supplemental_defaults({
        "revenue": build_metric(30000, "Revenue | 30,000", "income_statement"),
        "net_income": build_metric(4000, "Net income | 4,000", "income_statement"),
        "total_assets": build_metric(50000, "Total assets | 50,000", "balance_sheet"),
        "total_current_assets": build_metric(
            15000, "Total current assets | 15,000", "balance_sheet"
        ),
        "total_current_liabilities": build_metric(
            8000, "Total current liabilities | 8,000", "balance_sheet"
        ),
        "total_liabilities": build_metric(
            20000, "Total liabilities | 20,000", "balance_sheet"
        ),
        "total_shareholders_equity": build_metric(
            30000, "Total shareholders' equity | 30,000", "balance_sheet"
        ),
        "total_debt": build_metric(5000, "Debt lines", "debt_note"),
        "cash_and_cash_equivalents": build_metric(
            5000, "Cash and cash equivalents | 5,000", "balance_sheet"
        ),
        "interest_expense": build_metric(
            200, "Interest expense | (200)", "income_statement"
        ),
        "ebit": build_metric(5000, "Operating income | 5,000", "income_statement"),
        "ebitda": build_metric(
            5700,
            "Operating income | 5,000 || Depreciation and amortization | 700",
            "depreciation_note",
        ),
        "operating_cash_flow": build_metric(
            6000,
            "Net cash provided by operating activities | 6,000",
            "cash_flow_statement",
        ),
        "capital_expenditures": build_metric(
            1500,
            "Purchases of property and equipment | (1,500)",
            "cash_flow_statement",
        ),
    })

    ratios = compute_credit_ratios(metrics)

    assert round(ratios["debt_to_equity"].value or 0, 4) == 0.6667
    assert round(ratios["interest_coverage"].value or 0, 4) == 25.0
    assert round(ratios["current_ratio"].value or 0, 4) == 1.875
    assert ratios["net_debt_to_ebitda"].value == 0.0
    assert ratios["free_cash_flow_yield"].value is None


def test_normalize_extracted_metrics_scales_statement_values_but_not_explicit_billions() -> None:
    metrics = with_supplemental_defaults({
        "revenue": build_metric(39001623, "Revenue | 39,001,623", "income_statement"),
        "net_income": build_metric(11000000, "$11.0 billion", "income_statement"),
        "total_assets": build_metric(52000000, "Total assets | 52,000,000", "balance_sheet"),
        "total_current_assets": build_metric(12000000, "Total current assets | 12,000,000", "balance_sheet"),
        "total_current_liabilities": build_metric(10100000, "Total current liabilities | 10,100,000", "balance_sheet"),
        "total_liabilities": build_metric(28300000, "Total liabilities | 28,300,000", "balance_sheet"),
        "total_shareholders_equity": build_metric(26000000, "Total shareholders' equity | 26,000,000", "balance_sheet"),
        "total_debt": build_metric(14500000000, "$14.5 billion", "debt_note"),
        "cash_and_cash_equivalents": build_metric(9000000000, "$9.0 billion", "balance_sheet"),
        "interest_expense": build_metric(737000, "Interest expense | 737,000", "income_statement"),
        "ebit": build_metric(13659992, "Operating income | 13,659,992", "income_statement"),
        "ebitda": build_metric(13659992, "EBITDA | 13,659,992", "income_statement"),
        "operating_cash_flow": build_metric(9100000, "Net cash provided by operating activities | 9,100,000", "cash_flow_statement"),
        "capital_expenditures": build_metric(450000, "Purchases of property and equipment | 450,000", "cash_flow_statement"),
    })
    chunks = [
        FilingChunk(
            "balance_sheet",
            "financial_statement_anchors",
            0,
            100,
            "CONSOLIDATED BALANCE SHEETS (dollars in thousands)",
        ),
        FilingChunk(
            "income_statement",
            "financial_statement_anchors",
            100,
            200,
            "CONSOLIDATED STATEMENTS OF OPERATIONS (dollars in thousands)",
        ),
        FilingChunk(
            "cash_flow_statement",
            "financial_statement_anchors",
            200,
            300,
            "CONSOLIDATED STATEMENTS OF CASH FLOWS (dollars in thousands)",
        ),
        FilingChunk(
            "debt_note",
            "financial_statement_anchors",
            300,
            400,
            "Debt note excerpt",
        ),
    ]

    normalized_metrics, unit_label, multiplier, note = normalize_extracted_metrics(
        metrics,
        chunks,
        "CONSOLIDATED BALANCE SHEETS (dollars in thousands)",
    )
    ratios = compute_credit_ratios(normalized_metrics)

    assert unit_label == "thousands"
    assert multiplier == 1000
    assert note == "Values normalized from thousands to dollars before ratio computation."
    assert normalized_metrics["ebitda"].value == 13_659_992_000
    assert normalized_metrics["total_debt"].value == 14_500_000_000
    assert normalized_metrics["cash_and_cash_equivalents"].value == 9_000_000_000
    assert ratios["net_debt_to_ebitda"].value == pytest.approx(0.4026, rel=1e-3)


def test_financial_value_formatting_uses_b_m_and_k_consistently() -> None:
    assert format_financial_value(11_000_000_000) == "$11.0B"
    assert format_financial_value(11_000_000) == "$11.0M"
    assert format_financial_value(11_000) == "$11.0K"
    assert format_ratio_value(0.40, "net_debt_to_ebitda") == "0.40x"


def test_extract_financials_merges_first_non_null_metric_from_chunks() -> None:
    mocked_chunk_outputs = {
        "balance_sheet": {
            "revenue": build_metric(None, None, "balance_sheet"),
            "net_income": build_metric(None, None, "balance_sheet"),
            "total_assets": build_metric(
                50000, "Total assets | 50,000", "balance_sheet"
            ),
            "total_current_assets": build_metric(
                15000, "Total current assets | 15,000", "balance_sheet"
            ),
            "total_current_liabilities": build_metric(
                8000, "Total current liabilities | 8,000", "balance_sheet"
            ),
            "total_liabilities": build_metric(
                20000, "Total liabilities | 20,000", "balance_sheet"
            ),
            "total_shareholders_equity": build_metric(
                30000, "Total shareholders' equity | 30,000", "balance_sheet"
            ),
            "total_debt": build_metric(None, None, "balance_sheet"),
            "cash_and_cash_equivalents": build_metric(
                5000, "Cash and cash equivalents | 5,000", "balance_sheet"
            ),
            "interest_expense": build_metric(None, None, "balance_sheet"),
            "ebit": build_metric(None, None, "balance_sheet"),
            "ebitda": build_metric(None, None, "balance_sheet"),
            "operating_cash_flow": build_metric(None, None, "balance_sheet"),
            "capital_expenditures": build_metric(None, None, "balance_sheet"),
        },
        "income_statement": {
            "revenue": build_metric(30000, "Revenue | 30,000", "income_statement"),
            "net_income": build_metric(4000, "Net income | 4,000", "income_statement"),
            "total_assets": build_metric(None, None, "income_statement"),
            "total_current_assets": build_metric(None, None, "income_statement"),
            "total_current_liabilities": build_metric(None, None, "income_statement"),
            "total_liabilities": build_metric(None, None, "income_statement"),
            "total_shareholders_equity": build_metric(None, None, "income_statement"),
            "total_debt": build_metric(None, None, "income_statement"),
            "cash_and_cash_equivalents": build_metric(None, None, "income_statement"),
            "interest_expense": build_metric(
                200, "Interest expense | (200)", "income_statement"
            ),
            "ebit": build_metric(5000, "Operating income | 5,000", "income_statement"),
            "ebitda": build_metric(None, None, "income_statement"),
            "operating_cash_flow": build_metric(None, None, "income_statement"),
            "capital_expenditures": build_metric(None, None, "income_statement"),
        },
        "cash_flow_statement": {
            "revenue": build_metric(None, None, "cash_flow_statement"),
            "net_income": build_metric(None, None, "cash_flow_statement"),
            "total_assets": build_metric(None, None, "cash_flow_statement"),
            "total_current_assets": build_metric(None, None, "cash_flow_statement"),
            "total_current_liabilities": build_metric(None, None, "cash_flow_statement"),
            "total_liabilities": build_metric(None, None, "cash_flow_statement"),
            "total_shareholders_equity": build_metric(None, None, "cash_flow_statement"),
            "total_debt": build_metric(None, None, "cash_flow_statement"),
            "cash_and_cash_equivalents": build_metric(
                None, None, "cash_flow_statement"
            ),
            "interest_expense": build_metric(None, None, "cash_flow_statement"),
            "ebit": build_metric(None, None, "cash_flow_statement"),
            "ebitda": build_metric(None, None, "cash_flow_statement"),
            "operating_cash_flow": build_metric(
                6000,
                "Net cash provided by operating activities | 6,000",
                "cash_flow_statement",
            ),
            "capital_expenditures": build_metric(
                1500,
                "Purchases of property and equipment | (1,500)",
                "cash_flow_statement",
            ),
        },
        "debt_note": {
            "revenue": build_metric(None, None, "debt_note"),
            "net_income": build_metric(None, None, "debt_note"),
            "total_assets": build_metric(None, None, "debt_note"),
            "total_current_assets": build_metric(None, None, "debt_note"),
            "total_current_liabilities": build_metric(None, None, "debt_note"),
            "total_liabilities": build_metric(None, None, "debt_note"),
            "total_shareholders_equity": build_metric(None, None, "debt_note"),
            "total_debt": build_metric(
                5000,
                "Current portion of long-term debt | 500 || Long-term debt, net of current portion | 4,500",
                "debt_note",
            ),
            "cash_and_cash_equivalents": build_metric(None, None, "debt_note"),
            "interest_expense": build_metric(None, None, "debt_note"),
            "ebit": build_metric(None, None, "debt_note"),
            "ebitda": build_metric(None, None, "debt_note"),
            "operating_cash_flow": build_metric(None, None, "debt_note"),
            "capital_expenditures": build_metric(None, None, "debt_note"),
        },
        "property_equipment_note": {
            "revenue": build_metric(None, None, "property_equipment_note"),
            "net_income": build_metric(None, None, "property_equipment_note"),
            "total_assets": build_metric(None, None, "property_equipment_note"),
            "total_current_assets": build_metric(
                None, None, "property_equipment_note"
            ),
            "total_current_liabilities": build_metric(
                None, None, "property_equipment_note"
            ),
            "total_liabilities": build_metric(None, None, "property_equipment_note"),
            "total_shareholders_equity": build_metric(
                None, None, "property_equipment_note"
            ),
            "total_debt": build_metric(None, None, "property_equipment_note"),
            "cash_and_cash_equivalents": build_metric(
                None, None, "property_equipment_note"
            ),
            "interest_expense": build_metric(None, None, "property_equipment_note"),
            "ebit": build_metric(None, None, "property_equipment_note"),
            "ebitda": build_metric(None, None, "property_equipment_note"),
            "operating_cash_flow": build_metric(
                None, None, "property_equipment_note"
            ),
            "capital_expenditures": build_metric(
                1500,
                "Purchases of property and equipment | (1,500)",
                "property_equipment_note",
            ),
        },
        "depreciation_note": {
            "revenue": build_metric(None, None, "depreciation_note"),
            "net_income": build_metric(None, None, "depreciation_note"),
            "total_assets": build_metric(None, None, "depreciation_note"),
            "total_current_assets": build_metric(None, None, "depreciation_note"),
            "total_current_liabilities": build_metric(None, None, "depreciation_note"),
            "total_liabilities": build_metric(None, None, "depreciation_note"),
            "total_shareholders_equity": build_metric(None, None, "depreciation_note"),
            "total_debt": build_metric(None, None, "depreciation_note"),
            "cash_and_cash_equivalents": build_metric(
                None, None, "depreciation_note"
            ),
            "interest_expense": build_metric(None, None, "depreciation_note"),
            "ebit": build_metric(None, None, "depreciation_note"),
            "ebitda": build_metric(
                5700,
                "Operating income | 5,000 || Depreciation and amortization | 700",
                "depreciation_note",
            ),
            "operating_cash_flow": build_metric(None, None, "depreciation_note"),
            "capital_expenditures": build_metric(None, None, "depreciation_note"),
        },
    }
    mocked_chunk_outputs = {
        chunk_label: with_supplemental_defaults(chunk_metrics, chunk_label=chunk_label)
        for chunk_label, chunk_metrics in mocked_chunk_outputs.items()
    }

    def fake_extract_chunk_metrics(*args, **kwargs):
        chunk: FilingChunk = args[1]
        return mocked_chunk_outputs[chunk.label]

    with patch("ai_credit_analyst.extraction.build_openai_client", return_value=object()):
        with patch(
            "ai_credit_analyst.extraction._extract_chunk_metrics",
            side_effect=fake_extract_chunk_metrics,
        ):
            result = extract_financials(
                raw_filing_text=RAW_FILING_TEXT,
                company_name="NETFLIX INC",
                ticker="NFLX",
                filing_date="2026-01-23",
                api_key="test-key",
            )

    assert result.metrics["revenue"].value == 30000
    assert result.metrics["total_debt"].value == 5000
    assert result.metrics["ebitda"].value == 5700
    assert result.ratios["free_cash_flow_yield"].value is None


def test_extract_chunk_metrics_retries_on_rate_limit_then_succeeds() -> None:
    chunk = FilingChunk(
        label="income_statement",
        strategy="financial_statement_anchors",
        start_char=0,
        end_char=100,
        text="Revenue | 30,000",
    )
    payload = {
        field_name: {"value": None, "source_quote": None}
        for field_name in FIELD_ORDER
    }
    payload["revenue"] = {"value": 30000, "source_quote": "Revenue | 30,000"}

    response_object = type("DummyResponse", (), {"output_text": json.dumps(payload)})()
    client = type("DummyClient", (), {})()
    client.responses = type("DummyResponses", (), {})()
    client.responses.create = lambda **kwargs: response_object

    with patch.object(
        client.responses,
        "create",
        side_effect=[build_rate_limit_error(), build_rate_limit_error(), response_object],
    ) as create_mock:
        with patch("ai_credit_analyst.extraction.time.sleep") as sleep_mock:
            metrics = _extract_chunk_metrics(
                client,
                chunk,
                company_name="NETFLIX INC",
                ticker="NFLX",
                filing_date="2026-01-23",
                model="gpt-4o",
            )

    assert metrics["revenue"].value == 30000
    assert create_mock.call_count == 3
    assert [call.args[0] for call in sleep_mock.call_args_list] == [10, 20]


def test_extract_financials_waits_between_chunks_and_reports_progress() -> None:
    fake_chunks = [
        FilingChunk("chunk_1", "sliding_window_fallback", 0, 10, "one"),
        FilingChunk("chunk_2", "sliding_window_fallback", 10, 20, "two"),
        FilingChunk("chunk_3", "sliding_window_fallback", 20, 30, "three"),
    ]
    callback_events: list[tuple[int, int, str]] = []
    empty_metrics = {
        "revenue": build_metric(None, None, "chunk"),
        "net_income": build_metric(None, None, "chunk"),
        "total_assets": build_metric(None, None, "chunk"),
        "total_current_assets": build_metric(None, None, "chunk"),
        "total_current_liabilities": build_metric(None, None, "chunk"),
        "total_liabilities": build_metric(None, None, "chunk"),
        "total_shareholders_equity": build_metric(None, None, "chunk"),
        "total_debt": build_metric(None, None, "chunk"),
        "cash_and_cash_equivalents": build_metric(None, None, "chunk"),
        "interest_expense": build_metric(None, None, "chunk"),
        "ebit": build_metric(None, None, "chunk"),
        "ebitda": build_metric(None, None, "chunk"),
        "operating_cash_flow": build_metric(None, None, "chunk"),
        "capital_expenditures": build_metric(None, None, "chunk"),
    }
    empty_metrics = with_supplemental_defaults(empty_metrics, chunk_label="chunk")

    with patch(
        "ai_credit_analyst.extraction.build_filing_chunks",
        return_value=(fake_chunks, "sliding_window_fallback", 30, False),
    ):
        with patch("ai_credit_analyst.extraction.build_openai_client", return_value=object()):
            with patch(
                "ai_credit_analyst.extraction._extract_chunk_metrics",
                return_value=empty_metrics,
            ):
                with patch("ai_credit_analyst.extraction.time.sleep") as sleep_mock:
                    extract_financials(
                        raw_filing_text="raw",
                        company_name="NETFLIX INC",
                        ticker="NFLX",
                        filing_date="2026-01-23",
                        api_key="test-key",
                        chunk_progress_callback=lambda current, total, chunk: callback_events.append(
                            (current, total, chunk.label)
                        ),
                    )

    assert callback_events == [
        (1, 3, "chunk_1"),
        (2, 3, "chunk_2"),
        (3, 3, "chunk_3"),
    ]
    assert [call.args[0] for call in sleep_mock.call_args_list] == [3, 3]


def test_extract_chunk_metrics_raises_clear_error_after_rate_limit_retries_exhausted() -> None:
    chunk = FilingChunk(
        label="income_statement",
        strategy="financial_statement_anchors",
        start_char=0,
        end_char=100,
        text="Revenue | 30,000",
    )
    client = type("DummyClient", (), {})()
    client.responses = type("DummyResponses", (), {})()
    client.responses.create = lambda **kwargs: None

    with patch.object(
        client.responses,
        "create",
        side_effect=[
            build_rate_limit_error(),
            build_rate_limit_error(),
            build_rate_limit_error(),
            build_rate_limit_error(),
        ],
    ):
        with patch("ai_credit_analyst.extraction.time.sleep") as sleep_mock:
            with pytest.raises(OpenAIExtractionError, match="Wait 60 seconds and try again"):
                _extract_chunk_metrics(
                    client,
                    chunk,
                    company_name="NETFLIX INC",
                    ticker="NFLX",
                    filing_date="2026-01-23",
                    model="gpt-4o",
                )

    assert [call.args[0] for call in sleep_mock.call_args_list] == [10, 20, 40]
