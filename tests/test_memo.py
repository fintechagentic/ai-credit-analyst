from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest
from openai import RateLimitError

from ai_credit_analyst.memo import (
    build_credit_memo_pdf,
    build_financial_summary_rows,
    build_memo_context_sections,
    build_peer_comparison_pdf,
    generate_credit_memo,
)
from ai_credit_analyst.models import (
    AltmanComponent,
    AltmanZScoreResult,
    CashFlowAdequacyResult,
    CreditMemoResult,
    CreditOutlook,
    CreditRiskRating,
    DebtMaturityProfileResult,
    DebtMaturityTranche,
    FilingChunk,
    FinancialExtractionResult,
    FinancialSummaryRow,
    MaturityWallWarning,
    MemoContextSection,
    MemoPoint,
    MetricValue,
    PiotroskiCriterion,
    PiotroskiFScoreResult,
    QuantitativeModelResult,
    RatioValue,
    TrendAnalysisRow,
)
from ai_credit_analyst.exceptions import MemoGenerationError


RAW_FILING_TEXT = """
<SEC-DOCUMENT>
<DOCUMENT>
<TYPE>10-K
<TEXT>
<html><body>
<h1>ITEM 1. BUSINESS</h1>
<p>Netflix is one of the world's leading entertainment services with more than 300 million paid memberships in over 190 countries.</p>
<h1>ITEM 1A. RISK FACTORS</h1>
<p>The company faces intense competition for entertainment time, consumer attention, and content acquisition.</p>
<p>Streaming content obligations remain significant and require continued investment.</p>
<h1>ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS</h1>
<p>Operating margin expanded due to revenue growth and disciplined spending.</p>
<h1>LIQUIDITY AND CAPITAL RESOURCES</h1>
<p>Cash, cash equivalents and short-term investments provide liquidity for operations and debt servicing.</p>
<h1>LONG-TERM DEBT</h1>
<p>Long-term debt remains a key component of the capital structure.</p>
</body></html>
</TEXT>
</DOCUMENT>
"""


def build_financial_result() -> FinancialExtractionResult:
    metrics = {
        "revenue": MetricValue(39000000000, "Revenue | 39,000", "income_statement"),
        "net_income": MetricValue(8700000000, "Net income | 8,700", "income_statement"),
        "total_assets": MetricValue(52000000000, "Total assets | 52,000", "balance_sheet"),
        "total_current_assets": MetricValue(
            12000000000,
            "Total current assets | 12,000",
            "balance_sheet",
        ),
        "total_current_liabilities": MetricValue(
            10100000000,
            "Total current liabilities | 10,100",
            "balance_sheet",
        ),
        "total_liabilities": MetricValue(
            28300000000,
            "Total liabilities | 28,300",
            "balance_sheet",
        ),
        "total_shareholders_equity": MetricValue(
            26000000000,
            "Total shareholders' equity | 26,000",
            "balance_sheet",
        ),
        "total_debt": MetricValue(
            14500000000,
            "Total debt | 14,500",
            "debt_note",
        ),
        "cash_and_cash_equivalents": MetricValue(
            9000000000,
            "Cash and cash equivalents | 9,000",
            "balance_sheet",
        ),
        "interest_expense": MetricValue(
            740000000,
            "Interest expense | 740",
            "income_statement",
        ),
        "ebit": MetricValue(13700000000, "Operating income | 13,700", "income_statement"),
        "ebitda": MetricValue(13700000000, "EBITDA | 13,700", "income_statement"),
        "operating_cash_flow": MetricValue(
            9000000000,
            "Net cash provided by operating activities | 9,000",
            "cash_flow_statement",
        ),
        "capital_expenditures": MetricValue(
            450000000,
            "Purchases of property and equipment | 450",
            "cash_flow_statement",
        ),
        "retained_earnings": MetricValue(
            15000000000,
            "Retained earnings | 15,000",
            "balance_sheet",
        ),
        "common_stock": MetricValue(
            1000000000,
            "Common stock | 1,000",
            "balance_sheet",
        ),
        "additional_paid_in_capital": MetricValue(
            10000000000,
            "Additional paid-in capital | 10,000",
            "balance_sheet",
        ),
        "gross_profit": MetricValue(
            16000000000,
            "Gross profit | 16,000",
            "income_statement",
        ),
        "current_portion_of_long_term_debt": MetricValue(
            1200000000,
            "Current portion of long-term debt | 1,200",
            "debt_note",
        ),
    }
    ratios = {
        "debt_to_equity": RatioValue(1.09, None),
        "interest_coverage": RatioValue(18.54, None),
        "current_ratio": RatioValue(1.19, None),
        "net_debt_to_ebitda": RatioValue(0.40, None),
        "free_cash_flow_yield": RatioValue(
            None,
            "Skipped because market cap data is not available yet.",
        ),
    }
    chunks = (
        FilingChunk("balance_sheet", "financial_statement_anchors", 0, 1000, "Balance sheet chunk"),
        FilingChunk("income_statement", "financial_statement_anchors", 1000, 2000, "Income statement chunk"),
        FilingChunk("cash_flow_statement", "financial_statement_anchors", 2000, 3000, "Cash flow chunk"),
        FilingChunk("debt_note", "financial_statement_anchors", 3000, 4000, "Debt note chunk"),
    )
    return FinancialExtractionResult(
        metrics=metrics,
        ratios=ratios,
        chunk_strategy="financial_statement_anchors",
        chunk_count=len(chunks),
        prepared_text_length=5000,
        used_primary_document=True,
        chunks=chunks,
    )


def build_rate_limit_error() -> RateLimitError:
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx.Response(429, request=request)
    return RateLimitError("rate limited", response=response, body=None)


def build_quantitative_result() -> QuantitativeModelResult:
    return QuantitativeModelResult(
        altman_z_score=AltmanZScoreResult(
            score=3.42,
            zone="Safe Zone",
            note="Book value used as proxy for market capitalization.",
            components=(
                AltmanComponent("1.2 x Working Capital / Total Assets", 0.04, 0.05, "Liquidity relative to the asset base."),
                AltmanComponent("1.4 x Retained Earnings / Total Assets", 0.29, 0.41, "Cumulative profitability and balance-sheet seasoning."),
                AltmanComponent("3.3 x EBIT / Total Assets", 0.26, 0.87, "Operating earnings power relative to assets."),
                AltmanComponent("0.6 x Market Cap / Total Liabilities", 0.92, 0.55, "Equity cushion against total obligations."),
                AltmanComponent("1.0 x Revenue / Total Assets", 0.75, 0.75, "Asset turnover and commercial productivity."),
            ),
        ),
        piotroski_f_score=PiotroskiFScoreResult(
            score=8,
            interpretation="Strong",
            note="8/9 criteria passed.",
            criteria=(
                PiotroskiCriterion("Positive Net Income", True, "Pass"),
                PiotroskiCriterion("Positive Operating Cash Flow", True, "Pass"),
                PiotroskiCriterion("ROA Improved", True, "Pass"),
                PiotroskiCriterion("Cash Flow Exceeds Net Income", True, "Pass"),
                PiotroskiCriterion("Lower Debt-to-Equity", True, "Pass"),
                PiotroskiCriterion("Higher Current Ratio", True, "Pass"),
                PiotroskiCriterion("No Share Dilution", True, "Pass"),
                PiotroskiCriterion("Higher Margin", False, "Fail"),
                PiotroskiCriterion("Higher Asset Turnover", True, "Pass"),
            ),
            evaluated_count=9,
            missing_count=0,
        ),
        cash_flow_adequacy=CashFlowAdequacyResult(
            ratio=1.84,
            assessment="Strong - self-funding with margin",
            note=None,
            near_term_debt_used=1_200_000_000,
        ),
        trend_analysis=(
            TrendAnalysisRow("Revenue", 39_000_000_000, 35_000_000_000, 4_000_000_000, 0.114, True, True, False),
            TrendAnalysisRow("Debt-to-Equity", 1.09, 1.35, -0.26, -0.193, True, False, True),
        ),
        trend_analysis_note=None,
        ratio_history={
            "debt_to_equity": (1.09, 1.35),
            "interest_coverage": (18.54, 12.10),
            "current_ratio": (1.19, 1.05),
            "net_debt_to_ebitda": (0.40, 0.90),
        },
        debt_maturity_profile=DebtMaturityProfileResult(
            tranches=(
                DebtMaturityTranche(1000, 2027, "5.0%", "Senior notes due 2027"),
                DebtMaturityTranche(2500, 2028, "4.5%", "Senior notes due 2028"),
            ),
            weighted_average_maturity=2.7,
            warnings=(MaturityWallWarning(2028, 0.22),),
            note=None,
        ),
        prior_year_available=True,
        prior_year_filing_date="2025-01-26",
    )


def test_build_financial_summary_rows_formats_metrics_and_ratios() -> None:
    result = build_financial_result()

    rows = build_financial_summary_rows(result)

    assert rows[0].label == "Revenue"
    assert rows[0].value == "$39.0B"
    assert rows[-1].label == "Free Cash Flow Yield"
    assert rows[-1].value == "N/A"


def test_build_memo_context_sections_selects_narrative_and_chunk_context() -> None:
    result = build_financial_result()

    context_sections = build_memo_context_sections(RAW_FILING_TEXT, result)
    labels = {section.label for section in context_sections}

    assert "business_overview" in labels
    assert "risk_factors" in labels
    assert "liquidity" in labels
    assert "debt_note" in labels


def test_generate_credit_memo_parses_structured_response() -> None:
    result = build_financial_result()
    summary_rows = build_financial_summary_rows(result)
    model_payload = {
        "company_overview": (
            "Netflix operates a global subscription streaming platform and film and television studio. "
            "The company is a large-cap media and entertainment issuer with substantial recurring revenue scale."
        ),
        "financial_summary_table": [row.__dict__ for row in summary_rows],
        "credit_risk_rating": {
            "rating": "A-",
            "justification": (
                "The rating reflects strong scale, solid cash generation, and manageable net leverage, "
                "partly offset by elevated content investment requirements and competitive pressure."
            ),
        },
        "key_risk_factors": [
            {
                "title": "Content Investment Commitments",
                "explanation": "Netflix relies on sustained content spending to support subscriber engagement. This requirement keeps fixed obligations high and limits downside flexibility.",
                "evidence": "Streaming content obligations remain significant and require continued investment.",
            },
            {
                "title": "Competitive Intensity",
                "explanation": "Competition for consumer attention and content rights remains intense. That dynamic can pressure acquisition costs and retention economics.",
                "evidence": "The company faces intense competition for entertainment time, consumer attention, and content acquisition.",
            },
            {
                "title": "Debt-Funded Capital Structure Legacy",
                "explanation": "Debt remains an important part of the capital structure even as leverage has moderated. Continued shareholder returns or renewed content spend could slow further deleveraging.",
                "evidence": "Long-term debt remains a key component of the capital structure.",
            },
            {
                "title": "Consumer Demand Sensitivity",
                "explanation": "Subscription businesses remain exposed to churn if consumer value perception weakens. Revenue durability depends on consistent content performance across regions.",
                "evidence": "More than 300 million paid memberships in over 190 countries.",
            },
            {
                "title": "Execution Dependence on Operating Margin Discipline",
                "explanation": "Recent margin strength supports the profile, but credit quality still depends on maintaining disciplined spending. Any reversal in content efficiency would pressure cash flow.",
                "evidence": "Operating margin expanded due to revenue growth and disciplined spending.",
            },
        ],
        "key_strengths": [
            {
                "title": "Large Recurring Revenue Base",
                "explanation": "Netflix benefits from a broad subscription base that supports recurring revenue. Scale strengthens cash generation and market position.",
                "evidence": "More than 300 million paid memberships in over 190 countries.",
            },
            {
                "title": "Strong Interest Coverage",
                "explanation": "High interest coverage indicates substantial earnings capacity relative to debt service obligations. That supports financial flexibility.",
                "evidence": "Interest Coverage = 18.54x from extracted financial summary.",
            },
            {
                "title": "Meaningful Liquidity Cushion",
                "explanation": "Large on-balance-sheet cash improves resilience and supports debt servicing. Liquidity remains a meaningful offset to gross debt.",
                "evidence": "Cash, cash equivalents and short-term investments provide liquidity for operations and debt servicing.",
            },
        ],
        "outlook": {
            "direction": "Stable",
            "justification": (
                "The stable outlook reflects expectations for continued strong operating performance and manageable leverage. "
                "The outlook balances that stability against ongoing content spend requirements and competitive risk."
            ),
        },
    }

    class DummyResponses:
        def create(self, **kwargs):
            return type("DummyResponse", (), {"output_text": json.dumps(model_payload)})()

    class DummyClient:
        responses = DummyResponses()

    with patch("ai_credit_analyst.memo.build_openai_client", return_value=DummyClient()):
        memo = generate_credit_memo(
            raw_filing_text=RAW_FILING_TEXT,
            company_name="NETFLIX INC",
            ticker="NFLX",
            filing_date="2026-01-23",
            financial_result=result,
            api_key="test-key",
        )

    assert memo.credit_risk_rating.rating == "A-"
    assert len(memo.key_risk_factors) == 5
    assert len(memo.key_strengths) == 3
    assert memo.financial_summary[0].label == "Revenue"
    assert memo.outlook.direction == "Stable"


def test_build_credit_memo_pdf_returns_pdf_bytes() -> None:
    summary_rows = build_financial_summary_rows(build_financial_result())
    memo = CreditMemoResult(
        company_name="NETFLIX INC",
        ticker="NFLX",
        filing_date="2026-01-23",
        company_overview="Netflix operates a global streaming entertainment platform.",
        financial_summary=summary_rows,
        credit_risk_rating=CreditRiskRating(
            rating="A-",
            justification="Strong scale and solid cash generation support the rating.",
        ),
        key_risk_factors=(
            MemoPoint("Risk 1", "Explanation 1", "Evidence 1"),
            MemoPoint("Risk 2", "Explanation 2", "Evidence 2"),
            MemoPoint("Risk 3", "Explanation 3", "Evidence 3"),
            MemoPoint("Risk 4", "Explanation 4", "Evidence 4"),
            MemoPoint("Risk 5", "Explanation 5", "Evidence 5"),
        ),
        key_strengths=(
            MemoPoint("Strength 1", "Explanation 1", "Evidence 1"),
            MemoPoint("Strength 2", "Explanation 2", "Evidence 2"),
            MemoPoint("Strength 3", "Explanation 3", "Evidence 3"),
        ),
        outlook=CreditOutlook(
            direction="Stable",
            justification="Stable performance and liquidity support the outlook.",
        ),
        context_sections=(
            MemoContextSection("business_overview", "Business Overview", "Overview text"),
        ),
    )

    pdf_bytes = build_credit_memo_pdf(
        memo,
        filing_url="https://www.sec.gov/Archives/edgar/data/1065280/000106528026000034/form10k.txt",
        extraction_methodology="Test methodology for PDF rendering.",
    )

    assert pdf_bytes.startswith(b"%PDF")
    assert pdf_bytes.count(b"/Type /Page") >= 5
    assert b"CREDIT RISK MEMO" in pdf_bytes
    assert b"Executive Summary" in pdf_bytes
    assert b"CONFIDENTIAL" in pdf_bytes
    assert b"Page 1 of" in pdf_bytes
    assert b"DATA SOURCES" in pdf_bytes
    assert b"000106528026000034/form10k.txt" in pdf_bytes


def test_build_credit_memo_pdf_handles_missing_chart_ratio_value() -> None:
    summary_rows = []
    for row in build_financial_summary_rows(build_financial_result()):
        if row.label == "Net Debt / EBITDA":
            summary_rows.append(
                FinancialSummaryRow(
                    label=row.label,
                    value="N/A",
                    evidence="Unavailable for this test case.",
                )
            )
        else:
            summary_rows.append(row)

    memo = CreditMemoResult(
        company_name="NETFLIX INC",
        ticker="NFLX",
        filing_date="2026-01-23",
        company_overview="Netflix operates a global streaming entertainment platform.",
        financial_summary=tuple(summary_rows),
        credit_risk_rating=CreditRiskRating(
            rating="BBB",
            justification="Test justification.",
        ),
        key_risk_factors=(
            MemoPoint("Risk 1", "Explanation 1", "Evidence 1"),
            MemoPoint("Risk 2", "Explanation 2", "Evidence 2"),
            MemoPoint("Risk 3", "Explanation 3", "Evidence 3"),
            MemoPoint("Risk 4", "Explanation 4", "Evidence 4"),
            MemoPoint("Risk 5", "Explanation 5", "Evidence 5"),
        ),
        key_strengths=(
            MemoPoint("Strength 1", "Explanation 1", "Evidence 1"),
            MemoPoint("Strength 2", "Explanation 2", "Evidence 2"),
            MemoPoint("Strength 3", "Explanation 3", "Evidence 3"),
        ),
        outlook=CreditOutlook(direction="Stable", justification="Stable outlook."),
        context_sections=(MemoContextSection("business_overview", "Business Overview", "Overview text"),),
    )

    pdf_bytes = build_credit_memo_pdf(memo)

    assert pdf_bytes.startswith(b"%PDF")


def test_build_credit_memo_pdf_includes_quantitative_sections_when_provided() -> None:
    summary_rows = build_financial_summary_rows(build_financial_result())
    memo = CreditMemoResult(
        company_name="NETFLIX INC",
        ticker="NFLX",
        filing_date="2026-01-23",
        company_overview="Netflix operates a global streaming entertainment platform.",
        financial_summary=summary_rows,
        credit_risk_rating=CreditRiskRating(
            rating="A-",
            justification="Strong scale and solid cash generation support the rating.",
        ),
        key_risk_factors=(
            MemoPoint("Risk 1", "Explanation 1", "Evidence 1"),
            MemoPoint("Risk 2", "Explanation 2", "Evidence 2"),
            MemoPoint("Risk 3", "Explanation 3", "Evidence 3"),
            MemoPoint("Risk 4", "Explanation 4", "Evidence 4"),
            MemoPoint("Risk 5", "Explanation 5", "Evidence 5"),
        ),
        key_strengths=(
            MemoPoint("Strength 1", "Explanation 1", "Evidence 1"),
            MemoPoint("Strength 2", "Explanation 2", "Evidence 2"),
            MemoPoint("Strength 3", "Explanation 3", "Evidence 3"),
        ),
        outlook=CreditOutlook(direction="Stable", justification="Stable outlook."),
        context_sections=(MemoContextSection("business_overview", "Business Overview", "Overview text"),),
    )

    pdf_bytes = build_credit_memo_pdf(
        memo,
        quantitative_result=build_quantitative_result(),
    )

    assert b"QUANTITATIVE RISK MODELS" in pdf_bytes
    assert b"YEAR-OVER-YEAR TREND ANALYSIS" in pdf_bytes
    assert b"DEBT MATURITY PROFILE" in pdf_bytes


def test_build_peer_comparison_pdf_returns_pdf_bytes() -> None:
    summary_rows = build_financial_summary_rows(build_financial_result())
    nflx_memo = CreditMemoResult(
        company_name="NETFLIX INC",
        ticker="NFLX",
        filing_date="2026-01-23",
        company_overview="Netflix operates a global streaming entertainment platform.",
        financial_summary=summary_rows,
        credit_risk_rating=CreditRiskRating(
            rating="A-",
            justification="Strong scale and solid cash generation support the rating.",
        ),
        key_risk_factors=(
            MemoPoint("Content Spend", "Explanation 1", "Evidence 1"),
            MemoPoint("Competition", "Explanation 2", "Evidence 2"),
            MemoPoint("Execution", "Explanation 3", "Evidence 3"),
            MemoPoint("Demand Risk", "Explanation 4", "Evidence 4"),
            MemoPoint("Leverage", "Explanation 5", "Evidence 5"),
        ),
        key_strengths=(
            MemoPoint("Scale", "Explanation 1", "Evidence 1"),
            MemoPoint("Liquidity", "Explanation 2", "Evidence 2"),
            MemoPoint("Coverage", "Explanation 3", "Evidence 3"),
        ),
        outlook=CreditOutlook(direction="Stable", justification="Stable outlook."),
        context_sections=(MemoContextSection("business_overview", "Business Overview", "Overview text"),),
    )
    dis_memo = CreditMemoResult(
        company_name="THE WALT DISNEY COMPANY",
        ticker="DIS",
        filing_date="2026-01-12",
        company_overview="Disney is a diversified global media and entertainment company.",
        financial_summary=tuple(
            FinancialSummaryRow(
                label=row.label,
                value="1.80x" if row.label == "Debt-to-Equity" else row.value,
                evidence=row.evidence,
            )
            if row.label == "Debt-to-Equity"
            else row
            for row in summary_rows
        ),
        credit_risk_rating=CreditRiskRating(
            rating="BBB+",
            justification="Diversification supports the profile, offset by higher leverage.",
        ),
        key_risk_factors=(
            MemoPoint("Parks Cyclicality", "Explanation 1", "Evidence 1"),
            MemoPoint("Streaming Profitability", "Explanation 2", "Evidence 2"),
            MemoPoint("Capital Needs", "Explanation 3", "Evidence 3"),
            MemoPoint("Competition", "Explanation 4", "Evidence 4"),
            MemoPoint("Execution", "Explanation 5", "Evidence 5"),
        ),
        key_strengths=(
            MemoPoint("Brand Strength", "Explanation 1", "Evidence 1"),
            MemoPoint("Asset Diversity", "Explanation 2", "Evidence 2"),
            MemoPoint("Liquidity", "Explanation 3", "Evidence 3"),
        ),
        outlook=CreditOutlook(direction="Stable", justification="Stable outlook."),
        context_sections=(MemoContextSection("business_overview", "Business Overview", "Overview text"),),
    )

    pdf_bytes = build_peer_comparison_pdf(
        (nflx_memo, dis_memo),
        filing_urls={
            "NFLX": "https://www.sec.gov/Archives/edgar/data/1065280/000106528026000034/form10k.txt",
            "DIS": "https://www.sec.gov/Archives/edgar/data/1744489/000174448926000010/form10k.txt",
        },
        failed_tickers={"PARA": "No recent 10-K filing was found."},
    )

    assert pdf_bytes.startswith(b"%PDF")
    assert pdf_bytes.count(b"/Type /Page") >= 4
    assert b"PEER CREDIT COMPARISON" in pdf_bytes
    assert b"RATING COMPARISON" in pdf_bytes
    assert b"RATIO COMPARISON" in pdf_bytes
    assert b"NETFLIX INC" in pdf_bytes
    assert b"THE WALT DISNEY COMPANY" in pdf_bytes
    assert b"PARA" in pdf_bytes


def test_generate_credit_memo_raises_clear_error_after_rate_limit_retries_exhausted() -> None:
    result = build_financial_result()

    class DummyResponses:
        def create(self, **kwargs):
            raise build_rate_limit_error()

    class DummyClient:
        responses = DummyResponses()

    with patch("ai_credit_analyst.memo.build_openai_client", return_value=DummyClient()):
        with patch("ai_credit_analyst.extraction.time.sleep") as sleep_mock:
            with pytest.raises(MemoGenerationError, match="Wait 60 seconds and try again"):
                generate_credit_memo(
                    raw_filing_text=RAW_FILING_TEXT,
                    company_name="NETFLIX INC",
                    ticker="NFLX",
                    filing_date="2026-01-23",
                    financial_result=result,
                    api_key="test-key",
                )

    assert [call.args[0] for call in sleep_mock.call_args_list] == [10, 20, 40]
