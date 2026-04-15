from __future__ import annotations

import json
from unittest.mock import patch

from ai_credit_analyst.models import (
    DebtMaturityProfileResult,
    FilingChunk,
    FinancialExtractionResult,
    MetricValue,
    RatioValue,
)
from ai_credit_analyst.quantitative import (
    build_quantitative_model_result,
    build_trend_analysis,
    compute_altman_z_score,
    compute_cash_flow_adequacy,
    compute_piotroski_f_score,
    extract_debt_maturity_schedule,
    reconcile_prior_year_financial_result,
)


def build_financial_result(*, debt_to_equity: float, current_ratio: float, interest_coverage: float, net_debt_to_ebitda: float, revenue: int, net_income: int, operating_cash_flow: int, total_assets: int, total_liabilities: int, total_equity: int, total_debt: int, cash: int, ebit: int, ebitda: int, gross_profit: int, current_portion_debt: int | None) -> FinancialExtractionResult:
    metrics = {
        "revenue": MetricValue(revenue, "Revenue", "income_statement"),
        "net_income": MetricValue(net_income, "Net income", "income_statement"),
        "total_assets": MetricValue(total_assets, "Total assets", "balance_sheet"),
        "total_current_assets": MetricValue(12_000_000_000, "Current assets", "balance_sheet"),
        "total_current_liabilities": MetricValue(10_000_000_000, "Current liabilities", "balance_sheet"),
        "total_liabilities": MetricValue(total_liabilities, "Total liabilities", "balance_sheet"),
        "total_shareholders_equity": MetricValue(total_equity, "Total equity", "balance_sheet"),
        "total_debt": MetricValue(total_debt, "Total debt", "debt_note"),
        "cash_and_cash_equivalents": MetricValue(cash, "Cash", "balance_sheet"),
        "interest_expense": MetricValue(740_000_000, "Interest expense", "income_statement"),
        "ebit": MetricValue(ebit, "Operating income", "income_statement"),
        "ebitda": MetricValue(ebitda, "EBITDA", "income_statement"),
        "operating_cash_flow": MetricValue(operating_cash_flow, "Operating cash flow", "cash_flow_statement"),
        "capital_expenditures": MetricValue(450_000_000, "Capex", "cash_flow_statement"),
        "retained_earnings": MetricValue(15_000_000_000, "Retained earnings", "balance_sheet"),
        "common_stock": MetricValue(1_000_000_000, "Common stock", "balance_sheet"),
        "additional_paid_in_capital": MetricValue(10_000_000_000, "APIC", "balance_sheet"),
        "gross_profit": MetricValue(gross_profit, "Gross profit", "income_statement"),
        "current_portion_of_long_term_debt": MetricValue(current_portion_debt, "Current portion", "debt_note"),
    }
    ratios = {
        "debt_to_equity": RatioValue(debt_to_equity, None),
        "interest_coverage": RatioValue(interest_coverage, None),
        "current_ratio": RatioValue(current_ratio, None),
        "net_debt_to_ebitda": RatioValue(net_debt_to_ebitda, None),
        "free_cash_flow_yield": RatioValue(None, "Skipped because market cap data is not available yet."),
    }
    chunks = (
        FilingChunk("debt_note", "financial_statement_anchors", 0, 100, "2027 $1,000 million senior notes due 2027 at 5.0%"),
    )
    return FinancialExtractionResult(
        metrics=metrics,
        ratios=ratios,
        chunk_strategy="financial_statement_anchors",
        chunk_count=len(chunks),
        prepared_text_length=2000,
        used_primary_document=True,
        chunks=chunks,
    )


def test_compute_altman_z_score_uses_book_value_proxy_when_market_cap_unavailable() -> None:
    current_result = build_financial_result(
        debt_to_equity=1.09,
        current_ratio=1.20,
        interest_coverage=18.54,
        net_debt_to_ebitda=0.40,
        revenue=39_000_000_000,
        net_income=8_700_000_000,
        operating_cash_flow=9_000_000_000,
        total_assets=52_000_000_000,
        total_liabilities=28_300_000_000,
        total_equity=26_000_000_000,
        total_debt=14_500_000_000,
        cash=9_000_000_000,
        ebit=13_700_000_000,
        ebitda=13_700_000_000,
        gross_profit=16_000_000_000,
        current_portion_debt=1_200_000_000,
    )

    with patch("ai_credit_analyst.quantitative.fetch_market_cap", return_value=(None, "Market cap unavailable.")):
        altman = compute_altman_z_score(financial_result=current_result, ticker="NFLX")

    assert altman.score is not None
    assert altman.zone == "Gray Zone"
    assert "Book value used as proxy" in (altman.note or "")


def test_compute_piotroski_f_score_compares_two_years() -> None:
    current_result = build_financial_result(
        debt_to_equity=1.09,
        current_ratio=1.20,
        interest_coverage=18.54,
        net_debt_to_ebitda=0.40,
        revenue=39_000_000_000,
        net_income=8_700_000_000,
        operating_cash_flow=9_000_000_000,
        total_assets=52_000_000_000,
        total_liabilities=28_300_000_000,
        total_equity=26_000_000_000,
        total_debt=14_500_000_000,
        cash=9_000_000_000,
        ebit=13_700_000_000,
        ebitda=13_700_000_000,
        gross_profit=16_000_000_000,
        current_portion_debt=1_200_000_000,
    )
    prior_result = build_financial_result(
        debt_to_equity=1.35,
        current_ratio=1.05,
        interest_coverage=12.10,
        net_debt_to_ebitda=0.90,
        revenue=35_000_000_000,
        net_income=5_800_000_000,
        operating_cash_flow=6_500_000_000,
        total_assets=51_000_000_000,
        total_liabilities=30_500_000_000,
        total_equity=22_500_000_000,
        total_debt=16_000_000_000,
        cash=7_500_000_000,
        ebit=10_500_000_000,
        ebitda=10_900_000_000,
        gross_profit=13_000_000_000,
        current_portion_debt=1_400_000_000,
    )

    f_score = compute_piotroski_f_score(current_result=current_result, prior_result=prior_result)

    assert f_score.score == 9
    assert f_score.interpretation == "Strong"
    assert f_score.evaluated_count == 9
    assert f_score.missing_count == 0
    assert len(f_score.criteria) == 9


def test_compute_piotroski_marks_missing_prior_inputs_as_insufficient_data() -> None:
    current_result = build_financial_result(
        debt_to_equity=1.09,
        current_ratio=1.20,
        interest_coverage=18.54,
        net_debt_to_ebitda=0.40,
        revenue=39_000_000_000,
        net_income=8_700_000_000,
        operating_cash_flow=9_000_000_000,
        total_assets=52_000_000_000,
        total_liabilities=28_300_000_000,
        total_equity=26_000_000_000,
        total_debt=14_500_000_000,
        cash=9_000_000_000,
        ebit=13_700_000_000,
        ebitda=13_700_000_000,
        gross_profit=16_000_000_000,
        current_portion_debt=1_200_000_000,
    )
    prior_result = build_financial_result(
        debt_to_equity=1.35,
        current_ratio=1.05,
        interest_coverage=12.10,
        net_debt_to_ebitda=0.90,
        revenue=35_000_000_000,
        net_income=5_800_000_000,
        operating_cash_flow=6_500_000_000,
        total_assets=51_000_000_000,
        total_liabilities=30_500_000_000,
        total_equity=22_500_000_000,
        total_debt=16_000_000_000,
        cash=7_500_000_000,
        ebit=10_500_000_000,
        ebitda=10_900_000_000,
        gross_profit=13_000_000_000,
        current_portion_debt=1_400_000_000,
    )
    prior_result.metrics["gross_profit"] = MetricValue(None, None, "income_statement")
    prior_result.ratios["current_ratio"] = RatioValue(None, "Data unavailable")

    f_score = compute_piotroski_f_score(current_result=current_result, prior_result=prior_result)

    assert f_score.score == 8
    assert f_score.evaluated_count == 8
    assert f_score.missing_count == 1
    assert any(criterion.passed is None for criterion in f_score.criteria)
    assert "1 criteria lacked sufficient data" in (f_score.note or "")


def test_compute_cash_flow_adequacy_uses_proxy_when_current_portion_missing() -> None:
    current_result = build_financial_result(
        debt_to_equity=1.09,
        current_ratio=1.20,
        interest_coverage=18.54,
        net_debt_to_ebitda=0.40,
        revenue=39_000_000_000,
        net_income=8_700_000_000,
        operating_cash_flow=9_000_000_000,
        total_assets=52_000_000_000,
        total_liabilities=28_300_000_000,
        total_equity=26_000_000_000,
        total_debt=14_500_000_000,
        cash=9_000_000_000,
        ebit=13_700_000_000,
        ebitda=13_700_000_000,
        gross_profit=16_000_000_000,
        current_portion_debt=None,
    )

    adequacy = compute_cash_flow_adequacy(current_result)

    assert adequacy.ratio is not None
    assert adequacy.assessment.startswith("Strong")
    assert "10% of total debt" in (adequacy.note or "")


def test_extract_debt_maturity_schedule_parses_openai_output() -> None:
    current_result = build_financial_result(
        debt_to_equity=1.09,
        current_ratio=1.20,
        interest_coverage=18.54,
        net_debt_to_ebitda=0.40,
        revenue=39_000_000_000,
        net_income=8_700_000_000,
        operating_cash_flow=9_000_000_000,
        total_assets=52_000_000_000,
        total_liabilities=28_300_000_000,
        total_equity=26_000_000_000,
        total_debt=14_500_000_000,
        cash=9_000_000_000,
        ebit=13_700_000_000,
        ebitda=13_700_000_000,
        gross_profit=16_000_000_000,
        current_portion_debt=1_200_000_000,
    )
    payload = {
        "tranches": [
            {
                "amount_millions": 1000,
                "maturity_year": 2027,
                "interest_rate": "5.0%",
                "description": "Senior notes due 2027",
            },
            {
                "amount_millions": 3500,
                "maturity_year": 2028,
                "interest_rate": "4.5%",
                "description": "Senior notes due 2028",
            },
        ]
    }

    class DummyResponses:
        def create(self, **kwargs):
            return type("DummyResponse", (), {"output_text": json.dumps(payload)})()

    class DummyClient:
        responses = DummyResponses()

    with patch("ai_credit_analyst.quantitative.build_openai_client", return_value=DummyClient()):
        profile = extract_debt_maturity_schedule(
            raw_filing_text="LONG-TERM DEBT note text",
            company_name="NETFLIX INC",
            ticker="NFLX",
            filing_date="2026-01-23",
            financial_result=current_result,
            api_key="test-key",
            model="gpt-4o",
        )

    assert len(profile.tranches) == 2
    assert profile.weighted_average_maturity is not None
    assert any(warning.maturity_year == 2028 for warning in profile.warnings)


def test_extract_debt_maturity_schedule_filters_hallucinated_tranches() -> None:
    current_result = build_financial_result(
        debt_to_equity=1.09,
        current_ratio=1.20,
        interest_coverage=18.54,
        net_debt_to_ebitda=0.40,
        revenue=39_000_000_000,
        net_income=8_700_000_000,
        operating_cash_flow=9_000_000_000,
        total_assets=52_000_000_000,
        total_liabilities=28_300_000_000,
        total_equity=26_000_000_000,
        total_debt=14_500_000_000,
        cash=9_000_000_000,
        ebit=13_700_000_000,
        ebitda=13_700_000_000,
        gross_profit=16_000_000_000,
        current_portion_debt=1_200_000_000,
    )
    payload = {
        "tranches": [
            {
                "amount_millions": 59000,
                "maturity_year": 2027,
                "interest_rate": None,
                "description": "Bridge loan",
            },
            {
                "amount_millions": 1000,
                "maturity_year": 2028,
                "interest_rate": "4.5%",
                "description": "Senior notes due 2028",
            },
        ]
    }

    class DummyResponses:
        def create(self, **kwargs):
            return type("DummyResponse", (), {"output_text": json.dumps(payload)})()

    class DummyClient:
        responses = DummyResponses()

    with patch("ai_credit_analyst.quantitative.build_openai_client", return_value=DummyClient()):
        profile = extract_debt_maturity_schedule(
            raw_filing_text="LONG-TERM DEBT note text",
            company_name="NETFLIX INC",
            ticker="NFLX",
            filing_date="2026-01-23",
            financial_result=current_result,
            api_key="test-key",
            model="gpt-4o",
        )

    assert len(profile.tranches) == 1
    assert profile.tranches[0].description == "Senior notes due 2028"
    assert "Excluded 1 tranche" in (profile.note or "")


def test_extract_debt_maturity_schedule_discards_unreliable_schedule_and_marks_historical_only() -> None:
    current_result = build_financial_result(
        debt_to_equity=1.09,
        current_ratio=1.20,
        interest_coverage=18.54,
        net_debt_to_ebitda=0.40,
        revenue=39_000_000_000,
        net_income=8_700_000_000,
        operating_cash_flow=9_000_000_000,
        total_assets=52_000_000_000,
        total_liabilities=28_300_000_000,
        total_equity=26_000_000_000,
        total_debt=14_500_000_000,
        cash=9_000_000_000,
        ebit=13_700_000_000,
        ebitda=13_700_000_000,
        gross_profit=16_000_000_000,
        current_portion_debt=1_200_000_000,
    )

    unreliable_payload = {
        "tranches": [
            {"amount_millions": 12000, "maturity_year": 2027, "interest_rate": None, "description": "A"},
            {"amount_millions": 11000, "maturity_year": 2028, "interest_rate": None, "description": "B"},
        ]
    }
    historical_payload = {
        "tranches": [
            {"amount_millions": 1000, "maturity_year": 2024, "interest_rate": None, "description": "A"},
            {"amount_millions": 500, "maturity_year": 2025, "interest_rate": None, "description": "B"},
        ]
    }

    class DummyResponses:
        def __init__(self, payload):
            self._payload = payload

        def create(self, **kwargs):
            return type("DummyResponse", (), {"output_text": json.dumps(self._payload)})()

    class DummyClient:
        def __init__(self, payload):
            self.responses = DummyResponses(payload)

    with patch("ai_credit_analyst.quantitative.build_openai_client", return_value=DummyClient(unreliable_payload)):
        unreliable_profile = extract_debt_maturity_schedule(
            raw_filing_text="LONG-TERM DEBT note text",
            company_name="NETFLIX INC",
            ticker="NFLX",
            filing_date="2026-01-23",
            financial_result=current_result,
            api_key="test-key",
            model="gpt-4o",
        )

    with patch("ai_credit_analyst.quantitative.build_openai_client", return_value=DummyClient(historical_payload)):
        historical_profile = extract_debt_maturity_schedule(
            raw_filing_text="LONG-TERM DEBT note text",
            company_name="NETFLIX INC",
            ticker="NFLX",
            filing_date="2026-01-23",
            financial_result=current_result,
            api_key="test-key",
            model="gpt-4o",
        )

    assert unreliable_profile.tranches == tuple()
    assert unreliable_profile.note == "Maturity schedule could not be reliably extracted from this filing."
    assert historical_profile.weighted_average_maturity is None
    assert "historical" in (historical_profile.note or "").lower()


def test_build_trend_analysis_marks_missing_prior_values_as_unavailable() -> None:
    current_result = build_financial_result(
        debt_to_equity=1.09,
        current_ratio=1.20,
        interest_coverage=18.54,
        net_debt_to_ebitda=0.40,
        revenue=39_000_000_000,
        net_income=8_700_000_000,
        operating_cash_flow=9_000_000_000,
        total_assets=52_000_000_000,
        total_liabilities=28_300_000_000,
        total_equity=26_000_000_000,
        total_debt=14_500_000_000,
        cash=9_000_000_000,
        ebit=13_700_000_000,
        ebitda=13_700_000_000,
        gross_profit=16_000_000_000,
        current_portion_debt=1_200_000_000,
    )
    prior_result = build_financial_result(
        debt_to_equity=1.35,
        current_ratio=0.0,
        interest_coverage=12.10,
        net_debt_to_ebitda=0.90,
        revenue=0,
        net_income=5_800_000_000,
        operating_cash_flow=6_500_000_000,
        total_assets=51_000_000_000,
        total_liabilities=30_500_000_000,
        total_equity=22_500_000_000,
        total_debt=16_000_000_000,
        cash=7_500_000_000,
        ebit=10_500_000_000,
        ebitda=10_900_000_000,
        gross_profit=13_000_000_000,
        current_portion_debt=1_400_000_000,
    )

    rows, note, _ = build_trend_analysis(current_result=current_result, prior_result=prior_result)

    revenue_row = next(row for row in rows if row.label == "Revenue")
    current_ratio_row = next(row for row in rows if row.label == "Current Ratio")

    assert note is None
    assert revenue_row.prior_value == 0
    assert revenue_row.absolute_change is None
    assert revenue_row.percent_change is None
    assert revenue_row.improving is None
    assert revenue_row.prior_data_reliable is False
    assert current_ratio_row.prior_value == 0.0
    assert current_ratio_row.absolute_change is None
    assert current_ratio_row.percent_change is None
    assert current_ratio_row.improving is None
    assert current_ratio_row.prior_data_reliable is False


def test_build_trend_analysis_flags_extreme_ratio_changes_for_review() -> None:
    current_result = build_financial_result(
        debt_to_equity=1.09,
        current_ratio=1.20,
        interest_coverage=18.54,
        net_debt_to_ebitda=0.40,
        revenue=39_000_000_000,
        net_income=8_700_000_000,
        operating_cash_flow=9_000_000_000,
        total_assets=52_000_000_000,
        total_liabilities=28_300_000_000,
        total_equity=26_000_000_000,
        total_debt=14_500_000_000,
        cash=9_000_000_000,
        ebit=13_700_000_000,
        ebitda=13_700_000_000,
        gross_profit=16_000_000_000,
        current_portion_debt=1_200_000_000,
    )
    prior_result = build_financial_result(
        debt_to_equity=1.35,
        current_ratio=1.05,
        interest_coverage=2.0,
        net_debt_to_ebitda=0.90,
        revenue=35_000_000_000,
        net_income=5_800_000_000,
        operating_cash_flow=6_500_000_000,
        total_assets=51_000_000_000,
        total_liabilities=30_500_000_000,
        total_equity=22_500_000_000,
        total_debt=16_000_000_000,
        cash=7_500_000_000,
        ebit=10_500_000_000,
        ebitda=10_900_000_000,
        gross_profit=13_000_000_000,
        current_portion_debt=1_400_000_000,
    )

    rows, _, _ = build_trend_analysis(current_result=current_result, prior_result=prior_result)

    interest_coverage_row = next(row for row in rows if row.label == "Interest Coverage")
    assert interest_coverage_row.note == "⚠ Data quality - verify prior year extraction"
    assert interest_coverage_row.percent_change is None
    assert interest_coverage_row.improving is None
    assert interest_coverage_row.prior_data_reliable is False


def test_reconcile_prior_year_financial_result_rescales_likely_unit_mismatches() -> None:
    current_result = build_financial_result(
        debt_to_equity=1.09,
        current_ratio=1.20,
        interest_coverage=18.54,
        net_debt_to_ebitda=0.40,
        revenue=39_000_000_000,
        net_income=11_000_000_000,
        operating_cash_flow=9_000_000_000,
        total_assets=52_000_000_000,
        total_liabilities=28_300_000_000,
        total_equity=26_000_000_000,
        total_debt=14_500_000_000,
        cash=9_000_000_000,
        ebit=13_700_000_000,
        ebitda=13_700_000_000,
        gross_profit=16_000_000_000,
        current_portion_debt=1_200_000_000,
    )
    prior_result = build_financial_result(
        debt_to_equity=1.35,
        current_ratio=1.05,
        interest_coverage=12.10,
        net_debt_to_ebitda=0.90,
        revenue=35_000_000_000,
        net_income=3_300_000,
        operating_cash_flow=6_500_000,
        total_assets=51_000_000_000,
        total_liabilities=30_500_000_000,
        total_equity=22_500_000_000,
        total_debt=16_000_000_000,
        cash=7_500_000_000,
        ebit=10_500_000_000,
        ebitda=10_900_000_000,
        gross_profit=13_000_000,
        current_portion_debt=1_400_000_000,
    )

    reconciled = reconcile_prior_year_financial_result(
        current_result=current_result,
        prior_result=prior_result,
    )

    assert reconciled is not None
    assert reconciled.metrics["net_income"].value == 3_300_000_000
    assert reconciled.metrics["operating_cash_flow"].value == 6_500_000_000
    assert "scale reconciliation" in reconciled.normalization_note.lower()


def test_build_quantitative_model_result_handles_debt_extraction_failure_gracefully() -> None:
    current_result = build_financial_result(
        debt_to_equity=1.09,
        current_ratio=1.20,
        interest_coverage=18.54,
        net_debt_to_ebitda=0.40,
        revenue=39_000_000_000,
        net_income=8_700_000_000,
        operating_cash_flow=9_000_000_000,
        total_assets=52_000_000_000,
        total_liabilities=28_300_000_000,
        total_equity=26_000_000_000,
        total_debt=14_500_000_000,
        cash=9_000_000_000,
        ebit=13_700_000_000,
        ebitda=13_700_000_000,
        gross_profit=16_000_000_000,
        current_portion_debt=1_200_000_000,
    )

    with patch("ai_credit_analyst.quantitative.fetch_market_cap", return_value=(5_000_000_000, None)):
        with patch("ai_credit_analyst.quantitative.extract_debt_maturity_schedule", side_effect=RuntimeError("boom")):
            quantitative_result = build_quantitative_model_result(
                company_name="NETFLIX INC",
                ticker="NFLX",
                filing_date="2026-01-23",
                raw_filing_text="raw filing",
                current_result=current_result,
                prior_result=None,
                api_key="test-key",
            )

    assert quantitative_result.altman_z_score.score is not None
    assert quantitative_result.debt_maturity_profile.note == "Debt maturity extraction failed: boom"
    assert quantitative_result.prior_year_available is False
