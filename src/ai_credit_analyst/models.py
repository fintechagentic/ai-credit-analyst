from __future__ import annotations

from dataclasses import dataclass

Number = int | float


@dataclass(frozen=True)
class CompanyMatch:
    ticker: str
    cik: str
    name: str
    source_url: str


@dataclass(frozen=True)
class FilingMetadata:
    ticker: str
    company_name: str
    cik: str
    accession_number: str
    filing_date: str
    form: str
    filing_text_url: str
    submissions_url: str


@dataclass(frozen=True)
class FilingChunk:
    label: str
    strategy: str
    start_char: int
    end_char: int
    text: str


@dataclass(frozen=True)
class MetricValue:
    value: Number | None
    source_quote: str | None
    source_chunk_label: str | None = None


@dataclass(frozen=True)
class RatioValue:
    value: float | None
    note: str | None = None


@dataclass(frozen=True)
class FinancialExtractionResult:
    metrics: dict[str, MetricValue]
    ratios: dict[str, RatioValue]
    chunk_strategy: str
    chunk_count: int
    prepared_text_length: int
    used_primary_document: bool
    chunks: tuple[FilingChunk, ...]
    detected_statement_unit_label: str = "dollars"
    detected_statement_unit_multiplier: int = 1
    normalization_note: str = "Values already in dollars; no normalization applied."


@dataclass(frozen=True)
class FinancialSummaryRow:
    label: str
    value: str
    evidence: str


@dataclass(frozen=True)
class MemoContextSection:
    label: str
    title: str
    text: str


@dataclass(frozen=True)
class MemoPoint:
    title: str
    explanation: str
    evidence: str


@dataclass(frozen=True)
class CreditRiskRating:
    rating: str
    justification: str


@dataclass(frozen=True)
class CreditOutlook:
    direction: str
    justification: str


@dataclass(frozen=True)
class CreditMemoResult:
    company_name: str
    ticker: str
    filing_date: str
    company_overview: str
    financial_summary: tuple[FinancialSummaryRow, ...]
    credit_risk_rating: CreditRiskRating
    key_risk_factors: tuple[MemoPoint, ...]
    key_strengths: tuple[MemoPoint, ...]
    outlook: CreditOutlook
    context_sections: tuple[MemoContextSection, ...]


@dataclass(frozen=True)
class AltmanComponent:
    label: str
    raw_value: float | None
    weighted_contribution: float | None
    description: str


@dataclass(frozen=True)
class AltmanZScoreResult:
    score: float | None
    zone: str
    note: str | None
    components: tuple[AltmanComponent, ...]


@dataclass(frozen=True)
class PiotroskiCriterion:
    label: str
    passed: bool | None
    note: str


@dataclass(frozen=True)
class PiotroskiFScoreResult:
    score: int | None
    interpretation: str
    note: str | None
    criteria: tuple[PiotroskiCriterion, ...]
    evaluated_count: int = 0
    missing_count: int = 0
    total_criteria: int = 9


@dataclass(frozen=True)
class CashFlowAdequacyResult:
    ratio: float | None
    assessment: str
    note: str | None
    near_term_debt_used: float | None


@dataclass(frozen=True)
class TrendAnalysisRow:
    label: str
    current_value: float | None
    prior_value: float | None
    absolute_change: float | None
    percent_change: float | None
    improving: bool | None
    prefers_higher: bool
    is_ratio: bool
    note: str | None = None
    prior_data_reliable: bool = True


@dataclass(frozen=True)
class DebtMaturityTranche:
    amount_millions: float
    maturity_year: int
    interest_rate: str | None
    description: str


@dataclass(frozen=True)
class MaturityWallWarning:
    maturity_year: int
    percentage_of_total_debt: float


@dataclass(frozen=True)
class DebtMaturityProfileResult:
    tranches: tuple[DebtMaturityTranche, ...]
    weighted_average_maturity: float | None
    warnings: tuple[MaturityWallWarning, ...]
    note: str | None


@dataclass(frozen=True)
class QuantitativeModelResult:
    altman_z_score: AltmanZScoreResult
    piotroski_f_score: PiotroskiFScoreResult
    cash_flow_adequacy: CashFlowAdequacyResult
    trend_analysis: tuple[TrendAnalysisRow, ...]
    trend_analysis_note: str | None
    ratio_history: dict[str, tuple[float | None, float | None]]
    debt_maturity_profile: DebtMaturityProfileResult
    prior_year_available: bool
    prior_year_filing_date: str | None
