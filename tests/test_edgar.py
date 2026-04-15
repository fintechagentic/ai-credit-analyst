from __future__ import annotations

from unittest.mock import patch

import pytest

from ai_credit_analyst.edgar import (
    EdgarClient,
    build_sec_user_agent,
    fetch_latest_10k,
    fetch_recent_10ks,
)
from ai_credit_analyst.exceptions import (
    ConfigurationError,
    FilingNotFoundError,
    TickerNotFoundError,
)
from ai_credit_analyst.models import FilingMetadata


def build_client() -> EdgarClient:
    return EdgarClient(user_agent="AI Credit Analyst Agent test@example.com")


def test_build_sec_user_agent_requires_name_and_email() -> None:
    with pytest.raises(ConfigurationError):
        build_sec_user_agent(requester_name="", requester_email="")

    with pytest.raises(ConfigurationError):
        build_sec_user_agent(requester_name="Credit Agent", requester_email="invalid")


def test_get_latest_10k_metadata_selects_first_recent_10k() -> None:
    client = build_client()
    ticker_payload = {
        "0": {"ticker": "NFLX", "title": "Netflix, Inc.", "cik_str": 1065280}
    }
    submissions_payload = {
        "name": "NETFLIX INC",
        "filings": {
            "recent": {
                "form": ["8-K", "10-Q", "10-K", "8-K"],
                "filingDate": ["2026-02-01", "2025-10-18", "2025-01-26", "2025-01-03"],
                "accessionNumber": [
                    "0001065280-26-000010",
                    "0001065280-25-000040",
                    "0001065280-25-000006",
                    "0001065280-25-000001",
                ],
            }
        },
    }

    with patch.object(client, "_get_json", side_effect=[ticker_payload, submissions_payload]):
        metadata = client.get_latest_10k_metadata("nflx")

    assert metadata == FilingMetadata(
        ticker="NFLX",
        company_name="NETFLIX INC",
        cik="0001065280",
        accession_number="0001065280-25-000006",
        filing_date="2025-01-26",
        form="10-K",
        filing_text_url="https://www.sec.gov/Archives/edgar/data/1065280/0001065280-25-000006.txt",
        submissions_url="https://data.sec.gov/submissions/CIK0001065280.json",
    )


def test_get_recent_10k_metadata_returns_two_most_recent_filings() -> None:
    client = build_client()
    ticker_payload = {
        "0": {"ticker": "NFLX", "title": "Netflix, Inc.", "cik_str": 1065280}
    }
    submissions_payload = {
        "name": "NETFLIX INC",
        "filings": {
            "recent": {
                "form": ["10-Q", "10-K", "8-K", "10-K", "8-K"],
                "filingDate": ["2026-03-31", "2026-01-23", "2025-11-05", "2025-01-26", "2025-01-03"],
                "accessionNumber": [
                    "0001065280-26-000041",
                    "0001065280-26-000034",
                    "0001065280-25-000071",
                    "0001065280-25-000006",
                    "0001065280-25-000001",
                ],
            }
        },
    }

    with patch.object(client, "_get_json", side_effect=[ticker_payload, submissions_payload]):
        metadata = client.get_recent_10k_metadata("NFLX", limit=2)

    assert len(metadata) == 2
    assert metadata[0].accession_number == "0001065280-26-000034"
    assert metadata[1].accession_number == "0001065280-25-000006"


def test_get_latest_10k_metadata_raises_for_unknown_ticker() -> None:
    client = build_client()
    with patch.object(client, "_get_json", return_value={}):
        with pytest.raises(TickerNotFoundError):
            client.get_latest_10k_metadata("MISSING")


def test_get_latest_10k_metadata_raises_when_recent_window_has_no_10k() -> None:
    client = build_client()
    ticker_payload = {"0": {"ticker": "XYZ", "title": "XYZ Corp", "cik_str": 123456}}
    submissions_payload = {
        "name": "XYZ CORP",
        "filings": {
            "recent": {
                "form": ["8-K", "10-Q"],
                "filingDate": ["2026-02-01", "2025-10-18"],
                "accessionNumber": ["0000123456-26-000010", "0000123456-25-000004"],
            }
        },
    }

    with patch.object(client, "_get_json", side_effect=[ticker_payload, submissions_payload]):
        with pytest.raises(FilingNotFoundError):
            client.get_latest_10k_metadata("XYZ")


def test_fetch_latest_10k_downloads_text_after_metadata_lookup() -> None:
    metadata = FilingMetadata(
        ticker="NFLX",
        company_name="NETFLIX INC",
        cik="0001065280",
        accession_number="0001065280-25-000006",
        filing_date="2025-01-26",
        form="10-K",
        filing_text_url="https://www.sec.gov/Archives/edgar/data/1065280/0001065280-25-000006.txt",
        submissions_url="https://data.sec.gov/submissions/CIK0001065280.json",
    )

    with patch("ai_credit_analyst.edgar.EdgarClient") as client_cls:
        client_instance = client_cls.return_value
        client_instance.fetch_latest_10k.return_value = (metadata, "RAW FILING")

        resolved_metadata, filing_text = fetch_latest_10k(
            ticker="NFLX",
            requester_name="AI Credit Analyst Agent",
            requester_email="test@example.com",
        )

    assert resolved_metadata == metadata
    assert filing_text == "RAW FILING"
    client_cls.assert_called_once_with(
        user_agent="AI Credit Analyst Agent test@example.com"
    )


def test_fetch_recent_10ks_downloads_each_recent_filing() -> None:
    metadata = FilingMetadata(
        ticker="NFLX",
        company_name="NETFLIX INC",
        cik="0001065280",
        accession_number="0001065280-26-000034",
        filing_date="2026-01-23",
        form="10-K",
        filing_text_url="https://www.sec.gov/Archives/edgar/data/1065280/0001065280-26-000034.txt",
        submissions_url="https://data.sec.gov/submissions/CIK0001065280.json",
    )
    prior_metadata = FilingMetadata(
        ticker="NFLX",
        company_name="NETFLIX INC",
        cik="0001065280",
        accession_number="0001065280-25-000006",
        filing_date="2025-01-26",
        form="10-K",
        filing_text_url="https://www.sec.gov/Archives/edgar/data/1065280/0001065280-25-000006.txt",
        submissions_url="https://data.sec.gov/submissions/CIK0001065280.json",
    )

    with patch("ai_credit_analyst.edgar.EdgarClient") as client_cls:
        client_instance = client_cls.return_value
        client_instance.fetch_recent_10ks.return_value = (
            (metadata, "CURRENT"),
            (prior_metadata, "PRIOR"),
        )

        filings = fetch_recent_10ks(
            ticker="NFLX",
            requester_name="AI Credit Analyst Agent",
            requester_email="test@example.com",
            limit=2,
        )

    assert len(filings) == 2
    assert filings[0][1] == "CURRENT"
    assert filings[1][1] == "PRIOR"
