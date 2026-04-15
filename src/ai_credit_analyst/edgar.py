from __future__ import annotations

import os
import time
from typing import Any

import requests
from requests import Response, Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ai_credit_analyst.exceptions import (
    ConfigurationError,
    EdgarRateLimitError,
    EdgarResponseError,
    FilingNotFoundError,
    TickerNotFoundError,
)
from ai_credit_analyst.models import CompanyMatch, FilingMetadata

SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL_TEMPLATE = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVES_FILING_TEXT_URL_TEMPLATE = (
    "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}.txt"
)

REQUEST_TIMEOUT_SECONDS = 30
MIN_REQUEST_INTERVAL_SECONDS = 0.2


def build_sec_user_agent(
    requester_name: str | None = None,
    requester_email: str | None = None,
) -> str:
    name = (requester_name or os.getenv("SEC_REQUESTER_NAME", "")).strip()
    email = (requester_email or os.getenv("SEC_REQUESTER_EMAIL", "")).strip()

    if not name or not email:
        raise ConfigurationError(
            "SEC EDGAR requires a requester name and contact email. "
            "Enter both fields in the app sidebar or set SEC_REQUESTER_NAME "
            "and SEC_REQUESTER_EMAIL in the environment."
        )

    if "@" not in email:
        raise ConfigurationError(
            "The SEC contact email must be a valid email address."
        )

    return f"{name} {email}"


def build_filing_text_url(cik: str, accession_number: str) -> str:
    cik_without_leading_zeroes = str(int(cik))
    return SEC_ARCHIVES_FILING_TEXT_URL_TEMPLATE.format(
        cik=cik_without_leading_zeroes,
        accession=accession_number,
    )


def _build_retry_strategy() -> Retry:
    return Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )


def _build_session(user_agent: str) -> Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": user_agent,
            "Accept": "application/json, text/plain, */*",
            "Accept-Encoding": "gzip, deflate",
        }
    )
    adapter = HTTPAdapter(max_retries=_build_retry_strategy())
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


class EdgarClient:
    def __init__(
        self,
        user_agent: str,
        timeout_seconds: int = REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        if not user_agent.strip():
            raise ConfigurationError("SEC EDGAR requests require a non-empty User-Agent.")

        self.user_agent = user_agent
        self.timeout_seconds = timeout_seconds
        self.session = _build_session(user_agent)
        self._last_request_timestamp = 0.0

    def _wait_for_fair_access(self) -> None:
        elapsed = time.monotonic() - self._last_request_timestamp
        if elapsed < MIN_REQUEST_INTERVAL_SECONDS:
            time.sleep(MIN_REQUEST_INTERVAL_SECONDS - elapsed)

    def _request(self, url: str) -> Response:
        self._wait_for_fair_access()

        try:
            response = self.session.get(url, timeout=self.timeout_seconds)
        except requests.RequestException as exc:
            raise EdgarResponseError(f"Request to SEC EDGAR failed: {exc}") from exc
        finally:
            self._last_request_timestamp = time.monotonic()

        if "Request Rate Threshold Exceeded" in response.text:
            raise EdgarRateLimitError(
                "SEC EDGAR rejected the request because the fair access threshold was exceeded. "
                "Wait a moment and retry."
            )

        if response.status_code >= 400:
            raise EdgarResponseError(
                f"SEC EDGAR returned HTTP {response.status_code} for {url}."
            )

        return response

    def _get_json(self, url: str) -> dict[str, Any]:
        response = self._request(url)
        try:
            payload = response.json()
        except ValueError as exc:
            raise EdgarResponseError(
                f"SEC EDGAR returned a non-JSON response for {url}."
            ) from exc

        if not isinstance(payload, dict):
            raise EdgarResponseError(f"Unexpected JSON payload type from {url}.")

        return payload

    def _get_text(self, url: str) -> str:
        response = self._request(url)
        if not response.text.strip():
            raise EdgarResponseError(f"SEC EDGAR returned an empty response for {url}.")
        return response.text

    def resolve_company(self, ticker: str) -> CompanyMatch:
        normalized_ticker = ticker.strip().upper()
        if not normalized_ticker:
            raise TickerNotFoundError("Enter a ticker symbol before requesting a filing.")

        payload = self._get_json(SEC_COMPANY_TICKERS_URL)

        for company in payload.values():
            if not isinstance(company, dict):
                continue

            candidate_ticker = str(company.get("ticker", "")).upper()
            if candidate_ticker != normalized_ticker:
                continue

            cik_value = company.get("cik_str")
            title = str(company.get("title", "")).strip()
            if cik_value is None or not title:
                raise EdgarResponseError(
                    "SEC ticker lookup returned incomplete company metadata."
                )

            return CompanyMatch(
                ticker=normalized_ticker,
                cik=str(cik_value).zfill(10),
                name=title,
                source_url=SEC_COMPANY_TICKERS_URL,
            )

        raise TickerNotFoundError(
            f"Ticker '{normalized_ticker}' was not found in SEC company_tickers.json."
        )

    def get_recent_10k_metadata(
        self,
        ticker: str,
        *,
        limit: int = 2,
    ) -> tuple[FilingMetadata, ...]:
        if limit < 1:
            raise FilingNotFoundError("At least one recent 10-K filing must be requested.")

        company = self.resolve_company(ticker)
        submissions_url = SEC_SUBMISSIONS_URL_TEMPLATE.format(cik=company.cik)
        payload = self._get_json(submissions_url)

        company_name = str(payload.get("name", company.name)).strip() or company.name
        filings = payload.get("filings")
        if not isinstance(filings, dict):
            raise EdgarResponseError("SEC submissions response is missing the filings object.")

        recent = filings.get("recent")
        if not isinstance(recent, dict):
            raise EdgarResponseError("SEC submissions response is missing filings.recent.")

        forms = recent.get("form")
        filing_dates = recent.get("filingDate")
        accession_numbers = recent.get("accessionNumber")

        if not isinstance(forms, list) or not isinstance(filing_dates, list) or not isinstance(
            accession_numbers, list
        ):
            raise EdgarResponseError(
                "SEC submissions response does not contain the expected recent filing arrays."
            )

        if not (len(forms) == len(filing_dates) == len(accession_numbers)):
            raise EdgarResponseError(
                "SEC submissions response returned inconsistent recent filing array lengths."
            )

        matches: list[FilingMetadata] = []
        for index, form in enumerate(forms):
            if form != "10-K":
                continue

            accession_number = str(accession_numbers[index]).strip()
            filing_date = str(filing_dates[index]).strip()
            if not accession_number or not filing_date:
                raise EdgarResponseError("SEC submissions response contained an incomplete 10-K entry.")

            matches.append(
                FilingMetadata(
                    ticker=company.ticker,
                    company_name=company_name,
                    cik=company.cik,
                    accession_number=accession_number,
                    filing_date=filing_date,
                    form="10-K",
                    filing_text_url=build_filing_text_url(company.cik, accession_number),
                    submissions_url=submissions_url,
                )
            )
            if len(matches) >= limit:
                break

        if not matches:
            raise FilingNotFoundError(
                f"No 10-K filing was found in the recent SEC submissions window for {company.ticker}."
            )
        return tuple(matches)

    def get_latest_10k_metadata(self, ticker: str) -> FilingMetadata:
        return self.get_recent_10k_metadata(ticker, limit=1)[0]

    def download_filing_text(self, metadata: FilingMetadata) -> str:
        return self._get_text(metadata.filing_text_url)

    def fetch_latest_10k(self, ticker: str) -> tuple[FilingMetadata, str]:
        metadata = self.get_latest_10k_metadata(ticker)
        filing_text = self.download_filing_text(metadata)
        return metadata, filing_text

    def fetch_recent_10ks(
        self,
        ticker: str,
        *,
        limit: int = 2,
    ) -> tuple[tuple[FilingMetadata, str], ...]:
        recent_metadata = self.get_recent_10k_metadata(ticker, limit=limit)
        return tuple(
            (metadata, self.download_filing_text(metadata))
            for metadata in recent_metadata
        )


def fetch_latest_10k(
    ticker: str,
    requester_name: str | None = None,
    requester_email: str | None = None,
) -> tuple[FilingMetadata, str]:
    user_agent = build_sec_user_agent(
        requester_name=requester_name,
        requester_email=requester_email,
    )
    client = EdgarClient(user_agent=user_agent)
    return client.fetch_latest_10k(ticker)


def fetch_recent_10ks(
    ticker: str,
    requester_name: str | None = None,
    requester_email: str | None = None,
    *,
    limit: int = 2,
) -> tuple[tuple[FilingMetadata, str], ...]:
    user_agent = build_sec_user_agent(
        requester_name=requester_name,
        requester_email=requester_email,
    )
    client = EdgarClient(user_agent=user_agent)
    return client.fetch_recent_10ks(ticker, limit=limit)
