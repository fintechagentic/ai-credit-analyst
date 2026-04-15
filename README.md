# AI Credit Analyst Agent

MVP scaffold for an autonomous credit analyst workflow that starts by retrieving the latest 10-K filing for a public U.S. company directly from SEC EDGAR raw filing text.

## Retrieval flow

The current proof of concept uses these SEC endpoints:

1. `https://www.sec.gov/files/company_tickers.json`
   Returns JSON keyed by integer-like strings. Each value contains at least:
   - `ticker`
   - `title`
   - `cik_str`

2. `https://data.sec.gov/submissions/CIK##########.json`
   Returns a JSON object for a company. The app reads:
   - `name`
   - `filings.recent.form`
   - `filings.recent.filingDate`
   - `filings.recent.accessionNumber`

3. `https://www.sec.gov/Archives/edgar/data/{cik_without_leading_zeroes}/{accession_number}.txt`
   Returns the complete raw filing submission text as plain text.

SEC requires a descriptive `User-Agent` header that includes a requester or organization name and a contact email. The app will not call EDGAR until those fields are supplied.

## OpenAI extraction flow

The financial extraction step uses the OpenAI Responses API:

1. `POST https://api.openai.com/v1/responses`
   Request requirements:
   - Header: `Authorization: Bearer <OPENAI_API_KEY>`
   - Header: `Content-Type: application/json`
   - Body: model `gpt-4o`, a system instruction block, a user prompt containing a targeted filing excerpt, and `text.format` with a strict JSON schema

   Response format:
   - JSON `response` object from the Responses API
   - The Python SDK exposes the structured model output through `response.output_text`

The extraction pipeline does not send the full SEC submission to the model at once. It:

1. Isolates the primary `10-K` document from the EDGAR submission when possible
2. Cleans SGML / HTML into text while preserving line structure
3. Searches for targeted anchors such as:
   - `CONSOLIDATED BALANCE SHEETS`
   - `CONSOLIDATED STATEMENTS OF OPERATIONS`
   - `CONSOLIDATED STATEMENTS OF CASH FLOWS`
   - debt and property / depreciation note headings
4. Falls back to sliding-window chunking only if those anchors are not found
5. Calls GPT-4o on each relevant chunk and merges the first valid value for each required metric
6. Computes credit ratios locally after extraction

## OpenAI memo generation flow

The memo generation step also uses the OpenAI Responses API:

1. `POST https://api.openai.com/v1/responses`
   Request requirements:
   - Header: `Authorization: Bearer <OPENAI_API_KEY>`
   - Header: `Content-Type: application/json`
   - Body: model `gpt-4o`, a senior-analyst system prompt, structured user context containing financial summary rows and targeted filing excerpts, and `text.format` with a strict JSON schema

   Response format:
   - JSON `response` object from the Responses API
   - The Python SDK exposes the structured model output through `response.output_text`

The memo pipeline:

1. Builds a deterministic financial summary table from extracted metrics and ratios
2. Pulls targeted narrative evidence from the cleaned 10-K, including business, risk factor, liquidity, and debt-related sections
3. Sends that context to GPT-4o with conservative rating calibration instructions
4. Parses the memo into structured JSON sections
5. Renders the result in Streamlit and exports it as a PDF report

## Assumptions

- The latest annual report for a standard U.S. public company appears in `filings.recent`, which the SEC documents as the recent submissions window.
- For this retrieval-only milestone, the app targets exact `10-K` forms and does not treat `10-K/A` as the primary annual filing.
- The downloaded `.txt` filing is the full EDGAR submission text, which may include SGML wrappers and exhibits.
- The extraction pipeline assumes the primary `10-K` document can usually be isolated from the SEC submission's `<DOCUMENT>` blocks.
- `EBIT` is extracted as operating income / income from operations.
- `EBITDA` is extracted only if explicitly disclosed or if both operating income and depreciation and amortization are explicit in the same model chunk.
- `Free Cash Flow Yield` is intentionally left blank until market cap is added.
- The financial summary table in the memo is rendered from deterministic local values rather than relying on the model to restate numbers.
- The memo model is limited to the supplied filing excerpts and extracted financial context; it does not pull market or peer data.

## Local run

1. Create and activate a Python environment.
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Launch the Streamlit app:

```powershell
streamlit run app.py
```

4. In the sidebar, enter:
   - `SEC requester / organization`
   - `SEC contact email`
   - `OpenAI API key`

## Project layout

- `app.py`: Streamlit proof-of-concept UI
- `src/ai_credit_analyst/edgar.py`: SEC lookup and filing download logic
- `src/ai_credit_analyst/extraction.py`: filing chunking, GPT-4o extraction, and ratio computation
- `src/ai_credit_analyst/memo.py`: credit memo generation, memo context assembly, and PDF export
- `src/ai_credit_analyst/models.py`: retrieval and extraction dataclasses
- `src/ai_credit_analyst/exceptions.py`: domain-specific error types
- `tests/test_edgar.py`: retrieval unit tests
- `tests/test_extraction.py`: extraction and ratio unit tests
- `tests/test_memo.py`: memo generation and PDF unit tests
