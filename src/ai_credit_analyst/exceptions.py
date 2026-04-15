class EdgarError(Exception):
    """Base exception for SEC EDGAR retrieval failures."""


class ConfigurationError(EdgarError):
    """Raised when required application configuration is missing."""


class TickerNotFoundError(EdgarError):
    """Raised when a ticker is not present in the SEC ticker lookup file."""


class FilingNotFoundError(EdgarError):
    """Raised when the target filing cannot be found for a company."""


class EdgarRateLimitError(EdgarError):
    """Raised when SEC fair access throttling blocks a request."""


class EdgarResponseError(EdgarError):
    """Raised when SEC returns an unexpected or unusable response."""


class FinancialExtractionError(Exception):
    """Base exception for financial extraction failures."""


class OpenAIConfigurationError(ConfigurationError):
    """Raised when OpenAI client configuration is missing or invalid."""


class OpenAIExtractionError(FinancialExtractionError):
    """Raised when the OpenAI extraction request fails."""


class ExtractionParseError(FinancialExtractionError):
    """Raised when the model response cannot be parsed into the expected schema."""


class CreditMemoError(Exception):
    """Base exception for credit memo generation failures."""


class MemoGenerationError(CreditMemoError):
    """Raised when the OpenAI memo generation request fails."""


class MemoParseError(CreditMemoError):
    """Raised when the memo model response cannot be parsed into the expected schema."""


class PDFGenerationError(CreditMemoError):
    """Raised when PDF export generation fails."""
