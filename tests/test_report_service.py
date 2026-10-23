"""Unit tests for GeminiReportService and citation binding."""

from unittest.mock import MagicMock, patch
import pytest
from google.genai.errors import ClientError
from src.services.search_service import SearchResult
from src.services.report_service import (
    GeminiReportService,
    GroundedSource,
    ResearchReport,
    MissingGeminiKeyError,
    GeminiAuthError,
    GeminiQuotaError,
    GeminiGenerationError,
    bind_citations_to_urls,
)


def test_missing_gemini_key_raises_error():
    """Verify that MissingGeminiKeyError is raised if no key is configured."""
    service = GeminiReportService(api_key="")
    sample_results = [
        SearchResult(title="T1", url="https://example.com/1", content="Snippet 1")
    ]
    with pytest.raises(MissingGeminiKeyError) as exc_info:
        service.generate_report("test query", sample_results)
    assert "Gemini API key is missing" in str(exc_info.value)


def test_empty_results_raises_error():
    """Verify that GeminiGenerationError is raised when no evidence is supplied."""
    service = GeminiReportService(api_key="gemini-mock-key")
    with pytest.raises(GeminiGenerationError) as exc_info:
        service.generate_report("test query", [])
    assert "Cannot generate report without evidence" in str(exc_info.value)


def test_bind_citations_to_urls():
    """Verify citation tags [S1], [S2] are converted to clickable markdown links."""
    sources = {
        "S1": GroundedSource("S1", "Title One", "https://example.com/alpha", "Snippet 1", "example.com"),
        "S2": GroundedSource("S2", "Title Two", "https://example.com/beta", "Snippet 2", "example.com"),
    }

    input_text = "Solid state batteries offer high density [S1]. However, cooling is vital [s2] and [S9] is unknown."
    output_text = bind_citations_to_urls(input_text, sources)

    assert "[[S1]](https://example.com/alpha \"Title One\")" in output_text
    assert "[[S2]](https://example.com/beta \"Title Two\")" in output_text
    assert "[S9]" in output_text  # Unknown source preserved without hallucinated link


@patch("src.services.report_service.genai.Client")
def test_successful_report_generation(mock_client_cls):
    """Verify end-to-end report synthesis with mocked Gemini client."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    mock_response = MagicMock()
    mock_response.text = (
        "## Executive Summary\n"
        "Breakthroughs have occurred [S1].\n\n"
        "## Key Findings\n"
        "- High density achieved [S1][S2]."
    )
    mock_client.models.generate_content.return_value = mock_response

    service = GeminiReportService(api_key="gemini-test-key", model_name="gemini-3.8-flash")
    results = [
        SearchResult(title="Study 1", url="https://example.com/s1", content="New cathode design"),
        SearchResult(title="Study 2", url="https://example.com/s2", content="Electrolyte advances"),
    ]

    report = service.generate_report("Solid state batteries", results)

    assert isinstance(report, ResearchReport)
    assert report.model_used == "gemini-3.8-flash"
    assert len(report.sources) == 2
    assert report.sources[0].source_id == "S1"
    assert report.sources[1].source_id == "S2"

    # Verify citation link binding
    assert "[[S1]](https://example.com/s1 \"Study 1\")" in report.content
    assert "[[S2]](https://example.com/s2 \"Study 2\")" in report.content

    # Verify call arguments
    mock_client.models.generate_content.assert_called_once()
    call_kwargs = mock_client.models.generate_content.call_args.kwargs
    assert call_kwargs["model"] == "gemini-3.8-flash"
    assert "Solid state batteries" in call_kwargs["contents"]


@patch("src.services.report_service.genai.Client")
def test_gemini_auth_error_handling(mock_client_cls):
    """Verify that 401/403 ClientErrors are mapped to GeminiAuthError."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    mock_client.models.generate_content.side_effect = ClientError(401, {"error": "API_KEY_INVALID"})

    service = GeminiReportService(api_key="gemini-bad-key")
    results = [SearchResult(title="T", url="https://example.com", content="C")]

    with pytest.raises(GeminiAuthError) as exc_info:
        service.generate_report("test query", results)
    assert "invalid or unauthorized" in str(exc_info.value)


@patch("src.services.report_service.genai.Client")
def test_gemini_quota_error_handling(mock_client_cls):
    """Verify that 429 ClientErrors are mapped to GeminiQuotaError."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    mock_client.models.generate_content.side_effect = ClientError(429, {"error": "RESOURCE_EXHAUSTED"})

    service = GeminiReportService(api_key="gemini-test-key")
    results = [SearchResult(title="T", url="https://example.com", content="C")]

    with pytest.raises(GeminiQuotaError) as exc_info:
        service.generate_report("test query", results)
    assert "rate limit exceeded" in str(exc_info.value)


@patch("src.services.report_service.genai.Client")
def test_gemini_timeout_error_handling(mock_client_cls):
    """Verify that 504 Deadline Exceeded raises GeminiTimeoutError."""
    from google.genai.errors import ServerError
    from src.services.report_service import GeminiTimeoutError

    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.models.generate_content.side_effect = ServerError(504, {"error": "DEADLINE_EXCEEDED: timed out"})

    service = GeminiReportService(api_key="gemini-test-key")
    results = [SearchResult(title="T", url="https://example.com", content="C")]

    with pytest.raises(GeminiTimeoutError) as exc_info:
        service.generate_report("test query", results, timeout_seconds=14.0)
    assert "timed out" in str(exc_info.value)


@patch("src.services.report_service.genai.Client")
def test_concise_mode_generation(mock_client_cls):
    """Verify that concise_mode=True produces a concise answer prompt."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_resp = MagicMock()
    mock_resp.text = "## Summary\nConcise answer text [S1]."
    mock_client.models.generate_content.return_value = mock_resp

    service = GeminiReportService(api_key="gemini-test-key")
    results = [SearchResult(title="T", url="https://example.com", content="C")]

    report = service.generate_report("What is X?", results, concise_mode=True)
    assert isinstance(report, ResearchReport)
    assert "## Summary" in report.content

    call_kwargs = mock_client.models.generate_content.call_args.kwargs
    assert "under 350 words" in call_kwargs["contents"]


