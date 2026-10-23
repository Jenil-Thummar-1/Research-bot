"""Unit tests for TavilySearchService."""

from unittest.mock import MagicMock, patch
import pytest
from tavily.errors import (
    InvalidAPIKeyError as TavilyInvalidKeyError,
    UsageLimitExceededError as TavilyLimitError,
    BadRequestError as TavilyBadRequestError,
)
from src.services.search_service import (
    TavilySearchService,
    SearchResult,
    MissingAPIKeyError,
    InvalidAPIKeyError,
    SearchQueryError,
    SearchRateLimitError,
    SearchNetworkError,
)


def test_missing_api_key_raises_error():
    """Verify that MissingAPIKeyError is raised if no API key is provided or found."""
    service = TavilySearchService(api_key="")
    with pytest.raises(MissingAPIKeyError) as exc_info:
        service.search("test query")
    assert "Tavily API key is missing" in str(exc_info.value)


def test_empty_query_raises_error():
    """Verify that SearchQueryError is raised for blank or whitespace-only queries."""
    service = TavilySearchService(api_key="tvly-mock-key")
    with pytest.raises(SearchQueryError) as exc_info:
        service.search("   ")
    assert "cannot be empty" in str(exc_info.value)


@patch("src.services.search_service.TavilyClient")
def test_successful_search_parsing(mock_client_cls):
    """Verify that search results are correctly parsed into SearchResult objects."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.search.return_value = {
        "results": [
            {
                "title": "Quantum Computing 101",
                "url": "https://example.com/quantum",
                "content": "An introductory guide to qubits and superposition.",
                "score": 0.95,
            },
            {
                "title": "Qubit Hardware Overview",
                "url": "https://example.com/qubit-hardware",
                "content": "Superconducting and trapped-ion qubits.",
                "score": 0.88,
            },
        ]
    }

    service = TavilySearchService(api_key="tvly-test-key")
    results = service.search("quantum computing", max_results=5)

    assert len(results) == 2
    assert isinstance(results[0], SearchResult)
    assert results[0].title == "Quantum Computing 101"
    assert results[0].url == "https://example.com/quantum"
    assert results[0].content == "An introductory guide to qubits and superposition."
    assert results[0].score == 0.95
    mock_client.search.assert_called_once_with(
        query="quantum computing",
        search_depth="basic",
        max_results=5,
    )


@patch("src.services.search_service.TavilyClient")
def test_clamps_results_to_requested_limit(mock_client_cls):
    """Verify that search returns no more than max_results."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.search.return_value = {
        "results": [
            {"title": f"Result {i}", "url": f"https://example.com/{i}", "content": f"Snippet {i}"}
            for i in range(10)
        ]
    }

    service = TavilySearchService(api_key="tvly-test-key")
    results = service.search("deep learning", max_results=5)

    assert len(results) == 5
    assert results[0].title == "Result 0"
    assert results[4].title == "Result 4"


@patch("src.services.search_service.TavilyClient")
def test_invalid_api_key_handling(mock_client_cls):
    """Verify that Tavily's InvalidAPIKeyError is converted to our InvalidAPIKeyError."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.search.side_effect = TavilyInvalidKeyError("Unauthorized key")

    service = TavilySearchService(api_key="tvly-bad-key")
    with pytest.raises(InvalidAPIKeyError) as exc_info:
        service.search("ai agents")
    assert "invalid or unauthorized" in str(exc_info.value)


@patch("src.services.search_service.TavilyClient")
def test_rate_limit_exceeded_handling(mock_client_cls):
    """Verify that rate/usage limits are cleanly mapped to SearchRateLimitError."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.search.side_effect = TavilyLimitError("Limit exceeded")

    service = TavilySearchService(api_key="tvly-test-key")
    with pytest.raises(SearchRateLimitError) as exc_info:
        service.search("quantum teleportation")
    assert "usage limit exceeded" in str(exc_info.value)


@patch("src.services.search_service.TavilyClient")
def test_network_timeout_handling(mock_client_cls):
    """Verify that network/connection timeouts are cleanly mapped to SearchNetworkError."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.search.side_effect = ConnectionError("Connection timed out")

    service = TavilySearchService(api_key="tvly-test-key")
    with pytest.raises(SearchNetworkError) as exc_info:
        service.search("fusion energy")
    assert "Network error" in str(exc_info.value)
