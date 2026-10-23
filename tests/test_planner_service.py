"""Unit tests for ResearchPlannerService."""

from unittest.mock import MagicMock, patch
import pytest
from google.genai.errors import ClientError
from src.services.planner_service import ResearchPlannerService


@patch("src.services.planner_service.genai.Client")
def test_plan_sub_queries_clean_json(mock_client_cls):
    """Verify that valid JSON sub-queries are parsed correctly."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_resp = MagicMock()
    mock_resp.text = '["solid state battery electrolyte mechanisms", "solid state EV commercialization timeline"]'
    mock_client.models.generate_content.return_value = mock_resp

    service = ResearchPlannerService(api_key="gemini-test-key")
    sub_queries = service.plan_sub_queries("solid-state batteries", max_sub_queries=2)

    assert len(sub_queries) == 2
    assert "solid state battery electrolyte mechanisms" in sub_queries
    assert "solid state EV commercialization timeline" in sub_queries


@patch("src.services.planner_service.genai.Client")
def test_plan_sub_queries_markdown_fences(mock_client_cls):
    """Verify that JSON wrapped in markdown code blocks is cleanly handled."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_resp = MagicMock()
    mock_resp.text = '```json\n["quantum computing error correction benchmarks", "post-quantum cryptography migration"]\n```'
    mock_client.models.generate_content.return_value = mock_resp

    service = ResearchPlannerService(api_key="gemini-test-key")
    sub_queries = service.plan_sub_queries("quantum computing milestones", max_sub_queries=2)

    assert len(sub_queries) == 2
    assert sub_queries[0] == "quantum computing error correction benchmarks"


@patch("src.services.planner_service.genai.Client")
def test_plan_sub_queries_fallback_line_format(mock_client_cls):
    """Verify that non-JSON numbered list output is recovered through fallback line parser."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_resp = MagicMock()
    mock_resp.text = "1. AI agent evaluation benchmarks\n2. Tool use architectures for LLMs"
    mock_client.models.generate_content.return_value = mock_resp

    service = ResearchPlannerService(api_key="gemini-test-key")
    sub_queries = service.plan_sub_queries("AI agents", max_sub_queries=2)

    assert len(sub_queries) == 2
    assert "AI agent evaluation benchmarks" in sub_queries[0]
    assert "Tool use architectures for LLMs" in sub_queries[1]


@patch("src.services.planner_service.genai.Client")
def test_plan_sub_queries_filters_duplicates(mock_client_cls):
    """Verify that sub-queries matching the root query or each other are filtered."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_resp = MagicMock()
    mock_resp.text = '["nuclear fusion", "nuclear fusion", "stellarator vs tokamak efficiency"]'
    mock_client.models.generate_content.return_value = mock_resp

    service = ResearchPlannerService(api_key="gemini-test-key")
    sub_queries = service.plan_sub_queries("nuclear fusion", max_sub_queries=2)

    # "nuclear fusion" is filtered out because it matches original query
    assert len(sub_queries) == 1
    assert sub_queries[0] == "stellarator vs tokamak efficiency"


@patch("src.services.planner_service.genai.Client")
def test_plan_sub_queries_graceful_error_fallback(mock_client_cls):
    """Verify that if Gemini planning fails, it returns an empty list rather than crashing."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.models.generate_content.side_effect = ClientError(500, {"error": "Server error"})

    service = ResearchPlannerService(api_key="gemini-test-key")
    sub_queries = service.plan_sub_queries("deep research", max_sub_queries=2)

    assert sub_queries == []


def test_plan_sub_queries_empty_input():
    """Verify that empty inputs immediately return empty lists."""
    service = ResearchPlannerService(api_key="test-key")
    assert service.plan_sub_queries("   ") == []
