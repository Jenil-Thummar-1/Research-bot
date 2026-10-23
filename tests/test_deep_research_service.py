"""Unit tests for DeepResearchService: canonicalization, domain diversity, and fuller extraction."""

from unittest.mock import MagicMock, patch
import pytest
from src.services.search_service import SearchResult, MissingAPIKeyError
from src.services.deep_research_service import (
    DeepResearchService,
    EnrichedSource,
    canonicalize_url,
    apply_domain_diversity,
    extract_domain,
)


def test_canonicalize_url_tracking_removal():
    """Verify that tracking parameters, fragments, and casing are normalized."""
    dirty_url = "HTTPS://WWW.Example.com:443/research/?utm_source=twitter&topic=ai&utm_campaign=spring#overview"
    clean = canonicalize_url(dirty_url)
    assert clean == "https://www.example.com/research?topic=ai"


def test_canonicalize_url_trailing_slash():
    """Verify trailing slashes on sub-paths are normalized."""
    url1 = "https://example.com/article/"
    url2 = "https://example.com/article"
    assert canonicalize_url(url1) == canonicalize_url(url2)
    # Root slash is preserved
    assert canonicalize_url("https://example.com/") == "https://example.com/"


def test_extract_domain():
    """Verify clean domain extraction."""
    assert extract_domain("https://www.nature.com/articles/s41586-024") == "nature.com"
    assert extract_domain("http://sub.domain.org:8080/path") == "sub.domain.org"


def test_apply_domain_diversity():
    """Verify that results from the same domain are capped to max_per_domain."""
    sources = [
        EnrichedSource(title="E1", url="https://site-a.com/1", canonical_url="https://site-a.com/1", domain="site-a.com", content="C1", score=0.95),
        EnrichedSource(title="E2", url="https://site-a.com/2", canonical_url="https://site-a.com/2", domain="site-a.com", content="C2", score=0.90),
        EnrichedSource(title="E3", url="https://site-a.com/3", canonical_url="https://site-a.com/3", domain="site-a.com", content="C3", score=0.85),
        EnrichedSource(title="E4", url="https://site-b.com/1", canonical_url="https://site-b.com/1", domain="site-b.com", content="C4", score=0.80),
        EnrichedSource(title="E5", url="https://site-c.com/1", canonical_url="https://site-c.com/1", domain="site-c.com", content="C5", score=0.75),
    ]

    selected = apply_domain_diversity(sources, max_total=3, max_per_domain=2)
    assert len(selected) == 3
    domains = [s.domain for s in selected]
    # site-a should be capped at 2, and third slot filled by site-b
    assert domains == ["site-a.com", "site-a.com", "site-b.com"]


@patch("src.services.deep_research_service.TavilyClient")
def test_full_content_extraction_and_fallback(mock_tavily_cls):
    """Verify that full content is captured when available and fallback to snippet on paywall/error."""
    mock_client = MagicMock()
    mock_tavily_cls.return_value = mock_client
    mock_client.extract.return_value = {
        "results": [
            {
                "url": "https://example.com/open",
                "raw_content": "This is a comprehensive full text research paper discussing technical details." * 20,
            }
        ],
        "failed_results": [
            {"url": "https://example.com/paywall", "error": "403 Forbidden"}
        ],
    }

    service = DeepResearchService(tavily_key="tvly-mock-key")
    sources = [
        EnrichedSource(
            title="Open Paper",
            url="https://example.com/open",
            canonical_url="https://example.com/open",
            domain="example.com",
            content="Short snippet open.",
        ),
        EnrichedSource(
            title="Paywalled Paper",
            url="https://example.com/paywall",
            canonical_url="https://example.com/paywall",
            domain="example.com",
            content="Short snippet paywalled.",
        ),
    ]

    enriched, full_count, fallback_count = service._extract_full_content(sources, max_char_limit=500)

    assert len(enriched) == 2
    assert full_count == 1
    assert fallback_count == 1

    # First source received full text
    assert enriched[0].is_full_content is True
    assert "comprehensive full text" in enriched[0].content
    assert enriched[0].extraction_status == "Full article extracted"

    # Second source kept search snippet with honest disclosure
    assert enriched[1].is_full_content is False
    assert enriched[1].content == "Short snippet paywalled."
    assert "paywall" in enriched[1].extraction_status.lower() or "inaccessible" in enriched[1].extraction_status.lower()


@patch("src.services.deep_research_service.ResearchPlannerService.plan_sub_queries")
@patch("src.services.deep_research_service.TavilySearchService.search")
@patch.object(DeepResearchService, "_extract_full_content")
def test_run_deep_research_deduplication_and_bounds(mock_extract, mock_search, mock_plan):
    """Verify that deep research runs bounded searches and deduplicates by canonical URL."""
    mock_plan.return_value = ["sub query 1", "sub query 2"]

    # Mock search calls returning overlapping URLs across queries
    mock_search.side_effect = [
        # Call 1: Original query
        [
            SearchResult(title="Result 1", url="https://example.com/page1?utm_source=google", content="C1", score=0.9),
            SearchResult(title="Result 2", url="https://example.com/page2", content="C2", score=0.8),
        ],
        # Call 2: Sub-query 1 (returns duplicate page 1 with different UTM tag)
        [
            SearchResult(title="Result 1 Duplicate", url="https://example.com/page1?utm_source=twitter", content="C1", score=0.95),
            SearchResult(title="Result 3", url="https://example.com/page3", content="C3", score=0.7),
        ],
        # Call 3: Sub-query 2
        [
            SearchResult(title="Result 4", url="https://example.com/page4", content="C4", score=0.6),
        ],
    ]

    mock_extract.side_effect = lambda sources, max_char_limit: (sources, 0, len(sources))

    service = DeepResearchService(tavily_key="tvly-mock", gemini_key="gemini-mock")
    result = service.run_deep_research(
        query="broad topic",
        max_sub_queries=2,
        max_sources=4,
        extract_full_content=True,
    )

    # 1 root + 2 sub-queries = 3 searches run
    assert result.total_searches_run == 3
    assert len(result.sub_queries) == 2

    # Verify deduplication: page1 appeared twice with different UTM tags, only 1 kept
    canonical_urls = [s.canonical_url for s in result.sources]
    assert len(canonical_urls) == len(set(canonical_urls))
    assert len(result.sources) <= 4


def test_missing_tavily_key_raises_error():
    """Verify that MissingAPIKeyError is raised if Tavily key is absent."""
    service = DeepResearchService(tavily_key="")
    with pytest.raises(MissingAPIKeyError):
        service.run_deep_research("test query")
