"""Tavily search service integration for web evidence retrieval."""

from dataclasses import dataclass
import logging
from typing import Literal
from tavily import TavilyClient
from tavily.errors import (
    MissingAPIKeyError as TavilyMissingKeyError,
    InvalidAPIKeyError as TavilyInvalidKeyError,
    UsageLimitExceededError as TavilyLimitError,
    BadRequestError as TavilyBadRequestError,
)
from src.config import get_tavily_api_key

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchResult:
    """Represents a single verified web search result."""

    title: str
    url: str
    content: str
    score: float | None = None


class SearchServiceError(Exception):
    """Base exception for search service errors."""

    pass


class MissingAPIKeyError(SearchServiceError):
    """Raised when Tavily API key is not configured."""

    pass


class InvalidAPIKeyError(SearchServiceError):
    """Raised when Tavily API key is rejected as invalid or unauthorized."""

    pass


class SearchRateLimitError(SearchServiceError):
    """Raised when Tavily search quota or rate limits are exceeded."""

    pass


class SearchQueryError(SearchServiceError):
    """Raised when search query is malformed or invalid."""

    pass


class SearchNetworkError(SearchServiceError):
    """Raised when network connectivity fails during search."""

    pass


def duckduckgo_search(
    query: str,
    max_results: int = 5,
    timeout: float | None = None,
) -> list[SearchResult]:
    """Execute a free web search via DuckDuckGo (requires zero API keys).

    Args:
        query: The search query string.
        max_results: Maximum number of search results to return.
        timeout: Optional network request timeout in seconds.

    Returns:
        A list of SearchResult objects.

    Raises:
        SearchQueryError: If the query is blank.
        SearchServiceError: If the search request fails.
    """
    clean_query = (query or "").strip()
    if not clean_query:
        raise SearchQueryError("Search query cannot be empty.")

    clamped_max_results = max(1, min(max_results, 20))

    try:
        from ddgs import DDGS

        ddgs = DDGS(timeout=timeout or 10.0)
        raw_results = list(ddgs.text(clean_query, max_results=clamped_max_results))
        results: list[SearchResult] = []

        for item in raw_results:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip() or "Untitled Result"
            url = str(item.get("href", "")).strip()
            content = str(item.get("body", "")).strip() or "No snippet available."
            results.append(
                SearchResult(
                    title=title,
                    url=url,
                    content=content,
                    score=1.0,
                )
            )
        return results[:clamped_max_results]
    except Exception as err:
        logger.warning("DuckDuckGo search failed: %s", err)
        raise SearchServiceError(f"DuckDuckGo search failed: {err}") from err


class TavilySearchService:
    """Service wrapper for executing queries against Tavily Search API with optional DuckDuckGo fallback."""

    def __init__(self, api_key: str | None = None):
        """Initialize the search service with an optional explicit API key."""
        if api_key is not None:
            self._api_key = api_key.strip()
        else:
            self._api_key = (get_tavily_api_key() or "").strip()

    @property
    def is_configured(self) -> bool:
        """Check whether an API key is available."""
        return bool(self._api_key)

    def search(
        self,
        query: str,
        max_results: int = 5,
        search_depth: Literal["basic", "advanced"] = "basic",
        timeout: float | None = None,
        provider: Literal["auto", "tavily", "duckduckgo"] = "tavily",
        fallback_to_ddg: bool = False,
    ) -> list[SearchResult]:
        """Execute a web search and return structured search results.

        Args:
            query: The research query string.
            max_results: Maximum number of search results to return (default: 5).
            search_depth: "basic" for quick lookups or "advanced" for deeper research.
            timeout: Optional network request timeout in seconds.
            provider: Search provider to use ("tavily", "duckduckgo", or "auto").
            fallback_to_ddg: If True, automatically falls back to DuckDuckGo on Tavily rate limit or missing key.

        Returns:
            A list of SearchResult objects.

        Raises:
            MissingAPIKeyError: If no API key is configured (when fallback is disabled).
            SearchQueryError: If the search query is empty.
            InvalidAPIKeyError: If the API key is rejected.
            SearchRateLimitError: If API quotas are exceeded.
            SearchNetworkError: If a connection error occurs.
            SearchServiceError: For unexpected search failures.
        """
        clean_query = (query or "").strip()
        if not clean_query:
            raise SearchQueryError("Search query cannot be empty.")

        # If user explicitly requested DuckDuckGo, bypass Tavily completely (zero keys required)
        if provider == "duckduckgo":
            return duckduckgo_search(clean_query, max_results=max_results, timeout=timeout)

        if not self._api_key:
            if fallback_to_ddg or provider == "auto":
                logger.info("Tavily API key not configured; falling back to DuckDuckGo keyless search.")
                return duckduckgo_search(clean_query, max_results=max_results, timeout=timeout)
            raise MissingAPIKeyError(
                "Tavily API key is missing. Add TAVILY_API_KEY to your .env file."
            )

        clamped_max_results = max(1, min(max_results, 20))

        search_args = {
            "query": clean_query,
            "search_depth": search_depth,
            "max_results": clamped_max_results,
        }
        if timeout is not None:
            search_args["timeout"] = timeout

        try:
            client = TavilyClient(api_key=self._api_key)
            response = client.search(**search_args)
        except (TavilyMissingKeyError, ValueError) as err:
            logger.error("Missing Tavily API key: %s", err)
            raise MissingAPIKeyError("Tavily API key is missing or blank.") from err
        except TavilyInvalidKeyError as err:
            logger.error("Invalid Tavily API key: %s", err)
            raise InvalidAPIKeyError(
                "Tavily API key is invalid or unauthorized. Please verify your key at https://tavily.com."
            ) from err
        except TavilyLimitError as err:
            logger.error("Tavily rate/usage limit exceeded: %s", err)
            if fallback_to_ddg or provider == "auto":
                logger.info("Tavily usage limit hit; falling back to DuckDuckGo keyless search.")
                return duckduckgo_search(clean_query, max_results=clamped_max_results, timeout=timeout)
            raise SearchRateLimitError(
                "Tavily API usage limit exceeded. Check your plan quota at https://tavily.com."
            ) from err
        except TavilyBadRequestError as err:
            logger.error("Bad request to Tavily API: %s", err)
            raise SearchQueryError(f"Tavily API rejected request: {err}") from err
        except Exception as err:
            err_msg = str(err).lower()
            if "unauthorized" in err_msg or "401" in err_msg or "forbidden" in err_msg or "403" in err_msg:
                raise InvalidAPIKeyError(
                    "Tavily API authorization failed. Check your API key."
                ) from err
            if "timeout" in err_msg or "connection" in err_msg or "network" in err_msg:
                raise SearchNetworkError(
                    f"Network error while connecting to Tavily: {err}"
                ) from err
            logger.exception("Unexpected error during Tavily search: %s", err)
            raise SearchServiceError(f"Tavily search failed: {err}") from err

        raw_results = response.get("results", []) if isinstance(response, dict) else []
        results: list[SearchResult] = []

        for item in raw_results:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip() or "Untitled Result"
            url = str(item.get("url", "")).strip()
            content = str(item.get("content", "")).strip() or "No snippet available."
            score = item.get("score")
            score_val = float(score) if isinstance(score, (int, float)) else None

            results.append(
                SearchResult(
                    title=title,
                    url=url,
                    content=content,
                    score=score_val,
                )
            )

        return results[:clamped_max_results]
