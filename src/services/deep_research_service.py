"""Deep research orchestration: multi-query planning, URL canonicalization, domain diversity, and fuller content extraction."""

from dataclasses import dataclass
import logging
import time
import urllib.parse
from tavily import TavilyClient
from src.config import get_tavily_api_key, get_gemini_api_key
from src.services.search_service import TavilySearchService, SearchResult, MissingAPIKeyError
from src.services.planner_service import ResearchPlannerService

logger = logging.getLogger(__name__)

# Tracking parameters to strip during URL canonicalization
TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid", "gclsrc", "msclkid", "mc_eid", "ref", "referrer", "source"
}


@dataclass(frozen=True)
class EnrichedSource:
    """Represents a thoroughly collected evidence source with complete provenance."""

    title: str
    url: str
    canonical_url: str
    domain: str
    content: str
    score: float | None = None
    origin_query: str = ""
    is_full_content: bool = False
    extraction_status: str = "Search snippet"
    published_date: str | None = None


@dataclass(frozen=True)
class DeepResearchResult:
    """Result of a deep multi-query research workflow."""

    query: str
    sub_queries: list[str]
    sources: list[EnrichedSource]
    total_searches_run: int
    full_content_count: int
    snippet_fallback_count: int


def canonicalize_url(url: str) -> str:
    """Normalize a URL to its canonical form for reliable deduplication.

    - Strips fragment identifiers (#...)
    - Lowercases scheme and hostname
    - Strips default ports (:80, :443)
    - Strips common marketing/social tracking parameters
    - Normalizes trailing slashes on paths
    """
    clean_url = (url or "").strip()
    if not clean_url:
        return ""

    try:
        parts = urllib.parse.urlsplit(clean_url)
        scheme = parts.scheme.lower()
        netloc = parts.netloc.lower()

        # Remove default ports
        if netloc.endswith(":80") and scheme == "http":
            netloc = netloc[:-3]
        elif netloc.endswith(":443") and scheme == "https":
            netloc = netloc[:-4]

        # Filter query params
        query_dict = urllib.parse.parse_qsl(parts.query, keep_blank_values=False)
        filtered_params = [
            (k, v) for k, v in query_dict if k.lower() not in TRACKING_PARAMS
        ]
        # Sort parameters for deterministic equivalence
        filtered_params.sort(key=lambda x: x[0])
        clean_query = urllib.parse.urlencode(filtered_params)

        # Normalize path
        path = parts.path
        if len(path) > 1 and path.endswith("/"):
            path = path[:-1]
        elif not path:
            path = "/"

        return urllib.parse.urlunsplit((scheme, netloc, path, clean_query, ""))
    except Exception:
        return clean_url


def extract_domain(url: str) -> str:
    """Extract clean domain name without www or port."""
    try:
        parsed = urllib.parse.urlparse(url)
        netloc = parsed.netloc or parsed.path
        return netloc.replace("www.", "").split(":")[0].lower()
    except Exception:
        return "web source"


def apply_domain_diversity(
    sources: list[EnrichedSource],
    max_total: int = 6,
    max_per_domain: int = 2,
) -> list[EnrichedSource]:
    """Select high-quality sources enforcing domain diversity to avoid single-site dominance.

    Sources are assumed to be pre-sorted or ranked by relevance.
    """
    selected: list[EnrichedSource] = []
    domain_counts: dict[str, int] = {}
    deferred: list[EnrichedSource] = []

    for src in sources:
        dom = src.domain
        count = domain_counts.get(dom, 0)
        if count < max_per_domain:
            selected.append(src)
            domain_counts[dom] = count + 1
            if len(selected) >= max_total:
                return selected
        else:
            deferred.append(src)

    # If domain cap leaves us under max_total, fill remaining slots from deferred
    for src in deferred:
        if len(selected) >= max_total:
            break
        selected.append(src)

    return selected


class DeepResearchService:
    """Orchestrates multi-query search, deduplication, domain diversity, and fuller page extraction."""

    def __init__(
        self,
        tavily_key: str | None = None,
        gemini_key: str | None = None,
    ):
        """Initialize the deep research service."""
        if tavily_key is not None:
            self._tavily_key = tavily_key.strip()
        else:
            self._tavily_key = (get_tavily_api_key() or "").strip()

        if gemini_key is not None:
            self._gemini_key = gemini_key.strip()
        else:
            self._gemini_key = (get_gemini_api_key() or "").strip()

    def run_deep_research(
        self,
        query: str,
        max_sub_queries: int = 1,
        max_sources: int = 5,
        search_depth: str = "basic",
        extract_full_content: bool = True,
        max_char_per_source: int = 3500,
        time_budget: float = 25.0,
    ) -> DeepResearchResult:
        """Execute a complete, bounded deep research pipeline.

        Args:
            query: The root research question.
            max_sub_queries: Bounded number of sub-queries (1 to 2).
            max_sources: Maximum deduplicated evidence sources to retain.
            search_depth: "basic" or "advanced" for Tavily search.
            extract_full_content: Whether to fetch fuller article markdown via Tavily extract.
            max_char_per_source: Character limit per extracted page to prevent context overflow.
            time_budget: Maximum allowed seconds for the retrieval and extraction phases.

        Returns:
            A DeepResearchResult object containing provenance-backed sources and sub-queries.
        """
        start_time = time.time()
        clean_query = (query or "").strip()
        if not clean_query:
            raise ValueError("Research query cannot be empty.")

        if not self._tavily_key:
            raise MissingAPIKeyError("Tavily API key is missing. Add TAVILY_API_KEY to your .env file.")

        # 1. Query Planning with Gemini (skip if time budget is tight)
        sub_queries: list[str] = []
        elapsed = time.time() - start_time
        remaining_for_gather = max(5.0, time_budget - elapsed)

        if remaining_for_gather > 12.0 and max_sub_queries > 0:
            planner = ResearchPlannerService(api_key=self._gemini_key)
            sub_queries = planner.plan_sub_queries(
                clean_query,
                max_sub_queries=max_sub_queries,
                timeout_seconds=min(10.0, remaining_for_gather - 5.0),
            )

        # 2. Multi-Query Search with Tavily (Strictly bounded)
        all_queries = [clean_query] + sub_queries
        searches_run = 0
        raw_collected: list[EnrichedSource] = []
        seen_canonicals: set[str] = set()

        search_service = TavilySearchService(api_key=self._tavily_key)

        for q in all_queries:
            # Check elapsed time: if more than 50% of budget spent, stop adding searches
            if time.time() - start_time > (time_budget * 0.5) and searches_run > 0:
                logger.info("Search phase cutoff reached (%ds); proceeding with collected results.", int(time.time() - start_time))
                break

            try:
                # Clamp results per query to stay within rate/quota bounds
                results: list[SearchResult] = search_service.search(
                    query=q,
                    max_results=4,
                    search_depth=search_depth,
                    timeout=5.0,
                )
                searches_run += 1

                for item in results:
                    canonical = canonicalize_url(item.url)
                    if not canonical or canonical in seen_canonicals:
                        continue
                    seen_canonicals.add(canonical)

                    raw_collected.append(
                        EnrichedSource(
                            title=item.title,
                            url=item.url,
                            canonical_url=canonical,
                            domain=extract_domain(item.url),
                            content=item.content,
                            score=item.score,
                            origin_query=q,
                            is_full_content=False,
                            extraction_status="Search snippet",
                        )
                    )
            except Exception as err:
                logger.warning("Search failed for query '%s': %s", q, err)
                # Continue gracefully with remaining queries rather than aborting

        # 3. Sort by relevance score (descending) and apply domain diversity filter
        raw_collected.sort(key=lambda s: s.score if s.score is not None else 0.0, reverse=True)
        diverse_sources = apply_domain_diversity(
            raw_collected,
            max_total=max_sources,
            max_per_domain=2,
        )

        # 4. Fuller Content Extraction via Tavily's extract API (when enabled and time permits)
        final_sources: list[EnrichedSource] = []
        full_content_count = 0
        snippet_fallback_count = 0

        remaining_time = time_budget - (time.time() - start_time)
        if extract_full_content and diverse_sources and remaining_time > 8.0:
            final_sources, full_content_count, snippet_fallback_count = self._extract_full_content(
                diverse_sources,
                max_char_per_source,
            )
        else:
            final_sources = diverse_sources
            snippet_fallback_count = len(diverse_sources)

        return DeepResearchResult(
            query=clean_query,
            sub_queries=sub_queries,
            sources=final_sources,
            total_searches_run=searches_run,
            full_content_count=full_content_count,
            snippet_fallback_count=snippet_fallback_count,
        )

    def _extract_full_content(
        self,
        sources: list[EnrichedSource],
        max_char_limit: int,
        timeout: float = 7.0,
    ) -> tuple[list[EnrichedSource], int, int]:
        """Perform batch page extraction using TavilyClient.extract with graceful fallbacks."""
        urls_to_extract = [s.url for s in sources]
        extracted_map: dict[str, str] = {}
        failed_urls: set[str] = set()

        try:
            client = TavilyClient(api_key=self._tavily_key)
            response = client.extract(
                urls=urls_to_extract,
                extract_depth="basic",
                format="markdown",
                timeout=int(timeout),
            )
            # Parse successful extractions
            for item in response.get("results", []):
                u = item.get("url")
                raw = (item.get("raw_content") or "").strip()
                if u and raw:
                    extracted_map[u] = raw

            # Parse failed extractions (paywalls, 403s, timeouts)
            for item in response.get("failed_results", []):
                u = item.get("url")
                if u:
                    failed_urls.add(u)
        except Exception as err:
            logger.warning("Batch content extraction failed: %s; falling back to snippets.", err)

        enriched: list[EnrichedSource] = []
        full_count = 0
        fallback_count = 0

        for src in sources:
            full_text = extracted_map.get(src.url)
            if full_text and len(full_text) > 200:
                # Truncate to max_char_limit to protect synthesis context window
                truncated = full_text[:max_char_limit].strip()
                enriched.append(
                    EnrichedSource(
                        title=src.title,
                        url=src.url,
                        canonical_url=src.canonical_url,
                        domain=src.domain,
                        content=truncated,
                        score=src.score,
                        origin_query=src.origin_query,
                        is_full_content=True,
                        extraction_status="Full article extracted",
                        published_date=src.published_date,
                    )
                )
                full_count += 1
            else:
                status_reason = (
                    "Search snippet (inaccessible/paywall)"
                    if src.url in failed_urls
                    else "Search snippet"
                )
                enriched.append(
                    EnrichedSource(
                        title=src.title,
                        url=src.url,
                        canonical_url=src.canonical_url,
                        domain=src.domain,
                        content=src.content,
                        score=src.score,
                        origin_query=src.origin_query,
                        is_full_content=False,
                        extraction_status=status_reason,
                        published_date=src.published_date,
                    )
                )
                fallback_count += 1

        return enriched, full_count, fallback_count
