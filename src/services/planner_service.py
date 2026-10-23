"""Research query planner service using Gemini to decompose broad questions into targeted sub-queries."""

from dataclasses import dataclass
import json
import logging
import re
from google import genai
from google.genai import types
from google.genai.errors import APIError, ClientError, ServerError
from src.config import get_gemini_api_key

logger = logging.getLogger(__name__)

DEFAULT_PLANNER_MODEL = "gemini-3.8-flash"


@dataclass(frozen=True)
class QueryPlan:
    """Represents a decomposed research plan."""

    original_query: str
    sub_queries: list[str]


class ResearchPlannerService:
    """Decomposes a broad research question into high-signal, complementary search queries."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = DEFAULT_PLANNER_MODEL,
    ):
        """Initialize planner service with Gemini credentials."""
        if api_key is not None:
            self._api_key = api_key.strip()
        else:
            self._api_key = (get_gemini_api_key() or "").strip()
        self._model_name = model_name

    def plan_sub_queries(
        self,
        query: str,
        max_sub_queries: int = 2,
        timeout_seconds: float = 10.0,
    ) -> list[str]:
        """Generate targeted sub-queries to broaden and deepen evidence collection.

        Args:
            query: The root research question.
            max_sub_queries: Target number of sub-queries (bounded between 1 and 3).
            timeout_seconds: Maximum allowed socket timeout for planner generation.

        Returns:
            A list of distinct sub-query strings (may be empty if planning fails or is skipped).
        """
        clean_query = (query or "").strip()
        if not clean_query or not self._api_key:
            return []

        clamped_count = max(1, min(max_sub_queries, 3))

        prompt = f"""You are a research strategist. Decompose this research question into {clamped_count} distinct, high-signal search queries for a web search engine:

RESEARCH QUESTION:
"{clean_query}"

GOALS:
- Cover complementary angles: (1) technical mechanisms/foundations, (2) empirical benchmarks/data, (3) limitations/trade-offs.
- Make each search query concise (3 to 8 words) and keyword-dense for a web search engine.
- Do NOT repeat the exact wording of the original question.
- Return ONLY a valid JSON array of {clamped_count} strings. No explanation, no markdown text outside the array.

Example format:
["sub-query 1", "sub-query 2"]"""

        timeout_ms = max(10000, int(timeout_seconds * 1000))
        client = genai.Client(
            api_key=self._api_key,
            http_options=types.HttpOptions(timeout=timeout_ms),
        )
        try:
            response = client.models.generate_content(
                model=self._model_name,
                contents=prompt,
            )
            raw_text = (response.text or "").strip()
            return self._parse_sub_queries(raw_text, clean_query, clamped_count)
        except Exception as err:
            logger.warning("Query planning failed or timed out (%s); continuing with root query.", err)
            return []

    def _parse_sub_queries(
        self,
        raw_text: str,
        original_query: str,
        max_count: int,
    ) -> list[str]:
        """Safely parse sub-queries from model response, handling JSON or line formats."""
        if not raw_text:
            return []

        # Strip markdown fences if present
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw_text, flags=re.MULTILINE)
        cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.MULTILINE).strip()

        candidates: list[str] = []

        # Try JSON parsing
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                candidates = [str(item).strip() for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            # Fallback: line-by-line parsing for bulleted or numbered lists
            lines = cleaned.splitlines()
            for line in lines:
                sub = re.sub(r"^[\d\.\-\*\s]+", "", line).strip()
                # Remove quotes if present
                sub = sub.strip('"\'')
                if sub and len(sub) > 4:
                    candidates.append(sub)

        # Deduplicate, ensure not identical to original query, clamp to max_count
        filtered: list[str] = []
        orig_lower = original_query.lower()
        seen = set()

        for c in candidates:
            c_clean = c.strip()
            c_lower = c_clean.lower()
            if c_lower != orig_lower and c_lower not in seen:
                seen.add(c_lower)
                filtered.append(c_clean)
            if len(filtered) >= max_count:
                break

        return filtered
