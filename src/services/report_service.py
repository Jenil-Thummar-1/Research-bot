"""Gemini report generation service for synthesizing structured research reports."""

from dataclasses import dataclass
import logging
import re
import urllib.parse
from google import genai
from google.genai import types
from google.genai.errors import APIError, ClientError, ServerError
from src.config import get_gemini_api_key
from src.services.search_service import SearchResult

logger = logging.getLogger(__name__)

# Primary verified production model for synthesis
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"


@dataclass(frozen=True)
class GroundedSource:
    """Represents a verified evidence source mapped to a stable identifier with provenance."""

    source_id: str  # e.g., "S1", "S2"
    title: str
    url: str
    content: str
    domain: str
    is_full_content: bool = False
    origin_query: str = ""
    extraction_status: str = "Search snippet"
    score: float | None = None


@dataclass(frozen=True)
class ResearchReport:
    """Represents a complete synthesized research report with grounded citations."""

    query: str
    content: str  # Markdown content with linked citations
    raw_response: str
    sources: list[GroundedSource]
    model_used: str


class ReportServiceError(Exception):
    """Base exception for report generation errors."""

    pass


class GeminiTimeoutError(ReportServiceError):
    """Raised when Gemini synthesis exceeds the allocated time budget."""

    pass


class MissingGeminiKeyError(ReportServiceError):
    """Raised when the Gemini API key is missing or blank."""

    pass


class GeminiAuthError(ReportServiceError):
    """Raised when the Gemini API key is rejected or unauthorized."""

    pass


class GeminiQuotaError(ReportServiceError):
    """Raised when Gemini quota or rate limits are exceeded."""

    pass


class GeminiGenerationError(ReportServiceError):
    """Raised when content generation fails or returns an empty response."""

    pass


def extract_domain_name(url: str) -> str:
    """Extract clean domain name from a URL."""
    try:
        parsed = urllib.parse.urlparse(url)
        netloc = parsed.netloc or parsed.path
        return netloc.replace("www.", "").split(":")[0]
    except Exception:
        return "web source"


def bind_citations_to_urls(text: str, sources_by_id: dict[str, GroundedSource]) -> str:
    """Replace [S1], [S2] source markers with clickable markdown links to verified URLs.

    Only source identifiers matching known, retrieved sources will be linked.
    """
    def _replace_marker(match: re.Match) -> str:
        tag = match.group(1).upper()
        if tag in sources_by_id:
            src = sources_by_id[tag]
            # Output Markdown link with title tooltip: [S1](url "Title")
            safe_title = src.title.replace('"', "'")
            return f"[[{tag}]]({src.url} \"{safe_title}\")"
        return match.group(0)

    # Match patterns like [S1], [s2], [S10]
    pattern = re.compile(r"\[(S\d+)\]", re.IGNORECASE)
    return pattern.sub(_replace_marker, text)


class GeminiReportService:
    """Service for analyzing search evidence and generating structured research reports."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = DEFAULT_GEMINI_MODEL,
    ):
        """Initialize the report service.

        Args:
            api_key: Optional explicit Gemini API key. Falls back to GEMINI_API_KEY from .env.
            model_name: The Gemini model to use for synthesis.
        """
        if api_key is not None:
            self._api_key = api_key.strip()
        else:
            self._api_key = (get_gemini_api_key() or "").strip()
        self._model_name = model_name

    @property
    def is_configured(self) -> bool:
        """Check whether a Gemini API key is present."""
        return bool(self._api_key)

    @property
    def model_name(self) -> str:
        """Return the active model name."""
        return self._model_name

    def build_grounded_sources(self, results: list) -> list[GroundedSource]:
        """Assign stable identifiers [S1], [S2], ... to raw search or enriched sources."""
        grounded: list[GroundedSource] = []
        for idx, res in enumerate(results, start=1):
            source_id = f"S{idx}"
            domain = getattr(res, "domain", None) or extract_domain_name(res.url)
            is_full = getattr(res, "is_full_content", False)
            origin_query = getattr(res, "origin_query", "")
            ext_status = getattr(res, "extraction_status", "Search snippet")
            score = getattr(res, "score", None)

            grounded.append(
                GroundedSource(
                    source_id=source_id,
                    title=res.title or f"Source {idx}",
                    url=res.url,
                    content=res.content or "No snippet available.",
                    domain=domain,
                    is_full_content=is_full,
                    origin_query=origin_query,
                    extraction_status=ext_status,
                    score=score,
                )
            )
        return grounded

    def _build_prompt(self, query: str, sources: list[GroundedSource], concise_mode: bool = False) -> str:
        """Construct a grounding prompt for synthesis (concise fast mode or full deep mode)."""
        evidence_blocks: list[str] = []
        for src in sources:
            evidence_type = "Full Article Content" if src.is_full_content else "Search Snippet"
            query_ctx = f" (Found via: '{src.origin_query}')" if src.origin_query else ""
            evidence_blocks.append(
                f"[{src.source_id}] {src.title}\n"
                f"Source Domain: {src.domain}\n"
                f"URL: {src.url}\n"
                f"Evidence Type: {evidence_type}{query_ctx}\n"
                f"Content:\n{src.content}\n"
            )

        evidence_text = "\n---\n".join(evidence_blocks)

        if concise_mode:
            return f"""You are an expert AI research assistant providing a rapid, concise answer.

RESEARCH QUESTION:
"{query}"

VERIFIED EVIDENCE SOURCES:
{evidence_text}

RULES:
1. Strict Grounding: Rely strictly on the provided evidence. Cite sources inline using tags like [S1] or [S2].
2. Fast Structured Format:
   - ## Summary (Direct, high-impact answer answering the user's question, citing sources)
   - ## Key Insights (Bulleted findings highlighting core facts or data)
   - ## Caveats & Gaps (Disclose any missing details, disagreements, or limitations)
   - ## Sources (List [S1], [S2] with title)
3. Brevity: Keep the total response under 350 words so it generates quickly.

Start directly with ## Summary."""

        prompt = f"""You are a senior AI research analyst synthesizing a structured, production-quality technical research report.

RESEARCH QUESTION:
"{query}"

VERIFIED EVIDENCE SOURCES (Retrieved from web):
{evidence_text}

MANDATORY CITATION & SYNTHESIS RULES:
1. Strict Grounding: Base all factual assertions, metrics, and conclusions strictly on the provided evidence sources. Do not speculate or extrapolate without explicit qualification.
2. Citation Requirement: Every claim, statistic, or core insight MUST be cited inline using its exact source identifier, e.g., [S1] or [S2][S3].
3. Citation Integrity: Do NOT invent, guess, or create new source identifiers or external URLs. Use ONLY the identifiers provided above ([S1] through [S{len(sources)}]).
4. Disagreements & Contradictions: If sources disagree, present varying figures, or reflect differing industry perspectives, explicitly document the discrepancies in a dedicated subsection.
5. Incomplete Evidence & Paywalls: If certain facts could not be verified, or if sources were inaccessible/paywalled, clearly document those boundaries in the Limitations section. Do not claim the report is exhaustive.

REQUIRED REPORT STRUCTURE:
1. Executive Summary: High-level synthesis of findings with inline citations.
2. Key Findings: Bulleted core insights grounded with inline citations.
3. Comparative Analysis & Technical Assessment: Deep dive into mechanisms, trade-offs, and data.
4. Contradictions & Divergent Perspectives: Highlight discrepancies or nuances between sources.
5. Limitations & Evidence Gaps: Explicit disclosure of unanswered questions, source limitations, and boundaries.
6. Evidence Source Index: List each source tag ([S1], [S2], etc.) with its title and a 1-line summary of its contribution.

Generate the report in clean Markdown format. Start directly with the Executive Summary."""
        return prompt

    def generate_report(
        self,
        query: str,
        results: list,
        timeout_seconds: float = 14.0,
        concise_mode: bool = False,
    ) -> ResearchReport:
        """Synthesize a structured research report from verified search results.

        Args:
            query: The user's research query.
            results: List of SearchResult or EnrichedSource objects from Tavily.
            timeout_seconds: Maximum allowed socket timeout for generation (default: 14.0s).
            concise_mode: If True, uses a compact prompt for rapid responses under 15s.

        Returns:
            A ResearchReport object containing the report and clickable citations.

        Raises:
            GeminiTimeoutError: If Gemini synthesis times out or server deadline expires.
            MissingGeminiKeyError: If GEMINI_API_KEY is not configured.
            GeminiAuthError: If the key is invalid or unauthorized.
            GeminiQuotaError: If API rate limits are reached.
            GeminiGenerationError: If model response is empty or fails.
        """
        if not self._api_key:
            raise MissingGeminiKeyError(
                "Gemini API key is missing. Add GEMINI_API_KEY to your .env file."
            )

        if not results:
            raise GeminiGenerationError(
                "Cannot generate report without evidence sources. Please provide search results."
            )

        sources = self.build_grounded_sources(results)
        sources_by_id = {src.source_id: src for src in sources}
        prompt = self._build_prompt(query, sources, concise_mode=concise_mode)

        # Google API requires timeout in ms with minimum 10000ms
        timeout_ms = max(10000, int(timeout_seconds * 1000))
        client = genai.Client(
            api_key=self._api_key,
            http_options=types.HttpOptions(timeout=timeout_ms),
        )
        response = None

        try:
            response = client.models.generate_content(
                model=self._model_name,
                contents=prompt,
            )
        except ServerError as err:
            err_str = str(err).lower()
            if "504" in err_str or "deadline" in err_str or "timed out" in err_str:
                raise GeminiTimeoutError(
                    f"Gemini synthesis timed out after {timeout_seconds:.1f}s deadline."
                ) from err
            if "503" in err_str or "unavailable" in err_str:
                # 1 quick retry with 1s sleep
                try:
                    import time
                    time.sleep(1.0)
                    response = client.models.generate_content(
                        model=self._model_name,
                        contents=prompt,
                    )
                except Exception as retry_err:
                    retry_str = str(retry_err).lower()
                    if "504" in retry_str or "deadline" in retry_str or "timed out" in retry_str:
                        raise GeminiTimeoutError(
                            f"Gemini synthesis timed out after {timeout_seconds:.1f}s deadline."
                        ) from retry_err
                    raise ReportServiceError(f"Gemini service unavailable (503): {retry_err}") from retry_err
            else:
                raise ReportServiceError(f"Gemini server error: {err}") from err
        except ClientError as err:
            err_str = str(err).lower()
            if "401" in err_str or "403" in err_str or "unauthorized" in err_str or "api_key_invalid" in err_str:
                raise GeminiAuthError(
                    "Gemini API key is invalid or unauthorized. Check your key at https://aistudio.google.com."
                ) from err
            if "429" in err_str or "quota" in err_str or "resource_exhausted" in err_str:
                raise GeminiQuotaError(
                    "Gemini API quota or rate limit exceeded. Please wait a moment or check your quota."
                ) from err
            raise ReportServiceError(f"Gemini API client error: {err}") from err
        except APIError as err:
            err_str = str(err).lower()
            if "504" in err_str or "deadline" in err_str or "timed out" in err_str:
                raise GeminiTimeoutError(f"Gemini synthesis timed out after {timeout_seconds:.1f}s.") from err
            raise ReportServiceError(f"Gemini API error: {err}") from err
        except Exception as err:
            err_str = str(err).lower()
            if "deadline" in err_str or "timed out" in err_str or "timeout" in err_str:
                raise GeminiTimeoutError(f"Gemini synthesis timed out: {err}") from err
            logger.exception("Unexpected error during Gemini report generation: %s", err)
            raise ReportServiceError(f"Failed to generate research report: {err}") from err

        raw_text = (response.text or "").strip()
        if not raw_text:
            raise GeminiGenerationError("Gemini returned an empty response.")

        # Bind citation markers like [S1] to verified URLs
        linked_content = bind_citations_to_urls(raw_text, sources_by_id)

        return ResearchReport(
            query=query,
            content=linked_content,
            raw_response=raw_text,
            sources=sources,
            model_used=self._model_name,
        )
