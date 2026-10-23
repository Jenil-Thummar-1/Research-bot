"""Evidence verification and answer reliability service.

Performs claim-level factual verification, citation support auditing, conflict detection,
and source-grounded qualification of synthesized reports.
"""

from dataclasses import dataclass, field
from enum import Enum
import json
import logging
import re
import time
from google import genai
from google.genai import types
from google.genai.errors import APIError, ClientError, ServerError
from src.config import get_gemini_api_key
from src.services.report_service import GroundedSource, DEFAULT_GEMINI_MODEL

logger = logging.getLogger(__name__)


class ClaimStatus(str, Enum):
    """Classification status for a verified claim."""

    SUPPORTED = "Supported"
    PARTIALLY_SUPPORTED = "Partially Supported"
    UNSUPPORTED = "Unsupported"
    CONFLICTING = "Conflicting"


@dataclass(frozen=True)
class VerifiedClaim:
    """Represents a factual claim evaluated against retrieved evidence sources."""

    claim_text: str
    cited_source_ids: list[str]
    valid_source_ids: list[str]
    invalid_source_ids: list[str]
    status: ClaimStatus
    explanation: str
    revised_text: str | None = None
    conflicting_details: str | None = None


@dataclass(frozen=True)
class VerificationSummary:
    """Aggregate audit metrics for report reliability."""

    total_claims: int
    supported_count: int
    partially_supported_count: int
    unsupported_count: int
    conflicting_count: int
    unverified_count: int
    reliability_score: float  # 0.0 to 100.0%
    status_label: str
    is_fully_verified: bool
    claims: list[VerifiedClaim]
    discrepancy_notes: list[str] = field(default_factory=list)
    failure_reason: str | None = None


def extract_cited_source_ids(text: str) -> list[str]:
    """Extract source identifiers (e.g. ['S1', 'S2']) from text using regex."""
    pattern = re.compile(r"\[(S\d+)\]", re.IGNORECASE)
    matches = pattern.findall(text)
    # Deduplicate while preserving order, convert to uppercase
    seen = set()
    ordered = []
    for m in matches:
        tag = m.upper()
        if tag not in seen:
            seen.add(tag)
            ordered.append(tag)
    return ordered


def build_verification_summary(
    claims: list[VerifiedClaim],
    failure_reason: str | None = None,
    discrepancy_notes: list[str] | None = None,
) -> VerificationSummary:
    """Compute aggregate verification statistics and reliability scores."""
    total = len(claims)
    if total == 0 or failure_reason is not None:
        return VerificationSummary(
            total_claims=total,
            supported_count=0,
            partially_supported_count=0,
            unsupported_count=0,
            conflicting_count=0,
            unverified_count=total if total > 0 else 0,
            reliability_score=0.0,
            status_label="Verification Incomplete",
            is_fully_verified=False,
            claims=claims,
            discrepancy_notes=discrepancy_notes or [],
            failure_reason=failure_reason or "No claims evaluated.",
        )

    supported = sum(1 for c in claims if c.status == ClaimStatus.SUPPORTED)
    partially = sum(1 for c in claims if c.status == ClaimStatus.PARTIALLY_SUPPORTED)
    unsupported = sum(1 for c in claims if c.status == ClaimStatus.UNSUPPORTED)
    conflicting = sum(1 for c in claims if c.status == ClaimStatus.CONFLICTING)

    # Weighted reliability score: Supported = 1.0, Partially = 0.5, Unsupported = 0.0, Conflicting = 0.0
    weighted_pts = (supported * 1.0) + (partially * 0.5)
    reliability_pct = round((weighted_pts / total) * 100.0, 1)

    if reliability_pct >= 85.0 and unsupported == 0 and conflicting == 0:
        label = "High Reliability"
    elif reliability_pct >= 60.0:
        label = "Moderate (Minor Caveats)"
    else:
        label = "Low (Contains Unsupported/Conflicting Claims)"

    return VerificationSummary(
        total_claims=total,
        supported_count=supported,
        partially_supported_count=partially,
        unsupported_count=unsupported,
        conflicting_count=conflicting,
        unverified_count=0,
        reliability_score=reliability_pct,
        status_label=label,
        is_fully_verified=True,
        claims=claims,
        discrepancy_notes=discrepancy_notes or [],
        failure_reason=None,
    )


class EvidenceVerificationService:
    """Audits report claims against retrieved evidence for factual fidelity and citation integrity."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = DEFAULT_GEMINI_MODEL,
    ):
        """Initialize verification service."""
        if api_key is not None:
            self._api_key = api_key.strip()
        else:
            self._api_key = (get_gemini_api_key() or "").strip()
        self._model_name = model_name

    def verify_report(
        self,
        report_text: str,
        sources: list,
        max_claims: int = 6,
        timeout_seconds: float = 10.0,
    ) -> VerificationSummary:
        """Audit the factual claims in report_text against the provided sources.

        Args:
            report_text: The markdown report content containing citations.
            sources: List of GroundedSource or EnrichedSource objects.
            max_claims: Maximum number of representative claims to audit (default: 6).
            timeout_seconds: Maximum allowed socket timeout for verification model.

        Returns:
            A VerificationSummary containing claim evaluations, statistics, and revisions.
        """
        clean_text = (report_text or "").strip()
        if not clean_text:
            return build_verification_summary([], failure_reason="Report text is empty.")

        if not sources:
            return build_verification_summary([], failure_reason="No evidence sources provided for verification.")

        if not self._api_key:
            return build_verification_summary(
                [], failure_reason="Gemini API key is missing. Add GEMINI_API_KEY to your .env file."
            )

        # Build source lookup registry
        source_registry: dict[str, dict] = {}
        for idx, src in enumerate(sources, start=1):
            sid = getattr(src, "source_id", None) or f"S{idx}"
            source_registry[sid.upper()] = {
                "id": sid.upper(),
                "title": getattr(src, "title", f"Source {idx}"),
                "domain": getattr(src, "domain", ""),
                "url": getattr(src, "url", ""),
                "content": getattr(src, "content", ""),
            }

        prompt = self._build_audit_prompt(clean_text, source_registry, max_claims)

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
            raw_json_str = (response.text or "").strip()
        except ServerError as err:
            logger.warning("Gemini server error or 503 during verification: %s", err)
            return build_verification_summary(
                [], failure_reason="Gemini server is temporarily busy (503). Verification could not complete."
            )
        except (ClientError, APIError, Exception) as err:
            logger.warning("Gemini verification call failed or timed out: %s", err)
            return build_verification_summary([], failure_reason=f"Verification API error or timeout: {err}")

        if not raw_json_str:
            return build_verification_summary([], failure_reason="Verification model returned an empty response.")

        return self._parse_audit_response(raw_json_str, source_registry)

    def _build_audit_prompt(
        self,
        report_text: str,
        source_registry: dict[str, dict],
        max_claims: int,
    ) -> str:
        """Construct the prompt instructing Gemini to audit claims against evidence."""
        source_blocks = []
        for sid, sinfo in source_registry.items():
            # Truncate source content to avoid prompt explosion
            snippet = sinfo["content"][:2000].replace("\n", " ")
            source_blocks.append(f"[{sid}] Title: {sinfo['title']} | Content: {snippet}")

        sources_text = "\n\n".join(source_blocks)

        prompt = f"""You are a strict, objective fact-checking editor auditing an AI-generated research report.

EVIDENCE SOURCE REGISTRY:
{sources_text}

REPORT CONTENT TO AUDIT:
{report_text}

AUDIT INSTRUCTIONS:
1. Identify up to {max_claims} of the most important specific factual claims, statistics, or conclusions made in the report that cite sources.
2. For each claim, check whether the content of the cited source(s) genuinely supports it.
3. Classify each claim into exactly one status:
   - "Supported": The cited source content directly confirms and substantiates the claim.
   - "Partially Supported": The claim is mostly true according to the source, but slightly overstated, generalized, or missing nuances.
   - "Unsupported": The cited source content does NOT mention or justify the claim, or the claim extrapolates beyond evidence.
   - "Conflicting": Two or more cited sources contradict each other regarding this claim.
4. If a claim is "Unsupported" or "Partially Supported", provide a "revised_text" that tempers or qualifies the claim using ONLY the verified evidence.
5. If sources contradict each other, provide "conflicting_details" explaining the discrepancy objectively.
6. Note any overall discrepancies or source disagreements.

Return ONLY a valid JSON object matching this exact schema:
{{
  "claims": [
    {{
      "claim_text": "Exact or core claim from report",
      "cited_source_ids": ["S1", "S2"],
      "status": "Supported" | "Partially Supported" | "Unsupported" | "Conflicting",
      "explanation": "Short, objective explanation of how source content supports or fails to support the claim.",
      "revised_text": "Qualified revision or null",
      "conflicting_details": "Explanation of discrepancy or null"
    }}
  ],
  "discrepancy_notes": [
    "Short description of any genuine disagreement found across sources"
  ]
}}"""
        return prompt

    def _parse_audit_response(
        self,
        raw_text: str,
        source_registry: dict[str, dict],
    ) -> VerificationSummary:
        """Parse model JSON and enforce deterministic citation integrity against the registry."""
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw_text, flags=re.MULTILINE)
        cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.MULTILINE).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as err:
            logger.warning("Failed to decode verification JSON: %s. Raw text: %s", err, raw_text[:300])
            return build_verification_summary([], failure_reason=f"Model response was not valid JSON: {err}")

        raw_claims = data.get("claims", [])
        discrepancy_notes = [str(d) for d in data.get("discrepancy_notes", []) if str(d).strip()]

        if not isinstance(raw_claims, list):
            return build_verification_summary([], failure_reason="Malformed claims format in audit response.")

        verified_claims: list[VerifiedClaim] = []
        known_sids = set(source_registry.keys())

        for item in raw_claims:
            if not isinstance(item, dict):
                continue

            claim_text = str(item.get("claim_text", "")).strip()
            if not claim_text:
                continue

            # Parse cited source IDs
            raw_sids = item.get("cited_source_ids", [])
            if isinstance(raw_sids, list):
                cited_sids = [str(s).upper().strip() for s in raw_sids if str(s).strip()]
            else:
                cited_sids = extract_cited_source_ids(claim_text)

            # Deterministic Citation Integrity Check:
            # Separate valid IDs (in registry) from invalid/fabricated IDs
            valid_sids = [s for s in cited_sids if s in known_sids]
            invalid_sids = [s for s in cited_sids if s not in known_sids]

            raw_status = str(item.get("status", "Unsupported")).strip()
            explanation = str(item.get("explanation", "")).strip()
            revised_text = item.get("revised_text")
            conflicting_details = item.get("conflicting_details")

            # Map status string to enum safely
            try:
                status = ClaimStatus(raw_status)
            except ValueError:
                status = ClaimStatus.UNSUPPORTED

            # If model fabricated citations (cited source IDs not in registry), override to Unsupported
            if invalid_sids:
                status = ClaimStatus.UNSUPPORTED
                bad_tags = ", ".join(invalid_sids)
                explanation = (
                    f"Fabricated citation detected: source ID(s) [{bad_tags}] do not exist in the retrieved evidence registry. {explanation}".strip()
                )

            verified_claims.append(
                VerifiedClaim(
                    claim_text=claim_text,
                    cited_source_ids=cited_sids,
                    valid_source_ids=valid_sids,
                    invalid_source_ids=invalid_sids,
                    status=status,
                    explanation=explanation or "No explanation provided.",
                    revised_text=str(revised_text).strip() if revised_text else None,
                    conflicting_details=str(conflicting_details).strip() if conflicting_details else None,
                )
            )

        return build_verification_summary(
            claims=verified_claims,
            discrepancy_notes=discrepancy_notes,
        )
