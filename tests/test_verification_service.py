"""Unit tests for EvidenceVerificationService: claim evaluation, citation auditing, and conflict detection."""

from unittest.mock import MagicMock, patch
import json
import pytest
from google.genai.errors import ClientError, ServerError
from src.services.report_service import GroundedSource
from src.services.verification_service import (
    EvidenceVerificationService,
    ClaimStatus,
    VerifiedClaim,
    VerificationSummary,
    extract_cited_source_ids,
    build_verification_summary,
)


@pytest.fixture
def sample_sources():
    """Sample grounded evidence sources for testing."""
    return [
        GroundedSource(
            source_id="S1",
            title="Battery Density Study 2026",
            url="https://example.com/s1",
            domain="example.com",
            content="Solid state batteries achieved 450 Wh/kg in laboratory testing.",
        ),
        GroundedSource(
            source_id="S2",
            title="Commercial EV Projections",
            url="https://example.com/s2",
            domain="example.com",
            content="Automakers project commercial EV deployment by 2030, though scaling remains slow.",
        ),
    ]


def test_extract_cited_source_ids():
    """Verify regex extraction and deduplication of citation tags."""
    text = "Breakthroughs in solid state [S1] allow higher density [S2][s1], while [S3] is also cited."
    tags = extract_cited_source_ids(text)
    assert tags == ["S1", "S2", "S3"]


def test_build_verification_summary_metrics():
    """Verify reliability score calculation and status classification."""
    claims = [
        VerifiedClaim(
            claim_text="Claim 1",
            cited_source_ids=["S1"],
            valid_source_ids=["S1"],
            invalid_source_ids=[],
            status=ClaimStatus.SUPPORTED,
            explanation="Exact match in S1",
        ),
        VerifiedClaim(
            claim_text="Claim 2",
            cited_source_ids=["S2"],
            valid_source_ids=["S2"],
            invalid_source_ids=[],
            status=ClaimStatus.PARTIALLY_SUPPORTED,
            explanation="Partially verified",
            revised_text="Qualified claim 2",
        ),
    ]

    # 1.0 (Supported) + 0.5 (Partially) / 2.0 = 75.0%
    summary = build_verification_summary(claims)
    assert summary.total_claims == 2
    assert summary.supported_count == 1
    assert summary.partially_supported_count == 1
    assert summary.unsupported_count == 0
    assert summary.conflicting_count == 0
    assert summary.reliability_score == 75.0
    assert summary.is_fully_verified is True
    assert "Moderate" in summary.status_label


@patch("src.services.verification_service.genai.Client")
def test_verify_supported_claim(mock_client_cls, sample_sources):
    """Verify classification of a fully supported factual claim."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_resp = MagicMock()
    mock_resp.text = json.dumps({
        "claims": [
            {
                "claim_text": "Solid state batteries achieved 450 Wh/kg in testing [S1].",
                "cited_source_ids": ["S1"],
                "status": "Supported",
                "explanation": "S1 explicitly states 450 Wh/kg was achieved in laboratory testing.",
                "revised_text": None,
                "conflicting_details": None,
            }
        ],
        "discrepancy_notes": [],
    })
    mock_client.models.generate_content.return_value = mock_resp

    service = EvidenceVerificationService(api_key="gemini-mock")
    summary = service.verify_report("Test report with [S1]", sample_sources)

    assert summary.is_fully_verified is True
    assert summary.supported_count == 1
    assert summary.claims[0].status == ClaimStatus.SUPPORTED
    assert summary.claims[0].invalid_source_ids == []


@patch("src.services.verification_service.genai.Client")
def test_verify_partially_supported_claim_with_revision(mock_client_cls, sample_sources):
    """Verify partially supported claims include a qualified revision."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_resp = MagicMock()
    mock_resp.text = json.dumps({
        "claims": [
            {
                "claim_text": "All major carmakers will replace lithium-ion by 2030 [S2].",
                "cited_source_ids": ["S2"],
                "status": "Partially Supported",
                "explanation": "S2 mentions deployment by 2030, but does not state all carmakers will replace lithium-ion.",
                "revised_text": "Automakers project commercial solid-state EV deployment by 2030, but scaling remains slow [S2].",
                "conflicting_details": None,
            }
        ],
        "discrepancy_notes": [],
    })
    mock_client.models.generate_content.return_value = mock_resp

    service = EvidenceVerificationService(api_key="gemini-mock")
    summary = service.verify_report("Test report with [S2]", sample_sources)

    assert summary.partially_supported_count == 1
    assert summary.claims[0].status == ClaimStatus.PARTIALLY_SUPPORTED
    assert summary.claims[0].revised_text is not None


@patch("src.services.verification_service.genai.Client")
def test_verify_conflicting_claims(mock_client_cls, sample_sources):
    """Verify detection and explanation of conflicting sources."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_resp = MagicMock()
    mock_resp.text = json.dumps({
        "claims": [
            {
                "claim_text": "Commercial availability will occur by 2026 according to [S1], but [S2] states 2030.",
                "cited_source_ids": ["S1", "S2"],
                "status": "Conflicting",
                "explanation": "S1 suggests immediate near-term readiness while S2 points to 2030.",
                "revised_text": None,
                "conflicting_details": "Timeline discrepancy: S1 references lab tests whereas S2 focuses on commercial roadmaps (2030).",
            }
        ],
        "discrepancy_notes": ["Sources disagree on commercialization timeframe."],
    })
    mock_client.models.generate_content.return_value = mock_resp

    service = EvidenceVerificationService(api_key="gemini-mock")
    summary = service.verify_report("Report with conflict", sample_sources)

    assert summary.conflicting_count == 1
    assert summary.claims[0].status == ClaimStatus.CONFLICTING
    assert summary.claims[0].conflicting_details is not None
    assert len(summary.discrepancy_notes) >= 1


@patch("src.services.verification_service.genai.Client")
def test_fabricated_citation_intercepted(mock_client_cls, sample_sources):
    """Verify that citing non-existent source IDs is caught deterministically and marked Unsupported."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    # Model claims S9 is valid even though registry only has S1 and S2
    mock_resp = MagicMock()
    mock_resp.text = json.dumps({
        "claims": [
            {
                "claim_text": "Breakthrough announced by Acme Labs [S9].",
                "cited_source_ids": ["S9"],
                "status": "Supported",  # Model claims supported
                "explanation": "Found in source S9.",
            }
        ],
        "discrepancy_notes": [],
    })
    mock_client.models.generate_content.return_value = mock_resp

    service = EvidenceVerificationService(api_key="gemini-mock")
    summary = service.verify_report("Report with fabricated S9 citation", sample_sources)

    # Verification service must override status to Unsupported due to invalid ID
    claim = summary.claims[0]
    assert claim.status == ClaimStatus.UNSUPPORTED
    assert "S9" in claim.invalid_source_ids
    assert "Fabricated citation detected" in claim.explanation


@patch("src.services.verification_service.genai.Client")
def test_malformed_model_response_handling(mock_client_cls, sample_sources):
    """Verify that non-JSON output fails gracefully with failure_reason and is_fully_verified=False."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_resp = MagicMock()
    mock_resp.text = "Sorry, I cannot verify this as JSON right now."
    mock_client.models.generate_content.return_value = mock_resp

    service = EvidenceVerificationService(api_key="gemini-mock")
    summary = service.verify_report("Report text", sample_sources)

    assert summary.is_fully_verified is False
    assert "not valid JSON" in summary.failure_reason
    assert summary.status_label == "Verification Incomplete"


@patch("src.services.verification_service.genai.Client")
def test_api_client_error_handling(mock_client_cls, sample_sources):
    """Verify that API errors do not crash and clearly report incomplete verification."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.models.generate_content.side_effect = ClientError(429, {"error": "RESOURCE_EXHAUSTED"})

    service = EvidenceVerificationService(api_key="gemini-mock")
    summary = service.verify_report("Report text", sample_sources)

    assert summary.is_fully_verified is False
    assert "Verification API error" in summary.failure_reason


def test_empty_inputs():
    """Verify empty report or empty sources return incomplete summary."""
    service = EvidenceVerificationService(api_key="mock")
    summary1 = service.verify_report("", [GroundedSource("S1", "T", "U", "C", "D")])
    assert summary1.is_fully_verified is False
    assert "Report text is empty" in summary1.failure_reason

    summary2 = service.verify_report("Some report", [])
    assert summary2.is_fully_verified is False
    assert "No evidence sources" in summary2.failure_reason
