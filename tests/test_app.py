"""Automated UI tests for the Streamlit research app using AppTest."""

from pathlib import Path
from unittest.mock import patch, MagicMock
from streamlit.testing.v1 import AppTest
from src.services.search_service import SearchResult

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def test_app_initial_render():
    """Verify that app.py loads cleanly, renders headers, inputs, and buttons without exceptions."""
    at = AppTest.from_file(APP_PATH, default_timeout=15).run()
    assert not at.exception, f"App raised an exception on load: {at.exception}"

    # Verify key UI components exist
    assert any("AI Research Agent" in str(md.value) for md in at.markdown)
    assert len(at.text_input) >= 1
    assert len(at.button) >= 3  # quick prompt buttons
    assert len(at.form_submit_button) >= 3  # Deep Research, Standard Research, Search Only


def test_app_empty_query_warning():
    """Verify that submitting an empty query triggers a warning."""
    at = AppTest.from_file(APP_PATH, default_timeout=15).run()
    if at.form_submit_button:
        # Click the primary deep research button without input
        at.form_submit_button[0].click().run()
        assert not at.exception
        assert len(at.warning) >= 1
        assert "Please enter a research question" in at.warning[0].value


@patch("src.services.search_service.TavilySearchService.search")
def test_app_search_only_workflow(mock_search):
    """Verify that clicking Search Only invokes search and displays results."""
    mock_search.return_value = [
        SearchResult(
            title="Synthetic Biology Frontiers",
            url="https://example.com/synbio",
            content="An overview of CRISPR and metabolic engineering.",
            score=0.92,
        )
    ]

    at = AppTest.from_file(APP_PATH, default_timeout=15).run()
    at.text_input[0].set_value("synthetic biology")
    # Click "Search Only" button (index 2)
    if len(at.form_submit_button) >= 3:
        at.form_submit_button[2].click().run()
        assert not at.exception
        assert any("Found" in str(sc.value) for sc in at.success)


@patch("src.services.report_service.genai.Client")
@patch("src.services.search_service.TavilySearchService.search")
def test_app_fast_answer_workflow(mock_search, mock_genai_client_cls):
    """Verify that clicking Fast Answer runs single search and synthesizes concise report."""
    mock_search.return_value = [
        SearchResult(
            title="Post-Quantum NIST Standards",
            url="https://example.com/nist-pqc",
            content="ML-KEM and ML-DSA standards finalized in 2024.",
            score=0.98,
        )
    ]

    mock_client = MagicMock()
    mock_genai_client_cls.return_value = mock_client
    mock_resp = MagicMock()
    mock_resp.text = "## Executive Summary\nNIST announced finalized post-quantum algorithms [S1]."
    mock_client.models.generate_content.return_value = mock_resp

    at = AppTest.from_file(APP_PATH, default_timeout=15).run()
    at.text_input[0].set_value("Post-quantum cryptography standards")
    # Click "Fast Answer" button (index 0)
    if len(at.form_submit_button) >= 1:
        at.form_submit_button[0].click().run()

    assert not at.exception
    assert any("Research Report" in str(md.value) for md in at.markdown)
    assert any("NIST announced finalized post-quantum algorithms" in str(md.value) for md in at.markdown)


@patch("src.services.report_service.GeminiReportService.generate_report")
@patch("src.services.search_service.TavilySearchService.search")
def test_app_fast_fallback_on_timeout(mock_search, mock_gen_report):
    """Verify that if Gemini synthesis times out, the app immediately displays search fallback results."""
    from src.services.report_service import GeminiTimeoutError

    mock_search.return_value = [
        SearchResult(
            title="Post-Quantum Cryptography Backup",
            url="https://example.com/nist-backup",
            content="Useful web snippet preserved during fallback.",
            score=0.95,
        )
    ]
    mock_gen_report.side_effect = GeminiTimeoutError("Gemini synthesis reached deadline.")

    at = AppTest.from_file(APP_PATH, default_timeout=15).run()
    at.text_input[0].set_value("Post-quantum standards")
    # Click "Fast Answer" button (index 0)
    at.form_submit_button[0].click().run()

    assert not at.exception
    # Verify fallback notice and source results are displayed
    assert any("Fast Search Fallback" in str(md.value) for md in at.markdown)
    assert any("Post-Quantum Cryptography Backup" in str(md.value) for md in at.markdown)
    assert any("Useful web snippet preserved" in str(md.value) for md in at.markdown)


@patch("src.services.deep_research_service.ResearchPlannerService.plan_sub_queries")
@patch("src.services.deep_research_service.TavilySearchService.search")
@patch("src.services.deep_research_service.DeepResearchService._extract_full_content")
@patch("src.services.report_service.genai.Client")
def test_app_deep_research_workflow(mock_genai_cls, mock_extract, mock_search, mock_plan):
    """Verify that clicking Run Deep Research executes the multi-query pipeline and renders provenance."""
    from src.services.deep_research_service import EnrichedSource

    mock_plan.return_value = ["sub query battery"]
    mock_search.return_value = [
        SearchResult(
            title="Solid State Battery Progress",
            url="https://example.com/battery",
            content="Solid electrolyte designs show promise.",
            score=0.95,
        )
    ]
    mock_extract.return_value = (
        [
            EnrichedSource(
                title="Solid State Battery Progress",
                url="https://example.com/battery",
                canonical_url="https://example.com/battery",
                domain="example.com",
                content="Full article content discussing cathode stability.",
                score=0.95,
                origin_query="solid state",
                is_full_content=True,
                extraction_status="Full article extracted",
            )
        ],
        1,
        0,
    )

    mock_client = MagicMock()
    mock_genai_cls.return_value = mock_client
    mock_resp = MagicMock()
    mock_resp.text = "## Executive Summary\nDeep analysis completed [S1].\n## Discrepancies\nNone noted."
    mock_client.models.generate_content.return_value = mock_resp

    at = AppTest.from_file(APP_PATH, default_timeout=15).run()
    at.text_input[0].set_value("Solid-state batteries in EVs")
    # Click "Run Deep Research" button (index 1)
    at.form_submit_button[1].click().run()

    assert not at.exception
    # Check metrics rendered
    assert len(at.metric) >= 4
    # Check provenance notice and report
    assert any("Methodology & Grounding Notice" in str(md.value) for md in at.markdown)
    assert any("Deep analysis completed" in str(md.value) for md in at.markdown)
    assert any("Full Article" in str(md.value) for md in at.markdown)


@patch("src.services.verification_service.EvidenceVerificationService.verify_report")
@patch("src.services.deep_research_service.DeepResearchService.run_deep_research")
@patch("src.services.report_service.genai.Client")
def test_app_verification_ui_rendered(mock_genai_cls, mock_deep, mock_verify):
    """Verify that Milestone 4 verification summary card and audit details render properly."""
    from src.services.verification_service import VerificationSummary, VerifiedClaim, ClaimStatus
    from src.services.deep_research_service import DeepResearchResult, EnrichedSource

    mock_deep.return_value = DeepResearchResult(
        query="Post-quantum cryptography standards",
        sub_queries=["pqc algorithms"],
        sources=[
            EnrichedSource(
                title="NIST Post-Quantum Cryptography",
                url="https://example.com/pqc",
                canonical_url="https://example.com/pqc",
                domain="example.com",
                content="Standards finalized in August 2024.",
                score=0.95,
                origin_query="pqc",
                is_full_content=False,
            )
        ],
        total_searches_run=1,
        full_content_count=0,
        snippet_fallback_count=1,
    )

    mock_client = MagicMock()
    mock_genai_cls.return_value = mock_client
    mock_resp = MagicMock()
    mock_resp.text = "## Executive Summary\nNIST finalized post-quantum standards in August 2024 [S1]."
    mock_client.models.generate_content.return_value = mock_resp

    mock_verify.return_value = VerificationSummary(
        total_claims=1,
        supported_count=1,
        partially_supported_count=0,
        unsupported_count=0,
        conflicting_count=0,
        unverified_count=0,
        reliability_score=100.0,
        status_label="High Reliability",
        is_fully_verified=True,
        claims=[
            VerifiedClaim(
                claim_text="NIST finalized post-quantum standards in August 2024 [S1].",
                cited_source_ids=["S1"],
                valid_source_ids=["S1"],
                invalid_source_ids=[],
                status=ClaimStatus.SUPPORTED,
                explanation="S1 confirms August 2024 finalization date.",
            )
        ],
        discrepancy_notes=[],
    )

    at = AppTest.from_file(APP_PATH, default_timeout=15).run()
    # Enable verification checkbox
    if at.checkbox:
        at.checkbox[0].check()
    at.text_input[0].set_value("Post-quantum cryptography standards")
    # Click Deep Research (button 1)
    at.form_submit_button[1].click().run()

    assert not at.exception
    # Verify that the verification summary box is rendered in the UI
    assert any("Evidence Verification Audit" in str(md.value) for md in at.markdown)
    assert any("Reliability Score: 100.0%" in str(md.value) for md in at.markdown)
    assert any("Detailed Claim-by-Claim Evidence Audit" in str(exp.label) for exp in at.expander)


@patch("src.services.chat_service.genai.Client")
def test_app_chat_normal_flow(mock_genai_cls):
    """Verify that sending a normal chat message generates a response without web search."""
    mock_client = MagicMock()
    mock_genai_cls.return_value = mock_client
    mock_resp = MagicMock()
    mock_resp.text = "Python decorators are callables that accept a function and return an enhanced function."
    mock_client.models.generate_content.return_value = mock_resp
    mock_client.models.generate_content_stream.return_value = [mock_resp]

    at = AppTest.from_file(APP_PATH, default_timeout=15).run()
    assert not at.exception
    assert len(at.chat_input) >= 1

    at.chat_input[0].set_value("Explain Python decorators").run()
    assert not at.exception
    assert any("Python decorators are callables" in str(md.value) for md in at.markdown)


@patch("src.services.chat_service.TavilySearchService.search")
@patch("src.services.chat_service.genai.Client")
def test_app_chat_with_web_search_enabled(mock_genai_cls, mock_search):
    """Verify that toggling web search performs Tavily search and binds citations."""
    mock_search.return_value = [
        SearchResult(
            title="Next Gen Fusion Milestones",
            url="https://example.com/fusion-news",
            content="Recent net energy gain demonstrated in magnetic confinement experiment.",
            score=0.96,
        )
    ]
    mock_client = MagicMock()
    mock_genai_cls.return_value = mock_client
    mock_resp = MagicMock()
    mock_resp.text = "According to [S1], new magnetic confinement experiments demonstrated net energy gain."
    mock_client.models.generate_content.return_value = mock_resp
    mock_client.models.generate_content_stream.return_value = [mock_resp]

    at = AppTest.from_file(APP_PATH, default_timeout=15).run()
    assert not at.exception

    # Enable web search checkbox in sidebar
    search_checkboxes = [cb for cb in at.checkbox if "Search the Web" in str(cb.label)]
    if search_checkboxes:
        search_checkboxes[0].check()

    at.chat_input[0].set_value("What are the latest fusion energy breakthroughs?").run()
    assert not at.exception
    assert mock_search.called
    assert any("fusion-news" in str(md.value) for md in at.markdown)

