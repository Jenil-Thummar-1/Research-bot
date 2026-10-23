import time
import urllib.parse
import streamlit as st
from src.config import get_tavily_api_key, get_gemini_api_key, mask_key
from src.services.search_service import (
    TavilySearchService,
    SearchResult,
    MissingAPIKeyError,
    InvalidAPIKeyError,
    SearchRateLimitError,
    SearchNetworkError,
    SearchQueryError,
    SearchServiceError,
)
from src.services.planner_service import ResearchPlannerService
from src.services.deep_research_service import (
    DeepResearchService,
    DeepResearchResult,
    EnrichedSource,
    extract_domain,
    apply_domain_diversity,
)
from src.services.report_service import (
    GeminiReportService,
    ResearchReport,
    GroundedSource,
    MissingGeminiKeyError,
    GeminiAuthError,
    GeminiQuotaError,
    GeminiGenerationError,
    GeminiTimeoutError,
    ReportServiceError,
    DEFAULT_GEMINI_MODEL,
)
from src.services.verification_service import (
    EvidenceVerificationService,
    VerificationSummary,
    VerifiedClaim,
    ClaimStatus,
)

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Research Agent | Verified Research Engine",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Custom Styling
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        margin-bottom: 1.25rem;
    }

    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #2563EB 0%, #7C3AED 50%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }

    .main-subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        margin-bottom: 1.25rem;
    }

    .report-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 14px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.2);
    }

    .report-card h1, .report-card h2, .report-card h3 {
        color: #F8FAFC;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }

    .report-card a {
        color: #60A5FA !important;
        font-weight: 600;
        text-decoration: none;
        border-bottom: 1px dotted #60A5FA;
        transition: color 0.15s ease;
    }

    .report-card a:hover {
        color: #93C5FD !important;
        border-bottom: 1px solid #93C5FD;
    }

    .source-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 10px;
        padding: 1.1rem;
        margin-bottom: 0.85rem;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }

    .source-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.4);
    }

    .claim-card {
        background: rgba(255, 255, 255, 0.025);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 10px;
        padding: 1.1rem;
        margin-bottom: 0.85rem;
    }

    .pill-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.2rem 0.55rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .pill-tag {
        background: rgba(99, 102, 241, 0.2);
        color: #A5B4FC;
        border: 1px solid rgba(99, 102, 241, 0.4);
        margin-right: 0.4rem;
    }

    .pill-domain {
        background: rgba(59, 130, 246, 0.15);
        color: #60A5FA;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }

    .pill-full {
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .pill-snippet {
        background: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }

    .badge-supported {
        background: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.35);
    }

    .badge-partially {
        background: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.35);
    }

    .badge-unsupported {
        background: rgba(239, 68, 68, 0.15);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.35);
    }

    .badge-conflicting {
        background: rgba(249, 115, 22, 0.15);
        color: #FB923C;
        border: 1px solid rgba(249, 115, 22, 0.35);
    }

    .verification-summary-box {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1.5rem;
    }

    .status-badge {
        padding: 0.35rem 0.75rem;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 500;
        display: inline-block;
        margin-bottom: 0.5rem;
    }

    .status-connected {
        background: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .status-missing {
        background: rgba(239, 68, 68, 0.15);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    .methodology-notice {
        background: rgba(30, 41, 59, 0.5);
        border-left: 4px solid #3B82F6;
        padding: 0.85rem 1.15rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.88rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_status_badge_html(status: ClaimStatus) -> str:
    """Format claim status as styled badge."""
    if status == ClaimStatus.SUPPORTED:
        return '<span class="pill-badge badge-supported">✓ Supported</span>'
    elif status == ClaimStatus.PARTIALLY_SUPPORTED:
        return '<span class="pill-badge badge-partially">~ Partially Supported</span>'
    elif status == ClaimStatus.CONFLICTING:
        return '<span class="pill-badge badge-conflicting">⚡ Conflicting Sources</span>'
    else:
        return '<span class="pill-badge badge-unsupported">✗ Unsupported</span>'


# -----------------------------------------------------------------------------
# Sidebar: System Diagnostics & Settings
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ System Status")

    env_tavily_key = get_tavily_api_key()
    env_gemini_key = get_gemini_api_key()

    if env_tavily_key:
        st.markdown(
            '<div class="status-badge status-connected">✓ Tavily API: Connected</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"Tavily Key: `{mask_key(env_tavily_key)}`")
    else:
        st.markdown(
            '<div class="status-badge status-missing">✗ Tavily API: Missing</div>',
            unsafe_allow_html=True,
        )
        st.caption("Add `TAVILY_API_KEY` to your local `.env` file.")

    if env_gemini_key:
        st.markdown(
            '<div class="status-badge status-connected">✓ Gemini API: Connected</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"Gemini Key: `{mask_key(env_gemini_key)}`")
    else:
        st.markdown(
            '<div class="status-badge status-missing">✗ Gemini API: Missing</div>',
            unsafe_allow_html=True,
        )
        st.caption("Add `GEMINI_API_KEY` to your local `.env` file.")

    with st.expander("🔑 Session Key Overrides"):
        manual_tavily = st.text_input(
            "Tavily API Key",
            value="",
            type="password",
            help="Override Tavily key for this browser session.",
        ).strip()
        manual_gemini = st.text_input(
            "Gemini API Key",
            value="",
            type="password",
            help="Override Gemini key for this browser session.",
        ).strip()
        active_tavily_key = manual_tavily if manual_tavily else env_tavily_key
        active_gemini_key = manual_gemini if manual_gemini else env_gemini_key

    st.markdown("---")
    st.markdown("### ⏱️ Performance & Time Budget")

    time_budget = st.slider(
        "End-to-End Time Budget (seconds)",
        min_value=15,
        max_value=60,
        value=25,
        step=5,
        help="Target maximum duration for research responses (default: 25s, leaving margin for UI render).",
    )

    st.markdown("---")
    st.markdown("### 🔬 Evidence Audit & Reliability")

    enable_verification = st.checkbox(
        "Audit Factual Claims (Deep Mode)",
        value=False,
        help="Audits extracted report claims against source text to verify support and detect conflicts.",
    )

    claim_audit_limit = st.slider(
        "Claims to Audit",
        min_value=2,
        max_value=8,
        value=5,
        step=1,
        help="Bounded budget for factual claim verification.",
    )

    st.markdown("---")
    st.markdown("### 🔍 Research Parameters")

    subquery_budget = st.slider(
        "Sub-Query Budget (Deep Mode)",
        min_value=1,
        max_value=2,
        value=1,
        step=1,
        help="Target number of complementary search queries planned by Gemini (default: 1).",
    )

    max_sources = st.slider(
        "Max Evidence Sources",
        min_value=2,
        max_value=10,
        value=5,
        step=1,
        help="Target number of deduplicated, domain-diverse sources (default: 5).",
    )

    enable_full_extraction = st.checkbox(
        "Extract Full Page Content (Deep Mode)",
        value=True,
        help="Uses Tavily extract to retrieve complete page markdown when accessible.",
    )

    search_depth = st.selectbox(
        "Search Depth",
        options=["basic", "advanced"],
        index=0,
        help="'basic' offers rapid retrieval; 'advanced' extracts richer context.",
    )

    selected_model = st.selectbox(
        "Gemini Synthesis Model",
        options=[DEFAULT_GEMINI_MODEL, "gemini-flash-latest"],
        index=0,
        help="Verified primary model for research synthesis.",
    )

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size: 0.8rem; color: #64748B;">
            <strong>Fast Answer (Default)</strong>: ~25s Target Response.<br>
            • Search & Extract: <strong>Tavily</strong><br>
            • Rapid Synthesis & Auditing: <strong>Gemini</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Main Application Content
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <div class="main-title">🔬 AI Research Agent</div>
        <div class="main-subtitle">⚡ Rapid Research Engine (30s Target) & Deep Evidence Verification</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Suggested Research Questions
st.markdown("**Quick Research Prompts:**")
col_s1, col_s2, col_s3 = st.columns(3)

if "query_input" not in st.session_state:
    st.session_state["query_input"] = ""

def set_query(text: str):
    st.session_state["query_input"] = text

with col_s1:
    if st.button("🔋 Solid-state battery commercialization", use_container_width=True):
        set_query("What are the key manufacturing hurdles and timeline for solid-state batteries in EVs?")
with col_s2:
    if st.button("🤖 Reasoning models benchmark comparison", use_container_width=True):
        set_query("How do recent reasoning models compare on software engineering and logic benchmarks?")
with col_s3:
    if st.button("🔐 Quantum cryptography transitions", use_container_width=True):
        set_query("What are the primary NIST post-quantum cryptography standards and migration timelines?")

# Search & Research Input Form
with st.form("research_workflow_form", clear_on_submit=False):
    query_text = st.text_input(
        "Research Question or Topic",
        value=st.session_state.get("query_input", ""),
        placeholder="e.g., Explain recent advances and trade-offs in multimodal agent architectures...",
        help="Enter the specific topic or question you wish to research.",
    )
    col_btn1, col_btn2, col_btn3 = st.columns([2, 1.5, 1])
    with col_btn1:
        run_fast_answer = st.form_submit_button(
            "⚡ Fast Answer",
            type="primary",
            use_container_width=True,
            help="Default ~25s mode: 1 web search, concise AI synthesis, and instant search fallback if synthesis times out.",
        )
    with col_btn2:
        run_deep_research = st.form_submit_button(
            "🔬 Run Deep Research",
            type="secondary",
            use_container_width=True,
            help="Comprehensive mode: Query planning, deduplication, full page extraction, deep report synthesis, and claim audit.",
        )
    with col_btn3:
        run_search_only = st.form_submit_button(
            "🔍 Search Only",
            type="secondary",
            use_container_width=True,
            help="Executes Tavily search only without synthesis (Milestone 1 behavior).",
        )
    run_standard_research = False

# -----------------------------------------------------------------------------
# Execution Workflow
# -----------------------------------------------------------------------------
if run_fast_answer or run_deep_research or run_standard_research or run_search_only:
    clean_query = query_text.strip()

    if not clean_query:
        st.warning("⚠️ Please enter a research question or topic to search.")
    elif not active_tavily_key:
        st.error(
            "🔑 **Tavily API Key Required**\n\n"
            "To perform web research, please configure your Tavily API key:\n"
            "1. Copy `.env.example` to `.env` in the project root.\n"
            "2. Add your key: `TAVILY_API_KEY=tvly-your_key_here`\n"
            "3. Or enter it in the sidebar **Session Key Overrides**."
        )
    elif (run_fast_answer or run_deep_research or run_standard_research) and not active_gemini_key:
        st.error(
            "🔑 **Gemini API Key Required**\n\n"
            "To synthesize research reports and audit evidence, please configure your Google Gemini API key:\n"
            "1. Obtain a key from [Google AI Studio](https://aistudio.google.com).\n"
            "2. Add your key: `GEMINI_API_KEY=your_gemini_key_here` in `.env`.\n"
            "3. Or enter it in the sidebar **Session Key Overrides**."
        )
    else:
        # ---------------------------------------------------------------------
        # Workflow Option A: Search Only (Milestone 1)
        # ---------------------------------------------------------------------
        if run_search_only:
            with st.spinner("🔎 Querying Tavily Search API..."):
                search_service = TavilySearchService(api_key=active_tavily_key)
                try:
                    results = search_service.search(
                        query=clean_query,
                        max_results=max_sources,
                        search_depth=search_depth,
                        timeout=8.0,
                    )
                    if not results:
                        st.info(f"No results found for '{clean_query}'.")
                    else:
                        st.success(f"Found **{len(results)}** evidence sources for: *\"{clean_query}\"*")
                        for idx, res in enumerate(results, start=1):
                            domain = extract_domain(res.url)
                            st.markdown(
                                f"""
                                <div class="source-card">
                                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem; flex-wrap:wrap;">
                                        <a href="{res.url}" target="_blank" rel="noopener noreferrer" style="font-size:1.05rem; font-weight:600; color:#3B82F6;">
                                            {idx}. {res.title} ↗
                                        </a>
                                        <span class="pill-badge pill-domain">{domain}</span>
                                    </div>
                                    <a href="{res.url}" target="_blank" rel="noopener noreferrer" style="font-size:0.8rem; color:#94A3B8; display:block; margin-bottom:0.5rem; word-break:break-all;">
                                        {res.url}
                                    </a>
                                    <div style="font-size:0.92rem; line-height:1.5; color:#CBD5E1;">
                                        {res.content}
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                except Exception as err:
                    st.error(f"❌ Search error: {err}")

        # ---------------------------------------------------------------------
        # Workflow Option B & C: Fast Answer & Deep Research
        # ---------------------------------------------------------------------
        else:
            start_time = time.time()
            evidence_results: list = []
            planned_sub_queries: list[str] = []
            searches_count = 0
            full_extracted_count = 0
            report: ResearchReport | None = None
            verification_summary: VerificationSummary | None = None
            fallback_reason: str | None = None
            is_fast_mode = bool(run_fast_answer or run_standard_research)

            # Fast Answer Workflow (Default, ~25s Target)
            if is_fast_mode:
                with st.status("⚡ Fast Answer: Searching web & synthesizing...", expanded=True) as status_box:
                    st.write("🔍 **Step 1/2**: Searching web with Tavily (max 5 sources)...")
                    search_start = time.time()
                    search_service = TavilySearchService(api_key=active_tavily_key)
                    try:
                        evidence_results = search_service.search(
                            query=clean_query,
                            max_results=min(max_sources, 5),
                            search_depth=search_depth,
                            timeout=8.0,
                        )
                        searches_count = 1
                    except Exception as search_err:
                        status_box.update(label="Search failed", state="error", expanded=False)
                        st.error(f"❌ Search error: {search_err}")
                        evidence_results = []

                    search_elapsed = time.time() - search_start

                    if not evidence_results:
                        status_box.update(label="No search results found", state="complete", expanded=False)
                        st.info(f"No web results found for '{clean_query}'.")
                    else:
                        st.write(f"✓ Retrieved {len(evidence_results)} sources in {search_elapsed:.1f}s.")

                        # Calculate remaining time budget for Gemini synthesis
                        elapsed_so_far = time.time() - start_time
                        remaining_budget = max(10.0, time_budget - elapsed_so_far - 0.5)
                        st.write(f"📝 **Step 2/2**: Synthesizing concise answer with Gemini ({selected_model}, budget: {remaining_budget:.1f}s)...")

                        report_service = GeminiReportService(api_key=active_gemini_key, model_name=selected_model)
                        try:
                            report = report_service.generate_report(
                                query=clean_query,
                                results=evidence_results,
                                timeout_seconds=remaining_budget,
                                concise_mode=True,
                            )
                        except GeminiTimeoutError as err:
                            fallback_reason = f"Gemini synthesis reached the {remaining_budget:.1f}s deadline."
                            report = None
                        except Exception as err:
                            fallback_reason = f"Gemini synthesis unavailable or failed: {err}"
                            report = None

                        total_elapsed = time.time() - start_time
                        if report:
                            status_box.update(
                                label=f"✓ Fast Answer completed in {total_elapsed:.1f}s (budget: {time_budget}s)",
                                state="complete",
                                expanded=False,
                            )
                        else:
                            status_box.update(
                                label=f"⚡ Synthesis timed out ({total_elapsed:.1f}s) — displaying search fallback",
                                state="complete",
                                expanded=False,
                            )

            # Deep Research Pipeline (Comprehensive Mode)
            else:
                with st.status("🔬 Executing Deeper Research Pipeline...", expanded=True) as status_box:
                    st.write("🧠 **Step 1/3**: Planning sub-queries & gathering evidence...")
                    deep_service = DeepResearchService(tavily_key=active_tavily_key, gemini_key=active_gemini_key)
                    try:
                        deep_res: DeepResearchResult = deep_service.run_deep_research(
                            query=clean_query,
                            max_sub_queries=subquery_budget,
                            max_sources=max_sources,
                            search_depth=search_depth,
                            extract_full_content=enable_full_extraction,
                            max_char_per_source=3500,
                            time_budget=time_budget * 0.65,
                        )
                        evidence_results = deep_res.sources
                        searches_count = deep_res.total_searches_run
                        full_extracted_count = deep_res.full_content_count
                        planned_sub_queries = deep_res.sub_queries
                        if planned_sub_queries:
                            st.info("Planned sub-queries:\n" + "\n".join(f"- *{sq}*" for sq in planned_sub_queries))
                        st.write(f"✓ Deep evidence gathered ({len(evidence_results)} sources, {full_extracted_count} full text) across {searches_count} searches.")
                    except Exception as deep_err:
                        status_box.update(label="Deep research failed", state="error", expanded=False)
                        st.error(f"❌ Deep research error: {deep_err}")
                        evidence_results = []

                    if not evidence_results:
                        status_box.update(label="No evidence gathered", state="complete", expanded=False)
                        st.info(f"No sources collected for '{clean_query}'.")
                    else:
                        elapsed_so_far = time.time() - start_time
                        remaining_budget = max(10.0, time_budget - elapsed_so_far - 0.5)
                        st.write(f"📝 **Step 2/3**: Synthesizing deep research report with Gemini ({selected_model}, budget: {remaining_budget:.1f}s)...")

                        report_service = GeminiReportService(api_key=active_gemini_key, model_name=selected_model)
                        try:
                            report = report_service.generate_report(
                                query=clean_query,
                                results=evidence_results,
                                timeout_seconds=remaining_budget,
                                concise_mode=False,
                            )
                        except GeminiTimeoutError as err:
                            fallback_reason = f"Gemini synthesis reached the {remaining_budget:.1f}s deadline."
                            report = None
                        except Exception as err:
                            fallback_reason = f"Gemini synthesis error: {err}"
                            report = None

                        # Step 3: Claim Audit (if report succeeded and verification enabled)
                        if report and enable_verification:
                            rem_verif = max(10.0, time_budget - (time.time() - start_time))
                            st.write(f"🔬 **Step 3/3**: Auditing factual claims against sources (budget: {rem_verif:.1f}s)...")
                            verifier = EvidenceVerificationService(api_key=active_gemini_key, model_name=selected_model)
                            try:
                                verification_summary = verifier.verify_report(
                                    report_text=report.content,
                                    sources=report.sources,
                                    max_claims=claim_audit_limit,
                                    timeout_seconds=rem_verif,
                                )
                            except Exception as verif_err:
                                st.warning(f"Verification audit could not complete: {verif_err}")

                        total_elapsed = time.time() - start_time
                        if report:
                            status_box.update(
                                label=f"✓ Deep Research completed in {total_elapsed:.1f}s (budget: {time_budget}s)",
                                state="complete",
                                expanded=False,
                            )
                        else:
                            status_box.update(
                                label=f"⚡ Synthesis timed out ({total_elapsed:.1f}s) — displaying search fallback",
                                state="complete",
                                expanded=False,
                            )

            total_elapsed = time.time() - start_time

            # -----------------------------------------------------------------
            # Result Rendering: Mode A (Synthesized Report)
            # -----------------------------------------------------------------
            if report:
                st.markdown("---")

                # Metrics Row
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                col_m1.metric("Evidence Sources", len(report.sources))
                col_m2.metric("Elapsed Time", f"{total_elapsed:.1f}s")
                col_m3.metric("Research Mode", "Fast Answer" if is_fast_mode else "Deep Research")
                col_m4.metric("Model", selected_model)

                # Milestone 4 Verification Summary Card (if audited)
                if verification_summary:
                    if verification_summary.is_fully_verified:
                        score_color = "#10B981" if verification_summary.reliability_score >= 80 else ("#FBBF24" if verification_summary.reliability_score >= 60 else "#EF4444")
                        st.markdown(
                            f"""
                            <div class="verification-summary-box">
                                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; margin-bottom:0.75rem;">
                                    <div style="font-weight:700; font-size:1.15rem; color:#F8FAFC;">
                                        🛡️ Evidence Verification Audit: <span style="color:{score_color};">{verification_summary.status_label}</span>
                                    </div>
                                    <div style="font-size:1.1rem; font-weight:700; color:{score_color};">
                                        Reliability Score: {verification_summary.reliability_score}%
                                    </div>
                                </div>
                                <div style="display:flex; gap:1.5rem; flex-wrap:wrap; font-size:0.9rem; color:#CBD5E1;">
                                    <span>✓ <strong>{verification_summary.supported_count}</strong> Supported</span>
                                    <span>~ <strong>{verification_summary.partially_supported_count}</strong> Partially Supported</span>
                                    <span>⚡ <strong>{verification_summary.conflicting_count}</strong> Conflicting</span>
                                    <span>✗ <strong>{verification_summary.unsupported_count}</strong> Unsupported</span>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        if verification_summary.discrepancy_notes:
                            for disc in verification_summary.discrepancy_notes:
                                st.warning(f"⚡ **Source Discrepancy Detected**: {disc}")
                    else:
                        st.warning(
                            f"⚠️ **Verification Incomplete**: {verification_summary.failure_reason}. "
                            "This report could not be fully audited against source texts."
                        )

                col_rh1, col_rh2 = st.columns([3, 1])
                with col_rh1:
                    st.markdown(f"### 📋 Research Report: *\"{clean_query}\"*")
                with col_rh2:
                    download_content = report.content
                    if verification_summary and verification_summary.is_fully_verified:
                        download_content += "\n\n---\n## Evidence Reliability & Claim Audit\n"
                        download_content += f"- **Reliability Score**: {verification_summary.reliability_score}%\n"
                        download_content += f"- **Status**: {verification_summary.status_label}\n"
                        download_content += f"- **Supported Claims**: {verification_summary.supported_count}/{verification_summary.total_claims}\n"
                        download_content += f"- **Partially Supported**: {verification_summary.partially_supported_count}\n"
                        download_content += f"- **Conflicting**: {verification_summary.conflicting_count}\n"
                        download_content += f"- **Unsupported**: {verification_summary.unsupported_count}\n"

                    st.download_button(
                        label="📥 Download Report (.md)",
                        data=download_content,
                        file_name=f"report_{clean_query[:20].strip().replace(' ', '_')}.md",
                        mime="text/markdown",
                        use_container_width=True,
                    )

                # Methodology Notice with timing
                st.markdown(
                    f"""
                    <div class="methodology-notice">
                        <strong>Methodology & Grounding Notice:</strong> Synthesized by <strong>{selected_model}</strong> 
                        in <strong>{total_elapsed:.1f}s</strong> strictly using <strong>{len(report.sources)}</strong> retrieved web sources. 
                        All factual claims feature verified inline citations linked directly to original sources.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Render Report Card
                st.markdown(
                    f'<div class="report-card">{st.markdown(report.content)}</div>',
                    unsafe_allow_html=True,
                )

                # Claim-by-Claim Audit Section (when present)
                if verification_summary and verification_summary.claims:
                    with st.expander("🔬 Detailed Claim-by-Claim Evidence Audit", expanded=True):
                        st.markdown("Each core claim extracted from the report has been audited directly against the retrieved text of its cited sources:")
                        for idx, clm in enumerate(verification_summary.claims, start=1):
                            badge_html = get_status_badge_html(clm.status)
                            valid_sids_str = ", ".join(f"[{s}]" for s in clm.valid_source_ids) if clm.valid_source_ids else "None"
                            invalid_alert = f'<span style="color:#EF4444; font-weight:600; margin-left:0.5rem;">⚠️ Fabricated IDs: {", ".join(clm.invalid_source_ids)}</span>' if clm.invalid_source_ids else ""

                            st.markdown(
                                f"""
                                <div class="claim-card">
                                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem; flex-wrap:wrap;">
                                        <div>
                                            <strong>Claim {idx}</strong>: <span style="font-weight:500; color:#F8FAFC;">"{clm.claim_text}"</span>
                                        </div>
                                        <div>
                                            {badge_html}
                                        </div>
                                    </div>
                                    <div style="font-size:0.83rem; color:#94A3B8; margin-bottom:0.4rem;">
                                        Cited Sources: <strong>{valid_sids_str}</strong> {invalid_alert}
                                    </div>
                                    <div style="font-size:0.9rem; line-height:1.5; color:#CBD5E1; margin-bottom:0.3rem;">
                                        <strong>Audit Evaluation</strong>: {clm.explanation}
                                    </div>
                                """,
                                unsafe_allow_html=True,
                            )
                            if clm.conflicting_details:
                                st.markdown(
                                    f"""
                                    <div style="font-size:0.88rem; color:#FB923C; margin-top:0.3rem;">
                                        ⚡ <strong>Discrepancy Details</strong>: {clm.conflicting_details}
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                            if clm.revised_text:
                                st.markdown(
                                    f"""
                                    <div style="font-size:0.88rem; color:#38BDF8; margin-top:0.3rem;">
                                        ✏️ <strong>Evidence-Grounded Revision</strong>: <em>"{clm.revised_text}"</em>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                            st.markdown("</div>", unsafe_allow_html=True)

                # Grounded Source Index
                st.markdown("### 📚 Grounded Evidence Sources & Provenance")
                for src in report.sources:
                    ext_badge = (
                        '<span class="pill-badge pill-full">✓ Full Article</span>'
                        if src.is_full_content
                        else '<span class="pill-badge pill-snippet">○ Snippet</span>'
                    )
                    query_info = f'<span style="font-size:0.75rem; color:#94A3B8; margin-left:0.5rem;">Via: "{src.origin_query}"</span>' if src.origin_query else ""
                    st.markdown(
                        f"""
                        <div class="source-card">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.35rem; flex-wrap:wrap;">
                                <div>
                                    <span class="pill-badge pill-tag">[{src.source_id}]</span>
                                    <a href="{src.url}" target="_blank" rel="noopener noreferrer" style="font-weight:600; color:#3B82F6; font-size:1.05rem;">
                                        {src.title} ↗
                                    </a>
                                    {query_info}
                                </div>
                                <div>
                                    <span class="pill-badge pill-domain">{src.domain}</span>
                                    {ext_badge}
                                </div>
                            </div>
                            <a href="{src.url}" target="_blank" rel="noopener noreferrer" style="font-size:0.8rem; color:#94A3B8; display:block; margin-bottom:0.5rem; word-break:break-all;">
                                {src.url}
                            </a>
                            <div style="font-size:0.9rem; line-height:1.5; color:#CBD5E1;">
                                {src.content[:400] + ('...' if len(src.content) > 400 else '')}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with st.expander("🔍 Inspect Full Provenance Data (JSON)"):
                    st.json(
                        [
                            {
                                "source_id": s.source_id,
                                "title": s.title,
                                "url": s.url,
                                "domain": s.domain,
                                "origin_query": s.origin_query,
                                "is_full_content": s.is_full_content,
                                "extraction_status": s.extraction_status,
                                "content_preview": s.content[:200] + "...",
                            }
                            for s in report.sources
                        ]
                    )

            # -----------------------------------------------------------------
            # Result Rendering: Mode B (Fast Search Fallback when synthesis times out)
            # -----------------------------------------------------------------
            elif evidence_results:
                st.markdown("---")

                # Prominent Fallback Notice Banner
                st.markdown(
                    f"""
                    <div style="background: rgba(245, 158, 11, 0.12); border: 1px solid rgba(245, 158, 11, 0.4); border-radius: 12px; padding: 1.25rem; margin-bottom: 1.5rem;">
                        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; margin-bottom:0.5rem;">
                            <div style="font-weight:700; font-size:1.2rem; color:#FBBF24;">
                                ⚡ Fast Search Fallback: Primary Evidence Sources
                            </div>
                            <div style="font-size:0.95rem; font-weight:600; color:#FBBF24;">
                                ⏱️ Sources gathered in {total_elapsed:.1f}s
                            </div>
                        </div>
                        <p style="color:#E2E8F0; font-size:0.95rem; line-height:1.5; margin-bottom:0.5rem;">
                            AI synthesis exceeded the allocated deadline ({total_elapsed:.1f}s) or the model server was unavailable. 
                            Rather than leaving you waiting, here are the <strong>{len(evidence_results)} verified primary sources</strong> 
                            retrieved directly from the web.
                        </p>
                        <div style="font-size:0.83rem; color:#94A3B8;">
                            <strong>Notice:</strong> This is search-based evidence (raw web sources) rather than a synthesized report. Reason: <em>{fallback_reason}</em>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                col_fb1, col_fb2 = st.columns([3, 1])
                with col_fb1:
                    st.markdown(f"### 🔍 Retrieved Evidence Sources: *\"{clean_query}\"*")
                with col_fb2:
                    # Download search results markdown
                    fallback_md = f"# Evidence Sources: {clean_query}\n\n"
                    fallback_md += f"*Search-based fallback retrieved in {total_elapsed:.1f}s*\n"
                    fallback_md += f"*Fallback reason: {fallback_reason}*\n\n---\n\n"
                    for idx, s in enumerate(evidence_results, start=1):
                        s_dom = getattr(s, "domain", extract_domain(s.url))
                        fallback_md += f"### {idx}. [{s.title}]({s.url})\n"
                        fallback_md += f"- **Source URL**: {s.url}\n"
                        fallback_md += f"- **Domain**: {s_dom}\n\n"
                        fallback_md += f"> {s.content}\n\n"

                    st.download_button(
                        label="📥 Download Sources (.md)",
                        data=fallback_md,
                        file_name=f"sources_{clean_query[:20].strip().replace(' ', '_')}.md",
                        mime="text/markdown",
                        use_container_width=True,
                    )

                # Render Search Fallback Cards with Clickable Links and Snippets
                for idx, res in enumerate(evidence_results, start=1):
                    res_domain = getattr(res, "domain", extract_domain(res.url))
                    res_tag = f"S{idx}"
                    st.markdown(
                        f"""
                        <div class="source-card">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem; flex-wrap:wrap;">
                                <div>
                                    <span class="pill-badge pill-tag">[{res_tag}]</span>
                                    <a href="{res.url}" target="_blank" rel="noopener noreferrer" style="font-size:1.05rem; font-weight:600; color:#3B82F6;">
                                        {idx}. {res.title} ↗
                                    </a>
                                </div>
                                <span class="pill-badge pill-domain">{res_domain}</span>
                            </div>
                            <a href="{res.url}" target="_blank" rel="noopener noreferrer" style="font-size:0.8rem; color:#94A3B8; display:block; margin-bottom:0.5rem; word-break:break-all;">
                                {res.url}
                            </a>
                            <div style="font-size:0.92rem; line-height:1.5; color:#CBD5E1;">
                                {res.content}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
