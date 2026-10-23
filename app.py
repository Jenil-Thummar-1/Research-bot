import time
import urllib.parse
from pathlib import Path
import streamlit as st

from src.config import (
    get_tavily_api_key,
    get_gemini_api_key,
    get_gemini_api_keys,
    get_groq_api_key,
    get_ollama_base_url,
    mask_key,
)
from src.services.search_service import (
    TavilySearchService,
    SearchResult,
    MissingAPIKeyError,
    InvalidAPIKeyError,
    SearchRateLimitError,
    SearchNetworkError,
    SearchQueryError,
    SearchServiceError,
    duckduckgo_search,
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
    bind_citations_to_urls,
)
from src.services.verification_service import (
    EvidenceVerificationService,
    VerificationSummary,
    VerifiedClaim,
    ClaimStatus,
)
from src.services.chat_history_service import (
    ChatHistoryService,
    Conversation,
    ChatMessage,
)
from src.services.chat_service import (
    ConversationalChatService,
    ChatResponse,
    StreamResult,
    ChatServiceError,
    ChatTimeoutError,
    MissingChatKeyError,
    ChatAuthError,
    ChatQuotaError,
    is_ollama_running,
    get_available_ollama_models,
)

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Conversational Assistant | Chat & Research Agent",
    page_icon="🤖",
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

    .chat-hero {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem;
        color: #94A3B8;
    }
    .chat-hero h2 {
        color: #F8FAFC;
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .chat-hero p {
        font-size: 1.05rem;
        max-width: 600px;
        margin: 0 auto 1.5rem;
    }

    .chat-meta-bar {
        font-size: 0.78rem;
        color: #64748B;
        margin-top: 0.35rem;
        display: flex;
        gap: 1rem;
        align-items: center;
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
# Initialize Persistence & Session State
# -----------------------------------------------------------------------------
history_service = ChatHistoryService()

# Ensure active conversation ID is set
if "active_conversation_id" not in st.session_state:
    existing_convs = history_service.list_conversations()
    if existing_convs:
        st.session_state["active_conversation_id"] = existing_convs[0].id
    else:
        new_conv = history_service.create_conversation("New Chat")
        st.session_state["active_conversation_id"] = new_conv.id

# Ensure active conversation still exists in database
current_conv = history_service.get_conversation(st.session_state["active_conversation_id"])
if current_conv is None:
    existing_convs = history_service.list_conversations()
    if existing_convs:
        st.session_state["active_conversation_id"] = existing_convs[0].id
    else:
        new_conv = history_service.create_conversation("New Chat")
        st.session_state["active_conversation_id"] = new_conv.id
    current_conv = history_service.get_conversation(st.session_state["active_conversation_id"])

# -----------------------------------------------------------------------------
# Sidebar: Navigation, Conversations & System Status
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 💬 Conversations")

    # + New Chat Button
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        new_c = history_service.create_conversation("New Chat")
        st.session_state["active_conversation_id"] = new_c.id
        st.rerun()

    # List stored conversations
    all_conversations = history_service.list_conversations()
    if all_conversations:
        st.caption(f"Stored Chats ({len(all_conversations)})")
        for conv in all_conversations:
            is_active = conv.id == st.session_state.get("active_conversation_id")
            label_prefix = "👉 " if is_active else "💬 "
            btn_label = f"{label_prefix}{conv.title}"

            col_item1, col_item2 = st.columns([5, 1])
            with col_item1:
                if st.button(
                    btn_label,
                    key=f"conv_select_{conv.id}",
                    use_container_width=True,
                    help=f"Created: {time.strftime('%b %d, %H:%M', time.localtime(conv.created_at))}",
                ):
                    st.session_state["active_conversation_id"] = conv.id
                    st.rerun()
            with col_item2:
                # Delete conversation button
                if st.button("🗑️", key=f"del_{conv.id}", help="Delete chat"):
                    history_service.delete_conversation(conv.id)
                    remaining = history_service.list_conversations()
                    if remaining:
                        st.session_state["active_conversation_id"] = remaining[0].id
                    else:
                        new_created = history_service.create_conversation("New Chat")
                        st.session_state["active_conversation_id"] = new_created.id
                    st.rerun()

    # Rename active conversation
    if current_conv:
        with st.expander("✏️ Rename Active Chat", expanded=False):
            new_title_val = st.text_input("Title", value=current_conv.title, key="rename_title_input")
            if st.button("Save Title", use_container_width=True):
                if new_title_val.strip():
                    history_service.update_conversation_title(current_conv.id, new_title_val.strip())
                    st.rerun()

    st.markdown("---")
    st.markdown("### 🧠 AI Engine & Provider")

    provider_label = st.radio(
        "AI Backend Provider",
        options=[
            "Google Gemini (Key Pool / Auto-Rotate)",
            "Ollama (100% Free / Local Offline - 0 Keys)",
            "Groq Cloud (14,400 Free Req/Day)",
        ],
        index=0,
        help="Select the AI backend. Ollama runs 100% locally with zero API keys and no rate limits!",
    )

    if "Ollama" in provider_label:
        selected_provider = "ollama"
        ollama_online = is_ollama_running()
        if ollama_online:
            detected_models = get_available_ollama_models()
            model_options = detected_models if detected_models else ["gemma:2b", "gemma3:4b"]
            selected_model = st.selectbox("Ollama Local Model", options=model_options, index=0)
            st.success(f"🟢 Ollama Online ({len(model_options)} local models detected) — Unlimited Free Chat!")
        else:
            st.warning("⚠️ Local Ollama is offline. Run `ollama serve` in terminal.")
            selected_model = "gemma:2b"
    elif "Groq" in provider_label:
        selected_provider = "groq"
        selected_model = st.selectbox(
            "Groq Cloud Model",
            options=["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
            index=0,
        )
    else:
        selected_provider = "gemini"
        selected_model = st.selectbox(
            "Gemini Model",
            options=["gemini-3.8-flash", "gemini-3.5-flash"],
            index=0,
        )

    st.markdown("---")
    st.markdown("### 🌐 Chat Options")

    enable_web_search = st.checkbox(
        "🌐 Search the Web (Tavily)",
        value=False,
        help="When enabled, live web evidence is gathered and cited via Tavily. Default OFF for fast direct conversation.",
    )

    search_engine_choice = st.selectbox(
        "Search Engine Source",
        options=[
            "Auto (Tavily with DuckDuckGo Fallback)",
            "DuckDuckGo (100% Free / Zero API Key)",
            "Tavily (Direct)",
        ],
        index=0,
        help="DuckDuckGo performs free web search with zero API keys and zero limits.",
    )
    search_provider_map = {
        "Auto (Tavily with DuckDuckGo Fallback)": "auto",
        "DuckDuckGo (100% Free / Zero API Key)": "duckduckgo",
        "Tavily (Direct)": "tavily",
    }
    selected_search_provider = search_provider_map[search_engine_choice]

    context_window = st.slider(
        "Context Window (messages)",
        min_value=4,
        max_value=20,
        value=10,
        step=2,
        help="Number of recent conversational turns forwarded to AI.",
    )

    chat_timeout = st.slider(
        "Chat Timeout (seconds)",
        min_value=10,
        max_value=60,
        value=30,
        step=5,
        help="Bounded timeout budget for chat generation.",
    )

    # Export Chat Markdown
    current_messages = history_service.get_messages(st.session_state["active_conversation_id"])
    if current_messages:
        md_export = f"# {current_conv.title if current_conv else 'Chat Export'}\n\n"
        for m in current_messages:
            role_label = "**User**" if m.role == "user" else "**Assistant**"
            md_export += f"### {role_label}\n{m.content}\n\n"
        st.download_button(
            label="📥 Export Chat (.md)",
            data=md_export,
            file_name=f"chat_{current_conv.title[:20].strip().replace(' ', '_') if current_conv else 'export'}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    st.markdown("---")
    st.markdown("### ⚙️ System Status")

    env_tavily_key = get_tavily_api_key()
    env_gemini_key = get_gemini_api_key()
    gemini_key_pool = get_gemini_api_keys()
    env_groq_key = get_groq_api_key()
    ollama_is_up = is_ollama_running()

    if env_gemini_key:
        st.markdown(
            '<div class="status-badge status-connected">✓ Gemini API: Connected</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"Gemini Key: `{mask_key(env_gemini_key)}` ({len(gemini_key_pool)} key(s) in pool)")
    else:
        st.markdown(
            '<div class="status-badge status-missing">✗ Gemini API: Missing</div>',
            unsafe_allow_html=True,
        )
        st.caption("Add `GEMINI_API_KEY` to your local `.env` file.")

    if ollama_is_up:
        st.markdown(
            '<div class="status-badge status-connected">✓ Ollama: Running (Local Offline)</div>',
            unsafe_allow_html=True,
        )
        st.caption("Port 11434 • 0 API keys • Unlimited chats")
    else:
        st.markdown(
            '<div class="status-badge status-missing">✗ Ollama: Offline</div>',
            unsafe_allow_html=True,
        )
        st.caption("Run `ollama serve` in terminal for free offline AI.")

    if env_groq_key:
        st.markdown(
            '<div class="status-badge status-connected">✓ Groq Cloud: Connected</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"Groq Key: `{mask_key(env_groq_key)}` (14,400 free req/day)")
    else:
        st.caption("Groq API: Optional (Free tier available at console.groq.com)")

    if env_tavily_key:
        st.markdown(
            '<div class="status-badge status-connected">✓ Tavily API: Connected</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"Tavily Key: `{mask_key(env_tavily_key)}`")
    else:
        st.markdown(
            '<div class="status-badge status-connected">✓ DuckDuckGo: Ready (Keyless)</div>',
            unsafe_allow_html=True,
        )
        st.caption("Free search fallback active • 0 API keys required")

    with st.expander("🔑 Session Key Overrides"):
        manual_tavily = st.text_input(
            "Tavily API Key",
            value="",
            type="password",
            help="Override Tavily key for this browser session.",
            key="sb_tavily_override",
        ).strip()
        manual_gemini = st.text_input(
            "Gemini API Key",
            value="",
            type="password",
            help="Override Gemini key for this browser session.",
            key="sb_gemini_override",
        ).strip()
        manual_groq = st.text_input(
            "Groq API Key",
            value="",
            type="password",
            help="Override Groq key for this browser session.",
            key="sb_groq_override",
        ).strip()
        active_tavily_key = manual_tavily if manual_tavily else env_tavily_key
        active_gemini_key = manual_gemini if manual_gemini else env_gemini_key
        active_groq_key = manual_groq if manual_groq else env_groq_key


# -----------------------------------------------------------------------------
# Main Application Header & Tabs
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <div class="main-title">🤖 AI Assistant & Research Agent</div>
        <div class="main-subtitle">General-Purpose Conversational AI (ChatGPT Style) • Multi-Turn Q&A, Coding, Writing, & Optional Web Research</div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_chat, tab_research = st.tabs(["💬 Conversational Chat", "🔬 Deep Research & Evidence Studio"])

# -----------------------------------------------------------------------------
# TAB 1: Conversational Chat (Default Primary Interface)
# -----------------------------------------------------------------------------
with tab_chat:
    active_conv_id = st.session_state["active_conversation_id"]
    active_conv = history_service.get_conversation(active_conv_id)

    # Conversation Top Bar
    col_tb1, col_tb2 = st.columns([3, 1])
    with col_tb1:
        st.markdown(f"#### 💬 {active_conv.title if active_conv else 'Chat'}")
    with col_tb2:
        if enable_web_search:
            st.markdown('<span class="pill-badge pill-domain">🌐 Web Search Enabled</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="pill-badge pill-full">⚡ Direct Chat Mode</span>', unsafe_allow_html=True)

    # Retrieve all messages for the current conversation
    chat_messages = history_service.get_messages(active_conv_id)

    # Empty State: Suggestion Chips
    if not chat_messages:
        st.markdown(
            """
            <div class="chat-hero">
                <h2>How can I help you today?</h2>
                <p>Ask anything: everyday explanations, coding assistance, creative writing, step-by-step plans, or enable web search for live facts.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("**Suggested Prompts:**")
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        with col_p1:
            if st.button("💡 Explain Python Decorators", use_container_width=True):
                st.session_state["pending_chat_prompt"] = "Explain Python decorators with clear code examples and real-world use cases."
                st.rerun()
        with col_p2:
            if st.button("📝 Draft Project Proposal", use_container_width=True):
                st.session_state["pending_chat_prompt"] = "Help me draft a concise, compelling project proposal for an AI automation initiative."
                st.rerun()
        with col_p3:
            if st.button("🇮🇳 Hinglish Roadmap", use_container_width=True):
                st.session_state["pending_chat_prompt"] = "Mujhe Full-Stack Web Development sikhna hai, complete roadmap Hinglish me batao."
                st.rerun()
        with col_p4:
            if st.button("🌐 Web Search: AI News", use_container_width=True):
                st.session_state["pending_chat_prompt"] = "Search the web for the latest developments in generative AI models this month."
                st.rerun()

    # Render Conversation Messages
    for idx, msg in enumerate(chat_messages):
        with st.chat_message(msg.role):
            st.markdown(msg.content)

            # If web sources were recorded with this message, display them cleanly
            msg_sources = msg.metadata.get("sources", [])
            if msg_sources:
                with st.expander(f"🌐 Grounded in {len(msg_sources)} Web Evidence Sources", expanded=False):
                    for src in msg_sources:
                        s_dom = src.get("domain", "")
                        st.markdown(
                            f"""
                            <div class="source-card">
                                <div style="display:flex; justify-content:space-between; align-items:center;">
                                    <a href="{src.get('url')}" target="_blank" rel="noopener noreferrer" style="font-weight:600; color:#3B82F6;">
                                        [{src.get('source_id')}] {src.get('title')} ↗
                                    </a>
                                    <span class="pill-badge pill-domain">{s_dom}</span>
                                </div>
                                <div style="font-size:0.85rem; color:#94A3B8; margin-top:0.3rem;">
                                    {src.get('content', '')[:250]}...
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            # Metadata footer for assistant messages
            if msg.role == "assistant":
                elapsed = msg.metadata.get("elapsed_seconds")
                ttft = msg.metadata.get("time_to_first_token")
                model_used = msg.metadata.get("model_used", DEFAULT_GEMINI_MODEL)
                was_stopped = msg.metadata.get("stopped_early", False)

                p_used = msg.metadata.get("provider_used", "gemini").capitalize()
                meta_str = f"Engine: **{p_used}** • Model: `{model_used}`"
                if ttft:
                    meta_str += f" • TTFT: **{ttft:.2f}s**"
                if elapsed:
                    meta_str += f" • Total time: **{elapsed:.1f}s**"
                if msg.metadata.get("web_search_used"):
                    meta_str += " • 🌐 *Web search cited*"
                if was_stopped:
                    meta_str += " • ⏹️ *Stopped early*"

                col_m1, col_m2, col_m3 = st.columns([3, 1, 1])
                with col_m1:
                    st.caption(meta_str)
                with col_m2:
                    with st.popover("📋 Copy", help="View & copy raw text"):
                        st.code(msg.content, language=None)
                with col_m3:
                    # Retry response button for the latest assistant message
                    if idx == len(chat_messages) - 1:
                        if st.button("🔄 Retry", key=f"retry_msg_{msg.id}", help="Regenerate this response"):
                            history_service.delete_last_assistant_message(active_conv_id)
                            # Get the user prompt that triggered it
                            updated_msgs = history_service.get_messages(active_conv_id)
                            if updated_msgs and updated_msgs[-1].role == "user":
                                st.session_state["pending_chat_prompt"] = updated_msgs[-1].content
                                # Pop the user message so it gets re-added and re-executed cleanly
                                history_service.delete_last_assistant_message(active_conv_id)
                            st.rerun()

    # Chat Input Box (Enter to send, Shift+Enter for newline)
    is_busy = st.session_state.get("chat_is_generating", False)
    user_input = st.chat_input(
        "Message your AI assistant... (Enter to send, Shift+Enter for newline)",
        disabled=is_busy,
    )

    # Check for pending prompt from suggestion chips or retry
    if "pending_chat_prompt" in st.session_state and st.session_state["pending_chat_prompt"]:
        user_input = st.session_state.pop("pending_chat_prompt")

    # Handle incoming user message
    if user_input and user_input.strip():
        clean_user_prompt = user_input.strip()

        # If active conversation is named "New Chat", automatically rename it to a snippet of the prompt
        if active_conv and active_conv.title == "New Chat":
            clean_title = clean_user_prompt[:35].strip() + ("..." if len(clean_user_prompt) > 35 else "")
            history_service.update_conversation_title(active_conv_id, clean_title)

        # Store user message in SQLite
        history_service.add_message(active_conv_id, "user", clean_user_prompt)

        # Display immediately in UI
        with st.chat_message("user"):
            st.markdown(clean_user_prompt)

        # Validate selected provider readiness
        provider_ready = True
        if selected_provider == "gemini" and not active_gemini_key:
            provider_ready = False
            st.error(
                "🔑 **Gemini API Key Required**\n\n"
                "Please configure your `GEMINI_API_KEY` in `.env` or in the sidebar **Session Key Overrides**,\n"
                "or switch to **Ollama (100% Free / Local Offline - 0 Keys)** in the sidebar!"
            )
        elif selected_provider == "groq" and not active_groq_key:
            provider_ready = False
            st.error(
                "🔑 **Groq API Key Required**\n\n"
                "Please configure `GROQ_API_KEY` in `.env` or sidebar overrides "
                "(get a free key at https://console.groq.com for 14,400 free requests/day), "
                "or switch to **Ollama** in the sidebar!"
            )
        elif selected_provider == "ollama" and not is_ollama_running():
            provider_ready = False
            st.error(
                "💻 **Local Ollama is Offline**\n\n"
                "Start Ollama in your terminal using:\n```powershell\nollama serve\n```\n"
                "Once running, you can chat with zero API keys and zero limits!"
            )

        if provider_ready:
            with st.chat_message("assistant"):
                search_notice = None
                if enable_web_search:
                    s_label = "DuckDuckGo (Keyless)" if selected_search_provider == "duckduckgo" else "the Web"
                    search_notice = st.status(f"🌐 Searching {s_label} for live evidence...", expanded=False)

                stream_result = StreamResult()
                chat_service = ConversationalChatService(
                    gemini_api_key=active_gemini_key,
                    gemini_api_keys=gemini_key_pool,
                    tavily_api_key=active_tavily_key,
                    groq_api_key=active_groq_key,
                    provider=selected_provider,
                    search_provider=selected_search_provider,
                    model_name=selected_model,
                )

                past_history = history_service.get_messages(active_conv_id, limit=context_window)
                prior_turns = [m for m in past_history[:-1]] if len(past_history) > 1 else []

                col_stop_tray, _ = st.columns([1, 4])
                with col_stop_tray:
                    stop_clicked = st.button("⏹️ Stop Generation", key="btn_stop_stream")

                st.session_state["chat_is_generating"] = True

                try:
                    stream_gen = chat_service.stream_message(
                        user_prompt=clean_user_prompt,
                        conversation_history=prior_turns,
                        use_web_search=enable_web_search,
                        max_context_turns=context_window,
                        timeout_seconds=float(chat_timeout),
                        result_holder=stream_result,
                        should_stop_callback=lambda: bool(st.session_state.get("btn_stop_stream", False)),
                        provider=selected_provider,
                        model_name=selected_model,
                    )

                    # Real-time incremental rendering
                    full_text = st.write_stream(stream_gen)

                    if search_notice:
                        search_notice.update(label=f"✓ Gathered {len(stream_result.sources)} web sources", state="complete")

                    final_content = stream_result.full_content or full_text

                    # Package metadata for SQLite persistence
                    sources_meta = [
                        {
                            "source_id": s.source_id,
                            "title": s.title,
                            "url": s.url,
                            "domain": s.domain,
                            "content": s.content[:300],
                        }
                        for s in stream_result.sources
                    ]
                    resp_meta = {
                        "sources": sources_meta,
                        "web_search_used": stream_result.web_search_used,
                        "search_query": stream_result.search_query,
                        "model_used": stream_result.model_used,
                        "provider_used": stream_result.provider_used,
                        "time_to_first_token": stream_result.time_to_first_token,
                        "elapsed_seconds": stream_result.total_elapsed_seconds,
                        "stopped_early": stream_result.stopped_early,
                    }

                    if final_content and final_content.strip():
                        history_service.add_message(
                            conversation_id=active_conv_id,
                            role="assistant",
                            content=final_content,
                            metadata=resp_meta,
                        )

                    st.session_state["chat_is_generating"] = False
                    st.rerun()

                except MissingChatKeyError as err:
                    st.session_state["chat_is_generating"] = False
                    st.error(f"🔑 {err}")
                except ChatAuthError as err:
                    st.session_state["chat_is_generating"] = False
                    st.error(f"🚫 {err}")
                except ChatQuotaError as err:
                    st.session_state["chat_is_generating"] = False
                    st.error(f"⏳ {err}")
                except ChatTimeoutError as err:
                    st.session_state["chat_is_generating"] = False
                    st.error(f"⏱️ {err} You can click **🔄 Retry** to try again.")
                except ChatServiceError as err:
                    st.session_state["chat_is_generating"] = False
                    st.error(f"❌ Chat Generation Error: {err}")
                except Exception as err:
                    st.session_state["chat_is_generating"] = False
                    st.error(f"❌ Unexpected Error: {err}")


# -----------------------------------------------------------------------------
# TAB 2: Deep Research & Evidence Studio (Preserved Feature Set)
# -----------------------------------------------------------------------------
with tab_research:
    st.markdown("### 🔬 Deep Research & Evidence Studio (AI Research Agent)")
    st.markdown(
        "Conduct thorough multi-query web research, full-page extraction, and automated factual claim verification."
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

    # Research Input Form
    with st.form("research_workflow_form", clear_on_submit=False):
        query_text = st.text_input(
            "Research Question or Topic",
            value=st.session_state.get("query_input", ""),
            placeholder="e.g., Explain recent advances and trade-offs in multimodal agent architectures...",
            help="Enter the specific topic or question you wish to research.",
        )

        col_r_opt1, col_r_opt2, col_r_opt3 = st.columns(3)
        with col_r_opt1:
            enable_verification = st.checkbox(
                "Audit Factual Claims (Deep Mode)",
                value=False,
                help="Audits extracted report claims against source text to verify support and detect conflicts.",
            )
        with col_r_opt2:
            enable_full_extraction = st.checkbox(
                "Extract Full Page Content (Deep Mode)",
                value=True,
                help="Uses Tavily extract to retrieve complete page markdown when accessible.",
            )
        with col_r_opt3:
            search_depth = st.selectbox(
                "Search Depth",
                options=["basic", "advanced"],
                index=0,
                help="'basic' offers rapid retrieval; 'advanced' extracts richer context.",
            )

        col_btn1, col_btn2, col_btn3 = st.columns([2, 1.5, 1])
        with col_btn1:
            run_fast_answer = st.form_submit_button(
                "⚡ Fast Answer",
                type="primary",
                use_container_width=True,
                help="Fast ~25s mode: 1 web search, concise AI synthesis, and instant search fallback if synthesis times out.",
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

    # Execution Workflow for Research Studio
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
            # Workflow Option A: Search Only
            if run_search_only:
                with st.spinner("🔎 Querying Tavily Search API..."):
                    search_service = TavilySearchService(api_key=active_tavily_key)
                    try:
                        results = search_service.search(
                            query=clean_query,
                            max_results=5,
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

            # Workflow Option B & C: Fast Answer & Deep Research
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
                                max_results=5,
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
                            status_box.update(label="No evidence found", state="error", expanded=False)
                            st.warning(f"No web sources found for '{clean_query}'.")
                        else:
                            st.write(f"✓ Found {len(evidence_results)} sources in {search_elapsed:.1f}s.")
                            st.write("🤖 **Step 2/2**: Synthesizing concise answer with Gemini...")

                            report_service = GeminiReportService(
                                api_key=active_gemini_key,
                                model_name=DEFAULT_GEMINI_MODEL,
                            )

                            try:
                                report = report_service.generate_report(
                                    query=clean_query,
                                    results=evidence_results,
                                    timeout_seconds=14.0,
                                    concise_mode=True,
                                )
                                status_box.update(
                                    label=f"✓ Fast Answer complete in {time.time() - start_time:.1f}s!",
                                    state="complete",
                                    expanded=False,
                                )
                            except GeminiTimeoutError as timeout_err:
                                fallback_reason = f"Synthesis timed out: {timeout_err}"
                                status_box.update(
                                    label="⚡ Fast Search Fallback triggered (synthesis timed out)",
                                    state="complete",
                                    expanded=False,
                                )
                            except Exception as syn_err:
                                fallback_reason = f"Synthesis error: {syn_err}"
                                status_box.update(
                                    label="⚡ Fast Search Fallback triggered",
                                    state="complete",
                                    expanded=False,
                                )

                # Deep Research Workflow (Comprehensive)
                else:
                    with st.status("🔬 Executing Deep Research Pipeline...", expanded=True) as status_box:
                        deep_research_service = DeepResearchService(
                            tavily_key=active_tavily_key,
                            gemini_key=active_gemini_key,
                        )

                        st.write("🧠 **Phase 1/4**: Generating targeted sub-queries...")
                        research_result = deep_research_service.run_deep_research(
                            query=clean_query,
                            max_sub_queries=1,
                            max_sources=5,
                            extract_full_content=enable_full_extraction,
                            search_depth=search_depth,
                        )

                        evidence_results = research_result.sources
                        planned_sub_queries = research_result.sub_queries
                        searches_count = research_result.total_searches_run
                        full_extracted_count = research_result.full_content_count

                        st.write(f"✓ Collected {len(evidence_results)} deduplicated sources across {searches_count} queries.")
                        st.write("📝 **Phase 2/4**: Synthesizing comprehensive research report...")

                        report_service = GeminiReportService(
                            api_key=active_gemini_key,
                            model_name=DEFAULT_GEMINI_MODEL,
                        )

                        try:
                            report = report_service.generate_report(
                                query=clean_query,
                                results=evidence_results,
                                timeout_seconds=20.0,
                                concise_mode=False,
                            )
                        except Exception as syn_err:
                            fallback_reason = f"Deep synthesis failure: {syn_err}"
                            report = None

                        if report and enable_verification:
                            st.write("🔬 **Phase 3/4**: Auditing factual claims against source content...")
                            verifier = EvidenceVerificationService(
                                api_key=active_gemini_key,
                                model_name=DEFAULT_GEMINI_MODEL,
                            )
                            try:
                                verification_summary = verifier.verify_report(
                                    report_text=report.content,
                                    sources=report.sources,
                                    max_claims=5,
                                )
                            except Exception as v_err:
                                st.warning(f"Claim verification skipped: {v_err}")

                        st.write("✅ **Phase 4/4**: Complete!")
                        status_box.update(
                            label=f"✓ Deep Research completed in {time.time() - start_time:.1f}s!",
                            state="complete",
                            expanded=False,
                        )

                total_elapsed = time.time() - start_time

                # Render Synthesized Report
                if report:
                    st.markdown("---")
                    col_h1, col_h2 = st.columns([3, 1])
                    with col_h1:
                        st.markdown(f"## 📋 Research Report: *\"{clean_query}\"*")
                    with col_h2:
                        st.download_button(
                            label="📥 Download Report (.md)",
                            data=report.content,
                            file_name=f"report_{clean_query[:20].strip().replace(' ', '_')}.md",
                            mime="text/markdown",
                            use_container_width=True,
                        )

                    # Performance Metrics
                    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                    with mcol1:
                        st.metric("Total Elapsed Time", f"{total_elapsed:.1f}s")
                    with mcol2:
                        st.metric("Evidence Sources", len(report.sources))
                    with mcol3:
                        st.metric("Searches Executed", searches_count)
                    with mcol4:
                        st.metric("Model Used", report.model_used)

                    # Methodology Notice
                    mode_desc = "Fast Answer Mode (~25s budget, single-search concise synthesis)" if is_fast_mode else "Deep Research Mode (full query planning, deduplication, full page extraction)"
                    st.markdown(
                        f"""
                        <div class="methodology-notice">
                            <strong>Methodology & Grounding Notice:</strong> Generated using <strong>{mode_desc}</strong>. 
                            All factual assertions are grounded strictly on the verified sources indexed below. Inline citations like <code>[S1]</code> or <code>[S2]</code> link directly to primary URLs.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Render Verification Summary Box if available
                    if verification_summary:
                        st.markdown(
                            f"""
                            <div class="verification-summary-box">
                                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; margin-bottom:0.75rem;">
                                    <div style="font-weight:700; font-size:1.15rem; color:#F8FAFC;">
                                        🔬 Evidence Verification Audit ({verification_summary.status_label})
                                    </div>
                                    <div style="font-size:1.1rem; font-weight:700; color:{'#10B981' if verification_summary.reliability_score >= 80 else ('#FBBF24' if verification_summary.reliability_score >= 50 else '#EF4444')};">
                                        Reliability Score: {verification_summary.reliability_score:.1f}%
                                    </div>
                                </div>
                                <div style="display:flex; gap:1.5rem; flex-wrap:wrap; font-size:0.9rem; color:#CBD5E1;">
                                    <div>✓ Supported: <strong>{verification_summary.supported_count}</strong></div>
                                    <div>~ Partially Supported: <strong>{verification_summary.partially_supported_count}</strong></div>
                                    <div>✗ Unsupported: <strong>{verification_summary.unsupported_count}</strong></div>
                                    <div>⚡ Conflicting: <strong>{verification_summary.conflicting_count}</strong></div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    # Render Report Card
                    st.markdown(
                        f'<div class="report-card">{st.markdown(report.content)}</div>',
                        unsafe_allow_html=True,
                    )

                    # Claim-by-Claim Audit Section
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
                        is_full = getattr(src, "is_full_content", False)
                        ext_badge = (
                            '<span class="pill-badge pill-full">✓ Full Article</span>'
                            if is_full
                            else '<span class="pill-badge pill-snippet">○ Snippet</span>'
                        )
                        origin_q = getattr(src, "origin_query", "")
                        query_info = f'<span style="font-size:0.75rem; color:#94A3B8; margin-left:0.5rem;">Via: "{origin_q}"</span>' if origin_q else ""
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

                # Fallback rendering if synthesis timed out but sources were retrieved
                elif evidence_results:
                    st.markdown("---")
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
                                Rather than leaving you waiting, here are the <strong>{len(evidence_results)} verified primary sources</strong> 
                                retrieved directly from the web.
                            </p>
                            <div style="font-size:0.83rem; color:#94A3B8;">
                                <strong>Notice:</strong> Reason: <em>{fallback_reason}</em>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

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
