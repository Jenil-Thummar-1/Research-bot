import time
import urllib.parse
from pathlib import Path
import streamlit as st

from src.config import (
    get_tavily_api_key,
    get_gemini_api_key,
    get_gemini_api_keys,
    get_groq_api_key,
    get_default_provider,
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
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800;900&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    :root {
        --bg-dark: #080B12;
        --bg-surface: rgba(15, 23, 42, 0.72);
        --bg-surface-elevated: rgba(30, 41, 59, 0.55);
        --border-glass: rgba(255, 255, 255, 0.08);
        --border-glass-glow: rgba(99, 102, 241, 0.5);
        --primary-gradient: linear-gradient(135deg, #6366F1 0%, #8B5CF6 50%, #EC4899 100%);
        --aurora-gradient: linear-gradient(135deg, #38BDF8 0%, #818CF8 25%, #C084FC 50%, #F472B6 75%, #38BDF8 100%);
        --text-primary: #F8FAFC;
        --text-secondary: #94A3B8;
        --accent-indigo: #6366F1;
        --accent-violet: #8B5CF6;
        --accent-cyan: #06B6D4;
        --accent-emerald: #10B981;
    }

    /* Ambient Luxury Dark Background */
    .stApp {
        background: #080B12 !important;
        background-image: 
            radial-gradient(circle at 12% 8%, rgba(99, 102, 241, 0.16) 0px, transparent 45%),
            radial-gradient(circle at 88% 12%, rgba(139, 92, 246, 0.13) 0px, transparent 45%),
            radial-gradient(circle at 50% 92%, rgba(236, 72, 153, 0.09) 0px, transparent 55%),
            radial-gradient(circle at 50% 50%, rgba(6, 182, 212, 0.04) 0px, transparent 65%) !important;
        background-attachment: fixed !important;
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: #F8FAFC;
        letter-spacing: -0.01em;
    }

    h1, h2, h3, h4, .main-title, .chat-hero h2, .chat-top-title {
        font-family: 'Outfit', sans-serif !important;
        letter-spacing: -0.025em;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Main Chat Container Viewport with Generous Bottom Padding */
    .main .block-container {
        max-width: 960px !important;
        margin-left: auto !important;
        margin-right: auto !important;
        padding-top: 1.25rem !important;
        padding-bottom: 15rem !important;
    }

    /* Smooth Custom Scrollbars */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(99, 102, 241, 0.35);
        border-radius: 9999px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(139, 92, 246, 0.7);
    }

    /* Main Application Header (Clean & Compact) */
    .main-header {
        margin-bottom: 1.25rem;
        padding: 0.25rem 0 0.85rem 0;
        position: relative;
    }

    .main-title {
        font-size: 2.15rem;
        font-weight: 800;
        background: var(--aurora-gradient);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: auroraText 8s ease infinite;
        display: inline-block;
        margin-bottom: 0.15rem;
        filter: drop-shadow(0 2px 10px rgba(99, 102, 241, 0.25));
    }

    @keyframes auroraText {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .main-subtitle {
        color: #94A3B8;
        font-size: 0.95rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        flex-wrap: wrap;
    }

    /* Segmented Tabs Control */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(15, 23, 42, 0.65) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 14px !important;
        padding: 0.35rem !important;
        gap: 0.5rem !important;
        backdrop-filter: blur(16px) !important;
        margin-bottom: 1.5rem !important;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px !important;
        color: #94A3B8 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 0.6rem 1.4rem !important;
        transition: all 0.28s cubic-bezier(0.16, 1, 0.3, 1) !important;
        border: 1px solid transparent !important;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: #F8FAFC !important;
        background: rgba(255, 255, 255, 0.04) !important;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.28) 0%, rgba(139, 92, 246, 0.22) 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(99, 102, 241, 0.55) !important;
        box-shadow: 0 4px 18px rgba(99, 102, 241, 0.25) !important;
    }

    /* Sidebar Modern ChatGPT / Gemini Styling */
    [data-testid="stSidebar"] {
        background: rgba(10, 14, 26, 0.96) !important;
        backdrop-filter: blur(28px) !important;
        -webkit-backdrop-filter: blur(28px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }

    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.4rem 0.2rem 0.95rem 0.2rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.07);
        margin-bottom: 0.95rem;
    }

    .brand-avatar {
        width: 34px;
        height: 34px;
        border-radius: 10px;
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.15rem;
        color: #FFFFFF;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4);
    }

    .brand-info {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .brand-name {
        font-family: 'Outfit', sans-serif;
        font-size: 1.18rem;
        font-weight: 700;
        color: #F8FAFC;
        letter-spacing: -0.02em;
    }

    .brand-tag {
        font-size: 0.65rem;
        font-weight: 700;
        padding: 0.15rem 0.45rem;
        border-radius: 9999px;
        background: rgba(99, 102, 241, 0.2);
        border: 1px solid rgba(99, 102, 241, 0.4);
        color: #A5B4FC;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .sidebar-section-label {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #64748B;
        margin: 1.1rem 0 0.45rem 0.2rem;
    }

    /* Dedicated + New Chat Button Styling */
    .sidebar-new-chat button {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.88) 0%, rgba(139, 92, 246, 0.88) 100%) !important;
        border: 1px solid rgba(255, 255, 255, 0.18) !important;
        border-radius: 12px !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 0.94rem !important;
        padding: 0.65rem 1rem !important;
        box-shadow: 0 4px 18px rgba(99, 102, 241, 0.32) !important;
        margin-bottom: 0.85rem !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    .sidebar-new-chat button:hover {
        transform: translateY(-2px) scale(1.02) !important;
        box-shadow: 0 8px 24px rgba(139, 92, 246, 0.55) !important;
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%) !important;
        border-color: rgba(255, 255, 255, 0.3) !important;
    }

    /* Modern ChatGPT / Gemini Seamless Conversation Rows */
    .sidebar-chat-list {
        display: flex;
        flex-direction: column;
        gap: 2px !important;
    }

    /* REDUCE SPACE: Nested Vertical Block in Sidebar for Chat History */
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] [data-testid="stVerticalBlock"] {
        gap: 3px !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    [data-testid="stSidebar"] div[data-testid="stElementContainer"]:has([data-testid="stHorizontalBlock"]) {
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Base styling for every conversation row */
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {
        align-items: center !important;
        gap: 0 !important;
        border-radius: 7px !important;
        margin: 0 !important;
        padding: 0 0.25rem !important;
        min-height: 32px !important;
        height: 32px !important;
        background: transparent !important;
        background-color: transparent !important;
        background-image: none !important;
        border: 1px solid transparent !important;
        box-shadow: none !important;
        transition: background 0.15s ease, border-color 0.15s ease !important;
    }

    /* Inactive chat items: completely borderless, NO background box */
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:not(:has(button[kind="primary"])) {
        background: transparent !important;
        background-color: transparent !important;
        background-image: none !important;
        border: 1px solid transparent !important;
        box-shadow: none !important;
    }

    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:not(:has(button[kind="primary"])):hover {
        background: rgba(255, 255, 255, 0.05) !important;
        border-color: transparent !important;
    }

    /* Reset ALL buttons and columns inside sidebar conversation row so they have NO big bg box or border */
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] .stButton > button,
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stPopover"] > button,
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] button,
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stColumn"],
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="column"] {
        background: transparent !important;
        background-color: transparent !important;
        background-image: none !important;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
        transform: none !important;
        backdrop-filter: none !important;
        -webkit-backdrop-filter: none !important;
    }

    /* ACTIVE CHAT: ONLY the active item gets the translucent small box! */
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(button[kind="primary"]) {
        background: rgba(99, 102, 241, 0.15) !important;
        background-color: rgba(99, 102, 241, 0.15) !important;
        border: 1px solid rgba(99, 102, 241, 0.38) !important;
        border-radius: 7px !important;
    }

    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(button[kind="primary"]):hover {
        background: rgba(99, 102, 241, 0.22) !important;
        border-color: rgba(99, 102, 241, 0.48) !important;
    }

    /* Columns within the conversation row */
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stColumn"],
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="column"] {
        padding: 0 !important;
        margin: 0 !important;
        display: flex !important;
        align-items: center !important;
        height: 34px !important;
    }

    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stColumn"] [data-testid="stElementContainer"],
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="column"] [data-testid="stElementContainer"] {
        margin: 0 !important;
        padding: 0 !important;
        width: 100% !important;
    }

    /* Chat title button (first column): strict left aligned, clean text */
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stColumn"]:first-child,
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="column"]:first-child,
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stColumn"]:nth-child(1),
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="column"]:nth-child(1) {
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
    }

    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stColumn"]:first-child .stButton,
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="column"]:first-child .stButton,
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stColumn"]:nth-child(1) .stButton,
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="column"]:nth-child(1) .stButton {
        width: 100% !important;
        display: flex !important;
        justify-content: flex-start !important;
        align-items: center !important;
    }

    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stColumn"]:first-child button,
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="column"]:first-child button,
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stColumn"]:nth-child(1) button,
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="column"]:nth-child(1) button,
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stColumn"]:first-child .stButton > button,
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="column"]:first-child .stButton > button {
        color: #94A3B8 !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        display: flex !important;
        justify-content: flex-start !important;
        align-items: center !important;
        text-align: left !important;
        padding: 0 0.5rem !important;
        border-radius: 6px !important;
        width: 100% !important;
        height: 32px !important;
        min-height: 32px !important;
        line-height: 32px !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }

    /* Inner containers and text elements inside chat title button */
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stColumn"]:first-child button [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="column"]:first-child button [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stColumn"]:nth-child(1) button [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="column"]:nth-child(1) button [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stColumn"]:first-child button div,
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="column"]:first-child button div,
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stColumn"]:nth-child(1) button div,
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="column"]:nth-child(1) button div {
        display: flex !important;
        justify-content: flex-start !important;
        align-items: center !important;
        text-align: left !important;
        width: 100% !important;
        max-width: 100% !important;
        overflow: hidden !important;
    }

    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stColumn"]:first-child button p,
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="column"]:first-child button p,
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stColumn"]:nth-child(1) button p,
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="column"]:nth-child(1) button p,
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stColumn"]:first-child button span,
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="column"]:first-child button span,
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stColumn"]:nth-child(1) button span,
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="column"]:nth-child(1) button span {
        text-align: left !important;
        justify-content: flex-start !important;
        align-items: center !important;
        width: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        display: block !important;
    }

    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stColumn"]:first-child button:hover,
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="column"]:first-child button:hover {
        color: #FFFFFF !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    /* Active chat title text */
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(button[kind="primary"]) [data-testid="stColumn"]:first-child button,
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(button[kind="primary"]) [data-testid="column"]:first-child button {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }

    /* ACTION BUTTONS: Rename ✏️ & Delete 🗑️ HIDDEN BY DEFAULT */
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stColumn"]:not(:first-child) button,
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="column"]:not(:first-child) button,
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stPopover"] > button {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: #64748B !important;
        opacity: 0 !important;
        visibility: hidden !important;
        pointer-events: none !important;
        padding: 0 !important;
        border-radius: 6px !important;
        font-size: 0.82rem !important;
        width: 26px !important;
        height: 26px !important;
        min-width: 26px !important;
        min-height: 26px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transform: none !important;
        transition: opacity 0.18s ease, visibility 0.18s ease, background 0.15s ease !important;
    }

    /* ON HOVER over the row: action buttons smoothly appear */
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:hover [data-testid="stColumn"]:not(:first-child) button,
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:hover [data-testid="column"]:not(:first-child) button,
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:hover [data-testid="stPopover"] > button,
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stPopover"]:has([aria-expanded="true"]) > button {
        opacity: 0.75 !important;
        visibility: visible !important;
        pointer-events: auto !important;
    }

    /* Hover highlight for Rename ✏️ */
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stPopover"] > button:hover {
        opacity: 1 !important;
        color: #A5B4FC !important;
        background: rgba(99, 102, 241, 0.22) !important;
        transform: scale(1.08) !important;
    }

    /* Hover highlight for Delete 🗑️ */
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stColumn"]:last-child button:hover,
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="column"]:last-child button:hover {
        opacity: 1 !important;
        color: #F87171 !important;
        background: rgba(239, 68, 68, 0.22) !important;
        transform: scale(1.08) !important;
    }

    /* Clean popover popup styling */
    [data-testid="stPopoverBody"] {
        background: #0B101E !important;
        border: 1px solid rgba(255, 255, 255, 0.14) !important;
        border-radius: 12px !important;
        padding: 0.85rem !important;
        box-shadow: 0 12px 36px rgba(0, 0, 0, 0.75) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        min-width: 210px !important;
    }

    [data-testid="stPopoverBody"] input {
        background: rgba(255, 255, 255, 0.06) !important;
        border: 1px solid rgba(255, 255, 255, 0.14) !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
        font-size: 0.88rem !important;
    }

    /* Sidebar Expander & Subheadings (ChatGPT / Gemini style) */
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        background: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        overflow: hidden !important;
        margin-top: 1rem !important;
        backdrop-filter: blur(12px) !important;
    }

    [data-testid="stSidebar"] [data-testid="stExpander"] summary {
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        color: #CBD5E1 !important;
        padding: 0.65rem 0.85rem !important;
    }

    [data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {
        color: #FFFFFF !important;
        background: rgba(255, 255, 255, 0.03) !important;
    }

    [data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
        padding: 0.85rem !important;
        border-top: 1px solid rgba(255, 255, 255, 0.06) !important;
    }

    .sidebar-subheading {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: #818CF8;
        margin: 0.9rem 0 0.45rem 0;
        padding-bottom: 0.25rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }

    .chat-top-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.3rem;
        font-weight: 700;
        color: #F8FAFC;
        letter-spacing: -0.015em;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        text-align: left !important;
    }

    .prompts-grid-label {
        font-size: 0.88rem;
        font-weight: 600;
        color: #94A3B8;
        margin: 1.5rem 0 0.75rem 0;
    }

    /* 3D Animated AI Orb for Empty State */
    .hero-orb-wrapper {
        position: relative;
        width: 120px;
        height: 120px;
        margin: 0 auto 1.6rem auto;
        perspective: 1000px;
    }

    .hero-orb-core {
        position: absolute;
        top: 50%;
        left: 50%;
        width: 56px;
        height: 56px;
        margin: -28px 0 0 -28px;
        border-radius: 50%;
        background: radial-gradient(circle, #C7D2FE 0%, #818CF8 40%, #6366F1 75%, #4338CA 100%);
        box-shadow: 0 0 40px rgba(99, 102, 241, 0.85), inset 0 0 16px rgba(255, 255, 255, 0.8);
        animation: orbPulse 3.5s ease-in-out infinite alternate;
    }

    @keyframes orbPulse {
        0% { transform: scale(0.92); box-shadow: 0 0 25px rgba(99, 102, 241, 0.6); }
        100% { transform: scale(1.08); box-shadow: 0 0 50px rgba(139, 92, 246, 0.9), 0 0 70px rgba(236, 72, 153, 0.4); }
    }

    .hero-orb-ring-1 {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        border-radius: 50%;
        border: 1.5px dashed rgba(165, 180, 252, 0.55);
        transform-style: preserve-3d;
        animation: ringOrbit1 9s linear infinite;
    }

    @keyframes ringOrbit1 {
        0% { transform: rotateX(65deg) rotateY(15deg) rotateZ(0deg); }
        100% { transform: rotateX(65deg) rotateY(15deg) rotateZ(360deg); }
    }

    .hero-orb-ring-2 {
        position: absolute;
        top: 10%;
        left: 10%;
        width: 80%;
        height: 80%;
        border-radius: 50%;
        border: 1.5px dotted rgba(236, 72, 153, 0.65);
        transform-style: preserve-3d;
        animation: ringOrbit2 6s linear infinite reverse;
    }

    @keyframes ringOrbit2 {
        0% { transform: rotateX(25deg) rotateY(70deg) rotateZ(0deg); }
        100% { transform: rotateX(25deg) rotateY(70deg) rotateZ(360deg); }
    }

    .hero-orb-glow {
        position: absolute;
        top: 25%;
        left: 25%;
        width: 50%;
        height: 50%;
        border-radius: 50%;
        background: rgba(99, 102, 241, 0.3);
        filter: blur(24px);
    }

    /* Hero Section */
    .chat-hero {
        text-align: center;
        padding: 2.25rem 1rem 1.5rem;
        position: relative;
    }

    .chat-hero h2 {
        color: #F8FAFC;
        font-size: 2.15rem;
        font-weight: 800;
        margin-bottom: 0.6rem;
        background: linear-gradient(180deg, #FFFFFF 0%, #CBD5E1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .chat-hero p {
        color: #94A3B8;
        font-size: 1.05rem;
        max-width: 620px;
        margin: 0 auto 1.5rem auto;
        line-height: 1.6;
    }

    /* Universal Modern Button Styling with 3D Interaction */
    .stButton > button {
        border-radius: 12px !important;
        font-weight: 600 !important;
        transition: all 0.28s cubic-bezier(0.16, 1, 0.3, 1) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        background: rgba(255, 255, 255, 0.04) !important;
        color: #E2E8F0 !important;
        backdrop-filter: blur(10px) !important;
    }

    .stButton > button:hover {
        transform: translateY(-3.5px) scale(1.015) !important;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.18), rgba(139, 92, 246, 0.24)) !important;
        border-color: rgba(139, 92, 246, 0.6) !important;
        box-shadow: 0 12px 26px -5px rgba(99, 102, 241, 0.35), 0 0 15px rgba(99, 102, 241, 0.15) !important;
        color: #FFFFFF !important;
    }

    .stButton > button:active {
        transform: scale(0.98) !important;
    }

    button[kind="primary"] {
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 50%, #EC4899 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4) !important;
    }

    button[kind="primary"]:hover {
        transform: translateY(-2px) scale(1.02) !important;
        box-shadow: 0 8px 30px rgba(139, 92, 246, 0.65) !important;
    }

    /* Chat Messages Glass Bubbles */
    [data-testid="stChatMessage"] {
        border-radius: 18px !important;
        margin-bottom: 1.25rem !important;
        padding: 1.25rem 1.6rem !important;
        animation: chatMsgEnter 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        line-height: 1.65 !important;
        font-size: 0.98rem !important;
    }

    @keyframes chatMsgEnter {
        from { opacity: 0; transform: translateY(10px) scale(0.99); }
        to { opacity: 1; transform: translateY(0) scale(1); }
    }

    /* USER MESSAGE: Pinned strictly to the right-end side (modern chat app style) */
    div[data-testid="stElementContainer"]:has([data-testid="stChatMessageAvatarUser"]),
    div[data-testid="stElementContainer"]:has([data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])),
    div[data-testid="stElementContainer"]:has([aria-label*="user" i]) {
        display: flex !important;
        justify-content: flex-end !important;
        width: 100% !important;
    }

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),
    [data-testid="stChatMessage"]:has([aria-label*="user" i]) {
        background: linear-gradient(135deg, rgba(67, 56, 202, 0.38) 0%, rgba(99, 102, 241, 0.28) 100%) !important;
        border: 1px solid rgba(129, 140, 248, 0.42) !important;
        border-radius: 20px 20px 4px 20px !important;
        box-shadow: 0 8px 24px -4px rgba(79, 70, 229, 0.35) !important;
        margin-left: auto !important;
        margin-right: 0 !important;
        align-self: flex-end !important;
        width: fit-content !important;
        max-width: 78% !important;
        min-width: 100px !important;
        display: flex !important;
        flex-direction: row-reverse !important;
        align-items: flex-start !important;
        gap: 0.8rem !important;
    }

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) > div:first-child,
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageAvatarUser"],
    [data-testid="stChatMessage"]:has([aria-label*="user" i]) > div:first-child {
        margin-left: 0.4rem !important;
        margin-right: 0 !important;
        flex-shrink: 0 !important;
    }

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"],
    [data-testid="stChatMessage"]:has([aria-label*="user" i]) [data-testid="stChatMessageContent"] {
        text-align: left !important;
        color: #F8FAFC !important;
        flex: 1 1 auto !important;
        word-break: break-word !important;
    }

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] p,
    [data-testid="stChatMessage"]:has([aria-label*="user" i]) [data-testid="stChatMessageContent"] p {
        color: #F8FAFC !important;
        font-size: 0.98rem !important;
        line-height: 1.6 !important;
        margin: 0 !important;
    }

    /* USER & ASSISTANT AVATARS */
    [data-testid="stChatMessageAvatarUser"] {
        background: linear-gradient(135deg, #6366F1, #8B5CF6) !important;
        color: #FFFFFF !important;
        border-radius: 50% !important;
        box-shadow: 0 2px 8px rgba(99, 102, 241, 0.4) !important;
    }

    [data-testid="stChatMessageAvatarAssistant"] {
        background: linear-gradient(135deg, #3B82F6, #06B6D4) !important;
        color: #FFFFFF !important;
        border-radius: 50% !important;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.4) !important;
    }

    /* ASSISTANT MESSAGE: Left aligned with sleek dark glass backdrop */
    div[data-testid="stElementContainer"]:has([data-testid="stChatMessageAvatarAssistant"]),
    div[data-testid="stElementContainer"]:has([data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])),
    div[data-testid="stElementContainer"]:has([aria-label*="assistant" i]) {
        display: flex !important;
        justify-content: flex-start !important;
        width: 100% !important;
    }

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]),
    [data-testid="stChatMessage"]:has([aria-label*="assistant" i]) {
        background: rgba(15, 23, 42, 0.72) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 20px 20px 20px 4px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.38) !important;
        margin-right: auto !important;
        margin-left: 0 !important;
        align-self: flex-start !important;
        max-width: 95% !important;
        width: 100% !important;
        display: flex !important;
        flex-direction: row !important;
        align-items: flex-start !important;
        gap: 0.85rem !important;
    }

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) > div:first-child,
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageAvatarAssistant"],
    [data-testid="stChatMessage"]:has([aria-label*="assistant" i]) > div:first-child {
        margin-right: 0.4rem !important;
        margin-left: 0 !important;
        flex-shrink: 0 !important;
    }

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"],
    [data-testid="stChatMessage"]:has([aria-label*="assistant" i]) [data-testid="stChatMessageContent"] {
        text-align: left !important;
        color: #E2E8F0 !important;
        flex: 1 1 auto !important;
        word-break: break-word !important;
    }

    @media (max-width: 768px) {
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),
        [data-testid="stChatMessage"]:has([aria-label*="user" i]) {
            max-width: 90% !important;
        }
    }

    /* Clean Assistant Message Action Tray (Copy 📋 & Retry 🔄) */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stHorizontalBlock"],
    [data-testid="stChatMessage"]:has([aria-label*="assistant" i]) [data-testid="stHorizontalBlock"] {
        opacity: 0 !important;
        visibility: hidden !important;
        pointer-events: none !important;
        transition: opacity 0.2s cubic-bezier(0.16, 1, 0.3, 1), visibility 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
        margin-top: 0.45rem !important;
        gap: 0.25rem !important;
        align-items: center !important;
    }

    /* On hover of assistant message OR when popover is open: smoothly show action buttons */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]):hover [data-testid="stHorizontalBlock"],
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stHorizontalBlock"]:has([aria-expanded="true"]),
    [data-testid="stChatMessage"]:has([aria-label*="assistant" i]):hover [data-testid="stHorizontalBlock"],
    [data-testid="stChatMessage"]:has([aria-label*="assistant" i]) [data-testid="stHorizontalBlock"]:has([aria-expanded="true"]) {
        opacity: 1 !important;
        visibility: visible !important;
        pointer-events: auto !important;
    }

    /* Compact Column Sizing for Action Icons */
    [data-testid="stChatMessage"] [data-testid="stHorizontalBlock"] [data-testid="stColumn"]:first-child,
    [data-testid="stChatMessage"] [data-testid="stHorizontalBlock"] [data-testid="column"]:first-child,
    [data-testid="stChatMessage"] [data-testid="stHorizontalBlock"] [data-testid="stColumn"]:nth-child(2),
    [data-testid="stChatMessage"] [data-testid="stHorizontalBlock"] [data-testid="column"]:nth-child(2) {
        width: 34px !important;
        flex: 0 0 34px !important;
        min-width: 34px !important;
        padding: 0 !important;
        margin-right: 4px !important;
    }

    /* Minimal Icon Button Styling for Assistant Actions */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stHorizontalBlock"] .stButton > button,
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stHorizontalBlock"] [data-testid="stPopover"] > button,
    [data-testid="stChatMessage"]:has([aria-label*="assistant" i]) [data-testid="stHorizontalBlock"] .stButton > button,
    [data-testid="stChatMessage"]:has([aria-label*="assistant" i]) [data-testid="stHorizontalBlock"] [data-testid="stPopover"] > button {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        color: #94A3B8 !important;
        font-size: 0.95rem !important;
        width: 32px !important;
        height: 32px !important;
        min-width: 32px !important;
        min-height: 32px !important;
        padding: 0 !important;
        border-radius: 8px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        box-shadow: none !important;
        transform: none !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stHorizontalBlock"] .stButton > button:hover,
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stHorizontalBlock"] [data-testid="stPopover"] > button:hover,
    [data-testid="stChatMessage"]:has([aria-label*="assistant" i]) [data-testid="stHorizontalBlock"] .stButton > button:hover,
    [data-testid="stChatMessage"]:has([aria-label*="assistant" i]) [data-testid="stHorizontalBlock"] [data-testid="stPopover"] > button:hover {
        background: rgba(99, 102, 241, 0.28) !important;
        border-color: rgba(99, 102, 241, 0.6) !important;
        color: #FFFFFF !important;
        transform: scale(1.08) !important;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35) !important;
    }

    /* Pinned Static Bottom Type Bar (ChatGPT / Gemini Style) */
    [data-testid="stBottom"] {
        position: fixed !important;
        bottom: 0 !important;
        right: 0 !important;
        left: 0 !important;
        background: linear-gradient(180deg, rgba(8, 11, 18, 0) 0%, rgba(8, 11, 18, 0.94) 40%, #080B12 100%) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        padding-top: 0.35rem !important;
        padding-bottom: 1.25rem !important;
        border-top: none !important;
        z-index: 50 !important;
        transition: left 0.28s cubic-bezier(0.16, 1, 0.3, 1), width 0.28s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }

    /* Seamless desktop alignment: offset stBottom by sidebar width so chat input is centered in main area */
    @media (min-width: 768px) {
        body:has([data-testid="stSidebar"][aria-expanded="true"]) [data-testid="stBottom"],
        body:has([data-testid="stSidebar"]:not([data-collapsed="true"])) [data-testid="stBottom"],
        [data-testid="stSidebar"][aria-expanded="true"] ~ [data-testid="stMain"] [data-testid="stBottom"],
        [data-testid="stSidebar"]:not([data-collapsed="true"]) ~ [data-testid="stMain"] [data-testid="stBottom"] {
            left: var(--sidebar-width, 21rem) !important;
            width: calc(100% - var(--sidebar-width, 21rem)) !important;
        }
    }

    /* When sidebar is collapsed or on mobile: full viewport width */
    body:has([data-testid="stSidebar"][data-collapsed="true"]) [data-testid="stBottom"],
    body:has([data-testid="stSidebar"][aria-expanded="false"]) [data-testid="stBottom"],
    [data-testid="stSidebar"][data-collapsed="true"] ~ [data-testid="stMain"] [data-testid="stBottom"],
    [data-testid="stSidebar"][aria-expanded="false"] ~ [data-testid="stMain"] [data-testid="stBottom"] {
        left: 0 !important;
        width: 100% !important;
    }

    [data-testid="stBottom"] > div {
        max-width: 820px !important;
        margin: 0 auto !important;
        padding: 0 1rem !important;
    }

    /* Hide root bottom chat input when Tab 2 (Research Studio) is selected */
    body:has(.stTabs [data-baseweb="tab"]:nth-child(2)[aria-selected="true"]) [data-testid="stBottom"] {
        display: none !important;
    }

    [data-testid="stChatInput"] {
        background: rgba(18, 24, 38, 0.92) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 24px !important;
        box-shadow: 0 10px 35px rgba(0, 0, 0, 0.6), 0 0 0 1px rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
        max-width: 860px !important;
        margin: 0 auto !important;
    }

    [data-testid="stChatInput"]:focus-within {
        border-color: rgba(99, 102, 241, 0.85) !important;
        box-shadow: 0 12px 42px rgba(0, 0, 0, 0.7), 0 0 30px rgba(99, 102, 241, 0.35) !important;
        transform: translateY(-1px) !important;
    }

    [data-testid="stChatInput"] textarea {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 0.98rem !important;
        color: #F8FAFC !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #64748B !important;
    }

    [data-testid="stChatInputSubmitButton"] > button {
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%) !important;
        color: #FFFFFF !important;
        border-radius: 50% !important;
        border: none !important;
        box-shadow: 0 2px 10px rgba(99, 102, 241, 0.4) !important;
        transition: all 0.2s ease !important;
    }

    [data-testid="stChatInputSubmitButton"] > button:hover {
        transform: scale(1.08) !important;
        box-shadow: 0 4px 18px rgba(139, 92, 246, 0.6) !important;
    }

    /* 3D Elevated Cards */
    .report-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.78) 0%, rgba(30, 41, 59, 0.42) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 18px;
        padding: 2.25rem;
        margin-bottom: 2rem;
        box-shadow: 0 20px 50px -10px rgba(0, 0, 0, 0.55), inset 0 1px 0 rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(16px);
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
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.035) 0%, rgba(255, 255, 255, 0.01) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.25rem;
        margin-bottom: 0.95rem;
        backdrop-filter: blur(12px);
        transition: transform 0.3s cubic-bezier(0.16, 1, 0.3, 1), border-color 0.3s ease, box-shadow 0.3s ease;
        perspective: 1000px;
    }

    .source-card:hover {
        transform: translateY(-4px) scale(1.01) rotateX(1.5deg);
        border-color: rgba(99, 102, 241, 0.55);
        box-shadow: 0 16px 35px -8px rgba(99, 102, 241, 0.25), 0 0 15px rgba(99, 102, 241, 0.12);
    }

    .claim-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 14px;
        padding: 1.25rem;
        margin-bottom: 0.95rem;
        transition: all 0.25s ease;
    }

    .claim-card:hover {
        transform: translateY(-3px);
        border-color: rgba(139, 92, 246, 0.45);
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.35);
    }

    .verification-summary-box {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.85) 0%, rgba(30, 41, 59, 0.5) 100%);
        border: 1px solid rgba(99, 102, 241, 0.4);
        box-shadow: 0 15px 35px -5px rgba(99, 102, 241, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.75rem;
    }

    /* Badges & Pulsing Live Status Dots */
    .status-badge {
        padding: 0.4rem 0.85rem;
        border-radius: 8px;
        font-size: 0.84rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.5rem;
        letter-spacing: 0.02em;
    }

    .status-connected {
        background: rgba(16, 185, 129, 0.12);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.35);
        box-shadow: 0 0 12px rgba(16, 185, 129, 0.15);
    }

    .status-missing {
        background: rgba(239, 68, 68, 0.12);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.35);
    }

    .live-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10B981;
        display: inline-block;
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        animation: livePulse 2s infinite;
    }

    @keyframes livePulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    .pill-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        backdrop-filter: blur(8px);
    }

    .pill-tag {
        background: rgba(99, 102, 241, 0.18);
        color: #A5B4FC;
        border: 1px solid rgba(99, 102, 241, 0.45);
        margin-right: 0.4rem;
    }

    .pill-domain {
        background: rgba(6, 182, 212, 0.15);
        color: #38BDF8;
        border: 1px solid rgba(6, 182, 212, 0.35);
    }

    .pill-full {
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.35);
    }

    .pill-snippet {
        background: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.35);
    }

    .badge-supported {
        background: rgba(16, 185, 129, 0.16);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.4);
        box-shadow: 0 0 10px rgba(16, 185, 129, 0.2);
    }

    .badge-partially {
        background: rgba(245, 158, 11, 0.16);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }

    .badge-unsupported {
        background: rgba(239, 68, 68, 0.16);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }

    .badge-conflicting {
        background: rgba(236, 72, 153, 0.16);
        color: #F472B6;
        border: 1px solid rgba(236, 72, 153, 0.4);
        box-shadow: 0 0 10px rgba(236, 72, 153, 0.2);
    }

    .methodology-notice {
        background: rgba(30, 41, 59, 0.55);
        border-left: 4px solid #6366F1;
        padding: 0.95rem 1.25rem;
        border-radius: 0 10px 10px 0;
        font-size: 0.88rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(10px);
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
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="brand-avatar">✦</div>
            <div class="brand-info">
                <span class="brand-name">AI Assistant</span>
                <span class="brand-tag">PRO</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # + New Chat Button
    st.markdown('<div class="sidebar-new-chat">', unsafe_allow_html=True)
    if st.button("➕ New Chat", use_container_width=True, key="sidebar_new_chat_btn"):
        new_c = history_service.create_conversation("New Chat")
        st.session_state["active_conversation_id"] = new_c.id
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # List stored conversations
    all_conversations = history_service.list_conversations()
    if all_conversations:
        st.markdown('<div class="sidebar-section-label">Recent Chats</div>', unsafe_allow_html=True)
        with st.container():
            for conv in all_conversations:
                is_active = conv.id == st.session_state.get("active_conversation_id")
                label_prefix = "✦ " if is_active else "💬 "
                btn_label = f"{label_prefix}{conv.title}"
                btn_type = "primary" if is_active else "secondary"

                col_item1, col_item2, col_item3 = st.columns([0.76, 0.12, 0.12])
                with col_item1:
                    if st.button(
                        btn_label,
                        key=f"conv_select_{conv.id}",
                        use_container_width=True,
                        type=btn_type,
                        help=f"Created: {time.strftime('%b %d, %H:%M', time.localtime(conv.created_at))}",
                    ):
                        st.session_state["active_conversation_id"] = conv.id
                        st.rerun()
                with col_item2:
                    # Hover-activated rename popover
                    with st.popover("✏️", help="Rename chat"):
                        st.markdown("<p style='font-size:0.85rem; font-weight:700; color:#F8FAFC; margin-bottom:0.4rem;'>Rename Chat</p>", unsafe_allow_html=True)
                        ren_val = st.text_input(
                            "Title",
                            value=conv.title,
                            key=f"ren_input_{conv.id}",
                            label_visibility="collapsed",
                            placeholder="Chat title...",
                        )
                        if st.button("Save", key=f"save_ren_{conv.id}", use_container_width=True, type="primary"):
                            if ren_val.strip():
                                history_service.update_conversation_title(conv.id, ren_val.strip())
                                st.rerun()
                with col_item3:
                    # Hover-activated delete conversation button
                    if st.button("🗑️", key=f"del_{conv.id}", help="Delete chat"):
                        history_service.delete_conversation(conv.id)
                        remaining = history_service.list_conversations()
                        if remaining:
                            st.session_state["active_conversation_id"] = remaining[0].id
                        else:
                            new_created = history_service.create_conversation("New Chat")
                            st.session_state["active_conversation_id"] = new_created.id
                        st.rerun()

    # Consolidated Unified Settings Expander (ChatGPT / Gemini style)
    with st.expander("⚙️ Settings & Preferences", expanded=False):
        # 1. Export Active Chat
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

        # 2. AI Model & Search Settings
        st.markdown('<div class="sidebar-subheading">Model & Grounding</div>', unsafe_allow_html=True)
        groq_is_configured = bool(get_groq_api_key())
        default_provider_index = 0 if groq_is_configured else 1

        provider_label = st.radio(
            "AI Backend Provider",
            options=[
                "Groq Cloud (14,400 Free Req/Day)",
                "Google Gemini (Key Pool / Auto-Rotate)",
            ],
            index=default_provider_index,
            help="Select the AI backend. Groq offers fast inference with 14,400 free requests/day; Gemini provides multi-turn intelligence with key pool rotation.",
        )

        if "Groq" in provider_label:
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

        # 3. Connection & API Status
        st.markdown('<div class="sidebar-subheading">API Connections & Overrides</div>', unsafe_allow_html=True)
        env_tavily_key = get_tavily_api_key()
        env_gemini_key = get_gemini_api_key()
        gemini_key_pool = get_gemini_api_keys()
        env_groq_key = get_groq_api_key()

        if env_gemini_key:
            st.markdown(
                '<div class="status-badge status-connected"><span class="live-dot"></span> Gemini: Connected</div>',
                unsafe_allow_html=True,
            )
            st.caption(f"Gemini Key: `{mask_key(env_gemini_key)}` ({len(gemini_key_pool)} key(s) in pool)")
        else:
            st.markdown(
                '<div class="status-badge status-missing">✗ Gemini: Missing</div>',
                unsafe_allow_html=True,
            )
            st.caption("Add `GEMINI_API_KEY` to local `.env`")

        if env_groq_key:
            st.markdown(
                '<div class="status-badge status-connected"><span class="live-dot"></span> Groq Cloud: Connected</div>',
                unsafe_allow_html=True,
            )
            st.caption(f"Groq Key: `{mask_key(env_groq_key)}` (14,400 free req/day)")
        else:
            st.markdown(
                '<div class="status-badge status-missing">✗ Groq Cloud: Optional</div>',
                unsafe_allow_html=True,
            )
            st.caption("Add `GROQ_API_KEY` to `.env` (free tier at console.groq.com)")

        if env_tavily_key:
            st.markdown(
                '<div class="status-badge status-connected"><span class="live-dot"></span> Tavily API: Connected</div>',
                unsafe_allow_html=True,
            )
            st.caption(f"Tavily Key: `{mask_key(env_tavily_key)}`")
        else:
            st.markdown(
                '<div class="status-badge status-connected"><span class="live-dot"></span> DuckDuckGo: Ready (Keyless)</div>',
                unsafe_allow_html=True,
            )
            st.caption("Free search active • 0 API keys required")

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
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.75rem;">
            <div style="display:flex; align-items:center; gap:0.75rem;">
                <div class="main-title">🤖 AI Assistant & Research Agent</div>
                <span class="pill-badge status-connected" style="margin-bottom:0;"><span class="live-dot"></span> Cloud Ready</span>
            </div>
            <div class="main-subtitle">
                <span>Multi-Turn Dialogue</span>
                <span style="color:#475569;">•</span>
                <span>Real-Time Streaming</span>
                <span style="color:#475569;">•</span>
                <span>Live Web Grounding</span>
            </div>
        </div>
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

    # Clean Top Bar
    col_tb1, col_tb2 = st.columns([3, 2])
    with col_tb1:
        title_str = active_conv.title if active_conv else "New Chat"
        st.markdown(f'<div class="chat-top-title">{title_str}</div>', unsafe_allow_html=True)
    with col_tb2:
        model_pill = "⚡ Groq 70B" if "llama-3.3" in selected_model else ("⚡ Groq 8B" if "groq" in selected_provider else "✨ Gemini Flash")
        search_pill = '<span class="pill-badge pill-domain" style="margin-left:0.4rem;">🌐 Web Grounding ON</span>' if enable_web_search else ''
        st.markdown(
            f'<div style="text-align:right;"><span class="pill-badge pill-full"><span class="live-dot"></span> {model_pill}</span>{search_pill}</div>',
            unsafe_allow_html=True,
        )

    # Dedicated container for all conversation messages (strictly ABOVE bottom input dock)
    chat_container = st.container()

    # Retrieve all messages for the current conversation
    chat_messages = history_service.get_messages(active_conv_id)

    with chat_container:
        # Empty State: Suggestion Chips
        if not chat_messages:
            st.markdown(
                """
                <div class="chat-hero">
                    <div class="hero-orb-wrapper">
                        <div class="hero-orb-core"></div>
                        <div class="hero-orb-ring-1"></div>
                        <div class="hero-orb-ring-2"></div>
                        <div class="hero-orb-glow"></div>
                    </div>
                    <h2>How can I help you today?</h2>
                    <p>Ask anything: everyday explanations, coding assistance, creative writing, step-by-step plans, or enable web search for live facts.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown('<div class="prompts-grid-label">💡 Suggested Questions</div>', unsafe_allow_html=True)
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

                # Clean minimal actions below assistant response (ChatGPT / Gemini style)
                if msg.role == "assistant":
                    was_stopped = msg.metadata.get("stopped_early", False)
                    if was_stopped:
                        st.caption("⏹️ *Response stopped early*")

                    col_a1, col_a2, _ = st.columns([0.05, 0.05, 0.90])
                    with col_a1:
                        with st.popover("📋", help="Copy response"):
                            st.markdown("<p style='font-size:0.85rem; font-weight:700; color:#F8FAFC; margin-bottom:0.4rem;'>Copy Response</p>", unsafe_allow_html=True)
                            st.code(msg.content, language=None)
                    with col_a2:
                        # Retry response button for the latest assistant message
                        if idx == len(chat_messages) - 1:
                            if st.button("🔄", key=f"retry_msg_{msg.id}", help="Regenerate this response"):
                                history_service.delete_last_assistant_message(active_conv_id)
                                # Get the user prompt that triggered it
                                updated_msgs = history_service.get_messages(active_conv_id)
                                if updated_msgs and updated_msgs[-1].role == "user":
                                    st.session_state["pending_chat_prompt"] = updated_msgs[-1].content
                                    # Pop the user message so it gets re-added and re-executed cleanly
                                    history_service.delete_last_assistant_message(active_conv_id)
                                st.rerun()


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


# -----------------------------------------------------------------------------
# Static Bottom Chat Input Bar (Permanently Anchored to Viewport Bottom)
# -----------------------------------------------------------------------------
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

    if active_conv and active_conv.title == "New Chat":
        clean_title = clean_user_prompt[:35].strip() + ("..." if len(clean_user_prompt) > 35 else "")
        history_service.update_conversation_title(active_conv_id, clean_title)

    # Store user message in SQLite
    history_service.add_message(active_conv_id, "user", clean_user_prompt)

    # RENDER DIRECTLY INSIDE chat_container (strictly ABOVE the static bottom input bar!)
    with chat_container:
        with st.chat_message("user"):
            st.markdown(clean_user_prompt)

        provider_ready = True
        if selected_provider == "gemini" and not active_gemini_key and not active_groq_key:
            provider_ready = False
            st.error(
                "🔑 **Gemini API Key Required**\n\n"
                "Please configure your `GEMINI_API_KEY` in `.env` or in the sidebar **Session Key Overrides**,\n"
                "or configure `GROQ_API_KEY` for free Groq Cloud access."
            )
        elif selected_provider == "groq" and not active_groq_key and not active_gemini_key:
            provider_ready = False
            st.error(
                "🔑 **Groq API Key Required**\n\n"
                "Please configure `GROQ_API_KEY` in `.env` or sidebar overrides "
                "(get a free key at https://console.groq.com for 14,400 free requests/day),\n"
                "or switch to Google Gemini in the sidebar."
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
