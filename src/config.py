"""Configuration and environment variable management."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root if present
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATH = _PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=_ENV_PATH)


def get_tavily_api_key() -> str | None:
    """Retrieve Tavily API key from environment variables."""
    key = os.getenv("TAVILY_API_KEY", "").strip()
    return key if key else None


def get_gemini_api_key() -> str | None:
    """Retrieve Gemini API key from environment variables."""
    key = os.getenv("GEMINI_API_KEY", "").strip()
    return key if key else None


def mask_key(key: str | None) -> str:
    """Safely mask an API key for UI status display without leaking secrets."""
    if not key:
        return "Not Set"
    if len(key) <= 8:
        return "****"
    return f"{key[:4]}...{key[-4:]}"
