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
    """Retrieve primary Gemini API key from environment variables."""
    keys = get_gemini_api_keys()
    return keys[0] if keys else None


def get_gemini_api_keys() -> list[str]:
    """Retrieve all configured Gemini API keys for rotation.

    Supports comma-separated keys in GEMINI_API_KEY, as well as
    GEMINI_API_KEY_1, GEMINI_API_KEY_2, etc.
    """
    raw = os.getenv("GEMINI_API_KEY", "").strip()
    keys = [k.strip() for k in raw.split(",") if k.strip()]

    idx = 1
    while True:
        k = os.getenv(f"GEMINI_API_KEY_{idx}", "").strip()
        if not k:
            break
        if k not in keys:
            keys.append(k)
        idx += 1
    return keys


def get_groq_api_key() -> str | None:
    """Retrieve Groq API key (free tier with 14,400 requests/day)."""
    key = os.getenv("GROQ_API_KEY", "").strip()
    return key if key else None


def get_ollama_base_url() -> str:
    """Retrieve Ollama host URL (default: http://localhost:11434)."""
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip().rstrip("/")


def mask_key(key: str | None) -> str:
    """Safely mask an API key for UI status display without leaking secrets."""
    if not key:
        return "Not Set"
    if len(key) <= 8:
        return "****"
    return f"{key[:4]}...{key[-4:]}"
