"""Settings read from .env (see .env.example). No provider, URL or model name is written in the code.

The brain, speech-to-text and text-to-speech each talk to an OpenAI-compatible endpoint. A missing required value
is reported by the first code that needs it (see `require`), so tests and tools that do not call a provider still import.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, "").strip() or default


def require(**values: str) -> None:
    """Raise a clear error naming every setting in `values` that is empty."""
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise ValueError(f"Missing in .env: {', '.join(missing)} (see .env.example)")


# --- LLM (the brain) ---
LLM_API_KEY = _env("LLM_API_KEY")
LLM_BASE_URL = _env("LLM_BASE_URL")
LLM_MODEL = _env("LLM_MODEL")
# Optional, comma-separated: tried in order when the main model's daily budget runs out. Empty = none.
LLM_FALLBACK_MODELS = [m.strip() for m in _env("LLM_FALLBACK_MODELS").split(",") if m.strip()]
# Optional. "json_schema" (strict) when empty; "json_object" or "none" if the provider rejects json_schema.
LLM_JSON_MODE = _env("LLM_JSON_MODE", "json_schema")
# Optional. Sent as reasoning_effort only when set.
LLM_REASONING_EFFORT = _env("LLM_REASONING_EFFORT")

# --- Speech-to-text (voice agent). Key and URL default to the LLM ones. ---
STT_API_KEY = _env("STT_API_KEY", LLM_API_KEY)
STT_BASE_URL = _env("STT_BASE_URL", LLM_BASE_URL)
STT_MODEL = _env("STT_MODEL")
STT_LANGUAGE = _env("STT_LANGUAGE")  # optional

# --- Text-to-speech (voice agent). Key and URL default to the LLM ones. ---
TTS_API_KEY = _env("TTS_API_KEY", LLM_API_KEY)
TTS_BASE_URL = _env("TTS_BASE_URL", LLM_BASE_URL)
TTS_MODEL = _env("TTS_MODEL")
TTS_VOICE = _env("TTS_VOICE")
TTS_RESPONSE_FORMAT = _env("TTS_RESPONSE_FORMAT", "wav")  # audio format requested from the provider (wav, mp3, ...)
TTS_SAMPLE_RATE = int(_env("TTS_SAMPLE_RATE", "24000"))  # rate the agent plays at; the audio is resampled to it
