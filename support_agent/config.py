"""Settings and API keys. Keys come from .env (see .env.example)."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Model names with a "groq:" prefix go to Groq; anything else goes to the Gemini API.
VOICE_MODEL = "groq:openai/gpt-oss-120b"
MODEL = VOICE_MODEL
GEMINI_MODEL = "gemini-3.6-flash"

# Gemini only: thinking tokens count against the output limit, so short spoken replies run with none.
THINKING_BUDGET = 0

# Tried in order when a model's free daily budget runs out. gpt-oss-20b handles handoffs but not the transfer lookup.
VOICE_FALLBACK_MODELS = [m for m in [GEMINI_MODEL, "groq:openai/gpt-oss-20b"] if GEMINI_API_KEY or m.startswith("groq:")]
