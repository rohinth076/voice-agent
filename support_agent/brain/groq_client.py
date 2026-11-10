"""Groq chat adapter. Model names look like 'groq:openai/gpt-oss-120b'."""
import httpx
from google.genai import types

from support_agent.config import GROQ_API_KEY

URL = "https://api.groq.com/openai/v1/chat/completions"


class ProviderError(Exception):
    """An HTTP error from Groq. Has .code like the Gemini client errors."""

    def __init__(self, code: int, message: str):
        super().__init__(f"{code} {message}")
        self.code = code


def _messages(history: list[types.Content], system_prompt: str) -> list[dict]:
    msgs = [{"role": "system", "content": system_prompt}]
    for c in history:
        msgs.append({"role": "assistant" if c.role == "model" else "user", "content": "".join(p.text or "" for p in c.parts)})
    return msgs


def generate(history: list[types.Content], system_prompt: str, model: str, schema: dict) -> str:
    if not GROQ_API_KEY:
        raise ProviderError(401, "GROQ_API_KEY is not set in .env")
    body = {
        "model": model,
        "messages": _messages(history, system_prompt),
        "temperature": 0.2,
        "max_completion_tokens": 800,
        "response_format": {"type": "json_schema", "json_schema": {"name": "decision", "schema": schema}},
    }
    if "gpt-oss" in model:
        body["reasoning_effort"] = "low"
    for attempt in range(3):
        try:
            r = httpx.post(URL, json=body, headers={"Authorization": f"Bearer {GROQ_API_KEY}"}, timeout=30)
        except httpx.TimeoutException as e:
            raise ProviderError(504, "request timed out") from e
        except httpx.HTTPError as e:
            raise ProviderError(503, f"network error: {type(e).__name__}") from e
        # gpt-oss sometimes tries to call a tool we never offered; a retry usually returns plain JSON.
        if r.status_code == 400 and "tool_use_failed" in r.text and attempt < 2:
            continue
        break
    if r.status_code != 200:
        raise ProviderError(r.status_code, r.text[:300])
    return r.json()["choices"][0]["message"]["content"]
