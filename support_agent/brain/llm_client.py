"""Chat adapter for any OpenAI-compatible endpoint (LLM_BASE_URL + LLM_API_KEY from .env).

History is a list of {"role": "user" | "assistant", "content": str} dicts, the same shape the API uses.
"""
import json

import httpx

from support_agent.config import LLM_API_KEY, LLM_BASE_URL, LLM_JSON_MODE, LLM_REASONING_EFFORT, require


class ProviderError(Exception):
    """An HTTP error from the LLM provider. Has .code (the HTTP status)."""

    def __init__(self, code: int, message: str):
        super().__init__(f"{code} {message}")
        self.code = code


def _messages(history: list[dict], system_prompt: str) -> list[dict]:
    return [{"role": "system", "content": system_prompt}] + [{"role": m["role"], "content": m["content"]} for m in history]


def generate(history: list[dict], system_prompt: str, model: str, schema: dict) -> str:
    try:
        require(LLM_API_KEY=LLM_API_KEY, LLM_BASE_URL=LLM_BASE_URL, LLM_MODEL=model)
    except ValueError as e:
        raise ProviderError(401, str(e)) from e
    if LLM_JSON_MODE != "json_schema":  # the provider cannot enforce a schema, so describe it in the prompt
        system_prompt += "\n\nReply with a single JSON object matching this JSON schema:\n" + json.dumps(schema)
    body = {
        "model": model,
        "messages": _messages(history, system_prompt),
        "temperature": 0.2,
        "max_completion_tokens": 800,
    }
    if LLM_JSON_MODE == "json_schema":
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "decision", "schema": schema}}
    elif LLM_JSON_MODE == "json_object":
        body["response_format"] = {"type": "json_object"}
    if LLM_REASONING_EFFORT:
        body["reasoning_effort"] = LLM_REASONING_EFFORT
    url = LLM_BASE_URL.rstrip("/") + "/chat/completions"
    for attempt in range(3):
        try:
            r = httpx.post(url, json=body, headers={"Authorization": f"Bearer {LLM_API_KEY}"}, timeout=30)
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
