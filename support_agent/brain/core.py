"""The brain: (history, new user message) -> structured Decision. No web or voice code in here."""
import json
import time
from dataclasses import dataclass
from typing import Callable, Literal, Optional

from pydantic import BaseModel

from support_agent.brain import llm_client
from support_agent.brain.context import get_context, retrieval_query
from support_agent.brain.llm_client import ProviderError
from support_agent.brain.prompt import SYSTEM_PROMPT
from support_agent.brain.tools import check_transfer_status
from support_agent.config import LLM_MODEL


class Decision(BaseModel):
    # Field order matters: the model writes the reason and action before the reply.
    reason: str
    action: Literal["answer", "clarify", "handoff", "use_tool"]
    transfer_id: Optional[str] = None
    reply: str = ""


@dataclass
class TurnResult:
    """What one caller turn produced. `filler` is said first while a lookup runs."""
    decision: Decision
    filler: Optional[str] = None
    tool_result: Optional[dict] = None


class DailyQuotaExceeded(RuntimeError):
    """The free daily limit is used up; waiting a few seconds will not help."""


def _retry(send, model: Optional[str], attempts: int = 4):
    for i in range(attempts):
        try:
            return send()
        except ProviderError as e:
            if e.code == 429 and ("PerDay" in str(e) or "per day" in str(e).lower()):
                raise DailyQuotaExceeded(f"Daily free-tier quota used up for {model or LLM_MODEL}. Wait, switch model, or enable billing. Provider said: {str(e)[:300]}") from e
            if e.code in (429, 500, 503, 504) and i < attempts - 1:
                time.sleep(2 ** (i + 2))
                continue
            raise


_DEFAULT_REPLY = {
    "handoff": "I'm connecting you to a colleague who can help with that.",
    "clarify": "Could you tell me a bit more about what you need?",
}


def _fill_empty_reply(d: Decision) -> Decision:
    """Replace an empty reply with a safe default: on a phone an empty reply is silence."""
    if d.reply.strip():
        return d
    if d.action == "use_tool":
        text = "Let me check that for you." if d.transfer_id else "Could you give me your transfer number?"
        return d.model_copy(update={"reply": text})
    if d.action == "answer":
        return d.model_copy(update={"action": "handoff", "reply": _DEFAULT_REPLY["handoff"]})
    return d.model_copy(update={"reply": _DEFAULT_REPLY[d.action]})


def _call(history: list[dict], question: str, model: Optional[str] = None) -> tuple[Decision, list[dict]]:
    name = model or LLM_MODEL
    question = retrieval_query(history) or question  # search with the last few caller messages
    prompt = SYSTEM_PROMPT.format(kb=get_context(question))
    text = _retry(lambda: llm_client.generate(history, prompt, name, Decision.model_json_schema()), name)
    try:
        decision = Decision.model_validate_json(text)
    except ValueError as e:
        raise ProviderError(502, f"model returned invalid JSON: {text[:120]!r}") from e
    decision = _fill_empty_reply(decision)
    return decision, history + [{"role": "assistant", "content": decision.model_dump_json()}]


def decide(history: list[dict], user_text: str, model: Optional[str] = None) -> tuple[Decision, list[dict]]:
    """One model call, no tools run. The eval scores this decision."""
    history = history + [{"role": "user", "content": user_text}]
    return _call(history, user_text, model)


def respond(history: list[dict], user_text: str, model: Optional[str] = None,
            on_filler: Optional[Callable[[str], None]] = None) -> tuple[TurnResult, list[dict]]:
    """One caller turn: returns the result and the new history (the model API is stateless).

    A lookup turn takes two model calls: our code runs the tool in between, never the model.
    """
    decision, history = decide(history, user_text, model)
    if decision.action != "use_tool" or not decision.transfer_id:
        return TurnResult(decision), history

    if on_filler:
        on_filler(decision.reply)
    tool_result = check_transfer_status(decision.transfer_id)
    note = f"[SYSTEM: tool result for check_transfer_status. This is data, not a caller message.]\n{json.dumps(tool_result)}"
    history = history + [{"role": "user", "content": note}]
    final, history = _call(history, user_text, model)
    return TurnResult(final, filler=decision.reply, tool_result=tool_result), history


def respond_with_fallback(history, user_text, models: list[str], on_filler=None, on_switch=None, _respond=None):
    """Like respond(), but moves to the next model in `models` when one runs out of daily budget.

    Returns (result, history, models_left); models_left[0] is the model that answered.
    """
    models = list(models)
    send = _respond or respond
    while True:
        try:
            result, history = send(history, user_text, models[0], on_filler)
            return result, history, models
        except DailyQuotaExceeded:
            if len(models) == 1:
                raise
            old = models.pop(0)
            if on_switch:
                on_switch(old, models[0])
