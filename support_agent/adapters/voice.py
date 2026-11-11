"""Voice agent: LiveKit Agents around the same brain the CLI uses.

Run: python -m support_agent.adapters.voice console --input-device <mic> --output-device <headphones>
Pipeline: mic -> VAD -> STT -> brain -> TTS -> speaker. Providers come from .env (see config.py).
"""
import asyncio
import re

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli, llm
from livekit.plugins import openai, silero

from support_agent.brain import Decision, respond, respond_with_fallback
from support_agent import config
from support_agent.adapters.speech import SpeechTTS
from support_agent.transcripts import ConversationLog

GREETING = "Hi, thanks for calling. How can I help you today?"
FALLBACK = "Sorry, I'm having trouble right now. Let me connect you to a colleague."
WAIT_LINE = "One moment please."
SLOW_AFTER = 4.0  # seconds of silence from the brain before "One moment please."
GIVE_UP_AFTER = 25.0  # then apologise and end the call


def speakable(text: str) -> str:
    """Make text safe to speak: digit runs read one by one, links and markdown removed."""
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"[*_#`>]+", "", text)
    return re.sub(r"\d{4,}", lambda m: ", ".join(m.group()), text).strip()


def last_user_text(chat_ctx) -> str:
    """The caller's latest words as transcribed."""
    for item in reversed(chat_ctx.items):
        if getattr(item, "role", None) == "user":
            return (item.text_content or "").strip()
    return ""


class BrainPlaceholderLLM(llm.LLM):
    """Stand-in LLM: LiveKit silently drops caller turns when a session has none. Never called; llm_node replaces it."""

    @property
    def model(self) -> str:
        return "brain"

    def chat(self, **kwargs):
        raise RuntimeError("BrainPlaceholderLLM must never be called; SupportAgent.llm_node bypasses it")


class SupportAgent(Agent):
    def __init__(self, model: str = config.LLM_MODEL, fallbacks: list[str] | None = None) -> None:
        super().__init__(instructions="unused: the system prompt lives in the brain (brain/prompt.py)")
        self.models = [model] + (config.LLM_FALLBACK_MODELS if fallbacks is None else list(fallbacks))
        self.history = []
        self.log = ConversationLog("voice", model=model)

    async def on_enter(self) -> None:
        self.session.say(GREETING)

    def note_interrupted(self, heard: str) -> None:
        """The caller talked over the agent: keep only what they heard in the history and the transcript."""
        for i in range(len(self.history) - 1, -1, -1):
            if self.history[i]["role"] == "assistant":
                try:
                    d = Decision.model_validate_json(self.history[i]["content"])
                    self.history[i] = {"role": "assistant", "content": d.model_copy(update={"reply": heard}).model_dump_json()}
                except ValueError:
                    pass
                break
        self.log.note(f"AGENT was INTERRUPTED by the caller. Caller heard only: {heard!r}")

    def _respond_with_fallback(self, text: str, on_filler):
        """Runs in a worker thread; switches model for the rest of the call if one runs out of daily budget."""
        result, history, self.models = respond_with_fallback(
            self.history, text, self.models, on_filler,
            on_switch=lambda old, new: self.log.note(f"model {old} is out of free daily budget; switching to {new} for the rest of the call"),
            _respond=respond)
        return result, history

    async def llm_node(self, chat_ctx, tools, model_settings):
        text = last_user_text(chat_ctx)
        if not text:
            return
        loop = asyncio.get_running_loop()
        fillers: asyncio.Queue[str] = asyncio.Queue()
        work = asyncio.ensure_future(asyncio.to_thread(
            self._respond_with_fallback, text,
            lambda t: loop.call_soon_threadsafe(fillers.put_nowait, t)))
        started, said_wait = loop.time(), False
        while not work.done() or not fillers.empty():
            try:
                yield speakable(await asyncio.wait_for(fillers.get(), timeout=0.05)) + " "
                continue
            except asyncio.TimeoutError:
                pass
            waited = loop.time() - started
            if waited > GIVE_UP_AFTER:
                break
            if waited > SLOW_AFTER and not said_wait:
                said_wait = True
                yield WAIT_LINE + " "
        try:
            if not work.done():
                raise TimeoutError(f"brain did not answer within {GIVE_UP_AFTER:.0f} s")
            result, self.history = work.result()
        except Exception as e:
            self.log.note(f"CALLER said: {text!r} -> brain failed: {type(e).__name__}: {str(e)[:200]}")
            self.log.end("error")
            yield FALLBACK
            self.session.shutdown()
            return
        self.log.turn(text, result)
        yield speakable(result.decision.reply)
        if result.decision.action == "handoff":
            self.log.end("handoff")
            self.session.shutdown()


def build_stt():
    """Any OpenAI-compatible transcription endpoint (Groq, OpenAI, a local server...)."""
    config.require(STT_API_KEY=config.STT_API_KEY, STT_BASE_URL=config.STT_BASE_URL, STT_MODEL=config.STT_MODEL)
    extra = {"language": config.STT_LANGUAGE} if config.STT_LANGUAGE else {}
    return openai.STT(model=config.STT_MODEL, api_key=config.STT_API_KEY, base_url=config.STT_BASE_URL,
                      use_realtime=False, **extra,
                      prompt="A customer support call about a money transfer: where is my money, transfer number, proof of payment.")


def build_tts():
    """Any OpenAI-compatible /audio/speech endpoint."""
    config.require(TTS_API_KEY=config.TTS_API_KEY, TTS_BASE_URL=config.TTS_BASE_URL,
                   TTS_MODEL=config.TTS_MODEL, TTS_VOICE=config.TTS_VOICE)
    return SpeechTTS(model=config.TTS_MODEL, voice=config.TTS_VOICE, api_key=config.TTS_API_KEY,
                     base_url=config.TTS_BASE_URL, response_format=config.TTS_RESPONSE_FORMAT, sample_rate=config.TTS_SAMPLE_RATE)


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    agent = SupportAgent()
    session = AgentSession(
        vad=silero.VAD.load(),
        llm=BrainPlaceholderLLM(),
        stt=build_stt(),
        tts=build_tts(),
        turn_handling={
            "endpointing": {"min_delay": 0.6, "max_delay": 2.0},
            "preemptive_generation": {"enabled": False},  # would call the brain on a guess, before the caller finishes
        },
    )

    @session.on("conversation_item_added")
    def _on_item(ev) -> None:
        item = ev.item
        if getattr(item, "role", None) == "assistant" and getattr(item, "interrupted", False):
            agent.note_interrupted((item.text_content or "").strip())

    @session.on("close")
    def _on_close(_ev) -> None:
        agent.log.end("caller_left")

    await session.start(agent=agent, room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
