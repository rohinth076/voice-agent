"""Offline tests for the voice adapter's logic (no microphone, no network, no LiveKit session)."""
import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

from support_agent.adapters import voice as voice_agent
from support_agent.brain import Decision, TurnResult
from support_agent.transcripts import ConversationLog


def fake_ctx(*pairs):
    return SimpleNamespace(items=[SimpleNamespace(role=r, text_content=t) for r, t in pairs])


async def collect(agent, ctx):
    return [chunk async for chunk in agent.llm_node(ctx, [], None)]


class AgentCase(unittest.TestCase):
    """Shared setup only (no tests of its own): a SupportAgent with a fake session and a temp transcript folder."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.shutdowns = []
        session = SimpleNamespace(shutdown=lambda: self.shutdowns.append(1), say=lambda t: None)
        p = patch.object(voice_agent.SupportAgent, "session", new_callable=PropertyMock, return_value=session)
        p.start()
        self.addCleanup(p.stop)
        self.agent = voice_agent.SupportAgent(model="fake", fallbacks=[])
        self.agent.log = ConversationLog("voice", log_dir=Path(self.tmp.name), model="fake")

    def text(self):
        return "".join(p.read_text() for p in Path(self.tmp.name).glob("*.txt"))


class TestVoiceAgent(AgentCase):
    def test_last_user_text_picks_newest_user_message(self):
        ctx = fake_ctx(("user", "first"), ("assistant", "hi"), ("user", " latest "))
        self.assertEqual(voice_agent.last_user_text(ctx), "latest")

    def test_plain_answer_is_spoken_and_logged(self):
        def fake(history, text, model, on_filler=None):
            return TurnResult(Decision(reason="r", action="answer", reply="Sure.")), history + ["x"]
        with patch.object(voice_agent, "respond", fake):
            out = asyncio.run(collect(self.agent, fake_ctx(("user", "hello"))))
        self.assertEqual(out, ["Sure."])
        self.assertEqual(self.shutdowns, [])
        self.assertIn("CALLER: hello", self.text())

    def test_filler_is_spoken_before_the_final_answer_and_both_are_logged(self):
        def fake(history, text, model, on_filler=None):
            on_filler("Let me check.")
            return (TurnResult(Decision(reason="r", action="answer", reply="It is sent."), filler="Let me check.",
                               tool_result={"status": "sent"}), history)
        with patch.object(voice_agent, "respond", fake):
            out = asyncio.run(collect(self.agent, fake_ctx(("user", "yes"))))
        self.assertEqual([c.strip() for c in out], ["Let me check.", "It is sent."])
        log = self.text()
        self.assertIn("Let me check.", log)
        self.assertIn("It is sent.", log)

    def test_handoff_speaks_then_shuts_the_session_down(self):
        def fake(history, text, model, on_filler=None):
            return TurnResult(Decision(reason="wants a person", action="handoff", reply="Connecting you.")), history
        with patch.object(voice_agent, "respond", fake):
            out = asyncio.run(collect(self.agent, fake_ctx(("user", "a human please"))))
        self.assertEqual(out, ["Connecting you."])
        self.assertEqual(self.shutdowns, [1])
        self.assertIn("Ended (handoff)", self.text())

    def test_brain_failure_gives_a_spoken_fallback_not_silence(self):
        def boom(history, text, model, on_filler=None):
            raise RuntimeError("api down")
        with patch.object(voice_agent, "respond", boom):
            out = asyncio.run(collect(self.agent, fake_ctx(("user", "hello"))))
        self.assertEqual(out, [voice_agent.FALLBACK])
        self.assertEqual(self.shutdowns, [1])
        self.assertIn("brain failed", self.text())


if __name__ == "__main__":
    unittest.main()


class TestRealSession(unittest.TestCase):
    """Runs a whole turn through a REAL LiveKit AgentSession (text input, no audio). This is the test that would"""

    def test_caller_turn_reaches_our_brain_through_a_real_session(self):
        from livekit.agents import AgentSession

        calls = []

        def fake(history, text, model, on_filler=None):
            calls.append(text)
            return TurnResult(Decision(reason="r", action="answer", reply="Hello there.")), history

        async def go():
            with tempfile.TemporaryDirectory() as tmp, patch.object(voice_agent, "respond", fake):
                agent = voice_agent.SupportAgent(model="fake")
                agent.log = ConversationLog("voice", log_dir=Path(tmp), model="fake")
                async with AgentSession(llm=voice_agent.BrainPlaceholderLLM()) as session:
                    await session.start(agent)
                    await asyncio.wait_for(session.run(user_input="Where is my money?"), timeout=15)

        asyncio.run(go())
        self.assertEqual(calls, ["Where is my money?"])


if __name__ == "__main__":
    unittest.main()


class TestSlowBrain(AgentCase):
    def test_slow_brain_gets_a_one_moment_line(self):
        import time

        def slow(history, text, model, on_filler=None):
            time.sleep(0.4)
            return TurnResult(Decision(reason="r", action="answer", reply="Done.")), history
        with patch.object(voice_agent, "respond", slow), patch.object(voice_agent, "SLOW_AFTER", 0.1):
            out = asyncio.run(collect(self.agent, fake_ctx(("user", "hello"))))
        self.assertEqual([c.strip() for c in out], [voice_agent.WAIT_LINE, "Done."])

    def test_brain_that_never_answers_ends_with_the_fallback(self):
        import time

        def stuck(history, text, model, on_filler=None):
            time.sleep(0.6)
            return TurnResult(Decision(reason="r", action="answer", reply="late")), history
        with patch.object(voice_agent, "respond", stuck), patch.object(voice_agent, "GIVE_UP_AFTER", 0.2):
            out = asyncio.run(collect(self.agent, fake_ctx(("user", "hello"))))
        self.assertEqual(out[-1], voice_agent.FALLBACK)
        self.assertEqual(self.shutdowns, [1])
        self.assertIn("did not answer", self.text())


class TestSpeakableAndInterruption(AgentCase):
    def test_digits_are_read_one_by_one_and_links_dropped(self):
        self.assertEqual(voice_agent.speakable("Transfer 12345 is done."), "Transfer 1, 2, 3, 4, 5 is done.")
        self.assertEqual(voice_agent.speakable("See https://wise.com/help now"), "See  now")
        self.assertEqual(voice_agent.speakable("It takes 2 days, up to 10."), "It takes 2 days, up to 10.")
        self.assertEqual(voice_agent.speakable("**Bold** `x`"), "Bold x")

    def test_speaks_digits_apart_but_logs_the_original(self):
        def fake(history, text, model, on_filler=None):
            return TurnResult(Decision(reason="r", action="answer", reply="Your number is 12345.")), history
        with patch.object(voice_agent, "respond", fake):
            out = asyncio.run(collect(self.agent, fake_ctx(("user", "hi"))))
        self.assertEqual(out, ["Your number is 1, 2, 3, 4, 5."])
        self.assertIn("Your number is 12345.", self.text())

    def test_interruption_trims_history_to_what_was_heard_and_is_logged(self):
        from google.genai import types
        full = Decision(reason="r", action="answer", reply="Your transfer is on its way and should arrive soon.")
        self.agent.history = [types.Content(role="user", parts=[types.Part(text="hi")]),
                              types.Content(role="model", parts=[types.Part(text=full.model_dump_json())])]
        self.agent.note_interrupted("Your transfer is on its")
        kept = Decision.model_validate_json(self.agent.history[-1].parts[0].text)
        self.assertEqual(kept.reply, "Your transfer is on its")
        self.assertEqual(kept.action, "answer")
        self.assertIn("INTERRUPTED", self.text())
        self.assertIn("Your transfer is on its", self.text())


class TestModelFallback(AgentCase):
    def test_switches_to_the_next_model_when_the_daily_budget_is_gone(self):
        from support_agent.brain import DailyQuotaExceeded
        seen = []

        def fake(history, text, model, on_filler=None):
            seen.append(model)
            if model == "primary":
                raise DailyQuotaExceeded("out")
            return TurnResult(Decision(reason="r", action="answer", reply="From the backup.")), history
        self.agent.models = ["primary", "backup"]
        with patch.object(voice_agent, "respond", fake):
            out = asyncio.run(collect(self.agent, fake_ctx(("user", "hello"))))
        self.assertEqual(seen, ["primary", "backup"])
        self.assertEqual(out, ["From the backup."])
        self.assertEqual(self.agent.models[0], "backup")
        self.assertIn("switching to backup", self.text())

    def test_no_fallback_left_ends_with_the_spoken_apology(self):
        from support_agent.brain import DailyQuotaExceeded

        def fake(history, text, model, on_filler=None):
            raise DailyQuotaExceeded("out")
        self.agent.models = ["only"]
        with patch.object(voice_agent, "respond", fake):
            out = asyncio.run(collect(self.agent, fake_ctx(("user", "hello"))))
        self.assertEqual(out, [voice_agent.FALLBACK])
        self.assertEqual(self.shutdowns, [1])
