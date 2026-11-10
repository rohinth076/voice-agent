"""Offline tests for transcripts. No API calls. Uses a temp folder."""
import tempfile
import unittest
from pathlib import Path

from support_agent.brain import Decision, TurnResult
from support_agent.transcripts import ConversationLog


def result(action="answer", reply="Hi.", reason="because", filler=None, tool=None):
    return TurnResult(Decision(reason=reason, action=action, reply=reply), filler=filler, tool_result=tool)


class TranscriptTests(unittest.TestCase):
    def test_no_file_for_empty_call(self):
        with tempfile.TemporaryDirectory() as d:
            log = ConversationLog("cli", Path(d))
            log.end("caller_left")
            self.assertFalse(log.path.exists())

    def test_turns_are_written_in_order_and_flushed(self):
        with tempfile.TemporaryDirectory() as d:
            log = ConversationLog("cli", Path(d))
            log.turn("where is my money?", result(reply="Which transfer number?"))
            text = log.path.read_text()
            self.assertIn("CALLER: where is my money?", text)
            self.assertIn("AGENT : Which transfer number?", text)
            self.assertIn("action=answer | reason: because", text)

    def test_tool_filler_handoff_and_end(self):
        with tempfile.TemporaryDirectory() as d:
            log = ConversationLog("cli", Path(d))
            log.turn("yes", result(filler="Let me check.", tool={"status": "Transfer Sent"}))
            log.turn("cancel it", result(action="handoff", reply="Connecting you.", reason="wants to cancel"))
            log.end("handoff")
            log.end("handoff")
            text = log.path.read_text()
            self.assertIn("AGENT : Let me check.", text)
            self.assertIn("Transfer Sent", text)
            self.assertIn("HANDED OFF TO A HUMAN AGENT. Summary: wants to cancel", text)
            self.assertEqual(text.count("=== Ended"), 1)
            self.assertIn("after 2 turn(s)", text)


if __name__ == "__main__":
    unittest.main()
