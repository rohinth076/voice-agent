"""Offline tests for brain/core.py helpers (no network)."""
import unittest

from support_agent.brain.core import Decision, _fill_empty_reply


class TestFillEmptyReply(unittest.TestCase):
    def test_nonempty_reply_is_untouched(self):
        d = Decision(reason="r", action="answer", reply="Sure.")
        self.assertEqual(_fill_empty_reply(d), d)

    def test_empty_handoff_gets_a_spoken_line(self):
        d = _fill_empty_reply(Decision(reason="r", action="handoff", reply=""))
        self.assertEqual(d.action, "handoff")
        self.assertIn("colleague", d.reply)

    def test_empty_clarify_gets_a_question(self):
        self.assertTrue(_fill_empty_reply(Decision(reason="r", action="clarify", reply="  ")).reply.endswith("?"))

    def test_empty_use_tool_depends_on_whether_a_number_is_confirmed(self):
        ask = _fill_empty_reply(Decision(reason="r", action="use_tool", reply=""))
        check = _fill_empty_reply(Decision(reason="r", action="use_tool", transfer_id="12345", reply=""))
        self.assertIn("transfer number", ask.reply)
        self.assertIn("check", check.reply)

    def test_empty_answer_becomes_a_handoff_rather_than_silence(self):
        d = _fill_empty_reply(Decision(reason="r", action="answer", reply=""))
        self.assertEqual(d.action, "handoff")
        self.assertTrue(d.reply)


if __name__ == "__main__":
    unittest.main()


class TestRespondWithFallback(unittest.TestCase):
    def test_moves_to_the_next_model_on_daily_quota_and_reports_who_answered(self):
        from support_agent.brain.core import DailyQuotaExceeded, respond_with_fallback
        seen, switches = [], []

        def fake(history, text, model, on_filler=None):
            seen.append(model)
            if model != "c":
                raise DailyQuotaExceeded(model)
            return "RESULT", history + ["x"]
        result, hist, left = respond_with_fallback([], "hi", ["a", "b", "c", "d"], _respond=fake,
                                                   on_switch=lambda o, n: switches.append((o, n)))
        self.assertEqual((result, seen, left), ("RESULT", ["a", "b", "c"], ["c", "d"]))
        self.assertEqual(switches, [("a", "b"), ("b", "c")])

    def test_raises_when_the_last_model_is_also_out(self):
        from support_agent.brain.core import DailyQuotaExceeded, respond_with_fallback

        def fake(history, text, model, on_filler=None):
            raise DailyQuotaExceeded(model)
        with self.assertRaises(DailyQuotaExceeded):
            respond_with_fallback([], "hi", ["a", "b"], _respond=fake)
