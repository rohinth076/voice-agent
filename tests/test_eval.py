"""Offline tests for the eval scoring and cache keys. No API calls."""
import unittest

from scripts.eval import cache_key, metrics


def row(type_, label, got):
    return {"id": "x", "q": "q", "type": type_, "label": label, "got": got, "reason": "", "reply": ""}


class MetricTests(unittest.TestCase):
    def test_near_miss_answered_is_counted(self):
        rows = [row("near_miss", "handoff", "answer"), row("near_miss", "handoff", "handoff"), row("near_miss", "handoff", "clarify")]
        self.assertEqual(metrics(rows)["near_miss_wrongly_answered"][:2], (1, 3))

    def test_on_topic_and_wrong_handoff(self):
        rows = [row("on_topic", "answer", "answer"), row("on_topic", "answer", "handoff")]
        m = metrics(rows)
        self.assertEqual(m["on_topic_answered"][:2], (1, 2))
        self.assertEqual(m["wrong_handoff_on_topic"][:2], (1, 2))

    def test_human_honoured(self):
        rows = [row("human_request", "handoff", "handoff"), row("human_request", "handoff", "answer")]
        self.assertEqual(metrics(rows)["human_honoured"][:2], (1, 2))


class CacheKeyTests(unittest.TestCase):
    def test_key_changes_with_prompt_model_or_question(self):
        base = cache_key("m1", "fpA", "hello")
        self.assertEqual(base, cache_key("m1", "fpA", "hello"))
        self.assertNotEqual(base, cache_key("m1", "fpB", "hello"))
        self.assertNotEqual(base, cache_key("m2", "fpA", "hello"))
        self.assertNotEqual(base, cache_key("m1", "fpA", "bye"))


if __name__ == "__main__":
    unittest.main()


class TestSafeName(unittest.TestCase):
    def test_model_names_become_valid_file_names(self):
        import scripts.eval as ev
        self.assertEqual(ev.safe_name("openai/gpt-oss-120b"), "openai_gpt-oss-120b")
        self.assertEqual(ev.safe_name("gemma-4-26b-a4b-it"), "gemma-4-26b-a4b-it")


class TestCodeSignature(unittest.TestCase):
    def test_comments_and_docstrings_do_not_change_it_but_code_does(self):
        import inspect
        from unittest.mock import patch

        from scripts.eval import code_signature

        a = '"""doc"""\n# note\nx = 1\n\n\ndef f():\n    """why"""\n    return x  # trailing\n'
        b = 'x = 1\n\n\ndef f():\n    return x\n'
        c = 'x = 2\n\n\ndef f():\n    return x\n'
        with patch.object(inspect, "getsource", side_effect=[a, b, c]):
            sa, sb, sc = code_signature(None), code_signature(None), code_signature(None)
        self.assertEqual(sa, sb)
        self.assertNotEqual(sb, sc)
