"""Offline tests for the Groq adapter (httpx is faked: no network, no key spent)."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from google.genai import types

from support_agent.brain import groq_client
from support_agent.brain.groq_client import ProviderError


def resp(status, text="", content=None):
    return SimpleNamespace(status_code=status, text=text,
                           json=lambda: {"choices": [{"message": {"content": content}}]})


HIST = [types.Content(role="user", parts=[types.Part(text="hi")])]
BAD = resp(400, '{"error":{"code":"tool_use_failed"}}')


class TestGroqClient(unittest.TestCase):
    def setUp(self):
        p = patch.object(groq_client, "GROQ_API_KEY", "test-key")
        p.start()
        self.addCleanup(p.stop)

    def test_retries_when_the_model_tries_to_call_a_tool_we_did_not_offer(self):
        with patch.object(groq_client.httpx, "post", side_effect=[BAD, BAD, resp(200, content='{"ok": 1}')]) as post:
            out = groq_client.generate(HIST, "sys", "openai/gpt-oss-20b", {})
        self.assertEqual(out, '{"ok": 1}')
        self.assertEqual(post.call_count, 3)

    def test_gives_up_after_three_tries(self):
        with patch.object(groq_client.httpx, "post", side_effect=[BAD, BAD, BAD]) as post:
            with self.assertRaises(ProviderError) as cm:
                groq_client.generate(HIST, "sys", "openai/gpt-oss-20b", {})
        self.assertEqual(cm.exception.code, 400)
        self.assertEqual(post.call_count, 3)

    def test_other_errors_are_not_retried_here_and_keep_their_code(self):
        with patch.object(groq_client.httpx, "post", side_effect=[resp(429, "rate limited")]) as post:
            with self.assertRaises(ProviderError) as cm:
                groq_client.generate(HIST, "sys", "openai/gpt-oss-120b", {})
        self.assertEqual(cm.exception.code, 429)
        self.assertEqual(post.call_count, 1)

    def test_history_roles_are_translated_for_groq(self):
        hist = HIST + [types.Content(role="model", parts=[types.Part(text="hello")])]
        msgs = groq_client._messages(hist, "SYS")
        self.assertEqual([m["role"] for m in msgs], ["system", "user", "assistant"])

    def test_gpt_oss_gets_low_reasoning_effort_other_models_do_not(self):
        with patch.object(groq_client.httpx, "post", return_value=resp(200, content="{}")) as post:
            groq_client.generate(HIST, "sys", "openai/gpt-oss-20b", {})
            groq_client.generate(HIST, "sys", "qwen/qwen3.8-27b", {})
        self.assertEqual(post.call_args_list[0].kwargs["json"]["reasoning_effort"], "low")
        self.assertNotIn("reasoning_effort", post.call_args_list[1].kwargs["json"])


if __name__ == "__main__":
    unittest.main()
