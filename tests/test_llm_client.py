"""Offline tests for the OpenAI-compatible adapter (httpx is faked: no network, no key spent)."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from support_agent.brain import llm_client
from support_agent.brain.llm_client import ProviderError


def resp(status, text="", content=None):
    return SimpleNamespace(status_code=status, text=text,
                           json=lambda: {"choices": [{"message": {"content": content}}]})


HIST = [{"role": "user", "content": "hi"}]
BAD = resp(400, '{"error":{"code":"tool_use_failed"}}')


class TestLlmClient(unittest.TestCase):
    def setUp(self):
        p = patch.multiple(llm_client, LLM_API_KEY="test-key", LLM_BASE_URL="https://example.test/v1",
                           LLM_REASONING_EFFORT="", LLM_JSON_MODE="json_schema")
        p.start()
        self.addCleanup(p.stop)

    def test_retries_when_the_model_tries_to_call_a_tool_we_did_not_offer(self):
        with patch.object(llm_client.httpx, "post", side_effect=[BAD, BAD, resp(200, content='{"ok": 1}')]) as post:
            out = llm_client.generate(HIST, "sys", "openai/gpt-oss-20b", {})
        self.assertEqual(out, '{"ok": 1}')
        self.assertEqual(post.call_count, 3)

    def test_gives_up_after_three_tries(self):
        with patch.object(llm_client.httpx, "post", side_effect=[BAD, BAD, BAD]) as post:
            with self.assertRaises(ProviderError) as cm:
                llm_client.generate(HIST, "sys", "openai/gpt-oss-20b", {})
        self.assertEqual(cm.exception.code, 400)
        self.assertEqual(post.call_count, 3)

    def test_other_errors_are_not_retried_here_and_keep_their_code(self):
        with patch.object(llm_client.httpx, "post", side_effect=[resp(429, "rate limited")]) as post:
            with self.assertRaises(ProviderError) as cm:
                llm_client.generate(HIST, "sys", "openai/gpt-oss-120b", {})
        self.assertEqual(cm.exception.code, 429)
        self.assertEqual(post.call_count, 1)

    def test_history_is_sent_after_the_system_prompt(self):
        hist = HIST + [{"role": "assistant", "content": "hello"}]
        msgs = llm_client._messages(hist, "SYS")
        self.assertEqual([m["role"] for m in msgs], ["system", "user", "assistant"])

    def test_reasoning_effort_is_not_sent_unless_configured(self):
        with patch.object(llm_client.httpx, "post", return_value=resp(200, content="{}")) as post:
            llm_client.generate(HIST, "sys", "openai/gpt-oss-20b", {})
        self.assertNotIn("reasoning_effort", post.call_args.kwargs["json"])

    def test_missing_settings_are_named_in_the_error(self):
        with patch.object(llm_client, "LLM_BASE_URL", ""), patch.object(llm_client, "LLM_API_KEY", ""):
            with self.assertRaises(ProviderError) as cm:
                llm_client.generate(HIST, "sys", "m", {})
        self.assertIn("LLM_API_KEY", str(cm.exception))
        self.assertIn("LLM_BASE_URL", str(cm.exception))

    def test_url_and_key_come_from_config_so_any_provider_works(self):
        with patch.object(llm_client, "LLM_BASE_URL", "https://example.test/v1/"), \
                patch.object(llm_client.httpx, "post", return_value=resp(200, content="{}")) as post:
            llm_client.generate(HIST, "sys", "some-model", {})
        self.assertEqual(post.call_args.args[0], "https://example.test/v1/chat/completions")
        self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer test-key")

    def test_json_mode_changes_how_the_schema_is_requested(self):
        schema = {"title": "S"}
        with patch.object(llm_client.httpx, "post", return_value=resp(200, content="{}")) as post:
            llm_client.generate(HIST, "sys", "m", schema)
            with patch.object(llm_client, "LLM_JSON_MODE", "json_object"):
                llm_client.generate(HIST, "sys", "m", schema)
            with patch.object(llm_client, "LLM_JSON_MODE", "none"):
                llm_client.generate(HIST, "sys", "m", schema)
        strict, plain, none = (c.kwargs["json"] for c in post.call_args_list)
        self.assertEqual(strict["response_format"]["type"], "json_schema")
        self.assertEqual(plain["response_format"], {"type": "json_object"})
        self.assertNotIn("response_format", none)
        self.assertIn('"title": "S"', plain["messages"][0]["content"])  # schema described in the prompt instead
        self.assertNotIn('"title"', strict["messages"][0]["content"])

    def test_reasoning_effort_can_be_set_from_config(self):
        with patch.object(llm_client, "LLM_REASONING_EFFORT", "minimal"), \
                patch.object(llm_client.httpx, "post", return_value=resp(200, content="{}")) as post:
            llm_client.generate(HIST, "sys", "gpt-5-mini", {})
        self.assertEqual(post.call_args.kwargs["json"]["reasoning_effort"], "minimal")


if __name__ == "__main__":
    unittest.main()
