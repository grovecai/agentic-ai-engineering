"""Offline service tests for tool grounding, cache, and usage accounting."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx2
from openai import APITimeoutError

from service import Settings, SupportService, authorize


class Message:
    def __init__(self, content=None, calls=None):
        self.content = content
        self.tool_calls = calls

    def model_dump(self, exclude_none=True):
        return {
            "role": "assistant",
            "content": self.content,
            "tool_calls": [
                {
                    "id": c.id,
                    "function": {"name": c.function.name, "arguments": c.function.arguments},
                }
                for c in self.tool_calls or []
            ],
        }


class Client:
    def __init__(self, messages):
        self.messages = iter(messages)
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(
            choices=[SimpleNamespace(message=next(self.messages))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )


class ServiceTests(unittest.TestCase):
    def test_transient_timeout_is_retried_once(self):
        client = Client([Message(content="ok")])
        original = client.create
        attempts = 0

        def flaky(**kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise APITimeoutError(request=httpx2.Request("POST", "https://api.example.test"))
            return original(**kwargs)

        client.chat.completions.create = flaky
        service = SupportService(Settings(api_key="fake", max_attempts=2), client)
        with patch("service.time.sleep") as sleep:
            self.assertEqual(
                service._complete([{"role": "user", "content": "hi"}], None).content, "ok"
            )
        self.assertEqual(attempts, 2)
        sleep.assert_called_once()

    def test_application_key(self):
        with self.assertRaises(PermissionError):
            authorize("wrong", "expected")
        with self.assertRaises(RuntimeError):
            authorize("anything", None)
        authorize("expected", "expected")

    def test_cache_stops_duplicate_api_calls(self):
        call = SimpleNamespace(
            id="one",
            function=SimpleNamespace(name="lookup_policy", arguments='{"policy_id":"refund"}'),
        )
        client = Client([Message(calls=[call]), Message(content="Within 30 days [refund].")])
        service = SupportService(Settings(api_key="fake"), client)
        first = service.answer("Can I get a refund?", use_cache=True)
        second = service.answer("Can I get a refund?", use_cache=True)
        self.assertFalse(first["cached"])
        self.assertTrue(second["cached"])
        self.assertEqual(client.calls, 2)
        self.assertEqual(service.metrics["input_tokens"], 20)

    def test_input_limit_and_required_tool(self):
        client = Client([Message(content="Unsupported answer")])
        service = SupportService(Settings(api_key="fake", max_input_chars=10), client)
        with self.assertRaises(ValueError):
            service.answer("A" * 11)
        with self.assertRaisesRegex(RuntimeError, "required policy lookup"):
            service.answer("Hi")


if __name__ == "__main__":
    unittest.main()
