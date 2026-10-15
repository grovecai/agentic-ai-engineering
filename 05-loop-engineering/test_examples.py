"""Offline boundary checks for the shared loop and compaction."""

import runpy
import unittest
from pathlib import Path
from types import SimpleNamespace

from harness import LOOKUP, Agent, Hooks


class Message:
    def __init__(self, content=None, calls=None):
        self.content = content
        self.tool_calls = calls

    def model_dump(self, exclude_none=True):
        data = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            data["tool_calls"] = [
                {
                    "id": call.id,
                    "function": {"name": call.function.name, "arguments": call.function.arguments},
                }
                for call in self.tool_calls
            ]
        return data


class Client:
    def __init__(self, messages):
        self.messages = iter(messages)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=next(self.messages))], usage=None)


class ExampleTests(unittest.TestCase):
    def test_hook_can_block_tool_without_running_it(self):
        call = SimpleNamespace(
            id="one", function=SimpleNamespace(name="lookup", arguments='{"doc_id":"runbook"}')
        )
        client = Client([Message(calls=[call]), Message(content="Blocked safely")])
        agent = Agent(
            client,
            [LOOKUP],
            Hooks(before=[lambda *_: (_ for _ in ()).throw(PermissionError("no"))]),
        )
        self.assertEqual(agent.run("Question"), "Blocked safely")
        self.assertIn("no", agent.last_messages[-2]["content"])

    def test_compaction_rejects_orphaned_tool_result(self):
        compact = runpy.run_path(str(Path(__file__).parent / "06-context-compaction/demo.py"))
        with self.assertRaises(ValueError):
            compact["complete_turns"](
                [
                    {"role": "user", "content": "hi"},
                    {"role": "tool", "tool_call_id": "missing", "content": "bad"},
                ]
            )

    def test_skill_path_is_bounded(self):
        loader = runpy.run_path(str(Path(__file__).parent / "01-skills/demo.py"))["load_skill"]
        with self.assertRaises(ValueError):
            loader({"name": "../incident-summary"})


if __name__ == "__main__":
    unittest.main()
