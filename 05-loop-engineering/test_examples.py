"""Offline boundary checks for the shared loop and compaction."""

import runpy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from harness import LOOKUP, Agent, Hooks
from skills import SkillRegistry


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

    def test_skill_discovery_and_progressive_loading(self):
        registry = SkillRegistry(Path(__file__).parent / "01-skills/skills")
        self.assertIn("incident-summary", registry.discover())
        self.assertNotIn("When summarizing an incident", registry.catalog_prompt())
        with self.assertRaises(PermissionError):
            registry.read_skill_file({"name": "incident-summary", "path": "references/template.md"})
        self.assertIn(
            "When summarizing an incident", registry.load_skill({"name": "incident-summary"})
        )
        self.assertIn(
            "Timeline",
            registry.read_skill_file(
                {"name": "incident-summary", "path": "references/template.md"}
            ),
        )

    def test_new_skill_appears_without_agent_code_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "report-writer").mkdir()
            (root / "report-writer/SKILL.md").write_text(
                "---\nname: report-writer\ndescription: Write concise reports.\n---\n\nUse headings.\n"
            )
            registry = SkillRegistry(root)
            self.assertIn("report-writer", registry.catalog_prompt())
            self.assertIn(
                "report-writer",
                registry.tools()[0].schema()["function"]["parameters"]["properties"]["name"][
                    "enum"
                ],
            )

    def test_skill_path_is_bounded(self):
        registry = SkillRegistry(Path(__file__).parent / "01-skills/skills")
        with self.assertRaises(ValueError):
            registry.load_skill({"name": "../incident-summary"})
        registry.load_skill({"name": "incident-summary"})
        with self.assertRaises(ValueError):
            registry.read_skill_file({"name": "incident-summary", "path": "../../../../.env"})

    def test_agent_loads_skill_then_supporting_file(self):
        registry = SkillRegistry(Path(__file__).parent / "01-skills/skills")
        calls = [
            SimpleNamespace(
                id="skill",
                function=SimpleNamespace(
                    name="load_skill", arguments='{"name":"incident-summary"}'
                ),
            ),
            SimpleNamespace(
                id="file",
                function=SimpleNamespace(
                    name="read_skill_file",
                    arguments='{"name":"incident-summary","path":"references/template.md"}',
                ),
            ),
        ]
        client = Client(
            [
                Message(calls=[calls[0]]),
                Message(calls=[calls[1]]),
                Message(content="Summary with timeline"),
            ]
        )
        agent = Agent(client, registry.tools())
        self.assertIn("timeline", agent.run("Summarize incident", registry.catalog_prompt()))
        self.assertEqual(registry.loaded, {"incident-summary"})
        self.assertIn("Timeline", agent.last_messages[-2]["content"])


if __name__ == "__main__":
    unittest.main()
