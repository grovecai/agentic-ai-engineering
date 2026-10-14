"""Regression tests for avoiding duplicate memories during consolidation."""

import unittest

from memory.manager import MemoryManager
from memory.models import MemoryEntry


class FakeStore:
    def __init__(self) -> None:
        self.entries: list[MemoryEntry] = []

    def save(self, entry: MemoryEntry) -> None:
        self.entries.append(entry)

    def list_all(self) -> list[MemoryEntry]:
        return list(self.entries)


class ConsolidationTests(unittest.TestCase):
    def setUp(self) -> None:
        # Exercise the manager without creating JSON or ChromaDB files.
        self.manager = MemoryManager.__new__(MemoryManager)
        self.manager.episodic = FakeStore()
        self.manager.semantic = FakeStore()
        self.manager._session_writes = []

    def test_prompt_identifies_facts_already_saved_in_this_session(self) -> None:
        self.manager.remember("User prefers Python", memory_type="semantic")

        prompt = self.manager.build_consolidation_prompt(
            [{"role": "user", "content": "I prefer Python"}]
        )

        self.assertIn("Already saved during this session:", prompt)
        self.assertIn("[semantic] User prefers Python", prompt)
        self.assertIn("Do not repeat or paraphrase", prompt)

    def test_consolidation_skips_existing_and_repeated_items_across_tiers(self) -> None:
        self.manager.remember("User prefers Python", memory_type="semantic")

        saved = self.manager.save_consolidation(
            '[{"content":"User prefers Python!","type":"episodic"},'
            '{"content":"Orion launch moved to October 27","type":"episodic"},'
            '{"content":"orion launch moved to october 27.","type":"semantic"}]'
        )

        self.assertEqual(saved, ["[episodic] Orion launch moved to October 27"])
        self.assertEqual(len(self.manager.semantic.entries), 1)
        self.assertEqual(len(self.manager.episodic.entries), 1)

    def test_duplicate_only_response_saves_nothing(self) -> None:
        self.manager.remember("Orion launch moved to October 27", memory_type="episodic")

        saved = self.manager.save_consolidation(
            '[{"content":"orion launch moved to october 27!","type":"semantic"}]'
        )

        self.assertEqual(saved, [])
        self.assertEqual(len(self.manager.episodic.entries), 1)
        self.assertEqual(self.manager.semantic.entries, [])


if __name__ == "__main__":
    unittest.main()
