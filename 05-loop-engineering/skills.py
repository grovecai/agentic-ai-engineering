"""Discover filesystem skills and expose progressive-loading tools to an agent."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from harness import Tool

NAME_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
MAX_FILE_BYTES = 16_384


@dataclass
class SkillRegistry:
    root: Path
    loaded: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.root = self.root.resolve()

    def _skill_path(self, name: str) -> Path:
        if not isinstance(name, str) or not NAME_PATTERN.fullmatch(name):
            raise ValueError("Invalid skill name")
        directory = (self.root / name).resolve()
        if not directory.is_relative_to(self.root):
            raise ValueError("Skill path leaves the catalog")
        return directory

    def discover(self) -> dict[str, str]:
        """Read only front matter; keep full instructions out of the base prompt."""
        catalog: dict[str, str] = {}
        if not self.root.is_dir():
            return catalog
        for directory in sorted(self.root.iterdir()):
            if not directory.is_dir() or not NAME_PATTERN.fullmatch(directory.name):
                continue
            try:
                skill_file = self._skill_path(directory.name) / "SKILL.md"
                if not skill_file.is_file() or skill_file.stat().st_size > MAX_FILE_BYTES:
                    continue
                lines = skill_file.read_text(encoding="utf-8").splitlines()
                if not lines or lines[0] != "---":
                    continue
                end = lines.index("---", 1)
                metadata = dict(line.split(":", 1) for line in lines[1:end] if ":" in line)
                name = metadata.get("name", "").strip().strip("\"'")
                description = metadata.get("description", "").strip().strip("\"'")
                if name == directory.name and description:
                    catalog[name] = description
            except (OSError, UnicodeError, ValueError):
                continue
        return catalog

    def catalog_prompt(self) -> str:
        catalog = self.discover()
        lines = [f"- {name}: {description}" for name, description in catalog.items()]
        return (
            "Available skills (name and description only):\n"
            + ("\n".join(lines) if lines else "(none)")
            + "\nWhen a skill is relevant, call load_skill before answering. "
            "Follow its instructions. Read referenced files with read_skill_file when needed."
        )

    def load_skill(self, args: dict) -> str:
        name = args.get("name")
        if name not in self.discover():
            raise ValueError("Unknown skill")
        skill_file = self._skill_path(name) / "SKILL.md"
        text = skill_file.read_text(encoding="utf-8")
        self.loaded.add(name)
        return text

    def read_skill_file(self, args: dict) -> str:
        name = args.get("name")
        relative = args.get("path")
        if name not in self.loaded:
            raise PermissionError("Load the skill before reading its files")
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise ValueError("Invalid relative path")
        skill_root = self._skill_path(name)
        target = (skill_root / relative).resolve()
        if not target.is_relative_to(skill_root) or not target.is_file():
            raise ValueError("File is outside the loaded skill or does not exist")
        if target.stat().st_size > MAX_FILE_BYTES:
            raise ValueError("Skill file is too large")
        return target.read_text(encoding="utf-8")

    def tools(self) -> list[Tool]:
        names = list(self.discover())
        return [
            Tool(
                "load_skill",
                "Load the complete SKILL.md instructions for a relevant catalog skill.",
                {
                    "type": "object",
                    "properties": {"name": {"type": "string", "enum": names}},
                    "required": ["name"],
                },
                self.load_skill,
            ),
            Tool(
                "read_skill_file",
                "Read a referenced text file from a skill after loading that skill.",
                {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "enum": names},
                        "path": {"type": "string"},
                    },
                    "required": ["name", "path"],
                },
                self.read_skill_file,
            ),
        ]
