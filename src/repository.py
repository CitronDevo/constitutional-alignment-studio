"""
Repository module for managing tasks and golden examples.

Handles loading dev.jsonl (read-only) and golden_examples.json (read/write)
to keep training and testing data flows cleanly separated.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import List
import uuid
from datetime import datetime

from src.models import ConstitutionalExample, ConversationTurn

HISTORY_FILE = Path("data/output/history.jsonl")


class TaskRepository:
    """
    Central access point for tasks and saved constitutional examples.

    The repository enforces separation between trained tasks (those with a
    saved ConstitutionalExample) and untrained/test tasks (those without one).
    """

    def __init__(
        self,
        dataset_path: Path | str | None = None,
        examples_path: Path | str | None = None
    ):
        # Evaluate defaults at runtime so patched Path works in tests
        self.dataset_path = Path(dataset_path) if dataset_path is not None else Path("data/input/dev.jsonl")
        self.examples_path = Path(examples_path) if examples_path is not None else Path("data/output/golden_examples.json")
        self.history_path = HISTORY_FILE

        self.tasks: List[ConversationTurn] = self._load_dataset(self.dataset_path)
        self.examples: List[ConstitutionalExample] = self._load_examples(self.examples_path)

    def _load_dataset(self, path: Path) -> List[ConversationTurn]:
        """Load the raw tasks from dev.jsonl."""
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        tasks: List[ConversationTurn] = []
        with path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f, start=0):
                if not line.strip():
                    continue

                try:
                    payload = json.loads(line)
                    tasks.append(ConversationTurn(**payload))
                except Exception as exc:
                    warnings.warn(f"Skipping task at line {idx}: {exc}", UserWarning)

        if not tasks:
            raise ValueError(f"No valid tasks found in dataset: {path}")

        return tasks

    def _load_examples(self, path: Path) -> List[ConstitutionalExample]:
        """Load saved golden examples (if any)."""
        if not path.exists():
            return []

        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            warnings.warn(f"Failed to read examples file '{path}': {exc}", UserWarning)
            return []

        items = data if isinstance(data, list) else [data]

        examples: List[ConstitutionalExample] = []
        for idx, item in enumerate(items):
            if "source_id" not in item:
                warnings.warn(
                    f"Skipping example at index {idx}: missing required source_id",
                    UserWarning
                )
                continue
            if "active" not in item:
                item["active"] = True
            if "user_prompt" not in item:
                # Attempt to backfill from dataset using source_id
                try:
                    item["user_prompt"] = self.tasks[item["source_id"]].user
                except Exception:
                    item["user_prompt"] = ""

            try:
                examples.append(ConstitutionalExample(**item))
            except Exception as exc:
                warnings.warn(
                    f"Skipping example at index {idx} due to validation error: {exc}",
                    UserWarning
                )

        return examples

    def _persist_examples(self) -> None:
        """Write golden examples back to disk."""
        self.examples_path.parent.mkdir(parents=True, exist_ok=True)
        with self.examples_path.open("w", encoding="utf-8") as f:
            json.dump(
                [example.model_dump() for example in self.examples],
                f,
                indent=2,
                ensure_ascii=False
            )

    def get_untrained_tasks(self) -> List[ConversationTurn]:
        """
        Return tasks that have not yet been converted into golden examples.

        Tasks are identified by their index in dev.jsonl (0-indexed).
        """
        completed_ids = {example.source_id for example in self.examples}
        return [
            task for idx, task in enumerate(self.tasks)
            if idx not in completed_ids
        ]

    def get_test_tasks(self) -> List[ConversationTurn]:
        """
        Return tasks available for testing/generalization.

        By definition, these are the same tasks that are not part of the training pool.
        """
        return self.get_untrained_tasks()

    def get_training_pool(self) -> List[ConstitutionalExample]:
        """Return active saved golden examples."""
        return [ex for ex in self.examples if getattr(ex, "active", True)]

    def get_all_examples(self) -> List[ConstitutionalExample]:
        """Return all saved golden examples, including inactive ones."""
        return list(self.examples)

    def upsert_example(self, example: ConstitutionalExample) -> ConstitutionalExample:
        """
        Insert or overwrite a golden example, keyed by source_id.
        """
        if example.source_id is None:
            raise ValueError("source_id is required to upsert an example")

        existing_index = next(
            (i for i, ex in enumerate(self.examples) if ex.source_id == example.source_id),
            None
        )

        if existing_index is not None:
            self.examples[existing_index] = example
        else:
            self.examples.append(example)

        self._persist_examples()
        # Append to history log
        self._append_history(example)
        return example

    def delete_example(self, source_id: int) -> None:
        """Delete an example by source_id."""
        self.examples = [ex for ex in self.examples if ex.source_id != source_id]
        self._persist_examples()

    def toggle_example(self, source_id: int, active: bool) -> None:
        """Toggle active flag for an example."""
        for ex in self.examples:
            if ex.source_id == source_id:
                ex.active = active
        self._persist_examples()

    def _append_history(self, example: ConstitutionalExample) -> None:
        """Append an example version to history log."""
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        if not example.timestamp:
            example.timestamp = datetime.utcnow().isoformat()
        if not example.version_id:
            example.version_id = str(uuid.uuid4())
        with self.history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(example.model_dump(), ensure_ascii=False) + "\n")

    def get_history(self, source_id: int) -> List[ConstitutionalExample]:
        """Return history entries for a given source_id (newest first)."""
        if not self.history_path.exists():
            return []
        entries: List[ConstitutionalExample] = []
        with self.history_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if data.get("source_id") == source_id:
                        if "active" not in data:
                            data["active"] = True
                        entries.append(ConstitutionalExample(**data))
                except Exception:
                    continue
        entries.sort(key=lambda ex: ex.timestamp or "", reverse=True)
        return entries

    def import_dataset(self, json_content: str) -> None:
        """
        Overwrite the current golden examples with provided JSON/JSONL content.
        """
        # Try JSON array first
        examples: List[ConstitutionalExample] = []
        try:
            data = json.loads(json_content)
            if isinstance(data, list):
                raw_items = data
            else:
                raw_items = [data]
        except Exception:
            # Fallback to JSONL
            raw_items = []
            for line in json_content.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    raw_items.append(json.loads(line))
                except Exception:
                    continue

        for item in raw_items:
            if "active" not in item:
                item["active"] = True
            if "source_id" not in item:
                continue
            examples.append(ConstitutionalExample(**item))

        # Overwrite head file
        self.examples = examples
        self._persist_examples()

    def get_dataset_as_json(self) -> str:
        """Return current golden examples as JSON string."""
        return json.dumps([ex.model_dump() for ex in self.examples], ensure_ascii=False, indent=2)

    def get_export_json(self) -> str:
        """Return active examples as JSONL string for download."""
        lines = []
        for ex in self.get_training_pool():
            item = ex.model_dump()
            lines.append(json.dumps(item, ensure_ascii=False))
        return "\n".join(lines)
