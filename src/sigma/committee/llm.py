"""The committee's only contact with a language model: one ``Asker`` call, JSON out, results cached.

Every call has a stable ``task_id`` and its answer is appended to a JSONL file, so a run that is
stopped (quota, sleep, closing the laptop) resumes without repeating or paying for finished work.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Protocol

from sigma.ai.budgets import TaskClass
from sigma.ai.roles import Role


class Asker(Protocol):
    def ask(self, task_id: str, agent: str, role: Role, task_class: TaskClass, system: str, user: str
            ) -> str: ...


class OutOfBudget(RuntimeError):
    """The per-run call cap was reached; run again to continue where this stopped."""


class FileCheckpointStore:
    """Router-compatible checkpoint store backed by an append-only JSONL file."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._data: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self._data[rec["task_id"]] = rec["state"]

    def load(self, task_id: str) -> dict[str, Any] | None:
        s = self._data.get(task_id)
        return dict(s) if s else None

    def save(self, task_id: str, state: dict[str, Any]) -> None:
        self._data[task_id] = dict(state)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"task_id": task_id, "state": state}) + "\n")


class RouterAsker:
    """Adapts sigma.ai's ModelRouter to the Asker protocol and counts calls that really hit a model."""

    def __init__(self, router: Any, max_calls: int | None = None):
        self.router, self.max_calls, self.calls = router, max_calls, 0

    def ask(self, task_id: str, agent: str, role: Role, task_class: TaskClass, system: str, user: str
            ) -> str:
        if self.max_calls is not None and self.calls >= self.max_calls:
            state = self.router.checkpoints.load(task_id)
            if not (state and state.get("status") == "completed"):
                raise OutOfBudget(f"reached the cap of {self.max_calls} model calls for this run")
        res = self.router.run(task_id=task_id, agent=agent, role=role, task_class=task_class,
                              messages=[{"role": "system", "content": system},
                                        {"role": "user", "content": user}])
        if not res.from_checkpoint:
            self.calls += 1
        return str(res.text)


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def parse_json(text: str) -> dict[str, Any] | None:
    """The first JSON object in a model reply, tolerating code fences and chatter; None if there is none."""
    if not text:
        return None
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    for blob in candidates:
        start = blob.find("{")
        while start != -1:
            depth = 0
            for i in range(start, len(blob)):
                depth += (blob[i] == "{") - (blob[i] == "}")
                if depth == 0:
                    try:
                        obj = json.loads(blob[start:i + 1])
                    except ValueError:
                        break
                    return obj if isinstance(obj, dict) else None
            start = blob.find("{", start + 1)
    return None
