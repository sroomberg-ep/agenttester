"""Shared state for models waiting on user input."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class PendingQuestion:
    model_name: str
    question: str
    asked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    response: str | None = None
    answered: threading.Event = field(default_factory=threading.Event)


class QuestionRegistry:
    """Thread-safe registry of models waiting for user input."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: dict[str, PendingQuestion] = {}

    def ask(self, model_name: str, question: str, timeout: float | None = None) -> str:
        """Block until the user responds. Called from a tool executor thread."""
        entry = PendingQuestion(model_name=model_name, question=question)
        with self._lock:
            self._pending[model_name] = entry

        entry.answered.wait(timeout=timeout)

        with self._lock:
            self._pending.pop(model_name, None)

        if entry.response is None:
            return "[no response — timed out or session ended]"
        return entry.response

    def respond(self, model_name: str, response: str) -> bool:
        """Deliver a user response to a waiting model. Returns False if not waiting."""
        with self._lock:
            entry = self._pending.get(model_name)
            if entry is None:
                return False
        entry.response = response
        entry.answered.set()
        return True

    def pending(self) -> list[PendingQuestion]:
        """Return a snapshot of all pending questions."""
        with self._lock:
            return list(self._pending.values())

    def cancel_all(self) -> None:
        """Unblock all waiting models (e.g. on session exit)."""
        with self._lock:
            entries = list(self._pending.values())
        for entry in entries:
            if not entry.answered.is_set():
                entry.response = "[session ended]"
                entry.answered.set()
