"""Conversational + long-term memory design.

Short-term (session): last N turns + rolling summary — survives multi-turn chat.
Long-term (user): durable preferences / recurring topics keyed by username.

Why not dump full history into the LLM every turn?
- Cost & latency grow linearly.
- RLM-style agents already fetch evidence on demand.
- We keep a compact summary + last 6 turns for coherence.

Evidence continuity: the session also remembers which documents were cited so far. A follow-up
answered from memory ("what was the time limit for step 1?") may legitimately reference those
documents even though no retrieval ran this turn; without this the citation guardrail would
strip them as hallucinated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.app.config import get_settings
from backend.app.observability.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Turn:
    role: str
    content: str
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class SessionMemory:
    session_id: str
    user_id: str
    turns: list[Turn] = field(default_factory=list)
    summary: str = ""
    # doc_id → citation dict (title, metadata, snippet) for every document cited in this session.
    cited_docs: dict[str, dict[str, Any]] = field(default_factory=dict)


MAX_CITED_DOCS_PER_SESSION = 40


class MemoryStore:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._sessions: dict[str, SessionMemory] = {}
        self._long_term_path = Path(self.settings.local_index_path) / "long_term_memory.json"
        self._long_term_path.parent.mkdir(parents=True, exist_ok=True)
        self._long_term: dict[str, dict[str, Any]] = self._load_long_term()

    def _load_long_term(self) -> dict[str, dict[str, Any]]:
        if self._long_term_path.exists():
            return json.loads(self._long_term_path.read_text(encoding="utf-8"))
        return {}

    def _save_long_term(self) -> None:
        self._long_term_path.write_text(json.dumps(self._long_term, indent=2), encoding="utf-8")

    def get_session(self, session_id: str, user_id: str) -> SessionMemory:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionMemory(session_id=session_id, user_id=user_id)
        return self._sessions[session_id]

    def add_turn(self, session_id: str, user_id: str, role: str, content: str) -> SessionMemory:
        mem = self.get_session(session_id, user_id)
        mem.turns.append(Turn(role=role, content=content))
        # Rolling summary heuristic (no LLM call — predictable & cheap)
        if len(mem.turns) >= 4 and len(mem.turns) % 2 == 0:
            recent = mem.turns[-4:]
            mem.summary = " | ".join(f"{t.role}: {t.content[:120]}" for t in recent)
            logger.info("memory_summary_updated", session_id=session_id)
        return mem

    def context_block(self, session_id: str, user_id: str) -> str:
        mem = self.get_session(session_id, user_id)
        lt = self._long_term.get(user_id, {})
        lines = []
        if lt.get("preferences"):
            lines.append(f"Long-term preferences: {lt['preferences']}")
        if lt.get("topics"):
            lines.append(f"Recurring topics: {', '.join(lt['topics'][-5:])}")
        # Feedback loop: recent thumbs-down comments steer the next answers for this user.
        negative = [f for f in lt.get("feedback", []) if f.get("rating") == "down"][-3:]
        if negative:
            complaints = "; ".join(str(f.get("comment") or "no comment") for f in negative)
            lines.append(f"User recently rated answers as unhelpful ({len(negative)}x): {complaints}. Address this.")
        if mem.summary:
            lines.append(f"Session summary: {mem.summary}")
        for t in mem.turns[-6:]:
            lines.append(f"{t.role.upper()}: {t.content}")
        return "\n".join(lines)

    def remember_citations(self, session_id: str, user_id: str, citations: list[dict[str, Any]]) -> None:
        """Keep the documents cited this turn so later follow-ups can be validated against them."""
        mem = self.get_session(session_id, user_id)
        for c in citations:
            doc_id = str(c.get("doc_id") or "")
            if doc_id:
                mem.cited_docs[doc_id] = dict(c)
        while len(mem.cited_docs) > MAX_CITED_DOCS_PER_SESSION:  # drop the oldest first
            mem.cited_docs.pop(next(iter(mem.cited_docs)))

    def session_citations(self, session_id: str, user_id: str) -> list[dict[str, Any]]:
        """Documents already shown to the user in this session (most recent last)."""
        return list(self.get_session(session_id, user_id).cited_docs.values())

    def remember_topic(self, user_id: str, topic: str) -> None:
        entry = self._long_term.setdefault(user_id, {"preferences": "", "topics": [], "feedback": []})
        topics = entry.setdefault("topics", [])
        if topic and topic not in topics:
            topics.append(topic)
            self._save_long_term()
            logger.info("long_term_topic_saved", user_id=user_id, topic=topic)

    def record_feedback(self, user_id: str, payload: dict[str, Any]) -> None:
        entry = self._long_term.setdefault(user_id, {"preferences": "", "topics": [], "feedback": []})
        entry.setdefault("feedback", []).append(payload)
        self._save_long_term()


memory_store = MemoryStore()
