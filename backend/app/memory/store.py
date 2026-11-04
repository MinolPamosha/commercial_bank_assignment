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

Conversation history (user): the transcript of each chat is saved per user (``conversations.json``,
newest 20 kept) so the "Recent" list survives sign-out, a new browser and a UI restart. When a
saved chat is reopened after the API itself restarted, its session memory is re-hydrated from the
transcript (last turns + cited documents), so a follow-up still has context.
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
MAX_CONVERSATIONS_PER_USER = 20
REHYDRATE_TURNS = 6  # how much of a saved transcript becomes session memory again


class MemoryStore:
    def __init__(self) -> None:
        self.settings = get_settings()
        # Keyed per (user, session) so a guessed session id never reaches another user's working memory.
        self._sessions: dict[tuple[str, str], SessionMemory] = {}
        root = Path(self.settings.local_index_path)
        root.mkdir(parents=True, exist_ok=True)
        self._long_term_path = root / "long_term_memory.json"
        self._long_term: dict[str, dict[str, Any]] = self._load_json(self._long_term_path)
        self._conversations_path = root / "conversations.json"
        # user_id → session_id → {"title", "updated_at", "state"}; "state" is the UI transcript.
        self._conversations: dict[str, dict[str, dict[str, Any]]] = self._load_json(self._conversations_path)

    @staticmethod
    def _load_json(path: Path) -> dict:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return {}

    def _save_long_term(self) -> None:
        self._long_term_path.write_text(json.dumps(self._long_term, indent=2), encoding="utf-8")

    def _save_conversations(self) -> None:
        self._conversations_path.write_text(json.dumps(self._conversations), encoding="utf-8")

    def get_session(self, session_id: str, user_id: str) -> SessionMemory:
        key = (user_id, session_id)
        if key not in self._sessions:
            mem = SessionMemory(session_id=session_id, user_id=user_id)
            self._rehydrate(mem)
            self._sessions[key] = mem
        return self._sessions[key]

    def _rehydrate(self, mem: SessionMemory) -> None:
        """Seed a fresh session from its saved transcript (the API restarted, or a chat was
        reopened on another day). Only the owner's saved chats are consulted."""
        saved = self._conversations.get(mem.user_id, {}).get(mem.session_id)
        if not saved:
            return
        messages = [m for m in saved.get("state", {}).get("messages", []) if m.get("role") in ("user", "assistant")]
        for m in messages[-REHYDRATE_TURNS:]:
            mem.turns.append(Turn(role=m["role"], content=str(m.get("content", ""))[:2000]))
        for m in messages:
            for c in m.get("citations") or []:
                if c.get("doc_id"):
                    mem.cited_docs[str(c["doc_id"])] = dict(c)
        logger.info("session_memory_rehydrated", session_id=mem.session_id, turns=len(mem.turns), docs=len(mem.cited_docs))

    # --- Conversation history (per user) ---------------------------------------------------
    def save_conversation(self, user_id: str, session_id: str, title: str, state: dict[str, Any]) -> None:
        # Dict order is recency order (oldest first): re-insert on every save. Timestamps alone
        # tie on coarse clocks, so they are informational rather than the sort key.
        chats = self._conversations.setdefault(user_id, {})
        chats.pop(session_id, None)
        chats[session_id] = {"title": title, "updated_at": datetime.now(timezone.utc).isoformat(), "state": state}
        for sid in list(chats)[: max(0, len(chats) - MAX_CONVERSATIONS_PER_USER)]:  # drop the oldest
            chats.pop(sid, None)
        self._save_conversations()

    def list_conversations(self, user_id: str) -> list[dict[str, Any]]:
        chats = self._conversations.get(user_id, {})
        return [
            {"session_id": sid, "title": c["title"], "updated_at": c["updated_at"]}
            for sid, c in reversed(list(chats.items()))  # newest first
        ]

    def get_conversation(self, user_id: str, session_id: str) -> dict[str, Any] | None:
        saved = self._conversations.get(user_id, {}).get(session_id)
        return {"session_id": session_id, **saved} if saved else None

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
        # Durable preference: a thumbs-down with a real comment becomes a standing instruction
        # for this user (not only the "last 3 complaints" line), so the feedback loop survives
        # beyond the recent window.
        comment = str(payload.get("comment") or "").strip()
        if payload.get("rating") == "down" and len(comment) >= 8:
            prefs = [p.strip() for p in str(entry.get("preferences") or "").split(";") if p.strip()]
            if comment not in prefs:
                prefs.append(comment)
                entry["preferences"] = "; ".join(prefs[-8:])  # keep the newest eight
        self._save_long_term()

    def export_feedback(self) -> dict[str, Any]:
        """Administrator view of every user's long-term memory (feedback + preferences + topics)."""
        return {
            user_id: {
                "preferences": entry.get("preferences") or "",
                "topics": list(entry.get("topics") or []),
                "feedback": list(entry.get("feedback") or []),
            }
            for user_id, entry in self._long_term.items()
        }


memory_store = MemoryStore()
