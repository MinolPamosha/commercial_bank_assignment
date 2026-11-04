"""Validation + HITL nodes — guardrails before the user sees an answer.

HITL uses LangGraph's native `interrupt()`:
- `validate_node` decides whether a human must approve (research answers by default, or
  when the user asked for it) and routes to `hitl`.
- `hitl_node` calls `interrupt(payload)`; the graph pauses and the thread's checkpoint holds
  the full state (draft, citations, events). The API returns `hitl_required=True`.
- `/api/chat/hitl` resumes the same thread with `Command(resume={"approved": ..., "note": ...})`;
  `interrupt()` returns that value and the node finishes — nothing is re-generated, so the
  user receives exactly the draft they approved.
"""

from __future__ import annotations

import re
from difflib import get_close_matches
from typing import Any

from langgraph.types import interrupt

from backend.app.agents.state import AgentState
from backend.app.config import get_settings
from backend.app.models.schemas import AgentEvent
from backend.app.security.guardrails import check_citations

REJECTED_ANSWER = "A human reviewer rejected this answer. Please rephrase your question or contact your team lead."


def _referenced(answer: str, citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The subset of `citations` whose document id the answer mentions (as [doc:ID] or bare)."""
    return [c for c in citations if c.get("doc_id") and re.search(rf"\b{re.escape(str(c['doc_id']))}\b", answer)]


# Identifiers the model tends to retype from tool output: e-mail addresses and record ids
# (E1001, svc-upi, INC-2025-0142). Anything else is prose and left alone.
_IDENTIFIER_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+|\b(?:E\d{3,}|svc-[a-z0-9-]+|INC-\d{4}-\d{3,})\b", re.IGNORECASE)


def _leaf_strings(value: Any, out: set[str]) -> set[str]:
    if isinstance(value, dict):
        for v in value.values():
            _leaf_strings(v, out)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _leaf_strings(v, out)
    elif isinstance(value, str):
        out.add(value)
    return out


def repair_tool_identifiers(answer: str, tool_results: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    """Make identifiers in the answer match the MCP tool data exactly.

    LLMs occasionally mis-copy an address or id ("thariindu.silva@…"). When the answer was built
    from tool output we know the ground truth, so a near-miss is snapped back to the real value
    and an identifier the tool never returned is marked unverified rather than passed off as fact.
    """
    truth: set[str] = set()
    for tr in tool_results or []:
        res = tr.get("result") or {}
        if res.get("ok"):
            _leaf_strings(res.get("data"), truth)
    if not truth:
        return answer, {}
    known = {t for t in truth if _IDENTIFIER_RE.fullmatch(t)}
    if not known:
        return answer, {}
    lower = {k.lower(): k for k in known}

    fixed: dict[str, str] = {}
    unverified: list[str] = []

    def _sub(m: re.Match) -> str:
        found = m.group(0)
        if found.lower() in lower:
            return lower[found.lower()]
        close = get_close_matches(found.lower(), list(lower), n=1, cutoff=0.8)
        if close:
            fixed[found] = lower[close[0]]
            return lower[close[0]]
        unverified.append(found)
        return f"{found} (unverified)"

    repaired = _IDENTIFIER_RE.sub(_sub, answer)
    flags: dict[str, Any] = {}
    if fixed:
        flags["corrected_identifiers"] = fixed
    if unverified:
        flags["unverified_identifiers"] = sorted(set(unverified))
    return repaired, flags


async def validate_node(state: AgentState) -> dict:
    events = [AgentEvent(event_type="node", node="validate", message="Validating response & citations")]
    notes: list[str] = []
    answer = state.get("draft_answer") or state.get("final_answer") or ""
    citations = list(state.get("citations") or [])

    # Multi-turn grounding: a follow-up answered from conversation memory ran no retrieval, so
    # `citations` is empty, yet it may rightly point back at documents cited earlier in this
    # session. Those count as known evidence, and the ones the answer uses become its sources.
    prior = [c for c in (state.get("prior_citations") or []) if c.get("doc_id") not in {x.get("doc_id") for x in citations}]
    if not citations and prior:
        carried = _referenced(answer, prior)
        if carried:
            citations = carried
            events.append(
                AgentEvent(
                    event_type="validation",
                    node="validate",
                    message="Follow-up grounded on documents from earlier in this conversation",
                    detail={"doc_ids": [c["doc_id"] for c in carried]},
                )
            )

    cite_check = check_citations(answer, citations + prior)
    ok = cite_check.ok
    if not ok:
        notes.extend(cite_check.reasons)
        # Repair rather than fail closed (POC UX): strip invented [doc:…] citations and mark
        # bare document ids the evidence cannot back so the reader is never misled.
        for bad in cite_check.flags.get("hallucinated_citations", []):
            answer = answer.replace(f"[doc:{bad}]", "[citation-removed]")
        for bad in cite_check.flags.get("unverified_ids", []):
            answer = re.sub(rf"\b{re.escape(bad)}\b(?! \(unverified)", f"{bad} (unverified reference)", answer)
        events.append(
            AgentEvent(
                event_type="validation",
                node="validate",
                message="Removed hallucinated citations",
                detail=cite_check.flags,
            )
        )
        ok = True  # repaired
        notes.append("Auto-repaired unsupported document references")

    # Tool-grounded answers: identifiers must match the MCP data byte for byte.
    answer, id_flags = repair_tool_identifiers(answer, state.get("tool_results") or [])
    if id_flags:
        notes.append("Auto-repaired identifiers against tool data")
        events.append(
            AgentEvent(
                event_type="validation",
                node="validate",
                message="Corrected identifiers against the tool data",
                detail=id_flags,
            )
        )

    if state.get("degraded"):
        notes.append(f"Degraded mode: {state.get('degradation_reason')}")
        events.append(
            AgentEvent(
                event_type="validation",
                node="validate",
                message="Serving degraded response",
                detail={"reason": state.get("degradation_reason")},
            )
        )

    events.append(
        AgentEvent(
            event_type="validation",
            node="validate",
            message="Validation complete",
            detail={"ok": ok, "notes": notes},
        )
    )

    settings = get_settings()
    needs_human = bool(state.get("hitl_required")) or (settings.hitl_enabled and state.get("intent") == "research")
    route = "hitl" if needs_human else "end"

    return {
        "active_node": "validate",
        "validation_ok": ok,
        "validation_notes": notes,
        "final_answer": answer,
        "citations": citations,
        "hitl_required": needs_human,
        "events": events,
        "route": route,
    }


async def hitl_node(state: AgentState) -> dict:
    """Human-in-the-loop pause point (bonus) — a real LangGraph interrupt.

    First execution: `interrupt()` raises internally, the graph checkpoints and stops.
    Resumed execution: the node re-runs from the top and `interrupt()` returns the
    reviewer's decision.
    """
    draft = state.get("final_answer") or ""
    decision: dict[str, Any] = interrupt(
        {
            "reason": "Research answers are released only after a human review.",
            "draft_preview": draft[:280],
            "citations": [c.get("doc_id") for c in state.get("citations") or []],
        }
    )
    approved = bool((decision or {}).get("approved"))
    note = str((decision or {}).get("note") or "")

    if approved:
        events = [
            AgentEvent(
                event_type="final",
                node="hitl",
                message="Human reviewer approved the draft — answer released",
                detail={"note": note},
            )
        ]
        return {"active_node": "hitl", "hitl_required": False, "final_answer": draft, "events": events, "route": "end"}

    events = [
        AgentEvent(
            event_type="validation",
            node="hitl",
            message="Human reviewer rejected the draft — answer withheld",
            detail={"note": note},
        )
    ]
    return {"active_node": "hitl", "hitl_required": False, "final_answer": REJECTED_ANSWER, "events": events, "route": "end"}
