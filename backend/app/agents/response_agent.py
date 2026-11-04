"""Response Agent — final answer generation with citations + brand voice."""

from __future__ import annotations

import json

from backend.app.agents.llm import acomplete
from backend.app.agents.state import AgentState
from backend.app.config import get_settings
from backend.app.models.schemas import AgentEvent

GREETING_WORDS = {"hi", "hello", "hey", "thanks", "thank", "bye", "goodbye", "morning", "afternoon", "evening"}


def _is_greeting(message: str, memory_context: str) -> bool:
    """True for a short opener/closer; never for a follow-up once an assistant turn exists."""
    words = [w.strip("!.,?") for w in message.lower().split()]
    if "ASSISTANT:" in memory_context:
        return False
    return len(words) <= 4 and any(w in GREETING_WORDS for w in words)


async def response_node(state: AgentState) -> dict:
    settings = get_settings()
    events = [AgentEvent(event_type="node", node="response", message="Response agent composing final answer")]

    evidence = {
        "citations": state.get("citations") or [],
        "research_batches": state.get("research_batches") or [],
        "tool_results": state.get("tool_results") or [],
        "memory": state.get("memory_context") or "",
        # Documents already shown earlier in this conversation (id + title only): a follow-up
        # answered from memory may cite these; validation checks them against the session.
        "earlier_sources": [
            {"doc_id": c.get("doc_id"), "title": c.get("title")} for c in state.get("prior_citations") or []
        ],
    }

    # Canned greeting only for a real opener ("hi", "hello", "thanks"). A short follow-up in an
    # ongoing conversation ("say that shorter") must go to the model with the memory block, even
    # if the supervisor labelled it small talk.
    if state.get("intent") == "chitchat" and _is_greeting(state["message"], state.get("memory_context") or ""):
        answer = (
            f"Hello! I'm the {settings.assistant_name} for {settings.bank_name}. "
            "Ask me about policies, incidents, runbooks, or architecture."
        )
    else:
        user_prompt = (
            f"User question:\n{state['message']}\n\n"
            f"Evidence JSON:\n{json.dumps(evidence, default=str)[:12000]}\n\n"
            "Write a clear answer for a bank employee. Cite sources as [doc:<doc_id>] only when the id appears "
            "in citations or earlier_sources. Call out recurring root causes when research batches exist. "
            "If a tool_results entry has ok=false because the role is 'not permitted', say plainly that the "
            "user's role is not authorized for that data and suggest contacting an analyst/administrator — "
            "do not guess the data. When answering from tool_results, copy names, ids and e-mail addresses "
            "exactly as they appear in the data. If a tool failed or timed out, say live data is unavailable right now. "
            "If the question is a follow-up, answer from the memory block. If evidence is weak, say what is missing."
        )
        try:
            answer = await acomplete(
                system="Produce accurate, citation-grounded answers. Never invent document ids.",
                user=user_prompt,
            )
        except Exception as exc:  # noqa: BLE001
            # Graceful degradation
            cites = state.get("citations") or []
            bullets = "\n".join(f"- [{c['doc_id']}] {c['title']}: {c['snippet'][:160]}" for c in cites[:5])
            answer = (
                f"I hit an LLM failure ({exc}). Here is an evidence-only summary for "
                f"{settings.bank_name}:\n\n{bullets or 'No evidence retrieved.'}"
            )
            events.append(AgentEvent(event_type="error", node="response", message=f"LLM failure: {exc}"))
            return {
                "active_node": "response",
                "draft_answer": answer,
                "final_answer": answer,
                "events": events,
                "degraded": True,
                "degradation_reason": "llm_failure",
                "route": "validate",
            }

    events.append(AgentEvent(event_type="final", node="response", message="Draft answer ready"))
    return {
        "active_node": "response",
        "draft_answer": answer,
        "final_answer": answer,
        "events": events,
        "route": "validate",
    }
