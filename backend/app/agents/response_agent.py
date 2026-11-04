"""Response Agent — final answer generation with citations + brand voice."""

from __future__ import annotations

import json

from backend.app.agents.llm import acomplete
from backend.app.agents.state import AgentState
from backend.app.config import get_settings
from backend.app.models.schemas import AgentEvent

GREETING_WORDS = {"hi", "hello", "hey", "thanks", "thank", "bye", "goodbye", "morning", "afternoon", "evening"}


def _is_greeting(message: str, memory_context: str = "") -> bool:
    """True for a short greeting/thanks/goodbye made only of small-talk words.

    A short *follow-up* ("say that shorter", "which one?") contains no greeting word, so it still
    goes to the model with the memory block. A bare "hello" mid-conversation must not: the model
    would otherwise repeat the previous answer back.
    """
    words = [w.strip("!.,?") for w in message.lower().split()]
    if not words or len(words) > 4:
        return False
    filler = {"there", "again", "you", "so", "much", "a", "lot", "good", "ok", "okay", "and", "bye"}
    return any(w in GREETING_WORDS for w in words) and all(w in GREETING_WORDS or w in filler for w in words)


def _greeting_reply(message: str, memory_context: str, assistant_name: str, bank_name: str) -> str:
    words = {w.strip("!.,?") for w in message.lower().split()}
    if words & {"thanks", "thank"}:
        return "You're welcome. Ask whenever you need something else."
    if words & {"bye", "goodbye"}:
        return "Goodbye. Come back any time you need a policy, incident or runbook."
    if "ASSISTANT:" in memory_context:
        return "Hello again. What else can I help you with?"
    return f"Hello! I'm the {assistant_name} for {bank_name}. Ask me about policies, incidents, runbooks, or architecture."


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

    # Canned reply for pure small talk ("hi", "hello again", "thanks"), whatever the supervisor
    # labelled it. A short follow-up in an ongoing conversation ("say that shorter") has no
    # greeting word and goes to the model with the memory block.
    if _is_greeting(state["message"]):
        answer = _greeting_reply(state["message"], state.get("memory_context") or "", settings.assistant_name, settings.bank_name)
    else:
        revision = (state.get("revision_feedback") or "").strip()
        if revision:
            events.append(
                AgentEvent(
                    event_type="hitl",
                    node="response",
                    message="Revising the draft from the reviewer's send-back note",
                    detail={"note": revision, "revision": int(state.get("revision_count") or 0)},
                )
            )
            previous = state.get("draft_answer") or state.get("final_answer") or ""
            # Feedback first so it survives truncated offline stubs and leads the LLM's attention.
            user_prompt = (
                f"Reviewer feedback — address this in the revision:\n{revision}\n\n"
                f"User question:\n{state['message']}\n\n"
                f"Previous draft (a human reviewer sent this back):\n{previous[:3000]}\n\n"
                f"Evidence JSON:\n{json.dumps(evidence, default=str)[:10000]}\n\n"
                "Rewrite the answer so it satisfies the reviewer's feedback while staying grounded in the evidence. "
                "Cite sources as [doc:<doc_id>] only when the id appears in citations or earlier_sources. "
                "Do not mention the review process, HITL, or that this is a revision — write the improved answer directly."
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

    events.append(
        AgentEvent(
            event_type="final",
            node="response",
            message="Revised draft ready" if state.get("revision_feedback") else "Draft answer ready",
        )
    )
    return {
        "active_node": "response",
        "draft_answer": answer,
        "final_answer": answer,
        "revision_feedback": None,  # consumed for this pass
        "events": events,
        "route": "validate",
    }
