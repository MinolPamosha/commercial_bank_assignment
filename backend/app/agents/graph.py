"""LangGraph orchestration — multi-agent graph with explicit, observable nodes.

Butterfly-effect / failure handling notes:
- Each specialized node isolates failures (retrieval/MCP/LLM) into `degraded` flags
  instead of crashing the whole turn.
- Supervisor routing is re-entrant via conditional edges; research can recurse internally
  without poisoning session memory.
- Validation + HITL sit after response so bad citations never reach the user silently.

Checkpointing: the graph is compiled with a checkpointer keyed by `thread_id = session_id`.
That is what makes the HITL pause real — `interrupt()` in the `hitl` node persists the whole
turn state, and `/api/chat/hitl` resumes *that* state with `Command(resume=...)` instead of
re-running the agents. `MemorySaver` is in-process (single replica POC); swap in
`AsyncPostgresSaver`/Redis for multi-replica deployments without touching the graph.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any, AsyncIterator, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from langsmith import trace, tracing_context

from backend.app.agents.research_agent import research_node
from backend.app.agents.response_agent import response_node
from backend.app.agents.retrieval_agent import retrieval_node
from backend.app.agents.state import AgentState
from backend.app.agents.supervisor import supervisor_node
from backend.app.agents.tools_agent import tools_node
from backend.app.agents.validation import hitl_node, validate_node
from backend.app.config import get_settings
from backend.app.memory.store import memory_store
from backend.app.models.schemas import AgentEvent, Role
from backend.app.observability.logging import get_logger
from backend.app.observability.tracing import trace_step
from backend.app.security.guardrails import validate_user_input

logger = get_logger(__name__)

# Pre/post-graph steps are plain functions; give them spans so the LangSmith trace covers
# the *whole* turn (guardrail → memory → graph → memory update), not only the graph.
_traced_input_guardrail = trace_step("input_guardrail")(validate_user_input)
_traced_memory_load = trace_step("memory_load")(memory_store.context_block)
_traced_memory_update = trace_step("memory_update")(memory_store.add_turn)


class NoPendingApproval(Exception):
    """Raised when a resume is requested for a thread that is not paused at `hitl`."""


def _route_from_supervisor(state: AgentState) -> str:
    return {
        "retrieve": "retrieval",
        "research": "research",
        "tools": "tools",
        "respond": "response",
    }.get(state.get("route") or "retrieve", "retrieval")


def _route_after_validate(state: AgentState) -> str:
    return "hitl" if state.get("route") == "hitl" else "end"


def _route_after_hitl(state: AgentState) -> str:
    # Send-back → Response revises from the reviewer's note; Approve / stop → END.
    return "response" if state.get("route") == "revise" else "end"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("supervisor", supervisor_node)
    g.add_node("retrieval", retrieval_node)
    g.add_node("research", research_node)
    g.add_node("tools", tools_node)
    g.add_node("response", response_node)
    g.add_node("validate", validate_node)
    g.add_node("hitl", hitl_node)

    g.add_edge(START, "supervisor")
    g.add_conditional_edges(
        "supervisor",
        _route_from_supervisor,
        {
            "retrieval": "retrieval",
            "research": "research",
            "tools": "tools",
            "response": "response",
        },
    )
    g.add_edge("retrieval", "response")
    g.add_edge("research", "response")
    g.add_edge("tools", "response")
    g.add_edge("response", "validate")
    g.add_conditional_edges("validate", _route_after_validate, {"hitl": "hitl", "end": END})
    g.add_conditional_edges("hitl", _route_after_hitl, {"response": "response", "end": END})

    checkpointer = MemorySaver()
    return g.compile(checkpointer=checkpointer)


@lru_cache
def get_compiled_graph():
    return build_graph()


async def forget_thread(session_id: str) -> None:
    """Drop any LangGraph checkpoint for this chat (HITL pause, intermediate state)."""
    graph = get_compiled_graph()
    checkpointer = getattr(graph, "checkpointer", None)
    if checkpointer is None:
        return
    try:
        await checkpointer.adelete_thread(session_id)
    except Exception as exc:  # noqa: BLE001  — older savers may lack adelete_thread
        logger.warning("forget_thread_failed", session_id=session_id, error=str(exc))


# Node whose LLM tokens are streamed to the chat window (the user-facing answer).
ANSWER_STREAMING_NODE = "response"


def _graph_config(session_id: str, user_id: str, role: Role) -> dict[str, Any]:
    """thread_id = session → checkpointed multi-turn state; metadata/tags → filterable traces."""
    return {
        "configurable": {"thread_id": session_id},
        "run_name": "apex_agent_graph",
        "metadata": {"user_id": user_id, "role": role.value, "session_id": session_id},
        "tags": [f"role:{role.value}"],
    }


def _build_result(final: dict[str, Any], events: list[AgentEvent], session_id: str) -> dict[str, Any]:
    """Shape the final graph state into the API-facing turn result."""
    return {
        "answer": final.get("final_answer") or "No answer produced.",
        "citations": final.get("citations") or [],
        "events": events,
        "hitl_required": bool(final.get("hitl_required")),
        "degraded": bool(final.get("degraded")),
        "degradation_reason": final.get("degradation_reason"),
        "blocked": False,
        "session_id": session_id,
        "active_node": final.get("active_node"),
        "validation_notes": final.get("validation_notes") or [],
    }


async def _drain_graph(
    graph, graph_input: Any, config: dict[str, Any], final: dict[str, Any], events: list[AgentEvent]
) -> AsyncIterator[dict[str, Any]]:
    """Stream one graph execution (fresh turn or resume), yielding panel events + answer tokens.

    "updates"  → one chunk per completed node (its state delta, incl. new events)
    "messages" → LLM token chunks tagged with the LangGraph node that produced them
    A `__interrupt__` update means the graph paused at `hitl`; it is surfaced as a `hitl` event.
    """
    async for mode, chunk in graph.astream(graph_input, config=config, stream_mode=["updates", "messages"]):
        if mode == "updates":
            for node_name, update in chunk.items():
                if node_name == "__interrupt__":
                    for intr in update or ():
                        ev = AgentEvent(
                            event_type="hitl",
                            node="hitl",
                            message="Awaiting human approval before releasing research answer",
                            detail=dict(getattr(intr, "value", None) or {}),
                        )
                        events.append(ev)
                        yield {"type": "event", "event": ev}
                    final["hitl_required"] = True
                    final["active_node"] = "hitl"
                    continue
                if not isinstance(update, dict):
                    continue
                for ev in update.get("events") or []:
                    events.append(ev)
                    yield {"type": "event", "event": ev}
                for key, value in update.items():
                    if key != "events":
                        final[key] = value
                logger.debug("graph_node_complete", node=node_name)
        elif mode == "messages":
            msg_chunk, meta = chunk
            text = getattr(msg_chunk, "content", "")
            if text and (meta or {}).get("langgraph_node") == ANSWER_STREAMING_NODE:
                yield {"type": "token", "text": text}


async def stream_agent(
    *,
    message: str,
    session_id: str,
    user_id: str,
    role: Role,
    require_hitl: bool = False,
    resume: Optional[dict[str, Any]] = None,
) -> AsyncIterator[dict[str, Any]]:
    """Run one conversational turn and stream progress while the graph executes.

    Yields dict chunks in order:
      {"type": "event",  "event": AgentEvent}  — emitted as soon as the owning node finishes,
                                                 so the Activity Panel updates in real time
      {"type": "token",  "text": str}          — LLM tokens from the Response agent (streaming answer)
      {"type": "result", "result": {...}}      — exactly once, the completed turn (see `_build_result`)

    `resume` carries a human decision ({"approved": bool, "note": str}) for a thread paused at
    the `hitl` interrupt; the graph continues from its checkpoint instead of starting over.

    `run_agent` is a thin wrapper over this generator so the blocking `/chat`
    endpoint and the streaming `/chat/stream` endpoint share one code path.

    The whole turn runs inside one LangSmith root span (`chat_turn`) tagged with the
    user, role and session, so every conversation is one inspectable trace.
    `tracing_context(enabled=...)` makes the decision explicit per turn instead of relying
    on env-var lookups that langsmith caches at first use.
    """
    async with _turn_span(
        name="chat_turn" if resume is None else "chat_turn_hitl_resume",
        inputs={"message": message, "require_hitl": require_hitl, "resume": resume},
        user_id=user_id,
        role=role,
        session_id=session_id,
    ) as root:
        outcome: dict[str, Any] = {}
        inner = (
            _resume_turn(session_id=session_id, user_id=user_id, role=role, decision=resume)
            if resume is not None
            else _stream_turn(message=message, session_id=session_id, user_id=user_id, role=role, require_hitl=require_hitl)
        )
        async for chunk in inner:
            if chunk["type"] == "result":
                r = chunk["result"]
                outcome = {
                    "answer": r.get("answer"),
                    "route": r.get("active_node"),
                    "citations": [c.get("doc_id") for c in r.get("citations") or []],
                    "hitl_required": r.get("hitl_required"),
                    "degraded": r.get("degraded"),
                    "blocked": r.get("blocked", False),
                }
            yield chunk
        if root is not None:
            root.end(outputs=outcome)


@asynccontextmanager
async def _turn_span(*, name: str, inputs: dict[str, Any], user_id: str, role: Role, session_id: str):
    """Root LangSmith span for a turn, or a no-op when tracing is off (no client, no warnings)."""
    settings = get_settings()
    if not settings.tracing_enabled:
        yield None
        return
    with tracing_context(enabled=True, project_name=settings.langchain_project):
        async with trace(
            name=name,
            run_type="chain",
            inputs=inputs,
            metadata={"user_id": user_id, "role": role.value, "session_id": session_id},
            tags=[f"role:{role.value}"],
        ) as root:
            yield root


async def _stream_turn(
    *,
    message: str,
    session_id: str,
    user_id: str,
    role: Role,
    require_hitl: bool,
) -> AsyncIterator[dict[str, Any]]:
    guard = _traced_input_guardrail(message, role)
    events: list[AgentEvent] = []
    if not guard.ok:
        blocked = AgentEvent(
            event_type="guardrail",
            node="input_validation",
            message="Request blocked by prompt-injection / input guardrails",
            detail={"reasons": guard.reasons},
        )
        events.append(blocked)
        yield {"type": "event", "event": blocked}
        yield {
            "type": "result",
            "result": {
                "answer": (
                    "Your request was blocked by Apex Commercial Bank security guardrails. "
                    "Please rephrase without instruction-override or exfiltration attempts."
                ),
                "citations": [],
                "events": events,
                "hitl_required": False,
                "degraded": False,
                "blocked": True,
                "session_id": session_id,
            },
        }
        return

    _traced_memory_update(session_id, user_id, "user", guard.sanitized_text)
    memory_context = _traced_memory_load(session_id, user_id)
    prior_citations = memory_store.session_citations(session_id, user_id)
    mem_summary = memory_store.memory_summary(session_id, user_id)
    applied = []
    if mem_summary["prior_turns"] > 1:
        applied.append(f"{mem_summary['prior_turns'] - 1} earlier turn(s)")
    if mem_summary["preferences"]:
        applied.append("saved preferences")
    if mem_summary["recent_thumbs_down"]:
        applied.append(f"{mem_summary['recent_thumbs_down']} recent thumbs-down note(s)")
    if mem_summary["recurring_topics"]:
        applied.append("recurring topics")
    mem_event = AgentEvent(
        event_type="memory",
        node="memory",
        message="Loaded session + long-term memory context" + (f" ({', '.join(applied)})" if applied else " (first turn, nothing saved yet)"),
        detail={"chars": len(memory_context), "known_docs": [c.get("doc_id") for c in prior_citations], **mem_summary},
    )
    events.append(mem_event)
    yield {"type": "event", "event": mem_event}

    graph = get_compiled_graph()
    initial: AgentState = {
        "user_id": user_id,
        "role": role,
        "session_id": session_id,
        "raw_message": message,
        "message": guard.sanitized_text,
        "memory_context": memory_context,
        "prior_citations": prior_citations,
        "events": [],  # pre-graph events are tracked locally; keeps the panel scoped to this turn
        "errors": [],
        "degraded": False,
        "degradation_reason": None,
        "hitl_required": require_hitl,
        "revision_feedback": None,
        "revision_count": 0,
        "recursion_depth": 0,
        "retrieval_results": [],
        "research_batches": [],
        "tool_results": [],
        "citations": [],
        "validation_notes": [],
        "selected_tool": None,
        "tool_params": {},
        "research_topic": None,
        "draft_answer": "",
        "final_answer": "",
    }
    config = _graph_config(session_id, user_id, role)

    # Merge node updates locally so the final result reflects only this turn.
    final: dict[str, Any] = dict(initial)
    async for chunk in _drain_graph(graph, initial, config, final, events):
        yield chunk

    result = _build_result(final, events, session_id)
    _traced_memory_update(session_id, user_id, "assistant", result["answer"])
    memory_store.remember_citations(session_id, user_id, result["citations"])
    if final.get("intent") == "research" and final.get("research_topic"):
        memory_store.remember_topic(user_id, final["research_topic"])
    yield {"type": "result", "result": result}


async def _resume_turn(
    *,
    session_id: str,
    user_id: str,
    role: Role,
    decision: dict[str, Any],
) -> AsyncIterator[dict[str, Any]]:
    """Continue a thread paused at the `hitl` interrupt with the reviewer's decision."""
    graph = get_compiled_graph()
    config = _graph_config(session_id, user_id, role)

    snapshot = await graph.aget_state(config)
    if "hitl" not in (snapshot.next or ()):
        raise NoPendingApproval(f"Session {session_id} has no answer waiting for approval")

    # Start from the checkpointed state so the result carries the draft, citations, etc.
    final: dict[str, Any] = dict(snapshot.values)
    events: list[AgentEvent] = []
    async for chunk in _drain_graph(graph, Command(resume=decision), config, final, events):
        yield chunk

    result = _build_result(final, events, session_id)
    if decision.get("approved"):
        # The draft (original or last revision) was already written when it was produced.
        pass
    elif result.get("hitl_required"):
        # Send-back produced a revised draft that is waiting for approval again.
        _traced_memory_update(session_id, user_id, "assistant", result["answer"])
        memory_store.remember_citations(session_id, user_id, result["citations"])
    else:
        # Revision limit reached — record the stop message.
        _traced_memory_update(session_id, user_id, "assistant", result["answer"])
    logger.info(
        "hitl_resumed",
        session_id=session_id,
        approved=bool(decision.get("approved")),
        hitl_required=bool(result.get("hitl_required")),
        revision=int(final.get("revision_count") or 0),
    )
    yield {"type": "result", "result": result}


async def run_agent(**kwargs) -> dict[str, Any]:
    """Blocking variant of `stream_agent` — consumes the stream and returns the turn result."""
    result: dict[str, Any] = {}
    async for chunk in stream_agent(**kwargs):
        if chunk["type"] == "result":
            result = chunk["result"]
    return result
