"""Shared LangGraph state — every node reads/writes this contract."""

from __future__ import annotations

from typing import Any, Literal, Optional, TypedDict

from backend.app.models.schemas import AgentEvent, Role


class AgentState(TypedDict, total=False):
    # Identity / request
    user_id: str
    role: Role
    session_id: str
    raw_message: str
    message: str

    # Routing
    intent: Literal["qa", "research", "tool", "admin", "chitchat"]
    route: Literal["retrieve", "research", "tools", "respond", "hitl", "revise", "end"]
    plan: list[str]
    selected_tool: Optional[str]  # supervisor's tool choice (validated against the catalog)
    tool_params: dict[str, Any]  # LLM-proposed params; re-validated by the tool gate
    active_node: str

    # Evidence
    research_topic: Optional[str]  # slug from the RLM planner, persisted to long-term memory
    retrieval_results: list[dict[str, Any]]
    research_batches: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    citations: list[dict[str, Any]]

    # Memory
    memory_context: str
    # Documents cited earlier in this session. Follow-ups answered from memory may reference
    # them; validation treats them as grounded and surfaces the ones the answer actually uses.
    prior_citations: list[dict[str, Any]]

    # Output
    draft_answer: str
    final_answer: str
    validation_ok: bool
    validation_notes: list[str]

    # Control / resilience
    # `events` holds only the *last node's* events (no append reducer): the API layer
    # collects per-node updates as they stream, so the checkpointed thread stays small
    # across many turns instead of accumulating every event ever emitted.
    events: list[AgentEvent]
    errors: list[str]
    degraded: bool
    degradation_reason: Optional[str]
    hitl_required: bool  # True while the graph is paused at the `hitl` interrupt
    # Send-back revision: the reviewer's note steers the Response agent, then HITL pauses again.
    revision_feedback: Optional[str]
    revision_count: int
    recursion_depth: int
