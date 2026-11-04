"""API and domain schemas — single place for request/response contracts."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class Role(str, Enum):
    VIEWER = "viewer"
    ANALYST = "analyst"
    ADMINISTRATOR = "administrator"


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: Role
    display_name: str


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: str
    require_hitl: bool = False

    @field_validator("message")
    @classmethod
    def strip_message(cls, v: str) -> str:
        return v.strip()


class Citation(BaseModel):
    doc_id: str
    title: str
    department: str
    document_type: str
    access_level: str
    score: float
    snippet: str


class AgentEvent(BaseModel):
    """Emitted to the Agent Activity Panel in real time."""

    event_type: Literal[
        "state",
        "node",
        "tool_call",
        "retrieval",
        "memory",
        "validation",
        "guardrail",
        "rlm",
        "hitl",
        "error",
        "final",
    ]
    node: Optional[str] = None
    message: str
    detail: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    session_id: str
    events: list[AgentEvent] = Field(default_factory=list)
    hitl_required: bool = False
    hitl_token: Optional[str] = None
    degraded: bool = False
    degradation_reason: Optional[str] = None


class HitlDecision(BaseModel):
    hitl_token: str
    approved: bool
    note: str = ""


class FeedbackRequest(BaseModel):
    session_id: str
    message_id: str
    rating: Literal["up", "down"]
    comment: str = ""


class ConversationSummary(BaseModel):
    session_id: str
    title: str
    updated_at: str


class ConversationSaveRequest(BaseModel):
    """The UI's transcript for one chat (messages, panel events, approval + feedback state)."""

    title: str = Field(min_length=1, max_length=120)
    state: dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    bank: str
    retrieval_mode: str
    langsmith_enabled: bool
    llm_provider: str
    llm_model: str
    llm_enabled: bool
