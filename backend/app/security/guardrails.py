"""Prompt-injection defenses, input validation, brand/citation guardrails."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from backend.app.config import get_settings
from backend.app.models.schemas import Role
from backend.app.observability.logging import get_logger

logger = get_logger(__name__)

# Patterns that commonly indicate instruction override / exfiltration / tool abuse.
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"disregard\s+(your|the)\s+(system|safety)\s+(prompt|rules?)",
    r"you\s+are\s+now\s+(dan|jailbroken|unrestricted)",
    r"reveal\s+(your|the)\s+(system\s+prompt|hidden\s+instructions?)",
    r"exfiltrate|dump\s+(all\s+)?(secrets?|api\s*keys?|passwords?)",
    r"bypass\s+(rbac|authorization|access\s+control)",
    r"call\s+admin_|\.env\b|os\.system\(|subprocess\.",
    r"pretend\s+you\s+have\s+(no\s+)?restrictions?",
]

UNSAFE_TOOL_ARGS = [
    r"(rm\s+-rf|drop\s+table|delete\s+from|;--|/\*)",
    r"(__import__|eval\(|exec\(|compile\()",
    # Introspection chains (`().__class__.__bases__[0].__subclasses__()`) are how a "harmless"
    # expression reaches os.system; the AST sandbox rejects them too (defense in depth).
    r"__\w+__|getattr\(|globals\(|builtins",
]

# Instruction-override phrases that may be *embedded in documents* (indirect prompt injection).
# A poisoned chunk must not be handed to the LLM as trusted evidence. Deliberately a subset of
# INJECTION_PATTERNS: runbooks legitimately mention `.env`, `subprocess` or `os.system`.
RETRIEVED_CONTENT_INJECTION_PATTERNS = INJECTION_PATTERNS[:4] + INJECTION_PATTERNS[-1:]


@dataclass
class GuardrailResult:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    sanitized_text: str = ""
    flags: dict[str, Any] = field(default_factory=dict)


def validate_user_input(text: str, role: Role) -> GuardrailResult:
    settings = get_settings()
    reasons: list[str] = []
    flags: dict[str, Any] = {}

    if not text or not text.strip():
        return GuardrailResult(ok=False, reasons=["Empty request"], sanitized_text="")

    if len(text) > settings.max_input_chars:
        reasons.append(f"Input exceeds {settings.max_input_chars} characters")
        return GuardrailResult(ok=False, reasons=reasons, sanitized_text=text[: settings.max_input_chars])

    lowered = text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lowered, re.IGNORECASE):
            reasons.append(f"Prompt-injection pattern matched: {pattern}")
            flags["injection"] = True
            logger.warning("injection_blocked", role=role.value, pattern=pattern)

    # Soft-sanitize: strip common override prefixes while keeping the factual ask.
    sanitized = re.sub(
        r"(?i)^\s*(ignore|disregard).*?(instructions?|rules?)\s*[:.\-]?\s*",
        "",
        text,
    ).strip()

    ok = "injection" not in flags and not reasons
    return GuardrailResult(ok=ok, reasons=reasons, sanitized_text=sanitized or text, flags=flags)


def validate_tool_params(tool_name: str, params: dict[str, Any], role: Role) -> GuardrailResult:
    from backend.app.security.auth import role_allows_tool

    reasons: list[str] = []
    if not role_allows_tool(role, tool_name):
        reasons.append(f"Role '{role.value}' is not permitted to use tool '{tool_name}'")
        logger.warning("tool_rbac_denied", role=role.value, tool=tool_name)
        return GuardrailResult(ok=False, reasons=reasons)

    blob = " ".join(str(v) for v in params.values())
    for pattern in UNSAFE_TOOL_ARGS:
        if re.search(pattern, blob, re.IGNORECASE):
            reasons.append(f"Unsafe tool argument pattern: {pattern}")
            return GuardrailResult(ok=False, reasons=reasons)

    return GuardrailResult(ok=True, reasons=[], sanitized_text=blob)


def contains_embedded_injection(text: str) -> str | None:
    """Return the matched pattern if retrieved text carries instruction-override phrasing."""
    lowered = (text or "").lower()
    for pattern in RETRIEVED_CONTENT_INJECTION_PATTERNS:
        if re.search(pattern, lowered, re.IGNORECASE):
            return pattern
    return None


def screen_retrieved_content(chunks: list[dict], role: Role) -> tuple[list[dict], dict[str, int]]:
    """Validate evidence before it reaches the LLM (defense in depth).

    1. Access level: drop chunks the caller's role may not see, even if the index returned them.
    2. Indirect prompt injection: drop chunks whose *content* tries to instruct the model
       ("ignore previous instructions…"); a document is evidence, never an instruction source.
    Returns (surviving_chunks, dropped_counts) so the caller can surface the guardrail firing.
    """
    access_rank = {"public": 0, "internal": 1, "confidential": 2, "restricted": 3}
    role_max = {
        Role.VIEWER: "internal",
        Role.ANALYST: "confidential",
        Role.ADMINISTRATOR: "restricted",
    }[role]
    max_rank = access_rank[role_max]
    kept: list[dict] = []
    dropped = {"access_level": 0, "embedded_injection": 0}
    for c in chunks:
        level = (c.get("metadata") or {}).get("access_level", "internal")
        if access_rank.get(level, 1) > max_rank:
            dropped["access_level"] += 1
            logger.info("chunk_filtered_rbac", doc=c.get("doc_id"), level=level, role=role.value)
            continue
        pattern = contains_embedded_injection(c.get("text", ""))
        if pattern:
            dropped["embedded_injection"] += 1
            logger.warning("chunk_filtered_injection", doc=c.get("doc_id"), pattern=pattern)
            continue
        kept.append(c)
    return kept, dropped


def validate_retrieved_content(chunks: list[dict], role: Role) -> list[dict]:
    """Convenience wrapper over `screen_retrieved_content` for callers that only need the chunks."""
    kept, _ = screen_retrieved_content(chunks, role)
    return kept


# Corpus document-id shapes (INC-2025-0142, POL-INFOSEC-02, RB-PAY-017, ARCH-PAY-001, MEET-2025-02-RISK…).
# Used to catch invented ids that the model writes *without* the [doc:…] wrapper.
DOC_ID_PATTERN = re.compile(r"\b(?:INC|POL|RB|ARCH|PROD|MEET|BOARD)-[A-Z0-9][A-Z0-9-]{2,}\b")


def check_citations(answer: str, citations: list[dict], grounded_ids: set[str] | None = None) -> GuardrailResult:
    """Flag document references the retrieved evidence cannot back.

    Two checks:
    - `[doc:ID]` citations whose ID is not a retrieved document → hallucinated_citations
    - bare document ids (e.g. "INC-2025-0999") that appear neither as a retrieved document
      nor inside any retrieved snippet → unverified_ids. Ids mentioned *inside* evidence text
      (an incident report referencing a related incident) are grounded and allowed.

    `grounded_ids` are identifiers that came from another trusted source this turn, typically
    the records a live MCP tool returned (an open ticket id is real even if no document mentions it).
    """
    known = {str(c.get("doc_id") or c.get("id") or "") for c in citations}
    mentioned_in_evidence: set[str] = set(grounded_ids or ())
    for c in citations:
        mentioned_in_evidence.update(DOC_ID_PATTERN.findall(str(c.get("snippet") or "")))

    claimed = set(re.findall(r"\[doc:([^\]]+)\]", answer, flags=re.IGNORECASE))
    hallucinated = {c for c in claimed if c not in known}

    bare = set(DOC_ID_PATTERN.findall(re.sub(r"\[doc:[^\]]+\]", "", answer)))
    unverified = {b for b in bare if b not in known and b not in mentioned_in_evidence}

    if hallucinated or unverified:
        reasons = []
        if hallucinated:
            reasons.append(f"Hallucinated citations: {sorted(hallucinated)}")
        if unverified:
            reasons.append(f"Unverified document ids: {sorted(unverified)}")
        return GuardrailResult(
            ok=False,
            reasons=reasons,
            flags={"hallucinated_citations": sorted(hallucinated), "unverified_ids": sorted(unverified)},
        )
    return GuardrailResult(ok=True)


def brand_system_preamble() -> str:
    settings = get_settings()
    return (
        f"You are {settings.assistant_name}, the official internal knowledge assistant for "
        f"{settings.bank_name}. Protect customer trust and brand value: never invent policies, "
        f"never disclose secrets, never claim to override compliance. Prefer citing evidence as "
        f"[doc:<id>]. If evidence is insufficient, say so clearly."
    )
