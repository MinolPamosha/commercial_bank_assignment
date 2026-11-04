"""Research Agent — Recursive Language Model (RLM) style exploration.

Instead of stuffing every document into one prompt, the agent:
1) EXPLORES  — asks the LLM for a Python-executable search plan derived from *this* question:
               several targeted (query, metadata-filter) steps plus the analysis facets to count
2) DECOMPOSES — runs each step through hybrid search and batches the hits per document
3) ANALYZES   — scores each batch with the Python Analysis tool (deterministic theme counting)
4) RECURSES   — one depth-limited follow-up search driven by the dominant themes
5) AGGREGATES — hands compact batch findings (not raw documents) to the Response agent

The plan is validated against the corpus' real metadata vocabulary before execution, and a
deterministic plan is used when the LLM is unavailable, so the path is always runnable.
"""

from __future__ import annotations

import json
import re
from typing import Any

from backend.app.agents.llm import acomplete
from backend.app.agents.state import AgentState
from backend.app.models.schemas import AgentEvent, Role
from backend.app.observability.logging import get_logger
from backend.app.retrieval.hybrid import retriever
from backend.app.tools.knowledge_search import knowledge_search
from backend.app.tools.python_analysis import DEFAULT_THEME_PATTERNS, python_analysis

logger = get_logger(__name__)

MAX_DEPTH = 2
MAX_PLAN_STEPS = 4
FILTERABLE_KEYS = ("document_type", "department")

PLAN_SYSTEM = """You are the Research planner of a bank knowledge assistant.
Turn the user's broad question into a small search plan that a Python program will execute.
Reply with ONLY JSON:
{{
  "topic": <short snake_case slug for this research, e.g. payment_outage_root_causes>,
  "steps": [  // 2-{max_steps} steps, each a targeted hybrid search
    {{"label": <snake_case>, "query": <keyword-rich search string>, "filters": {{"document_type": <value or omit>, "department": <value or omit>}}}}
  ],
  "themes": {{ <theme_name>: [<3-6 literal words/phrases likely to appear in the documents>], ... }},   // 3-6 analysis facets
  "follow_up": {{"query_template": <search string containing the literal token {{themes}}>, "filters": {{...}}}} or null
}}
Allowed filter values — document_type: {doc_types}; department: {departments}.
Only use filters from those lists. Queries must be about the user's actual topic.
Themes are counted by literal keyword matching, so they must be CONCRETE causes/topics
(e.g. "connection_pool": ["pool", "max_connections", "connection exhausted"],
"idempotency": ["duplicate", "idempotency key", "replay"]) — never abstract buckets such as
"impact", "time_period", "root_causes" or "summary"."""


def _corpus_vocabulary() -> dict[str, set[str]]:
    vocab: dict[str, set[str]] = {k: set() for k in FILTERABLE_KEYS}
    for c in retriever.chunks:
        for k in FILTERABLE_KEYS:
            v = c.metadata.get(k)
            if v:
                vocab[k].add(str(v))
    return vocab


def _infer_filters(message: str, vocab: dict[str, set[str]]) -> dict[str, str]:
    """Cheap metadata hints from the question text (used by the deterministic plan)."""
    m = message.lower()
    filters: dict[str, str] = {}
    for dt in sorted(vocab["document_type"]):
        if dt in m or (dt == "incident" and "outage" in m):
            filters["document_type"] = dt
    for dept in sorted(vocab["department"]):
        if dept in m or (dept == "payments" and any(k in m for k in ("payment", "upi", "neft", "imps"))):
            filters["department"] = dept
    return filters


def deterministic_plan(message: str, vocab: dict[str, set[str]]) -> dict[str, Any]:
    """Fallback plan when the LLM is offline: broad → filtered → policy angle on the same question."""
    base = _infer_filters(message, vocab)
    words = " ".join(re.findall(r"[a-zA-Z0-9\-]{4,}", message))[:200]
    steps = [{"label": "broad_search", "query": words, "filters": {}}]
    if base:
        steps.append({"label": "filtered_search", "query": words, "filters": base})
    if "policy" in vocab["document_type"] and base.get("document_type") != "policy":
        steps.append({"label": "policy_context", "query": f"{words} policy requirement", "filters": {"document_type": "policy"}})
    return {
        "topic": re.sub(r"[^a-z0-9]+", "_", message.lower()).strip("_")[:48] or "research",
        "steps": steps,
        "themes": dict(DEFAULT_THEME_PATTERNS),
        "follow_up": {"query_template": "{themes} remediation runbook procedure", "filters": {"document_type": "runbook"}},
    }


def _sanitize_filters(raw: Any, vocab: dict[str, set[str]]) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    return {k: str(v) for k, v in raw.items() if k in FILTERABLE_KEYS and str(v) in vocab[k]}


def validate_plan(data: dict[str, Any], message: str, vocab: dict[str, set[str]]) -> dict[str, Any]:
    """Coerce an LLM plan into something safe to execute; fall back per-field when invalid."""
    fallback = deterministic_plan(message, vocab)

    steps: list[dict[str, Any]] = []
    for i, s in enumerate(data.get("steps") or []):
        if not isinstance(s, dict) or not str(s.get("query") or "").strip():
            continue
        steps.append(
            {
                "label": re.sub(r"[^a-z0-9_]", "_", str(s.get("label") or f"step_{i}").lower())[:40],
                "query": str(s["query"]).strip()[:300],
                "filters": _sanitize_filters(s.get("filters"), vocab),
            }
        )
        if len(steps) >= MAX_PLAN_STEPS:
            break

    themes: dict[str, list[str]] = {}
    for name, kws in (data.get("themes") or {}).items() if isinstance(data.get("themes"), dict) else []:
        if isinstance(kws, list) and kws:
            themes[re.sub(r"[^a-z0-9_]", "_", str(name).lower())[:40]] = [str(k)[:60] for k in kws[:6]]

    # Incident research additionally counts the known operational facets (connection pool,
    # idempotency, ...) so recurring-cause detection does not hinge on the planner's wording.
    if any(s["filters"].get("document_type") == "incident" for s in steps):
        themes = {**DEFAULT_THEME_PATTERNS, **themes}

    follow_up = data.get("follow_up")
    if isinstance(follow_up, dict) and "{themes}" in str(follow_up.get("query_template") or ""):
        follow_up = {
            "query_template": str(follow_up["query_template"])[:300],
            "filters": _sanitize_filters(follow_up.get("filters"), vocab),
        }
    else:
        follow_up = fallback["follow_up"]

    topic = re.sub(r"[^a-z0-9_]", "_", str(data.get("topic") or "").lower()).strip("_")[:48]
    return {
        "topic": topic or fallback["topic"],
        "steps": steps or fallback["steps"],
        "themes": themes or fallback["themes"],
        "follow_up": follow_up,
    }


async def generate_search_plan(message: str, memory: str) -> tuple[dict[str, Any], str]:
    """RLM step 1+2: LLM proposes the plan, Python validates it. Returns (plan, decided_by)."""
    vocab = _corpus_vocabulary()
    try:
        raw = await acomplete(
            system=PLAN_SYSTEM.format(
                max_steps=MAX_PLAN_STEPS,
                doc_types=sorted(vocab["document_type"]),
                departments=sorted(vocab["department"]),
            ),
            user=f"Conversation memory:\n{memory or '(none)'}\n\nResearch question:\n{message}",
        )
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(match.group(0)) if match else {}
        if data.get("steps"):
            return validate_plan(data, message, vocab), "llm"
    except Exception as exc:  # noqa: BLE001
        logger.warning("research_plan_llm_failed", error=str(exc))
    return deterministic_plan(message, vocab), "deterministic"


def _to_citation(r: dict[str, Any]) -> dict[str, Any]:
    md = r.get("metadata") or {}
    return {
        "doc_id": r["doc_id"],
        "title": r["title"],
        "department": md.get("department", ""),
        "document_type": md.get("document_type", ""),
        "access_level": md.get("access_level", ""),
        "score": float(r.get("score") or 0),
        "snippet": r.get("snippet", ""),
    }


async def research_node(state: AgentState) -> dict:
    role = state["role"] if isinstance(state["role"], Role) else Role(state["role"])
    depth = int(state.get("recursion_depth") or 0)
    events = [
        AgentEvent(event_type="node", node="research", message="Research agent (RLM) exploring corpus"),
        AgentEvent(event_type="rlm", node="research", message=f"Recursion depth={depth}/{MAX_DEPTH}"),
    ]

    plan, decided_by = await generate_search_plan(state["message"], state.get("memory_context") or "")
    events.append(
        AgentEvent(
            event_type="rlm",
            node="research",
            message=f"Generated Python search plan ({decided_by}): {len(plan['steps'])} searches, {len(plan['themes'])} analysis facets",
            detail={"plan": plan["steps"], "themes": plan["themes"], "topic": plan["topic"], "decided_by": decided_by},
        )
    )

    batches: list[dict] = []
    all_results: list[dict] = []
    snippets: list[str] = []
    analysis_denied = False
    errors = list(state.get("errors") or [])
    degraded = bool(state.get("degraded"))

    for step in plan["steps"]:
        events.append(AgentEvent(event_type="tool_call", node="research", message=f"RLM sub-search: {step['label']}", detail=step))
        result = await knowledge_search(step["query"], role=role, filters=step.get("filters") or None)
        if not result.get("ok"):
            errors.append(result.get("error", "retrieval failed"))
            degraded = degraded or bool(result.get("degraded"))
            events.append(AgentEvent(event_type="error", node="research", message=f"Sub-search failed: {step['label']}", detail=result))
            continue

        # Decompose: one batch per document so each is analyzed in isolation.
        by_doc: dict[str, list] = {}
        for d in result.get("results") or []:
            by_doc.setdefault(d["doc_id"], []).append(d)

        for doc_id, items in by_doc.items():
            batch_snips = [i.get("snippet") or "" for i in items]
            analysis = await python_analysis(role=role, snippets=batch_snips, theme_patterns=plan["themes"])
            if not analysis.get("ok"):
                analysis_denied = True
            batch = {
                "doc_id": doc_id,
                "plan_label": step["label"],
                "chunks": len(items),
                "analysis": analysis.get("analysis"),
                "titles": sorted({i.get("title") for i in items}),
            }
            batches.append(batch)
            snippets.extend(batch_snips)
            all_results.extend(items)
            events.append(AgentEvent(event_type="rlm", node="research", message=f"Analyzed batch {doc_id}", detail=batch))

    if analysis_denied:
        events.append(
            AgentEvent(
                event_type="guardrail",
                node="research",
                message=f"python_analysis not permitted for role '{role.value}' — batches carry evidence only",
            )
        )

    # Aggregate across all batches.
    aggregate = await python_analysis(role=role, snippets=snippets, theme_patterns=plan["themes"])
    agg = aggregate.get("analysis") or {}
    if aggregate.get("ok") and snippets and not agg.get("theme_counts") and plan["themes"] != DEFAULT_THEME_PATTERNS:
        # Self-correction: the planner's facets matched nothing in the evidence, so re-analyze
        # with the default operational facets rather than reporting "no themes".
        events.append(
            AgentEvent(
                event_type="rlm",
                node="research",
                message="Planner facets matched no evidence — re-analyzing with default facets",
                detail={"planner_facets": list(plan["themes"])},
            )
        )
        plan["themes"] = dict(DEFAULT_THEME_PATTERNS)
        aggregate = await python_analysis(role=role, snippets=snippets, theme_patterns=plan["themes"])
        agg = aggregate.get("analysis") or {}
    events.append(
        AgentEvent(
            event_type="rlm",
            node="research",
            message="Aggregated themes across batches" if aggregate.get("ok") else "Aggregation skipped (tool not permitted)",
            detail=agg,
        )
    )

    # Recursive refinement: one depth-limited follow-up search steered by the dominant themes.
    themes = agg.get("top_themes") or []
    if depth < MAX_DEPTH and themes and plan.get("follow_up"):
        follow_query = plan["follow_up"]["query_template"].replace("{themes}", " ".join(themes))
        events.append(
            AgentEvent(
                event_type="rlm",
                node="research",
                message="Recursive sub-agent call for supporting context",
                detail={"themes": themes, "query": follow_query, "filters": plan["follow_up"].get("filters")},
            )
        )
        follow = await knowledge_search(follow_query, role=role, filters=plan["follow_up"].get("filters") or None)
        all_results.extend(follow.get("results") or [])
        depth += 1

    # Dedupe citations by doc_id keeping best score.
    best: dict[str, dict] = {}
    for c in map(_to_citation, all_results):
        prev = best.get(c["doc_id"])
        if not prev or c["score"] > prev["score"]:
            best[c["doc_id"]] = c

    return {
        "active_node": "research",
        "research_topic": plan["topic"],
        "research_batches": batches,
        "retrieval_results": all_results,
        "citations": list(best.values()),
        "tool_results": [{"tool": "python_analysis", "result": aggregate}],
        "events": events,
        "errors": errors,
        "degraded": degraded,
        "degradation_reason": "retrieval_failure" if degraded and not state.get("degradation_reason") else state.get("degradation_reason"),
        "recursion_depth": depth,
        "route": "respond",
        "draft_answer": "",
    }
