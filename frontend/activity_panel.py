"""Agent Activity Panel. Renders backend `AgentEvent`s for two audiences at once.

* Readable view (default): one collapsible *step* per LangGraph node. The title carries a
  short summary (route decision, documents found, themes, validation verdict) plus the
  node id and start time. Inside: readable rows under a "Steps" tab and every raw event
  (type, node, message, timestamp, JSON) under a "Raw events" tab.
* Developer view (sidebar, Advanced): every step open, raw events shown directly.

Nothing the backend emits is hidden. The panel only adds a readable layer on top.

Spec coverage (Agent Activity Panel requirements):
  current agent state: status line       | active LangGraph node: pipeline line and status
  tool calls: 🛠 rows                     | retrieval status: 🔍 rows and evidence count
  memory updates: 🧠 rows                 | validation results: ✅ / ⚠️ rows and status
  final response: ✍️ / 🏁 rows            | HITL: 🙋 row and the "Approval" stage
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable

import streamlit as st

import ui

# --------------------------------------------------------------------------------------
# Static vocabulary: what each node / event type means in plain words
# --------------------------------------------------------------------------------------

NODE_LABELS: dict[str, str] = {
    "memory": "Memory",
    "input_validation": "Security guardrail",
    "supervisor": "Supervisor agent",
    "retrieval": "Retrieval agent",
    "research": "Research agent (RLM)",
    "tools": "Tools agent",
    "response": "Response agent",
    "validate": "Validation",
    "hitl": "Human approval",
    "graph": "Graph",
}

NODE_ICONS: dict[str, str] = {
    "memory": "🧠",
    "input_validation": "🛡️",
    "supervisor": "🧭",
    "retrieval": "🔍",
    "research": "🔬",
    "tools": "🛠️",
    "response": "✍️",
    "validate": "✅",
    "hitl": "🙋",
    "graph": "🏁",
}

NODE_PURPOSE: dict[str, str] = {
    "memory": "brings in earlier turns from this chat and anything already known about you",
    "input_validation": "checks the request for prompt injection or attempts to pull out secrets",
    "supervisor": "reads the question, decides what kind of task it is, and picks who should handle it",
    "retrieval": "searches documents by meaning and by keywords, then merges and reranks the hits",
    "research": "breaks a broad question into a search plan, studies documents in batches, and pulls themes together",
    "tools": "calls live bank tools (directory, catalogue, incidents, analysis), only if your role allows them",
    "response": "writes the answer from the evidence, citing documents as [doc:…]",
    "validate": "checks for made up citations before you see the answer",
    "hitl": "pauses so you can approve the answer or send it back",
}

EVENT_ICONS: dict[str, str] = {
    "memory": "🧠",
    "node": "▶️",
    "state": "📍",
    "retrieval": "🔍",
    "tool_call": "🛠️",
    "rlm": "🔁",
    "validation": "✅",
    "guardrail": "🛡️",
    "hitl": "🙋",
    "error": "❌",
    "final": "🏁",
}

ROUTE_LABELS: dict[str, str] = {
    "retrieve": "Retrieval agent (one pass over the documents)",
    "research": "Research agent (a deeper look across many documents, the RLM path)",
    "tools": "Tools agent (enterprise data through MCP)",
    "respond": "Response agent straight away (no search needed)",
}

INTENT_LABELS: dict[str, str] = {
    "qa": "a factual question",
    "research": "a research question that needs several documents",
    "tool": "a request for enterprise data (a tool call)",
    "admin": "an admin request",
    "chitchat": "small talk",
}

THEME_LABELS: dict[str, str] = {
    "connection_pool": "connection pool exhaustion",
    "idempotency": "duplicate processing (missing idempotency)",
    "partner_timeout": "partner and external timeouts",
    "capacity": "capacity and traffic spikes",
    "failover": "failover and cross region issues",
}

# Pipeline strip: (stage id, label, nodes that belong to the stage)
STAGES: list[tuple[str, str, frozenset[str]]] = [
    ("memory", "Memory", frozenset({"memory"})),
    ("supervisor", "Supervisor", frozenset({"supervisor"})),
    ("evidence", "Evidence", frozenset({"retrieval", "research", "tools"})),
    ("response", "Answer", frozenset({"response"})),
    ("validate", "Validation", frozenset({"validate"})),
    ("hitl", "Approval", frozenset({"hitl"})),
]


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _fmt_filters(filters: dict[str, Any] | None) -> str:
    if not filters:
        return "no metadata filter"
    return ", ".join(f"{k}={v}" for k, v in filters.items())


def _theme(name: str) -> str:
    return THEME_LABELS.get(name, name.replace("_", " "))


def _themes_sentence(analysis: dict[str, Any] | None) -> str:
    counts = (analysis or {}).get("theme_counts") or {}
    if not counts:
        return "no recurring theme detected"
    return ", ".join(f"{_theme(t)} ({n}×)" for t, n in counts.items())


def _time(ev: dict[str, Any]) -> str:
    ts = str(ev.get("timestamp") or "")
    return ts[11:19] if len(ts) >= 19 else ts


@dataclass
class Described:
    icon: str
    headline: str
    lines: list[str] = field(default_factory=list)
    tone: str = "normal"  # normal | good | warn | bad


# --------------------------------------------------------------------------------------
# Event → plain language
# --------------------------------------------------------------------------------------


def describe_event(ev: dict[str, Any]) -> Described:
    """Translate one backend AgentEvent into a readable headline plus summary lines."""
    et = ev.get("event_type", "?")
    node = ev.get("node") or ""
    msg = ev.get("message", "")
    d: dict[str, Any] = ev.get("detail") or {}
    icon = EVENT_ICONS.get(et, "•")

    if et == "memory":
        chars = d.get("chars", 0)
        note = "nothing stored yet, this is a fresh chat" if not chars else f"{chars} characters of context loaded"
        lines = [f"Recalled this chat's history and what is known about you ({note})."]
        prior = int(d.get("prior_turns") or 0) - 1
        if prior > 0:
            lines.append(f"**Earlier turns in this chat:** {prior}.")
        if d.get("preferences"):
            lines.append(f"**Saved preferences (from your feedback):** {d['preferences']}")
        if d.get("recent_thumbs_down"):
            lines.append(f"**Recent thumbs down:** {d['recent_thumbs_down']}, so the answer will try to address those notes.")
        if d.get("recurring_topics"):
            lines.append("**Recurring topics:** " + ", ".join(str(t).replace("_", " ") for t in d["recurring_topics"]) + ".")
        known = d.get("known_docs") or []
        if known:
            lines.append(f"**Documents already shown in this chat:** {', '.join(known[:6])}.")
        return Described(icon, "Memory loaded", lines)

    if et == "guardrail":
        if node == "input_validation":
            reasons = d.get("reasons") or []
            return Described(
                icon,
                "Request blocked by the security guardrails",
                [f"Reason: {r}" for r in reasons] or [msg],
                tone="bad",
            )
        if node == "retrieval":
            parts = []
            if d.get("access_level"):
                parts.append(f"{d['access_level']} above your access level")
            if d.get("embedded_injection"):
                parts.append(f"{d['embedded_injection']} containing instructions aimed at the assistant (prompt injection inside a document)")
            return Described(icon, "Some retrieved passages were withheld", [f"Dropped before the model saw them: {'; '.join(parts) or msg}."], tone="warn")
        error = d.get("error") or msg
        tool = msg.split(":")[-1].strip() if ":" in msg else "tool"
        denied = "not allowed" in str(error).lower() or "denied" in str(error).lower() or "role" in str(error).lower()
        headline = f"Tool `{tool}` {'is not allowed for your role' if denied else 'failed'}"
        return Described(icon, headline, [str(error)], tone="bad" if denied else "warn")

    if et == "node":
        label = NODE_LABELS.get(node, node)
        purpose = NODE_PURPOSE.get(node)
        return Described(NODE_ICONS.get(node, icon), f"{label} started", [purpose[0].upper() + purpose[1:] + "." if purpose else msg])

    if et == "state":
        if "route" in d:
            intent = d.get("intent", "?")
            route = d.get("route", "?")
            lines = [
                f"This looks like **{INTENT_LABELS.get(intent, intent)}**, so it goes to the "
                f"**{ROUTE_LABELS.get(route, route)}**."
            ]
            plan = d.get("plan") or []
            if plan:
                lines.append("The plan:")
                lines.extend(f"{i}. {step}" for i, step in enumerate(plan, 1))
            return Described("🧭", "Supervisor chose the route", lines)
        if msg.lower().startswith("active node"):
            return Described(icon, f"Now running: {NODE_LABELS.get(node, node)}", [])
        return Described(icon, msg, [])

    if et == "retrieval":
        if "doc_ids" in d:
            ids = d.get("doc_ids") or []
            uniq = list(dict.fromkeys(ids))
            if not uniq:
                return Described(icon, "No matching documents found", ["Either your role cannot see them, or nothing matched."], tone="warn")
            return Described(
                icon,
                f"Found {len(ids)} passages across {len(uniq)} document(s)",
                ["Documents: " + ", ".join(f"`{i}`" for i in uniq)],
                tone="good",
            )
        return Described(icon, "Searching the document library", ["Searching by meaning (embeddings) and by keywords (BM25), then merging both into one ranking."])

    if et == "tool_call":
        if msg.startswith("knowledge_search"):
            hits = d.get("hits", 0)
            ok = d.get("ok", True)
            return Described(
                icon,
                f"Knowledge search finished with {hits} hit(s)",
                [f"Filters used: {_fmt_filters(d.get('filters'))}."] + ([] if ok else ["The search reported a problem. See the raw events."]),
                tone="good" if ok else "warn",
            )
        if msg.startswith("RLM sub-search"):
            label = d.get("label") or msg.split(":")[-1].strip()
            return Described(
                icon,
                f"Focused search: {label.replace('_', ' ')}",
                [f"Query: “{d.get('query', '')}”. Filters: {_fmt_filters(d.get('filters'))}."],
            )
        if msg.startswith("Invoking"):
            tool = msg.replace("Invoking", "").strip()
            via = "the enterprise MCP server" if tool.startswith("mcp_") else "the sandboxed Python analysis tool"
            return Described(icon, f"Calling tool `{tool}`", [f"Through {via}. Your role is checked before the call goes out."])
        return Described(icon, msg, [])

    if et == "rlm":
        if msg.startswith("Recursion depth"):
            return Described(icon, f"Research depth: {msg.split('=')[-1]}", ["How many times the research agent has gone a level deeper."])
        if "plan" in d and msg.startswith("Generated"):
            lines = ["Rather than one giant prompt, the agent runs a small Python plan of focused searches:"]
            for i, step in enumerate(d.get("plan") or [], 1):
                lines.append(
                    f"{i}. **{str(step.get('label', '')).replace('_', ' ')}**: “{step.get('query', '')}” "
                    f"({_fmt_filters(step.get('filters'))})"
                )
            return Described(icon, "Search plan ready", lines)
        if msg.startswith("Analyzed batch"):
            doc = d.get("doc_id") or msg.split()[-1]
            titles = d.get("titles") or []
            title = f", {titles[0]}" if titles else ""
            return Described(
                "📄",
                f"Read document `{doc}`{title}",
                [f"{d.get('chunks', 0)} passage(s). Themes: {_themes_sentence(d.get('analysis'))}."],
            )
        if msg.startswith("Aggregated"):
            top = d.get("top_themes") or []
            lines = [f"Passages analyzed: {d.get('documents_analyzed', 0)}."]
            if top:
                lines.append("Root causes that keep coming back: " + ", ".join(f"**{_theme(t)}**" for t in top) + ".")
                lines.append("Full count: " + _themes_sentence(d) + ".")
            return Described("📊", "Findings pulled together across all batches", lines, tone="good")
        if msg.startswith("Recursive"):
            themes = d.get("themes") or []
            return Described(
                icon,
                "Going one level deeper",
                ["Looking up the runbooks that fix: " + ", ".join(_theme(t) for t in themes) + "."],
            )
        return Described(icon, msg, [])

    if et == "validation":
        if msg.startswith("Validation complete"):
            ok = d.get("ok", True)
            notes = d.get("notes") or []
            lines = ["Every citation matches a document that was actually retrieved. The answer is cleared."] if ok and not notes else notes
            return Described("✅" if ok else "❌", "Validation passed" if ok else "Validation failed", lines, tone="good" if ok else "bad")
        if msg.startswith("Removed hallucinated"):
            bad = d.get("hallucinated_citations") or []
            return Described(
                "⚠️",
                "Removed citations that did not match any retrieved document",
                [f"Removed: {', '.join(bad)}" if bad else msg],
                tone="warn",
            )
        if msg.startswith("Serving degraded"):
            return Described("⚠️", "Partial answer", [f"Reason: {d.get('reason')}. The system carried on with what it had."], tone="warn")
        if msg.startswith("Follow-up grounded"):
            ids = ", ".join(f"`{i}`" for i in d.get("doc_ids") or [])
            return Described("🔗", "Follow-up grounded on earlier evidence", [f"No new search was needed. The answer draws on documents already shown in this conversation: {ids}."], tone="good")
        return Described(icon, msg, [])

    if et == "hitl":
        preview = (d.get("draft_preview") or "").strip()
        lines = ["Use **Approve** or **Send back** in the chat. Send back needs a note saying what should change."]
        if preview:
            one_line = " ".join(preview.split())[:200]
            lines.append(f"> Draft preview: {one_line}{'…' if len(preview) > 200 else ''}")
        return Described(icon, "Waiting for your approval", lines, tone="warn")

    if et == "error":
        return Described(icon, "Something failed, so a fallback was used", [msg], tone="warn")

    if et == "final":
        if node == "response":
            return Described("✍️", "Draft answer written", ["Passing it to validation."], tone="good")
        return Described(icon, "Turn complete", [], tone="good")

    return Described(icon, msg, [])


# --------------------------------------------------------------------------------------
# Turn-level summary (current state)
# --------------------------------------------------------------------------------------


@dataclass
class TurnSummary:
    route: str | None = None
    intent: str | None = None
    docs: set[str] = field(default_factory=set)
    tool_calls: int = 0
    denied_tools: int = 0
    validation: str | None = None
    answer_ready: bool = False
    hitl_waiting: bool = False
    degraded_reason: str | None = None
    blocked: bool = False
    last_node: str | None = None
    touched: set[str] = field(default_factory=set)


def summarize(events: Iterable[dict[str, Any]]) -> TurnSummary:
    s = TurnSummary()
    for ev in events:
        et, node, msg, d = ev.get("event_type"), ev.get("node") or "", ev.get("message", ""), ev.get("detail") or {}
        s.last_node = node or s.last_node
        if node:
            s.touched.add(node)
        if et == "state" and "route" in d:
            s.route, s.intent = d.get("route"), d.get("intent")
        elif et == "retrieval" and "doc_ids" in d:
            s.docs.update(d.get("doc_ids") or [])
        elif et == "rlm" and msg.startswith("Analyzed batch") and d.get("doc_id"):
            s.docs.add(d["doc_id"])
        elif et == "tool_call":
            # The tools node emits two events per call ("Invoking X" then "X returned data"); count the call once.
            if not msg.endswith("returned data"):
                s.tool_calls += 1
        elif et == "guardrail":
            if node == "input_validation":
                s.blocked = True
            elif node != "retrieval":  # retrieval guardrails withhold passages; they are not tool denials
                s.denied_tools += 1
        elif et == "validation":
            if msg.startswith("Validation complete"):
                s.validation = "passed" if d.get("ok", True) else "failed"
            elif msg.startswith("Serving degraded"):
                s.degraded_reason = d.get("reason")
            elif msg.startswith("Removed"):
                s.validation = "repaired"
        elif et == "final":
            s.answer_ready = True
        elif et == "hitl":
            s.hitl_waiting = True
        if node == "hitl" and et in ("final", "validation"):
            s.hitl_waiting = False  # the reviewer has decided; the interrupt was resumed
    return s


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------


def _stage_strip(summary: TurnSummary, live: bool, answer_streaming: bool, approval: str | None = None) -> None:
    """Pipeline chips: done · running · waiting · pending · skipped · blocked. State is in words, not just colour."""
    active_stage = None
    if live:
        for sid, _, nodes in STAGES:
            if summary.last_node in nodes:
                active_stage = sid
        if answer_streaming and "response" not in summary.touched:
            active_stage = "response"

    chips: list[str] = []
    if summary.blocked:
        chips.append(ui.chip("Security check", "blocked", "blocked"))
    for sid, label, nodes in STAGES:
        touched = bool(summary.touched & nodes) or (sid == "response" and answer_streaming)
        if sid == "evidence" and summary.touched & nodes:
            label = {"retrieval": "Retrieval", "research": "Research", "tools": "Tools"}[next(iter(summary.touched & nodes))]
        if summary.blocked:
            chips.append(ui.chip(label, "skipped"))
        elif sid == active_stage:
            chips.append(ui.chip(label, "running", "running"))
        elif sid == "hitl" and summary.hitl_waiting:
            chips.append(ui.chip(label, "waiting", "waiting for you"))
        elif sid == "hitl" and approval:
            chips.append(ui.chip(label, "done", approval))
        elif touched:
            chips.append(ui.chip(label, "done"))
        elif not live and sid in ("hitl", "evidence"):
            chips.append(ui.chip(label, "skipped"))
        else:
            chips.append(ui.chip(label, "pending"))
    ui.stage_strip(chips)


def _status_box(summary: TurnSummary, live: bool, answer_streaming: bool) -> None:
    if summary.blocked:
        ui.status_card(
            "🛡️ <b>Right now:</b> the request was blocked by the security guardrail before any agent ran.", [], tone="bad"
        )
        return

    if live:
        if answer_streaming and "response" not in summary.touched:
            now = "✍️ The Response agent is writing the answer."
        elif summary.last_node:
            now = f"{NODE_ICONS.get(summary.last_node, '•')} {NODE_LABELS.get(summary.last_node, summary.last_node)} is working."
        else:
            now = "Starting."
    elif summary.hitl_waiting:
        now = "Paused. Waiting for you to approve or send it back."
    else:
        now = "Finished."

    facts = []
    if summary.route:
        facts.append(f"<b>Route:</b> {summary.route} ({INTENT_LABELS.get(summary.intent or '', summary.intent or '')})")
    if summary.docs:
        facts.append(f"<b>Evidence:</b> {len(summary.docs)} document(s)")
    if summary.tool_calls:
        denied = f", {summary.denied_tools} not allowed" if summary.denied_tools else ""
        facts.append(f"<b>Tool calls:</b> {summary.tool_calls}{denied}")
    if summary.validation:
        facts.append(f"<b>Validation:</b> {summary.validation}")
    if summary.degraded_reason:
        facts.append(f"<b>Partial:</b> {summary.degraded_reason}")

    # Colour only when the user needs to act or should know something is missing.
    tone = "warn" if (summary.hitl_waiting or summary.degraded_reason) else "ok"
    ui.status_card(f"<b>Right now:</b> {now}", facts, tone=tone)


def _legend() -> None:
    with st.expander("How to read this panel", expanded=False):
        st.markdown(
            """
Each step below is one LangGraph node the assistant ran, in order. Open a step to see what it did.

| Icon | What it means |
| --- | --- |
| 🧠 | Memory: recalling the earlier conversation |
| 🧭 | Supervisor: understanding the question and choosing a route |
| 🔍 | Retrieval: searching the documents by meaning and by keyword |
| 🔬 🔁 📄 📊 | Research (RLM): the plan, one batch per document, a deeper pass, the themes pulled together |
| 🛠️ | Tool call: knowledge search, MCP enterprise data, Python analysis |
| 🛡️ | Security: a guardrail block or a tool your role cannot use |
| ✍️ | Response agent writing the answer |
| ✅ ⚠️ | Validation: citation check, partial answer notice |
| 🙋 | Human approval: waiting for you to approve or send back |

Inside every step, the **Raw events** tab holds the exact JSON the backend sent. Nothing is hidden.
**Developer view** and **Open every step** are in the sidebar under *Advanced*. The same run is traced in LangSmith.
"""
        )


# --------------------------------------------------------------------------------------
# Step (one LangGraph node) rendering
# --------------------------------------------------------------------------------------


def _headline_md(desc: Described) -> str:
    color = {"good": None, "warn": "orange", "bad": "red", "normal": None}[desc.tone]
    text = f":{color}[**{desc.headline}**]" if color else f"**{desc.headline}**"
    return f"{desc.icon} {text}"


def _is_header_noise(ev: dict[str, Any]) -> bool:
    """Events already covered by the step title in the readable view."""
    return ev.get("event_type") == "node" or (
        ev.get("event_type") == "state" and str(ev.get("message", "")).lower().startswith("active node")
    )


def _is_batch(ev: dict[str, Any]) -> bool:
    return ev.get("event_type") == "rlm" and str(ev.get("message", "")).startswith("Analyzed batch")


def _render_batch_block(batch_events: list[dict[str, Any]]) -> None:
    """Fold consecutive 'Analyzed batch <doc>' events into one calm block."""
    st.markdown(f"📄 **Read {len(batch_events)} document(s), one batch each**")
    for ev in batch_events:
        d = ev.get("detail") or {}
        titles = d.get("titles") or []
        title = f" ({titles[0]})" if titles else ""
        st.markdown(
            f"- `{d.get('doc_id', '?')}`{title}: {d.get('chunks', 0)} passage(s), {_themes_sentence(d.get('analysis'))}"
        )


def _render_steps_plain(group: list[dict[str, Any]]) -> None:
    rows = [ev for ev in group if not _is_header_noise(ev)]
    i = 0
    while i < len(rows):
        if _is_batch(rows[i]):
            j = i
            while j < len(rows) and _is_batch(rows[j]):
                j += 1
            _render_batch_block(rows[i:j])
            i = j
            continue
        desc = describe_event(rows[i])
        st.markdown(_headline_md(desc))
        for line in desc.lines:
            st.markdown(line)
        i += 1


def _render_raw(group: list[dict[str, Any]]) -> None:
    for ev in group:
        st.caption(f"`{ev.get('event_type', '?')}` · node `{ev.get('node') or '-'}` · {_time(ev)} · {ev.get('message', '')}")
        detail = ev.get("detail") or {}
        st.code(json.dumps(detail, indent=2, default=str)[:4000] if detail else "{}", language="json")


def _step_summary(node: str, group: list[dict[str, Any]]) -> str:
    """Short phrase for the step title so a collapsed step still tells the story."""
    msgs = [str(ev.get("message", "")) for ev in group]
    details = [ev.get("detail") or {} for ev in group]

    if node == "memory":
        d = next((d for d in details if "chars" in d), {})
        chars = d.get("chars", 0)
        if not chars:
            return "fresh chat"
        bits = []
        prior = int(d.get("prior_turns") or 0) - 1
        if prior > 0:
            bits.append(f"{prior} earlier turn{'s' if prior != 1 else ''}")
        if d.get("preferences"):
            bits.append("your saved preferences")
        if d.get("recent_thumbs_down"):
            bits.append("recent feedback")
        return "recalled " + (", ".join(bits) if bits else f"{chars} characters of context")
    if node == "input_validation":
        return "request blocked"
    if node == "supervisor":
        d = next((d for d in details if "route" in d), {})
        return f"sent to {d.get('route', '?')} ({INTENT_LABELS.get(d.get('intent', ''), d.get('intent', '?'))})" if d else "choosing a route"
    if node == "retrieval":
        ids = next((d.get("doc_ids") for d in details if "doc_ids" in d), None)
        if ids is None:
            return "searching"
        return f"{len(ids)} passages from {len(set(ids))} documents" if ids else "no documents found"
    if node == "research":
        subs = sum(1 for m in msgs if m.startswith("RLM sub-search"))
        batches = sum(1 for m in msgs if m.startswith("Analyzed batch"))
        agg = next((d for m, d in zip(msgs, details) if m.startswith("Aggregated")), {})
        themes = agg.get("top_themes") or []
        parts = [f"{subs} searches", f"{batches} batches"]
        if themes:
            extra = f" and {len(themes) - 2} more" if len(themes) > 2 else ""
            parts.append("top themes: " + ", ".join(_theme(t) for t in themes[:2]) + extra)
        if any(m.startswith("Recursive") for m in msgs):
            parts.append("went one level deeper")
        return ", ".join(parts)
    if node == "tools":
        called = [m.replace("Invoking", "").strip() for m in msgs if m.startswith("Invoking")]
        denied = sum(1 for ev in group if ev.get("event_type") == "guardrail")
        s = ", ".join(f"`{c}`" for c in called) or "choosing a tool"
        return s + (f", {denied} not allowed" if denied else "")
    if node == "response":
        if any(ev.get("event_type") == "error" for ev in group):
            return "the model failed, so this is an evidence only summary"
        return "draft written" if any(m.startswith("Draft") for m in msgs) else "writing"
    if node == "validate":
        d = next((d for m, d in zip(msgs, details) if m.startswith("Validation complete")), None)
        if d is None:
            return "checking"
        bits = ["passed" if d.get("ok", True) else "failed"]
        if any(m.startswith("Removed") for m in msgs):
            bits.append("citations repaired")
        if any(m.startswith("Serving degraded") for m in msgs):
            bits.append("partial answer")
        return ", ".join(bits)
    if node == "hitl":
        if any(m.startswith("Human reviewer approved") for m in msgs):
            return "approved, answer released"
        if any(m.startswith("Human reviewer rejected") for m in msgs):
            return "sent back, answer withheld"
        return "waiting for your decision"
    return f"{len(group)} events"


def _groups(events: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    """Split the event stream into consecutive runs per LangGraph node."""
    out: list[tuple[str, list[dict[str, Any]]]] = []
    for ev in events:
        node = ev.get("node") or ""
        if out and out[-1][0] == node:
            out[-1][1].append(ev)
        else:
            out.append((node, [ev]))
    return out


def _render_step(node: str, group: list[dict[str, Any]], *, expanded: bool, developer_view: bool) -> None:
    icon = NODE_ICONS.get(node, "•")
    label = NODE_LABELS.get(node, node or "Graph")
    title = f"{icon} **{label}**: {_step_summary(node, group)} · `{node or 'graph'}` · {_time(group[0])}"
    with st.expander(title, expanded=expanded):
        purpose = NODE_PURPOSE.get(node)
        if purpose:
            st.caption(purpose[0].upper() + purpose[1:] + ".")
        if developer_view:
            _render_raw(group)
            return
        tab_plain, tab_raw = st.tabs(["Steps", f"Raw events ({len(group)})"])
        with tab_plain:
            _render_steps_plain(group)
        with tab_raw:
            _render_raw(group)


def render_activity_panel(
    container,
    events: list[dict[str, Any]],
    *,
    live: bool,
    developer_view: bool,
    expand_all: bool = False,
    answer_streaming: bool = False,
    approval: str | None = None,
) -> None:
    """Draw the whole panel into `container` (an `st.empty()` placeholder so it can be redrawn live).

    Collapsed steps keep the panel calm; the step running right now (or the last one, when the
    turn is over) is open so the evaluator always sees the current activity without clicking.
    `approval` ("approved" / "sent back") marks the Approval chip once a human has decided.
    """
    with container.container():
        ui.panel_header("How the answer is produced", "Agent Activity Panel", live=live)

        if not events and not answer_streaming:
            # Empty state previews the pipeline so people know what they are about to watch.
            ui.stage_strip([ui.chip(label, "pending") for _, label, _ in STAGES])
            st.caption("Ask something and each step the assistant takes will show up here as it happens.")
            return

        summary = summarize(events)
        _stage_strip(summary, live, answer_streaming, approval)
        _status_box(summary, live, answer_streaming)

        groups = _groups(events)
        for idx, (node, group) in enumerate(groups):
            is_last = idx == len(groups) - 1
            open_it = expand_all or developer_view or (is_last and not (answer_streaming and live))
            _render_step(node, group, expanded=open_it, developer_view=developer_view)

        if answer_streaming and "response" not in summary.touched:
            with st.expander("✍️ **Response agent**: writing the answer, you can see it appear in the chat · `response`", expanded=True):
                st.caption(NODE_PURPOSE["response"][0].upper() + NODE_PURPOSE["response"][1:] + ".")

        st.caption(f"{len(events)} events this turn. The full trace is in LangSmith (same run when LANGCHAIN_* is set).")
        _legend()


