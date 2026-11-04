"""Streamlit chat UI. Function and transparency first, kept calm.

Three screens:
  1. Signed out: one welcome card. Pick a demo identity and sign in (a real JWT login).
  2. Signed in, empty chat: greeting plus example questions you can click.
  3. Conversation: chat on the left (streaming answer, sources, approval, feedback) and the
     Agent Activity Panel on the right, fed live by `/api/chat/stream` (NDJSON).

Power features (Developer view, open all steps, custom credentials, API URL) sit under the
sidebar's "Advanced" expander so the first impression stays simple.
"""

from __future__ import annotations

import json
import os
import uuid

import httpx
import streamlit as st

import ui
from activity_panel import render_activity_panel

st.set_page_config(page_title="Apex Knowledge Assistant", page_icon="🏦", layout="wide")
ui.inject_css()

# Calm, consistent avatars (Material icons) instead of the default emoji faces.
AVATAR_USER = ":material/person:"
AVATAR_ASSISTANT = ":material/account_balance:"

# Demo identities (Option A auth: hardcoded users → JWT). Role decides which tools the agent may call.
DEMO_USERS: dict[str, dict[str, str]] = {
    "Kasun Fernando (Analyst)": {"username": "analyst1", "password": "analyst123", "can": "search the documents, run analysis and use the enterprise data tools"},
    "Nimali Perera (Viewer)": {"username": "viewer1", "password": "viewer123", "can": "chat and search the documents"},
    "Dilani Jayasinghe (Administrator)": {"username": "admin1", "password": "admin123", "can": "everything, including rebuilding the index"},
}

# Clickable starters shown on an empty conversation. (label, full prompt)
EXAMPLES: list[tuple[str, str]] = [
    ("Why do payments keep failing?", "Summarize all outage reports related to payment failures during the last year and identify recurring root causes."),
    ("What happens after a duplicate credit?", "What does the customer remediation policy require after duplicate credits are detected?"),
    ("How does UPI settlement work?", "Explain how the payment platform handles UPI settlement and failover."),
    ("Who is on call for payments?", "Who is on call for the payments team right now?"),
]

DEGRADED_REASONS: dict[str, str] = {
    "retrieval_failure": "the document index could not be reached, so some evidence may be missing",
    "mcp_failure": "the enterprise data server did not answer, so live tool data is missing",
    "llm_failure": "the language model failed, so this is a summary of the evidence only",
}

# --- Session state -----------------------------------------------------------------
_DEFAULTS = {
    "token": None,
    "session_id": str(uuid.uuid4()),
    "messages": [],
    "events": [],
    "hitl_token": None,
    "hitl_decision": None,  # "approved" / "sent back" for the current turn, shown on the Approval chip
    "feedback_given": {},
    "pending_prompt": None,
}
for key, value in _DEFAULTS.items():
    st.session_state.setdefault(key, value)

API_BASE = os.environ.get("API_BASE", "http://localhost:8000/api")


def auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {st.session_state.token}"}


def reset_session() -> None:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.events = []
    st.session_state.hitl_token = None
    st.session_state.hitl_decision = None
    st.session_state.feedback_given = {}
    st.session_state.pending_prompt = None


def sign_in(username: str, password: str) -> bool:
    try:
        r = httpx.post(f"{API_BASE}/auth/login", json={"username": username, "password": password}, timeout=30)
        r.raise_for_status()
        data = r.json()
        st.session_state.token = data["access_token"]
        st.session_state.role = data["role"]
        st.session_state.display_name = data["display_name"]
        reset_session()  # a new identity starts a fresh conversation thread
        return True
    except Exception as exc:  # noqa: BLE001
        st.error(f"Sign in did not work: {exc}")
        return False


def sign_out() -> None:
    st.session_state.token = None
    st.session_state.pop("role", None)
    st.session_state.pop("display_name", None)
    reset_session()


# --- Sidebar (kept to the essentials) ------------------------------------------------
with st.sidebar:
    ui.kicker("Apex Commercial Bank")
    st.markdown("### Knowledge Assistant")
    if st.session_state.token:
        st.markdown(
            f"**{st.session_state.get('display_name')}**  \n{ui.badge(str(st.session_state.get('role')))}",
            unsafe_allow_html=True,
        )
        st.write("")
        c1, c2 = st.columns(2)
        if c1.button("New chat", use_container_width=True):
            reset_session()
            st.rerun()
        if c2.button("Sign out", use_container_width=True):
            sign_out()
            st.rerun()
    else:
        st.caption("Sign in on the right to get started.")

    st.divider()
    with st.expander("Advanced", expanded=False):
        API_BASE = st.text_input("API base URL", value=API_BASE)
        require_hitl = st.checkbox(
            "Always ask me before releasing an answer",
            value=False,
            help="Every answer pauses as a draft until you approve it. "
            "Research questions pause anyway, even with this off.",
        )
        expand_all = st.checkbox("Open every step in the activity panel", value=False)
        developer_view = st.checkbox(
            "Developer view (raw JSON events)",
            value=False,
            help="Show every event with its raw data instead of the readable rows.",
        )
        st.caption(f"Session `{st.session_state.session_id[:8]}…`")
        with st.form("custom_login", clear_on_submit=False):
            st.caption("Sign in with your own username and password")
            cu = st.text_input("Username")
            cp = st.text_input("Password", type="password")
            if st.form_submit_button("Sign in"):
                if sign_in(cu, cp):
                    st.rerun()


# --- Screen 1: signed out → one welcome card -----------------------------------------
if not st.session_state.token:
    _, mid, _ = st.columns([1, 1.4, 1])
    with mid:
        st.write("")
        ui.kicker("Apex Commercial Bank")
        st.title("Knowledge Assistant")
        st.write("Ask about policies, incidents, runbooks and architecture. Every answer comes with its sources, and you can watch how it was put together.")
        st.write("")
        with st.container(border=True):
            choice = st.selectbox("Who are you today?", list(DEMO_USERS), index=0)
            st.caption(f"This person can {DEMO_USERS[choice]['can']}.")
            if st.button("Sign in", type="primary", use_container_width=True):
                if sign_in(DEMO_USERS[choice]["username"], DEMO_USERS[choice]["password"]):
                    st.rerun()
        st.caption("Demo identities. What each person can see and do is enforced on the server by role.")
    st.stop()


# --- Helpers for the conversation screen ---------------------------------------------
def show_citations(citations: list[dict]) -> None:
    with st.expander(f"Sources ({len(citations)})", expanded=False):
        for c in citations:
            st.markdown(
                f"**`{c.get('doc_id')}`** {c.get('title')}  \n"
                f"<small>{c.get('document_type')} · {c.get('department')} · access: {c.get('access_level')} · "
                f"relevance {float(c.get('score') or 0):.2f}</small>",
                unsafe_allow_html=True,
            )
            if c.get("snippet"):
                # Flatten the chunk to one plain line so document headings don't render as page titles.
                flat = " ".join(c["snippet"].replace("#", " ").split())
                st.caption(flat[:300] + ("…" if len(flat) > 300 else ""))


def send_feedback(idx: int, rating: str) -> None:
    try:
        httpx.post(
            f"{API_BASE}/feedback",
            headers=auth_headers(),
            json={
                "session_id": st.session_state.session_id,
                "message_id": str(idx),
                "rating": rating,
                "comment": "" if rating == "up" else "needs work",
            },
            timeout=15,
        )
        st.session_state.feedback_given[idx] = rating
        st.toast("Thanks, noted. This feedback is kept for future answers.")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not save your feedback: {exc}")


def render_panel(events: list[dict], *, live: bool, answer_streaming: bool = False) -> None:
    render_activity_panel(
        activity_box,
        events,
        live=live,
        developer_view=developer_view,
        expand_all=expand_all,
        answer_streaming=answer_streaming,
        approval=None if live else st.session_state.hitl_decision,
    )


def run_turn(prompt: str) -> None:
    """Send one chat turn over the NDJSON stream, updating chat + activity panel as chunks arrive."""
    events: list[dict] = []
    st.session_state.events = events
    st.session_state.hitl_decision = None
    answer = ""
    result: dict | None = None
    answer_streaming = False

    with st.chat_message("assistant", avatar=AVATAR_ASSISTANT):
        answer_box = st.empty()
        answer_box.markdown("_Working on it. You can follow each step on the right._")
        render_panel(events, live=True)

        try:
            with httpx.stream(
                "POST",
                f"{API_BASE}/chat/stream",
                headers=auth_headers(),
                json={
                    "message": prompt,
                    "session_id": st.session_state.session_id,
                    "require_hitl": require_hitl,
                },
                timeout=httpx.Timeout(240.0, connect=10.0),
            ) as resp:
                if resp.status_code == 429:
                    resp.read()
                    st.error("That was a bit quick. " + resp.json().get("detail", "Please wait a moment and try again."))
                    st.session_state.messages.pop()  # drop the user turn we could not process
                    return
                resp.raise_for_status()

                for line in resp.iter_lines():
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    kind = chunk.get("type")
                    if kind == "event":
                        events.append(chunk)
                        render_panel(events, live=True, answer_streaming=answer_streaming)
                    elif kind == "token":
                        if not answer_streaming:
                            answer_streaming = True
                            render_panel(events, live=True, answer_streaming=True)
                        answer += chunk.get("text", "")
                        answer_box.markdown(answer + "▌")
                    elif kind == "result":
                        result = chunk
                    elif kind == "error":
                        st.error(chunk.get("detail", "Something went wrong while answering."))
        except httpx.HTTPError as exc:
            st.error(f"Could not reach the assistant: {exc}")

        if result is None:
            answer_box.markdown("_No answer this time. Please try again._")
            return

        final_answer = result.get("answer", "")
        answer_box.markdown(final_answer)

    st.session_state.events = result.get("events") or events
    st.session_state.hitl_token = result.get("hitl_token")
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": final_answer,
            "citations": result.get("citations"),
            "degraded": result.get("degraded"),
            "degradation_reason": result.get("degradation_reason"),
            "hitl_required": result.get("hitl_required"),
        }
    )


def decide_hitl(approved: bool, note: str) -> None:
    try:
        r = httpx.post(
            f"{API_BASE}/chat/hitl",
            headers=auth_headers(),
            json={"hitl_token": st.session_state.hitl_token, "approved": approved, "note": note},
            timeout=120,
        )
    except httpx.HTTPError as exc:
        st.error(f"Could not reach the assistant: {exc}")
        return
    if r.status_code != 200:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        # Expired / invalid approval: clear the stale prompt so the user can simply ask again.
        st.session_state.hitl_token = None
        st.session_state.messages.append(
            {"role": "assistant", "content": f"⚠️ Your decision was not saved. {detail}", "citations": None}
        )
        return
    data = r.json()
    st.session_state.hitl_token = None
    st.session_state.hitl_decision = "approved" if approved else "sent back"
    # The graph resumed from its checkpoint, so only the `hitl` node ran: append its events
    # to this turn's panel rather than replacing the supervisor/research/response steps.
    st.session_state.events = (st.session_state.events or []) + (data.get("events") or [])
    verdict = "✅ **You approved this answer.**" if approved else "⛔ **You sent this draft back.**"
    if note:
        verdict += f" Note: _{note}_"
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": verdict if approved else f"{verdict}\n\n{data.get('answer', '')}",
            "citations": None,
            "degraded": data.get("degraded"),
            "degradation_reason": data.get("degradation_reason"),
        }
    )


# --- Screen 2/3: conversation ---------------------------------------------------------
col_chat, col_activity = st.columns([1.25, 1], gap="large")

with col_activity:
    activity_box = st.empty()

with col_chat:
    first_name = str(st.session_state.get("display_name", "")).split(" ")[0] or "there"

    # Read the input first (chat_input is pinned to the bottom regardless of call order) so the
    # greeting can step aside the moment a question is on its way.
    prompt = st.chat_input("Ask about a policy, an incident, a runbook…")
    if not prompt and st.session_state.pending_prompt:
        prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = None

    if not st.session_state.messages and not prompt:
        ui.kicker("Knowledge Assistant")
        st.title(f"Hi {first_name}")
        st.write("What would you like to know? Pick one of these or type your own question below.")
        cols = st.columns(2)
        for i, (label, full_prompt) in enumerate(EXAMPLES):
            if cols[i % 2].button(label, key=f"ex-{i}", use_container_width=True):
                st.session_state.pending_prompt = full_prompt
                st.rerun()

    last_assistant_idx = max((i for i, m in enumerate(st.session_state.messages) if m["role"] == "assistant"), default=None)
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"], avatar=AVATAR_ASSISTANT if msg["role"] == "assistant" else AVATAR_USER):
            st.markdown(msg["content"])
            if msg.get("citations"):
                show_citations(msg["citations"])
            if msg.get("degraded"):
                reason = msg.get("degradation_reason") or ""
                friendly = DEGRADED_REASONS.get(reason)
                st.warning(
                    f"Partial answer: {friendly} (`{reason}`)." if friendly else f"Partial answer, degraded mode: {reason}"
                )
            if msg["role"] == "assistant" and idx == last_assistant_idx and not msg.get("hitl_required"):
                given = st.session_state.feedback_given.get(idx)
                if given:
                    st.caption("Feedback recorded: " + ("👍 helpful" if given == "up" else "👎 needs work"))
                else:
                    c1, c2, _ = st.columns([1, 1, 3])
                    if c1.button("👍 Helpful", key=f"up-{idx}"):
                        send_feedback(idx, "up")
                        st.rerun()
                    if c2.button("👎 Needs work", key=f"down-{idx}"):
                        send_feedback(idx, "down")
                        st.rerun()

    if st.session_state.hitl_token:
        with st.container(border=True):
            st.markdown("🙋 **This answer is waiting for your approval.** Happy with it? Approve it. If not, send it back.")
            st.text_input("Note (optional)", key="hitl_note", label_visibility="collapsed", placeholder="Add a note if you like")
            a1, a2 = st.columns(2)
            if a1.button("Approve", type="primary", use_container_width=True):
                decide_hitl(True, st.session_state.get("hitl_note", ""))
                st.rerun()
            if a2.button("Send back", use_container_width=True):
                decide_hitl(False, st.session_state.get("hitl_note", ""))
                st.rerun()

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=AVATAR_USER):
            st.markdown(prompt)
        run_turn(prompt)
        st.rerun()

# Final (non-live) render of the panel for the current turn
render_panel(st.session_state.events, live=False)
