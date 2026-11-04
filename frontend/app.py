"""Streamlit chat UI. Function and transparency first, with the interaction patterns people
already know from ChatGPT so nobody needs instructions.

Three screens:
  1. Signed out: one welcome card. Pick a demo identity and sign in (a real JWT login).
  2. Signed in, empty chat: greeting plus example questions you can click.
  3. Conversation: the chat (inline "thinking" status while the agent works, streaming answer,
     source chips, approval card, thumbs feedback, ask again) and, beside it, the Agent Activity
     Panel fed live by `/api/chat/stream` (NDJSON). The panel can be hidden for a focused view;
     it is on by default because watching the agent work is the point of this assistant.

Sidebar: new chat, recent chats (switch back and the backend memory thread continues), panel
toggle, connection status, identity. Power features (Developer view, open all steps, custom
credentials, API URL) sit under "Advanced" so the first impression stays simple.
"""

from __future__ import annotations

import base64
import json
import os
import time
import uuid
from datetime import datetime, timezone

import httpx
import streamlit as st

import ui
from activity_panel import render_activity_panel

st.set_page_config(page_title="Apex Knowledge Assistant", page_icon="🏦", layout="wide")

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

# What the inline status says while each LangGraph node runs (the panel has the detail).
THINKING_LABELS: dict[str, str] = {
    "memory": "Looking back at this chat…",
    "input_validation": "Checking the request…",
    "supervisor": "Figuring out what you need…",
    "retrieval": "Searching the documents…",
    "research": "Going through the documents…",
    "tools": "Checking live bank data…",
    "response": "Putting the answer together…",
    "validate": "Double checking the sources…",
    "hitl": "Waiting for you",
}
ROUTE_WORDS: dict[str, str] = {"retrieve": "document search", "research": "deep research", "tools": "live bank data", "respond": "from earlier in this chat"}

# --- Session state -----------------------------------------------------------------
_DEFAULTS = {
    "token": None,
    "session_id": str(uuid.uuid4()),
    "messages": [],
    "events": [],
    "hitl_token": None,
    "hitl_decision": None,  # "approved" / "sent back" / "set aside" for the current turn (Approval chip)
    "feedback_given": {},
    "feedback_pending": None,  # message index whose 👎 is waiting for an optional comment
    "pending_prompt": None,
    "chats": {},  # session_id → saved conversation (recent chats in the sidebar)
}
for key, value in _DEFAULTS.items():
    st.session_state.setdefault(key, value)

API_BASE = os.environ.get("API_BASE", "http://localhost:8000/api")


def auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {st.session_state.token}"}


# --- Conversations (recent chats) ------------------------------------------------------
# A user's chats belong to the user, not to a browser tab: each transcript is saved to the API
# (`PUT /conversations/{session_id}`, per user, newest 20 kept) after every turn, and the list is
# loaded again on sign-in, so "Recent" survives sign-out, another browser or a UI restart. Switching
# back to a chat and asking a follow-up continues that memory thread (the backend re-hydrates
# session memory from the saved transcript if it has to). The UI keeps the transcript per chat in
# `chats`: session_id → {"title", "updated_at", "state" (the fields below, fetched lazily)}.
_CHAT_FIELDS = ("messages", "events", "hitl_token", "hitl_decision", "feedback_given")


# --- Surviving a browser refresh -----------------------------------------------------------
# Streamlit's session_state belongs to one browser tab's websocket session and is wiped on reload,
# which would sign the user out and blank the chat. So signed-in UI sessions are also kept in a
# process-wide store (cache_resource = one dict for the Streamlit process) under an opaque id that
# travels in the URL as ?s=<id>. On load, that id restores identity, open chat and recent chats; the
# backend memory thread continues because the same session_id is reused. The JWT itself never goes
# in the URL. Entries die with the token (8 h), on sign out, or when the Streamlit process restarts.
_RESUME_FIELDS = ("token", "role", "display_name", "session_id", "chats", *_CHAT_FIELDS)
RESUME_TTL_S = 8 * 3600


@st.cache_resource(show_spinner=False)
def _resume_store() -> dict[str, dict]:
    return {}


def _token_expired(token: str | None) -> bool:
    """Read the JWT's exp claim without verifying it (the server verifies on every request)."""
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        return float(json.loads(base64.urlsafe_b64decode(payload))["exp"]) <= time.time()
    except Exception:  # noqa: BLE001
        return True


def persist_session() -> None:
    rid = st.session_state.get("resume_id")
    if not rid or not st.session_state.token:
        return
    store = _resume_store()
    now = time.time()
    for stale in [k for k, v in store.items() if now - v.get("saved_at", 0) > RESUME_TTL_S]:
        store.pop(stale, None)
    store[rid] = {f: st.session_state.get(f) for f in _RESUME_FIELDS} | {"saved_at": now}


def restore_session() -> None:
    rid = st.query_params.get("s")
    if not rid:
        return
    saved = _resume_store().get(rid)
    if not saved or _token_expired(saved.get("token")):
        _resume_store().pop(rid, None)
        st.query_params.pop("s", None)
        if saved:
            st.session_state.resume_notice = True
        return
    for f in _RESUME_FIELDS:
        st.session_state[f] = saved[f]
    st.session_state.resume_id = rid


if not st.session_state.token:
    restore_session()


def chat_title(messages: list[dict]) -> str:
    first = next((m["content"] for m in messages if m["role"] == "user"), "New chat")
    return first if len(first) <= 44 else first[:43].rstrip() + "…"


def _chat_signature() -> tuple:
    """Cheap change detector so the transcript is only uploaded when something happened."""
    return (
        len(st.session_state.messages),
        st.session_state.hitl_token,
        st.session_state.hitl_decision,
        tuple(sorted(st.session_state.feedback_given.items())),
    )


def save_current_chat() -> None:
    if not st.session_state.messages:
        return
    sid = st.session_state.session_id
    state = {f: st.session_state[f] for f in _CHAT_FIELDS}
    title = chat_title(state["messages"])
    entry = st.session_state.chats.get(sid) or {}
    sig = _chat_signature()
    if entry.get("sig") == sig:
        return
    # Newest first, matching the order the API returns the list in.
    st.session_state.chats = {
        sid: {"title": title, "updated_at": datetime.now(timezone.utc).isoformat(), "state": state, "sig": sig},
        **{k: v for k, v in st.session_state.chats.items() if k != sid},
    }
    try:
        httpx.put(f"{API_BASE}/conversations/{sid}", headers=auth_headers(), json={"title": title, "state": state}, timeout=15)
    except httpx.HTTPError as exc:  # the chat still works from this tab; it just will not be in Recent next time
        st.toast(f"Could not save this chat to your history: {exc}")


def load_chats() -> None:
    """Recent list for the signed-in user. Full transcripts are fetched when a chat is opened."""
    try:
        r = httpx.get(f"{API_BASE}/conversations", headers=auth_headers(), timeout=15)
        r.raise_for_status()
        st.session_state.chats = {
            c["session_id"]: {"title": c["title"], "updated_at": c["updated_at"], "state": None, "sig": None}
            for c in r.json()
        }
    except httpx.HTTPError as exc:
        st.session_state.chats = {}
        st.toast(f"Could not load your recent chats: {exc}")


def _clear_current() -> None:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.events = []
    st.session_state.hitl_token = None
    st.session_state.hitl_decision = None
    st.session_state.feedback_given = {}
    st.session_state.feedback_pending = None
    st.session_state.pending_prompt = None


def new_chat() -> None:
    save_current_chat()
    _clear_current()


def open_chat(session_id: str) -> bool:
    save_current_chat()
    entry = st.session_state.chats[session_id]
    if entry.get("state") is None:  # first time this tab opens it: fetch the transcript
        try:
            r = httpx.get(f"{API_BASE}/conversations/{session_id}", headers=auth_headers(), timeout=15)
            r.raise_for_status()
            entry["state"] = r.json()["state"]
        except httpx.HTTPError as exc:
            st.error(f"Could not open that chat: {exc}")
            return False
    state = entry["state"]
    st.session_state.session_id = session_id
    for f in _CHAT_FIELDS:
        st.session_state[f] = state.get(f, _DEFAULTS[f])
    # JSON turned the message-index keys into strings on the way through the API.
    st.session_state.feedback_given = {int(k): v for k, v in (st.session_state.feedback_given or {}).items()}
    st.session_state.feedback_pending = None
    st.session_state.pending_prompt = None
    entry["sig"] = _chat_signature()
    return True


def delete_chat(session_id: str) -> bool:
    """Remove one chat on the server (transcript + session memory) and locally."""
    try:
        r = httpx.delete(f"{API_BASE}/conversations/{session_id}", headers=auth_headers(), timeout=15)
        if r.status_code not in (200, 404):
            r.raise_for_status()
    except httpx.HTTPError as exc:
        st.toast(f"Could not delete that chat: {exc}")
        return False
    st.session_state.chats.pop(session_id, None)
    if st.session_state.session_id == session_id:
        _clear_current()  # the open chat was deleted — start a blank one
    return True


def delete_all_chats(*, clear_long_term: bool = False) -> bool:
    """Remove every chat for this user; optionally forget long-term preferences too."""
    try:
        r = httpx.delete(
            f"{API_BASE}/conversations",
            headers=auth_headers(),
            params={"clear_long_term": str(clear_long_term).lower()},
            timeout=30,
        )
        r.raise_for_status()
    except httpx.HTTPError as exc:
        st.toast(f"Could not clear your chats: {exc}")
        return False
    st.session_state.chats = {}
    _clear_current()
    st.session_state.pop("confirm_clear_all", None)
    return True


# --- Auth ----------------------------------------------------------------------------
def sign_in(username: str, password: str) -> bool:
    try:
        r = httpx.post(f"{API_BASE}/auth/login", json={"username": username, "password": password}, timeout=30)
        r.raise_for_status()
        data = r.json()
        st.session_state.token = data["access_token"]
        st.session_state.role = data["role"]
        st.session_state.display_name = data["display_name"]
        _clear_current()
        load_chats()  # this person's own history, from the API
        st.session_state.resume_id = uuid.uuid4().hex
        st.query_params["s"] = st.session_state.resume_id
        persist_session()
        return True
    except Exception as exc:  # noqa: BLE001
        st.error(f"Sign in did not work: {exc}")
        return False


def sign_out() -> None:
    _resume_store().pop(st.session_state.pop("resume_id", None), None)
    st.query_params.pop("s", None)
    st.session_state.token = None
    st.session_state.pop("role", None)
    st.session_state.pop("display_name", None)
    st.session_state.chats = {}
    _clear_current()


@st.cache_data(ttl=30, show_spinner=False)
def fetch_health(api_base: str) -> dict | None:
    try:
        return httpx.get(f"{api_base}/health", timeout=5).json()
    except Exception:  # noqa: BLE001
        return None


# --- Sidebar -------------------------------------------------------------------------
with st.sidebar:
    ui.kicker("Apex Commercial Bank")
    st.markdown("### Knowledge Assistant")

    if st.session_state.token:
        if st.button("＋  New chat", type="primary", use_container_width=True, disabled=not st.session_state.messages):
            new_chat()
            st.rerun()

        save_current_chat()
        if st.session_state.chats:
            head, clear = st.columns([3, 1])
            head.caption("Recent")
            if clear.button("Clear all", key="clear-all-chats", type="tertiary", help="Delete every chat and its memory"):
                st.session_state.confirm_clear_all = True
            if st.session_state.get("confirm_clear_all"):
                st.warning("Delete every chat and its working memory for you?")
                forget = st.checkbox("Also forget saved preferences and feedback", key="clear-lt-too")
                y, n = st.columns(2)
                if y.button("Yes, delete all", key="clear-all-yes", type="primary", use_container_width=True):
                    if delete_all_chats(clear_long_term=bool(forget)):
                        st.toast("All chats cleared")
                        st.rerun()
                if n.button("Cancel", key="clear-all-no", type="tertiary", use_container_width=True):
                    st.session_state.confirm_clear_all = False
                    st.rerun()
            for sid, entry in list(st.session_state.chats.items()):  # already newest first
                current = sid == st.session_state.session_id
                row, trash = st.columns([5, 1])
                with row:
                    if st.button(
                        entry["title"],
                        key=f"chat-{sid}",
                        type="secondary" if current else "tertiary",
                        use_container_width=True,
                        disabled=current,
                    ):
                        if open_chat(sid):
                            st.rerun()
                with trash:
                    if st.button(
                        "×",
                        key=f"del-{sid}",
                        type="tertiary",
                        help="Delete this chat and its memory",
                        use_container_width=True,
                    ):
                        if delete_chat(sid):
                            st.toast("Chat deleted")
                            st.rerun()

        st.divider()        # The widget owns its default (on). Seeding the key in session_state before the widget exists
        # made Streamlit drop the value on the next rerun, hiding the panel unasked.
        st.toggle(
            "Show how answers are produced",
            value=True,
            key="show_panel",
            help="Shows the steps beside the chat. Turn it off if you want a quieter, wider chat.",
        )

        health = fetch_health(API_BASE)
        if health:
            st.markdown(
                ui.status_dot("ok", f"Connected · {health.get('retrieval_mode', '?')} · LangSmith {'on' if health.get('langsmith_enabled') else 'off'}"),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(ui.status_dot("bad", "API unreachable, check the backend"), unsafe_allow_html=True)

        st.divider()
        st.markdown(
            f"**{st.session_state.get('display_name')}**  \n{ui.badge(str(st.session_state.get('role')))}",
            unsafe_allow_html=True,
        )
        if st.button("Sign out", type="tertiary"):
            sign_out()
            st.rerun()
    else:
        st.caption("Sign in on the right to get started.")

    with st.expander("Advanced", expanded=False):
        API_BASE = st.text_input("API base URL", value=API_BASE)
        require_hitl = st.checkbox(
            "Always ask me before releasing an answer",
            value=False,
            help="Every answer waits for your OK. Research answers already do this even when this is off.",
        )
        expand_all = st.checkbox("Open every step in the activity panel", value=False)
        developer_view = st.checkbox(
            "Developer view (raw JSON events)",
            value=False,
            help="Show the raw event data as well as the readable steps.",
        )
        st.caption(f"Session `{st.session_state.session_id[:8]}…`")
        with st.form("custom_login", clear_on_submit=False):
            st.caption("Sign in with your own username and password")
            cu = st.text_input("Username")
            cp = st.text_input("Password", type="password")
            if st.form_submit_button("Sign in"):
                if sign_in(cu, cp):
                    st.rerun()

show_panel = bool(st.session_state.get("show_panel", True))
# Split-pane scroll lock only when signed in with the activity panel on; the welcome screen
# and focus mode keep normal page scroll.
ui.inject_css(focus=not (bool(st.session_state.token) and show_panel))


# --- Screen 1: signed out → one welcome card -----------------------------------------
if not st.session_state.token:
    _, mid, _ = st.columns([1, 1.4, 1])
    with mid:
        st.write("")
        ui.kicker("Apex Commercial Bank")
        st.title("Knowledge Assistant")
        st.write("Ask about policies, incidents, runbooks and architecture. Answers come with sources, and you can watch the steps as they run.")
        st.write("")
        with st.container(border=True):
            choice = st.selectbox("Who are you today?", list(DEMO_USERS), index=0)
            st.caption(f"This person can {DEMO_USERS[choice]['can']}.")
            if st.button("Sign in", type="primary", use_container_width=True):
                if sign_in(DEMO_USERS[choice]["username"], DEMO_USERS[choice]["password"]):
                    st.rerun()
        st.caption("Demo accounts. What each person can see is enforced on the server by role.")
        if st.session_state.pop("resume_notice", None):
            st.info("Your sign in expired. Please sign in again.")
        if fetch_health(API_BASE) is None:
            st.warning(f"The API at `{API_BASE}` is not answering. Start the backend, then sign in.")
    st.stop()


# --- Helpers for the conversation screen ---------------------------------------------
def show_sources(citations: list[dict]) -> None:
    """Source chips at a glance, full attribution one click away."""
    st.markdown(ui.source_pills(citations), unsafe_allow_html=True)
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


def send_feedback(idx: int, rating: str, comment: str = "") -> None:
    try:
        httpx.post(
            f"{API_BASE}/feedback",
            headers=auth_headers(),
            json={
                "session_id": st.session_state.session_id,
                "message_id": str(idx),
                "rating": rating,
                "comment": comment or ("" if rating == "up" else "needs work"),
            },
            timeout=15,
        )
        st.session_state.feedback_given[idx] = rating
        st.session_state.feedback_pending = None
        st.toast("Thanks, noted. This feedback shapes future answers for you.")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not save your feedback: {exc}")


def answer_actions(idx: int) -> None:
    """Thumbs (👎 asks for an optional comment, as ChatGPT does) plus 'Ask again' under the latest answer."""
    given = st.session_state.feedback_given.get(idx)
    if given:
        st.caption("Feedback recorded: " + ("👍 helpful" if given == "up" else "👎 needs work"))
        return
    if st.session_state.feedback_pending == idx:
        with st.form(key=f"fbform-{idx}", border=False):
            comment = st.text_input("What should be better?", placeholder="Optional: what was wrong or missing?", label_visibility="collapsed")
            f1, f2 = st.columns(2)
            if f1.form_submit_button("Send feedback", type="primary", use_container_width=True):
                send_feedback(idx, "down", comment)
                st.rerun()
            if f2.form_submit_button("Skip", type="tertiary", use_container_width=True):
                send_feedback(idx, "down")
                st.rerun()
        return
    c1, c2 = st.columns([1, 3])
    rating = c1.feedback("thumbs", key=f"fb-{idx}")
    if rating == 1:
        send_feedback(idx, "up")
        st.rerun()
    elif rating == 0:
        st.session_state.feedback_pending = idx
        st.rerun()
    last_question = next((m["content"] for m in reversed(st.session_state.messages[:idx]) if m["role"] == "user"), None)
    if last_question and c2.button("Ask again", key=f"again-{idx}", type="tertiary", icon=":material/refresh:"):
        st.session_state.pending_prompt = last_question
        st.rerun()


def render_panel(events: list[dict], *, live: bool, answer_streaming: bool = False) -> None:
    if activity_box is None:
        return
    render_activity_panel(
        activity_box,
        events,
        live=live,
        developer_view=developer_view,
        expand_all=expand_all,
        answer_streaming=answer_streaming,
        approval=None if live else st.session_state.hitl_decision,
    )


def summarize_turn(result: dict, elapsed: float) -> str:
    """One line kept above the answer afterwards, like ChatGPT's 'Thought for 4s'."""
    if result.get("blocked"):
        return "Blocked by the security guardrails"
    route = next((e.get("detail", {}).get("route") for e in result.get("events") or [] if e.get("event_type") == "state" and "route" in (e.get("detail") or {})), None)
    parts = [f"Answered in {elapsed:.1f}s"]
    if route:
        parts.append(ROUTE_WORDS.get(route, route))
    n = len(result.get("citations") or [])
    if n:
        parts.append(f"{n} source{'s' if n != 1 else ''}")
    if result.get("hitl_required"):
        parts.append("waiting for you")
    return " · ".join(parts)


def run_turn(prompt: str) -> None:
    """Send one chat turn over the NDJSON stream, updating chat + activity panel as chunks arrive."""
    events: list[dict] = []
    st.session_state.events = events
    st.session_state.hitl_decision = None
    answer = ""
    result: dict | None = None
    answer_streaming = False
    started = time.monotonic()
    seen_nodes: list[str] = []

    with st.chat_message("assistant", avatar=AVATAR_ASSISTANT):
        thinking = st.status("Thinking…", expanded=False)
        answer_box = st.empty()
        render_panel(events, live=True)

        def note_node(node: str) -> None:
            """Move the inline status along as nodes start; the panel keeps the full detail."""
            if node in THINKING_LABELS and node not in seen_nodes:
                seen_nodes.append(node)
                thinking.update(label=THINKING_LABELS[node])
                thinking.caption(THINKING_LABELS[node].rstrip("…"))

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
                if resp.status_code == 401:  # token expired (8 h): back to sign-in with a note
                    sign_out()
                    st.session_state.resume_notice = True
                    st.rerun()
                if resp.status_code == 429:
                    resp.read()
                    thinking.update(label="Too many requests", state="error")
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
                        if chunk.get("event_type") in ("node", "memory", "guardrail", "hitl"):
                            note_node(chunk.get("node") or "")
                        render_panel(events, live=True, answer_streaming=answer_streaming)
                    elif kind == "token":
                        if not answer_streaming:
                            answer_streaming = True
                            note_node("response")
                            render_panel(events, live=True, answer_streaming=True)
                        answer += chunk.get("text", "")
                        answer_box.markdown(answer + "▌")
                    elif kind == "result":
                        result = chunk
                    elif kind == "error":
                        st.error(chunk.get("detail", "Something went wrong while answering."))
        except httpx.HTTPError as exc:
            thinking.update(label="Could not reach the assistant", state="error")
            st.error(f"Could not reach the assistant: {exc}")

        if result is None:
            thinking.update(label="No answer this time", state="error")
            answer_box.markdown("_No answer this time. Please try again._")
            return

        meta = summarize_turn(result, time.monotonic() - started)
        thinking.update(label=meta, state="error" if result.get("blocked") else "complete")
        final_answer = result.get("answer", "")
        answer_box.markdown(final_answer)

    st.session_state.events = result.get("events") or events
    st.session_state.hitl_token = result.get("hitl_token")
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": final_answer,
            "meta": meta,
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
    # Append resume events (hitl → maybe response → validate → hitl) onto this turn's panel.
    st.session_state.events = (st.session_state.events or []) + (data.get("events") or [])

    if approved:
        st.session_state.hitl_token = None
        st.session_state.hitl_decision = "approved"
        verdict = "**Approved.** This answer is released."
        if note:
            verdict += f" Note: _{note}_"
        st.session_state.messages.append({"role": "assistant", "content": verdict, "citations": None})
        return

    # Send back: the backend revises from the note and (usually) pauses again for approval.
    feedback = (note or "").strip() or "make it clearer and more useful"
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": f"Sent back. Note: _{feedback}_",
            "citations": None,
        }
    )
    if data.get("hitl_required") and data.get("hitl_token"):
        st.session_state.hitl_token = data["hitl_token"]
        st.session_state.hitl_decision = None
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": data.get("answer", ""),
                "meta": "Revised · waiting for you",
                "citations": data.get("citations"),
                "degraded": data.get("degraded"),
                "degradation_reason": data.get("degradation_reason"),
                "hitl_required": True,
            }
        )
    else:
        st.session_state.hitl_token = None
        st.session_state.hitl_decision = "sent back"
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": data.get("answer", ""),
                "citations": data.get("citations"),
                "degraded": data.get("degraded"),
                "degradation_reason": data.get("degradation_reason"),
            }
        )


# --- Screen 2/3: conversation ---------------------------------------------------------
# The composer must be called at the top level of the script: only then does Streamlit pin it to
# the bottom of the page (inside a column it renders inline, above the messages). CSS in ui.py
# aligns it with the chat column. Reading it before the columns lets the greeting step aside the
# moment a question is on its way.
prompt = st.chat_input("Ask about a policy or an incident…")
if not prompt and st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

activity_box = None
if show_panel:
    col_chat, col_activity = st.columns([1.25, 1], gap="large")
    with col_activity:
        # Keyed container so CSS can find this column and keep it in view while the chat scrolls.
        with st.container(key="activity"):
            activity_box = st.empty()
else:
    col_chat = st.container(key="chat-focus")  # centred, reading-width column (see ui.py)

with col_chat:
    first_name = str(st.session_state.get("display_name", "")).split(" ")[0] or "there"

    if not st.session_state.messages and not prompt:
        st.markdown(ui.welcome(first_name), unsafe_allow_html=True)
        cols = st.columns(2)
        for i, (label, full_prompt) in enumerate(EXAMPLES):
            if cols[i % 2].button(label, key=f"ex-{i}", use_container_width=True):
                st.session_state.pending_prompt = full_prompt
                st.rerun()

    last_assistant_idx = max((i for i, m in enumerate(st.session_state.messages) if m["role"] == "assistant"), default=None)
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"], avatar=AVATAR_ASSISTANT if msg["role"] == "assistant" else AVATAR_USER):
            if msg.get("meta"):
                st.caption(msg["meta"])
            st.markdown(msg["content"])
            if msg.get("citations"):
                show_sources(msg["citations"])
            if msg.get("degraded"):
                reason = msg.get("degradation_reason") or ""
                friendly = DEGRADED_REASONS.get(reason)
                st.warning(
                    f"Partial answer: {friendly}." if friendly else f"Partial answer ({reason})."
                )
            if msg["role"] == "assistant" and idx == last_assistant_idx and not msg.get("hitl_required"):
                answer_actions(idx)

    if st.session_state.hitl_token and not prompt:
        with st.container(border=True, key="hitl-card"):
            ui.kicker("Needs your OK")
            st.markdown(
                "**Please review this answer before it goes out.** "
                "Approve it as is, or send it back with a short note and I'll rewrite it."
            )
            st.text_input(
                "Note (optional)",
                key="hitl_note",
                label_visibility="collapsed",
                placeholder="What should change? e.g. lead with the root causes",
            )
            a1, a2 = st.columns(2)
            if a1.button("Approve", type="primary", use_container_width=True, icon=":material/check:"):
                decide_hitl(True, st.session_state.get("hitl_note", ""))
                st.rerun()
            if a2.button("Send back", use_container_width=True, icon=":material/undo:"):
                decide_hitl(False, st.session_state.get("hitl_note", ""))
                st.rerun()
        # Extra space so the card clears the pinned composer even before CSS catches up.
        st.markdown('<div style="height: 4.5rem"></div>', unsafe_allow_html=True)

    if prompt:
        if st.session_state.hitl_token:
            # Asking a new question while a draft awaits approval moves the conversation on: the
            # backend thread restarts from the new input and the old interrupt is discarded.
            st.session_state.hitl_token = None
            st.session_state.hitl_decision = "set aside"
            note = "_Left the draft above and moved on to your new question._"
            st.session_state.messages.append({"role": "assistant", "content": note, "citations": None})
            with st.chat_message("assistant", avatar=AVATAR_ASSISTANT):
                st.markdown(note)
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=AVATAR_USER):
            st.markdown(prompt)
        run_turn(prompt)
        st.rerun()

# Final (non-live) render of the panel for the current turn
render_panel(st.session_state.events, live=False)
# Snapshot this tab's state so a browser refresh lands back here (see "Surviving a browser refresh").
persist_session()
