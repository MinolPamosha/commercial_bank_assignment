"""Small design system for the Streamlit UI: one CSS block plus a few HTML helpers.

Principles applied (kept deliberately plain, this is a POC and "UI beauty is not important"):
  * Hierarchy: one accent colour, one typeface, a kicker + title pattern for every screen.
  * Progressive disclosure: steps collapsed, raw data one tab away, power features under Advanced.
  * Status is never colour only: every chip and status card carries its state in words too.
  * Calm feedback: a single pulsing dot says "live"; nothing else moves.
  * Contrast: the accent (#1F7A6D) meets WCAG AA on white for text and icons.

Everything here is presentation only. No event, field or node id is dropped or renamed.
"""

from __future__ import annotations

import html

import streamlit as st
import streamlit.components.v1 as components

ACCENT = "#1F7A6D"

_CSS = """
<style>
/* Streamlit chrome: drop the rainbow bar, keep the header so the sidebar toggle stays reachable. */
[data-testid="stDecoration"] { display: none; }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; }
section[data-testid="stSidebar"] { border-right: 1px solid #E6EBF0; }
section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

/* Composer: Streamlit pins a top-level chat_input to the bottom of the page across the full main
   width. The layout-specific rules below (split / focus) align it with the chat column so it reads
   as part of the conversation, like ChatGPT. */
[data-testid="stBottomBlockContainer"] { padding-top: 0.6rem; padding-bottom: 1.4rem; }
[data-testid="stChatInput"] { border-radius: 14px; box-shadow: 0 6px 24px rgba(27, 39, 51, 0.08); }

/* Typography helpers */
.apx-kicker { font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase;
              color: #6B7780; font-weight: 600; margin-bottom: 2px; }
.apx-title  { font-size: 1.15rem; font-weight: 600; color: #1B2733; margin: 0 0 10px; }
.apx-muted  { color: #6B7780; font-size: 0.85rem; }
.apx-badge  { display: inline-block; font-size: 0.72rem; font-weight: 600; padding: 2px 8px;
              border-radius: 6px; background: #E8F4F1; color: #1F5F56; text-transform: capitalize; }

/* Example question cards: left aligned, a little taller, so they read as choices not buttons. */
[class*="st-key-ex-"] button { justify-content: flex-start; text-align: left; min-height: 3.2rem;
                               border-radius: 10px; border-color: #E1E7EC; }
[class*="st-key-ex-"] button:hover { border-color: #1F7A6D; color: #1F7A6D; }

/* Step expanders: slightly smaller title text so the summary fits on one line more often. */
[data-testid="stExpander"] summary p { font-size: 0.9rem; }

/* Pipeline chips */
.apx-stages { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin: 2px 0 10px; }
.apx-chip { display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px; border-radius: 999px;
            font-size: 0.78rem; line-height: 1.3; border: 1px solid #E1E7EC; background: #F4F7F9;
            color: #4A5661; white-space: nowrap; }
.apx-chip .dot { width: 8px; height: 8px; border-radius: 50%; background: #B8C2CC; flex: none; }
.apx-chip.done    { background: #E8F4F1; border-color: #CFE7E1; color: #1F5F56; }
.apx-chip.done .dot { background: #1F7A6D; }
.apx-chip.running { background: #E9F1FB; border-color: #CBDCF2; color: #1E4F8A; font-weight: 600; }
.apx-chip.running .dot { background: #2F6FD1; animation: apx-pulse 1.2s ease-in-out infinite; }
.apx-chip.waiting { background: #FFF6E5; border-color: #F5DFB0; color: #7A5200; font-weight: 600; }
.apx-chip.waiting .dot { background: #E0A100; }
.apx-chip.blocked { background: #FCEBEB; border-color: #F3C7C7; color: #8A1F1F; font-weight: 600; }
.apx-chip.blocked .dot { background: #C62828; }
.apx-chip.skipped { text-decoration: line-through; opacity: 0.55; }
.apx-arrow { color: #B8C2CC; font-size: 0.8rem; }
@keyframes apx-pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }

/* Status card */
.apx-status { border: 1px solid #E1E7EC; border-left: 4px solid #1F7A6D; border-radius: 8px;
              padding: 10px 14px; background: #FFFFFF; font-size: 0.9rem; margin-bottom: 12px; }
.apx-status.warn { border-left-color: #E0A100; background: #FFFBF2; }
.apx-status.bad  { border-left-color: #C62828; background: #FDF4F4; }
.apx-status .facts { color: #5B6771; font-size: 0.82rem; margin-top: 5px; line-height: 1.5; }

/* Live pill next to the panel title */
.apx-live { display: inline-flex; align-items: center; gap: 6px; font-size: 0.7rem; font-weight: 600;
            color: #1E4F8A; background: #E9F1FB; border-radius: 999px; padding: 2px 8px;
            margin-left: 8px; vertical-align: middle; letter-spacing: 0.04em; }
.apx-live .dot { width: 7px; height: 7px; border-radius: 50%; background: #2F6FD1;
                 animation: apx-pulse 1.2s ease-in-out infinite; }

/* Welcome (empty conversation) */
.apx-welcome { padding: 2.5rem 0 1rem; }
.apx-welcome h1 { font-size: 2rem; font-weight: 650; letter-spacing: -0.01em; margin: 0 0 6px; color: #1B2733; }
.apx-welcome p  { color: #5B6771; font-size: 1rem; margin: 0 0 1.2rem; }

/* Chat messages: user turns as a soft bubble, assistant turns as plain reading text (ChatGPT's
   convention), plus a quieter "Answered in …" line above each answer. */
[data-testid="stChatMessage"] { padding: 0.6rem 0.9rem; }
[data-testid="stChatMessage"] [data-testid="stCaptionContainer"] p { font-size: 0.78rem; }
[data-testid="stChatMessage"] [data-testid="stStatusWidget"],
[data-testid="stChatMessage"] [data-testid="stExpander"] details { border-radius: 10px; }
[data-testid="stChatMessage"] [data-testid="stExpander"] summary { padding: 0.45rem 0.8rem; }

/* Source chips under an answer */
.apx-sources { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0 6px; }
.apx-pill { display: inline-flex; align-items: center; gap: 6px; padding: 2px 10px; border-radius: 999px;
            border: 1px solid #DCE5EA; background: #F7FAFB; color: #2B3A46; font-size: 0.76rem;
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; max-width: 100%; }
.apx-pill .id { white-space: nowrap; }
.apx-pill .t { font-family: inherit; color: #6B7780; font-weight: 400; overflow: hidden;
               text-overflow: ellipsis; white-space: nowrap; max-width: 16rem; }

/* Sidebar: recent chats read as a list, not a stack of buttons */
section[data-testid="stSidebar"] [class*="st-key-chat-"] button { justify-content: flex-start; text-align: left;
    padding: 0.25rem 0.6rem; min-height: 2.1rem; font-weight: 400; }
section[data-testid="stSidebar"] [class*="st-key-chat-"] button p { font-size: 0.88rem; overflow: hidden;
    text-overflow: ellipsis; white-space: nowrap; }
section[data-testid="stSidebar"] [class*="st-key-chat-"] button:disabled { opacity: 1; color: #1B2733;
    background: #E8F4F1; border-color: transparent; }
section[data-testid="stSidebar"] [class*="st-key-del-"] button {
    min-height: 2.1rem; padding: 0; color: #9AA5AF; justify-content: center;
}
section[data-testid="stSidebar"] [class*="st-key-del-"] button:hover { color: #C62828; }
section[data-testid="stSidebar"] [class*="st-key-clear-all"] button {
    font-size: 0.72rem; min-height: 1.6rem; padding: 0; color: #9AA5AF;
}
section[data-testid="stSidebar"] [class*="st-key-clear-all"] button:hover { color: #C62828; }
.apx-dot { display: inline-flex; align-items: center; gap: 7px; font-size: 0.8rem; color: #5B6771; }
.apx-dot i { width: 8px; height: 8px; border-radius: 50%; background: #B8C2CC; display: inline-block; }
.apx-dot.ok i { background: #1F7A6D; }
.apx-dot.bad i { background: #C62828; }

/* "Ask again" stays quiet next to the thumbs */
[class*="st-key-again-"] button { color: #6B7780; font-size: 0.82rem; padding: 0 0.4rem; min-height: 2rem; }
[class*="st-key-again-"] button:hover { color: #1F7A6D; }
</style>
"""

# Split layout: chat column + activity panel. Streamlit sizes the chat column as 1.25 of 2.25
# (≈ 55.56%) minus the "large" 2rem gap; the composer block is given the same width so it sits
# under the conversation, not under the panel. Below Streamlit's 640px breakpoint the columns
# stack and the composer goes full width.
#
# Scroll model (desktop): two equal-height panes, each with its own scroll — no page scrollbar
# fighting the panel. Soft overlay scrollbars (visible on hover). A tiny helper script keeps the
# chat pane pinned to the latest message while an answer streams (Streamlit's native scroll-to-
# bottom only moves the page, which we have locked).
_CSS_SPLIT = """
<style>
@media (min-width: 641px) {
  /* Narrow the block, not the widget: Streamlit measures the block and sets the input's inner
     width in pixels from it, so a width on stChatInput alone leaves the textarea full width. */
  [data-testid="stBottomBlockContainer"] > [data-testid="stVerticalBlockBorderWrapper"] { width: calc(55.5556% - 2rem); }

  /* Lock the page: the two panes own scrolling, the composer stays pinned underneath. */
  [data-testid="stAppScrollToBottomContainer"] { overflow: hidden !important; height: 100vh; }
  section.stMain { height: 100vh; overflow: hidden; display: flex; flex-direction: column; }
  section.stMain .block-container {
    flex: 1 1 auto; min-height: 0; overflow: hidden;
    padding-top: 0.75rem !important; padding-bottom: 0 !important;
  }

  /* One row, viewport-tall, both columns stretch and scroll on their own. */
  [data-testid="stHorizontalBlock"]:has(.st-key-activity) {
    height: calc(100vh - 7.25rem);
    max-height: calc(100vh - 7.25rem);
    align-items: stretch !important;
  }
  [data-testid="stHorizontalBlock"]:has(.st-key-activity) > [data-testid="stColumn"] {
    height: 100% !important;
    max-height: 100%;
    min-height: 0;
    overflow-x: hidden;
    overflow-y: auto;
    overscroll-behavior: contain;
    align-self: stretch !important;
    padding-bottom: 0.75rem;
    scrollbar-width: thin;
    scrollbar-color: transparent transparent;
  }
  /* Chat column only: leave room under the last message / approval card so the pinned
     composer never covers Approve / Send back. */
  [data-testid="stHorizontalBlock"]:has(.st-key-activity) > [data-testid="stColumn"]:not(:has(.st-key-activity)) {
    padding-bottom: 7.5rem;
  }
  .st-key-hitl-card { margin-bottom: 1rem; }
  [data-testid="stHorizontalBlock"]:has(.st-key-activity) > [data-testid="stColumn"]:hover,
  [data-testid="stHorizontalBlock"]:has(.st-key-activity) > [data-testid="stColumn"]:focus-within {
    scrollbar-color: #C5CED6 transparent;
  }
  [data-testid="stHorizontalBlock"]:has(.st-key-activity) > [data-testid="stColumn"]::-webkit-scrollbar { width: 6px; }
  [data-testid="stHorizontalBlock"]:has(.st-key-activity) > [data-testid="stColumn"]::-webkit-scrollbar-thumb {
    background: transparent; border-radius: 4px;
  }
  [data-testid="stHorizontalBlock"]:has(.st-key-activity) > [data-testid="stColumn"]:hover::-webkit-scrollbar-thumb {
    background: #C5CED6;
  }

  /* Panel: divider only — height/scroll come from the shared row rule above. */
  [data-testid="stColumn"]:has(.st-key-activity) {
    border-left: 1px solid #E6EBF0;
    padding-left: 1.5rem;
  }
}
</style>
"""

# Runs in a hidden iframe; reaches the app DOM via window.parent. Keeps the chat pane following
# new tokens/messages when the user is already near the bottom (ChatGPT-style).
_SCROLL_HELPER = """
<script>
(function () {
  const doc = window.parent.document;
  if (doc.documentElement.dataset.apxScrollHelper === "1") return;
  doc.documentElement.dataset.apxScrollHelper = "1";

  function panes() {
    const row = doc.querySelector('[data-testid="stHorizontalBlock"]:has(.st-key-activity)');
    if (!row) return {};
    const cols = Array.from(row.querySelectorAll(':scope > [data-testid="stColumn"]'));
    const panel = cols.find((c) => c.querySelector(".st-key-activity"));
    const chat = cols.find((c) => c !== panel);
    return { chat, panel };
  }

  function stickBottom(el, force) {
    if (!el) return;
    const gap = el.scrollHeight - el.scrollTop - el.clientHeight;
    const needHitl = !!doc.querySelector(".st-key-hitl-card");
    if (force || needHitl || gap < 160 || el.dataset.apxForce === "1") {
      el.scrollTop = el.scrollHeight;
      el.dataset.apxForce = "0";
    }
  }

  const obs = new MutationObserver(function () {
    const { chat, panel } = panes();
    stickBottom(chat, false);
    stickBottom(panel, false);
  });
  obs.observe(doc.body, { childList: true, subtree: true, characterData: true });
  function forceChatBottom() {
    const { chat } = panes();
    stickBottom(chat, true);
  }
  setTimeout(forceChatBottom, 250);
  setTimeout(forceChatBottom, 900);
})();
</script>
"""

# Focus layout (panel hidden): one reading-width column centred on the page, composer aligned to it.
_CSS_FOCUS = """
<style>
.st-key-chat-focus { max-width: 50rem; margin: 0 auto; padding-bottom: 6rem; }
[data-testid="stBottomBlockContainer"] > [data-testid="stVerticalBlockBorderWrapper"] { max-width: 50rem; margin: 0 auto; }
.st-key-hitl-card { margin-bottom: 1rem; }
</style>
"""


def inject_css(*, focus: bool = False) -> None:
    """Base styles plus the rules for the current layout (split with the panel, or focus without)."""
    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown(_CSS_FOCUS if focus else _CSS_SPLIT, unsafe_allow_html=True)
    if not focus:
        # Hidden iframe → parent DOM. Only needed when the two-pane scroll lock is active.
        components.html(_SCROLL_HELPER, height=0, width=0)


def welcome(first_name: str) -> str:
    return (
        '<div class="apx-welcome">'
        f"<h1>Hi {html.escape(first_name)}, what do you need?</h1>"
        "<p>Policies, incidents, runbooks, architecture. Pick a starter below or type your own question.</p>"
        "</div>"
    )


def _plain_title(title: str) -> str:
    """Drop typographic dashes so titles read as plain product copy, not generated prose."""
    return title.replace("—", ":").replace("–", "-").strip()


def source_pills(citations: list[dict]) -> str:
    """One chip per cited document (deduplicated, order kept): id, then title in a quieter tone."""
    seen: dict[str, str] = {}
    for c in citations:
        doc_id = str(c.get("doc_id") or "")
        if doc_id and doc_id not in seen:
            seen[doc_id] = _plain_title(str(c.get("title") or ""))
    pills = "".join(
        f'<span class="apx-pill" title="{html.escape(title)}">📄 <span class="id">{html.escape(doc_id)}</span>'
        + (f'<span class="t">{html.escape(title)}</span>' if title else "")
        + "</span>"
        for doc_id, title in seen.items()
    )
    return f'<div class="apx-sources">{pills}</div>'


def status_dot(state: str, text: str) -> str:
    """Connection line in the sidebar. `state` is ok | bad; the text always carries the meaning."""
    return f'<span class="apx-dot {state}"><i></i>{html.escape(text)}</span>'


def kicker(text: str) -> None:
    st.markdown(f'<div class="apx-kicker">{html.escape(text)}</div>', unsafe_allow_html=True)


def badge(text: str) -> str:
    return f'<span class="apx-badge">{html.escape(text)}</span>'


def chip(label: str, state: str, note: str | None = None) -> str:
    """One pipeline chip. `state` is done | running | waiting | pending | skipped | blocked.

    The state is always spelled out (✓, "running", "waiting for you", "blocked") so the strip
    still reads correctly in greyscale or for colour blind users.
    """
    text = html.escape(label)
    if state == "done":
        text = f"✓ {text}"
    if note:
        text += f" · {html.escape(note)}"
    return f'<span class="apx-chip {state}"><span class="dot"></span>{text}</span>'


def stage_strip(chips: list[str]) -> None:
    arrow = '<span class="apx-arrow">›</span>'
    st.markdown(f'<div class="apx-stages">{arrow.join(chips)}</div>', unsafe_allow_html=True)


def status_card(headline: str, facts: list[str], tone: str = "ok") -> None:
    """Compact "right now" card. `headline` and `facts` may contain simple inline HTML."""
    facts_html = f'<div class="facts">{" &nbsp;·&nbsp; ".join(facts)}</div>' if facts else ""
    st.markdown(f'<div class="apx-status {tone}">{headline}{facts_html}</div>', unsafe_allow_html=True)


def panel_header(title: str, kicker_text: str, live: bool) -> None:
    pill = '<span class="apx-live"><span class="dot"></span>LIVE</span>' if live else ""
    st.markdown(
        f'<div class="apx-kicker">{html.escape(kicker_text)}</div>'
        f'<div class="apx-title">{html.escape(title)}{pill}</div>',
        unsafe_allow_html=True,
    )
