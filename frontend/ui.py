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
_CSS_SPLIT = """
<style>
@media (min-width: 641px) {
  /* Narrow the block, not the widget: Streamlit measures the block and sets the input's inner
     width in pixels from it, so a width on stChatInput alone leaves the textarea full width. */
  [data-testid="stBottomBlockContainer"] > [data-testid="stVerticalBlockBorderWrapper"] { width: calc(55.5556% - 2rem); }

  /* Activity panel column stays in view while the chat scrolls. It is found via the keyed
     container inside it, so nested columns (feedback, approval buttons) are not affected. The
     height leaves room for the pinned composer plus page padding, so the panel is never pushed
     up (and its header cut off) when the chat is scrolled to the very bottom. */
  [data-testid="stColumn"]:has(.st-key-activity) {
    position: sticky; top: 0.5rem; align-self: flex-start;
    max-height: calc(100vh - 8.5rem); overflow-y: auto; padding-bottom: 1rem;
  }
}
</style>
"""

# Focus layout (panel hidden): one reading-width column centred on the page, composer aligned to it.
_CSS_FOCUS = """
<style>
.st-key-chat-focus { max-width: 50rem; margin: 0 auto; }
[data-testid="stBottomBlockContainer"] > [data-testid="stVerticalBlockBorderWrapper"] { max-width: 50rem; margin: 0 auto; }
</style>
"""


def inject_css(*, focus: bool = False) -> None:
    """Base styles plus the rules for the current layout (split with the panel, or focus without)."""
    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown(_CSS_FOCUS if focus else _CSS_SPLIT, unsafe_allow_html=True)


def welcome(first_name: str) -> str:
    return (
        '<div class="apx-welcome">'
        f"<h1>What can I help with, {html.escape(first_name)}?</h1>"
        "<p>Ask about policies, incidents, runbooks or architecture. Pick one of these or type your own question below.</p>"
        "</div>"
    )


def source_pills(citations: list[dict]) -> str:
    """One chip per cited document (deduplicated, order kept): id, then title in a quieter tone."""
    seen: dict[str, str] = {}
    for c in citations:
        doc_id = str(c.get("doc_id") or "")
        if doc_id and doc_id not in seen:
            seen[doc_id] = str(c.get("title") or "")
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
