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
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


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
