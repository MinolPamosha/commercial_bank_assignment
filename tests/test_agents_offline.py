"""Agent logic that must hold without an LLM: supervisor decision validation, RLM plan
validation, theme analysis, and the heuristic router."""

from backend.app.agents.llm import heuristic_intent
from backend.app.agents.research_agent import deterministic_plan, validate_plan
from backend.app.agents.response_agent import _is_greeting
from backend.app.agents.supervisor import _validate_decision
from backend.app.agents.validation import validate_node
from backend.app.memory.store import MemoryStore
from backend.app.tools.catalog import TOOL_NAMES, catalog_for_prompt
from backend.app.tools.python_analysis import DEFAULT_THEME_PATTERNS, analyze_themes


def test_supervisor_decision_is_coerced_to_closed_vocabulary():
    d = _validate_decision(
        {"intent": "bogus", "route": "nowhere", "tool": "rm_rf", "tool_params": {"q": {"nested": 1}, "ok": "x"}, "plan": ["a"]},
        "qa",
        "retrieve",
    )
    assert d["intent"] == "qa" and d["route"] == "retrieve"
    assert d["tool"] is None and d["tool_params"] == {}  # unknown tool and nested params dropped


def test_supervisor_tool_choice_pins_route_and_intent():
    d = _validate_decision({"intent": "qa", "route": "retrieve", "tool": "mcp_employee_directory", "tool_params": {"q": "payments"}}, "qa", "retrieve")
    assert d["route"] == "tools" and d["intent"] == "tool" and d["tool_params"] == {"q": "payments"}


def test_catalog_lists_every_tool_for_the_prompt():
    text = catalog_for_prompt()
    assert all(name in text for name in TOOL_NAMES)
    assert {"knowledge_search", "python_analysis", "mcp_employee_directory"} <= TOOL_NAMES


def test_heuristic_router_covers_demo_questions():
    assert heuristic_intent("Summarize all outage reports and identify recurring root causes")[1] == "research"
    assert heuristic_intent("Who is on call for the payments team right now?")[1] == "tools"
    assert heuristic_intent("Show me the service catalog")[1] == "tools"
    assert heuristic_intent("What does the remediation policy require?")[1] == "retrieve"
    assert heuristic_intent("hello")[1] == "respond"


def test_rlm_plan_validation_sanitises_llm_output():
    vocab = {"document_type": {"incident", "policy", "runbook"}, "department": {"payments", "infosec"}}
    plan = validate_plan(
        {
            "topic": "Payment Outages!!",
            "steps": [
                {"label": "Bad Label", "query": "pool exhaustion", "filters": {"document_type": "incident", "department": "nope", "evil": "x"}},
                {"query": ""},  # dropped
                {"label": "s2", "query": "remediation", "filters": {"document_type": "policy"}},
            ],
            "themes": {"Pool": ["pool", "max_connections"], "bad": "not-a-list"},
            "follow_up": {"query_template": "no placeholder"},
        },
        "question",
        vocab,
    )
    assert plan["topic"] == "payment_outages"
    assert [s["label"] for s in plan["steps"]] == ["bad_label", "s2"]
    assert plan["steps"][0]["filters"] == {"document_type": "incident"}  # unknown value + key removed
    assert "pool" in plan["themes"] and "bad" not in plan["themes"]
    assert set(DEFAULT_THEME_PATTERNS) <= set(plan["themes"])  # incident research merges default facets
    assert "{themes}" in plan["follow_up"]["query_template"]  # invalid template → default


def test_deterministic_plan_targets_the_question_not_a_canned_topic():
    vocab = {"document_type": {"incident", "policy"}, "department": {"payments", "infosec"}}
    plan = deterministic_plan("Summarize all security policies and identify common themes", vocab)
    assert any("security" in s["query"] for s in plan["steps"])
    assert any(s["filters"].get("document_type") == "policy" for s in plan["steps"])


def test_theme_analysis_counts_each_snippet_once_and_escapes_keywords():
    snippets = [
        "Redis connection pool exhausted; pool raised to 200",
        "Duplicate NEFT credits — missing idempotency key",
        "Issuer network latency caused timeouts",
    ]
    res = analyze_themes(snippets, {"pool": ["pool"], "idempotency": ["duplicate", "idempotency"], "regex_injection": ["(", "pool"]})
    assert res["theme_counts"]["pool"] == 1  # two mentions in one snippet count once
    assert res["theme_counts"]["idempotency"] == 1
    assert res["documents_analyzed"] == 3
    assert "regex_injection" in res["facets"]  # "(" was escaped, not treated as regex


# --- Multi-turn conversation -------------------------------------------------------------

_POLICY = {"doc_id": "POL-OPS-04", "title": "Customer Remediation", "snippet": "Freeze ledger entries within 30 minutes."}


async def test_follow_up_from_memory_keeps_citations_seen_earlier_in_session():
    """Turn 2 ran no retrieval (route=respond) but cites the document turn 1 retrieved."""
    out = await validate_node(
        {
            "draft_answer": "The first step must happen within 30 minutes [doc:POL-OPS-04].",
            "citations": [],
            "prior_citations": [_POLICY],
            "intent": "qa",
        }
    )
    assert "[citation-removed]" not in out["final_answer"]
    assert out["validation_ok"] is True
    assert [c["doc_id"] for c in out["citations"]] == ["POL-OPS-04"]  # surfaced as this turn's source
    assert any("earlier in this conversation" in e.message for e in out["events"])


async def test_follow_up_still_strips_documents_never_seen_in_session():
    out = await validate_node(
        {
            "draft_answer": "See [doc:POL-OPS-04] and [doc:INC-2099-0001].",
            "citations": [],
            "prior_citations": [_POLICY],
            "intent": "qa",
        }
    )
    assert "[doc:POL-OPS-04]" in out["final_answer"]
    assert "[doc:INC-2099-0001]" not in out["final_answer"]


def test_session_memory_remembers_cited_documents_across_turns():
    store = MemoryStore()
    store.remember_citations("s1", "u1", [_POLICY, {"doc_id": "INC-2025-0033", "title": "NEFT dup"}])
    store.remember_citations("s1", "u1", [{**_POLICY, "score": 0.9}])  # same doc again → one entry, latest wins
    docs = store.session_citations("s1", "u1")
    assert [d["doc_id"] for d in docs] == ["POL-OPS-04", "INC-2025-0033"]
    assert docs[0]["score"] == 0.9
    assert store.session_citations("other-session", "u1") == []  # scoped per session


def _transcript(n_pairs: int, doc_id: str = "POL-OPS-04") -> dict:
    messages = []
    for i in range(n_pairs):
        messages.append({"role": "user", "content": f"question {i}"})
        messages.append({"role": "assistant", "content": f"answer {i}", "citations": [{"doc_id": doc_id, "title": "Policy"}]})
    return {"messages": messages, "events": [], "hitl_token": None, "hitl_decision": None, "feedback_given": {}}


def test_conversation_history_is_per_user_and_capped(tmp_path, monkeypatch):
    store = MemoryStore()
    store._conversations_path = tmp_path / "conversations.json"
    store._conversations = {}
    for i in range(25):
        store.save_conversation("u1", f"s{i}", f"chat {i}", _transcript(1))
    store.save_conversation("u2", "theirs", "private", _transcript(1))

    listed = store.list_conversations("u1")
    assert len(listed) == 20 and listed[0]["session_id"] == "s24"  # newest first, oldest 5 dropped
    assert store.get_conversation("u1", "theirs") is None  # another user's chat is invisible
    assert store.get_conversation("u2", "theirs")["title"] == "private"

    reloaded = MemoryStore()  # survives a restart: read back from disk
    reloaded._conversations = reloaded._load_json(store._conversations_path)
    assert [c["session_id"] for c in reloaded.list_conversations("u1")] == [c["session_id"] for c in listed]


def test_reopened_chat_rehydrates_session_memory_after_restart(tmp_path):
    store = MemoryStore()
    store._conversations_path = tmp_path / "conversations.json"
    store._conversations = {}
    store.save_conversation("u1", "old-chat", "chat", _transcript(5))

    fresh = MemoryStore()  # simulates the API restarting: no in-process sessions
    fresh._conversations = store._conversations
    mem = fresh.get_session("old-chat", "u1")
    # Only the last 6 of the 10 messages become turns, in order.
    assert [t.content for t in mem.turns] == ["question 2", "answer 2", "question 3", "answer 3", "question 4", "answer 4"]
    assert "POL-OPS-04" in mem.cited_docs  # evidence continuity carries over too
    assert "question 4" in fresh.context_block("old-chat", "u1")
    assert fresh.get_session("old-chat", "intruder").turns == []  # only the owner's transcript is used


def test_canned_greeting_only_for_a_real_opener():
    assert _is_greeting("hi", "")
    assert _is_greeting("Thanks!", "")
    assert not _is_greeting("Say that in one sentence.", "")
    assert not _is_greeting("thanks", "USER: hi\nASSISTANT: Hello!")  # mid-conversation → the model replies
