"""MCP server exposing dummy Apex Commercial Bank enterprise data.

A real Model Context Protocol server built with the official `mcp` SDK (FastMCP):
- transport: Streamable HTTP at `POST /mcp` (stateless, JSON responses — no sticky sessions,
  so it scales horizontally and is trivial to stop/start for the outage demo)
- tools:     employee_directory(q), service_catalog(tier), incident_records(status)

For humans and curl there is also `GET /health` and a read-only REST mirror of each tool
under `/rest/<tool>` — the agent never uses those; it goes through the MCP protocol
(`backend/app/tools/mcp_client.py`).

Run: uvicorn mcp_server.main:app --port 8100
"""

# NOTE: no `from __future__ import annotations` here — FastMCP 1.9 inspects the *runtime*
# parameter annotations to build each tool's JSON schema; stringified annotations break it.

from datetime import date
from typing import Any

from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

# ---- Dummy enterprise data ------------------------------------------------------------

EMPLOYEES = [
    {"id": "E1001", "name": "Nimali Perera", "role": "Viewer", "team": "Retail Ops", "email": "nimali.perera@apexbank.example"},
    {"id": "E2044", "name": "Kasun Fernando", "role": "Analyst", "team": "Payments Risk", "email": "kasun.fernando@apexbank.example"},
    {"id": "E3002", "name": "Dilani Jayasinghe", "role": "Administrator", "team": "Platform", "email": "dilani.jayasinghe@apexbank.example"},
    {"id": "E1888", "name": "Tharindu Silva", "role": "On-call", "team": "Payments SRE", "email": "tharindu.silva@apexbank.example"},
]

SERVICES = [
    {"id": "svc-upi", "name": "ApexPay UPI Gateway", "tier": 1, "owner": "Payments", "slo": "99.5% success"},
    {"id": "svc-neft", "name": "NEFT Clearing Adapter", "tier": 1, "owner": "Payments", "slo": "99.9% availability"},
    {"id": "svc-imps", "name": "IMPS Switch Connector", "tier": 1, "owner": "Payments", "slo": "99.5% success"},
    {"id": "svc-ledger-cache", "name": "Ledger Redis Cache", "tier": 1, "owner": "Platform", "slo": "p99 < 5ms"},
]

INCIDENTS = [
    {"id": "INC-2025-0142", "status": "closed", "severity": "SEV-2", "service": "svc-upi", "opened": "2025-03-12"},
    {"id": "INC-2025-0033", "status": "closed", "severity": "SEV-1", "service": "svc-neft", "opened": "2025-01-18"},
    {"id": "INC-2025-0201", "status": "closed", "severity": "SEV-2", "service": "svc-cards", "opened": "2025-06-04"},
    {"id": "INC-2025-0999", "status": "open", "severity": "SEV-3", "service": "svc-upi", "opened": str(date.today())},
]

# ---- MCP tools (plain functions; FastMCP derives each tool's JSON schema from the signature) ----


def employee_directory(q: str | None = None) -> dict[str, Any]:
    """Look up employees by name, team or role (e.g. 'payments on-call'). No filter returns everyone."""
    if not q:
        return {"employees": EMPLOYEES}
    # Token match so "payments team" / "payments on-call" still hit "Payments SRE" + role "On-call".
    tokens = [t for t in q.lower().replace("-", " ").split() if t not in {"team", "the", "for", "who", "is"}]

    def hit(e: dict) -> bool:
        hay = f"{e['name']} {e['team']} {e['role']}".lower().replace("-", " ")
        return any(t in hay for t in tokens)

    return {"employees": [e for e in EMPLOYEES if hit(e)] if tokens else EMPLOYEES}


def service_catalog(tier: int | None = None) -> dict[str, Any]:
    """List services with tier, owner and SLO; optionally filter by tier number."""
    if tier is None:
        return {"services": SERVICES}
    return {"services": [s for s in SERVICES if s["tier"] == tier]}


def incident_records(status: str | None = None) -> dict[str, Any]:
    """List incident tickets (id, status, severity, service); optionally filter by status 'open' or 'closed'."""
    if not status:
        return {"incidents": INCIDENTS}
    return {"incidents": [i for i in INCIDENTS if i["status"] == status.lower()]}


MCP_TOOLS = (employee_directory, service_catalog, incident_records)


# ---- Plain HTTP helpers (curl / health checks; not used by the agent) ----------------------


async def health(_: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "service": "apex-mcp", "protocol": "mcp/streamable-http", "tools": [t.__name__ for t in MCP_TOOLS]})


async def rest_employee_directory(request: Request) -> JSONResponse:
    return JSONResponse(employee_directory(request.query_params.get("q")))


async def rest_service_catalog(request: Request) -> JSONResponse:
    tier = request.query_params.get("tier")
    return JSONResponse(service_catalog(int(tier) if tier else None))


async def rest_incident_records(request: Request) -> JSONResponse:
    return JSONResponse(incident_records(request.query_params.get("status")))


# ---- Server factory -------------------------------------------------------------------


def create_server() -> FastMCP:
    """Build a fresh MCP server. A factory (not a module singleton) because each FastMCP instance
    owns one Streamable-HTTP session manager that can be started only once per process."""
    server = FastMCP(
        "apex-enterprise-data",
        instructions="Dummy Apex Commercial Bank enterprise data: employee directory, service catalog, incident records.",
        streamable_http_path="/mcp",
        stateless_http=True,
        json_response=True,
    )
    for tool in MCP_TOOLS:
        server.add_tool(tool)
    server.custom_route("/health", methods=["GET"])(health)
    server.custom_route("/rest/employee_directory", methods=["GET"])(rest_employee_directory)
    server.custom_route("/rest/service_catalog", methods=["GET"])(rest_service_catalog)
    server.custom_route("/rest/incident_records", methods=["GET"])(rest_incident_records)
    return server


def create_app():
    """ASGI app: MCP endpoint at /mcp plus the human-facing routes above."""
    return create_server().streamable_http_app()


app = create_app()
