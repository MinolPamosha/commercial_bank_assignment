"""The enterprise-data server is a real MCP server, and the agent's client speaks the protocol.

Runs offline: a fresh server app is driven through `httpx.ASGITransport`, so the official MCP
client performs `initialize` → `tools/list` → `tools/call` over Streamable HTTP without a socket.
Everything is opened inside the test body (not in fixtures) because anyio task groups must be
entered and exited by the same task.
"""

import json
from contextlib import asynccontextmanager

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from backend.app.tools.mcp_client import MCP_TOOL_NAMES, _parse_tool_result
from mcp_server.main import create_app


@asynccontextmanager
async def running_server():
    app = create_app()

    def asgi_client(headers=None, timeout=None, auth=None) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://mcp", headers=headers, timeout=timeout, auth=auth)

    async with app.router.lifespan_context(app):  # starts the Streamable-HTTP session manager
        yield asgi_client


@asynccontextmanager
async def mcp_session():
    async with running_server() as asgi_client:
        async with streamablehttp_client("http://mcp/mcp/", httpx_client_factory=asgi_client) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session


async def test_server_lists_the_three_enterprise_tools():
    async with mcp_session() as session:
        tools = {t.name: t for t in (await session.list_tools()).tools}
    assert set(tools) == set(MCP_TOOL_NAMES.values())
    assert "tier" in tools["service_catalog"].inputSchema["properties"]  # schema derived from the Python signature


async def test_tool_call_returns_structured_data_and_coerces_types():
    async with mcp_session() as session:
        people = await session.call_tool("employee_directory", {"q": "payments on-call"})
        services = await session.call_tool("service_catalog", {"tier": "1"})  # the supervisor LLM may send strings
    parsed = _parse_tool_result(people)
    assert parsed["ok"] and {e["team"] for e in parsed["data"]["employees"]} >= {"Payments SRE"}
    assert _parse_tool_result(services)["ok"] and len(json.loads(services.content[0].text)["services"]) == 4


async def test_invalid_tool_arguments_surface_as_tool_error_not_crash():
    async with mcp_session() as session:
        result = await session.call_tool("service_catalog", {"tier": "not-a-number"})
    parsed = _parse_tool_result(result)
    assert result.isError and not parsed["ok"] and "validation error" in parsed["error"]


async def test_health_and_rest_mirror_for_humans():
    async with running_server() as asgi_client:
        async with asgi_client() as client:
            health = (await client.get("/health")).json()
            open_incidents = (await client.get("/rest/incident_records?status=open")).json()["incidents"]
    assert health["protocol"] == "mcp/streamable-http" and set(health["tools"]) == set(MCP_TOOL_NAMES.values())
    assert open_incidents and all(i["status"] == "open" for i in open_incidents)
