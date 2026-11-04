"""Token-bucket rate limiter, JWT auth and signed HITL approval tokens."""

import time

import jwt
import pytest
from fastapi import HTTPException

from backend.app.config import get_settings
from backend.app.models.schemas import Role
from backend.app.security.auth import (
    authenticate,
    create_access_token,
    create_hitl_token,
    decode_hitl_token,
    decode_token,
    role_allows_tool,
)
from backend.app.security.rate_limit import TokenBucketRateLimiter


async def test_token_bucket_consumes_capacity_then_blocks_with_retry_after():
    limiter = TokenBucketRateLimiter(capacity=3, refill_per_sec=1.0)
    for _ in range(3):
        allowed, retry = await limiter.allow("u1")
        assert allowed and retry == 0.0
    allowed, retry = await limiter.allow("u1")
    assert not allowed
    assert 0 < retry <= 1.0  # one token deficit at 1 token/sec


async def test_token_bucket_is_per_user_and_refills():
    limiter = TokenBucketRateLimiter(capacity=1, refill_per_sec=50.0)
    assert (await limiter.allow("alice"))[0]
    assert not (await limiter.allow("alice"))[0]
    assert (await limiter.allow("bob"))[0]  # independent bucket
    time.sleep(0.05)  # 50 tokens/sec → refilled
    assert (await limiter.allow("alice"))[0]


def test_login_and_jwt_roundtrip():
    assert authenticate("viewer1", "wrong") is None
    user = authenticate("analyst1", "analyst123")
    assert user and user["role"] == Role.ANALYST and "password" not in user
    token = create_access_token(user["username"], user["role"], user["display_name"])
    payload = decode_token(token)
    assert payload["sub"] == "analyst1" and payload["role"] == "analyst"


def test_tampered_token_rejected():
    token = create_access_token("viewer1", Role.VIEWER, "V")
    forged = jwt.encode({"sub": "viewer1", "role": "administrator"}, "not-the-secret", algorithm="HS256")
    with pytest.raises(HTTPException) as exc:
        decode_token(forged)
    assert exc.value.status_code == 401
    assert decode_token(token)["role"] == "viewer"


def test_hitl_token_is_purpose_bound_and_expires():
    token = create_hitl_token(username="analyst1", session_id="s1", message="q")
    payload = decode_hitl_token(token)
    assert payload["purpose"] == "hitl" and payload["sid"] == "s1"

    # An ordinary access token must not be accepted as an approval token
    with pytest.raises(HTTPException) as exc:
        decode_hitl_token(create_access_token("analyst1", Role.ANALYST, "A"))
    assert exc.value.status_code == 400

    expired = jwt.encode(
        {"purpose": "hitl", "sub": "analyst1", "sid": "s1", "msg": "q", "exp": int(time.time()) - 10},
        get_settings().app_secret_key,
        algorithm="HS256",
    )
    with pytest.raises(HTTPException) as exc:
        decode_hitl_token(expired)
    assert exc.value.status_code == 410


def test_role_tool_map_matches_assignment_matrix():
    assert role_allows_tool(Role.VIEWER, "knowledge_search")
    assert not role_allows_tool(Role.VIEWER, "mcp_incident_records")
    assert role_allows_tool(Role.ANALYST, "mcp_incident_records")
    assert not role_allows_tool(Role.ANALYST, "admin_reindex")
    assert role_allows_tool(Role.ADMINISTRATOR, "admin_reindex")
