"""Option A: hardcoded users + JWT. RBAC is enforced here AND at tool gate."""

from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend.app.config import get_settings
from backend.app.models.schemas import Role

# Demo users for Apex Commercial Bank (password is plaintext for POC clarity).
USERS: dict[str, dict] = {
    "viewer1": {
        "password": "viewer123",
        "role": Role.VIEWER,
        "display_name": "Nimali Perera (Viewer)",
        "department": "retail",
    },
    "analyst1": {
        "password": "analyst123",
        "role": Role.ANALYST,
        "display_name": "Kasun Fernando (Analyst)",
        "department": "payments",
    },
    "admin1": {
        "password": "admin123",
        "role": Role.ADMINISTRATOR,
        "display_name": "Dilani Jayasinghe (Administrator)",
        "department": "platform",
    },
}

_bearer = HTTPBearer(auto_error=False)


def authenticate(username: str, password: str) -> Optional[dict]:
    user = USERS.get(username)
    if not user or user["password"] != password:
        return None
    return {"username": username, **{k: v for k, v in user.items() if k != "password"}}


def create_access_token(username: str, role: Role, display_name: str) -> str:
    settings = get_settings()
    payload = {
        "sub": username,
        "role": role.value,
        "name": display_name,
        "exp": datetime.now(timezone.utc) + timedelta(hours=8),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, settings.app_secret_key, algorithm="HS256")


def decode_token(token: str) -> dict:
    settings = get_settings()
    try:
        return jwt.decode(token, settings.app_secret_key, algorithms=["HS256"])
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from exc


HITL_TOKEN_TTL_MIN = 30


def create_hitl_token(*, username: str, session_id: str, message: str) -> str:
    """Signed, self-contained approval token for a paused (HITL) research answer.

    Stateless on purpose: the graph re-runs from the checkpointed thread on approval, so the
    server only needs *who* asked *what* in *which* session. Surviving API restarts and
    multiple replicas comes for free, unlike an in-memory pending-drafts dict.
    """
    settings = get_settings()
    now = datetime.now(timezone.utc)
    payload = {
        "purpose": "hitl",
        "sub": username,
        "sid": session_id,
        "msg": message,
        "iat": now,
        "exp": now + timedelta(minutes=HITL_TOKEN_TTL_MIN),
    }
    return jwt.encode(payload, settings.app_secret_key, algorithm="HS256")


def decode_hitl_token(token: str) -> dict:
    settings = get_settings()
    try:
        payload = jwt.decode(token, settings.app_secret_key, algorithms=["HS256"])
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail=f"This approval request expired after {HITL_TOKEN_TTL_MIN} minutes — ask the question again.",
        ) from exc
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid approval token") from exc
    if payload.get("purpose") != "hitl":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Not an approval token")
    return payload


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> dict:
    if not credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing credentials")
    payload = decode_token(credentials.credentials)
    return {
        "username": payload["sub"],
        "role": Role(payload["role"]),
        "display_name": payload.get("name", payload["sub"]),
    }


def role_allows_tool(role: Role, tool_name: str) -> bool:
    settings = get_settings()
    allowed = settings.role_tool_map.get(role.value, [])
    return tool_name in allowed
