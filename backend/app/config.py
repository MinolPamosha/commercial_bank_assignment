"""Central configuration. All tunables live here for explainability."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Brand
    bank_name: str = "Apex Commercial Bank"
    assistant_name: str = "Apex Knowledge Assistant"

    # LLM provider — "openai" (direct) or "requesty" (OpenAI-compatible gateway).
    # Both use the OpenAI SDK; only api_key + base_url + model naming differ.
    llm_provider: Literal["openai", "requesty"] = "openai"
    openai_api_key: str = ""
    requesty_api_key: str = ""
    requesty_base_url: str = "https://router.requesty.ai/v1"
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    llm_temperature: float = 0.1
    llm_timeout_sec: float = 60.0

    # ---- Derived provider helpers (single source of truth for all clients) ----
    @property
    def llm_api_key(self) -> str:
        return self.requesty_api_key if self.llm_provider == "requesty" else self.openai_api_key

    @property
    def llm_base_url(self) -> str | None:
        return self.requesty_base_url if self.llm_provider == "requesty" else None

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_api_key)

    def _qualified(self, model: str) -> str:
        """Requesty expects 'provider/model'; add 'openai/' when caller gave a bare name."""
        if self.llm_provider == "requesty" and "/" not in model:
            return f"openai/{model}"
        return model

    @property
    def resolved_llm_model(self) -> str:
        return self._qualified(self.llm_model)

    @property
    def resolved_embedding_model(self) -> str:
        return self._qualified(self.embedding_model)

    # Pinecone
    pinecone_api_key: str = ""
    pinecone_index: str = "apex-bank-knowledge"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    use_local_retrieval: bool = False
    hybrid_dense_weight: float = 0.6
    hybrid_sparse_weight: float = 0.4
    retrieval_top_k: int = 8
    rerank_top_n: int = 5

    # LangSmith
    langchain_tracing_v2: bool = True
    langchain_api_key: str = ""
    langchain_project: str = "apex-bank-assistant"
    langchain_endpoint: str = "https://api.smith.langchain.com"

    @property
    def tracing_enabled(self) -> bool:
        return bool(self.langchain_api_key and self.langchain_tracing_v2)

    # App
    app_secret_key: str = "dev-secret-change-me"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    mcp_base_url: str = "http://localhost:8100"
    documents_path: str = "data/documents"
    local_index_path: str = "data/index"

    # Rate limiting (token bucket)
    rate_limit_tokens: float = 20.0
    rate_limit_refill_per_sec: float = 0.5

    # Security / HITL
    hitl_enabled: bool = True
    max_input_chars: int = 4000
    max_tool_timeout_sec: float = 30.0

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # Role → allowed tools (authorization source of truth)
    role_tool_map: dict[str, list[str]] = {
        "viewer": ["knowledge_search"],
        "analyst": ["knowledge_search", "python_analysis", "mcp_employee_directory", "mcp_service_catalog", "mcp_incident_records"],
        # Admin HTTP endpoints (/admin/reindex, /admin/feedback) are gated by role on the route
        # itself, not via the agent tool catalog — listing phantom tool names here would lie.
        "administrator": [
            "knowledge_search",
            "python_analysis",
            "mcp_employee_directory",
            "mcp_service_catalog",
            "mcp_incident_records",
        ],
    }


@lru_cache
def get_settings() -> Settings:
    return Settings()
