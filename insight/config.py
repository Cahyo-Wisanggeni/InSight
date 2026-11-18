"""Runtime configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

PrecisionMode = Literal["4bit", "8bit", "full"]
LLMBackend = Literal["auto", "local_smart", "unsloth", "hf", "ollama", "retrieval_only"]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Application settings for storage, retrieval, and model execution."""

    data_dir: Path
    sqlite_path: Path
    qdrant_path: Path
    upload_dir: Path

    llm_backend: LLMBackend
    llm_model: str
    precision_mode: PrecisionMode
    max_seq_length: int
    max_new_tokens: int
    ollama_url: str

    embed_model: str
    use_reranker: bool
    rerank_model: str
    top_k_candidates: int
    top_k_final: int

    chunk_size: int
    chunk_overlap: int
    password: str
    auto_shutdown: bool
    shutdown_grace_seconds: float
    heartbeat_timeout_seconds: float

    @classmethod
    def from_env(cls, base_dir: Path | None = None) -> "Settings":
        root = Path(os.getenv("INSIGHT_DATA_DIR", str(base_dir or Path("./data")))).resolve()
        precision = os.getenv("INSIGHT_PRECISION", "4bit").strip().lower()
        if precision not in {"4bit", "8bit", "full"}:
            precision = "4bit"

        backend = os.getenv("INSIGHT_LLM_BACKEND", "auto").strip().lower()
        if backend not in {"auto", "local_smart", "unsloth", "hf", "ollama", "retrieval_only"}:
            backend = "auto"

        return cls(
            data_dir=root,
            sqlite_path=Path(os.getenv("INSIGHT_SQLITE_PATH", str(root / "insight.sqlite"))).resolve(),
            qdrant_path=Path(os.getenv("INSIGHT_QDRANT_PATH", str(root / "qdrant_db"))).resolve(),
            upload_dir=Path(os.getenv("INSIGHT_UPLOAD_DIR", str(root / "uploads"))).resolve(),
            llm_backend=backend,  # type: ignore[arg-type]
            llm_model=os.getenv("INSIGHT_LLM_MODEL", "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"),
            precision_mode=precision,  # type: ignore[arg-type]
            max_seq_length=int(os.getenv("INSIGHT_MAX_SEQ_LENGTH", "4096")),
            max_new_tokens=int(os.getenv("INSIGHT_MAX_NEW_TOKENS", "512")),
            ollama_url=os.getenv("INSIGHT_OLLAMA_URL", "http://localhost:11434").rstrip("/"),
            embed_model=os.getenv("INSIGHT_EMBED_MODEL", "BAAI/bge-small-en-v1.5"),
            use_reranker=_env_bool("INSIGHT_USE_RERANKER", False),
            rerank_model=os.getenv("INSIGHT_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            top_k_candidates=int(os.getenv("INSIGHT_TOP_K_CANDIDATES", "12")),
            top_k_final=int(os.getenv("INSIGHT_TOP_K_FINAL", "4")),
            chunk_size=int(os.getenv("INSIGHT_CHUNK_SIZE", "900")),
            chunk_overlap=int(os.getenv("INSIGHT_CHUNK_OVERLAP", "150")),
            password=os.getenv("INSIGHT_PASSWORD", ""),
            auto_shutdown=_env_bool("INSIGHT_AUTO_SHUTDOWN", False),
            shutdown_grace_seconds=float(os.getenv("INSIGHT_SHUTDOWN_GRACE", "4.0")),
            heartbeat_timeout_seconds=float(os.getenv("INSIGHT_HEARTBEAT_TIMEOUT", "25.0")),
        )

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.qdrant_path.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
