"""Runtime configuration loaded from environment variables."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

PrecisionMode = Literal["4bit", "8bit", "full"]
LLMBackend = Literal["auto", "llamacpp", "local_smart", "unsloth", "hf", "ollama", "retrieval_only"]
RAGMode = Literal["hybrid", "adaptive"]
DEFAULT_LLM_MODEL = "unsloth/Qwen3-1.7B-GGUF"
DEFAULT_GGUF_FILENAME = "Qwen3-1.7B-UD-Q4_K_XL.gguf"
DEFAULT_RESOURCES_DIR = Path(__file__).resolve().parent.parent.parent / "resources"
RESOURCE_SUBDIRS = ("huggingface", "huggingface/hub", "fastembed", "ollama", "gguf", "torch", "cache")


def normalize_rag_mode(
    raw: str | None,
    default: RAGMode = "hybrid",
    strict: bool = False,
) -> RAGMode:
    """Normalize a RAG mode string strictly to 'hybrid' or 'adaptive'."""
    if raw is None:
        return default
    cleaned = raw.strip().lower()
    if not cleaned:
        return default
    if cleaned in {"hybrid", "hybridrag", "hybrid_rag", "hybrid-rag"}:
        return "hybrid"
    if cleaned in {"adaptive", "adaptiverag", "adaptive_rag", "adaptive-rag"}:
        return "adaptive"
    if strict:
        raise ValueError(
            "Mode RAG tidak didukung. Hanya tersedia 2 opsi: 'hybrid' (HybridRAG) atau 'adaptive' (AdaptiveRAG)."
        )
    return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def configure_resource_env(resources_dir: Path | None = None) -> Path:
    """Ensure all model/cache downloads are stored under the centralized resources directory."""
    raw = str(resources_dir) if resources_dir is not None else os.getenv("INSIGHT_RESOURCES_DIR", str(DEFAULT_RESOURCES_DIR))
    target = Path(raw).resolve()
    for sub in RESOURCE_SUBDIRS:
        (target / sub).mkdir(parents=True, exist_ok=True)

    env_map = {
        "INSIGHT_RESOURCES_DIR": str(target),
        "HF_HOME": str(target / "huggingface"),
        "HUGGINGFACE_HUB_CACHE": str(target / "huggingface" / "hub"),
        "HF_HUB_CACHE": str(target / "huggingface" / "hub"),
        "FASTEMBED_CACHE_PATH": str(target / "fastembed"),
        "TORCH_HOME": str(target / "torch"),
        "OLLAMA_MODELS": str(target / "ollama"),
        "XDG_CACHE_HOME": str(target / "cache"),
    }
    for key, value in env_map.items():
        os.environ.setdefault(key, value)
    return target


def summarize_resources(resources_dir: Path) -> dict[str, Any]:
    """Return a live inventory of downloaded model/cache files inside resources_dir."""
    target = resources_dir.resolve()
    categories: dict[str, dict[str, Any]] = {}
    items: list[dict[str, Any]] = []
    total_bytes = 0
    file_count = 0

    for name in ("huggingface", "fastembed", "ollama", "gguf", "torch", "cache"):
        categories[name] = {"file_count": 0, "size_mb": 0.0}

    if target.is_dir():
        for path in sorted(target.rglob("*")):
            if not path.is_file():
                continue
            size = path.stat().st_size
            total_bytes += size
            file_count += 1
            rel = path.relative_to(target)
            top = rel.parts[0] if rel.parts else "other"
            if top not in categories:
                categories[top] = {"file_count": 0, "size_mb": 0.0}
            categories[top]["file_count"] += 1
            categories[top]["size_mb"] = round(categories[top]["size_mb"] + (size / (1024 * 1024)), 2)
            if len(items) < 60:
                items.append(
                    {
                        "path": rel.as_posix(),
                        "size_mb": round(size / (1024 * 1024), 2),
                    }
                )

    return {
        "resources_dir": str(target),
        "total_bytes": total_bytes,
        "total_mb": round(total_bytes / (1024 * 1024), 2),
        "file_count": file_count,
        "categories": categories,
        "items": items,
    }


_resources_cache_lock = threading.Lock()
_resources_cache: dict[str, tuple[float, dict[str, Any]]] = {}


def summarize_resources_cached(resources_dir: Path, ttl_seconds: float = 30.0) -> dict[str, Any]:
    """Return a TTL-cached resource summary to avoid repeated rglob scans on /health."""
    key = str(resources_dir.resolve())
    now = time.monotonic()
    with _resources_cache_lock:
        cached = _resources_cache.get(key)
        if cached is not None and (now - cached[0]) < ttl_seconds:
            return cached[1]
    summary = summarize_resources(resources_dir)
    with _resources_cache_lock:
        _resources_cache[key] = (time.monotonic(), summary)
    return summary


def load_env_file(env_path: Path | None = None) -> bool:
    """Load variables from a local .env file without overriding existing OS env vars."""
    loaded = False
    try:
        from dotenv import load_dotenv

        target = env_path or (Path(__file__).resolve().parent.parent / ".env")
        if target.is_file():
            loaded = bool(load_dotenv(dotenv_path=target, override=False))
    except ImportError:
        loaded = False
    configure_resource_env()
    return loaded


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
    token_ttl_seconds: int = 43200
    login_max_attempts: int = 5
    login_window_seconds: int = 60
    strict_embedder: bool = False
    max_upload_mb: int = 50
    rag_mode: RAGMode = "hybrid"
    resources_dir: Path | None = None

    @classmethod
    def from_env(cls, base_dir: Path | None = None) -> "Settings":
        root = Path(os.getenv("INSIGHT_DATA_DIR", str(base_dir or Path("./data")))).resolve()
        res_dir = configure_resource_env()
        precision = os.getenv("INSIGHT_PRECISION", "4bit").strip().lower()
        if precision not in {"4bit", "8bit", "full"}:
            precision = "4bit"

        backend = os.getenv("INSIGHT_LLM_BACKEND", "auto").strip().lower()
        if backend not in {"auto", "llamacpp", "local_smart", "unsloth", "hf", "ollama", "retrieval_only"}:
            backend = "auto"

        rag_mode = normalize_rag_mode(
            os.getenv("INSIGHT_RAG_MODE"),
            default="hybrid",
            strict=False,
        )

        return cls(
            data_dir=root,
            sqlite_path=Path(os.getenv("INSIGHT_SQLITE_PATH", str(root / "insight.sqlite"))).resolve(),
            qdrant_path=Path(os.getenv("INSIGHT_QDRANT_PATH", str(root / "qdrant_db"))).resolve(),
            upload_dir=Path(os.getenv("INSIGHT_UPLOAD_DIR", str(root / "uploads"))).resolve(),
            llm_backend=backend,  # type: ignore[arg-type]
            llm_model=os.getenv("INSIGHT_LLM_MODEL", DEFAULT_LLM_MODEL),
            precision_mode=precision,  # type: ignore[arg-type]
            max_seq_length=int(os.getenv("INSIGHT_MAX_SEQ_LENGTH", "4096")),
            max_new_tokens=int(os.getenv("INSIGHT_MAX_NEW_TOKENS", "512")),
            ollama_url=os.getenv("INSIGHT_OLLAMA_URL", "http://localhost:11434").rstrip("/"),
            embed_model=os.getenv(
                "INSIGHT_EMBED_MODEL",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            ),
            use_reranker=_env_bool("INSIGHT_USE_RERANKER", False),
            rerank_model=os.getenv("INSIGHT_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            top_k_candidates=int(os.getenv("INSIGHT_TOP_K_CANDIDATES", "12")),
            top_k_final=int(os.getenv("INSIGHT_TOP_K_FINAL", "4")),
            chunk_size=int(os.getenv("INSIGHT_CHUNK_SIZE", "900")),
            chunk_overlap=int(os.getenv("INSIGHT_CHUNK_OVERLAP", "150")),
            password=os.getenv("INSIGHT_PASSWORD", ""),
            auto_shutdown=_env_bool("INSIGHT_AUTO_SHUTDOWN", False),
            shutdown_grace_seconds=float(os.getenv("INSIGHT_SHUTDOWN_GRACE", "12.0")),
            heartbeat_timeout_seconds=float(os.getenv("INSIGHT_HEARTBEAT_TIMEOUT", "180.0")),
            token_ttl_seconds=max(1, int(os.getenv("INSIGHT_TOKEN_TTL_SECONDS", "43200"))),
            login_max_attempts=max(1, int(os.getenv("INSIGHT_LOGIN_MAX_ATTEMPTS", "5"))),
            login_window_seconds=max(1, int(os.getenv("INSIGHT_LOGIN_WINDOW_SECONDS", "60"))),
            strict_embedder=_env_bool("INSIGHT_STRICT_EMBEDDER", False),
            max_upload_mb=max(1, int(os.getenv("INSIGHT_MAX_UPLOAD_MB", "50"))),
            rag_mode=rag_mode,
            resources_dir=res_dir,
        )

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.qdrant_path.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        if self.resources_dir is not None:
            configure_resource_env(self.resources_dir)
