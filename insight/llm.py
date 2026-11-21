"""LLM backend adapters supporting Unsloth, Hugging Face, Ollama, and retrieval preview."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Protocol

from insight.config import DEFAULT_GGUF_FILENAME, DEFAULT_LLM_MODEL, DEFAULT_RESOURCES_DIR, Settings
from insight.vectorstore import STOPWORDS, RetrievedChunk

PRESET_MODELS: list[dict[str, Any]] = [
    {
        "id": "qwen2.5-0.5b",
        "name": "Qwen 3 - 0.5B",
        "backend": "llamacpp",
        "repo_id": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "filename": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "size_label": "~398 MB",
        "description": "Jawaban cepat",
    },
    {
        "id": "qwen3-1.7b",
        "name": "Qwen 3 - 1.7B",
        "backend": "llamacpp",
        "repo_id": "unsloth/Qwen3-1.7B-GGUF",
        "filename": "Qwen3-1.7B-UD-Q4_K_XL.gguf",
        "size_label": "~1.08 GB",
        "description": "Penalaran mendalam",
    },
]

QUERY_TOKEN_RE = re.compile(r"[A-Za-z0-9_]{2,}")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
THINK_OPEN_RE = re.compile(r"<think>.*$", re.DOTALL | re.IGNORECASE)


def _strip_thinking_tags(text: str) -> str:
    """Remove Qwen3 <think>...</think> reasoning blocks so only the final answer is returned."""
    cleaned = THINK_BLOCK_RE.sub("", text)
    cleaned = THINK_OPEN_RE.sub("", cleaned)
    return cleaned.strip()


def _stream_filter_think(chunks: Iterator[str]) -> Iterator[str]:
    """Filter <think>...</think> blocks from a token stream without leaking raw tags."""
    in_think = False
    buf = ""
    open_tag = "<think>"
    close_tag = "</think>"

    for delta in chunks:
        if not delta:
            continue
        buf += delta
        while buf:
            if in_think:
                end_idx = buf.lower().find(close_tag)
                if end_idx == -1:
                    keep = len(close_tag) - 1
                    buf = buf[-keep:] if len(buf) > keep else buf
                    break
                buf = buf[end_idx + len(close_tag) :].lstrip()
                in_think = False
            else:
                start_idx = buf.lower().find(open_tag)
                if start_idx != -1:
                    if start_idx > 0:
                        yield buf[:start_idx]
                    buf = buf[start_idx + len(open_tag) :]
                    in_think = True
                else:
                    hold = 0
                    lower_buf = buf.lower()
                    for k in range(min(len(buf), len(open_tag) - 1), 0, -1):
                        if open_tag.startswith(lower_buf[-k:]):
                            hold = k
                            break
                    if hold:
                        emit = buf[:-hold]
                        if emit:
                            yield emit
                        buf = buf[-hold:]
                        break
                    yield buf
                    buf = ""

    if not in_think and buf:
        yield buf


def _is_qwen3_model(name: str) -> bool:
    return "qwen3" in re.sub(r"[^a-z0-9]+", "", name.lower())


def _is_gguf_repo_or_file(model_name: str) -> bool:
    lower = model_name.strip().lower()
    return lower.endswith(".gguf") or ("/" in lower and "gguf" in lower)

SYSTEM_PROMPT = (
    "You are InSight, a private document assistant. "
    "Answer the user's question using ONLY the provided document excerpts. "
    "When presenting multiple items, steps, or points, format them as Markdown bullet points (`- `). "
    "Cite filenames when referencing information. "
    "Never invent file paths or details not present in the excerpts, and never repeat sentences. "
    "If the excerpts do not contain the answer, say so clearly in the same language as the question."
)


class LLMRunner(Protocol):
    label: str

    def generate(self, question: str, contexts: list[RetrievedChunk]) -> str: ...


def _format_chunk_loc(chunk: RetrievedChunk) -> str:
    page_part = f" | Hal. {chunk.page_label}" if chunk.page_label else ""
    return f"{chunk.section}{page_part}"


def _trim_chat_history(
    history: list[dict[str, str]] | None,
    max_messages: int = 8,
    max_chars: int = 800,
) -> list[dict[str, str]]:
    """Keep the most recent user/assistant turns within a compact character budget."""
    if not history:
        return []
    cleaned: list[dict[str, str]] = []
    for item in history[-max_messages:]:
        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "")).strip()
        if role not in {"user", "assistant"} or not content:
            continue
        cleaned.append({"role": role, "content": content[:max_chars]})
    return cleaned


def build_chat_messages(
    question: str,
    contexts: list[RetrievedChunk],
    history: list[dict[str, str]] | None = None,
    prompt_suffix: str = "",
) -> list[dict[str, str]]:
    """Build multi-turn chat messages with system prompt, trimmed history, and grounded user prompt."""
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for item in _trim_chat_history(history):
        messages.append({"role": item["role"], "content": item["content"]})
    user_prompt = build_prompt(question, contexts, history=None)
    if prompt_suffix:
        user_prompt = f"{user_prompt}{prompt_suffix}"
    messages.append({"role": "user", "content": user_prompt})
    return messages


def build_prompt(
    question: str,
    contexts: list[RetrievedChunk],
    history: list[dict[str, str]] | None = None,
) -> str:
    formatted_chunks = []
    for idx, chunk in enumerate(contexts, start=1):
        page_part = f" | Page: {chunk.page_label}" if chunk.page_label else ""
        formatted_chunks.append(
            f"[{idx}] File: {chunk.filename} | Section: {chunk.section}{page_part}\n{chunk.text}"
        )
    context_block = "\n\n---\n\n".join(formatted_chunks)
    trimmed = _trim_chat_history(history)
    history_block = ""
    if trimmed:
        lines = [f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in trimmed]
        history_block = "Recent conversation:\n" + "\n".join(lines) + "\n\n"
    return (
        f"Document excerpts:\n\n{context_block}\n\n"
        f"{history_block}"
        f"Question: {question}\n\n"
        "Answer clearly based strictly on the excerpts above:"
    )


class SmartLocalRunner:
    """Zero-download local synthesis runner using extractive summary."""

    label = "local_smart"

    def generate(self, question: str, contexts: list[RetrievedChunk]) -> str:
        if not contexts:
            return "Tidak ada potongan dokumen yang relevan ditemukan."

        q_lower = question.lower().strip()
        raw_q_tokens = set(QUERY_TOKEN_RE.findall(q_lower))
        q_tokens = {tok for tok in raw_q_tokens if tok not in STOPWORDS} or raw_q_tokens
        candidates: list[tuple[float, str, RetrievedChunk]] = []
        seen_sentences: set[str] = set()

        for rank, ctx in enumerate(contexts):
            raw_parts = [p.strip() for p in SENTENCE_SPLIT_RE.split(ctx.text) if len(p.strip()) >= 15]
            if not raw_parts and ctx.text.strip():
                raw_parts = [ctx.text.strip()[:300]]
            for part in raw_parts:
                clean_line = re.sub(r"^#+\s*", "", part).strip()
                norm_key = clean_line.lower()
                if len(clean_line) < 12 or norm_key in seen_sentences:
                    continue
                seen_sentences.add(norm_key)

                p_tokens = set(QUERY_TOKEN_RE.findall(norm_key))
                overlap = (len(q_tokens & p_tokens) / max(len(q_tokens), 1)) if q_tokens else 0.0
                phrase_bonus = 0.35 if q_lower and q_lower in norm_key else 0.0
                num_bonus = 0.1 if re.search(r"\d", clean_line) else 0.0
                def_bonus = 0.1 if (":" in clean_line or " adalah " in norm_key or " merupakan " in norm_key) else 0.0
                rank_bonus = max(0.0, 0.2 - (rank * 0.04))
                score = overlap + phrase_bonus + num_bonus + def_bonus + rank_bonus + (ctx.score * 0.15)
                candidates.append((score, clean_line[:320], ctx))

        candidates.sort(key=lambda item: item[0], reverse=True)
        top_items = candidates[:4]

        if not top_items:
            top_ctx = contexts[0]
            snippet = top_ctx.text[:350].replace("\n", " ").strip()
            return f"Berdasarkan **{top_ctx.filename}** ({_format_chunk_loc(top_ctx)}):\n\n{snippet}"

        lead_score, lead_text, lead_ctx = top_items[0]
        lines = [
            f"Berdasarkan dokumen **{lead_ctx.filename}** ({_format_chunk_loc(lead_ctx)}), poin paling relevan untuk pertanyaan Anda:",
            "",
            lead_text,
        ]
        if len(top_items) > 1:
            lines.append("")
            lines.append("Rincian pendukung dari dokumen:")
            for _, text, ctx in top_items[1:]:
                lines.append(f"- {text} [{ctx.filename} - {_format_chunk_loc(ctx)}]")
        return "\n".join(lines)


class RetrievalOnlyRunner(SmartLocalRunner):
    """Backward-compatible alias delegating to SmartLocalRunner."""

    label = "retrieval_only"


class OllamaRunner:
    """Calls a local Ollama server over HTTP, falling back to SmartLocalRunner if unreachable."""

    def __init__(self, model: str, base_url: str) -> None:
        self.model = model
        self.base_url = base_url
        self.label = f"ollama:{model}"

    def generate(
        self,
        question: str,
        contexts: list[RetrievedChunk],
        history: list[dict[str, str]] | None = None,
    ) -> str:
        import httpx

        messages = build_chat_messages(question, contexts, history=history)
        try:
            response = httpx.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "stream": False,
                    "messages": messages,
                },
                timeout=120.0,
            )
            response.raise_for_status()
            data = response.json()
            content = _strip_thinking_tags(str(data.get("message", {}).get("content", "")))
            if content:
                return content
        except Exception:
            pass
        return SmartLocalRunner().generate(question, contexts)

    def generate_stream(
        self,
        question: str,
        contexts: list[RetrievedChunk],
        history: list[dict[str, str]] | None = None,
    ) -> Iterator[str]:
        import httpx

        messages = build_chat_messages(question, contexts, history=history)
        emitted = False
        try:
            def _raw_tokens() -> Iterator[str]:
                with httpx.stream(
                    "POST",
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "stream": True,
                        "messages": messages,
                    },
                    timeout=120.0,
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if not line:
                            continue
                        data = json.loads(line)
                        delta = str(data.get("message", {}).get("content", ""))
                        if delta:
                            yield delta

            for token in _stream_filter_think(_raw_tokens()):
                emitted = True
                yield token
        except Exception:
            if not emitted:
                yield SmartLocalRunner().generate(question, contexts)
            return
        if not emitted:
            yield SmartLocalRunner().generate(question, contexts)


def find_ollama_executable() -> str | None:
    """Locate the local Ollama CLI binary across PATH and standard Windows install paths."""
    found = shutil.which("ollama")
    if found:
        return found
    local_app = os.getenv("LOCALAPPDATA", "")
    prog_files = os.getenv("ProgramFiles", r"C:\Program Files")
    candidates = [
        Path(local_app) / "Programs" / "Ollama" / "ollama.exe" if local_app else None,
        Path(prog_files) / "Ollama" / "ollama.exe",
    ]
    for candidate in candidates:
        if candidate and candidate.is_file():
            return str(candidate)
    return None


def resolve_ollama_model(settings: Settings, timeout: float = 0.5) -> str | None:
    """Return the active Ollama model tag if the local Ollama server is reachable and has models."""
    import httpx

    try:
        res = httpx.get(f"{settings.ollama_url}/api/tags", timeout=timeout)
        if res.status_code == 200:
            models = [m.get("name", "") for m in res.json().get("models", []) if m.get("name")]
            if models:
                target = settings.llm_model
                if target in models:
                    return target
                if _is_gguf_repo_or_file(target):
                    return None
                return models[0]
    except Exception:
        pass
    return None


def ensure_ollama_running(settings: Settings, wait_seconds: float = 5.0) -> bool:
    """Start local `ollama serve` with OLLAMA_MODELS pinned to resources_dir if installed but not yet running."""
    if settings.llm_backend not in {"auto", "ollama"}:
        return False

    import httpx

    try:
        res = httpx.get(f"{settings.ollama_url}/api/tags", timeout=0.4)
        if res.status_code == 200:
            return True
    except Exception:
        pass

    exe = find_ollama_executable()
    if not exe:
        return False

    env = os.environ.copy()
    if settings.resources_dir is not None:
        ollama_dir = settings.resources_dir / "ollama"
        ollama_dir.mkdir(parents=True, exist_ok=True)
        env["OLLAMA_MODELS"] = str(ollama_dir)

    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        subprocess.Popen(
            [exe, "serve"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
    except Exception:
        return False

    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        try:
            res = httpx.get(f"{settings.ollama_url}/api/tags", timeout=0.4)
            if res.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.25)
    return False


def _normalize_model_key(raw: str) -> str:
    leaf = raw.strip().split("/")[-1]
    leaf = re.sub(r"(?i)(?:[-_]?gguf)+$", "", leaf)
    return re.sub(r"[^a-z0-9]+", "", leaf.lower())


def resolve_gguf_model(settings: Settings) -> Path | None:
    """Locate a local .gguf model file under INSIGHT_GGUF_PATH or resources/gguf."""
    explicit = os.getenv("INSIGHT_GGUF_PATH", "").strip()
    if explicit:
        candidate = Path(explicit)
        if candidate.is_file():
            return candidate
    model_path = Path(settings.llm_model)
    if model_path.suffix.lower() == ".gguf" and model_path.is_file():
        return model_path

    gguf_dir = (settings.resources_dir / "gguf") if settings.resources_dir else None
    if gguf_dir is None or not gguf_dir.is_dir():
        return None
    ggufs = sorted(p for p in gguf_dir.rglob("*.gguf") if p.is_file())
    if not ggufs:
        return None

    norm_target = _normalize_model_key(settings.llm_model)

    preferred_file = os.getenv("INSIGHT_GGUF_FILE", "").strip()
    if preferred_file:
        pref_norm = _normalize_model_key(preferred_file)
        if not norm_target or norm_target in pref_norm or settings.llm_model.strip().lower() == DEFAULT_LLM_MODEL.lower():
            for item in ggufs:
                if item.name.lower() == preferred_file.lower():
                    return item

    matches = [
        item for item in ggufs
        if norm_target and norm_target in re.sub(r"[^a-z0-9]+", "", item.stem.lower())
    ]
    if matches:
        for item in matches:
            if item.name.lower() == DEFAULT_GGUF_FILENAME.lower():
                return item
        for item in matches:
            if "q4km" in re.sub(r"[^a-z0-9]+", "", item.stem.lower()):
                return item
        return matches[0]
    if _is_gguf_repo_or_file(settings.llm_model):
        return None
    return ggufs[0]


def ensure_gguf_model(settings: Settings) -> Path | None:
    """Download the configured Hugging Face GGUF file into resources/gguf if not yet cached locally."""
    if settings.llm_backend not in {"auto", "llamacpp"}:
        return None
    existing = resolve_gguf_model(settings)
    if existing is not None:
        return existing
    if "/" not in settings.llm_model or "gguf" not in settings.llm_model.lower():
        return None

    gguf_dir = ((settings.resources_dir or DEFAULT_RESOURCES_DIR) / "gguf").resolve()
    gguf_dir.mkdir(parents=True, exist_ok=True)
    norm_target = _normalize_model_key(settings.llm_model)

    filename = ""
    preferred_env = os.getenv("INSIGHT_GGUF_FILE", "").strip()
    if preferred_env and (not norm_target or norm_target in _normalize_model_key(preferred_env)):
        filename = preferred_env
    elif settings.llm_model.strip().lower() == DEFAULT_LLM_MODEL.lower():
        filename = DEFAULT_GGUF_FILENAME
    else:
        for p in PRESET_MODELS:
            if p.get("repo_id", "").lower() == settings.llm_model.lower():
                filename = p.get("filename", "")
                break

    try:
        from huggingface_hub import HfApi, hf_hub_download

        if not filename:
            files = [f for f in HfApi().list_repo_files(settings.llm_model) if f.lower().endswith(".gguf")]
            for preferred in ("UD-Q4_K_XL", "Q4_K_M", "Q4_K_S", "Q4_0", "Q5_K_M"):
                hit = next((f for f in files if preferred.lower() in f.lower()), None)
                if hit:
                    filename = hit
                    break
            if not filename and files:
                filename = files[0]
        if not filename:
            return None
        downloaded = hf_hub_download(
            repo_id=settings.llm_model,
            filename=filename,
            local_dir=str(gguf_dir),
        )
        return Path(downloaded)
    except Exception:
        return None


def list_available_models(settings: Settings) -> list[dict[str, Any]]:
    """Return all preset models, cached GGUF models, and running Ollama models."""
    gguf_dir = (settings.resources_dir / "gguf") if settings.resources_dir else None
    cached_ggufs: dict[str, Path] = {}
    if gguf_dir and gguf_dir.is_dir():
        for p in gguf_dir.rglob("*.gguf"):
            if p.is_file():
                cached_ggufs[p.name.lower()] = p

    norm_current = _normalize_model_key(settings.llm_model)
    results: list[dict[str, Any]] = []
    matched_active = False

    for preset in PRESET_MODELS:
        fname = preset.get("filename", "").lower()
        repo = preset.get("repo_id", "")
        norm_preset = _normalize_model_key(repo)
        is_downloaded = fname in cached_ggufs or any(
            norm_preset in re.sub(r"[^a-z0-9]+", "", k) for k in cached_ggufs
        )
        is_active = (
            settings.llm_backend in {"auto", "llamacpp"}
            and (
                settings.llm_model.lower() == repo.lower()
                or (norm_current and norm_current in norm_preset)
            )
        )
        if is_active and not matched_active:
            matched_active = True
        else:
            is_active = False

        results.append({
            "id": preset["id"],
            "name": preset["name"],
            "backend": preset["backend"],
            "repo_id": repo,
            "filename": preset["filename"],
            "size_label": preset["size_label"],
            "description": preset["description"],
            "downloaded": is_downloaded,
            "active": is_active,
        })

    preset_filenames = {p.get("filename", "").lower() for p in PRESET_MODELS}
    for fname, p in cached_ggufs.items():
        if fname not in preset_filenames:
            custom_id = f"custom:{p.name}"
            is_active = (
                not matched_active
                and settings.llm_backend in {"auto", "llamacpp"}
                and (settings.llm_model.lower() == p.name.lower() or norm_current in _normalize_model_key(p.name))
            )
            if is_active:
                matched_active = True
            results.append({
                "id": custom_id,
                "name": p.name,
                "backend": "llamacpp",
                "repo_id": p.name,
                "filename": p.name,
                "size_label": f"{round(p.stat().st_size / (1024 * 1024), 1)} MB",
                "description": "Custom local GGUF model",
                "downloaded": True,
                "active": is_active,
            })

    try:
        import httpx
        res = httpx.get(f"{settings.ollama_url}/api/tags", timeout=0.3)
        if res.status_code == 200:
            for item in res.json().get("models", []):
                tag = item.get("name", "")
                if tag:
                    is_active = settings.llm_backend == "ollama" and settings.llm_model.lower() == tag.lower()
                    results.append({
                        "id": f"ollama:{tag}",
                        "name": f"Ollama: {tag}",
                        "backend": "ollama",
                        "repo_id": tag,
                        "filename": "",
                        "size_label": "Ollama Service",
                        "description": "Model dari Ollama lokal",
                        "downloaded": True,
                        "active": is_active,
                    })
    except Exception:
        pass

    if not any(r["active"] for r in results) and results:
        results[0]["active"] = True

    return results


class LlamaCppRunner:
    """Runs quantized local GGUF models in-process via llama-cpp-python (low-RAM CPU)."""

    def __init__(self, model_path: Path, settings: Settings) -> None:
        from llama_cpp import Llama

        n_ctx = max(512, settings.max_seq_length)
        n_threads = max(1, min(8, os.cpu_count() or 4))
        self._lock = threading.Lock()
        self._model = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_threads=n_threads,
            last_n_tokens_size=256,
            verbose=False,
        )
        self.settings = settings
        self.model_path = model_path
        self._is_qwen3 = _is_qwen3_model(model_path.name) or _is_qwen3_model(settings.llm_model)
        self.label = f"llamacpp:{model_path.name}"

    _SAMPLING_INT_KEYS = ("max_tokens", "top_k")
    _SAMPLING_FLOAT_KEYS = (
        "temperature",
        "top_p",
        "min_p",
        "repeat_penalty",
        "frequency_penalty",
        "presence_penalty",
    )

    def _prepare_chat_call(
        self,
        question: str,
        contexts: list[RetrievedChunk],
        history: list[dict[str, str]] | None = None,
        sampling_params: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, str]], dict]:
        if self._is_qwen3:
            messages = build_chat_messages(question, contexts, history=history, prompt_suffix="\n/no_think")
            sampling: dict[str, Any] = {
                "max_tokens": min(self.settings.max_new_tokens, 2048),
                "temperature": 0.2,
                "top_p": 0.8,
                "top_k": 20,
                "min_p": 0.0,
                "repeat_penalty": 1.1,
            }
        else:
            messages = build_chat_messages(question, contexts, history=history)
            sampling = {
                "max_tokens": min(self.settings.max_new_tokens, 1024),
                "temperature": 0.2,
                "top_p": 0.9,
                "repeat_penalty": 1.18,
                "frequency_penalty": 0.2,
                "presence_penalty": 0.2,
            }
        if sampling_params:
            for key in self._SAMPLING_INT_KEYS:
                if key in sampling_params and sampling_params[key] is not None:
                    try:
                        val = int(sampling_params[key])
                        sampling[key] = max(1, val) if key == "max_tokens" else max(0, val)
                    except (TypeError, ValueError):
                        pass
            for key in self._SAMPLING_FLOAT_KEYS:
                if key in sampling_params and sampling_params[key] is not None:
                    try:
                        sampling[key] = float(sampling_params[key])
                    except (TypeError, ValueError):
                        pass
        return messages, sampling

    def _get_lock(self) -> threading.Lock:
        lock = getattr(self, "_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._lock = lock
        return lock

    def generate(
        self,
        question: str,
        contexts: list[RetrievedChunk],
        history: list[dict[str, str]] | None = None,
        sampling_params: dict[str, Any] | None = None,
    ) -> str:
        messages, sampling = self._prepare_chat_call(
            question, contexts, history=history, sampling_params=sampling_params
        )
        try:
            with self._get_lock():
                response = self._model.create_chat_completion(messages=messages, **sampling)
            content = _strip_thinking_tags(str(response["choices"][0]["message"]["content"] or ""))
            if content:
                return content
        except Exception:
            pass
        return SmartLocalRunner().generate(question, contexts)

    def generate_stream(
        self,
        question: str,
        contexts: list[RetrievedChunk],
        history: list[dict[str, str]] | None = None,
        sampling_params: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        messages, sampling = self._prepare_chat_call(
            question, contexts, history=history, sampling_params=sampling_params
        )
        emitted = False
        try:
            def _raw_tokens() -> Iterator[str]:
                with self._get_lock():
                    stream = self._model.create_chat_completion(
                        messages=messages,
                        stream=True,
                        **sampling,
                    )
                    for chunk in stream:
                        delta = str(chunk["choices"][0].get("delta", {}).get("content") or "")
                        if delta:
                            yield delta

            for token in _stream_filter_think(_raw_tokens()):
                emitted = True
                yield token
        except Exception:
            if not emitted:
                yield SmartLocalRunner().generate(question, contexts)
            return
        if not emitted:
            yield SmartLocalRunner().generate(question, contexts)


class AutoRunner:
    """Uses local GGUF (llama-cpp-python), local Ollama, or SmartLocalRunner fallback."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._delegate: LLMRunner = self._resolve(settings)
        self.label = self._delegate.label

    @staticmethod
    def _resolve(settings: Settings) -> LLMRunner:
        prefer_gguf = _is_gguf_repo_or_file(settings.llm_model)
        if prefer_gguf:
            gguf_path = resolve_gguf_model(settings)
            if gguf_path is not None:
                try:
                    return LlamaCppRunner(gguf_path, settings)
                except Exception:
                    pass
        chosen = resolve_ollama_model(settings)
        if chosen:
            return OllamaRunner(chosen, settings.ollama_url)
        if not prefer_gguf:
            gguf_path = resolve_gguf_model(settings)
            if gguf_path is not None:
                try:
                    return LlamaCppRunner(gguf_path, settings)
                except Exception:
                    pass
        return SmartLocalRunner()

    def generate(
        self,
        question: str,
        contexts: list[RetrievedChunk],
        history: list[dict[str, str]] | None = None,
    ) -> str:
        if isinstance(self._delegate, SmartLocalRunner):
            refreshed = self._resolve(self.settings)
            self._delegate = refreshed
            self.label = refreshed.label
        return _call_runner_generate(self._delegate, question, contexts, history=history)

    def generate_stream(
        self,
        question: str,
        contexts: list[RetrievedChunk],
        history: list[dict[str, str]] | None = None,
    ) -> Iterator[str]:
        if isinstance(self._delegate, SmartLocalRunner):
            refreshed = self._resolve(self.settings)
            self._delegate = refreshed
            self.label = refreshed.label
        return _call_runner_generate_stream(self._delegate, question, contexts, history=history)


class UnslothRunner:
    """Runs Hugging Face / Unsloth models with selectable 4bit, 8bit, or full precision."""

    def __init__(self, settings: Settings) -> None:
        from unsloth import FastLanguageModel

        load_4bit = settings.precision_mode == "4bit"
        load_8bit = settings.precision_mode == "8bit"
        self._model, self._tokenizer = FastLanguageModel.from_pretrained(
            model_name=settings.llm_model,
            max_seq_length=settings.max_seq_length,
            load_in_4bit=load_4bit,
            load_in_8bit=load_8bit,
        )
        FastLanguageModel.for_inference(self._model)
        self.settings = settings
        self.label = f"unsloth:{settings.llm_model} ({settings.precision_mode})"

    def generate(self, question: str, contexts: list[RetrievedChunk]) -> str:
        import torch

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, contexts)},
        ]
        if hasattr(self._tokenizer, "apply_chat_template"):
            prompt_text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt_text = f"{SYSTEM_PROMPT}\n\n{build_prompt(question, contexts)}"

        inputs = self._tokenizer([prompt_text], return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.settings.max_new_tokens,
                do_sample=False,
            )
        generated = output_ids[0][inputs["input_ids"].shape[1] :]
        return _strip_thinking_tags(self._tokenizer.decode(generated, skip_special_tokens=True))


class HuggingFaceRunner:
    """Runs standard Hugging Face transformers models with optional BitsAndBytes quantization."""

    def __init__(self, settings: Settings) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        kwargs: dict = {}
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            kwargs["device_map"] = "auto"
            if settings.precision_mode == "4bit":
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
            elif settings.precision_mode == "8bit":
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            else:
                kwargs["torch_dtype"] = "auto"
        else:
            kwargs["torch_dtype"] = torch.float32

        self._tokenizer = AutoTokenizer.from_pretrained(settings.llm_model)
        self._model = AutoModelForCausalLM.from_pretrained(settings.llm_model, **kwargs)
        self.settings = settings
        self.label = f"hf:{settings.llm_model} ({settings.precision_mode})"

    def generate(self, question: str, contexts: list[RetrievedChunk]) -> str:
        import torch

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, contexts)},
        ]
        if hasattr(self._tokenizer, "apply_chat_template"):
            prompt_text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt_text = f"{SYSTEM_PROMPT}\n\n{build_prompt(question, contexts)}"

        inputs = self._tokenizer(prompt_text, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.settings.max_new_tokens,
                do_sample=False,
            )
        generated = output_ids[0][inputs["input_ids"].shape[1] :]
        return _strip_thinking_tags(self._tokenizer.decode(generated, skip_special_tokens=True))


def _call_runner_generate(
    runner: LLMRunner,
    question: str,
    contexts: list[RetrievedChunk],
    history: list[dict[str, str]] | None = None,
    sampling_params: dict[str, Any] | None = None,
) -> str:
    if sampling_params:
        try:
            return runner.generate(question, contexts, history=history, sampling_params=sampling_params)  # type: ignore[call-arg]
        except TypeError:
            pass
    if history is not None:
        try:
            return runner.generate(question, contexts, history=history)  # type: ignore[call-arg]
        except TypeError:
            pass
    return runner.generate(question, contexts)


def _call_runner_generate_stream(
    runner: LLMRunner,
    question: str,
    contexts: list[RetrievedChunk],
    history: list[dict[str, str]] | None = None,
    sampling_params: dict[str, Any] | None = None,
) -> Iterator[str]:
    stream_fn = getattr(runner, "generate_stream", None)
    if callable(stream_fn):
        if sampling_params:
            try:
                yield from stream_fn(question, contexts, history=history, sampling_params=sampling_params)
                return
            except TypeError:
                pass
        try:
            yield from stream_fn(question, contexts, history=history)
            return
        except TypeError:
            yield from stream_fn(question, contexts)
            return
    yield _call_runner_generate(runner, question, contexts, history=history, sampling_params=sampling_params)


@dataclass
class LazyLLM:
    """Defers heavyweight model loading until the first chat request."""

    settings: Settings
    _runner: LLMRunner | None = None

    @property
    def label(self) -> str:
        if self._runner is not None:
            return self._runner.label
        return f"{self.settings.llm_backend}:{self.settings.llm_model} ({self.settings.precision_mode})"

    @property
    def active_backend(self) -> str:
        return self.settings.llm_backend

    @property
    def active_model(self) -> str:
        return self.settings.llm_model

    @property
    def active_model_id(self) -> str:
        models = list_available_models(self.settings)
        for m in models:
            if m.get("active"):
                return str(m.get("id"))
        return "custom"

    def switch_model(self, model_id: str) -> dict[str, Any]:
        """Switch active LLM to a preset model or custom GGUF/Ollama model."""
        from dataclasses import replace

        norm_id = model_id.strip().lower()
        target_preset = next((p for p in PRESET_MODELS if p["id"] == norm_id), None)

        if target_preset is not None:
            new_settings = replace(
                self.settings,
                llm_backend="llamacpp",
                llm_model=target_preset["repo_id"],
            )
            ensure_gguf_model(new_settings)
            self.settings = new_settings
            self._runner = None
            return {
                "id": target_preset["id"],
                "name": target_preset["name"],
                "backend": "llamacpp",
                "model": target_preset["repo_id"],
                "runtime": self.active_runtime,
            }

        if norm_id.startswith("ollama:"):
            ollama_tag = norm_id.split(":", 1)[1]
            new_settings = replace(
                self.settings,
                llm_backend="ollama",
                llm_model=ollama_tag,
            )
            self.settings = new_settings
            self._runner = None
            return {
                "id": norm_id,
                "name": f"Ollama: {ollama_tag}",
                "backend": "ollama",
                "model": ollama_tag,
                "runtime": self.active_runtime,
            }

        gguf_dir = (self.settings.resources_dir / "gguf") if self.settings.resources_dir else None
        if gguf_dir and gguf_dir.is_dir():
            for g in gguf_dir.rglob("*.gguf"):
                if g.name.lower() == norm_id or f"custom:{g.name.lower()}" == norm_id:
                    new_settings = replace(
                        self.settings,
                        llm_backend="llamacpp",
                        llm_model=g.name,
                    )
                    self.settings = new_settings
                    self._runner = None
                    return {
                        "id": f"custom:{g.name}",
                        "name": g.name,
                        "backend": "llamacpp",
                        "model": g.name,
                        "runtime": self.active_runtime,
                    }

        raise ValueError(f"Model ID '{model_id}' tidak ditemukan.")

    @property
    def active_precision(self) -> str:
        return self.settings.precision_mode

    @property
    def active_runtime(self) -> str:
        if self._runner is not None:
            return self._runner.label
        backend = self.settings.llm_backend
        if backend == "auto":
            gguf = resolve_gguf_model(self.settings)
            if gguf is not None:
                return f"llamacpp:{gguf.name} (ready)"
            return "auto (pending)"
        if backend == "llamacpp":
            gguf = resolve_gguf_model(self.settings)
            return f"llamacpp:{gguf.name} (ready)" if gguf is not None else "local_smart (gguf missing)"
        return backend

    def _get_runner(self) -> LLMRunner:
        if self._runner is not None:
            return self._runner

        backend = self.settings.llm_backend
        if backend == "auto":
            self._runner = AutoRunner(self.settings)
        elif backend == "llamacpp":
            gguf_path = resolve_gguf_model(self.settings)
            if gguf_path is not None:
                self._runner = LlamaCppRunner(gguf_path, self.settings)
            else:
                self._runner = SmartLocalRunner()
        elif backend == "local_smart":
            self._runner = SmartLocalRunner()
        elif backend == "ollama":
            self._runner = OllamaRunner(self.settings.llm_model, self.settings.ollama_url)
        elif backend == "unsloth":
            self._runner = UnslothRunner(self.settings)
        elif backend == "hf":
            self._runner = HuggingFaceRunner(self.settings)
        else:
            self._runner = RetrievalOnlyRunner()
        return self._runner

    def generate(
        self,
        question: str,
        contexts: list[RetrievedChunk],
        history: list[dict[str, str]] | None = None,
        sampling_params: dict[str, Any] | None = None,
    ) -> str:
        return _call_runner_generate(
            self._get_runner(), question, contexts, history=history, sampling_params=sampling_params
        )

    def generate_stream(
        self,
        question: str,
        contexts: list[RetrievedChunk],
        history: list[dict[str, str]] | None = None,
        sampling_params: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        return _call_runner_generate_stream(
            self._get_runner(), question, contexts, history=history, sampling_params=sampling_params
        )
