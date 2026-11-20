"""LLM backend adapters supporting Unsloth, Hugging Face, Ollama, and retrieval preview."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from insight.config import Settings
from insight.vectorstore import STOPWORDS, RetrievedChunk

GREETING_WORDS = {
    "halo",
    "hai",
    "hi",
    "hello",
    "hey",
    "pagi",
    "siang",
    "sore",
    "malam",
    "tes",
    "test",
    "ping",
    "bantuan",
    "help",
}
GREETING_PHRASES = {
    "apa kabar",
    "bisa apa",
    "selamat pagi",
    "selamat siang",
    "selamat sore",
    "selamat malam",
}
QUERY_TOKEN_RE = re.compile(r"[A-Za-z0-9_]{2,}")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def is_greeting(text: str) -> bool:
    """Return True when the user sent a short conversational greeting or help check."""
    cleaned = re.sub(r"[^\w\s]", " ", text.lower()).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if not cleaned:
        return False
    if cleaned in GREETING_WORDS or cleaned in GREETING_PHRASES:
        return True
    words = cleaned.split()
    return len(words) <= 3 and all(w in GREETING_WORDS for w in words)

SYSTEM_PROMPT = (
    "You are InSight, a private document assistant. "
    "Answer the user's question using ONLY the provided document excerpts. "
    "Cite filenames when referencing information. "
    "If the excerpts do not contain the answer, say so clearly in the same language as the question."
)


class LLMRunner(Protocol):
    label: str

    def generate(self, question: str, contexts: list[RetrievedChunk]) -> str: ...


def _format_chunk_loc(chunk: RetrievedChunk) -> str:
    page_part = f" | Hal. {chunk.page_label}" if chunk.page_label else ""
    return f"{chunk.section}{page_part}"


def build_prompt(question: str, contexts: list[RetrievedChunk]) -> str:
    formatted_chunks = []
    for idx, chunk in enumerate(contexts, start=1):
        page_part = f" | Page: {chunk.page_label}" if chunk.page_label else ""
        formatted_chunks.append(
            f"[{idx}] File: {chunk.filename} | Section: {chunk.section}{page_part}\n{chunk.text}"
        )
    context_block = "\n\n---\n\n".join(formatted_chunks)
    return (
        f"Document excerpts:\n\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer clearly based strictly on the excerpts above:"
    )


class SmartLocalRunner:
    """Zero-download local synthesis runner with greeting awareness and extractive summary."""

    label = "local_smart"

    def generate(self, question: str, contexts: list[RetrievedChunk]) -> str:
        if is_greeting(question):
            if not contexts:
                return (
                    "Halo! Saya InSight, asisten dokumen pribadi Anda. "
                    "Saat ini belum ada dokumen di ruang kerja. Klik '+' atau 'Unggah Dokumen' di panel kiri "
                    "untuk menambahkan file, lalu tanyakan isinya kepada saya."
                )
            names = list(dict.fromkeys(ctx.filename for ctx in contexts))
            return (
                f"Halo! Dokumen Anda siap ditelusuri ({', '.join(names)}). "
                "Silakan ajukan pertanyaan spesifik atau minta ringkasan isinya."
            )

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

    def generate(self, question: str, contexts: list[RetrievedChunk]) -> str:
        import httpx

        user_prompt = build_prompt(question, contexts)
        try:
            response = httpx.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                },
                timeout=120.0,
            )
            response.raise_for_status()
            data = response.json()
            content = str(data.get("message", {}).get("content", "")).strip()
            if content:
                return content
        except Exception:
            pass
        return SmartLocalRunner().generate(question, contexts)


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
                return target if target in models else models[0]
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
    # Prefer a GGUF file matching the configured model family (e.g. qwen2.5 / 0.5b)
    norm_target = re.sub(r"[^a-z0-9]+", "", settings.llm_model.lower())
    for item in ggufs:
        norm_name = re.sub(r"[^a-z0-9]+", "", item.stem.lower())
        if norm_target and norm_target in norm_name:
            return item
    return ggufs[0]


class LlamaCppRunner:
    """Runs quantized local GGUF models in-process via llama-cpp-python (low-RAM CPU)."""

    def __init__(self, model_path: Path, settings: Settings) -> None:
        from llama_cpp import Llama

        n_ctx = max(512, min(settings.max_seq_length, 2048))
        n_threads = max(1, min(4, os.cpu_count() or 2))
        self._model = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=False,
        )
        self.settings = settings
        self.model_path = model_path
        self.label = f"llamacpp:{model_path.name}"

    def generate(self, question: str, contexts: list[RetrievedChunk]) -> str:
        user_prompt = build_prompt(question, contexts)
        try:
            response = self._model.create_chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=min(self.settings.max_new_tokens, 512),
                temperature=0.2,
            )
            content = str(response["choices"][0]["message"]["content"] or "").strip()
            if content:
                return content
        except Exception:
            pass
        return SmartLocalRunner().generate(question, contexts)


class AutoRunner:
    """Uses local Ollama, local GGUF (llama-cpp-python), or SmartLocalRunner fallback."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._delegate: LLMRunner = self._resolve(settings)
        self.label = self._delegate.label

    @staticmethod
    def _resolve(settings: Settings) -> LLMRunner:
        chosen = resolve_ollama_model(settings)
        if chosen:
            return OllamaRunner(chosen, settings.ollama_url)
        gguf_path = resolve_gguf_model(settings)
        if gguf_path is not None:
            try:
                return LlamaCppRunner(gguf_path, settings)
            except Exception:
                pass
        return SmartLocalRunner()

    def generate(self, question: str, contexts: list[RetrievedChunk]) -> str:
        if isinstance(self._delegate, SmartLocalRunner):
            refreshed = self._resolve(self.settings)
            self._delegate = refreshed
            self.label = refreshed.label
        return self._delegate.generate(question, contexts)


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
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()


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
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()


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

    def generate(self, question: str, contexts: list[RetrievedChunk]) -> str:
        return self._get_runner().generate(question, contexts)
