"""LLM backend adapters supporting Unsloth, Hugging Face, Ollama, and retrieval preview."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from insight.config import Settings
from insight.vectorstore import RetrievedChunk

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


def build_prompt(question: str, contexts: list[RetrievedChunk]) -> str:
    formatted_chunks = []
    for idx, chunk in enumerate(contexts, start=1):
        formatted_chunks.append(
            f"[{idx}] File: {chunk.filename} | Section: {chunk.section}\n{chunk.text}"
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
        q_tokens = set(QUERY_TOKEN_RE.findall(q_lower))
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
            return f"Berdasarkan **{top_ctx.filename}** ({top_ctx.section}):\n\n{snippet}"

        lead_score, lead_text, lead_ctx = top_items[0]
        lines = [
            f"Berdasarkan dokumen **{lead_ctx.filename}** ({lead_ctx.section}), poin paling relevan untuk pertanyaan Anda:",
            "",
            lead_text,
        ]
        if len(top_items) > 1:
            lines.append("")
            lines.append("Rincian pendukung dari dokumen:")
            for _, text, ctx in top_items[1:]:
                lines.append(f"- {text} [{ctx.filename} - {ctx.section}]")
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


class AutoRunner:
    """Uses local Ollama when available, otherwise falls back seamlessly to SmartLocalRunner."""

    def __init__(self, settings: Settings) -> None:
        self._delegate: LLMRunner = self._resolve(settings)
        self.label = self._delegate.label

    @staticmethod
    def _resolve(settings: Settings) -> LLMRunner:
        import httpx

        try:
            res = httpx.get(f"{settings.ollama_url}/api/tags", timeout=0.5)
            if res.status_code == 200:
                models = [m.get("name", "") for m in res.json().get("models", []) if m.get("name")]
                if models:
                    target = settings.llm_model
                    chosen = target if target in models else models[0]
                    return OllamaRunner(chosen, settings.ollama_url)
        except Exception:
            pass
        return SmartLocalRunner()

    def generate(self, question: str, contexts: list[RetrievedChunk]) -> str:
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

        kwargs: dict = {"device_map": "auto"}
        if settings.precision_mode == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        elif settings.precision_mode == "8bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            kwargs["torch_dtype"] = "auto"

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

    def _get_runner(self) -> LLMRunner:
        if self._runner is not None:
            return self._runner

        backend = self.settings.llm_backend
        if backend == "auto":
            self._runner = AutoRunner(self.settings)
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
