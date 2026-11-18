"""LLM backend adapters supporting Unsloth, Hugging Face, Ollama, and retrieval preview."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from insight.config import Settings
from insight.vectorstore import RetrievedChunk

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


class RetrievalOnlyRunner:
    """Zero-model fallback returning retrieved passages directly."""

    label = "retrieval_only"

    def generate(self, question: str, contexts: list[RetrievedChunk]) -> str:
        if not contexts:
            return "Tidak ada potongan dokumen yang relevan ditemukan."
        lines = [
            "Mode preview aktif (belum terhubung ke LLM lokal/GPU). Berikut kutipan paling relevan dari dokumen Anda:",
            "",
        ]
        for idx, ctx in enumerate(contexts, start=1):
            snippet = ctx.text[:350].replace("\n", " ").strip()
            if len(ctx.text) > 350:
                snippet += "..."
            lines.append(f"{idx}. **{ctx.filename}** ({ctx.section}): {snippet}")
        return "\n".join(lines)


class OllamaRunner:
    """Calls a local Ollama server over HTTP (100% offline)."""

    def __init__(self, model: str, base_url: str) -> None:
        self.model = model
        self.base_url = base_url
        self.label = f"ollama:{model}"

    def generate(self, question: str, contexts: list[RetrievedChunk]) -> str:
        import httpx

        user_prompt = build_prompt(question, contexts)
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
        return str(data.get("message", {}).get("content", "")).strip()


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
        if backend == "ollama":
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
